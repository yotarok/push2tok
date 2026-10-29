mod beam;

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use rustfst::fst_impls::{ConstFst, VectorFst};
use rustfst::fst_traits::{CoreFst, ExpandedFst, MutableFst, SerializableFst};
use rustfst::semirings::{Semiring, TropicalWeight};
use rustfst::{Label, StateId, Tr, EPS_LABEL};

type ByteFst = VectorFst<TropicalWeight>;

// rustfst reserves label 0 for epsilon. Input bytes and output token IDs
// are both shifted by one; we select usize labels to hold all u32 token IDs
// on supported 64-bit targets.
fn output_label(token: u32) -> PyResult<Label> {
    (token as Label)
        .checked_add(1)
        .ok_or_else(|| PyValueError::new_err("token ID exceeds rustfst label range"))
}

fn add_output_path(
    fst: &mut ByteFst,
    from: StateId,
    input: Label,
    to: StateId,
    tokens: &[u32],
) -> PyResult<()> {
    let mut source = from;
    for (i, &token) in tokens.iter().enumerate() {
        let destination = if i + 1 == tokens.len() {
            to
        } else {
            fst.add_state()
        };
        fst.add_tr(
            source,
            Tr::new(
                if i == 0 { input } else { EPS_LABEL },
                output_label(token)?,
                TropicalWeight::one(),
                destination,
            ),
        )
        .expect("constructed states exist");
        source = destination;
    }
    if tokens.is_empty() {
        fst.add_tr(from, Tr::new(input, EPS_LABEL, TropicalWeight::one(), to))
            .expect("constructed states exist");
    }
    Ok(())
}

fn arc_tokens(tr: &Tr<TropicalWeight>, child: &[u32]) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(child.len() + usize::from(tr.olabel != EPS_LABEL));
    if tr.olabel != EPS_LABEL {
        tokens.push((tr.olabel - 1) as u32);
    }
    tokens.extend_from_slice(child);
    tokens
}

fn shared_length(candidate: &[u32], tr: &Tr<TropicalWeight>, child: &[u32]) -> usize {
    let first = (tr.olabel != EPS_LABEL).then(|| (tr.olabel - 1) as u32);
    candidate
        .iter()
        .zip(first.into_iter().chain(child.iter().copied()))
        .take_while(|(left, right)| **left == *right)
        .count()
}

struct Machine {
    fst: ByteFst,
    start: StateId,
    // Original state numbers are for reporting errors only. The graph itself
    // (including final outputs and intermediate epsilon states) lives in rustfst.
    state_names: Vec<usize>,
    // Longest output prefix common to every accepting continuation per state.
    // None denotes a state from which no final state is reachable.
    prefixes: Vec<Option<Vec<u32>>>,
}

impl Machine {
    fn new(
        transitions: Vec<(usize, u8, usize, Vec<u32>)>,
        finals: HashMap<usize, Vec<u32>>,
        start_state: usize,
    ) -> PyResult<Self> {
        let mut names = BTreeSet::from([start_state]);
        let mut seen = HashSet::new();
        for (from, byte, to, _) in &transitions {
            if !seen.insert((*from, *byte)) {
                return Err(PyValueError::new_err(format!(
                    "duplicate transition from state {from} on byte {byte}"
                )));
            }
            names.insert(*from);
            names.insert(*to);
        }
        names.extend(finals.keys().copied());

        let mut fst = ByteFst::new();
        let mut ids = HashMap::new();
        let state_names: Vec<_> = names.into_iter().collect();
        for &name in &state_names {
            ids.insert(name, fst.add_state());
        }
        let start = ids[&start_state];
        fst.set_start(start)
            .expect("constructed start state exists");

        for (from, byte, to, tokens) in transitions {
            add_output_path(
                &mut fst,
                ids[&from],
                Label::from(byte) + 1,
                ids[&to],
                &tokens,
            )?;
        }
        for (name, tokens) in finals {
            let state = ids[&name];
            if tokens.is_empty() {
                fst.set_final(state, TropicalWeight::one())
                    .expect("constructed final state exists");
            } else {
                let end = fst.add_state();
                add_output_path(&mut fst, state, EPS_LABEL, end, &tokens)?;
                fst.set_final(end, TropicalWeight::one())
                    .expect("constructed final state exists");
            }
        }

        let prefixes = Self::compute_prefixes(&fst);
        if prefixes[start].is_none() {
            return Err(PyValueError::new_err(
                "start state has no accepting continuation",
            ));
        }
        Ok(Self {
            fst,
            start,
            state_names,
            prefixes,
        })
    }

    fn from_loaded(fst: ByteFst) -> PyResult<Self> {
        let count = fst.num_states();
        let start = fst
            .start()
            .filter(|&state| state < count)
            .ok_or_else(|| PyValueError::new_err("FST has no valid start state"))?;
        // The stream accepts deterministic byte arcs, with at most one
        // epsilon arc per state for output-only paths or terminal outputs.
        let mut epsilons = vec![None; count];
        for state in fst.states_range() {
            let final_weight = fst.final_weight(state).expect("valid state");
            if final_weight.is_some_and(|weight| weight != TropicalWeight::one()) {
                return Err(PyValueError::new_err(format!(
                    "unsupported final weight at state {state}"
                )));
            }
            let mut labels = HashSet::new();
            for tr in fst.get_trs(state).expect("valid state").iter() {
                if tr.nextstate >= count
                    || tr.ilabel > 256
                    || (tr.olabel as u64) > u32::MAX as u64 + 1
                {
                    return Err(PyValueError::new_err(format!(
                        "unsupported transition label or destination at state {state}"
                    )));
                }
                if tr.weight != TropicalWeight::one() {
                    return Err(PyValueError::new_err(format!(
                        "unsupported transition weight at state {state}"
                    )));
                }
                if !labels.insert(tr.ilabel) {
                    return Err(PyValueError::new_err(format!(
                        "non-deterministic transitions at state {state} on label {}",
                        tr.ilabel
                    )));
                }
                if tr.ilabel == EPS_LABEL {
                    epsilons[state] = Some(tr.nextstate);
                }
            }
            if final_weight.is_some() && epsilons[state].is_some() {
                return Err(PyValueError::new_err(format!(
                    "ambiguous final state with an epsilon arc at state {state}"
                )));
            }
        }
        // An epsilon cycle cannot be consumed by an incremental byte stream.
        let mut marks = vec![0u8; count];
        for root in fst.states_range() {
            let mut state = root;
            let mut path = Vec::new();
            while marks[state] == 0 {
                marks[state] = 1;
                path.push(state);
                if let Some(next) = epsilons[state] {
                    state = next;
                } else {
                    break;
                }
            }
            if marks[state] == 1 && epsilons[state].is_some() {
                return Err(PyValueError::new_err("epsilon cycle in FST"));
            }
            for state in path {
                marks[state] = 2;
            }
        }
        // If epsilon and byte arcs coexist, epsilon must be a terminal-only
        // branch; otherwise there can be two paths for the same input bytes.
        for state in fst.states_range() {
            let trs = fst.get_trs(state).expect("valid state");
            if let Some(mut next) = epsilons[state] {
                if trs.iter().any(|tr| tr.ilabel != EPS_LABEL) {
                    loop {
                        let next_trs = fst.get_trs(next).expect("valid state");
                        if next_trs.iter().any(|tr| tr.ilabel != EPS_LABEL) {
                            return Err(PyValueError::new_err(format!(
                                "epsilon branch from state {state} can consume input"
                            )));
                        }
                        if fst.is_final(next).expect("valid state") {
                            break;
                        }
                        next = epsilons[next].ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "epsilon branch from state {state} is not terminal"
                            ))
                        })?;
                    }
                }
            }
        }
        let prefixes = Self::compute_prefixes(&fst);
        if prefixes[start].is_none() {
            return Err(PyValueError::new_err(
                "start state has no accepting continuation",
            ));
        }
        Ok(Self {
            fst,
            start,
            state_names: (0..count).collect(),
            prefixes,
        })
    }

    fn compute_prefixes(fst: &ByteFst) -> Vec<Option<Vec<u32>>> {
        let count = fst.num_states();
        let mut reverse = vec![Vec::new(); count];
        let mut distance = vec![None; count];
        let mut queue = VecDeque::new();
        for state in fst.states_range() {
            if fst.is_final(state).expect("valid state") {
                distance[state] = Some(0usize);
                queue.push_back(state);
            }
            for tr in fst.get_trs(state).expect("valid state").iter() {
                reverse[tr.nextstate].push(state);
            }
        }
        while let Some(state) = queue.pop_front() {
            let next_distance = distance[state].unwrap() + 1;
            for &parent in &reverse[state] {
                if distance[parent].is_none() {
                    distance[parent] = Some(next_distance);
                    queue.push_back(parent);
                }
            }
        }

        let mut ordered: Vec<_> = fst
            .states_range()
            .filter_map(|state| distance[state].map(|d| (state, d)))
            .collect();
        ordered.sort_unstable_by_key(|&(state, d)| (d, state));
        let mut prefixes: Vec<Option<Vec<u32>>> = vec![None; count];
        for &(state, d) in &ordered {
            let candidate = if d == 0 {
                Vec::new()
            } else {
                let trs = fst.get_trs(state).expect("valid state");
                let tr = trs
                    .iter()
                    .find(|tr| distance[tr.nextstate] == Some(d - 1))
                    .expect("productive state has a shorter path to a final state");
                arc_tokens(tr, prefixes[tr.nextstate].as_ref().unwrap())
            };
            prefixes[state] = Some(candidate);
        }

        // The candidate starts as the output of one accepting path. Intersect
        // it with every possible exit until cyclic constraints reach a fixed
        // point. Each update only removes tokens, so this terminates.
        loop {
            let mut changed = false;
            for &(state, _) in &ordered {
                let current = prefixes[state].as_ref().unwrap();
                let mut length = current.len();
                if fst.is_final(state).expect("valid state") {
                    length = 0;
                }
                for tr in fst.get_trs(state).expect("valid state").iter() {
                    if let Some(child) = &prefixes[tr.nextstate] {
                        length = length.min(shared_length(current, tr, child));
                    }
                }
                if length < current.len() {
                    prefixes[state].as_mut().unwrap().truncate(length);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        prefixes
    }

    fn name(&self, state: StateId) -> usize {
        self.state_names.get(state).copied().unwrap_or(state)
    }

    // Epsilon-only states are forced; a state with byte arcs may instead
    // have an optional epsilon branch for its end-of-input output.
    fn follow_forced(&self, mut state: StateId, output: &mut Vec<u32>) -> StateId {
        loop {
            if self.fst.is_final(state).expect("valid state") {
                break;
            }
            let trs = self.fst.get_trs(state).expect("valid state");
            if trs.iter().any(|tr| tr.ilabel != EPS_LABEL) {
                break;
            }
            let Some(tr) = trs.iter().find(|tr| tr.ilabel == EPS_LABEL) else {
                break;
            };
            if tr.olabel != EPS_LABEL {
                output.push((tr.olabel - 1) as u32);
            }
            state = tr.nextstate;
        }
        state
    }

    fn transition(&self, state: StateId, byte: u8) -> PyResult<(StateId, Vec<u32>)> {
        let mut output = Vec::new();
        let current = self.follow_forced(state, &mut output);
        if self.prefixes[current].is_none() {
            return Err(PyValueError::new_err(format!(
                "no accepting continuation from state {}",
                self.name(state)
            )));
        }
        let trs = self.fst.get_trs(current).expect("valid state");
        let tr = trs
            .iter()
            .find(|tr| tr.ilabel == Label::from(byte) + 1)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "no accepting transition on byte {byte} from state {}",
                    self.name(state)
                ))
            })?;
        if tr.olabel != EPS_LABEL {
            output.push((tr.olabel - 1) as u32);
        }
        let destination = self.follow_forced(tr.nextstate, &mut output);
        if self.prefixes[destination].is_none() {
            return Err(PyValueError::new_err(format!(
                "no accepting continuation after byte {byte} from state {}",
                self.name(state)
            )));
        }
        Ok((destination, output))
    }

    fn final_output(&self, state: StateId) -> PyResult<Vec<u32>> {
        let mut current = state;
        let mut output = Vec::new();
        while !self.fst.is_final(current).expect("valid state") {
            let trs = self.fst.get_trs(current).expect("valid state");
            let tr = trs
                .iter()
                .find(|tr| tr.ilabel == EPS_LABEL)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("state {} is not accepting", self.name(state)))
                })?;
            if tr.olabel != EPS_LABEL {
                output.push((tr.olabel - 1) as u32);
            }
            current = tr.nextstate;
        }
        Ok(output)
    }
}

/// An immutable deterministic transducer over input bytes and token-ID outputs.
#[pyclass(name = "FST", module = "push2tok._native")]
struct Fst {
    machine: Arc<Machine>,
}

#[pymethods]
impl Fst {
    #[new]
    #[pyo3(signature = (transitions, finals, start_state=0))]
    fn new(
        transitions: Vec<(usize, u8, usize, Vec<u32>)>,
        finals: HashMap<usize, Vec<u32>>,
        start_state: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            machine: Arc::new(Machine::new(transitions, finals, start_state)?),
        })
    }

    /// Load a rustfst/OpenFst VectorFst, ConstFst, or AT&T text FST.
    #[classmethod]
    #[pyo3(signature = (path, *, format="binary"))]
    fn from_file(
        _cls: &Bound<'_, PyType>,
        path: &Bound<'_, PyAny>,
        format: &str,
    ) -> PyResult<Self> {
        if !matches!(format, "binary" | "const" | "text") {
            return Err(PyValueError::new_err(
                "format must be 'binary', 'const', or 'text'",
            ));
        }
        let path: String = path
            .py()
            .import("os")?
            .getattr("fspath")?
            .call1((path,))?
            .extract()?;
        std::fs::metadata(&path).map_err(|err| PyOSError::new_err(err.to_string()))?;
        let fst = match format {
            "binary" => ByteFst::read(&path),
            "const" => ByteFst::read_from_const(&path),
            "text" => ByteFst::read_text(&path),
            _ => unreachable!(),
        }
        .map_err(|err| PyValueError::new_err(format!("invalid FST file: {err}")))?;
        Ok(Self {
            machine: Arc::new(Machine::from_loaded(fst)?),
        })
    }

    /// Save the rustfst graph as VectorFst/ConstFst binary or AT&T text.
    #[pyo3(signature = (path, *, format="binary"))]
    fn write(&self, path: &Bound<'_, PyAny>, format: &str) -> PyResult<()> {
        if !matches!(format, "binary" | "const" | "text") {
            return Err(PyValueError::new_err(
                "format must be 'binary', 'const', or 'text'",
            ));
        }
        let path: String = path
            .py()
            .import("os")?
            .getattr("fspath")?
            .call1((path,))?
            .extract()?;
        if format != "text"
            && self.machine.fst.states_range().any(|state| {
                self.machine
                    .fst
                    .get_trs(state)
                    .expect("valid state")
                    .iter()
                    .any(|tr| tr.ilabel > i32::MAX as Label || tr.olabel > i32::MAX as Label)
            })
        {
            return Err(PyValueError::new_err(
                "binary FST labels must fit in signed 32-bit integers; use format='text'",
            ));
        }
        let result = match format {
            "binary" => self.machine.fst.write(&path),
            "const" => ConstFst::from(self.machine.fst.clone()).write(&path),
            "text" => self.machine.fst.write_text(&path),
            _ => unreachable!(),
        };
        result.map_err(|err| PyOSError::new_err(err.to_string()))
    }

    fn stream(&self) -> Stream {
        Stream::new(self.machine.clone())
    }

    fn transduce(&self, data: &Bound<'_, PyBytes>) -> PyResult<Vec<u32>> {
        let mut stream = self.stream();
        let mut output = stream.push(data)?;
        output.extend(stream.finish()?);
        Ok(output)
    }
}

/// One independent, resettable input stream for an FST.
#[pyclass(module = "push2tok._native")]
struct Stream {
    machine: Arc<Machine>,
    state: StateId,
    begun: bool,
    finished: bool,
}

impl Stream {
    fn new(machine: Arc<Machine>) -> Self {
        Self {
            state: machine.start,
            machine,
            begun: false,
            finished: false,
        }
    }
}

#[pymethods]
impl Stream {
    fn push(&mut self, data: &Bound<'_, PyBytes>) -> PyResult<Vec<u32>> {
        if self.finished {
            return Err(PyRuntimeError::new_err("stream is finished; call reset()"));
        }
        // Keep changes local so an invalid byte rolls back the entire chunk.
        let mut state = self.state;
        let mut output = if self.begun {
            Vec::new()
        } else {
            self.machine.prefixes[state].as_ref().unwrap().clone()
        };
        for &byte in data.as_bytes() {
            let (destination, arc_output) = self.machine.transition(state, byte)?;
            let already_emitted = self.machine.prefixes[state].as_ref().unwrap();
            let destination_prefix = self.machine.prefixes[destination].as_ref().unwrap();
            let possible_output = arc_output.iter().chain(destination_prefix);
            debug_assert!(possible_output
                .clone()
                .take(already_emitted.len())
                .eq(already_emitted));
            output.extend(possible_output.skip(already_emitted.len()).copied());
            state = destination;
        }
        self.state = state;
        self.begun = true;
        Ok(output)
    }

    fn finish(&mut self) -> PyResult<Vec<u32>> {
        if self.finished {
            return Err(PyRuntimeError::new_err("stream is finished; call reset()"));
        }
        let final_output = self.machine.final_output(self.state)?;
        let output = if self.begun {
            let credit = self.machine.prefixes[self.state].as_ref().unwrap();
            debug_assert!(final_output.starts_with(credit));
            final_output[credit.len()..].to_vec()
        } else {
            final_output
        };
        self.finished = true;
        Ok(output)
    }

    fn reset(&mut self) {
        self.state = self.machine.start;
        self.begun = false;
        self.finished = false;
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Fst>()?;
    m.add_class::<Stream>()?;
    m.add_class::<beam::BeamFst>()?;
    m.add_class::<beam::BeamStream>()?;
    Ok(())
}
