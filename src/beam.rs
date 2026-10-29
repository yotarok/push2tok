//! Beam-limited streaming decode for weighted, non-deterministic byte FSTs.
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::{CoreFst, ExpandedFst, SerializableFst};
use rustfst::semirings::{Semiring, TropicalWeight};
use rustfst::{Label, StateId, EPS_LABEL};

type WeightedFst = VectorFst<TropicalWeight>;

#[derive(Clone, Copy)]
struct Trace {
    parent: usize,
    token: u32,
    depth: usize,
}

#[derive(Clone, Copy)]
struct Hypothesis {
    state: StateId,
    cost: f32,
    trace: usize,
}

#[derive(Clone)]
struct BeamState {
    traces: Vec<Trace>,
    intern: HashMap<(usize, u32), usize>,
    active: Vec<Hypothesis>,
    // Best end-of-input path is retained even if it falls outside the beam.
    // Otherwise a push could emit a prefix incompatible with finish().
    eos: Option<Hypothesis>,
}

impl BeamState {
    fn empty() -> Self {
        Self {
            traces: vec![Trace {
                parent: 0,
                token: 0,
                depth: 0,
            }],
            intern: HashMap::new(),
            active: Vec::new(),
            eos: None,
        }
    }

    fn append(&mut self, parent: usize, token: u32) -> usize {
        if let Some(&id) = self.intern.get(&(parent, token)) {
            return id;
        }
        let id = self.traces.len();
        self.traces.push(Trace {
            parent,
            token,
            depth: self.traces[parent].depth + 1,
        });
        self.intern.insert((parent, token), id);
        id
    }

    fn lca(&self, mut left: usize, mut right: usize) -> usize {
        while self.traces[left].depth > self.traces[right].depth {
            left = self.traces[left].parent;
        }
        while self.traces[right].depth > self.traces[left].depth {
            right = self.traces[right].parent;
        }
        while left != right {
            left = self.traces[left].parent;
            right = self.traces[right].parent;
        }
        left
    }

    fn path(&self, mut id: usize, stop: usize) -> Vec<u32> {
        let mut tokens = Vec::new();
        while id != stop {
            tokens.push(self.traces[id].token);
            id = self.traces[id].parent;
        }
        tokens.reverse();
        tokens
    }

    // Emit the shared output prefix and rebuild only reachable suffix traces.
    // Old frames and discarded hypotheses are freed every input byte.
    fn commit_and_compact(&mut self) -> Vec<u32> {
        let mut ancestors = self.active.iter().map(|hyp| hyp.trace);
        let first = ancestors.next().or_else(|| self.eos.map(|hyp| hyp.trace));
        let Some(mut common) = first else {
            return Vec::new();
        };
        for trace in ancestors {
            common = self.lca(common, trace);
        }
        if let Some(eos) = self.eos {
            common = self.lca(common, eos.trace);
        }
        let committed = self.path(common, 0);
        let old = std::mem::replace(self, BeamState::empty());
        for hyp in &old.active {
            let mut retained = *hyp;
            retained.trace = 0;
            for token in old.path(hyp.trace, common) {
                retained.trace = self.append(retained.trace, token);
            }
            self.active.push(retained);
        }
        if let Some(hyp) = old.eos {
            let mut retained = hyp;
            retained.trace = 0;
            for token in old.path(hyp.trace, common) {
                retained.trace = self.append(retained.trace, token);
            }
            self.eos = Some(retained);
        }
        committed
    }
}

// Smallest rank first. Sequence number makes equal-cost choices reproducible.
struct QueueEntry {
    rank: f32,
    order: usize,
    hyp: Hypothesis,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.rank.total_cmp(&other.rank) == Ordering::Equal && self.order == other.order
    }
}
impl Eq for QueueEntry {}
impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .total_cmp(&self.rank)
            .then_with(|| other.order.cmp(&self.order))
    }
}

struct BeamMachine {
    fst: WeightedFst,
    start: StateId,
    // Optimistic cost of reaching a final state (through any future bytes).
    // This prevents unfinished zero-cost piece prefixes from crowding out
    // completed tokens whose cost is paid at the last byte.
    remaining: Vec<f32>,
}

impl BeamMachine {
    fn from_fst(fst: WeightedFst) -> PyResult<Self> {
        let n = fst.num_states();
        let start = fst
            .start()
            .filter(|&state| state < n)
            .ok_or_else(|| PyValueError::new_err("FST has no valid start state"))?;
        let mut reverse = vec![Vec::<(StateId, f32)>::new(); n];
        let mut epsilon_indegree = vec![0usize; n];
        let mut final_costs = vec![f32::INFINITY; n];
        for state in fst.states_range() {
            if let Some(weight) = fst.final_weight(state).expect("valid state") {
                let value = *weight.value();
                if !value.is_finite() || value < 0.0 {
                    return Err(PyValueError::new_err(
                        "final weights must be finite and nonnegative",
                    ));
                }
                final_costs[state] = value;
            }
            for tr in fst.get_trs(state).expect("valid state").iter() {
                if tr.nextstate >= n || tr.ilabel > 256 || (tr.olabel as u64) > u32::MAX as u64 + 1
                {
                    return Err(PyValueError::new_err(format!(
                        "invalid byte/output label or destination at state {state}"
                    )));
                }
                let cost = *tr.weight.value();
                if !cost.is_finite() || cost < 0.0 {
                    return Err(PyValueError::new_err(
                        "arc weights must be finite and nonnegative",
                    ));
                }
                reverse[tr.nextstate].push((state, cost));
                if tr.ilabel == EPS_LABEL {
                    epsilon_indegree[tr.nextstate] += 1;
                }
            }
        }
        // Epsilon paths need finite closure. Input-consuming cycles are fine.
        let mut roots: VecDeque<_> = fst
            .states_range()
            .filter(|&state| epsilon_indegree[state] == 0)
            .collect();
        let mut visited = 0;
        while let Some(state) = roots.pop_front() {
            visited += 1;
            for tr in fst.get_trs(state).expect("valid state").iter() {
                if tr.ilabel == EPS_LABEL {
                    epsilon_indegree[tr.nextstate] -= 1;
                    if epsilon_indegree[tr.nextstate] == 0 {
                        roots.push_back(tr.nextstate);
                    }
                }
            }
        }
        if visited != n {
            return Err(PyValueError::new_err("input-epsilon cycle in FST"));
        }

        let mut remaining = final_costs;
        let mut heap = BinaryHeap::new();
        let mut order = 0;
        for state in fst.states_range() {
            if remaining[state].is_finite() {
                heap.push(QueueEntry {
                    rank: remaining[state],
                    order,
                    hyp: Hypothesis {
                        state,
                        cost: 0.0,
                        trace: 0,
                    },
                });
                order += 1;
            }
        }
        while let Some(entry) = heap.pop() {
            let state = entry.hyp.state;
            if entry.rank > remaining[state] {
                continue;
            }
            for &(parent, cost) in &reverse[state] {
                let candidate = entry.rank + cost;
                if candidate < remaining[parent] {
                    remaining[parent] = candidate;
                    heap.push(QueueEntry {
                        rank: candidate,
                        order,
                        hyp: Hypothesis {
                            state: parent,
                            cost: 0.0,
                            trace: 0,
                        },
                    });
                    order += 1;
                }
            }
        }
        if !remaining[start].is_finite() {
            return Err(PyValueError::new_err(
                "start state has no accepting continuation",
            ));
        }
        Ok(Self {
            fst,
            start,
            remaining,
        })
    }

    fn epsilon_close(&self, work: &mut BeamState, seeds: Vec<Hypothesis>, width: usize) {
        let mut frontier = BinaryHeap::new();
        let mut order = 0;
        for hyp in seeds {
            self.enqueue(&mut frontier, &mut order, hyp);
        }
        // Viterbi recombination: the future of a hypothesis depends only on
        // its FST state, so only the cheapest history per state survives.
        // The heap pops in rank order (cost + per-state constant), so the first
        // pop of each state is already its best; later ones are dropped.
        let mut seen: HashSet<StateId> = HashSet::new();
        let mut ready: Vec<(f32, usize, Hypothesis)> = Vec::new();
        let mut best_eos: Option<(f32, Hypothesis)> = None;
        while let Some(entry) = frontier.pop() {
            let hyp = entry.hyp;
            if !seen.insert(hyp.state) {
                continue;
            }
            if let Some(final_weight) = self.fst.final_weight(hyp.state).expect("valid state") {
                let total = hyp.cost + final_weight.value();
                if best_eos.is_none_or(|(best, _)| total < best) {
                    best_eos = Some((total, hyp));
                }
            }
            let trs = self.fst.get_trs(hyp.state).expect("valid state");
            if trs.iter().any(|tr| tr.ilabel != EPS_LABEL)
                || self.fst.is_final(hyp.state).expect("valid state")
            {
                ready.push((entry.rank, entry.order, hyp));
                ready.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                if ready.len() > width {
                    ready.pop();
                }
            }
            for tr in trs.iter().filter(|tr| tr.ilabel == EPS_LABEL) {
                let trace = if tr.olabel == EPS_LABEL {
                    hyp.trace
                } else {
                    work.append(hyp.trace, (tr.olabel - 1) as u32)
                };
                self.enqueue(
                    &mut frontier,
                    &mut order,
                    Hypothesis {
                        state: tr.nextstate,
                        cost: hyp.cost + tr.weight.value(),
                        trace,
                    },
                );
            }
            // An unexplored path cannot beat the current K survivors *or*
            // the best EOS path if its optimistic remaining cost is higher.
            if let (true, Some((eos_cost, _))) = (ready.len() == width, best_eos) {
                let cutoff = ready.last().unwrap().0.max(eos_cost);
                if frontier.peek().is_none_or(|next| next.rank > cutoff) {
                    break;
                }
            }
        }
        work.active = ready.into_iter().map(|(_, _, hyp)| hyp).collect();
        work.eos = best_eos.map(|(_, hyp)| hyp);
    }

    fn enqueue(&self, heap: &mut BinaryHeap<QueueEntry>, order: &mut usize, hyp: Hypothesis) {
        let rank = hyp.cost + self.remaining[hyp.state];
        if rank.is_finite() {
            heap.push(QueueEntry {
                rank,
                order: *order,
                hyp,
            });
            *order += 1;
        }
    }

    fn step(&self, work: &mut BeamState, byte: u8, width: usize) -> PyResult<Vec<u32>> {
        let mut seeds = Vec::new();
        let input = Label::from(byte) + 1;
        let active = work.active.clone();
        for hyp in &active {
            for tr in self.fst.get_trs(hyp.state).expect("valid state").iter() {
                if tr.ilabel != input {
                    continue;
                }
                let trace = if tr.olabel == EPS_LABEL {
                    hyp.trace
                } else {
                    work.append(hyp.trace, (tr.olabel - 1) as u32)
                };
                seeds.push(Hypothesis {
                    state: tr.nextstate,
                    cost: hyp.cost + tr.weight.value(),
                    trace,
                });
            }
        }
        self.epsilon_close(work, seeds, width);
        if work.active.is_empty() && work.eos.is_none() {
            return Err(PyValueError::new_err(format!(
                "beam exhausted after input byte {byte}; try a larger beam_width"
            )));
        }
        Ok(work.commit_and_compact())
    }

    fn initial(&self, width: usize) -> BeamState {
        let mut work = BeamState::empty();
        self.epsilon_close(
            &mut work,
            vec![Hypothesis {
                state: self.start,
                cost: 0.0,
                trace: 0,
            }],
            width,
        );
        work
    }
}

/// Immutable weighted, possibly nondeterministic byte-to-token FST.
#[pyclass(name = "BeamFST", module = "push2tok._native")]
pub struct BeamFst {
    machine: Arc<BeamMachine>,
}

#[pymethods]
impl BeamFst {
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
            "binary" => WeightedFst::read(&path),
            "const" => WeightedFst::read_from_const(&path),
            "text" => WeightedFst::read_text(&path),
            _ => unreachable!(),
        }
        .map_err(|err| PyValueError::new_err(format!("invalid FST file: {err}")))?;
        Ok(Self {
            machine: Arc::new(BeamMachine::from_fst(fst)?),
        })
    }

    #[pyo3(signature = (beam_width=30))]
    fn stream(&self, beam_width: usize) -> PyResult<BeamStream> {
        if beam_width == 0 {
            return Err(PyValueError::new_err("beam_width must be positive"));
        }
        let work = self.machine.initial(beam_width);
        Ok(BeamStream {
            machine: self.machine.clone(),
            width: beam_width,
            work,
            finished: false,
        })
    }

    #[pyo3(signature = (data, *, beam_width=30))]
    fn transduce(&self, data: &Bound<'_, PyBytes>, beam_width: usize) -> PyResult<Vec<u32>> {
        let mut stream = self.stream(beam_width)?;
        let mut output = stream.push(data)?;
        output.extend(stream.finish()?);
        Ok(output)
    }
}

/// Independent, resettable beam-search decoding stream.
#[pyclass(name = "BeamStream", module = "push2tok._native")]
pub struct BeamStream {
    machine: Arc<BeamMachine>,
    width: usize,
    work: BeamState,
    finished: bool,
}

#[pymethods]
impl BeamStream {
    #[getter]
    fn pending_nodes(&self) -> usize {
        // The root is not an output node.
        self.work.traces.len() - 1
    }

    fn push(&mut self, data: &Bound<'_, PyBytes>) -> PyResult<Vec<u32>> {
        if self.finished {
            return Err(PyRuntimeError::new_err("stream is finished; call reset()"));
        }
        // Work on a copy: invalid input never consumes a prefix of the chunk.
        let mut work = self.work.clone();
        let mut output = work.commit_and_compact();
        for &byte in data.as_bytes() {
            output.extend(self.machine.step(&mut work, byte, self.width)?);
        }
        self.work = work;
        Ok(output)
    }

    fn finish(&mut self) -> PyResult<Vec<u32>> {
        if self.finished {
            return Err(PyRuntimeError::new_err("stream is finished; call reset()"));
        }
        let eos = self
            .work
            .eos
            .ok_or_else(|| PyValueError::new_err("no accepting hypothesis at end of input"))?;
        let output = self.work.path(eos.trace, 0);
        self.finished = true;
        Ok(output)
    }

    fn reset(&mut self) {
        self.work = self.machine.initial(self.width);
        self.finished = false;
    }
}
