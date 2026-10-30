//! Shared rustfst operations for the CLI and the Python importer.

use rustfst::algorithms::compose::compose;
use rustfst::algorithms::{push, shortest_path, PushType, ReweightType};
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::{CoreFst, ExpandedFst, SerializableFst};
use rustfst::semirings::{Semiring, TropicalWeight};
use rustfst::utils::acceptor;
use rustfst::Label;

type Fst = VectorFst<TropicalWeight>;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn convert(
    source: &str,
    target: &str,
    push_labels: bool,
    text_output: bool,
) -> Result<(usize, usize)> {
    let mut fst = Fst::read_text(source)?;
    if push_labels {
        // Move only output labels to the earliest common arc; leave weights intact.
        fst = push(&fst, ReweightType::ReweightToInitial, PushType::PUSH_LABELS)?;
    }
    if text_output {
        fst.write_text(target)?;
    } else {
        fst.write(target)?;
    }
    let arcs = fst
        .states_range()
        .map(|s| fst.num_trs(s).unwrap_or(0))
        .sum();
    Ok((fst.num_states(), arcs))
}

pub fn encode(path: &str, normalized: &str) -> Result<Vec<Label>> {
    let fst = Fst::read(path)?;
    let labels: Vec<Label> = normalized
        .bytes()
        .map(|byte| Label::from(byte) + 1)
        .collect();
    let input: Fst = acceptor(&labels, TropicalWeight::one());
    let lattice: Fst = compose(input, fst)?;
    let best: Fst = shortest_path(&lattice)?;
    let mut state = best.start().ok_or("no accepting path")?;
    let mut output = Vec::new();
    loop {
        let trs = best.get_trs(state)?;
        if trs.is_empty() {
            if !best.is_final(state)? {
                return Err("shortest path does not end in a final state".into());
            }
            break;
        }
        if trs.len() != 1 {
            return Err("shortest path is not linear".into());
        }
        let tr = &trs[0];
        if tr.olabel != 0 {
            output.push(tr.olabel - 1);
        }
        state = tr.nextstate;
    }
    Ok(output)
}
