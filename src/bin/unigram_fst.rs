//! Convert a tropical-weighted rustfst text FST (optionally pushing output
//! labels toward the start) and inspect its best output for a
//! *pre-normalized* UTF-8 input string.
use std::env;
use std::io::{self, Read};

use rustfst::algorithms::compose::compose;
use rustfst::algorithms::{push, shortest_path, PushType, ReweightType};
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::{CoreFst, ExpandedFst, SerializableFst};
use rustfst::semirings::{Semiring, TropicalWeight};
use rustfst::utils::acceptor;
use rustfst::Label;

type Fst = VectorFst<TropicalWeight>;

const USAGE: &str =
    "usage: unigram_fst convert <input.txt> <output> [--push-labels] [--output-format binary|text]
       unigram_fst encode <model.fst> <normalized-text|->   ('-' reads stdin)";

fn convert(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut positional = Vec::new();
    let mut push_labels = false;
    let mut text_output = false;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--push-labels" => push_labels = true,
            "--output-format" => match rest.next().map(String::as_str) {
                Some("binary") => text_output = false,
                Some("text") => text_output = true,
                _ => return Err("--output-format must be 'binary' or 'text'".into()),
            },
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}").into()),
            _ => positional.push(arg),
        }
    }
    let [source, target] = positional.as_slice() else {
        return Err(USAGE.into());
    };
    let mut fst = Fst::read_text(source)?;
    if push_labels {
        // Move each output label to the earliest arc where *every* accepting
        // continuation agrees on it. Only labels move: the RmWeight-based
        // label distance keeps arc weights where they are. Labels that end up
        // grouped on one arc are split into epsilon-input chains.
        fst = push(&fst, ReweightType::ReweightToInitial, PushType::PUSH_LABELS)?;
    }
    if text_output {
        fst.write_text(target)?;
    } else {
        fst.write(target)?;
    }
    let arcs: usize = fst
        .states_range()
        .map(|s| fst.num_trs(s).unwrap_or(0))
        .sum();
    println!("{{\"states\": {}, \"arcs\": {}}}", fst.num_states(), arcs);
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    match args.as_slice() {
        [_, command, rest @ ..] if command == "convert" => convert(rest)?,
        [_, command, path, normalized] if command == "encode" => {
            let fst = Fst::read(path)?;
            let normalized = if normalized == "-" {
                let mut text = String::new();
                io::stdin().read_to_string(&mut text)?;
                text
            } else {
                normalized.clone()
            };
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
                    output.push((tr.olabel - 1).to_string());
                }
                state = tr.nextstate;
            }
            println!("[{}]", output.join(","));
        }
        _ => {
            return Err(USAGE.into());
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("unigram_fst: {error}");
        std::process::exit(1);
    }
}
