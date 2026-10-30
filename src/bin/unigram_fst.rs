//! Standalone entry point for the shared unigram FST operations.
#[path = "../unigram.rs"]
mod unigram;

use std::env;
use std::io::{self, Read};

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
    let (states, arcs) = unigram::convert(source, target, push_labels, text_output)?;
    println!("{{\"states\": {states}, \"arcs\": {arcs}}}");
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    match args.as_slice() {
        [_, command, rest @ ..] if command == "convert" => convert(rest)?,
        [_, command, path, normalized] if command == "encode" => {
            let normalized = if normalized == "-" {
                let mut text = String::new();
                io::stdin().read_to_string(&mut text)?;
                text
            } else {
                normalized.clone()
            };
            let output = unigram::encode(path, &normalized)?;
            println!("{:?}", output);
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("unigram_fst: {error}");
        std::process::exit(1);
    }
}
