# push2tok

`push2tok` is a Python package for incremental byte-to-token-ID transduction.

## Install with uv

In an existing [uv](https://docs.astral.sh/uv/) project:

```sh
uv add push2tok
```

To start a new project first, run `uv init my-project && cd my-project`, then
`uv add push2tok`. The PyPI wheel includes the native extension; installing it
does not require Rust. The following commands use `uv run` in that project.

## Converting the rinna tokenizer

Convert the SentencePiece Unigram model of
[rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium)
into a weighted, non-deterministic FST:

```sh
uv run python -m push2tok.importer.rinna
```

To choose an output path and use a local model (no download):

```sh
uv run python -m push2tok.importer.rinna --model ./spiece.model --output ./model.fst
```

## Usage

The FST consumes the UTF-8 bytes of **normalized** text, so first normalize
the input with the accompanying `.spiece.model`. (For example, the full-width
`？` becomes `?`, and a leading `▁` is added.)

From the command line:

```sh
echo -n "▁札幌から東京まで時速60キロで何時間かかる?" | \
  uv run push2tok --fst "dist/rinna-japanese-gpt2-medium.unigram.fst" --mode beam --beam-width 5
```

From Python (`uv run python`):

```
>>> import sentencepiece as spm
>>> import push2tok
>>> path = "dist/rinna-japanese-gpt2-medium.unigram.fst"
>>> fst = push2tok.BeamFST.from_file(path)

>>> text = '▁札幌から東京まで時速60キロで何時間かかる?'
>>> fst.transduce(text.encode("utf-8"), beam_width=5)
[9, 3401, 28, 238, 109, 23490, 630, 4659, 19, 1059, 364, 9525, 3017]

>>> stream = fst.stream(5)
>>> stream.push("▁札".encode("utf-8"))
[9]
>>> stream.push("幌から東".encode("utf-8"))
[3401]
>>> stream.push("京まで時".encode("utf-8"))
[28, 238, 109]
>>> stream.push("速60キ".encode("utf-8"))
[23490, 630]
>>> stream.push("ロで何時".encode("utf-8"))
[4659, 19, 1059]
>>> stream.push("間かかる?".encode("utf-8"))
[364, 9525]
>>> stream.finish()
[3017]
```

The token IDs match `normalizer.encode(...)`. This is beam search: a beam width
that is too small can change the result.

