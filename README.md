# push2tok

`push2tok` is a Python package for incremental byte-to-token-ID transduction.

## How to use

```
# Download and convert Rinna unigram SentencePiece into a (non-deterministic) FST.
./tools/convert_rinna.sh

# Test run
echo "札幌から東京まで時速60キロで何時間かかる？" | \
  push2tok --fst dist/rinna-japanese-gpt2-medium.unigram.fst --mode beam --beam-width 5
```

From Python:

```
>>> import push2tok
>>> stream = push2tok.BeamFST.from_file("dist/rinna-japanese-gpt2-medium.unigram.fst").stream(5)

>>> stream.push("札幌から東京まで時速60キロで何時間かかる？".encode("utf-8"))
[3401, 28, 238, 109, 23490, 630, 4659, 19, 1059, 364, 9525, 0]

>>> stream.push("札".encode("utf-8"))
[]
>>> stream.push("幌から東".encode("utf-8"))
[3401]
>>> stream.push("京まで時".encode("utf-8"))
[28, 238, 109]
>>> stream.push("速60キ".encode("utf-8"))
[23490, 630]
>>> stream.push("ロで何時".encode("utf-8"))
[4659, 19, 1059]
>>> stream.push("間かかる？".encode("utf-8"))
[364, 9525, 0]
```
