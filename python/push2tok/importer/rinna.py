#!/usr/bin/env python3
"""Compile rinna's SentencePiece Unigram *segmentation* to a weighted WFST.

The FST consumes UTF-8 bytes AFTER SentencePiece normalization and emits
piece IDs. Input and output labels are shifted by one (zero is epsilon).
To tokenize raw text, normalize it with the preserved .spiece.model first.
"""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from push2tok._native import convert_unigram_fst, encode_unigram_fst

REVISION = "8ce2399c33e99013a593ea9389378fd86662b9c7"
URL = f"https://huggingface.co/rinna/japanese-gpt2-medium/resolve/{REVISION}/spiece.model"
SHA256 = "b5cbdfa8aa7c54c8c5af85b78c309c54a5f2749a20468bf6f60eee007fe6fec1"


def utf8_next(state):
    """Valid UTF-8 for one scalar value, excluding overlongs and surrogates."""
    if state == "start":
        return ([(b, None) for b in range(128)]
                + [(b, "tail1") for b in range(0xC2, 0xE0)]
                + [(0xE0, "e0")]
                + [(b, "tail2") for b in (*range(0xE1, 0xED), *range(0xEE, 0xF0))]
                + [(0xED, "ed"), (0xF0, "f0")]
                + [(b, "tail3") for b in range(0xF1, 0xF4)]
                + [(0xF4, "f4")])
    first, last, following = {
        "tail1": (0x80, 0xC0, None),
        "tail2": (0x80, 0xC0, "tail1"),
        "tail3": (0x80, 0xC0, "tail2"),
        "e0": (0xA0, 0xC0, "tail1"),
        "ed": (0x80, 0xA0, "tail1"),
        "f0": (0x90, 0xC0, "tail2"),
        "f4": (0x80, 0x90, "tail2"),
    }[state]
    return [(b, following) for b in range(first, last)]


def load_model(path):
    model = pb.ModelProto()
    model.ParseFromString(path.read_bytes())
    if model.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM:
        raise ValueError("model_type must be UNIGRAM (this is not a BPE compiler)")
    if model.trainer_spec.byte_fallback:
        raise ValueError("byte_fallback is not implemented")
    if any(piece.type not in (pb.ModelProto.SentencePiece.NORMAL,
                              pb.ModelProto.SentencePiece.CONTROL,
                              pb.ModelProto.SentencePiece.UNKNOWN)
           for piece in model.pieces):
        raise ValueError("user-defined/byte/unused symbols require additional handling")
    if len([p for p in model.pieces if p.type == pb.ModelProto.SentencePiece.UNKNOWN]) != 1:
        raise ValueError("expected exactly one unknown piece")
    return model


class Writer:
    def __init__(self, stream):
        self.stream = stream
        self.next_state = 2  # 0: normal, 1: within a run of unknown characters
        self.arcs = 0

    def state(self):
        state = self.next_state
        self.next_state += 1
        return state

    def arc(self, source, destination, byte, token, cost=0.0):
        # Labels 0 are epsilon. A zero-valued token ID is encoded as label 1.
        ilabel = byte + 1
        olabel = 0 if token is None else token + 1
        self.stream.write(f"{source}\t{destination}\t{ilabel}\t{olabel}\t{cost:.9g}\n")
        self.arcs += 1


def emit_pieces(writer, pieces):
    """Shared prefix trie, with alternative 'complete piece' arcs."""
    prefix_states = {}
    singletons = set()
    normal = pb.ModelProto.SentencePiece.NORMAL
    for token, piece in enumerate(pieces):
        if piece.type != normal:
            continue
        data = piece.piece.encode("utf-8")
        if not data:
            raise ValueError("empty normal piece")
        if len(piece.piece) == 1:
            singletons.add(data)
        for length in range(1, len(data)):
            prefix = data[:length]
            if prefix not in prefix_states:
                destination = writer.state()
                prefix_states[prefix] = destination
                sources = (0, 1) if length == 1 else (prefix_states[prefix[:-1]],)
                for source in sources:
                    writer.arc(source, destination, data[length - 1], None)
        sources = (0, 1) if len(data) == 1 else (prefix_states[data[:-1]],)
        for source in sources:
            writer.arc(source, 0, data[-1], token, -piece.score)
    return singletons


def emit_unknowns(writer, singletons, unk_id, cost):
    """UTF-8 scalar DFA minus the vocabulary's one-character pieces.

    Only an unknown character without a single-character piece receives an
    UNK candidate in SentencePiece's lattice. A prefix trie of exceptions
    avoids enumerating the entire Unicode range in the FST.
    """
    exceptions = [{}]
    excluded = set()
    for data in singletons:
        node = 0
        for byte in data:
            node = exceptions[node].setdefault(byte, len(exceptions))
            if node == len(exceptions):
                exceptions.append({})
        excluded.add(node)

    queue = deque([(0, "start", 0, 0), (1, "start", 0, 1)])
    states = {(0, "start", 0): 0, (1, "start", 0): 1}

    def state_for(mode, valid, exception):
        key = (mode, valid, exception)
        if key not in states:
            states[key] = writer.state()
            queue.append((mode, valid, exception, states[key]))
        return states[key]

    while queue:
        mode, valid, exception, source = queue.popleft()
        for byte, following in utf8_next(valid):
            child = None if exception is None else exceptions[exception].get(byte)
            if following is None:
                if child not in excluded:
                    writer.arc(source, 1, byte, unk_id if mode == 0 else None, cost)
            else:
                writer.arc(source, state_for(mode, following, child), byte, None)


def compile_text(model, path):
    pieces = model.pieces
    normal = pb.ModelProto.SentencePiece.NORMAL
    unk_id = next(i for i, p in enumerate(pieces)
                  if p.type == pb.ModelProto.SentencePiece.UNKNOWN)
    minimum = min(p.score for p in pieces if p.type == normal)
    # SentencePiece's kUnkPenalty = 10, and min_score is over normal pieces.
    unknown_cost = -(minimum - 10.0)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = Writer(stream)
        singletons = emit_pieces(writer, pieces)
        emit_unknowns(writer, singletons, unk_id, unknown_cost)
        writer.stream.write("0\n1\n")  # empty input and an unknown run
    return writer.next_state, writer.arcs


def build(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        if args.model is None:
            model_path = Path(temp) / "spiece.model"
            with urlopen(URL, timeout=120) as response, model_path.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != SHA256:
                raise ValueError("downloaded model has an unexpected SHA-256")
        else:
            model_path = args.model.resolve()
        model = load_model(model_path)
        push_labels = args.push_labels
        text_path = output if args.format == "text" and not push_labels else Path(temp) / "model.txt"
        states, arcs = compile_text(model, text_path)
        if args.format == "binary" or push_labels:
            states, arcs = convert_unigram_fst(
                text_path, output, push_labels=push_labels, format=args.format)
        sidecar = Path(str(output) + ".spiece.model")
        if model_path != sidecar:
            shutil.copyfile(model_path, sidecar)
        metadata = {
            "source": "rinna/japanese-gpt2-medium" if args.model is None else str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "format": args.format,
            "push_labels": push_labels,
            "input": "UTF-8 bytes AFTER SentencePieceProcessor.normalize(raw_text)",
            "input_label": "byte + 1 (0 is epsilon)",
            "output_label": "piece_id + 1 (0 is epsilon)",
            "weight": "tropical cost = -piece_score, unknown = -(min_normal_score - 10)",
            "states": states,
            "arcs": arcs,
            "model_file": sidecar.name,
        }
        Path(str(output) + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({states} states, {arcs} arcs); normalizer: {sidecar}")
    if args.verify and args.format == "binary":
        sp = spm.SentencePieceProcessor(model_file=str(sidecar))
        got = encode_unigram_fst(output, sp.normalize(args.sample))
        want = sp.encode(args.sample)
        print("text         :", args.sample)
        print("sentencepiece:", want)
        print("fst          :", got)
        if got != want:
            raise ValueError("verification failed: outputs differ")
        print("verification OK")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="local spiece.model (defaults to pinned rinna download)")
    parser.add_argument("output", nargs="?", type=Path, help="output FST path")
    parser.add_argument("--output", dest="output_option", type=Path, help="output FST path")
    parser.add_argument("--format", choices=("binary", "text"), default="binary")
    parser.add_argument("--push-labels", action=argparse.BooleanOptionalAction, default=True,
                        help="push output labels toward the start state (default: enabled)")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                        help="compare the binary FST with SentencePiece (default: enabled)")
    parser.add_argument("--sample", default="吾輩は猫である。名前はまだ無い。",
                        help="text used to verify the binary FST")
    args = parser.parse_args(argv)
    if args.output and args.output_option:
        parser.error("specify either output or --output, not both")
    args.output = args.output_option or args.output or Path(
        "dist/rinna-japanese-gpt2-medium.unigram.fst")
    build(args)


if __name__ == "__main__":
    main()
