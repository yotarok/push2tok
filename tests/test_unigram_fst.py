"""Small offline tests for the SentencePiece -> weighted FST builder."""

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
except ImportError:
    spm = None
    pb = None

if spm is not None:
    from push2tok import BeamFST
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from build_unigram_fst import compile_text, load_model


@unittest.skipIf(spm is None, "install sentencepiece and protobuf for generator tests")
class UnigramBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        model = pb.ModelProto()
        model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
        model.trainer_spec.unk_id = 0
        model.normalizer_spec.name = "identity"
        model.normalizer_spec.add_dummy_prefix = False
        model.normalizer_spec.remove_extra_whitespaces = False
        model.normalizer_spec.escape_whitespaces = False
        for text, score, kind in [
            ("<unk>", 0, pb.ModelProto.SentencePiece.UNKNOWN),
            ("a", -0.1, pb.ModelProto.SentencePiece.NORMAL),
            ("b", -0.2, pb.ModelProto.SentencePiece.NORMAL),
            ("ab", -0.05, pb.ModelProto.SentencePiece.NORMAL),
            ("あ", -0.3, pb.ModelProto.SentencePiece.NORMAL),
        ]:
            piece = model.pieces.add()
            piece.piece, piece.score, piece.type = text, score, kind
        self.model_path = self.directory / "small.model"
        self.model_path.write_bytes(model.SerializeToString())
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    def test_all_known_and_unknown_segmentations(self):
        fst_path = self.directory / "small.txt"
        states, count = compile_text(load_model(self.model_path), fst_path)
        self.assertGreater(states, 10)
        self.assertGreater(count, 100)
        arcs = defaultdict(lambda: defaultdict(list))
        finals = set()
        for line in fst_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 1:
                finals.add(int(parts[0]))
            else:
                source, target, ilabel, olabel, weight = parts
                arcs[int(source)][int(ilabel)].append(
                    (int(target), int(olabel), float(weight)))
        self.assertEqual(finals, {0, 1})

        def best(text):
            states = {0: (0.0, [])}
            for byte in self.processor.normalize(text).encode("utf-8"):
                updated = {}
                for state, (cost, tokens) in states.items():
                    for target, label, weight in arcs[state][byte + 1]:
                        output = tokens + ([label - 1] if label else [])
                        candidate = (cost + weight, output)
                        if target not in updated or candidate[0] < updated[target][0]:
                            updated[target] = candidate
                states = updated
            return min((cost, tokens) for state, (cost, tokens) in states.items()
                       if state in finals)[1]

        samples = ["", "a", "b", "ab", "aba", "x", "xy", "axyb", "あ", "あx", "😀😀", "a😀b"]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(best(text), self.processor.encode(text))
        # SentencePiece never adds an unknown alternative when a single
        # character piece exists; both paths must be present for 'ab'.
        self.assertEqual([label for _, label, _ in arcs[0][ord("a") + 1]
                          if label != 0], [2])
        self.assertGreater(len(arcs[0][ord("a") + 1]), 1)

    def test_rejects_bpe(self):
        model = pb.ModelProto()
        model.trainer_spec.model_type = pb.TrainerSpec.BPE
        path = self.directory / "bpe.model"
        path.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ValueError, "UNIGRAM"):
            load_model(path)

    def test_streaming_beam_matches_small_sentencepiece_model(self):
        text = self.directory / "model.txt"
        compile_text(load_model(self.model_path), text)
        fst = BeamFST.from_file(text, format="text")
        for sample in ("", "a", "ab", "aba", "axyb", "あ😀😀b"):
            data = self.processor.normalize(sample).encode("utf-8")
            with self.subTest(sample=sample):
                expected = self.processor.encode(sample)
                self.assertEqual(fst.transduce(data, beam_width=30), expected)
                for split in range(len(data) + 1):
                    stream = fst.stream(beam_width=30)
                    output = stream.push(data[:split])
                    output += stream.push(data[split:])
                    output += stream.finish()
                    self.assertEqual(output, expected)

    @unittest.skipUnless(os.environ.get("PUSH2TOK_TEST_BINARY"),
                         "set PUSH2TOK_TEST_BINARY=1 for the rustfst integration test")
    def test_rustfst_binary_shortest_path(self):
        root = Path(__file__).resolve().parent.parent
        cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
        if not Path(cargo).exists():
            self.skipTest("cargo is required")
        env = os.environ.copy()
        env.setdefault("CARGO_HOME", str(root / ".cargo-home"))
        text = self.directory / "model.txt"
        binary = self.directory / "model.fst"
        pushed = self.directory / "pushed.fst"
        pushed_text = self.directory / "pushed.txt"
        compile_text(load_model(self.model_path), text)
        command = [cargo, "run", "--quiet", "--bin", "unigram_fst", "--"]
        subprocess.run(command + ["convert", str(text), str(binary)],
                       cwd=root, env=env, check=True)
        subprocess.run(command + ["convert", str(text), str(pushed), "--push-labels"],
                       cwd=root, env=env, check=True)
        subprocess.run(command + ["convert", str(text), str(pushed_text),
                                  "--push-labels", "--output-format", "text"],
                       cwd=root, env=env, check=True)
        pushed_fst = BeamFST.from_file(pushed)
        pushed_text_fst = BeamFST.from_file(pushed_text, format="text")
        for sample in ("", "ab", "axyb", "あ😀😀b"):
            with self.subTest(sample=sample):
                normalized = self.processor.normalize(sample)
                expected = self.processor.encode(sample)
                for model_file in (binary, pushed):
                    result = subprocess.run(command + ["encode", str(model_file), "-"],
                                            input=normalized, cwd=root, env=env,
                                            text=True, capture_output=True, check=True)
                    self.assertEqual(json.loads(result.stdout), expected)
                self.assertEqual(pushed_fst.transduce(normalized.encode(), beam_width=30), expected)
                self.assertEqual(pushed_text_fst.transduce(normalized.encode(), beam_width=30), expected)

    @unittest.skipUnless(os.environ.get("PUSH2TOK_TEST_BINARY"),
                         "set PUSH2TOK_TEST_BINARY=1 for the builder/wrapper integration test")
    def test_builder_and_wrapper_enable_label_pushing(self):
        root = Path(__file__).resolve().parent.parent
        cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
        if not Path(cargo).exists():
            self.skipTest("cargo is required")
        env = os.environ.copy()
        env.setdefault("CARGO_HOME", str(root / ".cargo-home"))
        env["PATH"] = str(Path(cargo).parent) + os.pathsep + env.get("PATH", "")
        text = self.directory / "builder-pushed.txt"
        subprocess.run(
            [sys.executable, str(root / "tools/build_unigram_fst.py"),
             "--model", str(self.model_path), "--output", str(text),
             "--format", "text", "--push-labels"],
            cwd=root, env=env, capture_output=True, text=True, check=True,
        )
        self.assertTrue(json.loads(Path(str(text) + ".json").read_text())["push_labels"])
        self.assertEqual(BeamFST.from_file(text, format="text").transduce(b"ab"),
                         self.processor.encode("ab"))

        binary = self.directory / "wrapper-pushed.fst"
        env.update(MODEL=str(self.model_path), VENV=sys.prefix, SAMPLE="ab", VERIFY="1")
        env.pop("PUSH_LABELS", None)
        subprocess.run(
            [str(root / "tools/convert_rinna.sh"), str(binary)],
            cwd=root, env=env, capture_output=True, text=True, check=True,
        )
        self.assertTrue(json.loads(Path(str(binary) + ".json").read_text())["push_labels"])
        self.assertEqual(BeamFST.from_file(binary).transduce(b"ab"),
                         self.processor.encode("ab"))


if __name__ == "__main__":
    unittest.main()
