import random
import tempfile
import unittest
from pathlib import Path

from push2tok import FST


class StreamTests(unittest.TestCase):
    def test_common_prefix_appears_before_finalization(self):
        fst = FST(
            [(0, 97, 1, []), (1, 98, 2, []), (1, 99, 3, [])],
            {2: [42, 1], 3: [42, 2]},
        )
        stream = fst.stream()
        self.assertEqual(stream.push(b"a"), [42])
        self.assertEqual(stream.push(b""), [])
        self.assertEqual(stream.push(b"b"), [1])
        self.assertEqual(stream.finish(), [])
        self.assertEqual(fst.transduce(b"ac"), [42, 2])
        with self.assertRaises(RuntimeError):
            stream.push(b"x")
        with self.assertRaises(RuntimeError):
            stream.finish()
        stream.reset()
        self.assertEqual(stream.push(b"ac"), [42, 2])
        self.assertEqual(stream.finish(), [])

    def test_transition_outputs_and_accepting_state_with_children(self):
        fst = FST(
            [(0, 97, 1, [3]), (1, 98, 2, [4, 5])],
            {1: [4], 2: [6]},
        )
        stream = fst.stream()
        self.assertEqual(stream.push(b"a"), [3, 4])
        self.assertEqual(stream.finish(), [])
        stream.reset()
        self.assertEqual(stream.push(b"a"), [3, 4])
        self.assertEqual(stream.push(b"b"), [5, 6])
        self.assertEqual(stream.finish(), [])
        self.assertEqual(fst.transduce(b"ab"), [3, 4, 5, 6])

    def test_empty_input_and_initial_guaranteed_output(self):
        fst = FST([(0, 97, 1, [])], {0: [7], 1: [7, 8]})
        stream = fst.stream()
        self.assertEqual(stream.push(b""), [7])
        self.assertEqual(stream.push(b"a"), [8])
        self.assertEqual(stream.finish(), [])
        stream.reset()
        self.assertEqual(stream.finish(), [7])
        self.assertEqual(fst.transduce(b""), [7])

    def test_premature_finish_and_invalid_chunk_do_not_advance(self):
        fst = FST([(0, 97, 1, [1]), (1, 98, 2, [2])], {2: [3]})
        stream = fst.stream()
        with self.assertRaisesRegex(ValueError, "not accepting"):
            stream.finish()
        with self.assertRaises(ValueError):
            stream.push(b"ax")
        self.assertEqual(stream.push(b"a"), [1, 2, 3])
        self.assertEqual(stream.push(b"b"), [])
        self.assertEqual(stream.finish(), [])
        with self.assertRaises(ValueError):
            fst.transduce(b"a")

    def test_dead_branch_and_validation(self):
        fst = FST([(0, 97, 1, []), (0, 98, 2, [])], {1: [4]})
        stream = fst.stream()
        with self.assertRaisesRegex(ValueError, "no accepting continuation"):
            stream.push(b"b")
        self.assertEqual(stream.push(b"a"), [4])
        self.assertEqual(stream.finish(), [])
        with self.assertRaisesRegex(ValueError, "duplicate transition"):
            FST([(0, 97, 1, []), (0, 97, 2, [])], {1: [], 2: []})
        with self.assertRaisesRegex(ValueError, "start state"):
            FST([(0, 97, 1, [])], {2: []})
        with self.assertRaises((ValueError, OverflowError)):
            FST([(0, 256, 1, [])], {1: []})
        with self.assertRaises((ValueError, OverflowError)):
            FST([], {0: [-1]})

    def test_cycle_with_nonempty_outputs(self):
        # All accepted strings are a* b. Each a contributes [8], b contributes
        # [8, 9]. The leading [8] is already certain before any input arrives.
        fst = FST([(0, 97, 0, [8]), (0, 98, 1, [8])], {1: [9]})
        stream = fst.stream()
        self.assertEqual(stream.push(b""), [8])
        self.assertEqual(stream.push(b"a"), [8])
        self.assertEqual(stream.push(b"ab"), [8, 9])
        self.assertEqual(stream.finish(), [])
        self.assertEqual(fst.transduce(b"b"), [8, 9])

    def test_sparse_states_and_full_u32_token_ids(self):
        fst = FST([(100, 0, 999_999, [0, 2**32 - 1])],
                  {999_999: [1, 2**32 - 1]}, start_state=100)
        stream = fst.stream()
        self.assertEqual(stream.push(b"\0"), [0, 2**32 - 1, 1, 2**32 - 1])
        self.assertEqual(stream.finish(), [])

    def test_file_round_trip_binary_and_text(self):
        fst = FST([(0, 0, 1, [0, 7]), (1, 255, 2, [8])],
                  {1: [3, 4], 2: [5, 6]})
        with tempfile.TemporaryDirectory() as directory:
            for format in ("binary", "const", "text"):
                path = Path(directory) / f"model.{format}"
                fst.write(path, format=format)
                loaded = FST.from_file(path, format=format)
                for data in (b"\0", b"\0\xff"):
                    self.assertEqual(loaded.transduce(data), fst.transduce(data))
                    for split in range(len(data) + 1):
                        stream = loaded.stream()
                        original = fst.stream()
                        output = stream.push(data[:split])
                        self.assertEqual(output, original.push(data[:split]))
                        next_output = stream.push(data[split:])
                        self.assertEqual(next_output, original.push(data[split:]))
                        output += next_output
                        output += stream.finish()
                        self.assertEqual(output, fst.transduce(data))

    def test_external_text_fst_with_epsilon_paths(self):
        # A state with both byte arcs and a terminal-output epsilon branch.
        # Label 98 consumes b'a', label 99 b'b'; output 5 means token 4.
        text = "0\t1\t98\t0\n1\t2\t0\t5\n1\t3\t99\t4\n2\n3\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.txt"
            path.write_text(text)
            fst = FST.from_file(path, format="text")
            self.assertEqual(fst.transduce(b"a"), [4])
            self.assertEqual(fst.transduce(b"ab"), [3])
            stream = fst.stream()
            self.assertEqual(stream.push(b"a"), [])
            self.assertEqual(stream.push(b"b"), [3])
            self.assertEqual(stream.finish(), [])

    def test_external_text_fst_with_initial_epsilon_output(self):
        # 0 -> 1 emits token 5 without consuming a byte. From 1, b'a'
        # emits token 7; a following epsilon arc emits token 8.
        text = "0\t1\t0\t6\n1\t2\t98\t8\n2\t3\t0\t9\n3\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.txt"
            path.write_text(text)
            fst = FST.from_file(path, format="text")
            stream = fst.stream()
            self.assertEqual(stream.push(b""), [5, 7, 8])
            self.assertEqual(stream.push(b"a"), [])
            self.assertEqual(stream.finish(), [])
            self.assertEqual(fst.transduce(b"a"), [5, 7, 8])
            with self.assertRaises(ValueError):
                fst.transduce(b"")

    def test_large_output_label_text_round_trip(self):
        fst = FST([], {0: [2**32 - 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            fst.write(path, format="text")
            self.assertEqual(FST.from_file(path, format="text").transduce(b""),
                             [2**32 - 1])

    def test_file_validation_and_io_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.txt"
            with self.assertRaises(OSError):
                FST.from_file(path)
            with self.assertRaises(ValueError):
                FST.from_file(path, format="unknown")
            path.write_text("0\t1\t98\t1\n0\t2\t98\t2\n1\n2\n")
            with self.assertRaisesRegex(ValueError, "non-deterministic"):
                FST.from_file(path, format="text")
            path.write_text("0\t1\t0\t1\n1\t0\t0\t1\n0\t2\t98\t1\n2\n")
            with self.assertRaisesRegex(ValueError, "epsilon cycle"):
                FST.from_file(path, format="text")
            path.write_text("0\t1\t98\t1\t2.0\n1\n")
            with self.assertRaisesRegex(ValueError, "weight"):
                FST.from_file(path, format="text")
            path.write_text("0\t1\t0\t0\n0\t2\t98\t0\n1\t3\t99\t0\n2\n3\n")
            with self.assertRaisesRegex(ValueError, "epsilon branch"):
                FST.from_file(path, format="text")
            fst = FST([], {0: [2**32 - 1]})
            with self.assertRaisesRegex(ValueError, "signed 32-bit"):
                fst.write(Path(directory) / "too-large.fst")

    def test_random_acyclic_fsts_match_all_accepting_continuations(self):
        rng = random.Random(42)
        for _ in range(40):
            transitions = []
            finals = {0: [rng.randrange(3)]}
            for state in range(5):
                if state and rng.randrange(2):
                    finals[state] = [rng.randrange(3) for _ in range(rng.randrange(3))]
                for byte in (97, 98):
                    if rng.randrange(2):
                        transitions.append(
                            (state, byte, rng.randrange(state + 1, 6),
                             [rng.randrange(3) for _ in range(rng.randrange(3))])
                        )
            if rng.randrange(2):
                finals[5] = [rng.randrange(3)]
            fst = FST(transitions, finals)
            arcs = {(source, byte): (destination, output)
                    for source, byte, destination, output in transitions}

            def all_suffixes(state):
                result = [([], finals[state])] if state in finals else []
                for (source, byte), (destination, output) in arcs.items():
                    if source == state:
                        result.extend(([byte] + suffix, output + tokens)
                                      for suffix, tokens in all_suffixes(destination))
                return result

            accepted = [(bytes(data), output) for data, output in all_suffixes(0)]
            for data, expected in accepted:
                for split in range(len(data) + 1):
                    stream = fst.stream()
                    emitted = []
                    for index, chunk in enumerate((data[:split], data[split:])):
                        emitted.extend(stream.push(chunk))
                        prefix = data[:split] if index == 0 else data
                        candidates = [tokens for key, tokens in accepted
                                      if key.startswith(prefix)]
                        common = list(candidates[0])
                        for tokens in candidates[1:]:
                            common = common[:next(
                                (i for i, (a, b) in enumerate(zip(common, tokens))
                                 if a != b), min(len(common), len(tokens))
                            )]
                        self.assertEqual(emitted, common)
                    emitted.extend(stream.finish())
                    self.assertEqual(emitted, expected)


if __name__ == "__main__":
    unittest.main()
