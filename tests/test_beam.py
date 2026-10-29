"""Beam-limited streaming decoding of non-deterministic, weighted FSTs."""

from pathlib import Path
import tempfile
import unittest

from push2tok import BeamFST, FST


class BeamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.txt"

    def load(self, text):
        self.path.write_text(text)
        return BeamFST.from_file(self.path, format="text")

    def test_shared_ancestor_and_beam_ranking(self):
        fst = self.load("0\t1\t98\t5\n"
                        "1\t2\t99\t6\t1\n"
                        "1\t3\t99\t7\t0\n"
                        "1\t4\t100\t6\n2\n3\n4\n")
        stream = fst.stream(beam_width=2)
        self.assertEqual(stream.push(b"a"), [4])
        self.assertEqual(stream.pending_nodes, 0)
        self.assertEqual(stream.push(b"b"), [])  # two different outputs remain
        self.assertEqual(stream.pending_nodes, 2)
        self.assertEqual(stream.finish(), [6])  # lower-cost branch wins
        with self.assertRaises(RuntimeError):
            stream.push(b"x")
        stream.reset()
        self.assertEqual(stream.push(b"ac"), [4, 5])
        self.assertEqual(stream.finish(), [])
        narrow = fst.stream(beam_width=1)
        self.assertEqual(narrow.push(b"a"), [4])
        self.assertEqual(narrow.push(b"b"), [6])
        self.assertEqual(narrow.finish(), [])
        self.assertEqual(fst.transduce(b"ab"), [4, 6])

    def test_epsilon_expansion_and_end_of_input_branch(self):
        fst = self.load("0\t1\t98\t0\n"
                        "1\t2\t0\t6\t0\n"
                        "1\t3\t0\t7\t1\n"
                        "2\t4\t99\t0\n3\t5\t99\t0\n4\n5\n")
        stream = fst.stream(2)
        self.assertEqual(stream.push(b"a"), [])
        self.assertEqual(stream.push(b"b"), [])
        self.assertEqual(stream.finish(), [5])
        self.assertEqual(fst.transduce(b"ab", beam_width=1), [5])

        # EOS must not be discarded just because an unfinished byte path has
        # a better optimistic score at the current frame.
        fst = self.load("0\t1\t98\t0\n"
                        "1\t2\t0\t2\t5\n"
                        "1\t3\t99\t3\t0\n2\n3\n")
        stream = fst.stream(1)
        self.assertEqual(stream.push(b"a"), [])
        self.assertEqual(stream.finish(), [1])
        stream.reset()
        self.assertEqual(stream.push(b"a"), [])
        self.assertEqual(stream.push(b"b"), [2])
        self.assertEqual(stream.finish(), [])

    def test_reconvergent_equal_outputs_share_history(self):
        fst = self.load("0\t1\t98\t2\t1\n"
                        "0\t2\t98\t2\t0\n"
                        "1\t3\t99\t3\n2\t4\t99\t3\n3\n4\n")
        stream = fst.stream(2)
        self.assertEqual(stream.push(b"a"), [1])
        self.assertEqual(stream.push(b"b"), [2])
        self.assertEqual(stream.pending_nodes, 0)
        self.assertEqual(stream.finish(), [])

    def test_same_state_keeps_only_cheapest_history(self):
        # Two different outputs reach the same state on 'a'. Any suffix from
        # there costs the same for both, so the more expensive history can
        # never win. This is Viterbi recombination, not just trace interning.
        fst = self.load("0\t1\t98\t2\t0\n"
                        "0\t1\t98\t3\t1\n"
                        "1\t2\t99\t4\t0\n2\n")
        stream = fst.stream(beam_width=30)
        self.assertEqual(stream.push(b"a"), [1])
        self.assertEqual(stream.pending_nodes, 0)
        self.assertEqual(stream.push(b"b"), [3])
        self.assertEqual(stream.finish(), [])

    def test_chunk_rollback_and_premature_finish(self):
        fst = self.load("0\t1\t98\t2\n1\t2\t99\t3\n2\n")
        stream = fst.stream()
        with self.assertRaisesRegex(ValueError, "no accepting hypothesis"):
            stream.finish()
        with self.assertRaisesRegex(ValueError, "beam exhausted"):
            stream.push(b"ax")
        self.assertEqual(stream.push(b"a"), [1])
        self.assertEqual(stream.push(b"b"), [2])
        self.assertEqual(stream.finish(), [])
        with self.assertRaises(ValueError):
            fst.stream(beam_width=0)

    def test_commits_and_releases_old_trace_frames(self):
        fst = self.load("0\t0\t98\t2\n0\n")
        stream = fst.stream()
        self.assertEqual(stream.push(b"a" * 1000), [1] * 1000)
        self.assertEqual(stream.pending_nodes, 0)
        self.assertEqual(stream.finish(), [])
        stream.reset()
        self.assertEqual(stream.pending_nodes, 0)
        self.assertEqual(stream.finish(), [])

    def test_binary_const_and_validation(self):
        deterministic = FST([(0, 0, 1, [10])], {1: [11]})
        for format in ("binary", "const", "text"):
            with self.subTest(format=format):
                path = Path(self.temp.name) / f"test.{format}"
                deterministic.write(path, format=format)
                loaded = BeamFST.from_file(path, format=format)
                self.assertEqual(loaded.transduce(b"\x00"), [10, 11])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.load("0\t1\t98\t2\t-1\n1\n")
        with self.assertRaisesRegex(ValueError, "epsilon cycle"):
            self.load("0\t0\t0\t0\n0\n")
        with self.assertRaises(OSError):
            BeamFST.from_file(Path(self.temp.name) / "missing.fst")

    def test_initial_epsilon_output_and_final_weight(self):
        fst = FST([], {0: [7]})
        path = Path(self.temp.name) / "initial.fst"
        fst.write(path)
        stream = BeamFST.from_file(path).stream()
        self.assertEqual(stream.push(b""), [7])
        self.assertEqual(stream.pending_nodes, 0)
        self.assertEqual(stream.finish(), [])
        stream.reset()
        self.assertEqual(stream.finish(), [7])

        fst = self.load("0\t1\t98\t2\t0\n0\t2\t98\t3\t0\n1\t5\n2\t1\n")
        self.assertEqual(fst.transduce(b"a", beam_width=1), [2])


if __name__ == "__main__":
    unittest.main()
