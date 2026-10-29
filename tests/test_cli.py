"""The CLI must emit tokens before stdin closes, not just after EOF."""

from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest

from push2tok import FST


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fst.txt"

    def run_cli(self, *args, data=b""):
        return subprocess.run(
            [sys.executable, "-m", "push2tok", *args],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )

    def test_deterministic_live_output_and_eof(self):
        fst = FST([(0, ord("a"), 1, []), (1, ord("b"), 2, [10])],
                  {0: [], 1: [10], 2: [20]})
        fst.write(self.path, format="text")
        proc = subprocess.Popen(
            [sys.executable, "-m", "push2tok", "--fst", str(self.path),
             "--format", "text", "--mode", "deterministic"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        def close_process():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()

        self.addCleanup(close_process)
        selector = selectors.DefaultSelector()
        self.addCleanup(selector.close)
        selector.register(proc.stdout, selectors.EVENT_READ)
        proc.stdin.write(b"a")
        proc.stdin.flush()
        self.assertTrue(selector.select(timeout=5), "the first token was not flushed")
        self.assertEqual(proc.stdout.readline(), b"10\n")
        proc.stdin.write(b"b")
        proc.stdin.flush()
        self.assertTrue(selector.select(timeout=5), "the second token was not flushed")
        self.assertEqual(proc.stdout.readline(), b"20\n")
        proc.stdin.close()
        self.assertEqual(proc.stdout.read(), b"")
        self.assertEqual(proc.wait(timeout=5), 0)
        self.assertEqual(proc.stderr.read(), b"")

        result = self.run_cli("--fst", str(self.path), "--format", "text", data=b"a")
        self.assertEqual((result.returncode, result.stdout), (0, b"10\n"))

    def test_beam_mode_weighted_nondeterministic_fst(self):
        self.path.write_text("0\t1\t98\t4\t0\n0\t2\t98\t4\t1\n0\n1\n2\n")
        result = self.run_cli("--fst", str(self.path), "--format", "text",
                              "--mode", "beam", "--beam-width", "2", data=b"a")
        self.assertEqual((result.returncode, result.stdout), (0, b"3\n"))
        self.assertEqual(result.stderr, b"")
        # Final-only outputs are flushed after EOF.
        self.path.write_text("0\t1\t98\t0\n1\t2\t0\t8\n2\n")
        result = self.run_cli("--fst", str(self.path), "--format", "text",
                              "--mode", "beam", data=b"a")
        self.assertEqual((result.returncode, result.stdout), (0, b"7\n"))

    def test_invalid_input_incomplete_and_bad_options(self):
        fst = FST([(0, ord("a"), 1, [])], {1: [5]})
        fst.write(self.path, format="text")
        args = ("--fst", str(self.path), "--format", "text")
        result = self.run_cli(*args, data=b"x")
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"byte offset 0", result.stderr)
        result = self.run_cli(*args, data=b"")
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"incomplete input", result.stderr)
        result = self.run_cli(*args, "--beam-width", "2")
        self.assertEqual(result.returncode, 2)
        result = self.run_cli(*args, "--mode", "beam", "--beam-width", "0")
        self.assertEqual(result.returncode, 2)
        result = self.run_cli("--fst", str(self.path) + ".missing")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
