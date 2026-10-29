"""Stream stdin through an FST, one byte at a time.

Each committed token ID is written to stdout on its own line and flushed
immediately, so downstream programs see IDs as soon as they are fixed.
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence

from ._native import BeamFST, FST


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="push2tok",
        description=(
            "Read stdin one byte at a time, push each byte into an FST stream, "
            "and print committed token IDs (one per line) as soon as they are known."
        ),
        epilog=(
            "Input is fed to the FST as raw bytes. For SentencePiece-derived FSTs "
            "(e.g. tools/convert_rinna.sh output), stdin must already be normalized "
            "with the accompanying .spiece.model."
        ),
    )
    parser.add_argument("--fst", required=True, metavar="PATH", help="FST file to load")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "beam"),
        default="deterministic",
        help=(
            "deterministic: exact incremental output for deterministic, unweighted FSTs; "
            "beam: beam search for weighted, non-deterministic FSTs (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("binary", "const", "text"),
        default="binary",
        help="FST file format: VectorFst binary, ConstFst binary, or AT&T text (default: %(default)s)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        metavar="N",
        help="number of hypotheses kept in --mode beam (default: 30)",
    )
    return parser


def _write(tokens: List[int]) -> None:
    if tokens:
        sys.stdout.write("".join(f"{token}\n" for token in tokens))
        sys.stdout.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "deterministic" and args.beam_width is not None:
        parser.error("--beam-width is only valid with --mode beam")
    beam_width = 30 if args.beam_width is None else args.beam_width
    if beam_width <= 0:
        parser.error("--beam-width must be positive")

    try:
        if args.mode == "deterministic":
            stream = FST.from_file(args.fst, format=args.format).stream()
        else:
            stream = BeamFST.from_file(args.fst, format=args.format).stream(beam_width)
    except (OSError, ValueError) as error:
        print(f"push2tok: cannot load {args.fst}: {error}", file=sys.stderr)
        return 2

    fd = sys.stdin.fileno()
    offset = 0
    try:
        # Tokens may already be fixed before any input arrives.
        _write(stream.push(b""))
        while True:
            byte = os.read(fd, 1)  # unbuffered: react to each byte as it arrives
            if not byte:
                break
            try:
                _write(stream.push(byte))
            except ValueError as error:
                print(f"push2tok: input rejected at byte offset {offset}: {error}",
                      file=sys.stderr)
                return 1
            offset += 1
        try:
            _write(stream.finish())
        except ValueError as error:
            print(f"push2tok: incomplete input at end of stream: {error}", file=sys.stderr)
            return 1
    except BrokenPipeError:
        # The reader went away (e.g. `| head`). Silence the flush at exit.
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)
        return 0
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
