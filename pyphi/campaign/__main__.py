"""Command-line entry point: ``python -m pyphi.campaign run TASK_FILE``."""

from __future__ import annotations

import argparse
import sys

from pyphi.campaign.runner import run_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyphi.campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute one campaign task file")
    run_parser.add_argument("task_file")
    run_parser.add_argument("--substrates", default="substrates")
    run_parser.add_argument("--outputs", default=".")
    args = parser.parse_args(argv)
    return run_task(args.task_file, args.substrates, args.outputs)


if __name__ == "__main__":
    sys.exit(main())
