"""Explicit local entrypoints for the existing private data-audit tools."""
from __future__ import annotations

import argparse
from importlib import import_module

COMMANDS = {
    "index": "build_exclusion_index",
    "exclusion": "hf_pool_exclusion",
    "review": "hf_pool_review",
    "pool": "pool_freeze",
    "export-approval": "local_run",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = (["export-approval"] if args.command == "export-approval" else []) + args.arguments
    return import_module(f"fine_tuning.notebook_rl.{COMMANDS[args.command]}").main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
