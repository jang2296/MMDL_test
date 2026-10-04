"""CPU prepare/dry-run or explicit local vLLM frozen answer audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", help="http://127.0.0.1:PORT/v1 or localhost SSH tunnel")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--revision", default="UNPINNED")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually call the local endpoint")
    parser.add_argument("--semantic-open", action="store_true", help="Enable second-stage equivalence")
    parser.add_argument("--retry-failures", action="store_true", help="Append a new attempt for errors")
    parser.add_argument("--extraction-mode", choices=("literal", "segments"), default="literal")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--deadline-utc", help="Absolute UTC cutoff, e.g. 2026-10-03T04:22:38Z")
    args = parser.parse_args()
    try:
        summary = execute(args.source, args.output, model=args.model, revision=args.revision,
                          endpoint=args.endpoint, identity_path=args.model_manifest,
                          run=args.execute, semantic_open=args.semantic_open,
                          retry_failures=args.retry_failures, timeout=args.timeout,
                          deadline_utc=args.deadline_utc, extraction_mode=args.extraction_mode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"gemma_verifier: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if args.execute and summary["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
