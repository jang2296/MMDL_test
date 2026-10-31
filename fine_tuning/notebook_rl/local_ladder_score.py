"""CPU replay of native ladder answers; does not authorize or prove inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .cli import artifact_path
from .dataset import _bytes, _sha, _write
from .local_ladder import _descriptor, _json, _path, load_ladder_actors
from .schema import to_json
from .vault import _read, _safe_root, _unique
from .verifier_service import VerifierService

VERSION = "local-native-ladder-cpu-score-v1"
_COUNTED = frozenset(("CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"))


def _code() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("local_ladder_score.py", "local_ladder.py", "cli.py", "dataset.py", "verifier_service.py",
             "verifier.py", "answers.py", "observations.py", "attribution.py", "schema.py", "guards.py", "vault.py",
             "generators/graph.py")
    return {name: _sha(_read(root / name)) for name in names}


def score_responses(cohort: Path, responses: list[dict], *, expected_preparation_sha256: str) -> dict:
    """Read only public inputs in this process; a fresh verifier owns private gold."""
    root = _safe_root(cohort)
    preparation_path = root / "preparation.json"
    identity = _descriptor(preparation_path)
    if (type(expected_preparation_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_preparation_sha256) is None
            or identity["file_sha256"] != expected_preparation_sha256):
        raise ValueError("ladder score: pinned preparation SHA required")
    preparation, code = _json(preparation_path), _code()
    actors = load_ladder_actors(root)
    if (preparation.get("status") != "CPU_COHORT_PREPARED_PENDING_GATES"
            or preparation.get("count") != 10 or preparation.get("gpu_evaluation_completed") is not False
            or any(a.mode != "obs" or a.answer_format != "unit_number" or a.choices for a in actors)):
        raise ValueError("ladder score: original native unit-answer cohort required")
    ids = [a.problem_id for a in actors]
    if (type(responses) is not list or len(responses) != 10
            or any(type(row) is not dict or set(row) != {"problem_id", "text", "finish_reason"}
                   or type(row["problem_id"]) is not str or type(row["text"]) is not str
                   or type(row["finish_reason"]) is not str or row["finish_reason"] not in ("stop", "length")
                   for row in responses)
            or [row["problem_id"] for row in responses] != ids):
        raise ValueError("ladder score: exactly ten ordered original response records required")
    response_sha = _sha(_bytes(responses))
    rows = []
    with VerifierService(root / "records", forbidden_roots=[Path.home() / "mmdl-data/evaluation"], timeout=60) as verifier:
        metadata = verifier.metadata
        if (metadata["item_count"] != 10
                or metadata["vault_manifest_sha256"] != preparation["records_manifest"]["file_sha256"]):
            raise ValueError("ladder score: fresh verifier cohort identity differs")
        for actor, response in zip(actors, responses, strict=True):
            result = verifier.grade(actor.problem_id, response["text"], response["finish_reason"], actor.mode)
            rows.append({"problem_id": actor.problem_id, "response_sha256": _sha(_bytes(response)),
                         "correct": result.status == "CORRECT" if result.status in _COUNTED else None,
                         "verifier": json.loads(to_json(result))})
    if (_descriptor(preparation_path) != identity or load_ladder_actors(root) != actors
            or _code() != code or _sha(_bytes(responses)) != response_sha):
        raise ValueError("ladder score: inputs or scorer changed during replay")
    counts = dict(Counter(row["verifier"]["status"] for row in rows))
    complete = all(row["correct"] is not None for row in rows)
    correct = sum(row["correct"] is True for row in rows)
    return {"status": "CPU_RESPONSES_SCORED" if complete else "INCOMPLETE_UNVERIFIABLE_RESPONSES",
            "scoring_version": VERSION, "preparation": identity, "responses_sha256": response_sha,
            "source_sha256": code, "verifier_process": metadata, "problem_ids": ids, "rows": rows,
            "n_expected": 10, "n_responses": len(rows), "n_correct": correct,
            "accuracy": correct / 10 if complete else None, "status_counts": counts,
            "training_allowed": False, "evaluation_execution_authorized": False,
            "model_generation_provenance_verified": False, "actual_input_tokens_verified": False,
            "cpu_replay_only": True, "module_causality_certified": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-preparation-sha256", required=True)
    parser.add_argument("--responses", type=Path, required=True,
                        help="JSON array of ten ordered problem_id/text/finish_reason records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cohort = _safe_root(args.cohort)
    responses_path = _path(args.responses)
    output = args.output.absolute()
    if (".." in output.parts or output.is_relative_to(cohort) or output == responses_path
            or output.exists() or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("ladder score: new output outside immutable inputs required")
    raw = _read(responses_path)
    responses = json.loads(raw, object_pairs_hook=_unique)
    result = score_responses(cohort, responses, expected_preparation_sha256=args.cohort_preparation_sha256)
    if _read(responses_path) != raw:
        raise ValueError("ladder score: response file changed during replay")
    result["response_file"] = {"path": str(responses_path), "file_sha256": _sha(raw)}
    _write(artifact_path(output), _bytes(result) + b"\n")
    print(json.dumps({key: result[key] for key in ("status", "n_correct", "accuracy", "cpu_replay_only")}))
    return 0 if result["status"] == "CPU_RESPONSES_SCORED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
