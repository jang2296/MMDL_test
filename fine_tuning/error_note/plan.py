"""Plan the error-note stages as pod/chain.py specs (`pod-chain-spec-v1`).

CLI: python -m fine_tuning.error_note.plan plan --stage {note-300,note-3417}
         --params PARAMS.json --output STEPS.json
Writes STEPS.json once (an existing file is refused) and prints its sha256. Same params -> same bytes.
Executes nothing; the job's main argv runs the steps with `pod.chain run --spec STEPS.json`.

params (`error-note-plan-params-v1`): exactly the keys below; the paths are canonical absolute paths.
pilot_limit 0 omits the pilot steps. Step timeouts by kind: pin-order and the collect steps -> collect;
train-* and pilot-train -> train; reload-* and pilot-reload -> reload; pilot-export -> export.
"""
from __future__ import annotations

import argparse
import json
import math
import posixpath
import sys
from pathlib import Path
from typing import Any

from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.pod.common import HEX40, HEX64, sha256_bytes

PARAMS_VERSION = "error-note-plan-params-v1"
SPEC_VERSION = "pod-chain-spec-v1"
STAGES = ("note-300", "note-3417")
ARMS = ("note", "solution")
TIMEOUT_KINDS = ("collect", "train", "reload", "export")
PATH_KEYS = ("python", "repo_root", "build_root", "model_path", "out_root")
PARAM_KEYS = {"version", "commit", "order_sha256", "limit", "pilot_limit", "seqs", "sweep_seconds",
              "kv_cache_gib", "max_num_batched_tokens", "token_budget", "gradient_checkpointing", "arms",
              "timeouts", *PATH_KEYS}
COLLECT = "fine_tuning.error_note.collect"
TRAIN = "fine_tuning.error_note.train"
CHAIN = "fine_tuning.notebook_rl.pod.chain"
PROOF = "fine_tuning.notebook_rl.proof_eval"


def _positive_int(params: dict[str, Any], key: str, low: int) -> None:
    value = params[key]
    if type(value) is not int or value < low:
        raise ValueError(f"{key} must be an integer >= {low}")


def _check(stage: str, params: Any) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    missing, unknown = sorted(PARAM_KEYS - set(params)), sorted(set(params) - PARAM_KEYS)
    if missing or unknown:
        raise ValueError(f"params keys: missing {missing}, unknown {unknown}")
    if params["version"] != PARAMS_VERSION:
        raise ValueError(f"version must be {PARAMS_VERSION}")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("commit must be 40 lowercase hex characters")
    if not isinstance(params["order_sha256"], str) or not HEX64.fullmatch(params["order_sha256"]):
        raise ValueError("order_sha256 must be 64 lowercase hex characters")
    for key in PATH_KEYS:
        value = params[key]
        if (not isinstance(value, str) or not value.startswith("/") or value == "/"
                or posixpath.normpath(value) != value):
            raise ValueError(f"{key} must be a canonical absolute path: {value!r}")
    if posixpath.basename(params["model_path"]) != BASE_REVISION:  # load_base and export refuse anything else, but only after GPU time
        raise ValueError(f"model_path must be the pinned snapshot directory {BASE_REVISION}")
    _positive_int(params, "limit", 1)
    _positive_int(params, "pilot_limit", 0)
    _positive_int(params, "sweep_seconds", 1)
    _positive_int(params, "max_num_batched_tokens", 1)
    _positive_int(params, "token_budget", 1)
    kv = params["kv_cache_gib"]
    if isinstance(kv, bool) or not isinstance(kv, (int, float)) or not math.isfinite(kv) or kv <= 0:
        raise ValueError("kv_cache_gib must be a positive finite number")
    seqs = params["seqs"]
    if not isinstance(seqs, list) or not seqs or any(type(s) is not int or s < 1 for s in seqs):
        raise ValueError("seqs must be a non-empty list of positive integers")
    if params["gradient_checkpointing"] not in ("on", "off"):
        raise ValueError('gradient_checkpointing must be "on" or "off"')
    arms = params["arms"]
    if not isinstance(arms, list) or not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be a non-empty list of unique values from {ARMS}")
    timeouts = params["timeouts"]
    if not isinstance(timeouts, dict) or set(timeouts) != set(TIMEOUT_KINDS):
        raise ValueError(f"timeouts must have exactly the keys {list(TIMEOUT_KINDS)}")
    for kind in TIMEOUT_KINDS:
        if type(timeouts[kind]) is not int or timeouts[kind] < 1:
            raise ValueError(f"timeouts.{kind} must be an integer >= 1")
    return params


def _step(name: str, argv: list[str], expect: list[str], kind: str, timeouts: dict[str, int]) -> dict[str, Any]:
    return {"name": name, "argv": argv, "expect_files": expect, "timeout_seconds": timeouts[kind]}


def plan(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    p = _check(stage, params)
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order = f"{p['repo_root']}/fine_tuning/error_note/item_order.json"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    collect_dir, limit, timeouts = f"{out}/collect", str(p["limit"]), p["timeouts"]

    def common(model_path: str) -> list[str]:
        return ["--build-root", build, "--order", order, "--model-path", model_path,
                "--kv-cache-gib", str(p["kv_cache_gib"]),
                "--max-num-batched-tokens", str(p["max_num_batched_tokens"]), "--engine", "vllm"]

    def train_run(rows: str, n: str, arm: str, output: str) -> list[str]:
        return [*train_cmd, "train", "--rows", rows, "--build-root", build, "--order", order, "--limit", n,
                "--arm", arm, "--precision", "bf16", "--model-path", model, "--output", output,
                "--code-commit", commit, "--token-budget", str(p["token_budget"]),
                "--gradient-checkpointing", p["gradient_checkpointing"], "--resume"]

    def reload_run(run: str) -> list[str]:
        return [*train_cmd, "reload", "--run", run, "--model-path", model, "--precision", "bf16"]

    steps = [_step("pin-order", [*chain_cmd, "pin", "--file", order, "--sha256", p["order_sha256"]],
                   [], "collect", timeouts)]
    if p["pilot_limit"] > 0:
        pilot_limit, pilot_rows = str(p["pilot_limit"]), f"{out}/pilot-rows"
        pilot_train, pilot_export = f"{out}/pilot-train", f"{out}/pilot-export"
        steps += [
            _step("pilot-solve", [*collect_cmd, "solve", *common(model), "--limit", pilot_limit,
                                  "--output", collect_dir, "--max-num-seqs", "16"],
                  [f"{collect_dir}/t1.jsonl"], "collect", timeouts),
            _step("pilot-note", [*collect_cmd, "note", *common(model), "--limit", pilot_limit,
                                 "--output", collect_dir, "--max-num-seqs", "16"],
                  [f"{collect_dir}/note.jsonl"], "collect", timeouts),
            _step("pilot-rows", [*collect_cmd, "rows", "--build-root", build, "--order", order,
                                 "--limit", pilot_limit, "--tokenizer-path", model,
                                 "--output", collect_dir, "--rows-output", pilot_rows],
                  [f"{pilot_rows}/rows.jsonl"], "collect", timeouts),
            _step("pilot-train", train_run(f"{pilot_rows}/rows.jsonl", pilot_limit, "note", pilot_train),
                  [f"{pilot_train}/checkpoint-final/metadata.json", f"{pilot_train}/result.json"],
                  "train", timeouts),
            _step("pilot-reload", reload_run(pilot_train), [f"{pilot_train}/reload.json"], "reload", timeouts),
            _step("pilot-export", [py, "-m", PROOF, "export", "--checkpoint", f"{pilot_train}/checkpoint-final",
                                   "--binding", f"{pilot_train}/binding.json", "--base", model,
                                   "--output", pilot_export, "--code-commit", commit],
                  [f"{pilot_export}/model_ref.json", f"{pilot_export}/model/export_manifest.json"],
                  "export", timeouts),
            _step("pilot-ft-check", [*collect_cmd, "solve", *common(f"{pilot_export}/model"), "--limit", "2",
                                     "--output", f"{out}/pilot-ft-check", "--max-num-seqs", "4"],
                  [f"{out}/pilot-ft-check/t1.jsonl"], "collect", timeouts),
        ]
    steps += [
        _step("sweep", [*collect_cmd, "sweep", *common(model), "--limit", limit, "--output", collect_dir,
                        "--seqs", ",".join(str(s) for s in p["seqs"]), "--seconds", str(p["sweep_seconds"])],
              [f"{collect_dir}/sweep.json"], "collect", timeouts),
        _step("note", [*collect_cmd, "note", *common(model), "--limit", limit, "--output", collect_dir,
                       "--max-num-seqs-from", f"{collect_dir}/sweep.json"],
              [f"{collect_dir}/note.jsonl"], "collect", timeouts),
        _step("rows", [*collect_cmd, "rows", "--build-root", build, "--order", order, "--limit", limit,
                       "--tokenizer-path", model, "--output", collect_dir],
              [f"{collect_dir}/rows.jsonl"], "collect", timeouts),
    ]
    for arm in p["arms"]:
        run = f"{out}/train-{arm}"
        steps.append(_step(f"train-{arm}", train_run(f"{collect_dir}/rows.jsonl", limit, arm, run),
                           [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts))
        steps.append(_step(f"reload-{arm}", reload_run(run), [f"{run}/reload.json"], "reload", timeouts))
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"version": SPEC_VERSION,
            "meta": {"stage": stage, "job_stage": stage, "params_sha256": sha256_bytes(canonical.encode()),
                     "generator": "fine_tuning.error_note.plan",
                     "total_timeout_seconds": sum(step["timeout_seconds"] for step in steps)},
            "steps": steps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write the chain spec of one error-note stage")
    plan_parser.add_argument("--stage", choices=STAGES, required=True)
    plan_parser.add_argument("--params", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        text = json.dumps(plan(args.stage, json.loads(args.params.read_text(encoding="utf-8"))),
                          indent=2, sort_keys=True) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(sha256_bytes(text.encode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
