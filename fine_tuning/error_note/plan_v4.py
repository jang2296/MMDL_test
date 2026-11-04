"""Plan one version-4 round as a pod/chain.py spec (`pod-chain-spec-v1`). Executes nothing.

CLI: python -m fine_tuning.error_note.plan_v4 plan --params PARAMS.json --output STEPS.json
Steps: pin order/rules (and a reused F0 T1) -> rollout -> note -> selfcheck -> resolve -> rows -> train -> reload ->
export to <volume_root>/eval-data/proof/export-v4-r<round>-<arm>. Generation uses auto KV/concurrency, vLLM prefix
caching and the training-only loop stop (plan section 10.3). The next round's rollout_model_path is this export's model.

params (`error-note-plan-params-v4`): exactly PARAM_KEYS; paths are canonical absolute paths; private inputs
(order, rules, reused T1) lie outside repo_root. init_run is null for round 1 and the previous run directory after.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import sys
from pathlib import Path
from typing import Any

from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT
from fine_tuning.notebook_rl.pod.common import HEX40, HEX64, sha256_bytes

from .plan import CHAIN, PROOF, SPEC_VERSION, TIMEOUT_KINDS, _step
from .v4 import ARMS

PARAMS_VERSION_V4 = "error-note-plan-params-v4"
STAGE = "note-v4-round"
COLLECT_V4, TRAIN_V4 = "fine_tuning.error_note.collect_v4", "fine_tuning.error_note.train_v4"
PATHS = ("python", "repo_root", "build_root", "base_model_path", "rollout_model_path", "out_root", "volume_root",
         "order_path", "training_rules_path")
OPTIONAL_PATHS = ("init_run", "reuse_t1_path")
HASHES = ("order_sha256", "training_rules_sha256")
PARAM_KEYS = {"version", "commit", "round", "start", "limit", "arm", "lesson_control", "max_num_batched_tokens",
              "token_budget", "gradient_checkpointing", "timeouts", "reuse_t1_sha256", *PATHS, *OPTIONAL_PATHS, *HASHES}


def _path(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value == "/" or posixpath.normpath(value) != value:
        raise ValueError(f"{key} must be a canonical absolute path: {value!r}")
    return value


def check(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    missing, unknown = sorted(PARAM_KEYS - set(params)), sorted(set(params) - PARAM_KEYS)
    if missing or unknown:
        raise ValueError(f"params keys: missing {missing}, unknown {unknown}")
    if params["version"] != PARAMS_VERSION_V4:
        raise ValueError(f"version must be {PARAMS_VERSION_V4}")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("commit must be 40 lowercase hex characters")
    for key in PATHS:
        _path(params[key], key)
    for key in OPTIONAL_PATHS:
        if params[key] is not None:
            _path(params[key], key)
    for key in HASHES:
        if not isinstance(params[key], str) or not HEX64.fullmatch(params[key]):
            raise ValueError(f"{key} must be 64 lowercase hex characters")
    if posixpath.basename(params["base_model_path"]) != BASE_REVISION:
        raise ValueError(f"base_model_path must be the pinned snapshot directory {BASE_REVISION}")
    for key in ("order_path", "training_rules_path", "build_root", "out_root", "reuse_t1_path"):
        if params[key] is not None and posixpath.commonpath((params["repo_root"], params[key])) == params["repo_root"]:
            raise ValueError(f"{key} must be outside repo_root (private evidence is not repository content)")
    for key, low in (("round", 1), ("start", 0), ("limit", 1), ("max_num_batched_tokens", 1), ("token_budget", 1)):
        if type(params[key]) is not int or params[key] < low:
            raise ValueError(f"{key} must be an integer >= {low}")
    if (params["round"] == 1) != (params["init_run"] is None):
        raise ValueError("round 1 has init_run null; a later round names the previous run directory")
    if (params["round"] == 1) != (params["rollout_model_path"] == params["base_model_path"]):
        raise ValueError("round 1 rolls out with the base snapshot; a later round with the previous export")
    if (params["reuse_t1_path"] is None) != (params["reuse_t1_sha256"] is None) or (
            params["reuse_t1_path"] is not None and (params["round"] != 1 or not HEX64.fullmatch(params["reuse_t1_sha256"]))):
        raise ValueError("reuse_t1_path and reuse_t1_sha256 go together, in round 1 only")
    if params["arm"] not in ARMS or type(params["lesson_control"]) is not bool:
        raise ValueError(f"arm must be one of {ARMS}; lesson_control a boolean")
    if params["gradient_checkpointing"] not in ("on", "off"):
        raise ValueError('gradient_checkpointing must be "on" or "off"')
    timeouts = params["timeouts"]
    if not isinstance(timeouts, dict) or set(timeouts) != set(TIMEOUT_KINDS) or any(
            type(timeouts[k]) is not int or timeouts[k] < 1 for k in TIMEOUT_KINDS):
        raise ValueError(f"timeouts must map exactly {list(TIMEOUT_KINDS)} to integers >= 1")
    return params


def steps_v4(p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, timeouts = p["python"], p["out_root"], p["timeouts"]
    collect_dir, run = f"{out}/collect-r{p['round']}", f"{out}/train-r{p['round']}-{p['arm']}"
    export = f"{p['volume_root']}/eval-data/proof/export-v4-r{p['round']}-{p['arm']}"
    collect, train, chain = [py, "-m", COLLECT_V4], [py, "-m", TRAIN_V4], [py, "-m", CHAIN]
    slice_args = ["--round", str(p["round"]), "--build-root", p["build_root"], "--order", p["order_path"],
                  "--start", str(p["start"]), "--limit", str(p["limit"]), "--output", collect_dir,
                  "--answer-contract", TRAINING_ANSWER_CONTRACT, "--training-rules", p["training_rules_path"],
                  "--training-rules-sha256", p["training_rules_sha256"]]
    engine = ["--model-path", p["rollout_model_path"], "--engine", "vllm", "--kv-cache-gib", "auto",
              "--max-num-seqs", "auto", "--max-num-batched-tokens", str(p["max_num_batched_tokens"]),
              "--prefix-caching", "--loop-stop"]
    pins = [("order", p["order_path"], p["order_sha256"]),
            ("training-rules", p["training_rules_path"], p["training_rules_sha256"])]
    reuse: list[str] = []
    if p["reuse_t1_path"] is not None:
        pins.append(("reuse-t1", p["reuse_t1_path"], p["reuse_t1_sha256"]))
        reuse = ["--reuse-t1", p["reuse_t1_path"], "--reuse-t1-sha256", p["reuse_t1_sha256"]]
    steps = [_step(f"pin-{name}", [*chain, "pin", "--file", path, "--sha256", digest], [], "collect", timeouts)
             for name, path, digest in pins]
    train_args = ["--rows", f"{collect_dir}/rows.jsonl", "--build-root", p["build_root"], "--order", p["order_path"],
                  "--round", str(p["round"]), "--start", str(p["start"]), "--limit", str(p["limit"]),
                  "--arm", p["arm"], "--precision", "bf16", "--model-path", p["base_model_path"], "--output", run,
                  "--code-commit", p["commit"], "--token-budget", str(p["token_budget"]),
                  *(["--init-run", p["init_run"]] if p["init_run"] is not None else [])]
    return steps + [
        _step("rollout", [*collect, "rollout", *slice_args, *engine, *reuse],
              [f"{collect_dir}/rollouts.jsonl", f"{collect_dir}/receipt-rollouts.json"], "collect", timeouts),
        _step("note", [*collect, "note", *slice_args, *engine], [f"{collect_dir}/gates-v4.json"], "collect", timeouts),
        _step("selfcheck", [*collect, "selfcheck", *slice_args, *engine], [f"{collect_dir}/selfcheck-v4.json"],
              "collect", timeouts),
        _step("resolve", [*collect, "resolve", *slice_args, *engine,
                          *(["--lesson-control"] if p["lesson_control"] else [])],
              [f"{collect_dir}/receipt-resolve.json"], "collect", timeouts),
        _step("rows", [*collect, "rows", *slice_args, "--tokenizer-path", p["base_model_path"]],
              [f"{collect_dir}/rows.jsonl", f"{collect_dir}/rows_summary.json"], "collect", timeouts),
        _step("train", [*train, "train", *train_args, "--gradient-checkpointing", p["gradient_checkpointing"], "--resume"],
              [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts),
        _step("reload", [*train, "reload", "--run", run, "--model-path", p["base_model_path"], "--precision", "bf16"],
              [f"{run}/reload.json"], "reload", timeouts),
        _step("export", [py, "-m", PROOF, "export", "--checkpoint", f"{run}/checkpoint-final",
                         "--binding", f"{run}/binding.json", "--base", p["base_model_path"], "--output", export,
                         "--code-commit", p["commit"]],
              [f"{export}/model_ref.json", f"{export}/model/export_manifest.json"], "export", timeouts),
    ]


def plan_v4(params: dict[str, Any]) -> dict[str, Any]:
    p = check(params)
    steps = steps_v4(p)
    canonical = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"version": SPEC_VERSION,
            "meta": {"stage": STAGE, "job_stage": STAGE, "params_sha256": sha256_bytes(canonical.encode()),
                     "generator": "fine_tuning.error_note.plan_v4",
                     "total_timeout_seconds": sum(step["timeout_seconds"] for step in steps)},
            "steps": steps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="write the chain spec of one version-4 round")
    plan_parser.add_argument("--params", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        text = json.dumps(plan_v4(json.loads(args.params.read_text(encoding="utf-8"))), indent=2, sort_keys=True) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(sha256_bytes(text.encode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
