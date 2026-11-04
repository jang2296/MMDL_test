"""Plan the error-note stages as pod/chain.py specs (`pod-chain-spec-v1`).

CLI: python -m fine_tuning.error_note.plan plan --stage {note-300,note-3417,note-v2,note-v3-collect,note-v3-resolve,note-v3-train}
         --params PARAMS.json --output STEPS.json
Writes STEPS.json once (an existing file is refused) and prints its sha256. Same params -> same bytes.
Executes nothing; the job's main argv runs the steps with `pod.chain run --spec STEPS.json`.

params (`error-note-plan-params-v1`): exactly the keys below; the paths are canonical absolute paths.
pilot_limit 0 omits the pilot steps. kv_cache_gib is a positive number or "auto"; seqs is a non-empty list of positive
integers or "auto". Both reach collect as given, and collect sizes the KV cache and the sweep from the GPU's memory.
Step timeouts by kind: pin-order and the collect steps -> collect;
train-* and pilot-train -> train; reload-* and pilot-reload -> reload; pilot-export -> export.

Stage note-v2 (claudedocs/plan_self_note_v2_20261102.md) takes `error-note-plan-params-v2`: the v1 keys without
pilot_limit and arms (no pilot; arm note only) plus volume_root and reuse_t1_sha256. Steps: pin-order (item_order_v2.json)
-> sweep (T1), or with reuse_t1_sha256 a pin of the copied <out_root>/collect/t1.jsonl instead -> note (v2) ->
resolve -> rows (v2) -> train-note (protocol v2) -> reload-note -> export to <volume_root>/eval-data/proof/export-note-v2.
Without the sweep, note and resolve take --max-num-seqs auto (the middle sweep candidate for the KV size).

Stages note-v3-collect, note-v3-resolve and note-v3-train take `error-note-plan-params-v3`. All use external order_path,
training_rules_path and source_provenance_path with their SHA256 pins, volume_root, auto KV/concurrency and
token_budget 24000. The collect stage stops after fresh T1 and v2 notes; the resolve stage additionally requires
reviews_path/reviews_sha256 and t1_sha256/note_sha256. It pins those collected files before audit -> resolve ->
candidate rows v3, then stops. No private file is read while planning. Train requires closed rows_path/audit_path/solution_audit_path
and their SHA256 pins, then performs admission and both training arms without regenerating targets. Shared generation
settings remain schema fields for all stages. Note and solution reviews are created outside these executable specs.
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
from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT
from fine_tuning.notebook_rl.pod.common import HEX40, HEX64, sha256_bytes

PARAMS_VERSION = "error-note-plan-params-v1"
SPEC_VERSION = "pod-chain-spec-v1"
PARAMS_VERSION_V2 = "error-note-plan-params-v2"
PARAMS_VERSION_V3 = "error-note-plan-params-v3"
V3_STAGES = ("note-v3-collect", "note-v3-resolve", "note-v3-train")
STAGES = ("note-300", "note-3417", "note-v2", *V3_STAGES)
ARMS = ("note", "solution")
TIMEOUT_KINDS = ("collect", "train", "reload", "export")
PATH_KEYS = ("python", "repo_root", "build_root", "model_path", "out_root")
PARAM_KEYS = {"version", "commit", "order_sha256", "limit", "pilot_limit", "seqs", "sweep_seconds",
              "kv_cache_gib", "max_num_batched_tokens", "token_budget", "gradient_checkpointing", "arms",
              "timeouts", *PATH_KEYS}
PARAM_KEYS_V2 = (PARAM_KEYS - {"pilot_limit", "arms"}) | {"volume_root", "reuse_t1_sha256"}
V3_PRIVATE_PATHS = ("order_path", "training_rules_path", "source_provenance_path")
V3_HASH_KEYS = ("training_rules_sha256", "source_provenance_sha256")
PARAM_KEYS_V3 = (PARAM_KEYS_V2 - {"reuse_t1_sha256"}) | {*V3_PRIVATE_PATHS, *V3_HASH_KEYS}
PARAM_KEYS_V3_RESOLVE = PARAM_KEYS_V3 | {"reviews_path", "reviews_sha256", "t1_sha256", "note_sha256"}
PARAM_KEYS_V3_TRAIN = PARAM_KEYS_V3 | {"rows_path", "rows_sha256", "audit_path", "audit_sha256",
                                     "solution_audit_path", "solution_audit_sha256"}
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
    v2, v3 = stage == "note-v2", stage in V3_STAGES
    extra_paths: tuple[str, ...] = ()
    extra_hashes: tuple[str, ...] = ()
    if v3:
        keys, version = PARAM_KEYS_V3, PARAMS_VERSION_V3
        if stage == "note-v3-resolve":
            keys = PARAM_KEYS_V3_RESOLVE
            extra_paths, extra_hashes = ("reviews_path",), ("reviews_sha256", "t1_sha256", "note_sha256")
        elif stage == "note-v3-train":
            keys = PARAM_KEYS_V3_TRAIN
            extra_paths = ("rows_path", "audit_path", "solution_audit_path")
            extra_hashes = ("rows_sha256", "audit_sha256", "solution_audit_sha256")
    else:
        keys, version = (PARAM_KEYS_V2, PARAMS_VERSION_V2) if v2 else (PARAM_KEYS, PARAMS_VERSION)
    missing, unknown = sorted(keys - set(params)), sorted(set(params) - keys)
    if missing or unknown:
        raise ValueError(f"params keys: missing {missing}, unknown {unknown}")
    if params["version"] != version:
        raise ValueError(f"version must be {version}")
    if not isinstance(params["commit"], str) or not HEX40.fullmatch(params["commit"]):
        raise ValueError("commit must be 40 lowercase hex characters")
    if not isinstance(params["order_sha256"], str) or not HEX64.fullmatch(params["order_sha256"]):
        raise ValueError("order_sha256 must be 64 lowercase hex characters")
    path_keys: tuple[str, ...] = (*PATH_KEYS, "volume_root") if v2 or v3 else PATH_KEYS
    if v3:
        path_keys += (*V3_PRIVATE_PATHS, *extra_paths)
    for key in path_keys:
        value = params[key]
        if (not isinstance(value, str) or not value.startswith("/") or value == "/"
                or posixpath.normpath(value) != value):
            raise ValueError(f"{key} must be a canonical absolute path: {value!r}")
    if posixpath.basename(params["model_path"]) != BASE_REVISION:  # load_base and export refuse anything else, but only after GPU time
        raise ValueError(f"model_path must be the pinned snapshot directory {BASE_REVISION}")
    _positive_int(params, "limit", 1)
    if v3:
        hash_keys = (*V3_HASH_KEYS, *extra_hashes)
        for key in hash_keys:
            if not isinstance(params[key], str) or not HEX64.fullmatch(params[key]):
                raise ValueError(f"{key} must be 64 lowercase hex characters")
        for key in (*V3_PRIVATE_PATHS, "build_root", "out_root", *extra_paths):
            if posixpath.commonpath((params["repo_root"], params[key])) == params["repo_root"]:
                raise ValueError(f"{key} must be outside repo_root (private evidence is not repository content)")
        if params["seqs"] != "auto" or params["kv_cache_gib"] != "auto" or params["token_budget"] != 24000:
            raise ValueError("v3 requires seqs=auto, kv_cache_gib=auto and token_budget=24000")
    elif v2:
        reuse = params["reuse_t1_sha256"]
        if reuse is not None and (not isinstance(reuse, str) or not HEX64.fullmatch(reuse)):
            raise ValueError("reuse_t1_sha256 must be null or 64 lowercase hex characters")
    else:
        _positive_int(params, "pilot_limit", 0)
    _positive_int(params, "sweep_seconds", 1)
    _positive_int(params, "max_num_batched_tokens", 1)
    _positive_int(params, "token_budget", 1)
    kv = params["kv_cache_gib"]
    if kv != "auto" and (isinstance(kv, bool) or not isinstance(kv, (int, float)) or not math.isfinite(kv) or kv <= 0):
        raise ValueError('kv_cache_gib must be a positive finite number or "auto"')
    seqs = params["seqs"]
    if seqs != "auto" and (not isinstance(seqs, list) or not seqs or any(type(s) is not int or s < 1 for s in seqs)):
        raise ValueError('seqs must be a non-empty list of positive integers or "auto"')
    if params["gradient_checkpointing"] not in ("on", "off"):
        raise ValueError('gradient_checkpointing must be "on" or "off"')
    arms = params.get("arms", ["note"])
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


def _plan_v2(p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order = f"{p['repo_root']}/fine_tuning/error_note/item_order_v2.json"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    collect_dir, limit, timeouts = f"{out}/collect", str(p["limit"]), p["timeouts"]
    run, export = f"{out}/train-note", f"{p['volume_root']}/eval-data/proof/export-note-v2"
    common = ["--build-root", build, "--order", order, "--model-path", model, "--kv-cache-gib", str(p["kv_cache_gib"]),
              "--max-num-batched-tokens", str(p["max_num_batched_tokens"]), "--engine", "vllm",
              "--limit", limit, "--output", collect_dir]
    steps = [_step("pin-order", [*chain_cmd, "pin", "--file", order, "--sha256", p["order_sha256"]], [], "collect", timeouts)]
    if p["reuse_t1_sha256"] is None:
        seqs = p["seqs"] if p["seqs"] == "auto" else ",".join(str(s) for s in p["seqs"])
        steps.append(_step("sweep", [*collect_cmd, "sweep", *common, "--seqs", seqs, "--seconds", str(p["sweep_seconds"])],
                           [f"{collect_dir}/sweep.json"], "collect", timeouts))
        concurrency = ["--max-num-seqs-from", f"{collect_dir}/sweep.json"]
    else:  # T1 copied from the v1 run (same ids, seeds and image budget): bound by its sha256, no sweep
        steps.append(_step("pin-t1", [*chain_cmd, "pin", "--file", f"{collect_dir}/t1.jsonl", "--sha256",
                                      p["reuse_t1_sha256"]], [], "collect", timeouts))
        concurrency = ["--max-num-seqs", "auto"]
    return steps + [
        _step("note", [*collect_cmd, "note", *common, *concurrency, "--note-version", "v2"],
              [f"{collect_dir}/note.jsonl", f"{collect_dir}/lessons.json"], "collect", timeouts),
        _step("resolve", [*collect_cmd, "resolve", *common, *concurrency],
              [f"{collect_dir}/receipt-resolve.json"], "collect", timeouts),
        _step("rows", [*collect_cmd, "rows", "--build-root", build, "--order", order, "--limit", limit,
                       "--tokenizer-path", model, "--output", collect_dir, "--rows-version", "v2"],
              [f"{collect_dir}/rows.jsonl"], "collect", timeouts),
        _step("train-note", [*train_cmd, "train", "--rows", f"{collect_dir}/rows.jsonl", "--build-root", build,
                             "--order", order, "--limit", limit, "--arm", "note", "--precision", "bf16",
                             "--model-path", model, "--output", run, "--code-commit", commit,
                             "--token-budget", str(p["token_budget"]), "--gradient-checkpointing",
                             p["gradient_checkpointing"], "--resume", "--protocol-version", "v2"],
              [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts),
        _step("reload-note", [*train_cmd, "reload", "--run", run, "--model-path", model, "--precision", "bf16",
                              "--protocol-version", "v2"], [f"{run}/reload.json"], "reload", timeouts),
        _step("export", [py, "-m", PROOF, "export", "--checkpoint", f"{run}/checkpoint-final",
                         "--binding", f"{run}/binding.json", "--base", model, "--output", export, "--code-commit", commit],
              [f"{export}/model_ref.json", f"{export}/model/export_manifest.json"], "export", timeouts),
    ]


def _plan_v3(stage: str, p: dict[str, Any]) -> list[dict[str, Any]]:
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order, timeouts, limit = p["order_path"], p["timeouts"], str(p["limit"])
    collect_dir = f"{out}/collect"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    common = ["--build-root", build, "--order", order, "--limit", limit, "--output", collect_dir,
              "--answer-contract", TRAINING_ANSWER_CONTRACT, "--training-rules", p["training_rules_path"],
              "--training-rules-sha256", p["training_rules_sha256"], "--code-commit", commit]
    engine = ["--model-path", model, "--kv-cache-gib", "auto", "--max-num-batched-tokens",
              str(p["max_num_batched_tokens"]), "--engine", "vllm"]
    pins = [("order", order, p["order_sha256"]),
            ("training-rules", p["training_rules_path"], p["training_rules_sha256"]),
            ("source-provenance", p["source_provenance_path"], p["source_provenance_sha256"])]
    if stage == "note-v3-resolve":
        pins += [("t1", f"{collect_dir}/t1.jsonl", p["t1_sha256"]),
                 ("note", f"{collect_dir}/note.jsonl", p["note_sha256"]),
                 ("reviews", p["reviews_path"], p["reviews_sha256"])]
    elif stage == "note-v3-train":
        pins += [("rows", p["rows_path"], p["rows_sha256"]),
                 ("audit", p["audit_path"], p["audit_sha256"]),
                 ("solution-audit", p["solution_audit_path"], p["solution_audit_sha256"])]
    steps = [_step(f"pin-{name}", [*chain_cmd, "pin", "--file", path, "--sha256", digest], [], "collect", timeouts)
             for name, path, digest in pins]
    if stage == "note-v3-collect":
        return steps + [
            _step("sweep", [*collect_cmd, "sweep", *common, *engine, "--seqs", "auto",
                            "--seconds", str(p["sweep_seconds"])], [f"{collect_dir}/sweep.json"], "collect", timeouts),
            _step("note", [*collect_cmd, "note", *common, *engine, "--max-num-seqs-from", f"{collect_dir}/sweep.json",
                           "--note-version", "v2"], [f"{collect_dir}/note.jsonl", f"{collect_dir}/lessons.json"], "collect", timeouts),
        ]
    if stage == "note-v3-resolve":
        audit, rows = f"{collect_dir}/audit.json", f"{collect_dir}/rows.jsonl"
        source = ["--source-provenance", p["source_provenance_path"]]
        return steps + [
            _step("audit", [*collect_cmd, "audit", *common, *source, "--reviews", p["reviews_path"]],
                  [audit, f"{collect_dir}/lessons-v3.json"], "collect", timeouts),
            _step("resolve", [*collect_cmd, "resolve", *common, *engine, "--max-num-seqs", "auto", *source, "--audit", audit],
                  [f"{collect_dir}/receipt-resolve.json"], "collect", timeouts),
            _step("rows", [*collect_cmd, "rows", *common, "--tokenizer-path", model, "--rows-version", "v3", *source, "--audit", audit],
                  [rows, f"{collect_dir}/rows_summary.json"], "collect", timeouts),
        ]
    audit, rows, admission = p["audit_path"], p["rows_path"], f"{out}/admission.json"
    contracts = ["--audit", audit, "--sidecar", p["training_rules_path"], "--solution-audit", p["solution_audit_path"]]
    train_common = ["--rows", rows, "--build-root", build, "--order", order, "--limit", limit,
                    "--precision", "bf16", "--model-path", model, "--code-commit", commit, *contracts]
    steps.append(_step("admit", [*train_cmd, "admit", *train_common, "--output", admission], [admission], "train", timeouts))
    for arm in ("note", "no-note"):
        run, export = f"{out}/train-{arm}", f"{p['volume_root']}/eval-data/proof/export-{arm}-v3"
        steps += [
            _step(f"train-{arm}", [*train_cmd, "train", *train_common, "--arm", arm, "--output", run,
                                  "--protocol-version", "v3", "--admission", admission, "--token-budget", "24000",
                                  "--gradient-checkpointing", p["gradient_checkpointing"], "--resume"],
                  [f"{run}/checkpoint-final/metadata.json", f"{run}/result.json"], "train", timeouts),
            _step(f"reload-{arm}", [*train_cmd, "reload", "--run", run, "--model-path", model,
                                   "--precision", "bf16", "--protocol-version", "v3"], [f"{run}/reload.json"], "reload", timeouts),
            _step(f"export-{arm}", [py, "-m", PROOF, "export", "--checkpoint", f"{run}/checkpoint-final",
                                   "--binding", f"{run}/binding.json", "--base", model, "--output", export, "--code-commit", commit],
                  [f"{export}/model_ref.json", f"{export}/model/export_manifest.json"], "export", timeouts),
        ]
    return steps


def plan(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    p = _check(stage, params)
    if stage in V3_STAGES:
        return _spec(stage, p, _plan_v3(stage, p))
    if stage == "note-v2":
        return _spec(stage, p, _plan_v2(p))
    py, out, model, build, commit = p["python"], p["out_root"], p["model_path"], p["build_root"], p["commit"]
    order = f"{p['repo_root']}/fine_tuning/error_note/item_order.json"
    collect_cmd, train_cmd, chain_cmd = [py, "-m", COLLECT], [py, "-m", TRAIN], [py, "-m", CHAIN]
    collect_dir, limit, timeouts = f"{out}/collect", str(p["limit"]), p["timeouts"]
    seqs = p["seqs"] if p["seqs"] == "auto" else ",".join(str(s) for s in p["seqs"])

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
                        "--seqs", seqs, "--seconds", str(p["sweep_seconds"])],
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
    return _spec(stage, p, steps)


def _spec(stage: str, p: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
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
