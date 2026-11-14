"""Offline selection, single-forward judging and provenance-checked reporting."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from functools import wraps
from pathlib import Path
import re
import sys

import yaml

from mmdl.evaluation.parsers import scoring_source_files
from mmdl.runtime.artifacts import atomic_text, default_path, digest, read_json, sha256_file, write_json

from . import CHECKER_VERSION
from . import backend, prompt, scoring, selection


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def _config():
    config = selection.load_config()
    selection.verify_base_parser(config, ROOT)
    return config


def _binding(config):
    return {"checker_version": CHECKER_VERSION,
            "config_sha256": sha256_file(HERE / "checker_v1.yaml"),
            "prompt_sha256": prompt.CV_PROMPT_SHA256,
            "v8_scoring_sha256": digest([config["base_parser"], *map(
                sha256_file, scoring_source_files(config["base_parser"]))])}


def _private(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Benchmark artifacts must be outside the repository")
    return path


def _jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL must contain objects")
            result.append(row)
    return result


def _write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                      allow_nan=False) + "\n" for row in rows))


def _vacant(directory, names):
    for name in names:
        if (directory / name).exists():
            raise ValueError(f"Refusing to overwrite {directory / name}")


def _exclusive_stage(stage):
    def decorate(function):
        @wraps(function)
        def run(args, **kwargs):
            source = args.targets if stage == "judge" else args.verdicts
            lock = _private(Path(source).parent) / f".{stage}.lock"
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            try:
                return function(args, **kwargs)
            finally:
                lock.unlink()
        return run
    return decorate


def _profile(args, config):
    reportable = backend.check_mode(args.mode, args.model, args.revision,
                                    args.quantization, config)
    limit = getattr(args, "limit", None)
    if limit is not None and (args.mode != "smoke" or limit < 1):
        raise ValueError("--limit must be positive and is allowed only in smoke mode")
    return {"mode": args.mode, "reportable": reportable, "model": args.model,
            "revision": args.revision, "quantization": args.quantization}


def _directory(args, identity):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", args.run_label):
        raise ValueError("Invalid --run-label")
    profile = identity["profile"]
    name = (f"{profile['mode']}-{profile['model'].split('/')[-1]}-"
            f"{profile['quantization']}-{digest(identity)[:12]}")
    directory = _private(args.output_root) / args.run_label / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _source_records(paths):
    return [{"path": str(Path(path).resolve()), "sha256": sha256_file(path)} for path in paths]


def _encoding(tokenizer):
    return (lambda text: tokenizer.encode(text, add_special_tokens=False),
            lambda ids: tokenizer.decode(ids, skip_special_tokens=False))


def select_run(args, *, tokenizer=None):
    config = _config()
    profile = _profile(args, config)
    rows = selection.load_rows(args.predictions)
    selection.replay_v8(rows)
    if tokenizer is None:
        tokenizer = backend.load_tokenizer(args.model, args.revision,
                                            path=getattr(args, "tokenizer", None))
        encode, decode = _encoding(tokenizer)
    else:
        encode, decode = tokenizer.encode, tokenizer.decode
    targets = selection.build_targets(rows, config, encode, decode)
    identity = {**_binding(config), "profile": profile,
                "source_predictions_sha256": _source_records(args.predictions)}
    directory = _directory(args, identity)
    _write_jsonl(directory / "targets.jsonl", targets)
    write_json(directory / "selection.json", {
        **identity, "denominator": len(rows), "v8_correct": sum(r["correct"] for r in rows),
        "categories": dict(Counter(selection.categorize(r) for r in rows)),
        "target_items": len({t["id"] for t in targets}), "target_inputs": len(targets),
        "targets_sha256": sha256_file(directory / "targets.jsonl")})
    return directory


def _selection_manifest(target_path, config):
    target_path = Path(target_path)
    manifest = read_json(target_path.parent / "selection.json")
    for key, value in _binding(config).items():
        if manifest.get(key) != value:
            raise ValueError(f"Selection identity mismatch: {key}")
    if manifest.get("targets_sha256") != sha256_file(target_path):
        raise ValueError("targets.jsonl SHA256 mismatch")
    targets = _jsonl(target_path)
    seen = set()
    for target in targets:
        key = (target["id"], target["alias_index"])
        if key in seen or target["category"] not in selection.TARGET_CATEGORIES:
            raise ValueError("Duplicate or forbidden target")
        seen.add(key)
        expected = digest([CHECKER_VERSION, prompt.CV_PROMPT_SHA256,
                           target["question"], target["gold_answer"], target["llm_response"]])
        if expected != target["input_sha256"]:
            raise ValueError("Target input SHA256 mismatch")
    if (manifest["target_inputs"] != len(targets)
            or manifest["target_items"] != len({t["id"] for t in targets})):
        raise ValueError("Target counts mismatch")
    return manifest, targets


def _limited(targets, limit):
    if limit is None:
        return targets
    ids = list(dict.fromkeys(t["id"] for t in targets))[:limit]
    return [t for t in targets if t["id"] in ids]


def _judge_profile(identity, profile):
    if identity.get("model") == "FakeJudge" and profile["mode"] == "smoke":
        return
    for field in ("model", "revision", "quantization"):
        if identity.get(field) != profile[field]:
            raise ValueError(f"Loaded judge identity mismatch: {field}")
    if identity.get("dtype") != "bfloat16":
        raise ValueError("Loaded judge must compute in bfloat16")


def _cases():
    cases = yaml.safe_load((HERE / "checkset_v1.yaml").read_text(encoding="utf-8"))
    if cases["version"] != "compassverifier-checkset-v1":
        raise ValueError("Unknown checkset version")
    ids = [case["id"] for case in cases["cases"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate checkset cases")
    if any(case["expect"] not in ("accept", "not_accept") for case in cases["cases"]):
        raise ValueError("Invalid checkset expectation")
    return cases["cases"]


def _require_checkset(receipt_path, profile, config):
    if receipt_path is None:
        raise ValueError("Official judge requires --checkset-receipt with zero false accepts")
    receipt = read_json(receipt_path)
    for key, value in {**_binding(config), "profile": profile,
                       "checkset_sha256": sha256_file(HERE / "checkset_v1.yaml")}.items():
        if receipt.get(key) != value:
            raise ValueError(f"Official checkset receipt mismatch: {key}")
    identity = receipt["judge"]
    if any(identity.get(key) != profile[key] for key in ("model", "revision", "quantization")):
        raise ValueError("Official checkset model mismatch")
    if identity.get("dtype") != "bfloat16":
        raise ValueError("Official checkset requires BF16")
    cases = _cases()
    results = receipt["results"]
    if len(results) != len(cases):
        raise ValueError("Incomplete official checkset")
    false_accepts = 0
    for case, result in zip(cases, results, strict=True):
        if result["id"] != case["id"] or result["expect"] != case["expect"]:
            raise ValueError("Official checkset cases mismatch")
        decision = scoring.verdict(result, identity["label_token_ids"],
                                    config["decision"]["accept_threshold"])
        false_accepts += case["expect"] == "not_accept" and decision == "ACCEPT"
    if false_accepts or receipt.get("false_accept_count") != 0:
        raise ValueError("Official checkset failed: false accepts present")
    return sha256_file(receipt_path)


@_exclusive_stage("judge")
def judge_run(args, *, judge=None):
    config = _config()
    profile = _profile(args, config)
    directory = _private(Path(args.targets).parent)
    _vacant(directory, ("verdicts.jsonl", "judge.json", "checkset.json"))
    manifest, targets = _selection_manifest(args.targets, config)
    if manifest["profile"] != profile:
        raise ValueError("Selected and requested judge profiles differ")
    receipt_sha = None
    if profile["reportable"]:
        receipt_sha = _require_checkset(args.checkset_receipt, profile, config)
    targets = _limited(targets, args.limit)
    if judge is None:
        judge = backend.load_judge(args.model, args.revision, quantization=args.quantization)
    _judge_profile(judge.identity, profile)
    results = []
    for index, target in enumerate(targets, 1):
        probability = judge.probabilities(prompt.build(target["question"], target["gold_answer"],
                                                       target["llm_response"]))
        scoring.verdict(probability, judge.identity["label_token_ids"],
                        config["decision"]["accept_threshold"])
        results.append({key: target[key] for key in ("id", "alias_index", "input_sha256")} | probability)
        print(f"judge {index}/{len(targets)}", file=sys.stderr, flush=True)
    if profile["reportable"]:
        atomic_text(directory / "checkset.json", args.checkset_receipt.read_text(encoding="utf-8"))
    _write_jsonl(directory / "verdicts.jsonl", results)
    write_json(directory / "judge.json", {
        **_binding(config), **profile, "judge": judge.identity, "limit": args.limit,
        "scope": "smoke-target-subset" if args.limit is not None else "complete-source",
        "targets_sha256": manifest["targets_sha256"],
        "verdicts_sha256": sha256_file(directory / "verdicts.jsonl"),
        "checkset_receipt_sha256": receipt_sha})
    return directory


@_exclusive_stage("report")
def report_run(args, *, tokenizer=None):
    config = _config()
    directory = _private(Path(args.verdicts).parent)
    _vacant(directory, ("merged.jsonl", "summary.json", "review.md"))
    manifest, stored_targets = _selection_manifest(args.targets, config)
    source = _source_records(args.predictions)
    if [r["sha256"] for r in source] != [r["sha256"] for r in manifest["source_predictions_sha256"]]:
        raise ValueError("Source predictions SHA256 mismatch")
    rows = selection.load_rows(args.predictions)
    selection.replay_v8(rows)
    profile = manifest["profile"]
    backend.check_mode(profile["mode"], profile["model"], profile["revision"],
                       profile["quantization"], config)
    if tokenizer is None:
        tokenizer = backend.load_tokenizer(profile["model"], profile["revision"])
        encode, decode = _encoding(tokenizer)
    else:
        encode, decode = tokenizer.encode, tokenizer.decode
    targets = selection.build_targets(rows, config, encode, decode)
    if stored_targets != targets:
        raise ValueError("Recomputed targets differ from selection")
    judged = read_json(Path(args.verdicts).parent / "judge.json")
    _judge_profile(judged["judge"], profile)
    for key, value in {**_binding(config), **profile,
                       "targets_sha256": manifest["targets_sha256"]}.items():
        if judged.get(key) != value:
            raise ValueError(f"Judge identity mismatch: {key}")
    if judged["verdicts_sha256"] != sha256_file(args.verdicts):
        raise ValueError("Verdicts SHA256 mismatch")
    if profile["reportable"]:
        receipt_sha = _require_checkset(directory / "checkset.json", profile, config)
        if judged.get("checkset_receipt_sha256") != receipt_sha:
            raise ValueError("Official checkset receipt identity mismatch")
    limit = judged["limit"]
    if limit is not None and (profile["mode"] != "smoke" or type(limit) is not int or limit < 1):
        raise ValueError("Invalid judge limit")
    expected_scope = "smoke-target-subset" if limit is not None else "complete-source"
    if judged["scope"] != expected_scope:
        raise ValueError("Judge scope disagrees with limit")
    source_denominator = len(rows)
    targets = _limited(targets, limit)
    if limit is not None:
        ids = {t["id"] for t in targets}
        rows = [r for r in rows if r["id"] in ids]
    results = _jsonl(args.verdicts)
    merged = scoring.merge(rows, targets, results,
                           config | {"label_token_ids": judged["judge"]["label_token_ids"]})
    identity = {**_binding(config), **profile, "judge": judged["judge"],
                "source_predictions_sha256": manifest["source_predictions_sha256"],
                "source_denominator": source_denominator, "scope": judged["scope"]}
    summary = scoring.summarize(merged, identity)
    _write_jsonl(directory / "merged.jsonl", merged)
    atomic_text(directory / "review.md", scoring.review_markdown(merged, rows, results))
    write_json(directory / "summary.json", summary)
    if profile["reportable"]:
        n = summary["denominator"]
        for label, key in (("V8", "v8_correct"), ("V8+checker", "final_correct")):
            count = summary[key]
            suffix = f" ({100 * count / n:.2f}%)" if n == 900 else ""
            print(f"{label:13}: {count}/{n}{suffix}")
        print("reportable=true", summary["verdicts"])
    else:
        print("Local smoke completed; reportable=false; not an evaluation score.")
    return directory


def checkset_run(args, *, judge=None):
    config = _config()
    profile = _profile(args, config)
    cases = _cases()
    identity = {**_binding(config), "profile": profile,
                "checkset_sha256": sha256_file(HERE / "checkset_v1.yaml")}
    directory = _directory(args, identity)
    if judge is None:
        judge = backend.load_judge(args.model, args.revision, quantization=args.quantization)
    _judge_profile(judge.identity, profile)
    results = []
    for index, case in enumerate(cases, 1):
        response, truncated = selection.tail(case["llm_response"],
                                              config["input"]["response_tail_tokens"],
                                              judge.encode, judge.decode)
        probability = judge.probabilities(prompt.build(case["question"], case["gold_answer"], response))
        decision = scoring.verdict(probability, judge.identity["label_token_ids"],
                                    config["decision"]["accept_threshold"])
        results.append({"id": case["id"], "expect": case["expect"], "verdict": decision,
                        "response_truncated": truncated, **probability})
        print(f"checkset {index}/{len(cases)}", file=sys.stderr, flush=True)
    negatives = [r for r in results if r["expect"] == "not_accept"]
    positives = [r for r in results if r["expect"] == "accept"]
    false_accepts = sum(r["verdict"] == "ACCEPT" for r in negatives)
    receipt = {**identity, "judge": judge.identity, "results": results,
               "false_accept_count": false_accepts,
               "accept_recall": sum(r["verdict"] == "ACCEPT" for r in positives) / len(positives)}
    write_json(directory / "checkset.json", receipt)
    print(f"Checkset: false_accept_count={false_accepts}; accept_recall={receipt['accept_recall']:.4f}")
    if profile["reportable"] and false_accepts:
        raise ValueError(f"Official checkset failed; receipt preserved at {directory}")
    return directory


def parser():
    config = selection.load_config()
    official = config["judge"]["official"]
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("select", "judge", "report", "run", "checkset"):
        sub = commands.add_parser(command)
        if command in ("select", "judge", "run", "checkset"):
            sub.add_argument("--mode", choices=("smoke", "official"), default="official")
            sub.add_argument("--model", default=official["model"])
            sub.add_argument("--revision", default=official["revision"])
            sub.add_argument("--quantization", choices=("none", "nf4"), default="none")
        if command in ("select", "run", "checkset"):
            sub.add_argument("--run-label", required=True)
            sub.add_argument("--output-root", type=Path, default=default_path(
                "MMDL_ARTIFACT_ROOT", "mmdl-artifacts") / "compassverifier_checker")
        if command in ("select", "run", "report"):
            sub.add_argument("--predictions", type=Path, nargs="+", required=True)
        if command in ("select", "run"):
            sub.add_argument("--tokenizer", type=Path)
        if command in ("judge", "report"):
            sub.add_argument("--targets", type=Path, required=True)
        if command in ("judge", "run"):
            sub.add_argument("--limit", type=int)
            sub.add_argument("--checkset-receipt", type=Path)
        if command == "report":
            sub.add_argument("--verdicts", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            directory = select_run(args)
            args.targets = directory / "targets.jsonl"
            judge_run(args)
            args.verdicts = directory / "verdicts.jsonl"
            directory = report_run(args)
        else:
            directory = {"select": select_run, "judge": judge_run,
                         "report": report_run, "checkset": checkset_run}[args.command](args)
        print(directory)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"checker: {error}", file=sys.stderr)
        return 2
