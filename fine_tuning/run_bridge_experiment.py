"""Fixed-module bounded Graph/CS or DC-circuit A0-A3 probes, not full G7."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter

from . import audit, bridge_evaluation, circuit_bridge, evaluate, run_bridge_smoke as bridge, run_experiment as runner
from .modules import ARMS, BASE_ID, BASE_REVISION
from .recover_evaluation import process_gone
from .report import compare_inputs, external_groups, paired_counts
from .run_control import checkpoint_layout, require_idle_gpu, verify_arm_training

CONDITIONS = ("A0", "A1", "A2", "A3")


def circuit_metric_gate(cohort, manifest):
    """A Circuit label cannot route rebound gold through the permissive V8 QA path."""
    if manifest.get("evaluation_domain") != bridge_evaluation.CIRCUIT_DOMAIN:
        return
    gold, ledger = (audit.read_rows(cohort/name) for name in ("gold.jsonl", "pairing.jsonl"))
    if (manifest.get("circuit_contract_version") != circuit_bridge.VERSION
            or manifest.get("pairing_sha256") != audit.sha(cohort/"pairing.jsonl")
            or len(gold) != manifest["count"] or len(ledger) != manifest["count"]
            or len({r["id"] for r in ledger}) != len(ledger)
            or [r["id"] for r in gold] != [r["id"] for r in ledger]
            or any(entry.get("view") not in {"qa", "structure", "grounded"}
                   or reference.get("metric_kind") != "circuit_"+entry["view"]
                   or reference.get("circuit_contract_version") != circuit_bridge.VERSION
                   for reference, entry in zip(gold, ledger, strict=True))):
        raise ValueError("Circuit gold metric/version/pairing contract mismatch")


def result_gate(output, cohort, protocol, checkpoint=None, evaluator_source=None, parent_evaluator_source=None):
    """Validate identity, observed inputs and independently replay stored CPU scores."""
    summary = runner.evaluation_gate(output, cohort, protocol, baseline=checkpoint is None)
    manifest = runner.cohort_gate(cohort)
    circuit_metric_gate(cohort, manifest)
    rows = audit.read_rows(output/"predictions.jsonl")
    inputs = audit.read_rows(cohort/"inputs.jsonl")
    gold = {row["id"]: row for row in audit.read_rows(cohort/"gold.jsonl")}
    external = manifest.get("evaluation_domain") in {bridge_evaluation.DOMAIN, bridge_evaluation.CIRCUIT_DOMAIN}
    source = evaluator_source or Path(evaluate.__file__)
    if (summary.get("evaluator_source_sha256") != audit.sha(source)
            or summary.get("benchmark_derived") is not (not external)
            or summary.get("training_allowed") is not False or summary.get("public_upload_allowed") is not False
            or checkpoint and (Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve()
                or summary.get("checkpoint_metadata_sha256") != audit.sha(checkpoint/"metadata.json"))
            or not math.isfinite(summary.get("seconds", 0)) or summary["seconds"] <= 0):
        raise ValueError("Evaluation checkpoint/source/domain/timing mismatch")
    if evaluator_source:
        # Older completed MMMU results are reusable only if the actual generation
        # functions are identical; the current CPU scorer must also reproduce them.
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name in {"digest", "sample_seed", "generation"}}
        original = functions(source)
        if len(original) != 3 or original != functions(Path(evaluate.__file__)) or external:
            raise ValueError("Baseline generation changed or historical graph scoring is incompatible")
        if summary.get("reused_completed", 0):
            if (parent_evaluator_source is None
                    or summary.get("parent_evaluator_source_sha256") != audit.sha(parent_evaluator_source)
                    or functions(parent_evaluator_source) != original):
                raise ValueError("Reused baseline prefix needs its own actual source/generation proof")
    for row, record in zip(rows, inputs, strict=True):
        if (any(row.get(field) is None for field in runner.PAIRED_INPUT_FIELDS)
                or type(row.get("correct")) is not bool
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("input_sha256", "")))
                or any(not isinstance(row.get(field), list) for field in ("prompt_token_ids", "output_token_ids", "image_grid_thw", "processed_image_sizes"))
                or any(type(token) is not int or token < 0 for field in ("prompt_token_ids", "output_token_ids") for token in row[field])
                or not row["prompt_token_ids"] or len(row["image_grid_thw"]) != len(record["image_paths"])
                or len(row["processed_image_sizes"]) != len(record["image_paths"])
                or any(not isinstance(size, list) or len(size) != length or any(type(n) is not int or n <= 0 for n in size)
                       for field,length in (("image_grid_thw", 3), ("processed_image_sizes", 2)) for size in row[field])
                or not isinstance(row.get("raw_response"), str) or row.get("finish_reason") not in {"stop", "length", "other"}
                or type(row.get("seed")) is not int or not math.isfinite(row.get("seconds", -1)) or row["seconds"] < 0):
            raise ValueError("Missing or invalid actual paired input/generation evidence")
        reference = gold[row["id"]]
        scored = evaluate.score_external(row["raw_response"], reference, row["finish_reason"]) if external else evaluate.score_with_parser(
            row["raw_response"], record["question_type"], record["options"], reference["answer"],
            "team-final-answer-v8", row["finish_reason"], question=record["question"])
        if (any(key not in row or row[key] != value for key, value in scored.items())
                or row.get("subject") != record["subject"]
                or row.get("seed") != evaluate.sample_seed(row["id"], 3407)
                or row.get("output_tokens") != len(row["output_token_ids"])
                or row.get("input_tokens") != len(row["prompt_token_ids"])):
            raise ValueError("Saved score/input/seed/token ledger differs from the frozen evaluation")
    correct = sum(row["correct"] is True for row in rows)
    if summary.get("correct") != correct or summary.get("accuracy") != correct/len(rows):
        raise ValueError("Saved evaluation arithmetic mismatch")
    if external and (summary.get("evaluation_domain") != manifest["evaluation_domain"]
                    or summary.get("metric_policy") != manifest["metric_policy"]):
        raise ValueError("Bridge development metric policy mismatch")
    return summary, rows


def graph_groups(cohort, before, after):
    ledger = audit.read_rows(cohort/"pairing.jsonl")
    if [r["id"] for r in ledger] != [r["id"] for r in after]:
        raise ValueError("Graph grouping ledger order differs")
    grouped = {}
    for field in ("condition", "root_id", "family_id"):
        grouped[field] = {}
        for value in sorted({row[field] for row in ledger}):
            selected = [i for i,row in enumerate(ledger) if row[field] == value]
            grouped[field][value] = external_groups([before[i] for i in selected], [after[i] for i in selected])
    return grouped


def training_bundle(args):
    datasets, frozen = {}, {}
    for condition in CONDITIONS:
        rows, manifest, records, contract, files = bridge.inputs(args.approval, args.image_root, args.input_protocol, condition)
        datasets[condition] = (rows, manifest, records, contract)
        frozen.update(files)
    reference_order = [(r["root_id"], r["view"]) for r in datasets["A0"][0]]
    if any([(r["root_id"], r["view"]) for r in rows] != reference_order
           or contract != datasets["A0"][3] for rows, _, _, contract in datasets.values()):
        raise ValueError("A0-A3 must share the exact root/view order and training input contract")
    graph = runner.cohort_gate(args.graph_cohort)
    circuit_metric_gate(args.graph_cohort, graph)
    circuit = graph.get("evaluation_domain") == bridge_evaluation.CIRCUIT_DOMAIN
    root_count = graph.get("root_count")
    if (type(root_count) is not int or root_count <= 0
            or graph.get("count") != 12*root_count or not circuit and graph.get("count") != 24
            or graph.get("evaluation_domain") not in {bridge_evaluation.DOMAIN, bridge_evaluation.CIRCUIT_DOMAIN}
            or any((rows[0]["source_id"] == circuit_bridge.SOURCE_ID) is not circuit
                   for rows, _, _, _ in datasets.values())
            or graph.get("builder_source_sha256") != audit.sha(Path(bridge_evaluation.__file__))
            or graph.get("pool_manifest_sha256") != audit.sha(args.image_root/"manifest.json")
            or graph.get("pairing_sha256") != audit.sha(args.graph_cohort/"pairing.jsonl")
            or graph.get("training_evidence") != [{"condition": c, "manifest_sha256": audit.sha(datasets[c][1]),
                "records_sha256": audit.sha(datasets[c][2]), "records": len(datasets[c][0])} for c in CONDITIONS]):
        raise ValueError("Common graph cohort/approval/source binding mismatch")
    audit.check_splits([r for rows, _, _, _ in datasets.values() for r in rows]
                       + audit.read_rows(args.graph_cohort/"pairing.jsonl"))
    for name, cohort in (("graph", args.graph_cohort), ("mmmu", args.mmmu_cohort)):
        manifest = runner.cohort_gate(cohort, graph["count"] if name == "graph" else 60)
        evaluate.verify_cohort_images(cohort, manifest)
        records = audit.read_rows(cohort/"inputs.jsonl")
        if (name == "graph" and manifest.get("benchmark_derived") is not False
                or name == "mmmu" and (manifest.get("dataset_id") != evaluate.DATA_ID
                    or manifest.get("dataset_revision") != evaluate.DATA_REVISION or manifest.get("split") != "validation"
                    or manifest.get("ids") != [r["id"] for r in records]
                    or Counter(r["subject"] for r in records) != {s: 2 for s in evaluate.SUBJECTS})
                or any(row.get("benchmark_derived") is not (name == "mmmu")
                    or row.get("training_allowed") is not False or row.get("public_upload_allowed") is not False for row in records)):
            raise ValueError("Benchmark provenance is not preserved")
        for filename in ("manifest.json", "inputs.jsonl", "gold.jsonl", "pairing.jsonl" if name == "graph" else "image_roles.jsonl"):
            frozen[str((cohort/filename).resolve())] = audit.sha(cohort/filename)
        for row in records:
            for filename in row["image_paths"]:
                path = audit.image_path(cohort, filename)
                frozen[str(path.resolve())] = audit.sha(path)
    return datasets, frozen


def reuse_a3(parent, datasets, args, reference, cfg, merge_size):
    """Rebind locally recovered paths without pretending their remote paths still exist."""
    if args.arm != "FL":
        raise ValueError("The existing A3 checkpoint is FL, not another fixed module")
    status, journal, frozen = runner.closed_controller(parent)
    if status["status"] != "BOUNDED_GRAPH_A3_GPU_SMOKE_COMPLETED":
        raise ValueError("Only the completed original A3 smoke is reusable here")
    rows, manifest, records, contract = datasets["A3"]
    output = parent/"A3_FL"
    metadata = verify_arm_training(output, rows, records, manifest, cfg, reference, "FL", contract)
    bridge.verify_processor_exposure(output, args.approval, "A3", metadata, merge_size)
    sources = json.loads((parent/"source_freeze.json").read_text())["sources"]
    declaration = json.loads((parent/"input_freeze.json").read_text())
    if (any(sources.get(n) != audit.sha(Path(__file__).with_name(n)) for n in ("train.py", "modules.py"))
            or declaration.get("condition") != "A3" or declaration.get("each_arm_starts_from") != "F0"
            or declaration.get("training_input_contract") != contract
            or declaration.get("spatial_merge_size") != merge_size
            or not set(bridge.inputs(args.approval, args.image_root, args.input_protocol, "A3")[4].values())
                <= set(declaration.get("files", {}).values())):
        raise ValueError("Reused A3 original input/source declaration differs")
    commands = {}
    for stage in ("train", "reload"):
        events = [e for e in journal if e.get("stage") == f"A3_FL_{stage}"]
        starts = [e for e in events if e.get("status") == "STARTING"]
        if not starts or events[-1].get("status") != "EXITED" or events[-1].get("returncode") != 0:
            raise ValueError("Reused A3 train/reload process was not closed successfully")
        commands[stage] = starts[-1]["command"]
    def argument(stage, flag):
        command = commands[stage]
        return command[command.index(flag)+1]
    original_output = Path(argument("train", "--output"))
    proof = [json.loads(line) for line in (parent/"A3_FL_reload.log").read_text().splitlines()
             if line.startswith('{"status": "RELOAD_VERIFIED"')]
    if (argument("train", "--arm") != "FL" or argument("train", "--seed") != "3407"
            or "--verify-checkpoint" in commands["train"]
            or Path(argument("reload", "--verify-checkpoint")) != original_output/"checkpoint"
            or argument("reload", "--model-path") != argument("train", "--model-path")
            or len(proof) != 1 or Path(proof[0]["checkpoint"]) != original_output/"checkpoint"
            or status.get("checkpoint_metadata_sha256") != audit.sha(output/"checkpoint/metadata.json")):
        raise ValueError("Original A3 fresh-F0/reload/path/hash proof mismatch")
    for name in ("input_freeze.json", "A3_FL_reload.log", "A3_FL/checkpoint/metadata.json", "A3_FL/consumed_ids.jsonl", "A3_FL/updates.jsonl"):
        frozen[str((parent/name).resolve())] = audit.sha(parent/name)
    frozen.update({str(audit.image_path(output/"checkpoint", n).resolve()): h for n,h in metadata["files"].items()})
    return output, metadata, frozen


def run(args, controller):
    if not process_gone({"pid": args.previous_pid, "pid_birth": args.previous_birth}):
        raise ValueError("The exact previous process is still live; no overlapping GPU work")
    _, _, frozen = runner.closed_controller(args.previous_controller)
    if args.arm not in ARMS or audit.sha(args.protocol) != audit.sha(Path(__file__).with_name("protocol.json")):
        raise ValueError("Fixed module and unchanged parent inference protocol required")
    datasets, files = training_bundle(args)
    frozen.update(files)
    contract = datasets["A0"][3]
    cfg = {**contract, "max_sequence_tokens": contract["sequence_budget"], "gradient_accumulation": 16}
    reference_path = args.reference_checkpoint/"metadata.json"
    reference = json.loads(reference_path.read_text())
    if (reference.get("arm") != args.arm or reference.get("base_id") != BASE_ID
            or reference.get("base_revision") != BASE_REVISION or not runner.exposure_matches(reference)):
        raise ValueError("Actual closed module inventory reference required; never used as initialization")
    frozen[str(reference_path.resolve())] = audit.sha(reference_path)
    frozen[str(args.protocol.resolve())] = audit.sha(args.protocol)
    model_config = args.model_path/"config.json"
    merge_size = json.loads(model_config.read_text())["vision_config"]["spatial_merge_size"]
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("Invalid pinned model vision merge size")
    frozen[str(model_config.resolve())] = audit.sha(model_config)
    bridge_manifest = json.loads((args.graph_cohort/"manifest.json").read_text())
    domain = bridge_manifest.get("evaluation_domain")
    if domain not in {bridge_evaluation.DOMAIN, bridge_evaluation.CIRCUIT_DOMAIN}:
        raise ValueError("Unknown bridge evaluation domain")
    circuit = domain == bridge_evaluation.CIRCUIT_DOMAIN
    external_name = "circuit" if circuit else "graph"
    cohorts = {external_name: args.graph_cohort, "mmmu": args.mmmu_cohort}
    baselines = {}
    if bool(args.baseline_mmmu) != bool(args.baseline_evaluator_source):
        raise ValueError("MMMU baseline reuse requires its actual original evaluator source")
    if args.baseline_mmmu:
        result_gate(args.baseline_mmmu, args.mmmu_cohort, args.protocol, evaluator_source=args.baseline_evaluator_source,
                    parent_evaluator_source=args.baseline_parent_evaluator_source)
        baselines["mmmu"] = args.baseline_mmmu
        for path in (args.baseline_mmmu/"summary.json", args.baseline_mmmu/"predictions.jsonl", args.baseline_evaluator_source,
                     args.baseline_parent_evaluator_source):
            if path:
                frozen[str(path.resolve())] = audit.sha(path)
    training, metadata = {}, {}
    if args.reuse_a3_controller:
        if circuit:
            raise ValueError("A historical Graph checkpoint cannot supply a Circuit condition")
        training["A3"], metadata["A3"], files = reuse_a3(args.reuse_a3_controller, datasets, args, reference, cfg, merge_size)
        frozen.update(files)
    prefix = "CIRCUIT" if circuit else "GRAPH"
    rows = datasets["A0"][0]
    declaration = {"kind": f"BOUNDED_{prefix}_FIXED_MODULE_A0_A3_ADDITIONAL_EXPERIMENT", "arm": args.arm,
        "module_selection": "explicit fixed recipe; not chosen from MMMU or a proven 10K winner",
        "each_condition_starts_from": "F0", "records_per_condition": len(rows),
        "roots_per_condition": len({r["root_id"] for r in rows}),
        "updates_per_condition": math.ceil(len(rows)/16), "seed": 3407, "training_input_contract": contract,
        "same_root_view_order": True, "equal_token_or_image_exposure_claimed": False,
        "bridge_evaluation_domain": domain, "common_bridge_evaluation_records": bridge_manifest["count"],
        "independent_bridge_roots": bridge_manifest["root_count"],
        "MMMU_validation_records": 60, "reuse_conditions": list(training),
        "training_cap_seconds": args.training_cap_seconds, "evaluation_cap_seconds": args.evaluation_cap_seconds,
        "reload_cap_seconds": 300, "full_professional_G7_completed": False, "public_upload_allowed": False,
        "files": frozen}
    audit.dump(controller.output/"input_freeze.json", declaration)
    frozen[str((controller.output/"input_freeze.json").resolve())] = audit.sha(controller.output/"input_freeze.json")
    if (controller.output/"launch.json").exists():
        frozen[str((controller.output/"launch.json").resolve())] = audit.sha(controller.output/"launch.json")
    if controller.remaining() < args.evaluation_cap_seconds+150:
        return {"status": "BUDGET_STOP", "GPU_started": False, "reason": "First evaluation and recovery reserve cannot fit"}

    def stage(name, command, limit):
        if any(audit.sha(Path(p)) != digest for p,digest in frozen.items()):
            raise ValueError("Frozen input/source/checkpoint evidence changed")
        for condition, output in training.items():
            checkpoint_layout(output/"checkpoint", metadata[condition])
        require_idle_gpu()
        if controller.remaining() < limit+150:
            raise TimeoutError("Stage cap plus termination/recovery reserve cannot fit")
        controller.stage(name, command, maximum_seconds=limit)

    def evaluate_stage(name, cohort, checkpoint=None):
        output = controller.output/name
        command = [sys.executable, "-m", "fine_tuning.evaluate", "run-external" if cohort == args.graph_cohort else "run",
            "--model-path", str(args.model_path), "--cohort", str(cohort), "--protocol", str(args.protocol),
            "--output", str(output), "--budget-seconds", str(args.evaluation_cap_seconds-60)]
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        stage(name, command, args.evaluation_cap_seconds)
        summary = json.loads((output/"summary.json").read_text())
        if summary["status"] == "BUDGET_STOP":
            raise TimeoutError("Partial evaluation preserved; no completed paired-score claim")
        result_gate(output, cohort, args.protocol, checkpoint)
        for filename in ("summary.json", "predictions.jsonl"):
            frozen[str((output/filename).resolve())] = audit.sha(output/filename)
        return output

    if "mmmu" not in baselines:
        baselines["mmmu"] = evaluate_stage("F0_mmmu", args.mmmu_cohort)
    baselines[external_name] = evaluate_stage(f"F0_{external_name}", args.graph_cohort)
    results, exposure_order = {}, None
    for condition in CONDITIONS:
        rows, manifest, records, _contract = datasets[condition]
        if condition not in training:
            output = controller.output/f"{condition}_{args.arm}"
            command = [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
                "--records", str(records), "--manifest", str(manifest), "--image-root", str(args.image_root),
                "--output", str(output), "--arm", args.arm, "--seed", "3407", "--accumulation", "16", "--max-steps", "0",
                "--input-protocol", str(args.input_protocol), "--budget-seconds", str(args.training_cap_seconds-60)]
            for flag in ("min_pixels", "max_pixels", "episode_pixels", "sequence_budget"):
                command.extend(["--"+flag.replace("_", "-"), str(contract[flag])])
            stage(f"{condition}_{args.arm}_train", command, args.training_cap_seconds)
            saved = json.loads((output/"checkpoint/metadata.json").read_text())
            if saved.get("termination") == "BUDGET_STOP":
                raise TimeoutError("Partial training checkpoint retained; no evaluation/comparison claim")
            metadata[condition] = verify_arm_training(output, rows, records, manifest, cfg, reference, args.arm, contract)
            bridge.verify_processor_exposure(output, args.approval, condition, metadata[condition], merge_size)
            for name in ("checkpoint/metadata.json", "consumed_ids.jsonl", "updates.jsonl"):
                frozen[str((output/name).resolve())] = audit.sha(output/name)
            frozen.update({str(audit.image_path(output/"checkpoint", n).resolve()): h for n,h in metadata[condition]["files"].items()})
            training[condition] = output
            stage(f"{condition}_{args.arm}_reload", [sys.executable, "-m", "fine_tuning.train", "--model-path", str(args.model_path),
                "--verify-checkpoint", str(output/"checkpoint")], 300)
            proof = [json.loads(line) for line in (controller.output/f"{condition}_{args.arm}_reload.log").read_text().splitlines()
                     if line.startswith('{"status": "RELOAD_VERIFIED"')]
            if len(proof) != 1 or Path(proof[0]["checkpoint"]).resolve() != (output/"checkpoint").resolve():
                raise ValueError("Fresh-process reload proof missing")
            training[condition] = output
        lookup = {r["id"]: (r["root_id"], r["view"]) for r in rows}
        observed = [lookup[r["id"]] for r in audit.read_rows(training[condition]/"consumed_ids.jsonl")]
        if exposure_order is not None and observed != exposure_order:
            raise ValueError("Condition consumed root/view order differs")
        exposure_order = observed
        results[condition] = {}
        for name, cohort in cohorts.items():
            output = evaluate_stage(f"{condition}_{args.arm}_{name}", cohort, training[condition]/"checkpoint")
            receipt = runner.paired_input_gate(output, baselines[name], cohort, args.protocol)
            audit.dump(controller.output/f"{condition}_{name}_paired_inputs.json", receipt)
            before, after = (audit.read_rows(p/"predictions.jsonl") for p in (baselines[name], output))
            compare_inputs(before, after)
            results[condition][name] = {"output": str(output.resolve()), "paired": paired_counts(before, after),
                "groups": external_groups(before, after) if name == external_name else {},
                f"{external_name}_membership_groups": graph_groups(cohort, before, after) if name == external_name else {}, "actual_inputs_equal": True}
        audit.dump(controller.output/f"{condition}_completed.json", {"condition": condition, "arm": args.arm,
            "checkpoint_metadata_sha256": audit.sha(training[condition]/"checkpoint/metadata.json"),
            "training_reused": condition == "A3" and bool(args.reuse_a3_controller),
            "exposure": metadata[condition]["exposure"], "results": results[condition]})
    contrasts = {}
    for after, before in (("A1", "A0"), ("A2", "A0"), ("A3", "A2"), ("A3", "A1")):
        contrasts[f"{after}-{before}"] = {}
        for name in cohorts:
            values = [audit.read_rows(Path(results[c][name]["output"])/"predictions.jsonl") for c in (before, after)]
            compare_inputs(*values)
            contrasts[f"{after}-{before}"][name] = {"paired": paired_counts(*values),
                "groups": external_groups(*values) if name == external_name else {},
                f"{external_name}_membership_groups": graph_groups(cohorts[name], *values) if name == external_name else {}}
    if any(audit.sha(Path(p)) != digest for p,digest in frozen.items()):
        raise ValueError("Final frozen input/checkpoint/evaluation evidence changed")
    for condition, output in training.items():
        checkpoint_layout(output/"checkpoint", metadata[condition])
    return {"status": f"BOUNDED_{prefix}_FIXED_MODULE_A0_A3_EVALUATED", "arm": args.arm,
        "training_exposures": {c: m["exposure"] for c,m in metadata.items()}, "results": results, "contrasts": contrasts,
        "MMMU_full_accuracy_claimed": False, f"{external_name}_development_only": True,
        "bridge_evaluation_domain": domain, "root_family_uncertainty_not_estimated": True,
        "full_common10K_G6_completed": False, "full_professional_G7_completed": False,
        "professional_non_synthetic_transfer_completed": False, "public_upload_allowed": False, "verified_files": frozen}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-cohort", "--graph-cohort", dest="graph_cohort", type=Path, required=True)
    for name in ("approval", "image-root", "input-protocol", "protocol", "mmmu-cohort",
                 "previous-controller", "reference-checkpoint", "model-path", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("baseline-mmmu", "baseline-evaluator-source", "baseline-parent-evaluator-source"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--reuse-a3-controller", type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--previous-pid", type=int, required=True)
    for name in ("previous-birth", "owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--"+name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--"+name, type=float, required=True)
    parser.add_argument("--training-cap-seconds", type=int, default=900)
    parser.add_argument("--evaluation-cap-seconds", type=int, default=3600)
    args = parser.parse_args()
    if (os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id or not args.owned_pod_id
            or args.previous_pid <= 0 or not args.previous_birth.isdecimal() or args.reserve_usd <= 0
            or not args.model_path.is_dir() or args.model_path.name != BASE_REVISION
            or args.training_cap_seconds <= 60 or args.evaluation_cap_seconds <= 60):
        parser.error("Exact owned Pod/process, positive caps/recovery reserve and pinned F0 snapshot required")
    deadline = runner.effective_deadline(runner.utc(args.gpu_cutoff_utc), runner.utc(args.billing_start_utc),
                                        args.balance_usd, args.reserve_usd, args.hourly_usd)
    controller = runner.Controller(args.output, deadline, Path(__file__).parent)
    audit.dump(args.output/"launch.json", {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    try:
        result = run(args, controller)
    except TimeoutError as error:
        result = {"status": "BUDGET_STOP", "reason": str(error)}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        result = {"status": "BLOCKED", "reason": f"{type(error).__name__}: {error}"}
    result.update(deadline_utc=deadline.isoformat(), stages=controller.stage_seconds,
        full_professional_G7_completed=False, public_upload_allowed=False)
    audit.dump(args.output/"status.json", result)
    controller.event(stage="controller", **result)
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
