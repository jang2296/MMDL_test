"""Immutable bridge from the original training checkout to a frozen evaluator."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess

from .audit import digest
from .prep_v16b import file_sha, read_sealed

VERSION = "v17-training-artifacts-v1"
TRAINING_VERSIONS = ("error-note-v17", "error-note-v18")
ROOT = Path(__file__).resolve().parents[2]


def _safe(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("v17 artifacts: symlink or path escape refused")
    return path


def _ref(path: Path) -> dict:
    path = _safe(path)
    if not path.is_file():
        raise ValueError("v17 artifacts: bound file is missing")
    return {"path": str(path), "file_sha256": file_sha(path)}


def _check_ref(ref: dict) -> Path:
    path = _safe(Path(ref["path"]))
    if _ref(path) != ref:
        raise ValueError("v17 artifacts: bound file bytes changed")
    return path


def _files(root: Path) -> list[dict]:
    root = _safe(root)
    if not root.is_dir():
        raise ValueError("v17 artifacts: artifact directory is missing")
    records = []
    for path in sorted(root.rglob("*")):
        _safe(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("v17 artifacts: nonregular artifact file refused")
        records.append({"path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size, "sha256": file_sha(path)})
    if not records:
        raise ValueError("v17 artifacts: empty artifact directory")
    return records


def _tree(root: Path) -> dict:
    records = _files(root)
    return {"root": str(_safe(root)), "files": records, "files_sha256": digest(records)}


def protected_evaluation(repo: Path) -> dict:
    """Hash protected evaluation bytes and the preregistration/ledger format, not engine glue."""
    repo = _safe(repo)
    paths = sorted((repo / "src/mmdl/evaluation").rglob("*.py"))
    paths += sorted((repo / "prompts").glob("mmmu_*_v1.txt"))
    paths += [repo / "configs/eval/mmmu_val_v8.yaml",
              repo / "fine_tuning/notebook_rl/eval_configs/mmmu_val_v8_b2_seed3407.yaml",
              repo / "third_party/mmmu/eval_utils.py", repo / "src/mmdl/runtime/contracts.py"]
    if not paths or not (repo / "src/mmdl/evaluation/engine.py").is_file():
        raise ValueError("v17 artifacts: protected evaluation source missing")
    records = [{"path": p.relative_to(repo).as_posix(), **{k: v for k, v in _ref(p).items() if k != "path"}}
               for p in paths]
    path = _safe(repo / "fine_tuning/notebook_rl/proof_eval.py")
    module = ast.parse(path.read_text(encoding="utf-8"))
    names = {"load_preregistration", "read_ledger", "_check_event", "_parse", "_append", "ledger_start", "ledger_fail",
             "_single_definition", "_paired_run_rules", "_variant_entry", "_scorer", "variant_registration"}
    contract = [ast.dump(node, include_attributes=False) for node in module.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (node.name.startswith("build_preregistration") or node.name in names)]
    if not contract or not any(isinstance(n, ast.FunctionDef) and n.name == "load_preregistration" for n in module.body):
        raise ValueError("v17 artifacts: preregistration contract missing")
    body = {"files": records, "preregistration_contract_sha256": digest(contract)}
    return body | {"sha256": digest(body)}


def evaluation_identity(repo: Path, commit: str) -> dict:
    from . import single

    repo = _safe(repo)
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != commit or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("v17 artifacts: frozen evaluation commit differs")
    if repo.resolve() != ROOT.resolve():
        raise ValueError("v17 artifacts: evaluator must run from its declared checkout")
    return {"code_commit": commit, "source_sha256": digest(single.source_manifest()["actual_files"])}


def create_training_artifacts(args, plan: dict) -> dict:
    from . import campaign_v16b as campaign
    from .v17_preflight import _write, bind_training
    from fine_tuning.notebook_rl.checkpoint import read_metadata

    protocol_version = plan["protocol"].get("version")
    if protocol_version not in TRAINING_VERSIONS:
        raise ValueError("v17 artifacts: unsupported training protocol")
    binding = bind_training(args, plan)
    if not binding.get("production_admitted"):
        raise ValueError("v17 artifacts: local smoke cannot seal production artifacts")
    output = _safe(Path(args.output))
    ready = campaign.require_complete(output, evaluation_decision=getattr(args, "evaluation_decision", None))
    checkpoint = _safe(Path(ready["checkpoint"]))
    metadata = read_metadata(checkpoint, binding=campaign._binding(plan))
    if metadata["protocol"] != plan["protocol"] or metadata["step"] != plan["total_slots"]:
        raise ValueError("v17 artifacts: final checkpoint identity changed")
    reload_path = output / f"reload-{plan['total_slots']}.json"
    reload = read_sealed(reload_path)
    if (reload.get("plan_sha256") != plan["sha256"] or reload.get("checkpoint") != ready["metadata_descriptor"]
            or reload.get("runtime_equal") is not True or reload.get("greedy_equal") is not True
            or reload.get("probe_max_abs_difference") != 0
            or not isinstance(reload.get("producer_pid"), int) or not isinstance(reload.get("reload_pid"), int)
            or reload["producer_pid"] == reload["reload_pid"]):
        raise ValueError("v17 artifacts: exact fresh-process final reload required")
    admission_path = output / "evaluation-admission.json"
    admission = read_sealed(admission_path)
    if protocol_version == "error-note-v17":
        decision = read_sealed(args.evaluation_decision)
        if (decision.get("decision") != "PROCEED" or decision.get("authorized_by") != "user"
                or decision.get("plan_sha256") != plan["sha256"] or decision.get("admission_sha256") != admission["sha256"]):
            raise ValueError("v17 artifacts: evaluation decision differs")
    export = _tree(Path(args.export_output))
    model_ref = _safe(Path(args.export_output) / "model_ref.json")
    model = json.loads(model_ref.read_text(encoding="utf-8"))
    if (model.get("code_commit") != binding["code_commit"] or model.get("kind") != "unmerged-lora"
            or model.get("dtype") != "bfloat16" or model.get("export", {}).get("binding") != campaign._binding(plan)
            or model.get("export", {}).get("campaign", {}).get("plan_sha256") != plan["sha256"]):
        raise ValueError("v17 artifacts: export is not the bound final unmerged model")
    if protocol_version == "error-note-v18" and model["export"].get("training_protocol") != plan["protocol"]:
        raise ValueError("v18 artifacts: exported training protocol differs from sealed plan")
    for record in model["files"]:
        path = _safe(Path(args.export_output) / "model" / record["path"])
        if (not path.is_relative_to(_safe(Path(args.export_output) / "model"))
                or path.stat().st_size != record["bytes"] or file_sha(path) != record["sha256"]):
            raise ValueError("v17 artifacts: model-ref file inventory differs")
    code_root = ROOT / "fine_tuning"
    code_files = [{"path": name, "file_sha256": _ref(code_root / name)["file_sha256"]}
                  for name in sorted(plan["source"]["actual_files"])]
    if {r["path"]: r["file_sha256"] for r in code_files} != plan["source"]["actual_files"]:
        raise ValueError("v17 artifacts: training source files changed")
    body = {"version": VERSION, "status": "SEALED", "training": {k: binding[k] for k in
            ("code_commit", "source_sha256", "campaign_plan_sha256", "protocol_sha256")},
        "training_code": {"root": str(code_root), "files": code_files, "files_sha256": digest(code_files)},
        "campaign_plan": _ref(output / "campaign-plan.json"),
        "checkpoint": _tree(checkpoint) | {"metadata": _ref(checkpoint / "metadata.json")},
        "export": export | {"model_ref": _ref(model_ref)}, "reload": _ref(reload_path),
        "functional_receipt": binding["functional_receipt"], "update_cost_receipt": binding["update_cost_receipt"],
        "evaluation_admission": _ref(admission_path),
        "export_ready": _ref(output / "export-ready.json"), "protected_evaluation": protected_evaluation(ROOT)}
    if protocol_version == "error-note-v18":
        body.update(training_protocol_version=protocol_version, monitor=_ref(output / "monitor-final.json"),
                    effect=_ref(output / "effect-final.json"))
        _v18_admission(body)
    else:
        body["evaluation_decision"] = _ref(args.evaluation_decision)
    destination = Path(getattr(args, "artifact_output", None) or output / "training-artifacts.json")
    if any(_safe(destination).is_relative_to(Path(t["root"])) for t in (body["checkpoint"], body["export"])):
        raise ValueError("v17 artifacts: receipt cannot be inside its bound artifact tree")
    return _write(destination, body)


def verify_training_artifacts(path: Path) -> dict:
    value = read_sealed(_safe(path))
    if value.get("version") != VERSION or value.get("status") != "SEALED":
        raise ValueError("v17 artifacts: sealed production training artifacts required")
    for name in ("campaign_plan", "reload", "functional_receipt", "update_cost_receipt",
                 "evaluation_admission", "export_ready"):
        _check_ref(value[name])
    if value.get("training_protocol_version") != "error-note-v18":
        _check_ref(value["evaluation_decision"])
    elif "evaluation_decision" in value:
        raise ValueError("v18 artifacts: training seal cannot authorize evaluation")
    for name in ("checkpoint", "export"):
        tree = value[name]
        if _tree(Path(tree["root"])) != {k: tree[k] for k in ("root", "files", "files_sha256")}:
            raise ValueError("v17 artifacts: checkpoint/export inventory changed")
    _check_ref(value["checkpoint"]["metadata"])
    _check_ref(value["export"]["model_ref"])
    for name, key, filename in (("checkpoint", "metadata", "metadata.json"),
                                 ("export", "model_ref", "model_ref.json")):
        if value[name][key] != _ref(Path(value[name]["root"]) / filename):
            raise ValueError("v17 artifacts: identity file escapes its artifact tree")
    code = value["training_code"]
    if digest(code["files"]) != code["files_sha256"]:
        raise ValueError("v17 artifacts: training code inventory changed")
    for record in code["files"]:
        source = _safe(Path(code["root"]) / record["path"])
        if not source.is_relative_to(_safe(Path(code["root"]))) or file_sha(source) != record["file_sha256"]:
            raise ValueError("v17 artifacts: original training source changed")
    plan = read_sealed(Path(value["campaign_plan"]["path"]))
    protocol_version = plan["protocol"].get("version")
    model = json.loads(Path(value["export"]["model_ref"]["path"]).read_text(encoding="utf-8"))
    if (protocol_version not in TRAINING_VERSIONS
            or (protocol_version == "error-note-v18" and (
                value.get("training_protocol_version") != protocol_version
                or model.get("export", {}).get("training_protocol") != plan["protocol"]))
            or (protocol_version == "error-note-v17" and value.get("training_protocol_version") not in (None, protocol_version))):
        raise ValueError("v17 artifacts: training protocol version was relabeled")
    if (plan["sha256"] != value["training"]["campaign_plan_sha256"]
            or plan["source"]["base_commit"] != value["training"]["code_commit"]
            or digest(plan["source"]["actual_files"]) != value["training"]["source_sha256"]
            or plan["protocol"]["sha256"] != value["training"]["protocol_sha256"]
            or {r["path"]: r["file_sha256"] for r in code["files"]} != plan["source"]["actual_files"]):
        raise ValueError("v17 artifacts: training identity was relabeled")
    observed = {p.relative_to(Path(code["root"])).as_posix(): file_sha(_safe(p))
                for p in Path(code["root"]).rglob("*.py")
                if not {"tests", "runs", "generated", "__pycache__"}.intersection(p.relative_to(Path(code["root"])).parts)}
    actual_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code["root"], check=True,
                                   capture_output=True, text=True).stdout.strip()
    if observed != plan["source"]["actual_files"] or actual_commit != value["training"]["code_commit"]:
        raise ValueError("v17 artifacts: original training checkout changed")
    if protocol_version == "error-note-v18":
        _v18_admission(value)
    return value


def _v18_admission(artifacts: dict) -> dict:
    """A training seal preserves the final admission, including failures, without approving an evaluation."""
    from .v17_eval import evaluation_gate

    plan = read_sealed(_check_ref(artifacts["campaign_plan"]))
    admission = read_sealed(_check_ref(artifacts["evaluation_admission"]))
    monitor = read_sealed(_check_ref(artifacts["monitor"]))
    effect = read_sealed(_check_ref(artifacts["effect"]))
    ready = read_sealed(_check_ref(artifacts["export_ready"]))
    checkpoint = artifacts["checkpoint"]["metadata"]
    if (plan["protocol"]["version"] != "error-note-v18"
            or artifacts["training"]["campaign_plan_sha256"] != plan["sha256"]
            or any(row.get("plan_sha256") != plan["sha256"] or row.get("checkpoint") != checkpoint
                   for row in (admission, monitor, effect))
            or ready.get("plan_sha256") != plan["sha256"] or ready.get("metadata_descriptor") != checkpoint
            or admission.get("monitor_sha256") != monitor["sha256"] or admission.get("effect_sha256") != effect["sha256"]
            or admission.get("decision") != evaluation_gate(monitor["comparison"], effect["comparison"])):
        raise ValueError("v18 artifacts: final admission/monitor/effect/checkpoint binding differs")
    return admission


def _v18_evaluation_decision(artifacts: dict, artifact_ref: dict, identity: dict, authorization: str) -> dict:
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError("v18 evaluation: separate non-empty user authorization required")
    admission = _v18_admission(artifacts)
    status = admission["decision"]["status"]
    if status not in ("EVAL_REVIEW_REQUIRED", "EVALUATION_CANDIDATE"):
        raise ValueError("v18 evaluation: DEGRADED or incomplete admission cannot be authorized")
    body = {"version": "v18-evaluation-decision-v1", "decision": "PROCEED", "authorized_by": "user",
            "pre_authorized": False, "authorization": authorization,
            "training_artifact": artifact_ref, "plan_sha256": artifacts["training"]["campaign_plan_sha256"],
            "admission": artifacts["evaluation_admission"], "admission_sha256": admission["sha256"],
            "admission_status": status, "checkpoint": artifacts["checkpoint"]["metadata"],
            "model_ref": artifacts["export"]["model_ref"], "evaluation": identity}
    return body | {"sha256": digest(body)}


def verify_evaluation_authorization(path: Path, training_artifacts: Path, identity: dict) -> dict:
    """Recheck consent against live sealed evidence before engine validation and every final/resumed run."""
    verified = read_sealed(_safe(path))
    artifacts = verify_training_artifacts(training_artifacts)
    ref = _ref(training_artifacts)
    decision = verified.get("evaluation_decision", {})
    if (artifacts.get("training_protocol_version") != "error-note-v18"
            or verified.get("version") != "v17-training-artifact-verification-v1" or verified.get("status") != "PASS"
            or verified.get("training_artifact") != ref or verified.get("training") != artifacts["training"]
            or verified.get("evaluation") != identity
            or verified.get("protected_evaluation_sha256") != artifacts["protected_evaluation"]["sha256"]
            or decision != _v18_evaluation_decision(artifacts, ref, identity, decision.get("authorization"))):
        raise ValueError("v18 evaluation: authorization belongs to another artifact, admission or evaluator")
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-protocol", choices=TRAINING_VERSIONS)
    parser.add_argument("--authorization")
    args = parser.parse_args(argv)
    value = verify_training_artifacts(args.receipt)
    version = read_sealed(Path(value["campaign_plan"]["path"]))["protocol"]["version"]
    if args.expected_protocol is not None and version != args.expected_protocol:
        raise ValueError("v17 artifacts: training seal does not match expected protocol")
    identity = evaluation_identity(args.repo, args.commit)
    if protected_evaluation(args.repo) != value["protected_evaluation"]:
        raise ValueError("v17 artifacts: protected evaluation/preregistration contract changed")
    from .v17_preflight import _write
    body = {"version": "v17-training-artifact-verification-v1", "status": "PASS",
        "training_artifact": _ref(args.receipt), "training": value["training"], "evaluation": identity,
        "protected_evaluation_sha256": value["protected_evaluation"]["sha256"]}
    if version == "error-note-v18":
        if args.expected_protocol != version:
            raise ValueError("v18 evaluation: explicit --expected-protocol error-note-v18 required")
        body["evaluation_decision"] = _v18_evaluation_decision(value, _ref(args.receipt), identity, args.authorization)
    _write(args.output, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
