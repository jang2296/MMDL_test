"""Create one allowlisted private SSH-transfer archive; never publish a repository."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from .ingest import read_rows, sha, write_json


def bundle(output, evidence, *, gemma_only=False):
    root = Path(__file__).resolve().parents[2]
    evidence = evidence.resolve(strict=True)
    output = output.resolve()
    if not output.is_relative_to(root / "analysis_exports"):
        raise ValueError("Bundle artifacts must remain in internal analysis_exports")
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise FileExistsError("Preserve previous transfer archive")
    include = ["experiments", "src/mmdl", "third_party/mmmu", "prompts",
               "env/requirements-vllm.lock", "manifests/models/baseline.json",
               "configs/eval/mmmu_val_v8.yaml", "configs/hardware/rtx3090_24gb.yaml"]
    if gemma_only:
        include = ["experiments/__init__.py", "experiments/gemma_verifier",
                   "experiments/bottleneck_diagnostic/__init__.py",
                   "experiments/bottleneck_diagnostic/finish_runpod.py",
                   "experiments/bottleneck_diagnostic/recovery.py",
                   "experiments/bottleneck_diagnostic/ingest.py",
                   "src/mmdl", "third_party/mmmu"]
    files = set()
    for name in include:
        path = root / name
        files.update(path.rglob("*") if path.is_dir() else [path])
    evidence_names = ["gemma_source_v8.jsonl"] if gemma_only else [
                 "prepared", "oracles/stress_reviewed.jsonl", "oracles/pilot_reviewed.jsonl",
                 "oracles/coverage_reviewed.jsonl", "oracles/review_receipt.json",
                 "oracles/pilot_review_receipt.json", "oracles/coverage_review_receipt.json",
                 "gemma_source_v8.jsonl"]
    for name in evidence_names:
        path = evidence / name
        files.update(path.rglob("*") if path.is_dir() else [path])
    files = sorted(p for p in files if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix != ".pyc"
                   and (not gemma_only or not p.is_relative_to(root / "experiments/gemma_verifier")
                        or p.parent in {root / "experiments/gemma_verifier",
                                        root / "experiments/gemma_verifier/tests"}))
    inventory = []
    merged = {}
    for name in (() if gemma_only else ("stress_reviewed.jsonl", "pilot_reviewed.jsonl", "coverage_reviewed.jsonl")):
        for row in read_rows(evidence / "oracles" / name):
            merged[row["sample_id"], row["level"]] = row
    oracle_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                           for _, row in sorted(merged.items())).encode()
    generated_path = str((evidence / "oracles/all_reviewed.jsonl").relative_to(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in files:
            relative = str(path.relative_to(root))
            if any(p in {"MMDL", "docs", "claudedocs", ".git", ".ssh"} for p in Path(relative).parts):
                raise ValueError("Non-allowlisted public/internal document or secret path")
            if path.is_symlink():
                raise ValueError("Transfer only real allowlisted files, no symlinks")
            inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
            archive.add(path, arcname="private_bundle/" + relative, recursive=False)
        if not gemma_only:
            inventory.append({"path": generated_path, "bytes": len(oracle_bytes),
                              "sha256": hashlib.sha256(oracle_bytes).hexdigest()})
            metadata = tarfile.TarInfo("private_bundle/" + generated_path)
            metadata.size = len(oracle_bytes)
            archive.addfile(metadata, io.BytesIO(oracle_bytes))
        manifest = json.dumps(inventory, sort_keys=True).encode()
        metadata = tarfile.TarInfo("private_bundle/TRANSFER_FILES.json")
        metadata.size = len(manifest)
        archive.addfile(metadata, io.BytesIO(manifest))
    receipt = {"status": "PRIVATE_BUNDLE_CREATED_NOT_TRANSFERRED", "archive": str(output),
               "sha256": sha(output), "bytes": output.stat().st_size, "files": len(inventory),
               "inventory_sha256": hashlib.sha256(manifest).hexdigest(),
               "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
               "public_upload_allowed": False, "private_source_worktree_not_public_commit": True}
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("analysis_exports/bottleneck_20261003"))
    parser.add_argument("--gemma-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(bundle(args.output, args.evidence, gemma_only=args.gemma_only), sort_keys=True))
