"""Remote-only exact benchmark exclusion inventory and TRACE audit downloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from mmdl.data.download import acquire
from mmdl.runtime.contracts import DATA_ID, DATA_REVISION, PRO_ID, PRO_REVISION, SUBJECTS

from .audit import dump, sha


def prepare_benchmarks(task_root):
    """Acquire EVERY released Parquet at pinned revisions for exclusion only.

    Gold-bearing records are never returned or mixed into training source rows.
    The separate validation symlinks are consumed only by the evaluator.
    """
    import pyarrow.parquet as pq

    if not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("HF payload download is remote-only")
    root = Path(task_root)
    target = root / "benchmark-exclusion"
    if target.exists():
        manifest = json.loads((target / "coverage.json").read_text())
        for record in manifest["files"]:
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Closed benchmark inventory changed")
        return manifest
    target.mkdir(parents=True)
    coverage, all_paths = [], []
    api = HfApi()
    for repo, revision, label in ((DATA_ID, DATA_REVISION, "mmmu"), (PRO_ID, PRO_REVISION, "mmmu_pro")):
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Benchmark revision mismatch")
        names = sorted(item.rfilename for item in info.siblings if item.rfilename.endswith(".parquet"))
        if not names:
            raise ValueError("Empty benchmark source inventory")
        manifest, snapshot = acquire(repo, revision, "dataset", names, root / "hf-cache")
        dump(target / f"{label}_downloads.json", manifest)
        checked = {record["path"]: record for record in manifest["files"]}
        for name in names:
            path = snapshot / name
            split = Path(name).name.split("-")[0]
            coverage.append({"path": str(path), "records": pq.ParquetFile(path).metadata.num_rows,
                "dataset_id": repo, "revision": revision, "configuration": str(Path(name).parent),
                "split": split, "sha256": checked[name]["sha256"]})
            all_paths.append(str(path))
        if repo == DATA_ID:
            validation = root / "data" / "evaluation" / "mmmu"
            validation.mkdir(parents=True, exist_ok=True)
            selected = []
            for subject in SUBJECTS:
                name = f"{subject}/validation-00000-of-00001.parquet"
                if name not in checked:
                    raise ValueError("Missing required validation subject")
                path = validation / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(os.path.relpath(snapshot / name, path.parent))
                selected.append(checked[name])
            dump(validation / "manifest.json", {"repo_id": repo, "revision": revision,
                "files": selected, "usage": "evaluation_only", "training_allowed": False})
    manifest = {"files": coverage, "record_count": sum(r["records"] for r in coverage),
        "inventory_policy": "Every Parquet in exact HF repository revisions; all splits/configurations",
        "all_paths": all_paths, "training_allowed": False, "public_upload_allowed": False}
    dump(target / "coverage.json", manifest)
    return manifest


def prepare_trace(task_root, inventory_path, shards=4):
    from .sources import download_exact

    root = Path(task_root)
    metadata = json.loads(Path(inventory_path).read_text())["sources"]["trace"]
    files = [r["path"] for r in metadata["files"] if r["path"].startswith("data/train/") and r["path"].endswith(".parquet")]
    files = sorted(files)[:shards]
    if len(files) != shards:
        raise ValueError("Requested TRACE source shard inventory incomplete")
    return download_exact(metadata["repo_id"], metadata["revision"], files, root / "hf-cache" / "hub",
                          root / "setup" / "trace-downloads.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["benchmarks", "trace"], required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--trace-shards", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "benchmarks":
        result = prepare_benchmarks(args.task_root)
        print(json.dumps({"status": "BENCHMARK_EXCLUSION_FILES_READY", "records": result["record_count"],
                          "files": len(result["files"])}), flush=True)
    else:
        if not args.inventory:
            parser.error("TRACE inventory required")
        print(json.dumps({"status": "TRACE_AUDIT_SHARDS_READY",
                          "paths": [str(p) for p in prepare_trace(args.task_root, args.inventory, args.trace_shards)]}), flush=True)


if __name__ == "__main__":
    main()
