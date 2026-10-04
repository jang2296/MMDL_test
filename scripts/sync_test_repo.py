"""Publish reviewed workspace sources to MMDL_test, never the submission MMDL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

REPOSITORY = "jang2296/MMDL_test"
CODE_DIRS = ("src", "tests", "scripts", "configs", "env", "prompts", "manifests",
             "third_party", "assignment", "reports", "experiments")
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".lock"}
AUTHORED_JSON = {
    "bridge_input_protocol_v1.json", "bridge_training_protocol_v1.json",
    "chartqa_transfer_source_v1.json", "circuit_clean_spec_v1.json", "clean_spec.json",
    "mixture_policy_v1.json", "protocol.json", "source_inventory_20261004.json",
    "source_registry_bridge_v2.json", "source_registry_v1.json",
}
EXPERIMENT_JSON = {
    "experiments/bottleneck_diagnostic/knowledge_cases.json",
    "experiments/bottleneck_diagnostic/knowledge_answer_key.json",
    "experiments/bottleneck_diagnostic/local_smoke_protocol.json",
    "experiments/bottleneck_diagnostic/protocol.json",
    "experiments/gemma_verifier/deployment.json",
}
SECRET = re.compile(rb"(?:hf_|sk-|gh[pousr]_|github_pat_|rpa_|rps_)[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
POOL = "fine_tuning/generated/graph_cs_bridge_split_pool32_v2"
PROOF = "fine_tuning/runs/recovered_graph_pool32_child_approval_v2/proof"
HISTORICAL = "fine_tuning/runs/recovered_control1600_final_v1/code/nvw_v18_control/fine_tuning"


def selected_files(root: Path) -> dict[str, Path]:
    files = {p.name: p for p in root.iterdir() if p.is_file()
             and (p.suffix == ".md" or p.name in {"pyproject.toml", ".env.example", ".gitattributes"})}
    for folder in CODE_DIRS + ("docs", "claudedocs", "fine_tuning", HISTORICAL):
      for base, directories, names in os.walk(root / folder, followlinks=False):
        directories[:] = [name for name in directories
                          if name not in {"__pycache__", ".git", "P0", "P1"}
                          and not name.endswith(".egg-info")
                          and not (folder == "fine_tuning" and name in {"runs", "generated"})]
        for name in names:
            path = Path(base) / name
            relative = path.relative_to(root)
            if any(part in {"__pycache__", ".git", "P0", "P1"} for part in relative.parts):
                continue
            if folder in {"docs", "claudedocs"} and path.suffix != ".md":
                continue
            authored_data = (path.suffix == ".json" and
                (folder in {"manifests", "configs"} or relative.as_posix() in EXPERIMENT_JSON
                 or folder in {"fine_tuning", HISTORICAL} and path.name in AUTHORED_JSON))
            plain_text = path.suffix == ".txt" and folder in {"env", "prompts"}
            if path.is_file() and (path.suffix in SUFFIXES or authored_data or plain_text
                                  or path.name in {"LICENSE", "test-repo.gitignore"}):
                files[relative.as_posix()] = path
    for folder in ("results/scoring-v6-20260924", "results/scoring-v8-20260924"):
        for path in (root / folder).glob("*"):
            if path.is_file() and path.suffix in {".json", ".csv"}:
                files[path.relative_to(root).as_posix()] = path
    manifest = json.loads((root / POOL / "manifest.json").read_text())
    files[f"{POOL}/manifest.json"] = root / POOL / "manifest.json"
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe own-fixture manifest path")
        path = root / POOL / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Own-fixture hash mismatch: {name}")
        files[f"{POOL}/{name}"] = path
    for arm in ("A0", "A1", "A2", "A3"):
        for name in (f"{arm}_processor.json", f"{arm}_approved/manifest.json", f"{arm}_approved/records.jsonl"):
            files[f"{PROOF}/{name}"] = root / PROOF / name
    pairing = "fine_tuning/generated/graph_cs_bridge_development_cohort_v1/pairing.jsonl"
    files[pairing] = root / pairing
    files[".gitignore"] = root / "env/test-repo.gitignore"
    for name, path in files.items():
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe publication path: {name}")
        if not path.is_file() or path.stat().st_size > 5 * 1024**2:
            raise ValueError(f"Missing or oversized source: {name}")
        data = path.read_bytes()
        if SECRET.search(data):
            raise ValueError(f"Possible credential; publication refused: {name}")
        if path.suffix == ".json" and not name.startswith((POOL, PROOF)):
            prohibited: tuple[str, ...] = ("raw_response", "gold_explanation")
            if name != "experiments/bottleneck_diagnostic/knowledge_cases.json":
                prohibited += ("question",)
            if any(f'"{k}"'.encode() in data for k in prohibited):
                raise ValueError(f"Benchmark payload in source metadata: {name}")
    return dict(sorted(files.items()))


def fingerprint(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), stat.S_IMODE(path.stat().st_mode)


def sync(root: Path, write: bool) -> dict:
    root = root.resolve()
    destination = root / "MMDL_test"
    if destination.is_symlink() or not (destination / ".git").is_dir():
        raise ValueError("Clone the real MMDL_test checkout first")
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=destination, text=True).strip()
    if remote not in {f"https://github.com/{REPOSITORY}.git", f"git@github.com:{REPOSITORY}.git"}:
        raise ValueError("Destination origin is not the approved MMDL_test repository")
    selected = selected_files(root)
    existing = {}
    for base, directories, names in os.walk(destination, followlinks=False):
      for directory in directories:
        if (Path(base) / directory).is_symlink():
            raise ValueError("Destination contains a symlink")
      directories[:] = [name for name in directories
                        if name not in {".git", "__pycache__", ".ruff_cache", ".mypy_cache", ".pytest_cache"}]
      for name in names:
        path = Path(base) / name
        if path.is_symlink():
            raise ValueError("Destination contains a symlink")
        if path.is_file():
            existing[path.relative_to(destination).as_posix()] = path
    extras = sorted(existing.keys() - selected.keys())
    if extras:
        raise ValueError("Preserve/reconcile destination-only files before syncing: " + ", ".join(extras))
    differences = [name for name, path in selected.items()
                   if name not in existing or fingerprint(path) != fingerprint(existing[name])]
    if write and differences:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=destination, text=True)
        incomplete_initial_copy = (bool(existing) and
            all(line.startswith("?? ") for line in dirty.splitlines()) and
            all(fingerprint(path) == fingerprint(selected[name]) for name, path in existing.items()))
        if dirty.strip() and not incomplete_initial_copy:
            raise ValueError("Destination has uncommitted work; sync refused")
        for name in differences:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected[name], target)
        differences = []
    entries = {name: {"sha256": fingerprint(path)[0], "bytes": path.stat().st_size}
               for name, path in selected.items()}
    return {"repository": REPOSITORY, "files": entries, "count": len(entries),
            "bytes": sum(value["bytes"] for value in entries.values()), "differences": differences,
            "published_GPU_success": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sync", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = sync(Path(__file__).resolve().parents[1], args.sync)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))
    if result["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
