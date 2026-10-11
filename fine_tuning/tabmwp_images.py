"""Acquire only frozen TabMWP PNGs; byte integrity never grants training approval."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from . import tabmwp
from .audit import dump, image_path, read_rows, sha

TREE_SHA = "88e72de0a0708f29fa4ba3440ce223c4bbca202e"
API = "https://api.github.com/repos/lupantech/PromptPG/git/"
WORKERS = 8
MAX_ATTEMPTS = 3


def git_hash(kind, data):
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def download(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "MMDL-TabMWP-pinned-acquisition"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Response exceeds declared acquisition bound")
    return data


def verify_tree(commit, tree):
    """Recompute every Git tree object, including the pinned repository root."""
    if (commit.get("sha") != tabmwp.REVISION or commit.get("tree", {}).get("sha") != TREE_SHA
            or tree.get("sha") != TREE_SHA or tree.get("truncated") is not False):
        raise ValueError("Wrong commit, root tree, or truncated tree")
    entries, children = {}, defaultdict(list)
    modes = {"tree": {"040000"}, "blob": {"100644", "100755", "120000"}, "commit": {"160000"}}
    for entry in tree.get("tree", []):
        path = entry.get("path")
        if (not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
                or any(p in ("", ".", "..") for p in path.split("/"))
                or path in entries or entry.get("mode") not in modes.get(entry.get("type"), set())
                or not re.fullmatch(r"[0-9a-f]{40}", entry.get("sha", ""))):
            raise ValueError("Malformed or unsafe Git tree entry")
        entries[path] = entry
        parent, _, name = path.rpartition("/")
        children[parent].append((name, entry))
    directories = {"": TREE_SHA, **{p: e["sha"] for p, e in entries.items() if e["type"] == "tree"}}
    if set(children) - set(directories):
        raise ValueError("Missing parent tree")
    for parent, expected in directories.items():
        objects = sorted(children[parent], key=lambda item: (
            item[0] + ("/" if item[1]["type"] == "tree" else "")).encode())
        payload = b"".join(e["mode"].lstrip("0").encode() + b" " + name.encode()
                           + b"\0" + bytes.fromhex(e["sha"]) for name, e in objects)
        if git_hash("tree", payload) != expected:
            raise ValueError(f"Git tree identity differs: {parent or 'root'}")
    return entries


def selection(prepared_root):
    """Bind immutable A/B records to the native fixed selection, without emitting gold."""
    prepared = Path(prepared_root)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashed = dict(manifest)
    expected = hashed.pop("preparation_hash", None)
    if (expected != tabmwp.digest(hashed) or manifest.get("revision") != tabmwp.REVISION
            or manifest.get("source_id") != tabmwp.SOURCE_ID
            or manifest.get("training_allowed") is not False):
        raise ValueError("Prepared manifest identity differs")
    source = prepared.parent
    problems, receipt = tabmwp.load_source(source)
    if (manifest.get("source_receipt_sha256") != sha(source / "source_receipt.json")
            or manifest.get("source_files") != receipt["files"]):
        raise ValueError("Prepared source receipt differs")
    policy = manifest["selection"].get("train_candidate_exclusion")
    if policy is not None and tabmwp.exclusion_policy(source, problems, prepared / "exclusion_receipt.json") != policy:
        raise ValueError("Frozen candidate exclusion policy or scan evidence differs")
    variants, planned = tabmwp.plan(problems, policy["excluded_train_families"] if policy else ())
    if manifest["selection"]["selected_ids"] != planned["selected_ids"]:
        raise ValueError("Frozen fixed selection differs from native deterministic plan")
    paths, bound = {}, {str(manifest_path): sha(manifest_path)}
    if policy:
        bound[str(prepared / "exclusion_receipt.json")] = policy["scan_sha256"]
        bound.update({name: value for name, value in policy["source_and_index_sha256"].items()
                      if name != policy["historical_converter_path"]})
    for variant, groups in variants.items():
        for role, expected_rows in groups.items():
            name = f"{variant}/{role}.jsonl"
            path = prepared / name
            if manifest["files"][name]["sha256"] != sha(path) or read_rows(path) != expected_rows:
                raise ValueError("Frozen records differ from native converter")
            bound[str(path)] = sha(path)
            for row in expected_rows:
                relative = row["image_paths"][0]
                identity = {"id": row["id"], "role": role}
                if relative in paths and paths[relative] != identity:
                    raise ValueError("Conflicting image selection")
                paths[relative] = identity
    for name in ["source_receipt.json", *tabmwp.ARTIFACTS]:
        bound[str(source / name)] = sha(source / name)
    if len(paths) != sum(tabmwp.COUNTS.values()):
        raise ValueError("Selected image count differs from fixed plan")
    return paths, bound, expected


def verify_png(data, entry):
    if (entry.get("type") != "blob" or entry.get("mode") != "100644"
            or type(entry.get("size")) is not int or len(data) != entry["size"]
            or git_hash("blob", data) != entry["sha"] or not data.startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Pinned PNG blob identity differs")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.width <= 0 or image.height <= 0:
            raise ValueError("Invalid PNG dimensions/format")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        size = list(image.size)
        rgb = image.convert("RGB")
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": entry["sha"], "size": size, "pixels_sha256": pixels}


def acquire(prepared_root, destination):
    """New bundle, or bounded resume of its exact incomplete acquisition; no overwrite."""
    paths, bound, preparation_hash = selection(prepared_root)
    root = Path(destination)
    if root.is_symlink():
        raise ValueError("Symlink destination forbidden")
    identity = {"revision": tabmwp.REVISION, "preparation_hash": preparation_hash,
                "selected_paths": paths, "source_and_prepared_sha256": bound}
    if root.exists():
        if json.loads((root / "acquisition.json").read_text()) != identity:
            raise ValueError("Existing acquisition belongs to another selection")
        attempts = sorted(root.glob("receipt_*.json"))
        if len(attempts) >= MAX_ATTEMPTS:
            raise ValueError("Bounded acquisition attempts exhausted; preserved files require review")
        if attempts and json.loads(attempts[-1].read_text()).get("status") == "COMPLETE":
            raise FileExistsError("Complete image acquisition is immutable")
        commit = json.loads((root / "commit.json").read_text())
        tree = json.loads((root / "tree.json").read_text())
    else:
        commit = json.loads(download(API + "commits/" + tabmwp.REVISION, 1024 * 1024))
        tree = json.loads(download(API + "trees/" + TREE_SHA + "?recursive=1", 32 * 1024 * 1024))
        verify_tree(commit, tree)
        root.mkdir(parents=True, exist_ok=False)
        dump(root / "acquisition.json", identity)
        dump(root / "commit.json", commit)
        dump(root / "tree.json", tree)
        attempts = []
    entries = verify_tree(commit, tree)
    selected = {}
    for relative in paths:
        native = "data/tabmwp/" + relative
        entry = entries.get(native)
        if (entry is None or entry.get("type") != "blob" or entry.get("mode") != "100644"
                or type(entry.get("size")) is not int or not 0 < entry["size"] <= 8 * 1024 * 1024):
            raise ValueError(f"Selected PNG absent or unbounded in pinned tree: {relative}")
        selected[relative] = entry
    unexpected = {p.relative_to(root).as_posix() for p in root.rglob("*.png")} - set(paths)
    if unexpected:
        raise ValueError("Unselected PNGs in acquisition destination")
    files, errors = {}, {}
    def fetch(relative):
        entry = selected[relative]
        path = root / relative
        url = tabmwp.RAW_URL + "data/tabmwp/" + relative
        if path.exists() or path.is_symlink():
            data = image_path(root, relative).read_bytes()
        else:
            data = download(url, entry["size"])
            verify_png(data, entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Existing data, including partial files, are never replaced.
            if path.parent.is_symlink():
                raise ValueError("Symlink image directory forbidden")
            with path.open("xb") as stream:
                stream.write(data)
        return {**verify_png(data, entry), **paths[relative], "url": url}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, relative): relative for relative in sorted(selected)}
        for future in as_completed(futures):
            relative = futures[future]
            try:
                files[relative] = future.result()
            except Exception as error:
                errors[relative] = f"{type(error).__name__}: {error}"
            if (len(files) + len(errors)) % 100 == 0:
                print(f"images checked {len(files) + len(errors)}/{len(paths)}; failed {len(errors)}", flush=True)
    changed = [name for name, digest in bound.items() if sha(name) != digest]
    receipt = {"status": "COMPLETE" if not errors and not changed and set(files) == set(paths) else "INCOMPLETE",
               "revision": tabmwp.REVISION, "tree_sha1": TREE_SHA, "preparation_hash": preparation_hash,
               "selection_sha256": tabmwp.digest(paths), "expected_count": len(paths),
               "verified_count": len(files), "total_verified_bytes": sum(v["bytes"] for v in files.values()),
               "files": files, "errors": errors, "changed_sources": changed,
               "tree_metadata_sha256": sha(root / "tree.json"),
               "commit_metadata_sha256": sha(root / "commit.json"),
               "dataset_license": tabmwp.DATASET_LICENSE, "rights_approved": False,
               "training_allowed": False, "public_upload_allowed": False,
               "human_semantic_certification": False, "pixel_semantics_verified": False,
               "verification_scope": "Pinned PNG bytes, decode, dimensions and pixel hash only"}
    dump(root / f"receipt_{len(attempts) + 1:02d}.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    receipt = acquire(args.prepared_root, args.destination)
    print(json.dumps({k: receipt[k] for k in ("status", "verified_count", "total_verified_bytes")}))
    if receipt["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
