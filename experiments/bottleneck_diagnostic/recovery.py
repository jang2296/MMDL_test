"""Private allowlisted byte recovery; never certifies experiments or authorizes deletion."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from mmdl.runtime.bundle import _manifest_from_archive, _safe_relative, _sha256_stream
from mmdl.runtime.environment import check_storage

from .ingest import sha, write_json

POD_ID = "h447a4u6c0vq3k"
AUTHORIZED_PODS = {POD_ID, "re5qljwwyq7yui"}  # Exact creations and local ownership receipts only.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = REPOSITORY_ROOT / "analysis_exports"
MANIFEST = "bundle_manifest.json"
RECEIPT = "recovery_receipt.json"
KIND = "PRIVATE_BOTTLENECK_BYTES"
STABILITY = "FILES_STABLE_NOT_PROCESS_COMPLETION_CERT"
EXTENSIONS = {".json", ".jsonl", ".csv", ".log", ".txt", ".py", ".sh", ".yaml", ".yml",
              ".toml", ".lock", ".md"}
FORBIDDEN_PARTS = {"MMDL", ".git", ".ssh", ".aws", ".azure", ".config", "venv", "venvs",
                   ".venv", "hf", "huggingface", "cache", ".cache", "caches", "__pycache__",
                   "site-packages", "node_modules"}
SECRET_NAME = re.compile(r"(?:secret|credential|password|private[_-]?key|api[_-]?key|id_rsa|id_ed25519)", re.I)


def _syntax(value: str) -> None:
    if not value or "\\" in value or ":" in value or any(ord(c) < 32 for c in value):
        raise ValueError(f"Unsafe path syntax: {value!r}")


def _label(value: str) -> PurePosixPath:
    _syntax(value)
    path = _safe_relative(value)
    if str(path) != value or path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"Noncanonical path or unsupported extension: {value!r}")
    if (FORBIDDEN_PARTS & set(path.parts) or SECRET_NAME.search(value)
            or any(part.startswith(".") for part in path.parts)):
        raise ValueError(f"Forbidden cache, environment, public, or secret path: {value!r}")
    if value in {MANIFEST, RECEIPT}:
        raise ValueError("Reserved recovery metadata path")
    return path


def _path(value: Path) -> Path:
    _syntax(str(value))
    if ".." in value.parts or "MMDL" in value.parts:
        raise ValueError("Path escape/public checkout is forbidden")
    path = value.absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink path component: {part}")
    return path


def _regular(path: Path) -> tuple[int, int, int, int, int]:
    path = _path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Only regular source/archive files are allowed: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _source_scope(path: Path) -> Path:
    path = _path(path)
    if not any(path.is_relative_to(root.resolve())
               for root in (REPOSITORY_ROOT, PRIVATE_ROOT, Path("/workspace"))):
        raise ValueError("Source must stay inside the current repository, private root, or owned Pod /workspace")
    if FORBIDDEN_PARTS & set(path.parts) or SECRET_NAME.search(str(path)):
        raise ValueError("Forbidden source cache, environment, public, or secret path")
    return path


def _fresh_output(path: Path, *, receive: bool = False) -> Path:
    path = _path(path)
    private = path.is_relative_to(PRIVATE_ROOT.resolve())
    if path.is_relative_to(REPOSITORY_ROOT) and not private:
        raise ValueError("Repository outputs must stay inside private analysis_exports")
    if not private and (receive or not path.is_relative_to(Path("/workspace"))):
        raise ValueError("Output must be private: local analysis_exports, or Pod /workspace for pack")
    if path in {PRIVATE_ROOT.resolve(), Path("/workspace")} or path.exists():
        raise FileExistsError(f"Refusing existing/broad output: {path}")
    return path


def _space(path: Path, required: int) -> dict[str, Any]:
    result = check_storage([path], required)
    if not result["sufficient"]:
        raise RuntimeError("Insufficient free space including existing storage reserve")
    return result


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _private_manifest(archive: tarfile.TarFile) -> tuple[dict[str, Any], dict[str, tarfile.TarInfo]]:
    manifest, members = _manifest_from_archive(archive)
    stream = archive.extractfile(members[MANIFEST])
    if stream is None:
        raise ValueError("Unreadable manifest")
    with stream:
        if json.load(stream, object_pairs_hook=_object) != manifest:
            raise ValueError("Manifest JSON differs from validated inventory")
    if (manifest.get("schema") != "mmdl-runpod-bundle-v2" or manifest.get("kind") != KIND
            or manifest.get("pod_id") not in AUTHORIZED_PODS or manifest.get("caller_asserted_closed") is not True
            or manifest.get("stability_scope") != STABILITY
            or manifest.get("cleanup_authorized") is not False):
        raise ValueError("Not an explicitly closed-asserted private byte-recovery bundle")
    for name in members:
        if name != MANIFEST:
            path = _label(name)
            if any(str(parent) in members for parent in path.parents if str(parent) != "."):
                raise ValueError("File/directory member collision")
    if not manifest["files"]:
        raise ValueError("Empty allowlist is not a recovery bundle")
    if any(type(item["bytes"]) is not int for item in manifest["files"]):
        raise ValueError("Inventory byte counts must be integers, not booleans")
    return manifest, members


def pack(files: Path, output: Path, pod_id: str, *, assert_closed: bool) -> dict[str, Any]:
    if pod_id not in AUTHORIZED_PODS or not assert_closed:
        raise ValueError("Exact authorized Pod ID and caller --assert-closed are required")
    output = _fresh_output(output)
    if not (output.name.endswith(".tar") or output.name.endswith(".tar.gz")
            or output.name.endswith(".tgz")):
        raise ValueError("Output must be .tar, .tar.gz, or .tgz")
    sidecar = _fresh_output(Path(str(output) + ".sha256"))
    receipt = _fresh_output(Path(str(output) + ".receipt.json"))
    files = _source_scope(files)
    allowlist_state, allowlist_hash = _regular(files), sha(files)
    selected = json.loads(files.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if not isinstance(selected, dict) or not selected:
        raise ValueError("--files must contain a nonempty label:exact-filepath JSON object")
    sources, initial, inventory = {}, {}, []
    for label, value in sorted(selected.items()):
        if not isinstance(label, str) or not isinstance(value, str):
            raise ValueError("Allowlist labels and source paths must be strings")
        _label(label)
        path = _source_scope(Path(value))
        _label(path.name)
        state = _regular(path)
        sources[label] = path
        initial[label] = (state, sha(path))
        inventory.append({"path": label, "bytes": state[2], "sha256": initial[label][1]})
    if any(str(p) in sources for name in sources for p in PurePosixPath(name).parents if str(p) != "."):
        raise ValueError("File/directory allowlist collision")
    manifest = {"schema": "mmdl-runpod-bundle-v2", "kind": KIND, "pod_id": pod_id,
                "caller_asserted_closed": True, "stability_scope": STABILITY,
                "cleanup_authorized": False, "allowlist_sha256": allowlist_hash,
                "source_files": {label: str(path) for label, path in sources.items()},
                "files": inventory}
    encoded = (json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n").encode()
    storage = _space(output.parent, sum(item["bytes"] for item in inventory)
                     + len(encoded) + 10240 + 1024 * (len(inventory) + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Failed/partial outputs deliberately remain for diagnosis; never erase or overwrite them.
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz" if output.suffix in {".gz", ".tgz"} else "w") as archive:
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mode = len(encoded), 0o600
            archive.addfile(info, io.BytesIO(encoded))
            for label, path in sources.items():
                info = tarfile.TarInfo(label)
                info.size, info.mode = initial[label][0][2], 0o600
                with path.open("rb") as source:
                    archive.addfile(info, source)
    archive_state, digest = _regular(output), sha(output)
    # Archive member hashes validate the bytes captured DURING streaming against pre-read hashes.
    with tarfile.open(output, "r:*") as archive:
        _private_manifest(archive)
    for label, path in sources.items():
        if (_regular(path), sha(path)) != initial[label]:
            raise ValueError(f"Source mutated during packing; partial output retained: {label}")
    if (_regular(files), sha(files)) != (allowlist_state, allowlist_hash):
        raise ValueError("Allowlist mutated during packing; output retained")
    if (_regular(output), sha(output)) != (archive_state, digest):
        raise ValueError("Archive mutated during packing verification; output retained")
    result = {"status": STABILITY, "kind": KIND, "pod_id": pod_id,
              "archive_path": str(output), "archive_sha256": digest, "archive_bytes": archive_state[2],
              "file_count": len(inventory), "files": inventory, "storage": storage,
              "experiment_verified": False, "cleanup_authorized": False}
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {output.name}\n")
    write_json(receipt, result)
    return result


def receive(archive_path: Path, expected_sha256: str, output: Path) -> dict[str, Any]:
    output = _fresh_output(output, receive=True)
    archive_path = _path(archive_path)
    before = _regular(archive_path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or sha(archive_path) != expected_sha256:
        raise ValueError("Exact archive SHA256 mismatch")
    with archive_path.open("rb") as stream:
        with tarfile.open(fileobj=stream, mode="r:*") as archive:
            manifest, members = _private_manifest(archive)
            inventory = []
            for name, member in sorted(members.items()):
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable regular member: {name}")
                with source:
                    inventory.append({"path": name, "bytes": member.size,
                                      "sha256": _sha256_stream(source)})
            storage = _space(output.parent, sum(item["bytes"] for item in inventory))
            if _regular(archive_path) != before or sha(archive_path) != expected_sha256:
                raise ValueError("Archive mutated during pre-extraction verification")
            output.mkdir(parents=True, exist_ok=False)
            for item in inventory:
                target = output / _safe_relative(item["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(members[item["path"]])
                if source is None:
                    raise ValueError("Previously verified member became unreadable")
                with source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                if target.stat().st_size != item["bytes"] or sha(target) != item["sha256"]:
                    raise ValueError(f"Extracted size/hash mismatch: {item['path']}")
    if _regular(archive_path) != before or sha(archive_path) != expected_sha256:
        raise ValueError("Archive mutated during extraction; partial output retained")
    for item in inventory:
        target = output / item["path"]
        _regular(target)
        if target.stat().st_size != item["bytes"] or sha(target) != item["sha256"]:
            raise ValueError("Extracted inventory changed before receipt")
    result = {"status": "PRIVATE_BYTES_VERIFIED_NOT_EXPERIMENT_VERIFIED", "kind": KIND,
              "pod_id": manifest["pod_id"], "archive_sha256": expected_sha256,
              "archive_path": str(archive_path), "output_directory": str(output),
              "archive_bytes": before[2], "file_count": len(manifest["files"]),
              "inventory_includes_manifest": True, "files": inventory, "storage": storage,
              "caller_asserted_closed": True, "stability_scope": STABILITY,
              "experiment_verified": False, "cleanup_authorized": False}
    write_json(output / RECEIPT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack")
    packing.add_argument("--files", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--pod-id", choices=sorted(AUTHORIZED_PODS), required=True)
    packing.add_argument("--assert-closed", action="store_true", required=True)
    receiving = commands.add_parser("receive")
    receiving.add_argument("--archive", type=Path, required=True)
    receiving.add_argument("--sha256", required=True)
    receiving.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (pack(args.files, args.output, args.pod_id, assert_closed=args.assert_closed)
                  if args.command == "pack" else receive(args.archive, args.sha256, args.output))
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        print(json.dumps({"status": "RECOVERY_FAILED_PARTIAL_OUTPUTS_RETAINED", "error": str(exc),
                          "cleanup_authorized": False}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
