"""Out-of-place FP32 fusion → sharded BF16 HF export, with immutable sources."""
from __future__ import annotations

import json
import re
import shutil
import stat
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from safetensors.torch import save_file

from fine_tuning.modules import BASE_ID, BASE_REVISION, tensor_sha256
from fine_tuning.notebook_rl.checkpoint import PolicyFactory, digest, file_hash, load_eval, read_metadata
from fine_tuning.notebook_rl.policy import assert_base_unchanged, merged_named_tensors
from fine_tuning.notebook_rl.resources import resource_path
from fine_tuning.notebook_rl.vault import _read, _safe_root, _unique

BASE_MANIFEST = resource_path("manifests/models/baseline.json")
_AUXILIARY = {"config.json", "generation_config.json", "chat_template.json", "merges.txt", "vocab.json",
              "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"}


def validate_export_identity(model_path: Path, arm: str) -> dict:
    """Validate a completed derived export without the historical evaluator package."""
    root = _safe_root(Path(model_path))
    manifest = json.loads(_read(root / "export_manifest.json"), object_pairs_hook=_unique)
    precision = {"fusion": "FP32", "export": "BF16",
        "merger_masters": "FP32 checkpoint preserved; export rounds to BF16",
        "unmerged_inference_bit_equivalent": False}
    if (not isinstance(arm, str) or not arm or not isinstance(manifest, dict) or manifest.get("status") != "EXPORTED"
            or manifest.get("format") != "derived-hf-bf16-v1" or manifest.get("arm") != arm
            or manifest.get("base_id") != BASE_ID or manifest.get("base_revision") != BASE_REVISION
            or manifest.get("precision_policy") != precision
            or manifest["precision_policy"]["unmerged_inference_bit_equivalent"] is not False):
        raise ValueError("export: complete pinned BF16 derived identity required")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("export: nonempty file/hash inventory required")
    for name, expected in files.items():
        if (not isinstance(name, str) or name in ("", ".", "..")
                or len(Path(name).parts) != 1 or PureWindowsPath(name).is_absolute()
                or len(PureWindowsPath(name).parts) != 1
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("export: unsafe model file/hash inventory")
        path = root / name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not path.resolve(strict=True).is_relative_to(root.resolve()) or file_hash(path) != expected):
            raise ValueError("export: model file/hash mismatch")
    return {"arm": arm, "manifest_sha256": digest(manifest), "files": files,
            "derived_precision_policy": manifest["precision_policy"]}


def _validate_base(model_path: Path, manifest_path: Path) -> tuple[dict, dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not model_path.is_dir() or model_path.name != BASE_REVISION
            or manifest["repo_id"] != BASE_ID or manifest["revision"] != BASE_REVISION):
        raise ValueError("export: validated local pinned base required; no downloads")
    files: dict[str, str] = {}
    for row in manifest["files"]:
        name = row["path"]
        if (Path(name).is_absolute() or len(Path(name).parts) != 1 or name in files
                or (model_path / name).stat().st_size != row["bytes"]
                or file_hash(model_path / name) != row["sha256"]):
            raise ValueError("export: pinned base file mismatch")
        files[name] = row["sha256"]
    if (not _AUXILIARY <= files.keys()
            or {p.name for p in model_path.glob("*.safetensors")} != {n for n in files if n.endswith(".safetensors")}
            or json.loads((model_path / "config.json").read_text())["model_type"] != "qwen3_vl"):
        raise ValueError("export: incomplete processor/tokenizer/config or weight inventory")
    return manifest, files


def validate_export_reference(model_path: Path, reference: dict) -> dict:
    """Bind the evaluated files to their declared checkpoint, not just a file inventory."""
    root, export = Path(model_path), reference["export"]
    identity = validate_export_identity(root, export["arm"])
    manifest_path = root / "export_manifest.json"
    manifest = json.loads(_read(manifest_path), object_pairs_hook=_unique)
    artifact_sha = file_hash(manifest_path)
    files = manifest["files"] | {"export_manifest.json": artifact_sha}
    inventory = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": sha}
                 for name, sha in sorted(files.items())]
    if (set(p.name for p in root.iterdir()) != set(files) or reference.get("files") != inventory
            or reference.get("artifact_revision") != artifact_sha
            or export.get("export_manifest_sha256") != identity["manifest_sha256"]
            or reference.get("train_config_sha256") != manifest["resolved_protocol_sha256"]
            or export.get("checkpoint_state_sha256") != digest(manifest["checkpoint_state_sha256"])
            or any(export.get(key) != manifest[key] for key in
                   ("checkpoint_files", "binding", "base_manifest_sha256", "mergers", "precision_policy"))):
        raise ValueError("export: evaluated files and declared checkpoint reference differ")
    return manifest


def export_merged(ckpt: Path, model_path: Path, out: Path, arm: str, *, binding: dict,
                  max_shard_size: int = 2_000_000_000,
                  _test_policy_factory: PolicyFactory | None = None,
                  _test_base_manifest: Path | None = None) -> dict:
    """Streaming shards contain canonical unique parameters, so tied weights are not duplicated.

    HF reconstructs tied lm_head/embed weights from the copied pinned config. A single
    parameter larger than max_shard_size occupies its own shard (the standard HF rule).
    Test-only injection is explicit; production always uses the pinned 4B policy builder.
    """
    ckpt, model_path, out = Path(ckpt), Path(model_path), Path(out)
    protected = Path(__file__).resolve().parents[2] / "MMDL"
    if (out.resolve().is_relative_to(protected) or out.resolve().is_relative_to(model_path.resolve())
            or out.resolve().is_relative_to(ckpt.resolve())):
        raise ValueError("export: output must be outside sources/protected submission")
    if out.exists() or out.is_symlink():
        raise FileExistsError("export: destination must be new")
    if type(max_shard_size) is not int or max_shard_size <= 0:
        raise ValueError("export: positive shard byte limit required")
    if (_test_policy_factory is None) != (_test_base_manifest is None):
        raise ValueError("export: test factory and test base manifest must be paired explicitly")
    manifest_path = _test_base_manifest or BASE_MANIFEST
    manifest_sha = file_hash(manifest_path)
    _, base_files = _validate_base(model_path, manifest_path)
    metadata = read_metadata(ckpt, binding=binding)
    if arm != metadata["arm"]:
        raise ValueError("export: checkpoint arm differs")
    source_checkpoint = {p.name: file_hash(p) for p in ckpt.iterdir()}
    bundle = load_eval(model_path, ckpt, binding=binding, _test_policy_factory=_test_policy_factory)
    before = {name: tensor_sha256(p) for name, p in bundle.model.named_parameters()}
    out.mkdir(parents=True, exist_ok=False)
    # Canonical names are deduplicated by named_parameters, matching HF safe serialization
    # for tied weights. No in-place PEFT merge and no mutation of config/source tensors.
    tensor_rows: list[dict[str, Any]] = []
    shards: list[tuple[str, list[str]]] = []
    weight_map: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f".part-{len(shards) + 1:05d}.safetensors"
        save_file(pending, str(out / filename), metadata={"format": "pt"})
        shards.append((filename, list(pending)))
        pending, pending_bytes = {}, 0

    for name, value in merged_named_tensors(bundle):
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("export: finite BF16 weights required")
        value = value.detach().cpu().contiguous()
        size = value.numel() * value.element_size()
        if pending and pending_bytes + size > max_shard_size:
            flush()
        pending[name] = value
        pending_bytes += size
        tensor_rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": tensor_sha256(value)})
    flush()
    for index, (filename, names) in enumerate(shards, 1):
        target = ("model.safetensors" if len(shards) == 1
                  else f"model-{index:05d}-of-{len(shards):05d}.safetensors")
        (out / filename).rename(out / target)
        weight_map.update({name: target for name in names})
    if len(shards) > 1:
        index_manifest = {"metadata": {"total_size": sum(
            2 * _numel(row["shape"]) for row in tensor_rows)}, "weight_map": weight_map}
        (out / "model.safetensors.index.json").write_text(json.dumps(index_manifest, indent=2) + "\n")
    for name in base_files:
        if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            shutil.copyfile(model_path / name, out / name)
    assert_base_unchanged(bundle)
    if before != {name: tensor_sha256(p) for name, p in bundle.model.named_parameters()}:
        raise ValueError("export: policy mutated during export")
    if (file_hash(manifest_path) != manifest_sha or _validate_base(model_path, manifest_path)[1] != base_files
            or source_checkpoint != {p.name: file_hash(p) for p in ckpt.iterdir()}
            or read_metadata(ckpt, binding=binding) != metadata):
        raise ValueError("export: source changed during export")
    manifest = {"status": "EXPORTED", "format": "derived-hf-bf16-v1", "arm": arm,
        "base_id": BASE_ID, "base_revision": BASE_REVISION, "base_manifest_sha256": manifest_sha,
        "base_files": base_files, "checkpoint_files": source_checkpoint,
        "checkpoint_state_sha256": metadata["state_sha256"], "binding": binding,
        "resolved_protocol_sha256": metadata["resolved_protocol_sha256"],
        "mergers": bundle.inventory["mergers"],
        "merger_tensor_names": sorted(r["name"] for r in tensor_rows if any(
            r["name"].startswith(root + ".") for root in bundle.inventory["mergers"])),
        "operation_order": ["verify_pinned_base_files", "fresh_BF16_F0", "restore_all_adapters_and_FP32_masters",
                            "out_of_place_FP32_fusion", "cast_BF16", "save_HF_shards"],
        "precision_policy": {"fusion": "FP32", "export": "BF16",
            "merger_masters": "FP32 checkpoint preserved; export rounds to BF16",
            "unmerged_inference_bit_equivalent": False},
        "tensors": tensor_rows, "files": {p.name: file_hash(p) for p in sorted(out.iterdir())}}
    # The final manifest is the completion marker; a failed directory is never accepted.
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _numel(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result
