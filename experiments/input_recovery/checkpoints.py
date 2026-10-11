"""Export verified selective checkpoints as derived, standalone BF16 HF models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from fine_tuning.modules import (ARMS, BASE_ID, BASE_REVISION, inspect_modules,
                                 load_checkpoint, tensor_sha256)

BASE_MANIFEST = Path(__file__).resolve().parents[2] / "manifests/models/baseline.json"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(checkpoint: Path, metadata: dict) -> dict[str, str]:
    files = metadata["files"]
    expected = {"metadata.json", *files}
    if {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()} != expected:
        raise ValueError("Checkpoint contains missing or unbound files")
    result = {"metadata.json": _hash(checkpoint / "metadata.json")}
    for name, digest in files.items():
        path = checkpoint / name
        if (Path(name).is_absolute() or not path.resolve().is_relative_to(checkpoint)
                or path.is_symlink() or _hash(path) != digest):
            raise ValueError(f"Checkpoint path/hash mismatch: {name}")
        result[name] = digest
    return result


def export_checkpoint(model_path: Path, checkpoint: Path, output: Path) -> dict:
    """Never modify sources; only export a new directory with a final commit manifest."""
    model_path, checkpoint, output = Path(model_path), Path(checkpoint), Path(output)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[2] / "MMDL"):
        raise ValueError("Professor submission checkout is protected")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Export destination must be new")
    model_path, checkpoint = model_path.resolve(), checkpoint.resolve()
    output = output.absolute()
    if any(output.resolve().is_relative_to(root) for root in (model_path, checkpoint)):
        raise ValueError("Export destination must be outside source directories")
    if not model_path.is_dir() or model_path.name != BASE_REVISION:
        raise ValueError("Use the local pinned snapshot; downloads are disabled")
    base_manifest_bytes = BASE_MANIFEST.read_bytes()
    base_manifest_digest = hashlib.sha256(base_manifest_bytes).hexdigest()
    base_manifest = json.loads(base_manifest_bytes)
    if base_manifest["repo_id"] != BASE_ID or base_manifest["revision"] != BASE_REVISION:
        raise ValueError("Base manifest identity mismatch")
    base_files = {}
    for row in base_manifest["files"]:
        name = row["path"]
        path = model_path / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or path.stat().st_size != row["bytes"] or _hash(path) != row["sha256"]):
            raise ValueError(f"Pinned base file mismatch: {name}")
        base_files[name] = row["sha256"]
    if ({p.name for p in model_path.glob("*.safetensors")}
            != {name for name in base_files if name.endswith(".safetensors")}):
        raise ValueError("Base has unbound or missing weight shards")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Wrong base architecture")
    metadata_bytes = (checkpoint / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("base_id") != BASE_ID or metadata.get("base_revision") != BASE_REVISION:
        raise ValueError("Checkpoint base identity mismatch")
    arm = metadata.get("arm")
    if arm not in ARMS or metadata["inventory"]["arm"] != arm:
        raise ValueError("Invalid checkpoint arm")
    checkpoint_files = _checkpoint_files(checkpoint, metadata)
    if checkpoint_files["metadata.json"] != hashlib.sha256(metadata_bytes).hexdigest():
        raise ValueError("Checkpoint metadata changed while reading")
    trainable = metadata["inventory"]["trainable"]
    if ({row["name"] for row in trainable} != set(metadata["state_sha256"])
            or len(trainable) != len(metadata["state_sha256"])
            or {row["group"] for row in trainable} != set(ARMS[arm])
            or ("mergers.pt" in checkpoint_files) != ("M" in ARMS[arm])
            or ("adapter/adapter_model.safetensors" in checkpoint_files) != (arm != "FM")):
        raise ValueError("Incomplete checkpoint trainable state")

    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, local_files_only=True, device_map=None,
        attn_implementation="sdpa",
    )
    inventory = inspect_modules(model)
    shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    if any(metadata["inventory"][key] != inventory[key] for key in ("visual", "language", "mergers", "targets")):
        raise ValueError("Checkpoint actual module inventory differs from pinned base")
    model = load_checkpoint(model, checkpoint)
    parameters = dict(model.named_parameters())
    for row in trainable:
        parameter = parameters[row["name"]]
        if list(parameter.shape) != row["shape"] or str(parameter.dtype) != row["dtype"]:
            raise ValueError(f"Restored tensor shape/dtype mismatch: {row['name']}")
    merger_names = {name for name in shapes
                    if any(name.startswith(root + ".") for root in inventory["mergers"])}
    restored_mergers = {row["name"].removeprefix("base_model.model.")
                        for row in trainable if row["group"] == "M"}
    if "M" in ARMS[arm] and restored_mergers != merger_names:
        raise ValueError("Checkpoint must restore every tensor of all four mergers")
    lora_modules = [name.removeprefix("base_model.model.") for name, module in model.named_modules()
                    if hasattr(module, "lora_A")]
    expected_lora = {name for group in ARMS[arm] if group != "M" for name in inventory["targets"][group]}
    if set(lora_modules) != expected_lora:
        raise ValueError("Restored LoRA targets differ from actual training partitions")
    # Fuse in FP32 so PEFT does not round the delta to BF16 before addition.
    model.float()
    if lora_modules:
        merge = getattr(model, "merge_and_unload", None)
        if not callable(merge):
            raise ValueError("Restored LoRA model does not support safe merging")
        model = merge(safe_merge=True)
    model.to(dtype=torch.bfloat16).eval()
    if {name: list(value.shape) for name, value in model.state_dict().items()} != shapes:
        raise ValueError("Merged export changes base tensor names/shapes")
    tensors = []
    for name, value in model.state_dict().items():
        if value.is_floating_point() and (value.dtype != torch.bfloat16 or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid BF16 export tensor: {name}")
        tensors.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": tensor_sha256(value)})
    def verify_sources():
        if (_hash(BASE_MANIFEST) != base_manifest_digest
                or {name: _hash(model_path / name) for name in base_files} != base_files
                or _checkpoint_files(checkpoint, metadata) != checkpoint_files):
            raise ValueError("Export source changed during restoration")

    verify_sources()

    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
        staging = Path(temporary)
        model.save_pretrained(staging, safe_serialization=True)
        for name in base_files:
            if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
                shutil.copyfile(model_path / name, staging / name)
        output_files = {str(path.relative_to(staging)): _hash(path)
                        for path in sorted(staging.rglob("*")) if path.is_file()}
        verify_sources()
        manifest = {
            "status": "EXPORTED", "format": "derived-hf-bf16-v1", "arm": arm,
            "base_id": BASE_ID, "base_revision": BASE_REVISION,
            "base_manifest_sha256": base_manifest_digest, "base_files": base_files,
            "checkpoint_files": checkpoint_files, "checkpoint_state_sha256": metadata["state_sha256"],
            "mergers": inventory["mergers"], "merger_tensor_names": sorted(merger_names),
            "merged_lora_modules": lora_modules,
            "operation_order": ["verify_pinned_base_files", "load_BF16_base",
                                "restore_and_hash_verify_adapter_and_FP32_merger_masters",
                                "promote_FP32", "safe_merge_all_LoRA", "cast_BF16", "save_HF"],
            "precision_policy": {"fusion": "FP32", "export": "BF16",
                                 "merger_masters": "FP32 checkpoint preserved; derived export rounds to BF16",
                                 "unmerged_inference_bit_equivalent": False},
            "tensors": tensors, "files": output_files,
            "source_sha256": {"checkpoints.py": _hash(Path(__file__)),
                              "modules.py": _hash(Path(__file__).resolve().parents[2] / "fine_tuning/modules.py")},
        }
        for path in list(staging.iterdir()):
            path.rename(output / path.name)
        pending = staging / "export_manifest.json"
        pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, output / "export_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_checkpoint(args.model_path, args.checkpoint, args.output)
    print(json.dumps({"status": manifest["status"], "arm": manifest["arm"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
