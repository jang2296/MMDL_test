"""Actual meta-model module/parameter inventory; no weights loaded or updated."""

import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from .ingest import MODEL_REVISION, sha, write_json


def module_inventory(model_path, manifest_path, output):
    from accelerate import init_empty_weights  # type: ignore[import-untyped]  # no packaged typing stubs
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    model_path = Path(model_path)
    manifest = json.loads(Path(manifest_path).read_text())
    config_entry = next(x for x in manifest["files"] if x["path"] == "config.json")
    if manifest["revision"] != MODEL_REVISION or sha(model_path / "config.json") != config_entry["sha256"]:
        raise ValueError("Pinned configuration mismatch; no automatic replacement")
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(cfg)
    model.tie_weights()
    def group(name):
        if name.startswith("model.visual."):
            return "M_MERGER" if name.startswith(("model.visual.merger", "model.visual.deepstack_merger_list")) else "V_VISION"
        return "L_LANGUAGE"
    parameters = [{"name": name, "shape": list(p.shape), "elements": p.numel(),
                   "device": str(p.device), "group": group(name)} for name, p in model.named_parameters()]
    modules = [{"name": name, "class": type(module).__name__, "group": group(name)}
               for name, module in model.named_modules() if name]
    counts = Counter()
    for p in parameters:
        counts[p["group"]] += p["elements"]
    mergers = [m for m in modules if m["class"] == "Qwen3VLVisionPatchMerger"]
    if not mergers or not any("deepstack_merger_list" in m["name"] for m in mergers):
        raise ValueError("Expected actual final and DeepStack merger instances not found")
    if not all(p["device"] == "meta" for p in parameters):
        raise ValueError("Inventory must not allocate real model weights")
    result = {"status": "ACTUAL_META_MODEL_INSPECTED", "model_revision": MODEL_REVISION,
              "config_sha256": config_entry["sha256"], "transformers_version": importlib.metadata.version("transformers"),
              "groups": dict(counts), "mergers": mergers, "modules": modules, "parameters": parameters,
              "training_updates": 0, "note": "Vision qkv/proj and language q_proj/k_proj/v_proj/o_proj remain separate actual names"}
    write_json(output, result)
    return {k: result[k] for k in ("status", "groups", "mergers", "training_updates")}
