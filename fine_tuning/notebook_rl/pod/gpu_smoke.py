"""Pod GPU smoke: GPU/torch facts, one real BF16 matmul, the pinned Qwen3-VL-4B load in BF16 from
the offline HF cache, and one short sampled image+text generation. The output is recorded, not judged.

CLI: python -m fine_tuning.notebook_rl.pod.gpu_smoke [--output PATH] [--max-new-tokens N]
The JSON report (`pod-gpu-smoke-v1`) is rewritten after every phase. Exit 0 PASS, 1 FAILED, 2 refused.
Never retries and never falls back to another dtype, device, device_map, model or revision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import utc_now, write_json

GIB = 2**30
VERSION = "pod-gpu-smoke-v1"
SMI_QUERY = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"]


def main(argv: list[str] | None = None) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"  # set before any torch/transformers/hub import
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output:
        output = Path(args.output)
    elif os.environ.get("MMDL_RUN_DIR"):
        output = Path(os.environ["MMDL_RUN_DIR"]) / "gpu_smoke.json"
    else:
        print("REFUSED: give --output or set MMDL_RUN_DIR", file=sys.stderr)
        return 2
    if output.exists():
        print(f"REFUSED: {output} already exists; never overwrite", file=sys.stderr)
        return 2
    if not 1 <= args.max_new_tokens <= 256:
        print("REFUSED: --max-new-tokens must be 1..256", file=sys.stderr)
        return 2

    report: dict = {"version": VERSION, "status": "RUNNING", "started_utc": utc_now(), "phases": {}}
    phase = "environment"

    def record(facts: dict, began: float) -> None:
        facts["seconds"] = round(time.monotonic() - began, 3)
        write_json(output, report)

    try:
        import torch
        import transformers
        from huggingface_hub import try_to_load_from_cache

        from mmdl.runtime.contracts import MODEL_ID, MODEL_REVISION

        began = time.monotonic()
        facts = report["phases"][phase] = {
            "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__, "HF_HOME": os.environ.get("HF_HOME")}
        smi = subprocess.run(SMI_QUERY, capture_output=True, text=True, timeout=30)
        facts["nvidia_smi"] = {"returncode": smi.returncode, "lines": smi.stdout.splitlines()}
        if smi.returncode != 0:
            raise RuntimeError(f"nvidia-smi exit {smi.returncode}: {smi.stderr.strip()[-500:]}")
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= 1):
            raise RuntimeError("CUDA not available")
        facts["device_name"] = torch.cuda.get_device_name(0)
        facts["capability"] = list(torch.cuda.get_device_capability(0))
        facts["bf16_supported"] = torch.cuda.is_bf16_supported()
        if not facts["bf16_supported"]:
            raise RuntimeError("cuda:0 does not support bfloat16")
        free, total = torch.cuda.mem_get_info(0)
        facts["free_gib"], facts["total_gib"] = round(free / GIB, 3), round(total / GIB, 3)
        record(facts, began)

        phase = "bf16_matmul"
        began = time.monotonic()
        torch.manual_seed(3407)
        a = torch.randn(2048, 2048, dtype=torch.float32, device="cuda:0")
        b = torch.randn(2048, 2048, dtype=torch.float32, device="cuda:0")
        product16 = a.to(torch.bfloat16) @ b.to(torch.bfloat16)
        c32 = a @ b
        torch.cuda.synchronize()
        rel = float((product16.float() - c32).norm() / c32.norm())
        if not (math.isfinite(rel) and rel < 0.05):
            raise RuntimeError(f"bf16 matmul rel_error {rel} is not finite or not below 0.05")
        facts = report["phases"][phase] = {"rel_error": rel, "bf16_product_dtype": str(product16.dtype)}
        record(facts, began)

        phase = "model_load"
        began = time.monotonic()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True, dtype=torch.bfloat16
        ).to("cuda:0").eval()
        wrong = sum(p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in model.parameters())
        if wrong:
            raise RuntimeError(f"{wrong} parameters are not bfloat16 on cuda")
        free, total = torch.cuda.mem_get_info(0)
        facts = report["phases"][phase] = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "config_cache_path": str(try_to_load_from_cache(MODEL_ID, "config.json", revision=MODEL_REVISION)),
            "free_gib": round(free / GIB, 3), "total_gib": round(total / GIB, 3)}
        record(facts, began)

        phase = "generation"
        began = time.monotonic()
        from PIL import Image, ImageDraw
        from transformers import GenerationConfig, LogitsProcessorList

        from fine_tuning.notebook_rl.engines import QWEN_OFFICIAL
        from mmdl.runtime.generation import GeneratedOnlyPresencePenalty

        image = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 80, 220, 240), fill="red")
        draw.ellipse((240, 200, 400, 360), fill="blue")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "Describe the image in one short sentence."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        input_length = inputs["input_ids"].shape[1]
        sampling = {"temperature": QWEN_OFFICIAL.temperature, "top_p": QWEN_OFFICIAL.top_p,
                    "top_k": QWEN_OFFICIAL.top_k, "presence_penalty": QWEN_OFFICIAL.presence_penalty,
                    "repetition_penalty": QWEN_OFFICIAL.repetition_penalty, "do_sample": True,
                    "max_new_tokens": args.max_new_tokens}
        cfg = GenerationConfig(
            do_sample=True, num_beams=1, temperature=QWEN_OFFICIAL.temperature,
            top_p=QWEN_OFFICIAL.top_p, top_k=QWEN_OFFICIAL.top_k,
            repetition_penalty=QWEN_OFFICIAL.repetition_penalty, max_new_tokens=args.max_new_tokens,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id)
        penalty = GeneratedOnlyPresencePenalty(input_length, QWEN_OFFICIAL.presence_penalty)
        torch.manual_seed(3407)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = model.generate(**inputs, generation_config=cfg,
                                 logits_processor=LogitsProcessorList([penalty]),
                                 use_model_defaults=False)
        torch.cuda.synchronize()
        ids = getattr(out, "sequences", out)[0, input_length:].tolist()
        if not ids:
            raise RuntimeError("generation returned no new tokens")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos) if isinstance(eos, list) else {eos}
        facts = report["phases"][phase] = {
            "sampling": sampling, "input_tokens": input_length, "generated_tokens": len(ids),
            "finish_reason": "eos" if ids[-1] in eos_ids else "length",
            "text": processor.tokenizer.decode(ids, skip_special_tokens=True),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3)}
        record(facts, began)

        report.update(status="PASS", finished_utc=utc_now())
        write_json(output, report)
        print(json.dumps({"status": "PASS", "output": str(output)}))
        return 0
    except Exception as exc:  # recorded and stopped; never retried or fallen back
        report.update(status="FAILED", failed_phase=phase, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc()[-4000:],
                      finished_utc=utc_now())
        write_json(output, report)
        print(f"FAILED in {phase}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
