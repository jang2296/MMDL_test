"""Reuse the continuous evaluator, adding only gold-blind text-only requests."""
from __future__ import annotations

from datetime import datetime, timezone

from mmdl.evaluation.backends.input_preparation import tensor_identity
from mmdl.evaluation.backends.vllm_backend import VLLMBackend
from mmdl.runtime.artifacts import digest


class BudgetStop(RuntimeError):
    pass


class RecoveryBackend(VLLMBackend):
    def __init__(self, *args, deadline_utc, **kwargs):
        self.deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if self.deadline.utcoffset() is None:
            raise ValueError("Timezone-aware deadline required")
        self._check_deadline()
        super().__init__(*args, **kwargs)

    def _check_deadline(self):
        if datetime.now(timezone.utc) >= self.deadline:
            raise BudgetStop("Absolute GPU deadline reached")

    def _prepare_request(self, request, *, final_only):
        self._check_deadline()
        if request["images"]:
            return super()._prepare_request(request, final_only=final_only)
        text = self.processor.apply_chat_template(
            request["messages"], tokenize=False, add_generation_prompt=True)
        inputs = self.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs["input_ids"][0].tolist()
        if self.model_config.image_token_id in token_ids:
            raise ValueError("Text-only payload contains an unbound image token")
        if len(token_ids) + 8192 > self.max_model_len:
            raise ValueError("Input plus output budget exceeds context; no truncation allowed")
        tensors = {name: tensor_identity(value) for name, value in inputs.items()}
        item = {"chat_prompt": text, "prompt_token_ids": token_ids,
                "input_tokens": len(token_ids), "image_grid_thw": [],
                "input_tensors": tensors, "input_sha256": digest(tensors),
                "processed_image_sizes": []}
        return item, {"prompt": text}, self._sampling_params(request["seed"], final_only=final_only)

def create_backend(model_path, base_path, arm, config, deadline_utc):
    return RecoveryBackend(str(model_path), config, {"placement": "gpu_only"},
                           kind="base" if arm == "F0" else "merged",
                           base_path=str(base_path), deadline_utc=deadline_utc)
