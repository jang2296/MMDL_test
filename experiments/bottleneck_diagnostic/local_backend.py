"""Small local BF16/offload checks; never reported as the RunPod vLLM protocol."""

import copy
import random
import time

from .runner import BudgetStop


def make_local_backend(model_path, cfg, hw):
    import numpy as np
    import torch
    from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    from mmdl.evaluation.backends.transformers_backend import TransformersBackend
    from mmdl.evaluation.backends.input_preparation import prepare_protocol_inputs, tensor_identity
    from mmdl.runtime.artifacts import digest
    from mmdl.runtime.generation import GeneratedOnlyPresencePenalty
    from .ingest import MODEL_REVISION

    if hw["placement"] != "cpu_offload" or hw["allow_disk_offload"]:
        raise ValueError("Local diagnostic requires explicit CPU offload, no disk/quantization fallback")
    local_cfg = copy.deepcopy(cfg)
    local_cfg["execution"].update(backend="transformers", batch_size=1, attention="sdpa", sdpa_kernel="auto")
    backend = TransformersBackend(model_path, local_cfg, hw)
    # Distinct local execution policy, not a claim of identical vLLM numerical behavior.
    torch.use_deterministic_algorithms(False)

    class Deadline(StoppingCriteria):
        def __init__(self, end):
            self.end = end
            self.expired = False

        def __call__(self, input_ids, scores, **kwargs):
            self.expired = time.monotonic() >= self.end
            return self.expired

    class LocalSmoke:
        actual_inference = True
        load_seconds = backend.load_seconds
        cfg = local_cfg
        hardware_profile = hw

        def __call__(self, request):
            if time.monotonic() >= request["deadline_monotonic"]:
                raise BudgetStop("LOCAL_DEADLINE_BEFORE_GENERATION")
            seed = request["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if request["images"]:
                item, _ = prepare_protocol_inputs(backend.processor, backend.model.config,
                                                 request["messages"], request["images"], cfg["image"])
            else:
                text = backend.processor.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
                inputs = backend.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                tensors = {name: tensor_identity(value) for name, value in inputs.items()}
                item = {"inputs": inputs, "input_tokens": len(inputs["input_ids"][0]),
                        "prompt_token_ids": inputs["input_ids"][0].tolist(), "image_grid_thw": [],
                        "processed_image_sizes": [], "input_sha256": digest(tensors)}
            if item["input_tokens"] + request["max_new_tokens"] > cfg["execution"]["max_model_len"]:
                raise ValueError("Local context allowance exceeded; refusing truncation")
            generation = copy.deepcopy(backend.generation)
            generation.max_new_tokens = request["max_new_tokens"]
            stop = Deadline(request["deadline_monotonic"])
            penalty = GeneratedOnlyPresencePenalty(item["input_tokens"], cfg["generation"]["presence_penalty"])
            torch.cuda.reset_peak_memory_stats()
            before = time.monotonic()
            with torch.inference_mode():
                result = backend.model.generate(**item["inputs"].to(backend.device),
                                                generation_config=generation,
                                                logits_processor=LogitsProcessorList([penalty]),
                                                stopping_criteria=StoppingCriteriaList([stop]))
            torch.cuda.synchronize()
            if stop.expired:
                raise BudgetStop("LOCAL_DEADLINE_DURING_GENERATION")
            tokens = result.sequences[0, item["input_tokens"]:].tolist()
            eos = generation.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos or []
            raw = backend.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            return {"raw_response": raw, "generated_tokens": len(tokens), "generated_token_ids": tokens,
                    "input_tokens": item["input_tokens"], "image_grid_thw": item["image_grid_thw"],
                    "processed_image_sizes": item.get("processed_image_sizes", []),
                    "image_tokens": sum(t == backend.model.config.image_token_id for t in item["prompt_token_ids"]),
                    "input_sha256": item["input_sha256"], "generation_seconds": time.monotonic() - before,
                    "finish_reason": "eos" if tokens and tokens[-1] in eos else "length",
                    "backend": "transformers-local-cpu-offload-not-runpod", "backend_version": "4.57.1",
                    "model_revision": MODEL_REVISION, "hardware": "RTX5060_Laptop_8GB_CPU_OFFLOAD",
                    "device_map": backend.describe()["device_map"],
                    "peak_memory": {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "reserved_bytes": torch.cuda.max_memory_reserved()}}
    return LocalSmoke()
