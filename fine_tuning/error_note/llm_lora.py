"""v15: apply the trained adapter unmerged in vLLM 0.11.0, shared by training rollouts and the evaluation commit.

LLM LoRA (252 targets) -> a PEFT file loaded by vLLM's own LoRA path (BF16 slots, B scaled by alpha/r).
Vision LoRA (48 targets) -> exact forward hooks on vLLM's eager vision tower, never merged (a BF16 merge keeps ~12 %).
Mergers -> FP32 masters rounded to BF16 and pushed alone; every other vLLM weight stays the original snapshot.
The preflight checks (PF0-PF5) are plain functions so training (single_vllm) and evaluation can both run them.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

import torch

ENGINE_LORA = "vllm-0.11.0-processed-logprobs-unmerged-lora-v1"
ENGINE_ARGS = {"enable_lora": True, "max_lora_rank": 16, "max_loras": 1, "max_cpu_loras": 1}  # lora_dtype auto = BF16
RANK, ALPHA = 16, 32
LLM_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PACKED = {"q_proj": ("self_attn.qkv_proj", 0), "k_proj": ("self_attn.qkv_proj", 1), "v_proj": ("self_attn.qkv_proj", 2),
          "o_proj": ("self_attn.o_proj", 0), "gate_proj": ("mlp.gate_up_proj", 0), "up_proj": ("mlp.gate_up_proj", 1),
          "down_proj": ("mlp.down_proj", 0)}
VLLM_LORA_MODULES = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
VLLM_MERGERS = ("visual.merger.", "visual.deepstack_merger_list.")
HF_MERGERS = tuple("model." + prefix for prefix in VLLM_MERGERS)
VISION_TARGET = re.compile(r"visual\.blocks\.\d+\.attn\.(qkv|proj)")
HOST_MEMORY_GIB = 16.0  # sleep level 1 copies the ~9 GiB weight pool to host memory
SLOPE_RANGE, MIN_CORRELATION, MIN_EFFECT = (0.85, 1.15), 0.9, 0.05
# PF5 amplified probe: every LoRA B x AMPLIFY on both engines. A power of two, so bf16(K*B) == K*bf16(B) and the
# amplified vLLM slots / hook buffers are exactly K times the real ones. Real v14 4B adapter: ~0.05 rms effect at
# ~0.018 mean |delta| BF16 cross-engine noise (slope test powerless); x16 puts it near 0.8 rms.
AMPLIFY = 16
# HF/PEFT reference columns of PF5: per-group scaling factors (absent group = as trained, 0 = off).
HF_PROBES = {"h_on": {}, "h_llm": {"V": 0.0}, "h_off": {"L": 0.0, "V": 0.0},
             "ha_on": {"L": AMPLIFY, "V": AMPLIFY}, "ha_llm": {"L": AMPLIFY, "V": 0.0}}


def scope_qwen3vl_lora() -> None:
    """vLLM 0.11.0 qwen3_vl.py:1605 returns HF prefixes ('model.visual.') that never match vLLM's 'visual.' module
    names, so vision and Merger linears get LoRA wrappers driven by the decoder's token map. Patch in-process only
    (VLLM_ENABLE_V1_MULTIPROCESSING=0) and before LLM(...). Idempotent."""
    from vllm.model_executor.models.module_mapping import MultiModelKeys
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as Q

    current = Q.get_mm_mapping
    if getattr(current, "mmdl_original", None) is not None:
        return
    upstream = current(Q.__new__(Q))
    if (upstream.connector, upstream.tower_model) != (["model.visual.merger"], ["model.visual."]):
        raise RuntimeError("llm_lora: vLLM Qwen3-VL LoRA scope changed; re-review the scope patch")

    # ponytail: remove on a vLLM whose Qwen3-VL mapping already uses "visual." (as qwen2_5_vl does).
    def scoped(self: Any) -> Any:
        return MultiModelKeys.from_string_field(language_model="language_model", connector=list(VLLM_MERGERS),
                                                tower_model="visual.")

    setattr(scoped, "mmdl_original", current)
    Q.get_mm_mapping = scoped


def adapter_scaling(bundle: Any) -> float:
    cfg = bundle.model.peft_config["default"]
    if (cfg.r != RANK or cfg.lora_alpha != ALPHA or cfg.use_dora or cfg.use_rslora or cfg.lora_dropout != 0.0
            or cfg.bias != "none"):
        raise ValueError("llm_lora: adapter contract r=16, alpha=32, no DoRA/rsLoRA/dropout/bias required")
    return cfg.lora_alpha / cfg.r


def _lora_weights(bundle: Any, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    layer = bundle.model.get_base_model().get_submodule(target)
    if layer.merged or "default" not in layer.lora_A or layer.scaling["default"] != adapter_scaling(bundle):
        raise ValueError(f"llm_lora: unmerged default adapter with the contract scaling required at {target}")
    return layer.lora_A["default"].weight.detach(), layer.lora_B["default"].weight.detach()


def adapter_config(bundle: Any) -> dict:
    adapter_scaling(bundle)
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": ALPHA, "lora_dropout": 0.0,
            "bias": "none", "target_modules": list(LLM_LEAVES), "use_rslora": False, "use_dora": False,
            "modules_to_save": None, "fan_in_fan_out": False}


def write_llm_adapter(bundle: Any, path: Path, factor: float = 1) -> None:
    """FP32 masters of the L targets only (vLLM casts to BF16 and folds alpha/r into B); no vision or Merger key.
    `factor` multiplies B (PF5 amplified probe only; the policy and the export always write factor 1)."""
    from safetensors.torch import save_file

    targets = bundle.inventory["targets"]["L"]
    if {t.rpartition(".")[2] for t in targets} != set(LLM_LEAVES):
        raise ValueError("llm_lora: LLM targets differ from the seven attention/MLP leaves")
    tensors = {}
    for target in targets:
        a, b = _lora_weights(bundle, target)
        tensors[f"base_model.model.{target}.lora_A.weight"] = a.float().cpu().contiguous().clone()
        tensors[f"base_model.model.{target}.lora_B.weight"] = (b.float() * factor).cpu().contiguous().clone()
    if len(tensors) != 2 * len(targets) or any("visual" in key for key in tensors):
        raise ValueError("llm_lora: adapter file must hold exactly the LLM A/B pairs")
    path.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(adapter_config(bundle), sort_keys=True))


def _vllm_vision_name(target: str) -> str:
    if not target.startswith("model.visual."):
        raise ValueError(f"llm_lora: unexpected vision target {target}")
    return target.removeprefix("model.")


def vision_lora(bundle: Any, factor: float = 1) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """vLLM name -> (bf16(A) [r, in], bf16(B) * scaling * factor [out, r]); x2 (and x AMPLIFY) is exact in BF16."""
    scaling = adapter_scaling(bundle) * factor
    out = {}
    for target in bundle.inventory["targets"]["V"]:
        a, b = _lora_weights(bundle, target)
        out[_vllm_vision_name(target)] = (a.to(torch.bfloat16).clone(), b.to(torch.bfloat16) * scaling)
    return out


class VisionHooks:
    """y += (x @ A.T) @ B.T after vLLM's visual.blocks.N.attn.{qkv,proj}. Allocated after engine start, outside the
    CuMem pool, so sleep never touches them. `on` is read at call time."""

    def __init__(self, model: Any):
        self.on = False
        self.buffers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles: dict[str, Any] = {}
        for name in sorted(n for n, _ in model.named_modules() if VISION_TARGET.fullmatch(n)):
            module = model.get_submodule(name)
            weight = getattr(module, "base_layer", module).weight
            a = torch.zeros(RANK, weight.shape[1], dtype=torch.bfloat16, device=weight.device)
            b = torch.zeros(weight.shape[0], RANK, dtype=torch.bfloat16, device=weight.device)
            self.buffers[name] = (a, b)
            self.handles[name] = module.register_forward_hook(functools.partial(self._hook, a, b))

    def _hook(self, a: torch.Tensor, b: torch.Tensor, module: Any, args: tuple, output: Any) -> Any:
        if not self.on:
            return output
        if not (isinstance(output, tuple) and len(output) == 2):
            raise RuntimeError("llm_lora: vLLM linear (output, bias) pair expected")
        y, bias = output
        return y + (args[0] @ a.T) @ b.T, bias

    def load(self, weights: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        if set(weights) != set(self.buffers):
            raise ValueError("llm_lora: vision LoRA names differ from the hooked modules")
        for name, (a, b) in weights.items():
            live_a, live_b = self.buffers[name]
            if (a.shape != live_a.shape or b.shape != live_b.shape
                    or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16):
                raise ValueError(f"llm_lora: vision LoRA shape/dtype mismatch at {name}")
            live_a.copy_(a)
            live_b.copy_(b)
        self.on = True

    def live(self, model: Any) -> bool:
        return all(h.id in model.get_submodule(n)._forward_hooks for n, h in self.handles.items())


def install_vision_hooks(model: Any) -> VisionHooks:
    return VisionHooks(model)


def push_mergers(engine: Any, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
    """engines.VLLMRolloutEngine._load_weights restricted to the four Mergers. Non-Merger names are consumed and
    checked against the inventory (merged_named_tensors' completeness check still runs) but never loaded."""
    from vllm.v1.engine.core_client import InprocClient

    from fine_tuning.notebook_rl.engines import _clear_encoder_cache

    engine._synchronized = engine._push_complete = False
    engine._last_loaded_version = None
    client = engine.llm.llm_engine.engine_core
    if not isinstance(client, InprocClient):
        raise RuntimeError("llm_lora: verified in-process vLLM V1 required")
    scheduler = client.engine_core.scheduler
    manager = scheduler.encoder_cache_manager
    if scheduler.has_unfinished_requests() or any(manager.cached.values()):
        raise RuntimeError("llm_lora: Merger push requires an idle scheduler/encoder")
    expected = {n for n in engine._weight_shapes if n.startswith(HF_MERGERS)}
    seen: set[str] = set()
    mergers: dict[str, torch.Tensor] = {}
    for name, value in named:
        if name in seen or name not in engine._weight_shapes:
            raise ValueError("llm_lora: duplicate or unexpected weight name")
        seen.add(name)
        if not name.startswith(HF_MERGERS):
            continue
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != engine._weight_shapes[name]
                or value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all())):
            raise ValueError("llm_lora: wrong shape/dtype or nonfinite Merger weight")
        mergers[name] = value
    if seen != engine._weight_shapes.keys() or set(mergers) != expected or not expected:
        raise ValueError("llm_lora: incomplete canonical inventory or Merger set")

    def load(model: Any) -> bool:
        target = {n for n, _ in model.named_parameters() if n.startswith(VLLM_MERGERS)}
        loaded = set(model.load_weights(iter(mergers.items())))
        return loaded == target and bool(loaded)

    if engine.llm.apply_model(load) != [True]:
        raise RuntimeError("llm_lora: complete Merger loading not confirmed")
    if scheduler.reset_prefix_cache() is not True:
        raise RuntimeError("llm_lora: prefix cache invalidation not acknowledged")
    if engine.llm.collective_rpc(_clear_encoder_cache) != [True]:
        raise RuntimeError("llm_lora: encoder cache invalidation not acknowledged")
    replacement = type(manager)(cache_size=manager.cache_size)
    scheduler.encoder_cache_manager = replacement
    if replacement.cached or replacement.freeable or replacement.freed:
        raise RuntimeError("llm_lora: encoder bookkeeping invalidation failed")
    engine.policy_version = engine._last_loaded_version = version
    engine._push_complete = True


# ---- preflight (PF0-PF5) -------------------------------------------------------------------------------------

def host_memory(sleep: bool) -> dict:
    """PF0: level-1 sleep needs host RAM for the weight pool; without sleep nothing is copied."""
    from .collect import _mem_available_kib

    kib = _mem_available_kib()
    gib = None if kib is None else kib / 2**20
    return {"sleep": bool(sleep), "mem_available_gib": gib, "required_gib": HOST_MEMORY_GIB if sleep else None,
            "pass": (not sleep) or (gib is not None and gib >= HOST_MEMORY_GIB)}


def check_scope(llm: Any, hooks: VisionHooks, num_layers: int, vision_depth: int) -> dict:
    """PF1: LoRA wrappers are exactly the language-model modules; no vision weight was renamed; 2 hooks per block."""
    def probe(model: Any) -> tuple[list[str], list[str], bool]:
        renamed = sorted(n for n, _ in model.named_parameters() if n.startswith("visual.") and ".base_layer." in n)
        return sorted(model.lora_manager.modules), renamed, hooks.live(model)

    modules, renamed, live = llm.apply_model(probe)[0]
    expected = sorted(f"language_model.model.layers.{i}.{m}" for i in range(num_layers) for m in VLLM_LORA_MODULES)
    hooked = sorted(f"visual.blocks.{i}.attn.{leaf}" for i in range(vision_depth) for leaf in ("qkv", "proj"))
    return {"lora_modules": len(modules), "expected_modules": len(expected),
            "foreign_modules": sorted(set(modules) - set(expected))[:8], "renamed_vision_weights": renamed[:8],
            "hooks": len(hooks.handles), "hooks_live": live,
            "pass": modules == expected and not renamed and sorted(hooks.handles) == hooked and live}


def llm_slots(bundle: Any) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Expected vLLM slot contents per packed module: [(bf16(A), bf16(B) * scaling)] in packing order."""
    scaling = adapter_scaling(bundle)
    groups: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for target in bundle.inventory["targets"]["L"]:
        parent, _, leaf = target.rpartition(".")
        layer = parent.rpartition(".")[0]
        if not layer.startswith("model.language_model.layers.") or leaf not in PACKED:
            raise ValueError(f"llm_lora: unexpected LLM target {target}")
        packed, index = PACKED[leaf]
        module = "language_model.model." + layer.removeprefix("model.language_model.") + "." + packed
        a, b = _lora_weights(bundle, target)
        groups.setdefault(module, {})[index] = (a.to(torch.bfloat16), b.to(torch.bfloat16) * scaling)
    if any(sorted(slices) != list(range(len(slices))) for slices in groups.values()):
        raise ValueError("llm_lora: incomplete packed LoRA slices")
    return {name: [slices[i] for i in range(len(slices))] for name, slices in groups.items()}


def merger_bf16(bundle: Any) -> dict[str, torch.Tensor]:
    out = {}
    for name, parameter in bundle.model.get_base_model().named_parameters():
        if ".modules_to_save.default." in name:
            canonical = name.replace(".modules_to_save.default.", ".")
            out[_vllm_vision_name(canonical)] = parameter.detach().to(torch.bfloat16)
    return out


def check_identity(llm: Any, bundle: Any, lora_id: int, hooks: VisionHooks) -> dict:
    """PF2, bit-exact: active slot == bf16(A) / bf16(B)*scaling, hook buffers == vision_lora, Mergers == bf16(master)."""
    slots, vision, mergers = llm_slots(bundle), vision_lora(bundle), merger_bf16(bundle)

    def probe(model: Any) -> tuple[list[str], list[str]]:
        manager, bad = model.lora_manager, []
        if lora_id not in manager.lora_index_to_id or set(manager.modules) != set(slots):
            return ["<inactive adapter or module set>"], []
        slot = manager.lora_index_to_id.index(lora_id)
        for name, expected in slots.items():
            module = manager.modules[name]
            if len(module.lora_a_stacked) != len(expected):
                bad.append(name)
                continue
            for i, (a, b) in enumerate(expected):
                live_a, live_b = module.lora_a_stacked[i][slot, 0], module.lora_b_stacked[i][slot, 0]
                if not (torch.equal(live_a, a.to(live_a.device)) and torch.equal(live_b, b.to(live_b.device))):
                    bad.append(f"{name}[{i}]")
        params = {n: p for n, p in model.named_parameters() if n.startswith(VLLM_MERGERS)}
        merger_bad = sorted(set(params) ^ set(mergers))
        merger_bad += [n for n in sorted(set(params) & set(mergers)) if not torch.equal(params[n], mergers[n].to(params[n].device))]
        return bad, merger_bad

    bad, merger_bad = llm.apply_model(probe)[0]
    hook_bad = sorted(set(vision) ^ set(hooks.buffers))
    hook_bad += [n for n in sorted(set(vision) & set(hooks.buffers))
                 if not all(torch.equal(live, want.to(live.device)) for live, want in zip(hooks.buffers[n], vision[n]))]
    return {"llm_slices": sum(len(s) for s in slots.values()), "llm_mismatch": bad[:8], "vision_hooks_on": hooks.on,
            "vision_mismatch": hook_bad[:8], "merger_mismatch": merger_bad[:8],
            "pass": not bad and not hook_bad and not merger_bad and hooks.on is True}


def base_fingerprint(llm: Any) -> dict[str, str]:
    """sha256 of every vLLM parameter outside the Mergers (the only weights a LoRA-mode push may change)."""
    from fine_tuning.modules import tensor_sha256

    return llm.apply_model(lambda model: {n: tensor_sha256(p) for n, p in model.named_parameters()
                                          if not n.startswith(VLLM_MERGERS)})[0]


def check_base(llm: Any, bundle: Any, frozen: dict[str, str]) -> dict:
    """PF3: no non-Merger weight changed since start, and vLLM's layer-0 / vision-block-0 bases are the HF bases."""
    base = bundle.model.get_base_model()

    def hf(name: str) -> torch.Tensor:
        module = base.get_submodule(name)
        return getattr(module, "base_layer", module).weight.detach()

    text = "model.language_model.layers.0."
    pairs = {"language_model.model.layers.0.self_attn.qkv_proj.base_layer.weight": [f"{text}self_attn.{x}_proj" for x in "qkv"],
             "language_model.model.layers.0.self_attn.o_proj.base_layer.weight": [f"{text}self_attn.o_proj"],
             "language_model.model.layers.0.mlp.gate_up_proj.base_layer.weight": [f"{text}mlp.gate_proj", f"{text}mlp.up_proj"],
             "language_model.model.layers.0.mlp.down_proj.base_layer.weight": [f"{text}mlp.down_proj"],
             "visual.blocks.0.attn.qkv.weight": ["model.visual.blocks.0.attn.qkv"]}
    wanted = {name: torch.cat([hf(n) for n in parts]) for name, parts in pairs.items()}

    def probe(model: Any) -> list[str]:
        params = dict(model.named_parameters())
        return [n for n, w in wanted.items() if n not in params or not torch.equal(params[n], w.to(params[n].device))]

    now = base_fingerprint(llm)
    changed = sorted(n for n in set(now) | set(frozen) if now.get(n) != frozen.get(n))
    unequal = llm.apply_model(probe)[0]
    return {"parameters": len(now), "changed": changed[:8], "hf_base_mismatch": unequal,
            "pass": bool(now) and not changed and not unequal}


@contextmanager
def scaled_adapters(bundle: Any, **factors: float) -> Iterator[None]:
    """HF reference with each named LoRA group's ('L', 'V') scaling multiplied by its factor (0 = off); Mergers stay
    on. Restored exactly on exit."""
    base, saved = bundle.model.get_base_model(), []
    try:
        for group, factor in factors.items():
            for target in bundle.inventory["targets"][group]:
                layer = base.get_submodule(target)
                saved.append((layer, layer.scaling["default"]))
                layer.scaling["default"] = layer.scaling["default"] * factor
        yield
    finally:
        for layer, value in reversed(saved):
            layer.scaling["default"] = value


def adapters_off(bundle: Any, *groups: str) -> Any:
    """HF reference with the named LoRA groups ('L', 'V') at scaling 0; Mergers stay on. Restored on exit."""
    return scaled_adapters(bundle, **dict.fromkeys(groups, 0.0))


def hf_columns(bundle: Any, prep: Any, pairs: list) -> dict[str, list[float]]:
    """PF5 HF/PEFT teacher-forced log-probs of (request, completion) pairs for every HF_PROBES column."""
    from fine_tuning.notebook_rl.policy import completion_logprobs

    device = next(bundle.model.parameters()).device
    out: dict[str, list[float]] = {key: [] for key in HF_PROBES}
    with torch.no_grad():
        for request, completion in pairs:
            inputs = prep.model_inputs(request, completion, device)
            for key, factors in HF_PROBES.items():
                with scaled_adapters(bundle, **factors):
                    out[key] += completion_logprobs(bundle, inputs, len(completion)).float().flatten().cpu().tolist()
    return out


def vllm_columns(bundle: Any, hooks: VisionHooks, probe: Any, pairs: list, lora: Any, swap: Any) -> tuple[dict, Any]:
    """PF5 vLLM teacher-forced log-probs, shared by training and evaluation. `probe(request, completion, lora_request)`
    returns one log-prob per completion token; `swap(factor)` makes the bundle's LLM adapter with B x factor the
    engine's only adapter and returns its LoRARequest (amplified: a never-used id; factor 1, the real adapter: a fresh
    id in training, the export's own request in evaluation, lora_preflight.adapter_swap).
    Real adapter: v_on (LoRA + hooks as found), v_llm (LoRA only), v_off (neither). Amplified: va_on / va_llm with
    swap(AMPLIFY) and the hook B buffers x AMPLIFY. Afterwards, also on error, swap(1) restores the real adapter and
    vision_lora(bundle) the hook buffers, hooks.on as found; the caller re-checks PF2 on the returned request."""
    columns: dict[str, list[float]] = {key: [] for key in ("v_on", "v_llm", "v_off", "va_on", "va_llm")}
    state = hooks.on

    def run(on: str, llm: str, request_lora: Any, off: bool) -> None:
        try:
            for request, completion in pairs:
                columns[on] += probe(request, completion, request_lora)  # the engine as it generates
                hooks.on = False
                columns[llm] += probe(request, completion, request_lora)
                if off:
                    columns["v_off"] += probe(request, completion, None)
                hooks.on = state
        finally:
            hooks.on = state

    run("v_on", "v_llm", lora, True)
    try:
        amplified = swap(AMPLIFY)
        hooks.load(vision_lora(bundle, AMPLIFY))
        hooks.on = state
        run("va_on", "va_llm", amplified, False)
    finally:
        hooks.load(vision_lora(bundle))
        hooks.on = state
        restored = swap(1)
    return columns, restored


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def _fit(dv: torch.Tensor, dh: torch.Tensor) -> dict:
    entry: dict[str, Any] = {"rms_effect": _rms(dh)}
    if float(dh @ dh) > 0:
        corr = float(torch.corrcoef(torch.stack([dv, dh]))[0, 1]) if _rms(dv - dv.mean()) > 0 else 0.0
        entry.update(slope=float(dv @ dh / (dh @ dh)), correlation=corr if math.isfinite(corr) else 0.0)
    return entry


def slope_check(v_on: list[float], v_llm: list[float], v_off: list[float], h_on: list[float], h_llm: list[float],
                h_off: list[float], va_on: list[float], va_llm: list[float], ha_on: list[float], ha_llm: list[float],
                cap: float) -> dict:
    """PF5 on teacher-forced log-probs. v_*: vLLM (on = LoRA + hooks, llm = LoRA only, off = neither); h_*: HF (on,
    vision scaling 0, all scaling 0); va_* / ha_*: the same with every LoRA B x AMPLIFY (vLLM adapter file and hook
    buffers, HF scaling). A trained adapter's effect can sit at the BF16 cross-engine noise (A100 4B v14: 0.05 rms vs
    0.018 mean |delta|), where the OLS slope is attenuated and the test has no power. So the REAL adapter must give
    finite values, mean |v_on - h_on| <= cap and, if its effect >= MIN_EFFECT, v_on closer to HF than v_off; its
    slope/correlation are diagnostics only (no pass/fail). The application MECHANISM is tested on the AMPLIFIED
    effects: the total effect must be >= MIN_EFFECT (else FAIL: no power) and every component (total / llm / vision)
    with effect >= MIN_EFFECT within SLOPE_RANGE and MIN_CORRELATION. Wrong scale gives slope ~0.5/2, a merged path
    ~0.45 (LLM) / ~0.12 (vision)."""
    columns = [torch.tensor(x, dtype=torch.float64) for x in (v_on, v_llm, v_off, h_on, h_llm, h_off, va_on, va_llm,
                                                                ha_on, ha_llm)]
    v1, vl, v0, h1, hl, h0, a1, al, g1, gl = columns
    if not (v1.numel() and all(t.shape == v1.shape for t in columns)):
        raise ValueError("llm_lora: probe vectors must be nonempty and aligned")
    finite = all(bool(torch.isfinite(t).all()) for t in columns)
    err_on, err_off = float((v1 - h1).abs().mean()), float((v0 - h1).abs().mean())
    real = {key: _fit(dv, dh) for key, dv, dh in (("total", v1 - v0, h1 - h0), ("llm", vl - v0, hl - h0),
                                                  ("vision", v1 - vl, h1 - hl))}
    amplified = {key: _fit(dv, dh) for key, dv, dh in (("total", a1 - v0, g1 - h0), ("llm", al - v0, gl - h0),
                                                       ("vision", a1 - al, g1 - gl))}
    ok = finite and err_on <= cap and amplified["total"]["rms_effect"] >= MIN_EFFECT
    if real["total"]["rms_effect"] >= MIN_EFFECT:
        ok = ok and err_on < err_off
    for entry in amplified.values():
        entry["pass"] = None  # None: effect too small to test
        if entry["rms_effect"] >= MIN_EFFECT:
            entry["pass"] = (SLOPE_RANGE[0] <= entry["slope"] <= SLOPE_RANGE[1]
                             and entry["correlation"] >= MIN_CORRELATION)
            ok = ok and entry["pass"]
    return {"tokens": int(v1.numel()), "cap": cap, "amplify": AMPLIFY, "mean_abs_on": err_on,
            "mean_abs_off_vs_on": err_off, "amplified_mean_abs_on": float((a1 - g1).abs().mean()),
            "real": real, "amplified": amplified, "pass": bool(ok)}
