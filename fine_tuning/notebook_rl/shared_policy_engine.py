"""Sequential HF smoke rollouts using the trainer's live, unmerged PEFT policy."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch

from .engines import HFRolloutEngine, InputPreparation, RolloutRequest, RolloutResult, SamplingSettings
from .policy import PolicyBundle, completion_logprobs


class SharedPolicyRolloutEngine(HFRolloutEngine):
    """One live model; fresh generation caches and actual probes after every update.

    HF retains one completion's generation scores until extraction. This path is
    for bounded sequential smoke runs; real GPU memory/time still need calibration.
    """

    def __init__(self, bundle: PolicyBundle, image_cfg: dict, max_model_len: int, *,
                 logprob_chunk: int = 1024, policy_version: int = 0):
        if (not isinstance(bundle, PolicyBundle) or not hasattr(bundle.model, "peft_config")
                or bundle.processor is None or type(logprob_chunk) is not int or logprob_chunk < 1
                or type(policy_version) is not int or policy_version < 0):
            raise ValueError("shared engine: live PEFT bundle, positive chunk and nonnegative version required")
        # Do not construct a second model, even on meta, or export merged weights.
        InputPreparation.__init__(self, bundle.processor, bundle.model.config, image_cfg, max_model_len)
        self.bundle, self.model, self.logprob_chunk = bundle, bundle.model, logprob_chunk
        self.policy_version = policy_version
        self._parameters = {name: (id(p), tuple(p.shape), p.dtype, p.device)
                            for name, p in self.model.named_parameters()}
        self._partitions = {group: tuple(id(p) for p in params) for group, params in bundle.params.items()}
        self._assert_owner(bundle)
        self._initialization_open = False
        self._synchronized = False

    def _assert_owner(self, bundle: PolicyBundle) -> None:
        if (bundle is not self.bundle or bundle.model is not self.model
                or bundle.processor is not self.processor
                or self._parameters != {name: (id(p), tuple(p.shape), p.dtype, p.device)
                                        for name, p in self.model.named_parameters()}
                or self._partitions != {group: tuple(id(p) for p in params)
                                        for group, params in bundle.params.items()}):
            raise ValueError("shared engine: policy identity or parameter ownership changed")
        owned = [identifier for partition in self._partitions.values() for identifier in partition]
        if (set(self._partitions) != {"L", "M", "V"} or not self._partitions["L"]
                or not self._partitions["M"] or len(owned) != len(set(owned))
                or not set(owned) <= {entry[0] for entry in self._parameters.values()}
                or tuple(id(p) for p in bundle.route.params) != self._partitions["V"]):
            raise ValueError("shared engine: invalid policy parameter partitions")
        if any(getattr(module, "merged_adapters", ()) or getattr(module, "disable_adapters", False) is True
               for module in self.model.modules()):
            raise ValueError("shared engine: enabled, unmerged live adapters required")

    def mark_updated(self, bundle: PolicyBundle, version: int) -> None:
        """Invalidate certification; shared storage alone never certifies sampling."""
        self._assert_owner(bundle)
        if type(version) is not int or version != self.policy_version + 1 or not self._synchronized:
            raise ValueError("shared engine: verified policy versions must advance by one")
        self.policy_version = version
        self._synchronized = False

    def initialize_weights(self, named_weights: Iterable[tuple[str, torch.Tensor]], policy_version: int) -> None:
        raise RuntimeError("shared engine: raw or merged weight initialization forbidden; bind the live bundle")

    def push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        raise RuntimeError("shared engine: raw or merged weight push forbidden; use mark_updated")

    def _load_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        raise RuntimeError("shared engine: raw or merged weight loading forbidden")

    def sample_with_settings(self, requests: list[RolloutRequest], sampling: SamplingSettings) -> list[RolloutResult]:
        self._assert_owner(self.bundle)
        vision_on = self.bundle.route.vision_on
        device = next(self.model.parameters()).device
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        try:
            with autocast:
                return super().sample_with_settings(requests, sampling)
        finally:
            self.bundle.route.apply(vision_on)

    def probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]:
        self._assert_owner(self.bundle)
        if not completion_ids:
            raise ValueError("shared engine: empty probe")
        device = next(self.model.parameters()).device
        batch = self.model_inputs(request, completion_ids, device)
        was_training, vision_on = self.model.training, self.bundle.route.vision_on
        self.model.eval()
        try:
            with torch.no_grad():
                return completion_logprobs(self.bundle, batch, len(completion_ids),
                                           chunk=self.logprob_chunk)[0].cpu().tolist()
        finally:
            self.model.train(was_training)
            self.bundle.route.apply(vision_on)
