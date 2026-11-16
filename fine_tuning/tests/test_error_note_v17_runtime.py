"""CPU contracts for v17 capacity, fixed schedule, signal gate and drift guards."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from fine_tuning.error_note.guard_v16b import (
    GuardV17, SignalGateV17, falsealarm_receipt_v17, length_distribution_v17,
)
from fine_tuning.error_note.llm_lora import pf5_contract_v17, slope_check
from fine_tuning.error_note.single_vllm import SingleVllmSampler
from fine_tuning.error_note.v16b_runtime import V16bScheduler, reserve_request_v17, run_v16b
from fine_tuning.tests.test_error_note_v16b_runtime import FakeSampler, fixture


def order(n=36, final=16):
    return [[f"p{b}:{i}" for i in range(final if b == n - 1 else 16)] for b in range(n)]


def batch(ids, short=2, cap=0):
    return [{"id": pid, "source": "s", "correct": True, "truncated": i < cap,
             "no_final": False, "tokens": 10 if i < short else 100, "f0_tokens": 100}
            for i, pid in enumerate(ids)]


class CapacitySampler(FakeSampler):
    def prepare_request_v17(self, request):
        return {"prompt_tokens": 1, "max_tokens": request["max_tokens"]}

    def can_submit_v17(self, prepared):
        return reserve_request_v17(prepared, 20, 1, [9] * len(self.pending), active=len(self.pending))["status"] == "ADMIT"

    def submit(self, request, *, prepared=None):
        assert prepared is not None and self.can_submit_v17(prepared)
        super().submit(request)


class CapacityTests(unittest.TestCase):
    def test_full_future_reservation_rounding_waiting_and_ceiling(self):
        request = {"prompt_tokens": 17, "max_tokens": 16}
        receipt = reserve_request_v17(request, 128, 16, [48], active=1)
        self.assertEqual((receipt["request_reserved_tokens"], receipt["usable_tokens"], receipt["status"]), (48, 115, "ADMIT"))
        self.assertEqual(reserve_request_v17(request, 128, 16, [48, 48], active=2)["status"], "WAIT")
        self.assertEqual(reserve_request_v17(request, 10000, 16, [48] * 16, active=16)["status"], "WAIT")
        with self.assertRaisesRegex(RuntimeError, "CAPACITY_INSUFFICIENT"):
            reserve_request_v17(request, 32, 16, [], active=0)
        with self.assertRaises(ValueError):
            reserve_request_v17(request, 128, 16, [48], active=0)

    def test_backpressure_keeps_order_policy_seed_and_history_cutoff(self):
        _, items, plan, callbacks, baselines, checks, _ = fixture(4)
        sampler = CapacitySampler(reverse=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_v16b(None, Mock(), None, sampler, plan, out=Path(tmp),
                             protocol={"version": "error-note-v17"}, items=items, callbacks=callbacks)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual([c[0] for c in checks], list(range(7)))
        self.assertTrue(all(version in (max(0, block - 1), block + 1, block + 2)
                            for version, block, _ in sampler.admissions))
        self.assertTrue(all(all(h["block"] <= cutoff for h in history)
                            for _, _, cutoff, history in baselines))

    def test_capacity_failure_never_hides_incomplete_stage_in_checkpoint(self):
        _, items, plan, callbacks, *_ = fixture(2)
        sampler = CapacitySampler()
        sampler.can_submit_v17 = lambda _: False
        sampler.add_version(None, 0)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = V16bScheduler(sampler, plan=plan, protocol={"version": "error-note-v17"},
                                    items=items, grade=None, verify=None, out=Path(tmp), callbacks=callbacks)
            with self.assertRaisesRegex(RuntimeError, "no live"):
                runtime.submit_for_version(0)
            with self.assertRaisesRegex(RuntimeError, "complete engine drain"):
                runtime.save_drain_state()
            self.assertEqual(runtime.requests, {})

    def test_mixed_note_alone_holds_whole_update_and_checkpoint_stop_blocks_admission(self):
        sampler, items, plan, callbacks, _, checks, _ = fixture(4)
        original_rows = callbacks.rows
        def rows(kind, item, record):
            result = copy.deepcopy(original_rows(kind, item, record))
            for row in result:
                row["generation"]["mixed_version"] = kind == "NOTE"
            return result
        callbacks.rows = rows
        callbacks.checkpoint_due = lambda v: v == 1
        callbacks.checkpoint = lambda v, state, pending: {"stop_reasons": ["SIGNAL_FAIL"]}
        with tempfile.TemporaryDirectory() as tmp:
            result = run_v16b(None, Mock(), None, sampler, plan, out=Path(tmp),
                             protocol={"version": "error-note-v17"}, items=items, callbacks=callbacks,
                             local_sequential=True)
        self.assertEqual(result["stop_reasons"], ["SIGNAL_FAIL"])
        self.assertEqual(checks[-1][2]["external_hold"], "MIXED_VERSION")
        self.assertEqual(len(checks), 2)
        self.assertFalse(any(v > 1 for v, _, _ in sampler.admissions))

    def test_failed_native_admission_aborts_before_reservation_or_policy_reference_leaks(self):
        from unittest.mock import patch
        from fine_tuning.error_note.collect import single_request_sha256

        _, items, _, callbacks, *_ = fixture(1)
        request = callbacks.make_request(items[0], "T1", 0, {})
        request.update(problem_id=items[0]["problem_id"], policy_version=0, policy_sha256="a" * 64, block_id=0, kind="T1")
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.protocol_version = "error-note-v17"
        sampler.pending, sampler.versions = {}, {0: {"policy_sha256": "a" * 64, "request": object(), "lora_int_id": 1, "refcount": 0}}
        sampler.actual_kv_tokens, sampler.kv_block_size, sampler.max_num_seqs = 128, 16, 16
        sampler.sampling, sampler.eos_token_ids = request["sampling"], [9]
        core = Mock()
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=core))
        sampler._scheduler = lambda: SimpleNamespace(requests={request["request_id"]: SimpleNamespace(lora_request=None)})
        prepared = {"request_sha256": single_request_sha256(request), "prompt_tokens": 1,
                    "max_tokens": 8, "item": {"prompt_token_ids": [1]}, "prompt": {}}
        with patch("vllm.SamplingParams"), self.assertRaisesRegex(RuntimeError, "completion cap changed"):
            sampler.submit(request, prepared=prepared | {"max_tokens": 1})
        core.add_request.assert_not_called()
        with patch("vllm.SamplingParams"), self.assertRaisesRegex(RuntimeError, "request LoRA"):
            sampler.submit(request, prepared=prepared)
        core.abort_request.assert_called_once_with([request["request_id"]])
        self.assertEqual((sampler.pending, sampler.versions[0]["refcount"]), ({}, 0))

    def test_abort_releases_native_waiting_reservation_exactly_once(self):
        sampler = SingleVllmSampler.__new__(SingleVllmSampler)
        sampler.pending = {"r": {"request": {"policy_version": 1}, "kv_reserved_tokens": 48}}
        sampler.versions = {1: {"refcount": 1}}
        core = Mock()
        sampler.engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=core))
        sampler.abort_request_v17("r")
        core.abort_request.assert_called_once_with(["r"])
        self.assertEqual((sampler.pending, sampler.versions[1]["refcount"]), ({}, 0))
        with self.assertRaises(RuntimeError):
            sampler.abort_request_v17("r")


class GuardTests(unittest.TestCase):
    def test_zero_reference_floor_and_short_final_denominator(self):
        planned = order(final=8)
        guard = GuardV17(planned)
        for b, ids in enumerate(planned, 1):
            result = guard.update(b, batch(ids, short=0))
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["thresholds"]["truncated"]["minimum"], 4)
        self.assertEqual(result["short_drift"]["recent12"]["n"], 184)
        self.assertEqual(result["length_summary"]["n"], 568)
        self.assertEqual(result["sources"]["s"]["n"], 568)

    def test_short_warning_stop_polarization_and_resume(self):
        planned = order()
        guard = GuardV17(planned)
        for b, ids in enumerate(planned, 1):
            result = guard.update(b, batch(ids, short=2 if b <= 24 else 3, cap=0 if b <= 24 else 1))
            if b == 24:
                guard = GuardV17(planned, json.loads(json.dumps(guard.state)))
        self.assertEqual(result["short_drift"]["status"], "WARNING")
        self.assertEqual(result["warnings"], ["SHORT_DRIFT_WARNING", "LENGTH_POLARIZATION_WARNING"])
        guard = GuardV17(planned)
        for b, ids in enumerate(planned, 1):
            result = guard.update(b, batch(ids, short=2 if b <= 24 else 5))
        self.assertIn("SHORT_DRIFT_STOP", result["reasons"])
        self.assertEqual(GuardV17(planned).update(1, batch(planned[0]))["short_drift"]["status"], "WARMUP")

    def test_length_bins_empty_null_and_exact_edges(self):
        self.assertIsNone(length_distribution_v17([])["median"])
        receipt = length_distribution_v17([31, 32, 127, 128, 255, 256, 1023, 1024, 4095, 4096, 8191, 8192])
        self.assertEqual(list(receipt["histogram"].values()), [1, 2, 2, 2, 2, 2, 1])

    def test_bootstrap_preserves_missing_positions_without_normal_zero(self):
        items = [{"problem_id": str(i), "source": "s", "layer": "chart", "capability_type": "table"}
                 for i in range(3000)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"items": items}))
            result = falsealarm_receipt_v17(path, trials=2)
            self.assertEqual(result, falsealarm_receipt_v17(path, trials=2))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIsNone(result["overall_rate"])
        self.assertIsNone(result["rule_rates"]["TRUNC_UP"])
        self.assertEqual(result["missing_positions"], list(range(3000)))
        self.assertEqual(result["block_sizes"][-1], 8)

    def test_bootstrap_available_priors_keep_unmeasured_tail_and_is_null(self):
        prior = {"binding": "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD", "status": "CORRECT",
                 "completion_tokens": 100, "finish_reason": "stop"}
        items = [{"problem_id": str(i), "source": "s", "layer": "chart", "f0_prior": prior}
                 for i in range(3000)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"items": items}))
            result = falsealarm_receipt_v17(path, trials=2)
        self.assertEqual(result["overall_rate"], 0)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIsNone(result["rule_rates"]["TAIL_RATE"])
        self.assertGreater(result["overall_upper95"], .02)


class GuardFillAndAcceptanceTests(unittest.TestCase):
    def test_fill_uses_stratum_then_extra_source_then_all_and_gate_needs_scope_or_acceptance(self):
        import hashlib
        from fine_tuning.error_note.guard_v16b import receipt_passes_v17

        prior = {"binding": "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD", "status": "CORRECT",
                 "completion_tokens": 100, "finish_reason": "stop"}
        items = [{"problem_id": str(i), "source": "a" if i % 3 == 0 else "b" if i % 3 == 1 else "c",
                  "layer": "x", "f0_prior": prior if i % 3 == 0 else None} for i in range(3000)]
        with tempfile.TemporaryDirectory() as tmp:
            path, extra = Path(tmp) / "manifest.json", Path(tmp) / "extra.json"
            path.write_text(json.dumps({"items": items}))
            extra.write_text(json.dumps({"version": "v17-guard-extra-priors-v1", "sources": {"b": [
                {"status": "INCORRECT", "completion_tokens": 50, "finish_reason": "stop"}]}}))
            result = falsealarm_receipt_v17(path, trials=2, fill_missing=True, extra_priors=extra)
            raw = path.read_bytes()
        self.assertEqual(result["filled_levels"], {"stratum": 0, "source": 0, "source_extra": 1000, "all": 1000})
        self.assertEqual(result["unmeasured_rules"], ["IS_DRIFT", "TAIL_RATE"])
        self.assertEqual(result["status"], "INCOMPLETE")  # fewer than 2,000 trials
        good = result | {"trials": 2000, "overall_upper95": .01, "status": "PASS_SCOPED", "measured_status": "PASS"}
        self.assertTrue(receipt_passes_v17(good, raw))
        failed = good | {"overall_upper95": .03, "status": "FAIL", "measured_status": "FAIL"}
        self.assertFalse(receipt_passes_v17(failed, raw))
        accepted = failed | {"status": "RISK_ACCEPTED", "risk_acceptance": {
            "by": "main", "at_utc": "2026-11-17T00:00:00Z", "reason": "r", "consequence": "c"}}
        self.assertTrue(receipt_passes_v17(accepted, raw))
        self.assertFalse(receipt_passes_v17(accepted | {"overall_upper95": .06}, raw))
        self.assertFalse(receipt_passes_v17(accepted | {"risk_acceptance": {"by": "main"}}, raw))
        self.assertFalse(receipt_passes_v17(good, hashlib.sha256(b"x").digest()))


class SignalTests(unittest.TestCase):
    def test_fixed_support_checkpoint_and_real_content_gradient_required(self):
        planned = order(3)
        ids = [pid for block in planned for pid in block]
        gate = SignalGateV17(planned, list(reversed(ids)))
        self.assertEqual((gate.parents, gate.due_slot), (ids[:32], 4))
        stats = {"optimizer_step": True, "verified_repair_parent_ids": ids[:3],
                 "n_verified_content_tokens": 5, "n_content_gradient_finite_nonzero": True,
                 "note_gradient_retained": True}
        gate.consume(stats | {"optimizer_step": False})
        self.assertFalse(gate.state["pending_parent_ids"])
        gate.consume(stats)
        self.assertEqual(gate.evaluate(4, ids, ids)["status"], "CHECKPOINT_REQUIRED")
        gate.mark_checkpoint("sealed-checkpoint-sha")
        gate = SignalGateV17(planned, ids, json.loads(json.dumps(gate.state)))
        self.assertEqual(gate.evaluate(4, ids, ids)["status"], "PASS")
        self.assertEqual(gate.state["verified_repair_parent_ids"], ids[:3])

    def test_coverage_inconclusive_and_local_smoke_are_not_pass(self):
        planned = order(3)
        ids = [pid for block in planned for pid in block]
        with self.assertRaisesRegex(ValueError, "SIGNAL_COVERAGE_INSUFFICIENT"):
            SignalGateV17(planned, ids[:31])
        gate = SignalGateV17(planned, ids)
        self.assertEqual(gate.evaluate(4, ids, [])["status"], "INCONCLUSIVE_SIGNAL")
        self.assertEqual(SignalGateV17([["one"]], [], local_smoke=True).evaluate(3, ["one"], [])["status"],
                         "LOCAL_SMOKE_UNVERIFIED")


class PreflightTests(unittest.TestCase):
    def test_f0_transport_contract_does_not_certify_trained_power(self):
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})
        model.layer.lora_B.default.weight.data.zero_()
        bundle = SimpleNamespace(model=model)
        self.assertFalse(pf5_contract_v17(bundle, "f0")["require_power"])
        self.assertTrue(pf5_contract_v17(bundle, "trained")["require_power"])
        zeros = dict.fromkeys(("v_on", "v_llm", "v_off", "h_on", "h_llm", "h_off",
                               "va_on", "va_llm", "ha_on", "ha_llm"), [0., 0.])
        self.assertTrue(slope_check(**zeros, cap=.04, require_power=False)["pass"])
        self.assertFalse(slope_check(**zeros, cap=.04, require_power=True)["pass"])
        model.layer.lora_B.default.weight.data.fill_(.1)
        with self.assertRaisesRegex(ValueError, "every LoRA B"):
            pf5_contract_v17(bundle, "f0")


if __name__ == "__main__":
    unittest.main()
