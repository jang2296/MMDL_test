"""Authored orchestration/contract tests; these are not CUDA measurements."""
import copy
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tuning.notebook_rl import approval, gate_sequence as seq
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.schema import load_protocol
from fine_tuning.tests import test_notebook_rl_approval as fixture


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(Path(seq.__file__).with_name("protocol_v1.yaml"))
        self.probes = [SimpleNamespace(request=RolloutRequest(str(n), None, [{"role": "user", "content": "probe"}],
            [object() for _ in range(n)], 1, 4, [3407]), completion_ids=[10, 11, 2]) for n in (1, 2)]
        self.runtime = {"device": {"index": 0}, "model_file_count": 13}

    def test_initial_components_order_and_actual_request_counts(self):
        calls, rows = [], []
        engine = object()
        def measured(name):
            def run(*args, **kwargs):
                calls.append(name)
                if name == "G7_INITIAL":
                    requests = args[2]
                    self.assertEqual(sum(r.n for r in requests), 32)
                    self.assertEqual([len(r.images) for r in requests], [1, 2])
                    self.assertEqual([len(r.seeds) for r in requests], [16, 16])
                    self.assertEqual(kwargs["stage"], "INITIAL")
                return {"authored": name}
            return run
        with ExitStack() as stack:
            stack.enter_context(patch.object(seq.gate_weights, "require_idle_engine", return_value={
                "device": "cuda:0", "policy_version": 0, "in_process": True}))
            for owner, method, name in ((seq.gate_images, "measure_image_parity", "G2"),
                    (seq.gate_logprobs, "measure_logprobs", "G7_INITIAL"),
                    (seq.gate_tokens, "measure_tokens", "G8"), (seq.gate_cache, "measure_current", "G6_BEFORE")):
                stack.enter_context(patch.object(owner, method, side_effect=measured(name)))
            value = seq.before_update(object(), engine, self.probes, self.protocol, self.runtime,
                seed=3407, record=rows.append, should_stop=lambda: False)
        self.assertEqual(calls, ["G2", "G7_INITIAL", "G8", "G6_BEFORE"])
        self.assertEqual(list(value), ["G0", *calls])
        self.assertEqual([r["gate"] for r in rows], list(value))
        self.assertEqual([p.request.n for p in self.probes], [1, 1])

    def test_seeded_probes_stable_and_stage_specific(self):
        self.assertEqual(seq._logprob_requests(self.probes, 0, 3407), seq._logprob_requests(self.probes, 0, 3407))
        self.assertNotEqual(seq._logprob_requests(self.probes, 0, 3407)[0].seeds,
                            seq._logprob_requests(self.probes, 1, 3407)[0].seeds)

    def test_before_engine_reference_has_no_rollout_engine_and_stops_on_deadline(self):
        events = []
        bundle = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:0")])))
        preparation = SimpleNamespace(model_inputs=Mock(return_value={}))
        with patch.object(seq.gpu_gates, "measure_inventory", side_effect=lambda *a: events.append("G1") or {}), \
                patch.object(seq, "_reference", side_effect=lambda *a, **k: events.append(k) or {}), \
                patch.object(seq.gate_gradients, "measure_gradient_routes", return_value={}), \
                patch.object(seq.gate_numerics, "measure_numerics", return_value={}):
            measured = seq.before_engine(bundle, preparation, self.probes, Path("unused"), self.protocol,
                record=lambda r: None, should_stop=lambda: False)
        self.assertEqual(events, ["G1", {"stage": "INITIAL"}])
        self.assertEqual(list(measured), ["G1", "G3_INITIAL", "G9", "G10"])
        self.assertIs(preparation.model_inputs.call_args.args[0], self.probes[1].request)
        rows = []
        with self.assertRaises(TimeoutError):
            seq._record("G1", {"measured": True}, rows.append, lambda: True)
        self.assertEqual(rows, [{"kind": "GPU_GATE_COMPONENT", "gate": "G1", "measurement": {"measured": True}}])

    def test_post_sequence_never_uses_live_step2(self):
        for engine_version, preparation_version in ((2, 1), (1, 2), (0, 0)):
            with self.subTest(engine_version=engine_version), \
                    patch.object(seq.gate_weights, "measure_weight_parity") as weights, \
                    self.assertRaisesRegex(ValueError, "shared step1"):
                seq.after_update(object(), SimpleNamespace(policy_version=engine_version),
                    SimpleNamespace(policy_version=preparation_version), self.probes, Path("unused"), self.protocol,
                    seed=3407, record=lambda r: None, should_stop=lambda: False)
            weights.assert_not_called()
        calls = []
        with ExitStack() as stack:
            for owner, method, name in ((seq.gate_weights, "measure_weight_parity", "G5"),
                    (seq.gate_cache, "measure_current", "G6_AFTER"),
                    (seq.gate_logprobs, "measure_logprobs", "G7_POST_SYNC"), (seq, "_reference", "G3_POST_SYNC")):
                stack.enter_context(patch.object(owner, method, side_effect=lambda *a, _n=name, **k: calls.append(_n) or {}))
            seq.after_update(object(), SimpleNamespace(policy_version=1), SimpleNamespace(policy_version=1),
                self.probes, Path("unused"), self.protocol, seed=3407, record=lambda r: None, should_stop=lambda: False)
        self.assertEqual(calls, ["G5", "G6_AFTER", "G7_POST_SYNC", "G3_POST_SYNC"])

    def measurements(self):
        # Authored scalar contract fixture, never a persisted gate proof.
        gates = fixture.ApprovalTests.gpu_measurements(None)
        measured = {k: gates[k] for k in ("G0", "G1", "G2", "G5", "G8", "G9", "G10")}
        for stage, version in (("INITIAL", 0), ("POST_SYNC", 1)):
            measured["G3_"+stage] = gates["G3"] | {"stage": stage, "base_sha256": "a"*64, "native_execution": True}
            measured["G7_"+stage] = gates["G7"] | {"stage": stage, "policy_version": version, "native_execution": True}
        shared = {"G11": gates["G11"] | {"G4": gates["G4"]}, "G13": gates["G13"]}
        replay = {"native_execution": True, "passed": True, "fresh_process": True, "next_loss_equal": True,
            "different_fields": [], "logit_errors": {f"{i}.{kind}.{field}": {"bit_equal": True}
                for i in (0, 1) for kind in ("policy", "reference") for field in ("logits", "logprobs")}}
        return [measured, shared, gates["G6"] | {"native_execution": True}, replay, self.protocol,
                {"arm": "MOD-FML", "device": {"total_memory_bytes": 1000}}]

    def test_all_scalar_contracts_and_g12_real_producer_field_mapping(self):
        args = self.measurements()
        before = copy.deepcopy(args)
        measured = seq.combine(*args)
        self.assertEqual(args, before)
        self.assertEqual(set(measured), {f"G{i}" for i in range(14)})
        self.assertEqual(measured["G7"]["trajectories"], 64)
        self.assertTrue(measured["G12"]["policy_logits_equal"])
        self.assertTrue(measured["G12"]["reference_logits_equal"])
        self.assertNotIn("training_allowed", measured)

    def test_unmeasured_or_wrong_stage_and_incomplete_roundtrip_are_not_pass(self):
        changes = [(0, "G3_POST_SYNC", "stage", "INITIAL"), (0, "G3_POST_SYNC", "base_sha256", "b"*64),
            (0, "G7_POST_SYNC", "policy_version", 2), (0, "G7_INITIAL", "trajectories", 31),
            (0, "G7_INITIAL", "native_execution", False), (0, "G7_POST_SYNC", "delta_seq", .1),
            (0, "G7_INITIAL", "step0_mean_abs", 1.)]
        for index, gate, field, value in changes:
            args = self.measurements()
            args[index][gate][field] = value
            with self.subTest(gate=gate, field=field), self.assertRaises((ValueError, approval._Blocked)):
                seq.combine(*args)
        for field, value in (("fresh_process", False), ("passed", False), ("different_fields", ["update"]),
                             ("logit_errors", {}), ("next_loss_equal", False)):
            args = self.measurements()
            args[3][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, approval._Blocked)):
                seq.combine(*args)
        args = self.measurements()
        args[3]["logit_errors"]["1.reference.logits"]["bit_equal"] = False
        with self.assertRaisesRegex(approval._Blocked, "G12"):
            seq.combine(*args)


if __name__ == "__main__":
    unittest.main()
