"""CPU orchestration contracts with explicit hardware mocks, not BF16 evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fine_tuning.error_note import campaign_v16b as campaign
from fine_tuning.error_note import prep_v16b as prep
from fine_tuning.error_note import v17_eval, v17_preflight as pf
from fine_tuning.error_note.audit import digest


def measurement():
    values = {phase: {"samples": [{"seconds": seconds} for seconds in (1., 2., 3.)], "median_seconds": 2.}
              for phase in ("A", "B", "C")}
    return {"version": "v17-update-calibration-v1", "measurement": "cuda-bf16", "optimizer_steps": 0,
            "measurements": values}


def capacity():
    return {"version": "v17-capacity-v1", "model_parameter_update": False,
            "report": {"version": "v17-update-memory-v1", "optimizer_steps": 0, "fixture_only": True,
                       "action_tokens": 8192, "measurement": "actual_cuda_bf16_shared_backward_only"}}


class V17Preflight(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"base_commit": "a" * 40, "actual_files": {"error_note/source.py": "b" * 64}}
        self.plan = {"sha256": "c" * 64, "source": self.source, "local_sequential": False,
                     "protocol": {"version": "error-note-v17", "sha256": "d" * 64, "precision": "bf16", "image": {}},
                     "prep_batches": [], "model_path": str(self.root / "base")}
        self.identity = {"code_commit": self.source["base_commit"], "source_sha256": digest(self.source["actual_files"]),
                         "campaign_plan_sha256": self.plan["sha256"], "protocol_sha256": "d" * 64}
        self.args = NS(output=self.root, receipt_output=self.root / "functional.json", code_commit="a" * 40,
                       precision="bf16", engine="vllm", allow_model_run=True, data_root=self.root / "data",
                       token_budget=12000, local_sequential=False)

    def _receipts(self, *, low=False, no_a1=False):
        probe = v17_eval.seal({"version": "v17-eval-probe-v1", "slots": 8, "items": 16, "wall_seconds": 120,
            "generated_tokens": 48000, "sixty_second_rates": [350., 450.], "dtype": "bfloat16",
            "backend": "vllm-0.11.0", "config_sha256": "e" * 64, "max_observed_context_tokens": 35000,
            "cumulative_preemptions": 0, "disjoint_receipt_sha256": "f" * 64,
            "load_finalize_allowance_seconds": 60., "long_tail_allowance_seconds": 1000.,
            "long_tail_basis": "synthetic CPU receipt fixture", "source_sha256": self.identity["source_sha256"],
            "code_commit": self.identity["code_commit"], "environment_sha256": "e" * 64})
        calibrated = v17_eval.calibration_receipt(probe)
        self.args.calibration = self.root / "eval-calibration.json"
        pf._write(self.args.calibration, {k: v for k, v in calibrated.items() if k != "sha256"})
        self.args.calibration_sha256 = prep.file_sha(self.args.calibration)
        coverage = {"actual_a1_count": 0 if no_a1 else 3, "missing_actual_a1_lengths": ["long"] if low else [], "actual_full_rows": 3}
        cost = pf._write(self.root / "update-cost-calibration.json", {"version": "v17-update-cost-receipt-v1",
            "status": "CALIBRATION_LOW_COVERAGE" if low else "PASS", **self.identity, "optimizer_steps": 0,
            "coverage": coverage, "measurement": None if no_a1 else measurement(),
            "actual_abc_status": "UNMEASURED_NO_ADMITTED_A1" if no_a1 else "MEASURED"})
        functional = {"version": pf.VERSION, "status": "PASS", **self.identity, "optimizer_steps": 0,
            "production_parents_consumed": 0, "local_smoke": False, "capacity": capacity(),
            "f0_transport": {"pass": True}, "coexistence": [{"pass": True}], "update_cost_sha256": cost["sha256"],
            "update_cost_receipt": {"path": str(self.root / "update-cost-calibration.json"),
                                    "file_sha256": prep.file_sha(self.root / "update-cost-calibration.json")}}
        self.args.functional_receipt = self.root / "functional.json"
        pf._write(self.args.functional_receipt, functional)
        self.args.functional_receipt_sha256 = prep.file_sha(self.args.functional_receipt)
        return functional

    def test_identity_checks_commit_and_exact_source_bytes(self):
        with patch.object(campaign, "_plan_check"), patch.object(campaign.single, "source_manifest", return_value=self.source):
            self.assertEqual(pf._identity(self.args, self.plan), self.identity)
            self.args.code_commit = "e" * 40
            with self.assertRaisesRegex(ValueError, "commit, source bytes"):
                pf._identity(self.args, self.plan)
            self.args.code_commit = "a" * 40
            changed = deepcopy(self.plan)
            changed["source"]["actual_files"]["error_note/source.py"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "commit, source bytes"):
                pf._identity(self.args, changed)

    def test_bind_execution_is_immutable_and_never_runs_a_model(self):
        self._receipts()
        with patch.object(pf, "_identity", return_value=self.identity), patch.object(campaign, "load_f0") as load:
            first = pf.bind_execution(self.args, self.plan)
            self.assertEqual(first["status"], "BOUND")
            self.assertTrue(first["production_admitted"])
            self.assertEqual(pf.bind_execution(self.args, self.plan), first)
            load.assert_not_called()
        with self.args.functional_receipt.open("a") as stream:
            stream.write(" ")
        with patch.object(pf, "_identity", return_value=self.identity), self.assertRaisesRegex(ValueError, "bytes changed"):
            pf.bind_execution(self.args, self.plan)

    def test_low_timing_coverage_is_preserved_without_stopping_production(self):
        self._receipts(low=True)
        with patch.object(pf, "_identity", return_value=self.identity):
            bound = pf.bind_execution(self.args, self.plan)
        self.assertTrue(bound["production_admitted"])
        self.assertTrue(bound["cost_operator_review_required"])
        self.assertEqual(bound["unmeasured_timing_strata"], ["long"])

    def test_no_actual_a1_is_not_faked_as_measured_abc(self):
        self._receipts(low=True, no_a1=True)
        with patch.object(pf, "_identity", return_value=self.identity):
            self.assertTrue(pf.bind_execution(self.args, self.plan)["cost_operator_review_required"])
        bad = {"status": "PASS", "coverage": {"actual_a1_count": 0}, "measurement": None}
        with self.assertRaisesRegex(ValueError, "unmeasured"):
            pf._validate_cost(bad)

    def test_status_pass_does_not_replace_real_capacity_or_three_repeats(self):
        functional = self._receipts()
        functional["capacity"]["report"]["action_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "maximum FULL"):
            pf._validate_functional(functional)
        cost = prep.read_sealed(self.root / "update-cost-calibration.json")
        cost["measurement"]["measurements"]["B"]["samples"].pop()
        with self.assertRaisesRegex(ValueError, "three A/B/C"):
            pf._validate_cost(cost)

    def test_v18_functional_requires_the_v18_capacity_version(self):
        good = {"optimizer_steps": 0, "production_parents_consumed": 0, "f0_transport": {"pass": True},
                "coexistence": [{"pass": True}], "training_protocol_version": "error-note-v18",
                "capacity": capacity() | {"version": "v18-capacity-v1"}}
        pf._validate_functional(good)
        for bad in (good | {"capacity": capacity()},  # v18 run with a v17 capacity receipt
                    {k: v for k, v in good.items() if k != "training_protocol_version"}):  # v17 run, v18 receipt
            with self.assertRaisesRegex(ValueError, "FULL capacity"):
                pf._validate_functional(bad)
        pf._validate_functional({k: v for k, v in good.items() if k != "training_protocol_version"} | {"capacity": capacity()})

    def test_stale_plan_and_missing_functional_file_are_rejected_before_gpu(self):
        self._receipts()
        changed = self.identity | {"campaign_plan_sha256": "0" * 64}
        with patch.object(pf, "_identity", return_value=changed), self.assertRaisesRegex(ValueError, "stale"):
            pf.bind_execution(self.args, self.plan)
        self.args.functional_receipt = None
        with patch.object(pf, "_identity", return_value=self.identity), self.assertRaisesRegex(ValueError, "path and SHA256"):
            pf.bind_execution(self.args, self.plan)

    def test_local_nf4_is_an_explicit_skip_not_a_pass(self):
        plan = self.plan | {"local_sequential": True}
        with patch.object(campaign, "load_f0") as load:
            skipped = pf.calibrate(self.args, plan)
            bound = pf.bind_execution(self.args, plan)
            load.assert_not_called()
        self.assertEqual(skipped["status"], "SKIPPED_LOCAL_SMOKE")
        self.assertEqual(bound["status"], "LOCAL_SMOKE_EXEMPT")
        self.assertFalse(bound["production_admitted"])

    def _mock_calibration(self, *, accepted=True, transport=True):
        from fine_tuning.error_note import rl, train_v17, v17_data, v17_rows, protocol, run_v16

        prior = pf._write(self.root / "prior.json", {"overall": .5, "source_rates": {"fixture": .5}})
        self.plan["inputs"] = {"baseline": {"path": str(self.root / "prior.json"), "file_sha256": prep.file_sha(self.root / "prior.json")}}
        item = {"problem_id": "perf-only", "source": "fixture", "layer": "graph", "image_paths": ["fixture.png"],
                "image_sha256": ["e" * 64], "evaluation": {"gold": "A", "choices": ["first", "second"]}}
        sampler = Mock(preflight_report={"pass": transport}, coexistence_reports=[{"pass": True}])
        sampler.has_unfinished_requests.return_value = False
        bundle = NS(processor=NS(tokenizer=object()))
        events = []

        def collect(*args, **kwargs):
            events.append("collect")
            baseline = kwargs["baseline_receipts"][item["problem_id"]]
            self.assertEqual(baseline["prior_sha256"], digest(prior))
            self.assertEqual(kwargs["split"], "performance")
            self.assertEqual(kwargs["sampling"]["top_p"], 1.)
            output = {"text": "Independent synthetic fixture.\nFinal answer: A", "finish_reason": "stop", "status": "CORRECT",
                      "extracted": "A", "prompt": "P0", "generation": {"completion_token_ids": [1, 2], "baseline_receipt": baseline}}
            return [{"problem_id": item["problem_id"], "output": output}]

        patches = [patch.object(pf, "_identity", return_value=self.identity), patch.object(v17_data, "validate", return_value={"valid": True}),
            patch.object(v17_data, "load_items", return_value=[item]), patch.object(run_v16, "_setup_execution"),
            patch.object(campaign, "load_f0", return_value=(bundle, {})), patch.object(campaign, "calibrate_capacity", return_value=capacity()),
            patch.object(campaign, "make_sampler", return_value=sampler), patch.object(prep, "collect", side_effect=collect),
            patch.object(prep, "p0", return_value="P0"), patch.object(v17_rows, "problem_contract_v17", return_value={"verifier": None, "fact_catalog": [], "rules": {}}),
            patch.object(v17_rows, "verify_solution_v17", return_value={"accepted": accepted}),
            patch.object(v17_rows, "a1_row_v17", return_value={"kind": "A1", "row_id": "perf-only:A1", "route": "FULL"} if accepted else None),
            patch.object(campaign, "_encode", side_effect=lambda b, r, *a: [x | {"target_len": 20} for x in r]),
            patch.object(rl, "policy_sha256", return_value="f" * 64), patch.object(protocol, "v17_contract_for_update", return_value={}),
            patch.object(train_v17, "calibrate_update_v17", return_value=measurement())]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        return sampler, mocks[-1], events

    def test_calibration_collects_only_performance_with_presealed_baselines(self):
        sampler, timing, events = self._mock_calibration()
        receipt = pf.calibrate(self.args, self.plan)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(events, ["collect"])
        self.assertEqual(receipt["production_parents_consumed"], 0)
        cost = prep.read_sealed(self.root / "update-cost-calibration.json")
        self.assertEqual(cost["status"], "CALIBRATION_LOW_COVERAGE")
        self.assertEqual(cost["actual_abc_status"], "MEASURED")
        timing.assert_called_once()
        sampler.release.assert_called_once()
        with self.assertRaises(FileExistsError):
            pf.calibrate(self.args, self.plan)

    def test_absent_a1_keeps_actual_timing_unmeasured_and_capacity_separate(self):
        sampler, timing, _ = self._mock_calibration(accepted=False)
        receipt = pf.calibrate(self.args, self.plan)
        self.assertEqual(receipt["status"], "PASS")
        cost = prep.read_sealed(self.root / "update-cost-calibration.json")
        self.assertEqual(cost["actual_abc_status"], "UNMEASURED_NO_ADMITTED_A1")
        self.assertIsNone(cost["measurement"])
        self.assertFalse(cost["capacity_fixture_is_solution_evidence"])
        timing.assert_not_called()
        sampler.release.assert_called_once()

    def test_failed_transport_never_writes_passing_receipt_and_releases(self):
        sampler, _, _ = self._mock_calibration(transport=False)
        with self.assertRaisesRegex(ValueError, "transport/coexistence"):
            pf.calibrate(self.args, self.plan)
        self.assertFalse(self.args.receipt_output.exists())
        sampler.release.assert_called_once()

    def test_actual_export_filename_and_ready_shape_with_pinned_execution_identity(self):
        from fine_tuning.error_note.protocol import V16B_CONTRACT_HASHES, plain_protocol_v17

        self.source = campaign.single.source_manifest()
        protocol = plain_protocol_v17("bf16", rl_blocks=188, prep_updates=0,
            contracts={name: "0" * 64 for name in V16B_CONTRACT_HASHES})
        items = [{"problem_id": f"fixture-{index}"} for index in range(3000)]
        blocks = [[item["problem_id"] for item in items[start:start + 16]] for start in range(0, len(items), 16)]
        body = {"version": campaign.CAMPAIGN_V17, "source": self.source, "local_sequential": False,
                "protocol": protocol, "items": items, "blocks": blocks, "prep_batches": [], "time_cut": None,
                "total_slots": protocol["training_contract"]["total_slots"],
                "capacity": {"engine": "vllm", "max_num_seqs": 16}, "model_path": str(self.root / "base")}
        self.plan = pf._write(self.root / "campaign-plan.json", body)
        self.args.code_commit = self.source["base_commit"]
        self.identity = pf._identity(self.args, self.plan)
        self._receipts()
        checkpoint = self.root / "checkpoint-final"
        checkpoint.mkdir()
        metadata = checkpoint / "metadata.json"
        metadata.write_text("{}")
        descriptor = campaign.descriptor(metadata)
        pf._write(self.root / "campaign-state.json", {"status": "COMPLETE", "stop_reasons": [],
            "plan_sha256": self.plan["sha256"], "checkpoint": descriptor,
            "cursor": {"slot": self.plan["total_slots"], "phase": "final"}})
        pf._write(self.root / "monitor-final.json", {"plan_sha256": self.plan["sha256"], "checkpoint": descriptor})
        pf._write(self.root / "evaluation-admission.json", {"plan_sha256": self.plan["sha256"],
            "checkpoint": descriptor, "decision": {"status": "EVALUATION_CANDIDATE"}})
        self.args.export_output = self.root / "export"
        self.args.export_output.mkdir()
        model_ref = self.args.export_output / "model_ref.json"
        model_ref.write_text('{"kind":"unmerged-lora"}')
        self.args.preflight_output = self.root / "trained-check.json"
        self.args.eval_protocol = self.root / "eval.yaml"
        self.args.hardware = self.root / "hardware.yaml"
        self.args.base = self.root / "base"

        def gpu_check(ref, saved, binding, base, eval_protocol, hardware, out):
            self.assertEqual(ref, model_ref)
            self.assertTrue(ref.is_file())
            self.assertEqual(saved, checkpoint)
            self.assertEqual(base.resolve(), Path(self.plan["model_path"]).resolve())
            self.assertEqual(prep.read_sealed(binding)["binding"], campaign._binding(self.plan))
            out.write_text('{"pass":true}')
            return {"pass": True}

        with patch("fine_tuning.error_note.lora_preflight.check", side_effect=gpu_check) as check:
            result = pf.trained_export_preflight(self.args, self.plan)
        ready = prep.read_sealed(self.root / "export-ready.json")
        self.assertEqual(ready["checkpoint"], str(checkpoint))
        self.assertEqual(ready["metadata_descriptor"], descriptor)
        self.assertEqual(result["model_ref_file_sha256"], prep.file_sha(model_ref))
        self.assertEqual(result["code_commit"], self.source["base_commit"])
        check.assert_called_once()
        self.args.base = self.root / "different-base"
        with patch("fine_tuning.error_note.lora_preflight.check") as check:
            with self.assertRaisesRegex(ValueError, "pinned model path"):
                pf.trained_export_preflight(self.args, self.plan)
            check.assert_not_called()

    def test_trained_preflight_uses_ready_checkpoint_and_existing_checker(self):
        self.args.export_output = self.root / "export"
        self.args.export_output.mkdir()
        (self.args.export_output / "model_ref.json").write_text("{}")
        self.args.preflight_output = self.root / "preflight.json"
        self.args.eval_protocol = self.root / "eval.yaml"
        self.args.hardware = self.root / "hardware.yaml"
        self.args.base = None
        ready = {"plan_sha256": self.plan["sha256"], "checkpoint": str(self.root / "checkpoint")}
        pf._write(self.root / "export-ready.json", ready)

        def check(*args):
            self.assertEqual(args[1], self.root / "checkpoint")
            self.assertEqual(prep.read_sealed(args[2])["binding"], {"exact": "binding"})
            args[-1].write_text('{"pass":true}')
            return {"pass": True}

        with patch.object(pf, "bind_execution", return_value={"sha256": "e" * 64}), patch.object(pf, "_identity", return_value=self.identity), \
                patch.object(campaign, "require_complete", return_value=ready), patch.object(campaign, "_binding", return_value={"exact": "binding"}), \
                patch("fine_tuning.error_note.lora_preflight.check", side_effect=check) as checker:
            result = pf.trained_export_preflight(self.args, self.plan)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["trained_effect_required"])
        checker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
