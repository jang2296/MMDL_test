"""CPU checks for the actual v18b evaluation entry points and their failure boundaries."""
from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fine_tuning.error_note import v17_eval as ev, v17_eval_engine as engine, v18b_eval as policy
from fine_tuning.notebook_rl import proof_eval as pe
from fine_tuning.tests.test_error_note_v17_split_engine import backend_value, engine_receipt
from mmdl.runtime.artifacts import digest, sha256_file, write_json


def settings(strategy="one_pod"):
    return {"run_id": "v18b-eval-test", "first_billed_at": "2026-11-19T00:00:00Z",
            "total_timeout_seconds": 50400, "recovery_reserve_seconds": 600,
            "post_eval_reserve_seconds": 1200, "evaluation_strategy": strategy,
            "training_cutoff_seconds": 24000 if strategy == "training_cutoff" else None,
            "monitor_stop_action": "evaluate_checkpoint", "reuse_f0_raw": False,
            "billing": [{"first_billed_at": "2026-11-19T00:00:00Z", "usd_per_hour": 1.4}] +
                       ([{"first_billed_at": "2026-11-19T08:00:00Z", "usd_per_hour": 1.4}]
                        if strategy == "two_pod" else [])}


def v18b_receipt(root):
    value = engine_receipt(root)
    observed = engine.observe_engine(backend_value(slots=9, blocks=30000), "eager", 9)
    selected = copy.deepcopy(value["selected_attempt"])
    for key in ("actual_length_probe", "long_context_probe", "quantiles"):
        del selected[key]
    tower = {"version": "v18-direct-vision-pf5-v1", "pass": True, "amplified": {"pass": True},
             "hook_removed": {"pass": False}, "real_effect_status": "UNCONFIRMED",
             "coordinates": 100, "inputs": [{"pixels_sha256": "a" * 64}]}
    evidence = selected["preflight_evidence"]
    for key in ("historical_pf5", "decode_pf5"):
        evidence[key].update(vision_gate="direct_tower", vision_tower=tower)
    selected["preflight_evidence"] = engine.seal(evidence)
    calibration = policy.live_calibration(observed, settings())
    selected.update(observed_engine=observed, calibration=calibration)
    identity = {"code_commit": value["code_commit"], "source_sha256": value["source_sha256"]}
    training_ref = {"path": "/cpu/training-artifacts.json", "file_sha256": value["training_artifact_sha256"]}
    decision = engine.seal({"version": "v18-evaluation-decision-v1", "decision": "PROCEED", "authorized_by": "user",
        "pre_authorized": False, "authorization": "CPU fixture decision", "training_artifact": training_ref,
        "evaluation": identity, "admission_status": "EVAL_REVIEW_REQUIRED",
        "model_ref": {"file_sha256": value["model_ref_sha256"]}})
    value.update(training_protocol_version="error-note-v18b", slots=9, safe_slots=9, observed_engine=observed,
        actual_kv_token_capacity=480000, selected_attempt=engine.seal(selected), calibration=calibration,
        execution_settings=settings(), evaluation_watchdog_seconds=50400, estimated_evaluation_seconds=None,
        artifact_verification={"path": "/cpu/verified.json", "file_sha256": "a" * 64},
        evaluation_authorization=engine.seal({"status": "PASS", "evaluation": identity,
            "training_artifact": training_ref, "evaluation_decision": decision}))
    return engine.seal(value)


class EngineChecks(unittest.TestCase):
    def test_real_attempt_skips_long_generators_but_calls_final_vision_and_decode_checks(self):
        from types import SimpleNamespace as NS
        from fine_tuning.error_note import llm_lora as lora, lora_preflight as pf
        from fine_tuning.notebook_rl import checkpoint
        from mmdl.evaluation import engine as evaluation
        from mmdl.evaluation.backends import vllm_backend
        import torch

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            ref = root / "model_ref.json"
            write_json(ref, {})
            bundle = NS(model=mock.Mock())
            prep = NS(tokenizer=NS(encode=lambda *args, **kwargs: [7]))
            backend = mock.Mock()
            observed = engine.observe_engine(backend_value(slots=9, blocks=30000), "eager", 9)
            context = {"config": {"image": {}, "protocol_id": "test"}, "hardware": {}, "model_ref": {},
                       "seal": {"checkpoint": {"root": "/cpu/checkpoint"}, "campaign_plan": {"path": "/cpu/plan"}},
                       "training_protocol_version": "error-note-v18b", "execution_settings": settings(), "requests": {}}
            for owner, name, result in ((checkpoint, "load_eval", bundle), (pe, "_binding_file", ({},)),
                    (pf, "check_files", {"pass": True}), (pf, "probe_preparation", prep),
                    (pf, "probe_pairs", [(NS(), [7])]), (lora, "hf_columns", {}),
                    (lora, "hf_vision_columns", {}), (lora, "vllm_vision_columns", {}),
                    (lora, "vision_tower_check", {"pass": True}),
                    (pf, "engine_checks", {"pass": True, "pf5": {"pass": True}}),
                    (engine, "decode_pf5", {"pass": True}), (engine, "probability_columns", []),
                    (engine, "observe_engine", observed), (vllm_backend, "VLLMBackend", backend)):
                stack.enter_context(mock.patch.object(owner, name, return_value=result))
            stack.enter_context(mock.patch.object(torch.cuda, "empty_cache"))
            stack.enter_context(mock.patch.object(evaluation, "verify_model"))
            stack.enter_context(mock.patch.object(pe, "registered_variants", lambda **kwargs: nullcontext()))
            stack.enter_context(mock.patch.object(engine, "engine_execution", lambda *args: nullcontext()))
            stack.enter_context(mock.patch.object(engine, "diagnostic_timing", lambda: nullcontext({})))
            forced = stack.enter_context(mock.patch.object(engine, "probe_generation", side_effect=AssertionError("forced generation")))
            result = engine._attempt(NS(base_path=root, model_ref=ref, requests=root / "requests.json"), "eager", root / "attempt", context, slots=9)
            self.assertEqual(result["status"], "MEASURED", result["reasons"])
            forced.assert_not_called()
            lora.hf_vision_columns.assert_called_once()
            engine.decode_pf5.assert_called_once()
            self.assertNotIn("long_context_probe", result)

    def test_short_receipt_retains_all_v18_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            value = v18b_receipt(Path(directory))
            self.assertEqual(engine.validate_engine_receipt(value), value)
            self.assertIsNone(value["estimated_evaluation_seconds"])
            for change in ("vision", "language", "authorization", "forced", "billing"):
                bad = copy.deepcopy(value)
                selected = bad["selected_attempt"]
                evidence = selected["preflight_evidence"]
                if change == "vision":
                    evidence["decode_pf5"]["vision_tower"]["hook_removed"]["pass"] = True
                elif change == "language":
                    evidence["historical_pf5"]["pass"] = False
                elif change == "authorization":
                    bad["evaluation_authorization"]["evaluation_decision"]["authorization"] = ""
                    bad["evaluation_authorization"]["evaluation_decision"] = engine.seal(bad["evaluation_authorization"]["evaluation_decision"])
                    bad["evaluation_authorization"] = engine.seal(bad["evaluation_authorization"])
                elif change == "forced":
                    selected["long_context_probe"] = {}
                else:
                    bad["execution_settings"]["first_billed_at"] = "2026-11-19T00:01:00Z"
                selected["preflight_evidence"] = engine.seal(evidence)
                bad["selected_attempt"] = engine.seal(selected)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    engine.validate_engine_receipt(engine.seal(bad))

    def test_short_deadlines_and_live_estimate_are_honest(self):
        self.assertEqual(engine.validation_timeout_seconds(300, "eager", training_protocol_version="error-note-v18b"), 660)
        self.assertLess(engine.validation_timeout_seconds(300, "auto", training_protocol_version="error-note-v18b"), 1800)
        live = policy.progress(elapsed=100, tokens=1000, completed=2, expected=900,
                               historical_tokens=4_116_922, remaining_seconds=200, preemptions=2)
        self.assertEqual(live["generated_tokens_per_second"], 10)
        self.assertTrue(live["estimate_exceeds_remaining_time"])
        self.assertFalse(live["automatic_retry"])

    def test_partitions_never_choose_an_answer_and_policies_are_exclusive(self):
        ids = [f"item-{i}" for i in range(900)]
        both = policy.partitions(ids, settings("two_pod"))
        self.assertEqual([len(value) for value in both.values()], [450, 450])
        self.assertFalse(set(both["half-a"]) & set(both["half-b"]))
        self.assertEqual(sorted(both["half-a"] + both["half-b"]), sorted(ids))
        self.assertEqual(both, policy.partitions(list(reversed(ids)), settings("two_pod")))
        for strategy in ("one_pod", "training_cutoff"):
            self.assertEqual(policy.partitions(ids, settings(strategy)), {"all": sorted(ids)})
        bad = settings("two_pod") | {"training_cutoff_seconds": 30000}
        with self.assertRaises(ValueError):
            policy.partitions(ids, bad)


class OfficialDeadlineChecks(unittest.TestCase):
    def test_live_telemetry_runs_inside_the_actual_stream_hooks(self):
        from fine_tuning.tests.test_error_note_v17_split_eval import SplitEvaluationTests
        helper = SplitEvaluationTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        registration = helper.registration() | {"execution_settings": settings()}
        helper.exercise(registration)
        heartbeat = json.loads((helper.root / "run/heartbeat.json").read_text())
        self.assertEqual(heartbeat["requests_completed"], 7)
        self.assertEqual(heartbeat["v18b_live"]["first_billed_at"], settings()["first_billed_at"])
        self.assertGreater(heartbeat["v18b_live"]["generated_tokens_per_second"], 0)
        self.assertIn("preemptions_observed", heartbeat["v18b_live"])

    def clock_fixture(self, decisions):
        from fine_tuning.error_note.prep_v16b import seal as seal_file
        from fine_tuning.error_note.v18b_execution import deadline_epoch
        from mmdl.evaluation import cli

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            paths = {key: root / key for key in ("prereg", "ledger", "calibration", "preflight", "seal", "ref", "protocol", "hardware")}
            for path in paths.values():
                write_json(path, {})
            cfg = {"protocol_id": pe.FINAL_VARIANT_V8}
            write_json(paths["protocol"], cfg)
            sealed_settings = root / "settings.json"
            seal_file(sealed_settings, {"version": "v18b-execution-settings-v1", "settings": decisions})
            ids = [f"item-{i:03}" for i in range(900)]
            declared = {"model_sha256": "a" * 64, "model_ref_sha256": sha256_file(paths["ref"])}
            validation = {"slots": 9, "training_protocol_version": "error-note-v18b", "execution_settings": decisions,
                "selected_mode": "eager", "environment_sha256": "b" * 64, "observed_engine": {}, "evaluation_commit": "a" * 40}
            prereg = {"format": pe.FORMAT_V12, "training_protocol_version": "error-note-v18b", "execution_settings": decisions,
                "calibration_sha256": sha256_file(paths["calibration"]), "evaluation_watchdog_seconds": 50400,
                "engine_validation": validation, "engine_validation_sha256": sha256_file(paths["calibration"]),
                "training_artifacts_file_sha256": sha256_file(paths["seal"]), "arms": {pe.ARM_SINGLE: declared},
                "evaluation_subsets": {"all": {"ids": ids}}, "execution_contract": {"version": "v17-evaluation-execution-v2"},
                "protocol_variants": {pe.FINAL_VARIANT_V8: {"config_digest": digest(cfg)}}}
            start = {"event": "START", "run_id": decisions["run_id"], "preregistration_sha256": "c" * 64,
                "arm": pe.ARM_SINGLE, "variant": pe.FINAL_VARIANT_V8, "subset": "all", "time": "2026-11-19T12:00:00+00:00"}
            for owner, name, result in ((pe, "load_preregistration", (prereg, "c" * 64)),
                    (pe, "read_ledger", ([start], "d" * 64)),
                    (pe, "_lora_model_ref", ({"export": {"training_protocol": {"version": "error-note-v18b"}, "campaign": {"stage": "final"}}}, declared)),
                    (pe, "gpu_info", ("CPU fixture", 0)), (pe, "variant_registration", {"slots": 9}),
                    (ev, "load_calibration", validation), (ev, "split_bindings", validation),
                    (ev, "calibration_timeout", 50400), (ev, "cpu_resume_finalize", False)):
                stack.enter_context(mock.patch.object(owner, name, return_value=result))
            stack.enter_context(mock.patch.dict("os.environ", {pe.VLLM_INPROCESS_ENV: "0"}))
            stack.enter_context(mock.patch.object(ev, "source_sha256", return_value="e" * 64))
            backend = stack.enter_context(mock.patch.object(cli, "main", side_effect=RuntimeError("model entry")))
            stack.enter_context(mock.patch.object(pe, "registered_variants", lambda **kwargs: nullcontext()))
            stack.enter_context(mock.patch.object(engine, "engine_execution", lambda *args: nullcontext()))
            seen = []

            @contextmanager
            def hooks(run, registration, deadline):
                seen.append(deadline)
                yield None

            stack.enter_context(mock.patch.object(ev, "evaluation_hooks", hooks))
            common = ["--prereg", str(paths["prereg"]), "--ledger", str(paths["ledger"]), "--calibration", str(paths["calibration"]),
                "--preflight", str(paths["preflight"]), "--engine-validation", str(paths["calibration"]),
                "--training-artifacts", str(paths["seal"]), "--execution-settings", str(sealed_settings), "--"]
            forwarded = ["--protocol", str(paths["protocol"]), "--hardware", str(paths["hardware"]), "--model-ref", str(paths["ref"]),
                "--model-path", str(root / "model"), "--artifact-root", str(root), "--data-root", str(root / "data"),
                "--base-path", str(root / "base"), "--run-id", decisions["run_id"], "--mode", "partial"]
            forwarded += [arg for sid in ids for arg in ("--sample-id", sid)]
            hard = deadline_epoch(decisions, reserve_seconds=decisions["post_eval_reserve_seconds"])
            with mock.patch.object(ev.time, "time", return_value=hard - 60), self.assertRaisesRegex(RuntimeError, "model entry"):
                ev.main(["run", *common, *forwarded])
            run = root / "runs" / decisions["run_id"]
            run.mkdir(parents=True)
            error, message = ((RuntimeError, "model entry") if decisions.get("limit_action") == "warn" else
                              (TimeoutError, "first-billing"))
            with mock.patch.object(ev.time, "time", return_value=hard + 1), self.assertRaisesRegex(error, message):
                ev.main(["resume", *common, *forwarded])
            if decisions.get("limit_action") == "warn":
                self.assertEqual(seen, [None, None])
                self.assertEqual(backend.call_count, 2)
            else:
                self.assertEqual(seen, [hard])
                self.assertEqual(backend.call_count, 1)
            self.assertEqual(json.loads((run / "v17_state.json").read_text())["status"], "INTERRUPTED_RESUMABLE")


    def test_original_billing_caps_the_canonical_run_and_resume_before_model_load(self):
        self.clock_fixture(settings())

    def test_completion_run_and_resume_load_after_the_planning_time(self):
        decisions = settings() | {"limit_action": "warn", "budget_limit_usd": 22.,
                                 "total_timeout_seconds": 86400}
        self.clock_fixture(decisions)


class ProductionArtifactChecks(unittest.TestCase):
    def test_real_model_ref_reader_accepts_only_the_authorized_preparation_checkpoint(self):
        from fine_tuning.error_note import protocol

        policy = protocol.plain_protocol_v18b("bf16", rl_blocks=188, prep_updates=7,
            contracts={key: "a" * 64 for key in protocol.V16B_CONTRACT_HASHES})
        section = pe.lora_section()
        selection = {"reason": "MONITOR_STOP", "authorized_action": "evaluate_checkpoint", "consumed_items": 0,
                     "total_items": 3000, "source_checkpoint_sha256": "c" * 64, "preparation_only": True}
        model = {"kind": pe.LORA_KIND, "dtype": "bfloat16", "base": {"id": pe.MODEL_ID, "revision": pe.MODEL_REVISION},
            "processor_revision": pe.MODEL_REVISION, "llm_lora": section,
            "files": [{"path": name, "bytes": 1, "sha256": "a" * 64} for name in section["files"]],
            "train_config_sha256": digest(policy), "data_manifest_sha256": "b" * 64,
            "export": {"training_protocol": policy, "arm": protocol.ARM, "step": 7,
                "checkpoint_files": {"metadata.json": "c" * 64},
                "binding": {"protocol_sha256": policy["sha256"], "prompts_sha256": policy["prompts_sha256"], "data_sha256": "b" * 64},
                "campaign": {"stage": "partial", "slot": 7, "updates": 7, "plan_sha256": "b" * 64,
                             "evaluation_selection": selection}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_ref.json"
            write_json(path, model)
            checked, declared = pe._lora_model_ref(path)
            self.assertEqual(checked, model)
            self.assertEqual(declared["checkpoint_metadata_sha256"], "c" * 64)
            self.assertIn("0/3000", pe._single_definition(checked))
            selection["reason"] = "TRAINING_CUTOFF"
            write_json(path, model)
            with self.assertRaisesRegex(ValueError, "only a monitor stop"):
                pe._lora_model_ref(path)
            selection.update(consumed_items=3000, preparation_only=False)
            contract = policy["training_contract"]
            first = contract["prep_updates"] + contract["rl_blocks"]
            for slot in range(first, contract["total_slots"] + 1):
                model["export"]["step"] = slot
                model["export"]["campaign"].update(slot=slot, updates=slot)
                write_json(path, model)
                with self.subTest(slot=slot):
                    self.assertEqual(pe._lora_model_ref(path)[0], model)
            for count, reason, slot in ((3001, "TRAINING_CUTOFF", first), (3000, "MONITOR_STOP", first),
                                        (3000, "TRAINING_CUTOFF", first - 1),
                                        (2999, "TRAINING_CUTOFF", contract["total_slots"]),
                                        (3000, "TRAINING_CUTOFF", contract["total_slots"] + 1)):
                selection.update(consumed_items=count, reason=reason)
                model["export"]["step"] = slot
                model["export"]["campaign"].update(slot=slot, updates=slot)
                write_json(path, model)
                with self.subTest(count=count, reason=reason, slot=slot), self.assertRaises(ValueError):
                    pe._lora_model_ref(path)

    def test_zero_main_model_ref_requires_explicit_postprep_monitor_selection(self):
        from fine_tuning.error_note import v17_artifacts as artifacts

        selection = {"reason": "MONITOR_STOP", "authorized_action": "evaluate_checkpoint", "consumed_items": 0,
                     "total_items": 3000, "source_checkpoint_sha256": "a" * 64, "preparation_only": True}
        model = {"export": {"training_protocol": {"training_contract": {"prep_updates": 7}},
                           "campaign": {"stage": "partial", "slot": 7, "evaluation_selection": selection}}}
        artifact = {"evaluation_selection": selection, "checkpoint": {"metadata": {"file_sha256": "a" * 64}}}
        self.assertEqual(artifacts._partial_selection(artifact, model), selection)
        for change in ({"reason": "TRAINING_CUTOFF"}, {"reason": "STOP_REQUESTED"},
                       {"preparation_only": False}, {"preparation_only": True, "consumed_items": 16}):
            bad = selection | change
            changed_model = copy.deepcopy(model)
            changed_model["export"]["campaign"]["evaluation_selection"] = bad
            with self.subTest(change=change), self.assertRaises(ValueError):
                artifacts._partial_selection(artifact | {"evaluation_selection": bad}, changed_model)
        model["export"]["campaign"]["slot"] = 8
        with self.assertRaises(ValueError):
            artifacts._partial_selection(artifact, model)

    def test_create_and_verify_bind_the_actual_live_producers_and_main_update(self):
        from fine_tuning.tests.test_error_note_v17_split_artifacts import SplitArtifacts
        from fine_tuning.error_note import campaign_v16b as campaign, prep_v16b as prep, v17_artifacts as artifacts

        helper = SplitArtifacts()
        helper.setUp()
        self.addCleanup(helper.doCleanups)

        def reseal(path, body):
            path.unlink(missing_ok=True)
            return prep.seal(path, {key: value for key, value in body.items() if key != "sha256"})

        plan = copy.deepcopy(helper.plan)
        plan["protocol"]["version"] = "error-note-v18b"
        plan["protocol"]["training_contract"] = {"prep_updates": 1, "rl_blocks": 1, "total_slots": 3}
        plan["prep_batches"] = [["prep"]]
        plan["execution_settings"] = settings()
        plan["inputs"] = {"payload": {"file_sha256": "e" * 64}}
        helper.plan = plan = reseal(helper.output / "campaign-plan.json", plan)
        helper.metadata.update(protocol=plan["protocol"], binding=campaign._binding(plan))
        metadata_path = helper.output / "checkpoint-3/metadata.json"
        write_json(metadata_path, helper.metadata)
        descriptor = artifacts._ref(metadata_path)
        helper.ready.update(plan_sha256=plan["sha256"], metadata_descriptor=descriptor)
        reseal(helper.output / "export-ready.json", helper.ready)
        reseal(helper.output / "reload-3.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
            "runtime_equal": True, "greedy_equal": True, "probe_max_abs_difference": 0, "producer_pid": 11, "reload_pid": 12})
        comparison = {"complete": True, "overall": {"net_change": 0, "p_value": 1}}
        monitor = reseal(helper.output / "monitor-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": comparison})
        effect = reseal(helper.output / "effect-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": {"complete": False}})
        reseal(helper.output / "evaluation-admission.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
            "monitor_sha256": monitor["sha256"], "effect_sha256": effect["sha256"],
            "decision": ev.evaluation_gate(comparison, effect["comparison"])})
        ref_path = helper.export / "model_ref.json"
        model = json.loads(ref_path.read_text())
        model["export"].update(binding=campaign._binding(plan), training_protocol=plan["protocol"],
                               campaign={"stage": "final", "plan_sha256": plan["sha256"]})
        write_json(ref_path, model)
        bounds = {"dataset_sha256": "e" * 64}
        capacity = {"version": "v18b-capacity-v1", "model_parameter_update": False, "shape_bounds": bounds,
            "report": {"version": "v17-update-memory-v1", "optimizer_steps": 0,
                "measurement": "actual_cuda_bf16_shared_backward_only", "live_engine": True, "fixture_only": True,
                "action_tokens": 8192, "prompt_tokens": 16384, "shape_bounds": bounds,
                "shape_bounds_sha256": digest(bounds), "post_backward_memory": {"free": 3 * 2**30}}}
        live_path = helper.output / "training-bindings/live-11.json"
        reseal(live_path, {"version": "v18b-live-training-preflight-v1", "status": "BOUND",
            "code_commit": "a" * 40, "source_sha256": digest(plan["source"]["actual_files"]),
            "campaign_plan_sha256": plan["sha256"], "protocol_sha256": plan["protocol"]["sha256"],
            "training_protocol_version": "error-note-v18b", "production_admitted": True, "producer_pid": 11,
            "capacity": capacity, "f0_transport": {"pass": True}, "coexistence": [{"pass": True}],
            "optimizer_steps": 0, "production_parents_consumed": 0, "maximum_shape_measurements": 1,
            "actual_abc_status": "NOT_RUN_USE_PRODUCTION_SLOT_TIMINGS"})
        for slot in range(3):
            transaction = reseal(helper.output / "transactions" / f"slot-{slot}.json", {"plan_sha256": plan["sha256"], "slot": slot})
            reseal(helper.output / "update-results" / f"slot-{slot}.json", {"version": "v18b-update-result-v1",
                "plan_sha256": plan["sha256"], "transaction_sha256": transaction["sha256"], "global_slot": slot,
                "phase": "prep" if slot == 0 else "rl", "producer_pid": 11, "timing_scope": "encode_backward_optimizer",
                "wall_seconds": 2., "result": {"optimizer_steps": 1}})
        with mock.patch("fine_tuning.error_note.v17_preflight.bind_training", side_effect=AssertionError("legacy receipt")):
            value = artifacts.create_training_artifacts(helper.args, plan)
        self.assertEqual(artifacts.verify_training_artifacts(helper.args.artifact_output), value)
        self.assertNotIn("functional_receipt", value)
        self.assertNotIn("update_cost_receipt", value)
        self.assertEqual(value["first_main_update"]["path"], str(helper.output / "update-results/slot-1.json"))
        # Simulate a validated restore mapping while preserving all historical descriptor bytes.
        import shutil
        from fine_tuning.error_note.v18b_recovery import recovery_paths
        original = helper.output
        restored = helper.root / "restored"
        shutil.copytree(original, restored)
        manifest = reseal(restored / "recovery-manifest.json", {"version": "v18b-recovery-closure-v1"})
        reseal(restored / "recovery-locations.json", {"manifest_sha256": manifest["sha256"],
                                                   "mappings": {str(original): str(restored)}})
        shutil.rmtree(original)
        helper.args.output = restored
        helper.args.artifact_output = restored / "training-artifacts.json"
        helper.args.artifact_output.unlink()
        with recovery_paths(restored):
            relocated = artifacts.create_training_artifacts(helper.args, plan)
        self.assertEqual(artifacts.verify_training_artifacts(helper.args.artifact_output), relocated)
        self.assertEqual(relocated["checkpoint"]["metadata"]["path"], str(restored / "checkpoint-3/metadata.json"))
        # A selected earlier boundary remains partial after restore and requires its own evaluation decision.
        for selected_step, consumed in ((2, 16), (3, 3000)):
            helper.metadata["step"] = selected_step
            metadata_path = restored / "checkpoint-3/metadata.json"
            write_json(metadata_path, helper.metadata)
            descriptor = {"path": str(original / "checkpoint-3/metadata.json"), "file_sha256": sha256_file(metadata_path)}
            helper.ready.update(status="PARTIAL_EVALUATION_CANDIDATE", metadata_descriptor=descriptor)
            reseal(restored / "export-ready.json", helper.ready)
            reseal(restored / f"reload-{selected_step}.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
                "runtime_equal": True, "greedy_equal": True, "probe_max_abs_difference": 0, "producer_pid": 11, "reload_pid": 12})
            monitor = reseal(restored / "monitor-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": comparison})
            effect = reseal(restored / "effect-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": {"complete": False}})
            reseal(restored / "evaluation-admission.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
                "monitor_sha256": monitor["sha256"], "effect_sha256": effect["sha256"], "decision": ev.evaluation_gate(comparison, effect["comparison"])})
            selection = {"reason": "TRAINING_CUTOFF", "authorized_action": "evaluate_checkpoint", "consumed_items": consumed,
                         "total_items": 3000, "source_checkpoint_sha256": descriptor["file_sha256"]}
            model["export"]["campaign"].update(stage="partial", slot=selected_step, updates=selected_step, evaluation_selection=selection)
            write_json(ref_path, model)
            helper.args.artifact_output.unlink()
            with recovery_paths(restored):
                partial = artifacts.create_training_artifacts(helper.args, plan)
            self.assertEqual(artifacts.verify_training_artifacts(helper.args.artifact_output), partial)
            self.assertEqual(partial["status"], "PARTIAL_EVALUATION_CANDIDATE")
            approved = restored / f"approved-evaluation-{selected_step}.json"
            artifacts.main(["verify", "--receipt", str(helper.args.artifact_output), "--repo", str(helper.repo),
                "--commit", "a" * 40, "--output", str(approved), "--expected-protocol", "error-note-v18b",
                "--authorization", "CPU fixture partial evaluation approval"])
            authorized = artifacts.verify_evaluation_authorization(approved, helper.args.artifact_output,
                {"code_commit": "a" * 40, "source_sha256": digest(helper.source["actual_files"])})
            self.assertEqual(authorized["evaluation_decision"]["evaluation_selection"], selection)
        # E8 may choose postprep even though the stopped run measured main updates afterwards.
        helper.metadata["step"] = 1
        write_json(metadata_path, helper.metadata)
        descriptor = {"path": str(original / "checkpoint-3/metadata.json"), "file_sha256": sha256_file(metadata_path)}
        selection = {"reason": "MONITOR_STOP", "authorized_action": "evaluate_checkpoint", "consumed_items": 0,
                     "total_items": 3000, "source_checkpoint_sha256": descriptor["file_sha256"], "preparation_only": True}
        stopped_metadata = restored / "checkpoint-stopped/metadata.json"
        write_json(stopped_metadata, {"step": 3, "protocol": plan["protocol"]})
        stopped_ref = {"path": str(original / "checkpoint-stopped/metadata.json"), "file_sha256": sha256_file(stopped_metadata)}
        stopped = reseal(restored / "campaign-state.json", {"plan_sha256": plan["sha256"], "status": "STOPPED",
            "stop_reasons": ["MONITOR_STOP"], "cursor": {"slot": 3}, "checkpoint": stopped_ref})
        write_json(restored / "prep-checkpoint.json", descriptor)
        reseal(restored / "evaluation-checkpoint-state.json", {"status": "PARTIAL_EVALUATION_CANDIDATE", "plan_sha256": plan["sha256"],
            "evaluation_selection": selection, "stopped_training_state_sha256": stopped["sha256"],
            "stopped_training_checkpoint": stopped_ref, "cursor": {"slot": 1, "consumed_problem_ids": []}, "checkpoint": descriptor})
        helper.ready.update(metadata_descriptor=descriptor, evaluation_selection=selection)
        reseal(restored / "export-ready.json", helper.ready)
        reseal(restored / "reload-1.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
            "runtime_equal": True, "greedy_equal": True, "probe_max_abs_difference": 0, "producer_pid": 11, "reload_pid": 12})
        monitor = reseal(restored / "monitor-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": comparison})
        effect = reseal(restored / "effect-final.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor, "comparison": {"complete": False}})
        reseal(restored / "evaluation-admission.json", {"plan_sha256": plan["sha256"], "checkpoint": descriptor,
            "monitor_sha256": monitor["sha256"], "effect_sha256": effect["sha256"], "decision": ev.evaluation_gate(comparison, effect["comparison"])})
        model["export"]["campaign"].update(slot=1, updates=1, evaluation_selection=selection)
        write_json(ref_path, model)
        helper.args.artifact_output.unlink()
        with recovery_paths(restored):
            preparation_only = artifacts.create_training_artifacts(helper.args, plan)
        self.assertEqual(artifacts.verify_training_artifacts(helper.args.artifact_output), preparation_only)
        self.assertNotIn("first_main_update", preparation_only)
        self.assertEqual(len(preparation_only["actual_update_results"]), 1)
        timing = preparation_only["stopped_run_timing"]
        self.assertFalse(timing["included_in_selected_checkpoint"])
        self.assertEqual(timing["first_main_update"]["path"], str(restored / "update-results/slot-1.json"))
        self.assertIn("0/3000", pe._single_definition(model))
        approved = restored / "approved-preparation-only.json"
        artifacts.main(["verify", "--receipt", str(helper.args.artifact_output), "--repo", str(helper.repo),
            "--commit", "a" * 40, "--output", str(approved), "--expected-protocol", "error-note-v18b",
            "--authorization", "CPU fixture preparation-only evaluation approval"])
        authorized = artifacts.verify_evaluation_authorization(approved, helper.args.artifact_output,
            {"code_commit": "a" * 40, "source_sha256": digest(helper.source["actual_files"])})
        self.assertTrue(authorized["evaluation_decision"]["evaluation_selection"]["preparation_only"])
        live_path = restored / "training-bindings/live-11.json"
        live_path.write_text(live_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            artifacts.verify_training_artifacts(helper.args.artifact_output)


class CanonicalPlanChecks(unittest.TestCase):
    def test_plans_parse_official_commands_with_single_or_disjoint_shards_and_resume(self):
        from fine_tuning.error_note import plan as planner
        from fine_tuning.tests.test_error_note_v17_split_plan import parse_only, split_params
        from fine_tuning.notebook_rl.pod.chain import check_spec
        import importlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for strategy in ("one_pod", "two_pod", "training_cutoff"):
                p = split_params(root / strategy, "v17-eval")
                p.update(version=planner.PARAMS_VERSION_V18B_SPLIT, execution_settings=settings(strategy),
                         evaluation_authorization="CPU fixture evaluation decision")
                p["funding"]["first_bill_at"] = settings(strategy)["first_billed_at"]
                p["timeouts"].update(calibrate=300, validate=1260)
                validation_path = Path(p["out_root"]) / "engine-validation.json"
                validation_path.parent.mkdir(parents=True)
                write_json(validation_path, {})
                validation = {"evaluation_watchdog_seconds": 50400, "slots": 9, "estimated_evaluation_seconds": None,
                    "selected_mode": "eager", "training_protocol_version": "error-note-v18b", "execution_settings": settings(strategy),
                    "evaluation_authorization": {"evaluation_decision": {"authorization": p["evaluation_authorization"]}},
                    "artifact_verification": {"path": str(validation_path.parent / "training-artifacts-verified.json")},
                    "training_artifact_path": p["training_artifacts_path"]}
                with mock.patch.object(engine, "load_engine_validation", return_value=validation):
                    for index in range(2 if strategy == "two_pod" else 1):
                        p["evaluation_partition_index"] = index
                        prefix = check_spec(planner.plan("v18b-eval", p))
                        continuation = check_spec(planner._evaluation_continuation(p, validation_path))
                        for step in prefix["steps"] + continuation["steps"]:
                            argv = step["argv"]
                            if "-m" in argv:
                                at = argv.index("-m")
                                parse_only(importlib.import_module(argv[at + 1]), argv[at + 2:])
                        evaluate = next(row for row in continuation["steps"] if row["name"] == "evaluate-900")
                        argv = evaluate["argv"]
                        self.assertIn("--execution-settings", argv)
                        self.assertEqual(argv.count("--sample-id"), 450 if strategy == "two_pod" else 900)
                        self.assertEqual(any(row["name"] == "compass-separate" for row in continuation["steps"]), strategy != "two_pod")
                        resumed = planner._resume_steps(continuation)
                        self.assertEqual(resumed["steps"][0]["argv"][argv.index("fine_tuning.error_note.v17_eval") + 1], "resume")
                if strategy == "two_pod":
                    shards = [{"prereg": str(root / f"p{i}.json"), "ledger": str(root / f"l{i}.jsonl"),
                               "run_dir": str(root / (settings(strategy)["run_id"] + f"-part-{i}"))} for i in range(2)]
                    result = planner.finalize_partition_evaluation(p, shards, Path(p["out_root"]) / "partition-finalization")
                    spec = check_spec(json.loads(Path(result["spec"]).read_text()))
                    self.assertEqual([row["name"] for row in spec["steps"]][-2:], ["merge-full-900-v8", "compass-separate"])
                    self.assertEqual(spec["meta"]["execution_settings"], settings(strategy))


class MergeChecks(unittest.TestCase):
    def test_real_v8_rescoring_requires_both_original_disjoint_partitions(self):
        from fine_tuning.tests.test_notebook_rl_proof_eval import IDS, write_run, variant_cfg

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(pe, "gpu_info", return_value=("CPU fixture", 0)):
            root = Path(directory)
            cfg = variant_cfg(pe.FINAL_VARIANT_V8)
            pieces = policy.partitions(IDS, settings("two_pod"))
            reference = {sid: {"correct": True, "finish_reason": "stop"} for sid in IDS}
            preregs, ledgers, runs = [], [], []
            for index, (tag, ids) in enumerate(pieces.items()):
                prereg = {"training_protocol_version": "error-note-v18b", "execution_settings": settings("two_pod"),
                    "b2_reference": {"wrong_ids": IDS[:342], "right_ids": IDS[342:]},
                    "evaluation_subsets": {key: {"ids": value} for key, value in pieces.items()},
                    "arms": {pe.ARM_SINGLE: {key: "a" * 64 for key in
                        ("model_sha256", "model_ref_sha256", "checkpoint_metadata_sha256", "train_config_sha256")}},
                    "reference_finish_strata": reference, "engine_validation": {"selected_mode": "eager"},
                    "engine_validation_sha256": str(index) * 64,
                    "protocol_variants": {pe.FINAL_VARIANT_V8: pe._variant_entry(pe.FINAL_VARIANT_V8)}}
                path = root / f"prereg-{index}.json"
                write_json(path, prereg)
                preregs.append(path)
                run_id = settings()["run_id"] + f"-part-{index}"
                run = write_run(root, run_id, cfg, "a" * 64, lambda i: i % 2 == 0, ids=ids, registration=False)
                registration = {"execution_settings": settings("two_pod"), "partition_index": index,
                    "preregistration_sha256": sha256_file(path), "engine_validation_sha256": str(index) * 64,
                    "expected_ids": ids}
                write_json(run / pe.REGISTRATION_FILE, registration)
                audit = {"last_event_sha256": None}
                ev._submission_event(run, registration, audit_state=audit, event="BEGIN", invocation_id="original",
                                     completed_ids=[], pending_ids=ev.submission_order(ids))
                for sid in ev.submission_order(ids):
                    for event in ("INTENT", "ACCEPTED"):
                        ev._submission_event(run, registration, audit_state=audit, event=event, invocation_id="original", id=sid)
                runs.append(run)
                ledger = root / f"ledger-{index}.jsonl"
                pe._append(ledger, "START", arm=pe.ARM_SINGLE, variant=pe.FINAL_VARIANT_V8, subset=tag,
                           run_id=run_id, preregistration_sha256=sha256_file(path))
                ledgers.append(ledger)

            def prereg(path):
                return json.loads(path.read_text()), sha256_file(path)

            with mock.patch.object(pe, "load_preregistration", side_effect=prereg):
                report = policy.merge(preregs, ledgers, runs, root / "merged")
                self.assertEqual((report["status"], report["correct"]), ("FULL900_VERIFIED", 450))
                self.assertEqual(len((root / "merged/predictions.jsonl").read_text().splitlines()), 900)
                self.assertEqual(policy.merge(preregs, ledgers, runs, root / "merged"), report)
                with self.assertRaisesRegex(ValueError, "repeated|replaced"):
                    policy.merge([preregs[0]] * 2, [ledgers[0]] * 2, [runs[0]] * 2, root / "duplicate")
                predictions = runs[1] / "predictions.jsonl"
                old = predictions.read_text()
                predictions.write_text("\n".join(old.splitlines()[:-1]) + "\n")
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    policy.merge(preregs, ledgers, runs, root / "missing")
                predictions.write_text(old)
                changed = json.loads(preregs[1].read_text())
                changed["arms"][pe.ARM_SINGLE]["model_sha256"] = "f" * 64
                write_json(preregs[1], changed)
                with self.assertRaisesRegex(ValueError, "model, partition"):
                    policy.merge(preregs, ledgers, runs, root / "model-changed")


if __name__ == "__main__":
    unittest.main()
