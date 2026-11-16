"""CPU crash-boundary and operating-contract checks; no model, GPU, downloads or benchmark inputs."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import time
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from types import MethodType
from unittest import mock

from fine_tuning.error_note import v17_eval as ev
from fine_tuning.notebook_rl import proof_eval as pe
from fine_tuning.notebook_rl.pod import collect
from mmdl.evaluation import engine as evaluation
from mmdl.evaluation.writer import RunWriter
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import sample_seed


def probe_value(**changes):
    value = {"version": "v17-eval-probe-v1", "slots": 8, "items": 16, "wall_seconds": 120,
             "generated_tokens": 48000, "sixty_second_rates": [350., 450.], "dtype": "bfloat16",
             "backend": "vllm-0.11.0", "config_sha256": "a" * 64,
             "max_observed_context_tokens": 35000, "cumulative_preemptions": 0,
             "disjoint_receipt_sha256": "b" * 64, "load_finalize_allowance_seconds": 60.,
             "long_tail_allowance_seconds": 1000., "long_tail_basis": "declared conservative tail scenario",
             "source_sha256": "c" * 64, "code_commit": "d" * 40, "environment_sha256": "e" * 64}
    return ev.seal(value | changes)


def engine_value(blocks=27776, size=16):
    config = NS(num_blocks=blocks, kv_cache_groups=[NS(kv_cache_spec=NS(block_size=size))])
    scheduler = NS(kv_cache_config=config, waiting=[], requests={},
                   kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda: blocks)))
    return NS(engine_core=NS(engine_core=NS(scheduler=scheduler)))


def record(sid):
    return {"id": sid, "subject": "fixture", "question_type": "multiple-choice", "question": "Pick one.",
            "options": ["first", "second"], "answer": "A", "seed": sample_seed(3407, sid),
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Pick one."}]}],
            "images": [], "prompt": "Pick one.", "inference_id": "infer-" + sid,
            "inference_started_at": "2026-11-17T00:00:00+00:00"}


def output(sid, raw="Final answer: A", finish="stop"):
    return {"sample_id": sid, "raw_response": raw, "generated_token_ids": [1, 2, 3],
            "generated_tokens": 3, "generation_seconds": .5, "finish_reason": finish,
            "input_sha256": "d" * 64, "input_tokens": 10}


def final_value(sid, registration, identity, run_id):
    return ev.seal({"version": "v17-final-raw-v1", "registration_sha256": digest(registration),
                    "identity": identity, "run_id": run_id, "invocation_id": "original-invocation",
                    "record": record(sid), "output": output(sid), "sealed_at": "fixed"})


class V17EvalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_actual_kv_boundary_and_unknown_structure_fail_closed(self):
        # 8 * 40960 <= floor(.9 * blocks * 16), exactly at the integer block boundary.
        blocks = math.ceil((8 * 40960) / (.9 * 16))
        value = ev.capacity_receipt(engine_value(blocks), "0.11.0")
        self.assertTrue(value["pass"])
        self.assertEqual(value["slots"], 8)
        for engine, version in ((engine_value(blocks - 1), "0.11.0"), (NS(), "0.11.0"),
                                (engine_value(), "0.12.0")):
            with self.subTest(version=version), self.assertRaises(ValueError):
                ev.capacity_receipt(engine, version)

    def test_calibration_uses_run_wall_and_window_median(self):
        value = ev.calibration_receipt(probe_value(sixty_second_rates=[300., 350.]))
        self.assertEqual(value["q_safe"], 325.)
        self.assertEqual(ev.calibration_timeout(value), math.ceil(1.25 * (1.5 * 4116922 / 325 + 1060)))
        path = self.root / "calibration.json"
        ev.exclusive(path, value)
        self.assertEqual(ev.load_calibration(path), value)
        altered = dict(value, evaluation_watchdog_seconds=23400)
        write_json(path, ev.seal(altered))
        with self.assertRaises(ValueError):
            ev.load_calibration(path)

    def test_generated_calibration_binds_to_real_preflight_full_source_identity(self):
        from fine_tuning.error_note import campaign_v16b as campaign, protocol, single, v17_preflight as pf
        from fine_tuning.tests.test_error_note_v17_preflight import capacity

        source = single.source_manifest()
        contracts = {key: "a" * 64 for key in protocol.V17_CONTRACT_HASHES}
        contracts["source"] = digest(source)
        training = protocol.plain_protocol_v17("bf16", rl_blocks=188, prep_updates=1, contracts=contracts)
        ids = [f"synthetic-parent-{i}" for i in range(3000)]
        plan = ev.seal({"version": campaign.CAMPAIGN_V17, "protocol": training, "source": source,
                        "blocks": [ids[i:i + 16] for i in range(0, len(ids), 16)],
                        "items": [{"problem_id": sid} for sid in ids], "prep_batches": [[]],
                        "total_slots": training["training_contract"]["total_slots"],
                        "time_cut": None, "local_sequential": False,
                        "capacity": {"engine": "vllm", "max_num_seqs": 16}})
        args = NS(output=self.root, code_commit=source["base_commit"])
        expected = pf._identity(args, plan)
        calibrated = ev.calibration_receipt(probe_value(
            source_sha256=ev.source_sha256(), code_commit=source["base_commit"]))
        args.calibration = self.root / "calibration.json"
        ev.exclusive(args.calibration, calibrated)
        args.calibration_sha256 = sha256_file(args.calibration)
        cost_path = self.root / "cost.json"
        cost = pf._write(cost_path, {
            "version": "v17-update-cost-receipt-v1", "status": "CALIBRATION_LOW_COVERAGE", **expected,
            "optimizer_steps": 0, "coverage": {"actual_a1_count": 0, "missing_actual_a1_lengths": ["long"]},
            "measurement": None, "actual_abc_status": "UNMEASURED_NO_ADMITTED_A1"})
        args.functional_receipt = self.root / "functional.json"
        pf._write(args.functional_receipt, {
            "version": pf.VERSION, "status": "PASS", **expected, "optimizer_steps": 0,
            "production_parents_consumed": 0, "local_smoke": False, "capacity": capacity(),
            "f0_transport": {"pass": True}, "coexistence": [{"pass": True}],
            "update_cost_sha256": cost["sha256"],
            "update_cost_receipt": {"path": str(cost_path), "file_sha256": sha256_file(cost_path)}})
        args.functional_receipt_sha256 = sha256_file(args.functional_receipt)
        bound = pf.bind_execution(args, plan)
        self.assertEqual(bound["status"], "BOUND")
        self.assertEqual(bound["source_sha256"], calibrated["source_sha256"])
        self.assertEqual(bound["source_sha256"], digest(source["actual_files"]))

        evaluator_only = digest({str(path.relative_to(pe.ROOT)): sha256_file(path)
                                 for path in (Path(ev.__file__), Path(pe.__file__))})
        self.assertNotEqual(evaluator_only, bound["source_sha256"])
        write_json(args.calibration, ev.calibration_receipt(probe_value(
            source_sha256=evaluator_only, code_commit=source["base_commit"])))
        args.calibration_sha256 = sha256_file(args.calibration)
        with self.assertRaisesRegex(ValueError, "stale"):
            pf.bind_execution(args, plan)

    def test_probe_low_coverage_never_seals_runnable_timeout(self):
        for changes in ({"generated_tokens": 0}, {"sixty_second_rates": []},
                        {"max_observed_context_tokens": 1024}, {"cumulative_preemptions": 1}):
            with self.subTest(changes=changes):
                value = ev.calibration_receipt(probe_value(**changes))
                self.assertEqual(value["status"], "CALIBRATION_LOW_COVERAGE")
                with self.assertRaises(ValueError):
                    ev.calibration_timeout(value)
        for changes in ({"wall_seconds": ev.PROBE_CAP_SECONDS + 1}, {"items": 17}, {"slots": 32}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ev.calibration_receipt(probe_value(**changes))

    def test_heartbeat_progress_and_stall_are_warnings_only(self):
        clock = [0.]
        heartbeat = ev.Heartbeat(self.root, clock=lambda: clock[0])
        heartbeat.active["long"] = 0
        clock[0] = 170
        heartbeat.observe(tokens=20, prefill=100, preemptions=0)
        clock[0] = 200
        self.assertEqual(heartbeat.emit()["warnings"], [])
        self.assertEqual(heartbeat.completed, 0)  # long output remains valid progress
        clock[0] = 400
        heartbeat.observe(preemptions=2)
        row = heartbeat.emit()
        self.assertEqual(row["warnings"], ["TOKEN_STALL_WARNING", "KV_RECOMPUTE_WARNING", "ENGINE_NO_PROGRESS_WARNING"])
        self.assertIn("long", heartbeat.active)
        self.assertEqual(read_json(self.root / "heartbeat.json"), row)

    def test_cpu_recovery_raw_and_sample_crash_boundaries(self):
        run_dir, ids = self.root / "runs/run", ["item-" + str(i) for i in range(10)]
        identity = {"model_sha256": "a" * 64, "mode": "partial"}
        registration = {"expected_ids": ids, "model_sha256": identity["model_sha256"]}
        with RunWriter(run_dir, identity, ids) as writer:
            for sid in ids[:4]:
                final = final_value(sid, registration, identity, run_dir.name)
                ev.exclusive(run_dir / "final_raw" / (sid + ".json"), final)
                writer.save(ev.record_from_final(final))
        old = {p.name: sha256_file(p) for p in (run_dir / "samples").glob("*.json")}
        # Crash after FINAL_RAW but before sample: this fifth ID also requires zero generation.
        ev.exclusive(run_dir / "final_raw" / (ids[4] + ".json"), final_value(ids[4], registration, identity, run_dir.name))
        with mock.patch("mmdl.evaluation.writer.render"):
            with ev.evaluation_hooks(run_dir, registration, time.time() + 60):
                with evaluation.RunWriter(run_dir, identity, ids, resume=True) as writer:
                    self.assertEqual(list(writer.completed), ids[:5])
                    self.assertEqual([i for i in ids if i not in writer.completed], ids[5:])
                    self.assertIn("recovery_provenance", writer.completed[ids[4]])
        self.assertEqual(old, {name: sha256_file(run_dir / "samples" / name) for name in old})
        self.assertIs(evaluation.RunWriter, RunWriter)

    def test_actual_stream_refills_eight_and_resume_generates_six_remaining_ids(self):
        from mmdl.evaluation.backends.vllm_backend import VLLMBackend

        ids = [f"item{i:02}" for i in range(10)]
        run_dir = self.root / "runs/run"
        identity = {"model_sha256": "a" * 64, "mode": "partial"}
        registration = {"expected_ids": ids, "model_sha256": identity["model_sha256"]}
        submitted, ordering, max_active = [], [], [0]

        def backend():
            engine = engine_value()
            active = engine.engine_core.engine_core.scheduler.requests
            def add(sid, prompt, params, **kwargs):
                active[sid] = NS(num_tokens=11)
                submitted.append(sid)
                ordering.append(("submit", sid))
                max_active[0] = max(max_active[0], len(active))
            def step():
                # item00 is deliberately long: item08 must start before item00 finishes.
                sid = next((sid for sid in active if sid != ids[0]), ids[0])
                active.pop(sid)
                ordering.append(("finish", sid))
                return [NS(request_id=sid, finished=True, outputs=[NS(token_ids=[1, 2, 3])])]
            engine.add_request, engine.step = add, step
            engine.abort_request = lambda pending: [active.pop(sid, None) for sid in pending]
            instance = NS(llm=NS(llm_engine=engine, get_metrics=lambda: []), vllm_version="0.11.0",
                          batch_size=8, max_model_len=40960, cfg={"execution": {"scheduling": "continuous"}},
                          _prepare_request=lambda request, **kwargs: ({}, {}, None),
                          _row=lambda request, item, params, result, **kwargs: output(request["sample_id"]))
            instance.generate_stream = MethodType(VLLMBackend.generate_stream, instance)
            return instance

        def prepare(sid):
            return {"sample_id": sid, "seed": sample_seed(3407, sid)}, record(sid)

        invocation_count = [0]
        def invocation():
            invocation_count[0] += 1
            current = "current-" + str(invocation_count[0])
            write_json(run_dir / "inference_invocations" / (current + ".json"), {"id": current, "status": "RUNNING",
                       "started_at": ev.utc(), "attempted_ids": ids, "completed_ids": []})

        with mock.patch.object(evaluation, "prepare_request", side_effect=prepare):
            with ev.evaluation_hooks(run_dir, registration, time.time() + 60) as ready:
                with evaluation.RunWriter(run_dir, identity, ids) as writer:
                    first = backend()
                    ready(first)
                    invocation()
                    requests = (evaluation.prepare_request(sid)[0] for sid in ids)
                    stream = first.generate_stream(requests)
                    try:
                        for _ in range(4):
                            result = next(stream)
                            path = run_dir / "final_raw" / (result["sample_id"] + ".json")
                            self.assertTrue(path.exists())  # durable before caller receives the result
                            writer.save(ev.record_from_final(read_json(path)))
                    finally:
                        stream.close()
        old = {p.name: sha256_file(p) for p in (run_dir / "samples").glob("*.json")}
        self.assertLess(ordering.index(("submit", ids[8])), ordering.index(("finish", ids[4])))
        self.assertNotIn(("finish", ids[0]), ordering)
        self.assertEqual(max_active[0], 8)
        submitted.clear()
        with mock.patch.object(evaluation, "prepare_request", side_effect=prepare):
            with ev.evaluation_hooks(run_dir, registration, time.time() + 60) as ready:
                with evaluation.RunWriter(run_dir, identity, ids, resume=True) as writer:
                    second = backend()
                    ready(second)
                    invocation()
                    requests = (evaluation.prepare_request(sid)[0] for sid in ids if sid not in writer.completed)
                    for result in second.generate_stream(requests):
                        writer.save(ev.record_from_final(read_json(run_dir / "final_raw" / (result["sample_id"] + ".json"))))
                    self.assertEqual(set(writer.completed), set(ids))
        self.assertEqual(len(submitted), 6)
        self.assertEqual(old, {name: sha256_file(run_dir / "samples" / name) for name in old})

    def test_conflicting_raw_and_completed_sample_is_integrity_error(self):
        run_dir, identity = self.root / "runs/run", {"model_sha256": "a" * 64, "mode": "partial"}
        registration = {"expected_ids": ["one"], "model_sha256": identity["model_sha256"]}
        final = final_value("one", registration, identity, "run")
        with RunWriter(run_dir, identity, ["one"]) as writer:
            writer.save(ev.record_from_final(final))
        altered = copy.deepcopy(final)
        altered["output"]["raw_response"] = "Final answer: B"
        ev.exclusive(run_dir / "final_raw/one.json", ev.seal(altered))
        with RunWriter(run_dir, identity, ["one"], resume=True) as writer:
            with self.assertRaisesRegex(ValueError, "conflicting"):
                ev.recover_finals(writer, digest(registration))

    def test_all_samples_without_summary_finalize_cpu_only(self):
        run_dir, identity = self.root / "runs/run", {"model_sha256": "a" * 64, "mode": "partial"}
        registration = {"expected_ids": ["one"], "model_sha256": identity["model_sha256"]}
        final = final_value("one", registration, identity, "run")
        with RunWriter(run_dir, identity, ["one"]) as writer:
            writer.save(ev.record_from_final(final))
        with mock.patch("mmdl.evaluation.writer.render"):
            with ev.evaluation_hooks(run_dir, registration, time.time() - 100) as backend_ready:
                with mock.patch.object(ev, "capacity_receipt", side_effect=AssertionError("model loaded")):
                    with evaluation.RunWriter(run_dir, identity, ["one"], resume=True):
                        self.assertEqual(read_json(run_dir / "summary.json")["status"], "PARTIAL")
        self.assertTrue(read_json(run_dir / "summary.json")["v17_cpu_finalize"])
        self.assertTrue(callable(backend_ready))

    def test_cpu_finalizer_recomputes_the_original_engine_identity_without_bf16_probe(self):
        import yaml

        cfg = yaml.safe_load(pe.VARIANTS[pe.FINAL_VARIANT_V8][0].read_text())
        hardware = pe.ROOT / "configs/hardware/rtx4090_24gb.yaml"
        hw = yaml.safe_load(hardware.read_text())
        data = self.root / "data"
        data.mkdir()
        write_json(data / "manifest.json", {"files": []})
        model_ref = self.root / "model_ref.json"
        write_json(model_ref, {"files": [], "kind": "base"})
        args = NS(artifact_root=self.root, data_root=data, model_ref=model_ref, model_path=self.root,
                  base_path=self.root, require_commit=None, hardware=hardware,
                  run_id="completed", job_id=None, run_role="standalone", mode="partial", subject=None,
                  sample_id=["one"], limit=None, resume=True)
        environment = {"python": "fixture", "platform": {}, "torch": {"cuda_runtime": "12.8",
                       "bf16_probe": {"passed": True}}, "nvidia_smi": {"gpus": []}}
        captured = {}
        def stop_at_writer(run_dir, identity, ids, **kwargs):
            captured.update(identity)
            raise RuntimeError("captured identity before model load")
        with mock.patch.object(evaluation, "collect_environment", return_value=environment), \
             mock.patch.object(evaluation, "check_storage", return_value={"sufficient": True}), \
             mock.patch.object(evaluation, "verify_model"), \
             mock.patch.object(evaluation, "git_commit", return_value="c" * 40), \
             mock.patch.object(evaluation, "load_validation", return_value=({"fixture": {"id": ["one"], 0: {"id": "one"}}}, {})), \
             mock.patch.object(evaluation, "RunWriter", side_effect=stop_at_writer):
            with self.assertRaisesRegex(RuntimeError, "captured identity"):
                evaluation.run(args, cfg, hw, pe.ROOT)
        registration = {"expected_ids": ["one"], "model_sha256": digest([])}
        run_dir = self.root / "runs/completed"
        with RunWriter(run_dir, captured, ["one"]) as writer:
            writer.save(ev.record_from_final(final_value("one", registration, captured, "completed")))
        with mock.patch("mmdl.runtime.environment.collect_environment", return_value=environment) as env, \
             mock.patch.object(evaluation, "verify_model"), \
             mock.patch.object(evaluation, "git_commit", return_value="c" * 40), \
             mock.patch.object(pe, "registered_variants", return_value=nullcontext()), \
             mock.patch("mmdl.evaluation.writer.render"):
            self.assertTrue(ev.cpu_resume_finalize(args, cfg, registration))
            env.assert_called_once_with(probe=False)
        self.assertTrue(read_json(run_dir / "summary.json")["v17_cpu_finalize"])

    def test_normal_engine_and_final_raw_recovery_have_equal_stable_record_and_v8_score(self):
        import yaml
        from mmdl.evaluation.backends import vllm_backend

        cfg = yaml.safe_load(pe.VARIANTS[pe.FINAL_VARIANT_V8][0].read_text())
        hardware = pe.ROOT / "configs/hardware/rtx4090_24gb.yaml"
        data = self.root / "data"
        data.mkdir()
        write_json(data / "manifest.json", {"files": []})
        model_ref = self.root / "model_ref.json"
        write_json(model_ref, {"files": [], "kind": "base"})
        args = NS(artifact_root=self.root, data_root=data, model_ref=model_ref, model_path=self.root,
                  base_path=self.root, require_commit=None, hardware=hardware, public_root=self.root / "public",
                  protocol=pe.VARIANTS[pe.FINAL_VARIANT_V8][0], run_id="normal", job_id=None,
                  run_role="standalone", mode="partial", subject=None, sample_id=["one"], limit=None, resume=False)
        environment = {"python": "fixture", "platform": {}, "torch": {"cuda_runtime": "12.8",
                       "bf16_probe": {"passed": True}}, "nvidia_smi": {"gpus": []}}
        registration = {"expected_ids": ["one"], "model_sha256": digest([])}
        run_dir = self.root / "runs/normal"
        def prepare(*args, **kwargs):
            return {"sample_id": "one", "seed": sample_seed(3407, "one")}, record("one")
        with mock.patch.object(evaluation, "prepare_request", side_effect=prepare), \
             mock.patch.object(evaluation, "collect_environment", return_value=environment), \
             mock.patch.object(evaluation, "check_storage", return_value={"sufficient": True}), \
             mock.patch.object(evaluation, "verify_model"), mock.patch.object(evaluation, "publish"), \
             mock.patch.object(evaluation, "git_commit", return_value="c" * 40), \
             mock.patch.object(evaluation, "load_validation", return_value=({"fixture": {"id": ["one"], 0: {"id": "one"}}}, {})), \
             mock.patch.object(evaluation, "code_records", return_value=[]), \
             mock.patch("mmdl.evaluation.writer.render"), mock.patch.object(evaluation, "render"):
            with ev.evaluation_hooks(run_dir, registration, time.time() + 60) as ready:
                def factory(*args, **kwargs):
                    engine = engine_value()
                    engine.add_request, engine.step = lambda *a, **k: None, lambda: []
                    backend = NS(llm=NS(llm_engine=engine), vllm_version="0.11.0", batch_size=8,
                                 max_model_len=40960, cfg=cfg, load_seconds=0, describe=lambda: {},
                                 _row=lambda *a, **k: output("one"))
                    def stream(requests):
                        for request in requests:
                            yield backend._row(request, {}, None, None)
                    backend.generate_stream = stream
                    ready(backend)
                    return backend
                with mock.patch.object(vllm_backend, "VLLMBackend", side_effect=factory):
                    summary = evaluation.run(args, cfg, yaml.safe_load(hardware.read_text()), pe.ROOT)
        self.assertEqual(summary["status"], "PARTIAL")
        normal = read_json(run_dir / "samples/one.json")
        reconstructed = ev.record_from_final(read_json(run_dir / "final_raw/one.json"))
        self.assertEqual(ev.stable_record(normal), ev.stable_record(reconstructed))
        self.assertEqual(normal["correct"], reconstructed["correct"])

    def test_v12_preregistration_freezes_measured_watchdog_and_b2_strata(self):
        from fine_tuning.tests.test_notebook_rl_proof_eval import IDS, REST, DECLARED

        baseline = self.root / "b2.jsonl"
        baseline.write_text("".join(json.dumps({"id": sid, "correct": sid in REST,
                             "finish_reason": "length" if i < 90 else "stop"}) + "\n"
                                     for i, sid in enumerate(IDS)))
        paths = {}
        for kind in ("wrong", "rest", "all"):
            paths[kind] = self.root / (kind + ".json")
            pe.make_subset(kind, paths[kind], baseline)
        calibration = self.root / "calibration.json"
        ev.exclusive(calibration, ev.calibration_receipt(probe_value()))
        ref = {"export": {"training_protocol": {"version": "error-note-v17", "training_contract": {
                     "prep_updates": 12, "rl_blocks": 188, "total_slots": 203}},
                     "campaign": {"stage": "final", "slot": 203, "updates": 201}}}
        output_path = self.root / "prereg.json"
        with mock.patch.object(pe, "_lora_model_ref", return_value=(ref, DECLARED)):
            pe.preregister(output_path, paths["wrong"], paths["rest"], baseline, "v12",
                           ft_single_model_ref=self.root / "model_ref.json", all_subset=paths["all"],
                           calibration=calibration)
        frozen, _ = pe.load_preregistration(output_path)
        self.assertEqual(frozen["execution_contract"]["slots"], 8)
        self.assertEqual(frozen["model_application"]["serving_kind"], "unmerged-lora")
        self.assertNotEqual(frozen["evaluation_watchdog_seconds"], 23400)
        self.assertEqual(sum(r["finish_reason"] == "length" for r in frozen["reference_finish_strata"].values()), 90)
        ledger = self.root / "ledger.jsonl"
        pe.ledger_start(output_path, pe.ARM_SINGLE, pe.FINAL_VARIANT_V8, "run", ledger, "all")
        with self.assertRaises(ValueError):
            pe.ledger_start(output_path, pe.ARM_SINGLE, pe.FINAL_VARIANT_V8, "replacement", ledger, "all")
        changed = copy.deepcopy(frozen)
        changed["execution_contract"]["slots"] = 32
        write_json(output_path, changed)
        output_path.with_name(output_path.name + ".sha256").write_text(sha256_file(output_path))
        with self.assertRaises(ValueError):
            pe.load_preregistration(output_path)

    def test_v17_cannot_bypass_capacity_and_resume_contract_through_old_wrapper(self):
        model_ref = self.root / "model_ref.json"
        write_json(model_ref, {"export": {"training_protocol": {"version": "error-note-v17"}}})
        with mock.patch.object(pe, "gpu_info", side_effect=AssertionError("backend preflight reached")):
            with self.assertRaisesRegex(ValueError, "v17_eval run/resume"):
                pe.run_eval(["--model-ref", str(model_ref), "--protocol", str(pe.VARIANTS[pe.FINAL_VARIANT_V8][0]),
                             "--run-id", "v17", "--artifact-root", str(self.root)])

    def test_completed_identity_and_registration_changes_refused_before_shortcut(self):
        run_dir, identity = self.root / "runs/run", {"model_sha256": "a" * 64, "mode": "partial"}
        registration = {"expected_ids": ["one"], "model_sha256": identity["model_sha256"]}
        with RunWriter(run_dir, identity, ["one"]) as writer:
            writer.save(ev.record_from_final(final_value("one", registration, identity, "run")))
        ev.exclusive(run_dir / pe.REGISTRATION_FILE, registration)
        write_json(run_dir / "summary.json", {"status": "PARTIAL", "completed_count": 1})
        for replacement in (identity | {"environment_sha256": "kernel-driver-package-change"},
                            identity | {"model_sha256": "different"}):
            with self.subTest(replacement=replacement):
                with ev.evaluation_hooks(run_dir, registration, time.time() + 60):
                    with self.assertRaises(ValueError):
                        evaluation.RunWriter(run_dir, replacement, ["one"], resume=True)
        with ev.evaluation_hooks(run_dir, registration | {"slots": 32}, time.time() + 60):
            with self.assertRaises(ValueError):
                evaluation.RunWriter(run_dir, identity, ["one"], resume=True)

    def test_cumulative_deadline_counts_downtime_and_receipt_extensions(self):
        start = {"time": datetime.fromtimestamp(1000, timezone.utc).isoformat(), "run_id": "one"}
        self.assertEqual(ev.deadline_seconds(start, 500, [], now=1400), 100)
        self.assertEqual(ev.deadline_seconds(start, 500, [], now=1600), -100)
        extension = ev.seal({"version": "v17-eval-extension-v1", "previous": "0" * 64,
                             "run_id": "one", "old_deadline": 1500., "new_deadline": 1800.,
                             "confirmed_by": "user", "authorization": "explicit user extension"})
        self.assertEqual(ev.deadline_seconds(start, 500, [extension], now=1600), 200)
        self.assertEqual(ev.deadline_seconds(start, 500, [extension], now=1600,
                                            funding_deadline=1750, recovery_reserve_seconds=60), 90)
        with self.assertRaises(ValueError):
            ev.deadline_seconds(start, 500, [ev.seal(extension | {"confirmed_by": "automatic"})], now=1600)

    def test_paired_strata_use_reference_and_gate_does_not_invent_effect_zero(self):
        ref = {"a": {"correct": False, "finish_reason": "length"},
               "b": {"correct": True, "finish_reason": "stop"},
               "c": {"correct": False, "finish_reason": "stop"}}
        candidate = {"a": {"correct": True, "finish_reason": "stop"},
                     "b": {"correct": False, "finish_reason": "length"},
                     "c": {"correct": False, "finish_reason": "stop"}}
        report = ev.paired_report(ref, candidate)
        self.assertEqual(report["reference_length_or_loop"]["n"], 1)
        self.assertEqual(report["reference_normal"]["n"], 2)
        self.assertEqual(report["candidate_only_length_or_loop"]["n"], 1)
        self.assertEqual(report["content_accuracy_improvement"], "UNPROVEN")
        self.assertEqual(ev.evaluation_gate(report, None)["status"], "EVAL_REVIEW_REQUIRED")
        report["eligible"] = True
        self.assertEqual(ev.evaluation_gate(report, report)["status"], "EVAL_REVIEW_REQUIRED")
        degraded = {"complete": True, "overall": {"net_change": -20, "p_value": .01}}
        self.assertEqual(ev.evaluation_gate(degraded, report)["status"], "DEGRADED")
        self.assertEqual(ev.paired_report({"a": ref["a"]}, {"a": candidate["a"]})["both_normal"]["p_value"], 1)

    def test_doctor_requires_cuda_bf16_and_disk_content(self):
        value = {"environment": {"torch": {"cuda_available": True, "bf16_probe": {"passed": True}}},
                 "storage": {"sufficient": True, "filesystems": [{"sufficient": True}]}}
        self.assertEqual(ev.doctor_check(value)["status"], "DOCTOR_VERIFIED")
        for bad in ({}, value | {"storage": None}, value | {"storage": {"sufficient": False}}):
            with self.assertRaises(ValueError):
                ev.doctor_check(bad)

    def test_compressed_pack_round_trip_uses_existing_safe_verifier(self):
        source = self.root / "source.json"
        source.write_text('{"optimizer": "fixture"}')
        def fill(tar, records):
            collect._add_member(tar, source, "state/source.json", records)
        manifest = collect._seal(self.root / "out", "closed", fill, compression=True,
                                 closure="CLOSED", resume_capability="RESULT_ONLY")
        self.assertTrue(manifest["archive"].endswith(".tar.gz"))
        receipt = collect.verify_archive(self.root / "out" / manifest["archive"],
                                          self.root / "out/closed.manifest.json", self.root / "extracted",
                                          self.root / "receipt.json")
        self.assertEqual(receipt["status"], "LOCAL_VERIFIED")
        self.assertEqual((self.root / "extracted/state/source.json").read_bytes(), source.read_bytes())

    def test_eval_resume_archive_keeps_raw_samples_invocations_and_input_assets(self):
        artifacts, inputs = self.root / "artifacts", self.root / "inputs"
        inputs.mkdir()
        sources = {name: inputs / (name + ".json") for name in ("prereg", "ledger", "model_ref", "calibration", "preflight")}
        for name, path in sources.items():
            write_json(path, {"kind": name})
        sources["prereg"].with_name("prereg.json.sha256").write_text(sha256_file(sources["prereg"]))
        prereg_sha = sha256_file(sources["prereg"])
        prereg = {"format": pe.FORMAT_V12, "calibration_sha256": sha256_file(sources["calibration"]),
                  "arms": {pe.ARM_SINGLE: {"model_ref_sha256": sha256_file(sources["model_ref"])}},
                  "evaluation_subsets": {"all": {"ids": ["one"]}}}
        registration = {"expected_ids": ["one"], "slots": 8, "serving_kind": "unmerged-lora",
                        "model_sha256": "a" * 64, "preregistration_sha256": prereg_sha,
                        "model_ref_sha256": sha256_file(sources["model_ref"]),
                        "calibration_sha256": sha256_file(sources["calibration"]),
                        "preflight_sha256": sha256_file(sources["preflight"])}
        run_dir, identity = artifacts / "runs/run", {"mode": "partial", "model_sha256": "a" * 64}
        asset = artifacts / "assets/images/temp.png"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"CPU fixture image bytes")
        hashed = asset.with_name(sha256_file(asset) + ".png")
        asset.rename(hashed)
        relative = "../../assets/images/" + hashed.name
        final = final_value("one", registration, identity, "run")
        final["record"]["images"] = [relative]
        final["record"]["messages"][0]["content"].insert(0, {"type": "image", "image": relative, "sha256": sha256_file(hashed)})
        final = ev.seal(final)
        with RunWriter(run_dir, identity, ["one"]) as writer:
            ev.exclusive(run_dir / "final_raw/one.json", final)
            writer.save(ev.record_from_final(final))
        for name in ("environment.json", "code_files.json", "locations.json", "backend.json"):
            write_json(run_dir / name, {})
        for name in ("resolved_eval_config.yaml", "resolved_hardware_config.yaml"):
            (run_dir / name).write_text("fixture: true\n")
        write_json(run_dir / pe.REGISTRATION_FILE, registration)
        write_json(run_dir / "inference_invocations/current.json", {"id": "original-invocation", "attempted_ids": ["one"]})
        events = [{"event": "START", "run_id": "run", "preregistration_sha256": prereg_sha}]
        with mock.patch.object(pe, "load_preregistration", return_value=(prereg, prereg_sha)), \
             mock.patch.object(pe, "read_ledger", return_value=(events, "a" * 64)), \
             mock.patch.dict("os.environ", {"MMDL_ARTIFACT_ROOT": str(self.root)}):
            manifest = collect.pack_eval_resume(run_dir, sources["prereg"], sources["ledger"], sources["model_ref"],
                sources["calibration"], sources["preflight"], self.root / "out", "resume")
            receipt = collect.verify_archive(self.root / "out/resume.tar.gz", self.root / "out/resume.manifest.json",
                                               self.root / "restored", self.root / "receipt.json")
            self.assertEqual(receipt["resume_capability"], "EVALUATION_ONLY")
            self.assertFalse(receipt["training_resume_certified"])
            names = {r["path"] for r in manifest["files"]}
            self.assertIn("eval/runs/run/samples/one.json", names)
            self.assertIn("eval/runs/run/final_raw/one.json", names)
            self.assertIn("eval/assets/images/" + hashed.name, names)
            (self.root / "restored/eval/runs/run/final_raw/one.json").unlink()
            with self.assertRaises(collect.CollectError):
                collect.verify_eval_resume_tree(self.root / "restored", manifest)
            with self.assertRaises(collect.CollectError):
                collect.verify_eval_resume_tree(self.root / "restored", manifest | {"training_resume_certified": True})


if __name__ == "__main__":
    unittest.main()
