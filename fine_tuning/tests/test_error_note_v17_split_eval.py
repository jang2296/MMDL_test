"""CPU checks for the split evaluator's lazy submission and sealed execution bindings."""
from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest import mock

from fine_tuning.error_note import v17_eval as ev
from fine_tuning.tests.test_error_note_v17_eval import engine_value, output, record
from mmdl.evaluation import engine as evaluation
from mmdl.evaluation.backends.vllm_backend import VLLMBackend
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import sample_seed


class SplitEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_reorders_only_ids_before_any_lazy_input_preparation(self):
        pending, prepared = ["a", "b", "c"], []

        def request_stream():
            for sid in pending:
                prepared.append(sid)
                yield sid

        requests = request_stream()
        self.assertEqual(ev.reorder_request_stream(requests, pending[:]), ev.submission_order(pending))
        self.assertEqual(prepared, [])
        self.assertEqual(list(requests), ev.submission_order(["a", "b", "c"]))
        for invalid in (iter(["a"]), (x for x in ["a"])):
            with self.assertRaises(ValueError):
                ev.reorder_request_stream(invalid, ["a"])
        with self.assertRaises(ValueError):
            ev.submission_order(["a", "a"])

    def backend(self, submitted, *, fail_enqueue=False):
        engine = engine_value()
        active = engine.engine_core.engine_core.scheduler.requests

        def add(sid, prompt, params, **kwargs):
            if fail_enqueue:
                raise RuntimeError("enqueue interrupted")
            active[sid] = NS(num_tokens=11)
            submitted.append(sid)

        def step():
            sid = next(iter(active))
            active.pop(sid)
            return [NS(request_id=sid, finished=True, outputs=[NS(token_ids=[1, 2, 3])])]

        engine.add_request, engine.step = add, step
        engine.abort_request = lambda pending: [active.pop(sid, None) for sid in pending]
        backend = NS(llm=NS(llm_engine=engine, get_metrics=lambda: []), vllm_version="0.11.0",
                     batch_size=2, max_model_len=40960, cfg={"execution": {"scheduling": "continuous"}},
                     _prepare_request=lambda request, **kwargs: ({}, {}, None),
                     _row=lambda request, *args, **kwargs: output(request["sample_id"]),
                     describe=lambda: {"vllm": {"enforce_eager": True}})
        backend.generate_stream = MethodType(VLLMBackend.generate_stream, backend)
        return backend

    def exercise(self, registration, *, resume=False, stop_after=None, fail_enqueue=False, observed=None):
        run_dir = self.root / "run"
        identity = {"model_sha256": "a" * 64, "mode": "partial"}
        submitted, prepared = [], []
        backend = self.backend(submitted, fail_enqueue=fail_enqueue)
        original_stream, original_describe = backend.generate_stream, backend.describe

        def prepare(sid):
            prepared.append(sid)
            return {"sample_id": sid, "seed": sample_seed(3407, sid)}, record(sid)

        with mock.patch.object(evaluation, "prepare_request", side_effect=prepare), \
             mock.patch("fine_tuning.error_note.v17_eval_engine.observe_engine",
                        return_value=registration["observed_engine"] if observed is None else observed):
            with ev.evaluation_hooks(run_dir, registration, time.time() + 60) as ready:
                with evaluation.RunWriter(run_dir, identity, registration["expected_ids"], resume=resume) as writer:
                    ready(backend)
                    self.assertFalse(backend.describe()["vllm"]["enforce_eager"])
                    self.assertEqual(prepared, [])
                    pending = [sid for sid in registration["expected_ids"] if sid not in writer.completed]
                    current = "second" if resume else "first"
                    invocation_path = run_dir / "inference_invocations" / (current + ".json")
                    invocation = {"id": current, "status": "RUNNING", "attempted_ids": []}
                    write_json(invocation_path, invocation)

                    def request_stream():
                        for sid in pending:
                            invocation["attempted_ids"].append(sid)
                            write_json(invocation_path, invocation)
                            yield evaluation.prepare_request(sid)[0]

                    stream = backend.generate_stream(request_stream())
                    try:
                        for index, result in enumerate(stream, 1):
                            writer.save(ev.record_from_final(read_json(run_dir / "final_raw" / (result["sample_id"] + ".json"))))
                            if index == stop_after:
                                break
                    finally:
                        stream.close()
                    self.assertEqual(invocation["attempted_ids"], submitted)
        self.assertIs(backend.generate_stream, original_stream)
        self.assertIs(backend.describe, original_describe)
        return submitted

    def registration(self):
        ids = [f"item{i:02}" for i in range(7)]
        return {"expected_ids": ids, "slots": 2, "model_sha256": "a" * 64,
                "execution_contract": {"version": "v17-evaluation-execution-v2", "engine_mode": "cuda_graph"},
                "observed_engine": ev.seal({"enforce_eager": False, "compilation_config": {"level": 3},
                                             "actual_kv_token_capacity": 444416})}

    def test_actual_enqueue_order_resume_and_final_raw_immutability(self):
        registration = self.registration()
        first = self.exercise(registration, stop_after=2)
        ordered = ev.submission_order(registration["expected_ids"])
        self.assertEqual(first, ordered[:3])  # two completions plus one still active at interruption
        run_dir = self.root / "run"
        old = {p.name: sha256_file(p) for p in (run_dir / "samples").glob("*.json")}
        completed = [p.stem for p in (run_dir / "samples").glob("*.json")]
        second = self.exercise(registration, resume=True)
        self.assertEqual(second, [sid for sid in ordered if sid not in completed])
        self.assertEqual(old, {name: sha256_file(run_dir / "samples" / name) for name in old})
        audit = ev.submission_audit(run_dir, registration)
        self.assertEqual(audit["actual_submitted_ids"], first + second)
        self.assertEqual(audit["actual_submission_sha256"], digest(first + second))
        self.assertEqual(audit["submission_order_sha256"], digest(ordered))
        self.assertEqual(len(list((run_dir / "samples").glob("*.json"))), 7)
        with self.assertRaises(ValueError):
            ev.submission_audit(run_dir, registration | {"slots": 3})

    def test_fresh_process_kv_capacity_may_differ_but_execution_settings_may_not(self):
        registration = self.registration()
        # A fresh 900-run process counts different non-torch memory, so its KV blocks differ from validation.
        fresh = dict(registration["observed_engine"], actual_kv_token_capacity=430000, sha256="0" * 64)
        self.assertEqual(len(self.exercise(registration, observed=fresh)), 7)
        changed = dict(registration["observed_engine"], enforce_eager=True)
        (self.root / "run").rename(self.root / "run-first")
        with self.assertRaisesRegex(ValueError, "actual engine differs"):
            self.exercise(registration, observed=changed)

    def test_enqueue_interruption_retains_intent_and_detects_tampering(self):
        registration = self.registration()
        with self.assertRaisesRegex(RuntimeError, "enqueue interrupted"):
            self.exercise(registration, fail_enqueue=True)
        audit = ev.submission_audit(self.root / "run", registration)
        self.assertEqual(audit["actual_submitted_ids"], [])
        self.assertEqual(audit["interrupted_intent_id"], ev.submission_order(registration["expected_ids"])[0])
        path = self.root / "run/submission_events.jsonl"
        text = path.read_text()
        path.write_text(text.replace('"previous": null', '"previous": "wrong"'))
        with self.assertRaises(ValueError):
            ev.submission_audit(self.root / "run", registration)

    def test_split_bindings_reject_changed_final_model_or_protected_evaluator(self):
        from fine_tuning.error_note import v17_artifacts as artifacts, v17_eval_engine as engine

        model, seal_path, validation_path = [self.root / name for name in ("model.json", "seal.json", "engine.json")]
        for path in (model, seal_path, validation_path):
            write_json(path, {"fixture": path.name})
        sealed = {"protected_evaluation": {"sha256": "protected"},
                  "export": {"model_ref": {"file_sha256": sha256_file(model)}}}
        validation = {"training_artifact_sha256": sha256_file(seal_path), "model_ref_sha256": sha256_file(model),
                      "evaluation_commit": "a" * 40, "source_sha256": "b" * 64}
        with mock.patch.object(engine, "load_engine_validation", return_value=validation), \
             mock.patch.object(artifacts, "verify_training_artifacts", return_value=sealed), \
             mock.patch.object(artifacts, "protected_evaluation", return_value=sealed["protected_evaluation"]), \
             mock.patch.object(artifacts, "evaluation_identity", return_value={"code_commit": "a" * 40, "source_sha256": "b" * 64}):
            self.assertEqual(ev.split_bindings(validation_path, seal_path, model), validation)
            write_json(model, {"changed": True})
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                ev.split_bindings(validation_path, seal_path, model)

    def test_protected_engine_generator_submits_hash_order_with_lazy_fake_backend(self):
        import yaml
        from fine_tuning.notebook_rl import proof_eval as pe
        from mmdl.evaluation.backends import vllm_backend

        registration = self.registration()
        registration["model_sha256"] = digest([])
        ids = registration["expected_ids"]
        data = self.root / "data"
        data.mkdir()
        write_json(data / "manifest.json", {"files": []})
        model_ref = self.root / "model_ref.json"
        write_json(model_ref, {"files": [], "kind": "base"})
        inputs = self.root / "inputs"
        inputs.mkdir()
        prereg_path, ledger, calibration, training_seal = [inputs / name for name in
            ("prereg.json", "ledger.jsonl", "calibration.json", "training-artifacts.json")]
        evidence = inputs / "pf5.json"
        write_json(evidence, {"fixture": "engine PF5"})
        write_json(calibration, {"attempts": [], "preflight_report": str(evidence),
                                 "preflight_file_sha256": sha256_file(evidence)})
        write_json(training_seal, {"fixture": "training seal"})
        prereg = {"format": pe.FORMAT_V12, "engine_validation": {},
                  "calibration_sha256": sha256_file(calibration),
                  "arms": {pe.ARM_SINGLE: {"model_ref_sha256": sha256_file(model_ref)}},
                  "evaluation_subsets": {"all": {"ids": ids}}}
        write_json(prereg_path, prereg)
        prereg_sha = sha256_file(prereg_path)
        prereg_path.with_name(prereg_path.name + ".sha256").write_text(prereg_sha)
        events = [{"event": "START", "run_id": "engine", "preregistration_sha256": prereg_sha}]
        write_json(ledger, events)
        registration.update({"serving_kind": pe.LORA_KIND, "preregistration_sha256": prereg_sha,
                             "model_ref_sha256": sha256_file(model_ref),
                             "calibration_sha256": sha256_file(calibration), "preflight_sha256": sha256_file(calibration),
                             "training_artifacts_path": str(training_seal)})
        cfg = yaml.safe_load(pe.VARIANTS[pe.FINAL_VARIANT_V8][0].read_text())
        hardware = pe.ROOT / "configs/hardware/rtx4090_24gb.yaml"
        args = NS(artifact_root=self.root, data_root=data, model_ref=model_ref, model_path=self.root,
                  base_path=self.root, require_commit=None, hardware=hardware, public_root=self.root / "public",
                  protocol=pe.VARIANTS[pe.FINAL_VARIANT_V8][0], run_id="engine", job_id=None,
                  run_role="standalone", mode="partial", subject=None, sample_id=ids, limit=None, resume=False)
        environment = {"python": "fixture", "platform": {}, "torch": {"cuda_runtime": "12.8",
                       "bf16_probe": {"passed": True}}, "nvidia_smi": {"gpus": []}}
        dataset = {"id": ids} | {i: {"id": sid} for i, sid in enumerate(ids)}
        submitted, prepared = [], []

        def prepare(row, *args, **kwargs):
            prepared.append(row["id"])
            return {"sample_id": row["id"], "seed": sample_seed(3407, row["id"])}, record(row["id"])

        run_dir = self.root / "runs/engine"
        with mock.patch.object(evaluation, "prepare_request", side_effect=prepare), \
             mock.patch.object(evaluation, "collect_environment", return_value=environment), \
             mock.patch.object(evaluation, "check_storage", return_value={"sufficient": True}), \
             mock.patch.object(evaluation, "verify_model"), mock.patch.object(evaluation, "publish"), \
             mock.patch.object(evaluation, "git_commit", return_value="c" * 40), \
             mock.patch.object(evaluation, "load_validation", return_value=({"fixture": dataset}, {})), \
             mock.patch.object(evaluation, "code_records", return_value=[]), \
             mock.patch("mmdl.evaluation.writer.render"), mock.patch.object(evaluation, "render"), \
             mock.patch("fine_tuning.error_note.v17_eval_engine.observe_engine", return_value=registration["observed_engine"]):
            with ev.evaluation_hooks(run_dir, registration, time.time() + 60) as ready:
                def factory(*args, **kwargs):
                    self.assertEqual(prepared, [])
                    backend = self.backend(submitted)
                    backend.cfg, backend.load_seconds = cfg, 0
                    ready(backend)
                    return backend

                with mock.patch.object(vllm_backend, "VLLMBackend", side_effect=factory):
                    summary = evaluation.run(args, cfg, yaml.safe_load(hardware.read_text()), pe.ROOT)
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(submitted, ev.submission_order(ids))
        self.assertEqual(prepared, submitted)
        self.assertEqual(read_json(run_dir / "backend.json")["vllm"]["enforce_eager"], False)
        audit = ev.submission_audit(run_dir, registration)
        self.assertEqual(audit["actual_submitted_ids"], submitted)

        from fine_tuning.notebook_rl.pod import collect

        # Artifact-input verification has its own fixtures; keep actual hook UUIDs and collection auditing connected.
        with mock.patch.object(pe, "load_preregistration", return_value=(prereg, prereg_sha)), \
             mock.patch.object(pe, "read_ledger", return_value=(events, "a" * 64)), \
             mock.patch.object(collect, "_split_resume_inputs", return_value=registration["execution_contract"]), \
             mock.patch.dict("os.environ", {"MMDL_ARTIFACT_ROOT": str(self.root)}):
            manifest = collect.pack_eval_resume(run_dir, prereg_path, ledger, model_ref,
                calibration, calibration, self.root / "archive", "actual-engine")
            receipt = collect.verify_archive(self.root / "archive/actual-engine.tar.gz",
                self.root / "archive/actual-engine.manifest.json", self.root / "restored",
                self.root / "verified.json")
        self.assertEqual(receipt["resume_capability"], "EVALUATION_ONLY")
        self.assertEqual(manifest["resume_capability"], "EVALUATION_ONLY")
        self.assertEqual(receipt["durable_final_raw"], len(ids))

    def test_split_preregistration_roundtrip_and_execution_tamper_rejection(self):
        import json
        from fine_tuning.error_note import v17_eval_engine as engine
        from fine_tuning.notebook_rl import proof_eval as pe
        from fine_tuning.tests.test_notebook_rl_proof_eval import DECLARED, IDS, REST

        baseline = self.root / "b2.jsonl"
        baseline.write_text("".join(json.dumps({"id": sid, "correct": sid in REST}) + "\n" for sid in IDS))
        paths = {}
        for kind in ("wrong", "rest", "all"):
            paths[kind] = self.root / (kind + ".json")
            pe.make_subset(kind, paths[kind], baseline)
        receipt, artifacts = self.root / "engine.json", self.root / "artifacts.json"
        write_json(artifacts, {"fixture": True})
        validated = {"version": "v17-eval-engine-validation-v1", "selected_mode": "cuda_graph", "slots": 5,
                     "actual_kv_token_capacity": 240000, "estimated_evaluation_seconds": 18000,
                     "evaluation_watchdog_seconds": 32000, "model_ref_sha256": DECLARED["model_ref_sha256"]}
        validated = ev.seal(validated)
        write_json(receipt, validated)
        ref = {"export": {"training_protocol": {"version": "error-note-v17", "training_contract": {
                     "prep_updates": 12, "rl_blocks": 188, "total_slots": 203}},
                     "campaign": {"stage": "final", "slot": 203, "updates": 201}}}
        prereg_path = self.root / "prereg.json"
        with mock.patch.object(pe, "_lora_model_ref", return_value=(ref, DECLARED)), \
             mock.patch.object(ev, "split_bindings", return_value=validated), \
             mock.patch.object(engine, "load_engine_validation", return_value=validated), \
             mock.patch.object(engine, "validate_engine_receipt", side_effect=lambda value: value):
            pe.preregister(prereg_path, paths["wrong"], paths["rest"], baseline, "v12",
                           ft_single_model_ref=self.root / "model_ref.json", all_subset=paths["all"],
                           calibration=receipt, engine_validation=receipt, training_artifacts=artifacts)
            value, _ = pe.load_preregistration(prereg_path)
            self.assertEqual(value["execution_contract"]["slots"], 5)
            self.assertEqual(value["execution_contract"]["engine_mode"], "cuda_graph")
            self.assertEqual(value["estimated_evaluation_seconds"], 18000)
            self.assertEqual(value["evaluation_watchdog_seconds"], 32000)
            value["execution_contract"]["engine_mode"] = "eager"
            write_json(prereg_path, value)
            prereg_path.with_name(prereg_path.name + ".sha256").write_text(sha256_file(prereg_path))
            with self.assertRaisesRegex(ValueError, "registration mismatch"):
                pe.load_preregistration(prereg_path)

    def test_execution_contract_changes_are_independent_of_b2_config(self):
        from fine_tuning.error_note import v17_eval_engine as engine

        validated = {"selected_mode": "cuda_graph", "slots": 5, "actual_kv_token_capacity": 240000}
        before = copy.deepcopy(validated)
        with mock.patch.object(engine, "validate_engine_receipt", side_effect=lambda value: value):
            value = ev.split_execution_contract(validated, ["a", "b"])
        self.assertEqual(validated, before)
        self.assertEqual(value["slots"], 5)
        self.assertEqual(value["b2_execution_change"]["selected"], "cuda_graph")
        self.assertFalse(value["b2_execution_change"]["bit_identical_claimed"])


if __name__ == "__main__":
    unittest.main()
