"""CPU concurrency/crash and real callback wiring checks; no model download or CUDA execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextlib import ExitStack, nullcontext
import copy
import json
import multiprocessing
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from fine_tuning.error_note import v17_eval as ev
from fine_tuning.error_note import v17_eval_engine as engine
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json


def snapshots(run_dir, count):
    for _ in range(count):
        ev.incremental_manifest(run_dir, force=True)


def die_at_publication(run_dir, after, entered):
    original = ev.exclusive

    def publish(path, value):
        if after:
            original(path, value)
        entered.set()
        multiprocessing.Event().wait(30)

    with mock.patch.object(ev, "exclusive", side_effect=publish):
        ev.incremental_manifest(run_dir, force=True)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "runs/run"
        (self.run_dir / "samples").mkdir(parents=True)
        (self.run_dir / "samples/one.json").write_text('{"id":"one"}\n')
        (self.run_dir / "heartbeat.json").write_text('{"count":1}\n')
        ev.append_event(self.run_dir / "submission_events.jsonl", {"id": "one"})

    def assert_chain(self, count):
        paths = sorted((self.run_dir / "recovery_manifests").glob("*.json"))
        self.assertEqual([p.name for p in paths], [f"{i:06d}.json" for i in range(count)])
        previous = None
        for path in paths:
            value = ev.checked(read_json(path))
            self.assertEqual(value["previous_sha256"], previous)
            for entry in value["files"]:
                source = self.root / entry["path"]
                self.assertEqual(source.stat().st_size, entry["bytes"])
                self.assertEqual(sha256_file(source), entry["sha256"])
            previous = value["sha256"]

    def test_threads_and_processes_share_one_sequence_and_immutable_log_ranges(self):
        ctx = multiprocessing.get_context("spawn")
        processes = [ctx.Process(target=snapshots, args=(self.run_dir, 5)) for _ in range(3)]
        for process in processes:
            process.start()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(snapshots, self.run_dir, 5) for _ in range(4)]
            for i in range(20):
                ev.append_event(self.run_dir / "submission_events.jsonl", {"id": i})
            for future in futures:
                future.result(timeout=15)
        for process in processes:
            process.join(15)
            self.assertEqual(process.exitcode, 0)
        (self.run_dir / "heartbeat.json").write_text('{"count":999}\n')
        self.assert_chain(35)
        self.assertEqual((self.run_dir / "samples/one.json").read_text(), '{"id":"one"}\n')

    def test_death_before_and_after_atomic_publication_releases_process_lock(self):
        ctx = multiprocessing.get_context("spawn")
        snapshots(self.run_dir, 1)
        count = 1
        for after in (False, True):
            entered = ctx.Event()
            process = ctx.Process(target=die_at_publication, args=(self.run_dir, after, entered))
            process.start()
            self.assertTrue(entered.wait(10))
            process.terminate()
            process.join(10)
            self.assertEqual(process.exitcode, -signal.SIGTERM)
            count += int(after)
            snapshots(self.run_dir, 1)
            count += 1
            self.assert_chain(count)


class DirectVisionTests(unittest.TestCase):
    def test_actual_tower_callbacks_and_hook_removal(self):
        import torch
        from fine_tuning.error_note import llm_lora as lora

        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(4, dtype=torch.bfloat16))

            def forward(self, value):
                return value @ self.weight.T, None

        class Tower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                block = torch.nn.Module()
                block.attn = torch.nn.Module()
                block.attn.qkv = Linear()
                self.blocks = torch.nn.ModuleList([block])
                self.merger = torch.nn.Identity()
                self.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Identity()])

            def forward(self, pixels, grid_thw):
                value = self.blocks[0].attn.qkv(pixels.to(torch.bfloat16))[0]
                self.deepstack_merger_list[0](value)
                return self.merger(value)

        hf, actual = torch.nn.Module(), torch.nn.Module()
        hf.model = torch.nn.Module()
        hf.model.visual, actual.visual = Tower(), Tower()
        hf.get_base_model = lambda: hf
        hh, vh = lora.VisionHooks(hf.model), lora.VisionHooks(actual)
        a, b = torch.zeros(16, 4, dtype=torch.bfloat16), torch.zeros(4, 16, dtype=torch.bfloat16)
        a[:4] = torch.eye(4, dtype=torch.bfloat16)
        b[:, :4] = torch.eye(4, dtype=torch.bfloat16) / 1024
        weights = {"visual.blocks.0.attn.qkv": (a, b)}
        hh.load(weights)
        vh.load(weights)

        @contextmanager
        def scaled(bundle, **factors):
            hh.load({name: (aa, bb * factors.get("V", 1)) for name, (aa, bb) in weights.items()})
            try:
                yield
            finally:
                hh.load(weights)

        prep = NS(model_inputs=lambda *args: {"pixel_values": torch.arange(1, 129).reshape(32, 4).float(),
                                              "image_grid_thw": torch.tensor([[1, 4, 8]])})
        backend = NS(vision_hooks=vh, llm=NS(apply_model=lambda callback: [callback(actual)]))
        with mock.patch.object(lora, "scaled_adapters", scaled), \
             mock.patch.object(lora, "vision_lora", side_effect=lambda bundle, factor=1:
                               {name: (aa, bb * factor) for name, (aa, bb) in weights.items()}):
            reference = lora.hf_vision_columns(NS(model=hf), prep, [(None, [])])
            columns = lora.vllm_vision_columns(backend, None, prep, [(None, [])])
        report = lora.vision_tower_check(reference, columns)
        self.assertTrue(report["pass"])
        self.assertTrue(report["amplified"]["pass"])
        self.assertFalse(report["hook_removed"]["pass"])
        self.assertEqual(report["real_effect_status"], "UNCONFIRMED")
        self.assertTrue(vh.live(actual))
        self.assertTrue(vh.on)
        broken = copy.deepcopy(columns)
        broken["columns"]["amplified"] = broken["columns"]["hook_removed"]
        self.assertFalse(lora.vision_tower_check(reference, broken)["pass"])
        broken["inputs"][0]["pixels_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pixels or grid"):
            lora.vision_tower_check(reference, broken)

    def test_final_output_and_language_still_gate_when_direct_tower_supplies_vision_evidence(self):
        from fine_tuning.error_note import llm_lora as lora
        from fine_tuning.tests.test_error_note_single_vllm import pf5_columns

        columns = pf5_columns(size=.08, vision_size=.004, noise=.005, joint_noise=.06, seed=11,
                              vision_only=True, vision=.4)
        self.assertFalse(lora.slope_check(**columns, cap=.04)["pass"])
        self.assertTrue(lora.slope_check(**columns, cap=.04, vision_tower={"pass": True})["pass"])
        self.assertFalse(lora.slope_check(**columns, cap=.04, vision_tower={"pass": False})["pass"])
        columns["v_on"] = [value + 1 for value in columns["v_on"]]
        self.assertFalse(lora.slope_check(**columns, cap=.04, vision_tower={"pass": True})["pass"])


class V18ReceiptTests(unittest.TestCase):
    def test_slot_discovery_is_capped_at_nine(self):
        calls = []

        def measure(slots):
            calls.append(slots)
            return {"safe_slots": 20}

        slots, _ = engine.stable_slots(measure, maximum=9)
        self.assertEqual((slots, calls), (9, [1, 9]))

    def test_v18_cannot_reuse_legacy_pf5_or_an_exception(self):
        from fine_tuning.tests.test_error_note_v17_split_engine import backend_value, engine_receipt

        with tempfile.TemporaryDirectory() as directory:
            value = engine_receipt(Path(directory))
            selected = copy.deepcopy(value["selected_attempt"])
            observed = engine.observe_engine(backend_value(slots=9, blocks=30000), "eager", 9)
            selected["observed_engine"] = observed
            value.update(training_protocol_version="error-note-v18", slots=9, safe_slots=9,
                         observed_engine=observed, actual_kv_token_capacity=480000)
            value["selected_attempt"] = engine.seal(selected)
            with self.assertRaisesRegex(ValueError, "direct vision"):
                engine.validate_engine_receipt(engine.seal(value))

            tower = {"version": "v18-direct-vision-pf5-v1", "pass": True, "amplified": {"pass": True},
                     "hook_removed": {"pass": False}, "real_effect_status": "UNCONFIRMED",
                     "coordinates": 100, "inputs": [{"pixels_sha256": "a" * 64}]}
            evidence = selected["preflight_evidence"]
            for name in ("historical_pf5", "decode_pf5"):
                evidence[name].update(vision_gate="direct_tower", vision_tower=tower)
            selected["preflight_evidence"] = engine.seal(evidence)
            value["selected_attempt"] = engine.seal(selected)
            identity = {"code_commit": value["code_commit"], "source_sha256": value["source_sha256"]}
            training_ref = {"path": "/cpu/training-artifacts.json", "file_sha256": value["training_artifact_sha256"]}
            decision = engine.seal({"version": "v18-evaluation-decision-v1", "decision": "PROCEED",
                "authorized_by": "user", "pre_authorized": False, "authorization": "CPU fixture approval",
                "training_artifact": training_ref, "evaluation": identity, "admission_status": "EVAL_REVIEW_REQUIRED",
                "model_ref": {"file_sha256": value["model_ref_sha256"]}})
            value["evaluation_authorization"] = engine.seal({"status": "PASS", "evaluation": identity,
                "training_artifact": training_ref, "evaluation_decision": decision})
            value["artifact_verification"] = {"path": "/cpu/verified.json", "file_sha256": "a" * 64}
            self.assertEqual(engine.validate_engine_receipt(engine.seal(value))["slots"], 9)
            evidence["decode_pf5"]["vision_exception"] = "old authorization"
            selected["preflight_evidence"] = engine.seal(evidence)
            value["selected_attempt"] = engine.seal(selected)
            with self.assertRaisesRegex(ValueError, "direct vision"):
                engine.validate_engine_receipt(engine.seal(value))

class OfficialResumeTests(unittest.TestCase):
    def test_official_run_and_resume_keep_start_registration_seeds_and_only_pending_ids(self):
        from fine_tuning.notebook_rl import proof_eval as pe
        from fine_tuning.tests.test_error_note_v17_split_eval import SplitEvaluationTests
        from fine_tuning.tests.test_error_note_v17_eval import record
        from mmdl.evaluation import cli, engine as evaluation
        from mmdl.runtime.contracts import sample_seed

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            helper = SplitEvaluationTests()
            ids = [f"item{i:02}" for i in range(20)]
            paths = {key: root / key for key in ("prereg", "ledger", "calibration", "preflight", "seal", "ref", "protocol", "hardware")}
            cfg = {"protocol_id": pe.FINAL_VARIANT_V8}
            for path in paths.values():
                write_json(path, {})
            write_json(paths["protocol"], cfg)
            ref = {"export": {"training_protocol": {"version": "error-note-v18"}, "campaign": {"stage": "final"}}}
            declared = {"model_sha256": "a" * 64, "model_ref_sha256": sha256_file(paths["ref"])}
            observed = helper.registration()["observed_engine"]
            validation = {"slots": 9, "training_protocol_version": "error-note-v18", "observed_engine": observed,
                          "selected_mode": "cuda_graph", "environment_sha256": "b" * 64,
                          "evaluation_commit": "a" * 40}
            prereg = {"format": pe.FORMAT_V12, "training_protocol_version": "error-note-v18",
                "calibration_sha256": sha256_file(paths["calibration"]), "evaluation_watchdog_seconds": 3600,
                "engine_validation": validation, "engine_validation_sha256": sha256_file(paths["calibration"]),
                "training_artifacts_file_sha256": sha256_file(paths["seal"]), "arms": {pe.ARM_SINGLE: declared},
                "evaluation_subsets": {"all": {"ids": ids}},
                "protocol_variants": {pe.FINAL_VARIANT_V8: {"config_digest": digest(cfg)}},
                "execution_contract": {"version": "v17-evaluation-execution-v2", "engine_mode": "cuda_graph"}}
            start = {"event": "START", "run_id": "v18-ft-single", "preregistration_sha256": "c" * 64,
                     "arm": pe.ARM_SINGLE, "variant": pe.FINAL_VARIANT_V8, "subset": "all", "time": ev.utc()}
            run_dir = root / "runs/v18-ft-single"
            identity = {"model_sha256": "a" * 64, "environment_sha256": "b" * 64, "mode": "partial"}
            ready, submissions = {}, []

            @contextmanager
            def variants(**kwargs):
                ready["callback"] = kwargs["after_backend"]
                self.assertEqual(kwargs["slots"], 9)
                yield

            def prepare(sid):
                return {"sample_id": sid, "seed": sample_seed(3407, sid)}, record(sid)

            def execute():
                submitted = []
                backend = helper.backend(submitted)
                backend.batch_size = 9
                resumed = "--resume" in sys.argv
                with evaluation.RunWriter(run_dir, identity, ids, resume=resumed) as writer:
                    ready["callback"](backend)
                    pending = [sid for sid in ids if sid not in writer.completed]
                    invocation = {"id": "second" if resumed else "first", "status": "RUNNING", "attempted_ids": []}
                    path = run_dir / "inference_invocations" / (invocation["id"] + ".json")
                    write_json(path, invocation)

                    def request_stream():
                        for sid in pending:
                            invocation["attempted_ids"].append(sid)
                            write_json(path, invocation)
                            yield evaluation.prepare_request(sid)[0]

                    stream = backend.generate_stream(request_stream())
                    try:
                        for i, result in enumerate(stream, 1):
                            writer.save(ev.record_from_final(read_json(run_dir / "final_raw" / (result["sample_id"] + ".json"))))
                            if not resumed and i == 2:
                                break
                    finally:
                        stream.close()
                    submissions.append(submitted)
                raise RuntimeError("fixture interruption before final report")

            patches = [(pe, "load_preregistration", mock.Mock(return_value=(prereg, "c" * 64))),
                (pe, "read_ledger", mock.Mock(return_value=([start], "d" * 64))),
                (pe, "_lora_model_ref", mock.Mock(return_value=(ref, declared))),
                (pe, "gpu_info", mock.Mock(return_value=("CPU fixture", 0))),
                (pe, "variant_registration", mock.Mock(return_value={"slots": 9})),
                (pe, "registered_variants", variants), (ev, "load_calibration", mock.Mock(return_value=validation)),
                (ev, "calibration_timeout", mock.Mock(return_value=3600)),
                (ev, "split_bindings", mock.Mock(return_value=validation)),
                (ev, "cpu_resume_finalize", mock.Mock(return_value=False)),
                (ev, "source_sha256", mock.Mock(return_value="e" * 64)),
                (engine, "observe_engine", mock.Mock(return_value=observed)),
                (engine, "engine_execution", lambda *args: nullcontext()),
                (evaluation, "prepare_request", prepare), (cli, "main", execute)]
            for owner, name, value in patches:
                stack.enter_context(mock.patch.object(owner, name, value))
            stack.enter_context(mock.patch.dict("os.environ", {pe.VLLM_INPROCESS_ENV: "0", pe.SLOTS_ENV: "9"}))
            common = ["--prereg", str(paths["prereg"]), "--ledger", str(paths["ledger"]),
                      "--calibration", str(paths["calibration"]), "--preflight", str(paths["preflight"]),
                      "--engine-validation", str(paths["calibration"]), "--training-artifacts", str(paths["seal"]), "--"]
            forwarded = ["--protocol", str(paths["protocol"]), "--hardware", str(paths["hardware"]),
                         "--model-ref", str(paths["ref"]), "--model-path", str(root / "model"),
                         "--artifact-root", str(root), "--data-root", str(root / "data"),
                         "--base-path", str(root / "base"), "--run-id", "v18-ft-single", "--mode", "partial"]
            for sid in ids:
                forwarded += ["--sample-id", sid]
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                ev.main(["run", *common, *forwarded])
            registration = (run_dir / pe.REGISTRATION_FILE).read_bytes()
            old = {p.name: sha256_file(p) for p in (run_dir / "samples").glob("*.json")}
            completed = {Path(name).stem for name in old}
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                ev.main(["resume", *common, *forwarded])
            self.assertEqual((run_dir / pe.REGISTRATION_FILE).read_bytes(), registration)
            self.assertEqual(read_json(run_dir / pe.REGISTRATION_FILE)["start"], start)
            self.assertEqual(submissions[1], [sid for sid in ev.submission_order(ids) if sid not in completed])
            self.assertEqual(old, {name: sha256_file(run_dir / "samples" / name) for name in old})
            self.assertEqual(len(list((run_dir / "samples").glob("*.json"))), len(ids))
            self.assertEqual(len(list((run_dir / "logical_invocations").glob("*.json"))), 2)


class V18PreflightTests(unittest.TestCase):
    def helper(self):
        from fine_tuning.tests.test_error_note_v17_preflight import V17Preflight

        helper = V17Preflight()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.plan["protocol"]["version"] = "error-note-v18"
        helper.identity["training_protocol_version"] = "error-note-v18"
        return helper

    def test_v18_identity_and_expected_protocol_reject_an_old_seal(self):
        from fine_tuning.error_note import campaign_v16b as campaign, v17_artifacts as artifacts, v17_preflight as pf
        from fine_tuning.error_note.prep_v16b import seal

        helper = self.helper()
        with mock.patch.object(campaign, "_plan_check"), \
             mock.patch.object(campaign.single, "source_manifest", return_value=helper.source):
            self.assertEqual(pf._identity(helper.args, helper.plan), helper.identity)
        path = helper.root / "legacy-plan.json"
        seal(path, {"protocol": {"version": "error-note-v17"}})
        with mock.patch.object(artifacts, "verify_training_artifacts", return_value={"campaign_plan": {"path": str(path)}}), \
             mock.patch.object(artifacts, "evaluation_identity") as identity:
            with self.assertRaisesRegex(ValueError, "expected protocol"):
                artifacts.main(["verify", "--receipt", str(helper.root / "seal"), "--repo", str(helper.root),
                                "--commit", "a" * 40, "--output", str(helper.root / "checked"),
                                "--expected-protocol", "error-note-v18"])
            identity.assert_not_called()

    def test_real_calibration_entry_uses_v18_policy_masks_rows_and_objective(self):
        from fine_tuning.error_note import campaign_v16b as campaign, v17_data, v17_preflight as pf, v18_rows, protocol

        helper = self.helper()
        sampler, timing, _ = helper._mock_calibration()
        policy = {"reward": 0., "reason": "FORMAT_ONLY_CERTIFIED", "pg": True,
                  "loss_token_mask": [False, True], "pg_mask_reason": "FORMAT_ONLY_CERTIFIED",
                  "pg_byte_spans": [[1, 2]], "policy_mask_receipt": {"fixture": "bound"}, "format_receipt": {"accepted": True}}
        with mock.patch.object(campaign, "data_api", return_value=v17_data), \
             mock.patch.object(v18_rows, "problem_contract_v17", return_value={"verifier": None, "fact_catalog": [], "rules": {}}), \
             mock.patch.object(v18_rows, "t1_policy_receipt_v18", return_value=policy), \
             mock.patch.object(v18_rows, "verify_solution_v17", return_value={"accepted": True}), \
             mock.patch.object(v18_rows, "a1_row_v17", return_value={"kind": "A1", "schema": v18_rows.ROW_SCHEMA,
                                  "row_id": "perf-only:A1", "route": "FULL"}), \
             mock.patch.object(protocol, "v18_contract_for_update", return_value={"version": "error-note-v18"}):
            receipt = pf.calibrate(helper.args, helper.plan)
        self.assertEqual(receipt["training_protocol_version"], "error-note-v18")
        encoded = timing.call_args.args[1]
        row = next(row for row in encoded if row["kind"] == "RL")
        self.assertEqual(row["schema"], v18_rows.ROW_SCHEMA)
        self.assertEqual(row["loss_token_mask"], policy["loss_token_mask"])
        self.assertEqual(row["policy_mask_receipt"], policy["policy_mask_receipt"])
        self.assertEqual(row["reason"], policy["reason"])
        cost = read_json(helper.root / "update-cost-calibration.json")
        self.assertEqual(cost["measurement"]["objective_version"], "error-note-v18")
        self.assertEqual(cost["measurement"]["row_schema"], v18_rows.ROW_SCHEMA)
        self.assertEqual(cost["production_parents_consumed"], 0)
        cost["measurement"]["objective_version"] = "error-note-v17"
        with self.assertRaisesRegex(ValueError, "objective and row schema"):
            pf._validate_cost(cost)
        sampler.release.assert_called_once()


class GeneratedEvaluationPlanTests(unittest.TestCase):
    def test_current_plan_cli_emits_real_verifier_and_engine_parser_arguments(self):
        from fine_tuning.error_note import plan as planner, v17_artifacts as artifacts
        from fine_tuning.tests.test_error_note_v17_split_plan import parse_only, split_params
        from fine_tuning.notebook_rl.pod.chain import check_spec

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = split_params(root / "private", "v17-eval")
            params["version"] = planner.PARAMS_VERSION_V18_SPLIT
            params["evaluation_authorization"] = "CPU fixture explicit evaluation approval"
            write_json(root / "params.json", params)
            with mock.patch("builtins.print"):
                self.assertEqual(planner.main(["plan", "--stage", "v18-eval", "--params", str(root / "params.json"),
                                               "--output", str(root / "spec.json")]), 0)
            spec = check_spec(json.loads((root / "spec.json").read_text()))
            steps = {row["name"]: row["argv"] for row in spec["steps"]}
            argv = steps["verify-training-artifacts"]
            parsed = parse_only(artifacts, argv[argv.index("fine_tuning.error_note.v17_artifacts") + 1:])
            self.assertEqual(parsed.expected_protocol, "error-note-v18")
            self.assertEqual(parsed.authorization, params["evaluation_authorization"])
            argv = steps["eval-engine-validation"]
            parsed = parse_only(engine, argv[argv.index("fine_tuning.error_note.v17_eval_engine") + 1:])
            self.assertEqual(parsed.training_seal, Path(params["training_artifacts_path"]))
            self.assertEqual(parsed.requests, Path(params["performance_requests_path"]))
            self.assertEqual(parsed.requests_sha256, params["performance_requests_sha256"])
            self.assertEqual(parsed.artifact_verification, Path(params["out_root"]) / "training-artifacts-verified.json")
            self.assertIsNone(parsed.pf5_vision_exception)



if __name__ == "__main__":
    unittest.main()
