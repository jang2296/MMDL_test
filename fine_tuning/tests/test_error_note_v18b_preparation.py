"""CPU cache and live-admission contracts; these are not BF16 execution evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fine_tuning.error_note import prep_v16b as prep, prep_reuse_v18b as reuse, v17_preflight as pf
from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.protocol import EVAL_SAMPLING_V16B, SINGLE_IMAGE_V14
from fine_tuning.tests.test_error_note_v16b_prep import item


class PreparationDispatch(unittest.TestCase):
    def test_public_preflight_dispatches_every_supported_version_without_replacing_preflight(self):
        for version in ("error-note-v16b", "error-note-v17", "error-note-v18", "error-note-v18b"):
            with self.subTest(version=version):
                result = prep.preflight_items([item("prep0")], version=version)
                self.assertEqual(result["version"], version.removeprefix("error-note-") + "-data-contract-preflight-v1")
                self.assertEqual(result["count"], 1)
                self.assertFalse(result["model_loaded"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            prep.preflight_items([], version="error-note-unknown")

    def test_v18b_selection_preserves_frozen_200_and_generates_only_149_plus_64(self):
        roles = {name: [item(f"{name}{i}") | {"training_allowed": name in {"train", "prep"}}
                        for i in range(count)] for name, count in {
                            "train": 3000, "prep": 512, "monitor": 200,
                            "effect": 128, "performance": 16, "reasoning_monitor": 64}.items()}
        for entry in roles["monitor"][:51]:
            entry["source"] = "hf-scienceqa"
        data = NS(validate=lambda root: {"fixture": True}, load_items=lambda root, role: deepcopy(roles[role]))
        with TemporaryDirectory() as directory, patch.object(prep, "data_api", return_value=data), patch.object(
                prep, "file_sha", return_value="b" * 64):
            root = Path(directory)
            selected = prep.select(root, root / "production", version="error-note-v18b")
            smoke = prep.select(root, root / "smoke", version="error-note-v18b", local_sequential=True)
        self.assertEqual(len(selected["frozen_role_ids"]["monitor"]), 200)
        self.assertEqual(len(selected["excluded_ceiling_monitor_ids"]), 51)
        self.assertEqual(len(selected["monitor_ids"]), 149)
        self.assertEqual(len(selected["reasoning_monitor_ids"]), 64)
        self.assertEqual(len(selected["prep_ids"]), 512)
        self.assertEqual(len(selected["role_ids"]["train"]), 3000)
        self.assertFalse(set(selected["monitor_ids"]) & set(selected["excluded_ceiling_monitor_ids"]))
        self.assertEqual(smoke["monitor_ids"], selected["monitor_ids"][:2])
        self.assertEqual(len(smoke["frozen_role_ids"]["monitor"]), 200)


class F0Reuse(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.item = item("prep0")
        self.request = prep.make_request(self.item, prep.p0(self.item), "v18-prep", cap=8192,
            sampling=EVAL_SAMPLING_V16B, protocol_sha256="b" * 64, policy_sha256="a" * 64)
        generation = {key: self.request[key] for key in ("request_id", "problem_id", "stage", "seed", "max_tokens",
            "request_sha256", "protocol_sha256", "policy_sha256")}
        generation.update(policy_version=0, visual_version=0, mixed_version=False, completion_count=1,
            eos_token_ids=[14],
            input_sha256="c" * 64, prompt_token_ids=[11, 12], completion_token_ids=[13, 14],
            sampling=EVAL_SAMPLING_V16B, sampling_sha256=digest(EVAL_SAMPLING_V16B),
            text="Final answer: A", finish_reason="stop")
        output = {"text": "Final answer: A", "finish_reason": "stop", "generation": generation,
                  "status": "WRONG", "extracted": None}
        self.row = prep.seal(self.root / "record.json", {"problem_id": "prep0", "stage": "v18-prep",
            "prompt": prep.p0(self.item), "request_sha256": self.request["request_sha256"],
            "protocol_sha256": "b" * 64, "policy_sha256": "a" * 64, "output": output})
        self.source = prep.seal(self.root / "source.json", {"version": "v18-preparation-v1",
            "initial_policy": "fresh_F0", "image": SINGLE_IMAGE_V14, "cap": 8192,
            "prep": [self.row], "monitor": [], "effect": []})
        self.evidence = prep.seal(self.root / "evidence.json", {"producer_commit": "d" * 40})
        self.args = NS(version="error-note-v18b", precision="bf16", local_sequential=False,
            output=self.root / "out", data_root=self.root / "data", model_path=self.root / "model",
            allow_f0_raw_reuse=True, f0_reuse_source=self.root / "source.json",
            f0_reuse_sha256=prep.file_sha(self.root / "source.json"),
            f0_reuse_evidence=self.root / "evidence.json",
            f0_reuse_evidence_sha256=prep.file_sha(self.root / "evidence.json"))
        self.processor = NS(tokenizer=NS(decode=lambda *a, **k: "Final answer: A"))
        self.prepared = {"input_sha256": "c" * 64, "prompt_token_ids": [11, 12]}

    def validate(self, row):
        with patch("fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs", return_value=(self.prepared, [])):
            reuse._validate_record(row, self.item, "prep", processor=self.processor,
                model_config=object(), image=SINGLE_IMAGE_V14, cap=8192, sampling=EVAL_SAMPLING_V16B)

    def test_stage_names_preserve_v18_prep_and_effect_seed_but_separate_b2_monitor(self):
        for role in ("prep", "effect"):
            self.assertEqual(prep.preparation_stage("error-note-v18b", role), "v18-" + role)
        self.assertEqual(prep.preparation_stage("error-note-v18b", "monitor"), "v18b-b2-monitor")
        self.assertEqual(prep.preparation_stage("error-note-v18b", "reasoning_monitor"), "v18b-b2-reasoning-monitor")

    def test_import_off_is_default_and_source_requires_explicit_local_authorization(self):
        self.assertEqual(reuse.reuse_f0(NS())["status"], "DISABLED")
        self.args.allow_f0_raw_reuse = False
        with self.assertRaisesRegex(ValueError, "allow-f0-raw-reuse"):
            reuse.reuse_f0(self.args)

    def test_request_and_raw_identity_are_each_checked(self):
        self.validate(self.row)
        mutations = (("seed", self.request["seed"] + 1), ("prompt_token_ids", [11, 15]),
            ("input_sha256", "f" * 64), ("sampling", EVAL_SAMPLING_V16B | {"top_p": 1.}),
            ("max_tokens", 32768), ("policy_version", 1), ("text", "invented"), ("completion_token_ids", []))
        for key, value in mutations:
            with self.subTest(key=key):
                bad = deepcopy(self.row)
                bad["output"]["generation"][key] = value
                bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
                with self.assertRaises(ValueError):
                    self.validate(bad)

    def test_cpu_import_regrades_and_preserves_original_generation_provenance(self):
        data = NS(load_items=lambda root, role: [self.item] if role == "prep" else [])
        original = deepcopy(self.row)
        with patch.object(prep, "select", return_value={"prep_ids": ["prep0"], "sha256": "a" * 64}), patch.object(
                prep, "data_api", return_value=data), patch.object(reuse, "_validate_evidence", return_value=[]), patch.object(
                prep, "grader", return_value=lambda *a: ("CORRECT", "A")), patch.object(
                prep.single, "source_manifest", return_value={"base_commit": "e" * 40, "actual_files": {"validator": "f" * 64}}), patch(
                "fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs", return_value=(self.prepared, [])):
            result = reuse.reuse_f0(self.args, processor=self.processor, model_config=object())
        self.assertEqual(result["status"], "VERIFIED_PARTIAL")
        imported = result["records"]["prep"][0]
        self.assertEqual(imported["output"]["status"], "CORRECT")
        self.assertEqual(imported["output"]["generation"], original["output"]["generation"])
        self.assertEqual(imported["reuse"]["producer_record_sha256"], original["sha256"])
        self.assertEqual(self.row, original)
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["upload_authorized"])

    def test_changed_source_bytes_are_rejected_before_processor_or_model_load(self):
        with self.args.f0_reuse_source.open("a") as handle:
            handle.write(" ")
        with patch("transformers.AutoProcessor.from_pretrained") as processor, self.assertRaisesRegex(ValueError, "unchanged"):
            reuse.reuse_f0(self.args)
        processor.assert_not_called()

    def test_model_bytes_and_dependency_scope_cannot_be_certified_by_status_only(self):
        evidence = {"version": reuse.SOURCE_VERSION, "producer_attempt": 7, "producer_commit": "d" * 40,
            "source_file_sha256": self.args.f0_reuse_sha256, "source_seal_sha256": self.source["sha256"],
            "initial_policy": "fresh_F0", "context_limit": 24576, "model": {}}
        with self.assertRaisesRegex(ValueError, "model and processor"):
            reuse._validate_evidence(evidence, self.source, self.args.f0_reuse_sha256, self.args.model_path)

    def test_legacy_generator_comparison_keeps_sampling_changes_and_side_effects_visible(self):
        original = """class Sampler:
    def run(self):
        if self.protocol_version in ('error-note-v17', 'error-note-v18'):
            return draw(seed=3407, cap=CONTEXT_LIMIT)
"""
        inherited = original.replace("'error-note-v18')", "'error-note-v18', 'error-note-v18b')").replace(
            "cap=CONTEXT_LIMIT", "cap=self.context_limit")
        before = reuse._legacy_generator_ast(original, version="error-note-v18")
        self.assertEqual(before, reuse._legacy_generator_ast(inherited, version="error-note-v18b"))
        self.assertNotEqual(before, reuse._legacy_generator_ast(inherited.replace("seed=3407", "seed=3408"),
                                                               version="error-note-v18b"))
        self.assertNotEqual(reuse._legacy_generator_ast("if side_effect() and False: pass", version="error-note-v18"),
                            reuse._legacy_generator_ast("if False: pass", version="error-note-v18b"))


class LiveAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = NS(output=self.root, precision="bf16", engine="vllm", allow_model_run=True)
        self.plan = {"sha256": "a" * 64, "protocol": {"version": "error-note-v18b", "precision": "bf16"},
                     "inputs": {"payload": {"file_sha256": "b" * 64}}}
        bounds = {"dataset_sha256": "b" * 64, "max_images": 1, "max_vision_patches": 16, "max_prompt_tokens": 16384}
        self.capacity = {"version": "v18b-capacity-v1", "model_parameter_update": False, "shape_bounds": bounds,
            "report": {"version": "v17-update-memory-v1", "optimizer_steps": 0, "measurement": "actual_cuda_bf16_shared_backward_only",
                "live_engine": True, "fixture_only": True, "action_tokens": 8192, "prompt_tokens": 16384,
                "shape_bounds": bounds, "shape_bounds_sha256": digest(bounds), "post_backward_memory": {"free": 3 * 2**30}}}

    def test_one_live_capacity_call_and_no_abc_or_production_consumption(self):
        sampler = Mock(preflight_report={"pass": True}, coexistence_reports=[{"pass": True}])
        sampler.has_unfinished_requests.return_value = False
        capacity = Mock(return_value=self.capacity)
        with patch.object(pf, "_identity", return_value={"code_commit": "c" * 40}):
            result = pf.live_training_preflight(self.args, self.plan, object(), sampler, capacity=capacity)
            capacity.assert_called_once_with()
            self.assertTrue(result["production_admitted"])
            self.assertEqual(result["maximum_shape_measurements"], 1)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["production_parents_consumed"], 0)
            with self.assertRaises(FileExistsError):
                pf.live_training_preflight(self.args, self.plan, object(), sampler, capacity=capacity)
        capacity.assert_called_once_with()

    def test_stale_shape_wrong_precision_and_low_headroom_fail(self):
        for field, value in (("live_engine", False), ("action_tokens", 256), ("shape_bounds_sha256", "0" * 64),
                             ("post_backward_memory", {"free": 1 * 2**30})):
            bad = deepcopy(self.capacity)
            bad["report"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "maximum FULL"):
                pf._validate_live_capacity(bad, self.plan)

    def test_v18b_rejects_separate_abc_job_before_loading_weights(self):
        self.args.receipt_output = self.root / "unused.json"
        with patch.object(pf, "_identity", return_value={}), patch(
                "fine_tuning.error_note.campaign_v16b.load_f0") as load, self.assertRaisesRegex(ValueError, "A/B/C jobs are removed"):
            pf.calibrate(self.args, self.plan)
        load.assert_not_called()

    def test_live_memory_probe_keeps_resident_engine_pool_and_rejects_low_headroom(self):
        from contextlib import ExitStack
        import torch
        from fine_tuning.error_note import capacity, train_v17 as train, train_v4
        from fine_tuning.error_note.contract import ROUTE_FULL
        from fine_tuning.error_note.protocol import ARM, plain_protocol
        from fine_tuning.notebook_rl.policy import configure_policy
        from fine_tuning.tests.test_training import tiny_model
        from fine_tuning.tests.test_error_note_v17_loss import encoded

        bundle = configure_policy(tiny_model().to(dtype=torch.bfloat16), None, ARM, plain_protocol("bf16"), enforce_counts=False)
        readings = {"free": 3 * 2**30, "total": 80 * 2**30, "allocated": 20 * 2**30, "reserved": 21 * 2**30}
        bounds = {"max_images": 1, "max_vision_patches": 16, "max_prompt_tokens": 12, "dataset_sha256": "f" * 64}
        with ExitStack() as stack:
            for name, value in (("MEMORY_PROBE_ACTIONS", 8), ("MEMORY_PROBE_PROMPT", 12)):
                stack.enter_context(patch.object(train, name, value))
            stack.enter_context(patch.object(capacity, "memory", return_value=readings))
            for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=30 * 2**30))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=29 * 2**30))
            stack.enter_context(patch.object(train_v4, "_autocast", side_effect=lambda _: torch.autocast("cpu", dtype=torch.bfloat16)))
            result = train.measure_update_memory_v17(bundle, encoded(kind="RL", route=ROUTE_FULL), shape_bounds=bounds,
                max_target_tokens=8, max_input_tokens=12, live_engine=True)
            self.assertTrue(result["live_engine"])
            self.assertIsNone(result["kv_cache_gib"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertGreater(result["gradient_norms"]["V"], 0)
            readings["free"] = 1 * 2**30
            with self.assertRaisesRegex(capacity.CapacityHold, "2 GiB headroom"):
                train.measure_update_memory_v17(bundle, encoded(kind="RL", route=ROUTE_FULL), shape_bounds=bounds,
                    max_target_tokens=8, max_input_tokens=12, live_engine=True)


if __name__ == "__main__":
    unittest.main()
