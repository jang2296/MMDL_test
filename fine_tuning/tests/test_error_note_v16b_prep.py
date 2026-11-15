"""CPU checks for preparation ownership, immutable rows, and reporting-only ETA."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fine_tuning.error_note import prep_v16b as prep


def item(pid, *, human=False):
    return {"problem_id": pid, "family_id": pid, "image_paths": [], "image_sha256": [],
        "source": "fixture", "training_allowed": not pid.startswith("monitor"),
        "evaluation": {"question": "Choose a letter", "answer_type": "mcq", "choices": ["one", "two"],
                       "gold": "A", "accept": {}},
        "references": {"human_solution": {"fixture": True}} if human else {}}


class Sampler:
    def __init__(self):
        self.pending, self.registry = [], {}
        self.engine_steps = 0
        self.submitted = 0

    def live_registry(self):
        return self.registry

    def add_version(self, bundle, version):
        self.registry[str(version)] = {"policy_sha256": "a" * 64}

    def submit(self, request):
        self.pending.append(request)
        self.submitted += 1

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        request = self.pending.pop(0)
        self.engine_steps += 1
        generation = {key: request[key] for key in ("policy_sha256", "policy_version", "seed", "request_sha256")}
        generation.update(text="Final answer: A", completion_token_ids=[1, 2, 3])
        return [(request, {"text": generation["text"], "generation": generation, "finish_reason": "stop"})]


class PreparationTest(unittest.TestCase):
    def test_direct_generate_uses_the_sealed_v16b_image_policy(self):
        from fine_tuning.error_note import protocol
        for local in (True, False):
            configs = []
            def factory(model, arm, cfg):
                configs.append(cfg)
                return SimpleNamespace(model=Mock())
            with self.subTest(local=local), patch.object(prep, "select"), patch.object(
                    prep, "generate", return_value={"sha256": "a" * 64}), patch(
                    "fine_tuning.error_note.campaign_v16b._factory", return_value=factory), patch(
                    "fine_tuning.error_note.campaign_v16b.make_sampler"), patch(
                    "fine_tuning.error_note.run_v16._setup_execution"):
                argv = ["generate", "--data-root", "/unused/data", "--output", "/unused/output",
                        "--model-path", "/unused/model", "--allow-model-run",
                        "--precision", "nf4" if local else "bf16", "--engine", "hf" if local else "vllm"]
                if local:
                    argv.append("--local-sequential")
                self.assertEqual(prep.main(argv), 0)
                self.assertEqual(configs[0]["image"], protocol.SMOKE_IMAGE_V16B if local
                                 else protocol.SINGLE_IMAGE_V14)

    def test_direct_generate_checks_native_data_before_loading_a_model(self):
        with patch.object(prep, "select", side_effect=ValueError("bad native data")), patch(
                "fine_tuning.error_note.campaign_v16b.load_f0") as load:
            with self.assertRaisesRegex(ValueError, "bad native data"):
                prep.main(["generate", "--data-root", "/unused/data", "--output", "/unused/output",
                           "--model-path", "/unused/model", "--allow-model-run", "--local-sequential",
                           "--precision", "nf4", "--engine", "hf"])
        load.assert_not_called()

    def test_selection_checks_disjoint_payload_and_smoke_human_coverage(self):
        splits = {"train": [item(f"train{i}") for i in range(3000)],
                  "prep": [item(f"prep{i}", human=i in (50, 70)) for i in range(512)],
                  "monitor": [item(f"monitor{i}") for i in range(200)]}
        with tempfile.TemporaryDirectory() as directory, patch(
                "fine_tuning.error_note.data_v16b.validate", return_value={"status": "checked"}), patch(
                "fine_tuning.error_note.data_v16b.load_items", side_effect=lambda root, split: splits[split]), patch(
                "fine_tuning.error_note.data_v16b._identities", side_effect=lambda entry: {entry["family_id"]}), patch.object(
                prep, "file_sha", return_value="b" * 64):
            root = Path(directory)
            selected = prep.select(root, root / "selected", local_sequential=True)
            self.assertEqual(set(selected["prep_ids"]), {"prep50", "prep70"})
            self.assertEqual(len(selected["monitor_ids"]), 2)
            splits["prep"][0]["family_id"] = "train0"
            with self.assertRaisesRegex(ValueError, "overlapping"):
                prep.select(root, root / "bad")

    def test_completed_collection_reuses_exact_response_without_sampling_again(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64), patch.object(
                prep, "grader", return_value=lambda item, output: ("CORRECT", "A")):
            sampler = Sampler()
            kwargs = dict(data_root=Path(directory), split="prep", out=Path(directory) / "collection",
                stage="v16b-prep", protocol_sha256="b" * 64,
                sampling={"top_p": .8, "top_k": 20, "temperature": .7, "presence_penalty": 1.5}, cap=256)
            original = prep.collect(SimpleNamespace(), sampler, [item("prep0")], **kwargs)
            repeated = prep.collect(SimpleNamespace(), sampler, [item("prep0")], **kwargs)
            self.assertEqual(original, repeated)
            self.assertEqual(sampler.submitted, 1)
            self.assertEqual(original[0]["output"]["generation"]["probability_space"], "processed-truncated-b2")

    def test_collection_refills_before_all_pending_finish(self):
        class Wide(Sampler):
            max_num_seqs = 3

            def __init__(self):
                super().__init__()
                self.depth = []

            def submit(self, request):
                super().submit(request)
                self.depth.append(len(self.pending))

        with tempfile.TemporaryDirectory() as directory, patch(
                "fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64), patch.object(
                prep, "grader", return_value=lambda item, output: ("CORRECT", "A")):
            sampler = Wide()
            items = [item(f"prep{i}") for i in range(7)]
            records = prep.collect(SimpleNamespace(), sampler, items, data_root=Path(directory), split="prep",
                out=Path(directory) / "collection", stage="v16b-prep", protocol_sha256="b" * 64,
                sampling={"top_p": .8, "top_k": 20}, cap=256)
        self.assertEqual([row["problem_id"] for row in records], [x["problem_id"] for x in items])
        self.assertEqual(sampler.submitted, 7)
        # Never more than max_num_seqs in flight, and the queue stays full (a drain-all would restart at 1).
        self.assertEqual(sampler.depth, [1, 2, 3, 3, 3, 3, 3])

    def test_started_request_without_result_is_never_reissued(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64), patch.object(
                prep, "grader", return_value=lambda item, output: ("CORRECT", "A")):
            root, sampler = Path(directory), Sampler()
            request_id = prep.digest({"problem_id": "prep0", "stage": "v16b-prep"})
            (root / "requests.jsonl").write_text(json.dumps({"request_id": request_id, "status": "STARTED"}) + "\n")
            with self.assertRaisesRegex(ValueError, "recovery required"):
                prep.collect(SimpleNamespace(), sampler, [item("prep0")], data_root=root, split="prep", out=root,
                    stage="v16b-prep", protocol_sha256="b" * 64, sampling={"top_p": .8, "top_k": 20}, cap=256)
            self.assertEqual(sampler.submitted, 0)

    def test_sealed_artifact_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            prep.seal(path, {"rows": [1]})
            path.write_text(json.dumps({"rows": [2], "sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "changed"):
                prep.read_sealed(path)

    def test_eta_is_only_a_report_and_rejects_unmeasured_numbers(self):
        schedule = {"blocks": 188, "lr_steps": 205}
        original = copy.deepcopy(schedule)
        report = prep.eta_report([10, 20, 30, 40], done=4, total=191)
        self.assertEqual(report["remaining_seconds"], 25 * 187)
        self.assertFalse(report["changes_plan"])
        self.assertEqual(schedule, original)
        for values in ([float("nan")], [-1], [True], []):
            with self.assertRaises(ValueError):
                prep.eta_report(values, done=4, total=191)


    def test_preflight_covers_every_parent_and_preserves_explicit_observation_skips(self):
        entries = [item("train0"), item("train1")]
        receipts = {"train0": {"eligible": False, "reason": "NO_NATIVE_OBSERVATION"},
                    "train1": {"eligible": True, "reason": None}}
        with patch("fine_tuning.error_note.v16b_rows.problem_contract_v16b", return_value={"fact_catalog": [], "rules": {}}), patch(
                "fine_tuning.error_note.v16b_rows.observation_eligibility_v16b", side_effect=lambda entry: receipts[entry["problem_id"]]), patch(
                "fine_tuning.error_note.v16b_rows.build_obs_prompt_v16b", return_value="Observe.") as prompt:
            result = prep.preflight_items(entries)
        self.assertEqual(result["first_solution_count"], 2)
        self.assertEqual(result["observation_eligible"], 1)
        self.assertEqual(result["items"][0]["observation_eligibility"]["reason"], "NO_NATIVE_OBSERVATION")
        prompt.assert_called_once_with(entries[1])
        with patch("fine_tuning.error_note.v16b_rows.problem_contract_v16b", side_effect=ValueError("CORRUPT_NATIVE_SOURCE")):
            with self.assertRaisesRegex(ValueError, "CORRUPT_NATIVE_SOURCE"):
                prep.preflight_items(entries)

    def test_observation_collection_logs_supported_skips_without_submitting_or_grading(self):
        receipt = {"problem_id": "prep0", "eligible": False, "reason": "UNSUPPORTED_NATIVE_SHAPE"}
        with tempfile.TemporaryDirectory() as directory, patch(
                "fine_tuning.error_note.rl.policy_sha256", return_value="a" * 64), patch.object(
                prep, "grader", return_value=lambda *args: self.fail("a skipped OBS must not be graded")), patch(
                "fine_tuning.error_note.v16b_rows.observation_eligibility_v16b", return_value=receipt):
            root, sampler = Path(directory), Sampler()
            records = prep.collect(SimpleNamespace(), sampler, [item("prep0")], data_root=root, split="prep", out=root,
                stage="v16b-prep-obs", protocol_sha256="b" * 64, sampling={"top_p": 1., "top_k": 0}, cap=256,
                observation=True)
            self.assertEqual(records, [])
            self.assertEqual(sampler.submitted, 0)
            self.assertEqual(prep.read_sealed(root / "observation-skips.json")["items"], [receipt])


if __name__ == "__main__":
    unittest.main()
