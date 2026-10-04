"""Small CPU contract checks, not a pretrained model evaluation."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

from fine_tuning.evaluate import (generation, inference_fields, sample_seed, score_external,
                                 selected_ids, structure_metrics, terminal_json, verify_cohort_images,
                                 visible_structure)
from mmdl.runtime.contracts import SUBJECTS
from fine_tuning.prepare_sources import prepare_trace
from fine_tuning.evaluate import completed_histogram, resume_results


class EvaluationTests(unittest.TestCase):
    def test_budget_stop_without_generation_fields_serializes_histograms(self):
        rows = [{"status":"COMPLETED", "finish_reason":"stop", "metric_kind":"qa"},
                {"status":"BUDGET_STOP", "correct":None}]
        self.assertEqual('{"stop": 1}', json.dumps(completed_histogram(rows, "finish_reason"), sort_keys=True))
        self.assertEqual('{"qa": 1}', json.dumps(completed_histogram(rows, "metric_kind"), sort_keys=True))

    def test_sampling_gold_independent_and_all_subjects(self):
        rows = [{"id": f"validation_{subject}_{i}", "subject": subject, "answer": "A"}
                for subject in SUBJECTS for i in range(1, 31)]
        first = selected_ids(rows)
        for row in rows:
            row["answer"] = "D"
        self.assertEqual(first, selected_ids(list(reversed(rows))))
        self.assertEqual(60, len(first))
        self.assertEqual(60, len(set(first)))
        self.assertEqual(sample_seed(first[0]), sample_seed(first[0]))
        self.assertNotEqual(sample_seed(first[0], 3408), sample_seed(first[0]))

    def test_input_projection_does_not_read_gold(self):
        class GoldGuard(dict):
            def __getitem__(self, key):
                if key in {"answer", "explanation"}:
                    raise AssertionError("Gold accessed")
                return super().__getitem__(key)
        row = GoldGuard(id="validation_Music_1", question="Read <image 1>.",
                        options="['one','two']", question_type="multiple-choice")
        value = inference_fields(row, "Music")
        self.assertNotIn("answer", value)
        self.assertNotIn("explanation", value)
        self.assertEqual(["one", "two"], value["options"])

    def test_generation_rejects_gold_before_any_model_call(self):
        with self.assertRaisesRegex(ValueError, "non-inference"):
            generation(None, None, {"id": "x", "answer": "B"}, ".", {})

    def test_private_overlay_resolves_public_prompt_before_gpu_generation(self):
        record = {"id": "x", "question_type": "multiple-choice", "question": "Which?",
                  "options": ["one", "two"], "image_paths": []}
        cfg = {"evaluation": {"min_pixels": 262144, "max_pixels": 1048576}}
        with patch("fine_tuning.evaluate.__file__", "/private/version/fine_tuning/evaluate.py"), \
                patch("fine_tuning.evaluate.prepare_protocol_inputs", side_effect=RuntimeError("before GPU")) as prepare:
            with self.assertRaisesRegex(RuntimeError, "before GPU"):
                generation(SimpleNamespace(config=None), None, record, ".", cfg)
        text = prepare.call_args.args[2][0]["content"][-1]["text"]
        self.assertEqual("Question: Which?\nOptions:\nA. one\nB. two\n"
                         "Please select the correct answer from the options above.", text)

    def test_partial_pool_cannot_fake_complete_subject_coverage(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            selected_ids([{"id": "validation_Music_1", "subject": "Music"}])

    def test_trace_uses_released_training_subdirectory_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"sources": {"trace": {
                "repo_id": "maveryn/trace", "revision": "a" * 40,
                "files": [{"path": name} for name in [
                    "data/train/part-1.parquet", "data/validation/part-1.parquet",
                    "data/train/part-0.parquet"]]}}}))
            with patch("fine_tuning.sources.download_exact", return_value=[]) as download:
                prepare_trace(root, inventory, shards=2)
            self.assertEqual(["data/train/part-0.parquet", "data/train/part-1.parquet"],
                             download.call_args.args[2])

    def test_external_answer_is_extracted_before_comparing_gold(self):
        gold = {"metric_kind": "qa", "target": "B", "options": ["zero", "one", "two", "three"]}
        result = score_external("Final answer: C.", gold, "stop")
        self.assertEqual("C", result["extracted_answer"])
        self.assertFalse(result["correct"])
        gold["target"] = "C"
        self.assertEqual("C", score_external("Final answer: C.", gold, "stop")["extracted_answer"])
        self.assertTrue(score_external("Final answer: C.", gold, "stop")["correct"])

    def test_terminal_json_never_searches_reasoning_for_gold(self):
        with self.assertRaises(ValueError):
            terminal_json('Intermediate: {"answer": 50}. Final answer: 75.')
        with self.assertRaises(ValueError):
            terminal_json('{"answer": 50} {"answer": 75}')
        self.assertEqual({"answer": 75}, terminal_json('```json\n{"answer": 75}\n```'))
        gold = {"metric_kind": "trace_qa", "target": '{"answer":50}', "answer_type": "integer"}
        self.assertFalse(score_external('{"answer":75}', gold, "stop")["correct"])
        self.assertTrue(score_external('{"answer":"50.0"}', gold, "stop")["correct"])
        self.assertFalse(score_external('{"answer":true}', gold, "stop")["correct"])
        for raw in ('{"answer":75,"answer":50}', '{"answer":NaN}', '{"answer":Infinity}'):
            with self.assertRaises(ValueError):
                terminal_json(raw)

    def test_structure_geometry_and_semantics_are_separate(self):
        target = {"objects": [{"id": 0, "bbox": [0,0,20,20], "initially_active": True}], "relations": []}
        predicted = {"objects": [{"id": 0, "bbox": [1,1,21,21], "initially_active": True}], "relations": []}
        result = structure_metrics(predicted, target)
        self.assertEqual(1, result["structure_f1"])
        self.assertEqual(1, result["bbox_hits"])
        self.assertEqual({"objects": [{"id":0}]}, visible_structure({"objects":[{"id":0,"glyph_code":8}]}))
        gold = {"metric_kind": "structure", "target": json.dumps(target)}
        self.assertEqual("SCHEMA_INVALID", score_external('{"objects":[{"id":[]}]}', gold, "stop")["parse_status"])

    def test_grounded_answer_does_not_rescue_wrong_states(self):
        target = {"answer": 1, "option": "B", "answer_identifiable": True,
                  "states": [{"step":0,"state":0}]}
        predicted = dict(target, states=[{"step":0,"state":1}])
        result = score_external(json.dumps(predicted), {"metric_kind":"grounded","target":json.dumps(target)}, "stop")
        self.assertTrue(result["correct"])
        self.assertFalse(result["supported_states_exact"])
        predicted = dict(target, answer=True, states=[{"step":0,"state":False}])
        result = score_external(json.dumps(predicted), {"metric_kind":"grounded","target":json.dumps(target)}, "stop")
        self.assertFalse(result["correct"])
        self.assertFalse(result["supported_states_exact"])

    def test_frozen_external_image_bytes_are_checked(self):
        from fine_tuning.audit import sha, write_rows
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"image.png").write_bytes(b"original image")
            write_rows(root/"inputs.jsonl", [{"id":"x", "image_paths":["image.png"]}])
            manifest = {"images":{"image.png":sha(root/"image.png")}}
            verify_cohort_images(root, manifest)
            (root/"image.png").write_bytes(b"different image")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                verify_cohort_images(root, manifest)

    def test_resume_reuses_only_closed_matching_prefix_without_mutating_it(self):
        from fine_tuning.audit import sha, write_rows
        from fine_tuning.modules import BASE_ID, BASE_REVISION
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort, parent = root/"cohort", root/"parent"
            cohort.mkdir()
            parent.mkdir()
            write_rows(cohort/"inputs.jsonl", [{"id":"a"},{"id":"b"}])
            (cohort/"manifest.json").write_text('{"count":2}')
            protocol = root/"protocol.json"
            protocol.write_text('{}')
            rows = [{"id":"a","status":"COMPLETED","raw_response":"Final answer: B."},
                    {"id":"b","status":"BUDGET_STOP","correct":None}]
            write_rows(parent/"predictions.jsonl", rows)
            summary = {"status":"BUDGET_STOP","model_id":BASE_ID,"model_revision":BASE_REVISION,
                "cohort_sha256":sha(cohort/"manifest.json"),"protocol_sha256":sha(protocol),
                "checkpoint":"F0-frozen","checkpoint_metadata_sha256":None,
                "benchmark_derived":True,"predictions_sha256":sha(parent/"predictions.jsonl"),"completed":1}
            (parent/"summary.json").write_text(json.dumps(summary))
            reused, _ = resume_results(parent, cohort, protocol, None, False)
            self.assertEqual({"a":rows[0]}, reused)
            self.assertEqual(summary["predictions_sha256"],sha(parent/"predictions.jsonl"))
            with self.assertRaisesRegex(ValueError, "matching"):
                resume_results(parent, cohort, protocol, None, True)
            (parent/"predictions.jsonl").write_text("\n".join(json.dumps(row) for row in reversed(rows))+"\n")
            with self.assertRaisesRegex(ValueError, "matching"):
                resume_results(parent, cohort, protocol, None, False)


if __name__ == "__main__":
    unittest.main()
