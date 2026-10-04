"""Synthetic CPU-only provenance/data-contract checks; no downloads or model calls."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fine_tuning import audit, sources
from fine_tuning.modules import BASE_REVISION


def png(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(stream, format="PNG")
    return stream.getvalue()


def trace_row(sid, task="task0", scene=None):
    result = {"instance_id": sid, "task": task, "domain": "synthetic-test",
              "answer_gt": json.dumps({"type": "integer", "value": 2}),
              "images": [{"bytes": png(), "path": None}], "prompt_answer": "<image>\nQuestion: Count the tokens.",
              "annotation_gt": json.dumps({"type": "answer_box"}),
              "reward_contract": json.dumps({"answer_type": "integer"})}
    if scene is not None:
        result["scene_grammar"] = scene
    return result


def parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class DataAuditTests(unittest.TestCase):
    def test_lineage_cycles_unknown_and_taint_rejected(self):
        source = {"benchmark_derived": False, "source_known": True, "rights_approved": True, "parents": []}
        row = {"id": "unit", "benchmark_derived": False, "lineage": ["a"]}
        registry = {"a": copy.deepcopy(source), "b": copy.deepcopy(source)}
        audit.check_lineage(row, registry)
        registry["a"]["parents"] = ["b"]
        registry["b"]["parents"] = ["a"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            audit.check_lineage(row, registry)
        registry["b"]["parents"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            audit.check_lineage(row, registry)
        registry["b"]["parents"] = []
        registry["b"]["benchmark_derived"] = True
        with self.assertRaisesRegex(ValueError, "tainted"):
            audit.check_lineage(row, registry)

    def test_short_question_wrapper_fingerprints_cannot_bypass(self):
        question = "Which token is larger?"
        expected = audit.text_hash(question)
        for wrapped in ("<image>\nQuestion: "+question, "Question： "+question,
                        "Question: "+question+"\nOptions:\nA. X\nB. Y"):
            self.assertEqual(audit.text_hash(wrapped), expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            Image.new("RGB", (32, 32)).save(path/"image.png")
            row = {"user_text": "<image>\nQuestion: "+question, "image_paths": ["image.png"]}
            self.assertIn("EXACT_BENCHMARK_TEXT", audit.contamination_candidates(
                row, path, {"text_sha256": [expected]}))

    def test_placeholder_count_strip_preserves_original(self):
        row = {"id": "unit", "user_text": "<image>\nQuestion: Test", "target_text": "A", "image_paths": ["a.png"]}
        prepared = audit.model_input_record(row)
        self.assertEqual(prepared["user_text"], "Question: Test")
        self.assertEqual(prepared["original_user_text"], row["user_text"])
        self.assertEqual(prepared["original_user_text_sha256"], hashlib.sha256(row["user_text"].encode()).hexdigest())
        bad = dict(row, user_text="<image> <image> Test")
        with self.assertRaisesRegex(ValueError, "placeholder count"):
            audit.model_input_record(bad)
        bad = dict(prepared, user_text="Different")
        with self.assertRaisesRegex(ValueError, "preserved original"):
            audit.model_input_record(bad)

    def test_image_paths_no_traversal_or_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (10, 10)).save(root/"valid.png")
            for name in ("../outside.png", str(root/"valid.png")):
                with self.assertRaises(ValueError):
                    audit.image_path(root, name)
            (root/"alias.png").symlink_to(root/"valid.png")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                audit.image_path(root, "alias.png")

    def test_trace_hashed_filenames_collision_safe_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [trace_row("../../escape"), trace_row("a:b"), trace_row("a_b")]
            path = parquet(root/"source.parquet", rows)
            records = sources.trace_records([path], root/"converted", train_limit=10)
            names = [r["image_paths"][0] for r in records]
            self.assertEqual(len(set(names)), 3)
            self.assertTrue(all(len(Path(name).stem) == 66 for name in names))
            self.assertFalse((root/"escape.png").exists())
            self.assertTrue(all(r["original_user_text"].startswith("<image>") for r in records))
            with self.assertRaises(FileExistsError):
                sources.trace_records([path], root/"converted")

    def test_trace_typed_values_and_prompt_mismatch_fail(self):
        for kind, value in (("integer", True), ("integer", "2"), ("number", float("nan")),
                            ("boolean", 1), ("option_letter", "ZZ"), ("string", "")):
            with self.assertRaises(ValueError):
                sources.typed_trace_answer({"type": kind, "value": value})
        for kind, value in (("integer", 2), ("number", 2.5), ("boolean", True), ("option_letter", "C")):
            self.assertEqual(sources.typed_trace_answer({"type": kind, "value": value})["value"], value)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = trace_row("bad-placeholder")
            row["prompt_answer"] = "<image> <image> Test"
            path = parquet(root/"source.parquet", [row])
            with self.assertRaisesRegex(ValueError, "placeholder count"):
                sources.trace_records([path], root/"converted")

    def test_actual_scene_grammar_groups_cross_task_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = parquet(root/"source.parquet", [trace_row("a", "task-a", "same-grammar"),
                                                     trace_row("b", "task-b", "same-grammar")])
            rows = sources.trace_records([path], root/"converted")
            self.assertEqual(rows[0]["family_id"], rows[1]["family_id"])
            self.assertEqual(rows[0]["split"], rows[1]["split"])
            conversion = json.loads((root/"converted/conversion.json").read_text())
            self.assertIn("scene_grammar", conversion["actual_grouping_dimensions"])

    def coverage(self, path, count, dataset="MMMU/MMMU", config="Accounting"):
        return {"files": [{"path": str(path), "records": count, "dataset_id": dataset,
                            "configuration": config, "split": "validation", "revision": "a"*40,
                            "sha256": audit.sha(path)}], "record_count": count}

    def test_benchmark_actual_mmmu_and_pro_schemas_exact_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number, (dataset, row) in enumerate((
                    ("MMMU/MMMU", {"id": "unit", "question": "Short question?", "options": "['x','y']",
                                    "image_1": {"bytes": png(), "path": None}, "image_2": None}),
                    ("MMMU/MMMU_Pro", {"id": "unit-pro", "question": "Other question?", "options": ["x", "y"],
                                        "images": [{"bytes": png("blue"), "path": None}]}))):
                path = parquet(root/f"{number}.parquet", [row])
                result = sources.benchmark_index([path], self.coverage(path, 1, dataset), root/f"{number}.json")
                self.assertEqual(result["status"], "COMPLETE")
                index = json.loads((root/f"{number}.json").read_text())
                self.assertTrue(index["coverage_verified"])
                self.assertEqual(index["image_count"], 1)
                with self.assertRaisesRegex(ValueError, "record coverage"):
                    sources.benchmark_index([path], self.coverage(path, 2, dataset), root/f"bad{number}.json")
                with self.assertRaisesRegex(ValueError, "file coverage"):
                    sources.benchmark_index([], self.coverage(path, 1, dataset), root/f"missing{number}.json")

    def test_missing_benchmark_image_bytes_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"question": "Test?", "options": ["A"], "image_1": {"path": "missing.png", "bytes": None}}
            path = parquet(root/"source.parquet", [row])
            with self.assertRaisesRegex(ValueError, "bytes missing"):
                sources.benchmark_index([path], self.coverage(path, 1), root/"index.json")
            self.assertFalse((root/"index.json").exists())

    def test_ambiguous_structure_safe_but_untrusted_letter_qa_rejected(self):
        row = {"id": "test", "root_id": "test-root", "answer_identifiable": False,
               "supervised_target_identifiable": True, "target_text": "A", "view": "structure"}
        audit.check_target_identifiability(row, "/missing")
        row["view"] = "qa"
        with self.assertRaisesRegex(ValueError, "trusted printed"):
            audit.check_target_identifiability(row, "/missing")
        row["target_text"] = "IN-SUFFICIENT text buried in target"
        with self.assertRaises(ValueError):
            audit.check_target_identifiability(row, "/missing")

    def test_approve_requires_actual_per_record_input_receipt_and_binds_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32), "green").save(root/"image.png")
            row = {"id": "unit", "root_id": "root-unit", "family_id": "family-unit", "split": "train",
                   "benchmark_derived": False, "lineage": ["clean"], "source_kind": "INDEPENDENT_SYNTHETIC",
                   "quality_status": "DERIVED_VERIFIED", "supervised_target_identifiable": True,
                   "answer_identifiable": True, "user_text": "<image>\nIndependent synthetic task",
                   "target_text": "A", "image_paths": ["image.png"]}
            records = root/"records.jsonl"
            audit.write_rows(records, [row])
            registry = {"clean": {"benchmark_derived": False, "rights_approved": True, "source_known": True, "parents": []}}
            index_path = root/"index.json"
            coverage = self.coverage(root/"image.png", 1)
            audit.dump(index_path, {"status": "COMPLETE", "coverage": coverage, "coverage_verified": True,
                "per_file_record_counts": {str(root/"image.png"): 1}, "record_count": 1,
                "source_files": {str(root/"image.png"): audit.sha(root/"image.png")}, "image_count": 1})
            with self.assertRaisesRegex(ValueError, "per-record"):
                audit.approve(records, root, registry, index_path, root/"unapproved")
            # A contract fixture is deliberately not an actual processor run.
            receipt = {"status": "PASS", "kind": "ACTUAL_PROCESSOR_PREFLIGHT", "producer_sha256": audit.sha(audit.__file__),
                       "processor_revision": BASE_REVISION, "records_sha256": audit.sha(records),
                       "processor_class": "Qwen3VLProcessor", "truncation": False, "chat_template_sha256": "fixture",
                       "sequence_budget": 4096, "episode_pixels": 1048576, "records": {"unit": {
                           "status": "PASS", "input_sha256": "wrong", "assistant_tokens": 1,
                           "tensor_sha256": {"input_ids": "fixture"}}}}
            receipt_path = root/"receipt.json"
            audit.dump(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "per-record"):
                audit.approve(records, root, registry, index_path, root/"still-unapproved", receipt_path)
            with self.assertRaisesRegex(ValueError, "Actual Qwen3VLProcessor"):
                audit.audit_model_inputs(records, root, object(), root/"fake-receipt.json")
            receipt["records"]["unit"].update(
                input_sha256=audit.input_fingerprint(row, root), sequence_tokens=5,
                image_grid_thw=[[1, 2, 2]], tensor_sha256={key: "fixture" for key in (
                    "input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")})
            good_receipt_path = root/"contract-fixture.json"
            audit.dump(good_receipt_path, receipt)
            manifest = audit.approve(records, root, registry, index_path, root/"approved-contract-test", good_receipt_path)
            approved = audit.read_rows(root/"approved-contract-test/records.jsonl")[0]
            self.assertEqual(manifest["model_input_audit"]["record_count"], 1)
            self.assertNotIn("<image>", approved["user_text"])
            self.assertEqual(approved["original_user_text"], row["user_text"])
            self.assertEqual(approved["processor_receipt_sha256"], audit.sha(good_receipt_path))

    def test_strict_ambiguous_qa_helper_binds_actual_printed_choices(self):
        from fine_tuning import nvw
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = next(e for i in range(200) if not (e := nvw.build_episode("finite_state", i))["quality"]["answer_identifiable"])
            image, observed = nvw.render(episode)
            episode["observed_structure"] = observed
            image = image.resize(nvw.OUTPUT_SIZE, Image.Resampling.LANCZOS)
            episode["image"] = "query.png"
            image.save(root/"query.png")
            episode["image_sha256"] = audit.sha(root/"query.png")
            audit.write_rows(root/"canonical_private.jsonl", [episode])
            row = nvw.view_record(episode, "query.png", "qa")
            audit.check_target_identifiability(row, root)
            row["target_text"] = "Z"
            with self.assertRaisesRegex(ValueError, "trusted canonical"):
                audit.check_target_identifiability(row, root)


if __name__ == "__main__":
    unittest.main()
