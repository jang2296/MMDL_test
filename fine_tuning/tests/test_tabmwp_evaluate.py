"""Self-authored CPU scope/leakage/resume/scoring fixtures; no model or source corpus."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning import tabmwp, tabmwp_evaluate as evaluation
from mmdl.runtime.artifacts import atomic_text, digest, read_json, sha256_file, write_json


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.gold = self.root / "cohort", self.root / "private-gold"
        self.data.mkdir()
        self.gold.mkdir()
        stream = io.BytesIO()
        Image.new("RGB", (3, 2), (70, 80, 90)).save(stream, format="PNG")
        self.png = stream.getvalue()
        (self.data / "images").mkdir()
        (self.data / "images/fixture.png").write_bytes(self.png)
        image = {"path": "images/fixture.png", "sha256": sha256_file(self.data / "images/fixture.png"),
                 "original_size": [3, 2]}
        self.samples, gold = [], []
        for mode, count in evaluation.COUNTS.items():
            for index in range(count):
                native_id = f"tabmwp-{index}" if mode != "RETENTION" else f"retained-{index}"
                sid = f"{native_id}__{mode}"
                sample = {"sample_id": sid, "native_id": native_id, "mode": mode,
                          "subject": "Self-authored", "question_type": "multiple-choice",
                          "question": f"Read fixture {index}.", "options": ["one", "two"],
                          "user_text": f"Question: Read fixture {index}.\nOptions:\nA. one\nB. two\nPlease select the correct answer from the options above.",
                          "images": [image], "family_id": "fixture-family-" + native_id,
                          "root_id": "fixture-root-" + native_id}
                if mode == "FACTS":
                    sample.update(user_text=evaluation.FACTS_PROMPT, question="", options=[], question_type="open")
                    target = {"sample_id": sid, "metric_kind": "table_facts", "target": {
                        "title": "Fixture", "headers": ["H"], "rows": [["cell"]]}}
                elif mode == "QA":
                    target = {"sample_id": sid, "metric_kind": "tabmwp_qa", "answer": "B", "native_answer": "two"}
                else:
                    kinds = ["qa"] * 12 + ["trace_qa"] * 24 + ["structure"] * 12 + ["grounded"] * 12
                    kind = kinds[index]
                    value = {"trace_qa": '{"answer":2}', "structure": '{"objects":[{"id":0}]}',
                             "grounded": '{"answer":2,"option":"B","answer_identifiable":true,"states":[]}'}
                    target = {"sample_id": sid, "metric_kind": kind, "target": value.get(kind, "B"),
                              "options": ["one", "two"], "answer_type": "number"}
                self.samples.append(sample)
                gold.append(target)
        self.targets = {row["sample_id"]: row for row in gold}
        atomic_text(self.data / "samples.jsonl", "".join(json.dumps(row) + "\n" for row in self.samples))
        atomic_text(self.gold / "gold.jsonl", "".join(json.dumps(row) + "\n" for row in gold))
        manifest = {"status": "COHORT_FROZEN", "protocol_id": evaluation.PROTOCOL_ID, "count": 132,
                    "counts": evaluation.COUNTS, "tabmwp_preparation_hash": "synthetic-preparation",
                    "qa_ids": [row["native_id"] for row in self.samples if row["mode"] == "QA"],
                    "facts_ids": [row["native_id"] for row in self.samples if row["mode"] == "FACTS"],
                    "retention_ids": [row["native_id"] for row in self.samples if row["mode"] == "RETENTION"],
                    "sample_hashes": {row["sample_id"]: digest(row) for row in self.samples},
                    "samples_sha256": sha256_file(self.data / "samples.jsonl"),
                    "gold_sha256": sha256_file(self.gold / "gold.jsonl"),
                    "facts_prompt_sha256": digest(evaluation.FACTS_PROMPT),
                    "training_allowed": False, "public_upload_allowed": False}
        manifest["cohort_hash"] = digest(manifest)
        write_json(self.data / "manifest.json", manifest)
        write_json(self.gold / "binding.json", {"cohort_hash": manifest["cohort_hash"], "gold_sha256": manifest["gold_sha256"]})
        self.deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    def tearDown(self):
        self.temp.cleanup()

    def fake_infer(self, candidate="F0", fail_after=None, deadline=None):
        targets = self.targets
        captured = []
        class Backend:
            def describe(self):
                return {"run_kind": "TEST_ONLY", "slots": 4}

            def generate_stream(self, requests):
                for index, request in enumerate(requests):
                    captured.append(request)
                    if fail_after is not None and index == fail_after:
                        raise RuntimeError("synthetic cutoff")
                    target = targets[request["original_sample_id"]]
                    kind = target["metric_kind"]
                    raw = (json.dumps(target["target"]) if kind == "table_facts"
                           else "Final answer: B." if kind in {"qa", "tabmwp_qa"} else target["target"])
                    yield {"sample_id": request["sample_id"], "raw_response": raw,
                           "input_tokens": 3, "generated_tokens": 2, "finish_reason": "stop"}
        def factory(model, base, arm, config, deadline):
            self.assertEqual(arm, "F0" if candidate == "F0" else "FM")
            self.assertEqual(config["generation"]["max_new_tokens"], 8192)
            self.assertEqual(config["execution"]["batch_size"], 4)
            return Backend()
        with patch.object(evaluation.recovery, "checkout"), patch.object(
                evaluation.recovery, "model_identity", side_effect=lambda path, arm: {"arm": arm}), patch.object(
                evaluation, "candidate_identity", return_value={"arm": "F0" if candidate == "F0" else "FM"}):
            evaluation.infer(self.data, self.root / "model", self.root / "base", self.root / candidate,
                             candidate, deadline or self.deadline, "a" * 40, backend_factory=factory)
        return self.root / candidate, captured

    def test_fixed_scope_seed_candidate_prompt_parity_and_no_gold(self):
        _, samples = evaluation.load_cohort(self.data)
        f0, a, b = [evaluation.make_requests(samples, candidate) for candidate in evaluation.CANDIDATES]
        self.assertEqual(Counter(row["mode"] for row in f0), {"QA": 48, "FACTS": 24, "RETENTION": 60})
        self.assertEqual(len(f0), 132)
        for requests in (a, b):
            for left, right in zip(f0, requests, strict=True):
                self.assertEqual(left["messages"], right["messages"])
                self.assertEqual(left["seed"], right["seed"])
                self.assertEqual(left["images"], right["images"])
                self.assertFalse({"answer", "target", "solution", "gold"} & set(right))
                self.assertEqual(right["max_new_tokens"], 8192)
        (self.gold / "gold.jsonl").unlink()
        output, _ = self.fake_infer()
        self.assertEqual(read_json(output / "status.json")["completed"], 132)

    def test_missing_denominator_and_test_only_expansion(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic cutoff"):
            self.fake_infer(fail_after=3)
        summary = evaluation.score(self.data, self.gold, [self.root / "F0"], self.root / "partial-score")
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(len(summary["missing_request_ids"]), 393)
        self.assertIsNone(summary["groups"]["F0"]["QA"]["accuracy"])
        self.assertIsNone(summary["groups"]["A"]["FACTS"]["cell_recall"])
        self.assertIsNone(summary["expansion_gate"]["A"]["expansion_eligible"])
        for candidate in evaluation.CANDIDATES:
            self.fake_infer(candidate)
        summary = evaluation.score(self.data, self.gold, [self.root / name for name in evaluation.CANDIDATES], self.root / "full-score")
        self.assertEqual(summary["completed"], 396)
        self.assertEqual(summary["status"], "COMPLETED")
        self.assertEqual(summary["groups"]["A"]["retention_metrics"]["qa"]["planned"], 12)
        self.assertEqual(summary["groups"]["A"]["retention_metrics"]["trace_qa"]["planned"], 24)
        self.assertEqual(summary["groups"]["A"]["retention_primary"], 1)
        self.assertIsNone(summary["expansion_gate"]["A"]["expansion_eligible"])

    def test_resume_preserves_bytes_and_refuses_changed_deadline_response_or_image(self):
        with self.assertRaises(RuntimeError):
            self.fake_infer(fail_after=1)
        path = next((self.root / "F0/responses").glob("*.json"))
        saved = path.read_bytes()
        self.fake_infer()
        self.assertEqual(path.read_bytes(), saved)
        with self.assertRaisesRegex(ValueError, "Resume changed"):
            self.fake_infer(deadline=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat())
        row = read_json(path)
        row["raw_response"] = "tampered"
        write_json(path, row)
        with self.assertRaisesRegex(ValueError, "altered"):
            self.fake_infer()
        (self.data / "images/fixture.png").write_bytes(b"different image")
        with self.assertRaisesRegex(ValueError, "image hash"):
            self.fake_infer("A")

    def test_whole_json_mcq_native_option_text_not_gold_and_no_reasoning_rescue(self):
        sample = self.samples[0]
        target = {"answer": "B"}
        for raw in ('{"answer":"two"}', '{"facts":{},"answer":"B"}', "Final answer: B."):
            self.assertTrue(evaluation.score_qa(raw, sample, target, "stop")["correct"])
        for raw in ('{"answer":"one","answer":"two"}', '{"answer":true}', '{"answer":"two"} {"answer":"one"}', '{"answer":"two","option":"A"}'):
            self.assertEqual(evaluation.score_qa(raw, sample, target, "stop")["parse_status"], "NO_PARSE")
        raw = 'Reasoning: {"answer":"two"}. Final answer: A.'
        self.assertFalse(evaluation.score_qa(raw, sample, target, "stop")["correct"])
        left = evaluation.score_qa('{"answer":"two"}', sample, {"answer": "B"}, "stop")
        right = evaluation.score_qa('{"answer":"two"}', sample, {"answer": "A"}, "stop")
        self.assertEqual(left["parsed_answer"], right["parsed_answer"])
        self.assertFalse(right["correct"])
        # A literal option "A" is native option text, not automatically option label A.
        special = dict(sample, options=["other", "A"])
        self.assertTrue(evaluation.score_qa('{"answer":"A"}', special, target, "stop")["correct"])
        opened = dict(sample, question_type="open", options=[], question="Read the number.")
        self.assertTrue(evaluation.score_qa('{"answer":"7"}', opened, {"answer": "7"}, "stop")["correct"])

    def test_ordered_facts_fields_are_independent_from_json_validity_and_no_numeric_coercion(self):
        target = {"title": None, "headers": ["Z", "A"], "rows": [["1.00", "x"], ["2", "y"]]}
        exact = evaluation.score_facts(json.dumps(target), target)
        self.assertTrue(exact["facts_exact"])
        wrong = dict(target, rows=[[1.0, "x"], ["2", "y"]])
        result = evaluation.score_facts(json.dumps(wrong), target)
        self.assertTrue(result["json_valid"])
        self.assertFalse(result["schema_valid"])
        self.assertFalse(result["facts_exact"])
        self.assertEqual(result["cell_hits"], 3)
        self.assertEqual(result["cell_denominator"], 4)
        self.assertFalse(evaluation.score_facts(json.dumps(dict(target, rows=list(reversed(target["rows"])))), target)["rows_exact"])
        self.assertFalse(evaluation.score_facts(json.dumps(dict(target, headers=["A", "Z"])), target)["headers_exact"])
        nested = evaluation.score_facts(json.dumps({"facts": target, "answer": "ignored"}), target)
        self.assertTrue(nested["facts_exact"])
        self.assertFalse(nested["schema_valid"])
        for raw in ('prefix ' + json.dumps(target), '{"rows":[],"rows":[["1"]]}', json.dumps(target) + json.dumps(target)):
            self.assertFalse(evaluation.score_facts(raw, target)["json_valid"])

    def test_changed_gold_or_input_extra_fields_or_shortened_scope_fail_closed(self):
        output, _ = self.fake_infer()
        (self.gold / "gold.jsonl").write_text((self.gold / "gold.jsonl").read_text().replace('"answer": "B"', '"answer": "A"'))
        with self.assertRaisesRegex(ValueError, "Gold differs"):
            evaluation.score(self.data, self.gold, [output], self.root / "bad-score")
        manifest = read_json(self.data / "manifest.json")
        sample = self.samples[0]
        sample["answer"] = "forbidden"
        atomic_text(self.data / "samples.jsonl", "".join(json.dumps(row) + "\n" for row in self.samples))
        manifest["samples_sha256"] = sha256_file(self.data / "samples.jsonl")
        manifest["sample_hashes"] = {row["sample_id"]: digest(row) for row in self.samples}
        manifest["cohort_hash"] = digest({k: v for k, v in manifest.items() if k != "cohort_hash"})
        write_json(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "gold-bearing"):
            evaluation.load_cohort(self.data)
        manifest["count"] = 131
        manifest["cohort_hash"] = digest({k: v for k, v in manifest.items() if k != "cohort_hash"})
        write_json(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "count differs"):
            evaluation.load_cohort(self.data)

    def test_candidate_identity_rejects_old_g6_and_wrong_or_incomplete_bound_fm(self):
        sources = {name: sha256_file(Path(evaluation.__file__).with_name(name)) for name in ("train.py", "modules.py")}
        contract = {"min_pixels": 1003520, "max_pixels": 4014080, "episode_pixels": 8028160, "sequence_budget": 8192}
        manifest = {"status": "APPROVED", "training_allowed": True, "benchmark_derived": False,
                    "record_count": 2000, "tabmwp_condition": "A", "tabmwp_preparation_hash": "synthetic-preparation",
                    "sources": {tabmwp.SOURCE_ID: {"rights_approved": True, "source_known": True, "benchmark_derived": False}},
                    "source_sha256": sources, "records_sha256": "c" * 64,
                    "consumption_order_sha256": "d" * 64,
                    "model_input_audit": {"status": "PASS", "record_count": 2000,
                                          "sequence_budget": 8192, "episode_pixels": 8028160}}
        mp, cp = self.root / "train-manifest.json", self.root / "checkpoint-metadata.json"
        model = self.root / "export"
        model.mkdir()
        write_json(mp, manifest)
        metadata = {"source_sha256": sources, "manifest_sha256": sha256_file(mp), "records_sha256": "c" * 64,
                    "base_id": evaluation.recovery.MODEL_ID, "base_revision": evaluation.recovery.MODEL_REVISION,
                    "arm": "FM", "seed": 3407, "termination": "EPOCH_COMPLETE", "epoch_exposure_complete": True,
                    "epochs": 1, "planned_records": 2000, "exposure": {"records": 2000, "updates": 125},
                    "consumed_order_sha256": "d" * 64, "record_order_sha256": "d" * 64,
                    "budgets": dict(contract, accumulation=16, max_steps=0), "base_precision": "BF16",
                    "trainable_master_precision": "FP32", "training_input_contract": contract}
        def exported():
            write_json(cp, metadata)
            write_json(model / "export_manifest.json", {"checkpoint_files": {"metadata.json": sha256_file(cp)},
                       "source_sha256": {"modules.py": sources["modules.py"],
                       "checkpoints.py": sha256_file(evaluation.recovery.ROOT / "experiments/input_recovery/checkpoints.py")}})
        exported()
        with patch.object(evaluation.recovery, "model_identity", return_value={"arm": "FM"}):
            self.assertEqual(evaluation.candidate_identity(model, "A", {"tabmwp_preparation_hash": "synthetic-preparation"}, mp, sha256_file(mp), cp)["candidate"], "A")
            for change in ({"planned_records": 10000}, {"seed": 42}, {"epoch_exposure_complete": False},
                           {"exposure": {"records": 2000, "updates": 124}}, {"source_sha256": {}},
                           {"consumed_order_sha256": "e" * 64}, {"record_order_sha256": "e" * 64}):
                saved = copy.deepcopy(metadata)
                metadata.update(change)
                exported()
                with self.assertRaisesRegex(ValueError, "Legacy/incomplete"):
                    evaluation.candidate_identity(model, "A", {"tabmwp_preparation_hash": "synthetic-preparation"}, mp, sha256_file(mp), cp)
                metadata.clear()
                metadata.update(saved)
            exported()
            with self.assertRaisesRegex(ValueError, "wrong-condition"):
                evaluation.candidate_identity(model, "B", {"tabmwp_preparation_hash": "synthetic-preparation"}, mp, sha256_file(mp), cp)
            with self.assertRaisesRegex(ValueError, "accepted frozen hash"):
                evaluation.candidate_identity(model, "A", {"tabmwp_preparation_hash": "synthetic-preparation"}, mp, "0" * 64, cp)
        with self.assertRaisesRegex(ValueError, "four continuous"):
            evaluation.infer(self.data, model, model, self.root / "invalid", "F0", self.deadline, "a" * 40, slots=8)

    def test_prepare_deterministic_exclusion_gold_separation_and_no_overwrite(self):
        source = self.root / "native-source"
        prepared, images, retention, training = [source / "prepared", source / "selected-images",
                                               self.root / "retention", self.root / "training"]
        for path in (prepared / "A", images / "tables", retention / "images", training):
            path.mkdir(parents=True)
        groups, paths, files, entries, native = {}, {}, {}, {}, {}
        proof = {"bytes": len(self.png), "sha256": hashlib_sha(self.png), "git_blob_sha1": tabmwp_images_hash(self.png),
                 "size": [3, 2], "pixels_sha256": evaluation.tabmwp_images.verify_png(self.png, {
                     "type": "blob", "mode": "100644", "size": len(self.png), "sha": tabmwp_images_hash(self.png)})["pixels_sha256"]}
        offset = 0
        for role, count in (("train", 1600), ("dev", 200), ("sealed", 200)):
            rows = []
            for i in range(offset, offset + count):
                name = f"tables/{i}.png"
                row = {"id": f"tabmwp-{i}", "source_native_id": str(i), "family_id": f"table-family-{i}",
                       "root_id": f"tabmwp-{i}", "user_text": f"Question: Read independent item {i}.", "image_paths": [name]}
                rows.append(row)
                paths[name] = {"id": row["id"], "role": role}
                files[name] = dict(proof, **paths[name])
                if role == "dev":
                    (images / name).write_bytes(self.png)
                    entries["data/tabmwp/" + name] = {"type": "blob", "mode": "100644", "size": len(self.png), "sha": tabmwp_images_hash(self.png)}
                    native[str(i)] = {"question": f"Read independent item {i}.", "answer": "native fixture answer",
                                     "choices": None, "ques_type": "free_text", "table_title": None,
                                     "table_for_pd": {"H": [f"cell-{i}"]}, "table": f"H\ncell-{i}", "row_num": 2, "column_num": 1}
            groups[role] = rows
            offset += count
        # A development item cannot borrow a train family even with a distinct ID/prompt.
        groups["dev"][0]["family_id"] = groups["train"][0]["family_id"]
        for role, rows in groups.items():
            atomic_text(prepared / f"A/{role}.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
        for name in ("tree.json", "commit.json"):
            write_json(images / name, {})
        receipt = {"status": "COMPLETE", "preparation_hash": "synthetic-preparation", "revision": tabmwp.REVISION,
                   "tree_sha1": evaluation.tabmwp_images.TREE_SHA, "selection_sha256": digest(paths), "files": files,
                   "expected_count": 2000, "verified_count": 2000, "errors": {}, "changed_sources": [],
                   "tree_metadata_sha256": sha256_file(images / "tree.json"), "commit_metadata_sha256": sha256_file(images / "commit.json")}
        write_json(images / "receipt_01.json", receipt)
        retained, retained_gold = [], []
        (retention / "images/fixture.png").write_bytes(self.png)
        for sample in self.samples:
            if sample["mode"] != "RETENTION":
                continue
            retained.append({"id": sample["native_id"], "subject": sample["subject"], "question_type": sample["question_type"],
                             "question": sample["question"], "options": sample["options"], "user_text": sample["user_text"],
                             "image_paths": ["images/fixture.png"]})
            retained_gold.append(dict(self.targets[sample["sample_id"]], id=sample["native_id"],
                                      family_id=sample["family_id"], root_id=sample["root_id"]))
        for name, rows in (("inputs.jsonl", retained), ("gold.jsonl", retained_gold)):
            atomic_text(retention / name, "".join(json.dumps(row) + "\n" for row in rows))
        write_json(retention / "manifest.json", {"status": "COHORT_FROZEN", "count": 60,
                   "inputs_sha256": sha256_file(retention / "inputs.jsonl"), "gold_sha256": sha256_file(retention / "gold.jsonl"),
                   "images": {"images/fixture.png": hashlib_sha(self.png)}})
        records_path = training / "records.jsonl"
        records_path.write_text("[]\n")
        evidence = {"path": str(records_path), "sha256": sha256_file(records_path), "record_count": 2000}
        with patch.object(evaluation.tabmwp_images, "selection", return_value=(paths, {}, "synthetic-preparation")), \
                patch.object(evaluation.tabmwp_images, "verify_tree", return_value=entries), \
                patch.object(evaluation.tabmwp, "load_source", return_value=({"dev": native}, {})), \
                patch.object(evaluation, "RETENTION_MANIFEST_SHA256", sha256_file(retention / "manifest.json")), \
                patch.object(evaluation, "_training_inputs", return_value=(set(), set(), set(), {str(records_path): sha256_file(records_path)}, evidence)):
            output, private = self.root / "prepared-eval-1", self.root / "gold-eval-1"
            first = evaluation.prepare(prepared, images, retention, records_path, training, output, private)
            self.assertNotIn(groups["dev"][0]["id"], first["qa_ids"])
            self.assertEqual(len(first["qa_ids"]), 48)
            self.assertEqual(first["facts_ids"], first["qa_ids"][:24])
            self.assertEqual(first["approved_training_records"], evidence)
            self.assertFalse((output / "gold.jsonl").exists())
            _, sample_map = evaluation.load_cohort(output)
            for row in sample_map.values():
                if row["mode"] == "FACTS":
                    self.assertNotIn("cell-", row["user_text"])
            for row in native.values():
                row["answer"] = "changed unseen gold"
            second = evaluation.prepare(prepared, images, retention, records_path, training,
                                        self.root / "prepared-eval-2", self.root / "gold-eval-2")
            self.assertEqual(first["qa_ids"], second["qa_ids"])
            self.assertEqual(first["samples_sha256"], second["samples_sha256"])
            self.assertNotEqual(first["gold_sha256"], second["gold_sha256"])
            with self.assertRaises(FileExistsError):
                evaluation.prepare(prepared, images, retention, records_path, training, output, private)

    def test_bad_backend_completion_is_not_committed_and_stream_closes(self):
        closed = []
        class BrokenBackend:
            def describe(self):
                return {}

            def generate_stream(self, requests):
                try:
                    request = next(iter(requests))
                    yield {"sample_id": request["sample_id"], "raw_response": "failure",
                           "finish_reason": "error", "generated_tokens": 0, "input_tokens": 2}
                finally:
                    closed.append(True)
        output = self.root / "bad-backend"
        with patch.object(evaluation.recovery, "checkout"), patch.object(
                evaluation.recovery, "model_identity", return_value={"arm": "F0"}):
            with self.assertRaisesRegex(ValueError, "non-completed"):
                evaluation.infer(self.data, self.root / "model", self.root / "base", output,
                                 "F0", self.deadline, "a" * 40, backend_factory=lambda *args: BrokenBackend())
        self.assertEqual(closed, [True])
        self.assertEqual(list((output / "responses").glob("*.json")), [])
        self.assertEqual(read_json(output / "status.json")["status"], "INCOMPLETE")
        with self.assertRaisesRegex(ValueError, "unexpired absolute"):
            evaluation.infer(self.data, self.root / "model", self.root / "base", self.root / "expired",
                             "F0", (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "a" * 40)
        self.assertFalse((self.root / "expired").exists())


def hashlib_sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def tabmwp_images_hash(data):
    return evaluation.tabmwp_images.git_hash("blob", data)


if __name__ == "__main__":
    unittest.main()
