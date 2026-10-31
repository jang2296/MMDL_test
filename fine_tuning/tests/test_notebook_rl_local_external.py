"""Authored CPU cohorts test isolation/scoring, not model inference or improvement."""
import copy
import io
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl import local_external as external
from fine_tuning.notebook_rl.dataset import _bytes, _sha, _write
from fine_tuning.tests.test_notebook_rl_schema import records


class LocalExternalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.cohort = self.base / "cohort"
        self.cohort.mkdir()
        inputs, gold, pairing, images = [], [], [], {}
        for root in range(10):
            stream = io.BytesIO()
            Image.new("RGB", (8, 8), (root, 10, 20)).save(stream, format="PNG")
            raw = stream.getvalue()
            path = f"images/{root}.png"
            _write(self.cohort / path, raw)
            images[path] = _sha(raw)
            for condition in ("A0", "A1", "A2", "A3"):
                for view in ("qa", "structure", "grounded"):
                    pid = f"authored-{root}-{condition}-{view}"
                    shared = {"id": pid, "root_id": f"root-{root}", "family_id": f"circuit-family-{root}",
                              "condition": condition, "view": view, "split": "development",
                              "benchmark_derived": False, "training_allowed": False}
                    pairing.append(shared)
                    inputs.append({"id": pid, "image_paths": [path], "question": "", "options": [],
                        "question_type": "open", "user_text": "Choose the source-current option. Give only its letter.",
                        "benchmark_derived": False, "training_allowed": False})
                    gold.append(shared | {"target": "A" if root % 2 == 0 else "B",
                        "metric_kind": "circuit_" + view, "circuit_contract_version": external.CIRCUIT_VERSION})
        for name, rows in (("inputs.jsonl", inputs), ("gold.jsonl", gold), ("pairing.jsonl", pairing)):
            _write(self.cohort / name, b"".join(_bytes(row) + b"\n" for row in rows))
        self.manifest = {"status": "COHORT_FROZEN", "count": 120, "root_count": 10,
            "split": "development", "circuit_contract_version": external.CIRCUIT_VERSION,
            "benchmark_derived": False, "training_allowed": False,
            "root_ids": [f"root-{i}" for i in range(10)], "images": images,
            **{key: _sha((self.cohort / name).read_bytes()) for name, key in
               (("inputs.jsonl", "inputs_sha256"), ("gold.jsonl", "gold_sha256"), ("pairing.jsonl", "pairing_sha256"))}}
        _write(self.cohort / "manifest.json", _bytes(self.manifest))
        self.expected_sha = _sha((self.cohort / "manifest.json").read_bytes())
        source = {"source_known": True, "rights_approved": True, "benchmark_derived": False, "parents": []}
        self.registry = self.base / "source_registry.json"
        _write(self.registry, _bytes({external.SOURCE: source}))
        self.approval = self.base / "historical_approval.json"
        _write(self.approval, _bytes({"status": "APPROVED", "benchmark_derived": False,
            "sources": {external.SOURCE: source}, "contamination_audit": {"status": "PASS", "quarantined": 0,
                "index_sha256": "1"*64, "coverage": {"record_count": 16740, "files": [{}]*100}}}))
        actor = records()[0]
        self.train = [replace(actor, problem_id=f"train-{i}", source="gen-graph-v1", family="graph")
                      for i in range(10)]
        self.structures = {a.problem_id: f"graph-family-{i}" for i, a in enumerate(self.train)}

    def actors(self):
        return external.load_external_actors(self.cohort, expected_manifest_sha256=self.expected_sha)

    def prepare(self, **kwargs):
        return external.prepare_external_data(self.cohort, kwargs.pop("train", self.train), self.structures,
            kwargs.pop("out", self.base / "prepared"), source_registry=self.registry,
            historical_approval=self.approval, expected_manifest_sha256=self.expected_sha, **kwargs)

    def responses(self):
        return [{"problem_id": a.problem_id, "text": "A" if i % 2 == 0 else "B", "finish_reason": "stop"}
                for i, a in enumerate(self.actors())]

    def score(self, responses):
        return external.score_external_responses(self.cohort, responses, expected_manifest_sha256=self.expected_sha)

    def test_actor_loader_never_opens_gold_or_canonical_solutions(self):
        opened, read = [], external._read

        def public_only(path):
            opened.append(path.name)
            self.assertNotIn(path.name, ("gold.jsonl", "canonical_private.jsonl"))
            return read(path)

        with patch.object(external, "_read", side_effect=public_only):
            actors = self.actors()
        self.assertEqual(len(actors), 10)
        self.assertTrue(all(a.source == external.SOURCE and not a.choices for a in actors))
        self.assertNotIn("gold.jsonl", opened)
        messages = external.external_messages(actors[0])
        self.assertEqual(messages[0]["content"][-1]["text"], actors[0].question)
        self.assertNotIn("Final answer", actors[0].question)
        self.assertFalse(any(k in asdict(actors[0]) for k in ("gold", "target", "solution")))

    def test_preparation_preserves_inputs_and_records_evaluation_only_separation(self):
        before = {p: p.read_bytes() for p in self.cohort.rglob("*") if p.is_file()}
        result = self.prepare()
        self.assertEqual(result["n_problems"], 10)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["GPU_evaluation_completed"])
        self.assertFalse(result["official_evaluation_recipe_claimed"])
        self.assertFalse(result["historical_exclusion"]["new_scan_performed"])
        self.assertFalse(result["historical_exclusion"]["used_as_current_training_approval"])
        self.assertTrue(result["separation"]["source_disjoint"])
        self.assertTrue(result["separation"]["family_id_disjoint"])
        self.assertTrue(result["separation"]["image_sha256_disjoint"])
        self.assertEqual(external.EVALUATION_PROTOCOL["sampler"]["temperature"], 1.)
        self.assertEqual(external.EVALUATION_PROTOCOL["max_new_tokens"], 1024)
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_same_source_or_image_overlap_fails_before_preparation(self):
        for change in ({"source": external.SOURCE}, {"image_sha256": self.actors()[0].image_sha256}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "overlaps"):
                self.prepare(train=[replace(self.train[0], **change)] + self.train[1:])
        self.assertFalse((self.base / "prepared").exists())

    def test_existing_output_and_output_inside_source_are_rejected(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "immutable cohort"):
            self.prepare(out=self.cohort / "child")

    def test_changed_frozen_input_image_or_manifest_is_rejected(self):
        for name in ("inputs.jsonl", "images/0.png", "manifest.json"):
            path = self.cohort / name
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.actors()
            path.write_bytes(before)

    def test_scoring_requires_complete_actual_responses_before_opening_gold(self):
        opened, read = [], external._read

        def observe(path):
            opened.append(path.name)
            return read(path)

        responses = self.responses()
        with patch.object(external, "_read", side_effect=observe):
            partial = self.score(responses[:-1])
        self.assertEqual(partial["status"], "INCOMPLETE")
        self.assertIsNone(partial["accuracy"])
        self.assertFalse(partial["gold_opened"])
        self.assertNotIn("gold.jsonl", opened)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.score(responses[:-1] + responses[:1])
        with self.assertRaisesRegex(ValueError, "fields"):
            self.score([r | {"gold": "A"} for r in responses])

    def test_existing_strict_circuit_scorer_and_paired_arithmetic(self):
        replies = self.responses()
        before = self.score(replies)
        self.assertEqual(before["n_correct"], 10)
        changed = copy.deepcopy(replies)
        changed[0]["text"] = "Final answer: A"
        changed[1]["finish_reason"] = "length"
        changed[2]["text"] = "A or B"
        after = self.score(changed)
        self.assertEqual(after["n_correct"], 7)
        comparison = external.compare_external_scores(before, after)
        self.assertEqual(comparison["correct_to_wrong"], 3)
        self.assertEqual(comparison["net_correct_change"], -3)
        self.assertFalse(comparison["improvement_claimed"])
        self.assertFalse(comparison["execution_conditions_verified"])
        with self.assertRaisesRegex(ValueError, "fixed"):
            external.compare_external_scores(before, after | {"cohort_manifest_sha256": "0"*64})

    def test_changed_trusted_gold_is_rejected_after_response_coverage(self):
        replies = self.responses()
        gold = self.cohort / "gold.jsonl"
        gold.write_bytes(gold.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "record ledger"):
            self.score(replies)


if __name__ == "__main__":
    unittest.main()
