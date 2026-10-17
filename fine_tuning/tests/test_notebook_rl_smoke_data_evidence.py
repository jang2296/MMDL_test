"""Authored review fixtures only; actual CPU renders/solvers and process denials."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl import smoke_data_evidence as s
from fine_tuning.notebook_rl.dataset import SPLITS, build_splits, generate_pool
from fine_tuning.notebook_rl.generators import graph, graph_pair


class SmokeDataEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_actual_interventions_change_each_answer_without_nonimage_changes(self):
        for kind_index in range(12):
            p = graph.generate(kind_index, 3407, "train", "obs")
            world = p.verifier.accept["graph_world"]
            changed = s.intervene(world, p.lineage.template_id)
            before = s._solvers(world, p.lineage.template_id)
            after = s._solvers(changed, p.lineage.template_id)
            self.assertEqual(before[0], before[1])
            self.assertEqual(after[0], after[1])
            self.assertNotEqual(before[0], after[0])
            self.assertNotEqual(graph._render(world, "default")[0], graph._render(changed, "default")[0])
            self.assertEqual(world, p.verifier.accept["graph_world"])
        p = graph_pair.generate(0, 3407, "train", "plain")
        changed = s.intervene(p.verifier.accept["graph_world"], p.lineage.template_id)
        before = p.image_bytes
        after, _ = graph_pair._render(changed, "default")
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])

    def test_intervention_journal_retains_every_failure_without_substitution(self):
        p = graph.generate(0, 3407, "train", "obs")
        row = (p.actor, p.verifier, p.lineage, p.image_bytes)
        with patch.object(s, "intervene", side_effect=lambda world, template: copy.deepcopy(world)):
            result = s.measure_interventions(p.actor.source, p.lineage.template_id, row, self.base, count=3)
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["answer_changed"], 0)
        self.assertEqual(len(list((self.base / "attempts").glob("*.json"))), 3)
        self.assertTrue(all(json.loads(path.read_bytes())["status"] == "FAIL"
                            for path in (self.base / "attempts").glob("*.json")))

    def test_true_separate_process_denies_actual_private_benchmark_and_teacher(self):
        root = self.base / "build"
        (root / "train").mkdir(parents=True)
        (root / "train/verifier_private.jsonl").write_text("authored confidential fixture")
        benchmark = self.base / "benchmark/native-fixture.parquet"
        benchmark.parent.mkdir()
        benchmark.write_bytes(b"authored fixture, not real benchmark bytes")
        actual = s.measure_isolation(root, benchmark)
        self.assertTrue(all(actual["measurements"].values()))
        with self.assertRaisesRegex(ValueError, "existing benchmark"):
            s.measure_isolation(root, self.base / "missing")

    def candidate_fixture(self):
        root = self.base / "build"
        build_splits(root, {split: generate_pool("graph", 1, 3407, split)
                           + generate_pool("graph_pair", 1, 3407, split) for split in SPLITS})
        items, files, manifests = s._inputs(root)
        review = {"status": "AI_REVIEWED_PROGRAM_VERIFIED", "build_root": str(root),
                  "build_manifest_sha256": s._sha((root / "manifest.yaml").read_bytes()),
                  "blockers": [], "provenance": {"reviewer_id": "authored-unit-test-not-an-actual-review",
                  "kind": "AI_REVIEWED_PROGRAM_VERIFIED", "method": "DIRECT_ORIGINAL_INPUT_AND_PROGRAM_REVIEW",
                  "reviewed_at_utc": "2026-10-17T00:00:00Z", "human_verified": False, "labels_modified": False},
                  "bound_files_sha256": {path: digest for path, digest in files.items()
                                         if not Path(path).is_relative_to(root / "images")},
                  "samples": [{"problem_id": pid, "image_sha256": list(row[0].image_sha256),
                               "mismatches": 0, "unresolved": [], "printed_text_slots_consistent": True}
                              for pid, row in items.items()],
                  "actor_leak_review": {"reviewed_actor_files": {p: sha for p, sha in files.items()
                                                                 if Path(p).name == "actor_inputs.jsonl"},
                  "warnings_sha256": s._hash({split: value["leak_review"] for split, value in manifests.items()}),
                  "unresolved": 0}}
        review_path = self.base / "authored-review-fixture.json"
        review_path.write_text(json.dumps(review))
        benchmark = self.base / "benchmark/fixture.parquet"
        benchmark.parent.mkdir()
        benchmark.write_bytes(b"authored isolation fixture, not benchmark content")
        return root, s._descriptor(review_path), benchmark

    def test_components_recompute_inputs_and_stay_pending_real_exclusion(self):
        root, review, benchmark = self.candidate_fixture()
        measured = s.measure_interventions
        with patch.object(s, "measure_interventions", side_effect=lambda *args: measured(*args, count=2)):
            result = s.produce_components(root, review, benchmark, self.base / "evidence")
        self.assertFalse(result["training_allowed"])
        self.assertNotIn("D6", result["gates"])
        self.assertEqual(result["gates"]["D1"]["repeat_mismatches"], 0)
        self.assertEqual(result["gates"]["D2"]["solver_mismatches"], 0)
        self.assertEqual(result["gates"]["D5"], {"structure_cross_split": 0, "image_cross_split": 0})
        for measured in result["gates"]["D4"]["templates"].values():
            self.assertEqual((measured["pairs"], measured["answer_changed"]), (2, 2))
        self.assertTrue(result["pending"])
        components = s._descriptor(self.base / "evidence/components.json")
        with self.assertRaisesRegex(ValueError, "actual complete exclusion"):
            s.assemble(components, root, s._descriptor(root / "contamination/receipt_v2.json"), {}, {},
                       self.base / "must-not-exist", source_registry={})
        self.assertFalse((self.base / "must-not-exist").exists())

    def test_review_tampering_or_missing_original_coverage_rejects_before_output(self):
        root, review, benchmark = self.candidate_fixture()
        bad = dict(review, file_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            s.produce_components(root, bad, benchmark, self.base / "bad")
        self.assertFalse((self.base / "bad").exists())
        path = Path(review["path"])
        payload = json.loads(path.read_bytes())
        payload["samples"].pop()
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "complete original-image"):
            s.produce_components(root, s._descriptor(path), benchmark, self.base / "bad")
        self.assertFalse((self.base / "bad").exists())


if __name__ == "__main__":
    unittest.main()
