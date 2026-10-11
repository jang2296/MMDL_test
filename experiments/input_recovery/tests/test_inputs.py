"""Self-made fixtures only; CPU selection and input-boundary checks."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from experiments.bottleneck_diagnostic.ingest import read_rows, write_rows
from experiments.bottleneck_diagnostic.input_builder import canonical_hash
from experiments.bottleneck_diagnostic.metrics import CLUSTERS, cluster
from experiments.bottleneck_diagnostic.oracle_validation import oracle_hash
from experiments.input_recovery.inputs import build_request, prepare
from mmdl.evaluation.prompt import build_prompt


class InputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "prepared/assets").mkdir(parents=True)
        (self.source / "oracles").mkdir()
        from PIL import Image
        Image.new("RGB", (2, 3), "white").save(self.source / "prepared/assets/image.png")
        digest = hashlib.sha256((self.source / "prepared/assets/image.png").read_bytes()).hexdigest()
        self.sample = {
            "sample_id": "synthetic_Music_1", "subject": "Music", "question_type": "multiple-choice",
            "question": "Which symbol appears in <image 1>?", "options": ["Alpha", "Beta"],
            "hint": None, "images": [{"field": "image_1", "marker": "<image 1>", "role": "question",
                "order": 0, "path": "assets/image.png", "sha256": digest, "original_size": [2, 3]}],
            "family_id": "self-made", "skill_tags": ["synthetic"],
            "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
        }
        self.oracle = {
            "sample_id": self.sample["sample_id"], "level": "O0", "annotation_version": "synthetic-v1",
            "facts": {"symbols": ["a mark"]}, "source_image_hashes": [digest],
            "evidence_locations": [{"marker": "<image 1>", "region": [0, 0, 1, 1]}],
            "completeness_status": "COMPLETE", "target_leakage_status": "VALID",
            "annotator_exposure_status": "GOLD_BLIND", "reviewer_status": "AI_INDEPENDENT_REVIEWED",
            "annotator_id": "synthetic-maker", "reviewer_id": "synthetic-reviewer", "approved": True,
            "provenance": "BENCHMARK_DERIVED", "training_allowed": False,
        }
        self.oracle["oracle_hash"] = oracle_hash(self.oracle)
        self.rule = {"subject": "Music", "version": "synthetic-v1", "text": "Compare visible symbols.",
                     "sources": ["https://example.org/synthetic"], "approved": True}
        self.samples, self.gold, self.baseline, self.oracles = [], [], [], []
        for subjects in CLUSTERS.values():
            for subject in sorted(subjects)[:3]:
                for number in range(12):
                    sample = self.sample | {"subject": subject, "sample_id": f"synthetic_{subject}_{number}"}
                    self.add(sample, number < 6, True)
        for number in range(900 - len(self.samples)):
            self.add(self.sample | {"subject": "Physics", "sample_id": f"synthetic_control_{number}"},
                     number < 558 - 72, False)
        write_rows(self.source / "prepared/samples.jsonl", self.samples)
        write_rows(self.source / "prepared/gold.jsonl", self.gold)
        self.predictions = self.root / "baseline.jsonl"
        write_rows(self.predictions, self.baseline)
        for name, rows in (("pilot_reviewed", self.oracles), ("coverage_reviewed", []),
                           ("expansion_reviewed", [])):
            write_rows(self.source / "oracles" / f"{name}.jsonl", rows)

    def add(self, sample, correct, oracle):
        self.samples.append(sample)
        self.gold.append({"sample_id": sample["sample_id"], "subject": sample["subject"],
                          "answer": "B", "historical_correct": correct,
                          "provenance": "BENCHMARK_DERIVED", "training_allowed": False})
        self.baseline.append({"id": sample["sample_id"], "correct": correct, "answer": "B",
                              **{key: sample[key] for key in ("subject", "question", "question_type", "options")}})
        if oracle:
            record = self.oracle | {"sample_id": sample["sample_id"]}
            record["oracle_hash"] = oracle_hash(record)
            self.oracles.append(record)

    def tearDown(self):
        self.temporary.cleanup()

    def test_deterministic_selection_strata_caps_subset_and_copied_images(self):
        first = prepare(self.source, self.root / "first", self.predictions)
        second = prepare(self.source, self.root / "second", self.predictions)
        self.assertEqual(first, second)
        self.assertEqual(len(first["selected_ids"]), 48)
        self.assertEqual(len(set(first["subset_ids"])), 12)
        self.assertTrue(set(first["subset_ids"]) <= set(first["selected_ids"]))
        for group in CLUSTERS:
            rows = [r for r in first["strata"] if r["cluster"] == group]
            self.assertEqual(Counter(r["historical_correct"] for r in rows), {True: 6, False: 6})
            self.assertLessEqual(max(Counter(r["subject"] for r in rows).values()), 4)
            self.assertGreaterEqual(len({r["subject"] for r in rows}), 3)
            small = [r for r in rows if r["sample_id"] in first["subset_ids"]]
            self.assertEqual(len(small), 3)
            self.assertEqual({r["historical_correct"] for r in small}, {True, False})
        self.assertEqual((self.root / "first/assets/image.png").read_bytes(),
                         (self.source / "prepared/assets/image.png").read_bytes())
        selected = read_rows(self.root / "first/samples.jsonl")
        self.assertFalse(any("answer" in row or "historical_correct" in row for row in selected))
        self.assertEqual(first["oracle_hashes"], {r["sample_id"]: r["oracle_hash"]
                                                  for r in read_rows(self.root / "first/oracles.jsonl")})
        selected_gold = read_rows(self.root / "first/gold.jsonl")
        self.assertEqual(first["gold_hashes"], {r["sample_id"]: canonical_hash(r)
                                                for r in selected_gold})
        changed_gold = selected_gold[0] | {"answer": "A"}
        self.assertNotEqual(first["gold_hashes"][changed_gold["sample_id"]],
                            canonical_hash(changed_gold))
        expected_hash = canonical_hash({key: value for key, value in first.items() if key != "selection_hash"})
        self.assertEqual(first["selection_hash"], expected_hash)

    def test_p0_all_modes_gold_blind_independent_and_seed_paired(self):
        requests = [build_request(self.sample, mode,
                    self.oracle if mode in {"T", "IT", "TR"} else None,
                    self.rule if mode in {"IR", "TR"} else None) for mode in ("I", "T", "IT", "IR", "TR")]
        self.assertEqual(requests[0]["messages"][0]["content"][-1]["text"],
                         build_prompt(self.sample, Path(__file__).resolve().parents[3]))
        self.assertEqual(len({r["seed"] for r in requests}), 1)
        self.assertEqual(len({r["request_id"] for r in requests}), 5)
        self.assertEqual(len({r["cache_key"] for r in requests}), 5)
        for request in requests:
            self.assertEqual(request["max_new_tokens"], 8192)
            self.assertEqual([m["role"] for m in request["messages"]], ["user"])
            self.assertNotIn("FINAL_ANSWER", json.dumps(request))
            self.assertNotIn("historical_correct", json.dumps(request))
        for request in (requests[1], requests[2], requests[4]):
            self.assertEqual(request["oracle_hash"], self.oracle["oracle_hash"])
            self.assertEqual(request["verification_kind"], "AI_REVIEWED_NOT_HUMAN_VERIFIED")
            self.assertIn('Provided visual information (O0):\n{"symbols": ["a mark"]}',
                          request["messages"][0]["content"][-1]["text"])
        self.assertEqual(requests[1]["messages"][0]["content"], requests[2]["messages"][0]["content"][-1:])

    def test_gold_unknown_fields_and_illegal_oracles_rejected(self):
        for key in ("gold", "answer", "correct", "gold_explanation"):
            with self.assertRaises(ValueError):
                build_request(self.sample | {key: "secret"}, "I")
        for patch in ({"approved": False}, {"facts": {"gold": "B"}},
                      {"source_image_hashes": []}, {"oracle_hash": "0" * 64},
                      {"target_leakage_status": "TARGET_REVEALED"}):
            oracle = self.oracle | patch
            if "oracle_hash" not in patch:
                oracle["oracle_hash"] = oracle_hash(oracle)
            with self.assertRaises(ValueError):
                build_request(self.sample, "T", oracle)
        for sid in ("validation_Math_16", "validation_Energy_and_Power_20"):
            with self.assertRaises(ValueError):
                build_request(self.sample | {"sample_id": sid}, "I")

    def test_image_order_role_paths_and_hashes(self):
        sample = copy.deepcopy(self.sample)
        sample["options"][1] += " <image 2>"
        sample["images"].append(sample["images"][0] | {
            "field": "image_2", "marker": "<image 2>", "role": "option", "order": 1,
            "path": "assets/second.png",
        })
        result = build_request(sample, "IT", self.oracle | {
            "source_image_hashes": [r["sha256"] for r in sample["images"]],
            "oracle_hash": oracle_hash(self.oracle | {
                "source_image_hashes": [r["sha256"] for r in sample["images"]]}),
        })
        self.assertEqual([r["image"] for r in result["messages"][0]["content"][:-1]],
                         [r["path"] for r in sample["images"]])
        for patch in ({"order": 0}, {"role": "question"}, {"sha256": "bad"}, {"path": "../escape"}):
            changed = copy.deepcopy(sample)
            changed["images"][1].update(patch)
            with self.assertRaises(ValueError):
                build_request(changed, "I")

    def test_rules_subject_approval_source_and_unknown_fields(self):
        for patch in ({"subject": "Math"}, {"approved": False}, {"version": ""},
                      {"sources": []}, {"sources": ["file:///private"]}, {"gold": "B"}):
            with self.assertRaises(ValueError):
                build_request(self.sample, "IR", rule=self.rule | patch)
        with self.assertRaises(ValueError):
            build_request(self.sample, "I", rule=self.rule)
        with self.assertRaises(ValueError):
            build_request(self.sample, "IR", oracle=self.oracle, rule=self.rule)

    def test_nooverwrite_and_source_image_tamper(self):
        output = self.root / "exists"
        output.mkdir()
        (output / "keep").write_text("preserve")
        with self.assertRaises(FileExistsError):
            prepare(self.source, output, self.predictions)
        self.assertEqual((output / "keep").read_text(), "preserve")
        with self.assertRaises(ValueError):
            prepare(self.source, self.source / "new", self.predictions)
        (self.source / "prepared/assets/image.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image"):
            prepare(self.source, self.root / "tampered", self.predictions)
        self.assertFalse((self.root / "tampered").exists())

    def test_source_change_and_baseline_mismatch(self):
        with mock.patch("experiments.input_recovery.inputs.sha", side_effect=["old", "changed"]):
            with self.assertRaisesRegex(ValueError, "changed during read"):
                prepare(self.source, self.root / "changed", self.predictions)
        self.baseline[0]["answer"] = "A"
        self.predictions.write_text("".join(json.dumps(row) + "\n" for row in self.baseline))
        with self.assertRaisesRegex(ValueError, "gold/baseline mismatch"):
            prepare(self.source, self.root / "mismatch", self.predictions)

    def test_scorer_rejects_changed_selected_gold_before_export(self):
        from experiments.input_recovery.runner import score

        prepared = self.root / "gold_bound"
        prepare(self.source, prepared, self.predictions)
        gold = read_rows(prepared / "gold.jsonl")
        gold[0]["answer"] = "A"
        (prepared / "gold.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gold))
        output = self.root / "scoring"
        with self.assertRaisesRegex(ValueError, "Selected gold differs"):
            score(prepared, [], output)
        self.assertFalse(output.exists())

    def test_infeasible_strata_refuse_before_output(self):
        group = next(iter(CLUSTERS.values()))
        allowed = sorted(group)[1:3]
        rows = [r for r in self.oracles if self.samples_by_id()[r["sample_id"]]["subject"] not in group
                or self.samples_by_id()[r["sample_id"]]["subject"] in allowed]
        (self.source / "oracles/pilot_reviewed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            prepare(self.source, self.root / "infeasible", self.predictions)
        self.assertFalse((self.root / "infeasible").exists())

    def samples_by_id(self):
        return {sample["sample_id"]: sample for sample in self.samples}

    def test_g6_old_history_never_invents_B_maintained(self):
        paths = {}
        for arm in ("F0", "FL", "FM", "FV", "FML"):
            rows = [{"id": r["id"], "correct":
                     cluster(r["subject"]) != "B_DIAGRAM_SPATIAL" and (arm != "FL")}
                    for r in self.baseline[:60]]
            paths[arm] = self.root / f"{arm}.jsonl"
            write_rows(paths[arm], rows)
        result = prepare(self.source, self.root / "history", self.predictions,
                         g6_predictions_paths=paths)
        for row in result["strata"]:
            if row["cluster"] == "B_DIAGRAM_SPATIAL":
                self.assertEqual(row["g6_kept_by"], [])
        self.assertTrue(result["g6_subset_limitations"]["B_DIAGRAM_SPATIAL"]["g6_kept_by"])
        self.assertEqual(len([key for key in result["source_files"] if key.startswith("g6_")]), 5)


if __name__ == "__main__":
    unittest.main()
