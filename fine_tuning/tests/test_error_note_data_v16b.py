"""Synthetic source-bound preparation tests; no benchmark or model access."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fine_tuning.error_note import curation_v16b as curation
from fine_tuning.error_note import data_v16b as data
from fine_tuning.notebook_rl import contamination


class DataV16bTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.curated = self.root / "curated"
        self.curated.mkdir()
        self.candidates = []
        for index in range(5):
            pid = f"native-{index}"
            image = self.root / f"{index}.png"
            Image.new("RGB", (8, 8), (index * 40, 20, 30)).save(image)
            with Image.open(image) as opened:
                pixels = contamination.image_fingerprints(opened)["pixels_sha256"]
            sha = curation._sha(image)
            actor = {"problem_id": pid, "source": "hf-scienceqa", "family": "diagram", "mode": "plain",
                "question": f"Question {index}?", "choices": ["first", "second"], "answer_format": "mcq",
                "image_paths": [f"native/{index}.png"], "image_sha256": [sha]}
            verifier = {"problem_id": pid, "answer_type": "mcq", "gold": "A", "accept": {},
                "choices": actor["choices"], "slots": {}, "slot_types": {}, "required_slots": [],
                "oracle_fact_lines": [], "printed_text": []}
            lineage = {"problem_id": pid, "source": "hf-scienceqa", "source_revision": "a" * 40,
                "license": "fixture", "label_provenance": "HUMAN", "image_provenance": "native:fixture",
                "upstream_origin": "dataset authors", "generator_version": None, "generator_seed": None,
                "template_id": "fixture", "structure_hash": f"structure-{index}", "split": "train",
                "parents": [], "benchmark_derived": False, "model_touched": False}
            item = {"problem_id": pid, "source": actor["source"], "native_id": f"train:{index}",
                "source_revision": lineage["source_revision"], "family_id": f"family-{index}",
                "historical_family_id": f"family-{index}", "structure_hash": lineage["structure_hash"],
                "actor": actor, "verifier": verifier, "lineage": lineage, "image_paths": [str(image)],
                "image_sha256": [sha], "pixel_sha256": [pixels], "eligible": True, "exclusion_reasons": [],
                "historical_bindings_status": "PASS_RECORDS_AND_IMAGES_RECHECKED", "record_paths": {},
                "record_sha256": {}, "native_source_refs": [], "annotation_refs": [],
                "annotation_verifier_supported": False, "annotation_support_status": "LABEL_ONLY",
                "raw_annotation": None, "native_metadata": {}, "human_solution": None, "f0_prior": None,
                "rights": "private fixture", "manifest_item": {"native_id": f"train:{index}"},
                "historical_manifest": self.ref(self.write(self.root / f"manifest-{index}.json", {"id": pid}))}
            self.candidates.append(item)
        for role, filename in (("actor", "actor_inputs.jsonl"), ("verifier", "verifier_private.jsonl"),
                               ("lineage", "lineage.jsonl")):
            path = self.root / filename
            path.write_text("".join(json.dumps(x[role]) + "\n" for x in self.candidates))
            for item in self.candidates:
                item["record_paths"][role] = str(path)
                item["record_sha256"][role] = curation._digest(item[role])
        receipt = self.write(self.root / "historical.json", {"allowlist": [x["problem_id"] for x in self.candidates],
            "input_files": {x["image_paths"][0]: x["image_sha256"][0] for x in self.candidates}})
        for item in self.candidates:
            item["historical_receipt"] = self.ref(receipt)
        self.fields = {}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    @staticmethod
    def ref(path):
        return {"path": str(path), "sha256": curation._sha(path)}

    def checked(self, item):
        return data._checked_candidate(item, {}, {}, {}, self.fields)

    def freeze(self):
        holdouts = self.root / "holdouts.jsonl"
        holdouts.write_text("")
        supplement = self.write(self.root / "supplement.json", {
            "schema": "v16b-scienceqa-visual-family-supplement-v1", "status": "COMPLETE",
            "source_scope": ["hf-scienceqa"], "links": [], "input_refs": {}})
        scan = self.write(self.curated / "new-text.json", {"status": "COMPLETE_FRESH_TEXT_SCAN_WITH_LIMITS",
            "thresholds_changed": False, "fields": list(self.fields.values())})
        bindings = {"inputs": {"holdouts": self.ref(holdouts)}, "family_supplement": self.ref(supplement)}
        self.write(self.curated / "train-3000-manifest.json", {"count": 2,
            "items": [curation._manifest_item(x) for x in self.candidates[:2]],
            "bindings": bindings, "new_text_checks": self.ref(scan)})
        self.write(self.curated / "monitor-manifest.json", {"count": 1,
            "items": [curation._manifest_item(self.candidates[2])]})
        self.write(self.curated / "order.json", {"items": [{"problem_id": x["problem_id"]} for x in self.candidates[:2]],
            "batches": [{"problem_ids": [x["problem_id"] for x in self.candidates[:2]]}]})
        (self.curated / "candidates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.candidates))
        validation = self.write(self.curated / "validation.json", {
            "status": "PASS_CURATED_NATIVE_3000_WITH_LIMITS", "artifacts": {
                name: self.ref(self.curated / name) for name in ("train-3000-manifest.json",
                    "monitor-manifest.json", "order.json", "candidates.jsonl")}})
        return curation._sha(validation)

    def test_deterministic_selection_and_no_model_outcome_selection(self):
        selected, receipt = data.select_prep(self.candidates, self.candidates[:1], self.candidates[1:2], [], count=2)
        altered = copy.deepcopy(list(reversed(self.candidates)))
        for item in altered:
            item["f0_prior"] = {"status": "INCORRECT"}
        other, _ = data.select_prep(altered, self.candidates[:1], self.candidates[1:2], [], count=2)
        self.assertEqual([x["problem_id"] for x in selected], [x["problem_id"] for x in other])
        self.assertFalse(receipt["selection_uses_model_outcomes"])

    def test_blocks_family_pixels_native_and_holdouts(self):
        altered = copy.deepcopy(self.candidates)
        altered[2]["family_id"] = altered[0]["family_id"]
        altered[3]["pixel_sha256"] = altered[0]["pixel_sha256"]
        altered[4]["native_id"] = altered[0]["native_id"]
        with self.assertRaisesRegex(ValueError, "insufficient"):
            data.select_prep(altered, altered[:2], [], [], count=1)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            data.select_prep(self.candidates, self.candidates[:2], self.candidates[2:3], self.candidates[3:], count=1)

    def test_largest_remainder_and_missing_sources(self):
        candidates = [dict(x, source="hf-mapqa" if i < 2 else "hf-scienceqa")
                      for i, x in enumerate(self.candidates)]
        train = [{"problem_id": f"train-{i}", "source": source} for i, source in enumerate(
            ["hf-mapqa", "hf-scienceqa", "hf-scienceqa", "hf-scienceqa", "hf-clevr"])]
        selected, receipt = data.select_prep(candidates, train, [], [], count=4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(receipt["source_counts"], {"hf-mapqa": 1, "hf-scienceqa": 3})
        self.assertEqual(receipt["missing_train_sources"], ["hf-clevr"])

    def test_native_annotation_aliases_preserve_facts_without_old_approval(self):
        geometry = {"diagram_logic_forms": ["Equals(LengthOf(Line(A, B)), 3)"]}
        graph = {"n": 4, "m": 2, "edges": [[0, 1], [1, 3]], "visible_nodes": [0, 1, 3]}
        scene = {"objects": [{"color": "red"}], "split": "train"}
        for source, raw, metadata, expected in (
            ("hf-geometry3k", geometry, {}, geometry),
            ("hf-gvlqa", graph, {"directed": False, "task": "cycle"},
             {"directed": False, "task": "cycle", "structure_text": "4 2\n0 1\n1 3\n"}),
            ("hf-clevr", {"scene": scene, "program": []}, {}, scene)):
            item = {"source": source, "references": {"raw_annotation": raw, "native_metadata": metadata}}
            self.assertEqual(data.native_annotation(item), expected)
        self.assertIsNone(data.native_annotation({"references": {}}))

    def test_rejects_changed_native_record(self):
        changed = copy.deepcopy(self.candidates[0])
        changed["actor"]["question"] = "unbound question"
        with self.assertRaisesRegex(ValueError, "native record"):
            self.checked(changed)

    def test_rejects_benchmark_and_teacher_lineage(self):
        for field in ("benchmark_derived", "model_touched"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.candidates[0])
                changed["lineage"][field] = True
                changed["record_sha256"]["lineage"] = curation._digest(changed["lineage"])
                records = {changed["record_paths"]["lineage"]: {changed["problem_id"]: changed["lineage"]}}
                with self.assertRaisesRegex(ValueError, "teacher/benchmark"):
                    data._checked_candidate(changed, {}, records, {}, {})

    def test_rejects_corrupt_source_reference(self):
        changed = copy.deepcopy(self.candidates[0])
        changed["native_source_refs"] = [{"path": str(self.root / "historical.json"), "sha256": "0" * 64}]
        with self.assertRaisesRegex(ValueError, "reference hash"):
            self.checked(changed)

    def test_rejects_unscanned_text_and_teacher_solution(self):
        changed = copy.deepcopy(self.candidates[0])
        changed["native_metadata"] = {"new": "uncleared words"}
        with self.assertRaisesRegex(ValueError, "not exactly cleared"):
            self.checked(changed)
        changed["human_solution"] = {"authorship": "teacher_model", "human_verified": False}
        with self.assertRaisesRegex(ValueError, "native authorship"):
            self.checked(changed)

    def test_rejects_image_receipt_and_pixel_mismatch(self):
        changed = copy.deepcopy(self.candidates[0])
        changed["pixel_sha256"] = ["0" * 64]
        with self.assertRaisesRegex(ValueError, "pixel binding"):
            self.checked(changed)

    def test_assembles_standard_vault_and_tamper_fails(self):
        pin = self.freeze()
        output = self.root / "assembled"
        receipt = data.prepare(self.curated, output, prep_count=2, validation_sha256=pin)
        self.assertEqual(receipt["counts"], {"train": 2, "monitor": 1, "prep": 2})
        self.assertFalse(data.load_items(output, "monitor")[0]["training_allowed"])
        self.assertEqual((output / "order.json").read_bytes(), (self.curated / "order.json").read_bytes())
        bundle = json.loads((output / "bundle-spec.json").read_text())["paths"]
        self.assertNotIn(str(self.root / "historical.json"), bundle)
        self.assertTrue(all(Path(x).is_relative_to(output) for x in bundle))
        with self.assertRaises(FileExistsError):
            data.prepare(self.curated, output, prep_count=2, validation_sha256=pin)
        with (output / "vault/prep/actor_inputs.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "payload hash"):
            data.validate(output)

    def test_native_human_solution_keeps_exact_text_and_clearance_receipt(self):
        item = self.candidates[3]
        item["human_solution"] = {
            "authorship": "dataset_native_expert_curated", "human_verified": False,
            "lecture": "Human authored fixture lecture.", "solution": "The first option follows from the example.",
            "source_ref": self.ref(self.write(self.root / "human-source.json", {"native_id": item["native_id"]})),
            "provenance_card": self.ref(self.write(self.root / "human-card.json", {"authors": "dataset experts"}))}
        for field in ("lecture", "solution"):
            name = f"new_native.human_solution.{field}"
            self.fields[item["problem_id"], name] = {"problem_id": item["problem_id"], "field": name,
                "text_sha256": data.hashlib.sha256(item["human_solution"][field].encode()).hexdigest(),
                "matches": []}
        pin = self.freeze()
        output = self.root / "human-assembled"
        data.prepare(self.curated, output, prep_count=2, validation_sha256=pin)
        human = next(x for x in data.load_items(output, "prep") if x["problem_id"] == item["problem_id"])
        reference = human["references"]["human_solution"]
        self.assertEqual(reference["solution"], item["human_solution"]["solution"])
        self.assertEqual(reference["contamination_status"], "PASS_FRESH_TEXT_SCAN_WITH_LIMITS")
        self.assertNotIn("path", reference["source_ref"])
        self.assertFalse(reference["human_verified"])
        receipt = reference["text_check_receipt"]
        self.assertEqual(curation._sha(Path(receipt["path"])), receipt["sha256"])

    def test_rejects_frozen_manifest_tampering(self):
        pin = self.freeze()
        (self.curated / "monitor-manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "reference hash"):
            data.prepare(self.curated, self.root / "assembled", prep_count=2, validation_sha256=pin)


if __name__ == "__main__":
    unittest.main()
