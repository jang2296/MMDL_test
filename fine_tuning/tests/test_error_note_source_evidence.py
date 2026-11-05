"""Authored CPU fixtures only; source references are never training targets."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning.error_note.answer_contract import load_training_rules
from fine_tuning.error_note.audit import digest, text_sha256
from fine_tuning.error_note.source_evidence import _annotations, _image_unit_references, _native_rows, _rule, prepare, prepare_dev
from fine_tuning.notebook_rl.answers import compare_training
from fine_tuning.notebook_rl.adapters.hf_circuit import SPECS as CIRCUIT_SPECS
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.tests.test_notebook_rl_schema import records


class SourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.vault, self.raw = self.root / "vault", self.root / "raw"
        self.vault.mkdir()
        self.raw.mkdir()
        (self.raw / "download_manifest.jsonl").write_text("")
        self.actor, self.item, self.lineage, *_ = records()
        image = b"authored fixture image bytes"
        self.actor = replace(self.actor, image_sha256=(hashlib.sha256(image).hexdigest(),))
        (self.vault / "images/images").mkdir(parents=True)
        (self.vault / "images/images/a.png").write_bytes(image)
        write_split(self.vault / "train", [self.actor], [self.item], [self.lineage])
        train_sha = text_sha256((self.vault / "train/manifest.yaml").read_text())
        (self.vault / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": train_sha}}))
        self.frozen = {"version": "pool-freeze-manifest-v1", "items": [{
            "problem_id": self.actor.problem_id, "source": self.actor.source,
            "source_revision": self.lineage.source_revision, "native_id": "train:1", "subject": "Math",
            "status": "INCLUDED", "split": "train", "image_paths": list(self.actor.image_paths),
            "image_sha256": list(self.actor.image_sha256), "adapter_code_sha256": "b" * 64,
            "record_sha256": {key: text_sha256(to_json(record)) for key, record in
                              (("actor", self.actor), ("verifier", self.item), ("lineage", self.lineage))}}]}
        self.manifest = self.root / "dataset_manifest.json"
        self.bind()
        self.order = self.root / "order.json"
        self.order.write_text(json.dumps({"version": "error-note-order-v2", "count": 1,
            "ids_sha256": text_sha256(self.actor.problem_id), "items": [{"problem_id": self.actor.problem_id,
                "subject": "Math", "source": self.actor.source}]}))

    def bind(self):
        self.manifest.write_text(json.dumps(self.frozen))
        (self.vault / "freeze_binding.json").write_text(json.dumps({
            "version": "pool-freeze-vault-binding-v1", "dataset_manifest_sha256": text_sha256(self.manifest.read_text()),
            "vault_manifest_sha256": text_sha256((self.vault / "manifest.yaml").read_text())}))

    def prepare(self, name="out", limit=1, **kwargs):
        return prepare(build_root=self.vault, manifest=self.manifest, order=self.order, limit=limit,
                       raw_root=self.raw, output=self.root / name, **kwargs)

    def circuit_cache(self, **changes):
        supplementary = self.root / "supplementary"
        repo = supplementary / "MirandaAbhilash__circuitvqa-dataset/questions"
        repo.mkdir(parents=True, exist_ok=True)
        row = {"splittype": "train", "qtype": "count", "question": "Count the voltage sources.",
               "answer": "2", "symbol": "voltage.dc", "bbox_len": 12,
               "bbox": "voltage-dc 0.2 0.3 0.1 0.1 . voltage-dc 0.6 0.7 0.1 0.1",
               "desc": "FORBIDDEN_TEACHER_DESCRIPTION"}
        row.update(changes)
        path = repo / "master_bbox.json"
        path.write_text(json.dumps([row]))
        entry = {"repo": CIRCUIT_SPECS["hf-circuitvqa"].repo,
                 "revision": CIRCUIT_SPECS["hf-circuitvqa"].revision,
                 "path": "questions/master_bbox.json", "sha256": text_sha256(path.read_text())}
        (supplementary / "download_manifest.jsonl").write_text(json.dumps(entry) + "\n")
        selected = {"fixture": {"source": "hf-circuitvqa", "native_id": "train:0",
                                 "source_revision": entry["revision"]}}
        return supplementary, selected, row

    def test_deterministic_private_files_and_existing_rule_loader(self):
        summary = self.prepare()
        self.assertEqual(summary, self.prepare("out2"))
        for name in ("rules.json", "source-provenance.json", "summary.json"):
            self.assertEqual((self.root / "out" / name).read_bytes(), (self.root / "out2" / name).read_bytes())
            self.assertEqual((self.root / "out" / name).stat().st_mode & 0o777, 0o600)
        rules = load_training_rules(self.root / "out/rules.json", summary["output_sha256"]["rules.json"],
            vault_root=self.vault / "train", actors={self.actor.problem_id: self.actor},
            items={self.item.problem_id: self.item}, lineages={self.lineage.problem_id: self.lineage})
        self.assertEqual(set(rules), {self.actor.problem_id})
        source = json.loads((self.root / "out/source-provenance.json").read_text())["items"][self.actor.problem_id]
        self.assertEqual(source["label_provenance"], "PROGRAM")
        self.assertEqual(source["annotations"], [])
        self.assertFalse(source["content_verification_available"])
        self.assertEqual(source["note_content_status"], "UNKNOWN_REVIEW_REQUIRED")
        self.assertFalse(summary["training_permission_granted"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.prepare()

    def test_selection_rejects_nontrain_excluded_unbound_or_unknown_ids(self):
        for change in ({"split": "dev"}, {"status": "EXCLUDED_QUALITY_FILTER"}, {"source": "other"},
                       {"record_sha256": {"actor": "a" * 64}}):
            original = copy.deepcopy(self.frozen)
            self.frozen["items"][0].update(change)
            self.bind()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
            self.frozen = original
        self.bind()
        order = json.loads(self.order.read_text())
        order["items"].append({"problem_id": "not-in-vault", "source": "fake", "subject": "Math"})
        order["count"] = 2
        order["ids_sha256"] = text_sha256(self.actor.problem_id + "\nnot-in-vault")
        self.order.write_text(json.dumps(order))
        self.assertEqual(self.prepare()["count"], 1)
        with self.assertRaisesRegex(ValueError, "selected id"):
            self.prepare("all", limit=2)

    def test_image_and_manifest_hash_changes_are_rejected(self):
        (self.vault / "images/images/a.png").write_bytes(b"changed fixture image bytes")
        with self.assertRaisesRegex(ValueError, "image hash"):
            self.prepare()
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest/vault"):
            self.prepare()

    def test_native_source_hash_and_revision_are_checked_before_using_fields(self):
        repo = self.raw / "liupf__ChEBI-20-MM"
        repo.mkdir()
        path = repo / "train.csv"
        path.write_text('CID,description,SMILES\n1,It has a role as an antibiotic.,C\n')
        self.download_record = {"repo": "liupf/ChEBI-20-MM", "revision": "fixture-v1",
                                "path": "train.csv", "sha256": text_sha256(path.read_text())}
        (self.raw / "download_manifest.jsonl").write_text(json.dumps(self.download_record) + "\n")
        selected = {"fixture": {"source": "hf-chebi-role", "native_id": "cid:1", "source_revision": "fixture-v1"}}
        native, _ = _native_rows(self.raw, selected)
        actor = replace(self.actor, source="hf-chebi-role", answer_format="mcq", choices=("antibiotic", "other"))
        item = replace(self.item, answer_type="mcq", gold="A", choices=actor.choices)
        refs, scope = _annotations(actor, item, selected["fixture"], native)
        description = next(ref for ref in refs if ref["field"] == "description")
        self.assertEqual(description["sha256"], digest(description["value"]))
        self.assertTrue(description["location"].startswith("raw:liupf__ChEBI-20-MM/train.csv#"))
        self.assertEqual(scope, "CURATED_ROLE_DESCRIPTION_FOR_REVIEW")
        path.write_text(path.read_text().replace("antibiotic", "other role"))
        with self.assertRaisesRegex(ValueError, "hash/revision"):
            _native_rows(self.raw, selected)
        path.write_text('CID,description,SMILES\n1,It has a role as an antibiotic.,C\n')
        selected["fixture"]["source_revision"] = "different-revision"
        with self.assertRaisesRegex(ValueError, "hash/revision"):
            _native_rows(self.raw, selected)

    def test_explicit_degree_units_preserve_tolerance_and_reject_other_units(self):
        actor = replace(self.actor, source="hf-geometry3k", answer_format="decimal", question="Find the angle in degrees.")
        item = replace(self.item, answer_type="decimal", gold="60", accept={"abs_tol": "0.000001"})
        frozen = self.frozen["items"][0]
        binding = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                   "source": actor.source, "source_revision": self.lineage.source_revision, "split": "train", "native_id": "train:1"}
        reference = {"field": "problem", "value": actor.question, "sha256": digest(actor.question)}
        rule, status = _rule(actor, item, self.lineage, frozen, [reference], binding)
        self.assertEqual(status, "EXPLICIT_NATIVE_DEGREES")
        self.assertEqual(rule["accept"]["abs_tol"], "0.000001")
        for answer, expected in (("60°", True), ("60 degrees", True), ("60.0000001", True),
                                 ("60 rad", False), ("60 cm", False), ("-60°", False)):
            with self.subTest(answer=answer):
                self.assertIs(compare_training(answer, item, rule), expected)
        for question, refs in (("Find the angle.", [reference]), ("An angle is 60°. Find the length.", [reference]),
                               (actor.question, [])):
            with self.subTest(question=question, refs=refs):
                rule, status = _rule(replace(actor, question=question), item, self.lineage, frozen, refs, binding)
                self.assertEqual(rule["answer_type"], "decimal")
                self.assertEqual(status, "UNSUPPORTED_NO_EXPLICIT_NATIVE_UNIT")
                self.assertIsNone(compare_training("60 rad", item, rule))

    def test_native_table_scale_is_bound_to_original_image_question_and_label(self):
        actor = replace(self.actor, source="hf-tatqa-table", answer_format="decimal", choices=(),
                        question="What is the average?\nAnswer scale should be: percent.")
        item = replace(self.item, answer_type="decimal", choices=(), gold="2.5", accept={"rel_tol": "0.01"})
        frozen = dict(self.frozen["items"][0], native_id="table:0:0")
        row = {"uid": "native-q", "question": "What is the average?", "answer": 2.5,
               "answer_type": "arithmetic", "answer_from": "table", "scale": "percent",
               "rel_paragraphs": [], "derivation": "SOURCE_ONLY_DERIVATION",
               "_native_table": {"uid": "native-table", "table": [["year", "value"], ["2019", "2.5"]]},
               "_origin": {"original_table_uid": "native-table"}, "_mirror_relative": "pinned.parquet",
               "_mirror_sha256": "c" * 64, "_normalized_image_sha256": actor.image_sha256[0],
               "_candidate_question": actor.question, "_candidate_gold": item.gold,
               "_candidate_answer_type": item.answer_type, "_candidate_accept": item.accept}
        native = {(actor.source, frozen["native_id"]): (row, "author.json", "d" * 64)}
        refs, scope = _annotations(actor, item, frozen, native)
        self.assertEqual(scope, "SOURCE_NATIVE_TABLE_ANNOTATIONS_FOR_REVIEW")
        self.assertEqual(next(r["value"] for r in refs if r["field"] == "native_table"), row["_native_table"])
        binding = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                   "source": actor.source, "source_revision": self.lineage.source_revision,
                   "split": "train", "native_id": frozen["native_id"]}
        rule, status = _rule(actor, item, self.lineage, frozen, refs, binding)
        self.assertEqual(status, "NATIVE_AUTHOR_SCALE_NO_RESCALING")
        self.assertEqual(rule["accept"]["rel_tol"], item.accept["rel_tol"])
        for answer, expected in (("2.5", True), ("2.5%", True), (r"2.5\%", True), ("2.5 percent", True),
                                 ("0.025", False), ("250%", False), ("2.5 million", False)):
            with self.subTest(answer=answer):
                self.assertIs(compare_training(answer, item, rule), expected)
        unbound, status = _rule(actor, item, self.lineage, frozen, [], binding)
        self.assertEqual(status, "NO_UNIT_EXTENSION")
        self.assertEqual(unbound["answer_type"], "decimal")
        for scale in ("thousand", "million", "billion"):
            magnitude_refs = [dict(r, value=scale, sha256=digest(scale)) if r["field"] == "scale" else r for r in refs]
            magnitude, _ = _rule(actor, item, self.lineage, frozen, magnitude_refs, binding)
            self.assertIs(compare_training("2.5 " + scale, item, magnitude), True)
            self.assertIs(compare_training("2500000", item, magnitude), False)
        for changed_actor, changed_item in ((replace(actor, image_sha256=("a" * 64,)), item),
                                          (replace(actor, question="Different question"), item),
                                          (actor, replace(item, gold="250"))):
            with self.subTest(changed_actor=changed_actor.question, changed_gold=changed_item.gold), self.assertRaises(ValueError):
                _annotations(changed_actor, changed_item, frozen, native)

    def test_exact_bound_reviewed_image_degrees_allow_only_selected_angular_target(self):
        actor = replace(self.actor, source="hf-geometry3k", answer_format="decimal",
                        question=r"Find the measure of $\angle 1$.")
        item = replace(self.item, answer_type="decimal", gold="123", accept={"abs_tol": "0.000001"})
        frozen = dict(self.frozen["items"][0], source=actor.source)
        pid = actor.problem_id
        entry = {"source": actor.source, "source_revision": frozen["source_revision"], "native_id": frozen["native_id"],
                 "split": "train", "actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                 "angular_target": r"\angle 1", "observed_degree_marks": ["95°", "109°", "33°", "24°"]}
        body = {"version": "geometry-image-unit-evidence-v1", "reviewer_id": "authored-fixture-reviewer",
                "human_verified": False, "labels_modified": False, "items": {pid: entry}}
        path = self.root / "image-units.json"

        def write(value):
            path.write_text(json.dumps(value))
            return text_sha256(path.read_text())

        expected = write(body)
        refs = _image_unit_references(path, expected, {pid: frozen}, {pid: actor})
        binding = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                   "source": actor.source, "source_revision": frozen["source_revision"], "split": "train",
                   "native_id": frozen["native_id"]}
        rule, status = _rule(actor, item, self.lineage, frozen, [refs[pid]], binding)
        self.assertEqual(status, "REVIEWED_NATIVE_IMAGE_DEGREES")
        self.assertEqual((item.gold, item.accept), ("123", {"abs_tol": "0.000001"}))
        self.assertEqual(rule["accept"]["abs_tol"], "0.000001")
        for answer, result in (("123°", True), ("123 degrees", True), ("123.0000001", True), ("123 rad", False)):
            self.assertIs(compare_training(answer, item, rule), result)
        self.assertEqual(_image_unit_references(None, None, {pid: frozen}, {pid: actor}), {})
        for change in ({"actor_sha256": "0" * 64}, {"image_sha256": ["0" * 64]}, {"native_id": "train:2"},
                       {"source_revision": "changed"}, {"source": "hf-scienceqa"}, {"angular_target": "length 1"},
                       {"angular_target": r"\angle 2"}, {"observed_degree_marks": ["95 radians"]}):
            changed = copy.deepcopy(body)
            changed["items"][pid].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _image_unit_references(path, write(changed), {pid: frozen}, {pid: actor})
        write(body)
        for args in ((path, "0" * 64), (path, None), (None, expected)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                _image_unit_references(*args, {pid: frozen}, {pid: actor})
        with self.assertRaisesRegex(ValueError, "unselected"):
            _image_unit_references(path, expected, {}, {})

    def test_v2_observed_target_degree_glyph_and_native_variable_are_bound(self):
        actor = replace(self.actor, source="hf-geometry3k", answer_format="decimal", question=r"Find $x$.")
        item = replace(self.item, answer_type="decimal", gold="11", accept={"abs_tol": "0.000001"})
        frozen = dict(self.frozen["items"][0], source=actor.source)
        pid = actor.problem_id
        entry = {"source": actor.source, "source_revision": frozen["source_revision"], "native_id": frozen["native_id"],
                 "split": "train", "actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                 "angular_target": "x", "target_degree_mark": "x°", "observed_degree_marks": ["220°"]}
        body = {"version": "geometry-image-unit-evidence-v2", "reviewer_id": "authored-target-glyph-reviewer",
                "human_verified": False, "labels_modified": False, "items": {pid: entry}}
        path = self.root / "image-units-v2.json"

        def load(value, candidate_actor=actor):
            path.write_text(json.dumps(value))
            return _image_unit_references(path, text_sha256(path.read_text()), {pid: frozen}, {pid: candidate_actor})

        refs = load(body)
        problem = {"field": "problem", "value": actor.question, "sha256": digest(actor.question)}
        binding = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                   "source": actor.source, "source_revision": frozen["source_revision"], "split": "train",
                   "native_id": frozen["native_id"]}
        rule, status = _rule(actor, item, self.lineage, frozen, [problem, refs[pid]], binding)
        self.assertEqual(status, "REVIEWED_NATIVE_IMAGE_DEGREES")
        self.assertEqual(rule["accept"]["variable"], "x")
        self.assertEqual(rule["accept"]["abs_tol"], "0.000001")
        for answer, expected in (("11°", True), ("x=11°", True), (r"\boxed{x=11^\circ}", True),
                                 ("11 rad", False), ("y=11°", None)):
            self.assertIs(compare_training(answer, item, rule), expected)
        self.assertEqual((item.gold, item.accept), ("11", {"abs_tol": "0.000001"}))
        for change in ({"angular_target": "y", "target_degree_mark": "y°"}, {"target_degree_mark": "x"},
                       {"target_degree_mark": "x rad"}, {"image_sha256": ["0" * 64]}):
            changed = copy.deepcopy(body)
            changed["items"][pid].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load(changed)
        radian_actor = replace(actor, question=r"Find $x$ in radians.")
        changed = copy.deepcopy(body)
        changed["items"][pid]["actor_sha256"] = text_sha256(to_json(radian_actor))
        with self.assertRaises(ValueError):
            load(changed, radian_actor)
        rule, _ = _rule(actor, item, self.lineage, frozen, [problem], binding)
        self.assertEqual(rule["answer_type"], "decimal")
        self.assertIs(compare_training("x=11", item, rule), True)
        self.assertIsNone(compare_training("x=11°", item, rule))  # variable text alone does not prove degrees
        unbound, _ = _rule(actor, item, self.lineage, frozen, [], binding)
        self.assertNotIn("variable", unbound["accept"])

    def test_prepare_combines_disjoint_pinned_v1_v2_and_rejects_overlap_or_changed_bindings(self):
        original_actor, original_item, original_lineage = self.actor, self.item, self.lineage
        actors = [replace(original_actor, source="hf-geometry3k", answer_format="decimal", choices=(),
                          question=r"Find the measure of $\angle 1$."),
                  replace(original_actor, problem_id=original_actor.problem_id + "-second", source="hf-geometry3k",
                          answer_format="decimal", choices=(), question=r"Find $x$.")]
        items = [replace(original_item, problem_id=a.problem_id, answer_type="decimal", gold=gold, choices=(),
                         accept={"abs_tol": "0.000001"}) for a, gold in zip(actors, ("123", "11"), strict=True)]
        lineages = [replace(original_lineage, problem_id=a.problem_id, source=a.source) for a in actors]
        self.vault = self.root / "mixed-vault"
        (self.vault / "images/images").mkdir(parents=True)
        (self.vault / "images/images/a.png").write_bytes(b"authored fixture image bytes")
        write_split(self.vault / "train", actors, items, lineages)
        self.frozen["items"] = []
        native, bodies = {}, []
        for i, (actor, item, lineage) in enumerate(zip(actors, items, lineages, strict=True), 1):
            row = {"problem_id": actor.problem_id, "source": actor.source, "source_revision": lineage.source_revision,
                   "native_id": f"train:{i}", "subject": "Math", "status": "INCLUDED", "split": "train",
                   "image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256),
                   "adapter_code_sha256": "b" * 64,
                   "record_sha256": {key: text_sha256(to_json(value)) for key, value in
                                     (("actor", actor), ("verifier", item), ("lineage", lineage))}}
            self.frozen["items"].append(row)
            native[actor.source, row["native_id"]] = ({"problem": actor.question, "answer": item.gold}, "native.parquet", "c" * 64)
            entry = {"source": actor.source, "source_revision": lineage.source_revision, "native_id": row["native_id"],
                     "split": "train", "actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                     "angular_target": r"\angle 1" if i == 1 else "x", "observed_degree_marks": ["95°"]}
            if i == 2:
                entry["target_degree_mark"] = "x°"
            bodies.append({"version": f"geometry-image-unit-evidence-v{i}", "reviewer_id": "authored-fixture",
                           "human_verified": False, "labels_modified": False, "items": {actor.problem_id: entry}})
        train_sha = text_sha256((self.vault / "train/manifest.yaml").read_text())
        (self.vault / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": train_sha}}))
        self.bind()
        self.order.write_text(json.dumps({"version": "error-note-order-v2", "count": 2,
            "ids_sha256": text_sha256("\n".join(a.problem_id for a in actors)),
            "items": [{"problem_id": a.problem_id, "subject": "Math", "source": a.source} for a in actors]}))
        paths = [self.root / f"pinned-units-{i}.json" for i in (1, 2)]
        for path, body in zip(paths, bodies, strict=True):
            path.write_text(json.dumps(body))
        self.enterContext(patch("fine_tuning.error_note.source_evidence._native_rows", return_value=(native, {})))
        args = {"image_unit_evidence": paths[0], "image_unit_evidence_sha256": text_sha256(paths[0].read_text()),
                "additional_image_unit_evidence": paths[1], "additional_image_unit_evidence_sha256": text_sha256(paths[1].read_text())}
        summary = self.prepare("both-files", limit=2, **args)
        self.assertEqual(summary["answer_policy"], {"REVIEWED_NATIVE_IMAGE_DEGREES": 2})
        self.assertEqual(summary["cache"]["additional_image_unit_evidence_sha256"], args["additional_image_unit_evidence_sha256"])
        rules = json.loads((self.root / "both-files/rules.json").read_text())["items"]
        for actor, item in zip(actors, items, strict=True):
            self.assertIs(compare_training(item.gold + "°", item, rules[actor.problem_id]), True)
            self.assertEqual(item.accept, {"abs_tol": "0.000001"})
        primary = {key: value for key, value in args.items() if not key.startswith("additional_")}
        self.prepare("single-default", limit=2, **primary)
        self.prepare("single-explicit-none", limit=2, **primary, additional_image_unit_evidence=None,
                     additional_image_unit_evidence_sha256=None)
        for name in ("rules.json", "source-provenance.json", "summary.json"):
            self.assertEqual((self.root / "single-default" / name).read_bytes(),
                             (self.root / "single-explicit-none" / name).read_bytes())
        duplicate = dict(args, additional_image_unit_evidence=paths[0],
                         additional_image_unit_evidence_sha256=args["image_unit_evidence_sha256"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.prepare("duplicate", limit=2, **duplicate)
        for change in ({"additional_image_unit_evidence": None}, {"additional_image_unit_evidence_sha256": None},
                       {"additional_image_unit_evidence_sha256": "0" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare("invalid-second", limit=2, **(args | change))
        changed = copy.deepcopy(bodies[1])
        changed["items"][actors[1].problem_id]["image_sha256"] = ["0" * 64]
        paths[1].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.prepare("changed-second-image", limit=2,
                         **(args | {"additional_image_unit_evidence_sha256": text_sha256(paths[1].read_text())}))

    def test_placeholder_slake_triple_does_not_claim_content_annotation(self):
        actor = replace(self.actor, source="hf-slake", question="What is shown?", answer_format="short_text")
        item = replace(self.item, answer_type="short_text", gold="lung")
        row = {"qid": 1, "question": actor.question, "answer": item.gold,
               "modality": "X-Ray", "location": "Chest", "triple": ["vhead", "_", "_"]}
        native = {(actor.source, "train:1"): (row, "BoKelvin__SLAKE/train.json", "a" * 64)}
        refs, scope = _annotations(actor, item, {"native_id": "train:1"}, native)
        self.assertEqual(scope, "SOURCE_METADATA_ONLY_PLACEHOLDER_TRIPLE")
        self.assertEqual(next(ref["value"] for ref in refs if ref["field"] == "triple"), row["triple"])

    def test_circuit_separate_root_binds_native_count_and_omits_descriptions(self):
        supplementary, selected, row = self.circuit_cache()
        with (supplementary / "download_manifest.jsonl").open("a") as journal:
            journal.write(json.dumps({"repo": CIRCUIT_SPECS["hf-circuitvqa"].repo,
                                      "revision": CIRCUIT_SPECS["hf-circuitvqa"].revision,
                                      "local": "unrelated-image-archive", "sha256": "0" * 64}) + "\n")
        native, cache = _native_rows(self.raw, selected, supplementary_raw_root=supplementary)
        actor = replace(self.actor, source="hf-circuitvqa", question=row["question"], answer_format="int", choices=())
        item = replace(self.item, answer_type="int", gold="2", choices=())
        refs, scope = _annotations(actor, item, selected["fixture"], native)
        self.assertEqual(scope, "SOURCE_EXACT_CIRCUIT_COUNT_FOR_REVIEW")
        self.assertEqual(cache["supplementary_download_manifest_sha256"],
                         text_sha256((supplementary / "download_manifest.jsonl").read_text()))
        self.assertTrue(all(ref["location"].startswith("raw:supplementary/") for ref in refs))
        self.assertIn("bbox", {ref["field"] for ref in refs})
        self.assertNotIn("FORBIDDEN", repr(refs))
        self.assertEqual(_native_rows(self.raw, selected)[0], {})

    def test_clevr_original_image_scene_program_and_gold_are_pinned(self):
        from fine_tuning.notebook_rl.adapters.hf_clevr import SPECS
        from fine_tuning.notebook_rl.data_tests.test_hf_v4_sources import clevr_scene, spatial_program

        root = self.root / "clevr-raw"
        repo = root / "HuggingFaceM4__clevr"
        scene = clevr_scene()
        question = {"image_index": 0, "image_filename": scene["image_filename"], "split": "train",
                    "question_index": 7, "program": spatial_program(), "question": "Are fewer metal objects left?",
                    "answer": "yes"}
        image = io.BytesIO()
        Image.new("RGB", (3, 4), "blue").save(image, format="PNG")
        data = {"questions/CLEVR_train_questions.json": json.dumps({"questions": [question]}).encode(),
                "scenes/CLEVR_train_scenes.json": json.dumps({"scenes": [scene]}).encode(),
                "images/train/CLEVR_train_000000.png": image.getvalue()}
        journal = []
        for relative, raw in data.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            journal.append({"repo": SPECS["hf-clevr"].repo, "revision": SPECS["hf-clevr"].revision,
                            "path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        (root / "download_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
        selected = {"fixture": {"source": "hf-clevr", "source_revision": SPECS["hf-clevr"].revision,
                                 "native_id": "train:7"}}
        native, cache = _native_rows(self.raw, selected, supplementary_raw_root=root)
        actor = replace(self.actor, source="hf-clevr", question=question["question"], choices=("Yes", "No"),
                        answer_format="mcq", image_sha256=(hashlib.sha256(image.getvalue()).hexdigest(),))
        item = replace(self.item, answer_type="mcq", choices=actor.choices, gold="A")
        refs, scope = _annotations(actor, item, selected["fixture"], native)
        self.assertEqual(scope, "SOURCE_EXECUTED_SPATIAL_PROGRAM_FOR_REVIEW")
        self.assertIn("scene", {ref["field"] for ref in refs})
        self.assertIn("program", {ref["field"] for ref in refs})
        self.assertEqual(len(cache["files"]), 3)
        self.assertTrue(all(ref["sha256"] == digest(ref["value"]) for ref in refs))
        with self.assertRaisesRegex(ValueError, "gold binding"):
            _annotations(actor, replace(item, gold="B"), selected["fixture"], native)
        with self.assertRaisesRegex(ValueError, "image binding"):
            _annotations(replace(actor, image_sha256=("0" * 64,)), item, selected["fixture"], native)
        (repo / "scenes/CLEVR_train_scenes.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash/revision"):
            _native_rows(self.raw, selected, supplementary_raw_root=root)

    def test_native_pdf_page_reconstruction_and_financial_context_are_review_only(self):
        from fine_tuning.notebook_rl.adapters import hf_docs
        from fine_tuning.notebook_rl.adapters.hf_common import split_for
        from fine_tuning.notebook_rl.data_tests.test_hf_docs import _tat_entry, _tat_q

        entry = next(e for i in range(30) if split_for(hf_docs._tatdqa_candidate(
            (e := _tat_entry(f"report{i}", 2, [_tat_q("q1", "What is the total?", 2, scale="million")]))["doc"]["uid"],
            e["doc"]["source"], e["questions"][0], b"native")) == "train")
        entry["doc"]["page"] = 1  # known upstream metadata can disagree with the two native layout/PDF pages
        repo = self.raw / "next-tat__TAT-DQA"
        repo.mkdir()
        path = repo / "tatdqa_dataset_train.json"
        path.write_text(json.dumps([entry]))
        archive = repo / "tatdqa_docs_train.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(f"train/{entry['doc']['uid']}.pdf", b"authored-native-pdf")
            handle.writestr(f"train/{entry['doc']['uid']}.json", json.dumps({"pages": [{}, {}]}))
        source, revision = "hf-tatdqa-pdf", hf_docs.SPECS["hf-tatdqa-pdf"].revision
        journal = [{"repo": hf_docs.SPECS[source].repo, "revision": revision, "path": p.name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (path, archive)]
        (self.raw / "download_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
        selected = {"fixture": {"source": source, "source_revision": revision,
                                 "native_id": f"{entry['doc']['uid']}:q1"}}
        image = io.BytesIO()
        Image.new("RGB", (3, 4), "white").save(image, format="PNG")
        second = io.BytesIO()
        Image.new("RGB", (3, 4), "blue").save(second, format="PNG")
        with patch.object(hf_docs, "render_pdf_pages", return_value=(image.getvalue(), second.getvalue())) as renderer:
            native, cache = _native_rows(self.raw, selected)
        renderer.assert_called_once_with(b"authored-native-pdf", 2)
        candidate = hf_docs._tatdqa_candidate(entry["doc"]["uid"], entry["doc"]["source"], entry["questions"][0], image.getvalue())
        actor = replace(self.actor, source=source, question=candidate.question, choices=(), answer_format="decimal",
                        image_sha256=(hashlib.sha256(image.getvalue()).hexdigest(), hashlib.sha256(second.getvalue()).hexdigest()))
        item = replace(self.item, answer_type="decimal", choices=(), gold="2", accept=candidate.accept)
        refs, scope = _annotations(actor, item, selected["fixture"], native)
        self.assertEqual(scope, "SOURCE_NATIVE_PDF_FINANCIAL_ANNOTATIONS_FOR_REVIEW")
        self.assertEqual({ref["field"] for ref in refs} & {"facts", "derivation", "pdf_page", "doc"},
                         {"facts", "derivation", "pdf_page", "doc"})
        self.assertTrue(all(ref["sha256"] == digest(ref["value"]) for ref in refs))
        self.assertEqual(len(cache["files"]), 2)
        self.assertEqual(next(ref["value"]["pages"] for ref in refs if ref["field"] == "pdf_page"), [1, 2])
        self.assertEqual(next(ref["value"]["doc_page_metadata"] for ref in refs if ref["field"] == "pdf_page"), 1)
        self.assertNotIn("SECRET", actor.question)
        with self.assertRaisesRegex(ValueError, "scale/gold"):
            _annotations(replace(actor, question="What is the total?"), item, selected["fixture"], native)

    def test_circuit_source_hash_revision_and_symlink_guard_are_not_weakened(self):
        supplementary, selected, _ = self.circuit_cache()
        path = supplementary / "MirandaAbhilash__circuitvqa-dataset/questions/master_bbox.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash/revision"):
            _native_rows(self.raw, selected, supplementary_raw_root=supplementary)
        supplementary, selected, _ = self.circuit_cache()
        selected["fixture"]["source_revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "pinned source"):
            _native_rows(self.raw, selected, supplementary_raw_root=supplementary)
        _, selected, _ = self.circuit_cache()
        alias = self.root / "raw-alias"
        alias.symlink_to(supplementary, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            _native_rows(self.raw, selected, supplementary_raw_root=alias)

    def test_circuit_count_requires_original_question_gold_split_and_box_count(self):
        supplementary, selected, row = self.circuit_cache()
        native, _ = _native_rows(self.raw, selected, supplementary_raw_root=supplementary)
        actor = replace(self.actor, source="hf-circuitvqa", question=row["question"], answer_format="int", choices=())
        item = replace(self.item, answer_type="int", gold="2", choices=())
        for changes in ({"question": "Changed question"}, {"answer": "3"}, {"splittype": "test"},
                        {"bbox": "voltage-dc 0.2 0.3 0.1 0.1"}, {"bbox": ""}):
            changed = copy.deepcopy(native)
            changed[actor.source, "train:0"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "CircuitVQA"):
                _annotations(actor, item, selected["fixture"], changed)

    def test_circuit_other_types_are_label_only_and_have_no_count_certification(self):
        for kind in ("junction", "count-complex"):
            supplementary, selected, row = self.circuit_cache(qtype=kind)
            native, _ = _native_rows(self.raw, selected, supplementary_raw_root=supplementary)
            actor = replace(self.actor, source="hf-circuitvqa", question=row["question"])
            refs, scope = _annotations(actor, self.item, selected["fixture"], native)
            self.assertEqual((refs, scope), ([], "LABEL_ONLY_UNSUPPORTED_CIRCUIT_TYPE"))

    def test_prepare_circuit_annotations_are_private_review_inputs_not_note_approval(self):
        supplementary, _, row = self.circuit_cache()
        self.actor = replace(self.actor, source="hf-circuitvqa", question=row["question"], answer_format="int", choices=())
        self.item = replace(self.item, answer_type="int", gold="2", choices=())
        self.lineage = replace(self.lineage, source="hf-circuitvqa", source_revision=CIRCUIT_SPECS["hf-circuitvqa"].revision)
        previous = self.vault
        self.vault = self.root / "circuit-vault"
        (self.vault / "images/images").mkdir(parents=True)
        (self.vault / "images/images/a.png").write_bytes((previous / "images/images/a.png").read_bytes())
        write_split(self.vault / "train", [self.actor], [self.item], [self.lineage])
        train_sha = text_sha256((self.vault / "train/manifest.yaml").read_text())
        (self.vault / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": train_sha}}))
        frozen = self.frozen["items"][0]
        frozen.update(source=self.actor.source, source_revision=self.lineage.source_revision, native_id="train:0",
                      record_sha256={name: text_sha256(to_json(record)) for name, record in
                                     (("actor", self.actor), ("verifier", self.item), ("lineage", self.lineage))})
        self.bind()
        order = json.loads(self.order.read_text())
        order["items"][0]["source"] = self.actor.source
        self.order.write_text(json.dumps(order))
        summary = self.prepare(supplementary_raw_root=supplementary)
        source = json.loads((self.root / "out/source-provenance.json").read_text())["items"][self.actor.problem_id]
        self.assertTrue(source["content_verification_available"])
        self.assertTrue(source["source_selection_eligible"])
        self.assertEqual(source["note_content_status"], "UNKNOWN_REVIEW_REQUIRED")
        self.assertEqual(source["reference_use"], "PRIVATE_REVIEW_INPUT_ONLY_NOT_TRAINING_TARGET")
        self.assertFalse(summary["human_verified"])
        self.assertNotIn("FORBIDDEN", json.dumps(source))

    def dev_fixture(self):
        actors, items, lineages = [], [], []
        for index, (kind, gold, accept, choices) in enumerate((
                ("decimal", "12.5", {"round_to": 1}, ()), ("short_text", "40-60", {}, ()),
                ("mcq", "A", {}, ("Yes", "No")))):
            pid = f"fake:family:dev:{index}"
            actor = replace(self.actor, problem_id=pid, answer_format=kind, choices=choices)
            item = replace(self.item, problem_id=pid, answer_type=kind, gold=gold, accept=accept, choices=choices)
            lineage = replace(self.lineage, problem_id=pid, split="dev")
            actors.append(actor)
            items.append(item)
            lineages.append(lineage)
            row = copy.deepcopy(self.frozen["items"][0])
            row.update(problem_id=pid, status="HELD_OUT_DEV", split="dev", native_id=f"source-train:{index}",
                       record_sha256={name: text_sha256(to_json(record)) for name, record in
                                      (("actor", actor), ("verifier", item), ("lineage", lineage))})
            self.frozen["items"].append(row)
        write_split(self.vault / "dev", actors, items, lineages)
        meta = json.loads((self.vault / "manifest.yaml").read_text())
        meta["split_manifests"]["dev"] = text_sha256((self.vault / "dev/manifest.yaml").read_text())
        (self.vault / "manifest.yaml").write_text(json.dumps(meta))
        self.bind()
        return actors, items, lineages

    def test_dev_rules_are_separate_complete_and_preserve_numeric_policy(self):
        actors, items, _ = self.dev_fixture()
        summary = prepare_dev(build_root=self.vault, manifest=self.manifest, output=self.root / "dev-out")
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["rules"], 2)
        self.assertFalse(summary["training_allowed"])
        document = json.loads((self.root / "dev-out/rules.json").read_text())
        self.assertEqual(set(document), {"version", "vault_manifest_sha256", "items"})
        self.assertEqual(document["version"], "external-dev-answer-rules-v1")
        self.assertEqual(document["vault_manifest_sha256"], text_sha256((self.vault / "dev/manifest.yaml").read_text()))
        rules = document["items"]
        self.assertEqual(set(rules), {items[0].problem_id, items[1].problem_id})
        numeric, interval = rules[items[0].problem_id], rules[items[1].problem_id]
        self.assertEqual(numeric["accept"], items[0].accept)
        self.assertEqual(numeric["actor_sha256"], text_sha256(to_json(actors[0])))
        self.assertTrue(all(rule["split"] == "dev" and rule["provenance"]["human_verified"] is False for rule in rules.values()))
        self.assertIs(compare_training("12.54", items[0], numeric), True)
        self.assertIsNone(compare_training("12.5 rad", items[0], numeric))
        self.assertEqual(interval["answer_type"], "range")
        self.assertIs(compare_training("40 to 60", items[1], interval), True)
        self.assertIs(compare_training("40 to 61", items[1], interval), False)
        self.assertIsNone(compare_training("40 or 60", items[1], interval))
        self.assertEqual(summary, prepare_dev(build_root=self.vault, manifest=self.manifest, output=self.root / "dev-out2"))
        self.assertEqual((self.root / "dev-out/rules.json").read_bytes(), (self.root / "dev-out2/rules.json").read_bytes())
        self.assertEqual((self.root / "dev-out/rules.json").stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_dev(build_root=self.vault, manifest=self.manifest, output=self.root / "dev-out")
        self.assertEqual(self.prepare()["count"], 1)

    def test_dev_rejects_wrong_split_missing_heldout_coverage_and_forged_hash(self):
        self.dev_fixture()
        for change in ({"split": "train"}, {"status": "HELD_OUT_SEALED"},
                       {"record_sha256": {"actor": "0" * 64}}):
            original = copy.deepcopy(self.frozen)
            self.frozen["items"][1].update(change)
            self.bind()
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_dev(build_root=self.vault, manifest=self.manifest, output=self.root / "bad-dev")
            self.frozen = original


if __name__ == "__main__":
    unittest.main()
