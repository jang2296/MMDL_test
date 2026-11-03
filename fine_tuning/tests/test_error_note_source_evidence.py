"""Authored CPU fixtures only; source references are never training targets."""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fine_tuning.error_note.answer_contract import load_training_rules
from fine_tuning.error_note.audit import digest, text_sha256
from fine_tuning.error_note.source_evidence import _annotations, _native_rows, _rule, prepare, prepare_dev
from fine_tuning.notebook_rl.answers import compare_training
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

    def prepare(self, name="out", limit=1):
        return prepare(build_root=self.vault, manifest=self.manifest, order=self.order, limit=limit,
                       raw_root=self.raw, output=self.root / name)

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

    def test_placeholder_slake_triple_does_not_claim_content_annotation(self):
        actor = replace(self.actor, source="hf-slake", question="What is shown?", answer_format="short_text")
        item = replace(self.item, answer_type="short_text", gold="lung")
        row = {"qid": 1, "question": actor.question, "answer": item.gold,
               "modality": "X-Ray", "location": "Chest", "triple": ["vhead", "_", "_"]}
        native = {(actor.source, "train:1"): (row, "BoKelvin__SLAKE/train.json", "a" * 64)}
        refs, scope = _annotations(actor, item, {"native_id": "train:1"}, native)
        self.assertEqual(scope, "SOURCE_METADATA_ONLY_PLACEHOLDER_TRIPLE")
        self.assertEqual(next(ref["value"] for ref in refs if ref["field"] == "triple"), row["triple"])

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
