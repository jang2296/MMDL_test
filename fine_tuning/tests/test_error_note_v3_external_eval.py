"""Invented external-dev fixtures: no pretrained weights, benchmark rows or GPU calls."""
import argparse
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning.error_note import evaluate as evaluation
from fine_tuning.error_note.order import _ids_sha256
from fine_tuning.error_note.protocol import ARM, plain_protocol_v3
from fine_tuning.notebook_rl.pool_freeze import _RECORDS, _lines
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.tests.test_error_note_collect import FakeStream, default_output, no_gpu
from fine_tuning.tests.test_notebook_rl_schema import records


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class ExternalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vault = self.root / "vault"
        self.vault.joinpath("images").mkdir(parents=True)
        self.manifest, self.rules, self.order, self.rows = (self.root / name for name in ("manifest.json", "rules.json", "order.json", "rows.jsonl"))
        self.models, self.prereg = self.root / "models.json", self.root / "prereg.json"
        self.items, self.actors, rule_items = [], {}, {}
        original_actor, original_item, original_lineage, *_ = records()
        types = [("decimal", "1.25", {"round_to": 1}), ("short_text", "10-20", {}),
                 ("unit_number", "60°", {"unit_aliases": ["°", "degrees"], "require_unit": True}),
                 ("mcq", "B", {})]
        for split, count in (("train", 1), ("dev", 4)):
            actors, items, lineages = [], [], []
            for index in range(count):
                pid = f"{split}-{index}"
                image = self.vault / "images" / f"{pid}.png"
                Image.new("RGB", (2, 2), (index + 20 * (split == "train"), 11, 23)).save(image)
                image_sha = evaluation.file_hash(image)
                kind, gold, accept = types[index]
                choices = ("alpha", "beta") if kind == "mcq" else ()
                actor = replace(original_actor, problem_id=pid, source="fixture", family="same-task-taxonomy",
                    mode="plain", question=f"Question {pid}?", choices=choices, answer_format=kind,
                    image_paths=(image.name,), image_sha256=(image_sha,))
                item = replace(original_item, problem_id=pid, answer_type=kind, gold=gold, accept=accept, choices=choices,
                               slots={}, slot_types={}, required_slots=(), oracle_fact_lines=(), printed_text=())
                lineage = replace(original_lineage, problem_id=pid, source="fixture", split=split,
                                  structure_hash=hashlib.sha256(pid.encode()).hexdigest())
                actors.append(actor)
                items.append(item)
                lineages.append(lineage)
                self.actors[pid] = actor
                self.items.append({"problem_id": pid, "source": "fixture", "source_revision": lineage.source_revision,
                    "native_id": pid, "split": split, "status": "INCLUDED" if split == "train" else "HELD_OUT_DEV",
                    "family": actor.family, "subject": "Authored fixture", "image_paths": [image.name],
                    "image_sha256": [image_sha], "pixel_sha256": [image_sha]})
                if split == "dev" and kind != "mcq":
                    rule_items[pid] = {"actor_sha256": hashlib.sha256(to_json(actor).encode()).hexdigest(),
                        "image_sha256": [image_sha], "source": actor.source, "source_revision": lineage.source_revision,
                        "split": "dev", "native_id": pid, "answer_type": "range" if index == 1 else kind, "accept": accept,
                        "provenance": {"label_provenance": lineage.label_provenance, "field": "fixture.gold",
                            "location": pid, "sha256": "f" * 64, "human_verified": False, "review_scope": "authored CPU fixture"}}
            write_split(self.vault / split, actors, items, lineages)
            raw = {kind: _lines(self.vault / split / name) for kind, name in _RECORDS}
            for row in self.items:
                if row["split"] == split:
                    row["record_sha256"] = {kind: hashlib.sha256(lines[row["problem_id"]]).hexdigest() for kind, lines in raw.items()}
        write(self.manifest, {"items": self.items})
        write(self.rules, {"version": "external-dev-answer-rules-v1",
            "vault_manifest_sha256": evaluation.file_hash(self.vault / "dev/manifest.yaml"), "items": rule_items})
        write(self.order, {"version": "error-note-order-v1", "count": 1, "items": [{"problem_id": "train-0"}],
                           "ids_sha256": _ids_sha256(["train-0"])})
        self.rows.write_text(json.dumps({"problem_id": "train-0", "row_id": "train-0:A"}) + "\n")
        self.base_ref = self.root / "baseline.json"
        models = {}
        for arm in evaluation.ARMS:
            model = self.root / f"model-{arm}"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
            (model / "model.safetensors").write_bytes(arm.encode())
            ref = {"kind": "base" if arm == "F0" else "merged", "dtype": "bfloat16",
                   "processor_revision": evaluation.BASE_REVISION}
            if arm == "F0":
                ref |= {"repo_id": evaluation.BASE_ID, "revision": evaluation.BASE_REVISION}
                ref_path = self.base_ref
            else:
                hashes = dict.fromkeys(("rows", "order", "audit", "sidecar", "solution_audit", "admission"), "a" * 64)
                hashes.update(rows=evaluation.file_hash(self.rows), order=evaluation.file_hash(self.order))
                protocol = plain_protocol_v3("bf16", 1, arm="note" if arm == "NOTE" else "no-note", contracts=hashes)
                binding = {"protocol_sha256": protocol["sha256"], "prompts_sha256": protocol["prompts_sha256"],
                           "data_sha256": hashes["rows"], "code_sha256": "b" * 64}
                precision = {"fusion": "FP32", "export": "BF16", "merger_masters": "FP32 checkpoint preserved; export rounds to BF16",
                             "unmerged_inference_bit_equivalent": False}
                exported = {"status": "EXPORTED", "format": "derived-hf-bf16-v1", "arm": ARM,
                    "base_id": evaluation.BASE_ID, "base_revision": evaluation.BASE_REVISION,
                    "base_manifest_sha256": "d" * 64, "binding": binding,
                    "checkpoint_files": {"metadata.json": "c" * 64}, "checkpoint_state_sha256": {"tensor": "e" * 64},
                    "resolved_protocol_sha256": evaluation.digest(protocol), "mergers": [], "precision_policy": precision,
                    "files": {path.name: evaluation.file_hash(path) for path in model.iterdir()}}
                write(model / "export_manifest.json", exported)
                ref |= {"base": {"id": evaluation.BASE_ID, "revision": evaluation.BASE_REVISION},
                    "train_config_sha256": evaluation.digest(protocol), "data_manifest_sha256": hashes["rows"],
                    "artifact_revision": evaluation.file_hash(model / "export_manifest.json"),
                    "export": {"arm": ARM, "step": 1, "binding": binding, "training_protocol": protocol,
                        "checkpoint_files": exported["checkpoint_files"],
                        "checkpoint_state_sha256": evaluation.digest(exported["checkpoint_state_sha256"]),
                        "export_manifest_sha256": evaluation.digest(exported), "base_manifest_sha256": "d" * 64,
                        "mergers": [], "precision_policy": precision}}
                ref_path = self.root / f"ref-{arm}.json"
            ref["files"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": evaluation.file_hash(path)}
                            for path in sorted(model.iterdir())]
            write(ref_path, ref)
            models[arm] = {"model_path": str(model), "model_ref": str(ref_path)}
        write(self.models, {"version": "error-note-external-models-v1", "arms": models})
        self.addCleanup(patch.stopall)
        patch.object(evaluation, "BASE_MODEL_REF", self.base_ref).start()
        patch.object(evaluation, "DEV_COUNT", 4).start()
        self.args = argparse.Namespace(manifest=self.manifest, vault_root=self.vault, train_order=self.order,
            train_limit=1, train_rows=self.rows, models=self.models, dev_rules=self.rules, code_commit="a" * 40,
            run_root=self.root / "runs", output=self.prereg)

    def register(self):
        return evaluation.preregister(self.args)

    def generate(self, arm, correct=True):
        answers = ["1.25", "10-20", "60°", "B"] if correct else ["9", "0-5", "0°", "A"]
        stream = FakeStream(lambda request: default_output(request) | {"text": "Final answer: " + answers[int(request["request_id"].split("-")[1])]})
        args = argparse.Namespace(prereg=self.prereg, arm=arm, output=self.args.run_root / arm)
        code = evaluation.run(args, stream=stream, load=lambda actor: [object()], guarded=False, gpu_probe=no_gpu)
        self.assertEqual(code, 0)
        return stream, args.output

    def grade(self, root):
        with patch.object(evaluation, "process_startticks", return_value=None):
            return evaluation.grade(argparse.Namespace(prereg=self.prereg, run=root))

    def test_registration_binds_all_dev_fixed_models_and_disjoint_source_structures(self):
        prereg = self.register()
        self.assertEqual(prereg["counts"]["items"], 4)
        self.assertEqual(prereg["counts"]["independent_families"], 4)
        self.assertEqual(prereg["counts"]["taxonomy_families"], {"same-task-taxonomy": 4})
        self.assertEqual(prereg["protocol"]["max_new_tokens"], 8192)
        self.assertEqual(prereg["protocol"]["sampling"]["n"], 1)
        self.assertEqual(prereg["gate"]["confidence_per_comparison"], .975)
        self.assertFalse(prereg["gate"]["automatic_expansion"])
        with self.assertRaises(FileExistsError):
            self.register()

    def test_generation_has_no_private_grading_and_each_arm_has_one_atomic_path(self):
        self.register()
        with patch.object(evaluation, "_context", side_effect=AssertionError("private context")), \
             patch.object(evaluation, "_dev_verifier", side_effect=AssertionError("private rules")):
            stream, root = self.generate("F0")
        self.assertEqual(len(stream.requests), 4)
        self.assertTrue(all(row["status"] == "NOT_GRADED" for row in evaluation.collect.read_jsonl_done(root / "t1.jsonl").values()))
        for request in stream.requests:
            self.assertNotIn("Correct answer", json.dumps(request["messages"]))
            self.assertEqual(request["max_tokens"], 8192)
        with self.assertRaises(FileExistsError):
            self.generate("F0")
        with self.assertRaisesRegex(ValueError, "preregistered run path"):
            evaluation.run(argparse.Namespace(prereg=self.prereg, arm="F0", output=self.root / "replacement"), guarded=False)

    def test_separate_grading_preserves_raw_and_gate_requires_both_positive_intervals(self):
        self.register()
        runs = {}
        for arm in evaluation.ARMS:
            _, root = self.generate(arm, correct=arm == "NOTE")
            raw = (root / "t1.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "must exit"):
                evaluation.grade(argparse.Namespace(prereg=self.prereg, run=root))
            self.assertEqual(self.grade(root), 0)
            self.assertEqual(raw, (root / "t1.jsonl").read_bytes())
            runs[arm] = root
        report = evaluation.analyze(argparse.Namespace(prereg=self.prereg, f0=runs["F0"], note=runs["NOTE"],
            no_note=runs["NO-NOTE"], output=self.root / "report.json"))
        self.assertEqual(report["decision"], "EXPAND_ELIGIBLE")
        self.assertEqual(report["paired"]["NOTE-F0"]["gains"], 4)
        self.assertEqual(report["paired"]["NOTE-NO-NOTE"]["item_accuracy_ci97_5"], [1., 1.])
        self.assertFalse(report["automatic_expansion"])

    def test_zero_crossing_interval_holds_and_cluster_primary_is_item_weighted(self):
        items = [{"problem_id": str(i), "independent_family": "large" if i < 9 else "small"} for i in range(10)]
        base = {str(i): i == 9 for i in range(10)}
        note = {str(i): i < 9 for i in range(10)}
        paired = evaluation.paired_report(items, base, note)
        self.assertAlmostEqual(paired["item_mean_difference"], .8)
        self.assertEqual(paired["equal_family_mean_difference_secondary"], 0)
        self.assertLessEqual(paired["item_accuracy_ci97_5"][0], 0)
        self.register()
        for arm in evaluation.ARMS:
            _, root = self.generate(arm)
            self.grade(root)
        report = evaluation.analyze(argparse.Namespace(prereg=self.prereg, f0=self.args.run_root / "F0", note=self.args.run_root / "NOTE",
            no_note=self.args.run_root / "NO-NOTE", output=self.root / "hold.json"))
        self.assertEqual(report["decision"], "HOLD")

    def test_changed_frozen_acceptance_or_overlap_is_rejected(self):
        data = evaluation.read_json(self.rules)
        data["items"]["dev-0"]["accept"] = {"abs_tol": "100"}
        write(self.rules, data)
        with self.assertRaisesRegex(ValueError, "acceptance policy changed"):
            self.register()
        data["items"]["dev-0"]["accept"] = {"round_to": 1}
        write(self.rules, data)
        manifest = evaluation.read_json(self.manifest)
        manifest["items"][1]["pixel_sha256"] = manifest["items"][0]["pixel_sha256"]
        write(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.register()

    def test_actual_export_must_match_declared_checkpoint_and_files(self):
        registered = self.register()
        descriptor = registered["models"]["NOTE"]
        ref = evaluation.read_json(Path(descriptor["model_ref"]))
        ref["export"]["checkpoint_files"]["metadata.json"] = "0" * 64
        write(Path(descriptor["model_ref"]), ref)
        descriptor["model_ref_sha256"] = evaluation.file_hash(Path(descriptor["model_ref"]))
        with self.assertRaisesRegex(ValueError, "declared checkpoint"):
            evaluation._verify_model(descriptor)

    def test_actual_parent_links_join_independent_families(self):
        lineage = records()[2]
        training, dev = self.items[:2]
        train_keys = evaluation._keys(training, replace(lineage, structure_hash="train", parents=()))
        dev_keys = evaluation._keys(dev, replace(lineage, structure_hash="dev", parents=(training["problem_id"],)))
        self.assertTrue(train_keys & dev_keys)

    def test_missing_duplicate_or_changed_raw_cannot_be_graded(self):
        prereg = self.register()
        _, root = self.generate("F0")
        raw = (root / "t1.jsonl").read_bytes()
        (root / "t1.jsonl").write_bytes(raw + raw.split(b"\n")[0] + b"\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluation._raw_rows(prereg, root / "t1.jsonl")
        (root / "t1.jsonl").write_bytes(raw.split(b"\n", 1)[1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluation._raw_rows(prereg, root / "t1.jsonl")
        (root / "t1.jsonl").write_bytes(raw.replace(b"Final answer: 1.25", b"Final answer: 9.99"))
        with self.assertRaisesRegex(ValueError, "generation binding"):
            self.grade(root)


if __name__ == "__main__":
    unittest.main()
