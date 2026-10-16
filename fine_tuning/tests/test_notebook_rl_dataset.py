"""Authored pools and phase-A counts; not benchmark exclusion certification."""
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl.dataset import build_splits, freeze_selection, generate_pool, load_images
from fine_tuning.notebook_rl.generators.common import GeneratedProblem
from fine_tuning.tests.test_notebook_rl_schema import records


def pool(split, i):
    actor, verifier, lineage, *_ = records()
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (i*32, 20, 50)).save(buf, format="PNG")
    raw = buf.getvalue()
    pid = "fixture:"+split
    actor = replace(actor, problem_id=pid, image_paths=(split+".png",), image_sha256=(hashlib.sha256(raw).hexdigest(),))
    return GeneratedProblem(actor, replace(verifier, problem_id=pid),
        replace(lineage, problem_id=pid, split=split, structure_hash=f"{i:016x}"), (raw,))


def phase_report():
    rows = [{"problem_id": f"p{i}", "family": "f", "correct": c, "n": 8, "blind_correct": 0,
             "blind_n": 4, "truncated": 0, "n_choices": 0, "context_ok": True}
            for i, c in enumerate([0, 0, 0, 2, 4, 7])]
    return {"status": "PHASE_A_COMPLETED", "policy_version": 0, "split": "train", "build_id": "fixture",
            "rows": rows, "max_new_tokens": {"1": 3072, "2": 3072, "3": 3072},
            "candidate_ids_sha256": hashlib.sha256(json.dumps(sorted(r["problem_id"] for r in rows),
                                               separators=(",", ":")).encode()).hexdigest()}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pools = {s: [pool(s, i+1)] for i, s in enumerate(("train", "dev", "sealed", "style_sealed"))}

    def test_write_once_all_splits_and_no_implicit_approval(self):
        target = self.root/"build"
        manifest = build_splits(target, self.pools)
        self.assertFalse(manifest["training_allowed"])
        self.assertFalse(manifest["human_render_review"])
        self.assertEqual(manifest["counts"], dict.fromkeys(self.pools, 1))
        self.assertEqual(json.loads((target/"contamination/receipt_v2.json").read_text())["status"], "BLOCKED_NOT_CHECKED")
        for rows in self.pools.values():
            images = load_images(rows[0].actor, target/"images")
            self.assertEqual(images[0].size, (4, 4))
        with self.assertRaises(FileExistsError):
            build_splits(target, self.pools)

    def test_cross_split_structure_and_image_are_rejected_before_write(self):
        original = self.pools["train"][0]
        dev = self.pools["dev"][0]
        self.pools["dev"] = [replace(dev, lineage=replace(dev.lineage, structure_hash=original.lineage.structure_hash))]
        with self.assertRaisesRegex(ValueError, "structure"):
            build_splits(self.root/"bad", self.pools)
        self.assertFalse((self.root/"bad").exists())
        self.pools["dev"] = [replace(dev, actor=replace(dev.actor, image_sha256=original.actor.image_sha256),
                                            image_bytes=original.image_bytes)]
        with self.assertRaisesRegex(ValueError, "image crosses"):
            build_splits(self.root/"bad", self.pools)

    def test_image_corruption_and_symlink_are_rejected(self):
        target = self.root/"build"
        build_splits(target, self.pools)
        actor = self.pools["train"][0].actor
        path = target/"images"/actor.image_paths[0]
        path.write_bytes(path.read_bytes()+b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_images(actor, target/"images")
        alias = self.root/"alias"
        alias.symlink_to(target/"images", target_is_directory=True)
        with self.assertRaises(ValueError):
            load_images(actor, alias)

    def test_source_bounded_generator_checks_every_record_and_determinism(self):
        record = self.pools["train"][0]
        class Module:
            __file__ = __file__
            checks = 0
            @staticmethod
            def generate(**kwargs):
                return replace(record, actor=replace(record.actor, mode=kwargs["mode"]))
            @classmethod
            def self_check(cls, r):
                cls.checks += 1
        with patch("fine_tuning.notebook_rl.dataset.get_generator", return_value=Module):
            generated = generate_pool("fixture", 1, 3407, "train")
        self.assertEqual(Module.checks, 1)
        self.assertEqual(len(generated[0].lineage.source_revision), 64)
        with patch("fine_tuning.notebook_rl.dataset.get_generator", return_value=Module), patch.object(Module,
                "generate", side_effect=[record, replace(record, image_bytes=(b"bad",))]):
            with self.assertRaisesRegex(ValueError, "nondeterministic"):
                generate_pool("fixture", 1, 3407, "train")

    def freeze(self, report=None, **rules):
        report = report or phase_report()
        path = self.root/"phase.json"
        path.write_text(json.dumps(report))
        values = {"max_pass_rate": .75, "max_zero_pass_share": .5, "blind_floor": .25,
                  "truncation_family_max": .1, "max_selected": 384, "min_open_answer_fraction": .6}
        return freeze_selection(path, values | rules, output=self.root/"selection.yaml")

    def test_selection_zero_cap_difficulty_and_write_once(self):
        selected = self.freeze()
        self.assertEqual(selected["selected_train"], ["p0", "p1", "p3", "p4"])
        self.assertEqual({r["reason"] for r in selected["excluded"]}, {"TOO_EASY", "ZERO_PASS_CAP"})
        self.assertFalse(selected["training_allowed"])
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_missing_coverage_heldout_or_bad_protocol_refused(self):
        for change in ({"split": "sealed"}, {"policy_version": 1}, {"candidate_ids_sha256": "a"*64},
                       {"status": "PARTIAL"}, {"max_new_tokens": {"1": 8192, "2": 8192, "3": 8192}}):
            with self.assertRaises(ValueError):
                self.freeze(phase_report() | change)
        self.assertFalse((self.root/"selection.yaml").exists())

    def test_blind_family_truncation_and_open_floor_fail_without_silent_resampling(self):
        for key, value in (("blind_correct", 4), ("truncated", 2), ("n_choices", 10)):
            report = phase_report()
            for row in report["rows"]:
                row[key] = value
            with self.assertRaisesRegex(ValueError, "insufficient"):
                self.freeze(report)


if __name__ == "__main__":
    unittest.main()
