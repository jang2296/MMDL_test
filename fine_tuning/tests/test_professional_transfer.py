"""Offline, independently authored fixtures; no HF payload/model/GPU evaluation."""
import copy
import io
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from fine_tuning import audit, professional_transfer as transfer
from fine_tuning.evaluate import score_external, verify_cohort_images, verify_transfer_checkpoint
from fine_tuning.modules import BASE_ID, BASE_REVISION


def png(seed):
    image = Image.frombytes("RGB", (24, 24), random.Random(seed).randbytes(24 * 24 * 3))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    image.close()
    return stream.getvalue()


def source_rows():
    return [{"query": f"Read independent fixture item {i}.", "image": {"bytes": png(i)},
             "label": [str(i)], "human_or_machine": 0} for i in range(4)]


def empty_index():
    return {"pixels": set(), "dhashes": set(), "texts": set(), "ancestors": set(), "evidence": []}


class ProfessionalTransferTests(unittest.TestCase):
    def fixture_training(self, root, chartqa=False):
        (root / "images").mkdir()
        (root / "images" / "own.png").write_bytes(png(99))
        source = root / "source"
        source.mkdir()
        for name in ("train.py", "modules.py"):
            (source / name).write_text("independently authored offline fixture; never executed")
        row = {"id": "own-train", "family_id": "own-family", "root_id": "own-root",
            "split": "train", "training_allowed": True, "benchmark_derived": False,
            "source_kind": "PUBLIC_INDEPENDENT_SYNTHETIC", "lineage": ["own"],
            "user_text": "Own fixture question.", "target_text": "Own fixture target.",
            "image_paths": ["images/own.png"]}
        audit.write_rows(root / "records.jsonl", [row])
        manifest = {"status": "APPROVED", "training_allowed": True, "benchmark_derived": False,
            "record_count": 1, "records_sha256": audit.sha(root / "records.jsonl"),
            "contamination_audit": {"status": "PASS"}, "known_parent_ids": ["own"],
            "sources": {"own": {"parents": [], "rights_approved": True, "source_known": True,
                "benchmark_derived": False, "repo_id": "HuggingFaceM4/ChartQA" if chartqa else "own"}},
            "files": {"images/own.png": audit.sha(root / "images/own.png")},
            "source_sha256": {name: audit.sha(source / name) for name in ("train.py", "modules.py")}}
        audit.dump(root / "manifest.json", manifest)
        metadata = {"base_id": BASE_ID, "base_revision": BASE_REVISION, "epoch_exposure_complete": True,
            "manifest_sha256": audit.sha(root / "manifest.json"), "records_sha256": manifest["records_sha256"],
            "planned_records": 1, "exposure": {"records": 1}, "source_sha256": manifest["source_sha256"]}
        audit.dump(root / "metadata.json", metadata)
        return {"records": str(root / "records.jsonl"), "manifest": str(root / "manifest.json"),
            "checkpoint_metadata": str(root / "metadata.json"),
            "checkpoint_metadata_sha256": audit.sha(root / "metadata.json"),
            "source_root": str(source), "image_root": str(root)}

    def test_training_index_checks_actual_closure_and_frozen_source_without_loading_weights(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.fixture_training(root)
            index = transfer.training_index([spec])
            self.assertEqual(index["ancestors"], {"own"})
            self.assertEqual(len(index["pixels"]), 1)
            self.assertEqual(index["evidence"][0]["records"], 1)
            (root / "source/train.py").write_text("changed fixture source")
            with self.assertRaisesRegex(ValueError, "frozen training source"):
                transfer.training_index([spec])

    def test_training_image_and_metadata_tamper_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self.fixture_training(root)
            (root / "images/own.png").write_bytes(png(100))
            with self.assertRaisesRegex(ValueError, "Trained image changed"):
                transfer.training_index([spec])
            (root / "metadata.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                transfer.training_index([spec])

    def test_chartqa_training_ancestor_blocks_source_holdout_claim(self):
        with TemporaryDirectory() as temporary:
            spec = self.fixture_training(Path(temporary), chartqa=True)
            with self.assertRaisesRegex(ValueError, "not held out"):
                transfer.training_index([spec])

    def test_multiple_training_registries_cannot_hide_changed_ancestry(self):
        with TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a", Path(temporary) / "b"
            a.mkdir()
            b.mkdir()
            specs = [self.fixture_training(a), self.fixture_training(b, chartqa=True)]
            with self.assertRaisesRegex(ValueError, "not held out"):
                transfer.training_index(specs)

    def test_selection_is_gold_blind_order_invariant_and_one_per_shared_chart(self):
        class GoldGuard(dict):
            def __getitem__(self, key):
                if key == "label":
                    raise AssertionError("Gold accessed during selection")
                return super().__getitem__(key)
        rows = [GoldGuard(row) for row in source_rows()]
        rows.append(GoldGuard(dict(rows[0], query="Another question on the same chart.")))
        observations = [transfer.observation(row, i) for i, row in enumerate(rows)]
        selected, _ = transfer.select(observations, empty_index(), 4)
        reordered, _ = transfer.select(list(reversed(observations)), empty_index(), 4)
        self.assertEqual(selected, reordered)
        self.assertEqual(len({row["family_id"] for row in selected}), 4)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            transfer.select(observations, empty_index(), 5)

    def test_train_question_pixel_and_near_image_matches_are_excluded_without_gold(self):
        observed = [transfer.observation(row, i) for i, row in enumerate(source_rows())]
        for key, value, reason in [("texts", audit.text_hash(observed[0]["query"]), "TRAIN_QUESTION_MATCH"),
                ("pixels", observed[0]["pixels_sha256"], "TRAIN_IMAGE_PIXEL_MATCH"),
                ("dhashes", observed[0]["dhash"], "TRAIN_IMAGE_NEAR_MATCH")]:
            index = empty_index()
            index[key].add(value)
            selected, excluded = transfer.select(observed, index, 1)
            self.assertNotEqual(selected[0]["id"], observed[0]["id"])
            self.assertIn({"id": observed[0]["id"], "reason": reason}, excluded)

    def test_nonhuman_and_unknown_provenance_are_not_accepted_as_human(self):
        row = source_rows()[0]
        self.assertIsNone(transfer.observation(dict(row, human_or_machine=1), 0))
        for value in (True, "human", None, 2):
            with self.assertRaisesRegex(ValueError, "class-label"):
                transfer.observation(dict(row, human_or_machine=value), 0)

    def test_fixture_cohort_separates_gold_binds_images_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"
            rows = source_rows()
            manifest = transfer.freeze(rows, empty_index(), output, {"kind": "OFFLINE_FIXTURE_NOT_REAL_SOURCE"}, 2)
            verify_cohort_images(output, manifest)
            first_inputs = audit.read_rows(output / "inputs.jsonl")
            self.assertTrue(all(not item["training_allowed"] for item in first_inputs))
            self.assertTrue(all(not {"label", "target", "answer", "family_id"} & item.keys() for item in first_inputs))
            self.assertEqual(len(manifest["ids"]), 2)
            altered = copy.deepcopy(rows)
            for row in altered:
                row["label"] = ["changed gold, not an input"]
            other = Path(temporary) / "other"
            changed = transfer.freeze(altered, empty_index(), other, {"kind": "OFFLINE_FIXTURE_NOT_REAL_SOURCE"}, 2)
            self.assertEqual(changed["inputs_sha256"], manifest["inputs_sha256"])
            self.assertEqual(changed["selection_sha256"], manifest["selection_sha256"])
            self.assertNotEqual(changed["gold_sha256"], manifest["gold_sha256"])
            with self.assertRaises(FileExistsError):
                transfer.freeze(rows, empty_index(), output, {}, 2)

    def test_selected_invalid_gold_aborts_without_changing_sample(self):
        rows = source_rows()
        chosen, _ = transfer.select([transfer.observation(row, i) for i, row in enumerate(rows)], empty_index(), 1)
        rows[chosen[0]["source_index"]]["label"] = []
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "cohort"
            with self.assertRaisesRegex(ValueError, "gold invalid"):
                transfer.freeze(rows, empty_index(), output, {}, 1)
            self.assertFalse(output.exists())

    def test_remote_guard_runs_before_training_reads_or_hf_downloads(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(transfer, "download_exact") as download:
            with self.assertRaisesRegex(ValueError, "RunPod-only"):
                transfer.prepare("missing", "missing", "missing")
        download.assert_not_called()

    def test_checkpoint_must_match_actual_frozen_training_exclusion_before_model_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            good, unrelated = root / "good", root / "unrelated"
            good.mkdir()
            unrelated.mkdir()
            (good / "metadata.json").write_text('owned frozen trained metadata fixture')
            (unrelated / "metadata.json").write_text('unrelated training metadata fixture')
            manifest = {"evaluation_domain": "PUBLISHED_CHARTQA_HUMAN_SOURCE_TRANSFER",
                "training_evidence": [{"checkpoint_metadata_sha256": audit.sha(good / "metadata.json")}]}
            verify_transfer_checkpoint(manifest, None)
            verify_transfer_checkpoint(manifest, good)
            with self.assertRaisesRegex(ValueError, "not bound"):
                verify_transfer_checkpoint(manifest, unrelated)

    def test_json_number_precision_is_not_rounded_into_a_false_positive(self):
        for submitted, reference in [("0.10000000000000001", "0.1"),
                                     ("9007199254740993.0", "9007199254740992")]:
            gold = {"metric_kind": "chartqa_human_exact", "target": '{"answers":["' + reference + '"]}'}
            result = score_external('{"answer":' + submitted + '}', gold, "stop")
            self.assertFalse(result["correct"])
            self.assertEqual(result["extracted_answer"], submitted)

    def test_one_submitted_scalar_is_extracted_without_reference_and_not_rescued(self):
        gold = {"metric_kind": "chartqa_human_exact", "target": '{"answers":["50"]}'}
        self.assertTrue(score_external('{"answer":"50.0"}', gold, "stop")["correct"])
        before = score_external('{"answer":75}', gold, "stop")
        self.assertFalse(before["correct"])
        after = score_external('{"answer":75}', dict(gold, target='{"answers":["75"]}'), "stop")
        self.assertEqual(before["extracted_answer"], after["extracted_answer"])
        self.assertTrue(after["correct"])
        for raw in ('{"answer":"50 or75"}', '{"answer":"the value is50"}', '{"answer":"50 V"}', '{"answer":52.4}'):
            self.assertFalse(score_external(raw, gold, "stop")["correct"])
        self.assertFalse(score_external('{"answer":1e999}', gold, "stop")["correct"])
        for raw in ('{"answer":true}', '{"answer":[50,75]}', '{"answer":NaN}', '{"answer":50,"answer":75}',
                    '{"answer":50,"reason":"but75"}', '{"answer":50} Actually75.', '{"answer":50'):
            self.assertEqual(score_external(raw, gold, "length")["parse_status"], "NO_PARSE")
        text = dict(gold, target='{"answers":["North America"]}')
        self.assertTrue(score_external('{"answer":" north america "}', text, "stop")["correct"])


if __name__ == "__main__":
    unittest.main()
