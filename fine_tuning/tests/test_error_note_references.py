"""Authored CPU fixtures only: small temp parquet/json files, no real dataset, MMMU or network."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from fine_tuning.error_note import references as ref


def _item(source: str, **fields) -> dict:
    return {"source": source, "annotations": [{"field": k, "value": v} for k, v in fields.items()]}


class NoteReferenceTests(unittest.TestCase):
    def test_fields_sha256_pins_config(self):
        self.assertEqual(ref.fields_sha256(), ref.fields_sha256())
        self.assertEqual(len(ref.fields_sha256()), 64)
        with patch.object(ref, "REFERENCE_MAX_CHARS", 10):
            changed = ref.fields_sha256()
        self.assertNotEqual(changed, ref.fields_sha256())

    def test_scienceqa_keeps_listed_fields_in_order_and_skips_blank(self):
        item = _item("hf-scienceqa", solution="Because X.", question="never shown", lecture="  Intro text.\nLine 2  ")
        self.assertEqual(ref.note_reference(item), "Lecture: Intro text.\nLine 2\n\nSolution: Because X.")
        self.assertEqual(ref.note_reference(_item("hf-scienceqa", lecture="  ", solution=None)), None)
        self.assertEqual(ref.note_reference(_item("hf-scienceqa", question="q", hint="h")), None)

    def test_clevr_scene_and_program_native_and_repr(self):
        scene = {"objects": [{"size": "small", "color": "blue", "material": "rubber", "shape": "sphere",
                              "pixel_coords": [286, 214, 8.78]},
                             {"size": "large", "color": "red", "material": "metal", "shape": "cube",
                              "pixel_coords": [100.0, 50.5, 3.0]}], "directions": {"left": [0, 1, 0]}}
        program = [{"function": "scene", "inputs": [], "value_inputs": []},
                   {"function": "filter_color", "inputs": [0], "value_inputs": ["purple"]}]
        want = ("Scene objects:\nobject 0: small blue rubber sphere at (286, 214)\n"
                "object 1: large red metal cube at (100, 50.5)\n\n"
                "Question program:\n0: scene\n1: filter_color(purple) <- 0")
        self.assertEqual(ref.note_reference(_item("hf-clevr", scene=scene, program=program, answer="yes")), want)
        self.assertEqual(ref.note_reference(_item("hf-clevr", scene=repr(scene), program=repr(program))), want)

    def test_clevr_unparsable_or_partial_scene_is_skipped(self):
        program = [{"function": "scene", "inputs": [], "value_inputs": []}]
        for bad in ("{not a dict", {"objects": [{"size": "small"}]}, {"objects": []}, "plain words"):
            self.assertEqual(ref.note_reference(_item("hf-clevr", scene=bad, program=program)),
                             "Question program:\n0: scene")
        self.assertIsNone(ref.note_reference(_item("hf-clevr", scene="{broken")))

    def test_slake_placeholder_triple_skipped_concrete_kept(self):
        base = {"modality": "CT", "location": "Abdomen"}
        for placeholder in (["_", "_", "_"], ["vhead", "_", "_"], "['_', '_', '_']"):
            self.assertEqual(ref.note_reference(_item("hf-slake", triple=placeholder, **base)),
                             "Modality: CT\n\nLocation: Abdomen")
        self.assertEqual(ref.note_reference(_item("hf-slake", triple=["liver", "position", "abdomen"], **base)),
                         "Modality: CT\n\nLocation: Abdomen\n\nTriple: liver, position, abdomen")
        self.assertEqual(ref.note_reference(_item("hf-slake", triple="['liver', 'position', 'abdomen']")),
                         "Triple: liver, position, abdomen")

    def test_chebi_labels_keep_smiles_uppercase(self):
        item = _item("hf-chebi-role", CID="1", description="A molecule.", SMILES="CCO")
        self.assertEqual(ref.note_reference(item), "Description: A molecule.\n\nSMILES: CCO")

    def test_supplement_item_fills_ptbxl_and_ssmr(self):
        supplement = {"source": "hf-ptbxl", "native_id": "ecg:1",
                      "annotations": [{"field": "report", "value": "sinus rhythm"},
                                      {"field": "statements", "value": "NORM, SR"}]}
        self.assertEqual(ref.note_reference(_item("hf-ptbxl"), supplement),
                         "Cardiologist report: sinus rhythm\n\nDiagnostic statements: NORM, SR")
        self.assertIsNone(ref.note_reference(_item("hf-ptbxl")))
        music = {"annotations": [{"field": "abc_context", "value": "X:1\nK:C"}]}
        self.assertEqual(ref.note_reference(_item("hf-ssmr"), music), "Score (ABC notation): X:1\nK:C")
        with self.assertRaises(ValueError):
            ref.note_reference(_item("hf-ssmr"), supplement)

    def test_provenance_value_wins_over_supplement(self):
        supplement = {"annotations": [{"field": "report", "value": "from supplement"}]}
        self.assertEqual(ref.note_reference(_item("hf-ptbxl", report="from provenance"), supplement),
                         "Cardiologist report: from provenance")

    def test_unlisted_source_returns_none(self):
        self.assertIsNone(ref.note_reference(_item("hf-chartqa", answer="x", lecture="y")))
        self.assertIsNone(ref.note_reference({"source": "hf-scienceqa"}))

    def test_cap_drops_whole_trailing_fields_never_cuts(self):
        lecture, solution = "L" * 3000, "S" * 1500
        item = _item("hf-scienceqa", lecture=lecture, solution=solution)
        self.assertEqual(ref.note_reference(item), f"Lecture: {lecture}")
        fits = _item("hf-scienceqa", lecture="L" * 1000, solution="S" * 1000)
        self.assertEqual(len(ref.note_reference(fits)), len("Lecture: ") + 1000 + 2 + len("Solution: ") + 1000)
        self.assertIsNone(ref.note_reference(_item("hf-scienceqa", lecture="L" * 4001, solution="S")))
        self.assertLessEqual(len(ref.note_reference(item)), ref.REFERENCE_MAX_CHARS)


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.raw = self.base / "raw"
        ptb = self.raw / "tirandazdylan__ptb-xl-ecg" / "data"
        ssmr = self.raw / "Sylence__SSMR-Bench" / "data"
        ptb.mkdir(parents=True)
        ssmr.mkdir(parents=True)
        for name, ids in (("train-00000-of-00002.parquet", [1, 2]), ("train-00001-of-00002.parquet", [3, 4])):
            pq.write_table(pa.table({"ecg_id": ids, "report": [f"report {i}" if i != 4 else None for i in ids],
                                     "statements": [["NORM", f"S{i}"] if i != 4 else [] for i in ids],
                                     "signal": [[0.0, 1.0]] * 2}), ptb / name)
        pq.write_table(pa.table({"abc_context": ["X:0\nK:C", None, "X:2\nK:G"], "question": ["a", "b", "c"]}),
                       ssmr / "Train_visual-00000-of-00001.parquet")
        rows = [("p1", "hf-ptbxl", "ecg:3"), ("p2", "hf-ptbxl", "ecg:4"), ("p3", "hf-ptbxl", "ecg:99"),
                ("p4", "hf-ssmr", "Train_visual:2"), ("p5", "hf-ssmr", "Train_visual:1"),
                ("p6", "hf-ssmr", "Train_visual:7"), ("p7", "hf-scienceqa", "train:5"), ("p8", "hf-ptbxl", "ecg:1")]
        self.manifest = self.base / "manifest.json"
        self.manifest.write_text(json.dumps({"items": [{"problem_id": p, "source": s, "native_id": n}
                                                       for p, s, n in rows + [("unordered", "hf-ssmr", "Train_visual:0")]]}))
        self.order = self.base / "order.json"
        self.order.write_text(json.dumps({"items": [{"problem_id": p} for p, _, _ in rows]}))

    def build(self) -> dict:
        return ref.build_supplement(manifest_path=self.manifest, order_path=self.order, raw_root=self.raw)

    def test_build_matches_rows_records_missing_and_hashes(self):
        value = self.build()
        self.assertEqual(value["version"], ref.SUPPLEMENT_VERSION)
        self.assertEqual(value["fields_sha256"], ref.fields_sha256())
        self.assertEqual(sorted(value["items"]), ["p1", "p4", "p8"])
        self.assertEqual(value["missing"], ["p2", "p3", "p5", "p6"])  # blank report+empty statements, absent, blank, absent
        ptb = value["items"]["p1"]
        self.assertEqual((ptb["source"], ptb["native_id"]), ("hf-ptbxl", "ecg:3"))
        fields = {a["field"]: a for a in ptb["annotations"]}
        self.assertEqual((fields["report"]["value"], fields["statements"]["value"]), ("report 3", "NORM, S3"))
        shard = self.raw / "tirandazdylan__ptb-xl-ecg/data/train-00001-of-00002.parquet"
        self.assertEqual(fields["report"]["location"],
                         "raw:tirandazdylan__ptb-xl-ecg/data/train-00001-of-00002.parquet#ecg:3:report")
        self.assertEqual(fields["report"]["file_sha256"], ref.file_sha(shard))
        self.assertEqual(fields["report"]["sha256"], ref.text_sha256("report 3"))
        music = value["items"]["p4"]["annotations"][0]
        self.assertEqual((music["field"], music["value"]), ("abc_context", "X:2\nK:G"))
        self.assertEqual(music["location"], "raw:Sylence__SSMR-Bench/data/Train_visual-00000-of-00001.parquet#Train_visual:2:abc_context")
        self.assertEqual(value["items"]["p8"]["annotations"][0]["value"], "report 1")

    def test_built_items_feed_note_reference(self):
        item = self.build()["items"]["p1"]
        self.assertEqual(ref.note_reference({"source": "hf-ptbxl", "annotations": []}, item),
                         "Cardiologist report: report 3\n\nDiagnostic statements: NORM, S3")

    def test_unknown_ordered_problem_or_bad_root_raises(self):
        self.order.write_text(json.dumps({"items": [{"problem_id": "ghost"}]}))
        with self.assertRaises(ValueError):
            self.build()
        with self.assertRaises(ValueError):
            ref.build_supplement(manifest_path=self.manifest, order_path=self.order, raw_root=self.base / "nope")

    def test_write_read_round_trip_and_tamper_checks(self):
        value = self.build()
        path = self.base / "out" / "supplement.json"
        sha = ref.write_supplement(value, path)
        self.assertEqual(sha, ref.file_sha(path))
        self.assertEqual(ref.read_supplement(path, sha), value["items"])
        self.assertEqual(ref.write_supplement(value, path), sha)  # canonical: deterministic bytes
        with self.assertRaises(ValueError):
            ref.read_supplement(path, "0" * 64)
        wrong = self.base / "wrong.json"
        wrong_sha = ref.write_supplement({**value, "version": "other"}, wrong)
        with self.assertRaises(ValueError):
            ref.read_supplement(wrong, wrong_sha)

    def test_cli_build_prints_status_json(self):
        out = self.base / "cli" / "supplement.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = ref.main(["build", "--manifest", str(self.manifest), "--order", str(self.order),
                             "--raw-root", str(self.raw), "--out", str(out)])
        report = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual((report["status"], report["items"], report["by_source"]),
                         ("OK", 3, {"hf-ptbxl": 2, "hf-ssmr": 1}))
        self.assertEqual(report["sha256"], ref.file_sha(out))


if __name__ == "__main__":
    unittest.main()
