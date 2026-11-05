"""Authored source fixtures only; no upstream images, questions or network."""
from __future__ import annotations

import json
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import MagicMock, patch

from PIL import Image

from fine_tuning.notebook_rl.adapters import hf_circuit, hf_clevr, hf_docs, hf_registry
from fine_tuning.notebook_rl.adapters.hf_common import split_for


def row(**changes):
    value = {"splittype": "train", "file": "d1_fixture_png.rf.abc", "question": "Count the switches.",
             "answer": "2", "qtype": "count", "desc": "FORBIDDEN_TEACHER_DESCRIPTION", "symbol": "switch",
             "bbox": "switch 0.2 0.3 0.1 0.1 . switch 0.6 0.7 0.1 0.1"}
    value.update(changes)
    return value


def candidates(rows):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "questions").mkdir()
        (root / "questions/master_bbox.json").write_text(json.dumps(rows))
        with zipfile.ZipFile(root / "384a.zip", "w") as archive:
            for record in rows:
                if record["splittype"] != "train" or "/" in record["file"]:
                    continue
                directory = "model-inputs-jn" if record["qtype"] == "junction" else "model-inputs"
                name = f"384a/{directory}/train/{record['file']}.jpg"
                if name not in archive.namelist():
                    archive.writestr(name, b"authored-image")
        return list(hf_circuit.iter_circuitvqa(root))


class CircuitSourceTests(unittest.TestCase):
    def test_upstream_train_only_and_original_indices(self):
        records = candidates([row(splittype="test", file="d1_other"), row()])
        self.assertEqual([c.native_id for c in records], ["train:1"])
        self.assertEqual((records[0].answer_type, records[0].gold, records[0].images),
                         ("int", "2", (b"authored-image",)))

    def test_float_count_is_exact_integer_and_bad_counts_rejected(self):
        self.assertEqual(candidates([row(answer="2.0")])[0].gold, "2")
        for answer in ("3.1", "-1", "two", "nan", None):
            self.assertEqual(candidates([row(answer=answer)]), [])

    def test_source_count_annotation_mismatch_is_excluded_without_answer_repair(self):
        self.assertEqual(candidates([row(answer="3")]), [])
        self.assertEqual(candidates([row(bbox=None)]), [])
        self.assertEqual(candidates([row(bbox="malformed")]), [])
        value = candidates([row(answer="1", symbol="voltage.dc", bbox="voltage-dc 0.2 0.3 0.1 0.1")])[0]
        self.assertEqual(value.gold, "1")

    def test_junction_gold_and_image_directory(self):
        for answer, gold in (("yes", "A"), ("no", "B")):
            value = candidates([row(qtype="junction", answer=answer)])[0]
            self.assertEqual((value.choices, value.answer_type, value.gold), (("Yes", "No"), "mcq", gold))
        self.assertEqual(candidates([row(qtype="junction", answer="maybe")]), [])

    def test_teacher_descriptions_and_other_target_types_never_copied(self):
        value = candidates([row()])[0]
        self.assertNotIn("FORBIDDEN", repr(value))
        self.assertEqual(value.question, "Count the switches.")
        for kind in ("position", "value", "unknown"):
            self.assertEqual(candidates([row(qtype=kind)]), [])

    def test_original_heldout_redrawing_family_is_excluded(self):
        self.assertEqual(candidates([row(file="d2_fixture_D1_P1"),
                                     row(file="d2_fixture_D2_P3", splittype="val")]), [])
        self.assertEqual(candidates([row(file="d1_fixture_png.rf.abc"),
                                     row(file="d1_fixture_png.rf.def", splittype="test")]), [])

    def test_family_assignment_preserved_across_redrawings(self):
        values = candidates([row(file="d2_fixture_D1_P1"), row(file="d2_fixture_D2_P2")])
        self.assertEqual(values[0].family_key, values[1].family_key)
        self.assertEqual(split_for(values[0]), split_for(values[1]))

    def test_unsafe_source_filename_rejected(self):
        for filename in ("../secret", "/root/image", ".", ".."):
            with self.assertRaisesRegex(ValueError, "source image name"):
                hf_circuit.circuit_family(filename)

    def test_missing_raw_files_are_not_silently_successful(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                list(hf_circuit.iter_circuitvqa(Path(temp)))

    def test_registry_pins_program_gold_and_upstream_terms(self):
        spec, iterate = hf_registry.sources()["hf-circuitvqa"]
        self.assertIs(iterate, hf_circuit.iter_circuitvqa)
        self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
        self.assertEqual((spec.license, spec.label_provenance), ("CC-BY-SA-3.0", "PROGRAM"))


def clevr_scene():
    return {"image_index": 0, "image_filename": "CLEVR_train_000000.png", "split": "train",
            "objects": [{"color": "red", "shape": "cube", "size": "large", "material": "metal"},
                        {"color": "blue", "shape": "sphere", "size": "small", "material": "metal"},
                        {"color": "green", "shape": "cylinder", "size": "small", "material": "rubber"}],
            "relationships": {"left": [[1, 2], [2], []], "right": [[], [0], [0, 1]]}}


def node(function, inputs=(), values=()):
    return {"function": function, "inputs": list(inputs), "value_inputs": list(values)}


def spatial_program():
    return [node("scene"), node("filter_color", [0], ["red"]), node("unique", [1]),
            node("relate", [2], ["left"]), node("filter_material", [3], ["metal"]), node("count", [4]),
            node("scene"), node("count", [6]), node("less_than", [5, 7])]


class ClevrSourceTests(unittest.TestCase):
    def test_native_program_executes_spatial_chain_and_comparisons(self):
        self.assertEqual(hf_clevr.program_answer(clevr_scene(), spatial_program()), "yes")
        prefix = spatial_program()[:6]
        for function, expected in (("greater_than", "no"), ("equal_integer", "no")):
            program = prefix + [node("scene"), node("count", [6]), node(function, [5, 7])]
            self.assertEqual(hf_clevr.program_answer(clevr_scene(), program), expected)

    def test_native_set_attribute_and_equality_operations(self):
        scene = clevr_scene()
        base = [node("scene"), node("filter_color", [0], ["red"]), node("unique", [1])]
        for attribute in ("color", "shape", "size", "material"):
            value = scene["objects"][0][attribute]
            self.assertEqual(hf_clevr.program_answer(scene, base + [node(f"query_{attribute}", [2])]), value)
            self.assertEqual(hf_clevr.program_answer(scene, base + [node(f"same_{attribute}", [2]), node("count", [3])]),
                             "1" if attribute == "material" else "0")
            program = base + [node(f"query_{attribute}", [2]), node(f"query_{attribute}", [2]),
                              node(f"equal_{attribute}", [3, 4])]
            self.assertEqual(hf_clevr.program_answer(scene, program), "yes")
        for function, expected in (("union", "3"), ("intersect", "2")):
            program = base + [node("relate", [2], ["left"]), node(function, [0, 3]), node("count", [4])]
            self.assertEqual(hf_clevr.program_answer(scene, program), expected)
        self.assertEqual(hf_clevr.program_answer(scene, [node("scene"), node("exist", [0])]), "yes")

    def test_unsupported_nonunique_forward_or_nonscalar_program_is_rejected(self):
        for program in ([node("scene")], [node("scene"), node("unique", [0])],
                        [node("count", [0])], [node("scene"), node("filter_secret", [0], ["x"])]):
            with self.subTest(program=program), self.assertRaises(ValueError):
                hf_clevr.program_answer(clevr_scene(), program)

    def test_original_question_gold_scene_and_train_images_are_bound(self):
        scene = clevr_scene()
        question = {"image_index": 0, "image_filename": scene["image_filename"], "split": "train",
                    "question_index": 7, "program": spatial_program(), "question": "Are fewer metal objects left?",
                    "answer": "yes", "description": "FORBIDDEN_TEACHER_TARGET"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ("questions", "scenes", "images/train"):
                (root / directory).mkdir(parents=True)
            (root / "scenes/CLEVR_train_scenes.json").write_text(json.dumps({"scenes": [scene]}))
            path = root / "questions/CLEVR_train_questions.json"
            path.write_text(json.dumps({"questions": [question, dict(question, split="val", question_index=9)]}))
            (root / "images/train" / scene["image_filename"]).write_bytes(b"authored-image")
            values = list(hf_clevr.iter_clevr(root))
            self.assertEqual(len(values), 1)
            value = values[0]
            self.assertEqual((value.native_id, value.gold, value.choices, value.images),
                             ("train:7", "A", ("Yes", "No"), (b"authored-image",)))
            self.assertNotIn("FORBIDDEN", repr(value))
            path.write_text(json.dumps({"questions": [dict(question, answer="no")]}))
            self.assertEqual(list(hf_clevr.iter_clevr(root)), [])  # never repair the native label
            path.write_text(json.dumps({"questions": [dict(question, image_index=1)]}))
            with self.assertRaisesRegex(ValueError, "binding"):
                list(hf_clevr.iter_clevr(root))

    def test_registry_pins_original_release_distribution(self):
        spec, iterate = hf_registry.sources()["hf-clevr"]
        self.assertIs(iterate, hf_clevr.iter_clevr)
        self.assertEqual((spec.repo, spec.license, spec.label_provenance),
                         ("HuggingFaceM4/clevr", "CC-BY-4.0", "PROGRAM"))
        self.assertRegex(spec.revision, r"^[a-f0-9]{40}$")


class OriginalPdfSourceTests(unittest.TestCase):
    def test_native_page_count_preserves_both_pages_in_order_and_rejects_missing_pages(self):
        document = MagicMock()
        document.__enter__.return_value = document
        document.__len__.return_value = 2
        with patch.dict("sys.modules", {"pypdfium2": SimpleNamespace(PdfDocument=lambda raw: document)}), patch.object(
                hf_docs, "render_pdf_page", side_effect=[b"first-original-page", b"second-original-page"]) as render:
            self.assertEqual(hf_docs.render_pdf_pages(b"native-two-page-pdf", 2),
                             (b"first-original-page", b"second-original-page"))
            self.assertEqual([call.args[1] for call in render.call_args_list], [1, 2])
            with self.assertRaisesRegex(ValueError, "page count mismatch"):
                hf_docs.render_pdf_pages(b"native-two-page-pdf", 1)
            for invalid in (0, 3, True):
                with self.assertRaises(ValueError):
                    hf_docs.render_pdf_pages(b"native-pdf", invalid)

    def test_renderer_closes_page_explicitly_without_requiring_page_context_manager(self):
        closed = []

        class Page:
            def get_size(self):
                return 200, 300

            def render(self, scale):
                self.scale = scale
                return SimpleNamespace(to_pil=lambda: Image.new("RGB", (3, 4), "blue"),
                                       close=lambda: closed.append("bitmap"))

            def close(self):
                closed.append("page")

        page = Page()

        class Document:
            def __init__(self, raw):
                self.raw = raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.append("document")

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return page

        with patch.dict("sys.modules", {"pypdfium2": SimpleNamespace(PdfDocument=Document)}), patch.object(
                hf_docs, "version", return_value=hf_docs.PDFIUM_VERSION):
            raw = hf_docs.render_pdf_page(b"authored-pdf", 1)
        self.assertTrue(raw.startswith(b"\x89PNG"))
        self.assertGreater(page.scale, 1)
        self.assertEqual(closed, ["bitmap", "page", "document"])

    def test_pdf_keeps_native_question_scale_gold_and_old_report_split(self):
        from fine_tuning.notebook_rl.data_tests.test_hf_docs import _tat_entry, _tat_q, _write_tatdqa

        entries = [_tat_entry(f"report{i}", 1, [_tat_q(f"q{i}", "What is the total?", i, scale="million")])
                   for i in range(30)]
        probes = [hf_docs._tatdqa_candidate(e["doc"]["uid"], e["doc"]["source"], e["questions"][0], b"native")
                  for e in entries]
        expected = {c.native_id for c in probes if split_for(c) == "train"}
        self.assertGreater(len(expected), 0)
        self.assertLess(len(expected), 30)
        members = {f"train/{e['doc']['uid']}.pdf": b"authored-original-pdf" for e in entries}
        members.update({f"train/{e['doc']['uid']}.json": json.dumps({"pages": [{}, {}]}).encode() for e in entries})
        with tempfile.TemporaryDirectory() as temp, patch.object(hf_docs, "render_pdf_pages", return_value=(b"first-native-page", b"second-native-page")) as render:
            values = list(hf_docs.iter_tatdqa_pdf(_write_tatdqa(Path(temp), entries, members)))
        self.assertEqual({c.native_id for c in values}, expected)
        self.assertEqual(render.call_args.args, (b"authored-original-pdf", 2))
        for candidate in values:
            self.assertEqual(candidate.source, "hf-tatdqa-pdf")
            self.assertEqual(candidate.images, (b"first-native-page", b"second-native-page"))
            self.assertIn("Answer scale should be: million.", candidate.question)
            self.assertNotIn("SECRET", repr(candidate))
            original = next(c for c in probes if c.native_id == candidate.native_id)
            self.assertEqual((candidate.gold, candidate.accept), (original.gold, original.accept))

    def test_one_native_question_per_report_chosen_without_gold_priority(self):
        from fine_tuning.notebook_rl.data_tests.test_hf_docs import _tat_entry, _tat_q, _write_tatdqa

        entries = [_tat_entry(f"r{i}", 1, [_tat_q("first", "Total?", 5), _tat_q("second", "Count?", 2)]) for i in range(10)]
        members = {f"train/{e['doc']['uid']}.pdf": b"authored-pdf" for e in entries}
        members.update({f"train/{e['doc']['uid']}.json": json.dumps({"pages": [{}]}).encode() for e in entries})
        with tempfile.TemporaryDirectory() as temp, patch.object(hf_docs, "render_pdf_pages", return_value=(b"native-raster",)):
            root = _write_tatdqa(Path(temp), entries, members)
            values = list(hf_docs.iter_tatdqa_pdf(root))
            self.assertEqual(values, list(hf_docs.iter_tatdqa_pdf(root)))
        self.assertEqual(len(values), len({c.family_key for c in values}))

    def test_three_page_native_documents_are_omitted_without_truncation(self):
        from fine_tuning.notebook_rl.data_tests.test_hf_docs import _tat_entry, _tat_q, _write_tatdqa

        entries = [_tat_entry(f"r{i}", 2, [_tat_q("q", "Total?", 5)]) for i in range(10)]
        members = {f"train/{e['doc']['uid']}.pdf": b"authored-three-page-pdf" for e in entries}
        members.update({f"train/{e['doc']['uid']}.json": json.dumps({"pages": [{}, {}, {}]}).encode() for e in entries})
        with tempfile.TemporaryDirectory() as temp, patch.object(hf_docs, "render_pdf_pages") as render:
            self.assertEqual(list(hf_docs.iter_tatdqa_pdf(_write_tatdqa(Path(temp), entries, members))), [])
        render.assert_not_called()


class NativeTableSourceTests(unittest.TestCase):
    def source(self, *, answer_from="table", rel_paragraphs=None, mirror_gold="2.5", duplicate=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from fine_tuning.notebook_rl.adapters import hf_tatqa_table as module

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        repo = root / "HuggingFaceM4__the_cauldron"
        author = root / "NExTplusplus__TAT-QA/dataset_raw/tatqa_dataset_train.json"
        author.parent.mkdir(parents=True)
        question = {"uid": "native-q", "question": "What is the average?", "answer": 2.5,
                    "answer_from": answer_from, "answer_type": "arithmetic", "scale": "percent",
                    "rel_paragraphs": [] if rel_paragraphs is None else rel_paragraphs,
                    "derivation": "FORBIDDEN_HUMAN_DERIVATION"}
        entry = {"table": {"uid": "authored-table", "table": [["year", "value"], ["2019", "2.5"]]},
                 "paragraphs": [{"order": 1, "text": "FORBIDDEN_PARAGRAPH"}], "questions": [question]}
        author.write_text(json.dumps([entry, entry] if duplicate else [entry]))
        image = io.BytesIO()
        Image.new("RGB", (4, 5), "blue").save(image, format="PNG")
        row = {"images": [{"bytes": image.getvalue(), "path": None}], "texts": [{
            "user": "FORBIDDEN_PARAGRAPH\n" + module.question_text(question),
            "assistant": "FORBIDDEN_MIRROR_DERIVATION\nAnswer: " + mirror_gold + ".", "source": "authored"}]}
        schema = pa.schema([("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
                            ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string()), ("source", pa.string())])))])
        path = repo / module.MIRROR_SHARD
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row], schema=schema), path)
        self.enterContext(patch.object(module, "AUTHOR_SHA256", hashlib.sha256(author.read_bytes()).hexdigest()))
        self.enterContext(patch.object(module, "MIRROR_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()))
        return module, repo, author, path

    def test_native_table_only_label_and_scale_without_context_or_derivation(self):
        module, repo, *_ = self.source()
        records = list(module.iter_table_records(repo))
        self.assertEqual(len(records), 1)
        candidate, origin = records[0]
        self.assertEqual((candidate.question, candidate.gold, candidate.answer_type),
                         ("What is the average?\nAnswer scale should be: percent.", "2.5", "decimal"))
        self.assertNotIn("FORBIDDEN", repr(candidate))
        self.assertEqual((origin["original_table_uid"], origin["native_answer_from"]), ("authored-table", "table"))
        expected = [candidate] if split_for(candidate) == origin["canonical_split"] else []
        self.assertEqual(list(module.iter_tatqa_table(repo)), expected)

    def test_native_mismatch_context_dependency_or_ambiguous_join_never_repaired(self):
        for kwargs in ({"mirror_gold": "250"}, {"answer_from": "text"}, {"rel_paragraphs": ["1"]}, {"duplicate": True}):
            with self.subTest(kwargs=kwargs):
                module, repo, *_ = self.source(**kwargs)
                self.assertEqual(list(module.iter_table_records(repo)), [])

    def test_pinned_source_changes_are_rejected(self):
        module, repo, author, _ = self.source()
        author.write_text(author.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            list(module.iter_table_records(repo))

    def test_new_alias_cannot_hide_source_native_family_image_or_heldout(self):
        from fine_tuning.notebook_rl.adapters.hf_common import family_digest

        module, repo, *_ = self.source()
        candidate, origin = list(module.iter_table_records(repo))[0]
        # Select an authored table UID that preserves the fixed legacy split for this branch.
        for i in range(100):
            candidate = replace(candidate, family_key=f"authored-table-{i}")
            if split_for(candidate) == origin["canonical_split"]:
                break
        origin = dict(origin, original_table_uid=candidate.family_key)
        empty = {"blocked_table_uids": set(), "blocked_families": set(), "old_native_ids": set(),
                 "blocked_raw_images": set(), "blocked_png_images": set(), "blocked_pixels": set(),
                 "png_sha256": "1" * 64, "pixel_sha256": "2" * 64}
        self.assertIsNone(module.admission_block_reason(candidate, origin, **empty))
        cases = [{"blocked_table_uids": {candidate.family_key}},
                 {"blocked_families": {("hf-tatqa", origin["canonical_family_digest"])}},
                 {"old_native_ids": {("hf-tatqa", candidate.native_id)}},
                 {"blocked_raw_images": {origin["image_raw_sha256"]}},
                 {"blocked_png_images": {"1" * 64}}, {"blocked_pixels": {"2" * 64}}]
        for change in cases:
            with self.subTest(change=change):
                self.assertIsNotNone(module.admission_block_reason(candidate, origin, **(empty | change)))
        for i in range(100, 200):
            other = replace(candidate, family_key=f"authored-heldout-{i}")
            if split_for(other) != origin["canonical_split"]:
                break
        self.assertEqual(module.admission_block_reason(other, dict(origin, original_table_uid=other.family_key), **empty),
                         "ANCESTRAL_SPLIT_DISAGREEMENT")
        self.assertNotEqual(family_digest(candidate), origin["canonical_family_digest"])


if __name__ == "__main__":
    unittest.main()
