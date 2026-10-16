"""Authored native-schema fixtures; no real dataset rows, images, or approval evidence."""
import copy
import hashlib
import unittest

from fine_tuning.notebook_rl.adapters import geometry
from fine_tuning.notebook_rl.answers import compare
from fine_tuning.notebook_rl.observations import compare_slots, parse_observations
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json, to_json


def provenance(source="geometry3k"):
    result = {"native_id": "17", "upstream_split": "train",
              "source_revision": geometry.GEOMETRY3K_REVISION if source == "geometry3k" else geometry.PGPS_DATA_REVISION,
              "label_provenance": "HUMAN", "annotation_provenance": "HUMAN",
              "model_touched": False, "benchmark_derived": False, "image_sha256": "a" * 64}
    if source == "pgps9k":
        result.update({"upstream_origin": "Geometry3K", "parentage_complete": True,
                       "geometry3k_parent_ids": ["17"], "geometry3k_revision": geometry.GEOMETRY3K_REVISION,
                       "geometry3k_test_image_hashes": {
                           str(i): hashlib.sha256(f"authored-test-index-{i}".encode()).hexdigest()
                           for i in range(2401, 3002)}})
    return result


def fixture(source="geometry3k"):
    if source == "geometry3k":
        return {"native": {"id": 17, "data_type": "train", "problem_text": "Find the marked length.",
                           "choices": ["3", "5", "7", "9"], "answer": "C",
                           "detailed_solution": "DO_NOT_CONSUME_NATIVE_SOLUTION"},
                "diagram_annotation": {"diagram_logic_form": [
                    "Equals(LengthOf(Line(A,B)), 3.5)", "Equals(MeasureOf(Angle(A,B,C)),90)",
                    "PointLiesOnLine(D,Line(A,B))", "Equals(LengthOf(Line(B,C)),x)"],
                    "text_logic_form": ["Equals(LengthOf(Line(C,D)),99)"],
                    "dissolved_text_logic_form": ["SOLUTION_MUST_NOT_BECOME_VISUAL_GOLD"]},
                "provenance": provenance(source)}
    return {"native": {"diagram": "synthetic-example.png", "text": "Find the marked length.",
                       "choices": ["3", "5", "7", "9"], "answer": "7.0",
                       "parsing_stru_seqs": ["line A B C"],
                       "parsing_sem_seqs": ["AB = 3.5", "m \\angle ABC = 90", "BC = x", "AB is parallel CD"],
                       "expression": "DO_NOT_CONSUME_PROGRAM"},
            "provenance": provenance(source)}


def convert(record, source="geometry3k", split="train"):
    function = geometry.convert_geometry3k if source == "geometry3k" else geometry.convert_pgps9k
    return function(record, split, "fixtures/geometry.png", "a" * 64, "obs")


class GeometryTests(unittest.TestCase):
    def test_native_choices_and_diagram_only_slots(self):
        for source in ("geometry3k", "pgps9k"):
            with self.subTest(source=source):
                actor, verifier, lineage = convert(fixture(source), source)
                self.assertEqual(actor.question, "Find the marked length.")
                self.assertEqual(actor.choices, ("3", "5", "7", "9"))
                self.assertEqual(actor.answer_format, "mcq")
                self.assertEqual(verifier.gold, "C")
                self.assertEqual(verifier.slots, {"angle[ABC]": "90", "len[AB]": "7/2"})
                self.assertEqual(verifier.required_slots, ())
                self.assertEqual(lineage.label_provenance, "HUMAN")
                self.assertEqual(lineage.structure_hash, "a" * 16)
                self.assertIs(compare("C", verifier), True)
                observed = "Observed facts:\n" + "\n".join(verifier.oracle_fact_lines) + "\nSolution:\n"
                report = compare_slots(parse_observations(observed), verifier)
                self.assertEqual(report.all_correct, report.all_total)
                for kind, record in ((ActorItem, actor), (VerifierItem, verifier), (Lineage, lineage)):
                    self.assertEqual(from_json(kind, to_json(record)), record)

    def test_native_questions_are_independent_of_solution_and_program_fields(self):
        for source in ("geometry3k", "pgps9k"):
            record = fixture(source)
            original = convert(record, source)
            for field in ("detailed_solution", "solution", "expression", "rationale", "target_text"):
                changed = copy.deepcopy(record)
                changed["native"][field] = "TEACHER_OUTPUT_DO_NOT_READ_OR_EMIT" * 30
                self.assertEqual(convert(changed, source), original)
            self.assertNotIn("DO_NOT_CONSUME", to_json(original[0]))
            self.assertNotIn("DO_NOT_CONSUME", to_json(original[1]))

    def test_original_mcq_order_and_no_closest_choice_recovery(self):
        g3k = fixture()
        g3k["native"]["choices"] = ["9", "7", "5", "3"]
        g3k["native"]["answer"] = "B"
        actor, verifier, _ = convert(g3k)
        self.assertEqual(actor.choices, ("9", "7", "5", "3"))
        self.assertEqual(verifier.gold, "B")
        pgps = fixture("pgps9k")
        pgps["native"]["choices"][0] = " 3 "
        self.assertEqual(convert(pgps, "pgps9k")[0].choices[0], " 3 ")
        pgps = fixture("pgps9k")
        pgps["native"]["answer"] = "7.01"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            convert(pgps, "pgps9k")
        pgps["native"]["answer"] = "7"
        pgps["native"]["choices"] = ["7.0", "7", "5", "9"]
        with self.assertRaises(ValueError):
            convert(pgps, "pgps9k")

    def test_open_numeric_precision_is_declared_independently_of_gold(self):
        for source in ("geometry3k", "pgps9k"):
            record = fixture(source)
            record["native"]["choices"] = []
            record["native"]["answer"] = "1.2345"
            with self.assertRaisesRegex(ValueError, "independent declared"):
                convert(record, source)
            record["provenance"]["numeric_precision"] = {"round_to": 2, "basis": "native_choice_precision"}
            actor, verifier, _ = convert(record, source)
            self.assertEqual(actor.answer_format, "decimal")
            self.assertEqual(verifier.accept, {"round_to": 2})
            self.assertEqual(verifier.gold, "1.2345")
            self.assertIs(compare("1.23", verifier), True)
            self.assertIs(compare("1.24", verifier), False)
            record["native"]["answer"] = "1.2"
            self.assertEqual(convert(record, source)[1].accept["round_to"], 2)
            record["provenance"]["numeric_precision"]["basis"] = "gold_digits"
            with self.assertRaises(ValueError):
                convert(record, source)

    def test_upstream_train_only_is_distinct_from_internal_split(self):
        for source in ("geometry3k", "pgps9k"):
            for split in ("train", "dev", "sealed", "style_sealed"):
                self.assertEqual(convert(fixture(source), source, split)[2].split, split)
            for forbidden in ("test", "val", "validation", "dev", None):
                record = fixture(source)
                record["provenance"]["upstream_split"] = forbidden
                with self.subTest(source=source, split=forbidden), self.assertRaises(ValueError):
                    convert(record, source)
        for native_id, data_type in ((2401, "train"), (2101, "train"), (17, "test"), (17, "val")):
            record = fixture()
            record["native"].update(id=native_id, data_type=data_type)
            record["provenance"]["native_id"] = str(native_id)
            with self.assertRaises(ValueError):
                convert(record)

    def test_fail_closed_provenance_and_image_binding(self):
        for source in ("geometry3k", "pgps9k"):
            for change in ({"source_revision": "main"}, {"label_provenance": "MODEL"},
                           {"annotation_provenance": "PREDICTED"}, {"model_touched": True},
                           {"benchmark_derived": True}, {"model_touched": 0},
                           {"image_sha256": "b" * 64}, {"native_id": "../17"}):
                record = fixture(source)
                record["provenance"].update(change)
                with self.subTest(source=source, change=change), self.assertRaises(ValueError):
                    convert(record, source)
            record = fixture(source)
            del record["provenance"]
            with self.assertRaises(ValueError):
                convert(record, source)
        record = fixture()
        record["native"]["id"] = 18
        with self.assertRaises(ValueError):
            convert(record)

    def test_pgps_rejects_unknown_and_nontrain_geometry3k_parentage(self):
        for change in ({"parentage_complete": False}, {"geometry3k_parent_ids": ["2401"]},
                       {"geometry3k_parent_ids": ["2101"]}, {"geometry3k_parent_ids": ["17", "17"]},
                       {"geometry3k_parent_ids": ["unknown"]}, {"geometry3k_parent_ids": []},
                       {"geometry3k_revision": "main"}, {"geometry3k_test_image_hashes": {}}):
            record = fixture("pgps9k")
            record["provenance"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                convert(record, "pgps9k")
        record = fixture("pgps9k")
        record["provenance"].update(upstream_origin="PGPS9K-textbooks", geometry3k_parent_ids=[])
        convert(record, "pgps9k")
        record["provenance"]["upstream_origin"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            convert(record, "pgps9k")

    def test_pgps_rejects_complete_index_exact_image_match(self):
        record = fixture("pgps9k")
        record["provenance"]["geometry3k_test_image_hashes"]["2800"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "test parent or diagram"):
            convert(record, "pgps9k")
        for bad in ("not-a-digest", "A" * 64):
            record = fixture("pgps9k")
            record["provenance"]["geometry3k_test_image_hashes"]["2800"] = bad
            with self.assertRaises(ValueError):
                convert(record, "pgps9k")

    def test_native_schema_and_path_validation(self):
        record = fixture()
        del record["diagram_annotation"]
        with self.assertRaises(ValueError):
            convert(record)
        record = fixture()
        record["diagram_annotation"]["diagram_logic_form"] = "predicted string"
        with self.assertRaises(ValueError):
            convert(record)
        for path in ("../diagram.png", "/diagram.png", "a\\b.png", "a//b.png", "a/./b.png"):
            record = fixture("pgps9k")
            record["native"]["diagram"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                convert(record, "pgps9k")
        record = fixture("pgps9k")
        record["native"]["id"] = "wrong-key"
        with self.assertRaisesRegex(ValueError, "key binding"):
            convert(record, "pgps9k")

    def test_logical_literals_and_clause_subset_conflicts(self):
        slots = geometry._logic_slots([
            "Equals(3.5,LengthOf(Line(B,A)))", "Equals(LengthOf(Line(A,B)),7/2)",
            "Equals(90,MeasureOf(Angle(C,B,A)))", "Equals(MeasureOf(Angle(1)),30)",
            "Equals(LengthOf(Line(C,D)),2*x)", "Equals(MeasureOf(Arc(A,B)),60)"])
        self.assertEqual(slots, {"angle[1]": "30", "angle[ABC]": "90", "len[AB]": "7/2"})
        self.assertEqual(geometry._pgps_slots(["B A = 3.5", "m ∠CBA = 90", "BC = x"]),
                         {"angle[ABC]": "90", "len[AB]": "7/2"})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            geometry._logic_slots(["Equals(LengthOf(Line(A,B)),3)", "Equals(LengthOf(Line(B,A)),4)"])
        with self.assertRaises(ValueError):
            geometry._pgps_slots(["AB = -3"])

    def test_empty_visual_slots_do_not_invent_required_facts_or_ocr(self):
        record = fixture()
        record["diagram_annotation"]["diagram_logic_form"] = ["Perpendicular(Line(A,B),Line(B,C))"]
        _, verifier, _ = convert(record)
        self.assertEqual(verifier.slots, {})
        self.assertEqual(verifier.required_slots, ())
        self.assertEqual(verifier.printed_text, ())

    def test_dataset_rights_remain_explicitly_unreviewed(self):
        self.assertIn("textbook-image-rights-unreviewed", convert(fixture())[2].license)
        self.assertEqual(convert(fixture("pgps9k"), "pgps9k")[2].license,
                         "CASIA-PGPS9K-academic-agreement-required")


if __name__ == "__main__":
    unittest.main()
