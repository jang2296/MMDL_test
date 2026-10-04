"""Hand-counted stdlib checks for explicit coverage, abstentions and immutable units."""

import unittest
from copy import deepcopy

from experiments.bottleneck_diagnostic.input_builder import canonical_hash
from experiments.bottleneck_diagnostic.oracle_validation import oracle_hash
from experiments.bottleneck_diagnostic.structure_review import (
    measures, packet, summarize, template, units, validated_decisions,
)


def hand_case():
    candidates, _ = units({"images": [{"marker": "<image 1>", "connections": [
        ["A", "B", "C"], "wrong", "unclear", "irrelevant"]}]}, "candidate", "p")
    references, _ = units({"images": [{"marker": "<image 1>", "connections": [
        ["A", "B"], ["B", "C"], ["C", "D"], "unclear", "irrelevant"]}]}, "reference", "o")
    case = {"candidate_units": candidates, "reference_units": references,
            "candidate_fields": [], "reference_fields": [], "sample_id": "synthetic",
            "runtime_status": "COMPLETED",
            "schema_parse_status": "VALID", "oracle_gate": {"status": "VALID"},
            "comparison_status": "ELIGIBLE"}
    case["packet_hash"] = canonical_hash(case)
    review = template(case)
    review.update(reviewer_id="synthetic-hand-check", verification_kind="AI_REVIEWED_NOT_HUMAN_VERIFIED",
                  coverage_status="COMPLETE", coverage_evidence="Synthetic whole packet review")
    cs, rs = review["candidate_decisions"], review["reference_decisions"]
    for d, status in zip(cs, ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "NOT_APPLICABLE"], strict=True):
        d.update(status=status, evidence="Synthetic explicit hand judgment")
    cs[0]["reference_ids"] = [r["unit_id"] for r in rs[:2]]
    for d, status in zip(rs, ["MATCHED", "MATCHED", "OMITTED", "UNVERIFIABLE", "NOT_APPLICABLE"], strict=True):
        d.update(status=status, evidence="Synthetic explicit hand judgment")
        if status == "MATCHED":
            d["candidate_id"] = cs[0]["unit_id"]
    return case, review


class StructureReviewTests(unittest.TestCase):
    def test_packet_gold_blind_projection_schema_and_empty_review_gate(self):
        import json
        sample = {"sample_id": "synthetic", "subject": "Biology",
                  "images": [{"marker": "<image 1>", "sha256": "a" * 64,
                              "path": "image.png", "original_size": [10, 10]}],
                  "question": "DO_NOT_EXPORT_QUESTION", "options": ["DO_NOT_EXPORT_OPTION"]}
        facts = {"images": [{"marker": "<image 1>", "visible_text": [], "symbols": [],
                             "values": [], "positions": [], "connections": [], "uncertainties": []}]}
        oracle = {"sample_id": "synthetic", "level": "O0", "annotation_version": "synthetic",
            "facts": facts, "source_image_hashes": ["a" * 64], "evidence_locations": ["image"],
            "completeness_status": "COMPLETE", "target_leakage_status": "VALID",
            "annotator_exposure_status": "GOLD_BLIND", "reviewer_status": "AI_INDEPENDENT_REVIEWED",
            "reviewer_id": "reviewer", "annotator_id": "author", "approved": True,
            "provenance": "BENCHMARK_DERIVED", "training_allowed": False}
        oracle["oracle_hash"] = oracle_hash(oracle)
        raw = json.dumps(facts)
        record = {"sample_id": "synthetic", "repetition": 3407, "runtime_status": "COMPLETED",
                  "raw_response": raw, "response_hash": canonical_hash(raw),
                  "oracle_review_policy": "ALLOW_AI_REVIEWED_DIAGNOSTIC", "gold": "DO_NOT_EXPORT_GOLD"}
        case = packet(record, sample, oracle)
        self.assertEqual(case["comparison_status"], "ELIGIBLE")
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(case))
        self.assertEqual(summarize([case], [template(case)])["reviewed_cases"], 0)
        record["raw_response"] = "not json"
        case = packet(record, sample, oracle)
        self.assertEqual(case["schema_parse_status"], "INVALID")
        self.assertEqual(case["comparison_status"], "NOT_COMPARABLE")
        self.assertEqual(case["candidate_units"], [])

    def test_hand_calculation_grouped_coverage_and_no_recursive_leaves(self):
        case, review = hand_case()
        self.assertEqual(len(case["candidate_units"]), 4)
        self.assertEqual(len(case["reference_units"]), 5)
        c, r = validated_decisions(case, review)
        result = measures(c, r, True)
        self.assertEqual(result["semantic_precision_adjudicable"], 1 / 2)
        self.assertEqual(result["supported_fraction_all_applicable"], 1 / 3)
        self.assertEqual(result["matched_reference_recall_adjudicable"], 2 / 3)
        self.assertEqual(result["matched_reference_fraction_all_applicable"], 2 / 4)
        self.assertEqual(result["omission_fraction_adjudicable"], 1 / 3)
        summary = summarize([case], [review])
        self.assertEqual(summary["reviewed_cases"], 1)
        self.assertEqual(summary["critical_full_structure_exact"]["status"], "NOT_MEASURED")

    def test_unreviewed_and_empty_denominators_never_become_perfect(self):
        case, _ = hand_case()
        summary = summarize([case], [template(case)])
        self.assertEqual(summary["overall"]["status"], "NOT_MEASURED")
        self.assertIsNone(summary["overall"]["semantic_precision_adjudicable"])
        self.assertEqual(summary["overall"]["candidate_denominator_status"], "UNREVIEWED")
        self.assertIsNone(summary["overall"]["empty_candidate_denominator"])
        result = measures([], [], True)
        self.assertIsNone(result["matched_reference_recall_adjudicable"])
        self.assertEqual(result["empty_reference_denominator"], "NOT_APPLICABLE")

    def test_rejects_coverage_status_identity_and_source_tampering(self):
        case, review = hand_case()
        changes = [
            lambda r: r["candidate_decisions"][1].update(reference_ids=r["candidate_decisions"][0]["reference_ids"]),
            lambda r: r["candidate_decisions"][0].update(reference_ids=["unknown"]),
            lambda r: r["reference_decisions"][0].update(candidate_id=None),
            lambda r: r["candidate_decisions"][0].update(status="PASS"),
            lambda r: r["candidate_decisions"].append(r["candidate_decisions"][0]),
            lambda r: r.update(verification_kind="HUMAN_VERIFIED"),
            lambda r: r.update(packet_hash="changed"),
            lambda r: r["candidate_decisions"][0].update(evidence=""),
            lambda r: r.update(coverage_status="UNREVIEWED"),
            lambda r: r.update(coverage_evidence=None),
            lambda r: r["candidate_decisions"][1].update(status="UNREVIEWED"),
        ]
        for change in changes:
            edited = deepcopy(review)
            change(edited)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_decisions(case, edited)
        changed = deepcopy(case)
        changed["candidate_units"][0]["value"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash"):
            validated_decisions(changed, review)

    def test_duplicate_supported_owner_rejected(self):
        case, review = hand_case()
        review["candidate_decisions"][1].update(status="SUPPORTED", reference_ids=[
            review["reference_decisions"][0]["unit_id"]])
        with self.assertRaisesRegex(ValueError, "one candidate owner"):
            validated_decisions(case, review)


if __name__ == "__main__":
    unittest.main()
