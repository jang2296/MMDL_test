"""CPU checks for deterministic selection; no benchmark or model fixture content."""
from collections import Counter
import copy
import hashlib
import random
import unittest

from fine_tuning.error_note.curation_policy_v16b import choose, feature_candidate, interleave


def candidate(source="hf-chartqa", index=0, **changes):
    pid = f"{source}:{index:04d}"
    return {
        "problem_id": pid, "source": source, "family_id": pid,
        "actor": {"question": "Which value is shown in the chart?", "choices": ["1", "2", "3", "4"]},
        "verifier": {"answer_type": "multiple_choice", "gold": "A"},
        "lineage": {}, "native_metadata": {}, "annotation_refs": [],
        "human_solution": None, "f0_prior": None, "image_metadata": [{"width": 100, "height": 100}],
        "eligible": True, "exclusion_reasons": [],
    } | changes


def prior(correct=True, tokens=100, **changes):
    return {"correct": correct, "completion_tokens": tokens,
            "finish_reason": "stop", "policy_kind": "F0"} | changes


class FeatureTests(unittest.TestCase):
    def test_unknown_and_updated_policy_priors_get_no_reward(self):
        base = candidate()
        unknown = feature_candidate(base)
        staged = feature_candidate(base | {"f0_prior": prior(False, policy_kind="updated")})
        unlabelled = feature_candidate(base | {"f0_prior": {"correct": False}})
        for result in (unknown, staged, unlabelled):
            self.assertEqual(result["score_components"]["f0_prior"], 0)
            self.assertEqual(result["f0_label"], "unknown")
            self.assertIn("NO_VERIFIED_F0_PRIOR", result["score_reason_codes"])

    def test_exact_source_bound_f0_statuses_are_supported(self):
        native = {"model_class": "F0_PREUPDATE",
                  "binding": "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD",
                  "completion_tokens": 100, "finish_reason": "stop"}
        for status in ("CORRECT", "INCORRECT", "TRUNCATED", "INVALID_NO_FINAL", "INVALID_MULTIPLE"):
            result = feature_candidate(candidate(f0_prior=native | {"status": status}))
            self.assertEqual(result["f0_label"], "correct" if status == "CORRECT" else "wrong")
        for change in ({"status": "UNVERIFIABLE"}, {"status": "UNKNOWN"},
                       {"status": "CORRECT", "model_class": "UPDATED"},
                       {"status": "CORRECT", "binding": "ID_ONLY"}):
            result = feature_candidate(candidate(f0_prior=native | change))
            self.assertEqual(result["f0_label"], "unknown")
            self.assertEqual(result["score_components"]["f0_prior"], 0)

    def test_verified_prior_scoring_direction(self):
        wrong = feature_candidate(candidate(f0_prior=prior(False)))
        long = feature_candidate(candidate(f0_prior=prior(True, 600)))
        short = feature_candidate(candidate(f0_prior=prior(True, 12)))
        length = feature_candidate(candidate(f0_prior=prior(False, finish_reason="length")))
        self.assertGreater(wrong["score"], long["score"])
        self.assertGreater(long["score"], short["score"])
        self.assertGreater(wrong["score"], length["score"])

    def test_which_following_or_counting_alone_is_not_multistep(self):
        for question in ("Which of the following is a plant?", "How many objects are shown?"):
            item = candidate("hf-scienceqa", actor={"question": question, "choices": []})
            self.assertEqual(feature_candidate(item)["reasoning_steps"], 1)

    def test_grade_annotation_solution_and_reasoning_have_visible_components(self):
        item = candidate(
            native_metadata={"grade": "grade10", "subject": "physics"},
            annotation_refs=[{"kind": "native_table", "sha256": "a" * 64}],
            human_solution={"text": "Human-written example solution."},
            actor={"question": "If values are 10 and 20, calculate their sum then the ratio.", "choices": []},
        )
        result = feature_candidate(item)
        self.assertEqual(result["reasoning_steps"], 4)
        self.assertEqual(result["level"], "high_school")
        self.assertIn("Physics", result["mmmu_subjects"])
        self.assertTrue(result["has_annotation"])
        self.assertTrue(result["has_human_solution"])
        self.assertEqual(result["score"], sum(result["score_components"].values()))
        self.assertIs(type(result["score"]), int)

    def test_missing_image_necessity_or_grade_is_unknown(self):
        item = candidate("hf-scienceqa", actor={"question": "Which animal is a mammal?"})
        result = feature_candidate(item)
        self.assertIsNone(result["image_needed"])
        self.assertEqual(result["level"], "unknown")
        self.assertEqual(result["score_components"]["visual_necessity"], 0)
        self.assertEqual(result["score_components"]["metadata_level"], 0)

    def test_review_overrides_take_precedence(self):
        base = candidate("hf-scienceqa")
        reviewed = feature_candidate(base | {"review_overrides": {
            "image_needed": False, "reasoning_steps": 3, "level": "college",
        }})
        self.assertFalse(reviewed["image_needed"])
        self.assertEqual(reviewed["reasoning_steps"], 3)
        self.assertEqual(reviewed["level"], "college")
        self.assertIn("REVIEWED_DECORATIVE_SCIENCEQA", reviewed["score_reason_codes"])
        with self.assertRaises(ValueError):
            feature_candidate(base | {"review_overrides": {"image_needed": "false"}})

    def test_calibration_disables_unreliable_heuristics_but_keeps_review(self):
        item = candidate("hf-geometry3k", disabled_rules=["image_needed", "reasoning_depth"])
        result = feature_candidate(item)
        self.assertIsNone(result["image_needed"])
        self.assertEqual(result["reasoning_steps"], 1)
        self.assertEqual(result["reasoning_kind"], "unknown_disabled_heuristic")
        alias = feature_candidate(item | {"disabled_rules": ["reasoning_steps"]})
        self.assertEqual(alias["reasoning_kind"], "unknown_disabled_heuristic")
        self.assertEqual(alias["score_components"]["reasoning_depth"], 0)
        reviewed = feature_candidate(item | {"review_overrides": {"image_needed": False, "reasoning_steps": 2}})
        self.assertFalse(reviewed["image_needed"])
        self.assertEqual(reviewed["reasoning_steps"], 2)
        self.assertEqual(reviewed["reasoning_kind"], "review_override")
        ai2d = feature_candidate(candidate("hf-ai2d", actor={"question": "Which object is a plant?"}))
        self.assertIsNone(ai2d["image_needed"])

    def test_native_human_fields_and_annotation_support_are_distinct(self):
        item = candidate(human_solution={"lecture": "Native lecture.", "solution": "Native solution."},
                         annotation_refs=[{"kind": "original_annotation"}])
        result = feature_candidate(item)
        self.assertTrue(result["has_human_solution"])
        self.assertEqual(result["human_solution_word_count"], 4)
        self.assertTrue(result["annotation_exists"])
        self.assertIsNone(result["annotation_verifier_supported"])
        item["annotation_refs"][0]["verifier_supported"] = False
        self.assertFalse(feature_candidate(item)["annotation_verifier_supported"])

    def test_simple_photo_and_legend_priorities_are_lower(self):
        chart = feature_candidate(candidate())
        for source in ("hf-nctcrc", "hf-plantvillage", "hf-deepweeds"):
            self.assertLess(feature_candidate(candidate(source))["score"], chart["score"])
        legend = candidate("hf-mapqa", actor={"question": "What does this legend symbol mean?"})
        self.assertIn("SINGLE_STEP_MAP_LEGEND", feature_candidate(legend)["score_reason_codes"])

    def test_explicit_plural_and_below_references_need_images(self):
        for question in ("The diagrams below show two containers. Compare their average kinetic energies.",
                         "Which pattern appears in the figures below?",
                         "Read the table below to select a value.", "Use the graph below."):
            result = feature_candidate(candidate("hf-scienceqa", actor={"question": question}))
            self.assertTrue(result["image_needed"], question)
        result = feature_candidate(candidate("hf-scienceqa", actor={
            "question": "The diagrams below show two containers. Compare their average kinetic energies."}))
        self.assertEqual(result["reasoning_steps"], 2)

    def test_long_generic_lecture_does_not_grant_multistep_credit(self):
        item = candidate("hf-scienceqa", actor={"question": "Which animal is a mammal?"},
                         human_solution={"lecture": "General background. " * 100})
        self.assertEqual(feature_candidate(item)["reasoning_steps"], 1)

    def test_scienceqa_named_regions_get_map_tag_and_lower_priority(self):
        item = candidate("hf-scienceqa", actor={"question": "What is the name of the colony shown?"})
        result = feature_candidate(item)
        self.assertEqual(result["image_type"], "map")
        self.assertIn("Geography", result["mmmu_subjects"])
        self.assertEqual(result["score_components"]["single_step_priority"], -12)
        self.assertEqual(result["reasoning_steps"], 1)
        self.assertEqual(len(choose([item], total=1)[0]), 1)

    def test_nested_clevr_relations_are_multistep_without_blanket_credit(self):
        nested = candidate("hf-clevr", actor={
            "question": "How many cubes left of the sphere are behind the object of the same color?"})
        result = feature_candidate(nested)
        self.assertEqual(result["reasoning_steps"], 2)
        self.assertEqual(result["level"], "elementary_middle")
        self.assertEqual(result["level_kind"], "source_heuristic")
        simple = candidate("hf-clevr", actor={"question": "How many cubes are shown?"})
        self.assertEqual(feature_candidate(simple)["reasoning_steps"], 1)

    def test_chart_skill_tag_does_not_certify_finance_domain(self):
        result = feature_candidate(candidate())
        self.assertNotIn("Finance", result["mmmu_subjects"])
        self.assertEqual(result["subject_mapping_kind"], "heuristic_skill_overlap_not_domain_coverage")


class SelectionTests(unittest.TestCase):
    def test_exact_3000_includes_all_eligible_ai2d_with_nonai2d_caps(self):
        sources = ["hf-chartqa", "hf-geometry3k", "hf-scienceqa", "hf-gvlqa",
                   "hf-ssmr", "hf-mapqa", "hf-slake", "hf-plantvillage"]
        candidates = [candidate(source, index) for source in sources for index in range(500)]
        candidates += [candidate("hf-ai2d", index) for index in range(183)]
        candidates += [candidate("hf-ai2d", 9999, eligible=False,
                                 exclusion_reasons=["HOLDOUT_FAMILY"])]
        selected, decisions = choose(candidates)
        counts = Counter(item["source"] for item in selected)
        self.assertEqual(len(selected), 3000)
        self.assertEqual(len({item["family_id"] for item in selected}), 3000)
        self.assertEqual(counts["hf-ai2d"], 183)
        self.assertTrue(all(count <= 450 for source, count in counts.items() if source != "hf-ai2d"))
        holdout = next(item for item in decisions if item["problem_id"] == "hf-ai2d:9999")
        self.assertEqual(holdout["reason_codes"], ["HOLDOUT_FAMILY"])

    def test_ai2d_is_mandatory_even_at_low_score_and_over_cap(self):
        candidates = [candidate("hf-ai2d", index, review_overrides={"image_needed": False})
                      for index in range(4)]
        candidates += [candidate("hf-geometry3k", index) for index in range(10)]
        selected, _ = choose(candidates, total=6, source_cap=2)
        self.assertEqual(Counter(item["source"] for item in selected), {"hf-ai2d": 4, "hf-geometry3k": 2})
        with self.assertRaisesRegex(ValueError, "mandatory eligible AI2D"):
            choose(candidates, total=3, source_cap=2)

    def test_calibrated_source_caps_reduce_exposure_but_not_ai2d(self):
        candidates = [candidate("hf-chartqa", index) for index in range(6)]
        candidates += [candidate("hf-geometry3k", index) for index in range(6)]
        candidates += [candidate("hf-ai2d", index) for index in range(3)]
        selected, decisions = choose(candidates, total=11, source_cap=6,
                                     source_caps={"hf-chartqa": 2, "hf-ai2d": 1})
        counts = Counter(item["source"] for item in selected)
        self.assertEqual(counts["hf-chartqa"], 2)
        self.assertEqual(counts["hf-ai2d"], 3)
        chart_exclusions = [item for item in decisions if item["source"] == "hf-chartqa" and not item["selected"]]
        self.assertTrue(all(item["reason_codes"] == ["SOURCE_CAP"] for item in chart_exclusions))
        for caps in ({"hf-chartqa": 0}, {"hf-chartqa": 7}, {"hf-chartqa": True}, {"": 1}, []):
            with self.assertRaisesRegex(ValueError, "source_caps"):
                choose(candidates, total=1, source_cap=6, source_caps=caps)

    def test_duplicate_family_is_resolved_by_priority_once(self):
        ai2d = candidate("hf-ai2d", family_id="shared")
        duplicate = candidate("hf-geometry3k", family_id="shared", f0_prior=prior(False))
        other = candidate("hf-chartqa", 2)
        selected, decisions = choose([duplicate, ai2d, other], total=2)
        self.assertEqual({item["problem_id"] for item in selected}, {ai2d["problem_id"], other["problem_id"]})
        excluded = next(item for item in decisions if item["problem_id"] == duplicate["problem_id"])
        self.assertEqual(excluded["reason_codes"], ["DUPLICATE_FAMILY"])

    def test_duplicate_input_ids_and_missing_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate input"):
            choose([candidate(), candidate()], total=1)
        with self.assertRaisesRegex(ValueError, "family_id"):
            choose([candidate(family_id="")], total=1)

    def test_insufficient_supply_and_source_cap_are_never_relaxed(self):
        candidates = [candidate(index=index) for index in range(4)]
        with self.assertRaisesRegex(ValueError, "insufficient eligible"):
            choose(candidates, total=3, source_cap=2)
        candidates[0]["exclusion_reasons"] = ["NO_RECEIPT"]
        with self.assertRaisesRegex(ValueError, "insufficient eligible"):
            choose(candidates, total=4)

    def test_global_score_order_does_not_force_named_source_quota(self):
        low = candidate("hf-plantvillage")
        high = candidate("hf-chartqa")
        selected, decisions = choose([low, high], total=1)
        self.assertEqual(selected[0]["problem_id"], high["problem_id"])
        self.assertEqual(next(item for item in decisions if not item["selected"])["reason_codes"],
                         ["BELOW_SELECTION_CUTOFF"])
        self.assertEqual(len(choose([low], total=1)[0]), 1)

    def test_selection_is_stable_and_does_not_mutate_input(self):
        candidates = [candidate(index=index) for index in range(30)]
        original = copy.deepcopy(candidates)
        first = choose(candidates, total=20)
        random.Random(2).shuffle(candidates)
        self.assertEqual(first, choose(candidates, total=20))
        self.assertEqual(sorted(candidates, key=lambda item: item["problem_id"]), original)


class InterleaveTests(unittest.TestCase):
    def test_3000_ends_with_8_without_padding_and_preserves_source_hash_order(self):
        selected = [candidate(f"source-{source}", index) for source in range(8) for index in range(375)]
        result = interleave(selected)
        self.assertEqual(result["count"], 3000)
        self.assertEqual(len(result["batches"]), 188)
        self.assertEqual(result["batches"][-1]["size"], 8)
        self.assertEqual(len({item["problem_id"] for item in result["items"]}), 3000)
        for source in {item["source"] for item in selected}:
            expected = sorted((item["problem_id"] for item in selected if item["source"] == source),
                              key=lambda pid: hashlib.sha256(("v16b-order:" + pid).encode()).hexdigest())
            actual = [item["problem_id"] for item in result["items"] if item["source"] == source]
            self.assertEqual(actual, expected)
        self.assertEqual(len(result["rule_sha256"]), 64)
        self.assertTrue(all(batch["prior_counts"]["unknown"] == batch["size"] for batch in result["batches"]))

    def test_unequal_source_proportions_remain_within_two_at_every_prefix(self):
        counts = {"a": 183, "b": 450, "c": 71, "d": 96}
        selected = [candidate(source, index) for source, count in counts.items() for index in range(count)]
        result = interleave(selected)
        seen, position = Counter(), 0
        for batch in result["batches"]:
            seen.update(batch["source_counts"])
            position += batch["size"]
            for source, count in counts.items():
                self.assertLessEqual(abs(seen[source] - position * count / 800), 2)
        self.assertEqual(result["prefix_balance"]["status"], "PASS")
        self.assertEqual(Counter(item["source"] for item in result["items"]), counts)

    def test_known_labels_mix_where_source_queues_allow(self):
        selected = [candidate(source, index, f0_prior=prior(source == "correct-source"))
                    for source in ("correct-source", "wrong-source") for index in range(16)]
        result = interleave(selected)
        self.assertTrue(all(batch["prior_balance_status"] == "MIXED_KNOWN_F0" for batch in result["batches"]))

    def test_unknown_prior_balance_is_reported_without_inventing_labels(self):
        selected = [candidate(index=index, f0_prior=prior(False, policy_kind="trained")) for index in range(17)]
        result = interleave(selected)
        self.assertEqual(result["f0_prior_counts"], {"unknown": 17})
        self.assertTrue(all(batch["prior_balance_status"] == "NO_BOTH_KNOWN_F0_LABELS_IN_BATCH"
                            for batch in result["batches"]))

    def test_joint_strata_balance_all_dimensions_and_preserve_within_stratum_hash(self):
        selected = []
        for source in ("a", "b", "c", "d"):
            items = [candidate(source, index) for index in range(200)]
            items.sort(key=lambda item: hashlib.sha256(("v16b-order:" + item["problem_id"]).encode()).hexdigest())
            for index, item in enumerate(items):
                item["f0_prior"] = None if index % 7 == 0 else prior(index < 100, 8192 if index >= 150 else 200,
                                                                   finish_reason="length" if index >= 150 else "stop")
                item["has_annotation"] = index >= 100
            selected.extend(items)
        result = interleave(selected)
        self.assertEqual(result["prefix_balance"]["status"], "PASS")
        fields = ("source", "f0_label", "annotation_label", "long_risk_label")
        populations = {field: Counter(item[field] for item in result["items"]) for field in fields}
        seen = {field: Counter() for field in fields}
        position = 0
        for batch in result["batches"]:
            position += batch["size"]
            for item in batch["items"]:
                for field in fields:
                    seen[field][item[field]] += 1
            for field in fields:
                for label, count in populations[field].items():
                    self.assertLessEqual(abs(seen[field][label] - position * count / 800), 2)
        strata = {}
        for item in result["items"]:
            strata.setdefault(tuple(item[field] for field in fields), []).append(item["problem_id"])
        for ids in strata.values():
            self.assertEqual(ids, sorted(ids, key=lambda pid: hashlib.sha256(("v16b-order:" + pid).encode()).hexdigest()))
        self.assertEqual(result, interleave(list(reversed(selected))))
        self.assertEqual(result["prefix_balance"]["checked_prefixes"], 50)

    def test_order_labels_do_not_turn_unknown_priors_into_short_responses(self):
        result = interleave([candidate("hf-ssmr"), candidate("hf-chartqa"),
                             candidate("hf-chartqa", 1, has_annotation=False, f0_prior=prior())])
        by_id = {item["problem_id"]: item for item in result["items"]}
        self.assertEqual(by_id["hf-ssmr:0000"]["long_risk_label"], "yes")
        self.assertEqual(by_id["hf-chartqa:0000"]["long_risk_label"], "unknown")
        self.assertEqual(by_id["hf-chartqa:0000"]["annotation_label"], "unknown")
        self.assertEqual(by_id["hf-chartqa:0001"]["long_risk_label"], "no")
        self.assertEqual(by_id["hf-chartqa:0001"]["annotation_label"], "absent")

    def test_order_is_stable_for_reversed_input(self):
        selected = [candidate(source, index) for source in ("a", "b", "c") for index in range(23)]
        first = interleave(selected)
        self.assertEqual(first, interleave(list(reversed(selected))))

    def test_order_refuses_duplicates_and_empty_selection(self):
        for selected in ([], [candidate(), candidate()], [candidate(), candidate(index=1, family_id="hf-chartqa:0000")]):
            with self.assertRaises(ValueError):
                interleave(selected)


if __name__ == "__main__":
    unittest.main()
