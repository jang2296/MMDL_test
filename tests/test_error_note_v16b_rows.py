"""Synthetic CPU contracts; no benchmark, model download, generation, or GPU work."""
from copy import deepcopy
import json
import unittest

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from fine_tuning.error_note import v16b_rows as rows
from fine_tuning.error_note.audit import digest, text_sha256
from fine_tuning.error_note.self_note_v16 import FORMULAS


class NativeBytes:
    all_special_tokens = ["<|im_end|>"]
    unk_token_id = None
    byte_encoder = bytes_to_unicode()

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 256

    def convert_ids_to_tokens(self, ids):
        return ["<|im_end|>" if value == 256 else self.byte_encoder[value] for value in ids]


TOKENIZER = NativeBytes()
LONG_PROSE = (
    "The displayed quantities describe separate groups whose sizes can be compared through their ratio. "
    "Before choosing an answer, identify the numerator and denominator from the question, preserve their "
    "original roles, perform the division, and inspect the resulting units for consistency. "
    "This reasoning connects the available evidence to a single numerical conclusion."
)


def output(text, *, status="INCORRECT", extracted="5", prompt="actual generation prompt", finish="stop"):
    actions = TOKENIZER.encode(text) + [256]
    generation = {"text": text, "completion_token_ids": actions, "finish_reason": finish,
                  "policy_version": 0, "sampling_logprobs": [-0.5] * len(actions),
                  "probability_space": "processed-full-vocab-v1"}
    return {"text": text, "status": status, "extracted": extracted, "prompt": prompt,
            "completion_tokens": len(actions), "finish_reason": finish, "generation": generation}


def fixture(*, catalog=True):
    facts = [{"fact_id": ref, "kind": "OBSERVATION", "entity": entity, "evidence_origin": "synthetic:" + ref,
              "value": value, "unit": "count"}
             for ref, entity, value in (("f0", "segment AB", "12"), ("f1", "segment CD", "3"))] if catalog else []
    rules = {"DIV_COUNTS": {"operation": "DIV", "formula": FORMULAS["DIV"],
                            "operand_units": ["count", "count"], "result_unit": "dimensionless"}} if catalog else {}
    note = {"observed_facts": [{"fact_ref": "f0", "value": "12"}] if catalog else [],
            "error": {"earlier_span": [0, 10], "earlier_quote": "12 / 3 = 5", "kind": "ARITHMETIC",
                      "operand_fact_refs": ["f0", "f1"] if catalog else [], "operation": "DIV",
                      "old_value": "5", "corrected_value": "4", "rounding": "none",
                      "unit": "dimensionless" if catalog else "", "rule_id": "DIV_COUNTS" if catalog else "DIV"},
            "lesson": {"text": "Recompute the division before committing the result."}}
    record = {"problem_id": "synthetic:1", "source": "creator", "image_paths": ["image.png"],
              "image_sha256": ["a" * 64], "p0_text": "Question: Divide the displayed counts.",
              "evaluation": {"gold": "4", "choices": []},
              "t1": output("12 / 3 = 5\nFinal answer: 5"), "note": output(json.dumps(note, ensure_ascii=False))}
    return record, facts, rules, note


def set_note(record, note):
    return record | {"note": output(json.dumps(note, ensure_ascii=False))}


def verify(record, facts=(), rules=None, gold="4"):
    return rows.verify_note_v16b(record, fact_catalog=list(facts), rules=rules or {}, choices=(), gold=gold)


class V16bRewardsAndBaselines(unittest.TestCase):
    def test_rewards_keep_normal_wrong_pg_but_exclude_cutoff_loops_and_unknown(self):
        cases = [("CORRECT", "stop", "Final answer: 4", "4", (1, "CORRECT", True)),
                 ("INCORRECT", "stop", "Final answer: 5", "5", (0, "INCORRECT", True)),
                 ("INVALID_NO_FINAL", "stop", "unfinished conclusion", None, (0, "INVALID_NO_FINAL", True)),
                 ("INVALID_MULTIPLE", "stop", "Final answer: A or B", None, (0, "INVALID_MULTIPLE", True)),
                 ("CORRECT", "length", "text", "4", (0, "LENGTH", False)),
                 ("UNVERIFIABLE", "stop", "text", None, (None, "UNVERIFIABLE", False))]
        for status, finish, text, extracted, expected in cases:
            self.assertEqual(rows.reward_v16b(status, finish, text, extracted), expected)

    def test_baseline_is_frozen_at_explicit_committed_history_cutoff(self):
        item = {"problem_id": "p", "source": "s", "f0_prior": {"correct": True}}
        prior = {"source_rates": {"s": 0.25}, "overall": 0.4}
        history = [{"block_index": block, "committed": True, "source": "s", "reward": reward}
                   for block, reward in ((0, 1.0), (1, 0.0), (2, 1.0))]
        history += [{"block_index": 0, "committed": False, "source": "s", "reward": 0.0}]
        receipt = rows.baseline_receipt_v16b(item, history, prior=prior, history_cutoff=0)
        self.assertAlmostEqual(receipt["source_mean"], 3 / 9)
        self.assertAlmostEqual(receipt["baseline"], (2 * (3 / 9) + 1) / 3)
        history[-2]["reward"] = 0.0
        self.assertEqual(receipt, rows.baseline_receipt_v16b(item, history, prior=prior, history_cutoff=0))
        self.assertEqual(receipt["sha256"], digest({key: value for key, value in receipt.items() if key != "sha256"}))
        with self.assertRaisesRegex(ValueError, "CUTOFF"):
            rows.baseline_v16b(item, history, prior=prior)

    def test_baseline_window_and_source_global_fallback(self):
        history = [{"block_index": i, "source": "s", "committed": True, "reward": float(i >= 6)} for i in range(70)]
        value = rows.baseline_v16b({"problem_id": "p", "source": "s"}, history,
                                   prior={"source_rates": {}, "overall": 0.5}, history_cutoff=69)
        self.assertAlmostEqual(value, 68 / 72)
        self.assertEqual(rows.baseline_v16b({"problem_id": "p", "source": "s"}, [],
                                           prior={"source_rates": {}, "overall": 0.5}, history_cutoff=-2), 0.5)

    def test_observation_initialization_waits_for_commit_and_preserves_zero_marker(self):
        empty = rows.obs_baseline_v16b("graph", [], history_cutoff=-1)
        self.assertTrue(empty["zero_advantage"])
        history = [{"block_index": 0, "kind": "OBS", "committed": True, "layer": "graph", "score": x}
                   for x in (0.2, 0.6)]
        later = rows.obs_baseline_v16b("graph", history, history_cutoff=0)
        self.assertFalse(later["zero_advantage"])
        self.assertAlmostEqual(later["baseline"], 0.4)
        self.assertIsNone(empty["baseline"])
        prior = rows.obs_baseline_v16b("graph", history, history_cutoff=-1, prior={"layer_rates": {"graph": 0.7}})
        self.assertEqual(prior["baseline"], 0.7)


class V16bObservationFacts(unittest.TestCase):
    def test_graph_native_visible_nodes_missing_extra_and_unreadable(self):
        item = {"annotation": {"directed": False, "structure_text": "5 2 7\n0 1\n1 3\n3 4\n"}}
        exact = {"nodes": [0, 1, 3], "edges": [[1, 0], [1, 3]]}
        self.assertEqual(rows.obs_score(item, "graph", json.dumps(exact))["score"], 1)
        missing = rows.obs_score(item, "graph", json.dumps({"nodes": [0, 1], "edges": [[0, 1]]}))
        self.assertEqual((missing["tp"], missing["fp"], missing["fn"]), (3, 0, 2))
        extra = deepcopy(exact)
        extra["nodes"].append(4)
        self.assertEqual(rows.obs_score(item, "graph", json.dumps(extra))["fp"], 1)
        self.assertEqual(rows.obs_score(item, "graph", "uncertain")["score"], 0)
        self.assertEqual(rows.obs_score(item, "graph", json.dumps(exact), finish_reason="length")["score"], 0)

    def test_geometry_labels_marks_and_scene_order(self):
        forms = ["Equals(LengthOf(Line(A, B)), 5)", "Equals(MeasureOf(Angle(A, B, C)), 30)",
                 "Parallel(Line(A, B), Line(C, D))"]
        item = {"annotation": {"matched_id": "g", "candidates": {"g": {"diagram_logic_forms": forms}}}}
        text = json.dumps({"labels": ["segment BA = 5.0", "angle CBA = 30°"], "marks": ["BA ∥ DC"]})
        self.assertEqual(rows.obs_score(item, "geometry", text)["score"], 1)
        objects = [{"size": "large", "color": "red", "material": "metal", "shape": "cube", "pixel_coords": [30, 0, 0]},
                   {"size": "small", "color": "blue", "material": "rubber", "shape": "sphere", "pixel_coords": [10, 0, 0]}]
        item = {"annotation": {"objects": objects}}
        ordered = [{key: obj[key] for key in ("size", "color", "material", "shape")} for obj in objects[::-1]]
        self.assertEqual(rows.obs_score(item, "scene", json.dumps({"objects": ordered}))["score"], 1)
        self.assertEqual(rows.obs_score(item, "scene", json.dumps({"objects": ordered[::-1]}))["score"], 0)


class V16bNativeSources(unittest.TestCase):
    def test_scene_catalog_and_observation_share_explicit_spatial_identity(self):
        objects = [{"size": "large", "color": "blue", "material": "metal", "shape": "cube",
                    "pixel_coords": [200, 0, 0]},
                   {"size": "small", "color": "red", "material": "rubber", "shape": "sphere",
                    "pixel_coords": [20, 0, 0]}]
        item = {"problem_id": "scene:spatial", "layer": "scene", "annotation": {"objects": objects}}
        contract = rows.native_note_contract(item)
        catalog, rules = contract["fact_catalog"], contract["rules"]
        self.assertEqual(next(f["value"] for f in catalog if f["fact_id"] == "left_object_1_color"), "red")
        correct = "object 1 from the left color is red"
        self.assertEqual(rows.t1_error_scan(correct, catalog, rules), [])
        self.assertEqual(rows.bind_statement(correct, catalog)[0]["fact_ref"], "left_object_1_color")
        wrong = "object 1 from the left color is blue"
        self.assertEqual(len(rows.t1_error_scan(wrong, catalog, rules)), 1)
        for ambiguous in ("object 0 color is red", "object 1 color is blue"):
            self.assertEqual(rows.bind_statement(ambiguous, catalog)[0]["status"], "UNVERIFIABLE")
            self.assertEqual(rows.t1_error_scan(ambiguous, catalog, rules), [])
        observed = [{k: obj[k] for k in ("size", "color", "material", "shape")} for obj in objects[::-1]]
        self.assertEqual(rows.obs_score(item, "scene", json.dumps({"objects": observed}))["score"], 1)
        objects[1]["pixel_coords"][0] = 200
        self.assertFalse(any(f["fact_id"].startswith("left_object")
                             for f in rows.native_note_contract(item)["fact_catalog"]))

    def test_native_alias_catalogs_and_observation_prompts_do_not_expose_answers(self):
        items = [
            {"problem_id": "geometry:1", "layer": "geometry", "gold": "secret",
             "annotation": {"diagram_logic_forms": ["Equals(LengthOf(Line(A, B)), 12)"]}},
            {"problem_id": "graph:1", "layer": "graph", "gold": "secret",
             "annotation": {"structure_text": "3 2\n0 1\n1 2", "directed": False}},
            {"problem_id": "scene:1", "layer": "scene", "gold": "secret",
             "annotation": {"objects": [{"size": "small", "color": "red", "shape": "cube",
                                          "material": "metal", "pixel_coords": [1, 0, 0]}]}},
        ]
        for item in items:
            catalog = rows.native_note_contract(item)
            self.assertTrue(catalog["supported"])
            self.assertNotIn("secret", rows.build_obs_prompt_v16b(item))
            self.assertNotIn("gold", catalog)
        graph = items[1]
        graph.pop("annotation")
        graph["references"] = {"raw_annotation": {"n": 3, "m": 2, "edges": [[0, 1], [1, 2]]},
                               "native_metadata": {"directed": False}}
        self.assertTrue(rows.native_note_contract(graph)["supported"])

    def test_natural_angle_scene_and_edge_bindings_require_exact_unambiguous_entity(self):
        facts = [{"fact_id": str(index), "entity": entity} for index, entity in enumerate(
            ("angle ABC", "object 1 from the left color", "edge 1 to 2", "weight of edge 2 to 3"))]
        text = "angle ABC = 30°\nobject 1 from the left color is red\nedge (1, 2)\n2–3 weight 7"
        claims = rows.bind_statement(text, facts)
        self.assertEqual([claim["fact_ref"] for claim in claims], ["0", "1", "2", "3"])
        self.assertTrue(all(claim["assertion"] == "ASSERTED" for claim in claims))
        self.assertIsNone(rows.bind_statement("angle ABD = 30°", facts)[0]["fact_ref"])

class V16bOptionalObservations(unittest.TestCase):
    def test_unannotated_and_unsupported_sources_keep_empty_catalog_arithmetic_notes(self):
        for item in ({"problem_id": "plain", "layer": None, "annotation": None},
                     {"problem_id": "plain", "layer": None, "annotation": {"unbounded": "source"}},
                     {"problem_id": "plain", "layer": "chart", "annotation": {"table": "source"}}):
            with self.subTest(item=item):
                contract = rows.native_note_contract(item)
                self.assertFalse(contract["supported"])
                self.assertEqual((contract["fact_catalog"], contract["rules"]), ([], {}))
                eligibility = contract["observation_eligibility"]
                self.assertFalse(eligibility["eligible"])
                self.assertEqual(eligibility["sha256"],
                                 digest({key: value for key, value in eligibility.items() if key != "sha256"}))
                record, _, _, _ = fixture(catalog=False)
                self.assertTrue(verify(record, contract["fact_catalog"], contract["rules"])["accepted"])
                with self.assertRaisesRegex(ValueError, "OBS_INELIGIBLE"):
                    rows.build_obs_prompt_v16b(item)

    def test_medium_graph_prompt_uses_problem_identity_without_parent_id(self):
        item = {"problem_id": "graph:medium", "layer": "graph",
                "annotation": {"structure_text": "40 31\n" + "\n".join(f"0 {node}" for node in range(1, 32)),
                               "directed": False}}
        self.assertTrue(rows.observation_eligibility_v16b(item)["eligible"])
        prompt = rows.build_obs_prompt_v16b(item)
        self.assertIn("Requested node:", prompt)
        self.assertIn("nodes: []", prompt)
        self.assertEqual(prompt, rows.build_obs_prompt_v16b(item | {"parent_id": item["problem_id"]}))

    def test_declared_unsupported_observations_are_receipts_but_corrupt_inputs_raise(self):
        cases = [
            ({"problem_id": "geometry:unsupported", "layer": "geometry",
              "annotation": {"diagram_logic_forms": ["Tangent(Line(A,B), Circle(C))"]}}, "UNSUPPORTED_FORM"),
            ({"problem_id": "geometry:empty", "layer": "geometry",
              "annotation": {"diagram_logic_forms": ["PointLiesOnLine(Q, Line(R, N))"]}}, "NO_FACTS"),
            ({"problem_id": "graph:large", "layer": "graph",
              "annotation": {"structure_text": "70 61\n" + "\n".join(f"0 {node}" for node in range(1, 62)),
                             "directed": False}}, "TOO_MANY_EDGES"),
        ]
        for item, reason in cases:
            eligibility = rows.observation_eligibility_v16b(item)
            self.assertFalse(eligibility["eligible"])
            self.assertEqual(eligibility["reason"], reason)
            self.assertEqual(rows.native_note_contract(item)["observation_eligibility"], eligibility)
        broken = {"problem_id": "graph:corrupt", "layer": "graph",
                  "annotation": {"structure_text": "3 2\n0 9", "directed": False}}
        with self.assertRaisesRegex(ValueError, "MALFORMED_GRAPH"):
            rows.observation_eligibility_v16b(broken)
        with self.assertRaisesRegex(ValueError, "MALFORMED_GRAPH"):
            rows.native_note_contract(broken)


class V16bNotes(unittest.TestCase):
    def test_json_spans_escape_unicode_duplicates_and_embedded_field_names(self):
        text = '{"observed_facts":[{"fact_ref":"f0","value":"삼각형 \\"error\\""}],"error":null,"lesson":{"text":"é😀"}}'
        spans = rows.json_spans(text)
        self.assertEqual(json.loads(text[slice(*spans["lesson.text"])]), "é😀")
        self.assertEqual(text[slice(*spans["error"])], "null")
        self.assertEqual(json.loads(text[slice(*spans["observed_facts[0]"])])["fact_ref"], "f0")
        for malformed in ('{"error":null,"err\\u006fr":{}}', '{"error.x":null}', '{"x":NaN}', '{"x":[1,]}'):
            with self.assertRaises(ValueError):
                rows.json_spans(malformed)

    def test_non_catalog_arithmetic_reuses_assertion_and_final_checks(self):
        record, _, _, _ = fixture(catalog=False)
        receipt = verify(record)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["verified"], 1)
        record["t1"] = output("12 / 3 = 5\n**Final answer:** 5")
        self.assertEqual(verify(record, gold="9")["reason"], "NOTE_GOLD_CONTRADICTION")

    def test_wrong_correction_contradiction_and_rounding_is_not_an_error(self):
        record, _, _, note = fixture(catalog=False)
        note["error"]["corrected_value"] = "8"
        receipt = verify(set_note(record, note))
        self.assertEqual(receipt["contradicted"], 1)
        self.assertFalse(receipt["resolve_allowed"])
        quote = "1 / 3 = 0.33"
        record["t1"] = output(quote + "\nFinal answer: 5")
        note["error"].update(earlier_span=[0, len(quote)], earlier_quote=quote, old_value="0.33",
                              corrected_value="0.333", rounding="decimal:2")
        receipt = verify(set_note(record, note))
        self.assertEqual(receipt["claims"][0]["reason"], "ORIGINAL_CLAIM_ALREADY_CORRECT")
        self.assertEqual(receipt["verified"], 0)

    def test_correct_t1_with_asserted_arithmetic_error_still_gets_note(self):
        record, facts, rules, _ = fixture()
        record["t1"] = output("12 / 3 = 5\nFinal answer: 4", status="CORRECT", extracted="4")
        self.assertEqual(len(rows.t1_error_scan(record["t1"]["text"], facts, rules)), 1)
        self.assertIn('"grading_result": "CORRECT"', rows.build_note_prompt_v16b(record, fact_catalog=facts, rules=rules, gold="4"))
        self.assertTrue(verify(record, facts, rules)["accepted"])

    def test_t1_scan_rejects_hypothetical_fragment_quotes_and_retractions(self):
        for text in ('If 12 / 3 = 5', '"12 / 3 = 5"', '112 / 3 = 5x', '12 / 3 = 5\nCorrection: 12 / 3 = 4'):
            self.assertEqual(rows.t1_error_scan(text, [], {}), [], text)
        self.assertEqual(len(rows.t1_error_scan("The ratio is 12 / 3 = 5.", [], {})), 1)

    def test_natural_catalog_binding_and_ambiguity_are_shared_by_scan_and_note(self):
        record, facts, _, note = fixture()
        facts = [facts[0]]
        rules = {"LOOKUP_COUNT": {"operation": "LOOKUP", "formula": FORMULAS["LOOKUP"],
                                  "operand_units": ["count"], "result_unit": "count"}}
        quote = "segment AB = 5"
        record["t1"] = output(quote + "\nFinal answer: 9", extracted="9")
        note["error"].update(earlier_span=[0, len(quote)], earlier_quote=quote, kind="OBSERVATION",
                              operand_fact_refs=["f0"], operation="LOOKUP", corrected_value="12",
                              unit="count", rule_id="LOOKUP_COUNT")
        self.assertEqual(rows.t1_error_scan(record["t1"]["text"], facts, rules)[0]["fact_ref"], "f0")
        self.assertTrue(verify(set_note(record, note), facts, rules)["accepted"])
        facts += [facts[0] | {"fact_id": "ambiguous"}]
        bound = rows.bind_statement(quote, facts)[0]
        self.assertIsNone(bound["fact_ref"])
        self.assertEqual(rows.t1_error_scan(quote, facts, rules), [])

    def test_masks_exclude_old_values_quotes_unknown_claims_and_lesson(self):
        record, facts, rules, note = fixture()
        note["observed_facts"].append({"fact_ref": "unknown", "value": "malicious unverified text"})
        note["lesson"] = {"text": "Check operations carefully."}
        record = set_note(record, note)
        original = deepcopy(record)
        row = rows.note_row(record, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold="4")
        selected = bytes(token for token, active in zip(row["generation"]["completion_token_ids"], row["loss_mask"])
                         if active).decode()
        self.assertNotIn("12 / 3 = 5", selected)
        self.assertNotIn('"5"', selected)
        self.assertNotIn("malicious", selected)
        self.assertNotIn("Check operations", selected)
        self.assertIn('"corrected_value": "4"', selected)
        self.assertFalse(row["loss_mask"][-1])
        self.assertEqual(row["route"], "FULL")
        self.assertIs(row["generation"], record["note"]["generation"])
        self.assertEqual(record, original)

    def test_extra_claim_fields_are_not_trained_and_fact_contradiction_rejects(self):
        record, facts, rules, note = fixture()
        note["error"]["unverified_prose"] = "secret training target"
        malformed = set_note(record, note)
        claim = next(c for c in verify(malformed, facts, rules)["claims"] if c["path"] == "error")
        self.assertEqual((claim["status"], claim["reason"]), ("UNVERIFIABLE", "ERROR_SCHEMA"))
        # The malformed claim no longer rejects the note, but only the verified fact is supervised.
        row = rows.note_row(malformed, tokenizer=TOKENIZER, fact_catalog=facts, rules=rules, choices=(), gold="4")
        trained = bytes(token for token, active in zip(row["generation"]["completion_token_ids"], row["loss_mask"])
                        if active).decode()
        self.assertNotIn("secret", trained)
        self.assertNotIn("corrected_value", trained)
        self.assertIn('"fact_ref": "f0"', trained)
        note["error"].pop("unverified_prose")
        note["observed_facts"][0]["value"] = "7"
        receipt = verify(set_note(record, note), facts, rules)
        self.assertFalse(receipt["accepted"])
        self.assertFalse(receipt["resolve_allowed"])

    def test_zero_verified_safe_lesson_allows_resolve_but_leak_does_not(self):
        record, _, _, note = fixture(catalog=False)
        note["error"] = None
        record = set_note(record, note)
        receipt = verify(record)
        self.assertFalse(receipt["accepted"])
        self.assertTrue(receipt["resolve_allowed"])
        self.assertIsNone(rows.note_row(record, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4"))
        note["lesson"] = {"text": "The answer is 4."}
        self.assertFalse(verify(set_note(record, note))["resolve_allowed"])

    def test_resolve_uses_p0_with_256_minimum_and_rejects_reasoning_mismatch(self):
        record, _, _, note = fixture(catalog=False)
        note["error"] = None
        record = set_note(record, note)
        text = LONG_PROSE
        text += "\n12 / 3 = 4\nConclusion: 4\nFinal answer: 4"
        record["resolve"] = output(text, status="CORRECT", extracted="4", prompt="lesson generation prompt")
        original = deepcopy(record)
        row, receipt = rows.c_row_with_receipt(
            record, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4")
        self.assertIsNotNone(row)
        self.assertEqual(row["prompt"], record["p0_text"])
        self.assertEqual(row["target"], text)
        evidence = {"version": "v16b-c-v1", "accepted": True,
                    "note_receipt_sha256": verify(record)["sha256"],
                    "generation_sha256": digest(record["resolve"]["generation"]),
                    "content_status": "PARTIALLY_CHECKED"}
        self.assertEqual(receipt, evidence | {"sha256": digest(evidence)})
        self.assertEqual(row["verification_receipt"], receipt)
        self.assertEqual(row, rows.c_row(
            record, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4"))
        self.assertEqual(record, original)
        record["resolve"] = output("Final answer: 4", status="CORRECT", extracted="4")
        self.assertIsNone(rows.c_row(record, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4"))
        record["resolve"] = output(text.replace("Conclusion: 4", "Conclusion: 9"), status="CORRECT", extracted="4")
        rejected, receipt = rows.c_row_with_receipt(
            record, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4")
        self.assertIsNone(rejected)
        self.assertEqual(receipt["reason"], "FINAL_CONTRADICTION")

    def test_resolve_receipts_preserve_specific_rejection_reasons_and_bindings(self):
        record, _, _, note = fixture(catalog=False)
        note["error"] = None
        record = set_note(record, note)
        text = LONG_PROSE + "\n12 / 3 = 4\nConclusion: 4\nFinal answer: 4"
        record["resolve"] = output(text, status="CORRECT", extracted="4")
        blocked_note = deepcopy(note)
        blocked_note["lesson"] = {"text": "The answer is 4."}
        changed = deepcopy(record)
        changed["resolve"]["generation"]["text"] += "changed"
        too_long = output(LONG_PROSE * 12, status="CORRECT", extracted="4")
        self.assertGreater(len(too_long["generation"]["completion_token_ids"]), 4096)
        cases = [
            (set_note(record, blocked_note), "RESOLVE_NOT_ALLOWED"),
            ({key: value for key, value in record.items() if key != "resolve"}, "MISSING_RESOLVE_OUTPUT"),
            (record | {"resolve": None}, "MISSING_RESOLVE_OUTPUT"),
            (changed, "CHANGED_ORIGINAL_GENERATION"),
            (record | {"resolve2": record["resolve"]}, "MULTIPLE_RESOLVE_ATTEMPTS"),
            (record | {"resolve3": record["resolve"]}, "MULTIPLE_RESOLVE_ATTEMPTS"),
            (record | {"resolve": too_long}, "RESOLVE_TOO_LONG"),
            (record | {"resolve": output("Final answer: 4", status="CORRECT", extracted="4")}, "A2_NOT_ADMISSIBLE"),
            (record | {"resolve": output("A calculation with separate groups.\nFinal answer: 4",
                                         status="CORRECT", extracted="4")}, "A2_UNDER_MIN_TOKENS"),
            (record | {"resolve": output(text.replace("12 / 3 = 4", "12 / 3 = 5"),
                                         status="CORRECT", extracted="4")}, "PROCESS_CONTRADICTION"),
            (record | {"resolve": output(text, status="CORRECT", extracted="5")}, "PARSE_GOLD_MISMATCH"),
            (record | {"resolve": output(LONG_PROSE + "\nFinal answer: 4\nFinal answer: 5",
                                         status="CORRECT", extracted="4")}, "PARSE_GOLD_MISMATCH"),
            (record | {"resolve": output(text.replace("Conclusion: 4", "Conclusion: 9"),
                                         status="CORRECT", extracted="4")}, "FINAL_CONTRADICTION"),
        ]
        for candidate, reason in cases:
            with self.subTest(reason=reason):
                original = deepcopy(candidate)
                row, receipt = rows.c_row_with_receipt(
                    candidate, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4")
                self.assertIsNone(row)
                self.assertFalse(receipt["accepted"])
                self.assertEqual(receipt["reason"], reason)
                self.assertEqual(receipt["note_receipt_sha256"], verify(candidate)["sha256"])
                self.assertEqual(receipt["generation_sha256"], digest(
                    (candidate.get("resolve") or {}).get("generation")))
                self.assertEqual(receipt["sha256"], digest(
                    {key: value for key, value in receipt.items() if key != "sha256"}))
                self.assertIsNone(rows.c_row(
                    candidate, tokenizer=TOKENIZER, fact_catalog=[], rules={}, choices=(), gold="4"))
                self.assertEqual(candidate, original)
                if reason == "RESOLVE_NOT_ALLOWED":
                    self.assertEqual(receipt["note_reason"], verify(candidate)["reason"])

    def test_resolve_receipt_mismatch_and_malformed_generation_still_raise(self):
        record, _, _, _ = fixture(catalog=False)
        kwargs = {"tokenizer": TOKENIZER, "fact_catalog": [], "rules": {}, "choices": (), "gold": "4"}
        for check in (rows.c_row_with_receipt, rows.c_row):
            with self.subTest(check=check.__name__):
                with self.assertRaisesRegex(ValueError, "NOTE_RECEIPT_MISMATCH"):
                    check(record, **kwargs, note_receipt=verify(record) | {"sha256": "changed"})
                with self.assertRaises(AttributeError):
                    check(record | {"resolve": {"generation": None}}, **kwargs)


class V16bAssertionRejections(unittest.TestCase):
    def test_explicitly_false_arithmetic_is_neither_trigger_nor_verified_note(self):
        quote = "12 / 3 = 5"
        for catalog in (False, True):
            for denial in (f"It is false that {quote}", f"{quote} is false",
                           f"It isn't true that {quote}", f"It is untrue that:\n{quote}"):
                with self.subTest(catalog=catalog, denial=denial):
                    record, facts, rules, note = fixture(catalog=catalog)
                    note["observed_facts"] = []
                    start = denial.index(quote)
                    note["error"]["earlier_span"] = [start, start + len(quote)]
                    record["t1"] = output(denial + "\nFinal answer: 4", status="CORRECT", extracted="4")
                    self.assertEqual(rows.t1_error_scan(record["t1"]["text"], facts, rules), [])
                    self.assertFalse(verify(set_note(record, note), facts, rules)["accepted"])
                    # A separate real error permits note generation but cannot certify the denied quote.
                    text = "2 + 2 = 5\n" + denial + "\nFinal answer: 4"
                    start = text.index(quote)
                    note["error"]["earlier_span"] = [start, start + len(quote)]
                    record["t1"] = output(text, status="CORRECT", extracted="4")
                    receipt = verify(set_note(record, note), facts, rules)
                    self.assertEqual(receipt["claims"][0]["status"], "UNVERIFIABLE", receipt)
                    self.assertEqual(receipt["verified_spans"], [])
                    self.assertTrue(receipt["resolve_allowed"])

    def test_false_or_later_replaced_natural_fact_is_unverifiable_in_scan_and_masks(self):
        record, facts, _, note = fixture()
        facts = facts[:1]
        rules = {"LOOKUP_COUNT": {"operation": "LOOKUP", "formula": FORMULAS["LOOKUP"],
                                  "operand_units": ["count"], "result_unit": "count"}}
        quote = "AB = 5"
        note["error"].update(earlier_quote=quote, kind="OBSERVATION", operand_fact_refs=["f0"],
                              operation="LOOKUP", corrected_value="12", unit="count", rule_id="LOOKUP_COUNT")
        for statement in (f"{quote} is false", f"It is false that {quote}",
                          quote + "\nAB = 12", quote + "\nsegment BA = 12"):
            with self.subTest(statement=statement):
                self.assertEqual(rows.t1_error_scan(statement, facts, rules), [])
                # A verified sibling may train, while the denied or retracted error must remain masked.
                text = "2 + 2 = 5\n" + statement + "\nFinal answer: 4"
                start = text.index(quote)
                note["error"]["earlier_span"] = [start, start + len(quote)]
                record["t1"] = output(text, status="CORRECT", extracted="4")
                receipt = verify(set_note(record, note), facts, rules)
                self.assertTrue(receipt["accepted"], receipt)
                self.assertEqual(receipt["claims"][-1]["status"], "UNVERIFIABLE", receipt)
                row = rows.note_row(set_note(record, note), tokenizer=TOKENIZER,
                                    fact_catalog=facts, rules=rules, choices=(), gold="4")
                selected = bytes(token for token, active in zip(row["generation"]["completion_token_ids"],
                                                               row["loss_mask"]) if active).decode()
                self.assertIn('"fact_ref": "f0"', selected)
                self.assertNotIn("corrected_value", selected)

    def test_only_later_asserted_same_entity_with_changed_value_retracts(self):
        _, facts, _, _ = fixture()
        for text, expected in (("AB = 5\nAB = 12", "RETRACTED"),
                               ("AB = 5\nAB = 5.0", "ASSERTED"),
                               ("AB = 5\nIf AB = 12", "ASSERTED"),
                               ("AB = 5\nAB = 12 is false", "ASSERTED"),
                               ("AB = 5\nCD = 12", "ASSERTED")):
            with self.subTest(text=text):
                self.assertEqual(rows.bind_statement(text, facts)[0]["assertion"], expected)
        ambiguous = facts + [facts[0] | {"fact_id": "duplicate"}]
        self.assertTrue(all(claim["status"] == "UNVERIFIABLE"
                            for claim in rows.bind_statement("AB = 5\nAB = 12", ambiguous)))
        self.assertEqual(rows.t1_error_scan("AB = 5\nAB = 12", ambiguous, {}), [])


class V16bPreparation(unittest.TestCase):
    def test_prep_preserves_native_self_target_and_requires_native_human_text_receipt(self):
        record, _, _, _ = fixture(catalog=False)
        text = LONG_PROSE + "\nFinal answer: 4"
        record["t1"] = output(text, status="CORRECT", extracted="4", prompt=record["p0_text"])
        fields = {"lecture": "The source provides a numerical method.",
                  "solution": "Combine the given counts through division to calculate the resulting ratio without changing units."}
        native = {"authorship": "dataset_native_expert_curated", "human_verified": False,
                  "contamination_status": "PASS_FRESH_TEXT_SCAN_WITH_LIMITS", **fields,
                  "text_sha256": {key: text_sha256(value) for key, value in fields.items()},
                  "source_ref": {"sha256": "a" * 64}, "provenance_card": {"sha256": "b" * 64},
                  "text_check_receipt": {"sha256": "c" * 64, "path": "/safe/receipt.json"}}
        record["references"] = {"human_solution": native}
        accepted = rows.prep_rows([record], tokenizer=TOKENIZER)
        self.assertEqual({row["kind"] for row in accepted}, {"PREP_SELF", "PREP_HUMAN"})
        self.assertTrue(all(row["route"] == "FULL" and row["prompt"] == record["p0_text"] for row in accepted))
        own = next(row for row in accepted if row["kind"] == "PREP_SELF")
        self.assertIs(own["generation"], record["t1"]["generation"])
        native["solution"] += " changed"
        self.assertEqual([row["kind"] for row in rows.prep_rows([record], tokenizer=TOKENIZER)], ["PREP_SELF"])

    def test_prep_hash_selection_is_deterministic_and_limits_are_not_expandable(self):
        record, _, _, _ = fixture(catalog=False)
        text = LONG_PROSE + "\nFinal answer: 4"
        record["t1"] = output(text, status="CORRECT", extracted="4", prompt=record["p0_text"])
        records = [record | {"problem_id": f"synthetic:{i}", "source": f"creator:{i % 2}"} for i in range(4)]
        first = rows.prep_rows(records, tokenizer=TOKENIZER, max_self=2)
        second = rows.prep_rows(records[::-1], tokenizer=TOKENIZER, max_self=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        with self.assertRaisesRegex(ValueError, "LIMIT"):
            rows.prep_rows(records, tokenizer=TOKENIZER, max_self=161)


if __name__ == "__main__":
    unittest.main()
