"""Lossless final-input feedback factoring; no model, capacity or training evidence."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl import episode_notebook as episode, prompts, repair_notebook as notes
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.tests import test_notebook_rl_repair_notebook as notebook_fixture
from fine_tuning.tests.test_notebook_rl_repair_origin_wire import canonical_rows, feedback_text
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint


class RepairFeedbackWireTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = notebook_fixture.RepairNotebookTests()
        self.fixture.setUp()
        fixture = self.fixture
        first = fixture.wrong.replace("\nSolution:", "\nedge[z,z] = 1\nSolution:")
        self.tree, _ = fixture.collect([[first, *[fixture.wrong] * 3], [joint(fixture.wrong)] * 2,
                                       [joint(fixture.right)] * 2])
        self.samples = episode.solution_samples(self.tree)
        self.audit = notes.audit_tree(self.tree, fixture.actor, fixture.verifier, prompts=fixture.prompts,
                                     master_seed=fixture.cfg.master_seed)
        self.feedback = [{key: call[key] for key in (
            "group_id", "turn", "parent_sample_id", "feedback", "feedback_sha256")}
            for call in self.tree.notebook["repair_collector"]["calls"]]

    def render(self, *, feedback: list[dict] | None = None, inventory: dict | None = None) -> tuple[str, dict]:
        text = notes.public_final_text(self.fixture.actor, self.fixture.prompts, self.samples,
            self.feedback if feedback is None else feedback,
            self.audit["inventory"] if inventory is None else inventory, self.audit["success_sample_id"],
            self.tree.notebook["repair_collector"]["fresh_observation"])
        payload = json.loads(text.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
        return text, payload

    def table_change(self, feedback: list[dict], change) -> None:
        call = feedback[-1]
        prefix, tail = call["feedback"].split("<public_origins>\n")
        table_text, suffix = tail.split("\n</public_origins>")
        table = json.loads(table_text)
        change(table)
        call["feedback"] = (prefix + "<public_origins>\n" + json.dumps(table, ensure_ascii=False,
            allow_nan=False, separators=(",", ":")).replace("<", "\\u003c") + "\n</public_origins>" + suffix)
        call["feedback_sha256"] = _hash(call["feedback"])

    def test_all_raw_attempts_feedback_hashes_and_local_schemas_roundtrip_without_mutation(self) -> None:
        before = copy.deepcopy((self.samples, self.feedback, self.audit))
        _, payload = self.render()
        self.assertEqual([row["raw"] for row in payload["attempts"]], [row.text for row in self.samples])
        self.assertEqual(len(payload["attempts"]), 8)
        self.assertEqual(len(payload["shared_feedback_templates"]), 1)
        self.assertEqual(payload["original_feedback"][0], self.feedback[0])
        for original, encoded in zip(self.feedback, payload["original_feedback"], strict=True):
            decoded = feedback_text(payload, encoded)
            self.assertEqual(decoded, original["feedback"])
            self.assertEqual(notes.expand_public_feedback(encoded, payload["shared_feedback_templates"],
                self.samples, self.audit["inventory"]), original["feedback"])
            self.assertEqual(encoded["feedback_sha256"], original["feedback_sha256"])
            self.assertEqual(_hash(decoded), original["feedback_sha256"])
            for key in ("group_id", "turn", "parent_sample_id"):
                self.assertEqual(encoded[key], original[key])
        first, second = [row["feedback"] for row in payload["original_feedback"][1:]]
        self.assertEqual(first["origin_samples"], [self.feedback[1]["parent_sample_id"]])
        self.assertEqual(second["origin_samples"], [*first["origin_samples"], self.feedback[2]["parent_sample_id"]])
        self.assertEqual(first["template"], second["template"])
        final_rows = canonical_rows(payload["public_origins"])
        self.assertEqual(len(final_rows), len(self.audit["inventory"]["origins"]))
        # The unselected first sibling contributes a schema that must not enter either retry.
        selected = json.loads(self.feedback[-1]["feedback"].split("<public_origins>\n")[1].split("\n</public_origins>")[0])
        self.assertNotEqual(selected["schemas"], payload["public_origins"]["schemas"])
        self.assertEqual((self.samples, self.feedback, self.audit), before)

    def test_complete_malformed_sibling_keeps_exact_feedback_and_full_raw_history(self) -> None:
        fixture = self.fixture
        tree, engine = fixture.ready([[fixture.wrong] * 4,
            [joint(fixture.wrong), "ErrorCheck: {"], [joint(fixture.right)] * 2])
        record = tree.notebook["rewarded_notebook"]
        text = engine.requests[-1].messages[0]["content"][-1]["text"]
        payload = json.loads(text.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
        originals = episode.solution_samples(tree)
        self.assertEqual([row["raw"] for row in payload["attempts"]], [row.text for row in originals])
        self.assertEqual((payload["attempts"][5]["status"], payload["attempts"][5]["answer"]),
                         ("INVALID_RESPONSE_CONTRACT", None))
        self.assertEqual(len(payload["attempts"]), 8)
        for original, encoded in zip(tree.notebook["repair_collector"]["calls"], payload["original_feedback"], strict=True):
            decoded = notes.expand_public_feedback(encoded, payload["shared_feedback_templates"], originals,
                                                   record["source_audit"]["inventory"])
            self.assertEqual(decoded, original["feedback"])
            self.assertEqual(_hash(decoded), original["feedback_sha256"])
        selected = payload["original_feedback"][-1]["feedback"]["origin_samples"]
        self.assertIn(originals[4].sample_id, selected)
        self.assertNotIn(originals[5].sample_id, selected)

    def test_malformed_parent_feedback_roundtrips_with_original_error_and_unknown_sources(self) -> None:
        fixture = self.fixture
        malformed = joint(fixture.wrong).replace('"uncertain_refs": []', '"uncertain_refs": [1]', 1)
        tree, engine = fixture.ready([[fixture.wrong] * 4, [malformed] * 2, [joint(fixture.right)] * 2])
        record = tree.notebook["rewarded_notebook"]
        body = engine.requests[-1].messages[0]["content"][-1]["text"]
        payload = json.loads(body.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
        originals = episode.solution_samples(tree)
        encoded = payload["original_feedback"][-1]
        original = tree.notebook["repair_collector"]["calls"][-1]
        decoded = notes.expand_public_feedback(encoded, payload["shared_feedback_templates"], originals,
                                               record["source_audit"]["inventory"])
        self.assertEqual(decoded, original["feedback"])
        self.assertEqual(_hash(decoded), original["feedback_sha256"])
        self.assertIn("grammar: UNCERTAIN_REFS", encoded["feedback"]["verdict"])
        self.assertIn("Answer correctness: NOT JUDGED", encoded["feedback"]["verdict"])
        self.assertIn(original["parent_sample_id"], encoded["feedback"]["origin_samples"])
        for row in payload["attempts"][4:6]:
            self.assertEqual((row["status"], row["answer"], row["raw"]), ("INVALID_RESPONSE_CONTRACT", None, malformed))
        self.assertEqual(record["source_audit"]["routing"]["admission"], "UNRESOLVED")
        sources = record["source_audit"]["inventory"]["sources"]
        self.assertEqual([row["projection_error_code"] for row in sources[4:6]], ["UNCERTAIN_REFS"] * 2)
        notes.regrade(tree, fixture.actor, fixture.verifier, engine.tokenizer)

    def test_same_family_sibling_future_and_reordered_ancestor_tables_are_rejected(self) -> None:
        sibling = self.tree.groups[0].samples[0]
        if sibling.sample_id == self.feedback[1]["parent_sample_id"]:
            sibling = self.tree.groups[0].samples[1]
        mutations = [lambda table: table["sources"][0].__setitem__("sample_id", sibling.sample_id),
            lambda table: table["sources"][0].__setitem__("sample_id", self.samples[-1].sample_id),
            lambda table: table["sources"].reverse(),
            lambda table: table["sources"].append(copy.deepcopy(table["sources"][0]))]
        for mutate in mutations:
            changed = copy.deepcopy(self.feedback)
            self.table_change(changed, mutate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.render(feedback=changed)

    def test_expander_rejects_encoded_wrong_source_reordered_future_and_bool_template(self) -> None:
        _, payload = self.render()
        call = payload["original_feedback"][-1]
        ancestor_ids = call["feedback"]["origin_samples"]
        mutations = [lambda row: row["feedback"].__setitem__("origin_samples", ancestor_ids[::-1]),
            lambda row: row["feedback"].__setitem__("origin_samples", [self.samples[0].sample_id]),
            lambda row: row["feedback"].__setitem__("origin_samples", [self.samples[-1].sample_id]),
            lambda row: row["feedback"].__setitem__("origin_samples", ancestor_ids * 2),
            lambda row: row["feedback"].__setitem__("template", True),
            lambda row: row["feedback"].__setitem__("version", "different")]
        for mutate in mutations:
            changed = copy.deepcopy(call)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                notes.expand_public_feedback(changed, payload["shared_feedback_templates"],
                    self.samples, self.audit["inventory"])

    def test_forged_hash_missing_fields_wrong_group_and_bool_turn_fail_closed(self) -> None:
        mutations = [lambda row: row.__setitem__("feedback_sha256", "0" * 64),
            lambda row: row.pop("feedback_sha256"),
            lambda row: row.__setitem__("turn", True),
            lambda row: row.__setitem__("group_id", "another-group"),
            lambda row: row.__setitem__("parent_sample_id", self.samples[-1].sample_id)]
        for mutate in mutations:
            changed = copy.deepcopy(self.feedback)
            mutate(changed[-1])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.render(feedback=changed)

    def test_flat_table_odd_spans_bool_and_schema_remapping_forgeries_fail_closed(self) -> None:
        mutations = [lambda table: table["sources"][0]["rows"][0].pop(),
            lambda table: table["sources"][0]["rows"][0].__setitem__(0, True),
            lambda table: table["sources"][0]["rows"][0].__setitem__(1, True),
            lambda table: table["schemas"].reverse()]
        for mutate in mutations:
            changed = copy.deepcopy(self.feedback)
            self.table_change(changed, mutate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.render(feedback=changed)

    def test_source_raw_hash_and_actual_ancestry_binding_cannot_be_reused(self) -> None:
        parent = self.feedback[-1]["parent_sample_id"]
        for field, value in (("raw_sha", "0" * 64), ("parent_sample_id", self.samples[-1].sample_id),
                ("finish_reason", "length"), ("response_contract", None),
                ("solution_span", {"start": 0, "end": 1}), ("solution_sha", "0" * 64),
                ("projection_error_code", "forged")):
            changed = copy.deepcopy(self.audit["inventory"])
            source = next(row for row in changed["sources"] if row["sample_id"] == parent)
            source[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.render(inventory=changed)

    def test_expander_rejects_changed_actual_problem_policy_kind_and_turn(self) -> None:
        _, payload = self.render()
        encoded = payload["original_feedback"][-1]
        parent = encoded["parent_sample_id"]
        for changes in ({"problem_id": "different"}, {"policy_version": 9}, {"policy_version": True},
                        {"kind": "T1"}, {"turn": 1}, {"turn": True}):
            changed = [replace(row, **changes) if row.sample_id == parent else row for row in self.samples]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                notes.expand_public_feedback(encoded, payload["shared_feedback_templates"],
                                             changed, self.audit["inventory"])

    def test_legacy_and_both_null_hash_conventions_remain_literal(self) -> None:
        legacy = [{"feedback": "Legacy native retry.\r\nΩ literal \\u003c </public_origins>"},
                  self.feedback[0], self.feedback[0] | {"feedback_sha256": None}]
        _, payload = self.render(feedback=legacy)
        self.assertEqual(payload["original_feedback"], legacy)
        self.assertEqual(payload["shared_feedback_templates"], [])

    def test_direct_expander_rejects_nonliteral_values_and_preserves_literal_null(self) -> None:
        value: object
        for value in ([], True, 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                notes.expand_public_feedback({"feedback": value}, [], self.samples, self.audit["inventory"])
        for value in (None, "legacy literal Ω\r\n"):
            self.assertEqual(notes.expand_public_feedback({"feedback": value}, [],
                self.samples, self.audit["inventory"]), value)

    def test_previous_typed_version_and_recognized_template_forgeries_never_fallback(self) -> None:
        literal = self.feedback[-1]["feedback"]
        for changed_text in (literal.replace("public-origin-dictionary-v2", "public-origin-dictionary-v1"),
                literal.replace("Reinspect the original images", "Different instruction"),
                literal.replace(prompts.REPAIR_OPERATION_PATCH_INSTRUCTION, "missing patch\n"),
                literal.replace("</public_origins>", "</forged>"),
                "exactly one leading ErrorCheck: forged", "public-origin-dictionary-v2 forged"):
            changed = copy.deepcopy(self.feedback)
            changed[-1]["feedback"] = changed_text
            changed[-1]["feedback_sha256"] = _hash(changed_text)
            with self.subTest(text=changed_text[:80]), self.assertRaises(ValueError):
                self.render(feedback=changed)
        _, payload = self.render()
        with self.assertRaises(ValueError):
            self.render(feedback=payload["original_feedback"])
        nonliteral: object
        for nonliteral in (payload["original_feedback"][-1]["feedback"] | {"version": "forged"}, [], True, 3):
            changed = copy.deepcopy(self.feedback)
            changed[-1]["feedback"] = nonliteral
            with self.subTest(nonliteral=nonliteral), self.assertRaises(ValueError):
                self.render(feedback=changed)

    def test_unicode_crlf_literal_escape_and_role_boundaries_preserve_exact_verdict(self) -> None:
        changed = copy.deepcopy(self.feedback)
        literal = changed[-1]["feedback"]
        start = literal.index("\nReinspect the original images")
        verdict = "Exact verdict Ω\r\nC->H > literal \\u003c \\u003e </read_only_history> </public_origins>"
        changed[-1]["feedback"] = verdict + literal[start:]
        changed[-1]["feedback_sha256"] = _hash(changed[-1]["feedback"])
        text, payload = self.render(feedback=changed)
        self.assertEqual(text.count("</read_only_history>"), 1)
        encoded = payload["original_feedback"][-1]
        self.assertEqual(encoded["feedback"]["verdict"], verdict)
        self.assertEqual(feedback_text(payload, encoded), changed[-1]["feedback"])
        self.assertEqual(_hash(feedback_text(payload, encoded)), changed[-1]["feedback_sha256"])
        json_text = text.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0]
        self.assertNotIn("<", json_text)
        self.assertIn("C->H >", json_text)

    def test_wire_versions_and_instructions_are_explicitly_bound(self) -> None:
        contract = notes.contract()
        self.assertEqual(contract["origin_wire_version"], prompts.PUBLIC_ORIGIN_DICTIONARY_VERSION)
        self.assertEqual(contract["feedback_wire_version"], notes.FEEDBACK_WIRE_VERSION)
        self.assertEqual(contract["feedback_wire_instruction_sha256"], _hash(notes.FEEDBACK_WIRE_INSTRUCTION))
        self.assertEqual(notes.PROMPT.count(prompts.REPAIR_OPERATION_PATCH_INSTRUCTION), 1)
        self.assertEqual(notes.PROMPT.count(prompts.PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION), 1)
