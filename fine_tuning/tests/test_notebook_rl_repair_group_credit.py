"""Hand-calculated CPU repair credit with actual authored audit fixtures; no model training."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from fine_tuning.notebook_rl import episode_notebook as episode
from fine_tuning.notebook_rl.repair_contract import FINAL_VERSION, ORIGIN_VERSION, VERSION, parse_retry_sections, public_origin_inventory
from fine_tuning.notebook_rl.repair_credit import (
    FinalRepairAudit, RepairAudit, _digest, audit_final_note, audit_online_repair, repair_group_credit,
)
from fine_tuning.notebook_rl.repair_tokens import emitted_patch_spans
from fine_tuning.notebook_rl.schema import Group, Sample, VerifierResult, to_json
from fine_tuning.tests.test_notebook_rl_notebook_synthesis import graph_native, graph_response
from fine_tuning.tests.test_notebook_rl_repair_credit import joint, sample
from fine_tuning.tests import test_notebook_rl_repair_notebook as notebook_fixture


class Characters:
    def __init__(self) -> None:
        self.chunks: dict[int, str] = {}

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(self.chunks[token] if token in self.chunks else chr(token) for token in ids)


def encoded(row: Sample) -> Sample:
    return replace(row, completion_token_ids=[ord(char) for char in row.text], sampler_logprobs=[-.1] * len(row.text))


class RepairGroupCreditTests(unittest.TestCase):
    audits: dict[str, RepairAudit | FinalRepairAudit | None]

    def setUp(self) -> None:
        self.actor, self.item, self.verifier = graph_native("shortest")
        self.trace = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None, "weights": [2, 2], "cost": 4}
        self.parent = encoded(sample(self.actor, "p", graph_response(self.item.slots, {**self.trace, "cost": 3}, "4")))
        self.tokenizer = Characters()

    def online(self, modes: tuple[str, str] = ("positive", "unknown"), *, pack_patch: bool = False):
        self.inventory = public_origin_inventory([self.parent], contract_version=ORIGIN_VERSION,
                                                response_contracts={"p": None}, parent_sample_id="p")
        ref = next(row.ref_id for row in self.inventory.origins if row.rule == "COST_SUM")
        children: list[Sample] = []
        audits: dict[str, RepairAudit | FinalRepairAudit | None] = {}
        for index, mode in enumerate(modes):
            body = graph_response(self.item.slots, self.trace, "4")
            if mode == "failed":
                body = graph_response(self.item.slots, {**self.trace, "cost": 3}, "4")
            text = joint(body, ref, intent="UNKNOWN" if mode == "unknown" else "RECHECK_OPERATION",
                         proposed=None if mode == "unknown" else {"field": "cost", "value": 4})
            if mode == "empty":
                text = 'ErrorCheck: {"version":"repair-check-v1","claims":[],"uncertain_refs":[]}\nRevisedSolution:\n' + body
            if mode == "malformed":
                text = "ErrorCheck: {"
            if mode == "version":
                text = text.replace('"repair-check-v1"', '"model-wrong-version"')
            child = encoded(replace(sample(self.actor, f"c{index}", text, self.parent), group_id="g2", seed=3407 + index))
            if pack_patch and index == 0:
                patch_spans = emitted_patch_spans(text, parse_retry_sections(text, contract_version=VERSION),
                                                  kind="TRACE", rule="COST_SUM", locus="cost")
                assert patch_spans is not None
                span, = patch_spans
                self.tokenizer.chunks[300000] = text[span.start - 1:span.end + 1]
                ids = [ord(char) for char in text[:span.start - 1]] + [300000] + [ord(char) for char in text[span.end + 1:]]
                child = replace(child, completion_token_ids=ids, sampler_logprobs=[-.1] * len(ids))
            children.append(child)
            audits[child.sample_id] = None if mode in ("malformed", "version") else audit_online_repair(
                [self.parent], child, self.actor, self.verifier, inventory=self.inventory,
                response_contracts={"p": None, child.sample_id: VERSION},
                sections=parse_retry_sections(child.text, contract_version=VERSION))
        self.group = Group("g2", "T2", self.actor.problem_id, _digest(children[0].prompt_token_ids), children,
                           True, None)
        self.audits, self.ref = audits, ref
        return repair_group_credit(self.group, audits, self.tokenizer, inventory=self.inventory,
                                   completion_cap=1024, component="online")

    def final(self, modes: tuple[str, str, str, str] = ("positive", "negative", "structural", "empty")):
        child = encoded(sample(self.actor, "c", joint(graph_response(self.item.slots, self.trace, "4"),
                                                     "p:r1", intent="UNKNOWN"), self.parent))
        history = [self.parent, child]
        contracts = {"p": None, "c": VERSION}
        self.inventory = public_origin_inventory(history, contract_version=ORIGIN_VERSION, response_contracts=contracts)
        ref = next(row.ref_id for row in self.inventory.origins if row.sample_id == "p" and row.rule == "COST_SUM")
        later = next(row.ref_id for row in self.inventory.origins if row.sample_id == "c" and row.rule == "COST_SUM")
        attempts = []
        for row in history:
            result = self.verifier.grade(row.problem_id, row.text, row.finish_reason, "obs", response_contract=contracts[row.sample_id])
            attempts.append({"sample_id": row.sample_id, "outcome": result.status, "answer": result.extracted})
        notes: list[Sample] = []
        audits: dict[str, RepairAudit | FinalRepairAudit | None] = {}
        for index, mode in enumerate(modes):
            claim: dict = {"origin_ref": ref, "later_ref": "foreign:r1" if mode == "structural" else later,
                     "relation": "CORRECTED", "rule": "COST_SUM",
                     "repaired_value": {"field": "cost", "value": 3 if mode == "negative" else 4}}
            if mode == "output":
                claim = {"origin_ref": next(row.ref_id for row in self.inventory.origins
                                            if row.sample_id == "p" and row.kind == "OUTPUT"),
                         "later_ref": next(row.ref_id for row in self.inventory.origins
                                           if row.sample_id == "c" and row.kind == "OUTPUT"),
                         "relation": "FINALIZED", "rule": "FINALIZATION", "repaired_value": None}
            data = {"version": FINAL_VERSION, "scope": "BOUNDED_CLAIMS_FULL_HISTORY", "attempts": attempts,
                    "success_sample_id": "p" if mode == "metadata" else "c", "claims": [] if mode == "empty" else [claim],
                    "uncertain_refs": [], "lesson_text": "FREE_PROSE_MUST_NOT_OWN_TOKENS"}
            text = json.dumps(data, separators=(",", ":"))
            note = encoded(Sample(f"n{index}", self.actor.problem_id, "diag", "DIAG", 1, None,
                                  [77], [88], [-.1], "stop", text, 3407 + index, 0))
            notes.append(note)
            audits[note.sample_id] = audit_final_note(note, history, self.actor, self.verifier,
                                                     inventory=self.inventory, response_contracts=contracts)
        self.group = Group("diag", "DIAG", self.actor.problem_id, _digest([77]), notes, True, None)
        self.audits, self.ref = audits, ref
        return repair_group_credit(self.group, audits, self.tokenizer, inventory=self.inventory,
                                   completion_cap=8192, component="final")

    def row(self, credit):
        return next(row for row in credit.unit_rows if row["origin_ref"] == self.ref)

    def test_online_hand_calculation_claim_patch_union_and_claim_only_negative(self) -> None:
        credits = self.online()
        self.assertEqual([self.row(credit)["applied_score"] for credit in credits], [1, 0])
        self.assertEqual([self.row(credit)["mean"] for credit in credits], [.5, .5])
        self.assertEqual([self.row(credit)["advantage"] for credit in credits], [.5, -.5])
        for credit in credits:
            row = self.row(credit)
            expected = 1024 * .0625 * row["advantage"] / (self.inventory.fixed_units * len(row["token_indices"]))
            self.assertEqual(row["coefficient"], expected)
            self.assertTrue(all(credit.coefficients[index] == expected for index in row["token_indices"]))
            self.assertEqual(credit.original_group_size, 2)
            self.assertFalse(credit.source_bindings["optimizer_admission"])
        positive, zero = [self.row(credit) for credit in credits]
        self.assertEqual(len(positive["raw_spans"]), 2)
        self.assertEqual(len(zero["raw_spans"]), 1)
        self.assertTrue(positive["positive_verified_corrected"])
        self.assertFalse(zero["corrected_channel"])

    def test_uniform_rewards_have_zero_coefficients_and_unsubmitted_owns_nothing(self) -> None:
        for mode in ("positive", "unknown", "empty"):
            for credit in self.online((mode, mode)):
                self.assertFalse(any(credit.coefficients))
        credits = self.online(("positive", "empty"))
        row = self.row(credits[1])
        self.assertEqual(row["advantage"], -.5)
        self.assertEqual(row["token_indices"], [])
        self.assertFalse(any(credits[1].coefficients))
        for credit in self.final(("negative",) * 4):
            self.assertFalse(any(credit.coefficients))
            self.assertEqual(self.row(credit)["mean"], -1.)

    def test_resolved_failed_recheck_retains_claim_and_actual_patch_ownership(self) -> None:
        credits = self.online(("positive", "failed"))
        row = self.row(credits[1])
        self.assertEqual((row["semantic_score"], row["applied_score"], row["advantage"]), (0, 0, -.5))
        self.assertEqual(len(row["raw_spans"]), 2)
        start, end = row["raw_spans"][-1]
        self.assertEqual(self.group.samples[1].text[start:end], "3")
        self.assertLess(credits[1].coefficients[start], 0.)

    def test_final_signed_hand_calculation_preserves_corrected_negative_and_free_prose(self) -> None:
        credits = self.final()
        self.assertEqual([self.row(credit)["applied_score"] for credit in credits], [1, -1, 0, 0])
        self.assertEqual([self.row(credit)["mean"] for credit in credits], [0.] * 4)
        self.assertTrue(self.row(credits[1])["corrected_channel"])
        self.assertFalse(self.row(credits[1])["positive_verified_corrected"])
        for sample_row, credit in zip(self.group.samples, credits, strict=True):
            row = self.row(credit)
            if row["token_indices"]:
                expected = 8192 * .0625 * row["advantage"] / (self.inventory.fixed_units * len(row["token_indices"]))
                self.assertTrue(all(credit.coefficients[index] == expected for index in row["token_indices"]))
            start = sample_row.text.index("FREE_PROSE_MUST_NOT_OWN_TOKENS")
            self.assertFalse(any(credit.coefficients[start:start + len("FREE_PROSE_MUST_NOT_OWN_TOKENS")]))

    def test_metadata_invalid_applied_zero_overrides_proposed_positive_and_output_is_zero(self) -> None:
        credits = self.final(("positive", "positive", "positive", "metadata"))
        audit = self.audits["n3"]
        assert isinstance(audit, FinalRepairAudit)
        self.assertTrue(any(unit.value == 1 for unit in audit.proposed_unit_scores))
        self.assertFalse(any(unit.value for unit in audit.unit_scores))
        row = self.row(credits[3])
        self.assertEqual((row["semantic_score"], row["applied_score"], row["mean"], row["advantage"]), (0, 0, .75, -.75))
        credits = self.final(("output",) * 4)
        self.assertFalse(any(value for credit in credits for value in credit.coefficients))
        self.assertFalse(any(row["corrected_channel"] for credit in credits for row in credit.unit_rows if row["kind"] == "OUTPUT"))

    def test_malformed_member_retains_original_n_in_usable_group(self) -> None:
        credits = self.online(("positive", "malformed"))
        self.assertEqual([self.row(credit)["mean"] for credit in credits], [.5, .5])
        self.assertFalse(any(credits[1].coefficients))
        self.assertIsNone(credits[1].source_bindings["audit_sha256"])
        with self.assertRaises(ValueError):
            repair_group_credit(replace(self.group, usable=False, reason="INVALID_RESPONSE_CONTRACT"), self.audits, self.tokenizer,
                                inventory=self.inventory, completion_cap=1024, component="online")
        self.online()
        with self.assertRaises(ValueError):
            repair_group_credit(replace(self.group, usable=False, reason="INVALID_RESPONSE_CONTRACT"), self.audits,
                                self.tokenizer, inventory=self.inventory, completion_cap=1024, component="online")

    def test_all_malformed_members_keep_full_n_without_signal(self) -> None:
        credits = self.online(("malformed", "malformed"))
        self.assertEqual([credit.original_group_size for credit in credits], [2, 2])
        self.assertFalse(any(value for credit in credits for value in credit.coefficients))
        self.assertTrue(all(row["mean"] == 0 for credit in credits for row in credit.unit_rows))

    def test_malformed_member_cannot_acquire_an_invented_native_result(self) -> None:
        self.online(("positive", "malformed"))
        self.group.samples[1].result = VerifierResult("CORRECT", "4", 1., None, "fixture-v1")
        with self.assertRaisesRegex(ValueError, "malformed audit or integrity failure"):
            repair_group_credit(self.group, self.audits, self.tokenizer, inventory=self.inventory,
                                completion_cap=1024, component="online")

    def test_model_version_failure_is_zero_but_caller_version_is_fatal(self) -> None:
        credits = self.online(("positive", "version"))
        self.assertEqual([self.row(credit)["mean"] for credit in credits], [.5, .5])
        self.assertFalse(any(credits[1].coefficients))
        self.assertIsNone(credits[1].source_bindings["audit_sha256"])

    def test_wrong_final_payload_version_retains_original_n4_at_zero(self) -> None:
        self.final()
        malformed = encoded(replace(self.group.samples[-1],
                                    text=self.group.samples[-1].text.replace(FINAL_VERSION, "repair-note-v1")))
        self.group.samples[-1] = malformed
        self.audits[malformed.sample_id] = None
        credits = repair_group_credit(self.group, self.audits, self.tokenizer, inventory=self.inventory,
                                      completion_cap=8192, component="final")
        self.assertEqual([credit.original_group_size for credit in credits], [4] * 4)
        self.assertFalse(any(credits[-1].coefficients))

    def test_positive_patch_mapping_failure_clears_unit_before_group_mean(self) -> None:
        credits = self.online(("positive", "positive"), pack_patch=True)
        failed, kept = [self.row(credit) for credit in credits]
        self.assertEqual((failed["semantic_score"], failed["applied_score"], failed["mapping_status"]), (1, 0, "MAPPING_FAILED"))
        self.assertEqual(failed["token_indices"], [])
        self.assertEqual(failed["advantage"], -.5)
        self.assertEqual(kept["advantage"], .5)
        self.assertFalse(any(credits[0].coefficients))

    def test_p08_complete_final_retains_malformed_member_and_original_group(self) -> None:
        fixture = notebook_fixture.RepairNotebookTests()
        fixture.setUp()
        tree, engine = fixture.collect([[fixture.right] * 4])
        engine.replies.append([fixture.final(tree)] * 3 + ["{"])
        fixture.append(tree, engine)
        before = to_json(tree)
        group = tree.groups[-1]
        history = episode.solution_samples(tree)
        contracts = tree.notebook["repair_collector"]["response_contracts"]
        inventory = public_origin_inventory(history, contract_version=ORIGIN_VERSION, response_contracts=contracts)
        audits: dict[str, RepairAudit | FinalRepairAudit | None] = {
            row.sample_id: None if row.text == "{" else audit_final_note(
            row, history, fixture.actor, fixture.verifier, inventory=inventory, response_contracts=contracts)
            for row in group.samples}
        credits = repair_group_credit(group, audits, engine.tokenizer, inventory=inventory,
                                      completion_cap=8192, component="final")
        self.assertEqual((group.usable, group.reason, len(group.samples)), (True, None, 4))
        self.assertEqual([credit.sample_id for credit in credits], [row.sample_id for row in group.samples])
        self.assertTrue(all(credit.original_group_size == 4 and credit.fixed_units == inventory.fixed_units
                            for credit in credits))
        self.assertIsNone(credits[-1].source_bindings["audit_sha256"])
        self.assertTrue(all(row["applied_score"] == 0 and not row["token_indices"] for row in credits[-1].unit_rows))
        self.assertFalse(any(credits[-1].coefficients))
        self.assertEqual(to_json(tree), before)

    def test_unsubmitted_units_reuse_one_initial_decode_validation_per_candidate(self) -> None:
        self.final(("empty",) * 4)
        self.assertGreaterEqual(self.inventory.fixed_units, 32)
        with patch.object(self.tokenizer, "decode", wraps=self.tokenizer.decode) as decode:
            credits = repair_group_credit(self.group, self.audits, self.tokenizer, inventory=self.inventory,
                                          completion_cap=8192, component="final")
        self.assertEqual(decode.call_count, sum(len(row.completion_token_ids) + 2 for row in self.group.samples))
        self.assertTrue(all(not any(credit.coefficients) for credit in credits))

    def test_missing_audit_token_raw_hash_inventory_and_group_tampering_rejected(self) -> None:
        self.online()
        for mutation in ("missing", "none", "hash", "unit", "witness", "token", "raw", "logprob", "cap", "state", "inventory"):
            group, audits, inventory, cap = copy.deepcopy(self.group), copy.deepcopy(self.audits), self.inventory, 1024
            if mutation == "missing":
                audits.pop("c0")
            elif mutation == "none":
                audits["c0"] = None
            elif mutation == "hash":
                audit = audits["c0"]
                assert isinstance(audit, RepairAudit)
                audit.source_bindings["sha256"] = "0" * 64
            elif mutation == "unit":
                audit = audits["c0"]
                assert isinstance(audit, RepairAudit)
                units = list(audit.unit_scores)
                index = next(i for i, row in enumerate(units) if row.origin_ref == self.ref)
                units[index] = replace(units[index], value=0)
                audits["c0"] = replace(audit, unit_scores=tuple(units))
            elif mutation == "witness":
                audit = audits["c0"]
                assert isinstance(audit, RepairAudit)
                audits["c0"] = replace(audit, witnesses=(replace(audit.witnesses[0], later_ref="c0:r999"),))
            elif mutation == "token":
                group.samples[0].completion_token_ids[0] += 1
            elif mutation == "raw":
                group.samples[0].text = group.samples[0].text.replace('"value":4', '"value":5')
            elif mutation == "logprob":
                group.samples[0].sampler_logprobs[0] = -.2
            elif mutation == "cap":
                cap = 8192
            elif mutation == "state":
                group = replace(group, state_sha256="0" * 64)
            else:
                inventory = replace(inventory, origins=tuple(reversed(inventory.origins)))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                repair_group_credit(group, audits, self.tokenizer, inventory=inventory, completion_cap=cap, component="online")

    def test_inter_unit_patch_overlap_is_rejected_without_redistribution(self) -> None:
        trace = {**self.trace, "nodes": ["H", "B", "A"]}
        self.parent = encoded(sample(self.actor, "p", graph_response(self.item.slots, trace, "4")))
        self.inventory = public_origin_inventory([self.parent], contract_version=ORIGIN_VERSION,
                                                response_contracts={"p": None}, parent_sample_id="p")
        origins = [next(row for row in self.inventory.origins if row.rule == rule and row.locus == locus)
                   for rule, locus in (("PATH_ENDPOINTS", "nodes"), ("DIRECTED_EDGE", "nodes[0:2]"))]
        claims = [{"origin_ref": row.ref_id, "intent": "RECHECK_OPERATION", "rule": row.rule,
                   "proposed_value": {"field": "nodes", "value": ["A", "B", "H"] if row.rule == "PATH_ENDPOINTS" else ["B", "H"]}}
                  for row in origins]
        raw = "ErrorCheck: " + json.dumps({"version": VERSION, "claims": claims, "uncertain_refs": []},
                                           separators=(",", ":")) + "\nRevisedSolution:\n" + graph_response(self.item.slots, self.trace, "4")
        children: list[Sample] = []
        audits: dict[str, RepairAudit | FinalRepairAudit | None] = {}
        for index in range(2):
            child = encoded(replace(sample(self.actor, f"c{index}", raw, self.parent), group_id="g2"))
            children.append(child)
            audits[child.sample_id] = audit_online_repair([self.parent], child, self.actor, self.verifier,
                inventory=self.inventory, response_contracts={"p": None, child.sample_id: VERSION},
                sections=parse_retry_sections(raw, contract_version=VERSION))
        group = Group("g2", "T2", self.actor.problem_id, _digest(children[0].prompt_token_ids), children, True, None)
        with self.assertRaisesRegex(ValueError, "collision"):
            repair_group_credit(group, audits, self.tokenizer, inventory=self.inventory, completion_cap=1024, component="online")

    def test_inputs_are_detached_and_zero_units_have_zero_vectors(self) -> None:
        self.online()
        before = to_json(self.group), copy.deepcopy(asdict(self.inventory)), copy.deepcopy(self.audits)
        original = self.tokenizer.decode

        def mutate(ids, **kwargs):
            self.group.samples[0].text = "changed outside snapshot"
            self.audits.clear()
            return original(ids, **kwargs)

        with patch.object(self.tokenizer, "decode", side_effect=mutate):
            credits = repair_group_credit(self.group, self.audits, self.tokenizer,
                                           inventory=self.inventory, completion_cap=1024, component="online")
        self.assertTrue(any(credits[0].coefficients))
        self.assertEqual(asdict(self.inventory), before[1])
        empty_inventory = public_origin_inventory([], contract_version=ORIGIN_VERSION, response_contracts={})
        rows = [encoded(Sample(f"bad{i}", self.actor.problem_id, "diag", "DIAG", 1, None,
                               [77], [88], [-.1], "stop", "{", i, 0)) for i in range(4)]
        group = Group("diag", "DIAG", self.actor.problem_id, _digest([77]), rows, True, None)
        credits = repair_group_credit(group, {row.sample_id: None for row in rows}, Characters(),
                                      inventory=empty_inventory, completion_cap=8192, component="final")
        self.assertTrue(all(credit.fixed_units == 0 and not any(credit.coefficients) for credit in credits))


if __name__ == "__main__":
    unittest.main()
