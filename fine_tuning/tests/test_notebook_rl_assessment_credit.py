"""Original authored assessment literals test CPU credit, never model improvement."""
import copy
import json
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from fine_tuning.notebook_rl import episode_notebook as episode, notebook_objective
from fine_tuning.notebook_rl import notebook_synthesis as synthesis, synthesis_credit as credit
from fine_tuning.notebook_rl import whole_reference_notebook as reference
from fine_tuning.notebook_rl.local_eval import digest
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as fixtures
from fine_tuning.tests import test_notebook_rl_whole_reference_training_credit as legacy_fixtures
from fine_tuning.tests.test_notebook_rl_updates import Characters
from fine_tuning.tests.test_notebook_rl_whole_reference_notebook import compact


class AssessmentCreditTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.item, self.verifier = fixtures.native()
        wrong = fixtures.response(self.item.slots)
        self.prepare([wrong] * 4 + [fixtures.response(self.item.slots, answer="accept")] * 2)

    def prepare(self, texts):
        self.tree = fixtures.tree_from(self.actor, self.verifier, texts)
        self.audit = synthesis.audit_episode(self.tree, self.actor, self.verifier, contract=synthesis.BOUNDED_VERSION)
        self.originals = episode.solution_samples(self.tree)
        self.note = compact(self.originals, fixtures.note_for(self.tree, self.audit, self.item))
        self.inventory = reference.public_inventory(self.originals)
        self.tree.notebook["rewarded_notebook"] = {"notebook_generation_policy": reference.VERSION}

    def score(self, note=None, *, text=None, finish="stop", index=0):
        raw = json.dumps(self.note if note is None else note, separators=(",", ":")) if text is None else text
        sample = replace(self.originals[0], sample_id=f"assessment-{index}", group_id="assessment-four", kind="DIAG",
            parent_sample_id=None, text=raw, finish_reason=finish, prompt_token_ids=[1, 2],
            completion_token_ids=list(map(ord, raw)), sampler_logprobs=[-.1] * len(raw), result=None, reward=0.)
        groups = [group for group in self.tree.groups if group.kind != "DIAG"]
        existing = [old for group in self.tree.groups if group.kind == "DIAG" for old in group.samples
                    if old.sample_id != sample.sample_id]
        self.tree.groups = [*groups, replace(groups[0], group_id=sample.group_id, kind="DIAG", samples=[*existing, sample],
            usable=bool(existing), reason=None if existing else "FIXTURE_GROUP_BUILDING")]
        row = credit.score_note(raw, finish, self.tree, self.audit, self.verifier, actor=self.actor,
            sample_id=sample.sample_id, objective_version=credit.ASSESSMENT_VERSION)
        sample.reward = row["strict_reward"]
        credit.validate_score(row, sample)
        return sample, row

    def group(self, notes, *, tokenizer=None):
        pairs = [self.score(note, index=index) for index, note in enumerate(notes)]
        samples = [sample for sample, _ in pairs]
        scores = {sample.sample_id: row for sample, row in pairs}
        return samples, scores, credit.group_credit(samples, scores, tokenizer or Characters())

    def without_decisions(self):
        note = copy.deepcopy(self.note)
        note["corrected_facts"] = {}
        for row in note["assessments"]:
            row[3] = None
        return note

    def test_legacy_contract_and_full_score_are_byte_compatible(self):
        from fine_tuning.notebook_rl import local_profile

        fixture = legacy_fixtures.WholeReferenceTrainingCreditTests()
        fixture.setUp()
        self.assertEqual(digest(credit.contract()), "9b4b9adc47fc431fd9e1fbaf88e8d4c3012ad027612849112829e6b6ff8be341")
        self.assertEqual(digest(fixture.score(fixture.partial())[1]),
                         "1339faadc275afd91956e66b9bd03a0b3638b7719c18b1038b9ef5778bcfe024")
        self.assertEqual(notebook_objective.contract(credit.ASSESSMENT_VERSION), credit.contract(credit.ASSESSMENT_VERSION))
        self.assertEqual(notebook_objective.ASSESSMENT_VERSION, credit.ASSESSMENT_VERSION)
        self.assertEqual(credit.ASSESSMENT_VERSION, "verified-whole-reference-assessment-credit-v3")
        self.assertEqual(local_profile.ASSESSMENT_PROTOCOL_ID, "local-nf4-whole-reference-assessment-credit-24k-v3")
        self.assertIn("authored JSON null has no assessment PG", credit.contract(credit.ASSESSMENT_VERSION)["assessment_tokens"])
        self.assertEqual(credit.ASSESSMENT_NORMALIZATION_CAP, 8192)
        self.assertEqual(credit.contract(credit.ASSESSMENT_VERSION)["assessment_normalization_cap"], 8192)
        for version in ("v1", "v2"):
            old = "verified-whole-reference-assessment-credit-" + version
            with self.subTest(version=version), self.assertRaises(ValueError):
                credit.contract(old)
            with self.subTest(version=version), self.assertRaises(ValueError):
                notebook_objective.contract(old)

    def test_v3_profile_rejects_resealed_old_versions_or_changed_normalization(self):
        from fine_tuning.notebook_rl import local_profile, updates

        current = local_profile.build_protocol(verified_note_synthesis_credit=True, pair_draft_synthesis=True,
            whole_reference_notebook=True, verified_note_assessment_credit=True, context_limit=24576)
        previous = local_profile.build_protocol(verified_note_synthesis_credit=True, pair_draft_synthesis=True,
            whole_reference_notebook=True, context_limit=24576)
        self.assertEqual({key for key in current if current[key] != previous.get(key)},
                         {"protocol_id", "notebook_objective", "notebook_update_requirement", "sha256"})
        self.assertEqual(local_profile.local_prompts(current), local_profile.local_prompts(previous))
        self.assertEqual(current["notebook_update_requirement"], "verified-native-assessment-aggregate-gradient-v2")
        self.assertEqual(updates.ASSESSMENT_REQUIREMENT, current["notebook_update_requirement"])
        self.assertEqual((current["rollout"]["diag"]["n"], current["rollout"]["diag"]["max_new_tokens"]), (4, 8192))
        self.assertEqual(local_profile.restore_protocol(json.loads(json.dumps(current))), current)
        changes: list[tuple[str, str | int]] = [
            ("protocol_id", "local-nf4-whole-reference-assessment-credit-24k-" + version) for version in ("v1", "v2")]
        changes += [("objective_version", "verified-whole-reference-assessment-credit-" + version)
                    for version in ("v1", "v2")]
        changes += [("assessment_normalization_cap", 4096), ("assessment_normalization_cap", True)]
        for key, value in changes:
            changed = copy.deepcopy(current)
            (changed if key == "protocol_id" else changed["notebook_objective"])[key] = value
            changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                local_profile.restore_protocol(changed)

    def test_objective_requires_explicit_monolithic_generation(self):
        from fine_tuning.notebook_rl import pair_action_notebook, whole_reference_pages

        for policy in (None, pair_action_notebook.VERSION, whole_reference_pages.VERSION):
            self.tree.notebook["rewarded_notebook"]["notebook_generation_policy"] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "monolithic"):
                self.score()
        with self.assertRaises(ValueError):
            credit.contract("unknown")
        with self.assertRaisesRegex(ValueError, "synthesis_credit.score_note.*actual raw sample ID"):
            notebook_objective.score_note("{}", "stop", self.tree, self.audit, self.verifier, actor=self.actor,
                objective_version=credit.ASSESSMENT_VERSION)

    def test_every_public_native_ref_including_n_and_roots_has_fixed_weight(self):
        _, row = self.score()
        refs = [ref for ref in self.inventory["assessment_refs"]
                if self.inventory["references"][ref]["kind"] in ("OBSERVATION", "TRACE")]
        self.assertEqual(row["assessment_refs"], refs)
        self.assertEqual(sum(row["assessment_rewards"].values()), len(refs))
        self.assertEqual(row["native_assessment_count"], len(refs))
        self.assertTrue(any(state["flags"] == "N" for state in row["assessment_expected_states"].values()))
        _, legacy = self.score()
        old_atomic = credit.FACT_WEIGHT * sum(legacy["fact_rewards"].values()) / legacy["required_fact_count"]
        old_atomic += credit.RELATIONSHIP_WEIGHT * sum(legacy["relationship_rewards"].values()) / len(legacy["original_issue_ids"])
        self.assertEqual(row["atomic_reward"], old_atomic + .0625)

    def test_false_i_false_u_and_missed_i_are_zero_without_erasing_other_rows(self):
        base = self.without_decisions()
        normal = next(row for row in base["assessments"] if row[1] == "N")
        issue = next(row for row in base["assessments"] if row[1] == "I")
        for ref, flag in ((normal[0], "I"), (normal[0], "U"), (issue[0], "N"), (issue[0], "U")):
            bad = copy.deepcopy(base)
            next(row for row in bad["assessments"] if row[0] == ref)[1] = flag
            _, score = self.score(bad)
            self.assertEqual(score["assessment_rewards"][ref], 0)
            self.assertEqual(sum(score["assessment_rewards"].values()), score["native_assessment_count"] - 1)
            self.assertIn(ref, score["assessment_spans"])

    def test_iu_requires_both_issue_and_supported_unknown_claim(self):
        mixed = fixtures.response(self.item.slots).replace("start = q0", "start = q1\nstart = ???")
        self.prepare([mixed] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        base = self.without_decisions()
        combined = next(row for row in base["assessments"] if row[1] == "IU")
        self.assertEqual(self.score(base)[1]["assessment_rewards"][combined[0]], 1)
        for flag in ("N", "I", "U"):
            bad = copy.deepcopy(base)
            next(row for row in bad["assessments"] if row[0] == combined[0])[1] = flag
            self.assertEqual(self.score(bad)[1]["assessment_rewards"][combined[0]], 0)

    def test_ordered_dependencies_and_direct_prerequisite_states_are_required(self):
        short = fixtures.graph_response(self.item.slots, {"kind": "dfa", "states": ["q0"], "decision": "reject"}, "reject")
        self.prepare([short] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        base = self.without_decisions()
        unknown = next(row for row in base["assessments"] if row[1] == "U" and row[2])
        self.assertEqual(self.score(base)[1]["assessment_rewards"][unknown[0]], 1)
        for dependencies in ([], [unknown[0]], [base["assessments"][-1][0]], unknown[2] * 2):
            bad = copy.deepcopy(base)
            next(row for row in bad["assessments"] if row[0] == unknown[0])[2] = dependencies
            self.assertEqual(self.score(bad)[1]["assessment_rewards"][unknown[0]], 0)
        for flag in ("N", "U", False):
            bad = copy.deepcopy(base)
            prerequisite = next(row for row in bad["assessments"] if row[0] == unknown[2][0])
            prerequisite[1] = flag
            self.assertEqual(self.score(bad)[1]["assessment_rewards"][unknown[0]], 0)

    def test_scope_unknowns_are_not_compliance_and_require_u(self):
        self.prepare(["Final answer: reject"] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        base = self.without_decisions()
        unknown = next(row for row in base["assessments"] if row[1] == "U")
        _, good = self.score(base)
        self.assertEqual(good["assessment_expected_states"][unknown[0]], {"flags": "U", "blocked_by": []})
        unknown[1] = "N"
        self.assertEqual(self.score(base)[1]["assessment_rewards"][unknown[0]], 0)

    def test_existing_circuit_unknown_dependencies_keep_u_grammar_and_order(self):
        from fine_tuning.tests import test_notebook_rl_circuit_notebook as circuit_fixtures

        fixture = circuit_fixtures.CircuitNotebookTests()
        fixture.setUp()
        trace = copy.deepcopy(fixture.right_trace)
        trace["nodes"][1][1] = None
        trace["ohm"][0][3:] = [None, None, None]
        text = circuit_fixtures.response(fixture.problem, trace).rsplit("Final answer:", 1)[0] + "Final answer: -999 A"
        fixture.prepare([text, fixture.right])
        inventory = reference.public_inventory(fixture.samples)
        expected = credit._assessment_states(inventory, fixture.audit)
        complete, rows, _, _ = credit._assessment_units(json.dumps(fixture.note), "stop", inventory)
        self.assertTrue(complete)
        ref = next(ref for ref, state in expected.items() if len(state["blocked_by"]) == 2)
        self.assertTrue(credit._assessment_state_matches(ref, rows, expected, inventory))
        self.assertTrue(all(expected[dep]["flags"] == "U" for dep in rows[ref][2]))
        bad = copy.deepcopy(rows)
        bad[ref][2].reverse()
        self.assertFalse(credit._assessment_state_matches(ref, bad, expected, inventory))
        bad = copy.deepcopy(rows)
        bad[rows[ref][2][0]][1] = "N"
        self.assertFalse(credit._assessment_state_matches(ref, bad, expected, inventory))

    def test_wrong_decision_null_string_and_bad_ancestry_zero_whole_row(self):
        source = next(row for row in self.note["assessments"] if row[3] and row[3][0] == "PERSISTED")
        for decision in ("null", [], ["CORRECTED", source[3][1]], ["PERSISTED", source[0]],
                         ["PERSISTED", "missing"], ["FINALIZED", source[3][1]]):
            bad = copy.deepcopy(self.note)
            next(row for row in bad["assessments"] if row[0] == source[0])[3] = decision
            self.assertEqual(self.score(bad)[1]["assessment_rewards"][source[0]], 0)
        null = copy.deepcopy(self.note)
        next(row for row in null["assessments"] if row[0] == source[0])[3] = None
        self.assertEqual(self.score(null)[1]["assessment_rewards"][source[0]], 1)

    def test_nonnull_source_and_destination_states_couple_relationship_rewards(self):
        source = next(row for row in self.note["assessments"] if row[3] and row[3][0] == "PERSISTED")
        endpoint = source[3][1]
        for target in (source[0], endpoint):
            bad = copy.deepcopy(self.note)
            next(row for row in bad["assessments"] if row[0] == target)[1] = "N"
            _, score = self.score(bad)
            self.assertEqual(score["assessment_rewards"][source[0]], 0)
            origin = credit.issue_id(self.inventory["references"][source[0]]["attempt"],
                                     self.inventory["references"][source[0]])
            self.assertEqual(score["relationship_rewards"][origin], 0)
        bad = copy.deepcopy(self.note)
        next(row for row in bad["assessments"] if row[0] == endpoint)[3] = "bad destination decision"
        _, score = self.score(bad)
        self.assertEqual(score["assessment_rewards"][source[0]], 1)
        self.assertEqual(score["assessment_rewards"][endpoint], 0)

    def test_bad_headers_and_output_rows_do_not_erase_valid_native_rows(self):
        variants = []
        for key, value in (("corrected_facts", "bad"), ("corrected_facts", {"start": 3}),
                           ("attempts", None), ("success_attempt", True), ("corrected_answer", None)):
            bad = copy.deepcopy(self.note)
            bad[key] = value
            variants.append(bad)
        bad = copy.deepcopy(self.note)
        output = next(row for row in bad["assessments"] if self.inventory["references"][row[0]]["kind"] == "OUTPUT")
        output[1:] = ["I", "bad dependencies", "bad decision"]
        variants.append(bad)
        for bad in variants:
            _, row = self.score(bad)
            self.assertEqual(row["strict_reward"], 0.)
            self.assertTrue(row["assessment_rows_complete"])
            self.assertEqual(sum(row["assessment_rewards"].values()), row["native_assessment_count"])

    def test_bad_identifiable_row_keeps_existing_literals_with_zero_reward(self):
        note = self.without_decisions()
        selected = note["assessments"][0]
        selected[1:] = [True, [], None, "extra"]
        sample, row = self.score(note)
        self.assertEqual(row["assessment_rewards"][selected[0]], 0)
        self.assertEqual([sample.text[a:b] for a, b in row["assessment_spans"][selected[0]]], ["true", "[]"])

    def test_missing_duplicate_reordered_refs_or_duplicate_keys_zero_entire_component(self):
        variants = []
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.reverse(),
                       lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0]))):
            bad = copy.deepcopy(self.note)
            mutate(bad["assessments"])
            variants.append(json.dumps(bad))
        valid = json.dumps(self.note)
        variants += [valid.replace('"success_attempt":', '"success_attempt": 1, "success_attempt":'), valid + " trailing", "{}"]
        for raw in variants:
            _, row = self.score(text=raw)
            self.assertFalse(row["assessment_rows_complete"])
            self.assertEqual(sum(row["assessment_rewards"].values()), 0)
            self.assertEqual(row["assessment_spans"], {})
        self.assertEqual(sum(self.score(finish="length")[1]["assessment_rewards"].values()), 0)

    def test_n4_mean_flags_keep_credit_and_null_decisions_get_none_in_both_layouts(self):
        good = self.without_decisions()
        bad = copy.deepcopy(good)
        bad["assessments"][0][1] = "I"
        for separators in ((",", ":"), None):
            with self.subTest(separators=separators):
                pairs = [self.score(text=json.dumps(note, separators=separators), index=index)
                         for index, note in enumerate((good, bad, {}, good))]
                samples = [sample for sample, _ in pairs]
                scores = {sample.sample_id: row for sample, row in pairs}
                mapped = credit.group_credit(samples, scores, Characters())
                ref = good["assessments"][0][0]
                positive = next(row for row in mapped[samples[0].sample_id]["assessments"] if row["ref"] == ref)
                negative = next(row for row in mapped[samples[1].sample_id]["assessments"] if row["ref"] == ref)
                self.assertEqual((positive["reward"], positive["group_mean"], positive["advantage"]), (1, .5, .5))
                self.assertEqual(negative["advantage"], -.5)
                self.assertEqual(positive["expected_flags"], "N")
                self.assertAlmostEqual(positive["coefficient"] * len(positive["token_indices"]),
                                       8192 * .0625 * .5 / scores[samples[0].sample_id]["native_assessment_count"])
                self.assertEqual([samples[0].text[a:b] for a, b in positive["literal_spans"]], ['"N"', "[]"])
                self.assertEqual(credit.group_credit(samples, json.loads(json.dumps(scores, sort_keys=True)), Characters()), mapped)
                for sample in samples:
                    row = mapped[sample.sample_id]
                    allowed = {index for entry in row["assessments"] for index in entry["token_indices"]}
                    self.assertTrue(all(not value for index, value in enumerate(row["assessment_token_advantages"])
                                        if index not in allowed))
                    self.assertEqual(row["partial_token_advantages"], row["assessment_token_advantages"])
                    note, offsets = reference._decode(sample.text)
                    for position, assessment in enumerate(note.get("assessments", [])):
                        self.assertIsNone(assessment[3])
                        start, end = offsets[("assessments", position, 3)]
                        self.assertEqual(sample.text[start:end], "null")
                        self.assertTrue(all(value == 0. for value in row["assessment_token_advantages"][start:end]))
                for index, sign in ((0, 1), (1, -1)):
                    row = mapped[samples[index].sample_id]
                    start, end = scores[samples[index].sample_id]["assessment_spans"][ref][0]
                    self.assertTrue(all(sign * value > 0 for value in row["assessment_token_advantages"][start:end]))
                self.assertTrue(credit.has_signal(samples, scores))

    def test_only_assessment_coefficients_and_vectors_scale_by_8192(self):
        bad = copy.deepcopy(self.note)
        bad["corrected_facts"] = {}
        for row in bad["assessments"]:
            row[1], row[3] = ("I" if row[1] == "N" else "N"), None
        samples, scores, mapped = self.group([self.note, bad, {}, self.note])
        legacy_scores = {sample.sample_id: credit.score_note(sample.text, sample.finish_reason, self.tree,
            self.audit, self.verifier, actor=self.actor, sample_id=sample.sample_id) for sample in samples}
        legacy = credit.group_credit(samples, legacy_scores, Characters())
        for sample in samples:
            row, score = mapped[sample.sample_id], scores[sample.sample_id]
            for key in ("strict_reward", "fact_rewards", "relationship_rewards", "fact_spans", "relationship_spans"):
                self.assertEqual(score[key], legacy_scores[sample.sample_id][key])
            self.assertAlmostEqual(score["atomic_reward"], legacy_scores[sample.sample_id]["atomic_reward"] +
                .0625 * sum(score["assessment_rewards"].values()) / score["native_assessment_count"])
            for key in ("facts", "relationships", "fact_token_advantages", "relationship_token_advantages",
                        "strict_advantage", "token_ids_sha256", "text_sha256", "full_group_size"):
                self.assertEqual(row[key], legacy[sample.sample_id][key])
            old_vector = [0.] * len(sample.completion_token_ids)
            for entry in row["assessments"]:
                old = (.0625 * entry["advantage"] / score["native_assessment_count"] / len(entry["token_indices"])
                       if entry["token_indices"] else 0.)
                self.assertEqual(entry["coefficient"], 8192 * old)
                for index in entry["token_indices"]:
                    old_vector[index] += old
            self.assertEqual(row["assessment_token_advantages"], [8192 * value for value in old_vector])
            self.assertEqual(row["partial_token_advantages"], [other + 8192 * old for other, old in
                zip(legacy[sample.sample_id]["partial_token_advantages"], old_vector, strict=True)])
        self.assertTrue(any(any(row["fact_token_advantages"]) for row in mapped.values()))
        self.assertTrue(any(any(row["relationship_token_advantages"]) for row in mapped.values()))
        self.assertTrue(any(row["strict_advantage"] for row in mapped.values()))

    def test_actual_group_loss_cancels_only_assessment_cap_and_scales_vjp_without_grad_mutation(self):
        from fine_tuning.notebook_rl.losses import group_loss

        samples, scores, mapped = self.group([self.without_decisions(), {}, {}, {}])
        sample = samples[0]
        coefficients = torch.tensor([mapped[sample.sample_id]["assessment_token_advantages"]], dtype=torch.float64)
        parameter = torch.nn.Parameter(torch.zeros_like(coefficients))
        parameter.grad = torch.full_like(parameter, 7.)
        old = torch.zeros_like(parameter)
        mask = torch.ones_like(parameter, dtype=torch.bool)
        keep = torch.tensor([True])
        kwargs = {"eps_low": .2, "eps_high": .28, "beta": 0., "norm_const": 4 * 8192}
        weight = .2
        scaled = weight * group_loss(parameter, old, old, torch.zeros(1, dtype=parameter.dtype), mask, keep,
                                    partial_adv=coefficients, **kwargs)
        prior = weight * group_loss(parameter, old, old, torch.zeros(1, dtype=parameter.dtype), mask, keep,
                                   partial_adv=coefficients / 8192, **kwargs)
        vjp = torch.autograd.grad(scaled, parameter, retain_graph=True)[0]
        prior_vjp = torch.autograd.grad(prior, parameter)[0]
        torch.testing.assert_close(scaled.detach(), prior.detach() * 8192, atol=0, rtol=0)
        torch.testing.assert_close(vjp, prior_vjp * 8192, atol=0, rtol=0)
        entry = mapped[sample.sample_id]["assessments"][0]
        expected = -weight * .0625 * entry["advantage"] / scores[sample.sample_id]["native_assessment_count"] / 4
        self.assertAlmostEqual(float(vjp[0, entry["token_indices"]].sum()), expected)
        masked = entry["token_indices"][0]
        mask[0, masked] = False
        for selected in (keep, torch.tensor([False])):
            loss = weight * group_loss(parameter, old, old, torch.zeros(1, dtype=parameter.dtype), mask, selected,
                                       partial_adv=coefficients, **kwargs)
            gradient = torch.autograd.grad(loss, parameter)[0]
            self.assertEqual(float(gradient[0, masked]), 0.)
            if not bool(selected[0]):
                self.assertEqual(int(torch.count_nonzero(gradient)), 0)
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.), atol=0, rtol=0)

    def test_finite_assessment_coefficients_may_exceed_one_without_clipping(self):
        self.prepare(["Final answer: reject"] * 4 + [fixtures.response(self.item.slots, wrong=False)] * 2)
        samples, _, mapped = self.group([self.without_decisions(), {}, {}, {}])
        coefficients = [entry["coefficient"] for entry in mapped[samples[0].sample_id]["assessments"]]
        self.assertTrue(all(math.isfinite(value) for value in coefficients))
        self.assertGreater(max(coefficients), 1.)
        with self.assertRaisesRegex(ValueError, "original complete unique n4"):
            credit.group_credit(samples[:-1], {}, Characters())

    def test_verified_nonnull_decision_keeps_credit_and_null_string_stays_malformed(self):
        source = next(row for row in self.note["assessments"] if row[3] and row[3][0] == "PERSISTED")
        bad = copy.deepcopy(self.note)
        next(row for row in bad["assessments"] if row[0] == source[0])[3] = "null"
        samples, scores, mapped = self.group([self.note, bad, {}, self.note])
        for index, sign in ((0, 1), (1, -1)):
            sample = samples[index]
            row = mapped[sample.sample_id]
            entry = next(entry for entry in row["assessments"] if entry["ref"] == source[0])
            self.assertEqual(entry["reward"], int(sign == 1))
            self.assertEqual(entry["group_mean"], .5)
            start, end = scores[sample.sample_id]["assessment_spans"][source[0]][2]
            self.assertEqual(json.loads(sample.text[start:end]), source[3] if sign == 1 else "null")
            self.assertTrue(all(token in entry["token_indices"] for token in range(start, end)))
            self.assertTrue(all(sign * value > 0 for value in row["assessment_token_advantages"][start:end]))

    def test_equal_n4_has_no_signal_and_unstable_prefix_has_no_mapping(self):
        good = self.without_decisions()
        bad = copy.deepcopy(good)
        for row in bad["assessments"]:
            row[1] = "I" if row[1] == "N" else "N"
        for note in (bad, good):
            samples, scores, result = self.group([note] * 4)
            self.assertFalse(credit.has_signal(samples, scores))
            self.assertTrue(all(not any(row["partial_token_advantages"]) for row in result.values()))
            if note is bad:
                self.assertTrue(all(not any(row["assessment_rewards"].values()) for row in scores.values()))
        unstable = SimpleNamespace(decode=lambda ids, **kwargs: "?" if len(ids) == 1 else "".join(map(chr, ids)))
        output = credit.group_credit(samples, scores, unstable)
        self.assertTrue(all(row["mapping_status"] == "UNSTABLE_TOKEN_PREFIX" for row in output.values()))
        with self.assertRaisesRegex(ValueError, "tokens and text differ"):
            credit.group_credit(samples, scores, SimpleNamespace(decode=lambda ids, **kwargs: "wrong"))

    def test_bound_version_count_spans_audit_and_original_token_tampering_rejected(self):
        sample, row = self.score(self.without_decisions())
        for mutate in (lambda score: score.update(objective_version=credit.VERSION),
                       lambda score: score.update(native_assessment_count=True),
                       lambda score: score["assessment_spans"][score["assessment_refs"][0]][0].__setitem__(0, 0),
                       lambda score: score["assessment_rewards"].update(unbound=1)):
            bad = copy.deepcopy(row)
            mutate(bad)
            bad = credit._seal({key: value for key, value in bad.items() if key != "score_sha256"})
            with self.assertRaises(ValueError):
                credit.validate_score(bad, sample)
        changed = replace(sample, completion_token_ids=sample.completion_token_ids + [2])
        with self.assertRaises(ValueError):
            credit.validate_score(row, changed)
        audit = copy.deepcopy(self.audit)
        audit["attempts"][0]["issues"] = []
        with self.assertRaisesRegex(ValueError, "source audit"):
            credit.score_note(sample.text, "stop", self.tree, audit, self.verifier, actor=self.actor,
                sample_id=sample.sample_id, objective_version=credit.ASSESSMENT_VERSION)


if __name__ == "__main__":
    unittest.main()
