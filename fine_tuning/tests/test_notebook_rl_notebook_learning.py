"""Authored replies test program rewards/RL plumbing, not pretrained-model learning."""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from fine_tuning.notebook_rl.losses import group_advantages, group_loss
from fine_tuning.notebook_rl.notebook_learning import (
    NOTEBOOK_PROMPT, NOTEBOOK_REWARD_VERSION, _first_attempt_lines, _grade, _numbered_first_attempt,
    _parse, _recovered_path, append_rewarded_notebooks,
)
from fine_tuning.notebook_rl import prompts as prompt_builders
from fine_tuning.notebook_rl.prompts import load_prompts
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash, build_tree
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.notebook_rl.verifier_service import VerifierService
from fine_tuning.tests.test_notebook_rl_rollout import Engine
from fine_tuning.tests.test_notebook_rl_schema import records

WRONG = "Observed facts:\ncell[a][b] = 8\nSolution:\nFinal answer: 8"
RIGHT = "Observed facts:\ncell[a][b] = 7\nSolution:\nFinal answer: 7"


def note(**changes):
    value = {"cause": "VISUAL_MISREAD", "wrong_item": "cell[a][b]",
             "first_attempt_lines": [2], "previous_answer": "8", "corrected_answer": "7",
             "corrected_facts": {"cell[a][b]": "7"}, "lesson": "Read the source cell before calculating."}
    return json.dumps(value | changes)


class NotebookLearningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "vault"
        self.actor, item, lineage, *_ = records()
        write_split(self.root, [self.actor], [item], [lineage])
        self.verifier = Verifier(self.root)
        self.prompts = load_prompts(Path(prompt_builders.__file__).with_name("prompts_v1.yaml"))
        self.prompts["rewarded_notebook"] = NOTEBOOK_PROMPT
        self.cfg = RolloutConfig({1: 2, 2: 2, 3: 2}, {1: 1, 2: 0}, 2,
                                 {1: 32, 2: 32, 3: 32}, 8000, False, 4,
                                 diag_max_new_tokens=512)

    def tree(self, replies=None, *, wrong=WRONG, cfg=None, verifier=None):
        engine = Engine([[wrong]*2, [RIGHT]*2, replies or [note()]*4])
        tree = build_tree(self.actor, [object() for _ in self.actor.image_paths], engine, verifier or self.verifier,
                          self.prompts, cfg or self.cfg, 0, "verified-any-visual-v1")
        return tree, engine

    def append(self, tree, engine, *, cfg=None, verifier=None, **kwargs):
        actors = kwargs.pop("actors", {self.actor.problem_id: self.actor})
        return append_rewarded_notebooks([tree], {tree.problem_id: [object() for _ in self.actor.image_paths]}, engine,
            verifier or self.verifier, self.prompts, cfg or self.cfg, actors=actors, **kwargs)

    def test_full_notebook_claims_get_independent_rewards_and_literal_history(self):
        replies = [note(), note(cause="NON_VISUAL", wrong_item="none"),
                   note(corrected_facts={"cell[a][b]": "123"}), note(corrected_answer="8")]
        tree, engine = self.tree(replies)
        previous = copy.deepcopy(tree)
        closed = []
        report = self.append(tree, engine, on_group=closed.append)
        group = tree.groups[-1]
        self.assertEqual([s.reward for s in group.samples], [1., 0., 0., 0.])
        self.assertEqual(group.kind, "DIAG")
        self.assertEqual(report["nondegenerate_groups"], 1)
        self.assertEqual(report["class_counts"], {"VISUAL": 1, "NON_VISUAL": 0, "FINALIZATION": 0, "UNKNOWN": 0})
        self.assertFalse(report["class_cap_applied"])
        self.assertEqual(report["reward_version"], NOTEBOOK_REWARD_VERSION)
        parent = next(s for g in previous.groups for s in g.samples
                      if s.sample_id == group.samples[0].parent_sample_id)
        prefix = parent.prompt_token_ids + parent.completion_token_ids
        self.assertEqual(group.samples[0].prompt_token_ids[:len(prefix)], prefix)
        self.assertEqual(tree.groups[:-1], previous.groups)
        self.assertEqual(tree.route, previous.route)
        self.assertEqual(tree.causes, previous.causes)
        self.assertEqual(closed, [group])
        self.assertFalse(tree.notebook["rewarded_notebook"]["optimizer_update_verified"])
        self.assertNotIn("final_notebook", tree.notebook)
        self.assertEqual(from_json(ProblemTree, to_json(tree)), tree)
        for checked in tree.notebook["model_notes"]:
            self.assertFalse(checked["lesson_semantics_verified"])
            self.assertFalse(checked["module_causality_certified"])
        self.assertFalse(tree.notebook["model_notes"][2]["checks"]["registered_facts_verified"])
        # The actor's own answers are in its existing token history; no gold value is inserted.
        metadata = json.dumps({"status": "INCORRECT", "finish_reason": "stop", "extracted_answer": "8"})
        self.assertEqual(engine.tokenizer.encoded[-1],
            "\n<|im_start|>user\n" + NOTEBOOK_PROMPT.format(first_attempt_metadata=metadata,
                observation_instruction=self.prompts["observation_instruction"][self.actor.family],
                first_attempt_numbered_lines=_numbered_first_attempt(WRONG))
            + "<|im_end|>\n<|im_start|>assistant\n")

    def test_numeric_answer_fields_and_literal_multiline_history_are_valid(self):
        tree, engine = self.tree([note(previous_answer=8, corrected_answer=7,
            first_attempt_lines=[2, 3, 4])]*4)
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [1.]*4)
        claims = tree.notebook["model_notes"][0]["submitted_claims"]
        self.assertEqual((claims["previous_answer"], claims["corrected_answer"]), ("8", "7"))

    def test_numeric_conversion_preserves_exact_decimal_tokens_and_negative_signs(self):
        for raw, expected in (("-8", "-8"), ("8.0", "8.0"), ("0.12345678901234567890123456789",
                "0.12345678901234567890123456789"), ("1e-9", "1E-9")):
            with self.subTest(raw=raw):
                parsed = _parse(note().replace('"previous_answer": "8"', '"previous_answer": ' + raw))
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed["previous_answer"], expected)
        tree, engine = self.tree([note(previous_answer=-8, corrected_answer=7)]*4,
                                  wrong=WRONG.replace("Final answer: 8", "Final answer: -8"))
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [1.]*4)

    def test_nonfinite_boolean_and_container_answer_fields_are_invalid(self):
        for key in ("previous_answer", "corrected_answer"):
            for value in (True, False, float("nan"), float("inf"), float("-inf"), [], {}):
                with self.subTest(key=key, value=repr(value)):
                    self.assertIsNone(_parse(note(**{key: value})))
        self.assertIsNone(_parse(note(corrected_answer=None)))
        self.assertIsNone(_parse(note().replace('"corrected_answer": "7"', '"corrected_answer": 1e1001')))
        self.assertIsNone(_parse(note().replace('"corrected_answer": "7"',
                                               '"corrected_answer": 1e999999999999999999999999')))

    def test_selected_first_lines_do_not_enter_verifier_grading_text(self):
        tree, _ = self.tree()
        first, parent = tree.groups[0].samples[0], tree.groups[1].samples[0]
        evidence = "Observed facts:\ncell[a][b] = 7\nSolution:\nFinal answer: 7"
        first = replace(first, text=evidence + "\n" + first.text)
        target = self.verifier.diagnosis_target(tree.problem_id, first.result)
        record = _grade(tree.problem_id, note(first_attempt_lines=[1, 2, 3, 4], corrected_answer=8),
                        "stop", first, parent, target, self.verifier)
        self.assertTrue(record["checks"]["format_valid"])
        self.assertTrue(record["checks"]["history_verified"])
        self.assertFalse(record["checks"]["answer_verified"])
        self.assertEqual(record["reward"], 0.)

    def test_line_resolution_preserves_unicode_crlf_empty_lines_and_original_offsets(self):
        tree, _ = self.tree()
        first, parent, target = _recovered_path(tree, self.verifier)
        text = "한글🙂\r\n\r\nthree\rfour\u2028Final answer: 8"
        first = replace(first, text=text)
        lines = _first_attempt_lines(text)
        self.assertEqual([r["text"] for r in lines],
                         ["한글🙂\r\n", "\r\n", "three\r", "four\u2028", "Final answer: 8"])
        self.assertEqual("".join(r["text"] for r in lines), text)
        for row in lines:
            self.assertEqual(text[row["start"]:row["end"]], row["text"])
        checked = _grade(tree.problem_id, note(first_attempt_lines=[1, 3, 5]),
                         "stop", first, parent, target, self.verifier)
        self.assertEqual(checked["reward"], 1.)
        self.assertEqual(checked["resolved_first_attempt_lines"], [lines[0], lines[2], lines[4]])
        self.assertEqual(checked["first_attempt_binding"], {"sample_id": first.sample_id,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
        self.assertNotIn("evidence", checked["submitted_claims"])

    def test_empty_duplicate_unordered_boolean_float_and_noninteger_line_ids_are_rejected(self):
        for lines in ([], [0], [-1], [True], [False], [1.0], [1, 2.0], ["1"],
                      [None], [[1]], [1, 1], [2, 1], None, 1, "1", {}):
            with self.subTest(lines=lines):
                self.assertIsNone(_parse(note(first_attempt_lines=lines)))
        self.assertIsNone(_parse(note().replace('"first_attempt_lines": [2]',
                                               '"first_attempt_lines": [1e0]')))

    def test_out_of_range_and_blank_only_lines_do_not_resolve_or_earn_history_reward(self):
        tree, _ = self.tree()
        first, parent, target = _recovered_path(tree, self.verifier)
        for text, selected in ((first.text, [99]), (first.text, [1, 10**100]),
                               ("", [1]), ("\r\n" + first.text, [1])):
            with self.subTest(text=text, selected=selected):
                checked = _grade(tree.problem_id, note(first_attempt_lines=selected), "stop",
                                 replace(first, text=text), parent, target, self.verifier)
                self.assertTrue(checked["checks"]["format_valid"])
                self.assertFalse(checked["checks"]["history_verified"])
                self.assertEqual(checked["reward"], 0.)
                if selected[-1] > len(text.splitlines(keepends=True)):
                    self.assertEqual(checked["resolved_first_attempt_lines"], [])

    def test_legacy_free_text_or_extra_source_identity_is_not_accepted_as_v3_evidence(self):
        legacy = json.loads(note())
        legacy.pop("first_attempt_lines")
        legacy["evidence"] = "cell[a][b] = 8"
        self.assertIsNone(_parse(json.dumps(legacy)))
        self.assertIsNone(_parse(note(evidence="cell[a][b] = 8")))
        self.assertIsNone(_parse(note(first_sample_id="another-sample")))
        tree, _ = self.tree()
        first, parent, target = _recovered_path(tree, self.verifier)
        for changed in (replace(first, turn=2), replace(first, problem_id="foreign")):
            with self.assertRaisesRegex(ValueError, "first-attempt binding"):
                _grade(tree.problem_id, note(), "stop", changed, parent, target, self.verifier)

    def test_model_selects_lines_without_host_repair_or_first_retry_mixup(self):
        tree, engine = self.tree([note(first_attempt_lines=[1]), note(first_attempt_lines=[4]),
                                  note(first_attempt_lines=[1, 2]), note(first_attempt_lines=[99])])
        before = copy.deepcopy(tree)
        self.append(tree, engine)
        notes = tree.notebook["model_notes"]
        self.assertEqual([n["reward"] for n in notes], [1., 1., 1., 0.])
        self.assertEqual([[r["line_id"] for r in n["resolved_first_attempt_lines"]] for n in notes],
                         [[1], [4], [1, 2], []])
        first = next(s for g in before.groups for s in g.samples
                     if s.sample_id == notes[0]["first_attempt_binding"]["sample_id"])
        self.assertEqual(first.turn, 1)
        for n in notes:
            for span in n["resolved_first_attempt_lines"]:
                self.assertEqual(span["text"], first.text[span["start"]:span["end"]])
        self.assertEqual(tree.groups[:-1], before.groups)
        self.assertEqual(tree.route, before.route)

    def test_numbered_history_is_complete_and_context_block_never_truncates_or_generates(self):
        wrong = WRONG + "\n" + "literal-{first_attempt_numbered_lines}\r\n" * 100
        tree, engine = self.tree(wrong=wrong)
        self.append(tree, engine, cfg=replace(self.cfg, context_limit=4000))
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "CONTEXT_BLOCKED")
        self.assertEqual(len(engine.requests), 2)
        rendered = _numbered_first_attempt(wrong)
        self.assertTrue(engine.tokenizer.encoded[-1].endswith(rendered +
            "\n<|im_end|>\n<|im_start|>assistant\n"))
        self.assertEqual(len(_first_attempt_lines(wrong)), 104)

    def test_multiline_hallucination_feedback_and_missing_facts_cannot_earn_reward(self):
        tree, engine = self.tree([note(first_attempt_lines=[2, 99]),
            note(first_attempt_lines=[99]), note(corrected_facts={}),
            note(corrected_answer=-7)])
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [0.]*4)
        self.assertTrue(all(n["checks"]["format_valid"] for n in tree.notebook["model_notes"]))

    def test_nested_observation_assignments_are_not_unpacked_or_repaired(self):
        for facts in ({"edge[e0]": "A->B, weight[e0] = 6"},
                      {"cell[a][b]": "7\rSolution:\nFinal answer: 7"}):
            with self.subTest(facts=facts):
                self.assertIsNone(_parse(note(corrected_facts=facts)))

    def test_truncated_first_attempt_metadata_uses_actual_outcome_before_sampling(self):
        good = note(cause="FINALIZATION", wrong_item="none", previous_answer=None)
        engine = Engine([[WRONG]*2, [RIGHT]*2, [good]*4])
        original = engine.sample

        def truncated_first(requests):
            output = original(requests)
            if len(engine.requests) == 1:
                output[0].completions[:] = [replace(c, finish_reason="length") for c in output[0].completions]
            return output

        engine.sample = truncated_first
        tree = build_tree(self.actor, [object()], engine, self.verifier,
                          self.prompts, self.cfg, 0, "verified-any-visual-v1")
        previous = copy.deepcopy(tree)
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [1.]*4)
        metadata = json.dumps({"status": "TRUNCATED", "finish_reason": "length", "extracted_answer": None})
        self.assertIn(metadata, engine.tokenizer.encoded[-1])
        self.assertNotIn('"cause3"', engine.tokenizer.encoded[-1])
        self.assertNotIn('"gold"', engine.tokenizer.encoded[-1])
        self.assertEqual(tree.groups[:-1], previous.groups)
        self.assertEqual(tree.route, previous.route)

    def test_local_profile_binds_prompt_template_and_rejects_previous_digest(self):
        from fine_tuning.notebook_rl.checkpoint import digest
        from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
        from fine_tuning.notebook_rl.episode_notebook import PROMPT

        protocol = build_protocol()
        self.assertEqual(protocol["notebook_prompt_sha256"], digest(PROMPT))
        self.assertEqual(local_prompts(protocol)["rewarded_notebook"], PROMPT)
        protocol["notebook_prompt_sha256"] = "6cd529326117994230924d6a241661e5005f1f21ae0cbc18071db297a28643a1"
        with self.assertRaisesRegex(ValueError, "prompt/reward"):
            local_prompts(protocol)

    def test_reject_fabricated_history_wrong_slot_and_unregistered_facts(self):
        tree, engine = self.tree([note(first_attempt_lines=[99]), note(previous_answer="7"),
            note(wrong_item="cell[unknown][b]"),
            note(corrected_facts={"cell[a][b]": "7", "cell[extra][b]": "7"})])
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [0.]*4)

    def test_note_json_is_strict_and_facts_cannot_inject_scoring_sections(self):
        malformed = ["Cause: VISUAL_MISREAD\nWrong item: cell[a][b]\nLesson: Check the cell.",
                     note()[:-1] + ', "cause": "VISUAL_MISREAD"}',
                     note(corrected_facts={"cell[a][b]": "7\nSolution:\nFinal answer: 7"}),
                     note(corrected_facts={"cell[a][b]": "7", "cell[ a ][b]": "7"})]
        tree, engine = self.tree(malformed)
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [0.]*4)
        self.assertTrue(all(not n["checks"]["format_valid"] for n in tree.notebook["model_notes"]))

    def test_truncated_and_empty_notebooks_get_zero_without_dropping_samples(self):
        tree, engine = self.tree([note(), "", note(), note()])
        original = engine.sample

        def truncated(requests):
            result = original(requests)
            result[0].completions[0] = replace(result[0].completions[0], finish_reason="length")
            return result

        engine.sample = truncated
        self.append(tree, engine)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [0., 0., 1., 1.])

    def test_existing_rl_loss_uses_actual_note_rewards_and_completion_actions(self):
        tree, engine = self.tree([note(), note(corrected_facts={"cell[a][b]": "8"})]*2)
        self.append(tree, engine)
        samples = tree.groups[-1].samples
        rewards = torch.tensor([s.reward for s in samples])
        advantages = group_advantages(rewards)
        current = torch.tensor([s.sampler_logprobs for s in samples], requires_grad=True)
        before = current.detach().clone()
        loss = group_loss(current, before, before, advantages, torch.ones_like(current, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool), eps_low=.2, eps_high=.28, beta=0., norm_const=4*512)
        loss.backward()
        self.assertLess(current.grad[0].sum(), 0.)
        self.assertGreater(current.grad[1].sum(), 0.)
        self.assertEqual(advantages.tolist(), [.5, -.5, .5, -.5])

    def test_degenerate_groups_are_preserved_and_report_zero_policy_advantage(self):
        tree, engine = self.tree()
        report = self.append(tree, engine)
        self.assertEqual(report["generated_groups"], 1)
        self.assertEqual(report["nondegenerate_groups"], 0)
        self.assertEqual(len(tree.groups[-1].samples), 4)
        self.assertEqual(group_advantages(torch.tensor([s.reward for s in tree.groups[-1].samples])).tolist(), [0.]*4)

    def test_context_budget_and_duplicate_generation_are_explicit(self):
        for expected, kwargs in (("BUDGET_STOP", {"should_stop": lambda: True}),
                                 ("CONTEXT_BLOCKED", {"cfg": replace(self.cfg, context_limit=100)})):
            with self.subTest(expected=expected):
                tree, engine = self.tree()
                report = self.append(tree, engine, **kwargs)
                self.assertEqual(report["generated_groups"], 0)
                self.assertEqual(tree.notebook["rewarded_notebook"]["status"], expected)
                self.assertEqual(len(engine.requests), 2)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    self.append(tree, engine)

    def test_no_actual_recovery_has_no_notebook(self):
        for replies in ([["Final answer: 7"]*2],
                        [["Final answer: 8"]*2, ["Final answer: 8"]*2],
                        [["Final answer: bananas"]*2]):
            engine = Engine(replies)
            tree = build_tree(self.actor, [object()], engine, self.verifier,
                              self.prompts, self.cfg, 0, "verified-any-visual-v1")
            report = self.append(tree, engine)
            self.assertEqual(report["generated_groups"], 0)
            self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "NO_VERIFIED_RECOVERY")
            self.assertFalse(tree.notebook["rewarded_notebook"]["recovery_verified"])

    def test_plain_unknown_recovery_generates_verified_claims_without_diagnosis_or_rerouting(self):
        self.actor = replace(self.actor, mode="plain")
        good = note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1])
        tree, engine = self.tree([good, note(cause="VISUAL_MISREAD"),
            note(cause="UNKNOWN", wrong_item="cell[a][b]", first_attempt_lines=[1]),
            good], wrong="Final answer: 8")
        before = copy.deepcopy(tree)
        self.assertEqual(tree.route, "VISION_OFF")
        self.assertEqual(tree.notebook["routing"]["observed_evidence"], "UNKNOWN")
        report = self.append(tree, engine)
        group = tree.groups[-1]
        self.assertEqual([s.reward for s in group.samples], [1., 0., 0., 1.])
        self.assertEqual(report["class_counts"]["UNKNOWN"], 1)
        self.assertEqual(report["nondegenerate_groups"], 1)
        self.assertEqual(group_advantages(torch.tensor([s.reward for s in group.samples])).tolist(), [.5, -.5, -.5, .5])
        self.assertEqual(tree.groups[:-1], before.groups)
        self.assertEqual(tree.route, before.route)
        self.assertEqual(tree.causes, before.causes)
        self.assertEqual(tree.notebook["routing"], before.notebook["routing"])
        record = tree.notebook["rewarded_notebook"]
        self.assertIsNone(record["target"])
        self.assertTrue(record["recovery_verified"])
        self.assertEqual(record["diagnosis_status"], "UNVERIFIABLE")
        self.assertFalse(record["used_for_routing"])
        for row in tree.notebook["model_notes"]:
            self.assertIsNone(row["checks"]["diagnosis_verified"])
            self.assertEqual(row["diagnosis_status"], "UNVERIFIABLE")
            self.assertFalse(row["module_causality_certified"])
            self.assertFalse(row["lesson_semantics_verified"])
        parent = next(s for g in before.groups for s in g.samples
                      if s.sample_id == group.samples[0].parent_sample_id)
        self.assertEqual(group.samples[0].prompt_token_ids[:len(parent.prompt_token_ids + parent.completion_token_ids)],
                         parent.prompt_token_ids + parent.completion_token_ids)
        self.assertEqual(len({tuple(s.prompt_token_ids) for s in group.samples}), 1)
        self.assertEqual(len({s.policy_version for s in group.samples}), 1)
        self.assertEqual(len(group.samples), 4)

    def test_unknown_claims_keep_all_history_answer_and_registered_fact_checks(self):
        self.actor = replace(self.actor, mode="plain")
        good = note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1])
        tree, _ = self.tree(wrong="Final answer: 8")
        first, parent, target = _recovered_path(tree, self.verifier)
        self.assertIsNone(target)
        cases = [good, note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[2]),
            note(cause="UNKNOWN", wrong_item="none", evidence="Verifier result for your previous answer"),
            note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1], previous_answer="7"),
            note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1], corrected_answer="8"),
            note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1], corrected_facts={}),
            note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1], corrected_facts={"cell[a][b]": "8"}),
            note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1],
                 corrected_facts={"cell[a][b]": "7", "cell[extra][b]": "7"}),
            note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1],
                 corrected_facts={"cell[a][b]": "7", "cell[ a ][b]": "8"}),
            good[:-1] + ', "cause": "UNKNOWN"}',
            good.replace('"corrected_answer": "7"', '"corrected_answer": NaN')]
        for index, raw in enumerate(cases):
            with self.subTest(index=index):
                checked = _grade(tree.problem_id, raw, "stop", first, parent, None, self.verifier)
                self.assertEqual(checked["reward"], float(index == 0))
                self.assertIsNone(checked["checks"]["diagnosis_verified"])
        self.assertEqual(_grade(tree.problem_id, good, "length", first, parent, None, self.verifier)["reward"], 0.)

    def test_unknown_does_not_bypass_a_known_diagnosis_with_or_without_target(self):
        tree, _ = self.tree()
        first, parent, target = _recovered_path(tree, self.verifier)
        good = note(cause="UNKNOWN", wrong_item="none")
        for supplied in (target, None):
            with self.subTest(target=supplied):
                checked = _grade(tree.problem_id, good, "stop", first, parent, supplied, self.verifier)
                self.assertEqual(checked["reward"], 0.)
        self.assertFalse(_grade(tree.problem_id, good, "stop", first, parent, None,
                                self.verifier)["checks"]["unknown_acknowledgment_valid"])

    def test_unknown_constant_rewards_preserve_full_group_and_zero_advantage(self):
        self.actor = replace(self.actor, mode="plain")
        for value in ("7", "8"):
            with self.subTest(corrected_fact=value):
                good = note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1],
                            corrected_facts={"cell[a][b]": value})
                tree, engine = self.tree([good]*4, wrong="Final answer: 8")
                report = self.append(tree, engine)
                rewards = [s.reward for s in tree.groups[-1].samples]
                self.assertEqual(rewards, [float(value == "7")]*4)
                self.assertEqual(report["nondegenerate_groups"], 0)
                self.assertEqual(group_advantages(torch.tensor(rewards)).tolist(), [0.]*4)

    def test_known_diagnosis_candidate_keeps_preference_over_earlier_unknown_recovery(self):
        cfg = replace(self.cfg, branch={1: 2, 2: 0})
        observed_earlier_unknown = False
        for wrongs in (["Final answer: 8", WRONG], [WRONG, "Final answer: 8"]):
            engine = Engine([wrongs, [RIGHT]*2, [RIGHT]*2, [note()]*4])
            tree = build_tree(self.actor, [object()], engine, self.verifier,
                              self.prompts, cfg, 0, "verified-any-visual-v1")
            firsts = {s.sample_id: s for s in tree.groups[0].samples}
            earliest = min((s for g in tree.groups[1:] for s in g.samples),
                           key=lambda s: (s.turn, _hash(s.sample_id)))
            observed_earlier_unknown |= self.verifier.cause(tree.problem_id,
                firsts[earliest.parent_sample_id].result) == "UNKNOWN"
            first, _, target = _recovered_path(tree, self.verifier)
            self.assertEqual(first.text, WRONG)
            self.assertEqual(target.cause3, "VISUAL")
            self.append(tree, engine, cfg=cfg)
            self.assertEqual(tree.notebook["rewarded_notebook"]["target"]["cause3"], "VISUAL")
        self.assertTrue(observed_earlier_unknown)

    def test_public_actor_schema_is_bound_for_plain_and_pair_notes_without_private_values(self):
        for source, mode, count in (("fake", "plain", 1), ("gen-graph-pair-v1", "obs", 2)):
            with self.subTest(source=source):
                self.actor = replace(self.actor, source=source, mode=mode,
                    image_paths=tuple(f"images/{i}.png" for i in range(count)), image_sha256=("a"*64,)*count)
                public = "PUBLIC_PAIR_SCHEMA: dir[<id>] = <source>-><target>"
                self.prompts["source_observation_instruction"] = {"gen-graph-pair-v1": public}
                tree, engine = self.tree(wrong="Final answer: 8")
                self.append(tree, engine)
                instruction = engine.tokenizer.encoded[-1]
                expected = public if count == 2 else self.prompts["observation_instruction"][self.actor.family]
                self.assertIn(expected, instruction)
                self.assertNotIn('"gold"', instruction)
                self.assertNotIn('cell[a][b] = 7', instruction)
                self.assertNotIn('"cause3"', instruction)

    def test_note_template_renders_metadata_and_public_schema_in_one_pass(self):
        self.actor = replace(self.actor, mode="plain", answer_format="short_text")
        _, item, lineage, *_ = records()
        root = self.root.parent / "text-vault"
        write_split(root, [self.actor], [replace(item, answer_type="short_text")], [lineage])
        verifier = Verifier(root)
        wrong = "Final answer: literal-{observation_instruction}-{first_attempt_metadata}"
        tree, engine = self.tree(wrong=wrong, verifier=verifier)
        self.append(tree, engine, verifier=verifier)
        metadata = json.dumps({"status": "INCORRECT", "finish_reason": "stop",
            "extracted_answer": "literal-{observation_instruction}-{first_attempt_metadata}"})
        self.assertIn(metadata, engine.tokenizer.encoded[-1])
        self.assertIn(self.prompts["observation_instruction"][self.actor.family], engine.tokenizer.encoded[-1])

    def test_missing_or_wrong_actor_binding_fails_before_note_generation(self):
        for actors in ({}, {"foreign": self.actor},
                       {self.actor.problem_id: replace(self.actor, problem_id="foreign")}):
            with self.subTest(actors=actors):
                tree, engine = self.tree()
                with self.assertRaises(ValueError):
                    self.append(tree, engine, actors=actors)
                self.assertEqual(len(engine.requests), 2)
                self.assertNotIn("rewarded_notebook", tree.notebook)

    def test_nonvisual_and_finalization_claims_are_checked(self):
        cases = [("Observed facts:\ncell[a][b] = 7\nSolution:\nFinal answer: 8", "NON_VISUAL", "8",
                  "Final answer: 8"), ("I forgot the required final line.", "FINALIZATION", None,
                                       "I forgot the required final line.")]
        for wrong, cause, previous, evidence in cases:
            with self.subTest(cause=cause):
                line_id = next(i for i, line in enumerate(wrong.splitlines(), 1) if line == evidence)
                correct = note(cause=cause, wrong_item="none", previous_answer=previous,
                               first_attempt_lines=[line_id])
                tree, engine = self.tree([correct]*4, wrong=wrong)
                self.append(tree, engine)
                self.assertEqual([s.reward for s in tree.groups[-1].samples], [1.]*4)
                self.assertEqual(tree.route, "VISION_OFF")

    def test_broken_histories_policy_changes_and_unbound_prompt_fail_before_sampling(self):
        tree, engine = self.tree()
        tree.groups[1].samples[0].prompt_token_ids = [99]
        tree.groups[1].samples[1].prompt_token_ids = [99]
        with self.assertRaisesRegex(ValueError, "history"):
            self.append(tree, engine)
        self.assertNotIn("rewarded_notebook", tree.notebook)
        tree, engine = self.tree()
        engine.policy_version = 1
        with self.assertRaisesRegex(ValueError, "same-policy"):
            self.append(tree, engine)
        engine.policy_version = 0
        self.prompts.pop("rewarded_notebook")
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.append(tree, engine)

    def test_program_grading_works_through_existing_isolated_service(self):
        with VerifierService(self.root, forbidden_roots=[], timeout=30) as service:
            for wrong, good in ((WRONG, note()), ("Final answer: 8",
                    note(cause="UNKNOWN", wrong_item="none", first_attempt_lines=[1]))):
                tree, engine = self.tree([good, note(corrected_facts={"cell[a][b]": "9"})]*2,
                                         wrong=wrong, verifier=service)
                self.append(tree, engine, verifier=service)
                self.assertEqual([s.reward for s in tree.groups[-1].samples], [1., 0., 1., 0.])


if __name__ == "__main__":
    unittest.main()
