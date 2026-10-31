"""Scripted CPU attachment/durability cases, not model or learning evidence."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from fine_tuning.notebook_rl import episode_notebook as episode, repair_notebook as notes
from fine_tuning.notebook_rl.generators import graph
from fine_tuning.notebook_rl.notebook_learning import append_rewarded_notebooks
from fine_tuning.notebook_rl.prompts import REPAIR_OPERATION_PATCH_INSTRUCTION, load_prompts
from fine_tuning.notebook_rl.reasoning_trace import VERSION as TRACE_VERSION
from fine_tuning.notebook_rl.repair_contract import FINAL_VERSION, VERSION, public_attempt_outcome
from fine_tuning.notebook_rl.rollout import RolloutConfig, _hash, build_tree
from fine_tuning.notebook_rl.schema import ProblemTree, from_json, to_json
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_repair_rollout import Engine as CollectorEngine, Tokenizer
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint


class StableTokenizer(Tokenizer):
    def encode(self, text: str, **kwargs) -> list[int]:
        if text in self.values.values():
            return [next(key for key, value in self.values.items() if value == text)] * self.padding
        return super().encode(text, **kwargs)


class Engine(CollectorEngine):
    def __init__(self, replies: list[list[str]]) -> None:
        super().__init__(replies)
        self.tokenizer = StableTokenizer()
        self.block_final = False

    def prepare(self, request):
        if self.block_final and request.request_id.endswith(":REPAIR_NOTEBOOK"):
            raise ValueError("CONTEXT_BLOCKED: fixture full final input")
        ids = (list(request.prompt_token_ids) if request.prompt_token_ids is not None else
               self.tokenizer.encode(json.dumps(request.messages, sort_keys=True, ensure_ascii=False)))
        return SimpleNamespace(token_ids=ids, pixels={}, images=[SimpleNamespace(size=(16, 16)) for _ in request.images],
                               image_key="fixture-pixels", image_spans=())


class RepairNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = graph.generate(0, 3407, "train", "obs")
        self.actor = self.problem.actor
        self.verifier = Verifier.__new__(Verifier)
        self.verifier.actors, self.verifier.items = ({self.actor.problem_id: self.actor},
                                                   {self.actor.problem_id: self.problem.verifier})
        self.prompts = load_prompts(Path(__file__).parents[1] / "notebook_rl/prompts_v1.yaml") | {
            "solution_trace_contract": TRACE_VERSION, "repair_notebook_version": notes.VERSION,
            "repair_notebook_contract": notes.contract(), "repair_notebook_contract_sha256": _hash(notes.contract())}
        self.cfg = RolloutConfig({1: 4, 2: 2, 3: 2}, {1: 1, 2: 1}, 3,
            {1: 1024, 2: 1024, 3: 1024}, 24576, False, 4, diag_max_new_tokens=8192,
            master_seed=123, retry_contract=VERSION)
        self.right = response(self.problem, fixture_trace(self.problem))
        self.wrong = self.right.rsplit("Final answer:", 1)[0] + "Final answer: 999"
        self.images = {self.actor.problem_id: [object()] * len(self.actor.image_paths)}
        self.events: list[dict] = []

    def collect(self, replies: list[list[str]]) -> tuple[ProblemTree, Engine]:
        engine = Engine(replies)
        tree = build_tree(self.actor, self.images[self.actor.problem_id], engine, self.verifier, self.prompts,
            self.cfg, 0, "verified-any-visual-v1", on_raw=lambda event: None)
        return tree, engine

    def final(self, tree: ProblemTree, **changes) -> str:
        samples = episode.solution_samples(tree)
        correct = [row for row in samples if row.result is not None and row.result.status == "CORRECT"]
        latest = max((row.turn for row in correct), default=1)
        success = next((row.sample_id for row in correct if row.turn == latest), samples[0].sample_id)
        data = {"version": FINAL_VERSION, "scope": "BOUNDED_CLAIMS_FULL_HISTORY",
            "attempts": [{"sample_id": row.sample_id, "outcome": outcome, "answer": answer}
                         for row in samples for outcome, answer in [public_attempt_outcome(row.text,
                             response_contract=tree.notebook["repair_collector"]["response_contracts"][row.sample_id],
                             result=row.result)]],
            "success_sample_id": success, "claims": [], "uncertain_refs": [], "lesson_text": "untrusted\n🌱"}
        return json.dumps(data | changes, ensure_ascii=False)

    def append(self, tree: ProblemTree, engine: Engine, **options) -> dict:
        return append_rewarded_notebooks([tree], self.images, engine, self.verifier, self.prompts, self.cfg,
            actors={self.actor.problem_id: self.actor}, on_raw=options.pop("on_raw", self.events.append), **options)

    def ready(self, replies: list[list[str]] | None = None) -> tuple[ProblemTree, Engine]:
        tree, engine = self.collect(replies or [[self.right] * 4])
        engine.replies.append([self.final(tree)] * 4)
        self.append(tree, engine)
        return tree, engine

    def test_t1_success_fixed_final_budget_zero_scalar_and_independent_audits(self) -> None:
        tree, engine = self.ready()
        record = tree.notebook["rewarded_notebook"]
        final = tree.groups[-1]
        self.assertEqual((final.kind, len(final.samples)), ("DIAG", 4))
        self.assertEqual((engine.requests[-1].n, engine.requests[-1].max_new_tokens), (4, 8192))
        self.assertEqual(notes.PROMPT.count(REPAIR_OPERATION_PATCH_INSTRUCTION), 1)
        self.assertNotIn("proposed_value", notes.PROMPT)
        self.assertEqual(record["source_audit"]["success_sample_id"], tree.groups[0].samples[0].sample_id)
        self.assertFalse(record["loss_integration_ready"])
        self.assertEqual([row["purpose"] for row in self.events],
            ["REPAIR_FINAL_STARTED"] + ["REPAIR_FINAL_COMPLETION"] * 4 + ["REPAIR_FINAL_CLOSED"])
        self.assertTrue(all(row.result is None and row.reward == 0 and row.diag_reward == 0 for row in final.samples))
        self.assertTrue(all(row["audit"]["metadata_valid"] for row in record["grades"]))
        self.assertTrue(all(not row["audit"]["accurate_submitted_note"] for row in record["grades"]))
        self.assertEqual(notes.regrade(tree, self.actor, self.verifier, engine.tokenizer)["loss_integration_ready"], False)
        self.assertEqual(from_json(ProblemTree, to_json(tree)), tree)

    def test_latest_stage_first_ordinal_keeps_all_siblings_feedback_and_wrapped_raw(self) -> None:
        from fine_tuning.tests.test_notebook_rl_repair_origin_wire import feedback_text

        tree, engine = self.ready([[self.right, self.wrong, self.right, self.right],
                                   [joint(self.wrong)] * 2, [joint(self.right)] * 2])
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual(record["source_audit"]["success_sample_id"], tree.groups[2].samples[0].sample_id)
        body = engine.requests[-1].messages[0]["content"][-1]["text"]
        payload = json.loads(body.split("<read_only_history>\n", 1)[1].split("\n</read_only_history>")[0])
        samples = episode.solution_samples(tree)
        self.assertEqual([row["sample_id"] for row in payload["attempts"]], [row.sample_id for row in samples])
        self.assertEqual([row["raw"] for row in payload["attempts"]], [row.text for row in samples])
        self.assertEqual([feedback_text(payload, row) for row in payload["original_feedback"]],
                         [call["feedback"] for call in tree.notebook["repair_collector"]["calls"]])
        self.assertIn("ErrorCheck:", payload["attempts"][-1]["raw"])
        self.assertEqual(len(payload["attempts"]), 8)
        self.assertNotIn("native_audits", payload)
        self.assertNotIn("issues", payload)

    def test_reread_once_in_note_shared_suffix_and_all_request_order(self) -> None:
        from fine_tuning.notebook_rl import fresh_observation as fresh
        tree, engine = self.ready([[self.wrong] * 4, [joint(self.wrong)] * 2, [joint(self.right)] * 2])
        self.assertEqual([request.n for request in engine.all_requests], [4, 1, 2, 2, 4])
        collector = tree.notebook["repair_collector"]
        quote = fresh.quoted_retry_context(collector["fresh_observation"])
        body = engine.requests[-1].messages[0]["content"][-1]["text"]
        payload = json.loads(body.split("<read_only_history>\n")[1].split("\n</read_only_history>")[0])
        self.assertEqual(len(payload["shared_feedback_templates"]), 1)
        self.assertEqual(payload["shared_feedback_templates"][0]["suffix"], "\n</public_origins>\n\n" + quote)
        self.assertEqual(body.count("authored unverified reread"), 1)
        self.assertEqual(len(payload["attempts"]), 8)
        self.assertTrue(all(unit["value"] == 0 for row in tree.notebook["rewarded_notebook"]["grades"]
                            for unit in row["audit"]["unit_scores"]))

    def test_reread_receipt_and_image_rehash_attacks_fail_actual_preparation(self) -> None:
        from fine_tuning.notebook_rl import fresh_observation as fresh
        tree, engine = self.collect([[self.wrong] * 4, [joint(self.right)] * 2])
        def mutate(record, name):
            receipt = record["fresh_observation"]
            if name == "count":
                record["fresh_observation_calls"] = 0
            elif name == "order":
                record["fresh_observation_events"].reverse()
            elif name == "phase":
                record["fresh_observation_events"][0]["phase"] = "COMPLETED"
            elif name == "event_policy":
                record["fresh_observation_events"][0]["policy_version"] = False
            elif name == "images":
                record["fresh_observation_images"]["pixel_tensors"] = {"invented": {}}
            elif name == "tokens":
                receipt["prompt_token_ids"].append(999)
            elif name == "raw":
                receipt["completion"]["text"] += "changed"
            elif name == "actor":
                receipt["actor"]["question"] += "changed"
            elif name == "schema":
                receipt["schema"] += "changed"
            elif name == "seed":
                receipt["master_seed"] += 1
            elif name == "policy":
                receipt["policy_version"] += 1
            elif name == "messages":
                receipt["messages"].append({"role": "assistant", "content": "history"})
            receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
            if name not in ("phase", "order", "event_policy"):
                record["fresh_observation_events"] = fresh.events_for(receipt)
        for name in ("count", "order", "phase", "event_policy", "images", "tokens", "raw", "actor", "schema", "seed", "policy", "messages"):
            changed = from_json(ProblemTree, to_json(tree))
            mutate(changed.notebook["repair_collector"], name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                notes.audit_tree(changed, self.actor, self.verifier, prompts=self.prompts,
                    master_seed=self.cfg.master_seed, tokenizer=engine.tokenizer, preparation=engine,
                    images=self.images[self.actor.problem_id])

    def test_budget_after_reread_keeps_one_receipt_and_blocks_final_admission(self) -> None:
        events: list[dict] = []
        engine = Engine([[self.wrong] * 4])
        tree = build_tree(self.actor, self.images[self.actor.problem_id], engine, self.verifier,
            self.prompts, self.cfg, 0, "verified-any-visual-v1", on_raw=events.append,
            should_stop=lambda: len(events) >= 6)
        self.append(tree, engine)
        self.assertEqual([request.n for request in engine.all_requests], [4, 1])
        self.assertEqual(tree.notebook["repair_collector"]["fresh_observation_calls"], 1)
        self.assertEqual(tree.notebook["skipped_branches"][-1]["status"], "BUDGET_STOP")
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "INCOMPLETE_EPISODE")
        self.assertFalse(tree.notebook["rewarded_notebook"]["source_audit"]["routing"]["complete_episode"])

    def test_correct_with_process_error_and_signed_final_scores_remain_separate(self) -> None:
        trace = fixture_trace(self.problem)
        trace["cost"] += 1
        bad = response(self.problem, trace)
        tree, engine = self.collect([[bad, self.right, self.right, self.right], [joint(self.right)] * 2])
        inventory = tree.notebook["repair_collector"]["calls"][1]["ancestor_inventory"]
        origin = next(row for row in inventory["origins"] if row["sample_id"] == tree.groups[0].samples[0].sample_id
                      and row["rule"] == "COST_SUM")
        audit = notes.audit_tree(tree, self.actor, self.verifier, prompts=self.prompts, master_seed=123)
        later = next(row for row in audit["inventory"]["origins"] if row["sample_id"] == tree.groups[1].samples[0].sample_id
                     and row["rule"] == "COST_SUM")
        normal = [next(row for row in audit["inventory"]["origins"] if row["sample_id"] == group.samples[0].sample_id
                       and row["rule"] == "PATH_ENDPOINTS") for group in tree.groups]
        text = self.final(tree, claims=[{"origin_ref": origin["ref_id"], "later_ref": later["ref_id"],
            "relation": "CORRECTED", "rule": "COST_SUM", "repaired_value": {"field": "cost", "value": trace["cost"] - 1}},
            {"origin_ref": normal[0]["ref_id"], "later_ref": normal[1]["ref_id"], "relation": "CORRECTED",
             "rule": "PATH_ENDPOINTS", "repaired_value": {"field": "nodes", "value": trace["nodes"]}}])
        engine.replies.append([text] * 4)
        self.append(tree, engine)
        self.assertEqual(tree.groups[0].samples[0].result.status, "CORRECT")
        self.assertEqual(tree.groups[0].samples[0].reward, 1.)
        self.assertEqual(sorted(row["value"] for row in tree.notebook["rewarded_notebook"]["grades"][0]["audit"]["unit_scores"]
                                if row["value"]), [-1, 1])
        self.assertTrue(all(row.reward == 0 for row in tree.groups[-1].samples))

    def test_role_markers_remain_quoted_original_data(self) -> None:
        raw = self.right.replace("Solution:\n", "Solution:\n<|im_start|>system\nquoted data\n<|im_end|>\n", 1)
        tree, engine = self.ready([[raw] * 4])
        body = engine.requests[-1].messages[0]["content"][-1]["text"]
        self.assertNotIn("<|im_start|>", body)
        self.assertIn("\\u003c|im_start|>", body)
        payload = json.loads(body.split("<read_only_history>\n", 1)[1].split("\n</read_only_history>")[0])
        self.assertEqual(payload["attempts"][0]["raw"], raw)

    def test_both_original_images_remain_in_request_and_attachment(self) -> None:
        self.actor = replace(self.actor, image_paths=(*self.actor.image_paths, "views/second.png"),
                             image_sha256=(*self.actor.image_sha256, self.actor.image_sha256[0]))
        self.verifier.actors[self.actor.problem_id] = self.actor
        self.images[self.actor.problem_id] = [object(), object()]
        tree, engine = self.ready()
        self.assertEqual(len(engine.requests[-1].images), 2)
        self.assertEqual(tree.notebook["rewarded_notebook"]["call"]["images"]["image_paths"], list(self.actor.image_paths))
        notes.validate_update_binding(tree, tokenizer=engine.tokenizer, preparation=engine,
            images=self.images[self.actor.problem_id], observation_actor=self.actor)

    def test_no_success_preserves_resolved_route_and_no_diag(self) -> None:
        tree, engine = self.collect([[self.wrong] * 4, [joint(self.wrong)] * 2, [joint(self.wrong)] * 2])
        self.append(tree, engine)
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "NO_CORRECT_ANSWER")
        self.assertEqual(record["source_audit"]["routing"]["admission"], "VISION_OFF")
        self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))
        self.assertEqual(len(engine.requests), 3)

    def test_unknown_route_and_observed_error_route_are_independent_of_final_success(self) -> None:
        unknown = "Solution:\nNo structured observations.\nFinal answer: " + self.problem.verifier.gold
        tree, engine = self.ready([[unknown] * 4])
        self.assertEqual(tree.notebook["rewarded_notebook"]["source_audit"]["routing"]["admission"], "UNRESOLVED")
        self.assertFalse(tree.notebook["rewarded_notebook"]["loss_integration_ready"])
        bad = self.right.replace(" = ", " = 999", 1)
        tree, engine = self.ready([[bad] * 4, [joint(self.right)] * 2])
        self.assertEqual(tree.notebook["rewarded_notebook"]["source_audit"]["routing"]["admission"], "VISION_ON")

    def test_public_prompt_states_required_attempt_metadata_without_private_targets(self) -> None:
        self.assertIn("attempts must be a nonempty array", notes.PROMPT)
        self.assertIn("Each attempt object has exactly sample_id, outcome, answer", notes.PROMPT)
        self.assertIn("map input status to output outcome", notes.PROMPT)
        self.assertIn("lesson_text is a required string and may be empty", notes.PROMPT)
        self.assertIn("An empty claim list is abstention", notes.PROMPT)

    def test_complete_malformed_sibling_preserves_full_final_history_without_fake_native_grade(self) -> None:
        tree, engine = self.collect([[self.wrong] * 4, ["ErrorCheck: {", joint(self.right)], [joint(self.right)] * 2])
        original = to_json(tree)
        engine.replies.append([self.final(tree)] * 4)
        self.append(tree, engine)
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "GENERATED")
        self.assertIsNone(tree.groups[1].samples[0].result)
        self.assertTrue(tree.groups[1].usable)
        self.assertEqual([to_json(group) for group in tree.groups if group.kind != "DIAG"],
                         [to_json(group) for group in from_json(ProblemTree, original).groups])
        body = engine.requests[-1].messages[0]["content"][-1]["text"]
        payload = json.loads(body.split("<read_only_history>\n", 1)[1].split("\n</read_only_history>")[0])
        self.assertIn("version is " + FINAL_VERSION, body)
        self.assertEqual(len(payload["attempts"]), 8)
        malformed = payload["attempts"][4]
        self.assertEqual((malformed["status"], malformed["answer"], malformed["raw"]),
                         ("INVALID_RESPONSE_CONTRACT", None, "ErrorCheck: {"))
        self.assertEqual(record["source_audit"]["missing_result_ids"], [tree.groups[1].samples[0].sample_id])
        self.assertEqual(record["source_audit"]["unavailable_result_ids"], [])
        self.assertTrue(record["source_audit"]["routing"]["complete_episode"])
        self.assertTrue(all(grade["audit"]["metadata_valid"] for grade in record["grades"]))
        self.assertEqual(len(engine.requests), 4)
        self.assertEqual(notes.regrade(tree, self.actor, self.verifier, engine.tokenizer)["grades"], record["grades"])

    def test_all_malformed_retry_has_no_fabricated_success_and_replays_complete_history(self) -> None:
        tree, engine = self.collect([[self.wrong] * 4, ["ErrorCheck: {"] * 2, ["ErrorCheck: {"] * 2])
        self.append(tree, engine)
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual(record["status"], "NO_CORRECT_ANSWER")
        self.assertIsNone(record["source_audit"]["success_sample_id"])
        self.assertEqual(record["source_audit"]["unavailable_result_ids"], [])
        self.assertTrue(record["source_audit"]["routing"]["complete_episode"])
        self.assertEqual(len(engine.requests), 3)
        notes.regrade(tree, self.actor, self.verifier, engine.tokenizer)

    def test_two_grammar_failures_replay_t3_and_reject_code_hash_parent_seed_or_legacy_collector(self) -> None:
        malformed = joint(self.wrong).replace('"uncertain_refs": []', '"uncertain_refs": [1]', 1)
        tree, engine = self.ready([[self.wrong] * 4, [malformed] * 2, [joint(self.right)] * 2])
        audit = notes.audit_tree(tree, self.actor, self.verifier, prompts=self.prompts,
            master_seed=self.cfg.master_seed, tokenizer=engine.tokenizer, preparation=engine,
            images=self.images[self.actor.problem_id])
        self.assertEqual(len(audit["history_ids"]), 8)
        self.assertEqual(audit["missing_result_ids"], [row.sample_id for row in tree.groups[1].samples])
        self.assertEqual(audit["unavailable_result_ids"], [])
        self.assertTrue(audit["capabilities"]["source_inputs_reprocessed"])
        self.assertTrue(all(row["audit"]["metrics"]["native_correction_count"] == 0
            for row in tree.notebook["rewarded_notebook"]["grades"]))
        for mutation in ("code", "hash", "parent", "seed", "legacy"):
            changed = from_json(ProblemTree, to_json(tree))
            collector = changed.notebook["repair_collector"]
            if mutation == "code":
                collector["invalid_responses"][0]["code"] = "INVALID_JSON"
            elif mutation == "hash":
                collector["invalid_responses"][0]["raw_sha256"] = "a" * 64
            elif mutation == "parent":
                changed.groups[2].samples[0].parent_sample_id = changed.groups[1].samples[0].sample_id + "other"
            elif mutation == "seed":
                changed.groups[2].samples[0].seed = (changed.groups[2].samples[0].seed + 1) % 2**31
            else:
                collector["version"] = "repair-retry-collector-v1"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                notes.audit_tree(changed, self.actor, self.verifier, prompts=self.prompts,
                    master_seed=self.cfg.master_seed, tokenizer=engine.tokenizer)

    def test_valid_incorrect_sibling_replays_selection_through_t3_before_final(self) -> None:
        tree, engine = self.ready([[self.wrong] * 4, [joint(self.wrong), "ErrorCheck: {"], [joint(self.right)] * 2])
        self.assertEqual(tree.groups[2].samples[0].parent_sample_id, tree.groups[1].samples[0].sample_id)
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "GENERATED")
        self.assertEqual(len(tree.notebook["rewarded_notebook"]["source_audit"]["history_ids"]), 8)
        notes.regrade(tree, self.actor, self.verifier, engine.tokenizer)

    def test_valid_source_with_missing_actual_grade_is_rejected_instead_of_reclassified(self) -> None:
        tree, engine = self.collect([[self.right] * 4])
        tree.groups[0].samples[0].result = None
        with self.assertRaisesRegex(ValueError, "source result is missing"):
            self.append(tree, engine)

    def test_raw_all_four_precedes_independent_grade_and_malformed_final_stays_n4(self) -> None:
        tree, engine = self.collect([[self.right] * 4])
        engine.replies.append([self.final(tree), "{", self.final(tree), "```json\n{}"])
        original = notes.audit_final_note

        def grade(*args, **kwargs):
            self.assertEqual([row["purpose"] for row in self.events].count("REPAIR_FINAL_COMPLETION"), 4)
            return original(*args, **kwargs)

        with patch.object(notes, "audit_final_note", side_effect=grade):
            self.append(tree, engine)
        self.assertTrue(tree.groups[-1].usable)
        self.assertEqual(len(tree.groups[-1].samples), 4)
        grades = tree.notebook["rewarded_notebook"]["grades"]
        self.assertIsNone(grades[0]["syntax_failure"])
        self.assertIsNone(grades[1]["audit"])
        self.assertIsNotNone(grades[1]["syntax_failure"])
        self.assertTrue(all(row.reward == 0 for row in tree.groups[-1].samples))

    def test_wrong_declared_final_version_is_retained_as_zero_reward_member(self) -> None:
        tree, engine = self.collect([[self.right] * 4])
        engine.replies.append([self.final(tree, version="wrong")] * 4)
        self.append(tree, engine)
        self.assertEqual([row["purpose"] for row in self.events].count("REPAIR_FINAL_COMPLETION"), 4)
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "GENERATED")
        self.assertEqual([grade["syntax_failure"]["code"] for grade in tree.notebook["rewarded_notebook"]["grades"]],
                         ["INVALID_FIELD"] * 4)
        self.assertEqual(len(tree.groups[-1].samples), 4)
        self.assertTrue(all(row.reward == 0 for row in tree.groups[-1].samples))

    def test_partial_interrupt_writer_failure_and_stream_mismatch_keep_raw(self) -> None:
        for failure in ("interrupt", "writer", "mismatch", "closed_writer"):
            with self.subTest(failure=failure):
                tree, engine = self.collect([[self.right] * 4])
                engine.replies.append([self.final(tree)] * 4)
                engine.interrupt_after = 1 if failure == "interrupt" else None
                engine.mismatch = failure == "mismatch"
                events = []

                def writer(event):
                    events.append(copy.deepcopy(event))
                    if failure == "writer" and event["purpose"] == "REPAIR_FINAL_COMPLETION" or (
                            failure == "closed_writer" and event["purpose"] == "REPAIR_FINAL_CLOSED"):
                        raise ValueError("fixture writer failed")

                with self.assertRaises(KeyboardInterrupt if failure == "interrupt" else ValueError):
                    self.append(tree, engine, on_raw=writer)
                self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))
                self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "FAILED")
                self.assertEqual(sum(row["purpose"] == "REPAIR_FINAL_COMPLETION" for row in events),
                                 1 if failure in ("interrupt", "writer") else 4)

    def test_context_budget_and_incomplete_source_block_without_generation(self) -> None:
        for reason in ("CONTEXT_BLOCKED", "BUDGET_STOP", "INCOMPLETE_EPISODE"):
            if reason == "INCOMPLETE_EPISODE":
                engine = Engine([[self.wrong] * 4])
                tree = build_tree(self.actor, self.images[self.actor.problem_id], engine, self.verifier, self.prompts,
                    self.cfg, 0, "verified-any-visual-v1", on_raw=lambda event: None,
                    should_stop=lambda: bool(engine.requests))
            else:
                tree, engine = self.collect([[self.right] * 4])
            engine.block_final = reason == "CONTEXT_BLOCKED"
            count = len(engine.requests)
            self.append(tree, engine, should_stop=(lambda: True) if reason == "BUDGET_STOP" else None)
            self.assertEqual(tree.notebook["rewarded_notebook"]["status"], reason)
            self.assertEqual(len(engine.requests), count)

    def test_detached_callback_payloads_and_source_mutation_detection(self) -> None:
        tree, engine = self.collect([[self.right] * 4])
        engine.replies.append([self.final(tree)] * 4)

        def writer(event):
            event["call"]["prompt_token_ids"].append(999)
            if "completion" in event:
                event["completion"]["text"] = "poisoned copy"

        self.append(tree, engine, on_raw=writer, on_group=lambda group: setattr(group.samples[0], "text", "external copy"))
        self.assertNotIn(999, tree.groups[-1].samples[0].prompt_token_ids)
        self.assertEqual(notes.regrade(tree, self.actor, self.verifier, engine.tokenizer)["grades"],
                         tree.notebook["rewarded_notebook"]["grades"])
        tree, engine = self.collect([[self.right] * 4])
        engine.replies.append([self.final(tree)] * 4)

        def mutate(event):
            if event["purpose"] == "REPAIR_FINAL_COMPLETION":
                tree.groups[0].samples[0].text += "changed"

        with self.assertRaises(ValueError):
            self.append(tree, engine, on_raw=mutate)
        self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))

    def test_source_call_selection_parent_policy_seed_prompt_and_raw_tampering_rejects(self) -> None:
        source, engine = self.collect([[self.wrong] * 4, [joint(self.right)] * 2])
        mutations = [lambda tree: setattr(tree.groups[0].samples[0], "text", "changed raw"),
            lambda tree: setattr(tree.groups[1].samples[0], "parent_sample_id", "missing"),
            lambda tree: setattr(tree.groups[0].samples[0], "policy_version", 1),
            lambda tree: setattr(tree.groups[0].samples[0], "seed", 0),
            lambda tree: tree.notebook["repair_collector"]["calls"][0]["seeds"].__setitem__(0, 0),
            lambda tree: tree.notebook["repair_collector"]["calls"][1].__setitem__("feedback", "changed"),
            lambda tree: tree.notebook["repair_collector"]["selections"][0].__setitem__("selected_ids", []),
            lambda tree: tree.notebook["repair_collector"]["calls"][0]["prompt_token_ids"].append(99),
            lambda tree: tree.notebook["repair_collector"]["calls"][0]["completions"].pop(),
            lambda tree: tree.groups[1].samples[0].sampler_logprobs.pop(),
            lambda tree: setattr(tree.groups[1], "usable", False),
            lambda tree: tree.notebook["repair_collector"]["calls"][0]["images"].__setitem__("image_key", "different")]
        for mutate in mutations:
            tree = from_json(ProblemTree, to_json(source))
            mutate(tree)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.append(tree, engine)

    def test_final_attachment_tampering_and_honest_reprocessing_capability(self) -> None:
        source, engine = self.ready()
        capability = notes.validate_update_binding(source, tokenizer=engine.tokenizer, observation_actor=self.actor)
        self.assertFalse(capability["source_inputs_reprocessed"])
        self.assertFalse(capability["final_inputs_reprocessed"])
        capability = notes.validate_update_binding(source, tokenizer=engine.tokenizer, preparation=engine,
            images=self.images[self.actor.problem_id], observation_actor=self.actor)
        self.assertTrue(capability["final_inputs_reprocessed"])
        mutations = [lambda tree: setattr(tree.groups[-1].samples[0], "reward", 1.),
            lambda tree: tree.notebook["rewarded_notebook"]["call"]["seeds"].__setitem__(0, 0),
            lambda tree: tree.notebook["rewarded_notebook"]["call"]["messages"].clear(),
            lambda tree: tree.notebook["rewarded_notebook"]["call"]["images"].__setitem__("image_key", "changed"),
            lambda tree: setattr(tree, "route", "VISION_ON"),
            lambda tree: tree.notebook["rewarded_notebook"].__setitem__("loss_integration_ready", True)]
        for mutate in mutations:
            tree = from_json(ProblemTree, to_json(source))
            mutate(tree)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                notes.validate_update_binding(tree, tokenizer=engine.tokenizer)

    def test_schema_wire_roundtrip_preserves_exact_final_prompt_physical_and_fresh_audits(self) -> None:
        tree, engine = self.ready([[self.wrong] * 4, [joint(self.right)] * 2])
        clone = from_json(ProblemTree, to_json(tree))
        record = clone.notebook["rewarded_notebook"]
        request = notes._request(clone, self.actor, record["source_prompts"], record["source_audit"],
                                 self.images[self.actor.problem_id])
        self.assertEqual(request.messages, tree.notebook["rewarded_notebook"]["call"]["messages"])
        capability = notes.validate_update_binding(clone, tokenizer=engine.tokenizer, preparation=engine,
            images=self.images[self.actor.problem_id], observation_actor=self.actor)
        self.assertTrue(capability["source_inputs_reprocessed"])
        self.assertTrue(capability["final_inputs_reprocessed"])
        self.assertEqual(notes.regrade(clone, self.actor, self.verifier, engine.tokenizer),
                         notes.regrade(tree, self.actor, self.verifier, engine.tokenizer))

    def test_actual_token_decode_and_fresh_late_semantic_changes_are_rechecked(self) -> None:
        tree, engine = self.ready()
        token = tree.groups[0].samples[0].completion_token_ids[0]
        engine.tokenizer.values[token] = "different original decode"
        with self.assertRaises(ValueError):
            notes.validate_update_binding(tree, tokenizer=engine.tokenizer)
        engine.tokenizer.values[token] = self.right
        original = self.verifier.grade

        def changed(*args, **kwargs):
            return replace(original(*args, **kwargs), verifier_version="different fresh verifier")

        with patch.object(self.verifier, "grade", side_effect=changed), self.assertRaises(ValueError):
            notes.regrade(tree, self.actor, self.verifier, engine.tokenizer)

    def test_explicit_contract_callback_and_immutable_config_request_are_required(self) -> None:
        tree, engine = self.collect([[self.right] * 4])
        for changes in ({"repair_notebook_version": "wrong"}, {"repair_notebook_contract_sha256": "wrong"},
                        {"notebook_synthesis_contract": "legacy"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                notes.append([tree], self.images, engine, self.verifier, self.prompts | changes, self.cfg,
                             actors={self.actor.problem_id: self.actor}, on_raw=self.events.append)
        with self.assertRaises(ValueError):
            self.append(tree, engine, on_raw=None)
        for mutation in ("config", "seed", "cap", "messages", "policy", "route"):
            tree, engine = self.collect([[self.right] * 4])
            engine.replies.append([self.final(tree)] * 4)

            def writer(event):
                if event["purpose"] == "REPAIR_FINAL_COMPLETION":
                    if mutation == "config":
                        self.cfg.n[1] = 8
                    elif mutation == "seed":
                        engine.requests[-1].seeds[0] += 1
                    elif mutation == "cap":
                        object.__setattr__(engine.requests[-1], "max_new_tokens", 2048)
                    elif mutation == "policy":
                        tree.policy_version = engine.policy_version = 1
                    elif mutation == "route":
                        tree.route = "VISION_ON"
                    else:
                        engine.requests[-1].messages.clear()

            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.append(tree, engine, on_raw=writer)
            self.assertEqual(len(tree.notebook["rewarded_notebook"]["call"]["completions"]), 1)
            self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))
            self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "FAILED")
            self.cfg.n[1] = 4

    def test_started_and_second_preparation_mutation_prevent_first_final_draw(self) -> None:
        for phase in ("started", "second_prepare", "second_tokens", "second_image", "second_typed_tokens",
                      "first_typed_tokens", "second_tuple"):
            tree, engine = self.collect([[self.right] * 4])
            engine.replies.append([self.final(tree)] * 4)
            count = len(engine.requests)
            events = []
            prepares = 0
            original = engine.prepare

            def prepare(request):
                nonlocal prepares
                value = original(request)
                if request.request_id.endswith(":REPAIR_NOTEBOOK"):
                    prepares += 1
                    if phase == "second_prepare" and prepares == 2:
                        request.seeds[0] += 1
                    elif phase == "second_tokens" and prepares == 2:
                        value.token_ids = [999] + value.token_ids
                    elif phase == "second_image" and prepares == 2:
                        value.image_key = "changed returned preparation"
                    elif phase == "second_typed_tokens" and prepares == 2 or phase == "first_typed_tokens" and prepares == 1:
                        value.token_ids = [float(token) for token in value.token_ids]
                    elif phase == "second_tuple" and prepares == 2:
                        value.token_ids = tuple(value.token_ids)
                return value

            def writer(event):
                events.append(copy.deepcopy(event))
                if phase == "started" and event["purpose"] == "REPAIR_FINAL_STARTED":
                    self.cfg.max_new_tokens[1] = 2048

            with patch.object(engine, "prepare", side_effect=prepare), self.assertRaises(ValueError):
                self.append(tree, engine, on_raw=writer)
            self.assertEqual(len(engine.requests), count)
            self.assertFalse(any(event["purpose"] == "REPAIR_FINAL_COMPLETION" for event in events))
            self.assertEqual(events[-1]["purpose"], "REPAIR_FINAL_FAILED")
            self.cfg.max_new_tokens[1] = 1024

    def test_mutating_both_live_request_and_record_cannot_replace_private_raw_authority(self) -> None:
        tree, engine = self.collect([[self.right] * 4])
        engine.replies.append([self.final(tree)] * 4)
        events = []

        def writer(event):
            events.append(copy.deepcopy(event))
            if event["purpose"] == "REPAIR_FINAL_COMPLETION":
                engine.requests[-1].seeds[1] += 1
                tree.notebook["rewarded_notebook"]["call"]["seeds"][1] += 1

        with self.assertRaises(ValueError):
            self.append(tree, engine, on_raw=writer)
        self.assertEqual(sum(event["purpose"] == "REPAIR_FINAL_COMPLETION" for event in events), 1)
        self.assertEqual(events[-1]["purpose"], "REPAIR_FINAL_FAILED")
        self.assertEqual(events[-1]["call"]["seeds"], events[0]["call"]["seeds"])

    def test_real_pil_pixel_mutation_aborts_at_zero_or_one_draw_boundary(self) -> None:
        for phase in ("started", "completion"):
            originals = [Image.new("RGB", (16, 16), "white") for _ in self.actor.image_paths]
            for image in originals:
                self.addCleanup(image.close)
            self.images[self.actor.problem_id] = originals
            tree, engine = self.collect([[self.right] * 4])
            engine.replies.append([self.final(tree)] * 4)
            count = len(engine.requests)
            events = []

            def writer(event):
                events.append(copy.deepcopy(event))
                if (phase == "started" and event["purpose"] == "REPAIR_FINAL_STARTED"
                        or phase == "completion" and event["purpose"] == "REPAIR_FINAL_COMPLETION"):
                    originals[0].putpixel((0, 0), (1, 2, 3))

            with self.assertRaises(ValueError):
                self.append(tree, engine, on_raw=writer)
            self.assertEqual(sum(event["purpose"] == "REPAIR_FINAL_COMPLETION" for event in events),
                             0 if phase == "started" else 1)
            self.assertEqual(len(engine.requests), count + (phase == "completion"))
            self.assertFalse(any(group.kind == "DIAG" for group in tree.groups))
            self.assertEqual(events[-1]["purpose"], "REPAIR_FINAL_FAILED")

    def test_legacy_adjacent_dispatch_does_not_activate_without_explicit_repair_key(self) -> None:
        tree, engine = self.ready()
        episode.validate_update_binding(tree, tokenizer=engine.tokenizer, observation_actor=self.actor)
        self.assertEqual(episode.regrade_synthesis(tree, self.actor, self.verifier, engine.tokenizer)["version"], notes.VERSION)
        tree, engine = self.collect([[self.right] * 4])
        engine.replies.append([self.final(tree)] * 4)
        episode.append_episode_notebooks([tree], self.images, engine, self.verifier, self.prompts, self.cfg,
            actors={self.actor.problem_id: self.actor}, on_raw=self.events.append)
        plain = {key: value for key, value in self.prompts.items() if not key.startswith("repair_notebook_")}
        with patch.object(notes, "append", side_effect=AssertionError("unexpected repair dispatch")):
            with self.assertRaises(ValueError):
                append_rewarded_notebooks([], {}, engine, self.verifier, plain, self.cfg, actors={})


if __name__ == "__main__":
    unittest.main()
