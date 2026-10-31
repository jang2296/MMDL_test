"""Authored CPU physical/source/credit preparation; no optimizer or model training."""
from __future__ import annotations

import hashlib
import copy
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.notebook_rl import episode_notebook as episode
from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, parse_retry_sections, public_origin_inventory
from fine_tuning.notebook_rl.repair_training import VERSION, prepare_tree_credit
from fine_tuning.notebook_rl.rollout import _hash, build_tree
from fine_tuning.notebook_rl.schema import ProblemTree, Sample, from_json, to_json
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_repair_credit import joint
from fine_tuning.tests import test_notebook_rl_repair_notebook as notebook_fixture


class OriginalTokenizer:
    unk_token_id = -1

    def __init__(self) -> None:
        self.values: dict[int, str] = {}
        self.ids: dict[str, int] = {}

    def convert_tokens_to_ids(self, text: str) -> int:
        return 151645

    def encode(self, text: str, **kwargs) -> list[int]:
        return self.pieces([text[index:index + 16] for index in range(0, len(text), 16)])

    def pieces(self, values: list[str]) -> list[int]:
        for value in values:
            if value not in self.ids:
                token = 1000 + len(self.ids)
                self.ids[value] = token
                self.values[token] = value
        return [self.ids[value] for value in values]

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(self.values[token] for token in ids if token != 151645)


class OriginalEngine(notebook_fixture.Engine):
    def __init__(self, replies: list[list[str]]) -> None:
        super().__init__(replies)
        self.tokenizer: Any = OriginalTokenizer()

    def prepare(self, request):
        prepared = super().prepare(request)
        prepared.pixels = {"pixel_values": torch.tensor([image.getpixel((0, 0)) for image in request.images])}
        return prepared

    def sample(self, requests, *, on_completion=None):
        request = requests[0]
        if request.request_id.endswith(":FRESH_OBSERVATION"):
            self.all_requests.append(request)
            self.fresh_requests.append(request)
            ids = self.tokenizer.pieces(list(self.fresh_text)) + [151645]
            return [RolloutResult(request.request_id, self.prepare(request).token_ids,
                [Completion(ids, [-.1] * len(ids), "stop", self.fresh_text)])]
        self.all_requests.append(request)
        texts = self.replies[len(self.requests)]
        self.requests.append(request)
        completions = []
        for index, text in enumerate(texts):
            ids = self.tokenizer.pieces(list(text)) + [151645]
            completion = Completion(ids, [-.1] * len(ids), "stop", text)
            completions.append(completion)
            on_completion(request.request_id, index, request.seeds[index], completion)
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, completions)]


class RepairTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = notebook_fixture.RepairNotebookTests()
        self.fixture.setUp()
        self.actor, self.verifier = self.fixture.actor, self.fixture.verifier
        self.images = [Image.new("RGB", (16, 16), "white") for _ in self.actor.image_paths]
        for image in self.images:
            self.addCleanup(image.close)
        self.fixture.images[self.actor.problem_id] = self.images
        self.trace = fixture_trace(self.fixture.problem)

    def prepare(self, tree: ProblemTree, engine: OriginalEngine, **options) -> dict:
        return prepare_tree_credit(tree, self.actor, self.verifier, engine.tokenizer,
            preparation=engine, images=self.images, master_seed=self.fixture.cfg.master_seed, **options)

    def collect(self, replies: list[list[str]], **options) -> tuple[ProblemTree, OriginalEngine]:
        engine = OriginalEngine(replies)
        tree = build_tree(self.actor, self.images, engine, self.verifier, self.fixture.prompts,
            self.fixture.cfg, 0, "verified-any-visual-v1", on_raw=lambda event: None, **options)
        return tree, engine

    def attach(self, tree: ProblemTree, engine: OriginalEngine, finals: list[str] | None = None) -> None:
        if finals is not None:
            engine.replies.append(finals)
        self.fixture.append(tree, engine)

    def online(self, *, wrong_answer: bool = True, malformed: bool = False,
               visual: bool = False) -> tuple[ProblemTree, OriginalEngine]:
        raw = response(self.fixture.problem, {**self.trace, "cost": self.trace["cost"] - 1})
        body = self.fixture.right
        if wrong_answer:
            raw = raw.rsplit("Final answer:", 1)[0] + "Final answer: 999"
            body = self.fixture.wrong
        if visual:
            lines = raw.splitlines()
            index = next(index for index, line in enumerate(lines) if " = " in line)
            lines[index] = lines[index].split(" = ")[0] + " = 999"
            raw = "\n".join(lines)
        gid = f"{self.actor.problem_id}:v0:T1:{_hash('root')[:16]}"
        sid = min((f"{gid}:{index}" for index in range(4)),
                  key=lambda value: hashlib.sha256(f"{self.actor.problem_id}:0:{value}".encode()).digest())
        parent = Sample(sid, self.actor.problem_id, gid, "T1", 1, None, [1], [2], [-.1], "stop", raw, 0, 0)
        inventory = public_origin_inventory([parent], contract_version=ORIGIN_VERSION,
                                            response_contracts={sid: None}, parent_sample_id=sid)
        origin = next(row.ref_id for row in inventory.origins if row.rule == "COST_SUM")
        positive = joint(body, origin, proposed={"field": "cost", "value": self.trace["cost"]})
        abstain = joint(body, origin, intent="UNKNOWN")
        replies = [[raw] * 4, [positive, "ErrorCheck: {" if malformed else abstain]]
        if wrong_answer:
            empty = 'ErrorCheck: {"version":"repair-check-v1","claims":[],"uncertain_refs":[]}\nRevisedSolution:\n' + body
            replies.append([empty] * 2)
        return self.collect(replies)

    def test_all_wrong_answers_no_final_still_has_online_signal_and_keeps_source(self) -> None:
        tree, engine = self.online()
        self.attach(tree, engine)
        before = to_json(tree)
        receipt = self.prepare(tree, engine)
        self.assertEqual(receipt["version"], VERSION)
        self.assertTrue(receipt["admitted"])
        self.assertTrue(receipt["has_signal"])
        self.assertEqual(receipt["signal_components"], {"answer": False, "online": True, "final": False})
        self.assertEqual(tree.notebook["rewarded_notebook"]["status"], "NO_CORRECT_ANSWER")
        self.assertEqual([row["original_group_size"] for row in receipt["groups"]], [4, 2, 2])
        self.assertEqual([row["group_weight"] for row in receipt["groups"]], [1., 1., 1.])
        self.assertTrue(all(row["answer_reward"] == 0 for group in receipt["groups"] for row in group["samples"]))
        positive = receipt["groups"][1]["samples"][0]
        self.assertTrue(any(value > 0 for value in positive["applied_online_coefficients"]))
        self.assertEqual(positive["component_credit"]["original_group_size"], 2)
        self.assertEqual(to_json(tree), before)
        self.assertEqual(self.prepare(tree, engine), receipt)
        receipt["groups"][0]["members"].clear()
        self.assertEqual(len(tree.groups[0].samples), 4)

    def test_online_positive_cannot_become_final_note_learning_when_final_abstains(self) -> None:
        tree, engine = self.online(wrong_answer=False)
        self.attach(tree, engine, [self.fixture.final(tree)] * 4)
        receipt = self.prepare(tree, engine)
        self.assertTrue(receipt["signal_components"]["online"])
        self.assertFalse(receipt["signal_components"]["final"])
        self.assertTrue(all(value == 0 for group in receipt["groups"] for sample in group["samples"]
                            for value in sample["applied_final_coefficients"]))
        self.assertTrue(any(value > 0 for group in receipt["groups"] for sample in group["samples"]
                            for value in sample["applied_online_coefficients"]))

    def test_fixed_solution_reward_and_denominators_are_independent_of_fresh_raw(self) -> None:
        original = OriginalEngine.sample
        reports = []
        for raw in ("unverified authored reread", "malformed <|im_start|> different raw"):
            def sample(engine, requests, **options):
                if requests[0].request_id.endswith(":FRESH_OBSERVATION"):
                    engine.fresh_text = raw
                return original(engine, requests, **options)
            with patch.object(OriginalEngine, "sample", sample):
                tree, engine = self.online()
                self.attach(tree, engine)
                receipt = self.prepare(tree, engine)
            reports.append(receipt)
        self.assertNotEqual(reports[0]["source_bindings"]["source_audit"]["sha256"],
                            reports[1]["source_bindings"]["source_audit"]["sha256"])
        for first, second in zip(reports[0]["groups"], reports[1]["groups"], strict=True):
            for key in ("original_group_size", "group_weight"):
                self.assertEqual(first[key], second[key])
            for before, after in zip(first["samples"], second["samples"], strict=True):
                for key in ("answer_reward", "applied_online_coefficients", "applied_final_coefficients"):
                    self.assertEqual(before[key], after[key])
                credit_before, credit_after = before["component_credit"], after["component_credit"]
                if credit_before is None:
                    self.assertIsNone(credit_after)
                else:
                    self.assertEqual({key: value for key, value in credit_before.items() if key != "source_bindings"},
                                     {key: value for key, value in credit_after.items() if key != "source_bindings"})

    def test_complete_malformed_sibling_keeps_n2_with_continuation_and_full_kl_mask(self) -> None:
        tree, engine = self.online(malformed=True, visual=True)
        self.attach(tree, engine)
        receipt = self.prepare(tree, engine)
        self.assertTrue(receipt["source_bindings"]["source_audit"]["routing"]["complete_episode"])
        self.assertEqual(receipt["admission"], "VISION_ON")
        self.assertTrue(receipt["admitted"])
        rows = receipt["groups"][1]["samples"]
        self.assertTrue(any(rows[0]["applied_online_coefficients"]))
        self.assertFalse(any(rows[1]["applied_online_coefficients"]))
        self.assertFalse(any(rows[1]["answer_coefficients"]))
        self.assertIsNone(rows[1]["semantic_audit"])
        self.assertTrue(all(rows[1]["action_mask"]))
        self.assertEqual(len(rows[1]["action_mask"]), len(tree.groups[1].samples[1].completion_token_ids))
        self.assertEqual([group["kind"] for group in receipt["groups"]], ["T1", "T2", "T3"])
        unit = next(row for row in rows[0]["component_credit"]["unit_rows"] if row["semantic_score"] == 1)
        self.assertEqual(unit["mean"], .5)
        self.assertEqual(rows[1]["component_credit"]["original_group_size"], 2)

    def test_two_malformed_sources_keep_n2_zero_coefficients_after_t3_syntax_recovery(self) -> None:
        from fine_tuning.tests.test_notebook_rl_repair_verifier import joint as wrapped

        malformed = wrapped(self.fixture.wrong).replace('"uncertain_refs": []', '"uncertain_refs": [1]', 1)
        tree, engine = self.collect([[self.fixture.wrong] * 4, [malformed] * 2, [wrapped(self.fixture.right)] * 2])
        self.attach(tree, engine, [self.fixture.final(tree)] * 4)
        receipt = self.prepare(tree, engine)
        self.assertEqual([group["original_group_size"] for group in receipt["groups"]], [4, 2, 2, 4])
        for source, row in zip(tree.groups[1].samples, receipt["groups"][1]["samples"], strict=True):
            self.assertIsNone(source.result)
            self.assertEqual(source.reward, 0)
            self.assertEqual(row["component_credit"]["original_group_size"], 2)
            self.assertFalse(any(row["answer_coefficients"]))
            self.assertFalse(any(row["applied_online_coefficients"]))
            self.assertIsNone(row["semantic_audit"])
        self.assertFalse(receipt["signal_components"]["online"])
        self.assertFalse(receipt["signal_components"]["final"])
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["admission"], "UNRESOLVED")

    def test_r0_preserves_base_credits_and_denominators_but_zeros_applied_repair(self) -> None:
        tree, engine = self.online()
        self.attach(tree, engine)
        r1, r0 = self.prepare(tree, engine), self.prepare(tree, engine, notebook_reward_enabled=False)
        self.assertFalse(r0["has_signal"])
        for first, second in zip(r1["groups"], r0["groups"], strict=True):
            for a, b in zip(first["samples"], second["samples"], strict=True):
                self.assertEqual(a["component_credit"], b["component_credit"])
                self.assertEqual(a["semantic_audit"], b["semantic_audit"])
                self.assertEqual(a["online_coefficients"], b["online_coefficients"])
                self.assertEqual(a["answer_coefficients"], b["applied_answer_coefficients"])
                self.assertFalse(any(b["applied_online_coefficients"]))
                self.assertTrue(all(b["action_mask"]))

    def test_answer_prefix_zero_and_t1_original_tokens_keep_unscaled_centering(self) -> None:
        tree, engine = self.collect([[self.fixture.right, self.fixture.wrong, self.fixture.wrong, self.fixture.wrong],
                                    [joint(self.fixture.right, "foreign:r1", intent="UNKNOWN"),
                                     joint(self.fixture.wrong, "foreign:r1", intent="UNKNOWN")],
                                    [joint(self.fixture.right, "foreign:r1", intent="UNKNOWN")] * 2])
        self.attach(tree, engine, [self.fixture.final(tree)] * 4)
        receipt = self.prepare(tree, engine)
        self.assertTrue(all(value == .75 for value in receipt["groups"][0]["samples"][0]["answer_coefficients"]))
        group, raw_group = receipt["groups"][1], tree.groups[1]
        for row, original in zip(group["samples"], raw_group.samples, strict=True):
            sections = parse_retry_sections(original.text, contract_version="repair-check-v1")
            self.assertTrue(all(value == 0 for value in row["answer_coefficients"][:sections.solution_span.start]))
            self.assertTrue(all(value in (0., .5, -.5) for value in row["answer_coefficients"]))
            self.assertTrue(any(row["answer_coefficients"]))
        for row in receipt["groups"][-1]["samples"]:
            self.assertFalse(any(row["answer_coefficients"]))
            self.assertTrue(all(row["action_mask"]))
        control = self.prepare(tree, engine, notebook_reward_enabled=False)
        self.assertTrue(control["signal_components"]["answer"])
        self.assertEqual(control["groups"][1]["samples"][0]["applied_answer_coefficients"],
                         group["samples"][0]["answer_coefficients"])

    def test_final_signed_claims_have_no_answer_or_lesson_credit(self) -> None:
        tree, engine = self.online(wrong_answer=False)
        history = episode.solution_samples(tree)
        contracts = tree.notebook["repair_collector"]["response_contracts"]
        inventory = public_origin_inventory(history, contract_version=ORIGIN_VERSION, response_contracts=contracts)
        origin = next(row for row in inventory.origins if row.sample_id == history[4].parent_sample_id and row.rule == "COST_SUM")
        later = next(row for row in inventory.origins if row.sample_id == history[4].sample_id and row.rule == "COST_SUM")
        claim = {"origin_ref": origin.ref_id, "later_ref": later.ref_id, "relation": "CORRECTED", "rule": "COST_SUM",
                 "repaired_value": {"field": "cost", "value": self.trace["cost"]}}
        good = self.fixture.final(tree, claims=[claim], lesson_text="FREE_LESSON")
        bad = self.fixture.final(tree, claims=[claim | {"repaired_value": {"field": "cost", "value": 999}}])
        self.attach(tree, engine, [good, bad, self.fixture.final(tree), self.fixture.final(tree)])
        receipt = self.prepare(tree, engine)
        rows = receipt["groups"][-1]["samples"]
        self.assertTrue(any(value > 0 for value in rows[0]["final_coefficients"]))
        self.assertTrue(any(value < 0 for value in rows[1]["final_coefficients"]))
        self.assertFalse(any(rows[0]["answer_coefficients"]))
        start = good.index("FREE_LESSON")
        self.assertFalse(any(rows[0]["final_coefficients"][start:start + len("FREE_LESSON")]))
        changed = from_json(ProblemTree, to_json(tree))
        record = changed.notebook["rewarded_notebook"]
        record["grades"][0]["audit"]["metadata_valid"] = False
        record["grades_sha256"] = _hash(record["grades"])
        changed.notebook["model_notes"] = copy.deepcopy(record["grades"])
        with self.assertRaisesRegex(ValueError, "fresh final semantics"):
            self.prepare(changed, engine)

    def test_malformed_complete_final_keeps_actual_usable_group_and_original_n4(self) -> None:
        tree, engine = self.collect([[self.fixture.right] * 4])
        self.attach(tree, engine, ["{"] + [self.fixture.final(tree)] * 3)
        before = to_json(tree)
        receipt = self.prepare(tree, engine)
        self.assertTrue(tree.groups[-1].usable)
        rows = receipt["groups"][-1]["samples"]
        self.assertIsNone(rows[0]["semantic_audit"])
        self.assertEqual(rows[0]["component_credit"]["original_group_size"], 4)
        self.assertFalse(any(rows[0]["final_coefficients"]))
        self.assertEqual(to_json(tree), before)

    def test_unresolved_and_actual_skip_deny_applied_vectors_without_dropping_sources(self) -> None:
        unknown = "Solution:\nNo authored observations.\nFinal answer: " + self.fixture.problem.verifier.gold
        tree, engine = self.collect([[unknown] * 4])
        self.attach(tree, engine, [self.fixture.final(tree)] * 4)
        receipt = self.prepare(tree, engine)
        self.assertFalse(receipt["admitted"])
        self.assertIn("UNRESOLVED_ROUTING", receipt["admission_reasons"])
        # Recollect a complete root and an explicitly skipped retry, without rewriting a receipt.
        engine = OriginalEngine([[self.fixture.wrong] * 4])
        tree = build_tree(self.actor, self.images, engine, self.verifier, self.fixture.prompts, self.fixture.cfg,
            0, "verified-any-visual-v1", on_raw=lambda event: None, should_stop=lambda: len(engine.requests) >= 1)
        self.attach(tree, engine)
        receipt = self.prepare(tree, engine)
        self.assertFalse(receipt["admitted"])
        self.assertIn("SKIPPED_SOLUTION_BRANCH", receipt["admission_reasons"])
        self.assertFalse(receipt["has_signal"])
        self.assertEqual(len(receipt["groups"][0]["members"]), 4)

    def test_incomplete_final_zeros_applied_vectors_but_keeps_online_evidence(self) -> None:
        tree, engine = self.online(wrong_answer=False)
        engine.block_final = True
        self.attach(tree, engine)
        receipt = self.prepare(tree, engine)
        self.assertFalse(receipt["admitted"])
        self.assertIn("INCOMPLETE_FINAL_COLLECTION", receipt["admission_reasons"])
        self.assertFalse(receipt["has_signal"])
        row = receipt["groups"][1]["samples"][0]
        self.assertTrue(any(row["online_coefficients"]))
        self.assertFalse(any(row["applied_online_coefficients"]))

    def test_raw_result_parent_contract_and_physical_mutations_rejected(self) -> None:
        tree, engine = self.online()
        self.attach(tree, engine)
        original = to_json(tree)
        for field in ("raw", "result", "parent", "contract", "missing_raw", "incomplete_call"):
            changed = from_json(ProblemTree, original)
            if field == "raw":
                changed.groups[1].samples[0].text += "!"
            elif field == "result":
                row = changed.groups[0].samples[0]
                row.result = replace(row.result, extracted="FORGED")
            elif field == "parent":
                changed.groups[1].samples[0].parent_sample_id = changed.groups[0].samples[0].sample_id + "wrong"
            elif field == "contract":
                changed.notebook["repair_collector"]["response_contracts"][changed.groups[1].samples[0].sample_id] = None
            elif field == "missing_raw":
                changed.groups[1].samples.pop()
            else:
                changed.notebook["repair_collector"]["calls"][1]["completions"].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare(changed, engine)
        for key, value in (("master_seed", 124), ("images", []), ("preparation", None),
                           ("notebook_reward_enabled", 1)):
            args: dict[str, Any] = {"preparation": engine, "images": self.images, "master_seed": 123}
            args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_tree_credit(tree, self.actor, self.verifier, engine.tokenizer, **args)
        self.images[0].putpixel((0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            self.prepare(tree, engine)
        self.assertEqual(to_json(tree), original)

    def test_callback_mutation_of_original_tree_or_pixels_is_detected(self) -> None:
        for field in ("tree", "pixels"):
            self.images[0].putpixel((0, 0), (255, 255, 255))
            tree, engine = self.online()
            self.attach(tree, engine)
            grade = self.verifier.grade
            mutated = False

            def callback(*args, **kwargs):
                nonlocal mutated
                result = grade(*args, **kwargs)
                if not mutated:
                    mutated = True
                    if field == "tree":
                        tree.groups[0].samples[0].text += "MUTATED"
                    else:
                        self.images[0].putpixel((0, 0), (0, 0, 0))
                return result

            with self.subTest(field=field), patch.object(self.verifier, "grade", side_effect=callback):
                with self.assertRaises(ValueError):
                    self.prepare(tree, engine)


if __name__ == "__main__":
    unittest.main()
