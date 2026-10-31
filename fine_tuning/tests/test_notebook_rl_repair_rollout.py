"""CPU scripted collector fixtures; persistence callbacks are not a durable-storage claim."""
from __future__ import annotations

import copy
import hashlib
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.generators import graph
from fine_tuning.notebook_rl.prompts import REPAIR_OPERATION_PATCH_INSTRUCTION, load_prompts, repair_retry_text
from fine_tuning.notebook_rl.repair_credit import operation_patch
from fine_tuning.notebook_rl.reasoning_trace import VERSION as TRACE_VERSION
from fine_tuning.notebook_rl.repair_contract import ORIGIN_VERSION, VERSION, public_origin_inventory
from fine_tuning.notebook_rl.rollout import (RolloutConfig, build_tree, seeds, select_failures,
                                            select_repair_parents)
from fine_tuning.notebook_rl.schema import Sample, from_json, ProblemTree, to_json
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_reasoning_trace import fixture_trace, response
from fine_tuning.tests.test_notebook_rl_repair_verifier import joint


class Tokenizer:
    unk_token_id = -1

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.values: dict[int, str] = {}
        self.padding = 1

    def convert_tokens_to_ids(self, token: str) -> int:
        return 151645

    def encode(self, text: str, **kwargs) -> list[int]:
        self.encoded.append(text)
        token = 1000 + len(self.values)
        self.values[token] = text
        return [token] * self.padding

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(self.values[token] for token in ids if token != 151645)


class Engine:
    policy_version = 0

    def __init__(self, replies: list[list[str]]) -> None:
        self.replies = replies
        self.requests: list = []
        self.fresh_requests: list = []
        self.all_requests: list = []
        self.fresh_text = "authored unverified reread"
        self.fresh_finish = "stop"
        self.tokenizer = Tokenizer()
        self.root_tokens = [4, 5]
        self.interrupt_after: int | None = None
        self.mismatch = False

    def prepare(self, request):
        return SimpleNamespace(token_ids=list(request.prompt_token_ids or self.root_tokens), pixels={},
            images=[SimpleNamespace(size=(16, 16)) for _ in request.images], image_key="fixture-pixels", image_spans=())

    def sample(self, requests, *, on_completion=None):
        request = requests[0]
        self.all_requests.append(request)
        if request.request_id.endswith(":FRESH_OBSERVATION"):
            self.fresh_requests.append(request)
            token = 1000 + len(self.tokenizer.values)
            self.tokenizer.values[token] = self.fresh_text
            return [RolloutResult(request.request_id, self.prepare(request).token_ids,
                [Completion([token, 151645], [-.1, -.2], self.fresh_finish, self.fresh_text)])]
        texts = self.replies[len(self.requests)]
        self.requests.append(request)
        completions = []
        for index, text in enumerate(texts):
            token = 1000 + len(self.tokenizer.values)
            self.tokenizer.values[token] = text
            completion = Completion([token, 151645], [-.1, -.2], "stop", text)
            completions.append(completion)
            if on_completion is not None:
                on_completion(request.request_id, index, request.seeds[index], completion)
            if self.interrupt_after == index + 1:
                raise KeyboardInterrupt("fixture interruption")
        if self.mismatch:
            completions[0] = replace(completions[0], text="changed after completion callback")
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, completions)]


class RepairRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = graph.generate(0, 3407, "train", "obs")
        self.actor = self.problem.actor
        self.verifier = Verifier.__new__(Verifier)
        self.verifier.actors, self.verifier.items = ({self.actor.problem_id: self.actor},
                                                  {self.actor.problem_id: self.problem.verifier})
        self.prompts = load_prompts(Path(__file__).parents[1] / "notebook_rl/prompts_v1.yaml") | {
            "solution_trace_contract": TRACE_VERSION}
        self.cfg = RolloutConfig({1: 4, 2: 2, 3: 2}, {1: 1, 2: 1}, 3, {1: 1024, 2: 1024, 3: 1024},
                                 24576, False, 4, diag_max_new_tokens=8192, retry_contract=VERSION)
        self.right = response(self.problem, fixture_trace(self.problem))
        self.wrong = self.right.rsplit("Final answer:", 1)[0] + "Final answer: 999"
        self.events: list[dict] = []

    def tree(self, engine: Engine, **kwargs) -> ProblemTree:
        return build_tree(self.actor, [object()] * len(self.actor.image_paths), engine, self.verifier,
            self.prompts, self.cfg, 0, "verified-any-visual-v1", on_raw=kwargs.pop("on_raw", self.events.append), **kwargs)

    def test_fixed_stages_prefix_seeds_public_inventory_and_raw_before_grade(self) -> None:
        engine = Engine([[self.wrong] * 4, [joint(self.wrong)] * 2, [joint(self.right)] * 2])
        original = self.verifier.grade

        def grade(problem_id, text, finish, mode, **options):
            self.assertIn(sum(event["purpose"] == "NATIVE_SOLUTION_COMPLETION" for event in self.events), (4, 6, 8))
            return original(problem_id, text, finish, mode, **options)

        with patch.object(self.verifier, "grade", side_effect=grade):
            tree = self.tree(engine)
        self.assertEqual([len(group.samples) for group in tree.groups], [4, 2, 2])
        self.assertEqual([request.max_new_tokens for request in engine.requests], [1024] * 3)
        self.assertEqual([request.n for request in engine.requests], [4, 2, 2])
        record = tree.notebook["repair_collector"]
        self.assertEqual([row["key"] for row in record["selections"]],
                         [self.actor.problem_id + ":0", self.actor.problem_id + ":0:2"])
        prior: dict[str, Sample] = {}
        for group, request, call in zip(tree.groups, engine.requests, record["calls"], strict=True):
            parent_id = group.samples[0].parent_sample_id
            branch = parent_id if parent_id is not None else "root"
            self.assertEqual(request.seeds, seeds(self.actor.problem_id, 0, group.samples[0].turn,
                                                group.kind + ":" + branch, len(group.samples)))
            self.assertEqual(len({tuple(sample.prompt_token_ids) for sample in group.samples}), 1)
            if parent_id is not None:
                parent = prior[parent_id]
                prefix = parent.prompt_token_ids + parent.completion_token_ids
                self.assertEqual(request.prompt_token_ids[:len(prefix)], prefix)
                inventory = call["ancestor_inventory"]
                source_ids = {source["sample_id"] for source in inventory["sources"]}
                self.assertIn(parent_id, source_ids)
                self.assertFalse(source_ids & {sample.sample_id for sample in group.samples})
                self.assertNotIn("graph_world", call["feedback"])
                self.assertNotIn('"expected"', call["feedback"])
                self.assertIn("ErrorCheck:", call["feedback"])
                self.assertIn("RevisedSolution:", call["feedback"])
            prior.update({sample.sample_id: sample for sample in group.samples})
        self.assertEqual(len(self.events), 10)
        self.assertFalse(record["loss_integration_ready"])
        self.assertEqual(record["full_history_admission"], "UNRESOLVED")
        self.assertEqual(from_json(ProblemTree, to_json(tree)), tree)

    def test_correct_known_violation_retries_but_correct_unknown_does_not(self) -> None:
        trace = fixture_trace(self.problem)
        trace["weights"][0] += 1
        contradicted = response(self.problem, trace)
        engine = Engine([[self.right, contradicted, self.right, self.right], [joint(self.right)] * 2])
        tree = self.tree(engine)
        parent = tree.groups[0].samples[1]
        self.assertEqual((parent.result.status, parent.reward), ("CORRECT", 1.))
        self.assertEqual(tree.groups[1].samples[0].parent_sample_id, parent.sample_id)
        record = tree.notebook["repair_collector"]
        self.assertEqual(record["selections"][0]["process_contradictions"], [parent.sample_id])
        self.assertTrue(any(row["process_assessment"] == "ANSWER_CORRECT_PROCESS_CONTRADICTED"
                            for row in record["sample_bindings"]))
        self.assertIn("previous answer was verified CORRECT", record["calls"][1]["feedback"])
        unknown = "Solution:\nNo structured trace.\nFinal answer: " + self.problem.verifier.gold
        tree = self.tree(Engine([[unknown] * 4]))
        self.assertEqual(len(tree.groups), 1)
        self.assertEqual(tree.notebook["repair_collector"]["selections"][0]["process_contradictions"], [])

    def test_union_selector_keeps_original_ordering_and_rejects_invalid_process_ids(self) -> None:
        statuses = ["INCORRECT", "CORRECT", "INVALID_NO_FINAL", "CORRECT", "INCORRECT"]
        samples = []
        for index, status in enumerate(statuses):
            text = self.right if status == "CORRECT" else self.wrong if status == "INCORRECT" else "No answer"
            result = self.verifier.grade(self.actor.problem_id, text, "stop", "obs")
            samples.append(Sample(str(index), self.actor.problem_id, "T1", "T1", 1, None,
                [1], [2], [-.1], "stop", text, index, 0, result))
        contradictions = {"1"}
        marked = [replace(sample, result=replace(sample.result, status="INCORRECT", reward_answer=0.))
                  if sample.sample_id in contradictions else sample for sample in samples]
        expected = [sample.sample_id for sample in select_failures(marked, 4, "mixed")]
        actual = select_failures(samples, 4, "mixed", process_contradictions=contradictions)
        self.assertEqual([sample.sample_id for sample in actual], expected)
        invalid_sets: tuple[Any, ...] = ({"missing"}, {"0"}, {True}, ["1"], frozenset({"1"}))
        for invalid in invalid_sets:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_failures(samples, 1, "mixed", process_contradictions=invalid)

    def test_malformed_retry_keeps_all_siblings_and_retries_without_fabricated_result(self) -> None:
        engine = Engine([[self.wrong] * 4, ['ErrorCheck: {', joint(self.right)], [joint(self.right)] * 2])
        with patch.object(self.verifier, "cause", wraps=self.verifier.cause) as cause:
            tree = self.tree(engine)
        self.assertEqual(len(tree.groups), 3)
        group = tree.groups[1]
        self.assertEqual(len(group.samples), 2)
        self.assertTrue(group.usable)
        self.assertIsNone(group.reason)
        self.assertIsNone(group.samples[0].result)
        self.assertEqual(group.samples[0].reward, 0.)
        self.assertEqual(group.samples[1].result.status, "CORRECT")
        self.assertEqual(cause.call_count, 7)
        self.assertEqual(len(self.events), 10)
        record = tree.notebook["repair_collector"]
        self.assertEqual(record["missing_result_ids"], [group.samples[0].sample_id])
        self.assertEqual(record["quarantined_groups"], [])
        self.assertEqual(record["selections"][1]["selected_ids"], [group.samples[0].sample_id])
        self.assertEqual(record["invalid_responses"][0]["raw_sha256"],
                         hashlib.sha256(group.samples[0].text.encode()).hexdigest())

    def test_valid_incorrect_sibling_reaches_t3_with_original_prefix_n_and_seeds(self) -> None:
        malformed = joint(self.wrong).replace('"uncertain_refs": []', '"uncertain_refs": [1]', 1)
        engine = Engine([[self.wrong] * 4, [joint(self.wrong), malformed], [joint(self.right)] * 2])
        tree = self.tree(engine)
        self.assertEqual([len(group.samples) for group in tree.groups], [4, 2, 2])
        parent = tree.groups[1].samples[0]
        invalid = tree.groups[1].samples[1]
        self.assertEqual(parent.result.status, "INCORRECT")
        self.assertIsNone(invalid.result)
        self.assertEqual(invalid.reward, 0.)
        self.assertEqual(tree.notebook["repair_collector"]["invalid_responses"][0]["code"], "UNCERTAIN_REFS")
        self.assertEqual(tree.notebook["repair_collector"]["selections"][1]["selected_ids"], [parent.sample_id])
        self.assertEqual(tree.groups[2].samples[0].parent_sample_id, parent.sample_id)
        prefix = parent.prompt_token_ids + parent.completion_token_ids
        self.assertEqual(engine.requests[2].prompt_token_ids[:len(prefix)], prefix)
        self.assertEqual(engine.requests[2].seeds, seeds(self.actor.problem_id, 0, 3,
            "T3:" + parent.sample_id, 2))
        self.assertEqual(engine.requests[2].max_new_tokens, 1024)
        self.assertEqual(len(self.events), 10)

    def test_all_malformed_retry_reaches_bounded_t3_with_sha_order_and_not_judged_feedback(self) -> None:
        malformed = joint(self.wrong).replace('"uncertain_refs": []', '"uncertain_refs": [1]', 1)
        engine = Engine([[self.wrong] * 4, [malformed] * 2, [joint(self.wrong)] * 2])
        tree = self.tree(engine)
        self.assertEqual(len(tree.groups), 3)
        self.assertTrue(tree.groups[1].usable)
        self.assertTrue(all(row.result is None and row.reward == 0 for row in tree.groups[1].samples))
        record = tree.notebook["repair_collector"]
        selection = record["selections"][1]
        parent = min(tree.groups[1].samples,
            key=lambda row: hashlib.sha256(f'{selection["key"]}:{row.sample_id}'.encode()).digest())
        self.assertEqual(selection["selected_ids"], [parent.sample_id])
        self.assertEqual(tree.groups[2].samples[0].parent_sample_id, parent.sample_id)
        prefix = parent.prompt_token_ids + parent.completion_token_ids
        self.assertEqual(engine.requests[2].prompt_token_ids[:len(prefix)], prefix)
        self.assertEqual(engine.requests[2].seeds, seeds(self.actor.problem_id, 0, 3, "T3:" + parent.sample_id, 2))
        self.assertEqual((engine.requests[2].n, engine.requests[2].max_new_tokens), (2, 1024))
        self.assertIn("grammar: UNCERTAIN_REFS", record["calls"][2]["feedback"])
        self.assertIn("Answer correctness: NOT JUDGED", record["calls"][2]["feedback"])
        self.assertNotIn("previous answer was verified CORRECT", record["calls"][2]["feedback"])
        selected, _ = select_repair_parents(tree.groups[1].samples, 2, selection["key"],
            native_audits=record["native_audits"], invalid_responses=record["invalid_responses"],
            response_contracts=record["response_contracts"])
        self.assertEqual([row.sample_id for row in selected], [parent.sample_id])
        native, _ = select_repair_parents(tree.groups[0].samples, 2, record["selections"][0]["key"],
            native_audits=record["native_audits"], invalid_responses=record["invalid_responses"],
            response_contracts=record["response_contracts"])
        self.assertEqual(len(native), 2)
        self.assertFalse(any(row.result is not None and row.result.status == "CORRECT"
                             for group in tree.groups for row in group.samples))

    def test_malformed_fallback_rechecks_code_raw_hash_and_missing_valid_grade(self) -> None:
        tree = self.tree(Engine([[self.wrong] * 4, ["ErrorCheck: {"] * 2, [joint(self.right)] * 2]))
        for mutation in ("code", "hash", "raw", "missing_valid_grade"):
            changed = from_json(ProblemTree, to_json(tree))
            record = changed.notebook["repair_collector"]
            frontier = changed.groups[1].samples
            if mutation == "code":
                record["invalid_responses"][0]["code"] = "UNCERTAIN_REFS"
            elif mutation == "hash":
                record["invalid_responses"][0]["raw_sha256"] = "a" * 64
            elif mutation == "raw":
                frontier[0].text += "changed"
            else:
                frontier[0].text = joint(self.right)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select_repair_parents(frontier, 1, "test", native_audits=record["native_audits"],
                    invalid_responses=record["invalid_responses"], response_contracts=record["response_contracts"])

    def test_length_malformed_completion_is_not_judged_and_stop_limits_still_apply(self) -> None:
        class LengthEngine(Engine):
            def sample(self, requests, *, on_completion=None):
                def completed(request_id, index, seed, completion):
                    on_completion(request_id, index, seed, replace(completion, finish_reason="length")
                        if ":T2:" in request_id else completion)
                output = super().sample(requests, on_completion=completed)
                if ":T2:" in requests[0].request_id:
                    output = [replace(row, completions=[replace(item, finish_reason="length")
                        for item in row.completions]) for row in output]
                return output

        engine: Engine = LengthEngine([[self.wrong] * 4, ["ErrorCheck: {"] * 2, [joint(self.right)] * 2])
        tree = self.tree(engine)
        self.assertTrue(all(row.finish_reason == "length" and row.result is None for row in tree.groups[1].samples))
        self.assertIn("output limit and may be incomplete", tree.notebook["repair_collector"]["calls"][2]["feedback"])
        for stop in ("budget", "context"):
            self.events.clear()
            engine = Engine([[self.wrong] * 4, ["ErrorCheck: {"] * 2])
            def persist(event):
                self.events.append(event)
                if stop == "context" and len(self.events) == 8:
                    engine.tokenizer.padding = 25000
            tree = self.tree(engine, on_raw=persist, should_stop=lambda: stop == "budget" and len(self.events) >= 8)
            self.assertEqual(len(engine.requests), 2)
            self.assertEqual(tree.notebook["skipped_branches"][-1]["status"],
                             "BUDGET_STOP" if stop == "budget" else "CONTEXT_BLOCKED")

    def test_callback_failure_partial_interrupt_and_stream_mismatch_preserve_raw(self) -> None:
        for failure in ("callback", "interrupt", "mismatch"):
            events = []
            engine = Engine([[self.wrong] * 4])
            engine.interrupt_after = 1 if failure == "interrupt" else None
            engine.mismatch = failure == "mismatch"

            def writer(event):
                events.append(copy.deepcopy(event))
                if failure == "callback":
                    raise ValueError("CONTEXT_BLOCKED: callback writer failed")

            with self.subTest(failure=failure), patch.object(self.verifier, "grade") as grade:
                with self.assertRaises(KeyboardInterrupt if failure == "interrupt" else ValueError):
                    self.tree(engine, on_raw=writer)
                grade.assert_not_called()
            self.assertEqual(len(events), 4 if failure == "mismatch" else 1)
            self.assertEqual(events[0]["completion"]["text"], self.wrong)

    def test_missing_logprobs_and_raw_decode_mismatch_remain_integrity_failures(self) -> None:
        for corruption in ("missing_logprobs", "decode"):
            self.events.clear()

            class CorruptEngine(Engine):
                def sample(self, requests, *, on_completion=None):
                    def completed(request_id, index, seed, completion):
                        damaged = (replace(completion, logprobs=completion.logprobs[:-1])
                            if corruption == "missing_logprobs" else replace(completion, text="changed raw"))
                        on_completion(request_id, index, seed, damaged)
                    return super().sample(requests, on_completion=completed)

            with self.subTest(corruption=corruption), patch.object(self.verifier, "grade") as grade:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.tree(CorruptEngine([[self.right] * 4]))
                grade.assert_not_called()
            self.assertEqual(self.events, [])

    def test_native_blocking_result_still_rejects_whole_group(self) -> None:
        original = self.verifier.grade

        def grade(*args, **kwargs):
            return replace(original(*args, **kwargs), status="SYSTEM_ERROR", reward_answer=0.)

        with patch.object(self.verifier, "grade", side_effect=grade):
            tree = self.tree(Engine([[self.right] * 4]))
        self.assertEqual(len(tree.groups), 1)
        self.assertFalse(tree.groups[0].usable)
        self.assertEqual(tree.groups[0].reason, "SYSTEM_ERROR")
        self.assertTrue(all(row.reward == 0 for row in tree.groups[0].samples))

    def test_callback_payload_and_group_payload_are_detached(self) -> None:
        def writer(event):
            self.events.append(copy.deepcopy(event))
            if event["purpose"] == "NATIVE_SOLUTION_COMPLETION":
                event["completion"]["text"] = "callback poisoning"
                event["prompt_token_ids"].append(999)

        def group_writer(group):
            group.samples[0].text = "changed external group"

        tree = self.tree(Engine([[self.wrong] * 4, [joint(self.right)] * 2]),
                         on_raw=writer, on_group=group_writer)
        self.assertEqual(tree.groups[0].samples[0].text, self.wrong)
        self.assertNotIn(999, tree.groups[0].samples[0].prompt_token_ids)

    def test_external_config_and_prompt_mutation_cannot_change_fixed_retry_contract(self) -> None:
        def writer(event):
            self.events.append(copy.deepcopy(event))
            if len(self.events) == 1:
                self.cfg.n[2] = 3
                self.cfg.branch[1] = 0
                self.cfg.max_new_tokens[2] = 2048
                self.prompts["observation_instruction"]["graph"] = "POISONED_EXTERNAL_PROMPT"

        engine = Engine([[self.wrong] * 4, [joint(self.right)] * 2])
        tree = self.tree(engine, on_raw=writer)
        self.assertEqual([len(group.samples) for group in tree.groups], [4, 2])
        self.assertEqual([request.max_new_tokens for request in engine.requests], [1024, 1024])
        self.assertEqual([request.n for request in engine.requests], [4, 2])
        self.assertNotIn("POISONED_EXTERNAL_PROMPT", tree.notebook["repair_collector"]["calls"][1]["feedback"])
        for group, call in zip(tree.groups, tree.notebook["repair_collector"]["calls"], strict=True):
            self.assertEqual(call["seeds"], [row.seed for row in group.samples])

    def test_callback_request_mutation_aborts_after_preserving_first_raw_before_next_draw(self) -> None:
        from PIL import Image

        for field in ("seeds", "n", "cap", "messages", "images", "pixels"):
            self.events.clear()
            image = Image.new("RGB", (16, 16), "white")
            engine = Engine([[self.right] * 4])

            def writer(event):
                self.events.append(copy.deepcopy(event))
                request = engine.requests[-1]
                if field == "seeds":
                    request.seeds[1] = (request.seeds[1] + 1) % (2**31)
                elif field == "n":
                    object.__setattr__(request, "n", 3)
                elif field == "cap":
                    object.__setattr__(request, "max_new_tokens", 2048)
                elif field == "messages":
                    request.messages[0]["content"].append({"type": "text", "text": "mutated"})
                elif field == "images":
                    request.images.append(object())
                else:
                    image.putpixel((0, 0), (0, 0, 0))

            with self.subTest(field=field), patch.object(self.verifier, "grade") as grade:
                with self.assertRaisesRegex(ValueError, "request state"):
                    build_tree(self.actor, [image] * len(self.actor.image_paths), engine, self.verifier,
                               self.prompts, self.cfg, 0, "verified-any-visual-v1", on_raw=writer)
                grade.assert_not_called()
            self.assertEqual(len(self.events), 1)
            self.assertEqual(self.events[0]["completion"]["text"], self.right)
            self.assertEqual(len(engine.requests), 1)

    def test_engine_preparation_mutation_is_rejected_before_sampling(self) -> None:
        class MutatingEngine(Engine):
            def prepare(self, request):
                prepared = super().prepare(request)
                request.seeds[0] = (request.seeds[0] + 1) % (2**31)
                return prepared

        engine = MutatingEngine([[self.right] * 4])
        with self.assertRaisesRegex(ValueError, "request state"):
            self.tree(engine)
        self.assertEqual(engine.requests, [])
        self.assertEqual(self.events, [])

    def test_changed_second_preparation_is_rejected_before_first_draw(self) -> None:
        for field in ("tokens", "images"):
            self.events.clear()

            class MutatingPreparedEngine(Engine):
                preparations = 0

                def prepare(self, request):
                    prepared = super().prepare(request)
                    self.preparations += 1
                    if self.preparations == 2:
                        if field == "tokens":
                            prepared.token_ids.append(999)
                        else:
                            prepared.images[0].size = (32, 16)
                    return prepared

            engine = MutatingPreparedEngine([[self.right] * 4])
            with self.subTest(field=field), patch.object(self.verifier, "grade") as grade:
                with self.assertRaisesRegex(ValueError, "prepared repair token or image"):
                    self.tree(engine)
                grade.assert_not_called()
            self.assertEqual(engine.preparations, 2)
            self.assertEqual(engine.requests, [])
            self.assertEqual(self.events, [])

    def test_context_and_budget_skips_keep_closed_root_and_no_fake_success(self) -> None:
        engine = Engine([[self.wrong] * 4])
        engine.tokenizer.padding = 25000
        tree = self.tree(engine)
        self.assertEqual(len(tree.groups), 1)
        self.assertEqual(tree.notebook["skipped_branches"][0]["status"], "CONTEXT_BLOCKED")
        self.events.clear()
        tree = self.tree(Engine([[self.wrong] * 4]), should_stop=lambda: len(self.events) >= 4)
        self.assertEqual(len(tree.groups), 1)
        self.assertEqual(tree.notebook["skipped_branches"][0]["status"], "BUDGET_STOP")
        self.assertNotIn("diagnosis_candidate", tree.notebook)

    def test_explicit_config_native_prompt_and_raw_callback_required(self) -> None:
        for cfg in (replace(self.cfg, retry_contract="other"), replace(self.cfg, n={1: 8, 2: 2, 3: 2}),
                    replace(self.cfg, diag=True), replace(self.cfg, final_notebook=True),
                    replace(self.cfg, max_new_tokens={1: 1024, 2: 2048, 3: 1024}),
                    replace(self.cfg, diag_max_new_tokens=2048)):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                cfg.validate()
        for raw_callback, prompts in ((None, self.prompts), (self.events.append, load_prompts(
                Path(__file__).parents[1] / "notebook_rl/prompts_v1.yaml"))):
            with self.subTest(callback=raw_callback), self.assertRaises(ValueError):
                build_tree(self.actor, [object()] * len(self.actor.image_paths), Engine([[self.right] * 4]),
                    self.verifier, prompts, self.cfg, 0, "verified-any-visual-v1", on_raw=raw_callback)
        protocol: dict = {"algorithms": ["ALG-NB3"], "context_limit": 24576, "reward": {"visual_lambda": 0.},
            "rollout": {"n": self.cfg.n, "branch": self.cfg.branch, "max_new_tokens": self.cfg.max_new_tokens,
                "diag": {"enabled_in": [], "n": 4, "max_new_tokens": 8192},
                "oracle_counterfactual": False, "self_perception_probe": False}}
        self.assertIsNone(RolloutConfig.from_protocol(protocol, "ALG-NB3").retry_contract)
        protocol["rollout"]["retry_contract"] = VERSION
        self.assertEqual(RolloutConfig.from_protocol(protocol, "ALG-NB3").retry_contract, VERSION)
        protocol["rollout"]["retry_contract"] = "other"
        with self.assertRaises(ValueError):
            RolloutConfig.from_protocol(protocol, "ALG-NB3")

    def test_prompt_uses_public_origins_and_wrong_declared_version_is_invalid_member(self) -> None:
        sample = Sample("parent", self.actor.problem_id, "T1", "T1", 1, None, [1], [2], [-.1],
                        "stop", self.wrong, 3407, 0)
        inventory = public_origin_inventory([sample], contract_version=ORIGIN_VERSION, response_contracts={"parent": None},
                                            parent_sample_id="parent")
        text = repair_retry_text("INCORRECT", "999", self.actor, self.prompts, inventory)
        self.assertIn("OBSERVATION_MATCH", text)
        self.assertIn("operation_patch", text)
        self.assertIn("uncertain_refs", text)
        self.assertNotIn("graph_world", text)
        wrong_contract = joint(self.right).replace('"version": "repair-check-v1"', '"version": "other"', 1)
        tree = self.tree(Engine([[self.wrong] * 4, [wrong_contract, joint(self.right)], [joint(self.right)] * 2]))
        self.assertTrue(tree.groups[1].usable)
        self.assertIsNone(tree.groups[1].samples[0].result)
        self.assertEqual(tree.notebook["repair_collector"]["invalid_responses"][0]["code"], "INVALID_FIELD")
        self.assertEqual(len(self.events), 10)

    def test_one_reread_is_between_t1_and_all_retries_and_has_no_training_members(self) -> None:
        from fine_tuning.notebook_rl import fresh_observation as fresh
        engine = Engine([[self.wrong] * 4, [joint(self.wrong)] * 2, [joint(self.right)] * 2])
        engine.fresh_text = "malformed raw\r\n<|im_start|>🙂"
        engine.fresh_finish = "length"
        tree = self.tree(engine)
        record = tree.notebook["repair_collector"]
        self.assertEqual([request.n for request in engine.all_requests], [4, 1, 2, 2])
        self.assertEqual(record["fresh_observation_calls"], 1)
        self.assertEqual(len(engine.fresh_requests), 1)
        self.assertEqual(record["fresh_observation"]["completion"]["text"], engine.fresh_text)
        self.assertEqual([event["phase"] for event in record["fresh_observation_events"]], ["STARTED", "COMPLETED"])
        quoted = fresh.quoted_retry_context(record["fresh_observation"])
        self.assertNotIn("designated", quoted)
        for call in record["calls"][1:]:
            self.assertTrue(call["feedback"].endswith("\n" + quoted))
            self.assertEqual(call["feedback"].count("<unverified_fresh_observation>"), 1)
        self.assertEqual(sum(len(group.samples) for group in tree.groups), 8)
        self.assertFalse(any(sample.text == engine.fresh_text for group in tree.groups for sample in group.samples))
        self.assertEqual([event["purpose"] for event in self.events],
            ["NATIVE_SOLUTION_COMPLETION"] * 4 + ["RETRY_FRESH_OBSERVATION"] * 2
            + ["NATIVE_SOLUTION_COMPLETION"] * 4)

    def test_no_eligible_parent_or_budget_before_first_reread_calls_zero(self) -> None:
        for stop in (False, True):
            self.events.clear()
            engine = Engine([[self.wrong if stop else self.right] * 4])
            tree = self.tree(engine, should_stop=lambda: stop and len(self.events) >= 4)
            self.assertEqual(tree.notebook["repair_collector"]["fresh_observation_calls"], 0)
            self.assertEqual(engine.fresh_requests, [])

    def test_reread_writer_context_failure_is_fatal_and_cannot_become_skip(self) -> None:
        engine = Engine([[self.wrong] * 4])
        events = []
        def writer(event):
            events.append(copy.deepcopy(event))
            if event["purpose"] == "RETRY_FRESH_OBSERVATION":
                raise ValueError("CONTEXT_BLOCKED: actual durable writer failed")
        with self.assertRaisesRegex(ValueError, "durable writer failed"):
            self.tree(engine, on_raw=writer)
        self.assertEqual(events[-1]["event"]["phase"], "STARTED")
        self.assertEqual(engine.fresh_requests, [])
        self.assertEqual(len(engine.requests), 1)

    def test_reread_callback_mutation_and_sampling_failure_preserve_started_only(self) -> None:
        for mode in ("mutate", "sample"):
            engine = Engine([[self.wrong] * 4])
            events = []
            original = engine.sample
            def sample(requests, **kwargs):
                if mode == "sample" and requests[0].request_id.endswith(":FRESH_OBSERVATION"):
                    raise RuntimeError("authored fresh sampler failure")
                return original(requests, **kwargs)
            def writer(event):
                events.append(copy.deepcopy(event))
                if mode == "mutate" and event["purpose"] == "RETRY_FRESH_OBSERVATION":
                    event["event"]["messages"].append({"role": "system", "content": "changed"})
            with self.subTest(mode=mode), patch.object(engine, "sample", side_effect=sample), \
                    self.assertRaises((RuntimeError, ValueError)):
                self.tree(engine, on_raw=writer)
            self.assertEqual([event["event"]["phase"] for event in events
                if event["purpose"] == "RETRY_FRESH_OBSERVATION"], ["STARTED"])

    def test_public_patch_instruction_matches_selected_row_slice_and_scalar_conventions(self) -> None:
        parent = Sample("parent", self.actor.problem_id, "T1", "T1", 1, None, [1], [2], [-.1],
                        "stop", self.wrong, 3407, 0)
        inventory = public_origin_inventory([parent], contract_version=ORIGIN_VERSION,
                                            response_contracts={"parent": None}, parent_sample_id="parent")
        text = repair_retry_text("INCORRECT", "999", self.actor, self.prompts, inventory)
        self.assertIn(REPAIR_OPERATION_PATCH_INSTRUCTION, text)
        shortest = {"kind": "shortest", "nodes": ["A", "B", "H"], "edge_ids": None,
                    "weights": [2, 2], "cost": 4}
        bfs = {"kind": "bfs", "steps": [{"node": "A", "new": ["B"]}]}
        self.assertIn("value only the two-node nodes[i:i+2] slice", text)
        self.assertEqual(operation_patch(shortest, "DIRECTED_EDGE", "nodes[1:3]"),
                         {"field": "nodes", "value": ["B", "H"]})
        self.assertIn("field steps, value the whole steps[i]", text)
        self.assertEqual(operation_patch(bfs, "BFS_FIFO", "steps[0].node"),
                         {"field": "steps", "value": {"node": "A", "new": ["B"]}})
        self.assertIn("COST_SUM: field cost, value the cost scalar", text)
        self.assertEqual(operation_patch(shortest, "COST_SUM", "cost"), {"field": "cost", "value": 4})
        self.assertIn("value the whole input list, even when the locus has an index", text)
        self.assertIn("Preserve JSON types, rational strings and units exactly", text)


if __name__ == "__main__":
    unittest.main()
