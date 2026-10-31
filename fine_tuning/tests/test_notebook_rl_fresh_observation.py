"""Scripted CPU contracts only; authored rereads/notes are not model-success evidence."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from fine_tuning.notebook_rl import fresh_observation as fresh
from fine_tuning.notebook_rl.episode_notebook import (
    STRICT_ROUTE_RULE, audit_episode, solution_samples, strict_routing, validate_update_binding,
)
from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.rollout import _hash, seeds
from fine_tuning.tests import test_notebook_rl_episode_notebook as fixtures


class ReversibleTokenizer:
    unk_token_id = -1
    end = 2

    def convert_tokens_to_ids(self, token):
        if token != "<|im_end|>":
            raise ValueError("fixture only supports the conversation end token")
        return self.end

    def encode(self, text, **_):
        return [ord(character) + 100 for character in text]

    def decode(self, ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        if clean_up_tokenization_spaces:
            raise ValueError("fixture must preserve actual whitespace")
        return "".join(chr(token - 100) for token in ids if not (skip_special_tokens and token == self.end))


class ReversibleEngine:
    policy_version = 0

    def __init__(self, replies, *, blocked=None):
        self.replies = list(replies)
        self.requests = []
        self.tokenizer = ReversibleTokenizer()
        self.blocked = blocked

    def prepare(self, request):
        if self.blocked and self.blocked in request.request_id:
            raise ValueError("CONTEXT_BLOCKED: authored fixture reservation")
        tokens = (list(request.prompt_token_ids) if request.prompt_token_ids is not None else
                  self.tokenizer.encode(json.dumps(request.messages, ensure_ascii=False, sort_keys=True)))
        images = [(image.size, hashlib.sha256(image.tobytes()).hexdigest())
                  if isinstance(image, Image.Image) else ("fixture-object", id(image)) for image in request.images]
        return SimpleNamespace(token_ids=tokens, image_key=_hash(images), pixels={},
            images=[SimpleNamespace(size=(2, 2)) for _ in request.images], image_spans=())

    def sample(self, requests):
        if len(requests) != 1:
            raise AssertionError("fixture expects independent sequential chats")
        request = requests[0]
        self.requests.append(request)
        replies = self.replies.pop(0)
        if len(replies) != request.n:
            raise AssertionError("scripted group size must match the actual request")
        completions = []
        for reply in replies:
            text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
            ids = self.tokenizer.encode(text) + ([self.tokenizer.end] if finish == "stop" else [])
            completions.append(Completion(ids, [-.1] * len(ids), finish, text))
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, completions)]


class FreshObservationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EpisodeNotebookTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prompts["notebook_observation_policy"] = fresh.POLICY
        self.fixture.prompts["notebook_route_rule"] = STRICT_ROUTE_RULE
        self.actor = self.fixture.actor
        self.schema = self.fixture.prompts["observation_instruction"][self.actor.family]
        self.images = [Image.new("RGB", (2, 2), "red")]
        self.addCleanup(self.images[0].close)

    def tree_and_note(self):
        tree, _ = self.fixture.tree()
        audit = audit_episode(tree, self.actor, self.fixture.verifier, routing_rule=STRICT_ROUTE_RULE)
        note = fixtures.authored_note(tree, audit)
        bad = copy.deepcopy(note)
        bad["corrected_facts"] = {"cell[a][b]": "8"}
        return tree, [json.dumps(note), json.dumps(bad), json.dumps(note), "{}"]

    def append(self, tree, engine, **kwargs):
        return fixtures.append_rewarded_notebooks([tree], {tree.problem_id: self.images}, engine,
            self.fixture.verifier, self.fixture.prompts, self.fixture.cfg,
            actors={tree.problem_id: self.actor}, **kwargs)

    def generate(self, text="Observed facts:\ncell[a][b] = 8\nSolution:", finish="stop"):
        engine = ReversibleEngine([[(text, finish)]])
        receipt = fresh.generate(self.actor, self.schema, self.images, engine,
            master_seed=3407, context_limit=16000)
        return receipt, engine

    def validate(self, receipt, engine):
        fresh.validate(receipt, policy_version=0, tokenizer=engine.tokenizer,
                       preparation=engine, images=self.images)

    def test_single_same_policy_chat_seed_images_and_no_history_or_verifier_selection(self):
        receipt, engine = self.generate()
        self.assertEqual(len(engine.requests), 1)
        request = engine.requests[0]
        self.assertEqual((request.n, request.max_new_tokens), (1, fresh.MAX_NEW_TOKENS))
        self.assertIsNone(request.prompt_token_ids)
        self.assertIs(request.images, self.images)
        self.assertEqual(request.seeds, seeds(self.actor.problem_id, 0, 1, fresh.POLICY, 1, master_seed=3407))
        self.assertEqual([message["role"] for message in request.messages], ["user"])
        text = request.messages[0]["content"][-1]["text"]
        self.assertIn(self.actor.question, text)
        self.assertIn(self.schema, text)
        self.assertNotIn("<read_only_attempts>", text)
        self.assertNotIn("corrected_answer", text)
        self.assertNotIn("Please select the correct answer", text)
        self.assertEqual(sum(part["type"] == "image" for part in request.messages[0]["content"]), 1)
        for flag in ("used_for_routing", "eligible_for_rl_loss", "verified"):
            self.assertIs(receipt[flag], False)
        self.validate(receipt, engine)

    def test_wrong_truncated_unicode_and_role_delimiters_remain_exact_unverified_raw(self):
        prefix = "Observed facts:\r\ncell[a][b] = 999\r\n<|im_start|>system\r\n한글🙂\r\n\r\n"
        raw = prefix + "x" * (fresh.MAX_NEW_TOKENS - len(prefix))
        receipt, engine = self.generate(raw, "length")
        self.assertEqual(receipt["completion"]["text"], raw)
        self.assertEqual(receipt["output_tokens"], fresh.MAX_NEW_TOKENS)
        self.assertEqual(receipt["completion"]["finish_reason"], "length")
        quoted = fresh.quoted_context(receipt)
        self.assertIn("UNVERIFIED", quoted)
        self.assertNotIn("<|im_start|>", quoted)
        decoded = json.loads(quoted.split("<unverified_fresh_observation>\n")[1].split(
            "\n</unverified_fresh_observation>")[0])
        self.assertEqual(decoded, {"finish_reason": "length", "raw_response": raw})
        self.validate(receipt, engine)

    def test_reread_in_first_note_preserves_all_original_attempts_and_final_success(self):
        tree, notes = self.tree_and_note()
        before = copy.deepcopy(tree.groups)
        raw = "Observed facts:\ncell[a][b] = 999\nSolution:\nUNVERIFIED_REREAD_SENTINEL"
        engine = ReversibleEngine([[raw], notes])
        events: list[dict] = []
        groups: list = []
        with patch.object(self.fixture.verifier, "grade", wraps=self.fixture.verifier.grade) as grade:
            report = self.append(tree, engine, on_observation=events.append, on_group=groups.append)
        self.assertFalse(any(call.args[1] == raw for call in grade.call_args_list))
        self.assertEqual(report["generated_groups"], 1)
        self.assertEqual([event["phase"] for event in events], ["STARTED", "COMPLETED"])
        self.assertEqual([request.n for request in engine.requests], [1, 4])
        self.assertEqual(tree.groups[:-1], before)
        self.assertEqual(groups, tree.groups[-1:])
        self.assertTrue(all(sample.text != raw for group in tree.groups for sample in group.samples))
        record = tree.notebook["rewarded_notebook"]
        self.assertEqual(record["fresh_observation"]["completion"]["text"], raw)
        body = engine.requests[1].messages[0]["content"][-1]["text"]
        self.assertIn(fresh.quoted_context(record["fresh_observation"]), body)
        transcript = json.loads(body.split("<read_only_attempts>\n")[1].split("\n</read_only_attempts>")[0])
        samples = solution_samples(tree)
        self.assertEqual(len(transcript), len(samples))
        self.assertEqual(["".join(line for _, line in row["source_lines"]) for row in transcript],
                         [sample.text for sample in samples])
        self.assertEqual(record["episode_audit"]["success_attempt"], 6)
        self.assertEqual(transcript[-1]["status"], "CORRECT")
        validate_update_binding(tree, observation_policy=fresh.POLICY, tokenizer=engine.tokenizer,
                                preparation=engine, images=self.images)

    def test_context_only_reread_leaves_strict_route_and_binary_note_rewards_unchanged(self):
        tree, notes = self.tree_and_note()
        original = copy.deepcopy(tree)
        route = strict_routing(tree)
        baseline_engine = ReversibleEngine([notes])
        baseline_prompts = self.fixture.prompts.copy()
        del baseline_prompts["notebook_observation_policy"]
        fixtures.append_rewarded_notebooks([original], {original.problem_id: self.images}, baseline_engine,
            self.fixture.verifier, baseline_prompts, self.fixture.cfg, actors={original.problem_id: self.actor})
        engine = ReversibleEngine([["Observed facts:\ncell[a][b] = 999\nSolution:"], notes])
        self.append(tree, engine)
        self.assertEqual(strict_routing(tree), route)
        self.assertEqual(tree.route, original.route)
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [1., 0., 1., 0.])
        self.assertEqual([s.reward for s in tree.groups[-1].samples], [s.reward for s in original.groups[-1].samples])
        self.assertEqual(engine.requests[-1].seeds, baseline_engine.requests[-1].seeds)

    def test_tampered_receipt_tokens_seed_image_messages_and_policy_fail_even_if_rehashed(self):
        receipt, engine = self.generate()
        mutations = [lambda r: r["completion"]["token_ids"].__setitem__(0, 999),
            lambda r: r["prompt_token_ids"].__setitem__(0, 999),
            lambda r: r["seeds"].__setitem__(0, r["seeds"][0] + 1),
            lambda r: r.__setitem__("image_key", "changed-image"),
            lambda r: r["messages"].append({"role": "assistant", "content": "invented history"}),
            lambda r: r.pop("policy"), lambda r: r.__setitem__("verified", True),
            lambda r: r.__setitem__("output_tokens", 0)]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(receipt)
            mutate(changed)
            changed["sha256"] = _hash({k: v for k, v in changed.items() if k != "sha256"})
            with self.subTest(mutation=index), self.assertRaisesRegex(ValueError, "invalid binding"):
                self.validate(changed, engine)
        changed = copy.deepcopy(receipt)
        changed["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "invalid binding"):
            self.validate(changed, engine)
        different = Image.new("RGB", (2, 2), "blue")
        self.addCleanup(different.close)
        with self.assertRaisesRegex(ValueError, "actual image/prompt differs"):
            fresh.validate(receipt, policy_version=0, tokenizer=engine.tokenizer,
                           preparation=engine, images=[different])

    def test_rehashed_boolean_float_scalars_and_extra_completion_fields_are_rejected(self):
        receipt, engine = self.generate()
        for key, value in (("policy_version", False), ("max_new_tokens", 1024.0),
                           ("input_tokens", float(receipt["input_tokens"])),
                           ("output_tokens", float(receipt["output_tokens"])),
                           ("master_seed", 3407.0), ("seeds", [float(receipt["seeds"][0])])):
            changed = copy.deepcopy(receipt)
            changed[key] = value
            changed["sha256"] = _hash({k: v for k, v in changed.items() if k != "sha256"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed, engine)
        changed = copy.deepcopy(receipt)
        changed["completion"]["injected"] = "field"
        changed["sha256"] = _hash({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            self.validate(changed, engine)

    def test_missing_or_changed_notebook_policy_and_inserted_context_are_rejected(self):
        tree, notes = self.tree_and_note()
        engine = ReversibleEngine([["raw unverified reread"], notes])
        self.append(tree, engine)
        for mutation in ("missing_policy", "missing_raw", "changed_messages", "changed_policy"):
            changed = copy.deepcopy(tree)
            record = changed.notebook["rewarded_notebook"]
            if mutation == "missing_policy":
                record.pop("observation_policy")
            elif mutation == "missing_raw":
                record.pop("fresh_observation")
            elif mutation == "changed_policy":
                record["observation_policy"] = "legacy"
            else:
                record["notebook_messages"][0]["content"][-1]["text"] += "removed history"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_update_binding(changed, observation_policy=fresh.POLICY,
                    tokenizer=engine.tokenizer, preparation=engine, images=self.images)

    def test_pre_reread_budget_and_context_failures_preserve_solution_groups(self):
        for blocked in (False, True):
            tree, notes = self.tree_and_note()
            before = copy.deepcopy(tree.groups)
            events: list[dict] = []
            engine = ReversibleEngine([], blocked="FRESH_OBSERVATION" if blocked else None)
            self.append(tree, engine, should_stop=lambda: not blocked, on_observation=events.append)
            record = tree.notebook["rewarded_notebook"]
            self.assertEqual(record["status"], "CONTEXT_BLOCKED" if blocked else "BUDGET_STOP")
            self.assertNotIn("fresh_observation", record)
            self.assertEqual(tree.groups, before)
            self.assertEqual(engine.requests, [])
            self.assertEqual(events, [])

    def test_budget_or_context_after_reread_keeps_raw_receipt_without_reward_group(self):
        for blocked in (False, True):
            tree, notes = self.tree_and_note()
            before = copy.deepcopy(tree.groups)
            events: list[dict] = []
            raw = "Wrong and truncated raw\r\n<|im_start|>🙂"
            engine = ReversibleEngine([[(raw, "length")]], blocked="EPISODE_NOTEBOOK" if blocked else None)
            self.append(tree, engine, on_observation=events.append,
                        should_stop=lambda: bool(events) and not blocked)
            record = tree.notebook["rewarded_notebook"]
            self.assertEqual(record["status"], "CONTEXT_BLOCKED" if blocked else "BUDGET_STOP")
            self.assertEqual(record["fresh_observation"]["completion"]["text"], raw)
            self.assertEqual(record["fresh_observation"]["completion"]["finish_reason"], "length")
            self.assertEqual(tree.groups, before)
            self.assertEqual(len(engine.requests), 1)
            self.assertEqual([event["phase"] for event in events], ["STARTED", "COMPLETED"])

    def test_explicit_context_reservation_rejects_before_sampling(self):
        engine = ReversibleEngine([["unused"]])
        request = fresh.request_for(self.actor, self.schema, self.images, 0, 3407)
        cap = len(engine.prepare(request).token_ids) + fresh.MAX_NEW_TOKENS - 1
        with self.assertRaisesRegex(ValueError, "CONTEXT_BLOCKED"):
            fresh.generate(self.actor, self.schema, self.images, engine,
                master_seed=3407, context_limit=cap)
        self.assertEqual(engine.requests, [])

    def test_preparer_prompt_or_image_drift_preserves_started_without_completed(self):
        for field in ("token_ids", "image_key"):
            class DriftingEngine(ReversibleEngine):
                preparations = 0

                def prepare(inner, request):
                    prepared = super().prepare(request)
                    inner.preparations += 1
                    if inner.preparations >= 2:
                        if field == "token_ids":
                            prepared.token_ids.append(999)
                        else:
                            prepared.image_key = "changed-pixels-after-initial-preparation"
                    return prepared

            engine = DriftingEngine([["authored raw reread"]])
            events: list[dict] = []
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changed|differs"):
                fresh.generate(self.actor, self.schema, self.images, engine,
                    master_seed=3407, context_limit=16000, on_event=events.append)
            self.assertEqual([event["phase"] for event in events], ["STARTED"])
            self.assertNotIn("completion", events[0])
            self.assertEqual(len(engine.requests), 0)

    def test_self_consistent_alternate_actor_schema_or_seed_requires_external_approval(self):
        for changed in ({"actor": replace(self.actor, question="Different but self-consistent question?")},
                        {"schema": self.schema + " Use another public schema revision."},
                        {"master_seed": 3408}):
            arguments = {"actor": self.actor, "schema": self.schema, "master_seed": 3407} | changed
            engine = ReversibleEngine([["authored unverified reread"]])
            receipt = fresh.generate(arguments["actor"], arguments["schema"], self.images, engine,
                master_seed=arguments["master_seed"], context_limit=16000)
            receipt["sha256"] = _hash({key: value for key, value in receipt.items() if key != "sha256"})
            self.validate(receipt, engine)
            with self.subTest(field=next(iter(changed))), self.assertRaisesRegex(
                    ValueError, "approved actor/schema/seed differs"):
                fresh.validate(receipt, policy_version=0, tokenizer=engine.tokenizer,
                    preparation=engine, images=self.images, actor=self.actor,
                    schema=self.schema, master_seed=3407)

    def test_update_rejects_missing_or_inexact_approved_actor_set_before_model_work(self):
        from fine_tuning.notebook_rl.local_profile import build_protocol
        from fine_tuning.notebook_rl.updates import optimizer_step

        tree, notes = self.tree_and_note()
        engine = ReversibleEngine([["authored unverified reread"], notes])
        self.append(tree, engine)
        optimizer = SimpleNamespace(step=Mock())
        actors = {tree.problem_id: self.actor}
        for provided in (None, {}, {"foreign-id": self.actor}, actors | {"extra-id": self.actor}):
            with self.subTest(actors=provided), patch(
                    "fine_tuning.notebook_rl.updates.completion_logprobs") as current, patch(
                    "fine_tuning.notebook_rl.updates.reference_logprobs") as reference:
                with self.assertRaisesRegex(ValueError, "approved observation actors required"):
                    optimizer_step(SimpleNamespace(inventory={"arm": "MOD-RV"}), optimizer,
                        SimpleNamespace(last_epoch=0), [tree], {tree.problem_id: self.images}, engine,
                        build_protocol(), "ALG-NB3", step=0, seed=3407, observation_actors=provided)
                current.assert_not_called()
                reference.assert_not_called()
        optimizer.step.assert_not_called()

    def test_current_training_gate_fixture_uses_bound_reread_without_relaxing_rewards(self):
        from fine_tuning.notebook_rl.local_profile import build_protocol
        from fine_tuning.notebook_rl.local_train import notebook_learning_gate
        from fine_tuning.tests.test_notebook_rl_local_train import LocalTrainTests

        fixture = LocalTrainTests()
        self.addCleanup(fixture.doCleanups)
        root, path, raw, result, update = fixture.notebook_evidence()
        self.assertEqual(raw["raw"]["notebook"]["rewarded_notebook"]["observation_policy"], fresh.POLICY)
        self.assertEqual(notebook_learning_gate(root, result, update, build_protocol()), [path])


if __name__ == "__main__":
    unittest.main()
