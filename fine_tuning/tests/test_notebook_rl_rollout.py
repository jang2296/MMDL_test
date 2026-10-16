"""Scripted responses exercise control flow, not learned self-correction performance."""
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fine_tuning.notebook_rl.engines import Completion, RolloutResult
from fine_tuning.notebook_rl.prompts import load_prompts
from fine_tuning.notebook_rl.rollout import (RolloutConfig, append_diagnoses, build_tree, continuation_ids,
                                            seeds, select_failures)
from fine_tuning.notebook_rl.schema import from_json, ProblemTree, to_json
from fine_tuning.notebook_rl.vault import write_split
from fine_tuning.notebook_rl.verifier import Verifier
from fine_tuning.tests.test_notebook_rl_schema import records


class Tokenizer:
    unk_token_id = -1

    def __init__(self):
        self.encoded = []

    def convert_tokens_to_ids(self, token):
        return 151645

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return [900 + ord(c) for c in text]


class Engine:
    policy_version = 0

    def __init__(self, replies):
        self.tokenizer = Tokenizer()
        self.replies = replies
        self.requests = []

    def prepare(self, request):
        return SimpleNamespace(token_ids=request.prompt_token_ids if request.prompt_token_ids is not None else [4, 5])

    def sample(self, requests):
        request = requests[0]
        texts = self.replies[len(self.requests)]
        self.requests.append(request)
        cs = [Completion([30+i, 151645], [-.1, -.2], "stop", text) for i, text in enumerate(texts)]
        return [RolloutResult(request.request_id, self.prepare(request).token_ids, cs)]


class RolloutTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.actor, v, lineage, *_ = records()
        self.verifier_root = Path(temp.name)
        write_split(self.verifier_root, [self.actor], [v], [lineage])
        self.verifier = Verifier(self.verifier_root)
        self.prompts = load_prompts(Path(__file__).parents[1] / "notebook_rl/prompts_v1.yaml")
        self.cfg = RolloutConfig({1: 8, 2: 4, 3: 4}, {1: 2, 2: 1}, 2, {1: 32, 2: 32, 3: 32},
                                 5000, False, 4)

    def tree(self, engine, cfg=None, verifier=None):
        return build_tree(self.actor, [object()], engine, verifier or self.verifier, self.prompts,
                          cfg or self.cfg, 0, "verified-any-visual-v1")

    def test_tree_groups_same_state_and_literal_history(self):
        engine = Engine([["MODEL_ONLY_SENTINEL\nFinal answer: 8", "MODEL_ONLY_SENTINEL\nFinal answer: 9"]*4, ["Final answer: 7"]*4,
                         ["Final answer: 8", "Final answer: 7"]*2])
        tree = self.tree(engine)
        self.assertEqual([len(g.samples) for g in tree.groups], [8, 4, 4])
        first = {s.sample_id: s for s in tree.groups[0].samples}
        for group in tree.groups[1:]:
            parent = first[group.samples[0].parent_sample_id]
            self.assertEqual(group.samples[0].prompt_token_ids[:4], [4, 5]+parent.completion_token_ids)
            self.assertEqual(len({tuple(s.prompt_token_ids) for s in group.samples}), 1)
            # The response string never passes back through tokenizer.encode.
        self.assertFalse(any("MODEL_ONLY_SENTINEL" in s for s in engine.tokenizer.encoded))
        self.assertEqual(from_json(ProblemTree, to_json(tree)), tree)

    def test_master_seed_reaches_actual_rollout_requests(self):
        actual = []
        for master in (3407, 3408, 3407):
            engine = Engine([["Final answer: 7"]*8])
            self.tree(engine, replace(self.cfg, master_seed=master))
            actual.append(engine.requests[0].seeds)
        self.assertEqual(actual[0], actual[2])
        self.assertNotEqual(actual[0], actual[1])
        self.assertEqual(actual[1], seeds(self.actor.problem_id, 0, 1, "T1:root", 8, master_seed=3408))

    def test_budget_between_groups_preserves_completed_first_without_retry(self):
        engine = Engine([["Final answer: 8"]*8])
        closed = []
        tree = build_tree(self.actor, [object()], engine, self.verifier, self.prompts,
            self.cfg, 0, "verified-any-visual-v1", on_group=closed.append,
            should_stop=lambda: bool(closed))
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0], tree.groups[0])
        self.assertEqual(len(engine.requests), 1)
        self.assertTrue(all(s["status"] == "BUDGET_STOP" for s in tree.notebook["skipped_branches"]))

    def test_all_correct_has_no_retries_and_unverifiable_group_not_used(self):
        tree = self.tree(Engine([["Final answer: 7"]*8]))
        self.assertEqual(len(tree.groups), 1)
        engine = Engine([["Final answer: bananas"]+["Final answer: 8"]*7])
        tree = self.tree(engine)
        self.assertFalse(tree.groups[0].usable)
        self.assertEqual(tree.groups[0].reason, "UNVERIFIABLE")
        self.assertEqual(len(engine.requests), 1)

    def test_third_turn_is_one_globally_selected_branch(self):
        engine = Engine([["Final answer: 8"]*8, ["Final answer: 8"]*4,
                         ["Final answer: 9"]*4, ["Final answer: 7"]*4])
        tree = self.tree(engine, replace(self.cfg, max_attempts=3))
        self.assertEqual([g.kind for g in tree.groups], ["T1", "T2", "T2", "T3"])
        third = tree.groups[-1].samples[0]
        parent = next(s for g in tree.groups for s in g.samples if s.sample_id == third.parent_sample_id)
        self.assertEqual(third.prompt_token_ids[:len(parent.prompt_token_ids)+2],
                         parent.prompt_token_ids+parent.completion_token_ids)

    def test_context_blocked_preserves_first_results_without_truncating(self):
        engine = Engine([["Final answer: 8"]*8])
        tree = self.tree(engine, replace(self.cfg, context_limit=40))
        self.assertEqual(len(tree.groups), 1)
        self.assertEqual(len(tree.notebook["skipped_branches"]), 2)
        self.assertTrue(all(s["status"] == "CONTEXT_BLOCKED" for s in tree.notebook["skipped_branches"]))

    def test_diagnosis_deferred_until_batch_and_single_class_not_generated(self):
        wrong = "Observed facts:\ncell[a][b] = 8\nSolution:\nFinal answer: 8"
        right = "Observed facts:\ncell[a][b] = 7\nSolution:\nFinal answer: 7"
        note = "Cause: VISUAL_MISREAD\nWrong item: cell[a][b]\nLesson: Recheck the source cell."
        engine = Engine([[wrong]*8, [right]*4, [right]*4, [note]*4])
        tree = self.tree(engine, replace(self.cfg, diag=True))
        self.assertEqual([g.kind for g in tree.groups], ["T1", "T2", "T2"])
        self.assertEqual(tree.route, "VISION_ON")
        self.assertEqual(tree.notebook["diagnosis_candidate"]["target"]["cause3"], "VISUAL")
        report = append_diagnoses([tree], {tree.problem_id: [object()]}, engine, self.verifier,
                                 self.prompts, replace(self.cfg, diag=True))
        self.assertEqual(report["generated_groups"], 0)
        self.assertEqual(report["excluded_by_cap"], 1)
        self.assertEqual(len(engine.requests), 3)
        self.assertFalse(any(n["lesson_semantics_verified"] for n in tree.notebook["model_notes"]))
        self.assertFalse(tree.notebook["verified_internal_module_responsibility"])
        # Unknown perception has no fabricated diagnosis target.
        engine = Engine([["Final answer: 8"]*8, ["Final answer: 7"]*4, ["Final answer: 7"]*4])
        tree = self.tree(engine, replace(self.cfg, diag=True))
        self.assertEqual(len(tree.groups), 3)
        self.assertIsNone(tree.notebook["diagnosis_candidate"])

    def test_class_cap_applied_before_generating_balanced_diagnoses(self):
        actor, verifier_item, lineage, *_ = records()
        pids = ["visual_a", "visual_b", "finalization_c"]
        root = self.verifier_root / "balanced"
        actors = [replace(actor, problem_id=pid) for pid in pids]
        write_split(root, actors, [replace(verifier_item, problem_id=pid) for pid in pids],
                    [replace(lineage, problem_id=pid) for pid in pids])
        verifier = Verifier(root)
        wrong = "Observed facts:\ncell[a][b] = 8\nSolution:\nFinal answer: 8"
        right = "Observed facts:\ncell[a][b] = 7\nSolution:\nFinal answer: 7"
        note = "Cause: VISUAL_MISREAD\nWrong item: cell[a][b]\nLesson: Recheck the source cell."
        replies = []
        for index in range(3):
            replies.extend([[wrong if index < 2 else "No final answer yet"]*8, [right]*4, [right]*4])
        replies.extend([[note]*4, [note]*4])
        engine, cfg = Engine(replies), replace(self.cfg, diag=True)
        trees = [build_tree(a, [object()], engine, verifier, self.prompts, cfg, 0,
                            "verified-any-visual-v1") for a in actors]
        self.assertEqual(len(engine.requests), 9)
        report = append_diagnoses(trees, {t.problem_id: [object()] for t in trees}, engine,
                                 verifier, self.prompts, cfg)
        self.assertEqual(report["class_counts"], {"VISUAL": 1, "NON_VISUAL": 0, "FINALIZATION": 1})
        self.assertEqual(report["excluded_by_cap"], 1)
        self.assertEqual(report["generated_groups"], 2)
        self.assertEqual(len(engine.requests), 11)
        visual = next(t for t in trees if t.notebook["diagnosis_target"]
                      and t.notebook["diagnosis_target"]["cause3"] == "VISUAL")
        self.assertTrue(all(s.diag_reward == 1.25 and s.result is None for s in visual.groups[-1].samples))
        self.assertFalse(any(n["lesson_semantics_verified"] for t in trees for n in t.notebook["model_notes"]))
        with self.assertRaisesRegex(ValueError, "fresh"):
            append_diagnoses(trees, {t.problem_id: [object()] for t in trees}, engine, verifier, self.prompts, cfg)

    def test_boundary_added_once_never_marked_generated(self):
        sample = records()[5]
        tokenizer = Tokenizer()
        for completion in ([30], [30, 151645]):
            changed = replace(sample, completion_token_ids=completion)
            ids = continuation_ids(changed, "retry", tokenizer)
            self.assertEqual(ids[:4], sample.prompt_token_ids+[30, 151645])
            self.assertEqual(changed.completion_token_ids, completion)

    def test_seeds_and_failure_selection_deterministic_diverse(self):
        expected = int.from_bytes(hashlib.sha256(b"3407|x|4|2|b|0").digest()[:8], "big") % 2**31
        self.assertEqual(seeds("x", 4, 2, "b", 2)[0], expected)
        sample = records()[5]
        samples = [replace(sample, sample_id=str(i), result=replace(sample.result, extracted=a))
                   for i, a in enumerate(["8"]*7+["9"])]
        selected = select_failures(samples, 2, "key")
        self.assertEqual({s.result.extracted for s in selected}, {"8", "9"})
        self.assertEqual(selected, select_failures(list(reversed(samples)), 2, "key"))

    def test_changed_version_ids_size_or_image_coverage_rejected(self):
        for mutation in ("version", "ids", "count"):
            engine = Engine([["Final answer: 7"]*8])
            original = engine.sample
            def altered(requests, mutation=mutation):
                outputs = original(requests)
                if mutation == "version":
                    engine.policy_version = 1
                elif mutation == "ids":
                    outputs[0].prompt_token_ids.append(18)
                else:
                    outputs[0].completions.pop()
                return outputs
            engine.sample = altered
            with self.assertRaises(RuntimeError):
                self.tree(engine)


if __name__ == "__main__":
    unittest.main()
