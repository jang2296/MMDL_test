"""Public source-format contracts only; no model output or learning certificate."""
import copy
import unittest
from dataclasses import replace

from fine_tuning.notebook_rl.local_profile import build_protocol, local_prompts
from fine_tuning.notebook_rl.local_data import project_observation_actors
from fine_tuning.notebook_rl.prompts import for_actor, retry_text, turn1_text
from fine_tuning.tests.test_notebook_rl_schema import records


class SourceObservationTests(unittest.TestCase):
    def actor(self, source="gen-graph-pair-v1"):
        return replace(records()[0], source=source, family="graph", mode="obs",
            question="Use both printed images to calculate the requested result.", choices=(),
            image_paths=("first.png", "second.png"), image_sha256=("a"*64, "b"*64))

    def test_pair_prompt_has_one_consistent_public_schema(self):
        prompts = local_prompts(build_protocol())
        before = copy.deepcopy(prompts)
        text = turn1_text(self.actor(), prompts)
        self.assertIn("dir[<edge ID>] = <source>-><target>", text)
        self.assertIn("weight[<edge ID>] = <integer>", text)
        self.assertIn("each assignment on its own line", text)
        self.assertNotIn("edge[<u>,<v>]", text)
        self.assertEqual(prompts, before)

    def test_all_retry_statuses_use_the_same_pair_schema(self):
        actor = self.actor()
        selected = for_actor(actor, local_prompts(build_protocol()))
        for status in ("INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
            text = retry_text(status, "previous-model-answer", actor.mode, actor.family, selected)
            self.assertIn("dir[<edge ID>]", text)
            self.assertIn("weight[<edge ID>]", text)
            self.assertNotIn("edge[<u>,<v>]", text)

    def test_single_image_source_uses_explicit_assignments_and_old_prompt_is_unchanged(self):
        prompts = local_prompts(build_protocol())
        actor = replace(self.actor("gen-graph-v1"), image_paths=("first.png",), image_sha256=("a"*64,))
        text = turn1_text(actor, prompts)
        self.assertIn("edge[<source>,<target>]", text)
        self.assertIn("one assignment on each line", text)
        self.assertIn("trans[<state>,<symbol>]", text)
        self.assertIn("child[<parent>,<L or R>]", text)
        self.assertIn("only the applicable form", text)
        old = {k: v for k, v in prompts.items() if k != "source_observation_instruction"}
        self.assertIs(for_actor(self.actor(), old), old)
        self.assertIn("edge[<u>,<v>]", turn1_text(self.actor(), old))

    def test_unbound_source_instruction_is_rejected(self):
        protocol = build_protocol()
        protocol["source_observation_instruction"]["gen-graph-pair-v1"] = "private answer value"
        with self.assertRaisesRegex(ValueError, "source observation instructions changed"):
            local_prompts(protocol)

    def test_projected_plain_actor_requests_observations_in_first_and_retry_prompts(self):
        protocol = build_protocol()
        original = replace(self.actor(), mode="plain")
        actor, = project_observation_actors([original], protocol)
        prompts = local_prompts(protocol)
        self.assertNotIn("dir[<edge ID>]", turn1_text(original, prompts))
        self.assertIn("dir[<edge ID>]", turn1_text(actor, prompts))
        selected = for_actor(actor, prompts)
        for status in ("INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
            self.assertIn("dir[<edge ID>]", retry_text(status, "own answer", actor.mode, actor.family, selected))
        self.assertEqual(original.mode, "plain")


if __name__ == "__main__":
    unittest.main()
