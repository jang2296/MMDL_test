"""Public raw-claim indexing tests; no model or semantic verification is involved."""
import copy
import unittest
from types import SimpleNamespace

from fine_tuning.notebook_rl import notebook_synthesis as synthesis
from fine_tuning.notebook_rl.episode_notebook import solution_samples
from fine_tuning.tests import test_notebook_rl_notebook_synthesis as native


class PublicCoverageTests(unittest.TestCase):
    def test_stable_wrong_claims_successful_attempts_and_all_retry_edges_remain(self):
        actor, item, verifier = native.native()
        facts = {**item.slots, "start": "q999"}
        tree = native.tree_from(actor, verifier, [native.response(facts, wrong=False)] * 6)
        samples = solution_samples(tree)
        table = synthesis.public_coverage(samples)
        self.assertEqual(len(table["attempts"]), 6)
        self.assertEqual([row[0] for row in table["slots"]["start"]], list(range(1, 7)))
        self.assertEqual([row[4] for row in table["slots"]["start"]], ["q999"] * 6)
        self.assertEqual(table["retry_pairs"], [[1, 3], [1, 4], [3, 5], [3, 6]])
        self.assertTrue(all(sample.result.status == "CORRECT" for sample in samples))
        for sample in samples:
            sample.result = None
            sample.reward, sample.diag_reward = -100, 900
        verifier.items[actor.problem_id] = native.replace(item, gold="a different gold", slots={})
        self.assertEqual(synthesis.public_coverage(samples), table)

    def test_reads_only_three_public_fields(self):
        class PublicOnly:
            def __getattribute__(self, name):
                return {"text": "", "sample_id": "public", "parent_sample_id": None}[name]

        table = synthesis.public_coverage([PublicOnly()])
        self.assertEqual(table["attempts"], [[1, "public", None, 0]])
        self.assertEqual((table["slots"], table["raw_observation_lines"], table["retry_pairs"]), ({}, [], []))

    def test_duplicates_malformed_values_and_raw_crlf_positions_are_lossless(self):
        text = ('Observed facts:\r\n - DIR[e4] = "A->B"  \r\n'
                'dir[e4] = "A->B"\r\nweight[e4] = malformed 12 prose\r\n'
                'weight[e4] =\r\nnot an assignment\r\n\r\nSolution:\r\nTrace: {}')
        sample = SimpleNamespace(text=text, sample_id="s", parent_sample_id=None)
        table = synthesis.public_coverage([sample])
        self.assertEqual([row[4] for row in table["slots"]["dir[e4]"]], ['"A->B"', '"A->B"'])
        self.assertEqual(table["slots"]["weight[e4]"][0][4], "malformed 12 prose")
        self.assertEqual([row[1] for row in table["raw_observation_lines"]], [5, 6, 7])
        for rows in table["slots"].values():
            for _, line, start, end, _, source in rows:
                self.assertEqual((text[start:end], text.splitlines(keepends=True)[line - 1]), (source, source))
        for _, line, start, end, source in table["raw_observation_lines"]:
            self.assertEqual((text[start:end], text.splitlines(keepends=True)[line - 1]), (source, source))

    def test_empty_unclosed_and_repeated_blocks_are_retained_without_inventing_values(self):
        samples = [SimpleNamespace(text="", sample_id="empty", parent_sample_id=None),
            SimpleNamespace(text="Observed facts:\ndir[e4] = A->B\nraw\n", sample_id="unclosed", parent_sample_id="empty"),
            SimpleNamespace(text="Observed facts:\nroot = A\nObserved facts:\nroot = B\nSolution:\n",
                            sample_id="repeat", parent_sample_id="unclosed")]
        before = copy.deepcopy(samples)
        table = synthesis.public_coverage(samples)
        self.assertEqual(table["attempts"][0], [1, "empty", None, 0])
        self.assertEqual(table["retry_pairs"], [[1, 2], [2, 3]])
        self.assertEqual([row[-1] for row in table["raw_observation_lines"]],
                         ["dir[e4] = A->B\n", "raw\n", "root = A\n"])
        self.assertEqual(table["slots"]["root"][0][4], "B")
        self.assertEqual(samples, before)

    def test_ambiguous_ids_or_missing_parent_are_rejected(self):
        sample = SimpleNamespace(text="", sample_id="s", parent_sample_id=None)
        with self.assertRaisesRegex(ValueError, "unique"):
            synthesis.public_coverage([sample, sample])
        sample.parent_sample_id = "absent"
        with self.assertRaisesRegex(ValueError, "parent missing"):
            synthesis.public_coverage([sample])


if __name__ == "__main__":
    unittest.main()
