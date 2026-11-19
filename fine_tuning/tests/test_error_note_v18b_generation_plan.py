"""Cost reports describe the actual monitor schedule and never control training admission."""
import unittest

from fine_tuning.error_note.v18b_generation_plan import generation_plan


class GenerationPlanTests(unittest.TestCase):
    def test_short_warning_and_accuracy_have_separate_workload(self):
        report = generation_plan("v18b-train", {
            "execution_settings": {"reuse_f0_raw": True},
            "generation_timing": {"short_monitor": {
                "output_tokens": 1000, "elapsed_seconds": 10, "source": "fixture"},
                "reference_only": {"output_tokens": 5000, "elapsed_seconds": 10, "source": "other panel"}},
        })
        stages = report["stages"]
        self.assertTrue(report["advisory_only"])
        self.assertEqual(stages["short_monitor"]["max_output_tokens_per_call"], 16896)
        self.assertEqual(stages["short_monitor"]["cap_scenario_seconds"], 844.8)
        self.assertEqual(stages["accuracy_monitor"]["max_output_tokens"], 4 * 200 * 8192)
        self.assertIsNone(stages["accuracy_monitor"]["cap_scenario_seconds"])
        self.assertEqual(stages["accuracy_monitor"]["reference_only_cap_seconds"], 13107.2)
        self.assertEqual(stages["prep_solutions"]["cap_scenario_seconds"], 0)
        self.assertEqual(stages["first_solutions"]["max_requests_per_call"], 3000)

    def test_official_eval_and_invalid_timing(self):
        params = {"execution_settings": {"evaluation_strategy": "one_pod"}}
        self.assertEqual(generation_plan("v18b-eval", params)["stages"]["mmmu"]["max_output_tokens"], 900 * 32768)
        for sample in (None, {"output_tokens": True, "elapsed_seconds": 1, "source": "foo"},
                       {"output_tokens": 10, "elapsed_seconds": float("nan"), "source": "foo"}):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                generation_plan("v18b-eval", params | {"generation_timing": {"mmmu": sample}})
