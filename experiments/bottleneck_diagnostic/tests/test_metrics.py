import unittest

from experiments.bottleneck_diagnostic.metrics import cluster, transition


class PairedStatisticsTests(unittest.TestCase):
    def test_recovery_regression_na_and_repetitions(self):
        samples = [{"sample_id": s, "family_id": "shared" if s in "ab" else s} for s in "abcd"]
        rows = []
        for sid, old, new in [("a", False, True), ("b", True, False), ("c", False, False), ("d", None, True)]:
            for rep in (3407, 3408):
                rows.extend({"sample_id": sid, "mode": mode, "repetition": rep, "diagnostic_score": value}
                            for mode, value in [("I", old), ("O0", new)])
        result = transition(rows, samples, "I", "O0", resamples=100)
        self.assertEqual(result["recovered"], 2)
        self.assertEqual(result["regressed"], 2)
        self.assertEqual(result["paired_questions"], 3)
        self.assertEqual(result["shared_image_text_families"], 2)
        self.assertEqual(result["recovery_rate"], .5)
        self.assertEqual(result["question_mean_delta"], 0)

    def test_cluster_is_predeclared_not_inferred(self):
        self.assertEqual(cluster("Music"), "A_SYMBOLIC_VISUAL")
        self.assertEqual(cluster("Geography"), "B_DIAGRAM_SPATIAL")
        self.assertEqual(cluster("Physics"), "OTHER_CONTROL")


if __name__ == "__main__":
    unittest.main()
