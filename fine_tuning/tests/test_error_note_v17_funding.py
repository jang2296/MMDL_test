"""Calibrated funding arithmetic; fixtures never read accounts or change cloud resources."""
import unittest

from fine_tuning.error_note.guard_v16b import funding_decision, funding_decision_v17


class FundingTests(unittest.TestCase):
    costs = {"evaluation_seconds": 7200, "other_post_train_seconds": 1800,
             "recovery_seconds": 600, "storage_usd_per_hour": .1, "reserve_usd": 1.5,
             "next_train_seconds": 3600}

    def test_actual_calibrated_work_and_storage_determine_threshold(self):
        receipt = funding_decision_v17(100, 1.9, 1000, **self.costs)
        self.assertEqual(receipt["post_train_seconds"], 9600)
        self.assertAlmostEqual(receipt["protected_usd"], 1.5 + 2 * 9600 / 3600)
        self.assertEqual(receipt["next_train_usd"], 2)
        threshold = receipt["protected_usd"] + receipt["next_train_usd"]
        self.assertEqual(funding_decision_v17(threshold, 1.9, 1000, **self.costs)["status"], "CONTINUE")
        self.assertEqual(funding_decision_v17(threshold - .001, 1.9, 1000, **self.costs)["status"], "RECHARGE_REQUESTED")
        longer = funding_decision_v17(threshold, 1.9, 1000, **(self.costs | {"evaluation_seconds": 28800}))
        self.assertEqual(longer["status"], "RECHARGE_REQUESTED")
        self.assertGreater(longer["protected_usd"], receipt["protected_usd"])

    def test_grace_stops_after_thirty_minutes_and_recharge_does_not_restart(self):
        first = funding_decision_v17(1, 1.9, 1000, **self.costs)
        self.assertEqual(first["requested_at"], 1000)
        self.assertEqual(funding_decision_v17(1, 1.9, 2799, requested_at=1000, **self.costs)["status"], "RECHARGE_REQUESTED")
        stopped = funding_decision_v17(1, 1.9, 2800, requested_at=1000, **self.costs)
        self.assertEqual(stopped["status"], "STOP_REQUESTED")
        restored = funding_decision_v17(100, 1.9, 2800, requested_at=1000, **self.costs)
        self.assertIsNone(restored["requested_at"])
        self.assertFalse(restored["auto_recharge"])
        self.assertFalse(restored["auto_restart"])

    def test_elapsed_target_is_not_a_training_stop(self):
        receipt = funding_decision_v17(100, 1.9, 35 * 3600, **self.costs)
        self.assertEqual(receipt["status"], "CONTINUE")
        self.assertIsNone(receipt["training_elapsed_cutoff"])

    def test_invalid_or_unmeasured_costs_fail_closed_and_legacy_is_unchanged(self):
        for key, value in (("evaluation_seconds", 0), ("evaluation_seconds", float("nan")),
                           ("storage_usd_per_hour", -.1), ("reserve_usd", True),
                           ("recovery_seconds", 0), ("next_train_seconds", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                funding_decision_v17(100, 1.9, 1000, **(self.costs | {key: value}))
        with self.assertRaises(ValueError):
            funding_decision_v17(100, 1.9, 1000, requested_at=1001, **self.costs)
        with self.assertRaisesRegex(ValueError, "derived cost"):
            funding_decision_v17(100, 1e308, 1000, **(self.costs | {"evaluation_seconds": 1e308}))
        self.assertEqual(funding_decision(100, 1.9, 1000)["protected_usd"], 5.65 * 1.9 + 1.5)


if __name__ == "__main__":
    unittest.main()
