"""Synthetic rule-C boundaries and budget decisions; no model or cloud access."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note.guard_v16b import GuardV16B, falsealarm_receipt, funding_decision


def order(n=40):
    return [[f"{b}-{i}" for i in range(16)] for b in range(n)]


def batch(ids, *, correct=10, cap=2, nofinal=2, tokens=100, f0=100):
    return [{"id": sid, "correct": i < correct, "truncated": i < cap, "no_final": i < nofinal,
             "tokens": tokens, "f0_tokens": f0} for i, sid in enumerate(ids)]


class GuardTests(unittest.TestCase):
    def test_counts_start36_ref_scaled24_to12_and_resume(self):
        ids = order()
        guard = GuardV16B(ids)
        for b in range(1, 36):
            result = guard.update(b, batch(ids[b - 1], cap=1 if b <= 24 else 5))
        self.assertNotIn("TRUNC_UP", result["reasons"])
        guard = GuardV16B(ids, json.loads(json.dumps(guard.state)))
        result = guard.update(36, batch(ids[35], cap=5))
        self.assertEqual(result["thresholds"]["truncated"]["ref"], 12)
        self.assertIn("TRUNC_UP", result["reasons"])
        self.assertNotIn("NOFINAL_DRIFT", result["reasons"])
        self.assertFalse(result["auto_evaluate"])
        with self.assertRaises(ValueError):
            GuardV16B(ids[::-1], guard.state)

    def test_accuracy_strict_boundary_and_frozen_prefix(self):
        ids = order(60)
        guard = GuardV16B(ids)
        for b in range(1, 37):
            result = guard.update(b, batch(ids[b - 1], correct=12 if b <= 24 else 9))
        self.assertIn("ACC_DROP_SELF", result["reasons"])
        self.assertEqual(result["thresholds"]["accuracy"], 0.60)

    def test_collapse_only_two_disjoint_six_block_windows(self):
        ids = order()
        guard = GuardV16B(ids)
        for b in range(1, 12):
            result = guard.update(b, batch(ids[b - 1], tokens=49, f0=100))
            self.assertNotIn("COLLAPSE_LEN", result["reasons"])
        result = guard.update(12, batch(ids[11], tokens=49))
        self.assertIn("COLLAPSE_LEN", result["reasons"])
        equal = GuardV16B(ids)
        for b in range(1, 13):
            result = equal.update(b, batch(ids[b - 1], tokens=50))
        self.assertNotIn("COLLAPSE_LEN", result["reasons"])

    def test_tail_short_hold_and_order_immediate(self):
        ids = order(8)
        guard = GuardV16B(ids)
        for b in range(1, 7):
            result = guard.update(b, batch(ids[b - 1]), tail_rate=0.1 if b <= 4 else 0.21,
                                  is_hold=b >= 5)
        self.assertEqual(set(result["reasons"]), {"TAIL_RATE", "IS_DRIFT"})
        short = GuardV16B(ids)
        short.update(1, batch(ids[0], tokens=31))
        result = short.update(2, batch(ids[1], tokens=31))
        self.assertIn("COLLAPSE_SHORT", result["reasons"])
        bad = GuardV16B(ids).update(1, batch(ids[0][::-1]))
        self.assertIn("ORDER_MISMATCH", bad["reasons"])
        with self.assertRaises(ValueError):
            GuardV16B(ids).update(2, batch(ids[0]))

    def test_count_zero_reference_is_literal_plan_not_tuned(self):
        ids = order()
        guard = GuardV16B(ids)
        for b in range(1, 37):
            result = guard.update(b, batch(ids[b - 1], cap=0, nofinal=0))
        self.assertIn("TRUNC_UP", result["reasons"])
        self.assertIn("NOFINAL_DRIFT", result["reasons"])

    def test_funding_threshold_recharge_and_no_training_timecut(self):
        price = 1.39
        threshold = price * 6.65 + 1.5
        self.assertEqual(funding_decision(threshold + .001, price, 1000000)["status"], "CONTINUE")
        first = funding_decision(threshold - .001, price, 1000000)
        self.assertEqual(first["status"], "RECHARGE_REQUESTED")
        self.assertEqual(funding_decision(5, price, 1001799, first["requested_at"])["status"], "RECHARGE_REQUESTED")
        self.assertEqual(funding_decision(5, price, 1001800, first["requested_at"])["status"], "STOP_REQUESTED")
        self.assertIsNone(funding_decision(40, price, 1001800, first["requested_at"])["requested_at"])
        self.assertFalse(first["auto_evaluate"])
        with self.assertRaises(ValueError):
            funding_decision(float("nan"), price, 1)

    def test_bootstrap_frozen_pool_receipt_unmeasured_rules_are_not_passed(self):
        rows = [{"problem_id": str(i), "source": "synthetic", "f0_prior": {
            "binding": "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD", "status": "CORRECT",
            "completion_tokens": 100, "finish_reason": "stop"}} for i in range(3000)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"items": rows}))
            a = falsealarm_receipt(path, trials=2)
            self.assertEqual(a, falsealarm_receipt(path, trials=2))
            self.assertEqual(a["pool_items"], 3000)
            self.assertEqual(a["prior_items"], 3000)
            self.assertEqual(a["status"], "FAIL")
            self.assertEqual(a["unmeasured_rules"], ["TAIL_RATE", "IS_DRIFT"])
            self.assertEqual(a["rule_rates"]["NOFINAL_DRIFT"], 1)
            altered = copy.deepcopy(rows)
            altered[0]["f0_prior"]["binding"] = "APPROXIMATE"
            path.write_text(json.dumps({"items": altered}))
            with self.assertRaises(ValueError):
                falsealarm_receipt(path, trials=1)


if __name__ == "__main__":
    unittest.main()
