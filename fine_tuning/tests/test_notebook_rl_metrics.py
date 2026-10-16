import tempfile
import unittest
from pathlib import Path

from fine_tuning.notebook_rl.metrics import append_record, health_check, step_record
from fine_tuning.notebook_rl.schema import load_protocol


def record(step=1, **kw):
    return step_record(step=step, policy_version=step, n_problems=8, n_discarded=0,
        n_groups_by_kind={"T1": 8}, reward_mean_by_kind={"T1": .5}, degenerate_fraction=0,
        kl_mean=kw.get("kl_mean", 0), masked_fraction=kw.get("masked_fraction", 0), mean_abs_lp_diff=0,
        grad_norm={"L": 1, "M": 1, "V": None}, vision_on_fraction=0, len_median=kw.get("len_median", 100),
        truncation_rate=kw.get("truncation_rate", 0), short_fraction_le32=kw.get("short_fraction_le32", 0),
        unverifiable_fraction=kw.get("unverifiable_fraction", 0),
        dev_first_attempt_acc=kw.get("dev_first_attempt_acc"),
        seconds=dict.fromkeys(("rollout", "old", "ref", "train", "push"), 1))


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(Path(__file__).parents[1]/"notebook_rl/protocol_v1.yaml")
        self.f0 = {"len_median": 100, "dev_first_attempt_acc": .6}

    def test_threshold_boundaries_and_consecutive_steps(self):
        for key, threshold, reason in (("kl_mean", .1, "KL"), ("truncation_rate", .3, "TRUNCATION"),
                                      ("masked_fraction", .05, "MISMATCH")):
            self.assertEqual(health_check([record(i, **{key: threshold}) for i in range(1, 4)], self.protocol, self.f0), [])
            self.assertEqual(health_check([record(i, **{key: threshold+.001}) for i in range(1, 3)], self.protocol, self.f0), [])
            self.assertIn(reason, health_check([record(i, **{key: threshold+.001}) for i in range(1, 4)], self.protocol, self.f0))

    def test_immediate_limits_and_periodic_dev(self):
        for values, reason in (({"len_median": 29.99}, "LENGTH_COLLAPSE"),
                               ({"short_fraction_le32": .2001}, "LENGTH_COLLAPSE"),
                               ({"unverifiable_fraction": .0101}, "VERIFIER")):
            self.assertIn(reason, health_check([record(**values)], self.protocol, self.f0))
        history = [record(i, dev_first_attempt_acc=.54 if i in (12, 24) else None) for i in range(1, 25)]
        self.assertIn("DEV_DROP", health_check(history, self.protocol, self.f0))
        history[-1]["dev_first_attempt_acc"] = .56
        self.assertNotIn("DEV_DROP", health_check(history, self.protocol, self.f0))

    def test_no_missing_nonfinite_or_fake_reference(self):
        with self.assertRaises(ValueError):
            step_record(step=1)
        with self.assertRaises(ValueError):
            record(kl_mean=float("nan"))
        with self.assertRaises(ValueError):
            health_check([record()], self.protocol, {})
        with self.assertRaises(ValueError):
            health_check([record(1), record(3)], self.protocol, self.f0)

    def test_append_preserves_complete_lines_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"metrics.jsonl"
            append_record(path, record(1))
            first = path.read_bytes()
            append_record(path, record(2))
            self.assertTrue(path.read_bytes().startswith(first))
            with self.assertRaises(ValueError):
                append_record(path, record(2))
            path.write_bytes(path.read_bytes()+b"{")
            with self.assertRaises(ValueError):
                append_record(path, record(3))


if __name__ == "__main__":
    unittest.main()
