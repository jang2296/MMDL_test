"""plan_v4: one round's chain spec from validated params; nothing is executed."""
import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import plan_v4
from fine_tuning.modules import BASE_REVISION

BASE = f"/vol/models/{BASE_REVISION}"


def params(**change):
    value = {"version": plan_v4.PARAMS_VERSION_V4, "commit": "a" * 40, "round": 1, "start": 0, "limit": 500,
             "arm": "hybrid", "lesson_control": True, "max_num_batched_tokens": 16384, "token_budget": 24000,
             "gradient_checkpointing": "on", "timeouts": {"collect": 100, "train": 200, "reload": 30, "export": 40},
             "python": "/venv/bin/python", "repo_root": "/work/MMDL_test", "build_root": "/vol/build",
             "base_model_path": BASE, "rollout_model_path": BASE, "out_root": "/vol/out", "volume_root": "/vol",
             "order_path": "/vol/private/order.json", "training_rules_path": "/vol/private/rules.json",
             "order_sha256": "b" * 64, "training_rules_sha256": "c" * 64, "init_run": None,
             "reuse_t1_path": None, "reuse_t1_sha256": None}
    return value | change


class PlanV4(unittest.TestCase):
    def test_round_one_steps(self):
        spec = plan_v4.plan_v4(params())
        names = [step["name"] for step in spec["steps"]]
        self.assertEqual(names, ["pin-order", "pin-training-rules", "rollout", "note", "selfcheck", "resolve", "rows",
                                 "train", "reload", "export"])
        rollout = spec["steps"][names.index("rollout")]["argv"]
        self.assertIn("--prefix-caching", rollout)
        self.assertIn("--loop-stop", rollout)
        self.assertIn("--lesson-control", spec["steps"][names.index("resolve")]["argv"])
        train = spec["steps"][names.index("train")]["argv"]
        self.assertNotIn("--init-run", train)
        self.assertEqual(spec["meta"]["total_timeout_seconds"], 2 * 100 + 5 * 100 + 200 + 30 + 40)
        self.assertEqual(spec, plan_v4.plan_v4(params()))

    def test_later_round_and_reuse(self):
        spec = plan_v4.plan_v4(params(round=2, start=500, limit=3000, init_run="/vol/out/train-r1-hybrid",
                                      rollout_model_path="/vol/eval-data/proof/export-v4-r1-hybrid/model"))
        train = next(step for step in spec["steps"] if step["name"] == "train")["argv"]
        self.assertEqual(train[train.index("--init-run") + 1], "/vol/out/train-r1-hybrid")
        reused = plan_v4.plan_v4(params(reuse_t1_path="/vol/private/t1.jsonl", reuse_t1_sha256="d" * 64))
        self.assertEqual([s["name"] for s in reused["steps"]][2], "pin-reuse-t1")

    def test_refusals(self):
        bad = [dict(version="x"), dict(commit="A" * 40), dict(round=2), dict(init_run="/vol/out/r1"),
               dict(round=2, init_run="/vol/out/r1"), dict(order_path="/work/MMDL_test/order.json"),
               dict(base_model_path="/vol/models/other", rollout_model_path="/vol/models/other"),
               dict(reuse_t1_path="/vol/t1.jsonl"), dict(arm="note"), dict(lesson_control=1),
               dict(timeouts={"collect": 1}), dict(limit=0), dict(extra=1)]
        for change in bad:
            with self.assertRaises(ValueError, msg=str(change)):
                plan_v4.plan_v4(params(**change))

    def test_cli_writes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / "p.json", Path(tmp) / "steps.json"
            path.write_text(json.dumps(params()))
            self.assertEqual(plan_v4.main(["plan", "--params", str(path), "--output", str(out)]), 0)
            self.assertEqual(plan_v4.main(["plan", "--params", str(path), "--output", str(out)]), 2)


if __name__ == "__main__":
    unittest.main()
