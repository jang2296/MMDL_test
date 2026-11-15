"""The v16b plan has no elapsed training timeout and cannot evaluate a stopped campaign."""
import copy
import unittest

from fine_tuning.error_note import plan as planner
from fine_tuning.modules import BASE_REVISION
from fine_tuning.notebook_rl.pod.chain import check_spec


def params():
    value = {"version": planner.PARAMS_VERSION_V16B, "commit": "a" * 40,
             "gpu_type": "NVIDIA A100-SXM4-80GB", "hourly_usd": 1.39,
             "timeouts": {"cpu": 600, "reload": 1800, "export": 1800, "evaluate": 23400, "compass": 1800}}
    value.update({key: f"/private/{key}" for key in planner.V16B_PATHS})
    value["repo_root"] = "/repo"
    value["out_root"] = "/private/run_dir/output"
    value["model_path"] = f"/cache/{BASE_REVISION}"
    value.update({key + "_sha256": "b" * 64 for key in planner.V16B_HASHED})
    return value


class V16BPlanTests(unittest.TestCase):
    def test_full_plan_has_separate_eval_watchdog_and_no_training_cut(self):
        spec = planner.plan("v16b", params())
        check_spec(spec)
        steps = {s["name"]: s for s in spec["steps"]}
        self.assertIsNone(spec["meta"]["total_timeout_seconds"])
        self.assertIsNone(spec["meta"]["training_elapsed_cutoff"])
        self.assertFalse(spec["meta"]["guard_stop_auto_evaluate"])
        self.assertNotIn("timeout_seconds", steps["campaign-advance"])
        self.assertNotIn("timeout_seconds", steps["prep-generate"])
        self.assertEqual(steps["evaluate-900"]["timeout_seconds"], 23400)
        names = list(steps)
        self.assertLess(names.index("require-complete"), names.index("export"))
        self.assertLess(names.index("require-complete"), names.index("evaluate-900"))
        self.assertLess(names.index("guard-falsealarm-gate"), names.index("prep-generate"))
        prereg = steps["preregister-v11"]["argv"]
        self.assertEqual(prereg[prereg.index("--format") + 1], "v11")
        self.assertIn("--all-subset", prereg)
        self.assertNotIn("--half-a-subset", prereg)
        argv = steps["evaluate-900"]["argv"]
        self.assertEqual(argv.count("--sample-id"), 900)
        self.assertIn("MMDL_EVAL_SLOTS=auto", argv)
        self.assertIn("VLLM_ENABLE_V1_MULTIPROCESSING=0", argv)
        rows = steps["prep-rows"]["argv"]
        self.assertEqual(rows[rows.index("--model-path") + 1], params()["model_path"])
        self.assertIn("compass-separate", steps)
        self.assertIn("pack", spec["meta"]["after_run_closed"])
        self.assertEqual(spec["meta"]["funding"]["post_train_hours"], 5.65)
        self.assertEqual(spec["meta"]["funding"]["next_train_hours"], 1)
        self.assertEqual(spec["meta"]["funding"]["recharge_wait_seconds"], 1800)

    def test_bad_gpu_training_timeout_private_path_and_revision_refused(self):
        cases = [{"gpu_type": "NVIDIA A40"}, {"gpu_type": "NVIDIA H100 80GB HBM3"}, {"out_root": "/repo/private"},
                 {"model_path": "/model/latest"}, {"hourly_usd": float("inf")}, {"commit": "main"}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                planner.plan("v16b", params() | change)
        pcie = planner.plan("v16b", params() | {"gpu_type": "NVIDIA A100 80GB PCIe"})
        self.assertIn("evaluate-900", [s["name"] for s in pcie["steps"]])
        for key, value in (("advance", 50000), ("evaluate", 50000)):
            altered = copy.deepcopy(params())
            altered["timeouts"][key] = value
            with self.assertRaises(ValueError):
                planner.plan("v16b", altered)


if __name__ == "__main__":
    unittest.main()
