"""Unit tests for fine_tuning.error_note.plan (stdlib unittest; the planner executes nothing)."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import plan as planner
from fine_tuning.error_note import single_campaign
from fine_tuning.notebook_rl.pod import chain
from fine_tuning.tests.test_error_note_single_plan import params as single_params

BASE = {
    "version": "error-note-plan-params-v1",
    "commit": "a" * 40,
    "python": "/volume/venv-0123456789abcdef/bin/python",
    "repo_root": "/volume/clone",
    "build_root": "/volume/error-note/build",
    "model_path": "/volume/hf-cache/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "out_root": "/volume/error-note/out",
    "order_sha256": "b" * 64,
    "limit": 300,
    "pilot_limit": 0,
    "seqs": [1, 2, 4],
    "sweep_seconds": 600,
    "kv_cache_gib": 8.5,
    "max_num_batched_tokens": 8192,
    "token_budget": 4096,
    "gradient_checkpointing": "on",
    "arms": ["note", "solution"],
    "timeouts": {"collect": 3600, "train": 7200, "reload": 900, "export": 900},
}
WITHOUT_PILOT = ["pin-order", "sweep", "note", "rows", "train-note", "reload-note",
                 "train-solution", "reload-solution"]
WITH_PILOT = ["pin-order", "pilot-solve", "pilot-note", "pilot-rows", "pilot-train", "pilot-reload",
              "pilot-export", "pilot-ft-check"] + WITHOUT_PILOT[1:]


def params(**changes: object) -> dict:
    value = copy.deepcopy(BASE)
    value.update(changes)
    return value


def run_cli(*args: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = planner.main(list(args))
    return code, out.getvalue() + err.getvalue()


class ErrorNotePlanTests(unittest.TestCase):
    def test_steps_order_without_pilot(self) -> None:
        spec = planner.plan("note-300", params())
        self.assertEqual([step["name"] for step in spec["steps"]], WITHOUT_PILOT)

    def test_steps_order_with_pilot(self) -> None:
        spec = planner.plan("note-3417", params(pilot_limit=20, limit=3417))
        self.assertEqual([step["name"] for step in spec["steps"]], WITH_PILOT)
        steps = {step["name"]: step for step in spec["steps"]}
        ft_check = steps["pilot-ft-check"]["argv"]
        self.assertEqual(ft_check[ft_check.index("--model-path") + 1], BASE["out_root"] + "/pilot-export/model")
        self.assertEqual(ft_check[ft_check.index("--limit") + 1], "2")

    def test_every_step_has_timeout_and_argv_strings(self) -> None:
        spec = planner.plan("note-3417", params(pilot_limit=20))
        for step in spec["steps"]:
            with self.subTest(step=step["name"]):
                self.assertIsInstance(step["timeout_seconds"], int)
                self.assertGreaterEqual(step["timeout_seconds"], 1)
                self.assertTrue(step["argv"])
                self.assertTrue(all(isinstance(arg, str) and arg for arg in step["argv"]))
        steps = {step["name"]: step for step in spec["steps"]}
        self.assertEqual(steps["train-note"]["timeout_seconds"], 7200)
        self.assertEqual(steps["reload-solution"]["timeout_seconds"], 900)
        self.assertEqual(steps["pilot-export"]["timeout_seconds"], 900)
        self.assertEqual(steps["sweep"]["timeout_seconds"], 3600)

    def test_bad_params_refused(self) -> None:
        cases = {
            "missing key": lambda p: p.pop("limit"),
            "unknown key": lambda p: p.__setitem__("extra", 1),
            "short commit": lambda p: p.__setitem__("commit", "a" * 39),
            "relative path": lambda p: p.__setitem__("out_root", "relative/out"),
            "empty arms": lambda p: p.__setitem__("arms", []),
            "unknown arm": lambda p: p.__setitem__("arms", ["note", "bogus"]),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                changed = params()
                change(changed)
                with self.assertRaises(ValueError):
                    planner.plan("note-300", changed)
        with self.assertRaises(ValueError):
            planner.plan("eval-ft", params())

    def test_auto_kv_and_seqs_pass_through(self) -> None:
        spec = planner.plan("note-3417", params(kv_cache_gib="auto", seqs="auto", pilot_limit=20))
        self.assertIs(chain.check_spec(spec), spec)
        for step in spec["steps"]:
            argv = step["argv"]
            if "--kv-cache-gib" in argv:
                self.assertEqual(argv[argv.index("--kv-cache-gib") + 1], "auto", step["name"])
        sweep = {step["name"]: step for step in spec["steps"]}["sweep"]["argv"]
        self.assertEqual(sweep[sweep.index("--seqs") + 1], "auto")
        fixed = {step["name"]: step for step in planner.plan("note-300", params())["steps"]}["sweep"]["argv"]
        self.assertEqual((fixed[fixed.index("--seqs") + 1], fixed[fixed.index("--kv-cache-gib") + 1]), ("1,2,4", "8.5"))

    def test_bad_kv_and_seqs_still_refused(self) -> None:
        cases = {
            "kv text": {"kv_cache_gib": "fast"}, "kv upper auto": {"kv_cache_gib": "AUTO"}, "kv zero": {"kv_cache_gib": 0},
            "kv negative": {"kv_cache_gib": -1}, "kv bool": {"kv_cache_gib": True}, "kv nan": {"kv_cache_gib": math.nan},
            "kv list": {"kv_cache_gib": ["auto"]}, "seqs text": {"seqs": "fast"}, "seqs upper auto": {"seqs": "AUTO"},
            "seqs csv string": {"seqs": "1,2"}, "seqs empty": {"seqs": []}, "seqs zero": {"seqs": [0]},
            "seqs bool": {"seqs": [True]}, "seqs float": {"seqs": [1.5]}, "seqs mixed": {"seqs": [4, "auto"]},
        }
        for name, changes in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                planner.plan("note-300", params(**changes))

    def test_same_params_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params_path = root / "params.json"
            params_path.write_text(json.dumps(params(pilot_limit=20)), encoding="utf-8")
            printed = []
            for name in ("first.json", "second.json"):
                code, message = run_cli("plan", "--stage", "note-3417", "--params", str(params_path),
                                        "--output", str(root / name))
                self.assertEqual(code, 0)
                printed.append(message.strip())
            first = (root / "first.json").read_bytes()
            self.assertEqual(first, (root / "second.json").read_bytes())
            self.assertEqual(printed, [hashlib.sha256(first).hexdigest()] * 2)

    def test_existing_output_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params_path = root / "params.json"
            params_path.write_text(json.dumps(params()), encoding="utf-8")
            output = root / "steps.json"
            output.write_text("keep", encoding="utf-8")
            code, message = run_cli("plan", "--stage", "note-300", "--params", str(params_path),
                                    "--output", str(output))
            self.assertEqual(code, 2)
            self.assertIn("REFUSED", message)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_spec_passes_chain_check_spec(self) -> None:
        for stage, changes in (("note-300", {}), ("note-3417", {"pilot_limit": 20, "limit": 3417})):
            with self.subTest(stage=stage):
                spec = planner.plan(stage, params(**changes))
                self.assertIs(chain.check_spec(spec), spec)
                self.assertEqual(spec["meta"]["stage"], stage)
                self.assertEqual(spec["meta"]["total_timeout_seconds"],
                                 sum(step["timeout_seconds"] for step in spec["steps"]))


class NoteV2PlanTests(unittest.TestCase):
    COLLECT = "fine_tuning.error_note.collect"
    TRAIN = "fine_tuning.error_note.train"
    PROOF = "fine_tuning.notebook_rl.proof_eval"
    ORDER = "/volume/clone/fine_tuning/error_note/item_order_v2.json"
    COLLECT_DIR = "/volume/error-note/out/collect"
    RUN = "/volume/error-note/out/train-note"
    EXPORT = "/volume/eval-data/proof/export-note-v2"
    TIMEOUTS = {"collect": 111, "train": 222, "reload": 333, "export": 444}
    SWEEP_NAMES = ["pin-order", "sweep", "note", "resolve", "rows", "train-note", "reload-note", "export"]
    REUSE_NAMES = ["pin-order", "pin-t1", "note", "resolve", "rows", "train-note", "reload-note", "export"]

    @staticmethod
    def v2_params(**changes: object) -> dict:
        value = copy.deepcopy(BASE)
        del value["pilot_limit"]
        del value["arms"]
        value.update(version="error-note-plan-params-v2", volume_root="/volume", reuse_t1_sha256=None)
        value.update(changes)
        return value

    @staticmethod
    def value_after(argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def assert_shared_steps(self, steps: dict[str, dict]) -> None:
        pin = steps["pin-order"]["argv"]
        self.assertEqual(self.value_after(pin, "--file"), self.ORDER)
        self.assertEqual(self.value_after(pin, "--sha256"), "b" * 64)
        self.assertEqual(self.value_after(steps["note"]["argv"], "--note-version"), "v2")
        self.assertEqual(steps["note"]["expect_files"],
                         [f"{self.COLLECT_DIR}/note.jsonl", f"{self.COLLECT_DIR}/lessons.json"])
        self.assertEqual(steps["resolve"]["expect_files"], [f"{self.COLLECT_DIR}/receipt-resolve.json"])
        self.assertEqual(self.value_after(steps["rows"]["argv"], "--rows-version"), "v2")
        train = steps["train-note"]["argv"]
        self.assertEqual(self.value_after(train, "--protocol-version"), "v2")
        self.assertEqual(self.value_after(train, "--arm"), "note")
        self.assertIn("--resume", train)
        self.assertEqual(self.value_after(steps["reload-note"]["argv"], "--protocol-version"), "v2")
        export = steps["export"]["argv"]
        self.assertEqual(export[1:4], ["-m", self.PROOF, "export"])
        self.assertEqual(self.value_after(export, "--output"), self.EXPORT)
        self.assertEqual(steps["export"]["expect_files"],
                         [f"{self.EXPORT}/model_ref.json", f"{self.EXPORT}/model/export_manifest.json"])

    def assert_collect_and_train_order(self, spec: dict, names: list[str]) -> None:
        seen = []
        for step in spec["steps"]:
            argv = step["argv"]
            if argv[1:3] in (["-m", self.COLLECT], ["-m", self.TRAIN]) and argv[3] != "reload":
                seen.append(step["name"])
                self.assertEqual(self.value_after(argv, "--order"), self.ORDER, step["name"])
        self.assertEqual(seen, names)
        self.assertNotIn("item_order.json", json.dumps(spec))

    def test_step_names_without_reuse(self) -> None:
        spec = planner.plan("note-v2", self.v2_params())
        self.assertEqual([step["name"] for step in spec["steps"]], self.SWEEP_NAMES)

    def test_step_names_with_reuse(self) -> None:
        spec = planner.plan("note-v2", self.v2_params(reuse_t1_sha256="c" * 64))
        self.assertEqual([step["name"] for step in spec["steps"]], self.REUSE_NAMES)

    def test_argv_without_reuse(self) -> None:
        spec = planner.plan("note-v2", self.v2_params())
        steps = {step["name"]: step for step in spec["steps"]}
        self.assert_shared_steps(steps)
        self.assert_collect_and_train_order(spec, ["sweep", "note", "resolve", "rows", "train-note"])
        self.assertEqual(self.value_after(steps["note"]["argv"], "--max-num-seqs-from"),
                         f"{self.COLLECT_DIR}/sweep.json")
        self.assertEqual(self.value_after(steps["resolve"]["argv"], "--max-num-seqs-from"),
                         f"{self.COLLECT_DIR}/sweep.json")
        self.assertEqual(steps["sweep"]["expect_files"], [f"{self.COLLECT_DIR}/sweep.json"])

    def test_argv_with_reuse(self) -> None:
        spec = planner.plan("note-v2", self.v2_params(reuse_t1_sha256="c" * 64))
        steps = {step["name"]: step for step in spec["steps"]}
        self.assert_shared_steps(steps)
        self.assert_collect_and_train_order(spec, ["note", "resolve", "rows", "train-note"])
        self.assertEqual(self.value_after(steps["pin-t1"]["argv"], "--file"), f"{self.COLLECT_DIR}/t1.jsonl")
        self.assertEqual(self.value_after(steps["pin-t1"]["argv"], "--sha256"), "c" * 64)
        for name in ("note", "resolve"):
            with self.subTest(step=name):
                argv = steps[name]["argv"]
                self.assertEqual(self.value_after(argv, "--max-num-seqs"), "auto")
                self.assertNotIn("--max-num-seqs-from", argv)

    def test_timeout_kind_per_step(self) -> None:
        tail = {"train-note": 222, "reload-note": 333, "export": 444}
        collect = {"pin-order": 111, "note": 111, "resolve": 111, "rows": 111}
        for changes, first in (({}, {"sweep": 111}), ({"reuse_t1_sha256": "c" * 64}, {"pin-t1": 111})):
            with self.subTest(changes=changes):
                spec = planner.plan("note-v2", self.v2_params(timeouts=self.TIMEOUTS, **changes))
                timeouts = {step["name"]: step["timeout_seconds"] for step in spec["steps"]}
                self.assertEqual(timeouts, {**collect, **first, **tail})

    def test_spec_passes_chain_check_spec(self) -> None:
        for changes in ({}, {"reuse_t1_sha256": "c" * 64}):
            with self.subTest(changes=changes):
                spec = planner.plan("note-v2", self.v2_params(**changes))
                self.assertIs(chain.check_spec(spec), spec)
                self.assertEqual(spec["meta"]["stage"], "note-v2")
                self.assertEqual(spec["meta"]["job_stage"], "note-v2")

    def test_same_params_same_json_bytes(self) -> None:
        first = json.dumps(planner.plan("note-v2", self.v2_params()), sort_keys=True)
        second = json.dumps(planner.plan("note-v2", self.v2_params()), sort_keys=True)
        self.assertEqual(first, second)

    def test_v2_refusals(self) -> None:
        cases = {
            "v1 version": lambda p: p.__setitem__("version", "error-note-plan-params-v1"),
            "pilot_limit present": lambda p: p.__setitem__("pilot_limit", 0),
            "arms present": lambda p: p.__setitem__("arms", ["note"]),
            "volume_root missing": lambda p: p.pop("volume_root"),
            "volume_root relative": lambda p: p.__setitem__("volume_root", "relative/path"),
            "reuse short sha": lambda p: p.__setitem__("reuse_t1_sha256", "abc"),
            "reuse not str": lambda p: p.__setitem__("reuse_t1_sha256", 5),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                changed = self.v2_params()
                change(changed)
                with self.assertRaises(ValueError):
                    planner.plan("note-v2", changed)
        with self.assertRaises(ValueError):
            planner.plan("note-300", self.v2_params())

    def test_note_v2_registered_and_cli_writes_spec(self) -> None:
        from fine_tuning.notebook_rl.pod import job as pod_job

        self.assertIn("note-v2", pod_job.TRAINING_STAGES)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params_path = root / "params.json"
            params_path.write_text(json.dumps(self.v2_params()), encoding="utf-8")
            output = root / "steps.json"
            code, message = run_cli("plan", "--stage", "note-v2", "--params", str(params_path),
                                    "--output", str(output))
            self.assertEqual(code, 0, message)
            self.assertEqual(message.strip(), hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["meta"]["stage"], "note-v2")


class SingleSupplementPlanTests(unittest.TestCase):
    """v11: the note-reference supplement is a fifth pinned campaign input of every single stage."""

    PATH, SHA = "/workspace/private/note-reference-supplement.json", "9" * 64

    def params(self, stage: str, **changes: object) -> dict:
        return single_params(stage, **({"note_reference_supplement_path": self.PATH,
                                        "note_reference_supplement_sha256": self.SHA} | changes))

    def test_every_campaign_call_carries_the_supplement_flags_and_a_pin_step(self) -> None:
        for stage in planner.SINGLE_STAGES:
            with self.subTest(stage=stage):
                steps = planner.plan(stage, self.params(stage))["steps"]
                pin = next(step for step in steps if step["name"] == "pin-note-reference-supplement")
                self.assertEqual(pin["argv"][3:], ["pin", "--file", self.PATH, "--sha256", self.SHA])
                campaigns = [step["argv"] for step in steps if step["argv"][2] == planner.CAMPAIGN]
                self.assertTrue(campaigns)
                for argv in campaigns:
                    self.assertEqual(argv[argv.index("--note-reference-supplement") + 1], self.PATH)
                    self.assertEqual(argv[argv.index("--note-reference-supplement-sha256") + 1], self.SHA)
                    parsed = single_campaign.build_parser().parse_args(argv[3:])
                    self.assertEqual((str(parsed.note_reference_supplement), parsed.note_reference_supplement_sha256),
                                     (self.PATH, self.SHA))

    def test_params_without_or_with_a_malformed_supplement_are_refused(self) -> None:
        for stage in planner.SINGLE_STAGES:
            full = self.params(stage)
            for key in ("note_reference_supplement_path", "note_reference_supplement_sha256"):
                with self.subTest(stage=stage, missing=key), self.assertRaisesRegex(ValueError, "missing"):
                    planner.plan(stage, {name: value for name, value in full.items() if name != key})
            for changes, message in (({"note_reference_supplement_sha256": "bad"}, "64 lowercase hex"),
                                     ({"note_reference_supplement_path": "relative.json"}, "canonical absolute path"),
                                     ({"note_reference_supplement_path": full["repo_root"] + "/s.json"}, "outside repo_root")):
                with self.subTest(stage=stage, changes=changes), self.assertRaisesRegex(ValueError, message):
                    planner.plan(stage, self.params(stage, **changes))


if __name__ == "__main__":
    unittest.main()
