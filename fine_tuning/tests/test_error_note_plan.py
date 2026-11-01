"""Unit tests for fine_tuning.error_note.plan (stdlib unittest; the planner executes nothing)."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from fine_tuning.error_note import plan as planner
from fine_tuning.notebook_rl.pod import chain

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


if __name__ == "__main__":
    unittest.main()
