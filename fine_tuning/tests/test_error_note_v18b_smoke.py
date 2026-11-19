"""CPU acceptance harness checks; fixtures are never counted as actual Qwen notes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18b_smoke as smoke
from fine_tuning.error_note.v16b_runtime import V16bScheduler, run_v16b
from fine_tuning.tests.test_error_note_v16b_runtime import fixture


class HarnessTests(unittest.TestCase):
    def test_five_gates_require_actual_gradient_save_reload_and_complete_controls(self):
        arguments = dict(cpu_receipt={"status": "PASS", "tests": 5},
            diagnostic_receipt={"accepted_n": 1, "false_acceptance": 0,
                "cases": [{"negative": True, "role": role, "problem_id": "fixture:" + suffix,
                    "status": "NO_REQUEST", "reason": "SKIP_NOTE_NO_ERROR", "accepted": False,
                    "ledgers": [{"stage": "T1", "problem_id": "fixture:" + suffix, "sha256": "a" * 64}]}
                    for role, suffix, negative in smoke.DIAGNOSTICS if negative], "n_row_ids": ["p:N"],
                "update": {"optimizer_steps": 1, "stats": {"n_parent_evidence": [{
                    "row_id": "p:N", "gradient_finite_nonzero": True, "note_gradient_retained": True,
                    "supervised_tokens": 1, "source_evidence_sha256": "a" * 64}]}}},
            diagnostic_reload={"status": "PASS", "resumed_learning": {"optimizer_steps": 1}}, resume_receipt={"status": "PASS", "resumed_slots": 1},
            monitoring_receipt={"status": "PASS"}, local_acceptance={"status": "PASS"})
        self.assertTrue(smoke.gates(**arguments)["paid_run_ready"])
        for key, value in (("cpu_receipt", {}), ("diagnostic_reload", {}),
                           ("resume_receipt", {}), ("monitoring_receipt", {})):
            changed = arguments | {key: value}
            self.assertFalse(smoke.gates(**changed)["paid_run_ready"])
        for replacement in ({"accepted_n": 0}, {"false_acceptance": 1},
                            {"update": {"optimizer_steps": 0}}, {"cases": [{"negative": True}]}):
            changed = deepcopy(arguments)
            changed["diagnostic_receipt"].update(replacement)
            self.assertFalse(smoke.gates(**changed)["paid_run_ready"])

    def test_kill_waits_for_an_active_request_after_the_committed_journal_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runtime").mkdir()
            journal = root / "runtime" / "requests.jsonl"
            prefix = json.dumps({"status": "COMPLETED", "request_id": "old"}).encode() + b"\n"
            journal.write_bytes(prefix)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"state": {"journal_bytes": len(prefix)}}))
            state = {"runtime": {"path": str(snapshot)}}
            self.assertIsNone(smoke.inflight_after_checkpoint(root, state))
            started = {"status": "STARTED", "request_id": "new", "kind": "NOTE", "block_id": 1}
            journal.write_bytes(prefix + json.dumps(started).encode() + b"\n")
            self.assertEqual(smoke.inflight_after_checkpoint(root, state), started)
            with journal.open("ab") as stream:
                stream.write(json.dumps({"status": "COMPLETED", "request_id": "new"}).encode() + b"\n")
                stream.write(b"{truncated")
            self.assertIsNone(smoke.inflight_after_checkpoint(root, state))

    def test_n_link_requires_exact_final_checkpoint_plan_cursor_and_fresh_pid(self):
        state = {"plan_sha256": "p", "checkpoint": {"path": "/a", "file_sha256": "m"},
                 "cursor": {"producer_pid": 10, "slot": 8, "consumed_row_ids": ["p:N"]}}
        receipt = {"runtime_equal": True, "greedy_equal": True, "probe_max_abs_difference": 0,
            "plan_sha256": "p", "checkpoint": state["checkpoint"], "cursor": state["cursor"],
            "producer_pid": 10, "reload_pid": 11}
        self.assertTrue(smoke.exact_campaign_reload(state, receipt, "p:N"))
        self.assertFalse(smoke.exact_campaign_reload(state, receipt, "different:N"))
        for field, changed in (("plan_sha256", "old"), ("checkpoint", {"path": "/old"}),
                               ("cursor", {"consumed_row_ids": ["p:N"]}), ("reload_pid", 10),
                               ("producer_pid", 9), ("probe_max_abs_difference", .001)):
            self.assertFalse(smoke.exact_campaign_reload(state, receipt | {field: changed}, "p:N"), field)

    def test_signals_require_same_pid_startticks_command_and_cwd(self):
        process = SimpleNamespace(pid=42, poll=lambda: None)
        identity = {"pid": 42, "startticks": "1", "cwd": "/a", "command_sha256": "x"}
        for key in ("startticks", "cwd", "command_sha256"):
            with patch.object(smoke, "process_identity", return_value=identity | {key: "changed"}), \
                    patch.object(smoke.os, "killpg") as send:
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    smoke.signal_owned(process, identity, 9)
                send.assert_not_called()

    def test_diagnostic_seed_uses_registered_v18_stream_under_the_v18b_contract(self):
        from fine_tuning.error_note.prep_v16b import make_request, preparation_stage
        from fine_tuning.error_note.protocol import TRAIN_SAMPLING_V16B
        from fine_tuning.error_note.contract import sample_seed

        item = {"problem_id": "seed-selfcheck", "image_paths": [], "image_sha256": []}
        for role in ("prep", "prep-obs", "monitor", "reasoning_monitor", "effect", "t1", "obs", "note", "resolve"):
            stage = preparation_stage(smoke.VERSION, role)
            request = make_request(item, "fixed request", stage, cap=256, sampling=TRAIN_SAMPLING_V16B,
                protocol_sha256="a" * 64, policy_sha256="b" * 64)
            self.assertEqual(request["seed"], sample_seed(item["problem_id"], stage))
        self.assertEqual(preparation_stage(smoke.VERSION, "note"), "v18-note")

    def test_cumulative_launch_publishes_two_workers_and_reaps_publication_failure(self):
        import sys
        import time
        import subprocess

        with tempfile.TemporaryDirectory() as tmp, patch.dict(smoke.os.environ,
                {"MMDL_V18B_SMOKE_DEADLINE": str(time.monotonic() + 15)}):
            root = Path(tmp)
            launch = {"workers": [], "deadline_monotonic": float(smoke.os.environ["MMDL_V18B_SMOKE_DEADLINE"])}
            for index in range(2):
                code, killed = smoke.run_child([sys.executable, "-c", "pass"], env=dict(smoke.os.environ),
                    log=root / f"child-{index}.log", launch=launch, output=root)
                self.assertEqual((code, killed), (0, None))
            self.assertEqual(len(smoke.read_sealed(root / "launch-current.json")["workers"]), 2)
            original, children = subprocess.Popen, []

            def spawn(*args, **kwargs):
                process = original(*args, **kwargs)
                children.append(process)
                return process

            with patch.object(smoke.subprocess, "Popen", side_effect=spawn), \
                    patch.object(smoke, "seal", side_effect=RuntimeError("publication failed")):
                with self.assertRaisesRegex(RuntimeError, "publication failed"):
                    smoke.run_child([sys.executable, "-c", "import time; time.sleep(30)"],
                        env=dict(smoke.os.environ), log=root / "failed.log", launch=launch, output=root)
            self.assertIsNotNone(children[0].poll())

    def test_single_deadline_is_absolute_and_has_no_per_child_reset(self):
        with patch.dict(smoke.os.environ, {"MMDL_V18B_SMOKE_DEADLINE": "1800"}), \
                patch.object(smoke.time, "monotonic", return_value=1000):
            self.assertEqual(smoke.remaining(), 800)
        with patch.dict(smoke.os.environ, {"MMDL_V18B_SMOKE_DEADLINE": "1800"}), \
                patch.object(smoke.time, "monotonic", return_value=1800):
            with self.assertRaises(TimeoutError):
                smoke.remaining()

    def test_eight_parent_campaign_enters_second_block_and_closes_delayed_notes(self):
        sampler, items, plan, callbacks, _, calls, saved = fixture(count=8)
        plan["blocks"] = [[item["problem_id"] for item in items[i:i + 2]] for i in range(0, 8, 2)]
        build, update = callbacks.rows, callbacks.update
        callbacks.rows = lambda kind, item, record: [
            row | {"row_id": item["problem_id"] + ":" + kind} for row in build(kind, item, record)]

        def consume(version, rows, contract):
            result = update(version, rows, contract)
            return result | {"consumed_row_ids": [row["row_id"] for row in rows]}

        callbacks.update = consume
        with tempfile.TemporaryDirectory() as tmp:
            runtime = V16bScheduler(sampler, plan=plan, protocol={"version": smoke.VERSION}, items=items,
                grade=None, verify=None, out=Path(tmp), callbacks=callbacks, local_sequential=True)
            result = run_v16b(None, None, None, sampler, plan, out=Path(tmp),
                protocol={"version": smoke.VERSION}, items=items, callbacks=callbacks,
                runtime=runtime, local_sequential=True)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual([version for version, *_ in calls], list(range(7)))
            self.assertEqual([row["problem_id"] for row in calls[1][1] if row["kind"] == "T1"], ["p2", "p3"])
            rows = [row["row_id"] for _, batch, *_ in calls for row in batch]
            self.assertEqual(len(rows), len(set(rows)))
            closure = runtime.terminal_closure(rows)
            self.assertEqual((closure["pending_requests"], closure["unconsumed_rows"]), (0, 0))
            self.assertFalse(saved[-1][-1])
            self.assertIn("p7:RESOLVE", rows)
            journal = [json.loads(line) for line in (Path(tmp) / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["status"] == "COMPLETED" for row in journal), 32)


if __name__ == "__main__":
    unittest.main()
