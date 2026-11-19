"""Readiness requires completed evidence; fixtures do not certify a model run."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from fine_tuning.error_note import v18b_smoke as smoke
from fine_tuning.error_note.prep_v16b import file_sha, seal


def controls():
    return [{"role": role, "problem_id": "fixture:" + suffix, "negative": True,
             "accepted": False, "status": "NO_REQUEST", "reason": "SKIP_NOTE_NO_ERROR",
             "ledgers": [{"stage": "T1", "problem_id": "fixture:" + suffix,
                          "sha256": "a" * 64}]}
            for role, suffix, negative in smoke.DIAGNOSTICS if negative]


def arguments():
    return dict(cpu_receipt={"status": "PASS", "tests": 1},
                diagnostic_receipt={"cases": controls(), "false_acceptance": 0},
                diagnostic_reload={}, resume_receipt={"status": "PASS", "resumed_slots": 1},
                monitoring_receipt={"status": "PASS"}, campaign_n_linked=1)


class AcceptanceTests(unittest.TestCase):
    def test_independent_gates_cannot_certify_an_unchecked_run(self):
        result = smoke.gates(**arguments())
        self.assertTrue(all(gate["status"] == "PASS" for gate in result["gates"].values()))
        self.assertFalse(result["paid_run_ready"])

    def test_controls_need_explicit_zero_and_both_audited_terminal_results(self):
        valid = arguments() | {"local_acceptance": {"status": "PASS"}}
        result = smoke.gates(**valid)
        self.assertTrue(result["paid_run_ready"])
        self.assertTrue(result["gates"]["negative_false_acceptance"]["coverage_limited"])
        mutations = [lambda receipt: receipt.pop("false_acceptance"),
                     lambda receipt: receipt.update(false_acceptance=False),
                     lambda receipt: receipt["cases"][0].pop("accepted"),
                     lambda receipt: receipt["cases"][0].update(status="STARTED"),
                     lambda receipt: receipt["cases"][0].update(reason="SKIP_NOTE_INCOMPLETE_WITHOUT_VERIFIED_ERROR"),
                     lambda receipt: receipt["cases"][0].update(ledgers=[]),
                     lambda receipt: receipt["cases"].__setitem__(1, deepcopy(receipt["cases"][0]))]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = deepcopy(valid)
                mutate(changed["diagnostic_receipt"])
                self.assertFalse(smoke.gates(**changed)["paid_run_ready"])

    def fixture(self, root):
        campaign = root / "campaign"
        (campaign / "update-results").mkdir(parents=True)
        (root / "diagnostic").mkdir()
        checkpoint = campaign / "checkpoint-metadata.json"
        checkpoint.write_text("checkpoint fixture")
        descriptor = {"path": str(checkpoint), "file_sha256": file_sha(checkpoint)}
        plan = seal(campaign / "campaign-plan.json", {"local_sequential": True, "total_slots": 8})
        before = {"slot": 3, "rl_block": 2, "producer_pid": 10, "consumed_row_ids": ["p:RL"]}
        cursor = {"slot": 8, "rl_block": 4, "producer_pid": 11, "consumed_row_ids": ["p:RL", "p:N"]}
        state = {"status": "COMPLETE", "plan_sha256": plan["sha256"], "cursor": cursor,
                 "checkpoint": descriptor, "stop_reasons": []}
        resume = {"status": "PASS", "resumed_slots": 5, "duplicates": 0,
                  "before_cursor": before, "after_cursor": cursor}
        reload = {"runtime_equal": True, "greedy_equal": True, "probe_max_abs_difference": 0,
                  "plan_sha256": plan["sha256"], "checkpoint": descriptor, "cursor": cursor,
                  "producer_pid": 11, "reload_pid": 12}
        run = {"status": "COMPLETE", "error": None, "wall_seconds": 1700, "deadline_seconds": 1800}
        files = {"campaign/campaign-state.json": state, "resume-receipt.json": resume,
                 "campaign/reload-8.json": reload, "run-receipt.json": run,
                 "cpu.json": {"status": "PASS", "tests": 1}, "monitor.json": {"status": "PASS"},
                 "sigkill-receipt.json": {"state": {"cursor": before}, "identity": {"pid": 10},
                    "after_main_blocks": 2, "actual_inflight_request": {"status": "STARTED"}},
                 "diagnostic/diagnostic-receipt.json": {"cases": controls(), "false_acceptance": 0},
                 "campaign/update-results/slot-7.json": {"result": {"optimizer_steps": 1,
                    "stats": {"n_parent_evidence": [{"problem_id": "p", "row_id": "p:N",
                        "gradient_finite_nonzero": True, "note_gradient_retained": True,
                        "supervised_tokens": 1, "source_evidence_sha256": "a" * 64}]}}}}
        return files, SimpleNamespace(output=root, cpu_receipt=root / "cpu.json",
                                      monitoring_receipt=root / "monitor.json")

    def summarize(self, files, args):
        for name, data in files.items():
            seal(args.output / name, data)
        return smoke.summarize(args)

    def test_supervisor_failure_cannot_be_hidden_by_independent_gate_success(self):
        for replacement in ({"error": "worker failed", "status": "PARTIAL"},
                            {"status": "PARTIAL"}, {"wall_seconds": 1801}):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                files, args = self.fixture(Path(tmp))
                files["run-receipt.json"].update(replacement)
                result = self.summarize(files, args)
                self.assertFalse(result["paid_run_ready"])
                self.assertTrue(all(g["status"] == "PASS" for g in result["gates"].values()))

    def test_complete_acceptance_requires_the_exact_final_checkpoint_reload(self):
        for fault in (None, "reload", "checkpoint", "resume"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                files, args = self.fixture(Path(tmp))
                if fault == "reload":
                    files["campaign/reload-8.json"]["checkpoint"] = {"path": "stale"}
                elif fault == "checkpoint":
                    (args.output / "campaign/checkpoint-metadata.json").unlink()
                elif fault == "resume":
                    files["resume-receipt.json"]["after_cursor"] = {"slot": 7}
                result = self.summarize(files, args)
                self.assertEqual(result["paid_run_ready"], fault is None)

    def test_intentional_time_reserve_stop_can_pass_without_full_eight_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, args = self.fixture(Path(tmp))
            state = files["campaign/campaign-state.json"]
            state.update(status="STOPPED", stop_reasons=["STOP_REQUESTED"])
            state["cursor"]["slot"] = 4
            files["resume-receipt.json"]["resumed_slots"] = 1
            files["run-receipt.json"]["status"] = "PARTIAL"
            (args.output / "campaign/STOP_REQUESTED").write_text("reserve final three minutes")
            result = self.summarize(files, args)
            self.assertTrue(result["paid_run_ready"])
            self.assertEqual(result["local_acceptance"]["completion"], "BOUNDED_PARTIAL")
            self.assertEqual(result["local_acceptance"]["remaining_slots"], 4)

    def test_latest_recovery_is_selected_numerically_without_requiring_old_failure_to_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, args = self.fixture(Path(tmp))
            files["recovery-run-receipt-9.json"] = {"status": "PARTIAL", "error": "old failure"}
            files["recovery-run-receipt-10.json"] = deepcopy(files["run-receipt.json"])
            files["run-receipt.json"].update(status="PARTIAL", error="original failure")
            result = self.summarize(files, args)
            self.assertTrue(result["paid_run_ready"])
            self.assertEqual(result["run"]["status"], "COMPLETE")
