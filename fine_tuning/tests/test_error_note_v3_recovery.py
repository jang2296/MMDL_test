"""A completed v3 chain is recovery evidence without a fabricated result/throughput receipt."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fine_tuning.notebook_rl.pod import chain, collect, common, job
from fine_tuning.tests.test_notebook_rl_pod import approved_job, make_run_dir


class V3ChainRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="note-v3-recovery-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def completed(self, name: str, *, stage: str = "note-v3-collect") -> tuple[Path, Path, Path]:
        run = make_run_dir(self.root, name, result=False)
        output = run / ("collect/note.jsonl" if stage.endswith("collect") else "train-note/checkpoint-final/metadata.json")
        script = f"from pathlib import Path; p=Path({str(output)!r}); p.parent.mkdir(parents=True); p.write_text('synthetic output')"
        spec = {"version": chain.SPEC_VERSION, "meta": {"job_stage": stage}, "steps": [
            {"name": "stage", "argv": [sys.executable, "-c", script], "expect_files": [str(output)]}]}
        source = self.root / f"{name}.spec.json"
        source.write_text(json.dumps(spec))
        spec_sha = common.sha256_file(source)
        recorded = approved_job(None, stage=stage, run_dir=str(run), main=[
            sys.executable, "-m", "fine_tuning.notebook_rl.pod.chain", "run", "--spec", str(source), "--spec-sha256", spec_sha])
        (run / "job.json").write_text(json.dumps(recorded))
        with mock.patch.dict(os.environ, {"MMDL_RUN_DIR": str(run), "MMDL_JOB_STAGE": stage}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(chain.run(source, spec_sha), 0)
        return run, source, output

    def test_successful_collect_and_train_chain_pack_all_outputs_and_the_pinned_spec(self) -> None:
        for stage in ("note-v3-collect", "note-v3-train"):
            run, source, output = self.completed(stage, stage=stage)
            with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "fixture-v3"}):
                manifest = collect.pack([run], self.root / "outbox", stage)
            files = {row["path"]: row for row in manifest["files"]}
            archived_spec = f"{run.name}/_recovery/chain-spec.json"
            self.assertEqual(files[archived_spec]["sha256"], common.sha256_file(source))
            self.assertIn(f"{run.name}/{output.relative_to(run)}", files)
            self.assertIn(f"{run.name}/chain.jsonl", files)
            self.assertFalse(any((run / name).exists() for name in collect.RESULT_FILES))
            receipt = collect.verify_archive(self.root / f"outbox/{stage}.tar", self.root / f"outbox/{stage}.manifest.json",
                                             self.root / f"received-{stage}", self.root / f"receipt-{stage}.json")
            self.assertEqual(receipt["status"], "LOCAL_VERIFIED")
            self.assertEqual(receipt["pod_id"], "fixture-v3")
            self.assertEqual((self.root / f"received-{stage}" / archived_spec).read_bytes(), source.read_bytes())

    def test_changed_spec_job_or_chain_and_missing_outputs_are_refused(self) -> None:
        for case in ("spec", "job", "chain", "output"):
            run, source, output = self.completed(case)
            if case == "spec":
                source.write_bytes(source.read_bytes() + b" ")
            elif case == "job":
                recorded = json.loads((run / "job.json").read_text())
                recorded["main"][-1] = "9" * 64
                (run / "job.json").write_text(json.dumps(recorded))
            elif case == "chain":
                log = run / chain.LOG_NAME
                log.write_bytes(log.read_bytes().replace(b'"COMPLETE"', b'"STOPPED"'))
            else:
                output.unlink()
            with self.subTest(case=case), self.assertRaises(collect.CollectError):
                collect.pack([run], self.root / "outbox", case)
        self.assertFalse((self.root / "outbox").exists())

    def test_complete_marker_without_step_history_is_refused_even_with_a_result(self) -> None:
        run, source, _ = self.completed("incomplete-history")
        (run / "result.json").write_text('{"status":"COMPLETE"}')
        log_path = run / chain.LOG_NAME
        log_path.unlink()
        log = chain.ChainLog(log_path, common.sha256_file(source))
        log.append("CHAIN_START", spec=str(source), spec_sha256=common.sha256_file(source), steps=["stage"])
        log.append("CHAIN_END", status="COMPLETE", exit_code=0)
        with self.assertRaisesRegex(collect.CollectError, "complete chain history"):
            collect.pack([run], self.root / "outbox", "invalid")

    def test_wrong_stage_and_incomplete_step_status_are_refused(self) -> None:
        for case in ("stage", "step"):
            run, source, _ = self.completed(case)
            spec = json.loads(source.read_text())
            if case == "stage":
                spec["meta"]["job_stage"] = "eval-ft"
                source.write_text(json.dumps(spec))
                recorded = json.loads((run / "job.json").read_text())
                recorded["main"][-1] = common.sha256_file(source)
                recorded["approval"]["approved_sha256"] = job.content_sha256(recorded)
                (run / "job.json").write_text(json.dumps(recorded))
            else:
                log_path = run / chain.LOG_NAME
                log_path.unlink()
                log = chain.ChainLog(log_path, common.sha256_file(source))
                log.append("CHAIN_START", spec=str(source), spec_sha256=common.sha256_file(source), steps=["stage"])
                log.append("STEP_START", name="stage", argv=spec["steps"][0]["argv"])
                log.append("STEP_END", name="stage", status="FAILED", exit_code=1,
                           expect_files={path: True for path in spec["steps"][0]["expect_files"]})
                log.append("CHAIN_END", status="COMPLETE", exit_code=0)
            with self.subTest(case=case), self.assertRaises(collect.CollectError):
                collect.pack([run], self.root / "outbox", case)

    def test_live_owner_is_refused_and_failed_run_keeps_its_partial_evidence(self) -> None:
        run, source, output = self.completed("live")
        owner = {"pid": os.getpid(), "startticks": common.process_startticks(os.getpid())}
        (run / "owner.json").write_text(json.dumps(owner))
        with self.assertRaisesRegex(collect.CollectError, "still alive"):
            collect.pack([run], self.root / "outbox", "live")
        failed, spec, partial = self.completed("failed")
        (failed / "run_state.json").write_text(json.dumps({"state": "MAIN_EXIT 3"}))
        spec.write_text("broken spec is preserved on failure")
        (failed / chain.LOG_NAME).write_text("broken history is preserved on failure")
        manifest = collect.pack([failed], self.root / "outbox", "failed")
        self.assertIn(f"{failed.name}/{partial.relative_to(failed)}", {row["path"] for row in manifest["files"]})
        self.assertEqual(manifest["closure"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
