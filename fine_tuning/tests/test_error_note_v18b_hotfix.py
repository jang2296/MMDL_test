"""Operational-fix continuation keeps the closed boundary and training contract intact."""
from copy import deepcopy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18b_hotfix as hotfix
from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.collect import write_json
from fine_tuning.error_note.prep_v16b import file_sha, read_sealed, seal


class HotfixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.output = self.root / "repo", self.root / "campaign"
        self.repo.mkdir()
        self.output.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.source = self.repo / "fine_tuning/error_note/single_vllm.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("class SingleVllmSampler:\n    def release_version(self, version):\n        return version\n\n    def sample(self):\n        return 'fixed prompt'\n")
        self.old = self.commit("initial fixture")
        self.manifest = hotfix.committed_source(self.repo, self.old)
        self.plan = seal(self.output / "campaign-plan.json", {"source": self.manifest,
            "protocol": {"version": "error-note-v18b", "sha256": "p" * 64}, "blocks": [["p0"], ["p1"]],
            "inputs": {}, "total_slots": 4})
        self.cursor = {"slot": 1, "updates": 1, "consumed_problem_ids": ["p0"], "consumed_row_ids": ["p0:T1"],
                       "phase": "rl", "last_transaction_sha256": None}
        checkpoint = self.output / "checkpoint-1-rl"
        checkpoint.mkdir()
        files = {}
        for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt"):
            (checkpoint / name).write_bytes(b"fixture payload")
            files[name] = file_sha(checkpoint / name)
        metadata = checkpoint / "metadata.json"
        write_json(metadata, {"step": 1, "protocol": self.plan["protocol"], "cursor": {"single": self.cursor}, "files": files})
        (checkpoint / "metadata.sha256").write_text(file_sha(metadata))
        self.state = seal(self.output / "campaign-state.json", {"plan_sha256": self.plan["sha256"],
            "checkpoint": {"path": str(metadata), "file_sha256": file_sha(metadata)}, "cursor": self.cursor,
            "runtime": None, "probe": {}, "status": "READY"})

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def operational_fix(self):
        self.source.write_text(self.source.read_text().replace("return version", "return version + 1"))
        return self.commit("fix adapter deletion bookkeeping")

    def authorization(self, new):
        body = hotfix.authorize(self.output, self.repo, self.old, new, "adapter registration bookkeeping")
        path = self.root / "authorization.json"
        seal(path, body)
        return path

    def test_operational_fix_resumes_same_boundary_and_keeps_lineage_on_later_invocations(self):
        new = self.operational_fix()
        args = SimpleNamespace(output=self.output, repo=self.repo, hotfix_resume=self.authorization(new))
        current = hotfix.committed_source(self.repo, new)
        result = hotfix.execution_source(args, self.plan, current)
        state = read_sealed(self.output / "campaign-state.json")
        self.assertEqual(state["cursor"], self.cursor)
        self.assertEqual(state["checkpoint"], self.state["checkpoint"])
        self.assertEqual(result["execution_code_commit"], new)
        self.assertEqual(hotfix.lineage(self.output, self.plan)[0]["old_commit"], self.old)
        self.assertEqual(len(state["hotfix_resumes"]), 1)
        args.hotfix_resume = None
        self.assertEqual(hotfix.execution_source(args, self.plan, current), result)
        state = {k: v for k, v in state.items() if k != "sha256"}
        state["cursor"] = self.cursor | {"slot": 2, "consumed_problem_ids": ["p0", "p1"]}
        write_json(self.output / "campaign-state.json", state | {"sha256": digest(state)})
        self.assertEqual(hotfix.execution_source(args, self.plan, current), result)
        self.assertEqual(read_sealed(self.output / "campaign-state.json")["cursor"]["slot"], 2)

    def test_malformed_authorization_leaves_checkpoint_unchanged(self):
        new = self.operational_fix()
        record = read_sealed(self.authorization(new))
        malformed = self.root / "missing-reason.json"
        seal(malformed, {key: value for key, value in record.items() if key not in {"reason", "sha256"}})
        args = SimpleNamespace(output=self.output, repo=self.repo, hotfix_resume=malformed)
        with self.assertRaisesRegex(ValueError, "both commits and a reason"):
            hotfix.execution_source(args, self.plan, hotfix.committed_source(self.repo, new))
        self.assertEqual(read_sealed(self.output / "campaign-state.json"), self.state)

    def test_bound_prompt_change_and_uncommitted_change_are_rejected(self):
        self.source.write_text(self.source.read_text().replace("fixed prompt", "different prompt"))
        new = self.commit("change prompt")
        with self.assertRaisesRegex(ValueError, "training contract"):
            self.authorization(new)
        self.source.write_text(self.source.read_text() + "# unpublished\n")
        with self.assertRaisesRegex(ValueError, "clean exact"):
            self.authorization(new)

    def test_syntax_only_change_is_accepted_but_signature_and_config_changes_are_rejected(self):
        self.source.write_text(self.source.read_text() + "# log comment\n")
        comment_commit = self.commit("comment only")
        self.assertEqual(len(hotfix.contract_diff(self.repo, self.old, comment_commit)), 1)
        self.source.write_text(self.source.read_text().replace("def release_version(self, version)", "def release_version(self, version, force=False)"))
        signature_commit = self.commit("signature change")
        with self.assertRaisesRegex(ValueError, "training contract"):
            hotfix.contract_diff(self.repo, self.old, signature_commit)
        (self.repo / "fine_tuning/protocol.yaml").write_text("reward: 4\n")
        config_commit = self.commit("new reward config")
        with self.assertRaises(ValueError):
            hotfix.contract_diff(self.repo, comment_commit, config_commit)

    def test_wrong_boundary_duplicate_rows_and_corrupt_weights_are_rejected(self):
        new = self.operational_fix()
        path = self.authorization(new)
        changed = {k: deepcopy(v) for k, v in self.state.items() if k != "sha256"}
        changed["status"] = "CHANGED"
        write_json(self.output / "campaign-state.json", changed | {"sha256": digest(changed)})
        with self.assertRaisesRegex(ValueError, "boundary changed"):
            hotfix.execution_source(SimpleNamespace(output=self.output, repo=self.repo, hotfix_resume=path),
                                    self.plan, hotfix.committed_source(self.repo, new))
        changed["cursor"]["consumed_row_ids"] = ["row", "row"]
        write_json(self.output / "campaign-state.json", changed | {"sha256": digest(changed)})
        with self.assertRaisesRegex(ValueError, "consumption"):
            hotfix.closed_boundary(self.output, self.plan)
        write_json(self.output / "campaign-state.json", self.state)
        (Path(self.state["checkpoint"]["path"]).parent / "adapter.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "payload"):
            hotfix.closed_boundary(self.output, self.plan)

    def test_unauthorized_source_and_added_source_are_rejected(self):
        new = self.operational_fix()
        args = SimpleNamespace(output=self.output, repo=self.repo, hotfix_resume=None)
        with self.assertRaisesRegex(ValueError, "explicit hotfix"):
            hotfix.execution_source(args, self.plan, hotfix.committed_source(self.repo, new))
        (self.source.parent / "replacement.py").write_text("reward = 99\n")
        extra = self.commit("add source")
        with self.assertRaisesRegex(ValueError, "adds or removes"):
            hotfix.contract_diff(self.repo, new, extra)

    def test_cli_creates_reviewable_receipt_without_changing_checkpoint_cursor(self):
        new = self.operational_fix()
        destination = self.root / "from-cli.json"
        self.assertEqual(hotfix.main(["authorize", "--campaign-output", str(self.output), "--repo", str(self.repo),
            "--output", str(destination), "--old-commit", self.old, "--new-commit", new,
            "--reason", "LoRA delete state fix"]), 0)
        self.assertEqual(read_sealed(destination)["slot"], 1)
        self.assertEqual(read_sealed(self.output / "campaign-state.json"), self.state)

    def test_only_closed_runtime_is_eligible_and_uncommitted_journal_suffix_can_be_rolled_back(self):
        journal = self.output / "runtime/requests.jsonl"
        journal.parent.mkdir()
        prefix = b'{"committed":"first"}\n'
        journal.write_bytes(prefix + b'{"uncommitted":"second"}\n')
        runtime = {"plan_sha256": digest(self.plan), "protocol_sha256": self.plan["protocol"]["sha256"],
            "requests": {"one": {}}, "closed": {"one": {}}, "registry": {"0": {"refcount": 0}},
            "journal_bytes": len(prefix), "journal_sha256": hashlib.sha256(prefix).hexdigest()}
        snapshot = self.output / "runtime-snapshot.json"

        def save():
            write_json(snapshot, {"state": runtime, "sha256": digest(runtime)})
            body = {key: value for key, value in self.state.items() if key != "sha256"}
            body["runtime"] = {"path": str(snapshot), "file_sha256": file_sha(snapshot)}
            write_json(self.output / "campaign-state.json", body | {"sha256": digest(body)})
        save()
        self.assertEqual(hotfix.closed_boundary(self.output, self.plan)["cursor"], self.cursor)
        runtime["registry"]["0"]["refcount"] = 1
        save()
        with self.assertRaisesRegex(ValueError, "open requests"):
            hotfix.closed_boundary(self.output, self.plan)
        runtime["registry"]["0"]["refcount"] = 0
        runtime["requests"]["two"] = {}
        save()
        with self.assertRaisesRegex(ValueError, "open requests"):
            hotfix.closed_boundary(self.output, self.plan)

    def test_preflight_retains_original_identity_and_records_the_new_execution_commit(self):
        from fine_tuning.error_note import campaign_v16b as campaign, v17_preflight as preflight
        new = self.operational_fix()
        current = hotfix.committed_source(self.repo, new)
        args = SimpleNamespace(output=self.output, repo=self.repo, code_commit=new, hotfix_resume=self.authorization(new))
        with patch.object(campaign, "_plan_check"), patch.object(campaign.single, "source_manifest", return_value=current):
            identity = preflight._identity(args, self.plan)
        self.assertEqual(identity["code_commit"], self.old)
        self.assertEqual(identity["source_sha256"], digest(self.manifest["actual_files"]))
        self.assertEqual(identity["execution_code_commit"], new)
        self.assertEqual(identity["campaign_plan_sha256"], self.plan["sha256"])

    def test_artifact_evidence_checks_each_producer_against_its_active_source(self):
        from fine_tuning.error_note import v17_artifacts as artifacts, v17_preflight as preflight
        new = self.operational_fix()
        current = hotfix.committed_source(self.repo, new)
        hotfix.execution_source(SimpleNamespace(output=self.output, repo=self.repo, hotfix_resume=self.authorization(new)),
                                self.plan, current)
        identity = {"code_commit": self.old, "source_sha256": digest(self.manifest["actual_files"]),
            "campaign_plan_sha256": self.plan["sha256"], "protocol_sha256": self.plan["protocol"]["sha256"],
            "training_protocol_version": "error-note-v18b"}
        fixes = hotfix.lineage(self.output, self.plan)
        for slot in range(2):
            transaction = seal(self.output / "transactions" / f"slot-{slot}.json",
                {"plan_sha256": self.plan["sha256"], "slot": slot})
            seal(self.output / "update-results" / f"slot-{slot}.json", {"version": "v18b-update-result-v1",
                "plan_sha256": self.plan["sha256"], "global_slot": slot, "producer_pid": slot + 10,
                "timing_scope": "encode_backward_optimizer", "wall_seconds": .25,
                "transaction_sha256": transaction["sha256"], "result": {"optimizer_steps": 1},
                "phase": "rl" if slot else "prep"})
            live = {"version": "v18b-live-training-preflight-v1", "status": "BOUND", **identity,
                "production_admitted": True, "producer_pid": slot + 10, "maximum_shape_measurements": 1,
                "optimizer_steps": 0, "production_parents_consumed": 0,
                "actual_abc_status": "NOT_RUN_USE_PRODUCTION_SLOT_TIMINGS", "f0_transport": {"pass": True},
                "coexistence": [{"pass": True}], "capacity": {}}
            if slot:
                live.update(execution_code_commit=new, hotfix_lineage_sha256=digest(fixes))
            seal(self.output / "training-bindings" / f"live-{slot + 10}.json", live)
        with patch.object(preflight, "_validate_live_capacity"):
            binding, evidence = artifacts._v18b_training_evidence(self.output, self.plan, 2)
        self.assertEqual(binding["execution_code_commit"], new)
        self.assertEqual(binding["code_commit"], self.old)
        self.assertEqual(len(evidence["live_training_preflights"]), 2)
        self.assertEqual(evidence["hotfix_resumes"], read_sealed(self.output / "campaign-state.json")["hotfix_resumes"])
        path = self.output / "training-bindings/live-11.json"
        changed = {key: value for key, value in read_sealed(path).items() if key != "sha256"}
        changed["execution_code_commit"] = self.old
        write_json(path, changed | {"sha256": digest(changed)})
        with patch.object(preflight, "_validate_live_capacity"), self.assertRaisesRegex(ValueError, "producer admission"):
            artifacts._v18b_training_evidence(self.output, self.plan, 2)

    def original_spec(self):
        py = "/test/python"
        exported = self.root / "export-final"
        artifact = self.root / "training-artifacts.json"
        steps = []
        for name, command in (("campaign-advance", "advance"), ("require-complete", "require-complete"),
                              ("reload", "reload"), ("export", "export"), ("training-artifact-seal", "training-artifact-seal")):
            argv = [py, "-m", "fine_tuning.error_note.campaign_v16b", command, "--output", str(self.output),
                    "--code-commit", self.old]
            expected = []
            if name == "reload":
                argv = [py, "-m", "fine_tuning.notebook_rl.pod.chain", "capture", "--output", str(self.root / "reload.json"), "--", *argv]
            if name in {"export", "training-artifact-seal"}:
                argv += ["--export-output", str(exported)]
            if name == "training-artifact-seal":
                argv += ["--artifact-output", str(artifact)]
                expected = [str(artifact)]
            steps.append({"name": name, "argv": argv, "expect_files": expected})
        return {"version": "pod-chain-spec-v1", "steps": steps, "meta": {"job_stage": "v18-main"}}

    def test_resume_spec_keeps_campaign_and_omits_completed_preparation(self):
        new = self.operational_fix()
        authorization = self.authorization(new)
        attempt = self.root / "hotfix-test"
        spec = hotfix.resume_spec(self.original_spec(), campaign_output=self.output, repo=self.repo, new_commit=new,
            authorization=authorization, attempt=attempt, run_dir=attempt / "job-run")
        self.assertEqual([row["name"] for row in spec["steps"]], ["verify-published-source", "campaign-advance",
            "require-complete", "reload", "export", "training-artifact-seal"])
        for row in spec["steps"][1:]:
            argv = row["argv"]
            self.assertIn(str(self.output), argv)
            self.assertIn(new, argv)
        self.assertIn("--hotfix-resume", spec["steps"][1]["argv"])
        self.assertIn(str(attempt / "reload-final.json"), spec["steps"][3]["argv"])
        self.assertIn(str(self.root / "export-final-hotfix-test"), spec["steps"][4]["argv"])
        self.assertEqual(read_sealed(self.output / "campaign-state.json"), self.state)

    def test_operator_plan_keeps_first_bill_and_builds_resumable_approved_job(self):
        from fine_tuning.notebook_rl.pod.job import content_sha256, validate_job
        body = {key: value for key, value in self.plan.items() if key != "sha256"}
        body["execution_settings"] = {"limit_action": "warn", "training_cutoff_seconds": None}
        self.plan = body | {"sha256": digest(body)}
        write_json(self.output / "campaign-plan.json", self.plan)
        state = {key: value for key, value in self.state.items() if key != "sha256"}
        state["plan_sha256"] = self.plan["sha256"]
        write_json(self.output / "campaign-state.json", state | {"sha256": digest(state)})
        new = self.operational_fix()
        source = self.root / "original-spec.json"
        write_json(source, self.original_spec())
        stamp = datetime.now(timezone.utc).isoformat()
        original = {"version": "pod-job-v1", "commit": self.old,
            "volume_root": str(self.root / "volume"), "run_dir": str(self.root / "original-job-run"),
            "repo_root": str(self.repo), "stage": "v18-main", "max_main_seconds": 0, "min_free_gib": 20,
            "smoke": ["/test/python", "-m", "fine_tuning.notebook_rl.pod.gpu_smoke"],
            "financial_mode": "measured_balance_reserve", "financial": {"owned_pod_id": "fixturepod",
                "billing_start_utc": stamp, "reserve_usd": 1.5, "hourly_usd": 1.43,
                "gpu_cutoff_utc": stamp, "balance_usd": 24., "balance_snapshot_utc": stamp},
            "main": ["/test/python", "-m", "fine_tuning.notebook_rl.pod.chain", "run", "--spec", str(source),
                     "--spec-sha256", file_sha(source)], "approval": {"approved_by_user": True}}
        original["approval"]["approved_sha256"] = content_sha256(original)
        job = self.root / "original-job.json"
        write_json(job, original)
        before = job.read_bytes()
        observation = self.root / "balance.json"
        write_json(observation, {"version": "v18b-balance-observation-v1", "owned_pod_id": "fixturepod",
                                "observed_at": stamp, "balance_usd": 24.})
        args = SimpleNamespace(job=job, source_spec=source, campaign_output=self.output, repo=self.repo,
            new_commit=new, approval="Recorded user authorization", reason="Operational LoRA fix",
            balance_observation=observation, output=self.root / "operator-attempt")
        with patch("fine_tuning.notebook_rl.pod.job.validate_job", side_effect=lambda row: validate_job(row, workspace=str(self.root))) as checked:
            result = hotfix.operator_plan(args)
        changed = json.loads(Path(result["job"]).read_text())
        self.assertEqual(changed["financial"]["billing_start_utc"], stamp)
        self.assertEqual(changed["financial"]["balance_usd"], 24.)
        self.assertEqual(changed["max_main_seconds"], 0)
        self.assertEqual(changed["commit"], new)
        self.assertEqual(changed["approval"]["approved_sha256"], content_sha256(changed))
        self.assertEqual(job.read_bytes(), before)
        checked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
