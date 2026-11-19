"""CPU closure transport and next-slot replay; no pretrained model or cloud access."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.collect import write_json
from fine_tuning.error_note.prep_v16b import file_sha, read_sealed
from fine_tuning.error_note import v18b_recovery as recovery
from fine_tuning.error_note.v16b_runtime import V16bScheduler, run_v16b
from fine_tuning.tests.test_error_note_v16b_runtime import fixture


def sealed(path, body):
    value = body | {"sha256": digest(body)}
    write_json(path, value)
    return value


def descriptor(path):
    return {"path": str(path.absolute()), "file_sha256": file_sha(path)}


class ClosureFixture:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        external = root.parent / "external"
        external.mkdir(exist_ok=True)
        (external / "input.json").write_text("{}")
        self.sampler, self.items, bare_plan, self.callbacks, _, self.calls, _ = fixture()
        self.protocol = {"version": "error-note-v16b", "sha256": "a" * 64, "protocol_sha256": "a" * 64}
        self.plan = sealed(root / "campaign-plan.json", bare_plan | {"protocol": self.protocol,
            "inputs": {"input": descriptor(external / "input.json")},
            "model_path": str(external), "data_root": str(external), "prep_root": str(external)})
        self.runtime = V16bScheduler(self.sampler, plan=self.plan, protocol=self.protocol,
            items=self.items, grade=None, verify=None, out=root / "runtime", callbacks=self.callbacks)
        self.callbacks.guard = lambda v, stats: ["fixture stop"] if v == 0 else []
        run_v16b(None, Mock(), None, self.sampler, self.plan, out=root / "runtime", protocol=self.protocol,
            items=self.items, callbacks=self.callbacks, runtime=self.runtime)
        self.envelope = json.loads((root / "runtime/drain-state.json").read_text())
        snapshot = root / "runtime-snapshots/slot-1.json"
        snapshot.parent.mkdir()
        write_json(snapshot, self.envelope)
        self.cursor = {"slot": 1, "consumed_row_ids": ["p0:T1", "p0:OBS"]}
        prep = self.checkpoint(root / "checkpoint-15-prep-end", {"slot": 0})
        write_json(root / "prep-checkpoint.json", prep)
        self.snapshot = descriptor(snapshot)
        probe = root / "probes/slot-1"
        probe.mkdir(parents=True)
        (probe / "probe.json").write_text("{}")
        (probe / "probe_inputs.safetensors").write_bytes(b"CPU probe fixture")
        self.probe = {"json": descriptor(probe / "probe.json"),
                      "tensors": descriptor(probe / "probe_inputs.safetensors")}
        self.state = self.commit(1)

    def checkpoint(self, path, cursor):
        path.mkdir(parents=True)
        files = {}
        for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt"):
            (path / name).write_bytes((name + ":CPU fixture:" + str(cursor["slot"])).encode())
            files[name] = file_sha(path / name)
        write_json(path / "metadata.json", {"files": files, "cursor": {"single": cursor}})
        (path / "metadata.sha256").write_text(file_sha(path / "metadata.json") + "\n")
        return descriptor(path / "metadata.json")

    def commit(self, slot):
        self.cursor = self.cursor | {"slot": slot}
        checkpoint = self.checkpoint(self.root / f"checkpoint-{slot}-rl", self.cursor)
        return sealed(self.root / "campaign-state.json", {"plan_sha256": self.plan["sha256"],
            "cursor": deepcopy(self.cursor), "checkpoint": checkpoint, "runtime": self.snapshot, "probe": self.probe})


class RecoveryTests(unittest.TestCase):
    def test_row_admission_receipts_restore_and_remain_bound_to_the_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = ClosureFixture(root / "campaign")
            rejected = {"row_id": "bad:N", "problem_id": "bad", "kind": "N",
                        "source_row_sha256": "a" * 64, "reason_code": "TOKEN_MASK", "reason": "synthetic bad mask"}
            report = {"schema": "v18b-row-quarantine-v1", "status": "CONTINUE", "attempted_rows": 3,
                      "auxiliary_rows": 3, "admitted_row_ids": ["good:N", "other:N"], "rejected_rows": [rejected]}
            for name in ("row-admissions", "transactions", "update-results"):
                (original.root / name).mkdir()
            short = original.root / "short-monitor-post-prep/generations"
            short.mkdir(parents=True)
            (short / "sample.json").write_text('{"synthetic_short_probe": true}')
            admission = sealed(original.root / "row-admissions/slot-0.json", {
                "plan_sha256": original.plan["sha256"], "slot": 0, "parent_checkpoint": None, "report": report})
            transaction = sealed(original.root / "transactions/slot-0.json", {
                "plan_sha256": original.plan["sha256"], "slot": 0, "parent_checkpoint": None,
                "row_admission_sha256": admission["sha256"]})
            update = sealed(original.root / "update-results/slot-0.json", {
                "global_slot": 0, "transaction_sha256": transaction["sha256"], "plan_sha256": original.plan["sha256"],
                "row_counts": {"N": 2}, "attempted_row_counts": {"N": 3}, "result": {"stats": {
                    "consumed_row_ids": report["admitted_row_ids"], "row_quarantine": report}}})
            hook = recovery.RecoveryCopyHook(original.root, root / "backups")
            hook.checkpoint(original.state, force=True)
            archive = Path(hook.close()[0]["archive"])
            shutil.rmtree(original.root)
            restored = root / "restored"
            self.assertEqual(recovery.restore_bundle(archive, restored)["status"], "RESUME_REFERENCES_VERIFIED")
            path = recovery.check_row_admission(restored, transaction, update)
            self.assertEqual(read_sealed(path), admission)
            self.assertEqual((restored / "short-monitor-post-prep/generations/sample.json").read_text(),
                             '{"synthetic_short_probe": true}')
            bad = deepcopy(update)
            bad["row_counts"]["N"] = 3
            with self.assertRaisesRegex(ValueError, "row counts"):
                recovery.check_row_admission(restored, transaction, bad)
            bad = deepcopy(update)
            bad["result"]["stats"]["consumed_row_ids"] = bad["result"]["stats"]["consumed_row_ids"] + ["bad:N"]
            with self.assertRaisesRegex(ValueError, "consumption"):
                recovery.check_row_admission(restored, transaction, bad)

    def test_archive_restores_delayed_results_and_consumes_same_next_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(original.root, root / "backups")
            self.assertEqual(hook.checkpoint(original.state, force=True)["status"], "COPY_STARTED")
            # Training may prune the latest directory immediately after staging; hardlinks retain it.
            shutil.rmtree(Path(original.state["checkpoint"]["path"]).parent)
            receipt = hook.close()[0]
            archive = Path(receipt["archive"])
            restored = root / "restored"
            result = recovery.restore_bundle(archive, restored)
            self.assertEqual(result["status"], "RESUME_REFERENCES_VERIFIED")
            self.assertFalse(result["model_reload_verified"])
            self.assertEqual(recovery.restore_bundle(archive, restored), result)
            state = read_sealed(restored / "campaign-state.json")
            with recovery.recovery_paths(restored):
                self.assertEqual(recovery.resolve_recovery_path(original.state["checkpoint"]["path"]),
                    restored / "checkpoint-1-rl/metadata.json")
                self.assertEqual(recovery.original_recovery_path(restored / "checkpoint-1-rl/metadata.json"),
                    Path(original.state["checkpoint"]["path"]))
                sampler, items, _, callbacks, _, calls, _ = fixture()
                runtime = V16bScheduler(sampler, plan=original.plan, protocol=original.protocol, items=items,
                    grade=None, verify=None, out=restored / "runtime", callbacks=callbacks, restore=True)
                runtime.restore_drain_state(recovery.resolve_recovery_path(state["runtime"]["path"]))
                resumed = run_v16b(None, Mock(), None, sampler, original.plan, out=restored / "runtime",
                    protocol=original.protocol, items=items, callbacks=callbacks, runtime=runtime)
            self.assertEqual(resumed["status"], "COMPLETE")
            self.assertEqual(calls[0][0], 1)
            self.assertIn("p1:T1", original.envelope["state"]["closed"])
            control_sampler, control_items, _, control_callbacks, _, control_calls, _ = fixture()
            control_runtime = V16bScheduler(control_sampler, plan=original.plan, protocol=original.protocol,
                items=control_items, grade=None, verify=None, out=root / "control", callbacks=control_callbacks)
            run_v16b(None, Mock(), None, control_sampler, original.plan, out=root / "control",
                protocol=original.protocol, items=control_items, callbacks=control_callbacks, runtime=control_runtime)
            self.assertEqual(original.calls + calls, control_calls)
            self.assertEqual(runtime.state(), control_runtime.state())

    def test_missing_snapshot_or_preparation_checkpoint_is_rejected(self):
        for missing in ("runtime-snapshots/slot-1.json", "checkpoint-15-prep-end/runtime.pt"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = ClosureFixture(root / "campaign")
                (fixture.root / missing).unlink()
                hook = recovery.RecoveryCopyHook(fixture.root, root / "backups")
                try:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        hook.checkpoint(fixture.state, force=True)
                    self.assertTrue((fixture.root / "RECOVERY_COPY_FAILED").exists())
                finally:
                    hook.close()

    def test_interval_idempotence_latest_two_and_receiver_failure_preserve_old_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups")
            self.assertEqual(hook.checkpoint(fixture.state)["status"], "NOT_DUE_OR_ALREADY_STAGED")
            for slot in (16, 32, 48):
                state = fixture.commit(slot)
                hook.checkpoint(state)
                published = hook.poll(wait=True)[0]
                received = recovery.receive_bundle(Path(published["archive"]), root / "receiver", external=False)
                self.assertEqual(received["status"], "RECOVERY_CLOSURE_VERIFIED")
                self.assertEqual(hook.checkpoint(state)["status"], "NOT_DUE_OR_ALREADY_STAGED")
            hook.close()
            self.assertEqual(len(list((root / "backups").glob("slot-*.tar"))), 2)
            self.assertEqual(len(list((root / "receiver").glob("slot-*.received.json"))), 2)
            self.assertEqual(len([path for path in (root / "receiver").iterdir() if path.is_dir()]), 2)
            again = recovery.RecoveryCopyHook(fixture.root, root / "backups")
            self.assertEqual(again.checkpoint(state)["status"], "ALREADY_VERIFIED")
            again.close()
            failing = recovery.RecoveryCopyHook(fixture.root, root / "backups",
                transfer_command=[sys.executable, "-c", "raise SystemExit(7)"])
            failing.checkpoint(fixture.commit(64))
            unconfirmed = failing.poll(wait=True)[0]
            failing.close()
            self.assertFalse(unconfirmed["receiver_verified"])
            self.assertEqual(unconfirmed["transfer_status"], "RECEIVER_UNCONFIRMED")
            self.assertFalse((fixture.root / "RECOVERY_COPY_FAILED").exists())
            self.assertTrue((fixture.root / "recovery-copy-warning.json").exists())
            self.assertEqual(len(list((root / "backups").glob("slot-*.tar"))), 2)
            self.assertTrue(Path(unconfirmed["archive"]).with_suffix(".receipt.pending.json").exists())

    def test_retention_rejects_forged_external_archive_path_and_signals_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups")
            unrelated = root / "unrelated.txt"
            unrelated.write_text("preserve")
            write_json(root / "backups/slot-0-forged.receipt.json", {"archive": str(unrelated)})
            try:
                for slot in (16, 32):
                    hook.checkpoint(fixture.commit(slot))
                    if slot == 16:
                        hook.poll(wait=True)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "retention failed"):
                            hook.poll(wait=True)
                self.assertEqual(unrelated.read_text(), "preserve")
                self.assertTrue((fixture.root / "RECOVERY_COPY_FAILED").exists())
            finally:
                hook.close()

    def test_receiver_verifies_actual_bundle_and_external_mounts_remain_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups", transfer_command=[
                sys.executable, "-m", "fine_tuning.error_note.v18b_recovery", "receive",
                "--archive", "{archive}", "--output", str(root / "receiver"), "--closure-only"])
            hook.checkpoint(fixture.state, force=True)
            receipt = hook.poll(wait=True)[0]
            hook.close()
            self.assertEqual(receipt["receiver"]["bundle_sha256"], receipt["bundle_sha256"])
            self.assertFalse(receipt["receiver"]["external_references_verified"])
            shutil.rmtree(root / "external")
            with self.assertRaisesRegex(ValueError, "reference"):
                recovery.restore_bundle(Path(receipt["archive"]), root / "no-mount")
            self.assertFalse((root / "no-mount").exists())

    def test_receiver_outages_keep_two_local_and_last_confirmed_and_allow_next_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups", transfer_command=["receiver", "{archive}"])
            def checksum(argv, stop):
                return {"status": "RECOVERY_CLOSURE_VERIFIED", "bundle_sha256": file_sha(Path(argv[1]))}
            try:
                with patch.object(recovery, "_receiver_result", side_effect=checksum):
                    hook.checkpoint(fixture.commit(16))
                    confirmed = hook.poll(wait=True)[0]
                for slot, error in ((32, FileNotFoundError("offline")), (48, TimeoutError("timeout")),
                                    (64, ValueError("no acknowledgement"))):
                    with patch.object(recovery, "_receiver_result", side_effect=error):
                        hook.checkpoint(fixture.commit(slot))
                        warning = hook.poll(wait=True)[0]
                    self.assertFalse(warning["receiver_verified"])
                    self.assertTrue(Path(confirmed["archive"]).exists())
                    self.assertLessEqual(len(list((root / "backups").glob("slot-*.tar"))), 3)
                    self.assertFalse((fixture.root / "RECOVERY_COPY_FAILED").exists())
                with patch.object(recovery, "_receiver_result", side_effect=checksum):
                    hook.checkpoint(fixture.commit(80))
                    recovered = hook.poll(wait=True)[0]
                self.assertTrue(recovered["receiver_verified"])
                self.assertEqual(len(list((root / "backups").glob("slot-*.tar"))), 2)
            finally:
                hook.close()

    def test_next_and_final_boundary_preempt_stalled_receiver_without_skipping_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            started = root / "receiver-started"
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups", transfer_command=[
                sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(120)", str(started)])
            hook.checkpoint(fixture.commit(16))
            deadline = time.monotonic() + 10
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(started.exists())
            before = time.monotonic()
            self.assertEqual(hook.checkpoint(fixture.commit(32))["status"], "COPY_STARTED")
            self.assertLess(time.monotonic() - before, 3)
            self.assertEqual(len(hook.pending), 1)
            self.assertEqual(hook.checkpoint(fixture.commit(33), force=True)["status"], "COPY_STARTED")
            before = time.monotonic()
            receipt = hook.close()[0]
            self.assertLess(time.monotonic() - before, 3)
            self.assertFalse(receipt["receiver_verified"])
            self.assertEqual(read_sealed(fixture.root / "campaign-state.json")["cursor"]["slot"], 33)
            restored = recovery.restore_bundle(Path(receipt["archive"]), root / "restored")
            self.assertEqual(restored["slot"], 33)
            self.assertEqual(len(list((root / "backups").glob("slot-*.tar"))), 2)

    def test_receiver_failure_cannot_hide_a_corrupted_pod_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups", transfer_command=[
                sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'broken'); sys.exit(4)",
                "{archive}"])
            hook.checkpoint(fixture.commit(16))
            with self.assertRaisesRegex(RuntimeError, "boundary backup failed"):
                hook.poll(wait=True)
            with self.assertRaises(RuntimeError):
                hook.close()
            self.assertTrue((fixture.root / "RECOVERY_COPY_FAILED").exists())
            self.assertFalse((fixture.root / "recovery-copy-warning.json").exists())

    def test_late_receipt_for_an_old_hash_cannot_certify_a_changed_local_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups",
                                            transfer_command=[sys.executable, "-c", "raise SystemExit(7)"])
            hook.checkpoint(fixture.commit(16))
            receipt = hook.poll(wait=True)[0]
            Path(receipt["archive"]).write_bytes(b"changed after delivery failed")
            write_json(Path(receipt["archive"] + ".remote.json"),
                       {"status": "RECOVERY_CLOSURE_VERIFIED", "bundle_sha256": receipt["bundle_sha256"]})
            with self.assertRaisesRegex(ValueError, "local archive hash changed"):
                hook.close()
            self.assertTrue((fixture.root / "RECOVERY_COPY_FAILED").exists())
            self.assertFalse(Path(receipt["archive"]).with_suffix(".receipt.json").exists())

    def test_late_receiver_acknowledgement_requires_matching_hash_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups", transfer_command=["receiver"])
            try:
                with patch.object(recovery, "_receiver_result", side_effect=TimeoutError("offline")):
                    hook.checkpoint(fixture.state, force=True)
                    receipt = hook.poll(wait=True)[0]
                pending = Path(receipt["archive"]).with_suffix(".receipt.pending.json")
                remote = Path(receipt["archive"] + ".remote.json")
                write_json(remote, {"status": "RECOVERY_CLOSURE_VERIFIED", "bundle_sha256": "0" * 64})
                self.assertEqual(hook.poll(), [])
                self.assertTrue(pending.exists())
                write_json(remote, {"status": "RECOVERY_CLOSURE_VERIFIED", "bundle_sha256": receipt["bundle_sha256"]})
                self.assertTrue(hook.poll()[0]["receiver_verified"])
                self.assertFalse(pending.exists())
            finally:
                hook.close()

    def test_explicit_external_root_remap_is_verified_and_preserves_descriptor_identity(self):
        from fine_tuning.error_note import campaign_v16b as campaign
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ClosureFixture(root / "campaign")
            hook = recovery.RecoveryCopyHook(fixture.root, root / "backups")
            hook.checkpoint(fixture.state, force=True)
            archive = Path(hook.close()[0]["archive"])
            shutil.move(str(root / "external"), str(root / "remounted"))
            restored = root / "relocated"
            recovery.restore_bundle(archive, restored, mappings={str(root / "external"): str(root / "remounted")})
            external = fixture.plan["inputs"]["input"]
            with recovery.recovery_paths(restored):
                actual = campaign.checked(external)
                self.assertEqual(actual, root / "remounted/input.json")
                self.assertEqual(campaign.descriptor(actual), external)
                following = recovery.RecoveryCopyHook(restored, root / "following-backup")
                following.checkpoint(read_sealed(restored / "campaign-state.json"), force=True)
                self.assertEqual(following.close()[0]["status"], "RESUME_REFERENCES_VERIFIED")

    def test_actual_v18b_gate_and_supply_paths_are_copied_and_latest_refs_checked(self):
        for missing in (None, "signal-reports/v18b-gate-slot-0.json", "supply-reports/v18b-slot-0.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = ClosureFixture(root / "campaign")
                gate = fixture.root / "signal-reports/v18b-gate-slot-0.json"
                supply = fixture.root / "supply-reports/v18b-slot-0.json"
                for path in (gate, supply):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sealed(path, {"plan_sha256": fixture.plan["sha256"], "slot": 0})
                    sealed(path.with_name(path.name.replace("-0.", "-1.")), {"plan_sha256": fixture.plan["sha256"], "slot": 1})
                write_json(fixture.root / "signal-gate.json", {"latest": descriptor(gate)})
                write_json(fixture.root / "supply-latest.json", {"latest": descriptor(supply)})
                if missing:
                    (fixture.root / missing).unlink()
                hook = recovery.RecoveryCopyHook(fixture.root, root / "backups")
                try:
                    if missing:
                        with self.assertRaises(ValueError):
                            hook.checkpoint(fixture.state, force=True)
                    else:
                        hook.checkpoint(fixture.state, force=True)
                        archive = Path(hook.poll(wait=True)[0]["archive"])
                        restored = root / "restore"
                        recovery.restore_bundle(archive, restored)
                        self.assertTrue((restored / gate.relative_to(fixture.root)).exists())
                        self.assertTrue((restored / supply.relative_to(fixture.root)).exists())
                        self.assertFalse((restored / "supply-reports/v18b-slot-1.json").exists())
                        manifest = read_sealed(restored / "recovery-manifest.json")
                        name = "supply-reports/v18b-slot-0.json"
                        (restored / name).unlink()
                        del manifest["files"][name]
                        body = {key: value for key, value in manifest.items() if key != "sha256"}
                        write_json(restored / "recovery-manifest.json", body | {"sha256": digest(body)})
                        with self.assertRaisesRegex(ValueError, "reference"):
                            recovery.verify_closure(restored)
                finally:
                    hook.close()

    def test_unsafe_archive_entry_never_escapes_destination(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar"
            with tarfile.open(archive, "w") as stream:
                entry = tarfile.TarInfo("../escape")
                entry.size = 1
                stream.addfile(entry, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                recovery.restore_bundle(archive, root / "restore")
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
