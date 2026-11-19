"""Drain checkpoint recovery never resumes KV or changes a STARTED request's seed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fine_tuning.error_note.v16b_runtime import V16bScheduler, run_v16b
from fine_tuning.tests.test_error_note_v16b_runtime import fixture


class ResumeTests(unittest.TestCase):
    def scheduler(self, directory, *, restore=False, seed_offset=0):
        sampler, items, plan, callbacks, *_ = fixture(seed_offset=seed_offset)
        scheduler = V16bScheduler(sampler, plan=plan, protocol={}, items=items, grade=None, verify=None,
            out=Path(directory), callbacks=callbacks, restore=restore)
        return scheduler, sampler, callbacks, items, plan

    def saved(self, directory):
        scheduler, sampler, callbacks, items, plan = self.scheduler(directory)
        callbacks.guard = lambda v, stats: ["test stop"] if v == 0 else []
        result = run_v16b(None, Mock(), None, sampler, plan, out=Path(directory), protocol={}, items=items,
                          callbacks=callbacks, runtime=scheduler)
        self.assertEqual(result["status"], "STOPPED")
        self.assertFalse(sampler.has_unfinished_requests())
        return scheduler, sampler, callbacks

    def test_save_rejects_open_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            scheduler, sampler, *_ = self.scheduler(directory)
            sampler.add_version(None, 0)
            scheduler.submit_for_version(0)
            with self.assertRaisesRegex(RuntimeError, "drain"):
                scheduler.save_drain_state()

    def test_resume_uses_closed_future_results_and_same_next_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            original, sampler, _ = self.saved(directory)
            sampler.add_version(None, 1)
            original.submit_for_version(1)
            original.step()
            old = {rid: dict(request) for rid, request in original.requests.items()}
            restored, fresh, callbacks, items, plan = self.scheduler(directory, restore=True)
            state = restored.restore_drain_state(Path(directory) / "drain-state.json")
            self.assertEqual(restored.version, 1)
            self.assertIn("p1:T1", restored.closed)
            self.assertEqual(state["pending"], restored.pending_descriptors())
            result = run_v16b(None, Mock(), None, fresh, plan, out=Path(directory), protocol={}, items=items,
                              callbacks=callbacks, runtime=restored)
            self.assertEqual(result["status"], "COMPLETE")
            for rid in ("p2:T1", "p2:OBS", "p0:NOTE"):
                self.assertEqual(restored.requests[rid], old[rid])
            self.assertEqual(len(list((Path(directory) / "rolled-back").iterdir())), 1)

    def test_resume_rejects_new_seed_for_rolled_back_started_request(self):
        with tempfile.TemporaryDirectory() as directory:
            original, sampler, _ = self.saved(directory)
            sampler.add_version(None, 1)
            original.submit_for_version(1)
            restored, fresh, *_ = self.scheduler(directory, restore=True, seed_offset=100)
            restored.restore_drain_state(Path(directory) / "drain-state.json")
            fresh.add_version(None, 1)
            with self.assertRaisesRegex(RuntimeError, "seed"):
                restored.submit_for_version(1)

    def test_two_crash_restores_preserve_archives_sidecars_and_original_started_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, sampler, _ = self.saved(directory)
            checkpoint = root / "drain-state.json"
            checkpoint_bytes = checkpoint.read_bytes()
            sampler.add_version(None, 1)
            original.submit_for_version(1)
            original.step()
            first_pins = {rid: dict(request) for rid, request in original.requests.items()}
            (root / "rows-1.json").write_text('{"attempt": 1}')
            (root / "rows-1.sha256.json").write_text('{"sha256": "first"}')
            restored, fresh, *_ = self.scheduler(directory, restore=True)
            restored.restore_drain_state(checkpoint)
            first_archive = root / "rolled-back" / "after-1"
            first_files = {path.relative_to(first_archive): path.read_bytes()
                           for path in first_archive.rglob("*") if path.is_file()}
            self.assertEqual(first_files[Path("rows-1.sha256.json")], b'{"sha256": "first"}')
            self.assertTrue(any(path.parts[0] == "items" for path in first_files))

            fresh.add_version(None, 1)
            restored.submit_for_version(1)
            restored.step()
            for rid in ("p2:T1", "p2:OBS", "p0:NOTE"):
                self.assertEqual(restored.requests[rid], first_pins[rid])
            (root / "rows-1.json").write_text('{"attempt": 2}')
            (root / "rows-1.sha256.json").write_text('{"sha256": "second"}')
            second, _, *_ = self.scheduler(directory, restore=True)
            second.restore_drain_state(checkpoint)
            second_archive = root / "rolled-back" / "after-1-2"
            self.assertTrue(second_archive.is_dir())
            self.assertEqual((second_archive / "rows-1.sha256.json").read_bytes(), b'{"sha256": "second"}')
            self.assertEqual(first_files, {path.relative_to(first_archive): path.read_bytes()
                                          for path in first_archive.rglob("*") if path.is_file()})
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            self.assertFalse((root / "rows-1.json").exists())
            self.assertFalse((root / "rows-1.sha256.json").exists())

            # Another process may die immediately after recovery, before writing a new journal tail.
            changed, changed_sampler, *_ = self.scheduler(directory, restore=True, seed_offset=100)
            changed.restore_drain_state(checkpoint)
            changed_sampler.add_version(None, 1)
            with self.assertRaisesRegex(RuntimeError, "seed"):
                changed.submit_for_version(1)
            self.assertEqual(len(list((root / "rolled-back").iterdir())), 2)

    def test_resume_rejects_tampered_pending_results(self):
        with tempfile.TemporaryDirectory() as directory:
            self.saved(directory)
            path = Path(directory) / "drain-state.json"
            envelope = json.loads(path.read_text())
            envelope["state"]["closed"]["p1:T1"]["text"] = "changed"
            path.write_text(json.dumps(envelope))
            restored, *_ = self.scheduler(directory, restore=True)
            with self.assertRaisesRegex(RuntimeError, "hash"):
                restored.restore_drain_state(path)

    def test_truncated_json_and_utf8_tails_are_archived_raw_across_two_restores(self):
        for suffix in (b'{"status": "START', b'{"text": "' + bytes([0xe2, 0x82])):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original, sampler, _ = self.saved(directory)
                sampler.add_version(None, 1)
                original.submit_for_version(1)
                journal = root / "requests.jsonl"
                with journal.open("ab") as stream:
                    stream.write(suffix)
                checkpoint = root / "drain-state.json"
                first, _, *_ = self.scheduler(directory, restore=True)
                first.restore_drain_state(checkpoint)
                self.assertTrue((root / "rolled-back/after-1/requests.jsonl").read_bytes().endswith(suffix))
                second, fresh, *_ = self.scheduler(directory, restore=True, seed_offset=100)
                second.restore_drain_state(checkpoint)
                fresh.add_version(None, 1)
                with self.assertRaisesRegex(RuntimeError, "seed"):
                    second.submit_for_version(1)

    def test_newline_terminated_corruption_and_sealed_prefix_corruption_still_fail(self):
        for suffix in (b'not-json' + bytes([10]), b'{"text":"' + bytes([0xff]) + b'"}' + bytes([10])):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                self.saved(directory)
                root = Path(directory)
                journal = root / "requests.jsonl"
                with journal.open("ab") as stream:
                    stream.write(suffix)
                before = journal.read_bytes()
                restored, *_ = self.scheduler(directory, restore=True)
                with self.assertRaises((json.JSONDecodeError, UnicodeDecodeError)):
                    restored.restore_drain_state(root / "drain-state.json")
                self.assertEqual(journal.read_bytes(), before)
        with tempfile.TemporaryDirectory() as directory:
            self.saved(directory)
            root = Path(directory)
            journal = root / "requests.jsonl"
            journal.write_bytes(b"x" + journal.read_bytes()[1:])
            restored, *_ = self.scheduler(directory, restore=True)
            with self.assertRaisesRegex(RuntimeError, "journal changed"):
                restored.restore_drain_state(root / "drain-state.json")


if __name__ == "__main__":
    unittest.main()
