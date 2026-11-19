"""C1 fixture commits stay inside the explicitly isolated local clone."""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tuning.tests.rehearse_v18b_hotfix import COMMENT, COMMENT_FILE, comment_commit, git, isolated_clone


class HotfixRehearsalDriverTests(unittest.TestCase):
    def test_live_preflight_keeps_distinct_immutable_receipts_on_resume(self):
        from fine_tuning.error_note.prep_v16b import read_sealed
        from fine_tuning.error_note.v18b_rehearsal import live_preflight

        with tempfile.TemporaryDirectory(prefix="v18b-live-admission-") as temporary:
            output = Path(temporary)
            sampler = SimpleNamespace(add_version=lambda *args: None,
                preflight_report={"pass": True}, coexistence_reports=[{"pass": True}])
            plan = {"inputs": {"payload": {"file_sha256": "a" * 64}}}
            with patch("fine_tuning.error_note.v18b_rehearsal.validate_execution"), patch(
                    "fine_tuning.error_note.v18b_rehearsal.os.getpid", side_effect=(101, 202, 303)):
                first = live_preflight(SimpleNamespace(output=output), plan, None, sampler)
                sampler.preflight_report = {"pass": True, "stage": "trained"}
                second = live_preflight(SimpleNamespace(output=output), plan, None, sampler)
                sampler.preflight_report = {"pass": True, "stage": "final"}
                third = live_preflight(SimpleNamespace(output=output), plan, None, sampler)
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(read_sealed(output / "training-bindings/rehearsal-live-101.json"), first)
            self.assertEqual(read_sealed(output / "training-bindings/rehearsal-live-202.json"), second)
            self.assertEqual(read_sealed(output / "training-bindings/rehearsal-live-303.json"), third)

    def test_clone_and_fixture_commit_preserve_the_callers_clean_commit(self):
        with tempfile.TemporaryDirectory(prefix="v18b-rehearsal-driver-") as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            subprocess.run(["git", "init", "--quiet", "-b", "test-candidate", str(source)], check=True)
            path = source / COMMENT_FILE
            path.parent.mkdir(parents=True)
            path.write_text("value = 1\n")
            (path.parent / "v18b_rehearsal.py").write_text("value = 1\n")
            git(source, "add", "--", "fine_tuning")
            git(source, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture candidate")
            original = git(source, "rev-parse", "HEAD")
            output = root / "run"
            output.mkdir()
            clone, old = isolated_clone(source, output)
            new = comment_commit(clone)
            self.assertEqual(old, original)
            self.assertNotEqual(new, old)
            self.assertEqual(git(clone, "diff", "--name-only", old, new), COMMENT_FILE)
            self.assertEqual((clone / COMMENT_FILE).read_text(), path.read_text() + COMMENT)
            self.assertEqual(git(source, "rev-parse", "HEAD"), original)
            self.assertEqual(git(source, "status", "--porcelain"), "")
            path.write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "clean committed"):
                isolated_clone(source, output / "rejected")


if __name__ == "__main__":
    unittest.main()
