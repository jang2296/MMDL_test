import tempfile
import unittest
from pathlib import Path

from scripts.sync_submission import _REQUIRED_FILES, sync


class SubmissionSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in _REQUIRED_FILES:
            self.source(name, b"fixture\n")

    def tearDown(self):
        self.temp.cleanup()

    def source(self, name, data, mode=0o640):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        return path

    def test_sync_copies_selected_files_and_check_matches(self):
        self.source("README.md", b"README\n")
        self.source("src/mmdl/app.py", b"print('ok')\n", 0o750)
        self.source("docs/internal.md", b"private internal notes\n")
        self.source("AGENTS.md", b"local instructions\n")
        self.source("Assignment_1.md", b"historical report\n")
        self.source("scripts/audit_v8.py", b"development tool\n")

        self.assertEqual(sync(self.root, write=True), [])
        destination = self.root / "MMDL"
        self.assertEqual((destination / "README.md").read_bytes(), b"README\n")
        self.assertEqual((destination / "src/mmdl/app.py").stat().st_mode & 0o777, 0o750)
        self.assertFalse((destination / "docs").exists())
        self.assertFalse((destination / "AGENTS.md").exists())
        self.assertFalse((destination / "Assignment_1.md").exists())
        self.assertTrue((destination / "reports/mmmu_baseline.md").is_file())
        self.assertFalse((destination / "scripts/audit_v8.py").exists())
        self.assertFalse((destination / "tests/unit/test_sync_submission.py").exists())
        self.assertEqual(sync(self.root, write=False), [])

    def test_check_reports_missing_different_and_extra(self):
        self.source("README.md", b"current\n")
        destination = self.root / "MMDL"
        self.assertEqual(sync(self.root, write=True), [])
        (destination / "README.md").write_bytes(b"old\n")
        (destination / "clutter.txt").write_bytes(b"old artifact\n")
        (destination / "empty-clutter").mkdir()
        (destination / "__pycache__").mkdir()
        (destination / "__pycache__/stale.pyc").write_bytes(b"stale bytecode")

        self.assertEqual(sync(self.root, write=False), ["different: README.md",
                                                        "extra: __pycache__/stale.pyc", "extra: clutter.txt",
                                                        "extra directory: __pycache__",
                                                        "extra directory: empty-clutter"])
        (destination / "README.md").unlink()
        self.assertEqual(sync(self.root, write=False), ["missing: README.md",
                                                        "extra: __pycache__/stale.pyc", "extra: clutter.txt",
                                                        "extra directory: __pycache__",
                                                        "extra directory: empty-clutter"])

    def test_sync_backs_up_displaced_files(self):
        self.source("README.md", b"current\n")
        destination = self.root / "MMDL"
        destination.mkdir()
        (destination / "README.md").write_bytes(b"user edit\n")
        (destination / "old.txt").write_bytes(b"old file\n")
        (destination / "__pycache__").mkdir()
        (destination / "__pycache__/stale.pyc").write_bytes(b"old bytecode")

        self.assertEqual(sync(self.root, write=True), [])
        backups = list((self.root / "analysis_exports/submission_backups").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "README.md").read_bytes(), b"user edit\n")
        self.assertEqual((backups[0] / "old.txt").read_bytes(), b"old file\n")
        self.assertEqual((backups[0] / "__pycache__/stale.pyc").read_bytes(), b"old bytecode")
        self.assertEqual((destination / "README.md").read_bytes(), b"current\n")
        self.assertFalse((destination / "__pycache__").exists())

    def test_sync_preserves_real_git_directory_and_rejects_git_link(self):
        self.source("README.md", b"README\n")
        destination = self.root / "MMDL"
        (destination / ".git").mkdir(parents=True)
        (destination / ".git/HEAD").write_text("ref: refs/heads/main\n")
        self.assertEqual(sync(self.root, write=True), [])
        self.assertEqual((destination / ".git/HEAD").read_text(), "ref: refs/heads/main\n")

        other = self.root / "other-git"
        other.mkdir()
        (destination / ".git").rename(destination / ".git.real")
        (destination / ".git").symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "MMDL/.git must be a real directory"):
            sync(self.root, write=True)

    def test_sync_refuses_destination_symlink(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (self.root / "MMDL").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real ROOT/MMDL"):
            sync(self.root, write=True)

    def test_sync_refuses_missing_required_file_before_destination_changes(self):
        missing = self.root / "results/scoring-v8-20260924/provenance.json"
        missing.unlink()
        destination = self.root / "MMDL"
        destination.mkdir()
        (destination / "user-file.txt").write_text("keep me\n")

        with self.assertRaisesRegex(ValueError, "Required publication files are missing:.*provenance.json"):
            sync(self.root, write=True)
        self.assertEqual((destination / "user-file.txt").read_text(), "keep me\n")
        self.assertFalse((self.root / "analysis_exports/submission_backups").exists())


if __name__ == "__main__":
    unittest.main()
