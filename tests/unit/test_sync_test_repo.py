"""Publication guards only; no GitHub/GPU writes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sync_test_repo import fingerprint, sync


class TestRepoSyncTests(unittest.TestCase):
    def test_wrong_remote_dirty_checkout_and_extra_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "MMDL_test"
            (destination / ".git").mkdir(parents=True)
            source = root / "README.md"
            source.write_text("source")
            with patch("scripts.sync_test_repo.selected_files", return_value={"README.md": source}):
                with patch("scripts.sync_test_repo.subprocess.check_output", return_value="git@github.com:jang2296/MMDL.git"):
                    with self.assertRaisesRegex(ValueError, "approved MMDL_test"):
                        sync(root, True)
                with patch("scripts.sync_test_repo.subprocess.check_output", side_effect=["git@github.com:jang2296/MMDL_test.git", " M README.md"]):
                    with self.assertRaisesRegex(ValueError, "uncommitted"):
                        sync(root, True)
                self.assertFalse((destination / "README.md").exists())
                with patch("scripts.sync_test_repo.subprocess.check_output", side_effect=["git@github.com:jang2296/MMDL_test.git", ""]):
                    self.assertEqual([], sync(root, True)["differences"])
                self.assertEqual(fingerprint(source), fingerprint(destination / "README.md"))
                extra = destination / "user.txt"
                extra.write_text("keep")
                with patch("scripts.sync_test_repo.subprocess.check_output", return_value="git@github.com:jang2296/MMDL_test.git"):
                    with self.assertRaisesRegex(ValueError, "destination-only"):
                        sync(root, True)
                self.assertEqual("keep", extra.read_text())

    def test_symlink_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "other").mkdir()
            (root / "MMDL_test").symlink_to(root / "other", target_is_directory=True)
            with self.assertRaises(ValueError):
                sync(root, True)


if __name__ == "__main__":
    unittest.main()
