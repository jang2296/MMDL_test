"""Publication guards only; no GitHub/GPU writes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sync_test_repo import POOL, PROOF, fingerprint, selected_files, sync


class TestRepoSyncTests(unittest.TestCase):
    def test_actual_selection_excludes_payloads_and_refuses_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_paths = [f"{POOL}/manifest.json", "env/test-repo.gitignore",
                "fine_tuning/generated/graph_cs_bridge_development_cohort_v1/pairing.jsonl"]
            for arm in ("A0", "A1", "A2", "A3"):
                fixture_paths += [f"{PROOF}/{arm}_processor.json",
                    f"{PROOF}/{arm}_approved/manifest.json", f"{PROOF}/{arm}_approved/records.jsonl"]
            for name in fixture_paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"files": {}}' if name.endswith("manifest.json") else "")
            excluded = ["experiments/demo/predictions.jsonl", "fine_tuning/benchmark_records.json",
                        "fine_tuning/model.safetensors", ".env", "docs/P1/raw.md",
                        "fine_tuning/notebook_rl/private.json",
                        "fine_tuning/private/benchmark_inventory_v1.json",
                        "fine_tuning/notebook_rl/private/benchmark_inventory_v1.json"]
            for name in excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("not for publication")
            source = root / "fine_tuning/example.py"
            source.write_text("print('source')")
            inventory_name = "fine_tuning/notebook_rl/benchmark_inventory_v1.json"
            inventory = root / inventory_name
            inventory.write_text('{"purpose":"exclusion_metadata_only","files":[]}')
            selected = selected_files(root)
            self.assertIn("fine_tuning/example.py", selected)
            self.assertIn(inventory_name, selected)
            self.assertFalse(set(excluded) & selected.keys())
            inventory.write_text('{"question":"private benchmark payload"}')
            with self.assertRaisesRegex(ValueError, "Benchmark payload"):
                selected_files(root)
            inventory.write_text('{"purpose":"exclusion_metadata_only","files":[]}')
            source.write_text("hf_" + "a" * 32)
            with self.assertRaisesRegex(ValueError, "credential"):
                selected_files(root)
            source.write_text("print('source')")
            payload = root / "manifests/datasets/leak.json"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"question":"benchmark payload"}')
            with self.assertRaisesRegex(ValueError, "Benchmark payload"):
                selected_files(root)

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
