"""Folder-only CPU checks; no model download, pretrained inference or cloud action."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fine_tuning.notebook_rl import resources
from fine_tuning.notebook_rl.input_preparation import digest


class PortabilityTests(unittest.TestCase):
    def test_bundled_resources_preserve_pinned_bytes_and_legacy_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for descriptor, name in resources.RESOURCE_FILES.items():
                bundled = resources.HERE / "resources" / name
                self.assertEqual(resources.resource_path(descriptor, project_root=root), bundled)
                original = resources.HERE.parents[1] / descriptor
                if original.exists():
                    self.assertEqual(bundled.read_bytes(), original.read_bytes())
                legacy = root / descriptor
                legacy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bundled, legacy)
                self.assertEqual(resources.resource_path(descriptor, project_root=root), legacy)
                legacy.unlink()
                legacy.symlink_to(bundled)
                with self.assertRaises(ValueError):
                    resources.resource_path(descriptor, project_root=root)
        invalid_descriptors: tuple[Any, ...] = (
            "../manifests/models/baseline.json", "/etc/passwd", "env/unknown.lock", None)
        for invalid in invalid_descriptors:
            with self.subTest(descriptor=invalid), self.assertRaises(ValueError):
                resources.resource_path(invalid)

    def test_input_identity_keeps_original_digest_serialization(self):
        value = {"z": [1, "한글"], "a": {"shape": [1, 2], "dtype": "torch.int64"}}
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest(value), expected)

    def test_folder_copy_runs_existing_engine_and_cloud_guard_checks(self):
        with tempfile.TemporaryDirectory(prefix="notebook-folder-copy-") as temporary:
            root = Path(temporary)
            shutil.copytree(Path(__file__).resolve().parents[1], root / "fine_tuning",
                            ignore=shutil.ignore_patterns("__pycache__", "generated", "runs", ".pytest_cache"))
            script = '''
import importlib.abc
import json
import sys
import unittest
from pathlib import Path

class NoProjectSiblings(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mmdl", "experiments"}:
            raise AssertionError("outside-folder import: " + fullname)

sys.meta_path.insert(0, NoProjectSiblings())
from fine_tuning.notebook_rl import approval, export, launch, resources
from fine_tuning.notebook_rl.schema import load_protocol

for sibling in ("src", "experiments", "env", "manifests", "prompts"):
    assert not Path(sibling).exists(), sibling
protocol = load_protocol(launch.HERE / "protocol_v1.yaml")
for descriptor in (protocol["initial_checkpoint"]["manifest"], protocol["environment"]["lock"]):
    assert resources.resource_path(descriptor).is_relative_to(resources.HERE)
assert json.loads(export.BASE_MANIFEST.read_text())["revision"] == protocol["initial_checkpoint"]["revision"]
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "fine_tuning.tests.test_notebook_rl_engines",
    "fine_tuning.tests.test_notebook_rl_launch",
    "fine_tuning.tests.test_notebook_rl_prompts",
    "fine_tuning.tests.test_notebook_rl_checkpoint",
    "fine_tuning.tests.test_notebook_rl_local_pipeline",
    "fine_tuning.tests.test_notebook_rl_local_profile",
    "fine_tuning.tests.test_notebook_rl_local_run",
    "fine_tuning.tests.test_notebook_rl_local_train",
    "fine_tuning.tests.test_notebook_rl_local_control",
    "fine_tuning.tests.test_notebook_rl_local_eval",
    "fine_tuning.tests.test_notebook_rl_notebook_learning",
    "fine_tuning.tests.test_notebook_rl_episode_notebook",
    "fine_tuning.tests.test_notebook_rl_episode_retry",
    "fine_tuning.tests.test_notebook_rl_source_observation",
])
result = unittest.TextTestRunner().run(suite)
assert result.testsRun > 40
sys.exit(not result.wasSuccessful())
'''
            env = os.environ | {"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
                                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            env.pop("MMDL_LOCAL_CONTRACT_EVIDENCE", None)
            env.pop("CIRCUIT_COHORT", None)
            result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
