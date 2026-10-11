"""Owned-stage orchestration checks, with no GPU/network or genuine checkpoint load."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mmdl.runtime.artifacts import write_json
from experiments.input_recovery.runpod import run


class ControllerTests(unittest.TestCase):
    def fixture(self, root):
        runtime = root / "runtime"
        model = runtime / "model"
        model.mkdir(parents=True)
        write_json(runtime / "setup/model.json", {"status": "MODEL_HASH_VERIFIED",
                   "code_commit": "a" * 40, "code_repository": "jang2296/MMDL_test",
                   "model_path": str(model)})
        data = root / "data"
        data.mkdir()
        return SimpleNamespace(runtime_root=runtime, data_root=data, commit="a" * 40,
                               checkpoint_root=root / "checkpoints", slots=4)

    def test_only_336_cells_smoke_reused_and_all_four_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            stages = []
            class Fake:
                deadline = datetime.now(timezone.utc) + timedelta(hours=1)
                def stage(self, name, command):
                    stages.append((name, command))
                    if name.endswith("export"):
                        return
                    output = Path(command[command.index("--output") + 1])
                    expected = 60 if name == "F0_smoke" else 240 if name == "F0_full" else 24
                    write_json(output / "status.json", {"status": "COMPLETED", "completed": expected})
            with patch("experiments.input_recovery.runpod.model_identity"):
                result = run(args, Fake())
            self.assertEqual(result["actual_cells"], 336)
            self.assertFalse(result["training_executed"])
            self.assertEqual([name for name, _ in stages], ["F0_smoke", "F0_full",
                "FL_export", "FL_full", "FM_export", "FM_full", "FV_export", "FV_full",
                "FML_export", "FML_full"])
            infer_commands = [command for name, command in stages if not name.endswith("export")]
            self.assertTrue(all(command[command.index("--slots") + 1] == "4" for command in infer_commands))
            self.assertEqual(infer_commands[0][infer_commands[0].index("--output") + 1],
                             infer_commands[1][infer_commands[1].index("--output") + 1])

    def test_gold_on_pod_or_incomplete_smoke_blocks_later_gpu_stages(self):
        for expose_gold in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                if expose_gold:
                    (args.data_root / "gold.jsonl").write_text("self-made")
                stages = []
                class Fake:
                    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
                    def stage(self, name, command):
                        stages.append(name)
                        output = Path(command[command.index("--output") + 1])
                        write_json(output / "status.json", {"status": "INCOMPLETE", "completed": 1})
                with patch("experiments.input_recovery.runpod.model_identity"), self.assertRaises(ValueError):
                    run(args, Fake())
                self.assertEqual(stages, [] if expose_gold else ["F0_smoke"])


if __name__ == "__main__":
    unittest.main()
