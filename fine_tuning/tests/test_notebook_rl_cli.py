"""CPU entry-point fixtures, never human data approval or GPU execution."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tuning.notebook_rl.cli import artifact_path, main, read_pool, write_pool
from fine_tuning.notebook_rl.dataset import SPLITS
from fine_tuning.tests.test_notebook_rl_dataset import phase_report, pool


class CliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_pool_roundtrip_and_four_split_build_are_not_approval(self):
        roots = []
        for i, split in enumerate(SPLITS):
            path = self.root/split
            row = pool(split, i+1)
            write_pool(path, [row], recipe={"fixture": True})
            self.assertEqual(read_pool(path), [row])
            roots.append(str(path))
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            code = main(["build", "--inputs", *roots, "--out", str(self.root/"build")])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(console.getvalue())["training_authorized"])
        manifest = json.loads((self.root/"build/manifest.yaml").read_text())
        self.assertEqual(manifest["counts"], dict.fromkeys(SPLITS, 1))
        self.assertFalse(manifest["human_leak_review"])
        with self.assertRaises(FileExistsError):
            main(["build", "--inputs", *roots, "--out", str(self.root/"build")])

    def test_generate_uses_declared_args_and_blocks_checkout_outputs(self):
        with patch("fine_tuning.notebook_rl.cli.generate_pool", return_value=[pool("train", 1)]) as generate:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["generate", "--generator", "chart", "--split", "train", "--count", "1",
                    "--seed", "3407", "--out", str(self.root/"pool")]), 0)
            generate.assert_called_once_with("chart", 1, 3407, "train", .5)
            with self.assertRaisesRegex(ValueError, "outside the checkout"):
                artifact_path(Path(__file__).parents[1]/"notebook_rl/private_data")

    def test_corrupt_pool_rejected_without_new_build(self):
        path = self.root/"pool"
        write_pool(path, [pool("train", 1)], recipe={})
        image = next((path/"images").glob("*.png"))
        image.write_bytes(image.read_bytes()+b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            read_pool(path)

    def test_selection_command_keeps_explicit_pending_status(self):
        report = self.root/"phase.json"
        report.write_text(json.dumps(phase_report()))
        rules = self.root/"rules.json"
        rules.write_text(json.dumps({"max_pass_rate": .75, "max_zero_pass_share": .5, "blind_floor": .25,
            "truncation_family_max": .1, "max_selected": 384, "min_open_answer_fraction": .6}))
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["select", "--phase-a", str(report), "--rules", str(rules),
                         "--out", str(self.root/"selection.yaml")])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads((self.root/"selection.yaml").read_text())["training_allowed"])

    def measurement_job(self):
        # Private payloads intentionally do not exist in the actor process.
        data = {"format": "notebook-rl-measurement-data-job-v1", "build_root": str(self.root/"build"),
            "expected_measurement_scope": {}, **{key: {"path": str(self.root/(key+".json")), "file_sha256": "0"*64}
            for key in ("index", "source_registry", "evidence", "data_report")}}
        path = self.root/"job.json"
        path.write_text(json.dumps(data))
        return path, data

    def test_measurement_check_serializes_only_public_result_without_opening_descriptors(self):
        path, job = self.measurement_job()
        response = {"status": "VALIDATED_FOR_MEASUREMENT", "training_allowed": False,
            "measurement_allowed": True, "approval_created": False, "actors": [pool("train", 1).actor]}
        out = self.root/"measured-data.json"
        with patch("fine_tuning.notebook_rl.approval.validate_for_measurement", return_value=response) as validate:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["check-measurement-data", "--job", str(path), "--out", str(out)]), 0)
            validate.assert_called_once_with(Path(job["build_root"]),
                **{k: v for k, v in job.items() if k not in ("format", "build_root")})
        saved = json.loads(out.read_text())
        self.assertFalse(saved["training_allowed"])
        self.assertEqual(saved["actors"][0]["problem_id"], response["actors"][0].problem_id)
        with self.assertRaises(FileExistsError):
            main(["check-measurement-data", "--job", str(path), "--out", str(out)])

    def test_measurement_check_rejects_inline_private_payload_and_unknown_keys(self):
        path, job = self.measurement_job()
        for changed in (job | {"index": {"private_benchmark_text": "not a descriptor"}}, job | {"skip_checks": True}):
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError), patch("fine_tuning.notebook_rl.approval.validate_for_measurement") as validate:
                main(["check-measurement-data", "--job", str(path), "--out", str(self.root/"not-created.json")])
            validate.assert_not_called()
        self.assertFalse((self.root/"not-created.json").exists())

    def test_real_missing_data_prerequisites_are_recorded_blocked(self):
        path, _ = self.measurement_job()
        out = self.root/"blocked.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["check-measurement-data", "--job", str(path), "--out", str(out)]), 2)
        result = json.loads(out.read_text())
        self.assertEqual(result["status"], "BLOCKED_NOT_CHECKED")
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["actors"], [])


if __name__ == "__main__":
    unittest.main()
