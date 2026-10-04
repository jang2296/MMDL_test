"""CPU checks for closed F0 recovery; original predictions and damaged summary stay intact."""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fine_tuning.audit import sha
from fine_tuning.evaluate import resume_results, sample_seed
from fine_tuning.modules import BASE_ID, BASE_REVISION
from fine_tuning.recover_evaluation import recover
from fine_tuning.run_experiment import evaluation_gate


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.parent, self.cohort, self.output = root / "parent", root / "cohort", root / "recovered"
        self.parent.mkdir()
        self.cohort.mkdir()
        self.protocol, self.launch_path = root / "protocol.json", root / "launch.json"
        self.model_path, self.log = root / "model.json", root / "terminal.log"
        self.protocol.write_text(json.dumps({"master_seed": 3407}))
        inputs = [{"id": f"sample_{i}", "subject": "Accounting", "question_type": "multiple-choice",
                   "question": "Choose.", "options": ["one", "two"]} for i in range(60)]
        self.write_rows(self.cohort / "inputs.jsonl", inputs)
        self.write_rows(self.cohort / "gold.jsonl", [{"id": r["id"], "answer": "A"} for r in inputs])
        (self.cohort / "manifest.json").write_text(json.dumps({"count": 60,
            "inputs_sha256": sha(self.cohort / "inputs.jsonl"),
            "gold_sha256": sha(self.cohort / "gold.jsonl")}))
        self.launch = {"owned_pod_id": "owned-test", "pid": 123, "pid_birth": "456",
            "started_utc": "2000-01-01T00:00:00+00:00",
            "command": ["python", "-m", "fine_tuning.evaluate", "run", "--model-path", "pinned-model",
                        "--cohort", str(self.cohort), "--output", str(self.parent),
                        "--protocol", str(self.protocol), "--budget-seconds", "120"],
            "sources": {"evaluate.py": "a" * 64, "protocol.json": sha(self.protocol)},
            "cohort_sha256": sha(self.cohort / "manifest.json")}
        self.model = {"status": "MODEL_HASH_VERIFIED", "model_path": "pinned-model",
                      "manifest": {"repo_id": BASE_ID, "revision": BASE_REVISION}}
        self.launch_path.write_text(json.dumps(self.launch))
        self.model_path.write_text(json.dumps(self.model))
        self.rows = [{"id": r["id"], "status": "BUDGET_STOP", "correct": None} for r in inputs]
        for row in self.rows[:2]:
            row.update(status="COMPLETED", correct=True, raw_response="Final answer: A.",
                       output_token_ids=[3], output_tokens=1, prompt_token_ids=[1, 2], input_tokens=2,
                       seed=sample_seed(row["id"]), seconds=1.0, finish_reason="stop",
                       input_sha256="c" * 64, image_grid_thw=[], processed_image_sizes=[], subject="Accounting")
        self.write_rows(self.parent / "predictions.jsonl", self.rows)
        prefix = {"accuracy": None, "benchmark_derived": True, "checkpoint": "F0-frozen",
                  "checkpoint_metadata_sha256": None, "cohort_sha256": self.launch["cohort_sha256"],
                  "completed": 2, "correct": 2, "denominator": 60, "evaluator_source_sha256": "b" * 64}
        (self.parent / "summary.json").write_text(json.dumps(prefix, sort_keys=True)[:-1]+', "finish_reasons": {')
        os.utime(self.parent / "summary.json", (946684810, 946684810))
        self.log.write_text("TypeError: '<' not supported between instances of 'NoneType' and 'str'\n")
        self.gone = self.enterContext(patch("fine_tuning.recover_evaluation.process_gone", return_value=True))
        self.enterContext(patch.dict("fine_tuning.recover_evaluation.os.environ", RUNPOD_POD_ID="owned-test"))
        self.enterContext(patch("fine_tuning.recover_evaluation.verify_cohort_images"))
        self.score = self.enterContext(patch("fine_tuning.recover_evaluation.score_with_parser",
                                            return_value={"correct": True}))

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def recover(self):
        return recover(self.parent, self.cohort, self.protocol, self.launch_path,
                       self.model_path, self.log, self.output, "owned-test")

    def test_recovered_summary_is_partial_and_parent_bytes_are_preserved(self):
        original = {p.name: p.read_bytes() for p in self.parent.iterdir()}
        summary = self.recover()
        self.assertEqual((summary["status"], summary["completed"], summary["budget_stopped"]),
                         ("BUDGET_STOP", 2, 58))
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["seconds"], 10)
        self.assertEqual(summary["seconds_kind"], "PROCESS_LAUNCH_TO_FAILED_SUMMARY_MTIME_OBSERVATION")
        self.assertIsNone(summary["peak_allocated_bytes"])
        self.assertIsNone(summary["peak_reserved_bytes"])
        self.assertEqual(summary["finish_reasons"], {"stop": 2})
        self.assertEqual(summary["evaluator_source_sha256"], "a" * 64)
        self.assertEqual(self.score.call_count, 2)
        self.assertEqual({p.name: p.read_bytes() for p in self.parent.iterdir()}, original)
        self.assertEqual((self.output / "predictions.jsonl").read_bytes(), original["predictions.jsonl"])
        receipt = json.loads((self.output / "recovery_receipt.json").read_text())
        self.assertTrue(receipt["parent_unchanged"])
        self.assertEqual(receipt["damaged_summary_sha256"], sha(self.parent / "summary.json"))
        self.assertEqual(len(resume_results(self.output, self.cohort, self.protocol, None, False)[0]), 2)
        with self.assertRaises(ValueError):
            evaluation_gate(self.output, self.cohort, self.protocol, baseline=True)

    def test_rejects_identity_hash_mismatch_and_checkpoint(self):
        for mutation in ("owned_pod_id", "cohort_sha256", "protocol_sha256", "model_revision", "checkpoint"):
            with self.subTest(mutation=mutation):
                launch, model = json.loads(json.dumps(self.launch)), json.loads(json.dumps(self.model))
                if mutation == "protocol_sha256":
                    launch["sources"]["protocol.json"] = "wrong"
                elif mutation == "model_revision":
                    model["manifest"]["revision"] = "wrong"
                elif mutation == "checkpoint":
                    launch["command"].extend(["--checkpoint", "adapter"])
                else:
                    launch[mutation] = "wrong"
                self.launch_path.write_text(json.dumps(launch))
                self.model_path.write_text(json.dumps(model))
                with self.assertRaises(ValueError):
                    self.recover()
                self.assertFalse(self.output.exists())

    def test_rejects_reordered_ids_completed_after_stop_and_missing_observation(self):
        for mutation in ("order", "completed_after_stop", "missing_field"):
            with self.subTest(mutation=mutation):
                rows = [dict(row) for row in self.rows]
                if mutation == "order":
                    rows[2], rows[3] = rows[3], rows[2]
                elif mutation == "completed_after_stop":
                    rows[0] = {"id": rows[0]["id"], "status": "BUDGET_STOP", "correct": None}
                else:
                    del rows[0]["prompt_token_ids"]
                self.write_rows(self.parent / "predictions.jsonl", rows)
                with self.assertRaises(ValueError):
                    self.recover()
                self.assertFalse(self.output.exists())

    def test_rejects_live_process_and_score_mismatch(self):
        self.gone.return_value = False
        with self.assertRaises(ValueError):
            self.recover()
        self.gone.return_value = True
        self.score.return_value = {"correct": False}
        with self.assertRaises(ValueError):
            self.recover()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
