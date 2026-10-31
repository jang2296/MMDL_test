"""CPU-only fixtures: native numeric scoring is not model-performance evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fine_tuning.notebook_rl import local_ladder_score as scorer, vault
from fine_tuning.notebook_rl.cli import write_pool
from fine_tuning.notebook_rl.dataset import _bytes, _sha, _write
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.local_ladder import INDICES, SEED
from fine_tuning.notebook_rl.schema import VerifierResult
from fine_tuning.notebook_rl.verifier_service import VerifierServiceError


class LadderScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [circuit.generate(i, SEED, "sealed", "obs") for i in INDICES]

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cohort = self.root / "cohort"
        write_pool(self.cohort, self.rows, recipe={"fixture_only": True})
        self.prep = {"format": "notebook-rl-local-ladder-preparation-v1",
            "status": "CPU_COHORT_PREPARED_PENDING_GATES", "training_allowed": False,
            "evaluation_allowed": False, "gpu_evaluation_completed": False, "count": 10,
            "records_manifest": scorer._descriptor(self.cohort / "records/manifest.yaml"),
            "problem_ids": [row.actor.problem_id for row in self.rows]}
        _write(self.cohort / "preparation.json", _bytes(self.prep))
        self.sha = _sha(_bytes(self.prep))
        self.responses = [{"problem_id": row.actor.problem_id,
            "text": "Final answer: " + row.verifier.gold,
            "finish_reason": "stop"} for row in self.rows]

    def score(self, rows=None):
        return scorer.score_responses(self.cohort, self.responses if rows is None else rows,
                                      expected_preparation_sha256=self.sha)

    def test_real_fresh_verifier_numeric_units_and_parent_never_reads_gold(self):
        original = vault._read
        opened = []
        def public_read(path):
            opened.append(Path(path).name)
            self.assertNotIn(Path(path).name, ("verifier_private.jsonl", "lineage.jsonl"))
            return original(path)
        with patch.object(vault, "_read", side_effect=public_read):
            result = self.score()
        self.assertEqual(result["status"], "CPU_RESPONSES_SCORED")
        self.assertEqual(result["n_correct"], 10)
        self.assertEqual(result["accuracy"], 1)
        self.assertNotEqual(result["verifier_process"]["pid"], os.getpid())
        self.assertEqual(result["verifier_process"]["vault_manifest_sha256"],
                         self.prep["records_manifest"]["file_sha256"])
        self.assertIn("actor_inputs.jsonl", opened)
        for field in ("training_allowed", "evaluation_execution_authorized",
                      "model_generation_provenance_verified", "actual_input_tokens_verified",
                      "module_causality_certified"):
            self.assertIs(result[field], False)

    def test_wrong_unit_missing_answer_multiple_and_length_keep_denominator(self):
        rows = copy.deepcopy(self.responses)
        rows[0]["text"] = "Final answer: " + self.rows[0].verifier.gold.rsplit(" ", 1)[0] + " bananas"
        rows[1]["text"] = "No conclusion."
        rows[2]["text"] += "\nFinal answer: 999999 A"
        rows[3]["finish_reason"] = "length"
        rows[4]["text"] = "Final answer: 999999 A"
        result = self.score(rows)
        self.assertEqual(result["n_correct"], 5)
        self.assertEqual(result["n_expected"], 10)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["rows"][1]["verifier"]["status"], "INVALID_NO_FINAL")
        self.assertEqual(result["rows"][2]["verifier"]["status"], "INVALID_MULTIPLE")
        self.assertEqual(result["rows"][3]["verifier"]["status"], "TRUNCATED")

    def test_no_gold_search_or_alternative_candidate_rescue(self):
        rows = copy.deepcopy(self.responses)
        rows[0]["text"] = rows[0]["text"].replace("Final answer:", "Intermediate:") + "\nFinal answer: 999999 A"
        rows[1]["text"] += " or 999999 A"
        result = self.score(rows)
        self.assertEqual(result["n_correct"], 8)

    def test_missing_duplicate_reordered_invalid_and_extra_fields_fail_before_worker(self):
        cases = [self.responses[:-1], self.responses + [self.responses[-1]],
                 self.responses[::-1], [self.responses[0]] * 10]
        for changes in ({"gold": "x"}, {"finish_reason": "error"}, {"text": None},
                        {"problem_id": True}, {"finish_reason": ["stop"]}):
            cases.append([self.responses[0] | changes, *self.responses[1:]])
        with patch.object(scorer, "VerifierService") as worker:
            for rows in cases:
                with self.subTest(rows=rows[:1]), self.assertRaises(ValueError):
                    self.score(rows)
            worker.assert_not_called()

    def test_wrong_pin_and_modified_manifest_are_rejected(self):
        with patch.object(scorer, "VerifierService") as worker:
            for digest in ("0" * 64, "bad", None):
                with self.assertRaisesRegex(ValueError, "pinned preparation"):
                    scorer.score_responses(self.cohort, self.responses, expected_preparation_sha256=digest)
            manifest = self.cohort / "records/manifest.yaml"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                self.score()
            worker.assert_not_called()

    def test_private_vault_corruption_fails_real_service_without_score(self):
        private = self.cohort / "records/verifier_private.jsonl"
        private.write_bytes(private.read_bytes() + b" ")
        with self.assertRaises(VerifierServiceError):
            self.score()

    def test_unverifiable_is_incomplete_not_an_incorrect_or_smaller_denominator(self):
        with patch.object(scorer, "VerifierService") as service:
            worker = service.return_value.__enter__.return_value
            worker.metadata = {"item_count": 10,
                "vault_manifest_sha256": self.prep["records_manifest"]["file_sha256"]}
            worker.grade.return_value = VerifierResult("UNVERIFIABLE", None, 0., None, "fixture-only")
            result = self.score()
        self.assertEqual(result["status"], "INCOMPLETE_UNVERIFIABLE_RESPONSES")
        self.assertIsNone(result["accuracy"])
        self.assertTrue(all(row["correct"] is None for row in result["rows"]))
        self.assertEqual(result["n_expected"], 10)

    def test_fresh_worker_wrong_identity_fails_before_grading(self):
        for metadata in ({"item_count": 9, "vault_manifest_sha256": self.prep["records_manifest"]["file_sha256"]},
                         {"item_count": 10, "vault_manifest_sha256": "0" * 64}):
            with patch.object(scorer, "VerifierService") as service:
                worker = service.return_value.__enter__.return_value
                worker.metadata = metadata
                with self.assertRaisesRegex(ValueError, "cohort identity"):
                    self.score()
                worker.grade.assert_not_called()

    def test_mid_replay_response_or_source_mutation_is_rejected(self):
        for source_changed in (False, True):
            rows = copy.deepcopy(self.responses)
            with patch.object(scorer, "VerifierService") as service, patch.object(scorer, "_code") as code:
                code.side_effect = [{"fixture": "before"}, {"fixture": "after" if source_changed else "before"}]
                worker = service.return_value.__enter__.return_value
                worker.metadata = {"item_count": 10,
                    "vault_manifest_sha256": self.prep["records_manifest"]["file_sha256"]}
                def grade(*_):
                    if not source_changed:
                        rows[0]["text"] = "changed while grading"
                    return VerifierResult("INCORRECT", "999999 A", 0., None, "fixture-only")
                worker.grade.side_effect = grade
                with self.assertRaisesRegex(ValueError, "changed during replay"):
                    self.score(rows)

    def test_cli_binds_source_bytes_and_preserves_output(self):
        responses, output = self.root / "responses.json", self.root / "scores.json"
        _write(responses, _bytes(self.responses))
        args = ["--cohort", str(self.cohort), "--cohort-preparation-sha256", self.sha,
                "--responses", str(responses), "--output", str(output)]
        self.assertEqual(scorer.main(args), 0)
        raw = output.read_bytes()
        result = json.loads(raw)
        self.assertEqual(result["response_file"]["file_sha256"], _sha(responses.read_bytes()))
        with self.assertRaises(ValueError):
            scorer.main(args)
        self.assertEqual(output.read_bytes(), raw)
        for forbidden in (self.cohort / "new-score.json", responses):
            with self.assertRaises(ValueError):
                scorer.main([*args[:-1], str(forbidden)])

    def test_import_does_not_load_model_libraries(self):
        code = ("import sys; import fine_tuning.notebook_rl.local_ladder_score; "
                "assert not {'torch', 'transformers', 'peft', 'bitsandbytes'} & set(sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_duplicate_keys_and_symlink_output_are_rejected(self):
        responses = self.root / "responses.json"
        _write(responses, b'[{"text":"first","text":"replacement"}]')
        output = self.root / "output.json"
        args = ["--cohort", str(self.cohort), "--cohort-preparation-sha256", self.sha,
                "--responses", str(responses), "--output", str(output)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            scorer.main(args)
        self.assertFalse(output.exists())
        responses.write_bytes(_bytes(self.responses))
        output.symlink_to(self.root / "missing-target")
        with self.assertRaises(ValueError):
            scorer.main(args)
        self.assertFalse((self.root / "missing-target").exists())


if __name__ == "__main__":
    unittest.main()
