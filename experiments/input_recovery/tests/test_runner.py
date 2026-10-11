"""Self-made CPU fixtures; these do not certify GPU throughput or semantic truth."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from mmdl.evaluation.backends.vllm_backend import VLLMBackend
from mmdl.runtime.artifacts import digest, read_json, write_json

from experiments.input_recovery.inputs import prepare
from experiments.input_recovery.runner import make_plan, run_inference, score
from experiments.input_recovery.tests import test_inputs


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_inputs.InputsTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.data = self.root / "bundle"
        prepare(self.fixture.source, self.data, self.fixture.predictions)
        self.future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    def tearDown(self):
        self.fixture.tearDown()

    def run_fake(self, arm="F0", phase="smoke", fail=False):
        output = self.root / arm
        class Fake:
            def describe(self):
                return {"run_kind": "TEST_ONLY", "max_num_seqs": 4}

            def generate_stream(self, requests):
                for index, request in enumerate(requests):
                    if fail and index == 1:
                        raise RuntimeError("self-made failure")
                    yield {"sample_id": request["sample_id"], "raw_response": "Final answer: B.",
                           "finish_reason": "stop", "generated_tokens": 4, "input_tokens": 3}
        with patch("builtins.print"), patch("experiments.input_recovery.runner.checkout"), patch(
                "experiments.input_recovery.runner.model_identity", side_effect=lambda path, arm: {"arm": arm}):
            run_inference(self.data, self.root / "model", self.root / "base", output,
                          arm, phase, 4, self.future, "a" * 40,
                          lambda *args: Fake())
        return output

    def test_plan_336_no_gold_in_model_requests(self):
        total = 0
        for arm in ("F0", "FL", "FM", "FV", "FML"):
            planned, _ = make_plan(self.data, arm, "full")
            total += len(planned)
            self.assertEqual(len(planned), 240 if arm == "F0" else 24)
            self.assertTrue(all(row["max_new_tokens"] == 8192 for row in planned))
            self.assertFalse(any("gold" in row or "answer" in row for row in planned))
        self.assertEqual(total, 336)
        selection = read_json(self.data / "selection.json")
        selection["selected_ids"] = selection["selected_ids"][:1]
        selection["selection_hash"] = digest({key: value for key, value in selection.items()
                                              if key != "selection_hash"})
        write_json(self.data / "selection.json", selection)
        with self.assertRaisesRegex(ValueError, "coverage"):
            make_plan(self.data, "F0", "full")

    def test_smoke_is_saved_once_and_full_resumes_only_remaining(self):
        output = self.run_fake()
        self.assertEqual(read_json(output / "status.json")["completed"], 60)
        before = {path.name: path.read_bytes() for path in (output / "responses").glob("*.json")}
        self.run_fake(phase="full")
        self.assertEqual(len(list((output / "responses").glob("*.json"))), 240)
        for name, content in before.items():
            self.assertEqual((output / "responses" / name).read_bytes(), content)

    def test_incomplete_accuracy_is_null_and_ft_other_conditions_na(self):
        output = self.run_fake()
        result = self.root / "score"
        score(self.data, [output], result)
        summary = read_json(result / "summary.json")
        self.assertEqual(summary["status"], "INCOMPLETE")
        condition = summary["groups"]["F0/all_selected"]["conditions"]["I"]
        self.assertEqual(condition["planned_applicable"], 48)
        self.assertEqual(condition["completed"], 12)
        self.assertIsNone(condition["accuracy"])
        self.assertEqual(condition["partial_completed_accuracy"], 1)
        self.assertEqual(summary["groups"]["FL/all_selected"]["conditions"]["TR"]["status"], "NOT_APPLICABLE")

    def test_failure_keeps_completion_and_tampering_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "self-made"):
            self.run_fake(fail=True)
        output = self.root / "F0"
        self.assertEqual(len(list((output / "responses").glob("*.json"))), 1)
        path = next((output / "responses").glob("*.json"))
        value = read_json(path)
        value["raw_response"] = "Final answer: A."
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "altered"):
            score(self.data, [output], self.root / "score")
        with self.assertRaisesRegex(ValueError, "Invalid completed"):
            self.run_fake()

    def test_gold_file_is_not_read_during_inference(self):
        (self.data / "gold.jsonl").unlink()
        self.assertEqual(read_json(self.run_fake() / "status.json")["completed"], 60)

    def test_changed_gold_is_rejected_without_touching_inference(self):
        output = self.run_fake()
        path = self.data / "gold.jsonl"
        original = path.read_text()
        path.write_text(original.replace('"answer": "B"', '"answer": "A"'))
        with self.assertRaisesRegex(ValueError, "gold differs"):
            score(self.data, [output], self.root / "score")


class ContinuousBackendTests(unittest.TestCase):
    def test_four_slots_refill_after_durable_completion_and_deadline_aborts(self):
        backend = VLLMBackend.__new__(VLLMBackend)
        backend.batch_size = 4
        added, aborted = [], []
        class Engine:
            def add_request(self, sid, prompt, params):
                added.append(sid)
            def step(self):
                return [SimpleNamespace(request_id=added[-1], finished=True)]
            def abort_request(self, ids):
                aborted.extend(ids)
        backend.llm = SimpleNamespace(llm_engine=Engine())
        backend._prepare_request = lambda request, **kw: ({}, {}, {})
        backend._row = lambda request, *args, **kw: {"sample_id": request["sample_id"]}
        stream = backend.generate_stream({"sample_id": str(index)} for index in range(6))
        self.assertEqual(next(stream)["sample_id"], "3")
        self.assertEqual(added, ["0", "1", "2", "3"])
        self.assertEqual(next(stream)["sample_id"], "4")
        self.assertEqual(added, ["0", "1", "2", "3", "4"])
        def deadline():
            raise TimeoutError("self-made deadline")
        backend._check_deadline = deadline
        with self.assertRaises(TimeoutError):
            next(stream)
        self.assertTrue({"0", "1", "2"} <= set(aborted))


if __name__ == "__main__":
    unittest.main()
