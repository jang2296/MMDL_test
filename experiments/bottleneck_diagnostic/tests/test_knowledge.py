"""Synthetic CPU checks; these are not actual knowledge-probe completions."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.bottleneck_diagnostic.input_builder import build_request
from experiments.bottleneck_diagnostic.knowledge import dry_run, protocol, samples_from_bank, score
from experiments.bottleneck_diagnostic.runner import run_diagnostics


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cases = self.root / "cases.json"
        self.items = [{"id": f"probe_{domain}_{pair}_{stage}", "subject": f"Domain{domain}",
                       "stage": stage, "pair_id": f"pair_{domain}_{pair}", "skill": f"skill{pair}",
                       "question": "Choose the named quantity.", "options": ["alpha", "beta", "gamma", "delta"]}
                      for domain in range(15) for pair in range(2) for stage in ("K1", "K2")]
        self.save()

    def save(self):
        self.cases.write_text(json.dumps({"version": "test-only", "items": self.items}))

    def tearDown(self):
        self.temp.cleanup()

    def test_same_frozen_final_prompt_no_images_or_gold(self):
        _, samples = samples_from_bank(self.cases)
        requests = [build_request(s, "T0", 3407) for s in samples]
        self.assertEqual(len(requests), 60)
        self.assertTrue(all(r["images"] == [] and r["max_new_tokens"] == 4096 for r in requests))
        self.assertTrue(all(r["training_allowed"] is False for r in requests))
        self.assertEqual(dry_run(self.cases, self.root / "dry")["actual_inference_calls"], 0)
        self.items[0]["answer"] = "A"
        self.save()
        with self.assertRaisesRegex(ValueError, "gold"):
            samples_from_bank(self.cases)

    def test_unpaired_skill_or_duplicate_id_rejected(self):
        self.items[0]["skill"] = "unrelated"
        self.save()
        with self.assertRaisesRegex(ValueError, "Freeze"):
            samples_from_bank(self.cases)

    def test_post_run_key_only_and_test_outputs_are_not_actual(self):
        _, samples = samples_from_bank(self.cases)
        def backend(request):
            self.assertNotIn("answer", request)
            return {"raw_response": "FINAL_ANSWER: B", "input_tokens": 10,
                    "generated_tokens": 3, "finish_reason": "stop"}
        run_diagnostics(self.root, self.root / "run", protocol(self.cases),
                        samples, [], backend, budget_seconds=120)
        key = self.root / "answers.json"
        key.write_text(json.dumps({"answers": {s["sample_id"]: {"answer": "B"} for s in samples}}))
        result = score(self.cases, key, self.root / "run", self.root / "scores")
        self.assertEqual(result["status"], "TEST_ONLY_NOT_ACTUAL")
        self.assertEqual(result["execution_kind"], "TEST_ONLY")
        self.assertEqual(result["groups"]["ALL"]["K1"]["correct"], 30)


if __name__ == "__main__":
    unittest.main()
