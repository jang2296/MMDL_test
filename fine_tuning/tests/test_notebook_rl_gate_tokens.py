"""Real tiny-HF generation and authored boundary failures, not native GPU proof."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import gate_tokens as gates
from fine_tuning.notebook_rl.engines import Completion, HFRolloutEngine, RolloutResult
from fine_tuning.tests.test_notebook_rl_engines import Processor, Tokenizer, request
from fine_tuning.tests.test_training import tiny_model


class BoundaryTokenizer(Tokenizer):
    unk_token_id = 1

    def convert_tokens_to_ids(self, text):
        return 2 if text == "<|im_end|>" else self.unk_token_id


def fixtures():
    processor = Processor()
    processor.tokenizer = BoundaryTokenizer()
    engine = HFRolloutEngine(tiny_model(), processor, {"preprocessing": "hf_processor"}, 64)
    requests = [replace(request(n, key=f"probe-{n}", n=1), max_new_tokens=1) for n in (1, 2)]
    return engine, requests


def authored_sample(engine):
    """Explicitly controlled EOS/pad-token cases, distinct from real HF outputs."""
    calls = []
    def sample(requests):
        r, = requests
        ids = [8, 2] if len(calls) == 0 else [0, 9]
        calls.append(r)
        return [RolloutResult(r.request_id, list(engine.prepare(r).token_ids),
            [Completion(ids, [-1., -2.], "stop" if ids[-1] == 2 else "length", "authored response")])]
    return sample, calls


class TokenGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def measure(self, engine, requests, journal):
        return gates._measure_tokens(engine, requests, feedback="Recheck the original image.",
                                     retry_max_new_tokens=2, record=journal.append)

    def test_real_model_two_turn_requests_retain_original_image_and_history(self):
        engine, requests = fixtures()
        journal = []
        result = self.measure(engine, requests, journal)
        self.assertEqual(result["actual_calls"], 4)
        self.assertEqual(len(result["probes"]), 2)
        self.assertEqual([r["kind"] for r in journal].count("G8_ENGINE_RETURN"), 4)
        self.assertEqual([r["kind"] for r in journal].count("G8_RETRY_REQUEST"), 2)
        self.assertEqual(result["conversation_boundary_token_id"], 2)
        self.assertTrue(all(result[k] for k in ("tokens_exact", "eos_checked", "generated_only_mask")))
        self.assertFalse(result["native_execution"])
        self.assertFalse(result["answers_scored"])
        self.assertFalse(result["weights_updated"])
        self.assertFalse(result["training_allowed"])
        for row in result["probes"]:
            first, second = row["first"], row["second"]
            self.assertEqual(first["input"], second["input"])
            parent = first["prompt_token_ids"] + first["completion_token_ids"]
            self.assertEqual(second["prompt_token_ids"][:len(parent)], parent)
            self.assertEqual(second["selected_tokens"], len(second["completion_token_ids"]))
            self.assertFalse(any(second["full_sequence_action_mask"][:len(second["prompt_token_ids"])]))

    def test_authored_closure_branches_and_generated_pad_are_not_dropped(self):
        engine, requests = fixtures()
        requests = [replace(r, max_new_tokens=2) for r in requests]
        sample, calls = authored_sample(engine)
        with patch.object(engine, "sample", side_effect=sample), \
                patch.object(engine.tokenizer, "decode", side_effect=AssertionError("must not retokenize")):
            result = self.measure(engine, requests, [])
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["closure_branches_observed"], ["GENERATED", "INSERTED"])
        for row in result["probes"]:
            first, second = row["first"], row["second"]
            boundary = len(first["prompt_token_ids"]) + len(first["completion_token_ids"])
            expected_suffix = [9, 10] if row["generated_im_end"] else [2, 9, 10]
            self.assertEqual(second["prompt_token_ids"][boundary:], expected_suffix)
            self.assertEqual(second["completion_token_ids"], [0, 9])
            self.assertEqual(second["full_sequence_action_mask"][-2:], [True, True])
            self.assertEqual(second["padding_tokens"], 0)

    def test_retry_prompt_tamper_is_journaled_then_rejected(self):
        engine, requests = fixtures()
        original = engine.sample
        def tamper(reqs):
            returned = original(reqs)
            if reqs[0].prompt_token_ids is not None:
                returned[0].prompt_token_ids[-1] += 1
            return returned
        journal = []
        with patch.object(engine, "sample", side_effect=tamper), self.assertRaisesRegex(ValueError, "actual prompt"):
            self.measure(engine, requests, journal)
        self.assertEqual(journal[-1]["kind"], "G8_ENGINE_RETURN")
        self.assertEqual(sum(r["kind"] == "G8_ENGINE_RETURN" for r in journal), 2)

    def test_changed_production_helper_or_action_selector_fails(self):
        for defect in ("continuation", "mask"):
            engine, requests = fixtures()
            target = "continuation_ids" if defect == "continuation" else "action_mask"
            value = [9] if defect == "continuation" else torch.zeros((1, 1), dtype=torch.bool)
            with self.subTest(defect=defect), patch.object(gates, target, return_value=value), \
                    self.assertRaisesRegex(ValueError, "continuation|selector"):
                self.measure(engine, requests, [])

    def test_training_pixels_must_equal_sampler_preparation(self):
        engine, requests = fixtures()
        original = engine.model_inputs
        def tamper(*args):
            inputs = original(*args)
            if args[1]:  # Change only the gate's training reconstruction, not generation.
                inputs["pixel_values"] = inputs["pixel_values"] + 1
            return inputs
        with patch.object(engine, "model_inputs", side_effect=tamper), \
                self.assertRaisesRegex(ValueError, "image tensors"):
            self.measure(engine, requests, [])

    def test_overflow_preserves_first_call_and_does_not_truncate_or_retry(self):
        engine, requests = fixtures()
        engine.context_limit = len(engine.prepare(requests[0]).token_ids) + 1
        journal = []
        with self.assertRaisesRegex(ValueError, "CONTEXT_BLOCKED"):
            self.measure(engine, requests, journal)
        self.assertEqual(journal[-1]["kind"], "G8_RETRY_REQUEST")
        self.assertEqual(sum(r["kind"] == "G8_ENGINE_RETURN" for r in journal), 1)

    def test_engine_exception_retains_error_not_unavailable_backend_payload(self):
        engine, requests = fixtures()
        journal = []
        with patch.object(engine, "sample", side_effect=RuntimeError("native failure")), \
                self.assertRaisesRegex(RuntimeError, "native failure"):
            self.measure(engine, requests, journal)
        self.assertEqual(journal[0]["kind"], "G8_SAMPLER_ERROR")
        self.assertFalse(journal[0]["backend_invalid_payload_preserved"])

    def test_bad_identity_seed_and_boundary_rejected_before_sampling(self):
        for defect in ("identity", "seed", "boundary", "synchronization"):
            engine, requests = fixtures()
            if defect == "identity":
                requests[1] = replace(requests[1], request_id=requests[0].request_id)
            elif defect == "seed":
                requests[1] = replace(requests[1], seeds=[-1])
            elif defect == "boundary":
                engine.tokenizer.unk_token_id = 2
            else:
                engine._synchronized = False
            with self.subTest(defect=defect), patch.object(engine, "sample") as sample, self.assertRaises(ValueError):
                self.measure(engine, requests, [])
            sample.assert_not_called()

    def test_version_change_fails_without_a_success_report(self):
        engine, requests = fixtures()
        original = engine.sample
        def change(reqs):
            result = original(reqs)
            engine.policy_version += 1
            return result
        with patch.object(engine, "sample", side_effect=change), self.assertRaisesRegex(ValueError, "policy version"):
            self.measure(engine, requests, [])

    def test_public_entry_rejects_cpu_fixture_before_sampling(self):
        engine, requests = fixtures()
        with patch.object(engine, "sample") as sample, self.assertRaises(ValueError):
            gates.measure_tokens(engine, requests, feedback="probe", retry_max_new_tokens=2, record=lambda row: None)
        sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
