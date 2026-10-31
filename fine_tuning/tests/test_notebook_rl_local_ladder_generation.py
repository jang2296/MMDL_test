"""CPU fake-engine contract checks; no model/performance/provenance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import local_ladder_generation as generation
from fine_tuning.notebook_rl.cli import write_pool
from fine_tuning.notebook_rl.dataset import _bytes, _sha, _write
from fine_tuning.notebook_rl.engines import Completion, Prepared, RolloutResult
from fine_tuning.notebook_rl.generators import circuit
from fine_tuning.notebook_rl.local_ladder import INDICES, SEED


class FakeEngine:
    context_limit = 16384

    def __init__(self, version=0):
        self.policy_version, self.calls = version, []
        self.overflow = self.missing_pixels = False
        self.after_sample = None
        self.model_config = SimpleNamespace(image_token_id=99)
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs:
            "Observed facts: fixture\r\nFinal answer: 3/2 A")

    def prepare(self, request):
        raw = _bytes(request.messages)
        token = int.from_bytes(hashlib.sha256(raw).digest()[:2], "big")
        pixels = {"pixel_values": torch.tensor([[float(token), 1., 2.]]),
                  "image_grid_thw": torch.tensor([[1, 2, 2]] * len(request.images))}
        if self.missing_pixels:
            pixels.pop("pixel_values")
        return Prepared([token] * 16000 if self.overflow else [token, 99, 8, 9], pixels, request.images,
                        _sha(request.images[0].tobytes()), ((1, 2),))

    def sample(self, requests):
        self.calls.extend(requests)
        request = requests[0]
        prepared = self.prepare(request)
        if self.after_sample:
            self.after_sample(self)
        return [RolloutResult(request.request_id, prepared.token_ids,
            [Completion([41, 42], [-0.3, -0.5], "stop", "Observed facts: fixture\r\nFinal answer: 3/2 A")])]


class LadderGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [circuit.generate(index, SEED, "sealed", "obs") for index in INDICES]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cohort = self.root / "cohort"
        write_pool(self.cohort, self.rows, recipe={"fixture_only": True})
        prep = {"format": "notebook-rl-local-ladder-preparation-v1",
                "status": "CPU_COHORT_PREPARED_PENDING_GATES", "training_allowed": False,
                "evaluation_allowed": False, "gpu_evaluation_completed": False, "count": 10,
                "records_manifest": generation._descriptor(self.cohort / "records/manifest.yaml"),
                "problem_ids": [row.actor.problem_id for row in self.rows]}
        _write(self.cohort / "preparation.json", _bytes(prep))
        self.sha = _sha(_bytes(prep))
        self.templates = {"turn1_obs": "{P0_BODY}\n{OBSERVATION_INSTRUCTION}",
                          "observation_instruction": {"circuit": "Record native circuit facts and a unit answer."}}
        self.engine = FakeEngine()

    def generate(self, name="out", **kwargs):
        return generation.generate_ladder(self.cohort, self.root / name, self.engine, self.templates,
            expected_preparation_sha256=kwargs.get("sha", self.sha), local_protocol_sha256="a" * 64,
            context_limit=kwargs.get("context_limit", 16384))

    def test_explicit_24k_preserves_seeds_and_accepts_long_inputs_blocked_at_16k(self):
        default = generation.protocol(self.templates, "a" * 64)
        self.assertEqual(default, generation.protocol(self.templates, "a" * 64, context_limit=16384))
        self.assertEqual(default | {"context_limit": 24576},
                         generation.protocol(self.templates, "a" * 64, context_limit=24576))
        self.engine.overflow = True
        with self.assertRaisesRegex(ValueError, "CONTEXT_BLOCKED"):
            self.generate("blocked")
        self.assertEqual(self.engine.calls, [])
        self.engine.context_limit = 24576
        long = self.generate("long", context_limit=24576)
        generation.validate_record(long)
        self.assertTrue(all(row["context_limit"] == 24576 and row["input"]["input_tokens"] == 16000
                            for row in long["rows"]))
        self.engine = FakeEngine(1)
        self.engine.context_limit = 24576
        short = self.generate("short", context_limit=24576)
        self.assertEqual([row["seed"] for row in long["rows"]], [row["seed"] for row in short["rows"]])
        self.engine = FakeEngine()
        legacy = self.generate("legacy")
        self.assertEqual([row["seed"] for row in legacy["rows"]], [row["seed"] for row in short["rows"]])
        with self.assertRaisesRegex(ValueError, "paired protocol"):
            generation.validate_paired_records(legacy, short)
        short["rows"][0]["context_limit"] = 16384
        short["rows"][0]["record_sha256"] = generation._digest({
            k: v for k, v in short["rows"][0].items() if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "response contract"):
            generation.validate_record(short)

    def test_only_exact_context_choices_and_matching_engine_are_allowed(self):
        for value in (True, False, 16384., 24576., 8192, 32768, "24576", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "canonical native context"):
                generation.protocol(self.templates, "a" * 64, context_limit=value)
        self.engine.context_limit = 24576
        with self.assertRaisesRegex(ValueError, "explicit policy and context"):
            self.generate()
        self.assertEqual(self.engine.calls, [])

    def test_v27_circuit_policy_preserves_graph_training_and_public_only_payloads(self):
        from fine_tuning.notebook_rl import local_profile, prompts
        from fine_tuning.notebook_rl.generators import graph
        from fine_tuning.notebook_rl.local_ladder_gate import _validate_protocol

        training = local_profile.build_protocol(verified_note_synthesis=True, context_limit=24576)
        original = copy.deepcopy(training)
        train_prompts = local_profile.local_prompts(training)
        graph_actor = graph.generate(0, SEED, "train", "obs").actor
        self.assertIn("Trace JSON", prompts.turn1_text(graph_actor, train_prompts))
        with self.assertRaises(ValueError):
            prompts.turn1_text(self.rows[0].actor, train_prompts)
        self.templates = generation.circuit_evaluation_prompts()
        self.engine.context_limit = 24576
        result = generation.generate_ladder(self.cohort, self.root / "generic", self.engine, self.templates,
            expected_preparation_sha256=self.sha, local_protocol_sha256=training["sha256"], context_limit=24576)
        generation.validate_record(result)
        _validate_protocol(result["protocol"])
        self.assertEqual(training, original)
        self.assertEqual(train_prompts, local_profile.local_prompts(training))
        self.assertEqual(result["protocol"]["evaluation_prompt_policy"], generation.CIRCUIT_PROMPT_POLICY)
        self.assertEqual(result["protocol"]["evaluation_prompt_policy_sha256"], generation._digest(self.templates))
        for row, native in zip(result["rows"], self.rows, strict=True):
            body = row["input"]["messages"][0]["content"][-1]["text"]
            self.assertIn(native.actor.question, body)
            self.assertIn("comp[<name>]", body)
            self.assertIn("wire[<name>]", body)
            self.assertIn("dir[<name>]", body)
            self.assertNotIn("Trace JSON", body)
            self.assertTrue({"gold", "slots", "accept", "verifier"}.isdisjoint(row["input"]["actor"]))
        changed = copy.deepcopy(result)
        changed["protocol"]["evaluation_prompt_policy_sha256"] = "0" * 64
        changed["protocol_sha256"] = generation._digest(changed["protocol"])
        with self.assertRaises(ValueError):
            generation.validate_record(changed)
        with self.assertRaises(ValueError):
            _validate_protocol(changed["protocol"])

    def test_generic_policy_rejects_mixed_templates_and_cross_policy_comparisons(self):
        legacy = self.generate("legacy-prompts")
        self.assertNotIn("evaluation_prompt_policy", legacy["protocol"])
        generic = generation.circuit_evaluation_prompts()
        for changed in (generic | {"solution_trace_contract": "native-reasoning-trace-v3"},
                        generic | {"turn1_obs": "changed"},
                        generic | {"evaluation_prompt_policy": "other"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "canonical circuit"):
                generation.protocol(changed, "a" * 64)
        self.templates = generic
        self.engine = FakeEngine(1)
        current = self.generate("generic-prompts")
        with self.assertRaisesRegex(ValueError, "paired protocol"):
            generation.validate_paired_records(legacy, current)
        self.engine = FakeEngine()
        f0 = self.generate("generic-f0")
        for version in (1, 2):
            self.engine = FakeEngine(version)
            ft = self.generate(f"generic-ft-{version}")
            self.assertEqual(generation.validate_paired_records(f0, ft)["n_pairs"], 10)

    def test_generic_policy_rejects_canonical_metadata_on_noncanonical_literal_messages(self):
        legacy = self.generate("legacy-literal")
        canonical = generation.protocol(generation.circuit_evaluation_prompts(), "a" * 64)
        forged = copy.deepcopy(legacy)
        forged.update(protocol=canonical, protocol_sha256=generation._digest(canonical))
        generation.validate_record(legacy)
        with self.assertRaisesRegex(ValueError, "literal canonical"):
            generation.validate_record(forged)
        self.templates = generation.circuit_evaluation_prompts()
        valid = self.generate("canonical-literal")
        generation.validate_record(valid)
        for index in (0, 9):
            forged = copy.deepcopy(valid)
            row = forged["rows"][index]
            row["input"]["messages"] = legacy["rows"][index]["input"]["messages"]
            row["input_sha256"] = generation._digest(row["input"])
            row["record_sha256"] = generation._digest({k: v for k, v in row.items() if k != "record_sha256"})
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "literal canonical"):
                generation.validate_record(forged)
            paired = copy.deepcopy(forged)
            paired["policy_version"] = 1
            for item in paired["rows"]:
                item["policy_version"] = 1
                item["record_sha256"] = generation._digest({k: v for k, v in item.items() if k != "record_sha256"})
            with self.subTest(paired_index=index), self.assertRaisesRegex(ValueError, "literal canonical"):
                generation.validate_paired_records(forged, paired)

    def test_ten_native_full_records_preserve_inputs_and_are_score_compatible(self):
        before = {path: path.read_bytes() for path in self.cohort.rglob("*") if path.is_file()}
        result = self.generate()
        self.assertEqual(len(self.engine.calls), 10)
        self.assertEqual(len(result["rows"]), 10)
        self.assertEqual(result, json.loads((self.root / "out/result.json").read_bytes()))
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertFalse(result["model_generation_provenance_verified"])
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["usage"], {"input_tokens": 40, "image_tokens": 10, "completion_tokens": 20})
        for index, row in enumerate(result["rows"]):
            self.assertEqual(row["text"], "Observed facts: fixture\r\nFinal answer: 3/2 A")
            self.assertEqual(row["completion_token_ids"], [41, 42])
            self.assertEqual(row["logprobs"], [-0.3, -0.5])
            self.assertIn("Question:", row["input"]["messages"][0]["content"][-1]["text"])
            self.assertEqual(row["input"]["image_grid_thw"], [[1, 2, 2]])
            self.assertEqual(row["input"]["actor"]["image_sha256"], list(self.rows[index].actor.image_sha256))
            self.assertEqual(set(row["input"]["pixel_tensors"]), {"pixel_values", "image_grid_thw"})
            self.assertTrue((self.root / "out" / f"response-{index:04d}.json").is_file())
            call = self.engine.calls[index]
            self.assertIsNone(call.prompt_token_ids)
            self.assertEqual((call.n, call.max_new_tokens), (1, 1024))

    def test_policy_independent_seeds_allow_different_outputs_and_versions(self):
        left = self.generate("f0")
        self.engine = FakeEngine(7)
        right = self.generate("ft")
        for row in right["rows"]:
            row["text"] = "different native answer"
            row["record_sha256"] = generation._digest({k: v for k, v in row.items() if k != "record_sha256"})
        checked = generation.validate_paired_records(left, right)
        self.assertEqual(checked["n_pairs"], 10)
        self.assertFalse(checked["model_generation_provenance_verified"])
        self.assertEqual([row["seed"] for row in left["rows"]], [row["seed"] for row in right["rows"]])

    def test_wrong_pin_and_changed_order_fail_without_sampling(self):
        with self.assertRaisesRegex(ValueError, "pinned preparation"):
            self.generate(sha="0" * 64)
        path = self.cohort / "preparation.json"
        prep = json.loads(path.read_bytes())
        prep["problem_ids"].reverse()
        path.write_bytes(_bytes(prep))
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.generate(sha=_sha(_bytes(prep)))
        self.assertEqual(self.engine.calls, [])
        self.assertFalse((self.root / "out").exists())

    def test_context_pixels_and_engine_context_reject_before_output_and_calls(self):
        for kind in ("overflow", "missing_pixels", "context_limit"):
            self.engine = FakeEngine()
            setattr(self.engine, kind, 8192 if kind == "context_limit" else True)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.generate()
            self.assertFalse((self.root / "out").exists())
            self.assertEqual(self.engine.calls, [])

    def test_existing_and_input_output_paths_preserved(self):
        self.generate()
        raw = (self.root / "out/result.json").read_bytes()
        for output in (self.root / "out", self.cohort / "new"):
            with self.assertRaisesRegex(ValueError, "preserve"):
                generation.generate_ladder(self.cohort, output, FakeEngine(), self.templates,
                    expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64)
        self.assertEqual((self.root / "out/result.json").read_bytes(), raw)

    def test_changed_policy_during_sampling_rejected_without_result(self):
        self.engine.after_sample = lambda engine: setattr(engine, "policy_version", 1)
        with self.assertRaisesRegex(RuntimeError, "sampler altered"):
            self.generate()
        self.assertFalse((self.root / "out/result.json").exists())

    def test_midrun_source_or_prompt_mutation_keeps_partial_without_complete_result(self):
        for changed in ("sources", "prompts"):
            self.engine = FakeEngine()
            original = generation.code_hashes()
            hashes = [original, original | {"fixture_only": "b" * 64}]
            def mutate(_):
                if changed == "prompts":
                    self.templates["extra"] = "changed while generating"
            self.engine.after_sample = mutate
            with patch.object(generation, "code_hashes", side_effect=hashes if changed == "sources" else [original, original]):
                with self.assertRaisesRegex(ValueError, "changed during generation"):
                    self.generate(changed)
            self.assertTrue((self.root / changed / "response-0000.json").is_file())
            self.assertFalse((self.root / changed / "result.json").exists())

    def test_pair_rejects_rehashed_actual_input_changes(self):
        left = self.generate()
        for key, value in (("prompt_token_ids", [66, 99, 8, 9]), ("image_grid_thw", [[1, 4, 4]]),
                           ("original_image_sizes", [[1, 1]]), ("processed_image_sizes", [[2, 2]]),
                           ("image_key", "b" * 64), ("messages", []),
                           ("pixel_tensors", {}), ("actor", {"problem_id": left["problem_ids"][0], "question": "changed"})):
            right = copy.deepcopy(left)
            row = right["rows"][0]
            row["input"][key] = value
            row["input_sha256"] = generation._digest(row["input"])
            row["record_sha256"] = generation._digest({k: v for k, v in row.items() if k != "record_sha256"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "paired actual"):
                generation.validate_paired_records(left, right)

    def test_pair_rejects_seed_budget_protocol_sources_coverage_and_record_changes(self):
        left = self.generate()
        changes = [("seed", 99), ("finish_reason", "error"), ("max_new_tokens", 512),
                   ("context_limit", 8192), ("n", 2), ("completion_token_ids", [1] * 1025),
                   ("logprobs", [float("nan")]), ("text", "tampered")]
        for key, value in changes:
            right = copy.deepcopy(left)
            right["rows"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generation.validate_paired_records(left, right)
        for key, value in (("rows", left["rows"][:-1]), ("source_sha256", {"fixture": "c" * 64}),
                           ("preparation", {"file_sha256": "c" * 64}), ("problem_ids", left["problem_ids"][::-1])):
            right = copy.deepcopy(left)
            right[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generation.validate_paired_records(left, right)
        right = copy.deepcopy(left)
        right["protocol"]["max_new_tokens"] = 512
        right["protocol_sha256"] = generation._digest(right["protocol"])
        with self.assertRaises(ValueError):
            generation.validate_paired_records(left, right)

    def test_same_untrained_policy_pair_and_decode_disagreement_rejected(self):
        left = self.generate()
        with self.assertRaisesRegex(ValueError, "updated FT"):
            generation.validate_paired_records(left, copy.deepcopy(left))
        self.engine.tokenizer.decode = lambda ids, **kwargs: "different text"
        with self.assertRaisesRegex(ValueError, "decode differs"):
            self.generate("decode-error")

    def test_invalid_completion_rejected_and_no_cuda_initialized(self):
        original = self.engine.sample
        def invalid(requests):
            result = original(requests)
            result[0].completions[0].logprobs.pop()
            return result
        self.engine.sample = invalid
        with self.assertRaisesRegex(ValueError, "response contract"):
            self.generate()
        self.assertFalse(torch.cuda.is_initialized())

    def test_deadline_before_first_or_after_final_response_never_completes(self):
        with self.assertRaises(TimeoutError):
            generation.generate_ladder(self.cohort, self.root / "expired", self.engine, self.templates,
                expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64, deadline=time.monotonic() - 1)
        self.assertEqual(self.engine.calls, [])
        late = False
        def finish(engine):
            nonlocal late
            late = len(engine.calls) == 10
        self.engine.after_sample = finish
        with patch.object(generation.time, "monotonic", side_effect=lambda: 1000. if late else 0.):
            with self.assertRaises(TimeoutError):
                generation.generate_ladder(self.cohort, self.root / "late", self.engine, self.templates,
                    expected_preparation_sha256=self.sha, local_protocol_sha256="a" * 64, deadline=100)
        self.assertTrue((self.root / "late/response-0009.json").is_file())
        self.assertFalse((self.root / "late/result.json").exists())

    def test_single_record_validation_preserves_policy_and_canonical_protocol_types(self):
        self.engine = FakeEngine(4)
        result = self.generate()
        generation.validate_record(result)
        changed = copy.deepcopy(result)
        changed["rows"][0]["policy_version"] = 0
        changed["rows"][0]["record_sha256"] = generation._digest({k: v for k, v in changed["rows"][0].items() if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "binding changed"):
            generation.validate_record(changed)
        result["protocol"]["n"] = True
        result["protocol_sha256"] = generation._digest(result["protocol"])
        with self.assertRaises(ValueError):
            generation.validate_record(result)


if __name__ == "__main__":
    unittest.main()
