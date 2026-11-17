"""CPU-only validation contracts: no model loading, GPU operation or external inputs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from fine_tuning.error_note import v17_eval_engine as engine
from fine_tuning.notebook_rl import proof_eval as pe
from mmdl.runtime.artifacts import sha256_file


def backend_value(*, mode="eager", slots=8, blocks=25000, block_size=16, sampler="raw_logprobs"):
    capacity = NS(num_blocks=blocks, kv_cache_groups=[NS(kv_cache_spec=NS(block_size=block_size))])
    model = NS(enforce_eager=mode == "eager", dtype="torch.bfloat16", quantization=None,
               logprobs_mode="raw_logprobs", max_model_len=40960)
    cfg = NS(model_config=model, scheduler_config=NS(max_num_seqs=slots),
             compilation_config=NS(level=0 if mode == "eager" else 3,
                                   cudagraph_mode=NS(name="NONE" if mode == "eager" else "FULL"),
                                   cudagraph_capture_sizes=[] if mode == "eager" else [1, 2, 4, 8]))
    runner = NS(sampler=NS(logprobs_mode=sampler))
    core = NS(scheduler=NS(kv_cache_config=capacity),
              model_executor=NS(driver_worker=NS(worker=NS(model_runner=runner))))
    llm = NS(llm_engine=NS(vllm_config=cfg, engine_core=NS(engine_core=core)))
    return NS(llm=llm, vllm_version="0.11.0", describe=lambda: {"enforce_eager": True})


def columns(offset=0.):
    return [{"sample_id": f"external-{i:02d}", "seed": i, "input_sha256": "a" * 64,
             "token_ids": [7] * 32, "logprobs": [-2. + offset] * 32, "graph_replay_calls": 31}
            for i in range(16)]


def probes():
    actual = engine.seal({"status": "COMPLETED", "completed_count": 16, "wall_seconds": 100.,
                           "generated_tokens": 10000, "sixty_second_rates": [90., 110.],
                           "forced_ignore_eos": True, "requested_count": 16,
                           "fixed_lengths": {f"external-{i:02d}": 625 for i in range(16)},
                           "cumulative_preemptions": 0})
    long = engine.seal({"status": "COMPLETED", "max_generated_tokens": 32768,
                        "max_observed_context_tokens": 33000, "forced_ignore_eos": True,
                        "cumulative_preemptions": 0})
    return actual, long


def lengths():
    result = [4574] * 900
    result[-1] += engine.HISTORICAL_TOKENS - sum(result)
    return result


def engine_receipt(root: Path) -> dict:
    historical = lengths()
    quantiles = engine.quantile_lengths(historical, [row["sample_id"] for row in columns()])
    actual, long = probes()
    actual["fixed_lengths"] = {row["sample_id"]: row["max_tokens"] for row in quantiles["items"]}
    actual["generated_tokens"] = sum(actual["fixed_lengths"].values())
    actual = engine.seal(actual)
    calibration = engine.calibration_receipt(actual, long, historical, load_seconds=0.,
                                              long_tail_seconds=0., long_tail_basis="CPU fixture")
    pf5 = {"pass": True, "model_ref_sha256": "a" * 64, "files": {"pass": True},
           "engine": {"pass": True, "pf5": {"pass": True, "power": True}},
           "decode_engine": {"pass": True, "power": True, "raw_logprobs": True, "graph_replay_deltas": [0] * 10}}
    pf5_path = root / "pf5.json"
    pf5_path.write_text(json.dumps(pf5))
    selected = engine.seal({"status": "PASS", "engine_mode": "eager",
        "observed_engine": engine.observe_engine(backend_value(), "eager", 8),
        "calibration": calibration, "actual_length_probe": actual, "long_context_probe": long,
        "quantiles": quantiles, "probabilities": columns(), "eager_reference_probabilities": columns(),
        "eager_comparison": engine.compare_probabilities(columns(), columns()),
        "preflight_evidence": engine.seal({"files_pass": True, "historical_engine_pass": True,
            "historical_pf5": pf5["engine"]["pf5"], "decode_pf5": pf5["decode_engine"]})})
    attempt_path = root / "attempt.json"
    attempt_path.write_text(json.dumps(selected))
    value = {"version": engine.VERSION, "status": "PASS", "engine_mode": "eager", "selected_mode": "eager",
             "slots": 8, "safe_slots": 8, "observed_engine": selected["observed_engine"],
             "actual_kv_token_capacity": 400000, "calibration": calibration, "selected_attempt": selected,
             "code_commit": "a" * 40, "baseline_generated_tokens": historical,
             "estimated_evaluation_seconds": calibration["estimated_evaluation_seconds"],
             "evaluation_watchdog_seconds": calibration["evaluation_watchdog_seconds"],
             "preflight_report": str(pf5_path), "preflight_file_sha256": sha256_file(pf5_path),
             "attempts": [{"path": str(attempt_path), "sha256": sha256_file(attempt_path)}]}
    for key in ("training_artifact_sha256", "training_artifact_file_sha256", "model_ref_sha256",
                "source_sha256", "config_sha256", "hardware_sha256", "environment_sha256"):
        value[key] = "a" * 64

    return engine.seal(value)


class SplitEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_capacity_exact_floor_including_graph_allocation(self):
        for blocks in (3000, 10000, 25000, 27776):
            backend = backend_value(blocks=blocks)
            value = engine.capacity_observation(backend.llm.llm_engine, "0.11.0")
            self.assertEqual(value["safe_slots"], math.floor(.9 * blocks * 16 / 40960))
            self.assertLessEqual(value["safe_slots"] * 40960, value["usable_tokens"])

    def test_capacity_rejects_zero_slot_mixed_blocks_and_wrong_version(self):
        with self.assertRaises(ValueError):
            engine.capacity_observation(backend_value(blocks=1).llm.llm_engine, "0.11.0")
        with self.assertRaises(ValueError):
            engine.capacity_observation(backend_value().llm.llm_engine, "0.12.0")
        backend = backend_value()
        backend.llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config.kv_cache_groups.append(
            NS(kv_cache_spec=NS(block_size=32)))
        with self.assertRaises(ValueError):
            engine.capacity_observation(backend.llm.llm_engine, "0.11.0")

    def test_observation_reads_live_graph_configuration_not_describe_literal(self):
        backend = backend_value(mode="cuda_graph")
        value = engine.observe_engine(backend, "cuda_graph", 8)
        self.assertFalse(value["enforce_eager"])
        self.assertTrue(backend.describe()["enforce_eager"])
        self.assertEqual(value["compilation_config"]["cudagraph_mode"], "FULL")
        self.assertEqual(value["actual_kv_token_capacity"], 400000)

    def test_observation_rejects_stale_mode_and_processed_sampler(self):
        with self.assertRaises(ValueError):
            engine.observe_engine(backend_value(), "cuda_graph", 8)
        with self.assertRaises(ValueError):
            engine.observe_engine(backend_value(sampler="processed_logprobs"), "eager", 8)

    def test_constructor_override_survives_registered_variant_and_restores_on_error(self):
        import os
        import yaml
        from mmdl.evaluation.backends import vllm_backend

        calls = []

        def constructor(**kwargs):
            calls.append(kwargs)
            return NS()

        def init(self, model_path, cfg, hw, kind="base", base_path=None):
            self.cfg, self.batch_size = cfg, cfg["execution"]["batch_size"]
            llm, _, _ = vllm_backend._vllm()
            self.llm = llm(max_num_seqs=self.batch_size, enforce_eager=True)

        cfg = yaml.safe_load(pe.VARIANTS[pe.FINAL_VARIANT_V8][0].read_text())
        hw = yaml.safe_load((pe.ROOT / "configs/hardware/rtx4090_24gb.yaml").read_text())
        with mock.patch.object(vllm_backend, "_vllm", return_value=(constructor, object, "0.11.0")) as original:
            with mock.patch.object(vllm_backend.VLLMBackend, "__init__", init):
                before = {key: os.environ.get(key) for key in (pe.SLOTS_ENV, pe.VLLM_INPROCESS_ENV)}
                with self.assertRaisesRegex(RuntimeError, "bounded failure"):
                    with engine.engine_execution("cuda_graph", 8) as execution:
                        with pe.registered_variants(slots=8):
                            vllm_backend.VLLMBackend("unused", cfg, hw)
                        self.assertEqual(execution["constructor_calls"], [{"enforce_eager": False, "max_num_seqs": 8}])
                        raise RuntimeError("bounded failure")
                self.assertIs(vllm_backend._vllm, original)
                self.assertEqual(before, {key: os.environ.get(key) for key in before})
        self.assertFalse(calls[0]["enforce_eager"])
        self.assertEqual(calls[0]["max_num_seqs"], 8)

    def test_real_length_sidecar_is_checked_and_fixed_quantiles_do_not_follow_manifest_order(self):
        path = self.root / "lengths.json"
        value = engine.seal({"version": "v17-b2-length-distribution-v1", "source_sha256": "c" * 64,
                             "generated_tokens": lengths()})
        path.write_text(json.dumps(value))
        loaded = engine.baseline_lengths(path, sha256_file(path))
        ids = [row["sample_id"] for row in columns()]
        self.assertEqual(engine.quantile_lengths(loaded, ids), engine.quantile_lengths(loaded, ids[::-1]))
        path.write_text(json.dumps(value) + " ")
        with self.assertRaises(ValueError):
            engine.baseline_lengths(path, "0" * 64)
        value["generated_tokens"][0] += 1
        path.write_text(json.dumps(engine.seal(value)))
        with self.assertRaises(ValueError):
            engine.baseline_lengths(path, sha256_file(path))

    def test_calibration_uses_actual_length_rate_for_eta_and_watchdog(self):
        actual, long = probes()
        value = engine.calibration_receipt(actual, long, lengths(), load_seconds=10., long_tail_seconds=30.,
                                             long_tail_basis="fixed scenario")
        self.assertEqual(value["status"], "CALIBRATED")
        self.assertEqual(value["q_run"], 100.)
        self.assertEqual(value["estimated_evaluation_seconds"], math.ceil(engine.HISTORICAL_TOKENS / 100 + 40))
        self.assertEqual(value["evaluation_watchdog_seconds"], math.ceil(1.25 * (1.5 * engine.HISTORICAL_TOKENS / 100 + 40)))
        self.assertGreater(value["evaluation_watchdog_seconds"], value["estimated_evaluation_seconds"])

    def test_calibration_does_not_accept_missing_history_partial_actual_or_preemption(self):
        actual, long = probes()
        for a, b in ((actual | {"status": "DIAGNOSTIC_CAP_PARTIAL"}, long),
                     (actual, long | {"max_generated_tokens": 32767}),
                     (actual, long | {"cumulative_preemptions": 1})):
            value = engine.calibration_receipt(engine.seal(a), engine.seal(b), lengths(), load_seconds=0,
                                                 long_tail_seconds=0, long_tail_basis="none observed")
            self.assertEqual(value["status"], "CALIBRATION_LOW_COVERAGE")

    def test_all_sixteen_decode_prefixes_must_agree_with_existing_threshold(self):
        self.assertTrue(engine.compare_probabilities(columns(.01), columns())["pass"])
        changed = columns()
        changed[-1]["logprobs"] = [-1.] * 32
        self.assertFalse(engine.compare_probabilities(changed, columns())["pass"])
        with self.assertRaises(ValueError):
            engine.compare_probabilities(columns()[:-1], columns())
        changed = columns()
        changed[-1]["token_ids"][0] = 9
        with self.assertRaises(ValueError):
            engine.compare_probabilities(changed, columns())

    def test_masked_single_candidate_logprob_zero_is_rejected(self):
        output = NS(outputs=[NS(token_ids=[7] * 32, logprobs=[{7: NS(logprob=0.)}] * 32)])
        with self.assertRaisesRegex(ValueError, "masked zero"):
            engine._raw_decoded(output, 7, 32)
        output.outputs[0].logprobs = [{7: NS(logprob=-5.)}] * 32
        self.assertEqual(engine._raw_decoded(output, 7, 32), [-5.] * 32)

    def test_graph_failure_falls_back_before_benchmark_and_preserves_attempts(self):
        calls = []

        def run(mode, path, *, reference_only):
            calls.append((mode, reference_only))
            if mode == "cuda_graph":
                return {"engine_mode": mode, "status": "FAIL", "reasons": [{"message": "PF5 no replay"}]}
            return {"engine_mode": mode, "status": "MEASURED", "probabilities": columns(), "reasons": []}

        selected, attempts = engine.select_attempts(run, "auto", self.root)
        self.assertEqual(selected["engine_mode"], "eager")
        self.assertEqual(calls, [("cuda_graph", False), ("eager", False), ("eager", True)])
        self.assertEqual(len(attempts), 3)
        rejected = next(row for row in attempts if row["engine_mode"] == "cuda_graph")
        self.assertEqual(rejected["reasons"][0]["message"], "PF5 no replay")
        for row in attempts:
            self.assertEqual(row["sha256"], sha256_file(Path(row["path"])))

    def test_graph_selection_requires_independent_eager_decode_comparison(self):
        calls = []

        def run(mode, path, *, reference_only):
            calls.append((mode, reference_only))
            return {"engine_mode": mode, "status": "MEASURED", "probabilities": columns(), "reasons": []}

        selected, _ = engine.select_attempts(run, "auto", self.root)
        self.assertEqual(selected["engine_mode"], "cuda_graph")
        self.assertEqual(calls, [("cuda_graph", False), ("eager", True)])
        self.assertTrue(selected["eager_comparison"]["pass"])

    def test_graph_probability_failure_rebuilds_eager_without_reusing_graph_calibration(self):
        calls = []

        def run(mode, path, *, reference_only):
            calls.append((mode, reference_only))
            return {"engine_mode": mode, "status": "MEASURED", "probabilities": columns(1. if mode == "cuda_graph" else 0.),
                    "reasons": [], "calibration": {"mode": mode}}

        selected, attempts = engine.select_attempts(run, "auto", self.root)
        self.assertEqual(selected["calibration"]["mode"], "eager")
        self.assertEqual(calls, [("cuda_graph", False), ("eager", True), ("eager", False)])
        self.assertEqual(next(a for a in attempts if a["engine_mode"] == "cuda_graph")["status"], "FAIL")

    def test_cpu_timing_hooks_restore_sampling_penalties_and_replay_on_failure(self):
        def penalties(value):
            return value + 1

        class Sampler:
            def forward(self, value):
                return module.apply_all_penalties(value)

        class Graph:
            def replay(self):
                return "replayed"

        module = NS(Sampler=Sampler, apply_all_penalties=penalties)
        original_forward, original_replay = Sampler.forward, Graph.replay
        fake_torch = NS(cuda=NS(CUDAGraph=Graph))
        with mock.patch.dict("sys.modules", {"torch": fake_torch}), \
                mock.patch.object(engine.importlib, "import_module", return_value=module):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with engine.diagnostic_timing() as timing:
                    self.assertEqual(Sampler().forward(2), 3)
                    self.assertEqual(Graph().replay(), "replayed")
                    self.assertEqual(timing["sampling"]["calls"], 1)
                    self.assertEqual(timing["penalties"]["calls"], 1)
                    self.assertEqual(timing["graph_replay_calls"], 1)
                    raise RuntimeError("stop")
        self.assertIs(Sampler.forward, original_forward)
        self.assertIs(module.apply_all_penalties, penalties)
        self.assertIs(Graph.replay, original_replay)

    def test_technical_validation_bound_covers_full_fallback_and_independent_references(self):
        self.assertEqual(engine.validation_timeout_seconds(2700), 18180)
        self.assertEqual(engine.validation_timeout_seconds(2700, "cuda_graph"), 18180)
        self.assertEqual(engine.validation_timeout_seconds(2700, "eager"), 9120)
        with self.assertRaises(ValueError):
            engine.validation_timeout_seconds(2701)

    def test_decode_probabilities_require_all_sixteen_actual_generation_calls(self):
        requests = [{"sample_id": row["sample_id"], "messages": [], "images": [], "seed": row["seed"]}
                    for row in columns()]
        tokenizer = NS(encode=lambda *a, **k: [7])
        def generate(prompts, params, **kwargs):
            self.assertEqual(params.max_tokens, 32)
            self.assertEqual(params.allowed_token_ids, [7])
            return [NS(outputs=[NS(token_ids=[7] * 32, logprobs=[{7: NS(logprob=-2.)}] * 32)])]
        backend = NS(processor=NS(tokenizer=tokenizer), lora_request="final",
                     _prepare_request=lambda r, **k: ({"input_sha256": "a" * 64}, {}, NS()),
                     llm=NS(generate=mock.Mock(side_effect=generate)))
        with mock.patch.object(engine, "_requests", return_value=iter(requests)):
            result = engine.probability_columns(backend, {}, self.root, {"graph_replay_calls": 0}, "eager")
        self.assertEqual(len(result), 16)
        self.assertEqual(backend.llm.generate.call_count, 16)

    def test_receipt_reconstructs_calibration_and_checks_preserved_attempt_and_pf5(self):
        value = engine_receipt(self.root)
        pf5_path = self.root / "pf5.json"
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(engine.seal(value)))
        self.assertEqual(engine.load_engine_validation(receipt_path)["status"], "PASS")
        with self.assertRaises(ValueError):
            engine.validate_engine_receipt(engine.seal(value | {"slots": 9}))
        with self.assertRaises(ValueError):
            engine.validate_engine_receipt(engine.seal(value | {"evaluation_watchdog_seconds": 5}))
        pf5_path.write_text(pf5_path.read_text() + " ")
        with self.assertRaises(ValueError):
            engine.load_engine_validation(receipt_path)

    def test_cli_parses_chain_arguments_without_running_a_model(self):
        args = ["validate", "--training-seal", "seal.json", "--model-ref", "ref.json", "--base-path", "base",
                "--requests", "requests.json", "--requests-sha256", "a" * 64, "--hardware", "hardware.yaml",
                "--eval-protocol", "protocol.yaml", "--output", "receipt.json", "--code-commit", "b" * 40,
                "--baseline-raw", "lengths.json", "--baseline-raw-sha256", "c" * 64, "--mode", "auto"]
        result = {"status": "PASS", "selected_mode": "eager", "safe_slots": 8, "estimated_evaluation_seconds": 100,
                  "evaluation_watchdog_seconds": 200}
        with mock.patch.object(engine, "validate", return_value=result) as validate, mock.patch("builtins.print"):
            self.assertEqual(engine.main(args), 0)
        parsed = validate.call_args.args[0]
        self.assertEqual(parsed.training_seal, Path("seal.json"))
        self.assertEqual(parsed.max_seconds, engine.PROBE_CAP_SECONDS)


if __name__ == "__main__":
    unittest.main()
