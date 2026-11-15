"""CPU fixtures check runtime mechanics; they are never L0 or model-effect evidence."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fine_tuning.error_note import run_v16 as runtime
from fine_tuning.error_note.contract import sample_seed
from fine_tuning.notebook_rl.input_preparation import digest


def sealed(value):
    value = {key: val for key, val in value.items() if key != "sha256"}
    return value | {"sha256": digest(value)}


def record(pid, correct=True, tokens=100, *, observation=False):
    body = {"item_id": pid, "seed": sample_seed(pid, "t1"), "input_sha256": digest(pid),
            "text": "Final answer: " + ("A" if correct else "B"), "finish_reason": "stop",
            "completion_token_ids": [1] * tokens}
    if observation:
        body["target"] = "Drawn nodes: 1, 2."
    else:
        body["evaluation"] = {"question": "Which option?", "gold": "A", "choices": ["one", "two"],
                              "answer_type": "mcq", "accept": {}}
    return sealed(body)


def gate_fixture():
    dev, sentinel = [f"dev-{i}" for i in range(40)], [f"sentinel-{i}" for i in range(16)]
    files = {"error_note/run_v16.py": digest("CPU fixture only")}
    pre = {"schema": "v16-local-prereg-v1", "mode": "l0", "source_files": files,
           "source_sha256": digest(files), "dataset_sha256": digest("data"), "f0_sha256": digest("f0"),
           "train_blocks": [[f"train-{step * 8 + i}" for i in range(8)] for step in range(16)], "dev_ids": dev,
           "sentinel_ids": sentinel, "min_net": math.ceil(13 * len(dev) / 256), "master_seed": 3407,
           "max_new_tokens": 8192, "image": runtime.LOCAL_IMAGE, "generation_batch_size": 2}
    protocol = runtime._local_protocol(pre | {"rows": [{"source": "graph", "layer": "graph"}]})
    binding = {key: pre[key] for key in ("source_sha256", "dataset_sha256", "f0_sha256")}
    binding |= {"protocol_sha256": protocol["sha256"], "preregistration_sha256": digest(pre)}
    reload = sealed({"schema": "v16-local-reload-v1", "binding": binding, "step": 16,
                     "producer_pid": 100, "reload_pid": 200, "probe_max_abs_difference": 0,
                     "greedy_equal": True, "state_equal": True, "runtime_equal": True})
    return sealed({"schema": "v16-l0-gate-v1", "scope": "l0-full", "binding": binding,
        "preregistration": pre, "protocol": protocol,
        "updates": [{"step": i + 1, "problem_ids": pre["train_blocks"][i], "loss": 0.3} for i in range(16)],
        "metrics": {"dev": {"F0": [record(pid, i >= 10) for i, pid in enumerate(dev)],
                             "FT": [record(pid) for pid in dev]},
                    "sentinel": {str(i): [record(pid) for pid in sentinel] for i in (0, 8, 16)},
                    "observations": {arm: [record(pid, observation=True) for pid in dev] for arm in ("F0", "FT")}},
        "diagnostics": {"ablations": {"on": [-1.] * 40, "vision_off": [-1.1] * 40,
                                       "language_off": [-1.2] * 40, "merger_f0": [-1.3] * 40},
                        "bf16": {"changed_fraction": 0.5, "cosine": 0.8, "projection": 0.7,
                                 "true_norm": 1., "visible_norm": 0.7}}, "reload": reload})


def functional_fixture():
    gate = gate_fixture()
    pre = gate["preregistration"] | {"schema": "v16-local-prereg-v1", "mode": "smoke",
        "train_blocks": [["train-0", "train-1"]], "image": runtime.SMOKE_IMAGE,
        "f0_files": {"model.safetensors": digest("fixture BF16 weights")}}
    pre["f0_sha256"] = digest(pre["f0_files"])
    protocol = sealed(gate["protocol"] | {"image": runtime.SMOKE_IMAGE})
    binding = {key: pre[key] for key in ("source_sha256", "dataset_sha256", "f0_sha256")}
    binding |= {"protocol_sha256": protocol["sha256"], "preregistration_sha256": digest(pre)}
    reload = sealed(gate["reload"] | {"binding": binding, "step": 1})
    result = {"mode": "smoke", "step": 1, "binding": binding, "producer_pid": 100,
        "changed": {group: 1 for group in ("L", "M", "V")}, "functional_only": True, "stop_reasons": [],
        "updates": [{"step": 1, "problem_ids": pre["train_blocks"][0], "loss": 0.4,
                     "gradient_norms": {group: 0.1 for group in ("L", "M", "V")}}]}
    metadata = {"step": 1, "cursor": {"position": 1, "optimizer_steps": 1}, "protocol": protocol,
        "binding": {"code_sha256": pre["source_sha256"], "data_sha256": pre["dataset_sha256"],
                    "protocol_sha256": protocol["sha256"]}, "state_sha256": {"tiny-fixture": digest("weights")}}
    return sealed({"schema": "v16-local-functional-v1", "scope": "local-nf4-functional", "binding": binding,
                   "preregistration": pre, "protocol": protocol, "result": result, "reload": reload,
                   "checkpoint_metadata": metadata, "checkpoint_metadata_file_sha256": digest("metadata")})


def bf16_gate_fixture():
    gate = gate_fixture()
    image = runtime.LOCAL_IMAGE | {"min_pixels": 1003520, "max_pixels": 4014080}
    protocol = sealed(gate["protocol"] | {"precision": "bf16", "image": image})
    pre = gate["preregistration"] | {"schema": "v16-effect-prereg-v1", "mode": "bf16-effect",
                                     "precision": "bf16", "protocol": protocol, "image": image}
    binding = gate["binding"] | {"protocol_sha256": protocol["sha256"], "preregistration_sha256": digest(pre),
                                 "checkpoint_sha256": digest("BF16 trained checkpoint")}
    gate.update(schema="v16-bf16-effect-v1", scope="bf16-O-effect", binding=binding,
                preregistration=pre, protocol=protocol,
                checkpoint={"path": "/fixture/metadata.json", "file_sha256": binding["checkpoint_sha256"]},
                reload=sealed(gate["reload"] | {"schema": "v16-effect-reload-v1", "binding": binding}))
    return sealed(gate)


class GateTests(unittest.TestCase):
    def test_functional_receipt_is_not_effect_evidence(self):
        receipt = functional_fixture()
        result = runtime.validate_local_functional_receipt(receipt)
        self.assertTrue(result["ready"])
        self.assertFalse(result["effect_established"])
        with self.assertRaises(ValueError):
            runtime.validate_bf16_effect_receipt(receipt)
        for group in ("L", "M", "V"):
            broken = copy.deepcopy(receipt)
            broken["result"]["changed"][group] = 0
            with self.assertRaises(ValueError):
                runtime.validate_local_functional_receipt(sealed(broken))

    def test_bf16_gate_requires_its_own_precision_and_checkpoint(self):
        gate = bf16_gate_fixture()
        result = runtime.validate_bf16_effect_receipt(gate,
            expected_binding={"checkpoint_sha256": digest("BF16 trained checkpoint")})
        self.assertTrue(result["passed"])
        self.assertEqual(result["precision"], "bf16")
        self.assertEqual(result["baseline_correct_length_median"], 100)
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(gate)
        with self.assertRaises(ValueError):
            runtime.validate_bf16_effect_receipt(gate_fixture())
        with self.assertRaises(ValueError):
            runtime.validate_bf16_effect_receipt(gate, expected_binding={"checkpoint_sha256": digest("other")})

    def test_real_manifest_sentinel_split_never_replaced_by_dev(self):
        manifest = {"rows": [{"problem_id": f"train-{i}"} for i in range(128)], "items":
            [{"problem_id": f"dev-{i}", "split": "dev"} for i in range(64)] +
            [{"problem_id": f"sentinel-{i}", "split": "sentinel"} for i in range(12)]}
        with patch("fine_tuning.error_note.content_v16.load_content_manifest", return_value=manifest):
            args = argparse.Namespace(mode="l0", manifest=Path("/fixture/manifest.json"),
                                      manifest_sha256=digest("fixture"))
            with self.assertRaisesRegex(ValueError, "sixteen sentinels"):
                runtime.prepare(args)

    def test_exact_paired_known_values(self):
        self.assertEqual(runtime.exact_paired(0, 0)["p_value"], 1)
        self.assertEqual(runtime.exact_paired(6, 0)["p_value"], 0.03125)
        self.assertAlmostEqual(runtime.exact_paired(33, 28)["p_value"], 0.6089207154)
        self.assertEqual(runtime.exact_paired(28, 33)["net"], -5)
        for args in ((True, 1), (-1, 0), (2.0, 1)):
            with self.assertRaises(ValueError):
                runtime.exact_paired(*args)

    def test_full_raw_fixture_is_recomputed(self):
        gate = gate_fixture()
        gate["PASS"] = False
        result = runtime.validate_l0_gate(sealed(gate), expected_binding={"dataset_sha256": digest("data")})
        self.assertEqual(result["verdict"], "GO")
        self.assertEqual(result["paired"]["net"], 10)
        self.assertEqual(result["paired"]["n"], 40)
        self.assertFalse(result["benchmark_effect_established"])

    def test_no_improvement_stops_despite_pass_boolean(self):
        gate = gate_fixture()
        gate["metrics"]["dev"]["FT"] = copy.deepcopy(gate["metrics"]["dev"]["F0"])
        gate["passed"] = True
        decision = runtime.validate_l0_gate(sealed(gate))
        self.assertFalse(decision["passed"])
        self.assertIn("P0_GAIN_NOT_ESTABLISHED", decision["reasons"])

    def test_smoke_or_incomplete_updates_never_promote(self):
        for mutate in (lambda gate: gate.update(scope="smoke"),
                       lambda gate: gate["updates"].pop(),
                       lambda gate: gate["metrics"]["dev"]["FT"].pop()):
            gate = gate_fixture()
            mutate(gate)
            with self.assertRaises(ValueError):
                runtime.validate_l0_gate(sealed(gate))

    def test_threshold_cannot_be_relaxed_and_resealed(self):
        gate = gate_fixture()
        gate["preregistration"]["min_net"] = 0
        gate["binding"]["preregistration_sha256"] = digest(gate["preregistration"])
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(sealed(gate))

    def test_expected_source_binding_and_input_identity(self):
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(gate_fixture(), expected_binding={"source_sha256": digest("other source")})
        gate = gate_fixture()
        row = gate["metrics"]["dev"]["FT"][0]
        row["input_sha256"] = digest("changed prompt")
        gate["metrics"]["dev"]["FT"][0] = sealed(row)
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(sealed(gate))

    def test_raw_text_is_regraded_instead_of_caller_correct_flag(self):
        gate = gate_fixture()
        rows = gate["metrics"]["dev"]["FT"]
        for index, row in enumerate(rows):
            row.update(text="Final answer: B", correct=True)
            rows[index] = sealed(row)
        self.assertEqual(runtime.validate_l0_gate(sealed(gate))["verdict"], "STOP")

    def test_seed_duplicate_and_corruption_rejected(self):
        for kind in ("seed", "duplicate", "corrupt"):
            gate = gate_fixture()
            rows = gate["metrics"]["dev"]["FT"]
            if kind == "seed":
                rows[0] = sealed(rows[0] | {"seed": 99})
            elif kind == "duplicate":
                rows[1] = copy.deepcopy(rows[0])
            else:
                rows[0]["text"] += " changed"
            with self.assertRaises(ValueError):
                runtime.validate_l0_gate(sealed(gate))

    def test_required_ablation_columns_and_positive_effects(self):
        gate = gate_fixture()
        gate["diagnostics"]["ablations"]["merger_f0"] = [-0.9] * 40
        self.assertIn("NO_POSITIVE_CONTRIBUTION:merger_f0", runtime.validate_l0_gate(sealed(gate))["reasons"])
        del gate["diagnostics"]["ablations"]["language_off"]
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(sealed(gate))

    def test_bf16_receipt_is_diagnostic_not_fabricated_success(self):
        gate = gate_fixture()
        gate["diagnostics"]["bf16"]["changed_fraction"] = 0.1
        result = runtime.validate_l0_gate(sealed(gate))
        self.assertTrue(result["passed"])
        self.assertFalse(result["merger_measurable"])
        gate["diagnostics"]["bf16"]["cosine"] = float("nan")
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(sealed(gate))

    def test_reference_coverage_reports_its_exact_matching_limit(self):
        rows = [record("p", observation=True)]
        rows[0]["text"] = rows[0]["target"]
        self.assertEqual(runtime.observation_summary(rows)["strict_reference_coverage"], 1)
        rows[0]["text"] = "A paraphrased pair of vertices."
        self.assertEqual(runtime.observation_summary(rows)["strict_reference_coverage"], 0)
        self.assertEqual(runtime.observation_summary([{"target": "red cube\nblue sphere", "text": "red cube"}])[
            "strict_reference_coverage"], 0.5)
        graph = runtime.observation_summary([{"target": "Edges: 1-2, 2-3.", "text": "Edges: 1-2."}])
        self.assertEqual(graph["reference_units"], 2)
        self.assertEqual(graph["reference_units_matched"], 1)

    def test_duplicate_train_parent_across_blocks_rejected(self):
        gate = gate_fixture()
        gate["preregistration"]["train_blocks"][1][0] = gate["preregistration"]["train_blocks"][0][0]
        gate["binding"]["preregistration_sha256"] = digest(gate["preregistration"])
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(sealed(gate))

    def test_reload_producer_same_pid_is_rejected(self):
        gate = gate_fixture()
        receipt = gate["reload"]
        receipt["reload_pid"] = receipt["producer_pid"]
        gate["reload"] = sealed(receipt)
        with self.assertRaises(ValueError):
            runtime.validate_l0_gate(sealed(gate))

    def test_sentinel_exact_boundaries(self):
        baseline = [record(str(i), tokens=100) for i in range(16)]
        self.assertEqual(runtime.sentinel_stop(baseline, [record(str(i), tokens=50) for i in range(16)]), [])
        current = [record(str(i), tokens=49) for i in range(16)]
        self.assertIn("SENTINEL_MEDIAN_LOW", runtime.sentinel_stop(baseline, current))
        current = copy.deepcopy(baseline)
        for index in range(4):
            current[index] = sealed(current[index] | {"text": "No conclusion."})
        self.assertIn("SENTINEL_FINAL_MISSING", runtime.sentinel_stop(baseline, current))
        current = copy.deepcopy(baseline)
        for index in range(3):
            current[index] = sealed(current[index] | {"finish_reason": "length"})
        self.assertIn("SENTINEL_LENGTH_LIMIT", runtime.sentinel_stop(baseline, current))


class RequestLedgerTests(unittest.TestCase):
    def test_started_request_without_result_is_never_regenerated(self):
        from fine_tuning.error_note.collect import append_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "generations").mkdir()
            request = digest({"phase": "smoke", "item_id": "p"})
            append_jsonl(out / "requests.jsonl", {"request_id": request, "status": "STARTED"})
            with patch("fine_tuning.error_note.collect._single_generate_batch") as generate:
                with self.assertRaisesRegex(ValueError, "manual recovery"):
                    runtime.generate_once(None, {"problem_id": "p"}, prompt="Q", target=None,
                        manifest=out / "manifest.json", pre={}, phase="smoke", out=out)
                generate.assert_not_called()

    def test_completed_single_response_reused_without_sampling(self):
        from fine_tuning.error_note.collect import append_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "generations").mkdir()
            request = digest({"phase": "FT-P0", "item_id": "p"})
            value = record("p")
            append_jsonl(out / "requests.jsonl", {"request_id": request, "status": "STARTED",
                                                   "input_sha256": value["input_sha256"]})
            runtime._write_new(out / "generations" / f"{request}.json", value)
            bundle = SimpleNamespace(processor=None, model=SimpleNamespace(config=None))
            with patch("fine_tuning.error_note.collect._single_generate_batch") as generate, patch(
                    "fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs",
                    return_value=({"input_sha256": value["input_sha256"]}, [])), patch.object(runtime, "_images", return_value=[]):
                actual = runtime.generate_once(bundle, {"problem_id": "p", "evaluation": value["evaluation"]}, prompt="Q", target=None,
                    manifest=out / "manifest.json", pre={"image": runtime.SMOKE_IMAGE}, phase="FT-P0", out=out)
                self.assertEqual(actual, value)
                generate.assert_not_called()

    def test_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            runtime._write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                runtime._write_new(path, {"original": False})
            self.assertTrue(json.loads(path.read_text())["original"])


def _job(pid, target=None):
    return {"item": {"problem_id": pid, "evaluation": record("p")["evaluation"]}, "prompt": f"Q {pid}",
            "target": target, "phase": "FT-P0"}


def _fake_bundle():
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "tok" + ",".join(map(str, tokens)))
    model = SimpleNamespace(config=None, training=True, eval=lambda: None, train=lambda flag: None,
                            generation_config=SimpleNamespace(eos_token_id=[7], pad_token_id=0))
    return SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer), model=model)


class FakeSampler:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, bundle, rows, prepared, dimensions, **kwargs):
        self.calls.append([row["problem_id"] for row in rows])
        if self.error:
            raise self.error
        return {"tokens": [[len(self.calls), row["seed"] % 100] for row in rows], "reasons": ["stop"] * len(rows)}


class GenerateManyTests(unittest.TestCase):
    pre = {"image": runtime.SMOKE_IMAGE, "master_seed": 3407, "max_new_tokens": 8192}

    def run_many(self, out, pids, sampler, batch_size, *, many=True):
        stub = lambda processor, config, messages, images, image: (
            {"input_sha256": digest(messages[0]["content"][-1]["text"])}, [])
        with patch("fine_tuning.error_note.collect._single_generate_batch", sampler), patch(
                "fine_tuning.notebook_rl.input_preparation.prepare_protocol_inputs", stub), patch.object(
                runtime, "_images", return_value=[]):
            if many:
                return runtime.generate_many(_fake_bundle(), [_job(pid) for pid in pids], manifest=out / "m.json",
                                             pre=self.pre, out=out, batch_size=batch_size)
            item = _job(pids[0])["item"]
            return [runtime.generate_once(_fake_bundle(), item, prompt=f"Q {pids[0]}", target=None,
                                          manifest=out / "m.json", pre=self.pre, phase="FT-P0", out=out)]

    @staticmethod
    def ledger(out):
        return [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]

    @staticmethod
    def rid(pid):
        return digest({"phase": "FT-P0", "item_id": pid})

    def test_generate_many_chunks_in_job_order(self):
        pids = [f"p{i}" for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            out, sampler = Path(tmp), FakeSampler()
            (out / "generations").mkdir()
            records = self.run_many(out, pids, sampler, 2)
            self.assertEqual(sampler.calls, [["p0", "p1"], ["p2", "p3"], ["p4"]])
            self.assertEqual([row["item_id"] for row in records], pids)
            self.assertEqual([row["seed"] for row in records], [sample_seed(pid, "t1", 3407) for pid in pids])
            expected = []
            for chunk in (pids[0:2], pids[2:4], pids[4:]):
                expected += [("STARTED", self.rid(pid)) for pid in chunk] + [("COMPLETED", self.rid(pid)) for pid in chunk]
            self.assertEqual([(row["status"], row["request_id"]) for row in self.ledger(out)], expected)
            for pid, stored in zip(pids, records, strict=True):
                self.assertEqual(json.loads((out / "generations" / f"{self.rid(pid)}.json").read_text()), stored)
            self.assertEqual([row["sha256"] for row in self.ledger(out) if row["status"] == "COMPLETED"],
                             [row["sha256"] for row in records])

    def test_generate_many_resumes_completed_and_generates_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "generations").mkdir()
            first = self.run_many(out, ["p0", "p1"], FakeSampler(), 2)
            sampler = FakeSampler()
            records = self.run_many(out, ["p0", "p1", "p2", "p3"], sampler, 2)
            self.assertEqual(sampler.calls, [["p2", "p3"]])
            self.assertEqual(records[:2], first)
            self.assertEqual([row["item_id"] for row in records], ["p0", "p1", "p2", "p3"])

    def test_generate_many_started_without_record_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "generations").mkdir()
            sampler = FakeSampler(error=RuntimeError("crash between STARTED and COMPLETED"))
            with self.assertRaisesRegex(RuntimeError, "crash"):
                self.run_many(out, ["p0", "p1"], sampler, 2)
            self.assertEqual([row["status"] for row in self.ledger(out)], ["STARTED", "STARTED"])
            retry = FakeSampler()
            with self.assertRaisesRegex(ValueError, "manual recovery"):
                self.run_many(out, ["p0", "p1"], retry, 2)
            self.assertEqual(retry.calls, [])
            self.assertEqual(list((out / "generations").iterdir()), [])

    def test_generate_once_equals_generate_many_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            outs = [Path(tmp) / "once", Path(tmp) / "many"]
            for out in outs:
                (out / "generations").mkdir(parents=True)
            once = self.run_many(outs[0], ["p0"], FakeSampler(), 1, many=False)
            many = self.run_many(outs[1], ["p0"], FakeSampler(), 1)
            self.assertEqual(once, many)
            self.assertEqual(self.ledger(outs[0]), self.ledger(outs[1]))

    def test_prereg_generation_batch_size_bound(self):
        def with_batch(gate, value):
            pre = {k: v for k, v in gate["preregistration"].items() if k != "generation_batch_size"}
            if value is not None:
                pre["generation_batch_size"] = value
            binding = gate["binding"] | {"preregistration_sha256": digest(pre)}
            return sealed(gate | {"preregistration": pre, "binding": binding,
                                  "reload": sealed(gate["reload"] | {"binding": binding})})
        for fixture, validate in ((gate_fixture, runtime.validate_l0_gate),
                                  (bf16_gate_fixture, runtime.validate_bf16_effect_receipt)):
            for good in (1, 32):
                self.assertEqual(validate(with_batch(fixture(), good))["verdict"], "GO")
            for bad in (0, 33, None, True, 2.0, "2"):
                with self.assertRaises(ValueError):
                    validate(with_batch(fixture(), bad))
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "nf4_base.json").write_text("{}")
            files, f0 = {"run_v16.py": digest("source")}, {"model.safetensors": digest("weights")}
            pre = {"schema": "v16-local-prereg-v1", "mode": "l0", "dataset_sha256": digest("data"),
                   "source_files": files, "source_sha256": digest(files), "f0_files": f0, "f0_sha256": digest(f0),
                   "model_path": "/model", "master_seed": 3407, "image": runtime.LOCAL_IMAGE, "cache": tmp,
                   "cache_metadata_sha256": runtime.file_sha256(cache / "nf4_base.json")}
            with patch.object(runtime, "_source_files", return_value=files), patch.object(
                    runtime, "_f0_files", return_value=f0):
                runtime._verify_prereg(pre | {"generation_batch_size": 32}, digest("data"))
                for bad in (0, 33, None, True):
                    with self.assertRaises(ValueError):
                        runtime._verify_prereg(pre | {"generation_batch_size": bad}, digest("data"))
                with self.assertRaises(ValueError):
                    runtime._verify_prereg(pre, digest("data"))
        with self.assertRaises(ValueError):
            runtime.generate_many(None, [], manifest=Path("m"), pre={}, out=Path("o"), batch_size=33)

    def test_cli_generation_batch_size_defaults(self):
        parser = runtime.build_parser()
        local = parser.parse_args(["prepare", "--mode", "l0", "--manifest", "m", "--out", "o", "--manifest-sha256", "x"])
        effect = parser.parse_args(["prepare-effect", "--campaign-plan", "p", "--out", "o"])
        self.assertEqual((local.generation_batch_size, effect.generation_batch_size), (2, 8))


class TinyRuntimeTests(unittest.TestCase):
    def test_actual_tiny_bf16_effect_phases_and_changed_checkpoint_resume(self):
        import torch
        from fine_tuning.error_note.train_v16 import backward_content_block
        from fine_tuning.notebook_rl import checkpoint as ckpt
        from fine_tuning.tests.test_error_note_train_v16 import content_row
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_path, _, protocol, binding = fixture(root)
                protocol["training_contract"] = {"source_weights": {"graph": 1.0},
                    "weights": {"O": 1.0, "J": 0.0, "C": 0.0, "N": 0.0, "A1": 0.0}, "pg_weight": 0}
                protocol["precision"] = "bf16"
                protocol["image"] = runtime.LOCAL_IMAGE | {"min_pixels": 1003520, "max_pixels": 4014080}
                protocol = sealed(protocol)
                binding["protocol_sha256"] = protocol["sha256"]
                bundle = fresh_tiny(model_path, "MOD-RV", protocol)
                optimizer, scheduler = ckpt.make_optimizer(bundle, protocol)
                encoded = content_row()
                files = {"fixture.py": digest("tiny CPU BF16 fixture")}
                sentinel = [f"sentinel-{i}" for i in range(16)]
                manifest = {"items": [{"problem_id": pid, "layer": "graph", "evaluation": record(pid)["evaluation"]}
                                      for pid in ["dev", *sentinel]], "rows": []}
                pre = {"schema": "v16-effect-prereg-v1", "precision": "bf16", "mode": "bf16-effect",
                    "protocol": protocol, "source_files": files, "source_sha256": digest(files),
                    "dataset_sha256": digest("fixture dataset"), "f0_sha256": digest("fixture F0"),
                    "runtime_binding": binding, "dev_ids": ["dev"], "sentinel_ids": sentinel,
                    "master_seed": 3407, "max_new_tokens": 8192, "image": protocol["image"],
                    "generation_batch_size": 3}
                def authored(item, **kwargs):
                    return {"problem_id": item["problem_id"], "prompt": "Describe", "target": "reference"}
                def generated(policy, jobs, **kwargs):
                    return [record(job["item"]["problem_id"], observation=job["target"] is not None) for job in jobs]
                with patch.object(runtime, "_source_files", return_value=files), patch(
                        "fine_tuning.error_note.content_v16.load_content_manifest", return_value=manifest), patch(
                        "fine_tuning.error_note.content_v16.observation_row", side_effect=authored), patch(
                        "fine_tuning.error_note.content_v16.validate_content_row"), patch.object(
                        runtime, "_encode", return_value=[encoded]), patch.object(runtime, "generate_many", side_effect=generated) as generate:
                    effect_root = root / "effect"
                    runtime.evaluate_effect_bundle(bundle, pre, manifest, root / "manifest.json", stage="0", output=effect_root)
                    self.assertEqual(generate.call_args.kwargs["batch_size"], 3)
                    for step in range(1, 17):
                        optimizer.zero_grad(set_to_none=True)
                        backward_content_block(bundle, [encoded], protocol["training_contract"], components=("O",))
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        if step in (8, 16):
                            saved = root / f"checkpoint-{step}"
                            ckpt.save(bundle, optimizer, scheduler, step,
                                {"epoch": 0, "position": step, "discarded_problem_ids": []}, ckpt.capture_rng(),
                                saved, protocol=protocol, binding=binding)
                            descriptor = {"path": str(saved / "metadata.json"), "sha256": runtime.file_sha256(saved / "metadata.json")}
                            before = ckpt.capture_rng()
                            phase = runtime.evaluate_effect_bundle(bundle, pre, manifest, root / "manifest.json",
                                stage=str(step), output=effect_root, checkpoint=descriptor)
                            self.assertTrue(runtime._state_equal(before, ckpt.capture_rng()))
                            self.assertIn("file_sha256", phase["checkpoint"])
                    self.assertEqual(set(phase["diagnostics"]["ablations"]), {"on", "vision_off", "language_off", "merger_f0"})
                    self.assertEqual(len(phase["probe"]["greedy_ids"]), 16)
                    (effect_root / "phase16.json").unlink()  # Simulate an interrupted receipt commit.
                    with torch.no_grad():
                        bundle.params["L"][0].add_(0.01)
                    changed = root / "changed-checkpoint-16"
                    ckpt.save(bundle, optimizer, scheduler, 16,
                        {"epoch": 0, "position": 16, "discarded_problem_ids": []}, ckpt.capture_rng(), changed,
                        protocol=protocol, binding=binding)
                    descriptor = {"path": str(changed / "metadata.json"), "file_sha256": runtime.file_sha256(changed / "metadata.json")}
                    previous_calls = generate.call_count
                    with self.assertRaisesRegex(ValueError, "interrupted phase"):
                        runtime.evaluate_effect_bundle(bundle, pre, manifest, root / "manifest.json", stage="16",
                            output=effect_root, checkpoint=descriptor)
                    self.assertEqual(generate.call_count, previous_calls)
        finally:
            torch.set_num_threads(threads)

    def test_runtime_state_comparison_includes_tensor_values_and_structure(self):
        import torch

        state = {"rng": torch.tensor([1, 2]), "scheduler": [1.0, (2,)]}
        self.assertTrue(runtime._state_equal(state, copy.deepcopy(state)))
        changed = copy.deepcopy(state)
        changed["rng"][0] = 9
        self.assertFalse(runtime._state_equal(state, changed))
        self.assertFalse(runtime._state_equal([1], (1,)))

    def test_one_real_cpu_tiny_update_uses_existing_checkpoint(self):
        import torch
        from fine_tuning.tests.test_error_note_train_v16 import content_row
        from fine_tuning.tests.test_notebook_rl_checkpoint import fixture, fresh_tiny

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_path, _, protocol, binding = fixture(root)
                protocol["training_contract"] = {"source_weights": {"graph": 1.0},
                    "weights": {"O": 1.0, "J": 0.0, "C": 0.0, "N": 0.0, "A1": 0.0}, "pg_weight": 0}
                protocol = sealed(protocol)
                row = content_row()
                pid = row["problem_id"]
                pre = {"mode": "smoke", "train_blocks": [[pid]], "model_path": str(model_path),
                       "source_sha256": binding["code_sha256"], "dataset_sha256": binding["data_sha256"],
                       "f0_sha256": digest("tiny-test-only"), "image": runtime.SMOKE_IMAGE, "master_seed": 3407}
                manifest = {"items": [{"problem_id": pid, "split": "train"}], "rows": [row | {"loss_mask": []}]}
                args = argparse.Namespace(command="smoke", out=root / "run", manifest=root / "manifest.json",
                                          manifest_sha256=binding["data_sha256"], resume=False)
                with patch.object(runtime, "_load_inputs", return_value=(manifest, pre)), patch.object(
                        runtime, "_local_protocol", return_value=protocol), patch.object(runtime, "_encode", return_value=[row]):
                    self.assertEqual(runtime.run(args, policy_factory=fresh_tiny, device="cpu"), 0)
                result = json.loads((args.out / "result.json").read_text())
                self.assertEqual(result["step"], 1)
                self.assertTrue(result["functional_only"])
                self.assertFalse(result["benchmark_effect_established"])
                self.assertTrue(all(value > 0 for value in result["changed"].values()))
                self.assertTrue((args.out / "checkpoint-1" / "runtime.pt").is_file())
                self.assertFalse((args.out / "l0-gate.json").exists())
                # CPU fixture exercises the reload path; the parent runs the real fresh-process GPU check.
                with patch.object(runtime, "_verify_prereg"), patch.object(runtime, "validate_local_functional_receipt"), patch.object(
                        runtime, "_encode", return_value=[row]), patch(
                        "fine_tuning.error_note.content_v16.load_content_manifest", return_value=manifest), patch(
                        "fine_tuning.error_note.run_v16.os.getpid", return_value=result["producer_pid"] + 1):
                    self.assertEqual(runtime.reload_run(argparse.Namespace(run=args.out),
                                                        policy_factory=fresh_tiny, device="cpu"), 0)
                receipt = json.loads((args.out / "reload.json").read_text())
                self.assertTrue(receipt["runtime_equal"])
                self.assertEqual(receipt["probe_max_abs_difference"], 0)
                self.assertFalse((args.out / "l0-gate.json").exists())
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
