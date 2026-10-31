"""Opt-in scale options: CPU checks with scripted rollouts and real tiny optimizer/checkpoint payloads.

Collection is scripted here (no Qwen generation); the row-seeded sampler itself is exercised on the
tiny real Qwen3-VL in test_notebook_rl_batched_engine and below. Nothing here is GPU, NF4 or BF16
GPU evidence: the "bf16" runs load a tiny CPU model through the bf16 code path only.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from fine_tuning.notebook_rl import checkpoint, local_train as training, scale
from fine_tuning.notebook_rl.compact_note import VERSION as COMPACT
from fine_tuning.notebook_rl.engines import ROW_SEEDED_BATCH_SAMPLER
from fine_tuning.notebook_rl.local_profile import build_protocol, restore_protocol
from fine_tuning.notebook_rl.policy import assert_base_unchanged, configure_policy
from fine_tuning.notebook_rl.schema import ProblemTree
from fine_tuning.tests.test_notebook_rl_policy import config
from fine_tuning.tests.test_notebook_rl_repair_replay import probe_fixture
from fine_tuning.tests.test_training import tiny_model

# Recorded before this change on the unmodified working copy (scratchpad hashes.py, same environment).
HISTORICAL_SHA256 = {
    "default": "cf9146ed9a09119941bff30b2b12481208f38a08b6eaddb33f85a0ff2931b336",
    "repair": "89181dfc7e8e4d306c03d0b340f198a6d2d59acee5b988eaa6ab86eb36dbad33",
    "compact": "e87c4ab05cce479c173b0c94eb287d49f2050bad99320d955794425b09aeb730",
    "flagged": "299c40fb5a030bba7ec65b43ff12bbf4794032c530d10393f2f4b75c731f25b5",
    "synthesis": "d3f3e583e350d6f9ff7f38a7929191197b1093193bcbb0aafe3b302e055ef391"}
HISTORICAL_OPTIONS: dict[str, dict] = {
    "default": {}, "repair": dict(repair_notebook=True, context_limit=24576),
    "compact": dict(repair_notebook=True, context_limit=24576, note_contract=COMPACT),
    "flagged": dict(repair_notebook=True, context_limit=24576, note_contract=COMPACT, answer_verified_route=True,
                    finalization_reward=True, slotless_sources=("hf-x",)),
    "synthesis": dict(verified_note_synthesis_credit=True, context_limit=24576)}
VLLM = {"gpu_memory_utilization": .55, "max_num_seqs": 16, "kv_cache_memory_bytes": 10737418240,
        "enable_prefix_caching": True}


def repair(**options) -> dict:
    return build_protocol(repair_notebook=True, context_limit=24576, scale=options)


def sync_rows(step: int) -> list[dict]:
    return [{"request_id": "local-sync:" + pid, "policy_version": step, "mean_abs_delta": 0., "max_abs_delta": 0.}
            for pid in ("0", "1")]


class ProtocolTests(unittest.TestCase):
    def test_default_compact_flagged_and_legacy_hashes_are_unchanged(self):
        for name, options in HISTORICAL_OPTIONS.items():
            with self.subTest(name=name):
                protocol = build_protocol(**options)
                self.assertEqual(protocol["sha256"], HISTORICAL_SHA256[name])
                self.assertNotIn("scale", protocol)
                self.assertEqual(scale.options(protocol), scale.DEFAULTS)

    def test_group_binds_only_when_given_and_round_trips(self):
        cases = [{"rollout_batch_rows": 4}, {"policy_precision": "bf16"},
                 {"problems_per_update": 4, "max_updates": 4, "train_problems": 16, "checkpoint_every_updates": 2},
                 {"evidence_audit_rule": {"first": True, "last": True, "every": 5}},
                 {"policy_precision": "bf16", "rollout_engine": "vllm", "vllm": VLLM},
                 {"park_adam_moments": False}]
        hashes = set()
        for options in cases:
            with self.subTest(options=options):
                protocol = repair(**options)
                self.assertEqual(protocol["scale"], {"version": scale.VERSION, **options})
                self.assertEqual(restore_protocol(json.loads(json.dumps(protocol)))["sha256"], protocol["sha256"])
                hashes.add(protocol["sha256"])
                self.assertEqual(protocol["batch"]["problems_per_step"], scale.options(protocol)["problems_per_update"])
                self.assertEqual(protocol["local_budget"]["train_updates"], scale.options(protocol)["max_updates"])
                self.assertIs("quantization" in protocol, scale.options(protocol)["policy_precision"] == "nf4")
        self.assertEqual(len(hashes), len(cases))
        self.assertNotIn(HISTORICAL_SHA256["repair"], hashes)
        forged = repair(rollout_batch_rows=4)
        forged["scale"]["rollout_batch_rows"] = 8
        with self.assertRaises(ValueError):
            restore_protocol(forged)

    def test_invalid_groups_fail_closed(self):
        for options in ({}, {"unknown": 1}, {"rollout_batch_rows": 0}, {"rollout_batch_rows": True},
                        {"problems_per_update": 3}, {"evidence_audit_rule": {"first": False, "last": False, "every": 0}},
                        {"evidence_audit_rule": {"first": 1, "last": True, "every": 0}},
                        {"policy_precision": "fp8"}, {"rollout_engine": "vllm", "vllm": VLLM},
                        {"policy_precision": "bf16", "rollout_engine": "vllm"},
                        {"policy_precision": "bf16", "rollout_engine": "vllm", "vllm": VLLM, "rollout_batch_rows": 2},
                        {"policy_precision": "bf16", "rollout_engine": "vllm", "vllm": VLLM | {"max_num_seqs": 0}},
                        {"vllm": VLLM}, {"park_adam_moments": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                repair(**options)
        with self.assertRaises(ValueError):
            build_protocol(scale={"rollout_batch_rows": 2})

    def test_cli_flags_reach_every_protocol_builder_only_when_given(self):
        from fine_tuning.notebook_rl import local_run, passrate

        base = ["minimal", "--preparation", "p", "--model-path", "m", "--output", "o", "--repair-notebook",
                "--context-limit", "24576"]
        with patch.object(local_run, "capture") as called:
            self.assertEqual(local_run.main(base), 0)
            self.assertNotIn("scale", called.call_args.kwargs)
            self.assertEqual(local_run.main(base + ["--rollout-batch-rows", "4", "--policy-precision", "bf16",
                "--evidence-audit-rule", '{"first": true, "last": true, "every": 2}']), 0)
            self.assertEqual(called.call_args.kwargs["scale"], {"rollout_batch_rows": 4, "policy_precision": "bf16",
                "evidence_audit_rule": {"first": True, "last": True, "every": 2}})
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            local_run.main(base[:-3] + ["--rollout-batch-rows", "4"])
        args = SimpleNamespace(rollout_batch_rows=None, problems_per_update=None, checkpoint_every_updates=None,
            train_problems=None, max_updates=None, evidence_audit_rule=None, policy_precision="bf16",
            rollout_engine="vllm", vllm_gpu_memory_utilization=.55, vllm_max_num_seqs=16,
            vllm_kv_cache_memory_bytes=10737418240, vllm_enable_prefix_caching=True, no_park_adam_moments=False)
        self.assertEqual(scale.from_args(args)["scale"]["vllm"], VLLM)
        with patch.object(passrate, "_cli_measure", return_value={"status": "x"}) as measure, patch("builtins.print"):
            passrate.main(["measure", "--build-root", "b", "--model-path", "m", "--k", "4", "--seed-namespace", "n",
                           "--output", "o", "--rollout-batch-rows", "4"])
        self.assertEqual(measure.call_args.args[0].rollout_batch_rows, 4)

    def test_resume_and_problem_range_require_the_scale_group(self):
        protocol = build_protocol(repair_notebook=True, context_limit=24576)
        parent = {"launch": {"protocol": protocol, "model_path": "m", "cache_check": "c"}, "files": []}
        with patch.object(training, "training_parent_gate", return_value=parent), \
                patch.object(training, "read_preparation") as read:
            for options in ({"resume_from": Path("/x")}, {"problem_range": (0, 10)}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "bound scale group"):
                    training.train(Path("p"), Path("m"), Path("c"), Path("min"), Path("rel"), Path("o"),
                                   arm="notebook", admission=training.REPAIR_FUNCTIONAL_EXPLORATION, **options)
        read.assert_not_called()
        for value in ("3:2", "a:4", "4", "-1:3"):
            with self.subTest(value=value), self.assertRaises(Exception):
                training._problem_range(value)
        self.assertEqual(training._problem_range("300:600"), (300, 600))

    def test_audit_rule_and_three_state_fold(self):
        protocol = repair(evidence_audit_rule={"first": True, "last": True, "every": 2},
                          problems_per_update=2, max_updates=5, train_problems=10)
        self.assertEqual([scale.audited(protocol, step, final_batch=False) for step in range(5)],
                         [True, True, False, True, True])
        self.assertTrue(scale.audited(repair(rollout_batch_rows=1), 3, final_batch=False))
        self.assertIsNone(scale.fold_s(None, None))
        self.assertIs(scale.fold_s(None, False), False)
        self.assertIs(scale.fold_s(False, None), False)
        self.assertIs(scale.fold_s(False, True), True)
        self.assertIs(scale.fold_s(True, None), True)

    def test_vllm_memory_fraction_is_bound_and_default_unchanged(self):
        from fine_tuning.notebook_rl.engines import vllm_kwargs

        image = {"preprocessing": "qwen_vl_utils_0_0_14", "min_pixels": 1, "max_pixels": 2}
        self.assertEqual(vllm_kwargs(Path("m"), image, 1, 1, 1)["gpu_memory_utilization"], .7)
        self.assertEqual(vllm_kwargs(Path("m"), image, 1, 1, 1, gpu_memory_utilization=.55)["gpu_memory_utilization"], .55)
        for value in (0., 1.5, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vllm_kwargs(Path("m"), image, 1, 1, 1, gpu_memory_utilization=value)


class CacheReadinessTests(unittest.TestCase):
    """quantized_cache check -> protocol.json, then the real local_run minimal -> capture -> cache_readiness."""
    REPAIR = ["--repair-notebook", "--context-limit", "24576"]
    FLAGGED = [*REPAIR, "--note-contract", COMPACT, "--answer-verified-route", "--slotless-source", "hf-x",
               "--finalization-reward", "--rollout-batch-rows", "4",
               "--evidence-audit-rule", '{"first": true, "last": true, "every": 2}']

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "model").mkdir()
        (self.root / "model" / "nf4_base.json").write_text('{"fixture_only":true}')

    def cache_check(self, name: str, flags: list[str]) -> Path:
        from fine_tuning.notebook_rl import quantized_cache

        output = self.root / name
        with patch("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={}), \
                patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
                patch("torch.cuda.max_memory_reserved", return_value=0), patch("traceback.print_exc"), \
                patch("builtins.print"):
            code = quantized_cache.main(["check", "--cache", str(self.root / "cache"), "--output", str(output),
                                         "--parent", str(self.root / "absent-parent"), *flags])
        self.assertEqual(code, 1)  # stops at the absent parent, after sources.json/protocol.json are written
        (output / "result.json").write_text(json.dumps({
            "status": "NF4_CACHE_FRESH_PROCESS_RELOAD_GENERATION_VERIFIED", "denied_original_access_attempts": 0,
            "cache_metadata_sha256": checkpoint.file_hash(self.root / "model" / "nf4_base.json")}))
        return output

    def readiness_passes(self, check: Path, flags: list[str]) -> bool:
        from fine_tuning.notebook_rl import local_run

        with patch.object(local_run, "read_preparation", side_effect=RuntimeError("past readiness")) as read, \
                patch("traceback.print_exc"):
            code = local_run.main(["minimal", "--preparation", str(self.root / "prep.json"), "--model-path",
                                   str(self.root / "model"), "--standalone-base", "--cache-check", str(check),
                                   "--output", str(self.root / "run"), *flags])
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "run").exists())
        return read.called

    def test_flagged_and_scaled_runs_pass_only_with_an_identically_flagged_cache_check(self):
        flagged = self.cache_check("flagged", self.FLAGGED)
        compact = self.cache_check("compact", [*self.REPAIR, "--note-contract", COMPACT])
        self.assertEqual(json.loads((compact / "protocol.json").read_text())["sha256"], HISTORICAL_SHA256["compact"])
        self.assertEqual(json.loads((flagged / "protocol.json").read_text())["scale"]["rollout_batch_rows"], 4)
        self.assertTrue(self.readiness_passes(flagged, self.FLAGGED))
        self.assertTrue(self.readiness_passes(compact, [*self.REPAIR, "--note-contract", COMPACT]))
        mismatches = [[flag for flag in self.FLAGGED if flag != "--finalization-reward"],
                      [*self.FLAGGED[:-4], "--rollout-batch-rows", "2", *self.FLAGGED[-2:]],
                      [*self.REPAIR, "--note-contract", COMPACT]]
        for flags in mismatches:
            with self.subTest(flags=flags):
                self.assertFalse(self.readiness_passes(flagged, flags))
        self.assertFalse(self.readiness_passes(compact, self.FLAGGED))

    def test_cache_check_refuses_bf16_and_unbound_flags_before_any_output(self):
        from fine_tuning.notebook_rl import quantized_cache

        for flags in ([*self.REPAIR, "--policy-precision", "bf16"], ["--rollout-batch-rows", "4"],
                      [*self.REPAIR, "--answer-verified-route"], [*self.REPAIR, "--finalization-reward"]):
            with self.subTest(flags=flags), patch("sys.stderr"), self.assertRaises(SystemExit):
                quantized_cache.main(["check", "--cache", "c", "--output", str(self.root / "never"),
                                      "--parent", "p", *flags])
            self.assertFalse((self.root / "never").exists())


class ReceiptTests(unittest.TestCase):
    def test_peak_memory_counts_device_use_held_by_another_process(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "is_initialized", return_value=True), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=0), \
                patch.object(torch.cuda, "mem_get_info", return_value=(38 * 2**30, 48 * 2**30)):
            self.assertEqual(scale.peak_memory(), (10.0, "cuda_device_used_at_receipt"))
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "is_initialized", return_value=True), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=12 * 2**30), \
                patch.object(torch.cuda, "mem_get_info", return_value=(38 * 2**30, 48 * 2**30)):
            self.assertEqual(scale.peak_memory(), (12.0, "cuda_max_memory_reserved"))

    def test_receipt_is_read_by_the_pod_cost_model(self):
        from fine_tuning.notebook_rl.pod import cost

        with tempfile.TemporaryDirectory() as directory:
            receipt = scale.write_receipt(Path(directory), items=4, completion_tokens=1200, wall_seconds=100.,
                updates=2, update_seconds=30., n_samples=4, retries_per_item=.5, startup_seconds=10.,
                extra={"sampler": scale.sampler_label(repair(rollout_batch_rows=4))})
            path = Path(directory) / scale.RECEIPT_NAME
            self.assertEqual(json.loads(path.read_text()), receipt)
            plan = {"version": cost.PLAN_VERSION, "hourly_usd": .59, "stages": [
                {"name": "pilot", "items": 300, "n_samples": 4, "retries_per_item": .5}]}
            projection = cost.project(json.loads(path.read_text()), plan, receipt_sha256="a" * 64, plan_sha256="b" * 64)
        self.assertEqual(receipt["label"], "measured")
        self.assertEqual(receipt["sampler"], {"engine": "hf", "sampler_version": ROW_SEEDED_BATCH_SAMPLER, "batch_rows": 4})
        self.assertGreater(receipt["peak_memory_gib"], 0)
        self.assertEqual(projection["derived"]["generation_seconds_per_item"], 15.)
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"MMDL_RUN_DIR": directory}):
            run = Path(directory) / "train"
            run.mkdir()
            written = scale.write_receipt(run, items=1, completion_tokens=5, wall_seconds=2., updates=0,
                update_seconds=0., n_samples=4, retries_per_item=0., startup_seconds=1.)
            mirrored = json.loads((Path(directory) / scale.RECEIPT_NAME).read_text())
            self.assertEqual(mirrored, written | {"mirrored_from": str((run / scale.RECEIPT_NAME).resolve())})
        for bad in ({"items": -1}, {"wall_seconds": math.nan}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                scale.write_receipt(Path(directory), **({"items": 1, "completion_tokens": 1, "wall_seconds": 2.,
                    "updates": 0, "update_seconds": 0., "n_samples": 1, "retries_per_item": 0.,
                    "startup_seconds": 0.} | bad))


class TinyEngineTests(unittest.TestCase):
    """Actual tiny Qwen3-VL on CPU float32; the probe label path, not GPU drift."""

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def bundle(self):
        from fine_tuning.tests.test_notebook_rl_shared_policy_engine import processor

        return configure_policy(tiny_model(), processor(), "MOD-RV", config(), enforce_counts=False)

    def protocol(self, **options):
        value = repair(**options) if options else build_protocol(repair_notebook=True, context_limit=24576)
        return value | {"image": {"preprocessing": "hf_processor"}, "context_limit": 64}

    def probes(self, engine):
        from fine_tuning.notebook_rl.trainer import SyncProbe
        from fine_tuning.tests.test_notebook_rl_shared_policy_engine import request

        return [SyncProbe(replace_id(request(count), "local-sync:" + pid), [7, 8]) for pid, count in (("0", 1), ("1", 2))]

    def test_bound_batching_labels_probes_and_replay_accepts_them_only_when_bound(self):
        from fine_tuning.notebook_rl.local_run import probe_evidence
        from fine_tuning.notebook_rl.repair_replay import validate_probe_evidence

        serial = scale.shared_engine(self.bundle(), self.protocol(), context_limit=64)
        batched = scale.shared_engine(self.bundle(), self.protocol(rollout_batch_rows=4), context_limit=64)
        self.assertIsNone(serial.batch_rows)
        self.assertEqual(batched.batch_rows, 4)
        outputs = {}
        for name, engine in (("serial", serial), ("batched", batched)):
            engine.validate_sync([(p.request, p.completion_ids, engine.probe_logprobs(p.request, p.completion_ids))
                                  for p in self.probes(engine)], .02)
            outputs[name] = probe_evidence(engine, self.probes(engine))
        rows = outputs["batched"]["generated"]
        self.assertTrue(all(row["batch_rows"] == 4 and row["sampler_version"] == ROW_SEEDED_BATCH_SAMPLER for row in rows))
        # n=1 rows: the batched path is token- and logprob-identical to serial on CPU float32.
        self.assertEqual([row["completions"] for row in rows],
                         [row["completions"] for row in outputs["serial"]["generated"]])
        for probes, rows_bound, accepted in ((outputs["serial"], None, True), (outputs["batched"], 4, True),
                                             (outputs["batched"], None, False), (outputs["serial"], 4, False),
                                             (outputs["batched"], 2, False)):
            with self.subTest(rows=rows_bound, batched=probes is outputs["batched"]):
                if accepted:
                    validate_probe_evidence(copy.deepcopy(probes), ["0", "1"], [1, 2], batch_rows=rows_bound)
                else:
                    with self.assertRaises(ValueError):
                        validate_probe_evidence(copy.deepcopy(probes), ["0", "1"], [1, 2], batch_rows=rows_bound)

    def test_vllm_sync_pushes_merged_weights_after_transformers_sync_and_before_sampler_probes(self):
        from fine_tuning.notebook_rl.policy import merged_named_tensors

        bundle = self.bundle()
        protocol = self.protocol(policy_precision="bf16", rollout_engine="vllm", vllm=VLLM)
        shared = scale.shared_engine(bundle, protocol, context_limit=64)
        events: list[tuple] = []
        expected = {name for name, _ in merged_named_tensors(bundle)}

        class FakeVLLM:
            policy_version = 0

            def push_weights(self, named, version):
                tensors = dict(named)
                events.append(("push", version, set(tensors) == expected,
                               {value.dtype for value in tensors.values()} == {torch.bfloat16}))
                self.policy_version = version

            def validate_sync(self, probes, threshold):
                events.append(("sampler-probes", self.policy_version, [p[0].request_id for p in probes]))
                return sync_rows(self.policy_version)

        original = shared.validate_sync

        def hf_sync(probes, threshold):
            events.append(("hf-probes", shared.policy_version))
            return original(probes, threshold)

        shared.validate_sync = hf_sync
        sampler = FakeVLLM()
        hf, vllm = scale.synchronize(bundle, sampler, shared, self.probes(shared), protocol, 0, push=False)
        self.assertEqual(vllm, sync_rows(0))
        self.assertEqual(len(hf), 2)
        with torch.no_grad():
            bundle.params["L"][0].add_(.01)
        scale.synchronize(bundle, sampler, shared, self.probes(shared), protocol, 1, push=True)
        self.assertEqual(events, [("hf-probes", 0), ("sampler-probes", 0, ["local-sync:0", "local-sync:1"]),
                                  ("hf-probes", 1), ("push", 1, True, True),
                                  ("sampler-probes", 1, ["local-sync:0", "local-sync:1"])])
        self.assertEqual((shared.policy_version, sampler.policy_version), (1, 1))
        hf, vllm = scale.synchronize(bundle, shared, shared, self.probes(shared), protocol, 1, push=False)
        self.assertIsNone(vllm)

    def test_bf16_guard_detects_frozen_base_changes_and_nf4_guard_is_unchanged(self):
        bundle = self.bundle()
        guard = scale.base_guard(bundle, self.protocol(policy_precision="bf16"))
        guard(bundle)
        with torch.no_grad():
            next(p for name, p in bundle.model.get_base_model().named_parameters() if "lora_" not in name).add_(1.)
        with self.assertRaisesRegex(ValueError, "checksum"):
            guard(bundle)
        with patch("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged") as nf4:
            scale.base_guard(bundle, self.protocol())(bundle)
        nf4.assert_called_once_with(bundle)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            with patch("torch.cuda.is_available", return_value=False):
                scale.build_bf16_policy(Path("m"), "MOD-RV", self.protocol(policy_precision="bf16"))

    def test_bf16_checksum_converts_bounded_chunks_and_sees_every_single_element_change(self):
        from fine_tuning.notebook_rl.policy import _canonical_frozen

        bundle = configure_policy(tiny_model().to(torch.bfloat16), None, "MOD-RV", config(), enforce_counts=False)
        name, tensor = max(_canonical_frozen(bundle), key=lambda row: row[1].numel())
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertGreater(tensor.numel(), 4 * 16)
        converted: list[int] = []
        original = torch.Tensor.to

        def spy(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            if result.dtype == torch.float64:
                converted.append(result.numel())
            return result

        with patch.object(scale, "_CHECKSUM_CHUNK", 16), patch.object(torch.Tensor, "to", spy):
            first = scale._checksum(bundle)
            self.assertEqual(first, scale._checksum(bundle))
            # Large tensors never become one float64 temporary; every conversion is at most one chunk.
            self.assertTrue(converted and max(converted) <= 16)
            row = next(row for row in first if row[0] == name)
            exact = tensor.detach().double()
            self.assertAlmostEqual(row[1], float(exact.sum()), places=9)
            self.assertAlmostEqual(row[2], float(exact.square().sum().sqrt()), places=9)
            guard = scale.base_guard(bundle, self.protocol(policy_precision="bf16"))
            guard(bundle)
            bits = tensor.data.view(-1).view(torch.int16)
            for index in (0, 17, tensor.numel() - 1):
                with self.subTest(index=index):
                    bits[index] ^= 1  # one mantissa bit of one bfloat16 element
                    self.assertNotEqual(scale._checksum(bundle), first)
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        guard(bundle)
                    bits[index] ^= 1
                    guard(bundle)
            self.assertEqual(scale._checksum(bundle), first)


def replace_id(request, request_id):
    from dataclasses import replace

    return replace(request, request_id=request_id)


class AuditVJPTests(unittest.TestCase):
    """Actual tiny optimizer_step on an authored repair tree with and without evidence VJPs."""

    @classmethod
    def setUpClass(cls):
        from fine_tuning.notebook_rl import repair_evidence
        from fine_tuning.tests import test_notebook_rl_repair_update as fixture

        cls.audit = repair(evidence_audit_rule={"first": True, "last": False, "every": 0})
        cls.runs: dict[bool, Any] = {}
        for measured in (True, False):
            f = fixture.RepairUpdateTests()
            f.setUp()
            cls.addClassCleanup(f.doCleanups)
            tree, engine = f.online()
            update, _, _ = f.step(tree, engine, protocol=cls.audit, measure_evidence_vjp=measured)
            receipts = {tree.problem_id: f.f.prepare(tree, engine)}
            inventory = [{key: value for key, value in row.items() if key != "index"}
                         for row in update["repair_aggregate_gradient"]["parameter_inventory"]]
            cls.runs[measured] = SimpleNamespace(f=f, tree=tree, update=update, receipts=receipts, inventory=inventory,
                weights={key: value.detach().clone() for key, value in f.model.items()})
        cls.verify = staticmethod(lambda run, update=None, protocol=None, **options: repair_evidence.repair_update_evidence(
            run.update if update is None else update, protocol or cls.audit, trees=[run.tree],
            preparations=run.receipts, parameter_inventory=run.inventory, **options))

    def test_non_audit_step_is_unmeasured_with_unknown_s_and_identical_weights(self):
        measured, skipped = self.runs[True], self.runs[False]
        aggregate = skipped.update["repair_aggregate_gradient"]
        self.assertEqual(skipped.update["status"], "UPDATED_UNSYNCHRONIZED")
        self.assertIs(aggregate["measured"], False)
        self.assertIsNone(aggregate["S"])
        self.assertNotIn("channels", aggregate)
        self.assertEqual(aggregate["members"], measured.update["repair_aggregate_gradient"]["members"])
        for key in measured.weights:
            torch.testing.assert_close(skipped.weights[key], measured.weights[key], atol=0, rtol=0)
        result = self.verify(skipped)
        self.assertIsNone(result["S_gradient_verified"])
        self.assertEqual(result["scope"]["vjp_audit"], "NOT_AUDITED_FIXED_RULE_S_UNKNOWN")
        audited = self.verify(measured)
        self.assertIs(audited["S_gradient_verified"], True)
        self.assertEqual(audited["scope"]["vjp_audit"], "AUDITED_FIXED_RULE")

    def test_unmeasured_aggregate_cannot_claim_s_or_escape_the_bound_rule(self):
        skipped = self.runs[False]
        with self.assertRaisesRegex(ValueError, "positive aggregate S"):
            self.verify(skipped, require_s=True)
        mutations = [lambda a: a.update(S=True), lambda a: a.update(S=False), lambda a: a["members"].pop(),
                     lambda a: a.update(basis="same_forward_vjp_sum_by_parameter_identity_cpu_float64"),
                     lambda a: a.update(retained_positive_verified_tokens=0),
                     lambda a: a.update(channels={})]
        for mutate in mutations:
            update = copy.deepcopy(skipped.update)
            mutate(update["repair_aggregate_gradient"])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.verify(skipped, update)
        from fine_tuning.notebook_rl import updates
        from fine_tuning.notebook_rl.local_profile import build_protocol as profile

        default = profile(repair_notebook=True, context_limit=24576)
        execution = skipped.update["repair_execution"] | {"protocol_sha256": default["sha256"]}
        with self.assertRaisesRegex(ValueError, "unmeasured aggregate requires the bound audit rule"):
            self.verify(skipped, skipped.update | {"repair_execution": execution}, protocol=default)
        with self.assertRaisesRegex(ValueError, "repair audit rule"):
            updates.optimizer_step(None, None, SimpleNamespace(last_epoch=0), [], {}, None, config(), "ALG-NB3",
                                   step=0, seed=3407, measure_evidence_vjp=False)


class MinimalAndJournalTests(unittest.TestCase):
    def test_bf16_minimal_capture_loads_the_original_snapshot_without_nf4_cache_or_guard(self):
        from fine_tuning.notebook_rl import local_profile, local_run
        from fine_tuning.tests.test_notebook_rl_repair_local_run import RepairLocalRunTests

        fixture = RepairLocalRunTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        real = local_profile.build_protocol
        bf16 = {"policy_precision": "bf16", "rollout_batch_rows": 2}
        loaded: list[str] = []

        def loader(model_path, arm, protocol, **kwargs):
            loaded.append(scale.options(protocol)["policy_precision"])
            return SimpleNamespace(inventory={"trainable": []}, base_sha256="frozen")

        with patch.object(local_run, "build_protocol", side_effect=lambda **kw: real(**kw, scale=bf16)), \
                patch("fine_tuning.notebook_rl.scale.build_bf16_policy", side_effect=loader), \
                patch("fine_tuning.notebook_rl.scale.base_guard", return_value=lambda bundle: None), \
                patch("fine_tuning.notebook_rl.quantized_cache.block_original_access") as guard:
            result, calls = fixture.run_capture()
        self.assertEqual(result["status"], "NO_MATCHED_ACTIONS")
        self.assertEqual(loaded, ["bf16"])
        guard.assert_not_called()
        launch = json.loads((fixture.root / "capture" / "launch.json").read_text())
        self.assertIs(launch["standalone_base"], False)
        self.assertEqual(launch["sampler"]["batch_rows"], 2)
        self.assertNotIn("quantization", launch["protocol"])
        self.assertTrue((fixture.root / "capture" / "original_files.json").is_file())
        with self.assertRaisesRegex(ValueError, "not an NF4 cache"):
            local_run.capture(fixture.root / "p", fixture.root / "m", fixture.root / "o", standalone=True,
                cache_check=fixture.root / "c", repair_notebook=True, context_limit=24576, scale=bf16)

    def test_resume_prefix_journal_tolerates_only_trailing_abandoned_records(self):
        from fine_tuning.notebook_rl import repair_replay

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = ProblemTree("a", 0, [], "VISION_OFF", {}, {})
            event = {"kind": "TREE", "raw": {"fixture": 1}}
            for name in ("raw-0000.json", "raw-0001.json"):
                (root / name).write_text(json.dumps(event))
            records = [{"file": "raw-0000.json", "kind": "TREE", "sha256": checkpoint.file_hash(root / "raw-0000.json")}]
            with patch.object(repair_replay, "_expected_journal", return_value=[event]):
                self.assertEqual(set(repair_replay.validate_journal(root, records, [tree], complete=False)), {"raw-0000.json"})
                with self.assertRaisesRegex(ValueError, "contiguous complete raw journal"):
                    repair_replay.validate_journal(root, records, [tree])
                (root / "raw-0000.json").rename(root / "raw-0007.json")
                with self.assertRaisesRegex(ValueError, "contiguous complete raw journal"):
                    repair_replay.validate_journal(root, records, [tree], complete=False)


class ScaledTrainFixture:
    """Scripted CPU collection, real tiny AdamW/checkpoint payloads, real scaled replay and reload."""
    options: dict = {}
    count = 10
    stop_after = 99

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.model = self.root / "data", self.root / "base"
        self.data.mkdir()
        self.model.mkdir()
        (self.data / "manifest.yaml").write_text(json.dumps({"split_manifests": {"train": "fixture-vault"}}))
        (self.model / "nf4_base.json").write_text('{"fixture_only":true}')
        self.cache = self.root / "cache"
        self.protocol = repair(**self.options)
        self.bf16 = scale.bf16(self.protocol)
        self.actors = [SimpleNamespace(problem_id=str(i), image_paths=["image"] * (2 if i == 1 else 1))
                       for i in range(self.count)]
        self.prepared = {"build_root": str(self.data), "image_root": str(self.data / "images")}
        parent_capture, parent_reload = self.root / "minimum", self.root / "reload"
        parent_capture.mkdir()
        parent_reload.mkdir()
        (parent_capture / "launch.json").write_text("{}")
        (parent_reload / "sync.json").write_text("{}")
        self.parent = {"launch": {"protocol": self.protocol, "model_path": str(self.model),
                                  "cache_check": None if self.bf16 else str(self.cache)},
            "admission_policy": training.REPAIR_FUNCTIONAL_EXPLORATION, "repair_functional_verified": True,
            "S_gradient_verified": False, "repair_learning_verified": False,
            "files": [{"path": str(path), "sha256": checkpoint.file_hash(path)}
                      for path in (parent_capture / "launch.json", parent_reload / "sync.json")]}
        self.now, self.parked, self.built = 0., [], []
        self.update_calls: list[dict] = []
        self.sync_calls: list[tuple] = []
        self.engines: list[Any] = []
        self.s_true_step: int | None = None
        stack = self.enterContext(ExitStack())

        def patched(name, **kwargs):
            return stack.enter_context(patch(name, **kwargs))

        @contextmanager
        def park(model, optimizer, scheduler, protocol, step):
            self.assertEqual(scheduler.last_epoch, step)
            self.parked.append(step)
            yield lambda: None

        def engine(bundle, image, limit, **kwargs):
            value = SimpleNamespace(kind="shared", policy_version=kwargs.get("policy_version", 0), tokenizer=object(),
                                    kwargs=kwargs)
            self.engines.append(value)
            return value

        self.vllm_class = patched("fine_tuning.notebook_rl.engines.VLLMRolloutEngine", side_effect=self.vllm)
        patched("fine_tuning.notebook_rl.checkpoint.park_adam_moments", side_effect=park)
        patched("fine_tuning.notebook_rl.local_train.training_parent_gate", return_value=self.parent)
        patched("fine_tuning.notebook_rl.local_train.cache_readiness",
                return_value={"original_model_root": str(self.root / "original")})
        patched("fine_tuning.notebook_rl.local_train.read_preparation", return_value=(self.prepared, {}, self.actors))
        patched("fine_tuning.notebook_rl.local_train.local_prompts", return_value={})
        patched("fine_tuning.notebook_rl.local_train._install_actor_guards", return_value=[])
        patched("fine_tuning.notebook_rl.policy.enable_deterministic_execution", return_value={"fixture_only": True})
        self.original_access = patched("fine_tuning.notebook_rl.quantized_cache.block_original_access",
                                       return_value={"denied_original_access_attempts": 0})
        patched("fine_tuning.notebook_rl.quantized_policy.load_quantized_policy", side_effect=self.policy)
        patched("fine_tuning.notebook_rl.scale.build_bf16_policy", side_effect=self.policy)
        patched("fine_tuning.notebook_rl.local_calibrate.verify_original",
                return_value={"model.safetensors": {"bytes": 1, "sha256": "c" * 64}})
        patched("fine_tuning.notebook_rl.quantized_policy.assert_quantized_base_unchanged", side_effect=assert_base_unchanged)
        patched("fine_tuning.notebook_rl.shared_policy_engine.SharedPolicyRolloutEngine", side_effect=engine)
        patched("fine_tuning.notebook_rl.local_train.load_images", side_effect=lambda *_: [MagicMock()])
        service = patched("fine_tuning.notebook_rl.verifier_service.VerifierService")
        service.return_value.__enter__.return_value.metadata = {"vault_manifest_sha256": "fixture-vault"}
        patched("fine_tuning.notebook_rl.local_train.make_probes", return_value=[])
        patched("fine_tuning.notebook_rl.rollout.RolloutConfig.from_protocol", return_value=object())
        patched("fine_tuning.notebook_rl.rollout.build_tree", side_effect=self.build)
        patched("fine_tuning.notebook_rl.notebook_learning.append_rewarded_notebooks")
        patched("fine_tuning.notebook_rl.updates.optimizer_step", side_effect=self.update)
        patched("fine_tuning.notebook_rl.trainer.synchronize", side_effect=self.synchronize)
        patched("fine_tuning.notebook_rl.local_train.probe_evidence", side_effect=lambda *_: self.forward())
        patched("fine_tuning.notebook_rl.repair_training.prepare_tree_credit", side_effect=self.prepare)
        patched("fine_tuning.notebook_rl.repair_evidence.repair_update_evidence", side_effect=self.evidence)
        patched("fine_tuning.notebook_rl.local_train.time.monotonic", side_effect=lambda: self.now)
        for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
            patched("torch.cuda." + name, return_value=0)
        patched("builtins.print")

    # ---- scripted collaborators
    def policy(self, *args, **kwargs):
        bundle = configure_policy(tiny_model(), None, "MOD-RV", config(), enforce_counts=False)
        if not self.bf16:
            bundle.inventory["quantization"] = {"fixture_only": True}
        return bundle

    def vllm(self, model_path, image, kv, seqs, limit, **kwargs):
        value = SimpleNamespace(kind="vllm", policy_version=0, init=(kv, seqs, limit, kwargs))
        self.engines.append(value)
        return value

    def forward(self):
        probes = probe_fixture(["0", "1"])
        rows = scale.options(self.protocol)["rollout_batch_rows"]
        if rows is not None:
            for row in probes["generated"]:
                row.update(batch_rows=rows, sampler_version=ROW_SEEDED_BATCH_SAMPLER)
        return probes

    def build(self, actor, images, engine, verifier, prompts, cfg, version, rule, **options):
        self.built.append((actor.problem_id, version, engine.kind))
        options["on_raw"]({"problem_id": actor.problem_id, "policy_version": version})
        return ProblemTree(actor.problem_id, version, [], "VISION_OFF", {}, {})

    def prepare(self, tree, actor, verifier, tokenizer, *, notebook_reward_enabled=True, preparation=None, **kwargs):
        self.assertEqual(preparation.kind, "shared")
        value = {"admitted": True, "has_signal": True, "admission": "VISION_OFF", "admission_reasons": [],
                 "groups": [], "scripted_boundary_only": True, "notebook_reward_enabled": notebook_reward_enabled}
        return value | {"sha256": checkpoint.digest(value)}

    def update(self, model, optimizer, scheduler, trees, images, preparation, protocol, algorithm, *, step,
               before_optimizer_step, **kwargs):
        measure_evidence_vjp = kwargs.get("measure_evidence_vjp", True)
        self.update_calls.append({"step": step, "ids": [t.problem_id for t in trees], "measure": measure_evidence_vjp,
                                  "audit_kwarg": "measure_evidence_vjp" in kwargs, "preparation": preparation.kind})
        model.route.apply(False)
        sum(p.float().sum() for group in ("L", "M") for p in model.params[group]).backward()
        before_optimizer_step()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step + 1 == self.stop_after:
            self.now = 10 ** 9
        return {"status": "UPDATED_UNSYNCHRONIZED", "step_before": step, "step_after": step + 1,
            "notebook_backward": [], "routes": {t.problem_id: False for t in trees},
            "grad_norm": {"L": 1., "M": 1., "V": None},
            "n_groups_by_kind": {"T1": len(trees), "T2": len(trees), "T3": 0, "DIAG": len(trees)},
            "repair_aggregate_gradient": {"measured": False} if not measure_evidence_vjp else {"fixture": True}}

    def evidence(self, update, protocol, **kwargs):
        measured = update["repair_aggregate_gradient"].get("measured") is not False
        status = (update["step_before"] == self.s_true_step) if measured else None
        return {"version": "repair-update-evidence-v1", "S_gradient_verified": status, "step": update["step_before"],
                "functional_verified": False, "repair_learning_verified": False, "benefit_verified": False}

    def synchronize(self, bundle, preparation, engine, probes, protocol, step, *, push):
        self.sync_calls.append((preparation.kind, engine.kind, step, push))
        engine.policy_version = step
        return sync_rows(step)

    # ---- drivers
    def train(self, output: Path, **kwargs):
        return training.train(self.root / "preparation.json", self.model, None if self.bf16 else self.cache,
            self.root / "minimum", self.root / "reload", output, arm="notebook",
            admission=training.REPAIR_FUNCTIONAL_EXPLORATION, **kwargs)

    def receipts_by_id(self) -> dict:
        receipts = {}
        for path in self.root.glob("**/repair-preparation-*.json"):
            value = json.loads(path.read_text())
            receipts[value["sha256"]] = value
        return receipts

    @contextmanager
    def replaying(self):
        def trees(values, actors, protocol, prepared, model_path, *, control=False):
            return {tree.problem_id: self.prepare(tree, SimpleNamespace(), None, None,
                                                  notebook_reward_enabled=not control,
                                                  preparation=SimpleNamespace(kind="shared")) for tree in values}

        def journal(capture, raw, values, *, complete=True):
            return {row["file"]: row["sha256"] for row in raw}

        with patch("fine_tuning.notebook_rl.repair_replay.replay_trees", side_effect=trees), \
                patch("fine_tuning.notebook_rl.repair_replay.validate_journal", side_effect=journal), \
                patch("psutil.pid_exists", return_value=False):
            yield

    def close(self, capture: Path) -> dict:
        """Simulate the owner process exiting: a different, dead PID (as the existing fixed-ten tests do)."""
        launch = json.loads((capture / "launch.json").read_text())
        launch["pid"] += 100000 + len(list(self.root.glob("*")))
        (capture / "launch.json").write_text(json.dumps(launch))
        return launch

    def replay(self, capture: Path) -> dict:
        launch = json.loads((capture / "launch.json").read_text())
        result = json.loads((capture / "result.json").read_text())
        metadata = checkpoint.read_metadata(capture / "checkpoint", binding=launch["binding"])
        with self.replaying():
            return training._repair_training_replay(capture, result, launch["execution_contract"], self.protocol,
                                                    parameter_inventory=metadata["inventory"]["trainable"])

    def reload(self, capture: Path, output: Path) -> dict:
        original = checkpoint.load_resume

        def cpu_load(model_path, saved, protocol, *, binding, device, policy_factory):
            return original(model_path, saved, protocol, binding=binding, device="cpu", policy_factory=policy_factory)

        launch = json.loads((capture / "launch.json").read_text())
        with self.replaying(), patch.object(checkpoint, "load_resume", side_effect=cpu_load), \
                patch.object(training, "training_parent_gate", return_value=launch["execution_contract"]["parent_gate"]):
            return training.reload_training(capture, output)


class BatchedTrainTests(ScaledTrainFixture, unittest.TestCase):
    options = {"rollout_batch_rows": 4}

    def test_batched_protocol_collect_update_save_replay_reload_with_labelled_receipts(self):
        capture = self.root / "run"
        result = self.train(capture)
        self.assertEqual((result["status"], result["optimizer_updates"]), ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", 5))
        self.assertTrue(self.engines and all(e.kwargs["batch_rows"] == 4 for e in self.engines))
        self.assertFalse(any(call["audit_kwarg"] for call in self.update_calls))  # no rule: default call shape
        label = {"engine": "hf", "sampler_version": ROW_SEEDED_BATCH_SAMPLER, "batch_rows": 4}
        launch = self.close(capture)
        self.assertEqual(launch["sampler"], label)
        receipt = json.loads((capture / scale.RECEIPT_NAME).read_text())
        self.assertEqual((receipt["sampler"], receipt["items"], receipt["updates"], receipt["label"]),
                         (label, 10, 5, "measured"))
        self.assertEqual(receipt["group_weight_denominators"], [2 * (1. + 1. + .5)] * 5)
        pair = json.loads((capture / "pair-00.json").read_text())
        self.assertEqual(pair["group_weight_scale"]["per_group_weight"]["T1"], 1. / 5.)
        proof = self.replay(capture)
        self.assertEqual((proof["version"], proof["optimizer_updates"]), ("repair-scaled-training-replay-v1", 5))
        restored = self.reload(capture, self.root / "fresh")
        self.assertEqual((restored["step"], restored["repair_replay_sha256"]), (5, proof["sha256"]))
        self.assertTrue(restored["optimizer_restored"] and restored["repair_functional_verified"])
        # A replay must not accept serial-labelled probes for a batched protocol.
        expected = json.loads((capture / "reload_expected.json").read_text())
        for row in expected["forward_probes"]["generated"]:
            del row["batch_rows"]
        (capture / "reload_expected.json").write_text(json.dumps(expected))
        with self.assertRaises(ValueError):
            self.replay(capture)


class LargerBatchTests(ScaledTrainFixture, unittest.TestCase):
    options = {"problems_per_update": 4, "max_updates": 4, "train_problems": 16, "checkpoint_every_updates": 2}
    count = 16

    def test_four_problem_updates_checkpoint_every_two_then_replay_and_reload(self):
        capture = self.root / "run"
        result = self.train(capture)
        self.assertEqual(result["optimizer_updates"], 4)
        self.assertEqual([call["ids"] for call in self.update_calls],
                         [[str(i) for i in range(start, start + 4)] for start in (0, 4, 8, 12)])
        self.assertEqual(sorted(p.name for p in capture.iterdir() if p.is_dir()),
                         ["checkpoint", "checkpoint-pair-01", "checkpoint-pair-03"])
        self.assertEqual(len(self.sync_calls), 5)  # initial + after every update, not only at checkpoints
        committed = json.loads((capture / "checkpoint-pair-01.json").read_text())
        self.assertTrue(committed["resume_supported"])
        self.assertEqual(committed["cursor"]["position"], 8)
        launch = self.close(capture)
        self.assertEqual(launch["execution_contract"]["max_optimizer_updates"], 4)
        self.assertEqual(json.loads((capture / "pair-00.json").read_text())["group_weight_scale"]["denominator"],
                         4 * (1. + 1. + .5))
        proof = self.replay(capture)
        self.assertEqual(proof["optimizer_updates"], 4)
        self.assertEqual(self.reload(capture, self.root / "fresh")["step"], 4)
        metadata = (capture / "checkpoint-pair-01" / "metadata.json").read_bytes()
        (capture / "checkpoint-pair-01" / "metadata.json").write_bytes(metadata + b" ")
        with self.assertRaises(ValueError):
            self.replay(capture)
        (capture / "checkpoint-pair-01" / "metadata.json").write_bytes(metadata)
        (capture / "checkpoint-pair-00").mkdir()
        with self.assertRaisesRegex(ValueError, "off the bound cadence"):
            self.replay(capture)

    def test_resume_recollects_after_last_checkpoint_and_matches_an_uninterrupted_run(self):
        full = self.root / "full"
        self.train(full)
        self.close(full)
        straight = json.loads((full / "reload_expected.json").read_text())["trainable_sha256"]
        self.update_calls.clear()
        self.built.clear()
        self.stop_after = 3  # update 3 commits after the cadence checkpoint at update 2, then the run stops
        first = self.root / "segment-1"
        stopped = self.train(first)
        self.assertEqual((stopped["status"], stopped["optimizer_updates"]), ("BUDGET_STOP", 3))
        self.assertTrue((first / "checkpoint-pair-01").is_dir() and not (first / "checkpoint-pair-02").exists())
        self.close(first)
        self.stop_after, self.now = 99, 0.
        built_before = len(self.built)
        second = self.root / "segment-2"
        original = checkpoint.load_resume

        def cpu_load(model_path, saved, protocol, *, binding, device, policy_factory):
            return original(model_path, saved, protocol, binding=binding, device="cpu", policy_factory=policy_factory)

        with self.replaying(), patch.object(checkpoint, "load_resume", side_effect=cpu_load):
            resumed = self.train(second, resume_from=first)
        self.assertEqual((resumed["status"], resumed["optimizer_updates"]), ("LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD", 4))
        self.assertEqual(resumed["resumed_from"]["checkpoint"], "checkpoint-pair-01")
        # Batch 2 (after the last checkpoint) is re-collected at the same policy version, then batch 3.
        self.assertEqual(self.built[built_before:], [(str(i), 2 if i < 12 else 3, "shared") for i in range(8, 16)])
        self.assertEqual(json.loads((second / "reload_expected.json").read_text())["trainable_sha256"], straight)
        self.assertEqual(resumed["attempted_ids"], [str(i) for i in range(16)])
        self.close(second)
        proof = self.replay(second)
        self.assertEqual([link["capture"] for link in proof["chain"]], [str(first)])
        self.assertEqual(self.reload(second, self.root / "fresh")["step"], 4)
        # Changing the closed parent segment breaks the chain.
        evidence = first / "repair-evidence-00.json"
        evidence.write_text(evidence.read_text() + " ")
        with self.assertRaises(ValueError):
            self.replay(second)

    def test_resume_skips_an_interrupted_checkpoint_save(self):
        self.stop_after = 3
        first = self.root / "segment-1"
        self.train(first)
        # A crash/outer kill inside checkpoint.save leaves a directory without its commit record.
        (first / "checkpoint-pair-02").mkdir()
        (first / "checkpoint").mkdir()
        self.close(first)
        self.stop_after, self.now = 99, 0.
        original = checkpoint.load_resume

        def cpu_load(model_path, saved, protocol, *, binding, device, policy_factory):
            return original(model_path, saved, protocol, binding=binding, device="cpu", policy_factory=policy_factory)

        with self.replaying(), patch.object(checkpoint, "load_resume", side_effect=cpu_load):
            resumed = self.train(self.root / "segment-2", resume_from=first)
        self.assertEqual((resumed["optimizer_updates"], resumed["resumed_from"]["checkpoint"]), (4, "checkpoint-pair-01"))

    def test_disk_low_stops_at_the_first_checkpoint_without_writing_one(self):
        from fine_tuning.notebook_rl.local_run import host_identity

        capture = self.root / "low"
        with patch("shutil.disk_usage", return_value=SimpleNamespace(total=10, used=9, free=1)):
            result = self.train(capture, min_free_gib=1.0)
        self.assertEqual(result["status"], "DISK_LOW")
        self.assertEqual(result["disk_low"]["free_bytes"], 1)
        self.assertEqual([p.name for p in capture.iterdir() if p.name.startswith("checkpoint")], [])
        self.assertFalse((capture / "reload_expected.json").exists())
        self.assertEqual(json.loads((capture / "launch.json").read_text())["host"], host_identity())
        self.assertEqual(json.loads((capture / "result.json").read_text())["status"], "DISK_LOW")

    def test_disk_low_before_the_final_save_leaves_committed_intermediates_and_no_partial_final(self):
        capture = self.root / "low-final"
        low = {"path": str(capture), "free_bytes": 1, "min_free_bytes": 2**30}
        with patch.object(training, "disk_low", side_effect=[None, None, low]) as guard:
            result = self.train(capture, min_free_gib=1.0)
        self.assertEqual(guard.call_count, 3)  # two intermediate saves, then the final one
        self.assertEqual((result["status"], result["optimizer_updates"]), ("DISK_LOW", 4))
        self.assertEqual(sorted(p.name for p in capture.iterdir() if p.is_dir() and p.name.startswith("checkpoint")),
                         ["checkpoint-pair-01", "checkpoint-pair-03"])
        self.assertFalse((capture / "reload_expected.json").exists())

    def test_default_has_no_disk_check(self):
        with patch.object(training, "disk_low", wraps=training.disk_low) as guard, \
                patch("shutil.disk_usage", side_effect=AssertionError("no minimum bound")):
            result = self.train(self.root / "unbounded")
        self.assertEqual(result["status"], "LOCAL_TEN_TRAINING_SAVED_PENDING_FRESH_RELOAD")
        self.assertTrue(all(call.args[1] is None for call in guard.call_args_list))

    def test_problem_range_runs_a_bound_chunk_of_a_longer_selection(self):
        self.actors.extend(SimpleNamespace(problem_id=str(i), image_paths=["image"]) for i in range(16, 40))
        capture = self.root / "chunk"
        with patch("fine_tuning.notebook_rl.local_train.read_preparation", return_value=(self.prepared, {}, self.actors)):
            result = self.train(capture, problem_range=(0, 16))
            self.assertEqual(result["optimizer_updates"], 4)
            launch = json.loads((capture / "launch.json").read_text())
            self.assertEqual(launch["execution_contract"]["problem_range"], [0, 16])
            self.assertEqual(launch["problem_ids"], [str(i) for i in range(16)])
            for window in ((16, 40), (0, 8)):
                with self.subTest(window=window), self.assertRaisesRegex(ValueError, "exactly 16|problem range"):
                    self.train(self.root / f"bad-{window[0]}", problem_range=window)


class AuditTrainTests(ScaledTrainFixture, unittest.TestCase):
    options = {"evidence_audit_rule": {"first": True, "last": True, "every": 0}}

    def test_only_rule_steps_measure_vjps_and_s_claims_come_only_from_audits(self):
        self.s_true_step = 2  # a non-audit step: its would-be S must never surface
        capture = self.root / "run"
        result = self.train(capture)
        self.assertEqual([call["measure"] for call in self.update_calls], [True, False, False, False, True])
        self.assertIs(result["S_gradient_verified"], False)
        self.assertEqual([row["S_gradient_verified"] for row in result["repair_commits"]], [False, None, None, None, False])
        self.close(capture)
        proof = self.replay(capture)
        self.assertIs(proof["S_gradient_verified"], False)
        restored = self.reload(capture, self.root / "fresh")
        self.assertIs(restored["S_gradient_verified"], False)
        # Flipping a non-audit step to "measured" violates the fixed rule.
        update = json.loads((capture / "pair-01.json").read_text())
        update["repair_aggregate_gradient"] = {"fixture": True}
        (capture / "pair-01.json").write_text(json.dumps(update))
        with self.assertRaisesRegex(ValueError, "fixed audit rule"):
            self.replay(capture)

    def test_audited_positive_step_sets_s_and_partial_runs_fold_only_audited_steps(self):
        self.s_true_step = 4
        result = self.train(self.root / "run")
        self.assertIs(result["S_gradient_verified"], True)
        self.update_calls.clear()
        self.now = 0.
        self.stop_after = 2
        partial = self.train(self.root / "partial")
        self.assertEqual(partial["status"], "BUDGET_STOP")
        self.assertIs(partial["S_gradient_verified"], False)


class SparseAuditTrainTests(ScaledTrainFixture, unittest.TestCase):
    options = {"evidence_audit_rule": {"first": False, "last": True, "every": 0}}
    stop_after = 2

    def test_run_without_an_audited_update_keeps_s_unknown(self):
        self.s_true_step = 0
        result = self.train(self.root / "run")
        self.assertEqual((result["status"], result["optimizer_updates"]), ("BUDGET_STOP", 2))
        self.assertEqual([call["measure"] for call in self.update_calls], [False, False])
        self.assertIsNone(result["S_gradient_verified"])


class Bf16TrainTests(ScaledTrainFixture, unittest.TestCase):
    options = {"policy_precision": "bf16"}

    def test_bf16_collect_update_save_reload_without_nf4_cache_guard_or_parking(self):
        capture = self.root / "run"
        result = self.train(capture)
        self.assertEqual(result["optimizer_updates"], 5)
        self.assertEqual(self.parked, [])
        self.original_access.assert_not_called()
        self.assertIsNone(result["original_access_guard"])
        launch = self.close(capture)
        self.assertIs(launch["standalone_base"], False)
        self.assertIsNone(launch["cache_check"])
        self.assertEqual(launch["optimizer_state_collection_policy"], scale.RESIDENT_POLICY)
        self.assertNotIn("quantization", launch["protocol"])
        initial = json.loads((capture / "initialization.json").read_text())
        self.assertIsNone(initial["quantization"])
        self.assertIn("original_files_sha256", initial)
        self.assertTrue((capture / "original_files.json").is_file())
        restored = self.reload(capture, self.root / "fresh")
        self.assertEqual((restored["step"], restored["policy_precision"]), (5, "bf16"))
        self.original_access.assert_not_called()

    def test_bf16_policy_trains_with_a_bfloat16_frozen_base_on_cpu(self):
        def bf16_policy(*args, **kwargs):
            model = tiny_model().to(torch.bfloat16)
            return configure_policy(model, None, "MOD-RV", config(), enforce_counts=False)

        with patch("fine_tuning.notebook_rl.scale.build_bf16_policy", side_effect=bf16_policy):
            result = self.train(self.root / "bfloat16")
        self.assertEqual(result["optimizer_updates"], 5)
        inventory = json.loads((self.root / "bfloat16" / "inventory.json").read_text())
        self.assertEqual({row["dtype"] for row in inventory["trainable"] if row["group"] == "M"}, {"torch.float32"})


class VllmTrainTests(ScaledTrainFixture, unittest.TestCase):
    options = {"policy_precision": "bf16", "rollout_engine": "vllm", "vllm": VLLM}

    def test_mocked_vllm_collector_samples_and_receives_a_push_after_every_update(self):
        capture = self.root / "run"
        result = self.train(capture)
        self.assertEqual(result["optimizer_updates"], 5)
        kv, seqs, limit, kwargs = next(e for e in self.engines if e.kind == "vllm").init
        self.assertEqual((kv, seqs, limit, kwargs), (VLLM["kv_cache_memory_bytes"], 16, 24576,
            {"enable_prefix_caching": True, "gpu_memory_utilization": .55}))
        self.assertTrue(all(engine == "vllm" for _, _, engine in self.built))
        self.assertTrue(all(call["preparation"] == "shared" for call in self.update_calls))
        expected = [("shared", "shared", 0, False), ("shared", "vllm", 0, False)]
        for step in range(1, 6):
            expected += [("shared", "shared", step, True), ("shared", "vllm", step, True)]
        self.assertEqual(self.sync_calls, expected)
        self.assertTrue(all((capture / f"sampler-sync-{i:02d}.json").is_file() for i in range(5)))
        self.close(capture)
        self.assertEqual(self.replay(capture)["optimizer_updates"], 5)
        (capture / "sampler-sync-02.json").unlink()
        with self.assertRaises((ValueError, OSError)):
            self.replay(capture)


if __name__ == "__main__":
    unittest.main()
