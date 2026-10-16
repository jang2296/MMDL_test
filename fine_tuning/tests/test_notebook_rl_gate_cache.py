"""Authored cache-control boundaries, never native vLLM/GPU evidence."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil
import torch

from fine_tuning.notebook_rl import gate_cache as gate
from fine_tuning.notebook_rl.engines import InputPreparation
from fine_tuning.notebook_rl.trainer import SyncProbe
from fine_tuning.tests.test_notebook_rl_engines import Processor, request, pushed_engine, weights
from fine_tuning.tests.test_notebook_rl_gate_images import Qwen3VLForConditionalGeneration


def stages():
    roots = ["visual.merger.", *[f"visual.deepstack_merger_list.{i}." for i in range(3)]]
    parameters = {root + str(i): {"shape": [2], "dtype": "torch.bfloat16", "sha256": "a" * 64}
                  for root in roots for i in range(6)}
    before = {"stage": "BEFORE", "state": {"parameters": parameters}, "probes": []}
    for count in (1, 2):
        before["probes"].append({"input": {"request_id": f"probe-{count}",
            "prompt_token_ids": [3] * count + [9], "completion_ids": [10, 11], "seed": 31,
            "image": {"image_key": f"image-{count}", "image_spans": [[i, i+1] for i in range(count)],
                      "tensors": {"authored": True}}}, "logprobs": [-2., -3.]})
    after = deepcopy(before)
    after["stage"] = "AFTER"
    after["state"]["parameters"]["visual.merger.0"]["sha256"] = "b" * 64
    fresh = deepcopy(after)
    fresh["stage"] = "FRESH"
    control = {"stage": "DEFAULT_CONTROL", "state_before": deepcopy(before["state"]),
               "state_after": deepcopy(after["state"]), "probes": []}
    for count, row in enumerate(fresh["probes"], 1):
        row["control_action"] = {"token_ids": [12], "logprobs": [-1.]}
        control["probes"].append({"input": deepcopy(row["input"]),
            "fixed_after": {"token_ids": [10, 11], "logprobs": [-2.3, -3.4], "visual_call_count": 0,
                "encoder_checks": [{"identifier": f"G6:image-{count}:{i}", "hit": True,
                                    "worker_entry_present": True} for i in range(count)]},
            "prefix_after": {"token_ids": [12], "logprobs": [-1.5], "num_cached_tokens": 16}})
    return before, after, control, fresh


def preparation():
    return InputPreparation(Processor(), SimpleNamespace(image_token_id=3, vision_start_token_id=4,
        vision_end_token_id=5), {"preprocessing": "hf_processor"}, 128)


class CacheGateTests(unittest.TestCase):
    def test_measured_difference_is_not_native_or_training_approval(self):
        result = gate._compare(*stages(), .02)
        self.assertTrue(result["fresh_engine_match"])
        self.assertTrue(result["default_cache_comparison_measured"])
        self.assertTrue(result["default_mismatch_observed"])
        self.assertEqual([r["encoder_hit_count"] for r in result["probes"]], [1, 2])
        for key in ("native_execution", "training_allowed", "gpu_gates_certified"):
            self.assertFalse(result[key])

    def test_real_reuse_without_numeric_difference_does_not_manufacture_a_mismatch(self):
        b, a, c, f = stages()
        for control, fresh in zip(c["probes"], f["probes"], strict=True):
            control["fixed_after"]["logprobs"] = fresh["logprobs"][:]
            control["prefix_after"]["logprobs"] = fresh["control_action"]["logprobs"][:]
        result = gate._compare(b, a, c, f, .02)
        self.assertTrue(result["default_cache_comparison_measured"])
        self.assertFalse(result["default_mismatch_observed"])

    def test_missing_effective_bf16_update_or_weight_identity_fails(self):
        for fault in ("unchanged", "fresh", "dtype", "shape"):
            b, a, c, f = stages()
            if fault == "unchanged":
                b["state"] = deepcopy(a["state"])
                c["state_before"] = deepcopy(a["state"])
            elif fault == "fresh":
                f["state"]["parameters"]["visual.merger.1"]["sha256"] = "c" * 64
            else:
                for row in (a["state"], c["state_after"], f["state"]):
                    row["parameters"]["visual.merger.0"][fault] = "torch.float32" if fault == "dtype" else [3]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                gate._compare(b, a, c, f, .02)

    def test_missing_actual_cache_reuse_fails(self):
        for fault in ("prefix", "hit", "worker", "key", "recompute"):
            b, a, c, f = stages()
            row = c["probes"][1]
            if fault == "prefix":
                row["prefix_after"]["num_cached_tokens"] = 0
            elif fault == "recompute":
                row["fixed_after"]["visual_call_count"] = 1
            else:
                key = {"hit": "hit", "worker": "worker_entry_present", "key": "identifier"}[fault]
                row["fixed_after"]["encoder_checks"][0][key] = "wrong" if fault == "key" else False
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "reuse"):
                gate._compare(b, a, c, f, .02)

    def test_one_large_token_error_cannot_hide_in_a_small_mean(self):
        b, a, c, f = stages()
        for stage in (b, a, c, f):
            for row in stage["probes"]:
                row["input"]["completion_ids"] = [10] * 32
                if stage is c:
                    row["fixed_after"]["token_ids"] = [10] * 32
                    row["fixed_after"]["logprobs"] = [-2.] * 32
                else:
                    row["logprobs"] = [-2.] * 32
        a["probes"][0]["logprobs"][0] -= .5
        self.assertLess(.5/32, .02)
        with self.assertRaisesRegex(ValueError, "differs from fresh"):
            gate._compare(b, a, c, f, .02)

    def test_inputs_actions_probabilities_and_coverage_are_bound(self):
        for fault in ("input", "action", "normal", "nan", "count", "coverage", "identity"):
            b, a, c, f = stages()
            if fault == "input":
                f["probes"][0]["input"]["seed"] += 1
            elif fault == "action":
                c["probes"][0]["prefix_after"]["token_ids"] = [13]
            elif fault in ("normal", "nan", "count"):
                a["probes"][0]["logprobs"] = {"normal": [-9., -3.], "nan": [float("nan"), -3.], "count": [-2.]}[fault]
            else:
                for stage in (b, a, c, f):
                    item = stage["probes"][1]["input"]
                    if fault == "coverage":
                        item["image"]["image_spans"] = [[0, 1]]
                    else:
                        item["request_id"] = "probe-1"
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                gate._compare(b, a, c, f, .02)
        for threshold in (True, 0, .021, float("nan")):
            with self.assertRaises(ValueError):
                gate._compare(*stages(), threshold)

    def test_uncleared_control_load_preserves_cache_but_validates_all_weights(self):
        engine, scheduler, worker, parameters = pushed_engine()
        gate._uncleared_load(engine.llm, weights(), engine._weight_shapes)
        self.assertTrue(all(bool((p == 1).all()) for p in parameters.values()))
        self.assertIn("old-v0", worker.model_runner.encoder_cache)
        self.assertIn("old-v0", scheduler.encoder_cache_manager.cached)
        self.assertEqual(engine.policy_version, 0)
        self.assertIsNone(engine._last_loaded_version)
        for fault in ("missing", "duplicate", "dtype", "nonfinite"):
            engine, _, _, _ = pushed_engine()
            values = weights()
            if fault == "missing":
                values.pop()
            elif fault == "duplicate":
                values.append(values[0])
            else:
                values[0] = (values[0][0], values[0][1].float() if fault == "dtype" else values[0][1].fill_(float("nan")))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                gate._uncleared_load(engine.llm, values, engine._weight_shapes)

    def test_actual_hit_hook_and_visual_hook_restore_even_on_failure(self):
        class Manager:
            def check_and_update_cache(self, request, input_id):
                return request.mm_features[input_id].identifier == "cached"
        manager, model = Manager(), Qwen3VLForConditionalGeneration()
        runner = SimpleNamespace(encoder_cache={"cached": object()})
        hits, visual = [], []
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with ExitStack() as stack:
                gate._cache_trace(manager, runner, model, stack, hits, visual)
                req = SimpleNamespace(mm_features=[SimpleNamespace(identifier=x) for x in ("cached", "new")])
                self.assertTrue(manager.check_and_update_cache(req, 0))
                self.assertFalse(manager.check_and_update_cache(req, 1))
                model.visual(torch.ones((16, 24), dtype=torch.bfloat16), grid_thw=[[1, 4, 4]])
                raise RuntimeError("intentional")
        self.assertNotIn("check_and_update_cache", vars(manager))
        self.assertFalse(model.visual._forward_hooks)
        self.assertEqual([r["hit"] for r in hits], [True, False])
        self.assertEqual([r["worker_entry_present"] for r in hits], [True, False])
        self.assertEqual(len(visual), 1)

    def test_predecessor_must_actually_exit_before_next_stage(self):
        with self.assertRaises(ValueError):
            gate._closed_previous({"process": gate._process()})
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
        identity = {"pid": child.pid, "parent_pid": os.getpid(), "create_time": psutil.Process(child.pid).create_time(),
                    "boot_id": gate._process()["boot_id"]}
        try:
            with self.assertRaisesRegex(ValueError, "still live"):
                gate._closed_previous({"process": identity})
        finally:
            child.communicate(timeout=15)
        self.assertEqual(gate._closed_previous({"process": identity}), identity)

    def test_fresh_allocation_waits_for_closed_predecessor_and_always_shuts_down(self):
        probes = [SyncProbe(request(n, key=f"p{n}", n=1), [10]) for n in (1, 2)]
        control = {"stage": "DEFAULT_CONTROL", "pid": -1, "probes": [{}, {}], "process": gate._process()}
        kwargs = dict(kv_cache_memory_bytes=1, max_num_seqs=1, max_model_len=128,
                      probes=probes, control=control, record=lambda row: None)
        with patch.object(gate, "VLLMRolloutEngine") as constructor, self.assertRaises(ValueError):
            gate.measure_fresh(Path("/authored/not-loaded"), {}, **kwargs)
        constructor.assert_not_called()
        shutdown = []
        engine = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: shutdown.append(True)))))
        with patch.object(gate, "_closed_previous", return_value={}), \
                patch.object(gate, "_dedicated_process"), \
                patch.object(gate, "VLLMRolloutEngine", return_value=engine), \
                patch.object(gate, "require_idle_engine", side_effect=ValueError("authored native failure")), \
                self.assertRaisesRegex(ValueError, "authored native failure"):
            gate.measure_fresh(Path("/authored/not-loaded"), {}, **kwargs)
        self.assertEqual(shutdown, [True])

    def test_parent_gpu_blocks_disposable_allocation(self):
        parent = psutil.Process().ppid()
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
                patch.object(torch.cuda, "memory_allocated", return_value=0), \
                patch.object(gate.subprocess, "check_output", return_value=f"{parent}\n"), \
                self.assertRaisesRegex(ValueError, "ancestors"):
            gate._dedicated_process()

    def test_control_uses_real_prefix_mode_and_preserves_raw_failure(self):
        prep = preparation()
        prep._call = 0
        probe = SyncProbe(request(2, key="two", n=1), [10, 11])
        item = prep.prepare(probe.request)
        calls, journal = [], []
        bad = False
        def generate(prompts, params, **kwargs):
            calls.append((prompts, params))
            ids = list(item.token_ids) + ([] if params.prompt_logprobs is None else probe.completion_ids)
            def lp(value):
                return SimpleNamespace(logprob=value)
            output = SimpleNamespace(prompt_token_ids=ids, finished=not bad, num_cached_tokens=16,
                outputs=[SimpleNamespace(token_ids=[12], finish_reason="length", logprobs=[{12: lp(-1.)}])],
                prompt_logprobs=None if params.prompt_logprobs is None else
                    [None]*len(item.token_ids) + [{10: lp(-2.)}, {11: lp(-3.)}])
            return [output]
        llm = SimpleNamespace(generate=generate)
        with patch.object(gate, "import_module", return_value=SimpleNamespace(SamplingParams=SimpleNamespace)):
            prefix = gate._control_call(llm, prep, probe, prefix=True, hits=[], visual=[], record=journal.append)
            fixed = gate._control_call(llm, prep, probe, prefix=False, hits=[], visual=[], record=journal.append)
            bad = True
            with self.assertRaisesRegex(ValueError, "coverage"):
                gate._control_call(llm, prep, probe, prefix=True, hits=[], visual=[], record=journal.append)
        self.assertEqual(prefix["token_ids"], [12])
        self.assertEqual(fixed["logprobs"], [-2., -3.])
        self.assertIsNone(calls[0][1].prompt_logprobs)
        self.assertEqual(calls[1][1].prompt_logprobs, 0)
        self.assertEqual(calls[0][0][0]["multi_modal_uuids"], calls[1][0][0]["multi_modal_uuids"])
        self.assertEqual(len(journal), 3)
        self.assertFalse(journal[-1]["finished"])

    def test_probe_contract_and_fake_completed_handles_are_rejected(self):
        probes = [SyncProbe(request(n, key=f"p{n}", n=1), [10]) for n in (1, 2)]
        gate._probes(probes)
        with self.assertRaises(ValueError):
            gate._probes([probes[0], replace(probes[1], request=probes[0].request)])
        with self.assertRaisesRegex(ValueError, "separate native processes"):
            gate.compare_cache(*stages(), threshold=.02, current_process=SimpleNamespace(pid=os.getpid(), poll=lambda: 0),
                               control_process=None, fresh_process=None)


if __name__ == "__main__":
    unittest.main()
