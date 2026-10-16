"""Authored CPU visual-hook fixtures; not native vLLM or pretrained GPU evidence."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import gate_images as gates
from fine_tuning.notebook_rl.engines import Prepared, RolloutRequest


class Qwen3_VisionTransformer(torch.nn.Module):
    """Named native-boundary fixture, with deliberately synthetic features."""
    def __init__(self):
        super().__init__()
        self.spatial_merge_size = 2
        self.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.bad = None

    def forward(self, pixels, grid_thw):
        rows = sum(t*h*w//4 for t, h, w in grid_thw)
        output = torch.zeros((rows, 32), dtype=torch.bfloat16, device=pixels.device)
        if self.bad == "rows":
            return output[:-1]
        if self.bad == "width":
            return output[:, :8]
        if self.bad == "dtype":
            return output.float()
        if self.bad == "nan":
            output[0, 0] = float("nan")
        return output


class Qwen3VLForConditionalGeneration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Qwen3_VisionTransformer()
        self.config = SimpleNamespace(vision_config=SimpleNamespace(out_hidden_size=8))
        self.use_data_parallel = False


def prepared(count=1):
    grid = [[1, 4, 4]] * count
    return Prepared([9] + [3] * (4*count) + [10],
        {"pixel_values": torch.arange(count*16*24).reshape(-1, 24).float(),
         "image_grid_thw": torch.tensor(grid)}, [object() for _ in grid], "fixture")


class ImageParityTests(unittest.TestCase):
    def test_hook_observes_real_module_call_without_changing_outputs_and_is_removed(self):
        model, records = Qwen3VLForConditionalGeneration(), []
        item = prepared(2)
        pixels = item.pixels["pixel_values"].bfloat16()
        grid = item.pixels["image_grid_thw"].tolist()
        baseline = model.visual(pixels, grid_thw=grid)
        with ExitStack() as stack:
            gates._observe_visual(model, stack, records)
            actual = model.visual(pixels, grid_thw=grid)
            self.assertTrue(torch.equal(actual, baseline))
        self.assertEqual(len(model.visual._forward_hooks), 0)
        result = gates._compare(item, records, 3)
        self.assertEqual(result["image_tokens"], 8)
        self.assertEqual(records[0]["device"], "cpu")
        self.assertTrue(result["encoder_order_matches_input"])

    def test_invalid_actual_features_fail_and_hook_cleanup_runs(self):
        for fault in ("rows", "width", "dtype", "nan"):
            model, records = Qwen3VLForConditionalGeneration(), []
            model.visual.bad = fault
            with self.subTest(fault=fault), self.assertRaisesRegex(RuntimeError, "feature tensor"):
                with ExitStack() as stack:
                    gates._observe_visual(model, stack, records)
                    model.visual(torch.ones((16, 24), dtype=torch.bfloat16), grid_thw=[[1, 4, 4]])
            self.assertFalse(model.visual._forward_hooks)
            self.assertEqual(records, [])

    def test_malformed_grid_and_patch_rows_rejected(self):
        pixels = torch.ones((16, 24), dtype=torch.bfloat16)
        for grid in ([[True, 4, 4]], [[2, 2, 4]], [[1, 3, 4]], [[1, 2, 4]], []):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                gates._image_rows(pixels, grid, 2)
        for value in (pixels.float(), pixels[:12], pixels.clone().fill_(float("nan"))):
            with self.assertRaises(ValueError):
                gates._image_rows(value, [[1, 4, 4]], 2)

    def test_image_identity_and_order_across_calls_not_just_grid_size(self):
        item = prepared(2)
        expected = gates._image_rows(item.pixels["pixel_values"].bfloat16(), [[1, 4, 4]]*2, 2)
        result = gates._compare(item, [{"images": [row]} for row in expected], 3)
        self.assertTrue(result["encoder_order_matches_input"])
        for rows in (list(reversed(expected)), [], expected[:1],
                     [expected[0], expected[0]], expected + expected[:1]):
            with self.assertRaisesRegex(RuntimeError, "coverage"):
                gates._compare(item, [{"images": rows}], 3)
        item.token_ids.pop(1)
        with self.assertRaisesRegex(RuntimeError, "literal image tokens"):
            gates._compare(item, [{"images": expected}], 3)

    def controlled_engine(self):
        model = Qwen3VLForConditionalGeneration()
        config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(multimodal_config=SimpleNamespace(mm_processor_cache_gb=0)))
        engine = SimpleNamespace(model_config=SimpleNamespace(image_token_id=3),
            llm=SimpleNamespace(apply_model=lambda fn: [fn(model)],
                                llm_engine=SimpleNamespace(vllm_config=config)))
        engine.prepare = lambda request: prepared(len(request.images))
        def probe(request, ids):
            self.assertEqual(request.max_new_tokens, 1)
            item = engine.prepare(request)
            model.visual(item.pixels["pixel_values"].bfloat16(),
                         grid_thw=item.pixels["image_grid_thw"].tolist())
            return [-1.]
        engine.probe_logprobs = probe
        requests = [RolloutRequest(str(n), None, [], [object()]*n, 1, 3072, [3407+n]) for n in (1, 2)]
        return engine, model, requests

    def test_controlled_entrypoint_records_three_images_never_gpu_approval(self):
        engine, model, requests = self.controlled_engine()
        with patch("fine_tuning.notebook_rl.gate_weights.require_idle_engine", return_value={"device": "cpu"}):
            result = gates.measure_image_parity(engine, requests)
        self.assertEqual(result["images_checked"], 3)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["gpu_gates_certified"])
        self.assertEqual([r.max_new_tokens for r in requests], [3072, 3072])
        self.assertFalse(model.visual._forward_hooks)

    def test_failed_probe_missing_forward_wrong_device_and_runtime_drift_clean_hooks(self):
        for fault in ("raises", "missing", "device", "drift"):
            engine, model, requests = self.controlled_engine()
            if fault == "raises":
                engine.probe_logprobs = lambda *a: (_ for _ in ()).throw(RuntimeError("fixture failure"))
            elif fault == "missing":
                engine.probe_logprobs = lambda *a: [-1.]
            values = [{"device": "cuda:0" if fault == "device" else "cpu"},
                      {"device": "cpu", "policy_version": 9}]
            with self.subTest(fault=fault), patch("fine_tuning.notebook_rl.gate_weights.require_idle_engine",
                    side_effect=values), self.assertRaises(RuntimeError):
                gates.measure_image_parity(engine, requests)
            self.assertFalse(model.visual._forward_hooks)

    def test_cache_and_worker_mode_rejected_without_leftover_hooks(self):
        for fault in ("cache", "parallel", "workers", "input"):
            engine, model, requests = self.controlled_engine()
            if fault == "cache":
                engine.llm.llm_engine.vllm_config.cache_config.enable_prefix_caching = True
            elif fault == "parallel":
                model.use_data_parallel = True
            elif fault == "workers":
                engine.llm.apply_model = lambda fn: [fn(model), fn(model)]
            else:
                requests = requests[:1]
            with self.subTest(fault=fault), patch("fine_tuning.notebook_rl.gate_weights.require_idle_engine",
                    return_value={"device": "cpu"}), self.assertRaises((RuntimeError, ValueError)):
                gates.measure_image_parity(engine, requests)
            self.assertFalse(model.visual._forward_hooks)


if __name__ == "__main__":
    unittest.main()
