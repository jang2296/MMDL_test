"""Focused v16 loss, source normalization, encoding, and reversible ablation checks on CPU."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from fine_tuning.error_note import train_v16 as train
from fine_tuning.error_note.audit import text_sha256
from fine_tuning.error_note.content_v16 import O_INSTRUCTION, ROW_SCHEMA
from fine_tuning.error_note.contract import ROUTE_FULL, ROUTE_LM
from fine_tuning.modules import tensor_sha256
from fine_tuning.tests.test_error_note_train import encoded, tiny_factory, tiny_processor


def content_row(index=0, kind="O", layer="graph", route=ROUTE_FULL):
    row = encoded(f"{kind}:{index}", target=(12, 13), kind=kind, route=route, seed=index)
    return row | {"schema": ROW_SCHEMA, "problem_id": f"problem-{index}", "parent_id": f"parent-{index}",
                  "layer": layer, "loss_mask": torch.tensor([True, False, True])}


def parameter_hashes(bundle):
    return {name: tensor_sha256(value) for name, value in bundle.model.named_parameters()}


def gradients(bundle):
    return {name: value.grad.clone() for name, value in bundle.model.named_parameters() if value.grad is not None}


class ContentLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_masked_raw_ce_matches_manual_value_and_gradients_without_update(self):
        bundle = tiny_factory()
        bundle.model.eval()
        bundle.route.apply(False)
        rows = [content_row()]
        before = parameter_hashes(bundle)
        report = train.masked_backward(bundle, rows, [[0]], [.4])
        actual = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        bundle.route.apply(True)
        values = train.batch_logprobs(bundle, rows)[0]
        manual = -.4 * values[rows[0]["loss_mask"]].mean()
        manual.backward()
        self.assertAlmostEqual(report["weighted_loss"], float(manual.detach()), places=6)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(actual[name], value, atol=1e-7, rtol=1e-4)
        self.assertEqual(parameter_hashes(bundle), before)
        self.assertTrue(report["task_signal"] and report["has_gradient"])
        self.assertTrue(all(value is not None and value > 0 for value in report["gradient_norms"].values()))
        self.assertFalse(bundle.model.training)

    def test_masked_logit_gradient_is_exactly_zero(self):
        bundle = tiny_factory()
        row = content_row()
        logits = torch.randn(3, 19, requires_grad=True)
        values = logits.log_softmax(-1)[:, 3]
        with patch.object(train, "batch_logprobs", return_value=[values]):
            train.masked_backward(bundle, [row], [[0]], [1.0])
        torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]), atol=0, rtol=0)
        self.assertGreater(float(logits.grad[[0, 2]].abs().sum()), 0)

    def test_bad_masks_and_plans_fail_before_partial_gradient(self):
        bundle = tiny_factory()
        row = content_row()
        for mask in (torch.zeros(3, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.ones(3)):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "mask/target"):
                train.masked_backward(bundle, [row | {"loss_mask": mask}], [[0]], [1.0])
        for plan in ([], [[0], [0]], [[1]]):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                train.masked_backward(bundle, [row], plan, [1.0])
        self.assertTrue(all(value.grad is None for value in bundle.model.parameters()))

    def test_nonfinite_loss_stops_and_restores_route_and_mode(self):
        bundle = tiny_factory()
        bundle.model.eval()
        bundle.route.apply(False)
        flags = {name: value.requires_grad for name, value in bundle.model.named_parameters()}
        with patch.object(train, "batch_logprobs", return_value=[torch.tensor([float("nan"), 0., 0.])]):
            with self.assertRaises(train.NonFiniteStop):
                train.masked_backward(bundle, [content_row()], [[0]], [1.0])
        self.assertFalse(bundle.route.vision_on or bundle.model.training)
        self.assertEqual(flags, {name: value.requires_grad for name, value in bundle.model.named_parameters()})

    def test_nonfinite_actual_gradient_stops_before_step_admission(self):
        bundle = tiny_factory()
        handle = bundle.params["M"][0].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            with self.assertRaisesRegex(train.NonFiniteStop, "gradient"):
                train.masked_backward(bundle, [content_row()], [[0]], [1.0])
        finally:
            handle.remove()

    def test_closed_block_coefficients_and_gradients_ignore_microbatch_partition(self):
        bundle = tiny_factory()
        rows = [content_row(index) for index in range(3)]
        rows[2]["layer"] = "scene"
        contract = {"source_weights": {"graph": .25, "scene": .25, "chart": .5},
                    "weights": {"O": 1., "J": .5, "C": 0., "N": 0.}, "pg_weight": 0}
        coefficients = train.block_coefficients(rows, contract["source_weights"], contract["weights"])
        self.assertEqual(coefficients, [.125, .125, .25])
        first = train.backward_content_block(bundle, rows, contract)
        actual = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        second = train.backward_content_block(bundle, rows, contract, token_budget=1000)
        self.assertAlmostEqual(first["loss"], second["loss"], places=6)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(actual[name], value, atol=2e-6, rtol=2e-4)
        with self.assertRaisesRegex(ValueError, "one row per parent and kind"):
            train.block_coefficients(rows + [rows[0]], contract["source_weights"], contract["weights"])
        bundle.model.zero_grad(set_to_none=True)
        zero = train.backward_content_block(bundle, rows, contract, components=("J",))
        self.assertFalse(zero["task_signal"] or zero["has_gradient"])

    def test_source_kind_cells_keep_parent_mass_when_another_kind_is_added(self):
        rows = [content_row(0), content_row(1), content_row(0, kind="J", route=ROUTE_LM)]
        values = train.block_coefficients(rows, {"graph": .5}, {"O": .6, "J": .4})
        self.assertEqual(values, [.15, .15, .2])

    def test_empty_admission_requires_explicit_exclusions_and_never_has_task_gradient(self):
        bundle = tiny_factory()
        contract = {"source_weights": {"graph": 1.}, "weights": {"O": 1., "A1": 0.},
                    "expected_problem_ids": ["missing-parent"], "excluded_problem_ids": {"missing-parent": "NO_TASK_SIGNAL"}}
        before = parameter_hashes(bundle)
        result = train.backward_content_block(bundle, [], contract, components=())
        self.assertEqual((result["loss"], result["microbatches"]), (0., []))
        self.assertFalse(result["task_signal"] or result["has_gradient"])
        self.assertTrue(all(value is None for value in result["gradient_norms"].values()))
        self.assertEqual(parameter_hashes(bundle), before)
        self.assertTrue(all(value.grad is None for value in bundle.model.parameters()))
        with self.assertRaisesRegex(ValueError, "closed block"):
            train.backward_content_block(bundle, [], contract | {"excluded_problem_ids": {}})
        with self.assertRaises(ValueError):
            train.backward_content_block(bundle, [], contract | {"excluded_problem_ids": {"missing-parent": ""}})
        with self.assertRaises(ValueError):
            train.backward_content_block(bundle, [], contract | {"weights": {"O": float("nan")}})
        empty = train.backward_content_block(bundle, [], contract | {"expected_problem_ids": [], "excluded_problem_ids": {}})
        self.assertFalse(empty["task_signal"] or empty["has_gradient"])

    def test_self_rows_use_stored_support_and_zero_singletons_keep_no_gradient(self):
        from fine_tuning.error_note import rl
        from fine_tuning.notebook_rl.input_preparation import digest

        bundle = tiny_factory()
        row = content_row(kind="C", route=ROUTE_LM)
        actions = row["input_ids"][-row["target_len"]:].tolist()
        sampling = {"temperature": .7, "top_p": .8, "top_k": 20, "presence_penalty": 1.5}
        supports = [[action, 18] for action in actions]
        policy = rl.policy_sha256(bundle)
        row["generation"] = {"completion_token_ids": actions, "support_token_ids": supports,
                             "support_sha256": [digest(value) for value in supports],
                             "sampling_sha256": rl.sampling_sha256(sampling), "policy_sha256": policy}
        contract = {"source_weights": {"graph": 1.}, "weights": {"O": 0., "J": 0., "C": 1., "N": 0.},
                    "sampling": sampling, "policy_sha256": policy}
        from fine_tuning.error_note.train_v4 import single_batch_statistics

        bundle.route.apply(False)
        value = single_batch_statistics(bundle, [row], sampling)[0]["logprobs"]
        manual = -value[row["loss_mask"]].mean()
        manual.backward()
        expected = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        report = train.backward_content_block(bundle, [row], contract)
        self.assertAlmostEqual(report["loss"], float(manual.detach()), places=6)
        for name, actual in gradients(bundle).items():
            torch.testing.assert_close(actual, expected[name], atol=1e-7, rtol=1e-4)
        bundle.model.zero_grad(set_to_none=True)
        row["generation"]["support_token_ids"] = [[action] for action in actions]
        row["generation"]["support_sha256"] = [digest([action]) for action in actions]
        zero = train.backward_content_block(bundle, [row], contract)
        self.assertEqual(zero["loss"], 0.)
        self.assertTrue(zero["task_signal"])
        self.assertFalse(zero["has_gradient"])
        bundle.model.zero_grad(set_to_none=True)
        row["generation"]["support_token_ids"][0] = [-1, actions[0]]
        row["generation"]["support_sha256"][0] = digest([-1, actions[0]])
        with self.assertRaisesRegex(ValueError, "binding changed"):
            train.backward_content_block(bundle, [row], contract)
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_merger_ablation_restores_hooks_flags_and_weights_even_on_exception(self):
        bundle = tiny_factory()
        roots = bundle.inventory["mergers"]
        wrapper = bundle.model.get_base_model().get_submodule(roots[0])
        wrapper.enable_adapters(False)
        next(wrapper.original_module.parameters()).requires_grad_(False)
        calls = []
        handle = wrapper.register_forward_hook(lambda *_: calls.append(True))
        before = parameter_hashes(bundle)
        flags = {name: value.requires_grad for name, value in bundle.model.named_parameters()}
        hook_ids = set(wrapper._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with train.mergers_original(bundle, roots):
                self.assertTrue(all(bundle.model.get_base_model().get_submodule(root).disable_adapters for root in roots))
                raise RuntimeError("probe failed")
        self.assertTrue(wrapper.disable_adapters)
        self.assertEqual(flags, {name: value.requires_grad for name, value in bundle.model.named_parameters()})
        self.assertEqual(parameter_hashes(bundle), before)
        self.assertEqual(set(wrapper._forward_hooks), hook_ids)
        handle.remove()

    def test_visual_contribution_four_columns_restore_every_state(self):
        bundle = tiny_factory()
        with torch.no_grad():
            for group in ("L", "V"):
                for parameter in bundle.params[group]:
                    parameter.add_(.01 * torch.randn_like(parameter))
            for parameter in bundle.params["M"]:
                parameter.add_(.02 * torch.randn_like(parameter))
        bundle.route.apply(False)
        bundle.model.train()
        before = parameter_hashes(bundle)
        flags = {name: value.requires_grad for name, value in bundle.model.named_parameters()}
        values = train.visual_contribution(bundle, [content_row()], bundle.inventory["mergers"])
        self.assertEqual(set(values), {"on", "vision_off", "language_off", "merger_f0"})
        self.assertTrue(all(len(value) == 1 and torch.isfinite(torch.tensor(value)).all() for value in values.values()))
        self.assertTrue(any(values[key] != values["on"] for key in values if key != "on"))
        self.assertFalse(bundle.route.vision_on)
        self.assertTrue(bundle.model.training)
        self.assertEqual(parameter_hashes(bundle), before)
        self.assertEqual(flags, {name: value.requires_grad for name, value in bundle.model.named_parameters()})
        self.assertTrue(all(value.grad is None for value in bundle.model.parameters()))


class ContentEncodingTests(unittest.TestCase):
    def row(self, kind="O"):
        prompt, target = O_INSTRUCTION["graph"], "a b"
        return {"schema": ROW_SCHEMA, "kind": kind, "route": ROUTE_FULL, "row_id": "O:1", "layer": "graph",
                "target_origin": "program" if kind == "O" else "human", "prompt": prompt, "target": target,
                "verification_receipt": {"training_prompt_sha256": text_sha256(prompt), "target_sha256": text_sha256(target)}}

    def test_o_eos_supervised_j_eos_excluded_and_token_limits_never_truncate(self):
        processor = tiny_processor()
        config = SimpleNamespace(image_token_id=3, video_token_id=6, vision_config=SimpleNamespace(spatial_merge_size=2))
        image = [Image.new("RGB", (8, 8))]
        for kind in ("O", "J"):
            encoded_row = train.encode_content_row(processor, config, self.row(kind), image, image={})
            self.assertEqual(encoded_row["target_len"], 3)
            self.assertEqual(encoded_row["loss_mask"].tolist(), [True, True, kind == "O"])
            self.assertTrue(encoded_row["input_sha256"])
        with self.assertRaisesRegex(ValueError, "TARGET_TOO_LONG"):
            train.encode_content_row(processor, config, self.row(), image, image={}, max_target_tokens=2)
        with self.assertRaisesRegex(ValueError, "CONTEXT_BLOCKED"):
            train.encode_content_row(processor, config, self.row(), image, image={}, max_context_tokens=2)

    def test_real_o_j_encoding_same_parent_stays_separate_in_capacity_plan(self):
        from fine_tuning.error_note.capacity import shape

        class Capacity:
            def __init__(self):
                self.checked = []

            def check(self, rows, phase):
                shape(rows)
                self.checked.append((len(rows), phase))

            def partition(self, rows, plan, phase):
                for batch in plan:
                    self.check([rows[index] for index in batch], phase)
                return plan

        bundle = tiny_factory()
        processor = tiny_processor()
        image = [Image.new("RGB", (8, 8))]
        config = bundle.model.get_base_model().config
        rows = []
        for kind in ("O", "J"):
            row = self.row(kind) | {"problem_id": "same-problem", "parent_id": "same-parent", "row_id": kind}
            rows.append(train.encode_content_row(processor, config, row, image, image={}))
        capacity = Capacity()
        contract = {"source_weights": {"graph": 1.}, "weights": {"O": .5, "J": .5}, "pg_weight": 0}
        result = train.backward_content_block(bundle, rows, contract, token_budget=4096, capacity=capacity)
        self.assertEqual(result["microbatches"], [[0], [1]])
        self.assertEqual(set(result["loss_by_kind"]), {"O", "J"})
        self.assertTrue(all(count == 1 for count, _ in capacity.checked))

    def test_a1_replays_original_p0_native_actions_with_stored_support_loss(self):
        import random

        from fine_tuning.error_note import rl
        from fine_tuning.error_note.train_v4 import single_batch_statistics
        from fine_tuning.notebook_rl.input_preparation import digest, prepare_protocol_inputs
        from fine_tuning.notebook_rl.prompts import user_message
        from fine_tuning.error_note.train import END_TOKEN

        bundle, processor = tiny_factory(), tiny_processor()
        config, images = bundle.model.get_base_model().config, [Image.new("RGB", (8, 8))]
        prompt = "Question: question"
        prepared, _ = prepare_protocol_inputs(processor, config, [user_message(prompt, 1)], images, {})
        rng = random.Random(16)
        body = [rng.randrange(8, 30) for _ in range(256)]
        end_id = processor.tokenizer.convert_tokens_to_ids(END_TOKEN)
        actions = [*body, end_id]
        tokens = [chr(97 + token - 8) for token in body]
        target = "".join(tokens)
        sampling = {"temperature": .7, "top_p": .8, "top_k": 20, "presence_penalty": 1.5}
        support = [[action, 31] for action in actions]
        policy = rl.policy_sha256(bundle)
        # Real stream generation dicts carry no completion_tokens; only the outer output does.
        generation = {"text": target, "completion_token_ids": actions, "completion_count": 1,
                      "finish_reason": "stop", "prompt_token_ids": prepared["prompt_token_ids"],
                      "input_sha256": prepared["input_sha256"], "support_token_ids": support,
                      "support_sha256": [digest(value) for value in support], "sampling_sha256": rl.sampling_sha256(sampling),
                      "policy_sha256": policy}
        row = {"schema": ROW_SCHEMA, "kind": "A1", "route": ROUTE_FULL, "row_id": "a1", "layer": "graph",
               "problem_id": "first", "parent_id": "first", "prompt": prompt, "target": target,
               "target_source": "T1", "status": "CORRECT", "generation": generation,
               "verified_byte_spans": [[0, len(target)]], "loss_mask": [True] * 256 + [False]}
        original = processor.tokenizer.convert_ids_to_tokens
        def native_tokens(ids, *args, **kwargs):
            return [*tokens, END_TOKEN] if ids == actions else original(ids, *args, **kwargs)
        with patch.object(processor.tokenizer, "convert_ids_to_tokens", side_effect=native_tokens):
            actual = train.encode_content_row(processor, config, row, images, image={})
            self.assertEqual(actual["input_ids"][-len(actions):].tolist(), actions)
            self.assertEqual(actual["input_ids"][:-len(actions)].tolist(), prepared["prompt_token_ids"])
            self.assertNotEqual(processor.tokenizer.encode(target, add_special_tokens=False), actions)
            with self.assertRaisesRegex(ValueError, "original P0"):
                train.encode_content_row(processor, config, row | {"prompt": "Question: a"}, images, image={})
            with self.assertRaisesRegex(ValueError, "admitted correct"):
                train.encode_content_row(processor, config, row | {"status": "INCORRECT"}, images, image={})
        bundle.route.apply(True)
        value = single_batch_statistics(bundle, [actual], sampling)[0]["logprobs"]
        manual = -.05 * value[actual["loss_mask"]].mean()
        manual.backward()
        expected = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        contract = {"source_weights": {"graph": 1.}, "weights": {"O": 0., "A1": .05}, "sampling": sampling,
                    "policy_sha256": policy, "expected_problem_ids": ["first"]}
        result = train.backward_content_block(bundle, [actual], contract)
        self.assertAlmostEqual(result["loss"], float(manual.detach()), places=7)
        self.assertEqual(set(result["loss_by_kind"]), {"A1"})
        for name, gradient in gradients(bundle).items():
            torch.testing.assert_close(gradient, expected[name], atol=1e-7, rtol=1e-4)

    def test_observations_reject_p0_no_images_control_tokens_and_sampled_targets(self):
        processor = tiny_processor()
        for changes, images in (({"prompt": "Question: a"}, [object()]), ({"route": ROUTE_LM}, [object()]),
                                ({"generation": {}}, [object()]), ({"target": "<|im_end|>"}, [object()]), ({}, [])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train.encode_content_row(processor, SimpleNamespace(), self.row() | changes, images, image={})

    def test_byte_spans_select_only_whole_utf8_tokens_without_retokenizing(self):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        encoder = bytes_to_unicode()
        parts = ["A ".encode(), "한".encode(), b" B"]
        tokenizer = SimpleNamespace(all_special_tokens=["<end>"],
                                    convert_ids_to_tokens=lambda _: ["".join(encoder[b] for b in part) for part in parts] + ["<end>"])
        text = "A 한 B"
        mask = train.token_byte_mask(tokenizer, [1, 2, 3, 9], text, [[3, 5], [5, 7]], end_token_id=9)
        self.assertEqual(mask, [False, False, True, False])
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            train.token_byte_mask(tokenizer, [1, 2, 3, 9], text + "!", [[0, 7]], end_token_id=9)

    def test_documented_module_entry_delegates_lazily(self):
        import sys

        calls = []
        runner = SimpleNamespace(main=lambda argv: calls.append(argv) or 7)
        with patch.dict(sys.modules, {"fine_tuning.error_note.run_v16": runner}):
            self.assertEqual(train.main(["--help"]), 7)
        self.assertEqual(calls, [["--help"]])


class BF16Tests(unittest.TestCase):
    def test_bf16_fraction_cosine_and_no_base_cast_noise(self):
        origin = {"w": torch.ones(4)}
        self.assertEqual(train.bf16_survival(origin, origin)["changed_fraction"], 0.)
        large = train.bf16_survival({"w": origin["w"] + 1 / 128}, origin)
        self.assertEqual(large["changed_fraction"], 1.)
        self.assertAlmostEqual(large["cosine"], 1.)
        self.assertEqual(train.bf16_survival({"w": origin["w"] + 1 / 1280}, origin)["changed_fraction"], 0.)
        fractional = {"w": torch.tensor([1.001, .123456])}
        self.assertEqual(train.bf16_survival(fractional, fractional)["changed_fraction"], 0.)
        self.assertEqual(train.bf16_survival(fractional, fractional)["visible_norm"], 0.)

    def test_empty_mismatched_or_nonfinite_diagnostics_rejected(self):
        for masters, base in (({}, {}), ({"w": torch.ones(1)}, {"q": torch.ones(1)}),
                              ({"w": torch.ones(2)}, {"w": torch.ones(1)}),
                              ({"w": torch.tensor([float("nan")])}, {"w": torch.ones(1)}),
                              ({"w": torch.tensor([torch.finfo(torch.float32).max])}, {"w": torch.ones(1)})):
            with self.subTest(masters=masters), self.assertRaises(ValueError):
                train.bf16_survival(masters, base)


if __name__ == "__main__":
    unittest.main()
