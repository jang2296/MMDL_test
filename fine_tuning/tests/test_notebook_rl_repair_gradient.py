"""Authored native-repair CPU VJPs; no model quality or optimizer admission claim."""
import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl import updates
from fine_tuning.notebook_rl.engines import RolloutRequest
from fine_tuning.notebook_rl.policy import VisionRoute
from fine_tuning.notebook_rl.repair_credit import ComponentCredit
from fine_tuning.notebook_rl.rollout import _hash
from fine_tuning.notebook_rl.schema import Sample, to_json


class RepairGradientTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
                                             for key in ("L", "M", "V")})
        self.params = {key: [self.model[key]] for key in ("L", "M", "V")}
        self.bundle = SimpleNamespace(model=self.model, params=self.params, route=VisionRoute(self.params["V"]))
        self.bundle.route.apply(False)
        self.config = {"clip": {"eps_low": .2, "eps_high": .28}, "kl": {"beta": .01}}
        self.serial = 0

    def fixture(self, *, component="final", rows=None, current=None, keep=True, vision_on=False):
        self.serial += 1
        size, cap = (2, 1024) if component == "online" else (4, 8192)
        sample_id = f"sample{self.serial}"
        sample = Sample(sample_id, "problem", f"group{self.serial}", "T2" if component == "online" else "DIAG",
            2 if component == "online" else 1, "parent" if component == "online" else None,
            [1], [1, 2, 3, 4], [-.1] * 4, "stop", "abcd", 1, 0)
        current = self.model["L"].unsqueeze(0) if current is None else current
        mask = torch.ones_like(current, dtype=torch.bool)
        old = torch.zeros_like(current)
        request = RolloutRequest("request", [1], None, [], 1, cap, [1])
        action = updates._PreparedAction(sample, request, mask, 17., .5, size * cap,
            vision_on, {"partial_token_advantages": [100.] * 4}, old=old, keep=keep)
        rows = copy.deepcopy([{}] if rows is None else rows)
        coefficients = [0.] * 4
        for index, row in enumerate(rows):
            default = {"origin_ref": f"origin{index}", "kind": "TRACE", "semantic_score": 1,
                "applied_score": 1, "mean": 0., "relation": "CORRECTED", "verdict": "VERIFIED",
                "mapping_status": "MAPPED", "reason": "VERIFIED_REPAIR", "raw_spans": [[index, index + 1]],
                "token_indices": [index], "excluded_crossing_token_indices": []}
            default.update(row)
            row.clear()
            row.update(default)
            row["advantage"] = row["applied_score"] - row["mean"]
            row["coefficient"] = cap * .0625 * row["advantage"] / (len(rows) * len(row["token_indices"])) if row["token_indices"] else 0.
            row["corrected_channel"] = row["kind"] in ("OBSERVATION", "TRACE") and row["relation"] == "CORRECTED"
            row["positive_verified_corrected"] = bool(row["corrected_channel"] and row["verdict"] == "VERIFIED"
                and row["applied_score"] == 1 and row["advantage"] > 0)
            for token in row["token_indices"]:
                coefficients[token] += row["coefficient"]
        members = [sample_id, *[f"other{index}" for index in range(size - 1)]]
        binding = {"version": "repair-group-credit-v1", "group_id": sample.group_id,
            "problem_id": sample.problem_id, "group_kind": sample.kind, "group_members": members,
            "group_members_sha256": _hash(members), "sample_index": 0,
            "group_sha256": "a" * 64, "sample_sha256": _hash(json.loads(to_json(sample))),
            "inventory_sha256": "b" * 64, "audit_sha256": "c" * 64,
            "token_text_correspondence_verified": True, "semantic_truth_reverified": False,
            "engine_attachment_verified": False, "optimizer_admission": False}
        credit = ComponentCredit(sample_id, component, tuple(coefficients), size, len(rows), cap,
                                 _hash(rows), tuple(rows), binding)
        return action, credit, current

    def measure(self, accumulator, fixture, *, reference_shift=.5):
        action, credit, current = fixture
        return updates._repair_gradient(self.bundle, action, current, action.old,
            action.old + reference_shift, action.mask, self.config, credit, accumulator)

    def test_same_forward_excludes_answer_kl_and_preserves_none_and_existing_grad_objects(self):
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        fixture = self.fixture()
        self.model["L"].grad = torch.arange(4, dtype=torch.float64)
        self.model["V"].grad = torch.ones(4, dtype=torch.float64)
        saved = {key: (value.grad, None if value.grad is None else value.grad.clone())
                 for key, value in self.model.items()}
        with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad, patch.object(
                updates, "group_loss", wraps=updates.group_loss) as loss:
            member = self.measure(accumulator, fixture)
        self.assertEqual(grad.call_count, 2)
        for call in grad.call_args_list:
            self.assertTrue(call.kwargs["retain_graph"])
            self.assertTrue(call.kwargs["allow_unused"])
            self.assertNotIn(id(self.model["V"]), [id(p) for p in call.args[1]])
        for call in loss.call_args_list:
            self.assertEqual(call.kwargs["beta"], 0)
            self.assertEqual(call.args[3].tolist(), [0.])
            self.assertEqual(call.kwargs["norm_const"], 4 * 8192)
        self.assertEqual(member["disposition"], "VJP")
        receipt = accumulator.receipt(self.bundle)
        expected = {"L": .5 * .0625 / 4, "M": 0., "V": 0.}
        self.assertEqual(receipt["combined"], {"signed": expected, "positive": expected})
        self.assertTrue(receipt["S"])
        self.assertFalse(receipt["benefit_verified"])
        for key, value in self.model.items():
            original, clone = saved[key]
            self.assertIs(value.grad, original)
            if clone is not None:
                torch.testing.assert_close(value.grad, clone, atol=0, rtol=0)
        fixture[2].sum().backward()
        previous = saved["L"][1]
        assert previous is not None
        torch.testing.assert_close(self.model["L"].grad, previous + 1, atol=0, rtol=0)
        self.assertIsNone(self.model["M"].grad)

    def test_actual_online_group_credit_dto_reaches_same_forward_measurement(self):
        from fine_tuning.notebook_rl.repair_credit import repair_group_credit
        from fine_tuning.notebook_rl.schema import Group
        from fine_tuning.tests.test_notebook_rl_fresh_observation import ReversibleTokenizer
        from fine_tuning.tests.test_notebook_rl_repair_credit import OnlineRepairTests

        fixture = OnlineRepairTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        tokenizer = ReversibleTokenizer()
        good, unknown = fixture.child(), replace(fixture.child(intent="UNKNOWN"), sample_id="unknown", seed=3408)
        for sample in (good, unknown):
            sample.completion_token_ids = tokenizer.encode(sample.text)
            sample.sampler_logprobs = [-.1] * len(sample.completion_token_ids)
        group = Group(good.group_id, "T2", good.problem_id, _hash(good.prompt_token_ids), [good, unknown], True, None)
        credits = repair_group_credit(group, {row.sample_id: fixture.audit(row) for row in group.samples},
            tokenizer, inventory=fixture.inventory(), completion_cap=1024, component="online")
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        for sample, credit in zip(group.samples, credits, strict=True):
            current = self.model["L"][0].expand(1, len(sample.completion_token_ids))
            old, mask = torch.zeros_like(current), torch.ones_like(current, dtype=torch.bool)
            action = updates._PreparedAction(sample, RolloutRequest("original", sample.prompt_token_ids,
                None, [], 1, 1024, [sample.seed]), mask, 0., .5, 2048, False, old=old, keep=True)
            self.measure(accumulator, (action, credit, current))
        receipt = accumulator.receipt(self.bundle)
        self.assertTrue(receipt["S"])
        self.assertGreater(receipt["retained_positive_verified_tokens"], 0)
        self.assertEqual([member["source_bindings"]["group_members"] for member in receipt["members"]],
                         [[good.sample_id, unknown.sample_id]] * 2)

    def test_signed_vjps_cancel_but_positive_verified_sum_remains(self):
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        self.measure(accumulator, self.fixture())
        self.measure(accumulator, self.fixture(rows=[{"semantic_score": -1, "applied_score": -1, "verdict": "REFUTED"}]))
        receipt = accumulator.receipt(self.bundle)
        self.assertEqual(receipt["combined"]["signed"]["L"], 0.)
        self.assertGreater(receipt["combined"]["positive"]["L"], 0.)
        self.assertFalse(receipt["S"])
        self.assertEqual(receipt["retained_positive_verified_tokens"], 1)

    def test_positive_verified_vjps_cancel_independently_of_signed_sum(self):
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        self.measure(accumulator, self.fixture())
        self.measure(accumulator, self.fixture(current=-self.model["L"].unsqueeze(0)))
        self.measure(accumulator, self.fixture(rows=[{"semantic_score": -1, "applied_score": -1, "verdict": "REFUTED"}]))
        receipt = accumulator.receipt(self.bundle)
        self.assertGreater(receipt["combined"]["signed"]["L"], 0.)
        self.assertEqual(receipt["combined"]["positive"]["L"], 0.)
        self.assertFalse(receipt["S"])

    def test_cross_channel_parameter_addition_cancels_before_norm(self):
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        self.measure(accumulator, self.fixture(component="online", rows=[{"mean": .5}]))
        self.measure(accumulator, self.fixture(rows=[{"semantic_score": -1, "applied_score": -1, "verdict": "REFUTED"}]))
        receipt = accumulator.receipt(self.bundle)
        self.assertEqual(receipt["channels"]["online"]["signed"]["L"], receipt["channels"]["final"]["signed"]["L"])
        self.assertGreater(receipt["channels"]["online"]["signed"]["L"], 0.)
        self.assertEqual(receipt["combined"]["signed"]["L"], 0.)
        self.assertFalse(receipt["S"])
        self.assertEqual(len(accumulator.channels), 2)
        self.assertTrue(all(value.device.type == "cpu" and value.dtype == torch.float64
            for channel in accumulator.channels.values() for values in channel.values() for value in values.sums.values()))

    def test_noncorrected_provenance_is_excluded_even_with_positive_coefficients(self):
        for relation in ("PERSISTED", "UNKNOWN", "OTHER_BRANCH_COMPARISON", "FINALIZED"):
            with self.subTest(relation=relation):
                accumulator = updates._RepairGradientAccumulator(self.bundle)
                fixture = self.fixture(rows=[{"relation": relation}])
                self.assertGreater(fixture[1].coefficients[0], 0.)
                with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
                    member = self.measure(accumulator, fixture)
                grad.assert_not_called()
                self.assertEqual(member["disposition"], "ZERO_VECTOR")
                self.assertFalse(accumulator.receipt(self.bundle)["S"])

    def test_negative_only_and_zero_derivative_cannot_satisfy_gate(self):
        for rows, current in (([{"semantic_score": -1, "applied_score": -1, "verdict": "REFUTED"}], None),
                              (None, self.model["L"].unsqueeze(0) * 0)):
            with self.subTest(rows=rows):
                accumulator = updates._RepairGradientAccumulator(self.bundle)
                self.measure(accumulator, self.fixture(rows=rows, current=current))
                self.assertFalse(accumulator.receipt(self.bundle)["S"])

    def test_mis_fullmask_and_dtype_zero_have_explicit_dispositions(self):
        for mode in ("MIS_EXCLUDED", "MASKED", "CONVERTED_ZERO"):
            with self.subTest(mode=mode):
                accumulator = updates._RepairGradientAccumulator(self.bundle)
                fixture = self.fixture(keep=mode != "MIS_EXCLUDED")
                if mode == "MASKED":
                    fixture[0].mask[0, 0] = False
                elif mode == "CONVERTED_ZERO":
                    fixture = self.fixture(current=self.model["L"].unsqueeze(0).to(torch.float16),
                        rows=[{"applied_score": 0, "semantic_score": 0, "mean": -1e-50, "verdict": "UNVERIFIABLE"}])
                with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
                    member = self.measure(accumulator, fixture)
                grad.assert_not_called()
                self.assertEqual(member["disposition"], "MIS_EXCLUDED" if mode == "MIS_EXCLUDED" else "ZERO_VECTOR")
                self.assertEqual(member["positive_verified_tokens"], 0)
                self.assertFalse(accumulator.receipt(self.bundle)["S"])

    def test_off_on_off_preserves_fixed_inventory_and_active_route(self):
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        for enabled in (False, True, False):
            self.bundle.route.apply(enabled)
            current = (self.model["L"] + self.model["M"] + (self.model["V"] if enabled else 0)).unsqueeze(0)
            self.measure(accumulator, self.fixture(current=current, vision_on=enabled))
        receipt = accumulator.receipt(self.bundle)
        self.assertEqual([row["group"] for row in receipt["parameter_inventory"]], ["L", "M", "V"])
        self.assertEqual(receipt["combined"]["signed"]["V"] * 3, receipt["combined"]["signed"]["L"])
        self.assertEqual([row["vision_on"] for row in receipt["members"]], [False, True, False])

    def test_nonfinite_vjp_aborts_receipt_without_changing_grad(self):
        accumulator = updates._RepairGradientAccumulator(self.bundle)
        fixture = self.fixture()
        self.model["L"].grad = torch.ones_like(self.model["L"])
        original = self.model["L"].grad
        hook = self.model["L"].register_hook(lambda value: value * float("nan"))
        try:
            with self.assertRaisesRegex(RuntimeError, "NONFINITE_REPAIR_GRADIENT"):
                self.measure(accumulator, fixture)
        finally:
            hook.remove()
        self.assertIs(self.model["L"].grad, original)
        torch.testing.assert_close(original, torch.ones_like(original), atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            accumulator.receipt(self.bundle)

    def test_inventory_rename_or_shape_change_is_rejected_before_vjp_and_receipt(self):
        for stage in ("measure", "receipt"):
            for mutation in ("rename", "shape"):
                with self.subTest(stage=stage, mutation=mutation):
                    self.setUp()
                    accumulator = updates._RepairGradientAccumulator(self.bundle)
                    fixture = self.fixture()
                    if stage == "receipt":
                        self.measure(accumulator, fixture)
                    if mutation == "rename":
                        parameter = self.model["L"]
                        del self.model["L"]
                        self.model["renamed"] = parameter
                    else:
                        self.model["L"].data = torch.zeros(2, dtype=torch.float64)
                    with self.assertRaisesRegex(ValueError, "parameter inventory changed"):
                        self.measure(accumulator, fixture) if stage == "measure" else accumulator.receipt(self.bundle)

    def test_shape_arithmetic_and_semantic_provenance_mutations_are_rejected(self):
        for mutation in ("shape", "coefficient", "ownership", "positive", "signed", "membership", "cap", "norm", "route"):
            with self.subTest(mutation=mutation):
                accumulator = updates._RepairGradientAccumulator(self.bundle)
                action, credit, current = self.fixture()
                if mutation == "shape":
                    credit = replace(credit, coefficients=credit.coefficients[:-1])
                elif mutation == "coefficient":
                    credit = replace(credit, coefficients=(float("nan"), 0., 0., 0.))
                elif mutation == "ownership":
                    credit.unit_rows[0]["token_indices"] = [1]
                elif mutation in ("positive", "signed"):
                    credit.unit_rows[0]["positive_verified_corrected" if mutation == "positive" else "corrected_channel"] = False
                    credit = replace(credit, ownership_sha256=_hash(credit.unit_rows))
                elif mutation == "membership":
                    credit.source_bindings["group_members"] = ["forged"] * 4
                elif mutation == "cap":
                    credit = replace(credit, completion_cap=1024)
                elif mutation == "norm":
                    action.norm = 1
                else:
                    action.vision_on = True
                with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad, self.assertRaises(ValueError):
                    self.measure(accumulator, (action, credit, current))
                grad.assert_not_called()

    def test_mutation_after_measurement_or_during_vjp_cannot_issue_receipt(self):
        for mutation in ("credit", "sample", "old", "member", "during"):
            with self.subTest(mutation=mutation):
                accumulator = updates._RepairGradientAccumulator(self.bundle)
                fixture = self.fixture()
                action, credit, _ = fixture
                if mutation == "during":
                    def mutate(gradient):
                        credit.unit_rows[0]["verdict"] = "REFUTED"
                        return gradient
                    hook = self.model["L"].register_hook(mutate)
                    try:
                        with self.assertRaises(ValueError):
                            self.measure(accumulator, fixture)
                    finally:
                        hook.remove()
                    continue
                self.measure(accumulator, fixture)
                if mutation == "credit":
                    credit.source_bindings["audit_sha256"] = "d" * 64
                elif mutation == "sample":
                    action.sample.text = "changed"
                elif mutation == "old":
                    action.old[0, 0] = .1
                else:
                    accumulator.members[0]["positive_verified_tokens"] = 0
                with self.assertRaises(ValueError):
                    accumulator.receipt(self.bundle)


if __name__ == "__main__":
    unittest.main()
