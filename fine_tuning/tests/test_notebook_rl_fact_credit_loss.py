"""Independent CPU arithmetic checks for separately clipped fact-token credit."""
import math
import unittest

import torch

from fine_tuning.notebook_rl.losses import group_loss, mis_mask


def tensor(values, *, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def loss(current, *, old=None, reference=None, strict=None, partial=None,
         mask=None, keep=None, beta=0.0, norm=16.0):
    batch, length = current.shape
    return group_loss(
        current, current.detach().clone() if old is None else old,
        current.detach().clone() if reference is None else reference,
        tensor([0.0] * batch) if strict is None else strict,
        torch.ones((batch, length), dtype=torch.bool) if mask is None else mask,
        torch.ones(batch, dtype=torch.bool) if keep is None else keep,
        eps_low=0.2, eps_high=0.28, beta=beta, norm_const=norm, partial_adv=partial)


def handwritten(current, old, reference, strict, partial, mask, keep, beta, norm):
    """Evaluate each clipped term and its derivative using scalar arithmetic."""
    value = 0.0
    gradients = [[0.0] * len(row) for row in current]
    for i, row in enumerate(current):
        for j, logprob in enumerate(row):
            if not keep[i] or not mask[i][j]:
                continue
            ratio = math.exp(logprob - old[i][j])
            clipped = min(max(ratio, 0.8), 1.28)
            derivative = 0.0
            for advantage in (strict[i], partial[i][j]):
                value -= min(ratio * advantage, clipped * advantage) / norm
                saturated = ((advantage > 0 and ratio > 1.28)
                             or (advantage < 0 and ratio < 0.8))
                if not saturated:
                    derivative -= ratio * advantage
            delta = reference[i][j] - logprob
            value += beta * (math.expm1(delta) - delta) / norm
            gradients[i][j] = (derivative + beta * (1 - math.exp(delta))) / norm
    return value, tensor(gradients)


class FactCreditLossTests(unittest.TestCase):
    def test_handwritten_loss_and_gradient_with_fixed_full_group_denominator(self):
        old = [[-2.0] * 4, [-3.0] * 4]
        ratios = [[1.5, 0.5, 1.1, 0.9], [0.6, 1.6, 0.95, 1.05]]
        current = [[old[i][j] + math.log(ratio) for j, ratio in enumerate(row)]
                   for i, row in enumerate(ratios)]
        reference = [[value + 0.15 for value in row] for row in current]
        strict = [0.7, -0.7]
        partial = [[-0.2, 0.3, 0.0, -0.4], [0.1, -0.3, 0.5, 0.0]]
        mask = [[1, 1, 1, 0], [1, 1, 0, 1]]
        value = tensor(current, grad=True)
        actual = loss(value, old=tensor(old), reference=tensor(reference),
                      strict=tensor(strict), partial=tensor(partial), mask=torch.tensor(mask),
                      beta=0.13, norm=32)
        expected, gradient = handwritten(current, old, reference, strict, partial,
                                         mask, [True, True], 0.13, 32)
        self.assertAlmostEqual(actual.item(), expected, places=13)
        torch.testing.assert_close(torch.autograd.grad(actual, value)[0], gradient,
                                   rtol=1e-12, atol=1e-14)

    def test_opposing_terms_are_clipped_separately_in_both_directions(self):
        current = tensor([[math.log(1.5)], [math.log(0.5)]], grad=True)
        actual = loss(current, old=torch.zeros_like(current), strict=tensor([2, -2]),
                      partial=tensor([[-1], [1]]), norm=1)
        self.assertAlmostEqual(actual.item(), -(2 * 1.28 - 1.5 - 2 * 0.8 + 0.5))
        torch.testing.assert_close(torch.autograd.grad(actual, current)[0], tensor([[1.5], [-0.5]]))
        collapsed = loss(current, old=torch.zeros_like(current), strict=tensor([1, -1]), norm=1)
        self.assertNotAlmostEqual(actual.item(), collapsed.item())
        torch.testing.assert_close(torch.autograd.grad(collapsed, current)[0], tensor([[0], [0]]))

    def test_zero_partial_is_exactly_backward_compatible(self):
        current = tensor([[-1, -2, -3], [-2, -3, -4]], grad=True)
        old = current.detach() - 0.17
        reference = current.detach() + 0.23
        strict = tensor([0.4, -0.4])
        mask = torch.tensor([[1, 1, 0], [1, 0, 1]])
        keep = torch.tensor([True, True])
        original = group_loss(current, old, reference, strict, mask, keep,
                              eps_low=0.2, eps_high=0.28, beta=0.07, norm_const=32)
        expanded = loss(current, old=old, reference=reference, strict=strict,
                        partial=torch.zeros_like(current), mask=mask, keep=keep, beta=0.07, norm=32)
        self.assertTrue(torch.equal(original, expanded))
        original_gradient = torch.autograd.grad(original, current)[0]
        self.assertTrue(torch.equal(original_gradient, torch.autograd.grad(expanded, current)[0]))

    def test_zero_and_negative_partial_advantages_need_no_positive_token(self):
        for values in ([[0.0, 0.0, 0.0]], [[-0.25, 0.0, -0.5]]):
            with self.subTest(values=values):
                current = tensor([[-1, -2, -3]], grad=True)
                partial = tensor(values)
                actual = loss(current, partial=partial)
                self.assertEqual(actual.item(), -partial.sum().item() / 16)
                torch.testing.assert_close(torch.autograd.grad(actual, current)[0], -partial / 16)

    def test_partial_old_reference_and_strict_are_detached(self):
        current = tensor([[-1, -2]], grad=True)
        old = current.detach().clone().requires_grad_()
        reference = tensor([[-0.8, -1.7]], grad=True)
        strict = tensor([0.2], grad=True)
        partial = tensor([[0.1, -0.3]], grad=True)
        actual = loss(current, old=old, reference=reference, strict=strict, partial=partial, beta=0.2)
        gradients = torch.autograd.grad(actual, (current, old, reference, strict, partial), allow_unused=True)
        expected = (-(strict.detach().unsqueeze(1) + partial.detach())
                    + 0.2 * (1 - (reference.detach() - current.detach()).exp())) / 16
        torch.testing.assert_close(gradients[0], expected)
        self.assertEqual(gradients[1:], (None, None, None, None))

    def test_kl_is_counted_once_over_all_completion_tokens(self):
        current = tensor([[-1, -2, -3]], grad=True)
        reference = current.detach() + tensor([[0.1, 0.3, -0.2]])
        partial = tensor([[0.4, 0.0, 0.0]])
        actual = loss(current, reference=reference, partial=partial, beta=0.2)
        delta = reference - current.detach()
        expected = (-0.4 + 0.2 * (delta.exp() - delta - 1).sum()) / 16
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual, current)[0]
        torch.testing.assert_close(gradient, (-partial + 0.2 * (1 - delta.exp())) / 16)
        self.assertNotEqual(gradient[0, 1].item(), 0)
        self.assertNotEqual(gradient[0, 2].item(), 0)

    def test_mis_keeps_full_completion_scope_and_does_not_drop_zero_credit_samples(self):
        old = tensor([[-1, -1], [-1, -1]])
        sampled = old - tensor([[0, 0.12], [0.01, 0.03]])
        mask = torch.ones_like(old, dtype=torch.bool)
        before = mask.clone()
        keep, delta = mis_mask(old, sampled, mask)
        self.assertEqual(keep.tolist(), [False, True])
        torch.testing.assert_close(delta, tensor([0.06, 0.02]))
        current = old.clone().requires_grad_()
        partial = tensor([[0.5, 0], [0, 0]])
        actual = loss(current, reference=old + 0.2, partial=partial, mask=mask, keep=keep, beta=0.1)
        gradient = torch.autograd.grad(actual, current)[0]
        torch.testing.assert_close(gradient[0], tensor([0, 0]))
        torch.testing.assert_close(gradient[1], tensor([0.1 * (1 - math.exp(0.2)) / 16] * 2))
        self.assertTrue(torch.equal(mask, before))
        after_keep, after_delta = mis_mask(old, sampled, mask)
        self.assertTrue(torch.equal(keep, after_keep))
        self.assertTrue(torch.equal(delta, after_delta))

    def test_action_and_sequence_masks_still_exclude_partial_gradients(self):
        current = tensor([[-1, -1000], [-1000, -1000]], grad=True)
        old = tensor([[-1, 1000], [1000, 1000]])
        partial = tensor([[0.25, -0.5], [0.75, -1.0]])
        actual = loss(current, old=old, partial=partial, mask=torch.tensor([[1, 0], [1, 1]]),
                      keep=torch.tensor([True, False]))
        torch.testing.assert_close(torch.autograd.grad(actual, current)[0], tensor([[-0.25, 0], [0, 0]]) / 16)

    def test_invalid_partial_shape_type_and_nonfinite_values_are_rejected(self):
        current = tensor([[-1, -2], [-3, -4]])
        malformed = [tensor([0, 0]), tensor([[0], [0]]), tensor([[[0, 0], [0, 0]]]),
                     tensor([[0, float("nan")], [0, 0]]),
                     tensor([[0, float("inf")], [0, 0]]),
                     torch.zeros((2, 2), dtype=torch.bool), torch.zeros((2, 2), dtype=torch.int64),
                     torch.zeros((2, 2), dtype=torch.complex128), [[0, 0], [0, 0]], True]
        for partial in malformed:
            with self.subTest(partial=partial), self.assertRaises(ValueError):
                loss(current, partial=partial)

    def test_partial_device_mismatch_is_rejected_without_cuda(self):
        current = tensor([[-1, -2]])
        with self.assertRaises(ValueError):
            loss(current, partial=torch.zeros((1, 2), dtype=torch.float64, device="meta"))


if __name__ == "__main__":
    unittest.main()
