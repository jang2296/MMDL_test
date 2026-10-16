"""Synthetic CPU numeric/autograd checks for the v2 RL objective only."""
import math
import unittest

import torch

from fine_tuning.notebook_rl.losses import combine, group_advantages, group_loss, k3, mis_mask


def tensor(values, *, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def loss(cur, old=None, ref=None, adv=None, mask=None, keep=None, **kwargs):
    b, t = cur.shape
    return group_loss(cur, cur.detach().clone() if old is None else old,
                      cur.detach().clone() if ref is None else ref,
                      tensor([1.0] * b) if adv is None else adv,
                      torch.ones((b, t), dtype=torch.bool) if mask is None else mask,
                      torch.ones(b, dtype=torch.bool) if keep is None else keep,
                      **({"eps_low": 0.2, "eps_high": 0.28, "beta": 0.01,
                          "norm_const": float(b * 4)} | kwargs))


def handwritten(cur, old, ref, adv, mask, keep, norm, beta):
    """Independent scalar formula and derivative away from clipping boundaries."""
    total = 0.0
    gradients = [[0.0] * len(row) for row in cur]
    for i, row in enumerate(cur):
        for j, value in enumerate(row):
            if not mask[i][j] or not keep[i]:
                continue
            ratio = math.exp(value - old[i][j])
            clipped = min(max(ratio, 0.8), 1.28)
            pg = min(ratio * adv[i], clipped * adv[i])
            delta = ref[i][j] - value
            total += (-pg + beta * (math.exp(delta) - delta - 1)) / norm
            saturated = (adv[i] > 0 and ratio > 1.28) or (adv[i] < 0 and ratio < 0.8)
            pg_grad = 0.0 if saturated else -ratio * adv[i]
            gradients[i][j] = (pg_grad + beta * (1 - math.exp(delta))) / norm
    return total, tensor(gradients)


class LossTests(unittest.TestCase):
    def test_group_mean_without_std_and_detached(self):
        rewards = tensor([0.0, 0.0, 3.0], grad=True)
        actual = group_advantages(rewards)
        torch.testing.assert_close(actual, tensor([-1.0, -1.0, 2.0]))
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(group_advantages(tensor([7.0, 7.0])), tensor([0.0, 0.0]))

    def test_two_groups_handwritten_loss_and_gradients(self):
        inputs = [
            ([[-1.0, -2.0, -7.0], [-2.0, -1.0, -3.0]],
             [[-1.1, -2.5, -7.0], [-1.5, -1.1, -3.0]],
             [[-0.8, -2.3, -7.0], [-1.8, -1.0, -3.0]],
             [1.5, -1.5], [[1, 1, 0], [1, 1, 1]], [True, True], 8.0),
            ([[-1.4, -0.9], [-1.0, -2.0]], [[-1.2, -1.3], [-1.0, -2.0]],
             [[-1.5, -1.1], [-0.7, -1.0]], [-0.7, 0.7],
             [[1, 1], [1, 1]], [True, False], 6.0),
        ]
        currents, losses, expected_losses, expected_grads = [], [], [], []
        weights = [1.0, 0.5]
        for c, o, r, a, m, keep, norm in inputs:
            cur = tensor(c, grad=True)
            currents.append(cur)
            losses.append(loss(cur, tensor(o), tensor(r), tensor(a), torch.tensor(m),
                               torch.tensor(keep), norm_const=norm, beta=0.07))
            value, gradient = handwritten(c, o, r, a, m, keep, norm, 0.07)
            expected_losses.append(value)
            expected_grads.append(gradient)
        objective = combine(losses, weights)
        self.assertAlmostEqual(objective.item(), sum(x * w for x, w in
                               zip(expected_losses, weights, strict=True)) / 1.5, places=13)
        for actual, expected, weight in zip(torch.autograd.grad(objective, currents),
                                             expected_grads, weights, strict=True):
            torch.testing.assert_close(actual, expected * weight / 1.5, rtol=1e-12, atol=1e-14)

    def test_asymmetric_clipping_and_advantage_sign(self):
        ratios = tensor([[1.5], [0.5], [1.1], [0.9]])
        cur = ratios.log().requires_grad_()
        actual = loss(cur, torch.zeros_like(cur), adv=tensor([2, -2, 2, -2]),
                      beta=0, norm_const=1)
        self.assertAlmostEqual(actual.item(), -(2 * 1.28 - 2 * 0.8 + 2 * 1.1 - 2 * 0.9))
        torch.testing.assert_close(torch.autograd.grad(actual, cur)[0],
                                   tensor([[0], [0], [-2.2], [1.8]]))

    def test_ratio_one_clip_inactive_and_old_ref_adv_detached(self):
        cur = tensor([[-1, -2], [-2, -3]], grad=True)
        old = cur.detach().clone().requires_grad_()
        ref = tensor([[-0.9, -1.5], [-2.4, -3.1]], grad=True)
        adv = tensor([0.5, -0.5], grad=True)
        actual = loss(cur, old, ref, adv, beta=0)
        gradients = torch.autograd.grad(actual, (cur, old, ref, adv), allow_unused=True)
        torch.testing.assert_close(gradients[0], tensor([[-0.5, -0.5], [0.5, 0.5]]) / 8)
        self.assertEqual(gradients[1:], (None, None, None))

    def test_equal_rewards_leave_only_kl_without_ratio_weighting(self):
        cur = tensor([[-1, -2], [-3, -4]], grad=True)
        ref = tensor([[-1.3, -1.5], [-3, -3.8]], grad=True)
        actual = loss(cur, old=cur.detach() - 0.4, ref=ref,
                      adv=group_advantages(tensor([1, 1])), beta=0.2)
        expected = 0.2 * k3(ref, cur).sum() / 8
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, cur)[0],
                                   0.2 * (1 - (ref.detach() - cur.detach()).exp()) / 8)

    def test_k3_values_nonnegative_and_only_current_gradient(self):
        cur = tensor([-1, -2, -3, -4], grad=True)
        delta = tensor([0, 1e-8, -0.5, 0.7])
        ref = (cur.detach() + delta).requires_grad_()
        actual = k3(ref, cur)
        expected = tensor([math.expm1(x) - x for x in delta.tolist()])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-16)
        self.assertTrue(bool((actual >= 0).all()))
        grad_cur, grad_ref = torch.autograd.grad(actual.sum(), (cur, ref), allow_unused=True)
        torch.testing.assert_close(grad_cur, 1 - delta.exp())
        self.assertIsNone(grad_ref)

    def test_mis_uses_action_token_mean_not_sum_or_absolute_mean(self):
        old = tensor([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], grad=True)
        samp = tensor([[-1.1, -1.1, -100], [-1.01, -1.01, -100],
                       [-1.1, -0.9, -100]], grad=True)
        mask = torch.tensor([[1, 1, 0]] * 3)
        keep, delta = mis_mask(old, samp, mask)
        self.assertEqual(keep.tolist(), [False, True, True])
        torch.testing.assert_close(delta, tensor([0.1, 0.01, 0]), atol=1e-15, rtol=0)
        self.assertFalse(delta.requires_grad)
        exact_old = torch.zeros((2, 1), dtype=torch.float64)
        self.assertEqual(mis_mask(exact_old, tensor([[-0.05], [0.05]]),
                                  torch.ones_like(exact_old))[0].tolist(), [True, True])

    def test_microbatch_global_group_and_step_normalization(self):
        cur = tensor([[-1, -2], [-2, -1], [-3, -2]], grad=True)
        old = cur.detach() - tensor([[0.1, 0.4], [-0.3, 0.1], [0.2, 0.2]])
        ref = cur.detach() + 0.2
        adv = tensor([1, -0.5, -0.5])
        keep = torch.tensor([True, True, False])
        mask = torch.tensor([[1, 0], [1, 1], [1, 1]])
        whole = loss(cur, old, ref, adv, mask, keep, norm_const=12)
        pieces = [loss(cur[i:i+1], old[i:i+1], ref[i:i+1], adv[i:i+1],
                       mask[i:i+1], keep[i:i+1], norm_const=12) for i in range(3)]
        torch.testing.assert_close(sum(pieces), whole)
        other = loss(cur[:2], old[:2], ref[:2], adv[:2], mask[:2], keep[:2], norm_const=16)
        full_step = combine([whole, other], [1.0, 0.5])
        sliced_step = sum(pieces) / 1.5 + other * 0.5 / 1.5
        torch.testing.assert_close(sliced_step, full_step)
        torch.testing.assert_close(torch.autograd.grad(sliced_step, cur, retain_graph=True)[0],
                                   torch.autograd.grad(full_step, cur)[0])

    def test_masks_zero_gradients_even_with_extreme_inactive_logprobs(self):
        cur = tensor([[-1, -1000], [-1000, -1000]], grad=True)
        old = tensor([[-1, 1000], [1000, 1000]])
        ref = tensor([[-0.8, 1000], [1000, 1000]])
        actual = loss(cur, old, ref, mask=torch.tensor([[1, 0], [1, 1]]),
                      keep=torch.tensor([True, False]))
        gradient = torch.autograd.grad(actual, cur)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        torch.testing.assert_close(gradient[0, 1:], tensor([0]))
        torch.testing.assert_close(gradient[1], tensor([0, 0]))
        excluded = loss(cur, old, ref, keep=torch.tensor([False, False]))
        self.assertEqual(excluded.item(), 0)
        torch.testing.assert_close(torch.autograd.grad(excluded, cur)[0], torch.zeros_like(cur))

    def test_low_precision_work_is_fp32(self):
        cur = torch.tensor([[-1, -2]], dtype=torch.bfloat16, requires_grad=True)
        result = loss(cur, adv=torch.ones(1, dtype=torch.bfloat16))
        self.assertEqual(result.dtype, torch.float32)
        result.backward()
        self.assertTrue(bool(torch.isfinite(cur.grad).all()))

    def test_invalid_tensor_shapes_types_and_nonfinite_values(self):
        cur = tensor([[-1, -2], [-2, -3]])
        for rewards in (tensor([1]), tensor([[1, 2]]), tensor([1, float("nan")]),
                        torch.tensor([1, 2]), [1, 2]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                group_advantages(rewards)
        changes = [dict(old=tensor([[-1]])), dict(ref=tensor([[-1]])),
                   dict(adv=tensor([[1], [-1]])), dict(adv=tensor([1, float("inf")])),
                   dict(keep=torch.tensor([1, 0])), dict(keep=torch.tensor([True])),
                   dict(mask=tensor([[1, 0.5], [1, 1]])),
                   dict(mask=torch.zeros((2, 2))), dict(mask=torch.ones((2, 1))),
                   dict(old=tensor([[-1, float("nan")], [-2, -3]]))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                loss(cur, **change)
        for malformed in (tensor([]), tensor([-1, -2]), torch.ones((2, 2), dtype=torch.int64),
                          tensor([[float("inf")]])):
            with self.subTest(cur=malformed), self.assertRaises(ValueError):
                loss(malformed)
        for call in (lambda: mis_mask(cur, cur, torch.zeros_like(cur)),
                     lambda: mis_mask(cur, cur[:, :1], torch.ones_like(cur)),
                     lambda: k3(cur[:, :1], cur),
                     lambda: k3(tensor([float("nan")]), tensor([0]))):
            with self.assertRaises(ValueError):
                call()

    def test_invalid_scalars_combinations_and_overflow(self):
        cur = tensor([[-1]])
        for name, values in (("eps_low", [-1, 1, float("nan"), True]),
                             ("eps_high", [-0.1, float("inf")]),
                             ("beta", [-1, float("nan")]),
                             ("norm_const", [0, -1, float("inf"), True])):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    loss(cur, **{name: value})
        for value in (-1, float("nan"), True):
            with self.assertRaises(ValueError):
                mis_mask(cur, cur, torch.ones_like(cur), value)
        for losses, weights in (([], []), ([tensor(1)], []), ([tensor([1])], [1]),
                                ([tensor(1)], [0]), ([tensor(1)], [-1]),
                                ([tensor(1)], [float("inf")]), ([tensor(float("nan"))], [1])):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                combine(losses, weights)
        for call in (lambda: k3(tensor([1000]), tensor([-1000])),
                     lambda: loss(cur, old=tensor([[-1000]]))):
            with self.assertRaisesRegex(ValueError, "finite"):
                call()


if __name__ == "__main__":
    unittest.main()
