"""Hand-calculated CPU objectives; these fixtures are not model learning evidence."""
import math
import unittest

import torch

from fine_tuning.notebook_rl.losses import repair_component_losses


def tensor(values, *, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def evaluate(cur, *, old=None, ref=None, answer=None, online=None, final=None,
             keep=None, mask=None, kind="T2", n=2, cap=1024):
    return repair_component_losses(cur, cur.detach().clone() if old is None else old,
        cur.detach().clone() if ref is None else ref,
        torch.ones_like(cur, dtype=torch.bool) if mask is None else mask,
        torch.ones(cur.shape[0], dtype=torch.bool) if keep is None else keep,
        answer=torch.zeros_like(cur) if answer is None else answer,
        online=torch.zeros_like(cur) if online is None else online,
        final=torch.zeros_like(cur) if final is None else final,
        kind=kind, original_group_size=n, completion_cap=cap)


class RepairLossTests(unittest.TestCase):
    def test_opposing_channels_are_clipped_separately_and_kl_once(self):
        cur = tensor([[math.log(1.5), math.log(.5)]], grad=True)
        old = torch.zeros_like(cur)
        ref = cur.detach() + .2
        answer, online = tensor([[2., -2.]]), tensor([[-2., 2.]])
        result = evaluate(cur, old=old, ref=ref, answer=answer, online=online)
        norm = 2 * 1024
        expected_answer = -(1.28 * 2 + .8 * -2) / norm
        expected_online = -(1.5 * -2 + .5 * 2) / norm
        expected_kl = .01 * 2 * (math.expm1(.2) - .2) / norm
        self.assertAlmostEqual(result["answer"].item(), expected_answer, places=14)
        self.assertAlmostEqual(result["online"].item(), expected_online, places=14)
        self.assertAlmostEqual(result["kl"].item(), expected_kl, places=14)
        self.assertAlmostEqual(result["total"].item(), expected_answer + expected_online + expected_kl, places=14)
        self.assertNotEqual(result["total"].item(), result["kl"].item())
        torch.testing.assert_close(torch.autograd.grad(result["total"], cur)[0],
            tensor([[3., -1.]]) / norm + .01 * (1 - math.exp(.2)) / norm)

    def test_repair_cap_cancels_once_but_answer_keeps_original_normalization(self):
        for kind, n, cap, channel in (("T2", 2, 1024, "online"), ("DIAG", 4, 8192, "final")):
            cur = tensor([[-1., -2., -3.]], grad=True)
            coeff = torch.full_like(cur, cap * .0625 * .5 / (7 * 3))
            result = evaluate(cur, kind=kind, n=n, cap=cap, **{channel: coeff})
            self.assertAlmostEqual(result["total"].item(), -.0625 * .5 / (n * 7), places=14)
            torch.testing.assert_close(torch.autograd.grad(result["total"], cur)[0],
                torch.full_like(cur, -.0625 * .5 / (n * 7 * 3)))
        cur = tensor([[-1., -2.]])
        self.assertAlmostEqual(evaluate(cur, answer=torch.full_like(cur, .5))["total"].item(), -1 / 2048)

    def test_microbatch_matches_complete_group_and_mask_does_not_recenter(self):
        cur = tensor([[-1., -2.], [-3., -4.]], grad=True)
        coefficients = tensor([[.5, .5], [-.5, -.5]])
        keep = torch.tensor([True, False])
        whole = evaluate(cur, online=coefficients, keep=keep)["total"]
        pieces = sum(evaluate(cur[i:i+1], online=coefficients[i:i+1], keep=keep[i:i+1])["total"] for i in range(2))
        torch.testing.assert_close(whole, pieces)
        self.assertAlmostEqual(whole.item(), -1 / 2048)
        torch.testing.assert_close(torch.autograd.grad(whole, cur)[0], tensor([[-.5, -.5], [0., 0.]]) / 2048)

    def test_only_current_receives_gradients_and_zero_channels_leave_kl(self):
        cur = tensor([[-1., -2.]], grad=True)
        old = cur.detach().clone().requires_grad_()
        ref = (cur.detach() + .3).requires_grad_()
        coefficients = tensor([[.5, 0.]], grad=True)
        result = evaluate(cur, old=old, ref=ref, online=coefficients)
        grads = torch.autograd.grad(result["total"], (cur, old, ref, coefficients), allow_unused=True)
        self.assertEqual(grads[1:], (None, None, None))
        zero = evaluate(cur, old=old, ref=ref)
        torch.testing.assert_close(zero["total"], zero["kl"])

    def test_solution_mask_keeps_answer_pg_off_error_check_tokens(self):
        cur = tensor([[-1., -2., -3., -4.]], grad=True)
        answer = tensor([[0., 0., .5, .5]])
        online = tensor([[1., 1., .25, .25]])
        result = evaluate(cur, answer=answer, online=online)
        torch.testing.assert_close(torch.autograd.grad(result["answer"], cur, retain_graph=True)[0], -answer / 2048)
        torch.testing.assert_close(torch.autograd.grad(result["online"], cur)[0], -online / 2048)

    def test_invalid_contract_and_channels_are_rejected(self):
        cur = tensor([[-1., -2.]])
        cases = ({"kind": "bad"}, {"n": True}, {"cap": True}, {"n": 4}, {"cap": 8192},
                 {"final": torch.ones_like(cur)}, {"online": tensor([[float("nan"), 0.]])},
                 {"online": tensor([[1.]])}, {"kind": "T1", "n": 4, "online": torch.ones_like(cur)},
                 {"kind": "DIAG", "n": 4, "cap": 8192, "answer": torch.ones_like(cur)},
                 {"mask": torch.tensor([[True, False]]), "online": tensor([[0., 1.]])})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate(cur, **kwargs)
        with self.assertRaises(ValueError):
            evaluate(torch.zeros((3, 2), dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
