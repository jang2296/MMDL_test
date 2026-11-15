"""Full-vocabulary probabilities retain generated history across projection chunks."""
import unittest

import torch

from fine_tuning.error_note.rl import negative_token_mask, processed_full_logprobs


class FullLogprobTests(unittest.TestCase):
    def test_vectorized_matches_reference_at_chunk_boundaries_and_gradients(self):
        torch.manual_seed(31)
        actions = torch.randint(0, 23, (67,))
        logits = torch.randn(67, 23, requires_grad=True)
        reference = []
        for i, row in enumerate(logits):
            seen = torch.zeros(23)
            seen[actions[:i]] = 1
            reference.append(((row - 1.5 * seen) / .7).log_softmax(-1)[actions[i]])
        reference = torch.stack(reference)
        for chunk in (1, 13, 32, 67):
            actual = torch.cat([processed_full_logprobs(logits[i:i + chunk], actions, i)
                                for i in range(0, len(actions), chunk)])
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=0)
        actual.sum().backward()
        gradient = logits.grad.clone()
        logits.grad = None
        reference.sum().backward()
        torch.testing.assert_close(logits.grad, gradient, atol=1e-6, rtol=0)

    def test_negative_mask_boundary_and_exactly_zero_gradient(self):
        values = torch.tensor([.8, .5, .2]).log().requires_grad_()
        mask = negative_token_mask(values)
        self.assertEqual(mask.tolist(), [False, True, True])
        (values * mask).sum().backward()
        self.assertEqual(values.grad.tolist(), [0., 1., 1.])

    def test_invalid_inputs_are_rejected(self):
        for logits, actions, offset in ((torch.zeros(2, 3), torch.tensor([0, 3]), 0),
                                        (torch.full((2, 3), float("nan")), torch.tensor([0, 1]), 0),
                                        (torch.zeros(2, 3), torch.tensor([0, 1]), 1)):
            with self.assertRaises(ValueError):
                processed_full_logprobs(logits, actions, offset)
        for value in (torch.tensor([float("nan")]), torch.tensor([.1])):
            with self.assertRaises(ValueError):
                negative_token_mask(value)


if __name__ == "__main__":
    unittest.main()
