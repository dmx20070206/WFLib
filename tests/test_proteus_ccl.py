"""Partial optimal transport and unmatched-token behavior."""

import unittest
import torch
from exp.proteus import partial_ot_loss, unbalanced_transport


class PartialOTTest(unittest.TestCase):
    def test_uot_prefers_low_cost_and_relaxes_marginals(self):
        cost = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
        plan = unbalanced_transport(cost, 0.1, 0.05, 0.05, 80)
        self.assertGreater(plan[0, 0, 0], plan[0, 0, 1])
        expensive = torch.ones(1, 2, 2)
        relaxed = unbalanced_transport(expensive, 0.1, 0.01, 0.01, 80)
        balanced = unbalanced_transport(expensive, 0.1, 10.0, 10.0, 80)
        self.assertLess(relaxed.sum(), balanced.sum())

    def test_rejected_tokens_receive_no_alignment_gradient(self):
        source = torch.eye(3).unsqueeze(0).requires_grad_()
        target = (
            torch.diag(torch.tensor([1.0, 1.0, -1.0])).unsqueeze(0).requires_grad_()
        )
        loss = partial_ot_loss(source, target, dustbin_cost=0.2, position_weight=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(source.grad[0, :, 2].abs().sum(), 0)
        self.assertEqual(target.grad[0, :, 2].abs().sum(), 0)

    def test_direction_prior_increases_cross_direction_cost(self):
        features = torch.ones(1, 2, 1)
        upload = torch.tensor([[[[1.0], [0.0]]]])
        download = torch.tensor([[[[0.0], [1.0]]]])
        without = partial_ot_loss(
            features, features, upload, download, direction_weight=0, dustbin_cost=2
        )
        with_prior = partial_ot_loss(
            features, features, upload, download, direction_weight=0.8, dustbin_cost=2
        )
        self.assertGreater(with_prior, without)


if __name__ == "__main__":
    unittest.main()
