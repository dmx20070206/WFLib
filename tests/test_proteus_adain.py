"""Support sampling remains class conditioned under OT adaptation."""

import unittest
import torch
from exp.proteus import sample_target_support


class SupportSamplingTest(unittest.TestCase):
    def test_class_pairing_and_reproducibility(self):
        support = [torch.tensor([[1.0], [2.0]]), torch.tensor([[10.0], [20.0]])]
        labels = torch.arange(128) % 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            first = sample_target_support(support, labels, "cpu")
            torch.manual_seed(7)
            second = sample_target_support(support, labels, "cpu")
        torch.testing.assert_close(first, second)
        self.assertEqual(set(first[labels == 0].flatten().tolist()), {1.0, 2.0})
        self.assertEqual(set(first[labels == 1].flatten().tolist()), {10.0, 20.0})


if __name__ == "__main__":
    unittest.main()
