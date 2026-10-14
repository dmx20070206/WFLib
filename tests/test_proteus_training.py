"""Supervised adaptation integration checks."""

import unittest
import torch
from exp.proteus import (
    Adapter,
    adaptation_losses,
    make_loader,
    make_parser,
    train_adaptation,
)
from WFlib.models import DMX


class ProteusTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.source = torch.rand(4, 1, 2, 180)
        self.target = torch.rand(4, 1, 2, 180)
        self.labels = torch.tensor([0, 0, 1, 1])

    def test_target_supervision_updates_encoder_without_ot(self):
        adapter = Adapter(DMX(2))
        adapter.configure_training(True)
        loss = adaptation_losses(
            adapter, self.source, self.target, self.labels, source_weight=0, ot_weight=0
        )
        loss["loss"].backward()
        self.assertGreater(adapter.model.first_layer[0].weight.grad.abs().sum(), 0)
        self.assertEqual(float(loss["ot"]), 0.0)

    def test_one_epoch_uses_same_class_support_and_partial_ot(self):
        adapter = Adapter(DMX(2))
        before = adapter.model.first_layer[0].weight.detach().clone()
        source = make_loader(self.source, self.labels, 4, False, 0)
        support = make_loader(self.target, self.labels, 4, False, 0)
        train_adaptation(adapter, source, support, 2, "cpu", epochs=1, ot_iterations=5)
        self.assertFalse(torch.equal(before, adapter.model.first_layer[0].weight))

    def test_parser_exposes_unbalanced_ot_controls(self):
        args = make_parser().parse_args(["--dataset", "tiny"])
        self.assertEqual(args.tau_source, 0.5)
        self.assertEqual(args.dustbin_cost, 1.0)
        self.assertEqual(args.ot_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
