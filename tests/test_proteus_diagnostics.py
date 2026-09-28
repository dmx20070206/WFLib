"""Run with python -m unittest discover -s tests -p test_proteus_diagnostics.py."""
import contextlib
import io
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from exp.log import Stage2Metrics, compute_prototype_geometry
from exp.proteus import FeatureTranslator, freeze, train_stage3, translator_loss


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        embedding = self.backbone(x).tanh()
        return self.head(embedding), embedding

    def logits_from_embedding(self, x):
        return self.head(x)


class DiagnosticsTest(unittest.TestCase):
    def test_geometry_and_source_weighting(self):
        prototypes = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]])
        result = compute_prototype_geometry(prototypes, torch.tensor([0, 0, 1, 2]), 1., 2.)
        self.assertAlmostEqual(result['prototype_pair_fraction_below_alpha'], 1 / 3)
        self.assertAlmostEqual(result['ideal_alignment_contrast_source_weighted'], .375)
        self.assertAlmostEqual(result['ideal_alignment_total_source_weighted'], .75)
        single = compute_prototype_geometry(prototypes[:1], torch.tensor([0]))
        self.assertEqual(single['ideal_alignment_contrast_source_weighted'], 0)

    def test_gradient_probe_preserves_backward(self):
        torch.manual_seed(3)
        translator = FeatureTranslator(2, 4)
        prototypes = torch.eye(2)
        labels = torch.tensor([0, 1])
        output = translator(prototypes)
        total, align, contrast = translator_loss(output, prototypes, labels, alpha=2.)
        expected = torch.autograd.grad(total, tuple(translator.parameters()), retain_graph=True)
        metrics = Stage2Metrics()
        metrics.measure_gradient_conflict(align, contrast, translator, 1.)
        self.assertTrue(all(p.grad is None for p in translator.parameters()))
        total.backward()
        for p, g in zip(translator.parameters(), expected):
            torch.testing.assert_close(p.grad, g)
        self.assertTrue(-1.00001 <= metrics.gradient_metrics['gradient_cosine'] <= 1.00001)

    def test_stage3_updates_encoder_but_not_translator(self):
        torch.manual_seed(7)
        encoder = TinyEncoder()
        translator = FeatureTranslator(4, 8)
        loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.arange(8) % 2), batch_size=4)
        freeze(encoder)  # State left behind by Stage 2.
        before = {k: p.detach().clone() for k, p in encoder.named_parameters()}
        translator_before = {k: p.detach().clone() for k, p in translator.named_parameters()}
        with contextlib.redirect_stdout(io.StringIO()):
            train_stage3(encoder, translator, loader, loader, 'cpu', epochs=1, lr=.01)
        self.assertTrue(all(p.requires_grad for p in encoder.parameters()))
        for name, p in encoder.named_parameters():
            self.assertFalse(torch.equal(p, before[name]), name)
        for name, p in translator.named_parameters():
            self.assertFalse(p.requires_grad)
            torch.testing.assert_close(p, translator_before[name])

    def test_opposing_gradients_are_reported(self):
        module = nn.Linear(1, 1, bias=False)
        align = module.weight.sum()
        contrast = -module.weight.sum()
        metrics = Stage2Metrics()
        metrics.measure_gradient_conflict(align, contrast, module, 1.)
        self.assertAlmostEqual(metrics.gradient_metrics['gradient_cosine'], -1.)
        self.assertAlmostEqual(metrics.gradient_metrics['gradient_cancellation_ratio'], 0.)
        self.assertIsNone(module.weight.grad)


if __name__ == '__main__':
    unittest.main()
