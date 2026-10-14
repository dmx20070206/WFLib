"""Checks for repository conventions without requiring model inference."""
import argparse
from pathlib import Path
import tempfile
import unittest

from experiments.insight_a1.run import resolve_inputs


class InputTests(unittest.TestCase):
    def test_directory_deduplicates_tam_and_sorts_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('valid', 'train', 'tam_valid', 'day90', 'tam_day90', 'day14', 'tam_day270'):
                (root / (name + '.npz')).touch()
            args = argparse.Namespace(
                datasets=[tmp], period=None, input_mode='model', model=None,
                model_factory=None, checkpoint=Path('checkpoints/TemporalDrift/DMX/max_f1.pth'),
                feature=None, loader=None, device='cpu', seq_len=None,
            )
            resolve_inputs(args, argparse.ArgumentParser())
            self.assertEqual(args.model, 'DMX')
            self.assertEqual((args.feature, args.seq_len), ('TAM', 1800))
            self.assertEqual([name for name, _ in args.period], ['day0', 'day14', 'day90', 'day270'])
            self.assertEqual(Path(args.period[0][1]).name, 'valid.npz')
            self.assertEqual(Path(args.period[2][1]).name, 'day90.npz')


if __name__ == '__main__':
    unittest.main()
