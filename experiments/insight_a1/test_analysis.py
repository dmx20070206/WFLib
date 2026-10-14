import itertools
import unittest

import numpy as np

from experiments.insight_a1.analysis import analyze, confusion, solve


class AnalysisTests(unittest.TestCase):
    def test_known_swap_and_heldout(self):
        y = np.tile(np.arange(3), 100)
        result, _, _ = analyze(y, (y+1) % 3, 3, repeats=10)
        self.assertEqual([result[k] for k in ('A_raw', 'A_perm', 'G_perm')], [0, 1, 1])
        self.assertEqual(result['mapping_pred_to_true'], [2, 0, 1])
        self.assertEqual(result['heldout']['A_mapped'], 1)

    def test_collapsed_predictions_cannot_be_recovered(self):
        y = np.tile(np.arange(4), 20)
        values, _ = solve(confusion(y, np.zeros_like(y), 4))
        np.testing.assert_allclose(values, [.25, .25, 0])

    def test_matches_exhaustive_search_and_missing_classes(self):
        rng = np.random.default_rng(4)
        for _ in range(10):
            matrix = rng.integers(0, 10, (4, 4))
            matrix[3] = 0
            values, mapping = solve(matrix)
            best = max(sum(matrix[i, p[i]] for i in range(4))
                       for p in itertools.permutations(range(4)))
            self.assertAlmostEqual(values[1], best / matrix.sum())
            self.assertGreaterEqual(values[2], 0)
            self.assertEqual(len(set(mapping)), 4)

    def test_reproducibility_and_singleton(self):
        a, _, draws_a = analyze(np.array([2]), np.array([0]), 4, repeats=10)
        b, _, draws_b = analyze(np.array([2]), np.array([0]), 4, repeats=10)
        self.assertEqual(a, b)
        self.assertIsNone(a['heldout'])
        for key in draws_a:
            np.testing.assert_array_equal(draws_a[key], draws_b[key])

    def test_float_integer_labels_from_project(self):
        y = np.array([0., 1., 2.] * 10)
        report, _, _ = analyze(y, (y + 1) % 3, 3, repeats=3)
        self.assertEqual(report["A_perm"], 1)
        for invalid in ([np.nan], [np.inf], [-1.], [1.5]):
            with self.assertRaises(ValueError):
                confusion(invalid, [0], 3)

    def test_reject_invalid_labels(self):
        for y, pred in [([], []), ([.5], [0]), ([0], [3]), ([0, 1], [0]), ([[0]], [0])]:
            with self.assertRaises(ValueError):
                confusion(y, pred, 3)


if __name__ == '__main__':
    unittest.main()
