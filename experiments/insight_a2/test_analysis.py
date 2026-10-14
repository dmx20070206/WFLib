import unittest
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.insight_a2.analysis import cluster, oracle, prepare, variance


class StructureTests(unittest.TestCase):
    def test_rectangular_oracle_does_not_merge_clusters(self):
        y = np.array([0, 0, 1, 1])
        acc, mapping, _ = oracle(y, np.array([0, 1, 2, 3]), 4, 2)
        self.assertEqual(acc, .5)
        self.assertEqual(sum(mapping == -1), 2)
        acc, _, _ = oracle(y, np.zeros(4, dtype=int), 1, 2)
        self.assertEqual(acc, .5)

    def test_variance_decomposition(self):
        result = variance(np.array([[0.], [2.], [8.], [10.]]), np.array([0, 0, 1, 1]))
        self.assertEqual(result['within_variance'], 1.)
        self.assertEqual(result['between_variance'], 16.)
        self.assertAlmostEqual(result['within_total_ratio'], 1/17)

    def test_separated_clusters_and_reproducibility(self):
        rng = np.random.default_rng(1)
        z = np.concatenate([rng.normal(c, .01, (15, 2)) for c in (-5, 5)])
        y = np.repeat([0, 1], 15)
        with threadpool_limits(limits=1):
            a = cluster(z, y, 2, [2], [1, 2], 30, 'kmeans', 2)
            b = cluster(z, y, 2, [2], [1, 2], 30, 'kmeans', 2)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1][0]['stability']['mean'], 1)
        for run in a[0]:
            self.assertEqual(run['NMI'], 1)
            self.assertEqual(run['ARI'], 1)
            self.assertEqual(run['oracle_accuracy'], 1)
            self.assertGreater(run['silhouette'], .9)

    def test_transform_does_not_need_labels(self):
        z = np.random.default_rng(5).normal(size=(20, 8))
        normalized, projected, info = prepare(z, 3, 1)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1, atol=1e-6)
        self.assertEqual(projected.shape, (20, 3))
        self.assertEqual(info['pca_dimensions'], 3)
        with self.assertRaises(ValueError):
            prepare(np.array([[np.nan]] * 3), 1, 1)


if __name__ == '__main__':
    unittest.main()
