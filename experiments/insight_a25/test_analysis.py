import tempfile
from pathlib import Path
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.insight_a25.analysis import protocol, split, fit_probes, evaluate, export_linear


class ProbeTests(unittest.TestCase):
    def test_matched_disjoint_splits(self):
        a = np.repeat([0, 1, 2], [20, 30, 4])
        b = np.repeat([0, 1, 3], [40, 50, 20])
        design = protocol([a, b])
        self.assertEqual(design['classes'], [0, 1])
        for y in (a, b):
            parts = split(y, design, 1)
            self.assertEqual([len(v) for v in parts], [20, 8, 12])
            self.assertEqual(len(np.unique(np.concatenate(parts))), sum(map(len, parts)))
            for part, again in zip(parts, split(y, design, 1)):
                np.testing.assert_array_equal(part, again)
                self.assertEqual(set(y[part]), {0, 1})

    def test_no_test_label_or_feature_leakage(self):
        y = np.repeat([0, 1, 2], 30)
        z = np.eye(3)[y] + np.random.default_rng(1).normal(0, .01, (90, 3))
        train, val, test = split(y, protocol([y]), 2)
        with threadpool_limits(limits=1):
            models, report = fit_probes(z, y, train, val, 1., [.1, 1.], [1, 3], 500)
            changed_z, changed_y = z.copy(), y.copy()
            changed_z[test], changed_y[test] = 10000, 0
            other, other_report = fit_probes(changed_z, changed_y, train, val, 1., [.1, 1.], [1, 3], 500)
        self.assertEqual(report, other_report)
        for name in models:
            np.testing.assert_array_equal(models[name].predict(z[test]), other[name].predict(z[test]))
        scaler = models['linear_fixed'].steps[0][1]
        np.testing.assert_allclose(scaler.mean_, z[train].mean(0))
        self.assertEqual(models['linear_fixed'].score(z[test], y[test]), 1.)
        with tempfile.TemporaryDirectory() as tmp:
            export_linear(models['linear_fixed'], Path(tmp)/'model.npz')
            with np.load(Path(tmp)/'model.npz') as saved:
                pred = saved['classes'][np.argmax(((z[test]-saved['mean'])/saved['scale']) @ saved['coef'].T + saved['intercept'], axis=1)]
                np.testing.assert_array_equal(pred, y[test])

    def test_paired_gain(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        result = evaluate(y, dict(source_head=(y+1)%3, linear_fixed=y), 1, 50)
        self.assertEqual(result['linear_fixed']['G_probe'], 1.)
        self.assertEqual(result['linear_fixed']['gain_ci95'], [1., 1.])
        self.assertEqual(result['source_head']['gain_ci95'], [0., 0.])

    def test_insufficient_common_support(self):
        with self.assertRaises(ValueError):
            protocol([np.array([0, 1, 1]), np.array([2, 2, 2])])


if __name__ == '__main__':
    unittest.main()
