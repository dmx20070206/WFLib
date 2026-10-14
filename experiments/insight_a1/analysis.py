"""Offline one-to-one class identity diagnostics (no model training)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


METRICS = ("A_raw", "A_perm", "G_perm")


def labels(values, classes):
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf":
        raise ValueError("Labels must be a nonempty 1D numeric array of integer class IDs")
    if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError("Labels must contain finite integer class IDs")
    if classes < 1 or np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Labels must be in [0, num_classes)")
    return values.astype(np.int64)


def confusion(y, pred, classes):
    y, pred = labels(y, classes), labels(pred, classes)
    if y.shape != pred.shape:
        raise ValueError("Truth and predictions must have identical shapes")
    # Rows: predicted source identity; columns: true target identity.
    return np.bincount(pred * classes + y, minlength=classes ** 2).reshape(classes, classes)


def solve(matrix):
    rows, cols = linear_sum_assignment(-matrix)
    mapping = np.empty(matrix.shape[0], dtype=np.int64)
    mapping[rows] = cols
    raw = float(np.trace(matrix) / matrix.sum())
    perm = float(matrix[rows, cols].sum() / matrix.sum())
    return np.array([raw, perm, perm - raw]), mapping


def summary(samples, confidence):
    alpha = (1 - confidence) / 2
    return {name: {"mean": float(samples[:, i].mean()),
                   "interval": np.quantile(samples[:, i], [alpha, 1-alpha]).tolist()}
            for i, name in enumerate(METRICS)}


def analyze(y, pred, classes, repeats=1000, confidence=.95, seed=2024, holdout=.5):
    matrix = confusion(y, pred, classes)
    y, pred = labels(y, classes), labels(pred, classes)
    if repeats < 1 or not 0 < confidence < 1 or not 0 < holdout < 1:
        raise ValueError("Require repeats >= 1 and confidence/holdout in (0, 1)")
    rng = np.random.default_rng(seed)
    observed, mapping = solve(matrix)
    draws = {key: [] for key in ("bootstrap", "random_prediction", "shuffled_labels")}
    random_mapping = []
    for _ in range(repeats):
        # Multinomial counts equal paired IID resampling of (truth, prediction).
        boot = rng.multinomial(len(y), matrix.ravel() / len(y)).reshape(matrix.shape)
        draws["bootstrap"].append(solve(boot)[0])
        random_pred = rng.integers(classes, size=len(y))
        draws["random_prediction"].append(solve(confusion(y, random_pred, classes))[0])
        draws["shuffled_labels"].append(solve(confusion(rng.permutation(y), pred, classes))[0])
        random_mapping.append(float(np.mean(rng.permutation(classes)[pred] == y)))
    draws = {k: np.asarray(v) for k, v in draws.items()}
    report = dict(zip(METRICS, observed.tolist()))
    report.update(n=len(y), num_classes=classes, mapping_pred_to_true=mapping.tolist(),
                  true_support=matrix.sum(axis=0).tolist(), predicted_support=matrix.sum(axis=1).tolist(),
                  confidence=confidence,
                  distributions={k: summary(v, confidence) for k, v in draws.items()})
    alpha = (1-confidence)/2
    report["random_mapping_accuracy"] = {
        "mean": float(np.mean(random_mapping)),
        "interval": np.quantile(random_mapping, [alpha, 1-alpha]).tolist()}
    # Null test of association after optimizing identity, not of a causal drift mechanism.
    report["shuffled_labels_A_perm_pvalue"] = float(
        (1 + np.sum(draws["shuffled_labels"][:, 1] >= observed[1])) / (repeats + 1))
    if len(y) >= 2:
        order = rng.permutation(len(y))
        n_test = min(len(y)-1, max(1, int(round(len(y)*holdout))))
        test, fit = order[:n_test], order[n_test:]
        _, fitted = solve(confusion(y[fit], pred[fit], classes))
        raw = float(np.mean(pred[test] == y[test]))
        mapped = float(np.mean(fitted[pred[test]] == y[test]))
        report["heldout"] = {"n_fit": len(fit), "n_test": len(test), "A_raw": raw,
                             "A_mapped": mapped, "G_mapped": mapped-raw,
                             "mapping_pred_to_true": fitted.tolist()}
    else:
        report["heldout"] = None
    draws["random_mapping_accuracy"] = np.asarray(random_mapping)
    return report, matrix, draws
