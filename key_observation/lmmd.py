import numpy as np


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rbf_kernel(x, y, gamma):
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True)
    dist2 = np.maximum(x2 - 2.0 * (x @ y.T) + y2.T, 0.0)
    return np.exp(-gamma * dist2)


def _median_heuristic_gamma(x, y, eps=1e-12):
    z = np.concatenate([x, y], axis=0)
    z2 = np.sum(z * z, axis=1, keepdims=True)
    dist2 = np.maximum(z2 - 2.0 * (z @ z.T) + z2.T, 0.0)
    upper = dist2[np.triu_indices(dist2.shape[0], k=1)]
    upper = upper[upper > 0]
    if upper.size == 0:
        return 1.0
    return 1.0 / max(2.0 * np.median(upper), eps)


def _mmd2_rbf(x, y, gamma=None):
    if gamma is None:
        gamma = _median_heuristic_gamma(x, y)
    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)
    return float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))


def compute_lmmd(data, label, source_domain=0, target_domain=1, gamma=None):
    x = _to_numpy(data).reshape(_to_numpy(data).shape[0], -1)

    if isinstance(label, (tuple, list)):
        class_label = _to_numpy(label[0]).reshape(-1)
        domain_label = _to_numpy(label[1]).reshape(-1)
    else:
        label_arr = _to_numpy(label)
        class_label = label_arr[:, 0].reshape(-1)
        domain_label = label_arr[:, 1].reshape(-1)

    src_mask = domain_label == source_domain
    tgt_mask = domain_label == target_domain
    shared_classes = np.intersect1d(np.unique(class_label[src_mask]), np.unique(class_label[tgt_mask]))

    lmmd_values = []
    for c in shared_classes:
        x_s = x[(class_label == c) & src_mask]
        x_t = x[(class_label == c) & tgt_mask]
        lmmd_values.append(_mmd2_rbf(x_s, x_t, gamma=gamma))

    return np.asarray(lmmd_values, dtype=np.float64)
