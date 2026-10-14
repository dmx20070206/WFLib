"""Unsupervised clustering followed by label-informed diagnostics."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize


def oracle(y, clusters, k, classes):
    matrix = np.bincount(clusters * classes + y, minlength=k * classes).reshape(k, classes)
    rows, cols = linear_sum_assignment(-matrix)
    mapping = np.full(k, -1, dtype=int)
    mapping[rows] = cols
    return float(matrix[rows, cols].sum() / len(y)), mapping, matrix


def variance(z, y):
    """Population mean squared distance, not the unbiased sample estimator."""
    total = float(np.square(z - z.mean(axis=0)).sum(axis=1).mean())
    records = []
    within = 0.
    for label in np.unique(y):
        points = z[y == label]
        v = float(np.square(points - points.mean(axis=0)).sum(axis=1).mean())
        within += len(points) * v / len(y)
        records.append(dict(label=int(label), n=len(points), within_variance=v))
    return dict(total_variance=total, within_variance=within,
                between_variance=max(0., total-within),
                within_total_ratio=within/total if total > 0 else None,
                per_class=records)


def prepare(z, dimensions, seed):
    if z.ndim != 2 or len(z) < 3 or not np.isfinite(z).all():
        raise ValueError("Need finite [N,D] features with at least 3 samples")
    normalized = normalize(z).astype(np.float32)
    effective = min(dimensions, len(z)-1, z.shape[1]) if dimensions else 0
    info = dict(input_dimensions=z.shape[1], normalization="L2 before PCA", pca_dimensions=0)
    if effective and effective < z.shape[1]:
        pca = PCA(n_components=effective, svd_solver="randomized", random_state=seed)
        projected = pca.fit_transform(normalized)
        explained = float(pca.explained_variance_ratio_.sum())
        info.update(pca_dimensions=effective, explained_variance_ratio=explained if np.isfinite(explained) else None)
    else:
        projected = normalized
    return normalized, projected, info


def cluster(z, y, classes, ks, seeds, silhouette_samples=2000, method="minibatch", n_init=5):
    runs, assignments, matrices = [], {}, {}
    # Same label-independent subset for every run, making silhouettes comparable.
    indices = np.random.default_rng(seeds[0]).choice(len(z), min(len(z), silhouette_samples), replace=False)
    for k in ks:
        for seed in seeds:
            kwargs = dict(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
            estimator = (MiniBatchKMeans(batch_size=1024, **kwargs) if method == "minibatch" else KMeans(**kwargs))
            pred = estimator.fit_predict(z)
            accuracy, mapping, matrix = oracle(y, pred, k, classes)
            sampled = pred[indices]
            distinct = len(np.unique(sampled))
            silhouette = (float(silhouette_score(z[indices], sampled))
                          if 1 < distinct < len(indices) else None)
            key = f"k{k}_seed{seed}"
            assignments[key], matrices[key] = pred, matrix
            runs.append(dict(k=k, seed=seed, effective_k=len(np.unique(pred)),
                             NMI=float(normalized_mutual_info_score(y, pred)),
                             ARI=float(adjusted_rand_score(y, pred)),
                             silhouette=silhouette, oracle_accuracy=accuracy,
                             inertia=float(estimator.inertia_),
                             cluster_sizes=np.bincount(pred, minlength=k).tolist(),
                             mapping_cluster_to_class=mapping.tolist()))
            print(f"K={k} seed={seed}: NMI={runs[-1]['NMI']:.4f} ARI={runs[-1]['ARI']:.4f}", flush=True)
    summaries = []
    for k in ks:
        subset = [r for r in runs if r['k'] == k]
        result = dict(k=k)
        for metric in ('NMI', 'ARI', 'silhouette', 'oracle_accuracy'):
            values = [r[metric] for r in subset if r[metric] is not None]
            result[metric] = dict(mean=float(np.mean(values)) if values else None,
                                  std=float(np.std(values)) if values else None, n_valid=len(values))
        pairs = [float(adjusted_rand_score(assignments[f'k{k}_seed{a}'], assignments[f'k{k}_seed{b}']))
                 for a, b in itertools.combinations(seeds, 2)]
        result['stability'] = dict(pairwise_ARI=pairs, mean=float(np.mean(pairs)), std=float(np.std(pairs)))
        summaries.append(result)
    return runs, summaries, assignments, matrices, indices
