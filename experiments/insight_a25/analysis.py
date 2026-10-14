"""Matched, disjoint probe splits and supervised frozen-feature evaluation."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


def protocol(label_sets, train_cap=100, val_cap=40, test_cap=100):
    supports = [{int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))} for y in label_sets]
    common = set.intersection(*(set(s) for s in supports))
    classes = sorted(c for c in common if min(s[c] for s in supports) >= 5)
    if len(classes) < 2:
        raise ValueError('Need >=2 shared classes with >=5 samples each in every period')
    minimum = min(s[c] for s in supports for c in classes)
    train = min(train_cap, max(2, int(minimum * .5)))
    val = min(val_cap, max(1, int(minimum * .2)))
    test = min(test_cap, minimum-train-val)
    return dict(classes=classes, min_support=minimum, train_per_class=train,
                val_per_class=val, test_per_class=test, supports=supports,
                excluded_classes=[sorted(set(s)-set(classes)) for s in supports])


def split(y, design, seed):
    rng = np.random.default_rng(seed)
    buckets = [[], [], []]
    counts = [design[k] for k in ('train_per_class', 'val_per_class', 'test_per_class')]
    for c in design['classes']:
        indices = rng.permutation(np.flatnonzero(y == c))
        if len(indices) < sum(counts):
            raise ValueError('Insufficient class support for matched split')
        a, b, d = counts
        for bucket, chunk in zip(buckets, (indices[:a], indices[a:a+b], indices[a+b:a+b+d])):
            bucket.extend(chunk)
    return tuple(np.asarray(bucket, dtype=int) for bucket in buckets)


def linear(z, y, c, max_iter):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver='lbfgs', max_iter=max_iter))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(z, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def fit_probes(z, y, train, val, fixed_c, grid, neighbors, max_iter):
    candidates, search = {}, []
    for c in sorted(set([fixed_c]+grid)):
        model, converged = linear(z[train], y[train], c, max_iter)
        score = float(model.score(z[val], y[val]))
        candidates[c] = model
        search.append(dict(C=c, val_accuracy=score, converged=converged))
    # Ties prefer stronger regularization. No refit on validation/test data.
    best_c = max(sorted(set(grid)), key=lambda c: next(r['val_accuracy'] for r in search if r['C'] == c))
    knn_search, knn_models = [], {}
    for k in sorted(set(neighbors)):
        if k > len(train):
            continue
        model = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean'))
        model.fit(z[train], y[train])
        knn_search.append(dict(k=k, val_accuracy=float(model.score(z[val], y[val]))))
        knn_models[k] = model
    if not knn_search:
        raise ValueError('No kNN k is <= training sample count')
    best_k = max(knn_search, key=lambda row: row['val_accuracy'])['k']
    return dict(linear_fixed=candidates[fixed_c], linear_tuned=candidates[best_c], knn=knn_models[best_k]), dict(
        linear_search=search, selected_C=best_c, knn_search=knn_search, selected_k=best_k)


def evaluate(y, predictions, seed, repeats=1000):
    source = predictions['source_head'] == y
    rng = np.random.default_rng(seed)
    result = {}
    for name, pred in predictions.items():
        correct = pred == y
        gain = correct.astype(int)-source.astype(int)
        # Paired bootstrap of test accuracy differences; fixed fitted models.
        counts = np.bincount(gain+1, minlength=3)
        draws = rng.multinomial(len(y), counts/len(y), size=repeats)
        intervals = np.quantile((draws[:, 2]-draws[:, 0])/len(y), [.025, .975]).tolist()
        result[name] = dict(accuracy=float(correct.mean()), G_probe=float(gain.mean()),
                            gain_ci95=intervals)
    return result


def export_linear(model, path):
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    np.savez_compressed(path, mean=scaler.mean_, scale=scaler.scale_,
                        coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_)
