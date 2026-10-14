"""A2: frozen source features, multiple cluster counts and random seeds."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.insight_a1.run import build_model, load_inputs, resolve_inputs, save_config
from experiments.insight_a2.analysis import cluster, prepare, variance


def extract(args, path, model):
    import torch
    X, y = load_inputs(args, path)
    indices = np.arange(len(y))
    if args.max_samples and len(y) > args.max_samples:
        indices = np.sort(np.random.default_rng(args.seed).choice(len(y), args.max_samples, replace=False))
        X, y = X[indices], y[indices]
    values, capture = [], []
    # The model's returned "features" are after the class projection. Hook the
    # preceding 512-channel layer instead; pool time without target labels.
    layer = model.get_submodule(args.feature_layer)
    handle = layer.register_forward_hook(lambda module, inputs, output: capture.append(output))
    try:
        with torch.inference_mode():
            for start in range(0, len(y), args.batch_size):
                capture.clear()
                model(X[start:start+args.batch_size].to(args.device))
                if len(capture) != 1 or not isinstance(capture[0], torch.Tensor):
                    raise ValueError("Feature layer must run once and return a Tensor")
                z = capture[0]
                if z.ndim > 2:
                    z = z.flatten(2).mean(dim=2)
                if z.ndim != 2:
                    raise ValueError("Feature layer must return [N,D,...]")
                values.append(z.cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(values), y, indices


def plots(reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = ('NMI', 'ARI', 'silhouette', 'oracle_accuracy')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.flat, metrics):
        for report in reports:
            rows = report['summary']
            rows = [r for r in rows if r[metric]['mean'] is not None]
            ax.errorbar([r['k'] for r in rows], [r[metric]['mean'] for r in rows],
                        yerr=[r[metric]['std'] for r in rows], marker='o', label=report['period'])
        ax.set(xlabel='Requested K', ylabel=metric)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('A2: seed means ± SD (not confidence intervals)')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / ('metrics_vs_k.'+extension), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    all_ks = sorted({row['k'] for report in reports for row in report['summary']})
    for ax, metric in zip(axes.flat, metrics):
        for k in all_ks:
            values, deviations = [], []
            for report in reports:
                row = next((r for r in report['summary'] if r['k'] == k), None)
                value = row[metric]['mean'] if row else None
                values.append(value if value is not None else np.nan)
                deviations.append(row[metric]['std'] if value is not None else np.nan)
            ax.errorbar(range(len(reports)), values, yerr=deviations, marker='o', label=f'K={k}')
        ax.set_xticks(range(len(reports)))
        ax.set_xticklabels([r['period'] for r in reports])
        ax.set_ylabel(metric)
        ax.grid(alpha=.2)
        ax.legend(fontsize='small')
    fig.suptitle('A2 across periods (categorical spacing): seed means ± SD')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / ('metrics_vs_period.'+extension), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in reports:
        axes[0].plot([v['k'] for v in r['summary']], [v['stability']['mean'] for v in r['summary']], 'o-', label=r['period'])
    axes[0].set(xlabel='Requested K', ylabel='Between-seed partition ARI')
    axes[0].legend()
    axes[1].bar([r['period'] for r in reports],
                [r['variance_normalized']['within_total_ratio']
                 if r['variance_normalized']['within_total_ratio'] is not None else np.nan for r in reports])
    axes[1].set(ylabel='Within / total variance (L2 features)')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / ('structure.'+extension), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--datasets', '--dataset', nargs='+')
    p.add_argument('--period', nargs=2, action='append', metavar=('NAME', 'PATH'))
    p.add_argument('--model')
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--feature-layer', default='features.18')
    p.add_argument('--ks', type=int, nargs='+', help='Default: 0.5, 0.75, 1, 1.25, 1.5, 2 times source C')
    p.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026])
    p.add_argument('--seed', type=int, default=2024, help='Sampling and PCA seed')
    p.add_argument('--method', choices=['minibatch', 'kmeans'], default='minibatch')
    p.add_argument('--n-init', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=10000, help='Uniform label-free sample; 0 means all')
    p.add_argument('--pca-dim', type=int, default=64, help='Per-period unsupervised PCA; 0 disables')
    p.add_argument('--silhouette-samples', type=int, default=2000)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if (min(args.batch_size, args.n_init, args.threads) < 1 or args.num_workers < 0
            or args.max_samples < 0 or args.pca_dim < 0 or args.silhouette_samples < 3
            or args.seed < 0 or len(set(args.seeds)) < 2 or min(args.seeds) < 0
            or (args.ks and min(args.ks) < 2)):
        p.error('Require positive counts, >=2 distinct nonnegative seeds, K>=2, silhouette sample>=3')
    args.seeds = sorted(set(args.seeds))
    # Share repository model/data conventions with A1.
    args.input_mode = 'model'
    args.model_factory = args.loader = args.checkpoint_key = None
    args.model_kwargs = args.loader_kwargs = '{}'
    args.num_classes = args.feature = args.seq_len = None
    resolve_inputs(args, p)
    if args.model not in ('DMX', 'RF'):
        p.error('Automatic feature extraction supports repository DMX/RF models')
    if len({name for name, _ in args.period}) != len(args.period):
        p.error('Period names must be unique')
    import torch
    torch.set_num_threads(args.threads)
    model = build_model(args)
    model.requires_grad_(False)
    args.resolved_inputs = []
    if args.ks is None:
        args.ks = sorted({max(2, int(round(args.num_classes*r))) for r in (.5, .75, 1., 1.25, 1.5, 2.)})
    else:
        args.ks = sorted(set(args.ks))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / 'results' / timestamp
    elif args.output_dir.exists():
        args.output_dir = args.output_dir / timestamp
    args.output_dir.mkdir(parents=True)
    save_config(args)
    print(f'Output: {args.output_dir.resolve()}', flush=True)
    reports, csv_rows = [], []
    with threadpool_limits(limits=args.threads):
        for index, (name, path) in enumerate(args.period):
            z, y, indices = extract(args, path, model)
            save_config(args)
            normalized, projected, transform = prepare(z, args.pca_dim, args.seed)
            ks = [k for k in args.ks if k < len(y)]
            if not ks:
                raise ValueError(f'{name}: no K < sample count {len(y)}; use smaller --ks')
            runs, summaries, assignments, matrices, silhouette_indices = cluster(
                projected, y, args.num_classes, ks, args.seeds,
                args.silhouette_samples, args.method, args.n_init)
            report = dict(period=name, n=len(y), num_classes=args.num_classes,
                          observed_classes=len(np.unique(y)), transform=transform,
                          skipped_ks=[k for k in args.ks if k not in ks],
                          variance_raw=variance(z, y), variance_normalized=variance(normalized, y),
                          runs=runs, summary=summaries)
            dest = args.output_dir / f'period_{index:03d}'
            dest.mkdir()
            np.savez_compressed(dest / 'features.npz', features=z, clustering_features=projected,
                                y_true=y, source_indices=indices, silhouette_indices=silhouette_indices)
            np.savez_compressed(dest / 'clusters.npz', **assignments)
            np.savez_compressed(dest / 'contingencies.npz', **matrices)
            (dest / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            reports.append(report)
            for r in runs:
                csv_rows.append({k: r[k] for k in ('k', 'seed', 'effective_k', 'NMI', 'ARI', 'silhouette', 'oracle_accuracy')})
                csv_rows[-1]['period'] = name
            (args.output_dir / 'results.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
            with (args.output_dir / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
    plots(reports, args.output_dir)


if __name__ == '__main__':
    main()
