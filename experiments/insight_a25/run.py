"""A2.5: source head versus supervised probes on frozen source features."""
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
from experiments.insight_a25.analysis import protocol, split, fit_probes, evaluate, export_linear


def extract(args, path, model):
    import torch
    X, y = load_inputs(args, path)
    features, predictions, captured = [], [], []
    handle = model.get_submodule(args.feature_layer).register_forward_hook(
        lambda module, inputs, output: captured.append(output))
    try:
        with torch.inference_mode():
            for start in range(0, len(y), args.batch_size):
                captured.clear()
                output = model(X[start:start+args.batch_size].to(args.device))
                logits = output[0] if isinstance(output, (tuple, list)) else output
                if len(captured) != 1:
                    raise ValueError('Feature layer must execute exactly once')
                z = captured[0]
                if z.ndim > 2:
                    z = z.flatten(2).mean(dim=2)
                if z.ndim != 2 or not torch.isfinite(z).all() or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite or invalid features/scores')
                features.append(z.cpu().numpy())
                predictions.append(logits.argmax(1).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(features), y, np.concatenate(predictions)


def plot(reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('source_head', 'linear_fixed', 'linear_tuned', 'knn', 'day0_linear_transfer')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name in names:
        for ax, metric in zip(axes, ('accuracy', 'G_probe')):
            means, deviations = [], []
            for report in reports:
                values = [r['metrics'][name][metric] for r in report['runs'] if name in r['metrics']]
                means.append(np.mean(values) if values else np.nan)
                deviations.append(np.std(values) if values else np.nan)
            ax.errorbar(range(len(reports)), means, yerr=deviations, marker='o', label=name)
    axes[0].set(ylabel='Held-out accuracy', ylim=(0, 1.02))
    axes[1].set(ylabel='Gain over source head', xlabel='Period (categorical spacing)')
    axes[1].axhline(0, color='gray', linewidth=.8)
    axes[1].set_xticks(range(len(reports)))
    axes[1].set_xticklabels([r['period'] for r in reports])
    for ax in axes:
        ax.legend(fontsize='small')
        ax.grid(alpha=.2)
    fig.suptitle('A2.5: frozen features, supervised probes; mean ± split-seed SD')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output/('probe_curves.'+ext), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--datasets', '--dataset', nargs='+')
    p.add_argument('--period', nargs=2, action='append', metavar=('NAME', 'PATH'))
    p.add_argument('--model')
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--feature-layer', default='features.18')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026])
    p.add_argument('--train-per-class', type=int, default=100, help='Cap; matched across all periods')
    p.add_argument('--val-per-class', type=int, default=40)
    p.add_argument('--test-per-class', type=int, default=100)
    p.add_argument('--fixed-c', type=float, default=1.)
    p.add_argument('--c-grid', type=float, nargs='+', default=[.01, .1, 1., 10., 100.])
    p.add_argument('--neighbors', type=int, nargs='+', default=[1, 3, 5, 11, 21])
    p.add_argument('--max-iter', type=int, default=2000)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    if (min(args.batch_size, args.threads, args.train_per_class, args.val_per_class,
            args.test_per_class, args.max_iter, args.bootstrap, *args.neighbors) < 1
            or args.num_workers < 0 or min(args.seeds) < 0
            or not all(np.isfinite(c) and c > 0 for c in [args.fixed_c]+args.c_grid)):
        p.error('Counts and C must be positive; seeds/workers nonnegative')
    args.seeds = sorted(set(args.seeds))
    args.seed = args.seeds[0]
    args.input_mode = 'model'
    args.model_factory = args.loader = args.checkpoint_key = None
    args.model_kwargs = args.loader_kwargs = '{}'
    args.num_classes = args.feature = args.seq_len = None
    resolve_inputs(args, p)
    if args.model not in ('DMX', 'RF'):
        p.error('Automatic extraction supports repository DMX/RF')
    if len({name for name, _ in args.period}) != len(args.period):
        p.error('Period names must be unique')
    references = [i for i, (name, path) in enumerate(args.period)
                  if name.lower() in ('day0', 'd0', 'valid', 'tam_valid')]
    reference = references[0] if references else None
    if reference is not None and Path(args.period[reference][1]).stem in ('train', 'tam_train'):
        p.error('Day0 must use held-out source data (valid.npz), not source training data')
    if reference is None:
        print('No Day0 identified: target probes will run without Day0 transfer control.', flush=True)
    import torch
    torch.set_num_threads(args.threads)
    model = build_model(args)
    model.requires_grad_(False)
    args.resolved_inputs = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent/'results'/timestamp
    elif args.output_dir.exists():
        args.output_dir /= timestamp
    args.output_dir.mkdir(parents=True)
    print(f'Output: {args.output_dir.resolve()}', flush=True)
    save_config(args)
    label_sets = []
    # Extract once; disk cache avoids holding every period's features in RAM.
    for i, (name, path) in enumerate(args.period):
        dest = args.output_dir/f'period_{i:03d}'
        dest.mkdir()
        z, y, raw = extract(args, path, model)
        np.savez_compressed(dest/'features.npz', features=z, y_true=y, source_pred=raw)
        label_sets.append(y)
        save_config(args)
        print(f'{name}: extracted {z.shape}', flush=True)
    design = protocol(label_sets, args.train_per_class, args.val_per_class, args.test_per_class)
    (args.output_dir/'protocol.json').write_text(json.dumps(design, indent=2))
    print('Matched per-class train/val/test:', design['train_per_class'], design['val_per_class'], design['test_per_class'], flush=True)
    reports = [dict(period=name, runs=[]) for name, _ in args.period]
    rows = []
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            reference_model = None
            order = ([reference] if reference is not None else []) + [i for i in range(len(reports)) if i != reference]
            for i in order:
                dest = args.output_dir/f'period_{i:03d}'
                with np.load(dest/'features.npz') as data:
                    z, y, raw = data['features'], data['y_true'], data['source_pred']
                train, val, test = split(y, design, seed)
                fitted, tuning = fit_probes(z, y, train, val, args.fixed_c, args.c_grid, args.neighbors, args.max_iter)
                if i == reference:
                    reference_model = fitted['linear_fixed']
                predictions = dict(source_head=raw[test])
                predictions.update({name: probe.predict(z[test]) for name, probe in fitted.items()})
                if reference_model is not None:
                    predictions['day0_linear_transfer'] = reference_model.predict(z[test])
                metrics = evaluate(y[test], predictions, seed, args.bootstrap)
                run = dict(seed=seed, n_train=len(train), n_val=len(val), n_test=len(test),
                           tuning=tuning, metrics=metrics)
                reports[i]['runs'].append(run)
                np.savez_compressed(dest/f'seed_{seed}_predictions.npz', train_indices=train,
                                    val_indices=val, test_indices=test, y_test=y[test], **predictions)
                for name in ('linear_fixed', 'linear_tuned'):
                    export_linear(fitted[name], dest/f'seed_{seed}_{name}.npz')
                for name, m in metrics.items():
                    rows.append(dict(period=reports[i]['period'], seed=seed, classifier=name,
                                     accuracy=m['accuracy'], G_probe=m['G_probe'],
                                     gain_ci95_low=m['gain_ci95'][0], gain_ci95_high=m['gain_ci95'][1]))
                if any(not r['converged'] for r in tuning['linear_search']):
                    print('Warning: linear candidate did not converge; see report and increase --max-iter.', flush=True)
                print(reports[i]['period'], seed, {k: round(v['accuracy'], 4) for k, v in metrics.items()}, flush=True)
                (dest/'report.json').write_text(json.dumps(reports[i], indent=2))
                (args.output_dir/'results.json').write_text(json.dumps(reports, indent=2))
                with (args.output_dir/'summary.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    plot(reports, args.output_dir)


if __name__ == '__main__':
    main()
