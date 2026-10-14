"""Run A1 on external predictions or a frozen source-domain classifier."""
import argparse
import csv
from datetime import datetime
import re
import importlib
import json
from pathlib import Path
import sys

import numpy as np

# Allow both direct script execution and python -m invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.insight_a1.analysis import METRICS, analyze, labels


def symbol(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def build_model(args):
    import torch
    torch.manual_seed(args.seed)
    if args.model_factory:
        factory = symbol(args.model_factory)
    else:
        from WFlib import models
        factory = getattr(models, args.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.checkpoint_key:
        state = state[args.checkpoint_key]
    elif "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if args.model in ("DMX", "RF") and not args.model_factory:
        inferred = int(state["features.19.weight"].shape[0])
        if args.num_classes is not None and args.num_classes != inferred:
            raise ValueError(f"--num-classes={args.num_classes} disagrees with checkpoint: {inferred}")
        args.num_classes = inferred
    if args.num_classes is None:
        raise ValueError("External model factories require --num-classes")
    model = factory(num_classes=args.num_classes, **json.loads(args.model_kwargs))
    model.load_state_dict(state, strict=True)
    return model.to(args.device).eval()


def load_inputs(args, path):
    import torch
    if args.loader:
        X, y = symbol(args.loader)(path, **json.loads(args.loader_kwargs))
    else:
        from WFlib.tools import data_processor
        source = Path(path)
        # DMX/RF consume TAM, not raw signed timestamps. Reuse project artifacts.
        if args.feature == "TAM" and not source.stem.startswith("tam_"):
            sibling = source.with_name("tam_" + source.name)
            if sibling.exists():
                with np.load(source, allow_pickle=False) as raw, np.load(sibling, allow_pickle=False) as tam:
                    if not np.array_equal(raw["y"], tam["y"]):
                        raise ValueError(f"TAM labels do not match source: {sibling}")
                source = sibling
        with np.load(source, allow_pickle=False) as data:
            X, y = data["X"], labels(data["y"], args.num_classes)
        if args.feature == "TAM":
            if X.ndim == 2:
                print(f"Generating TAM from {source} (first 5000 packets)", flush=True)
                X = data_processor.extract_TAM(data_processor.length_align(X, 5000),
                                               num_workers=max(1, args.num_workers))
            if X.ndim != 3 or X.shape[1] != 2:
                raise ValueError(f"Expected TAM [N, 2, L], got {X.shape}: {source}")
            X = data_processor.length_align(X, args.seq_len)[:, np.newaxis]
        else:
            X, _ = data_processor.load_data(str(source), args.feature, args.seq_len)
        args.resolved_inputs.append({"requested": str(path), "loaded": str(source.resolve())})
        print(f"Loaded {source}: X={tuple(X.shape)}, classes={args.num_classes}", flush=True)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = labels(y, args.num_classes)
    X = torch.as_tensor(X, dtype=torch.float32)
    if len(X) != len(y):
        raise ValueError("Features and labels have different lengths")
    return X, y


def predict(args, path, model):
    import torch
    X, y = load_inputs(args, path)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X),
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    predictions = []
    with torch.inference_mode():
        for (batch,) in loader:
            output = model(batch.to(args.device))
            if isinstance(output, (tuple, list)):
                output = output[args.logits_index]
            if output.shape != (len(batch), args.num_classes) or not torch.isfinite(output).all():
                raise ValueError("Model must return finite [batch, num_classes] class scores")
            predictions.append(output.argmax(dim=1).cpu().numpy())
    return y, np.concatenate(predictions)


def plot(reports, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(reports))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name in ("A_raw", "A_perm"):
        values = [r[name] for r in reports]
        bounds = np.array([r["distributions"]["bootstrap"][name]["interval"] for r in reports])
        axes[0].plot(x, values, "o-", label=name)
        axes[0].fill_between(x, bounds[:, 0], bounds[:, 1], alpha=.15)
    axes[1].plot(x, [r["G_perm"] for r in reports], "o-", label="G_perm (oracle)")
    bounds = np.array([r["distributions"]["bootstrap"]["G_perm"]["interval"] for r in reports])
    axes[1].fill_between(x, bounds[:, 0], bounds[:, 1], alpha=.15)
    for key in ("random_prediction", "shuffled_labels"):
        axes[1].plot(x, [r["distributions"][key]["G_perm"]["mean"] for r in reports],
                     "--", label=key + " mean gain")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Accuracy gain")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["period"] for r in reports])
    axes[1].set_xlabel("Period (input order; categorical spacing)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("A1: label-informed diagnostic upper bound")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=180)
    fig.savefig(output / "curves.pdf")
    plt.close(fig)


def save_config(args):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def resolve_inputs(args, parser):
    if args.datasets and args.period:
        parser.error("Use --datasets or --period, not both")
    if args.datasets:
        args.period = []
        for value in args.datasets:
            path = Path(value)
            if path.is_dir():
                # Use held-out source data when available, then chronological targets.
                baseline = next((path / name for name in ("valid.npz", "tam_valid.npz", "train.npz", "tam_train.npz")
                                 if (path / name).exists()), None)
                if baseline:
                    args.period.append(("day0", str(baseline)))
                targets = {}
                for candidate in path.glob("*.npz"):
                    match = re.fullmatch(r"(?:tam_)?day(\d+)", candidate.stem)
                    if match:
                        day = int(match.group(1))
                        if day not in targets or not candidate.stem.startswith("tam_"):
                            targets[day] = candidate
                if 0 in targets and baseline:
                    args.period.pop()
                args.period.extend((f"day{day}", str(targets[day])) for day in sorted(targets))
            else:
                args.period.append((re.sub(r"^tam_", "", path.stem), str(path)))
    if not args.period:
        parser.error("Provide --datasets PATH [PATH ...] or --period NAME PATH")
    for _, path in args.period:
        if not Path(path).is_file():
            parser.error(f"Dataset does not exist: {path}")
    if args.input_mode == "model":
        if not args.model and not args.model_factory and args.checkpoint:
            if args.checkpoint.parent.name in ("DMX", "RF"):
                args.model = args.checkpoint.parent.name
            else:
                parser.error("Cannot infer model from checkpoint directory; supply --model DMX/RF")
        if args.feature is None:
            if args.model in ("DMX", "RF"):
                args.feature = "TAM"
            elif not args.loader:
                parser.error("External models require --feature or --loader")
        if args.device == "auto":
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.num_classes is None:
        parser.error("Prediction mode requires --num-classes (full source label space)")
    if args.seq_len is None:
        args.seq_len = 1800 if args.feature == "TAM" else 5000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-mode", choices=["predictions", "model"], default="model")
    p.add_argument("--period", nargs=2, action="append", metavar=("NAME", "PATH"))
    p.add_argument("--num-classes", type=int, help="Fixed source output class count")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--datasets", "--dataset", nargs="+", help="NPZ paths or a dataset directory")
    p.add_argument("--true-key", default="y_true")
    p.add_argument("--pred-key", default="y_pred")
    p.add_argument("--model", help="WFlib model name, e.g. DMX or RF")
    p.add_argument("--model-factory", help="External module:callable accepting num_classes")
    p.add_argument("--model-kwargs", default="{}", help="JSON extra model constructor arguments")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-key", help="Optional state_dict key in checkpoint")
    p.add_argument("--loader", help="External module:callable returning preprocessed (X, y)")
    p.add_argument("--loader-kwargs", default="{}")
    p.add_argument("--feature", choices=["DIR", "DT", "DT2", "TAM", "TAF", "MTAF"])
    p.add_argument("--seq-len", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--logits-index", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--confidence", type=float, default=.95)
    p.add_argument("--holdout", type=float, default=.5, help="Fraction reserved to evaluate fitted mapping")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()
    resolve_inputs(args, p)
    if min(args.num_classes or 1, args.batch_size, args.seq_len, args.repeats) < 1 or args.num_workers < 0:
        p.error("Counts must be positive (num-workers may be zero)")
    if args.seed < 0 or not 0 < args.confidence < 1 or not 0 < args.holdout < 1:
        p.error("Require nonnegative seed and confidence/holdout in (0, 1)")
    if len({name for name, _ in args.period}) != len(args.period):
        p.error("Period names must be unique")
    if args.input_mode == "model" and (not args.checkpoint or bool(args.model) == bool(args.model_factory)):
        p.error("Model mode requires --checkpoint and exactly one of --model/--model-factory")
    model = build_model(args) if args.input_mode == "model" else None
    args.resolved_inputs = []
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    elif args.output_dir.exists():
        args.output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output: {args.output_dir.resolve()}", flush=True)
    save_config(args)
    reports = []
    seeds = np.random.SeedSequence(args.seed).spawn(len(args.period))
    for index, (name, path) in enumerate(args.period):
        if args.input_mode == "predictions":
            with np.load(path, allow_pickle=False) as data:
                y, pred = data[args.true_key], data[args.pred_key]
        else:
            y, pred = predict(args, path, model)
        save_config(args)
        seed = int(seeds[index].generate_state(1)[0])
        report, matrix, draws = analyze(y, pred, args.num_classes, args.repeats,
                                         args.confidence, seed, args.holdout)
        report.update(period=name, input_path=str(Path(path).resolve()), seed=seed)
        dest = args.output_dir / f"period_{index:03d}"
        dest.mkdir()
        np.savez_compressed(dest / "predictions.npz", y_true=y, y_pred=pred)
        np.save(dest / "confusion.npy", matrix)
        np.savez_compressed(dest / "samples.npz", **draws)
        (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        print(name, " ".join(f"{key}={report[key]:.4f}" for key in METRICS), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["period", "n", *METRICS])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in reports)
    plot(reports, args.output_dir)


if __name__ == "__main__":
    main()
