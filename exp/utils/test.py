import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from WFlib import models
from WFlib.tools import data_processor


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(model_name, num_classes, num_tabs):
    if model_name in ["BAPM", "TMWF"]:
        return eval(f"models.{model_name}")(num_classes, num_tabs)
    return eval(f"models.{model_name}")(num_classes)


def extract_features(model, data_array, batch_size, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data_array), batch_size), desc="Extracting features", leave=False):
            batch_np = data_array[i:i + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            out = model(batch)
            if isinstance(out, (tuple, list)):
                _, f = out
            else:
                # Fallback: if model only returns logits, use logits as feature view.
                f = out
            feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def maybe_filter_by_class(data_array, labels_array, target_class):
    if target_class < 0:
        return data_array, labels_array
    idx = np.where(labels_array == target_class)[0]
    return data_array[idx], labels_array[idx]


def maybe_subsample(data_array, labels_array, max_samples, seed=2024):
    if max_samples <= 0 or len(data_array) <= max_samples:
        return data_array, labels_array
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data_array), size=max_samples, replace=False)
    idx = np.sort(idx)
    return data_array[idx], labels_array[idx]


def tsne_2d(features, perplexity, seed=2024):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def plot_before_after(
    src_pre_2d, tgt_pre_2d,
    src_post_2d, tgt_post_2d,
    save_path,
    title_suffix=""
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(src_pre_2d[:, 0], src_pre_2d[:, 1], c="royalblue", s=10, alpha=0.6, label="Source")
    axes[0].scatter(tgt_pre_2d[:, 0], tgt_pre_2d[:, 1], c="crimson", s=10, alpha=0.6, label="Target", marker="^")
    axes[0].set_title(f"Before Fine-tuning{title_suffix}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(src_post_2d[:, 0], src_post_2d[:, 1], c="royalblue", s=10, alpha=0.6, label="Source")
    axes[1].scatter(tgt_post_2d[:, 0], tgt_post_2d[:, 1], c="crimson", s=10, alpha=0.6, label="Target", marker="^")
    axes[1].set_title(f"After Fine-tuning{title_suffix}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE before/after fine-tuning")
    parser.add_argument("--dataset", type=str, required=True, default="CW")
    parser.add_argument("--model", type=str, required=True, default="DF")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_tabs", type=int, default=1)

    parser.add_argument("--train_file", type=str, default="train")
    parser.add_argument("--test_file", type=str, default="test")
    parser.add_argument("--feature", type=str, default="DIR")
    parser.add_argument("--seq_len", type=int, default=5000)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--output_dir", type=str, default="temp/test")
    parser.add_argument("--output_name", type=str, default="tsne_before_after.png")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--target_class", type=int, default=-1, help="-1 means all classes")

    # Output parameters
    parser.add_argument("--eval_metrics", nargs="+", required=True, type=str, help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]",)
    parser.add_argument("--load_name", type=str, default="base", help="Name of the model file")
    parser.add_argument("--result_file", type=str, default="result", help="File to save test results")
    parser.add_argument("--model_save_name", type=str, default="proteus", help="Name used to save the model")

    args = parser.parse_args()
    set_seed(2024)
    device = get_device(args.device)

    dataset_path = os.path.join("./datasets", args.dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    pre_ckpt = os.path.join(ckp_path, "max_f1.pth")
    post_ckpt = os.path.join(ckp_path, "proteus.pth")
    if not os.path.exists(pre_ckpt):
        raise FileNotFoundError(f"Pre-finetune checkpoint not found: {pre_ckpt}")
    if not os.path.exists(post_ckpt):
        raise FileNotFoundError(f"Post-finetune checkpoint not found: {post_ckpt}")

    train_data, train_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.train_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )
    test_data, test_labels = data_processor.load_data(
        os.path.join(dataset_path, f"{args.test_file}.npz"),
        args.feature,
        args.seq_len,
        args.num_tabs,
    )

    train_labels_np = train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else train_labels
    test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels

    if args.num_tabs == 1:
        num_classes = len(np.unique(test_labels_np))
    else:
        num_classes = test_labels.shape[1]

    src_data, src_labels = maybe_filter_by_class(train_data, train_labels_np, args.target_class)
    tgt_data, tgt_labels = maybe_filter_by_class(test_data, test_labels_np, args.target_class)

    if len(src_data) == 0 or len(tgt_data) == 0:
        raise ValueError("No samples left after filtering by target_class.")

    # class list:
    # target_class >= 0: only draw that class
    # target_class < 0: draw all classes
    if args.target_class >= 0:
        class_list = [args.target_class]
    else:
        src_classes = set(np.unique(train_labels_np).tolist())
        tgt_classes = set(np.unique(test_labels_np).tolist())
        class_list = sorted(list(src_classes & tgt_classes))  # only classes existing in both domains

    if len(class_list) == 0:
        raise ValueError("No common classes found between source and target domains.")

    model_pre = build_model(args.model, num_classes, args.num_tabs).to(device)
    model_pre.load_state_dict(torch.load(pre_ckpt, map_location="cpu", weights_only=True))

    model_post = build_model(args.model, num_classes, args.num_tabs).to(device)
    model_post.load_state_dict(torch.load(post_ckpt, map_location="cpu", weights_only=True))
    
    for cls in class_list:
        src_data, src_labels = maybe_filter_by_class(train_data, train_labels_np, cls)
        tgt_data, tgt_labels = maybe_filter_by_class(test_data, test_labels_np, cls)

        if len(src_data) == 0 or len(tgt_data) == 0:
            print(f"Skip class {cls}: empty in source or target.")
            continue

        src_data, src_labels = maybe_subsample(src_data, src_labels, args.max_samples, seed=2024 + int(cls))
        tgt_data, tgt_labels = maybe_subsample(tgt_data, tgt_labels, args.max_samples, seed=3024 + int(cls))

        print(f"[Class {cls}] Source: {len(src_data)} | Target: {len(tgt_data)}")

        # extract features (before)
        src_pre = extract_features(model_pre, src_data, args.batch_size, device)
        tgt_pre = extract_features(model_pre, tgt_data, args.batch_size, device)

        # extract features (after)
        src_post = extract_features(model_post, src_data, args.batch_size, device)
        tgt_post = extract_features(model_post, tgt_data, args.batch_size, device)

        # t-SNE before
        pre_all = np.vstack([src_pre, tgt_pre])
        pre_2d = tsne_2d(pre_all, perplexity=args.perplexity, seed=2024)
        src_pre_2d = pre_2d[:len(src_pre)]
        tgt_pre_2d = pre_2d[len(src_pre):]

        # t-SNE after
        post_all = np.vstack([src_post, tgt_post])
        post_2d = tsne_2d(post_all, perplexity=args.perplexity, seed=2024)
        src_post_2d = post_2d[:len(src_post)]
        tgt_post_2d = post_2d[len(src_post):]

        # save one image per class
        base, ext = os.path.splitext(args.output_name)
        if ext == "":
            ext = ".png"
        class_output_name = f"{base}_class{cls}{ext}"
        save_path = os.path.join(args.output_dir, class_output_name)

        plot_before_after(
            src_pre_2d, tgt_pre_2d,
            src_post_2d, tgt_post_2d,
            save_path,
            title_suffix=f" (Class {cls})"
        )

    suffix = f" (Class {args.target_class})" if args.target_class >= 0 else " (All Classes)"
    save_path = os.path.join(args.output_dir, args.output_name)
    plot_before_after(
        src_pre_2d, tgt_pre_2d,
        src_post_2d, tgt_post_2d,
        save_path,
        title_suffix=suffix
    )


if __name__ == "__main__":
    main()