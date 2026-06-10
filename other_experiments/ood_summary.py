import argparse

from other_experiments.ood_common import format_table, run_ood_eval
from other_experiments.ood_energy import build_score_fn
from other_experiments.ood_entropy import score_fn as entropy_score_fn
from other_experiments.ood_kplus1 import score_fn as kplus1_score_fn
from other_experiments.ood_msp import score_fn as msp_score_fn


def parse_args():
    parser = argparse.ArgumentParser(description="OOD evaluation summary across four scoring strategies")
    parser.add_argument("--id_model", type=str, required=True, help="Base closed-set model for energy/MSP/entropy")
    parser.add_argument("--id_model_file", type=str, required=True, help="Checkpoint file for the base model")
    parser.add_argument("--kplus1_model", type=str, required=True, help="K+1 model name")
    parser.add_argument("--kplus1_model_file", type=str, required=True, help="Checkpoint file for the K+1 model")
    parser.add_argument("--test_file", type=str, required=True, help="Known-class test file")
    parser.add_argument("--extra_test_file", type=str, required=True, help="Unknown-class test file")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--feature", type=str, default="DIR", help="Feature type")
    parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Data loader workers")
    parser.add_argument("--num_tabs", type=int, default=1, help="Number of tabs")
    parser.add_argument("--energy_temperature", type=float, default=1.0, help="Temperature used by energy score")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser.parse_args()


def build_eval_args(args, model_name, model_file):
    namespace = argparse.Namespace(**vars(args))
    namespace.model = model_name
    namespace.model_file = model_file
    return namespace


def build_strategy_specs(args):
    return [
        {
            "model": args.id_model,
            "model_file": args.id_model_file,
            "score_name": f"Energy: -E(x) = T * logsumexp(logits / T), T={args.energy_temperature:g}",
            "score_fn": build_score_fn(args.energy_temperature),
            "score_tag": "energy",
            "model_class_offset": 0,
        },
        {
            "model": args.id_model,
            "model_file": args.id_model_file,
            "score_name": "MSP = max softmax probability",
            "score_fn": msp_score_fn,
            "score_tag": "msp",
            "model_class_offset": 0,
        },
        {
            "model": args.id_model,
            "model_file": args.id_model_file,
            "score_name": "-Entropy = sum p(y|x) * log p(y|x)",
            "score_fn": entropy_score_fn,
            "score_tag": "entropy",
            "model_class_offset": 0,
        },
        {
            "model": args.kplus1_model,
            "model_file": args.kplus1_model_file,
            "score_name": "1 - P(y = K)",
            "score_fn": kplus1_score_fn,
            "score_tag": "kplus1",
            "model_class_offset": 1,
        },
    ]


def print_summary_table(args, results):
    shared_rows = [
        ["Known test", results[0]["known_test_file"]],
        ["Unknown test", results[0]["unknown_test_file"]],
        ["Feature", args.feature],
        ["Seq len", args.seq_len],
        ["Base model", f"{args.id_model} ({args.id_model_file})"],
        ["K+1 model", f"{args.kplus1_model} ({args.kplus1_model_file})"],
    ]
    metric_rows = [
        [result["score_name"], f"{result['auroc']:.6f}", f"{result['fpr90']:.6f}", f"{result['eer']:.6f}"]
        for result in results
    ]

    print(format_table(["Field", "Value"], shared_rows))
    print()
    print(format_table(["Strategy Formula", "AUROC", "FPR@90", "EER"], metric_rows))


def main():
    args = parse_args()
    results = []
    for spec in build_strategy_specs(args):
        eval_args = build_eval_args(args, spec["model"], spec["model_file"])
        results.append(
            run_ood_eval(
                eval_args,
                spec["score_name"],
                spec["score_fn"],
                spec["score_tag"],
                model_class_offset=spec["model_class_offset"],
                print_summary=False,
            )
        )
    print_summary_table(args, results)


if __name__ == "__main__":
    main()