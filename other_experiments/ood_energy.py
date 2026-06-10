import torch
from other_experiments.ood_common import parse_common_args, run_ood_eval


def parse_args():
    return parse_common_args("OOD evaluation with energy score")


def build_score_fn(temperature):
    def score_fn(logits, probs, unknown_label):
        del probs, unknown_label
        return temperature * torch.logsumexp(logits / temperature, dim=1)

    return score_fn


def main():
    args = parse_args()
    score_name = f"Energy (-E(x) = T * logsumexp(logits / T), T={args.energy_temperature:g})"
    run_ood_eval(args, score_name, build_score_fn(args.energy_temperature), "energy")


if __name__ == "__main__":
    main()