import torch
from other_experiments.ood_common import parse_common_args, run_ood_eval


def parse_args():
    return parse_common_args("OOD evaluation with negative entropy")


def score_fn(logits, probs, unknown_label):
    del logits, unknown_label
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return -entropy


def main():
    args = parse_args()
    run_ood_eval(args, "-Entropy = sum p(y|x) * log p(y|x)", score_fn, "entropy")


if __name__ == "__main__":
    main()