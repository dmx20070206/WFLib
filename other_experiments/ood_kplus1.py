import torch
from other_experiments.ood_common import parse_common_args, run_ood_eval


def parse_args():
    return parse_common_args("OOD evaluation with K+1 score")


def score_fn(logits, probs, unknown_label):
    del logits
    return 1.0 - probs[:, unknown_label]


def main():
    args = parse_args()
    run_ood_eval(args, "1 - P(y = K)", score_fn, "kplus1", model_class_offset=1)


if __name__ == "__main__":
    main()