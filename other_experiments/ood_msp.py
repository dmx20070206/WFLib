from other_experiments.ood_common import parse_common_args, run_ood_eval


def parse_args():
    return parse_common_args("OOD evaluation with MSP")


def score_fn(logits, probs, unknown_label):
    del logits, unknown_label
    return probs.max(dim=1).values


def main():
    args = parse_args()
    run_ood_eval(args, "MSP = max softmax probability", score_fn, "msp")


if __name__ == "__main__":
    main()