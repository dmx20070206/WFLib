"""
score_search.py
---------------
Read the per-(n,m) evaluation JSON files written by search_rand_augment.sh
and produce a ranked summary table.

Ranking criterion
-----------------
Primary  : mean Accuracy across all drift splits × all scenarios
Secondary: mean Accuracy across ONLY 50-50 scenario (closed-world-like)
Tertiary : mean Accuracy across ONLY 10-90 scenario (harder)

Usage (called by bash script)
-----
    python key_observation/score_search.py \
        --logs    logs/AugSearch/TemporalDrift/DF \
        --drifts  "day14 day30 day90 day150 day270" \
        --scenarios "5050:test 1090:test_1090"

Standalone usage (inspect results after the fact)
-----
    python key_observation/score_search.py \
        --logs    logs/AugSearch/TemporalDrift/DF
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

# ── Defaults (must match search_rand_augment.sh) ──────────────────────────────
DEFAULT_N_VALUES = [1, 2, 3]
DEFAULT_M_VALUES = [1, 2, 3, 4, 5]
DEFAULT_DRIFTS   = "day14 day30 day90 day150 day270"
DEFAULT_SCENARIOS = "5050:test 1090:test_1090"
DEFAULT_METRIC   = "Accuracy"
# ──────────────────────────────────────────────────────────────────────────────


def load_result(log_dir: Path, tag: str) -> float | None:
    """Return the primary metric value from  <log_dir>/<tag>.json, or None."""
    path = log_dir / f"{tag}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get(DEFAULT_METRIC)
    except (json.JSONDecodeError, KeyError):
        return None


def score(
    log_dir: Path,
    n_values: list[int],
    m_values: list[int],
    drifts: list[str],
    scenario_tags: list[str],
) -> list[dict]:
    """Compute per-(n,m) mean accuracy and return a list of result dicts."""
    rows = []
    for n, m in product(n_values, m_values):
        tag_prefix = f"n{n}_m{m}"
        all_scores, by_scenario = [], {s: [] for s in scenario_tags}

        for scenario_tag, drift in product(scenario_tags, drifts):
            result_tag = f"{tag_prefix}_{scenario_tag}_{drift}"
            val = load_result(log_dir, result_tag)
            if val is not None:
                all_scores.append(val)
                by_scenario[scenario_tag].append(val)

        if not all_scores:
            continue

        row = {
            "n": n,
            "m": m,
            "mean_acc": sum(all_scores) / len(all_scores),
            "n_runs": len(all_scores),
        }
        for scenario_tag, vals in by_scenario.items():
            key = f"mean_acc_{scenario_tag}"
            row[key] = sum(vals) / len(vals) if vals else float("nan")
        rows.append(row)

    rows.sort(key=lambda r: r["mean_acc"], reverse=True)
    return rows


def print_table(rows: list[dict], scenario_tags: list[str]) -> None:
    if not rows:
        print("No results found. Have you run the search yet?")
        return

    # Header
    scenario_cols = "".join(f"  acc_{s:<6}" for s in scenario_tags)
    print(f"\n{'n':>3} {'m':>3}  {'mean_acc':>8}{scenario_cols}  {'runs':>5}")
    print("─" * (3 + 3 + 8 + 10 * len(scenario_tags) + 7 + 6))

    for i, row in enumerate(rows):
        prefix = "★ " if i == 0 else "  "
        scen_vals = "".join(
            f"  {row.get(f'mean_acc_{s}', float('nan')):>8.4f}" for s in scenario_tags
        )
        print(
            f"{prefix}{row['n']:>2} {row['m']:>3}  "
            f"{row['mean_acc']:>8.4f}{scen_vals}  {row['n_runs']:>5}"
        )

    best = rows[0]
    print(
        f"\nBest: n={best['n']}, m={best['m']}  "
        f"(mean_acc={best['mean_acc']:.4f}, runs={best['n_runs']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank RandAugment search results")
    parser.add_argument("--logs",      required=True,  help="Directory with result JSON files")
    parser.add_argument("--drifts",    default=DEFAULT_DRIFTS,    help="Space-separated drift split names")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Space-separated scenario_tag:test_suffix pairs")
    parser.add_argument("--n-values",  default=None,   help="Comma-separated n values (default: 1,2,3)")
    parser.add_argument("--m-values",  default=None,   help="Comma-separated m values (default: 1,2,3,4,5)")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    n_values = [int(x) for x in args.n_values.split(",")] if args.n_values else DEFAULT_N_VALUES
    m_values = [int(x) for x in args.m_values.split(",")] if args.m_values else DEFAULT_M_VALUES
    drifts   = args.drifts.split()
    # scenario_tags: only the tag part (before colon)
    scenario_tags = [s.split(":")[0] for s in args.scenarios.split()]

    rows = score(log_dir, n_values, m_values, drifts, scenario_tags)
    print_table(rows, scenario_tags)


if __name__ == "__main__":
    main()
