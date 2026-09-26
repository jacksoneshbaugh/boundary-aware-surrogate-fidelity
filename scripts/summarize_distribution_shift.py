"""
Combine per-seed distribution-shift results and summarize across seeds.

This summarizer supports both Experiment 2 modes:

1. Original decision-tree capacity sweep
   Typical input directory:
       results/distribution_shift_3bins/

2. Cross-family robustness check
   Typical input directory:
       results/distribution_shift_family/

The mode is detected automatically unless --mode is supplied explicitly.

Outputs (inside --result-dir):
    runs.csv
    summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/distribution_shift_3bins"),
    )

    parser.add_argument(
        "--pattern",
        default="distribution_shift_3bins_seed_*.csv",
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "depth", "family"],
        default="auto",
        help=(
            "Summary mode. 'depth' preserves the original decision-tree "
            "capacity analysis; 'family' summarizes the representative "
            "cross-family robustness check; 'auto' infers the mode from "
            "the loaded rows."
        ),
    )

    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=10,
        help=(
            "Expected number of seeds. Used only for an integrity warning; "
            "set to 3 for a three-seed smoke test."
        ),
    )

    return parser.parse_args()


def detect_mode(runs):
    """Infer whether rows represent the depth sweep or family check."""
    if "surrogate_family" not in runs.columns:
        return "depth"

    families = set(
        runs["surrogate_family"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(families) > 1 or any(
        family != "decision_tree"
        for family in families
    ):
        return "family"

    return "depth"


def available_metrics(runs):
    candidates = [
        "teacher_accuracy",
        "n_source",
        "n_target_pool",
        "n_target",
        "source_mean_confidence",
        "target_mean_confidence",
        "source_global_fidelity",
        "true_target_fidelity",
        "pred_global",
        "pred_profile",
        "error_global",
        "error_profile",
        "error_improvement",
    ]

    return [
        column
        for column in candidates
        if column in runs.columns
    ]


def print_headline_stats(runs, mode):
    global_mae = runs["error_global"].mean()
    profile_mae = runs["error_profile"].mean()

    relative_reduction = (
        (global_mae - profile_mae) / global_mae
        if global_mae > 0
        else float("nan")
    )

    profile = runs["error_profile"].to_numpy(float)
    global_ = runs["error_global"].to_numpy(float)

    ties_mask = profile == global_
    wins = int(np.sum((profile < global_) & ~ties_mask))
    losses = int(np.sum((profile > global_) & ~ties_mask))
    ties = int(np.sum(ties_mask))

    print()
    print("Overall:")
    print(f"  Global MAE:  {100 * global_mae:.2f} pp")
    print(f"  Profile MAE: {100 * profile_mae:.2f} pp")
    print(
        "  Relative reduction: "
        f"{100 * relative_reduction:.1f}%"
    )
    print(
        "  Profile wins / ties / losses: "
        f"{wins} / {ties} / {losses}"
    )

    def compact_table(group_cols, title):
        grouped = (
            runs.groupby(group_cols, dropna=False)
            .agg(
                global_mae=("error_global", "mean"),
                profile_mae=("error_profile", "mean"),
                n_conditions=("seed", "size"),
            )
            .reset_index()
        )

        grouped["relative_reduction_pct"] = np.where(
            grouped["global_mae"] > 0,
            100
            * (
                grouped["global_mae"]
                - grouped["profile_mae"]
            )
            / grouped["global_mae"],
            np.nan,
        )

        grouped["global_mae"] *= 100
        grouped["profile_mae"] *= 100

        print()
        print(title)
        print(grouped.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    compact_table(["dataset"], "By dataset:")
    compact_table(["shift"], "By shift:")

    if mode == "family":
        compact_table(
            ["surrogate_family"],
            "By surrogate family:",
        )
        compact_table(
            ["dataset", "surrogate_family"],
            "By dataset × surrogate family:",
        )


def main():
    args = parse_args()

    files = sorted(args.result_dir.glob(args.pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching {args.result_dir / args.pattern}"
        )

    print(f"Found {len(files)} seed files.")

    runs = pd.concat(
        [pd.read_csv(path) for path in files],
        ignore_index=True,
    )

    required = {
        "dataset",
        "seed",
        "shift",
        "error_global",
        "error_profile",
        "profile_wins",
    }

    missing = required - set(runs.columns)
    if missing:
        raise ValueError(
            "Distribution-shift results missing required columns: "
            f"{sorted(missing)}"
        )

    mode = (
        detect_mode(runs)
        if args.mode == "auto"
        else args.mode
    )

    if mode == "family":
        family_required = {
            "surrogate_family",
            "surrogate_config",
        }
        missing = family_required - set(runs.columns)
        if missing:
            raise ValueError(
                "Family-mode results missing columns: "
                f"{sorted(missing)}"
            )

        sort_cols = [
            "dataset",
            "surrogate_family",
            "surrogate_config",
            "seed",
            "shift",
        ]
        group_cols = [
            "dataset",
            "surrogate_family",
            "surrogate_config",
            "shift",
        ]
    else:
        if "depth" not in runs.columns:
            raise ValueError(
                "Depth-mode results require a 'depth' column."
            )

        sort_cols = [
            "dataset",
            "depth",
            "seed",
            "shift",
        ]
        group_cols = [
            "dataset",
            "depth",
            "shift",
        ]

    print(f"Detected summary mode: {mode}")

    seeds = sorted(runs["seed"].unique())
    print(f"Seeds found: {seeds}")
    print(f"Total rows: {len(runs)}")

    expected_seeds = set(range(args.expected_seeds))
    missing_seeds = expected_seeds - set(seeds)

    if missing_seeds:
        print(
            "WARNING: missing expected seeds: "
            f"{sorted(missing_seeds)}"
        )

    runs = runs.sort_values(sort_cols)

    runs_path = args.result_dir / "runs.csv"
    runs.to_csv(runs_path, index=False)

    metric_cols = available_metrics(runs)

    summary = (
        runs.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if x)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    extra = (
        runs.groupby(group_cols, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_conditions=("seed", "size"),
            profile_win_rate=("profile_wins", "mean"),
        )
        .reset_index()
    )

    summary = summary.merge(
        extra,
        on=group_cols,
        how="left",
    )

    summary = summary.sort_values(group_cols)

    summary_path = args.result_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print_headline_stats(runs, mode)

    print()
    print(f"Saved {runs_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
