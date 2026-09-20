"""
Combine per-seed distribution-shift results and summarize across seeds.

By default, this script summarizes the 3-bin experiment:

    results/distribution_shift_3bins/distribution_shift_3bins_seed_*.csv

Outputs:
    results/distribution_shift_3bins/runs.csv
    results/distribution_shift_3bins/summary.csv
"""

import argparse
from pathlib import Path

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

    return parser.parse_args()


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

    # Basic integrity checks.
    seeds = sorted(runs["seed"].unique())

    print(f"Seeds found: {seeds}")
    print(f"Total rows: {len(runs)}")

    expected_seeds = set(range(10))
    missing = expected_seeds - set(seeds)

    if missing:
        print(f"WARNING: missing seeds: {sorted(missing)}")

    # Keep all raw experimental observations.
    sort_cols = [
        "dataset",
        "seed",
        "depth",
        "shift",
    ]

    runs = runs.sort_values(sort_cols)

    runs.to_csv(
        args.result_dir / "runs.csv",
        index=False,
    )

    # Summarize independent runs across seeds.
    group_cols = [
        "dataset",
        "depth",
        "shift",
    ]

    metric_cols = [
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

    summary = (
        runs.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if x)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    # Add number of seeds and profile win rate.
    extra = (
        runs.groupby(group_cols)
        .agg(
            n_seeds=("seed", "nunique"),
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

    summary.to_csv(
        args.result_dir / "summary.csv",
        index=False,
    )

    # Overall headline statistics.
    global_mae = runs["error_global"].mean()
    profile_mae = runs["error_profile"].mean()

    relative_reduction = (
        (global_mae - profile_mae) / global_mae
        if global_mae > 0
        else float("nan")
    )

    wins = (runs["error_profile"] < runs["error_global"]).sum()
    ties = (runs["error_profile"] == runs["error_global"]).sum()
    losses = (runs["error_profile"] > runs["error_global"]).sum()

    print()
    print("Overall:")
    print(f"  Global MAE:  {global_mae:.4f}")
    print(f"  Profile MAE: {profile_mae:.4f}")
    print(
        f"  Relative reduction: "
        f"{100 * relative_reduction:.1f}%"
    )
    print(
        f"  Profile wins / ties / losses: "
        f"{wins} / {ties} / {losses}"
    )

    print()
    print(f"Saved {args.result_dir / 'runs.csv'}")
    print(f"Saved {args.result_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()