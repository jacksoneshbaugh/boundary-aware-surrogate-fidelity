"""
Combine per-seed boundary-fidelity results and summarize them across seeds.

Inputs:
    results/boundary_fidelity/boundary_fidelity_seed_*.csv

Outputs:
    results/boundary_fidelity/runs.csv
    results/boundary_fidelity/summary.csv

The raw `runs.csv` preserves every individual experimental observation.
The summary groups results by dataset, surrogate depth, boundary definition,
and boundary-region width, then reports means and variation across seeds.
"""

from pathlib import Path

import pandas as pd


RESULT_DIR = Path("results/boundary_fidelity")
INPUT_PATTERN = "boundary_fidelity_seed_*.csv"


def main():
    files = sorted(RESULT_DIR.glob(INPUT_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"No files matching {RESULT_DIR / INPUT_PATTERN}"
        )

    print(f"Found {len(files)} seed files.")

    frames = [pd.read_csv(path) for path in files]
    runs = pd.concat(frames, ignore_index=True)

    # Basic integrity checks.
    seeds = sorted(runs["seed"].unique())

    print(f"Seeds found: {seeds}")
    print(f"Total rows: {len(runs)}")

    # Preserve all individual runs.
    runs = runs.sort_values(
        [
            "dataset",
            "seed",
            "depth",
            "boundary_fraction",
            "boundary_type",
        ]
    )

    runs.to_csv(
        RESULT_DIR / "runs.csv",
        index=False,
    )

    # Summarize across random seeds.
    group_cols = [
        "dataset",
        "depth",
        "boundary_type",
        "boundary_fraction",
    ]

    metric_cols = [
        "teacher_accuracy",
        "global_fidelity",
        "aurc",
        "n_boundary",
        "boundary_fidelity",
        "boundary_gap",
        "boundary_set_jaccard",
        "boundary_overlap_n",
        "boundary_union_n",
    ]

    summary = (
        runs.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten pandas' multi-level column names:
    # ('global_fidelity', 'mean') -> global_fidelity_mean
    summary.columns = [
        "_".join(
            str(part) for part in col if part
        )
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    # Number of independent seeds in each condition.
    n_seeds = (
        runs.groupby(group_cols)["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )

    summary = summary.merge(
        n_seeds,
        on=group_cols,
        how="left",
    )

    summary = summary.sort_values(group_cols)

    summary.to_csv(
        RESULT_DIR / "summary.csv",
        index=False,
    )

    print(
        f"Saved:\n"
        f"  {RESULT_DIR / 'runs.csv'}\n"
        f"  {RESULT_DIR / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()