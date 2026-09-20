"""
Combine and validate per-seed counterfactual-transfer results.

Inputs
------
results/counterfactual_transfer/
    counterfactual_transfer_seed_*.csv
    counterfactual_transfer_queries_seed_*.csv

Outputs
-------
results/counterfactual_transfer/
    runs.csv
        One row per dataset x seed x surrogate depth.

    queries.csv
        One row per attempted example-based counterfactual query.

    summary.csv
        Mean/std across independent seeds for each dataset x depth.

The primary statistical unit is the independently seeded run. Therefore,
mean counterfactual-transfer rates in summary.csv are averages over seeds,
rather than treating individual counterfactual queries as independent
experimental replicates.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_PATTERN = "counterfactual_transfer_seed_*.csv"
QUERY_PATTERN = "counterfactual_transfer_queries_seed_*.csv"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/counterfactual_transfer"),
    )

    return parser.parse_args()


def read_files(files):
    if not files:
        return pd.DataFrame()

    return pd.concat(
        [pd.read_csv(path) for path in files],
        ignore_index=True,
    )


def normalize_bool(series):
    """Convert common CSV boolean encodings to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad = series[normalized.isna()].unique()
        raise ValueError(
            f"Could not parse boolean values: {bad}"
        )

    return normalized.astype(bool)


def validate_runs(runs):
    required = {
        "dataset",
        "seed",
        "depth",
        "teacher_accuracy",
        "global_fidelity",
        "boundary_fraction",
        "boundary_fidelity",
        "aurc",
        "eligible_start_points",
        "excluded_initial_disagreement",
        "no_counterfactual_candidate",
        "total_recourse_attempts",
        "transfers",
        "failures",
        "recourse_transfer_rate",
        "mean_recourse_distance",
        "median_recourse_distance",
    }

    missing = required - set(runs.columns)

    if missing:
        raise ValueError(
            f"Summary files are missing columns: {sorted(missing)}"
        )

    key = ["dataset", "seed", "depth"]

    if runs.duplicated(key).any():
        duplicates = runs.loc[
            runs.duplicated(key, keep=False),
            key,
        ]
        raise ValueError(
            "Duplicate dataset/seed/depth rows found:\n"
            f"{duplicates}"
        )

    if not np.all(
        runs["transfers"] + runs["failures"]
        == runs["total_recourse_attempts"]
    ):
        raise ValueError(
            "transfers + failures != total_recourse_attempts"
        )

    if not np.all(
        runs["total_recourse_attempts"]
        + runs["no_counterfactual_candidate"]
        == runs["eligible_start_points"]
    ):
        raise ValueError(
            "attempts + no-counterfactual cases != eligible starts"
        )

    if not np.all(
        runs["eligible_start_points"]
        + runs["excluded_initial_disagreement"]
        == runs["n_test"]
    ):
        raise ValueError(
            "eligible starts + excluded starts != n_test"
        )

    expected_rates = np.where(
        runs["total_recourse_attempts"] > 0,
        runs["transfers"]
        / runs["total_recourse_attempts"],
        np.nan,
    )

    if not np.allclose(
        runs["recourse_transfer_rate"],
        expected_rates,
        equal_nan=True,
    ):
        raise ValueError(
            "Saved counterfactual transfer rates are inconsistent "
            "with transfer/attempt counts."
        )


def validate_queries(runs, queries):
    required = {
        "dataset",
        "seed",
        "depth",
        "test_index",
        "start_teacher_class",
        "start_surrogate_class",
        "target_class",
        "start_teacher_confidence",
        "counterfactual_distance",
        "teacher_counterfactual_class",
        "transfer_success",
    }

    missing = required - set(queries.columns)

    if missing:
        raise ValueError(
            f"Query files are missing columns: {sorted(missing)}"
        )

    queries["transfer_success"] = normalize_bool(
        queries["transfer_success"]
    )

    # Every evaluated query must begin at teacher-surrogate agreement.
    if not np.all(
        queries["start_teacher_class"]
        == queries["start_surrogate_class"]
    ):
        raise ValueError(
            "Found query starting from teacher-surrogate disagreement."
        )

    # Binary counterfactual target must be opposite surrogate class.
    expected_target = 1 - queries["start_surrogate_class"]

    if not np.all(
        queries["target_class"] == expected_target
    ):
        raise ValueError(
            "Found query whose target is not the opposite class."
        )

    # Success flag must exactly correspond to teacher reaching target.
    expected_success = (
        queries["teacher_counterfactual_class"]
        == queries["target_class"]
    )

    if not np.all(
        queries["transfer_success"] == expected_success
    ):
        raise ValueError(
            "transfer_success is inconsistent with teacher predictions."
        )

    if (queries["counterfactual_distance"] <= 0).any():
        raise ValueError(
            "Found non-positive counterfactual distance."
        )

    key = ["dataset", "seed", "depth"]

    query_summary = (
        queries.groupby(key)
        .agg(
            query_count=(
                "transfer_success",
                "size",
            ),
            query_transfers=(
                "transfer_success",
                "sum",
            ),
            query_mean_distance=(
                "counterfactual_distance",
                "mean",
            ),
            query_median_distance=(
                "counterfactual_distance",
                "median",
            ),
        )
        .reset_index()
    )

    checked = runs.merge(
        query_summary,
        on=key,
        how="left",
    )

    checked["query_count"] = (
        checked["query_count"]
        .fillna(0)
        .astype(int)
    )

    checked["query_transfers"] = (
        checked["query_transfers"]
        .fillna(0)
        .astype(int)
    )

    if not np.all(
        checked["query_count"]
        == checked["total_recourse_attempts"]
    ):
        raise ValueError(
            "Number of query rows does not match total attempts."
        )

    if not np.all(
        checked["query_transfers"]
        == checked["transfers"]
    ):
        raise ValueError(
            "Query-level transfer successes do not match summary counts."
        )

    attempted = checked["total_recourse_attempts"] > 0

    if not np.allclose(
        checked.loc[
            attempted,
            "query_mean_distance",
        ],
        checked.loc[
            attempted,
            "mean_recourse_distance",
        ],
    ):
        raise ValueError(
            "Query mean distances do not match saved summary values."
        )

    if not np.allclose(
        checked.loc[
            attempted,
            "query_median_distance",
        ],
        checked.loc[
            attempted,
            "median_recourse_distance",
        ],
    ):
        raise ValueError(
            "Query median distances do not match saved summary values."
        )


def make_summary(runs):
    group_cols = [
        "dataset",
        "depth",
        "boundary_fraction",
    ]

    metric_cols = [
        "teacher_accuracy",
        "global_fidelity",
        "boundary_fidelity",
        "aurc",
        "eligible_start_points",
        "excluded_initial_disagreement",
        "no_counterfactual_candidate",
        "total_recourse_attempts",
        "recourse_transfer_rate",
        "mean_recourse_distance",
        "median_recourse_distance",
    ]

    summary = (
        runs.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in column if x != "")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    # Additional counts and pooled descriptive statistics.
    #
    # The pooled transfer rate is useful descriptively, but the
    # seed-level mean/std above should be treated as the main result.
    extras = (
        runs.groupby(group_cols)
        .agg(
            n_seeds=("seed", "nunique"),
            total_attempts=("total_recourse_attempts", "sum"),
            total_transfers=("transfers", "sum"),
            total_failures=("failures", "sum"),
        )
        .reset_index()
    )

    extras["pooled_transfer_rate"] = (
        extras["total_transfers"]
        / extras["total_attempts"]
    )

    summary = summary.merge(
        extras,
        on=group_cols,
        how="left",
    )

    return summary.sort_values(group_cols)


def print_headlines(runs):
    print("\nDataset-level averages across all seeds and depths:")
    print()

    headline = (
        runs.groupby("dataset")
        .agg(
            global_fidelity=(
                "global_fidelity",
                "mean",
            ),
            boundary_fidelity=(
                "boundary_fidelity",
                "mean",
            ),
            counterfactual_transfer=(
                "recourse_transfer_rate",
                "mean",
            ),
            teacher_accuracy=(
                "teacher_accuracy",
                "mean",
            ),
        )
    )

    print(
        headline.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()

    total_queries = int(
        runs["total_recourse_attempts"].sum()
    )

    print(
        f"Configurations: {len(runs)}"
    )
    print(
        f"Counterfactual queries: {total_queries}"
    )
    print(
        f"Seeds: {sorted(runs['seed'].unique())}"
    )


def main():
    args = parse_args()
    result_dir = args.result_dir

    summary_files = sorted(
        result_dir.glob(SUMMARY_PATTERN)
    )

    query_files = sorted(
        result_dir.glob(QUERY_PATTERN)
    )

    if not summary_files:
        raise FileNotFoundError(
            f"No files matching "
            f"{result_dir / SUMMARY_PATTERN}"
        )

    if not query_files:
        raise FileNotFoundError(
            f"No files matching "
            f"{result_dir / QUERY_PATTERN}"
        )

    print(
        f"Found {len(summary_files)} summary files."
    )
    print(
        f"Found {len(query_files)} query files."
    )

    runs = read_files(summary_files)
    queries = read_files(query_files)

    # Stable ordering makes diffs and inspection easier.
    runs = runs.sort_values(
        [
            "dataset",
            "seed",
            "depth",
        ]
    ).reset_index(drop=True)

    queries = queries.sort_values(
        [
            "dataset",
            "seed",
            "depth",
            "test_index",
        ]
    ).reset_index(drop=True)

    print("\nValidating summary rows...")
    validate_runs(runs)

    print("Validating query rows...")
    validate_queries(runs, queries)

    seeds = sorted(runs["seed"].unique())

    if seeds != list(range(10)):
        print(
            f"WARNING: expected seeds 0-9, found {seeds}"
        )

    summary = make_summary(runs)

    runs_path = result_dir / "runs.csv"
    queries_path = result_dir / "queries.csv"
    summary_path = result_dir / "summary.csv"

    runs.to_csv(
        runs_path,
        index=False,
    )

    queries.to_csv(
        queries_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nAll integrity checks passed.")

    print_headlines(runs)

    print()
    print(f"Saved {runs_path}")
    print(f"Saved {queries_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()