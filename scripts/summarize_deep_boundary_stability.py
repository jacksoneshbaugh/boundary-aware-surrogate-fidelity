"""
Pairwise analysis for Experiment 4: deep boundary stability.

Loads cached predictions and top-two logit margins from independently
trained teachers and compares every unique pair.

For each teacher and boundary fraction, its boundary-proximal set is the
fixed fraction of test examples with the smallest classification margins.

For every teacher pair i,j, this script records:
    - test accuracy of both teachers
    - global predictive agreement
    - agreement on teacher i's boundary set
    - agreement on teacher j's boundary set
    - agreement on the union of their boundary sets
    - agreement on the intersection of their boundary sets
    - boundary-set intersection / union sizes
    - boundary-set Jaccard overlap

Pairwise rows are descriptive comparisons and are not treated as
independent statistical replicates, because each trained teacher appears
in multiple pairs.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FRACTIONS = [
    0.10,
    0.20,
    0.25,
]


def lowest_fraction_mask(
    values,
    fraction,
):
    k = max(
        1,
        int(np.ceil(
            fraction * len(values)
        )),
    )

    indices = np.argsort(
        values,
        kind="mergesort",
    )[:k]

    mask = np.zeros(
        len(values),
        dtype=bool,
    )

    mask[indices] = True

    return mask


def load_runs(cache_dir):
    files = sorted(
        cache_dir.glob("*_seed_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No cached runs in {cache_dir}"
        )

    runs = {}

    for path in files:
        data = np.load(path)

        dataset = str(data["dataset"])
        seed = int(data["seed"])

        runs[(dataset, seed)] = {
            "path": path,
            "labels": data["labels"],
            "predictions": data["predictions"],
            "margins": data["margins"],
            "accuracy": float(data["accuracy"]),
            "epochs": int(data["epochs"]),
        }

    return runs


def validate_dataset_runs(
    dataset,
    dataset_runs,
):
    seeds = sorted(dataset_runs)

    reference_labels = (
        dataset_runs[seeds[0]]["labels"]
    )

    for seed in seeds:
        run = dataset_runs[seed]

        if len(run["predictions"]) != len(
            reference_labels
        ):
            raise ValueError(
                f"{dataset} seed {seed}: "
                "test-set length mismatch"
            )

        if not np.array_equal(
            run["labels"],
            reference_labels,
        ):
            raise ValueError(
                f"{dataset} seed {seed}: "
                "test labels/order differ"
            )

        if np.any(
            run["margins"] < 0
        ):
            raise ValueError(
                f"{dataset} seed {seed}: "
                "negative top-two margin"
            )


def masked_agreement(
    agreement,
    mask,
):
    n = int(mask.sum())

    if n == 0:
        return np.nan

    return float(
        np.mean(agreement[mask])
    )


def compare_pair(
    dataset,
    seed_i,
    run_i,
    seed_j,
    run_j,
    fractions,
):
    pred_i = run_i["predictions"]
    pred_j = run_j["predictions"]

    agreement = pred_i == pred_j

    global_agreement = float(
        np.mean(agreement)
    )

    rows = []

    for fraction in fractions:
        mask_i = lowest_fraction_mask(
            run_i["margins"],
            fraction,
        )

        mask_j = lowest_fraction_mask(
            run_j["margins"],
            fraction,
        )

        intersection = (
            mask_i & mask_j
        )

        union = (
            mask_i | mask_j
        )

        n_i = int(mask_i.sum())
        n_j = int(mask_j.sum())
        n_intersection = int(
            intersection.sum()
        )
        n_union = int(
            union.sum()
        )

        jaccard = (
            n_intersection / n_union
            if n_union > 0
            else np.nan
        )

        agreement_i = masked_agreement(
            agreement,
            mask_i,
        )

        agreement_j = masked_agreement(
            agreement,
            mask_j,
        )

        union_agreement = masked_agreement(
            agreement,
            union,
        )

        intersection_agreement = (
            masked_agreement(
                agreement,
                intersection,
            )
        )

        mean_directed_boundary_agreement = (
            0.5
            * (
                agreement_i
                + agreement_j
            )
        )

        rows.append(
            {
                "dataset": dataset,
                "seed_i": seed_i,
                "seed_j": seed_j,

                "accuracy_i":
                    run_i["accuracy"],

                "accuracy_j":
                    run_j["accuracy"],

                "epochs_i":
                    run_i["epochs"],

                "epochs_j":
                    run_j["epochs"],

                "global_agreement":
                    global_agreement,

                "boundary_fraction":
                    fraction,

                "boundary_i_n":
                    n_i,

                "boundary_j_n":
                    n_j,

                "boundary_intersection_n":
                    n_intersection,

                "boundary_union_n":
                    n_union,

                "boundary_jaccard":
                    jaccard,

                "agreement_on_boundary_i":
                    agreement_i,

                "agreement_on_boundary_j":
                    agreement_j,

                "mean_directed_boundary_agreement":
                    mean_directed_boundary_agreement,

                "agreement_on_union":
                    union_agreement,

                "agreement_on_intersection":
                    intersection_agreement,

                "global_minus_union_agreement":
                    (
                        global_agreement
                        - union_agreement
                    ),
            }
        )

    return rows


def make_model_table(runs):
    rows = []

    for (
        dataset,
        seed,
    ), run in runs.items():
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "epochs": run["epochs"],
                "test_accuracy":
                    run["accuracy"],
                "n_test":
                    len(run["labels"]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["dataset", "seed"]
    )


def make_pairwise_table(
    runs,
    fractions,
):
    rows = []

    datasets = sorted(
        set(
            dataset
            for dataset, _ in runs
        )
    )

    for dataset in datasets:
        dataset_runs = {
            seed: run
            for (
                run_dataset,
                seed,
            ), run in runs.items()
            if run_dataset == dataset
        }

        validate_dataset_runs(
            dataset,
            dataset_runs,
        )

        seeds = sorted(
            dataset_runs
        )

        for seed_i, seed_j in (
            itertools.combinations(
                seeds,
                2,
            )
        ):
            rows.extend(
                compare_pair(
                    dataset,
                    seed_i,
                    dataset_runs[seed_i],
                    seed_j,
                    dataset_runs[seed_j],
                    fractions,
                )
            )

    return pd.DataFrame(rows).sort_values(
        [
            "dataset",
            "boundary_fraction",
            "seed_i",
            "seed_j",
        ]
    )


def make_summary(pairs):
    metrics = [
        "global_agreement",
        "boundary_jaccard",
        "mean_directed_boundary_agreement",
        "agreement_on_union",
        "agreement_on_intersection",
        "global_minus_union_agreement",
    ]

    grouped = (
        pairs.groupby(
            [
                "dataset",
                "boundary_fraction",
            ]
        )[metrics]
        .agg(
            ["mean", "std", "min", "max"]
        )
        .reset_index()
    )

    grouped.columns = [
        "_".join(
            str(x)
            for x in col
            if x != ""
        )
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]

    counts = (
        pairs.groupby(
            [
                "dataset",
                "boundary_fraction",
            ]
        )
        .agg(
            n_pairs=(
                "global_agreement",
                "size",
            )
        )
        .reset_index()
    )

    return grouped.merge(
        counts,
        on=[
            "dataset",
            "boundary_fraction",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "results/deep_boundary_stability/cache"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/deep_boundary_stability"
        ),
    )

    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    runs = load_runs(
        args.cache_dir
    )

    models = make_model_table(
        runs
    )

    pairs = make_pairwise_table(
        runs,
        args.fractions,
    )

    summary = make_summary(
        pairs
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models.to_csv(
        args.output_dir
        / "models.csv",
        index=False,
    )

    pairs.to_csv(
        args.output_dir
        / "pairs.csv",
        index=False,
    )

    summary.to_csv(
        args.output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print("Models:")
    print(
        models.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Pairwise headline:")
    print()

    headline = (
        pairs.groupby(
            [
                "dataset",
                "boundary_fraction",
            ]
        )
        .agg(
            global_agreement=(
                "global_agreement",
                "mean",
            ),
            boundary_jaccard=(
                "boundary_jaccard",
                "mean",
            ),
            boundary_union_agreement=(
                "agreement_on_union",
                "mean",
            ),
            agreement_drop=(
                "global_minus_union_agreement",
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
    print("Saved models.csv")
    print("Saved pairs.csv")
    print("Saved summary.csv")


if __name__ == "__main__":
    main()