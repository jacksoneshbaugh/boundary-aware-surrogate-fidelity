"""
Experiment 2: Fidelity profiles under evaluation-distribution shift.

This experiment tests whether a confidence-stratified fidelity profile
provides a better estimate of surrogate fidelity under changes in the
composition of the evaluation distribution than a single global fidelity
score.

Training, source-evaluation, and target-evaluation roles are kept disjoint.
A surrogate fidelity profile is estimated on the source evaluation set by
stratifying examples according to teacher confidence. That profile is then
combined with the confidence composition of a shifted target set to predict
the surrogate's target fidelity.

The primary comparison is:

    - prediction from source global fidelity
    - prediction from the source confidence-stratified fidelity profile

Predicted target fidelity is compared with actually observed target fidelity
across datasets, surrogate complexities, distribution shifts, and random
seeds.

The experiment evaluates whether retaining conditional information about
where surrogate agreement occurs improves fidelity estimation when evaluation
mass shifts across regions of the teacher's decision space.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from common import (
    load_tabular_datasets,
    split_and_scale,
    teacher_logits,
    train_mlp_teacher,
)


DEFAULT_SEEDS = list(range(10))
DEFAULT_DEPTHS = [3, 5, 7]
DEFAULT_DATASETS = [
    "breast_cancer",
    "wine",
    "diabetes",
]


def build_profile(
    source_confidence,
    source_agreement,
    n_bins=5,
):
    """
    Estimate fidelity as a function of teacher confidence.

    Bin boundaries are determined entirely from the source evaluation set.
    Tied quantiles may reduce the number of distinct bins.

    Returns
    -------
    edges : ndarray
        Confidence-bin edges, including -inf and +inf.
    fidelities : ndarray
        Source fidelity estimated within each confidence bin.
        Empty bins contain NaN.
    counts : ndarray
        Number of source examples in each confidence bin.
    """
    internal = np.quantile(
        source_confidence,
        np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
    )

    internal = np.unique(internal)

    edges = np.concatenate(
        ([-np.inf], internal, [np.inf])
    )

    bin_ids = np.digitize(
        source_confidence,
        edges[1:-1],
        right=False,
    )

    n_actual_bins = len(edges) - 1

    fidelities = np.full(
        n_actual_bins,
        np.nan,
        dtype=float,
    )

    counts = np.zeros(
        n_actual_bins,
        dtype=int,
    )

    for bin_id in range(n_actual_bins):
        mask = bin_ids == bin_id
        counts[bin_id] = int(mask.sum())

        if counts[bin_id] > 0:
            fidelities[bin_id] = float(
                np.mean(source_agreement[mask])
            )

    return edges, fidelities, counts


def predict_from_profile(
    target_confidence,
    edges,
    fidelities,
    fallback,
):
    """
    Predict target fidelity from the source confidence profile.

    If a target example falls in a source bin for which no source fidelity
    estimate exists, source global fidelity is used as a conservative
    fallback.
    """
    if len(target_confidence) == 0:
        return np.nan

    bin_ids = np.digitize(
        target_confidence,
        edges[1:-1],
        right=False,
    )

    predictions = np.empty(
        len(target_confidence),
        dtype=float,
    )

    for i, bin_id in enumerate(bin_ids):
        fidelity = fidelities[bin_id]

        predictions[i] = (
            fallback
            if np.isnan(fidelity)
            else fidelity
        )

    return float(np.mean(predictions))


def run_dataset(
    dataset_name,
    X,
    y,
    seed,
    depths,
    n_bins,
    epochs,
):
    (
        X_train,
        X_eval,
        y_train,
        y_eval,
        _,
    ) = split_and_scale(
        X,
        y,
        seed=seed,
        test_size=0.40,
    )

    teacher = train_mlp_teacher(
        X_train,
        y_train,
        seed=seed,
        epochs=epochs,
    )

    train_logits = teacher_logits(
        teacher,
        X_train,
    )

    eval_logits = teacher_logits(
        teacher,
        X_eval,
    )

    teacher_train = (
        train_logits >= 0.0
    ).astype(int)

    teacher_eval = (
        eval_logits >= 0.0
    ).astype(int)

    confidence_eval = np.abs(eval_logits)

    teacher_accuracy = float(
        np.mean(teacher_eval == y_eval)
    )

    # ------------------------------------------------------------
    # Split the held-out evaluation data into two disjoint roles:
    #
    #   source_idx:
    #       used to estimate the fidelity profile
    #
    #   target_idx:
    #       used only to construct and evaluate distribution shifts
    # ------------------------------------------------------------

    all_idx = np.arange(len(X_eval))

    source_idx, target_idx = train_test_split(
        all_idx,
        test_size=0.50,
        random_state=seed + 10_000,
        stratify=teacher_eval,
    )

    rows = []

    for depth in depths:
        surrogate = DecisionTreeClassifier(
            max_depth=depth,
            random_state=seed,
        )

        surrogate.fit(
            X_train,
            teacher_train,
        )

        surrogate_eval = surrogate.predict(
            X_eval
        )

        agreement = (
            surrogate_eval == teacher_eval
        ).astype(float)

        # -------------------------
        # Source fidelity profile
        # -------------------------

        source_agreement = agreement[source_idx]
        source_confidence = confidence_eval[source_idx]

        source_global_fidelity = float(
            np.mean(source_agreement)
        )

        (
            profile_edges,
            profile_fidelities,
            profile_counts,
        ) = build_profile(
            source_confidence,
            source_agreement,
            n_bins=n_bins,
        )

        # -------------------------
        # Independent target pool
        # -------------------------

        target_confidence_all = confidence_eval[
            target_idx
        ]

        target_agreement_all = agreement[
            target_idx
        ]

        median_target_confidence = float(
            np.median(target_confidence_all)
        )

        shifts = {
            "hard": (
                target_confidence_all
                < median_target_confidence
            ),
            "easy": (
                target_confidence_all
                >= median_target_confidence
            ),
        }

        for shift_name, shift_mask in shifts.items():
            target_confidence = (
                target_confidence_all[shift_mask]
            )

            target_agreement = (
                target_agreement_all[shift_mask]
            )

            if len(target_agreement) == 0:
                continue

            true_target_fidelity = float(
                np.mean(target_agreement)
            )

            # Baseline: assume fidelity does not change.
            pred_global = source_global_fidelity

            # Profile estimate: integrate source conditional fidelity
            # over the target confidence composition.
            pred_profile = predict_from_profile(
                target_confidence,
                profile_edges,
                profile_fidelities,
                fallback=source_global_fidelity,
            )

            error_global = float(
                abs(
                    pred_global
                    - true_target_fidelity
                )
            )

            error_profile = float(
                abs(
                    pred_profile
                    - true_target_fidelity
                )
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "depth": depth,
                    "shift": shift_name,

                    "teacher_accuracy": teacher_accuracy,

                    "n_train": len(X_train),
                    "n_source": len(source_idx),
                    "n_target_pool": len(target_idx),
                    "n_target": int(shift_mask.sum()),

                    "n_profile_bins_requested": n_bins,
                    "n_profile_bins_actual": len(
                        profile_fidelities
                    ),
                    "n_empty_profile_bins": int(
                        np.isnan(
                            profile_fidelities
                        ).sum()
                    ),

                    "source_mean_confidence": float(
                        np.mean(source_confidence)
                    ),
                    "target_mean_confidence": float(
                        np.mean(target_confidence)
                    ),

                    "source_global_fidelity":
                        source_global_fidelity,

                    "true_target_fidelity":
                        true_target_fidelity,

                    "pred_global":
                        pred_global,

                    "pred_profile":
                        pred_profile,

                    "error_global":
                        error_global,

                    "error_profile":
                        error_profile,

                    "error_improvement": (
                        error_global
                        - error_profile
                    ),

                    "profile_wins": bool(
                        error_profile
                        < error_global
                    ),
                }
            )

            winner = (
                "PROFILE"
                if error_profile < error_global
                else "GLOBAL/TIE"
            )

            print(
                f"{dataset_name:14s} | "
                f"seed={seed:2d} | "
                f"d={depth:2d} | "
                f"{shift_name:4s} | "
                f"true={true_target_fidelity:.3f} | "
                f"global={pred_global:.3f} "
                f"(err={error_global:.3f}) | "
                f"profile={pred_profile:.3f} "
                f"(err={error_profile:.3f}) | "
                f"{winner}"
            )

    return rows


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=DEFAULT_DEPTHS,
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/distribution_shift/"
            "runs.csv"
        ),
    )

    return parser.parse_args()


def main(args):
    datasets = load_tabular_datasets()

    unknown = (
        set(args.datasets)
        - set(datasets.keys())
    )

    if unknown:
        raise ValueError(
            f"Unknown datasets: {sorted(unknown)}"
        )

    rows = []

    for dataset_name in args.datasets:
        X, y = datasets[dataset_name]

        for seed in args.seeds:
            rows.extend(
                run_dataset(
                    dataset_name,
                    X,
                    y,
                    seed=seed,
                    depths=args.depths,
                    n_bins=args.bins,
                    epochs=args.epochs,
                )
            )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rows:
        with output.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    print(
        f"\nSaved {len(rows)} rows to "
        f"{output}"
    )


if __name__ == "__main__":
    main(parse_args())