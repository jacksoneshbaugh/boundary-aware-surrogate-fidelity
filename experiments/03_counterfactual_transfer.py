"""
Experiment 3: Example-based counterfactual transfer.

This experiment tests whether high global teacher-surrogate fidelity implies
that surrogate-derived counterfactual interventions behave similarly under
the teacher model.

For each test example on which teacher and surrogate initially agree, the
surrogate identifies the nearest example in a disjoint candidate pool that
it predicts as the opposite class. The teacher is then evaluated on that
candidate to determine whether the surrogate's intended prediction change
transfers.

The data are divided into three disjoint roles:

    - training set:
        trains the teacher and surrogate

    - candidate set:
        supplies possible surrogate-derived counterfactual destinations

    - test set:
        supplies starting examples for counterfactual queries

The experiment records global fidelity, boundary-local fidelity,
risk-coverage, counterfactual transfer rate, and counterfactual distances
across surrogate complexities and random seeds.

This is an example-based behavioral stress test rather than a claim about
actionable real-world recourse: candidate examples are selected by proximity
in standardized feature space and are not constrained by domain-specific
actionability or causal feasibility.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (
    compute_aurc,
    load_tabular_datasets,
    lowest_fraction_mask,
    teacher_logits,
    train_mlp_teacher,
)


DEFAULT_SEEDS = list(range(10))
DEFAULT_DEPTHS = [2, 3, 5, 7, 10]
DEFAULT_DATASETS = [
    "breast_cancer",
    "wine",
    "diabetes",
]


def split_train_candidate_test(
    X,
    y,
    seed,
    train_fraction=0.60,
    candidate_fraction=0.20,
):
    """
    Split data into disjoint train, candidate, and test sets.

    Scaling parameters are estimated using the training set only.
    """
    holdout_fraction = 1.0 - train_fraction

    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=y,
    )

    candidate_share_of_holdout = (
        candidate_fraction / holdout_fraction
    )

    (
        X_test_raw,
        X_candidate_raw,
        y_test,
        y_candidate,
    ) = train_test_split(
        X_holdout_raw,
        y_holdout,
        test_size=candidate_share_of_holdout,
        random_state=seed + 10_000,
        stratify=y_holdout,
    )

    scaler = StandardScaler().fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_candidate = scaler.transform(X_candidate_raw)
    X_test = scaler.transform(X_test_raw)

    return (
        X_train,
        X_candidate,
        X_test,
        y_train,
        y_candidate,
        y_test,
    )


def find_example_based_counterfactual(
    surrogate,
    x0,
    X_candidate,
    surrogate_candidate_predictions,
):
    """
    Find the nearest candidate example assigned to the opposite class
    by the surrogate.
    """
    start_class = int(
        surrogate.predict(
            x0.reshape(1, -1)
        )[0]
    )

    target_class = 1 - start_class

    target_mask = (
        surrogate_candidate_predictions
        == target_class
    )

    if not np.any(target_mask):
        return None, None, target_class

    candidates = X_candidate[target_mask]

    distances = np.linalg.norm(
        candidates - x0,
        axis=1,
    )

    nearest = int(np.argmin(distances))

    return (
        candidates[nearest],
        float(distances[nearest]),
        target_class,
    )


def run_dataset(
    dataset_name,
    X,
    y,
    seed,
    depths,
    epochs,
    boundary_fraction,
):
    (
        X_train,
        X_candidate,
        X_test,
        y_train,
        y_candidate,
        y_test,
    ) = split_train_candidate_test(
        X,
        y,
        seed=seed,
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

    candidate_logits = teacher_logits(
        teacher,
        X_candidate,
    )

    test_logits = teacher_logits(
        teacher,
        X_test,
    )

    teacher_train = (
        train_logits >= 0.0
    ).astype(int)

    teacher_candidate = (
        candidate_logits >= 0.0
    ).astype(int)

    teacher_test = (
        test_logits >= 0.0
    ).astype(int)

    confidence_test = np.abs(test_logits)

    boundary_mask = lowest_fraction_mask(
        confidence_test,
        boundary_fraction,
    )

    teacher_accuracy = float(
        np.mean(teacher_test == y_test)
    )

    summary_rows = []
    query_rows = []

    for depth in depths:
        surrogate = DecisionTreeClassifier(
            max_depth=depth,
            random_state=seed,
        )

        surrogate.fit(
            X_train,
            teacher_train,
        )

        surrogate_test = surrogate.predict(
            X_test
        )

        surrogate_candidate = surrogate.predict(
            X_candidate
        )

        agreement = (
            surrogate_test == teacher_test
        ).astype(float)

        global_fidelity = float(
            np.mean(agreement)
        )

        boundary_fidelity = float(
            np.mean(
                agreement[boundary_mask]
            )
        )

        aurc = compute_aurc(
            confidence_test,
            agreement,
        )

        transfers = 0
        failures = 0
        no_candidate = 0
        excluded_initial_disagreement = 0

        distances = []

        for test_index, x0 in enumerate(X_test):
            # Transfer is interpretable only when teacher and surrogate
            # agree on the starting prediction.
            if (
                surrogate_test[test_index]
                != teacher_test[test_index]
            ):
                excluded_initial_disagreement += 1
                continue

            (
                x_cf,
                distance,
                target_class,
            ) = find_example_based_counterfactual(
                surrogate,
                x0,
                X_candidate,
                surrogate_candidate,
            )

            if x_cf is None:
                no_candidate += 1
                continue

            teacher_cf_logit = teacher_logits(
                teacher,
                x_cf.reshape(1, -1),
            )[0]

            teacher_cf_class = int(
                teacher_cf_logit >= 0.0
            )

            success = (
                teacher_cf_class
                == target_class
            )

            transfers += int(success)
            failures += int(not success)
            distances.append(distance)

            query_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "depth": depth,
                    "test_index": test_index,

                    "start_teacher_class":
                        int(teacher_test[test_index]),

                    "start_surrogate_class":
                        int(surrogate_test[test_index]),

                    "target_class":
                        int(target_class),

                    "start_teacher_confidence":
                        float(
                            confidence_test[test_index]
                        ),

                    "counterfactual_distance":
                        float(distance),

                    "teacher_counterfactual_class":
                        teacher_cf_class,

                    "transfer_success":
                        bool(success),
                }
            )

        total_attempts = (
            transfers + failures
        )

        transfer_rate = (
            transfers / total_attempts
            if total_attempts > 0
            else np.nan
        )

        eligible_starts = (
            len(X_test)
            - excluded_initial_disagreement
        )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "depth": depth,

                "teacher_accuracy":
                    teacher_accuracy,

                "n_train":
                    len(X_train),

                "n_candidate":
                    len(X_candidate),

                "n_test":
                    len(X_test),

                "global_fidelity":
                    global_fidelity,

                "boundary_fraction":
                    boundary_fraction,

                "boundary_fidelity":
                    boundary_fidelity,

                "aurc":
                    aurc,

                "eligible_start_points":
                    eligible_starts,

                "excluded_initial_disagreement":
                    excluded_initial_disagreement,

                "no_counterfactual_candidate":
                    no_candidate,

                "total_recourse_attempts":
                    total_attempts,

                "transfers":
                    transfers,

                "failures":
                    failures,

                "recourse_transfer_rate":
                    float(transfer_rate),

                "mean_recourse_distance":
                    (
                        float(np.mean(distances))
                        if distances
                        else np.nan
                    ),

                "median_recourse_distance":
                    (
                        float(np.median(distances))
                        if distances
                        else np.nan
                    ),
            }
        )

        print(
            f"{dataset_name:14s} | "
            f"seed={seed:2d} | "
            f"d={depth:2d} | "
            f"global={global_fidelity:.3f} | "
            f"boundary={boundary_fidelity:.3f} | "
            f"transfer={transfer_rate:.3f} "
            f"({transfers}/{total_attempts}) | "
            f"excluded="
            f"{excluded_initial_disagreement} | "
            f"no-candidate={no_candidate}"
        )

    return summary_rows, query_rows


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
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--boundary-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/counterfactual_transfer/"
            "runs.csv"
        ),
    )

    parser.add_argument(
        "--query-output",
        default=None,
    )

    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def main(args):
    datasets = load_tabular_datasets()

    unknown = (
        set(args.datasets)
        - set(datasets.keys())
    )

    if unknown:
        raise ValueError(
            f"Unknown datasets: "
            f"{sorted(unknown)}"
        )

    summary_rows = []
    query_rows = []

    for dataset_name in args.datasets:
        X, y = datasets[dataset_name]

        for seed in args.seeds:
            summaries, queries = run_dataset(
                dataset_name,
                X,
                y,
                seed=seed,
                depths=args.depths,
                epochs=args.epochs,
                boundary_fraction=(
                    args.boundary_fraction
                ),
            )

            summary_rows.extend(summaries)
            query_rows.extend(queries)

    output = Path(args.output)

    write_csv(
        output,
        summary_rows,
    )

    query_output = (
        Path(args.query_output)
        if args.query_output
        else output.with_name(
            output.stem
            + "_queries.csv"
        )
    )

    write_csv(
        query_output,
        query_rows,
    )

    print(
        f"\nSaved {len(summary_rows)} "
        f"summary rows to {output}"
    )

    print(
        f"Saved {len(query_rows)} "
        f"query rows to {query_output}"
    )


if __name__ == "__main__":
    main(parse_args())