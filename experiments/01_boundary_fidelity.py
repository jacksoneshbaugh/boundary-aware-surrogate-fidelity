"""
Experiment 1: Boundary-aware characterization of surrogate fidelity.

This experiment tests whether global teacher-surrogate agreement masks
systematically lower fidelity near the teacher's decision boundary.

For each tabular dataset, teacher seed, and surrogate complexity, the script
measures:

    - teacher predictive accuracy
    - global surrogate fidelity
    - confidence-local boundary fidelity
    - geometric transition-region fidelity
    - area under the risk-coverage curve (AURC)
    - the gap between global and boundary-local fidelity

Boundary-local fidelity is evaluated using two independent notions of
proximity to the teacher boundary:

    1. teacher confidence, using low-|logit| examples
    2. approximate geometric distance to a teacher decision transition

The experiment is repeated across multiple boundary-region widths to test
whether the observed global-vs-boundary fidelity gap is robust to the specific
boundary threshold.

All individual runs are written to a tidy results file so that means,
variation across seeds, confidence intervals, and paper figures can be
computed separately from model training.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from common import (
    compute_aurc,
    load_tabular_datasets,
    lowest_fraction_mask,
    segment_crossing_distances,
    split_and_scale,
    teacher_logits,
    train_mlp_teacher,
)


DEFAULT_SEEDS = list(range(10))
DEFAULT_DEPTHS = [1, 2, 3, 5, 7, 10]
DEFAULT_FRACTIONS = [0.10, 0.20, 0.25, 0.40]


def run(args):
    rows = []

    datasets = load_tabular_datasets()

    if args.datasets is not None:
        datasets = {
            name: datasets[name]
            for name in args.datasets
        }

    for dataset_name, (X, y) in datasets.items():
        for seed in args.seeds:
            print(f"\n{dataset_name} | seed={seed}")

            (
                X_train,
                X_test,
                y_train,
                y_test,
                _,
            ) = split_and_scale(
                X,
                y,
                seed=seed,
                test_size=0.30,
            )

            teacher = train_mlp_teacher(
                X_train,
                y_train,
                seed=seed,
                epochs=args.epochs,
            )

            train_logits = teacher_logits(teacher, X_train)
            test_logits = teacher_logits(teacher, X_test)

            teacher_train = (train_logits >= 0.0).astype(int)
            teacher_test = (test_logits >= 0.0).astype(int)

            teacher_accuracy = float(
                np.mean(teacher_test == y_test)
            )

            # Low confidence = close to teacher transition.
            confidence = np.abs(test_logits)

            # Independently defined geometric proxy.
            crossing_distance = segment_crossing_distances(
                teacher,
                X_test,
                teacher_test,
            )

            valid_crossing = np.isfinite(crossing_distance)

            print(
                f"  teacher accuracy={teacher_accuracy:.4f} | "
                f"valid crossing distances="
                f"{valid_crossing.sum()}/{len(X_test)}"
            )

            for depth in args.depths:
                surrogate = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=seed,
                )

                surrogate.fit(
                    X_train,
                    teacher_train,
                )

                surrogate_test = surrogate.predict(X_test)

                agreement = (
                    surrogate_test == teacher_test
                ).astype(float)

                global_fidelity = float(
                    np.mean(agreement)
                )

                aurc = compute_aurc(
                    confidence,
                    agreement,
                )

                for fraction in args.fractions:
                    confidence_mask = lowest_fraction_mask(
                        confidence,
                        fraction,
                    )

                    transition_mask = lowest_fraction_mask(
                        crossing_distance,
                        fraction,
                        valid=valid_crossing,
                    )

                    # Diagnostic: overlap between the two boundary definitions.
                    intersection = np.sum(
                        confidence_mask & transition_mask
                    )
                    union = np.sum(
                        confidence_mask | transition_mask
                    )

                    boundary_jaccard = (
                        intersection / union
                        if union > 0
                        else np.nan
                    )

                    for boundary_type, mask in (
                        ("confidence", confidence_mask),
                        ("segment_crossing", transition_mask),
                    ):
                        if not np.any(mask):
                            continue

                        boundary_fidelity = float(
                            np.mean(agreement[mask])
                        )

                        rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "depth": depth,
                                "teacher_accuracy": teacher_accuracy,
                                "n_test": len(X_test),
                                "global_fidelity": global_fidelity,
                                "aurc": aurc,
                                "boundary_type": boundary_type,
                                "boundary_fraction": fraction,
                                "n_boundary": int(mask.sum()),
                                "boundary_fidelity": boundary_fidelity,
                                "boundary_gap": (
                                    global_fidelity
                                    - boundary_fidelity
                                ),
                                "boundary_set_jaccard": float(
                                    boundary_jaccard
                                ),
                                "boundary_overlap_n": int(intersection),
                                "boundary_union_n": int(union)
                            }
                        )

                print(
                    f"  depth={depth:2d} | "
                    f"global={global_fidelity:.4f} | "
                    f"AURC={aurc:.4f}"
                )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rows:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {output}")


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run; default is all datasets.",
    )

    parser.add_argument(
        "--output",
        default="results/boundary_fidelity/runs.csv",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())