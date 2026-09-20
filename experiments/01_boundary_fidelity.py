"""
Experiment 1: Boundary-aware characterization of surrogate fidelity.

This experiment tests whether global teacher-surrogate agreement masks
systematically lower fidelity near the teacher's decision boundary.

The experiment contains two complementary robustness analyses:

    1. Decision-tree capacity sweep
       Decision-tree surrogates are evaluated across multiple maximum depths.

    2. Surrogate-family robustness
       Representative surrogates from several model families are compared:
           - decision tree
           - logistic regression
           - k-nearest neighbors
           - multilayer perceptron

For each tabular dataset, teacher seed, and surrogate configuration, the
script measures:

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

The experiment is repeated across multiple boundary-region widths.

All individual runs are written to a tidy results file so that the tree
capacity sweep and the surrogate-family robustness analysis can be summarized
separately without retraining models.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
DEFAULT_TREE_DEPTHS = [1, 2, 3, 5, 7, 10]
DEFAULT_FRACTIONS = [0.10, 0.20, 0.25, 0.40]

DEFAULT_FAMILIES = [
    "decision_tree",
    "logistic_regression",
    "knn",
    "mlp",
]


@dataclass(frozen=True)
class SurrogateSpec:
    family: str
    name: str
    complexity_name: str
    complexity_value: float
    is_family_reference: bool
    is_tree_capacity_sweep: bool


def build_surrogate_specs(args):
    """
    Construct the surrogate configurations used in Experiment 1.

    Decision trees retain the original depth sweep. The depth selected by
    --tree-reference-depth is also used as the decision-tree representative
    in the cross-family robustness comparison.

    The other families use one representative configuration each. These are
    intended as robustness checks across substantially different inductive
    biases, not as exhaustive hyperparameter sweeps.
    """
    specs = []

    if "decision_tree" in args.families:
        if args.tree_reference_depth not in args.tree_depths:
            raise ValueError(
                "--tree-reference-depth must be included in --tree-depths."
            )

        for depth in args.tree_depths:
            specs.append(
                SurrogateSpec(
                    family="decision_tree",
                    name=f"tree_depth_{depth}",
                    complexity_name="max_depth",
                    complexity_value=float(depth),
                    is_family_reference=(
                        depth == args.tree_reference_depth
                    ),
                    is_tree_capacity_sweep=True,
                )
            )

    if "logistic_regression" in args.families:
        specs.append(
            SurrogateSpec(
                family="logistic_regression",
                name=f"logistic_C_{args.logistic_c:g}",
                complexity_name="C",
                complexity_value=float(args.logistic_c),
                is_family_reference=True,
                is_tree_capacity_sweep=False,
            )
        )

    if "knn" in args.families:
        specs.append(
            SurrogateSpec(
                family="knn",
                name=f"knn_k_{args.knn_k}",
                complexity_name="n_neighbors",
                complexity_value=float(args.knn_k),
                is_family_reference=True,
                is_tree_capacity_sweep=False,
            )
        )

    if "mlp" in args.families:
        specs.append(
            SurrogateSpec(
                family="mlp",
                name=f"mlp_width_{args.mlp_width}",
                complexity_name="hidden_width",
                complexity_value=float(args.mlp_width),
                is_family_reference=True,
                is_tree_capacity_sweep=False,
            )
        )

    return specs


def make_surrogate(spec, seed, args):
    """
    Instantiate one surrogate model from a SurrogateSpec.
    """
    if spec.family == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=int(spec.complexity_value),
            random_state=seed,
        )

    if spec.family == "logistic_regression":
        return LogisticRegression(
            C=args.logistic_c,
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )

    if spec.family == "knn":
        return KNeighborsClassifier(
            n_neighbors=args.knn_k,
        )

    if spec.family == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(args.mlp_width,),
            activation="relu",
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown surrogate family: {spec.family}"
    )


def run(args):
    rows = []

    datasets = load_tabular_datasets()

    if args.datasets is not None:
        datasets = {
            name: datasets[name]
            for name in args.datasets
        }

    surrogate_specs = build_surrogate_specs(args)

    for dataset_name, (X, y) in datasets.items():
        for seed in args.seeds:
            print()
            print(
                f"{dataset_name} | seed={seed}"
            )

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

            # ---------------------------------------------------------
            # Teacher
            # ---------------------------------------------------------

            teacher = train_mlp_teacher(
                X_train,
                y_train,
                seed=seed,
                epochs=args.epochs,
            )

            train_logits = teacher_logits(
                teacher,
                X_train,
            )

            test_logits = teacher_logits(
                teacher,
                X_test,
            )

            teacher_train = (
                train_logits >= 0.0
            ).astype(int)

            teacher_test = (
                test_logits >= 0.0
            ).astype(int)

            teacher_accuracy = float(
                np.mean(
                    teacher_test == y_test
                )
            )

            # ---------------------------------------------------------
            # Teacher-relative boundary definitions
            # ---------------------------------------------------------

            # Confidence-based proximity:
            # smaller |logit| = closer to the teacher transition.
            confidence = np.abs(
                test_logits
            )

            # Confidence-independent geometric proxy.
            crossing_distance = (
                segment_crossing_distances(
                    teacher,
                    X_test,
                    teacher_test,
                )
            )

            valid_crossing = np.isfinite(
                crossing_distance
            )

            print(
                f"  teacher accuracy="
                f"{teacher_accuracy:.4f} | "
                f"valid crossing distances="
                f"{valid_crossing.sum()}/"
                f"{len(X_test)}"
            )

            # ---------------------------------------------------------
            # Boundary masks depend only on the teacher, not surrogate.
            # Compute them once and reuse them for every family.
            # ---------------------------------------------------------

            boundary_masks = {}

            for fraction in args.fractions:
                confidence_mask = (
                    lowest_fraction_mask(
                        confidence,
                        fraction,
                    )
                )

                transition_mask = (
                    lowest_fraction_mask(
                        crossing_distance,
                        fraction,
                        valid=valid_crossing,
                    )
                )

                intersection = int(
                    np.sum(
                        confidence_mask
                        & transition_mask
                    )
                )

                union = int(
                    np.sum(
                        confidence_mask
                        | transition_mask
                    )
                )

                boundary_jaccard = (
                    intersection / union
                    if union > 0
                    else np.nan
                )

                boundary_masks[fraction] = {
                    "confidence":
                        confidence_mask,
                    "segment_crossing":
                        transition_mask,
                    "jaccard":
                        float(boundary_jaccard),
                    "intersection":
                        intersection,
                    "union":
                        union,
                }

            # ---------------------------------------------------------
            # Surrogates
            # ---------------------------------------------------------

            for spec in surrogate_specs:
                surrogate = make_surrogate(
                    spec,
                    seed,
                    args,
                )

                surrogate.fit(
                    X_train,
                    teacher_train,
                )

                surrogate_test = (
                    surrogate.predict(
                        X_test
                    )
                )

                agreement = (
                    surrogate_test
                    == teacher_test
                ).astype(float)

                global_fidelity = float(
                    np.mean(agreement)
                )

                aurc = compute_aurc(
                    confidence,
                    agreement,
                )

                for fraction in args.fractions:
                    masks = boundary_masks[
                        fraction
                    ]

                    for (
                        boundary_type,
                        mask,
                    ) in (
                        (
                            "confidence",
                            masks["confidence"],
                        ),
                        (
                            "segment_crossing",
                            masks[
                                "segment_crossing"
                            ],
                        ),
                    ):
                        if not np.any(mask):
                            continue

                        boundary_fidelity = float(
                            np.mean(
                                agreement[mask]
                            )
                        )

                        # Keep the old depth column for tree-specific
                        # downstream analyses while adding a general schema.
                        depth = (
                            int(
                                spec.complexity_value
                            )
                            if spec.family
                            == "decision_tree"
                            else ""
                        )

                        rows.append(
                            {
                                "dataset":
                                    dataset_name,

                                "seed":
                                    seed,

                                "surrogate_family":
                                    spec.family,

                                "surrogate_config":
                                    spec.name,

                                "complexity_name":
                                    spec.complexity_name,

                                "complexity_value":
                                    spec.complexity_value,

                                "is_family_reference":
                                    spec.is_family_reference,

                                "is_tree_capacity_sweep":
                                    spec.is_tree_capacity_sweep,

                                # Backward-compatible tree column.
                                "depth":
                                    depth,

                                "teacher_accuracy":
                                    teacher_accuracy,

                                "n_test":
                                    len(X_test),

                                "global_fidelity":
                                    global_fidelity,

                                "aurc":
                                    aurc,

                                "boundary_type":
                                    boundary_type,

                                "boundary_fraction":
                                    fraction,

                                "n_boundary":
                                    int(mask.sum()),

                                "boundary_fidelity":
                                    boundary_fidelity,

                                "boundary_gap":
                                    (
                                        global_fidelity
                                        - boundary_fidelity
                                    ),

                                "boundary_set_jaccard":
                                    masks["jaccard"],

                                "boundary_overlap_n":
                                    masks[
                                        "intersection"
                                    ],

                                "boundary_union_n":
                                    masks["union"],
                            }
                        )

                print(
                    f"  {spec.name:20s} | "
                    f"global="
                    f"{global_fidelity:.4f} | "
                    f"AURC="
                    f"{aurc:.4f}"
                )

    # -------------------------------------------------------------
    # Write tidy results
    # -------------------------------------------------------------

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
        f"\nSaved {len(rows)} rows "
        f"to {output}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--tree-depths",
        "--depths",
        nargs="+",
        type=int,
        default=DEFAULT_TREE_DEPTHS,
    )

    parser.add_argument(
        "--tree-reference-depth",
        type=int,
        default=5,
        help=(
            "Tree depth used as the representative "
            "decision tree in the cross-family comparison."
        ),
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=DEFAULT_FAMILIES,
        default=DEFAULT_FAMILIES,
    )

    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--mlp-width",
        type=int,
        default=32,
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
        help=(
            "Datasets to run; "
            "default is all datasets."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/boundary_fidelity/"
            "runs.csv"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())