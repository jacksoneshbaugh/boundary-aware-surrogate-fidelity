"""
Generate paper-ready figures for Experiments 1--4.

Inputs
------
results/boundary_fidelity/runs.csv
results/distribution_shift_3bins/runs.csv
results/counterfactual_transfer/runs.csv
results/deep_boundary_stability/summary.csv

Outputs
-------
figures/
    fig_boundary_fidelity.pdf/png
    fig_distribution_shift.pdf/png
    fig_counterfactual_transfer.pdf/png
    fig_deep_boundary_stability.pdf/png
    appendix/
        fig_boundary_fidelity_families.pdf/png

Uncertainty intervals
---------------------
For experiments with multiple surrogate configurations per independent seed,
configurations are first averaged within each seed. Error bars therefore
reflect variation across seeds rather than treating multiple configurations
from the same seed as independent replicates.

Experiment 1 now contains two complementary views:

    1. fig_boundary_fidelity
       Main-text decision-tree capacity sweep.

    2. appendix/fig_boundary_fidelity_families
       Appendix cross-family robustness across all evaluated boundary
       fractions, using the designated representative configuration for each
       surrogate family.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = (
    ROOT
    / "results"
    / "boundary_fidelity"
    / "runs.csv"
)

SHIFT_FILE = (
    ROOT
    / "results"
    / "distribution_shift_3bins"
    / "runs.csv"
)

COUNTERFACTUAL_FILE = (
    ROOT
    / "results"
    / "counterfactual_transfer"
    / "runs.csv"
)

DEEP_STABILITY_FILE = (
    ROOT
    / "results"
    / "deep_boundary_stability"
    / "summary.csv"
)

FIGURE_DIR = (
    ROOT
    / "figures"
)

APPENDIX_FIGURE_DIR = (
    FIGURE_DIR
    / "appendix"
)


# =====================================================================
# Shared labels / style
# =====================================================================

DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "Breast_Cancer": "Breast Cancer",
    "wine": "Wine",
    "Wine": "Wine",
    "diabetes": "Diabetes",
    "Diabetes": "Diabetes",
    "iris": "Iris",
    "Iris": "Iris",
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
}

SURROGATE_LABELS = {
    "decision_tree": "Decision tree",
    "logistic_regression": "Logistic",
    "knn": "kNN",
    "mlp": "MLP",
}


def dataset_label(name):
    return DATASET_LABELS.get(
        name,
        str(name),
    )


def surrogate_label(name):
    return SURROGATE_LABELS.get(
        name,
        str(name)
        .replace("_", " ")
        .title(),
    )


def boundary_type_label(boundary_type):
    normalized = (
        str(boundary_type)
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized == "confidence":
        return "Confidence"

    if normalized in {
        "geometric",
        "segment_crossing",
        "segment_crossing_distance",
        "transition",
    }:
        return "Segment-crossing proxy"

    return (
        str(boundary_type)
        .replace("_", " ")
        .title()
    )


def normalize_bool_column(series):
    """
    Robustly recover booleans after CSV serialization.
    """
    if pd.api.types.is_bool_dtype(series):
        return series

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if mapped.isna().any():
        bad = sorted(
            series[
                mapped.isna()
            ]
            .astype(str)
            .unique()
        )

        raise ValueError(
            "Could not interpret boolean values: "
            f"{bad}"
        )

    return mapped


def preferred_order(
    available,
    preferred,
):
    """
    Return available values in a preferred order, followed by any extras.
    """
    available = list(
        pd.unique(available)
    )

    ordered = [
        value
        for value in preferred
        if value in available
    ]

    ordered.extend(
        value
        for value in available
        if value not in ordered
    )

    return ordered


def apply_paper_style():
    """
    Consistent compact styling suitable for conference figures.
    """
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.0,
    )

    plt.rcParams.update(
        {
            "font.family":
                "sans-serif",

            "font.size":
                8.5,

            "axes.titlesize":
                10,

            "axes.labelsize":
                9,

            "xtick.labelsize":
                8,

            "ytick.labelsize":
                8,

            "legend.fontsize":
                8,

            "axes.titleweight":
                "regular",

            "axes.spines.top":
                False,

            "axes.spines.right":
                False,

            "grid.alpha":
                0.22,

            "grid.linewidth":
                0.55,

            "lines.linewidth":
                1.6,

            "lines.markersize":
                5.5,

            "figure.dpi":
                150,

            "savefig.dpi":
                300,

            "pdf.fonttype":
                42,

            "ps.fonttype":
                42,
        }
    )


def mean_ci(values):
    """
    Mean and approximate 95% confidence interval across independent seeds.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    mean = float(
        np.mean(values)
    )

    if len(values) == 1:
        return mean, np.nan

    sem = (
        np.std(
            values,
            ddof=1,
        )
        / np.sqrt(
            len(values)
        )
    )

    return (
        mean,
        float(
            1.96 * sem
        ),
    )


def save_figure(
    fig,
    stem,
    output_dir=FIGURE_DIR,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = (
        output_dir
        / f"{stem}.pdf"
    )

    png = (
        output_dir
        / f"{stem}.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    print(
        f"Saved {pdf}"
    )

    print(
        f"Saved {png}"
    )


# =====================================================================
# Figure 1:
# Boundary-local fidelity degradation across tree capacities
# =====================================================================

def make_boundary_figure():
    df = pd.read_csv(
        BOUNDARY_FILE
    )

    required = {
        "dataset",
        "seed",
        "surrogate_family",
        "is_tree_capacity_sweep",
        "depth",
        "boundary_fraction",
        "boundary_type",
        "global_fidelity",
        "boundary_fidelity",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Boundary results missing columns: "
            f"{sorted(missing)}"
        )

    df[
        "is_tree_capacity_sweep"
    ] = normalize_bool_column(
        df[
            "is_tree_capacity_sweep"
        ]
    )

    # -------------------------------------------------------------
    # Preserve the original Experiment 1 comparison.
    #
    # Only decision-tree capacity-sweep rows belong here.
    # The additional surrogate families must not be averaged into
    # this figure.
    # -------------------------------------------------------------

    df = df[
        (
            df[
                "surrogate_family"
            ]
            == "decision_tree"
        )
        & df[
            "is_tree_capacity_sweep"
        ]
    ].copy()

    if df.empty:
        raise ValueError(
            "No decision-tree capacity-sweep rows "
            "were found for Figure 1."
        )

    df["boundary_gap"] = (
        df["global_fidelity"]
        - df["boundary_fidelity"]
    )

    # Multiple tree depths are configurations within a seed,
    # not independent replicates.
    seed_level = (
        df.groupby(
            [
                "dataset",
                "seed",
                "boundary_fraction",
                "boundary_type",
            ],
            as_index=False,
        )[
            "boundary_gap"
        ]
        .mean()
    )

    datasets = preferred_order(
        seed_level[
            "dataset"
        ].unique(),
        [
            "breast_cancer",
            "diabetes",
            "wine",
            "iris",
        ],
    )

    boundary_types = preferred_order(
        seed_level[
            "boundary_type"
        ].unique(),
        [
            "confidence",
            "segment_crossing",
        ],
    )

    palette = sns.color_palette(
        "colorblind",
        n_colors=max(
            2,
            len(boundary_types),
        ),
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(7.15, 2.25),
        sharey=True,
    )

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(
        axes,
        datasets,
    ):
        subset = seed_level[
            seed_level[
                "dataset"
            ]
            == dataset
        ]

        for index, boundary_type in enumerate(
            boundary_types
        ):
            part = subset[
                subset[
                    "boundary_type"
                ]
                == boundary_type
            ]

            if part.empty:
                continue

            rows = []

            for (
                fraction,
                group,
            ) in part.groupby(
                "boundary_fraction"
            ):
                mean, ci = mean_ci(
                    group[
                        "boundary_gap"
                    ]
                )

                rows.append(
                    {
                        "fraction":
                            fraction,

                        "mean":
                            mean,

                        "ci":
                            ci,
                    }
                )

            curve = (
                pd.DataFrame(
                    rows
                )
                .sort_values(
                    "fraction"
                )
            )

            ax.errorbar(
                curve[
                    "fraction"
                ] * 100,

                curve[
                    "mean"
                ] * 100,

                yerr=(
                    curve[
                        "ci"
                    ] * 100
                ),

                color=palette[
                    index
                ],

                marker=markers[
                    index
                ],

                markeredgewidth=0.7,
                capsize=2.5,
                elinewidth=1.0,
                linewidth=1.6,

                label=boundary_type_label(
                    boundary_type
                ),
            )

        ax.axhline(
            0,
            color="0.45",
            linewidth=0.75,
            linestyle="--",
            zorder=0,
        )

        ax.set_title(
            dataset_label(
                dataset
            ),
            pad=6,
        )

        ax.set_xlabel(
            "Boundary fraction (%)"
        )

        ax.grid(
            axis="x",
            visible=False,
        )

        sns.despine(
            ax=ax,
        )

    axes[0].set_ylabel(
        "Global − boundary fidelity (pp)"
    )

    handles, labels = (
        axes[-1]
        .get_legend_handles_labels()
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.07,
            ),
            ncol=len(
                handles
            ),
            frameon=False,
        )

    fig.tight_layout(
        pad=0.6,
        rect=(
            0,
            0,
            1,
            0.93,
        ),
    )

    save_figure(
        fig,
        "fig_boundary_fidelity",
    )

    plt.close(
        fig
    )


# =====================================================================
# Appendix Figure B.1:
# Surrogate-family robustness across boundary fractions
# =====================================================================

def make_boundary_family_fraction_figure():
    """
    Appendix figure:
    surrogate-family robustness across all evaluated boundary fractions.

    Rows correspond to boundary-proximity definitions:
        1. Confidence
        2. Segment-crossing proxy

    Columns correspond to datasets.

    Curves show mean global-to-boundary fidelity gaps across ten seeds,
    with approximate 95% confidence intervals.
    """
    df = pd.read_csv(
        BOUNDARY_FILE
    )

    required = {
        "dataset",
        "seed",
        "surrogate_family",
        "surrogate_config",
        "is_family_reference",
        "boundary_fraction",
        "boundary_type",
        "global_fidelity",
        "boundary_fidelity",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Boundary results missing columns "
            "for surrogate-family appendix figure: "
            f"{sorted(missing)}"
        )

    df[
        "is_family_reference"
    ] = normalize_bool_column(
        df[
            "is_family_reference"
        ]
    )

    # Only the designated representative configuration
    # from each surrogate family belongs in this figure.
    df = df[
        df[
            "is_family_reference"
        ]
    ].copy()

    if df.empty:
        raise ValueError(
            "No family-reference rows were found "
            "for the appendix robustness figure."
        )

    df["boundary_gap"] = (
        df["global_fidelity"]
        - df["boundary_fidelity"]
    )

    datasets = preferred_order(
        df[
            "dataset"
        ].unique(),
        [
            "breast_cancer",
            "diabetes",
            "wine",
            "iris",
        ],
    )

    families = preferred_order(
        df[
            "surrogate_family"
        ].unique(),
        [
            "decision_tree",
            "logistic_regression",
            "knn",
            "mlp",
        ],
    )

    boundary_types = preferred_order(
        df[
            "boundary_type"
        ].unique(),
        [
            "confidence",
            "segment_crossing",
        ],
    )

    palette = sns.color_palette(
        "colorblind",
        n_colors=len(families),
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    fig, axes = plt.subplots(
        len(boundary_types),
        len(datasets),
        figsize=(7.4, 4.4),
        sharey=True,
        sharex=True,
    )

    # Handle degenerate cases cleanly.
    axes = np.atleast_2d(
        axes
    )

    for row_index, boundary_type in enumerate(
        boundary_types
    ):
        for col_index, dataset in enumerate(
            datasets
        ):
            ax = axes[
                row_index,
                col_index,
            ]

            subset = df[
                (
                    df[
                        "dataset"
                    ]
                    == dataset
                )
                & (
                    df[
                        "boundary_type"
                    ]
                    == boundary_type
                )
            ]

            for family_index, family in enumerate(
                families
            ):
                part = subset[
                    subset[
                        "surrogate_family"
                    ]
                    == family
                ]

                if part.empty:
                    continue

                rows = []

                for (
                    fraction,
                    group,
                ) in part.groupby(
                    "boundary_fraction"
                ):
                    mean, ci = mean_ci(
                        group[
                            "boundary_gap"
                        ]
                    )

                    rows.append(
                        {
                            "fraction":
                                fraction,

                            "mean":
                                mean,

                            "ci":
                                ci,
                        }
                    )

                curve = (
                    pd.DataFrame(
                        rows
                    )
                    .sort_values(
                        "fraction"
                    )
                )

                ax.errorbar(
                    curve[
                        "fraction"
                    ] * 100,

                    curve[
                        "mean"
                    ] * 100,

                    yerr=(
                        curve[
                            "ci"
                        ] * 100
                    ),

                    color=palette[
                        family_index
                    ],

                    marker=markers[
                        family_index
                    ],

                    markeredgewidth=0.7,
                    capsize=2.5,
                    elinewidth=1.0,
                    linewidth=1.6,

                    label=surrogate_label(
                        family
                    ),
                )

            ax.axhline(
                0,
                color="0.45",
                linewidth=0.75,
                linestyle="--",
                zorder=0,
            )

            if row_index == 0:
                ax.set_title(
                    dataset_label(
                        dataset
                    ),
                    pad=6,
                )

            if row_index == (
                len(boundary_types) - 1
            ):
                ax.set_xlabel(
                    "Boundary fraction (%)"
                )

            ax.set_xticks(
                [10, 20, 25, 40]
            )

            ax.grid(
                axis="x",
                visible=False,
            )

            sns.despine(
                ax=ax,
            )

    # -------------------------------------------------------------
    # Row identifiers.
    #
    # These are categorical facet labels rather than axis labels,
    # so place them inside the first panel of each row.
    # -------------------------------------------------------------

    row_labels = {
        "confidence":
            "Confidence",

        "segment_crossing":
            "Segment-crossing proxy",
    }

    for row_index, boundary_type in enumerate(
        boundary_types
    ):
        axes[
            row_index,
            0,
        ].text(
            0.04,
            0.93,
            row_labels.get(
                boundary_type,
                boundary_type_label(
                    boundary_type
                ),
            ),

            transform=axes[
                row_index,
                0,
            ].transAxes,

            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="semibold",

            bbox={
                "facecolor":
                    "white",

                "edgecolor":
                    "none",

                "alpha":
                    0.85,

                "pad":
                    1.5,
            },
        )

    # One shared quantitative y-axis label for both rows.
    fig.supylabel(
        "Global − boundary fidelity (pp)",
        x=0.012,
        fontsize=9,
    )

    # Shared legend across the top.
    handles, labels = (
        axes[
            0,
            -1,
        ]
        .get_legend_handles_labels()
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.01,
            ),
            ncol=len(
                handles
            ),
            frameon=False,
        )

    fig.tight_layout(
        pad=0.6,
        rect=(
            0.035,
            0,
            1,
            0.94,
        ),
    )

    save_figure(
        fig,
        "fig_boundary_fidelity_families",
        output_dir=APPENDIX_FIGURE_DIR,
    )

    plt.close(
        fig
    )


# =====================================================================
# Figure 2:
# Prediction under distribution shift
# =====================================================================

def make_distribution_shift_figure():
    df = pd.read_csv(
        SHIFT_FILE
    )

    required = {
        "dataset",
        "seed",
        "depth",
        "shift",
        "error_global",
        "error_profile",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Distribution-shift results "
            "missing columns: "
            f"{sorted(missing)}"
        )

    # Surrogate depths are configurations within a seed,
    # not independent replicates.
    seed_level = (
        df.groupby(
            [
                "dataset",
                "seed",
                "shift",
            ],
            as_index=False,
        )
        .agg(
            global_error=(
                "error_global",
                "mean",
            ),
            profile_error=(
                "error_profile",
                "mean",
            ),
        )
    )

    datasets = preferred_order(
        seed_level[
            "dataset"
        ].unique(),
        [
            "breast_cancer",
            "diabetes",
            "wine",
        ],
    )

    shifts = preferred_order(
        seed_level[
            "shift"
        ].unique(),
        [
            "hard",
            "easy",
        ],
    )

    palette = sns.color_palette(
        "colorblind",
        2,
    )

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.8, 2.35),
        sharey=True,
    )

    if len(datasets) == 1:
        axes = [axes]

    x = np.arange(
        len(shifts)
    )

    offset = 0.10

    for ax, dataset in zip(
        axes,
        datasets,
    ):
        part = seed_level[
            seed_level[
                "dataset"
            ]
            == dataset
        ]

        global_means = []
        global_cis = []

        profile_means = []
        profile_cis = []

        for shift in shifts:
            group = part[
                part[
                    "shift"
                ]
                == shift
            ]

            global_mean, global_ci = mean_ci(
                group[
                    "global_error"
                ]
            )

            profile_mean, profile_ci = mean_ci(
                group[
                    "profile_error"
                ]
            )

            global_means.append(
                global_mean * 100
            )

            global_cis.append(
                global_ci * 100
            )

            profile_means.append(
                profile_mean * 100
            )

            profile_cis.append(
                profile_ci * 100
            )

        ax.errorbar(
            x - offset,
            global_means,
            yerr=global_cis,
            fmt="o",
            color=palette[0],
            markersize=5.8,
            markeredgewidth=0.7,
            capsize=2.5,
            elinewidth=1.1,
            linewidth=0,
            label="Global fidelity",
        )

        ax.errorbar(
            x + offset,
            profile_means,
            yerr=profile_cis,
            fmt="s",
            color=palette[1],
            markersize=5.8,
            markeredgewidth=0.7,
            capsize=2.5,
            elinewidth=1.1,
            linewidth=0,
            label="Stratified profile",
        )

        ax.set_xticks(
            x,
            [
                shift.capitalize()
                for shift in shifts
            ],
        )

        ax.set_title(
            dataset_label(
                dataset
            ),
            pad=6,
        )

        ax.set_xlabel(
            "Target composition"
        )

        ax.grid(
            axis="x",
            visible=False,
        )

        sns.despine(
            ax=ax,
        )

    axes[0].set_ylabel(
        "Target-fidelity prediction error (pp)"
    )

    handles, labels = (
        axes[-1]
        .get_legend_handles_labels()
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.07,
            ),
            ncol=2,
            frameon=False,
        )

    fig.tight_layout(
        pad=0.6,
        rect=(
            0,
            0,
            1,
            0.93,
        ),
    )

    save_figure(
        fig,
        "fig_distribution_shift",
    )

    plt.close(
        fig
    )


# =====================================================================
# Figure 3:
# Counterfactual transfer
# =====================================================================

def find_matched_example(df):
    """
    Find a within-dataset, within-seed pair of surrogate depths with
    nearly identical global AND boundary fidelity but maximally different
    counterfactual transfer rates.
    """
    best = None

    for (
        dataset,
        seed,
    ), group in df.groupby(
        [
            "dataset",
            "seed",
        ]
    ):
        rows = list(
            group.to_dict(
                "records"
            )
        )

        for i in range(
            len(rows)
        ):
            for j in range(
                i + 1,
                len(rows),
            ):
                a = rows[i]
                b = rows[j]

                global_delta = abs(
                    a[
                        "global_fidelity"
                    ]
                    - b[
                        "global_fidelity"
                    ]
                )

                boundary_delta = abs(
                    a[
                        "boundary_fidelity"
                    ]
                    - b[
                        "boundary_fidelity"
                    ]
                )

                # Near-match:
                # <= 1 percentage point on both
                # observational fidelity measures.
                if (
                    global_delta > 0.01
                    or boundary_delta > 0.01
                ):
                    continue

                transfer_delta = abs(
                    a[
                        "recourse_transfer_rate"
                    ]
                    - b[
                        "recourse_transfer_rate"
                    ]
                )

                if (
                    best is None
                    or transfer_delta
                    > best[
                        "transfer_delta"
                    ]
                ):
                    best = {
                        "dataset":
                            dataset,

                        "seed":
                            seed,

                        "a":
                            a,

                        "b":
                            b,

                        "transfer_delta":
                            transfer_delta,
                    }

    return best


def make_counterfactual_figure():
    df = pd.read_csv(
        COUNTERFACTUAL_FILE
    )

    required = {
        "dataset",
        "seed",
        "depth",
        "global_fidelity",
        "boundary_fidelity",
        "recourse_transfer_rate",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Counterfactual results missing columns: "
            f"{sorted(missing)}"
        )

    datasets = preferred_order(
        df[
            "dataset"
        ].unique(),
        [
            "breast_cancer",
            "wine",
            "diabetes",
        ],
    )

    palette = sns.color_palette(
        "colorblind",
        n_colors=len(
            datasets
        ),
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    fig, ax = plt.subplots(
        figsize=(4.7, 3.3)
    )

    for index, dataset in enumerate(
        datasets
    ):
        part = df[
            df[
                "dataset"
            ]
            == dataset
        ]

        ax.scatter(
            part[
                "global_fidelity"
            ] * 100,

            part[
                "recourse_transfer_rate"
            ] * 100,

            color=palette[
                index
            ],

            marker=markers[
                index
            ],

            s=30,
            alpha=0.58,

            edgecolors=palette[
                index
            ],

            linewidths=0.6,

            label=dataset_label(
                dataset
            ),
        )

    matched = find_matched_example(
        df
    )

    if matched is not None:
        a = matched[
            "a"
        ]

        b = matched[
            "b"
        ]

        # Put the larger-transfer configuration first.
        if (
            a[
                "recourse_transfer_rate"
            ]
            < b[
                "recourse_transfer_rate"
            ]
        ):
            a, b = b, a

        x1 = (
            a[
                "global_fidelity"
            ]
            * 100
        )

        y1 = (
            a[
                "recourse_transfer_rate"
            ]
            * 100
        )

        x2 = (
            b[
                "global_fidelity"
            ]
            * 100
        )

        y2 = (
            b[
                "recourse_transfer_rate"
            ]
            * 100
        )

        ax.plot(
            [
                x1,
                x2,
            ],
            [
                y1,
                y2,
            ],
            color="0.20",
            linewidth=1.6,
            zorder=4,
        )

        ax.scatter(
            [
                x1,
                x2,
            ],
            [
                y1,
                y2,
            ],
            facecolors="white",
            edgecolors="0.15",
            linewidths=1.25,
            s=64,
            zorder=5,
        )

        global_mean = (
            0.5
            * (
                a[
                    "global_fidelity"
                ]
                + b[
                    "global_fidelity"
                ]
            )
            * 100
        )

        boundary_mean = (
            0.5
            * (
                a[
                    "boundary_fidelity"
                ]
                + b[
                    "boundary_fidelity"
                ]
            )
            * 100
        )

        annotation = (
            f"{dataset_label(matched['dataset'])}, "
            f"seed {matched['seed']}\n"
            f"Same global fidelity: "
            f"{global_mean:.1f}%\n"
            f"Same boundary fidelity: "
            f"{boundary_mean:.1f}%\n"
            f"d={int(a['depth'])}: "
            f"{y1:.1f}% transfer\n"
            f"d={int(b['depth'])}: "
            f"{y2:.1f}% transfer"
        )

        ax.annotate(
            annotation,

            xy=(
                (
                    x1 + x2
                )
                / 2,

                (
                    y1 + y2
                )
                / 2,
            ),

            xytext=(
                18,
                0,
            ),

            textcoords="offset points",
            va="center",
            fontsize=7.5,

            bbox={
                "boxstyle":
                    "round,pad=0.35",

                "facecolor":
                    "white",

                "edgecolor":
                    "0.75",

                "alpha":
                    0.95,

                "linewidth":
                    0.7,
            },
        )

        print()

        print(
            "Matched-fidelity "
            "counterfactual example:"
        )

        print(
            f"  dataset = "
            f"{matched['dataset']}"
        )

        print(
            f"  seed = "
            f"{matched['seed']}"
        )

        print(
            f"  depths = "
            f"{a['depth']} vs "
            f"{b['depth']}"
        )

        print(
            f"  global fidelity = "
            f"{a['global_fidelity']:.4f} "
            f"vs "
            f"{b['global_fidelity']:.4f}"
        )

        print(
            f"  boundary fidelity = "
            f"{a['boundary_fidelity']:.4f} "
            f"vs "
            f"{b['boundary_fidelity']:.4f}"
        )

        print(
            f"  transfer rate = "
            f"{a['recourse_transfer_rate']:.4f} "
            f"vs "
            f"{b['recourse_transfer_rate']:.4f}"
        )

    ax.set_xlabel(
        "Global fidelity (%)"
    )

    ax.set_ylabel(
        "Counterfactual transfer (%)"
    )

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    ax.grid(
        alpha=0.20,
        linewidth=0.55,
    )

    sns.despine(
        ax=ax,
    )

    fig.tight_layout(
        pad=0.6,
    )

    save_figure(
        fig,
        "fig_counterfactual_transfer",
    )

    plt.close(
        fig
    )


# =====================================================================
# Figure 4:
# Deep-teacher boundary stability
# =====================================================================

def make_deep_stability_figure():
    df = pd.read_csv(
        DEEP_STABILITY_FILE
    )

    required = {
        "dataset",
        "boundary_fraction",
        "boundary_jaccard_mean",
        "global_minus_union_agreement_mean",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Deep-stability results missing columns: "
            f"{sorted(missing)}"
        )

    datasets = preferred_order(
        df[
            "dataset"
        ].unique(),
        [
            "mnist",
            "cifar10",
        ],
    )

    palette = sns.color_palette(
        "colorblind",
        n_colors=len(
            datasets
        ),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.6, 2.55),
    )

    # -------------------------------------------------------------
    # Panel A:
    # Boundary-set overlap
    # -------------------------------------------------------------

    ax = axes[0]

    for index, dataset in enumerate(
        datasets
    ):
        part = (
            df[
                df[
                    "dataset"
                ]
                == dataset
            ]
            .sort_values(
                "boundary_fraction"
            )
        )

        ax.plot(
            part[
                "boundary_fraction"
            ] * 100,

            part[
                "boundary_jaccard_mean"
            ] * 100,

            marker="o",
            color=palette[
                index
            ],
            linewidth=1.7,
            markersize=5.5,

            label=dataset_label(
                dataset
            ),
        )

    ax.set_xlabel(
        "Boundary fraction (%)"
    )

    ax.set_ylabel(
        "Boundary-set Jaccard (%)"
    )

    ax.set_title(
        "Boundary-set stability"
    )

    # Keep an absolute 0--100 scale so overlap
    # is not visually exaggerated.
    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.legend(
        frameon=False,
    )

    sns.despine(
        ax=ax,
    )

    # -------------------------------------------------------------
    # Panel B:
    # Loss of predictive agreement near boundary
    # -------------------------------------------------------------

    ax = axes[1]

    for index, dataset in enumerate(
        datasets
    ):
        part = (
            df[
                df[
                    "dataset"
                ]
                == dataset
            ]
            .sort_values(
                "boundary_fraction"
            )
        )

        ax.plot(
            part[
                "boundary_fraction"
            ] * 100,

            part[
                "global_minus_union_agreement_mean"
            ] * 100,

            marker="o",
            color=palette[
                index
            ],
            linewidth=1.7,
            markersize=5.5,

            label=dataset_label(
                dataset
            ),
        )

    ax.axhline(
        0,
        color="0.45",
        linewidth=0.75,
        linestyle="--",
        zorder=0,
    )

    ax.set_xlabel(
        "Boundary fraction (%)"
    )

    ax.set_ylabel(
        "Global − union agreement (pp)"
    )

    ax.set_title(
        "Agreement drop on boundary-set union"
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.legend(
        frameon=False,
    )

    sns.despine(
        ax=ax,
    )

    fig.tight_layout(
        pad=0.7,
    )

    save_figure(
        fig,
        "fig_deep_boundary_stability",
    )

    plt.close(
        fig
    )


# =====================================================================
# Main
# =====================================================================

def main():
    apply_paper_style()

    print(
        "Generating Experiment 1 "
        "tree-capacity figure..."
    )

    make_boundary_figure()

    print()

    print(
        "Generating Experiment 1 appendix "
        "surrogate-family fraction figure..."
    )

    make_boundary_family_fraction_figure()

    print()

    print(
        "Generating Experiment 2 figure..."
    )

    make_distribution_shift_figure()

    print()

    print(
        "Generating Experiment 3 figure..."
    )

    make_counterfactual_figure()

    print()

    print(
        "Generating Experiment 4 figure..."
    )

    make_deep_stability_figure()

    print()

    print(
        "Done."
    )


if __name__ == "__main__":
    main()
