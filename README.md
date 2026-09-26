# "Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity

Code and experimental artifacts for the paper **"Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity**.

## Overview

This repository contains the experiments, result summaries, and figure-generation code used in the paper.

The paper studies four questions:

1. **Boundary-local fidelity:** Is teacher--surrogate disagreement systematically concentrated near the teacher's decision boundary?
2. **Fidelity prediction under changes in evaluation composition:** Does retaining agreement across teacher-confidence regions improve prediction of fidelity when the composition of the evaluation set changes?
3. **Counterfactual transfer:** Do surrogates with similar global and boundary-local fidelity necessarily behave similarly under changes selected using the surrogate?
4. **Deep-teacher boundary stability:** Do independently trained deep teachers identify the same examples as lying near their decision boundaries?

For the tabular experiments, boundary proximity is measured in two ways:

- **Confidence-based proximity:** examples with the smallest absolute teacher logits.
- **Approximate segment-crossing proximity:** examples closest to an estimated teacher prediction transition along a segment to a nearby oppositely predicted example.

## Repository Structure

```text
experiments/
    01_boundary_fidelity.py
    02_distribution_shift.py
    03_counterfactual_transfer.py
    04_deep_boundary_stability.py
    common.py

scripts/
    make_paper_figures.py
    summarize_boundary_fidelity.py
    summarize_distribution_shift.py
    summarize_counterfactual_transfer.py
    summarize_deep_boundary_stability.py

results/
    boundary_fidelity/
    distribution_shift_3bins/
    counterfactual_transfer/
    deep_boundary_stability/

figures/                          # generated paper figures

hpc/
    setup_venv.sh
    slurm/                        # SLURM launch scripts

notebooks/
    exploratory/                  # exploratory analyses not required for reproduction
    paper_figures.ipynb
```

The numbered scripts in `experiments/` correspond directly to Sections 4.1--4.4 of the paper.

## Environment

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

The deep-teacher experiments use PyTorch and will use CUDA automatically when available.

## Reproducing the Main Experiments

Run commands from the repository root.

### Experiment 1: Boundary-local structure of surrogate disagreement

```bash
python experiments/01_boundary_fidelity.py
```

This runs the decision-tree capacity sweep and the representative cross-family robustness analysis over the tabular datasets.

Outputs:

```text
results/boundary_fidelity/runs.csv
```

To regenerate the summary tables:

```bash
python scripts/summarize_boundary_fidelity.py
```

which writes:

```text
results/boundary_fidelity/summary.csv
results/boundary_fidelity/family_summary.csv
```

### Experiment 2: Predicting fidelity under changes in evaluation composition

The primary paper analysis uses three teacher-confidence bins:

```bash
python experiments/02_distribution_shift.py \
    --bins 3 \
    --output results/distribution_shift_3bins/runs.csv
```

For the representative surrogate-family robustness analysis:

```bash
python experiments/02_distribution_shift.py \
    --bins 3 \
    --family-check \
    --output results/distribution_shift_3bins/family_runs.csv
```

The five-bin sensitivity analysis can be reproduced with:

```bash
python experiments/02_distribution_shift.py \
    --bins 5 \
    --output results/distribution_shift_5bins/runs.csv
```

### Experiment 3: Example-based counterfactual transfer

```bash
python experiments/03_counterfactual_transfer.py \
    --query-output results/counterfactual_transfer/queries.csv
```

Outputs include:

```text
results/counterfactual_transfer/runs.csv
results/counterfactual_transfer/queries.csv
```

To regenerate the summary:

```bash
python scripts/summarize_counterfactual_transfer.py
```

### Experiment 4: Boundary stability across deep teachers

Train five independently initialized teachers for MNIST:

```bash
python experiments/04_deep_boundary_stability.py --dataset mnist
```

and five for CIFAR-10:

```bash
python experiments/04_deep_boundary_stability.py --dataset cifar10
```

The default training schedules are 30 epochs for MNIST and 200 epochs for CIFAR-10.

After the teacher runs finish, compute the pairwise stability statistics:

```bash
python scripts/summarize_deep_boundary_stability.py
```

This produces:

```text
results/deep_boundary_stability/models.csv
results/deep_boundary_stability/pairs.csv
results/deep_boundary_stability/summary.csv
```

The per-seed deep-teacher cache files are generated locally and are not required once the summary outputs have been produced.

## Generate the Paper Figures

After the required result files are present, run:

```bash
python scripts/make_paper_figures.py
```

This generates the main-text figures:

```text
figures/
    fig_boundary_fidelity.pdf
    fig_boundary_fidelity.png
    fig_distribution_shift.pdf
    fig_distribution_shift.png
    fig_counterfactual_transfer.pdf
    fig_counterfactual_transfer.png
    fig_deep_boundary_stability.pdf
    fig_deep_boundary_stability.png
```

and the appendix robustness figure:

```text
figures/appendix/
    fig_boundary_fidelity_families.pdf
    fig_boundary_fidelity_families.png
```

## Mapping to the Paper

| Paper section | Experiment                                                  |
|---------------|-------------------------------------------------------------|
| Section 4.1   | Boundary-local structure of surrogate disagreement          |
| Section 4.2   | Predicting fidelity under changes in evaluation composition |
| Section 4.3   | Counterfactual transfer under matched fidelity              |
| Section 4.4   | Boundary stability across deep teachers                     |
| Appendix B    | Additional boundary-fidelity results                        |
| Appendix C    | Additional fidelity-prediction results                      |
| Appendix D    | Additional counterfactual-transfer results                  |
| Appendix E    | Additional deep-teacher stability results                   |

## Datasets

The tabular experiments use four datasets available through `scikit-learn`:

- Breast Cancer Wisconsin
- Diabetes
- Wine
- Iris

Wine and Iris are converted to binary one-vs-rest tasks using class 0 as the positive class. The Diabetes regression target is binarized at its median.

The deep-teacher experiments use:

- MNIST
- CIFAR-10

Dataset downloads for the deep experiments are handled by `torchvision`.

## Reproducibility Notes

Unless otherwise stated, tabular experiments are repeated over ten random seeds.

When an experiment contains multiple surrogate configurations within a seed, configurations are first averaged within that seed before uncertainty intervals are computed. Error bars therefore reflect variation across independent seeds rather than treating multiple configurations from the same seed as independent replicates.

Features in the tabular experiments are standardized using statistics estimated from the corresponding training partition only.

Surrogates are trained to reproduce the teacher's hard predictions rather than the ground-truth labels.

The segment-crossing measure is an approximate geometric proxy and should not be interpreted as the exact minimum Euclidean distance to the teacher's full decision boundary.

The deep-teacher stability experiment uses five independently trained teachers per dataset, corresponding to seeds 0--4. Boundary proximity is defined by the top-two logit margin and compared by rank rather than by a shared absolute threshold.

Full model architectures, hyperparameters, partitioning procedures, and metric definitions are described in Appendix A of the paper.

## HPC

The `hpc/` directory contains a virtual-environment setup script and SLURM launch scripts used for cluster execution. These are optional; all experiment scripts can also be run directly as shown above.

## Exploratory Notebooks

The notebooks under `notebooks/exploratory/` document exploratory analyses conducted during development. They are not required to reproduce the reported experiments.

The reported results are generated by the numbered experiment scripts and the scripts under `scripts/`.