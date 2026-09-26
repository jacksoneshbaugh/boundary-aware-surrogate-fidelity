# "Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity

Code and experimental artifacts for:

> **"Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity**  
> Jackson Eshbaugh, 2026.

## Overview

Surrogate models are commonly evaluated using a single global fidelity score: the proportion of examples on which the surrogate agrees with its teacher. This repository accompanies a study of how that agreement is structured relative to the teacher's decision boundary and what information is lost when fidelity is reduced to a single scalar.

The paper investigates four questions:

1. **Boundary-local fidelity:** Is teacher--surrogate disagreement systematically concentrated near the teacher's decision boundary?
2. **Fidelity prediction under changes in evaluation composition:** Does retaining agreement across teacher-confidence regions improve prediction of fidelity when the composition of the evaluation set changes?
3. **Counterfactual transfer:** Do surrogates with similar global and boundary-local fidelity necessarily behave similarly under changes selected using the surrogate?
4. **Deep-teacher boundary stability:** Do independently trained deep teachers identify the same examples as lying near their decision boundaries?

For the tabular experiments, boundary proximity is operationalized in two ways:

- **Confidence-based proximity:** examples with the smallest absolute teacher logits.
- **Approximate segment-crossing proximity:** examples closest to an estimated teacher prediction transition along a line segment to a nearby oppositely predicted example.

Precomputed result files are included so that the paper figures can be regenerated without rerunning model training.

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
    distribution_shift_family/
    distribution_shift/
    counterfactual_transfer/
    deep_boundary_stability/

figures/                          # generated paper figures

hpc/
    setup_venv.sh
    slurm/                        # optional SLURM launch scripts

notebooks/
    exploratory/                  # exploratory analyses; not required
    paper_figures.ipynb           # earlier figure-development notebook

requirements.txt
```

The result directories for Experiment 2 correspond to:

```text
results/distribution_shift_3bins/   primary three-bin analysis
results/distribution_shift_family/  cross-family robustness analysis
results/distribution_shift/         five-bin sensitivity analysis
```

The numbered experiment scripts correspond directly to Sections 4.1--4.4 of the paper.

## Environment

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

The deep-teacher experiments use PyTorch and will use CUDA automatically when available.

## Reproducing the Paper Figures

The precomputed results required for the paper figures are included in `results/`.

From the repository root, run:

```bash
python scripts/make_paper_figures.py
```

This generates the four main-text figures:

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

and the appendix surrogate-family robustness figure:

```text
figures/appendix/
    fig_boundary_fidelity_families.pdf
    fig_boundary_fidelity_families.png
```

## Reproducing the Experiments

Run all commands from the repository root.

### Experiment 1: Boundary-local structure of surrogate disagreement

```bash
python experiments/01_boundary_fidelity.py
```

This runs both the decision-tree capacity sweep and the representative cross-family analysis over the tabular datasets.

Primary output:

```text
results/boundary_fidelity/runs.csv
```

To regenerate the summary tables:

```bash
python scripts/summarize_boundary_fidelity.py
```

### Experiment 2: Predicting fidelity under changes in evaluation composition

The primary analysis uses three teacher-confidence bins:

```bash
python experiments/02_distribution_shift.py \
    --bins 3 \
    --output results/distribution_shift_3bins/runs.csv
```

The representative surrogate-family robustness analysis uses the same experimental protocol with one configuration from each surrogate family:

```bash
python experiments/02_distribution_shift.py \
    --bins 3 \
    --family-check \
    --output results/distribution_shift_family/runs.csv
```

The five-bin sensitivity analysis is:

```bash
python experiments/02_distribution_shift.py \
    --bins 5 \
    --output results/distribution_shift/runs.csv
```

### Experiment 3: Example-based counterfactual transfer

```bash
python experiments/03_counterfactual_transfer.py \
    --query-output results/counterfactual_transfer/queries.csv
```

Primary outputs:

```text
results/counterfactual_transfer/runs.csv
results/counterfactual_transfer/queries.csv
```

To regenerate the summary:

```bash
python scripts/summarize_counterfactual_transfer.py
```

### Experiment 4: Boundary stability across deep teachers

Train five independently initialized MNIST teachers:

```bash
python experiments/04_deep_boundary_stability.py --dataset mnist
```

and five independently initialized CIFAR-10 teachers:

```bash
python experiments/04_deep_boundary_stability.py --dataset cifar10
```

The default training schedules are 30 epochs for MNIST and 200 epochs for CIFAR-10.

After training, compute the pairwise stability statistics:

```bash
python scripts/summarize_deep_boundary_stability.py
```

This produces:

```text
results/deep_boundary_stability/models.csv
results/deep_boundary_stability/pairs.csv
results/deep_boundary_stability/summary.csv
```

Per-seed model caches are generated locally and are not included in the repository; the derived summary outputs needed to reproduce the reported analyses are included.

## Mapping to the Paper

| Paper section | Experiment |
|---|---|
| Section 4.1 | Boundary-local structure of surrogate disagreement |
| Section 4.2 | Predicting fidelity under changes in evaluation composition |
| Section 4.3 | Counterfactual transfer under matched fidelity |
| Section 4.4 | Boundary stability across deep teachers |
| Appendix B | Additional boundary-fidelity results |
| Appendix C | Additional fidelity-prediction results |
| Appendix D | Additional counterfactual-transfer results |
| Appendix E | Additional deep-teacher stability results |

## Datasets

The tabular experiments use four datasets distributed through `scikit-learn`:

- Breast Cancer Wisconsin
- Diabetes
- Wine
- Iris

Wine and Iris are converted to binary one-vs-rest tasks using class 0 as the positive class. The Diabetes regression target is binarized at its median.

The deep-teacher experiments use:

- MNIST
- CIFAR-10

These datasets are downloaded through `torchvision` when required.

## Reproducibility Notes

Unless otherwise stated, the tabular experiments are repeated over ten random seeds.

When an experiment contains multiple surrogate configurations within a seed, configurations are first averaged within that seed before uncertainty intervals are computed. Error bars therefore reflect variation across independent seeds rather than treating multiple configurations from the same seed as independent observations.

Features in the tabular experiments are standardized using statistics estimated from the corresponding training partition only.

Surrogates are trained to reproduce the teacher's hard predictions rather than the ground-truth labels.

The segment-crossing measure is an approximate geometric proxy and should not be interpreted as the exact minimum Euclidean distance to the teacher's full decision boundary.

The deep-teacher stability experiment uses five independently trained teachers per dataset, corresponding to seeds 0--4. Boundary proximity is defined using the top-two logit margin and compared by rank rather than by a shared absolute threshold.

Full model architectures, hyperparameters, partitioning procedures, and metric definitions are provided in Appendix A of the paper.

## HPC

The `hpc/` directory contains a virtual-environment setup script and SLURM launch scripts used for cluster execution. These are optional; the experiment scripts can also be run directly using the commands above.

## Exploratory Analyses

The notebooks under `notebooks/exploratory/` record analyses conducted during development of the project. They are retained for transparency but are **not required to reproduce the results reported in the paper**.

The reported experiments are implemented in `experiments/`, and the final paper figures are generated by `scripts/make_paper_figures.py`.

## Citation

If you use this work, please cite:

```bibtex
@article{eshbaugh2026where,
  title   = {"Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity},
  author  = {Eshbaugh, Jackson},
  year    = {2026},
  note    = {Preprint}
}
```

The citation will be updated with the archival publication information when available.