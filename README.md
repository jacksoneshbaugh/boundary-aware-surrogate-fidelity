# "Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity

Code and experimental artifacts for the anonymous ICLR 2027 submission  
**"Where Can I Trust You?": Boundary-Aware Evaluation of Surrogate Fidelity**.

## Overview

This repository contains the code used to reproduce the experiments and figures in the paper.

The experiments study four questions:

1. **Boundary-local fidelity:** Is teacher--surrogate disagreement systematically concentrated near the teacher's decision boundary?
2. **Fidelity prediction under changes in evaluation composition:** Does retaining agreement across teacher-confidence regions improve prediction of fidelity when the composition of the evaluation set changes?
3. **Counterfactual transfer:** Do surrogates with similar global and boundary-local fidelity necessarily behave similarly under changes selected using the surrogate?
4. **Deep-teacher boundary stability:** Do independently trained deep teachers identify the same examples as lying near their decision boundaries?

The paper evaluates two notions of boundary proximity for the tabular experiments:

- teacher-confidence proximity, based on absolute logit magnitude; and
- an approximate segment-crossing distance based on teacher predictions and the geometry of the evaluation set.

## Experiments

### Experiment 1: Boundary-local structure of surrogate disagreement

The primary analysis varies decision-tree surrogate capacity and compares global fidelity with fidelity near the teacher's decision boundary.

Additional robustness experiments evaluate representative:

- decision-tree,
- logistic-regression,
- \(k\)-nearest-neighbor, and
- multilayer-perceptron

surrogates across multiple boundary fractions.

Results are written to:

```text
results/boundary_fidelity/
```

### Experiment 2: Predicting fidelity under changes in evaluation composition

This experiment compares:

- a source-set global-fidelity baseline; and
- a confidence-stratified fidelity profile

for predicting fidelity on target sets with different confidence compositions.

The primary analysis uses three confidence bins, with a five-bin sensitivity analysis reported in the appendix.

Results are written to:

```text
results/distribution_shift_3bins/
```

### Experiment 3: Example-based counterfactual transfer

This experiment measures whether a change selected using the surrogate that changes the surrogate's prediction also induces the corresponding target-class change in the teacher.

The analysis additionally identifies surrogate pairs with closely matched global and boundary-conditioned fidelity but substantially different transfer behavior.

Results are written to:

```text
results/counterfactual_transfer/
```

### Experiment 4: Boundary stability across deep teachers

Five independently trained teachers are evaluated for each of MNIST and CIFAR-10.

For each pair of teachers, we measure:

- global predictive agreement,
- Jaccard overlap between their lowest-margin boundary sets,
- agreement on the union of those boundary sets, and
- agreement on their intersection.

Results are written to:

```text
results/deep_boundary_stability/
```

## Datasets

The tabular experiments use four datasets distributed with `scikit-learn`:

- Breast Cancer Wisconsin
- Diabetes
- Wine
- Iris

Wine and Iris are converted to binary one-vs-rest tasks using class 0 as the positive class.  
The Diabetes regression target is binarized at its median.

The deep-teacher experiments use:

- MNIST
- CIFAR-10

No generated or synthetic data are used in the reported experiments.

## Random Seeds and Uncertainty

Unless otherwise stated, tabular experiments are repeated over ten random seeds.

When an experiment contains multiple surrogate configurations within a seed, configurations are first averaged within that seed before uncertainty intervals are computed. Error bars therefore reflect variation across independent seeds rather than treating configurations from the same seed as independent replicates.

The deep-teacher stability experiment uses five independently trained teachers per dataset, corresponding to seeds 0--4.

## Reproducing the Experiments

### Environment

Install the dependencies using the environment specification provided with this repository:

```bash
# Replace with the command appropriate to the supplied environment file.
<environment setup command>
```

### Run Experiments

Run the four experiment pipelines:

```bash
<boundary-fidelity command>
<fidelity-prediction command>
<counterfactual-transfer command>
<deep-teacher-stability command>
```

Each experiment writes its per-run or summary results under `results/`.

### Generate Figures

After the experiment outputs have been generated, run:

```bash
<figure-generation command>
```

This produces the main-text figures:

```text
figures/
    fig_boundary_fidelity.pdf
    fig_distribution_shift.pdf
    fig_counterfactual_transfer.pdf
    fig_deep_boundary_stability.pdf
```

and the appendix robustness figure:

```text
figures/appendix/
    fig_boundary_fidelity_families.pdf
```

PNG versions are also generated.

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

## Reproducibility Notes

Features in the tabular experiments are standardized using statistics estimated from the corresponding training partition only.

Surrogates are trained to reproduce the teacher's hard predictions rather than the ground-truth labels.

The segment-crossing measure is an approximate geometric proxy and should not be interpreted as the exact minimum Euclidean distance to the teacher's full decision boundary.

Full model architectures, hyperparameters, partitioning procedures, and boundary definitions are provided in Appendix A of the paper.

## Anonymous Review

This repository is provided for anonymous peer review.

Author-identifying information and the permanent repository location will be added after the review process.