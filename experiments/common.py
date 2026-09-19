"""
Shared utilities for the surrogate-fidelity experiments.

This module contains the reusable components used across the paper's
experiments, including:

    - dataset loading and preprocessing
    - deterministic seeding
    - teacher-model training and inference
    - fidelity and risk-coverage calculations
    - confidence-based boundary definitions
    - approximate geometric boundary-distance calculations

The goal is to keep core scientific definitions implemented in one place so
that all experiments use the same preprocessing, teacher models, boundary
definitions, and metrics.

The geometric boundary-distance routine used here is not an exact minimum
Euclidean distance to the teacher's full decision boundary. It approximates
boundary proximity by locating an opposite-prediction evaluation point and
binary-searching for a teacher decision change along the connecting segment.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_iris,
    load_diabetes,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BinaryMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tabular_datasets():
    datasets = {}

    d = load_breast_cancer()
    datasets["breast_cancer"] = (d.data, d.target)

    d = load_wine()
    datasets["wine"] = (
        d.data,
        (d.target == 0).astype(int),
    )

    d = load_iris()
    datasets["iris"] = (
        d.data,
        (d.target == 0).astype(int),
    )

    d = load_diabetes()
    median = np.median(d.target)
    datasets["diabetes"] = (
        d.data,
        (d.target >= median).astype(int),
    )

    return datasets


def split_and_scale(X, y, seed, test_size=0.30):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler


def train_mlp_teacher(
    X_train,
    y_train,
    seed,
    epochs=500,
    lr=1e-2,
):
    seed_everything(seed)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = BinaryMLP(X_train.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def teacher_logits(model, X):
    X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        return model(X_t).numpy().reshape(-1)


def teacher_predictions(model, X):
    return (teacher_logits(model, X) >= 0.0).astype(int)


def lowest_fraction_mask(scores, fraction, valid=None):
    scores = np.asarray(scores)

    if valid is None:
        valid = np.ones(len(scores), dtype=bool)

    valid_idx = np.flatnonzero(valid)

    mask = np.zeros(len(scores), dtype=bool)

    if len(valid_idx) == 0:
        return mask

    k = max(1, int(np.ceil(fraction * len(valid_idx))))

    order = np.argsort(scores[valid_idx], kind="mergesort")
    chosen = valid_idx[order[:k]]

    mask[chosen] = True
    return mask


def compute_aurc(confidence, agreement):
    """
    Area under the selective risk-coverage curve.

    Higher confidence points are retained first.
    """
    confidence = np.asarray(confidence)
    agreement = np.asarray(agreement, dtype=float)

    order = np.argsort(-confidence, kind="mergesort")
    errors = 1.0 - agreement[order]

    n = len(errors)

    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)

    return float(np.trapezoid(risk, coverage))


def segment_crossing_distance(
    teacher,
    x0,
    X_eval,
    teacher_predictions_eval,
    steps=30,
):
    """
    Approximate teacher-boundary proximity.

    Finds the nearest observed evaluation point with the opposite teacher
    prediction, then binary-searches for a teacher decision-boundary crossing
    along the line segment connecting the two points.

    This is NOT the exact minimum Euclidean distance to the full teacher
    decision boundary.
    """
    with torch.no_grad():
        f0 = float(
            teacher(
                torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
            ).item()
        )

    y0 = int(f0 >= 0.0)

    opposite = teacher_predictions_eval != y0

    if not np.any(opposite):
        return np.nan

    X_opposite = X_eval[opposite]

    distances = np.linalg.norm(X_opposite - x0, axis=1)
    x1 = X_opposite[np.argmin(distances)]

    lo = 0.0
    hi = 1.0

    for _ in range(steps):
        mid = (lo + hi) / 2.0
        x_mid = x0 + mid * (x1 - x0)

        with torch.no_grad():
            f_mid = float(
                teacher(
                    torch.tensor(
                        x_mid,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                ).item()
            )

        if (f_mid >= 0.0) == (f0 >= 0.0):
            lo = mid
        else:
            hi = mid

    t = (lo + hi) / 2.0
    crossing = x0 + t * (x1 - x0)

    return float(np.linalg.norm(crossing - x0))


def segment_crossing_distances(
    teacher,
    X_eval,
    teacher_predictions_eval,
):
    return np.array(
        [
            segment_crossing_distance(
                teacher,
                x,
                X_eval,
                teacher_predictions_eval,
            )
            for x in X_eval
        ],
        dtype=float,
    )