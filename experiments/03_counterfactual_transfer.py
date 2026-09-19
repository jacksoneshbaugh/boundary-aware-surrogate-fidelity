"""Idea 6 (fixed): example-based counterfactual/recourse transfer.

Fixes:
- deterministic seeds
- split before scaling; fit scaler only on training data
- logits + BCEWithLogitsLoss; rank-based boundary evaluation
- only evaluate recourse where teacher and surrogate agree at the starting point
- transfer succeeds iff the teacher reaches the SURROGATE'S INTENDED TARGET CLASS
  (the original merely checked whether the teacher changed class)
"""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

def compute_aurc(confidence, agreements):
    order = np.argsort(-confidence, kind="mergesort")
    sorted_agree = agreements[order]
    risks = 1.0 - sorted_agree
    cumulative_risk = np.cumsum(risks)
    coverages = np.arange(1, len(sorted_agree) + 1) / len(sorted_agree)
    risk_at_coverage = cumulative_risk / np.arange(1, len(sorted_agree) + 1)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(risk_at_coverage, coverages))

def lowest_fraction_mask(scores, fraction=0.25):
    k = max(1, int(np.ceil(fraction * len(scores))))
    idx = np.argsort(scores, kind="mergesort")[:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[idx] = True
    return mask

def find_recourse(surrogate, x0, X_pool, h_surr_pool):
    y0 = int(surrogate.predict(x0.reshape(1, -1))[0])
    target = 1 - y0
    opp_mask = h_surr_pool == target
    if not np.any(opp_mask):
        return None, None, target
    X_opp = X_pool[opp_mask]
    dists = np.linalg.norm(X_opp - x0, axis=1)
    j = int(np.argmin(dists))
    return X_opp[j], float(dists[j]), target

def run_recourse_test(name, X, y):
    print(f"\n{'='*60}\n  Dataset: {name}\n{'='*60}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    teacher = MLP(X.shape[1])
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    teacher.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(teacher(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    teacher.eval()
    with torch.no_grad():
        test_logits = teacher(X_test_t).numpy().flatten()
        train_logits = teacher(X_train_t).numpy().flatten()
    h_teacher_test = (test_logits >= 0.0).astype(int)
    h_teacher_train = (train_logits >= 0.0).astype(int)
    confidence = np.abs(test_logits)
    boundary_mask = lowest_fraction_mask(confidence, 0.25)

    results = {"dataset": name, "surrogates": []}
    for depth in [2, 3, 5, 7, 10]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        dt.fit(X_train, h_teacher_train)
        h_surr_test = dt.predict(X_test)
        h_surr_train = dt.predict(X_train)
        agreement = (h_surr_test == h_teacher_test).astype(float)

        global_fid = float(np.mean(agreement))
        boundary_fid = float(np.mean(agreement[boundary_mask]))
        aurc = compute_aurc(confidence, agreement)

        transfers = 0
        fails = 0
        total_recourse = 0
        excluded_initial_disagreement = 0
        distances = []

        for i in range(len(X_test)):
            # A surrogate recommendation only has a well-defined transfer target if the
            # surrogate is faithful at the starting point.
            if h_surr_test[i] != h_teacher_test[i]:
                excluded_initial_disagreement += 1
                continue

            x_cf, dist, target = find_recourse(dt, X_test[i], X_train, h_surr_train)
            if x_cf is None:
                continue
            total_recourse += 1
            distances.append(dist)

            with torch.no_grad():
                teacher_at_cf = int(teacher(torch.tensor(x_cf, dtype=torch.float32).unsqueeze(0)).item() >= 0.0)

            if teacher_at_cf == target:
                transfers += 1
            else:
                fails += 1

        transfer_rate = transfers / total_recourse if total_recourse > 0 else np.nan
        s = {
            "depth": depth,
            "global_fidelity": global_fid,
            "boundary_fidelity": boundary_fid,
            "aurc": aurc,
            "recourse_transfer_rate": float(transfer_rate),
            "total_recourse_attempts": int(total_recourse),
            "transfers": int(transfers),
            "fails": int(fails),
            "excluded_initial_disagreement": int(excluded_initial_disagreement),
            "mean_recourse_distance": float(np.mean(distances)) if distances else None,
        }
        results["surrogates"].append(s)
        print(
            f"  d={depth:2d} | GlobFid={global_fid:.4f} | BoundFid={boundary_fid:.4f} | "
            f"AURC={aurc:.4f} | Transfer={transfer_rate:.4f} ({transfers}/{total_recourse}) | "
            f"excluded={excluded_initial_disagreement}"
        )
    return results

def datasets():
    d = load_breast_cancer(); bc = (d.data, d.target)
    d = load_wine(); wine = (d.data, (d.target == 0).astype(int))
    d = load_diabetes(); med = np.median(d.target); diab = (d.data, (d.target >= med).astype(int))
    return {"Breast_Cancer": bc, "Wine": wine, "Diabetes": diab}

all_results = [run_recourse_test(name, X, y) for name, (X, y) in datasets().items()]
with open("idea6_recourse_transfer_results_fixed.json", "w") as f:
    json.dump(all_results, f, indent=2, allow_nan=False)
print("\nDone!")
