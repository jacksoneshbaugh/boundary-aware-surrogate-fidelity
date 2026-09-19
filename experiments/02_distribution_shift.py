"""Idea 5 (fixed): distribution-shift prediction from a fidelity profile.

Fixes:
- deterministic seeds
- split before scaling; fit scaler only on training data
- logits for confidence to avoid sigmoid saturation
- DISJOINT source-evaluation and target pools (removes target leakage)
- source-only confidence-bin construction
- bool(...) around NumPy comparisons so JSON serialization cannot crash
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

def build_profile(source_conf, source_agree, n_bins=5):
    internal = np.quantile(source_conf, np.linspace(0, 1, n_bins + 1)[1:-1])
    internal = np.unique(internal)
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    profile = []
    for i in range(len(edges) - 1):
        mask = (source_conf >= edges[i]) & (source_conf < edges[i + 1])
        if not np.any(mask):
            continue
        profile.append({
            "bin": i,
            "low": None if np.isneginf(edges[i]) else float(edges[i]),
            "high": None if np.isposinf(edges[i + 1]) else float(edges[i + 1]),
            "fidelity": float(np.mean(source_agree[mask])),
            "n": int(np.sum(mask)),
        })
    return profile, edges

def predict_from_profile(target_conf, profile, edges, fallback):
    if len(target_conf) == 0:
        return np.nan
    values = np.full(len(target_conf), fallback, dtype=float)
    for i, p in enumerate(profile):
        mask = (target_conf >= edges[i]) & (target_conf < edges[i + 1])
        values[mask] = p["fidelity"]
    return float(np.mean(values))

def run_shift_test(name, X, y):
    print(f"\n{'='*60}\n  Dataset: {name}\n{'='*60}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_eval = scaler.transform(X_eval_raw)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32)

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
        eval_logits = teacher(X_eval_t).numpy().flatten()
        train_logits = teacher(X_train_t).numpy().flatten()
    h_teacher_eval = (eval_logits >= 0.0).astype(int)
    conf_eval = np.abs(eval_logits)
    h_teacher_train = (train_logits >= 0.0).astype(int)

    # Disjoint source and target evaluation pools.
    all_idx = np.arange(len(X_eval))
    source_idx, target_idx = train_test_split(
        all_idx, test_size=0.5, random_state=SEED + 1, stratify=h_teacher_eval
    )

    results = {"dataset": name, "n_source": len(source_idx), "n_target_pool": len(target_idx), "depths": {}}

    for depth in [3, 5, 7]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        dt.fit(X_train, h_teacher_train)
        h_surr_eval = dt.predict(X_eval)
        agreement = (h_surr_eval == h_teacher_eval).astype(float)

        source_agree = agreement[source_idx]
        source_conf = conf_eval[source_idx]
        source_global_fid = float(np.mean(source_agree))
        profile, edges = build_profile(source_conf, source_agree, n_bins=5)

        target_conf_all = conf_eval[target_idx]
        target_agree_all = agreement[target_idx]
        median_target_conf = float(np.median(target_conf_all))
        shifts = {
            "easy": target_conf_all >= median_target_conf,
            "hard": target_conf_all < median_target_conf,
        }

        depth_result = {"source_global_fidelity": source_global_fid, "profile": profile, "targets": {}}
        for target_name, local_mask in shifts.items():
            t_conf = target_conf_all[local_mask]
            t_agree = target_agree_all[local_mask]
            if len(t_agree) == 0:
                continue
            target_true_fid = float(np.mean(t_agree))
            pred_global = source_global_fid
            pred_profile = predict_from_profile(t_conf, profile, edges, source_global_fid)
            error_global = float(abs(pred_global - target_true_fid))
            error_profile = float(abs(pred_profile - target_true_fid))
            profile_wins = bool(error_profile < error_global)

            print(
                f"  d={depth} | Target={target_name:4s} | True={target_true_fid:.4f} | "
                f"PredGlobal={pred_global:.4f} (err={error_global:.4f}) | "
                f"PredProfile={pred_profile:.4f} (err={error_profile:.4f}) | "
                f"{'PROFILE WINS' if profile_wins else 'GLOBAL WINS/TIES'}"
            )
            depth_result["targets"][target_name] = {
                "n": int(len(t_agree)),
                "true_fidelity": target_true_fid,
                "pred_global": pred_global,
                "pred_profile": pred_profile,
                "error_global": error_global,
                "error_profile": error_profile,
                "profile_wins": profile_wins,
            }
        results["depths"][str(depth)] = depth_result
    return results

def datasets():
    d = load_breast_cancer(); bc = (d.data, d.target)
    d = load_wine(); wine = (d.data, (d.target == 0).astype(int))
    d = load_diabetes(); med = np.median(d.target); diab = (d.data, (d.target >= med).astype(int))
    return {"Breast_Cancer": bc, "Wine": wine, "Diabetes": diab}

all_results = [run_shift_test(name, X, y) for name, (X, y) in datasets().items()]
with open("idea5_distribution_shift_results_fixed.json", "w") as f:
    json.dump(all_results, f, indent=2, allow_nan=False)
print("\nDone!")
