"""
GNN-Based Disease Risk Prediction (simple base project)
-------------------------------------------------------
Idea: each patient is a node. Patients with similar clinical profiles are
connected by edges (k-nearest-neighbour graph). A Graph Convolutional Network
(GCN) learns from a patient's own features AND the features of similar
patients to predict disease risk (high risk = 1, low risk = 0).

Requirements:  pip install torch numpy scikit-learn
(Replace the synthetic data with a real CSV, e.g. the Pima Indians Diabetes
dataset, using the `load_csv_data` function below.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def generate_synthetic_data(n=1000):
    """Create fake patient records with a plausible risk relationship."""
    age = np.random.randint(20, 80, n)
    bmi = np.random.normal(27, 5, n)
    systolic_bp = np.random.normal(125, 18, n)
    glucose = np.random.normal(105, 25, n)
    cholesterol = np.random.normal(200, 35, n)
    smoker = np.random.binomial(1, 0.25, n)
    family_history = np.random.binomial(1, 0.30, n)

    X = np.column_stack([age, bmi, systolic_bp, glucose, cholesterol,
                         smoker, family_history]).astype(np.float32)

    # Hidden risk score -> binary label (with noise)
    score = (0.04 * (age - 50) + 0.08 * (bmi - 27) + 0.02 * (systolic_bp - 125)
             + 0.03 * (glucose - 105) + 0.01 * (cholesterol - 200)
             + 0.8 * smoker + 0.9 * family_history
             + np.random.normal(0, 0.8, n))
    y = (score > np.median(score)).astype(np.int64)
    return X, y


def load_csv_data(path, label_col):
    """Optional: load a real dataset (all other columns must be numeric)."""
    import pandas as pd
    df = pd.read_csv(path)
    y = df[label_col].values.astype(np.int64)
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    return X, y


# ----------------------------------------------------------------------
# 2. GRAPH CONSTRUCTION
# ----------------------------------------------------------------------
def build_patient_graph(X, k=8):
    """kNN patient-similarity graph -> normalised adjacency D^-1/2 (A+I) D^-1/2."""
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    A = A.toarray()
    A = np.maximum(A, A.T)            # make undirected
    A = A + np.eye(A.shape[0])        # add self-loops
    deg = A.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


# ----------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------
class GCNLayer(nn.Module):
    """H' = A_norm @ H @ W"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return adj @ self.linear(x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, self.training)
        return self.out(x)


class MLP(nn.Module):
    """Baseline that ignores the graph."""

    def __init__(self, in_dim, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes))

    def forward(self, x, adj=None):
        return self.net(x)


# ----------------------------------------------------------------------
# 4. TRAIN / EVALUATE
# ----------------------------------------------------------------------
def train_model(model, x, adj, y, idx_train, idx_val, epochs=200, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, adj)
        loss = F.cross_entropy(out[idx_train], y[idx_train])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x, adj)[idx_val].argmax(1)
            val_acc = (val_pred == y[idx_val]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val acc {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model


def evaluate(model, x, adj, y, idx_test, name):
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)[idx_test]
        prob = F.softmax(logits, dim=1)[:, 1].numpy()
        pred = logits.argmax(1).numpy()
    true = y[idx_test].numpy()
    print(f"[{name}] Acc {accuracy_score(true, pred):.4f} | "
          f"F1 {f1_score(true, pred):.4f} | AUC {roc_auc_score(true, prob):.4f}")


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
def main():
    X, y = generate_synthetic_data(n=1000)
    # X, y = load_csv_data("diabetes.csv", label_col="Outcome")   # real data

    X = StandardScaler().fit_transform(X)
    adj = build_patient_graph(X, k=8)
    x = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    idx = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp], random_state=SEED)
    idx_train, idx_val, idx_test = map(torch.tensor, (idx_train, idx_val, idx_test))

    print(f"Patients: {len(y)} | Features: {X.shape[1]} | High-risk rate: {y.mean():.2f}")
    print(f"Graph edges (incl. self-loops): {int((adj > 0).sum())}\n")

    print("Training GCN...")
    gcn = train_model(GCN(X.shape[1], 32, 2), x, adj, y_t, idx_train, idx_val)
    print("\nTraining MLP baseline...")
    mlp = train_model(MLP(X.shape[1], 32, 2), x, adj, y_t, idx_train, idx_val)

    print("\n=== Test Results ===")
    evaluate(gcn, x, adj, y_t, idx_test, "GCN")
    evaluate(mlp, x, adj, y_t, idx_test, "MLP")

    # Risk score for individual patients
    gcn.eval()
    with torch.no_grad():
        risk = F.softmax(gcn(x, adj), dim=1)[:, 1]
    print("\nSample predicted risk probabilities (first 5 test patients):")
    for i in idx_test[:5]:
        print(f"  Patient {int(i):4d} -> risk {risk[i]:.2f} (true label {int(y_t[i])})")


if __name__ == "__main__":
    main()
