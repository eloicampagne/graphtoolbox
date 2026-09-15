"""Build a directed graph from validation-set forecasting gains."""
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("EXPL_ADJ", "space")
import run_gnnexplainer_load as R

ALPHA = float(os.environ.get("CEIL_ALPHA", 10.0))
TOPK = int(os.environ.get("USE_TOPK", 3))
GRAPH_DIR = os.path.join(os.path.dirname(HERE), "graph_representations")
NAME = os.environ.get("USE_NAME", "usefulness")


def stack(ds):
    return (np.stack([d.x.numpy() for d in ds]), np.stack([d.y.numpy() for d in ds]))


def fit_pred(Xtr, ytr, Xte):
    sx = StandardScaler().fit(Xtr)
    return Ridge(alpha=ALPHA).fit(sx.transform(Xtr), ytr).predict(sx.transform(Xte))


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    tr, va, te = R.datasets()
    Xtr, Ytr = stack(tr); Xva, Yva = stack(va)
    N = Xtr.shape[1]

    # Usefulness estimated on validation only: fit on train, score on val.
    base = np.array([rmse(fit_pred(Xtr[:, i, :], Ytr[:, i, :], Xva[:, i, :]), Yva[:, i, :])
                     for i in range(N)])
    gain = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            A = np.hstack([Xtr[:, i, :], Xtr[:, j, :]])
            B = np.hstack([Xva[:, i, :], Xva[:, j, :]])
            gain[i, j] = 100 * (base[i] - rmse(fit_pred(A, Ytr[:, i, :], B), Yva[:, i, :])) / base[i]
    off = ~np.eye(N, dtype=bool)
    print(f"validation usefulness: mean {gain[off].mean():+.2f}%  "
          f"max {gain[off].max():+.2f}%  helping pairs {(gain[off] > 0).mean():.2f}")
    asym = np.abs(gain - gain.T)[off].mean()
    print(f"asymmetry |gain_ij - gain_ji|: mean {asym:.2f} points "
          f"(a symmetric criterion cannot represent this)")

    # Keep, for each target i, the TOPK sources j with the largest positive gain.
    W = np.zeros((N, N))
    for i in range(N):
        order = [j for j in np.argsort(-gain[i]) if j != i and gain[i, j] > 0][:TOPK]
        for j in order:
            W[j, i] = gain[i, j]          # edge j -> i, weighted by what j brings to i
    W = W / (W.max() + 1e-12)
    out = os.path.join(GRAPH_DIR, NAME)
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, "W.txt"), W)
    print(f"\nwrote {NAME}/W.txt: {int((W > 0).sum())} directed edges over {N} nodes "
          f"(mean in-degree {(W > 0).sum() / N:.1f})")
    src = (W > 0).sum(1)
    print("sources (edges out) :", {i: int(s) for i, s in enumerate(src) if s})
    print("\nNow train on it, e.g.:")
    print(f"  EXPL_CONV=SAGEConv EXPL_ADJ={NAME} python seed_graph_reliance.py")


if __name__ == "__main__":
    main()
