"""Measure pairwise forecasting gains and compare them with existing graphs."""
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
GRAPH_DIR = os.path.join(os.path.dirname(HERE), "graph_representations")


def stack(ds):
    return (np.stack([d.x.numpy() for d in ds]), np.stack([d.y.numpy() for d in ds]))


def fit(Xtr, ytr, Xte):
    sx = StandardScaler().fit(Xtr)
    return Ridge(alpha=ALPHA).fit(sx.transform(Xtr), ytr).predict(sx.transform(Xte))


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    tr, va, te = R.datasets()
    Xtr, Ytr = stack(tr); Xva, Yva = stack(va); Xte, Yte = stack(te)
    Xtr = np.concatenate([Xtr, Xva]); Ytr = np.concatenate([Ytr, Yva])
    N = Xtr.shape[1]
    nodes = [str(n) for n in tr.data.nodes] if hasattr(tr, "data") else [str(i) for i in range(N)]

    base = np.array([rmse(fit(Xtr[:, i, :], Ytr[:, i, :], Xte[:, i, :]), Yte[:, i, :])
                     for i in range(N)])
    print("own-only regional RMSE:", np.round(base, 1))

    gain = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            A = np.hstack([Xtr[:, i, :], Xtr[:, j, :]])
            B = np.hstack([Xte[:, i, :], Xte[:, j, :]])
            gain[i, j] = 100 * (base[i] - rmse(fit(A, Ytr[:, i, :], B), Yte[:, i, :])) / base[i]
        print(f"  row {i+1}/{N} done", flush=True)
    np.save(os.path.join(HERE, "results_usefulness_gain.npy"), gain)

    off = ~np.eye(N, dtype=bool)
    print(f"\n=== pairwise usefulness (%, ridge, {N}x{N-1} pairs) ===")
    print(f"mean {gain[off].mean():+.2f}   min {gain[off].min():+.2f}   max {gain[off].max():+.2f}   "
          f"share of pairs that help {(gain[off] > 0).mean():.2f}")

    Yall = np.concatenate([Ytr, Yte])
    series = Yall.transpose(1, 0, 2).reshape(N, -1)
    corr = np.corrcoef(series)
    print("\n=== is usefulness related to similarity? ===")
    print(f"corr(usefulness, load correlation) = {np.corrcoef(gain[off], corr[off])[0,1]:+.3f}")
    print("   (negative => the useful neighbors are the LEAST similar ones)")

    print("\n=== do the existing graphs pick useful pairs? ===")
    for g in ("dtw", "correlation", "precision", "space", "gl3sr"):
        f = os.path.join(GRAPH_DIR, g, "W.txt")
        if not os.path.exists(f):
            continue
        W = np.loadtxt(f)
        if W.shape != (N, N):
            continue
        m = off & (W > 0)
        if m.sum() == 0:
            continue
        print(f"  {g:12} keeps {m.sum():3d}/{off.sum()} pairs   "
              f"mean usefulness of kept {gain[m].mean():+.2f}%   "
              f"of dropped {gain[off & (W <= 0)].mean() if (off & (W <= 0)).sum() else float('nan'):+.2f}%   "
              f"corr(W, usefulness) {np.corrcoef(W[off], gain[off])[0,1]:+.3f}")

    print("\n=== the graph a forecaster would want (top-3 useful neighbors per node) ===")
    for i in range(N):
        order = np.argsort(-gain[i])
        top = [f"{nodes[j] if j < len(nodes) else j}({gain[i,j]:+.1f}%)" for j in order[:3] if j != i]
        print(f"  {nodes[i] if i < len(nodes) else i:<28} <- {', '.join(top)}")


if __name__ == "__main__":
    main()
