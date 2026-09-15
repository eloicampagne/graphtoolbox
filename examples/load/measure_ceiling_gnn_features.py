"""Measure spatial forecasting gains using the GNN feature set."""
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
USE_XGB = os.environ.get("CEIL_XGB", "0") == "1"


def stack(ds):
    """[days, nodes, feat] and [days, nodes, 48] from a GraphDataset."""
    X = np.stack([d.x.numpy() for d in ds])
    Y = np.stack([d.y.numpy() for d in ds])
    return X, Y


def fit_eval(Xtr, ytr, Xte, yte):
    sx = StandardScaler().fit(Xtr)
    if USE_XGB:
        import xgboost as xgb
        from sklearn.multioutput import MultiOutputRegressor
        m = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=300, max_depth=5, eta=0.05,
                                                  subsample=0.9, colsample_bytree=0.8,
                                                  n_jobs=-1, random_state=42))
    else:
        m = Ridge(alpha=ALPHA)
    m.fit(sx.transform(Xtr), ytr)
    return m.predict(sx.transform(Xte))


def main():
    tr, va, te = R.datasets()
    Xtr, Ytr = stack(tr)
    Xva, Yva = stack(va)
    Xte, Yte = stack(te)
    # The GNN trains on train+val, so give the reference model the same data.
    Xtr = np.concatenate([Xtr, Xva]); Ytr = np.concatenate([Ytr, Yva])
    D, N, F = Xtr.shape
    print(f"nodes {N}, features per node {F}, train days {D}, test days {Xte.shape[0]}, "
          f"model {'XGBoost' if USE_XGB else f'Ridge(alpha={ALPHA})'}")

    flat_tr = Xtr.reshape(len(Xtr), -1)      # [days, nodes*F]
    flat_te = Xte.reshape(len(Xte), -1)

    out = {}
    for name in ("own", "all"):
        pred = np.zeros_like(Yte)
        for i in range(N):
            A = Xtr[:, i, :] if name == "own" else flat_tr
            B = Xte[:, i, :] if name == "own" else flat_te
            pred[:, i, :] = fit_eval(A, Ytr[:, i, :], B, Yte[:, i, :])
        pn = pred.sum(1).ravel(); yn = Yte.sum(1).ravel()
        out[name] = float(np.sqrt(np.mean((pn - yn) ** 2)))
        reg = np.sqrt(((pred - Yte) ** 2).mean(axis=(0, 2)))
        print(f"[{name:3}] national RMSE {out[name]:7.0f}   mean regional {reg.mean():6.1f}   "
              f"({A.shape[1]} features)")

    d = out["own"] - out["all"]
    print("\n===== ceiling at the GNN's feature set =====")
    print(f"own {out['own']:.0f} -> all {out['all']:.0f} MW   ({d:+.0f} MW, {100*d/out['own']:+.2f}%)")
    print("reference: GNN reads ~5% from neighbors and reaches 877 +/- 42 MW (5 seeds)")


if __name__ == "__main__":
    main()
