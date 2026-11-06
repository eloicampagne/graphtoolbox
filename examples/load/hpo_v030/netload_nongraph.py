"""Non-graph baselines of the companion study on the net-load inputs of the sweep (weather trajectories included).

Direct: the shared and per-node MLPs and XGBoost b-global and b-local of nongraph_models.py (run_mlps, run_xgb),
on the datasets of netload_sweep_v030.py --leads. Decomposed: XGBoost b-global, with the search and seeds of
run_xgb, fitted on each component (load, wind and solar power, with the inputs of the decomposed sweep) and
recombined as load - wind - solar, like the decomposed graph models.

    python netload_nongraph.py     # writes netload_nongraph.npz and netload_nongraph.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NETLOAD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "netload")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, NETLOAD)
sys.path.insert(0, HERE)
os.chdir(NETLOAD)
import netload_pipeline as P  # noqa: E402
import decomp_sweep_netload as D  # noqa: E402
import netload_data_v030 as ND  # noqa: E402
from graphtoolbox.data.dataset import DataClass, GraphDataset  # noqa: E402

H = 48


def stack(ds):
    X = np.stack([d.x.numpy() for d in ds.pyg_data])
    Y = np.stack([d.y.numpy() for d in ds.pyg_data])
    return X.astype(np.float32), Y.astype(np.float64)


def xgb_global(tr, va, te, n_seeds=3):
    """b-global branch of nongraph_models.run_xgb: same search space, trials and seeds."""
    import xgboost as xgb
    (Xtr, Ytr), (Xva, Yva), (Xte, _) = tr, va, te
    S, N, F = Xtr.shape
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    Xs, Ys = Xtr.reshape(-1, F), ((Ytr - mu) / sd).reshape(-1, H)
    Xv, Yv = Xva.reshape(-1, F), ((Yva - mu) / sd).reshape(-1, H)
    space = lambda rng: {
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1, 0.2])),
        "max_depth": int(rng.choice([4, 6, 8, 10])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 6.0, 10.0])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2])),
        "reg_lambda": float(rng.choice([0.1, 1.0, 5.0, 10.0])),
        "n_estimators": int(rng.choice([120, 180, 260, 340]))}
    rng = np.random.default_rng(42); best, best_cfg = np.inf, None
    for _ in range(10):
        cfg = space(rng)
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, **cfg)
        m.fit(Xs, Ys[:, H // 2]); v = float(np.sqrt(np.mean((m.predict(Xv) - Yv[:, H // 2]) ** 2)))
        if v < best:
            best, best_cfg = v, cfg
    preds = []
    for s in range(n_seeds):
        Pm = np.zeros((Xte.shape[0] * N, H))
        for h in range(H):
            m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42 + s, **best_cfg)
            m.fit(Xs, Ys[:, h]); Pm[:, h] = m.predict(Xte.reshape(-1, F))
        preds.append(Pm.reshape(Xte.shape[0], N, H) * sd + mu)
    return {"cfg": best_cfg, "preds": preds}


# Direct datasets of the sweep, weather trajectories included.
direct = ND.direct_datasets(True)
for ds in direct:
    ds._set_adj_matrix(adj_matrix="dtw")
d_tr, d_va, d_te = (stack(ds) for ds in direct)
print("direct samples", len(d_tr[0]), len(d_va[0]), len(d_te[0]), "features", d_tr[0].shape[2], flush=True)
y = d_te[1].sum(1).reshape(-1)
stored = torch.load(os.path.join(HERE, "netload_v030", "leads_seed42", "target__NetLoad.pt"), weights_only=False)
assert np.max(np.abs(np.asarray(stored).sum(0) - y)) < 1e-3, "net-load target differs from the sweep"

# Component datasets of the decomposed sweep.
dkw = dict(D.data_kwargs)
dkw["features_to_lag"] = {**D.data_kwargs["features_to_lag"], **{v: (-47, -1) for v in ND.LEAD_VARS}}
data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                 data_kwargs=dkw, folder_config=str(P.HERE))
extra = {"Load": ND.lead_cols({"wind", "cloud"}), "Wind_power": ND.lead_cols({"wind"}),
         "Solar_power": ND.lead_cols({"cloud"})}
comps = {}
for comp, feats in D.COMPONENTS.items():
    dk = {"batch_size": 32, "adj_matrix": "dtw", "features_base": feats + extra[comp], "target_base": comp}
    common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    comps[comp] = tuple(stack(ds) for ds in (tr, va, te))

rec = comps["Load"][2][1] - comps["Wind_power"][2][1] - comps["Solar_power"][2][1]
assert np.allclose(rec.sum(1).reshape(-1), y, atol=1.0), "component targets do not recombine into net-load"

import nongraph_models as B  # noqa: E402

out = B.run_mlps(d_tr, d_va, d_te) | B.run_xgb(d_tr, d_va, d_te)
comp_out = {c: xgb_global(*v) for c, v in comps.items()}
for c, o in comp_out.items():
    print(f"{c}: cfg {o['cfg']}", flush=True)
out["XGBoost (b-global), decomposed"] = {
    "cfg": {c: o["cfg"] for c, o in comp_out.items()},
    "preds": [l - w - s for l, w, s in zip(*(comp_out[c]["preds"] for c in ("Load", "Wind_power", "Solar_power")))]}

rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
smape = lambda p: float(100 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y))))
res, arrays = {}, {}
for name, o in out.items():
    nat = np.stack([B.national(p) for p in o["preds"]])
    r = [rmse(p) for p in nat]
    res[name] = {"cfg": o["cfg"], "rmse": r, "rmse_mean": float(np.mean(r)),
                 "rmse_sd": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                 "smape_mean": float(np.mean([smape(p) for p in nat]))}
    arrays[name] = nat
    print(f"{name}: RMSE {res[name]['rmse_mean']:.0f} ± {res[name]['rmse_sd']:.0f} MW, "
          f"sMAPE {res[name]['smape_mean']:.2f}%", flush=True)
json.dump(res, open(os.path.join(HERE, "netload_nongraph.json"), "w"), indent=1)
np.savez_compressed(os.path.join(HERE, "netload_nongraph.npz"), **arrays)
