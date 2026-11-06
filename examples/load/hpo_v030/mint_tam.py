"""MinT reconciliation of the best GNN of each task onto TAM, the strongest reference of both tables.

The top level is the national sum of the regional static TAM (tam_baselines.py --static writes its
2018 and 2019 forecasts, fitted before each year, to <task>/tam_static_forecasts.npz). The GNNs are
reloaded from their checkpoints, never retrained: GatedGraphConv on load (the sweep configuration),
FastRGCNConv on net-load, direct and decomposed (seed 42, weather trajectories of the forecast day).

The projection is the one of the Trainer: weighted least squares with a diagonal W holding the
2018 error variance of each regional forecast and of the top-level forecast. On load it runs through
the Trainer itself (external top-level forecasts) and once more offline, which checks that the
offline function reproduces it; the net-load forecasts are reconciled offline. A second estimator,
MinT-shrink (full 2018 error covariance with the shrinkage of Schafer and Strimmer), is reported
alongside. Writes ./mint_tam.json.

With --top xgb, the top level is instead the national sum of pooled XGBoost (b-global of nongraph_baselines.py
and netload_nongraph.py), refitted here with its selected configuration and seed 42 on the datasets of the GNN,
which gives its 2018 forecasts for the error variances. With --top mlp, it is the national sum of the MLP fitted
per region (nongraph_models.mlp_fit, configuration of the tables, seed 42). Writes ./mint_<top>.json and the
forecasts to ./mint_<top>_forecasts.npz.

    python mint_tam.py [load] [netload] [--top xgb|mlp]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import FastRGCNConv, GatedGraphConv

from graphtoolbox.evaluation import diebold_mariano
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(os.path.dirname(HERE))
H = 48


def per_node(trainer, dataset, split):
    """Unreconciled forecasts and targets [num_nodes, T] on a split, one graph per batch."""
    trainer.reconcile = False
    _, pred, true = trainer._predict(DataLoader(dataset, shuffle=False), split=split)
    return pred.numpy().astype(float), true.numpy().astype(float)


def reconcile(base, top, base_err, top_err, kind):
    """National forecast after projecting [base; top] onto the coherent subspace (MinT/WLS)."""
    n = base.shape[0]
    S = np.vstack([np.eye(n), np.ones((1, n))])
    E = np.vstack([base_err, top_err[None]])
    if kind == "wls":
        W = np.diag(E.var(axis=1, ddof=1))
    else:  # MinT-shrink (Wickramasuriya et al., 2019), shrinkage intensity of Schafer and Strimmer as in hts
        T = E.shape[1]
        cov = E @ E.T / T
        xs = E / np.sqrt(np.diag(cov))[:, None]
        v = ((xs ** 2) @ (xs ** 2).T - (xs @ xs.T) ** 2 / T) / (T * (T - 1))
        cor = xs @ xs.T / T
        np.fill_diagonal(v, 0.0)
        np.fill_diagonal(cor, 0.0)
        lam = float(np.clip(v.sum() / (cor ** 2).sum(), 0.0, 1.0))
        W = lam * np.diag(np.diag(cov)) + (1 - lam) * cov
    Wi = np.linalg.inv(W)
    G = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
    return (S @ G @ np.vstack([base, top[None]]))[-1]


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


FORECASTS = {}  # national forecasts of the 2019 test set, written to mint_<top>_forecasts.npz


def score(name, base_val, y_val_nodes, base_test, top_val, top_test, y_test, ref=None):
    """RMSE and Diebold-Mariano p-values of the unreconciled and reconciled national forecasts."""
    y_val = y_val_nodes.sum(0)
    base_err, top_err = base_val - y_val_nodes, top_val - y_val
    unrec = base_test.sum(0)
    FORECASTS.update({f"{name}__target": y_test, f"{name}__unreconciled": unrec, f"{name}__top": top_test})
    out = {"unreconciled": rmse(unrec, y_test), "top_level_TAM": rmse(top_test, y_test),
           "top_level_TAM_2018": rmse(top_val, y_val), "gnn_2018": rmse(base_val.sum(0), y_val)}
    for kind in ("wls", "shrink"):
        rec = reconcile(base_test, top_test, base_err, top_err, kind)
        out[f"mint_{kind}"] = rmse(rec, y_test)
        FORECASTS[f"{name}__mint_{kind}"] = rec
        out[f"dm_p_{kind}_vs_unreconciled"] = float(diebold_mariano(rec, unrec, y_test, loss="squared", h=H).pvalue)
        out[f"dm_p_{kind}_vs_TAM"] = float(diebold_mariano(rec, top_test, y_test, loss="squared", h=H).pvalue)
    if ref is not None:
        out["trainer_wls"] = ref
    print(name, json.dumps(out, indent=1), flush=True)
    return out


TOP = sys.argv[sys.argv.index("--top") + 1] if "--top" in sys.argv else "tam"


def xgb_pooled(task, tr, va, te):
    """2018 and 2019 national forecasts of pooled XGBoost, one model per half-hour, seed 42."""
    import xgboost as xgb
    name = {"load": "nongraph_baselines.json", "netload": "netload_nongraph.json"}[task]
    res = json.load(open(os.path.join(HERE, name)))["XGBoost (b-global)"]
    stack = lambda ds: (np.stack([d.x.numpy() for d in ds.pyg_data]).astype(np.float32),
                        np.stack([d.y.numpy() for d in ds.pyg_data]).astype(np.float64))
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = stack(tr), stack(va), stack(te)
    N, F = Xtr.shape[1], Xtr.shape[2]
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    Ys = ((Ytr - mu) / sd).reshape(-1, H)
    pv, pt = np.zeros((Xva.shape[0] * N, H)), np.zeros((Xte.shape[0] * N, H))
    for h in range(H):
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, **res["cfg"])
        m.fit(Xtr.reshape(-1, F), Ys[:, h])
        pv[:, h], pt[:, h] = m.predict(Xva.reshape(-1, F)), m.predict(Xte.reshape(-1, F))
    nat = lambda P, X: (P.reshape(X.shape[0], N, H) * sd + mu).sum(1).reshape(-1)
    top_val, top_test = nat(pv, Xva), nat(pt, Xte)
    y_test = Yte.sum(1).reshape(-1)
    print(f"{task} pooled XGBoost refitted: 2019 RMSE {rmse(top_test, y_test):.1f} MW "
          f"(stored seed 42: {res['rmse'][0]:.1f})", flush=True)
    return top_val, Yva.sum(1).reshape(-1), top_test


def mlp_regional(task, tr, va, te):
    """2018 and 2019 national forecasts of the MLP fitted per region (nongraph_models.mlp_fit), seed 42."""
    sys.path.insert(0, HERE)
    import nongraph_models as B
    name = {"load": "nongraph_baselines.json", "netload": "netload_nongraph.json"}[task]
    res = json.load(open(os.path.join(HERE, name)))["Per-node MLP"]
    stack = lambda ds: (np.stack([d.x.numpy() for d in ds.pyg_data]).astype(np.float32),
                        np.stack([d.y.numpy() for d in ds.pyg_data]).astype(np.float64))
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = stack(tr), stack(va), stack(te)
    pt, pv = B.mlp_fit(Xtr, Ytr, Xva, Yva, Xte, res["cfg"], 42, True)
    top_val, top_test, y_test = B.national(pv), B.national(pt), Yte.sum(1).reshape(-1)
    print(f"{task} regional MLP refitted: 2019 RMSE {rmse(top_test, y_test):.1f} MW "
          f"(stored seed 42: {res['rmse'][0]:.1f})", flush=True)
    return top_val, Yva.sum(1).reshape(-1), top_test


def tam(task, n_val_days, first_val_offset, n_test):
    z = np.load(os.path.join(EXAMPLES, task, "tam_static_forecasts.npz"))
    a = first_val_offset * H
    return z["val"][a:a + n_val_days * H], z["y_val"][a:a + n_val_days * H], z["test"][:n_test]


def load_task():
    src = open(os.path.join(HERE, "mint_all.py")).read().split("with np.load")[0]
    g = {"__file__": os.path.join(HERE, "mint_all.py"), "__name__": "mint_setup"}
    exec(compile(src, "mint_all.py", "exec"), g)               # data and dataset kwargs of the sweep
    data = g["DataClass"](path_train="./train.csv", path_test="./test.csv",
                          data_kwargs=g["data_kwargs"], folder_config=".")
    common = dict(graph_folder="../graph_representations", dataset_kwargs=g["dataset_kwargs"], out_channels=48)
    tr = g["GraphDataset"](data=data, period="train", **common)
    va = g["GraphDataset"](data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te = g["GraphDataset"](data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix="dtw")
    days = sorted(pd.to_datetime(va.dataframe["date"], utc=True).dt.tz_localize(None).dt.normalize().unique())
    offset = int((days[0] - np.datetime64("2018-01-01")) / np.timedelta64(1, "D"))
    with np.load(os.path.join(EXAMPLES, "load", "forecasts_paper", "load_sweep.npz")) as z:
        archive, y_test = z["GatedGraphConv"].astype(float), z["target"].astype(float)
    top_val, y_val_tam, top_test = (tam("load", len(va), offset, len(y_test)) if TOP == "tam"
                                    else xgb_pooled("load", tr, va, te))

    def trainer(top):
        seed_everything(42)
        model = myGNN(in_channels=tr.num_node_features, num_layers=2, hidden_channels=64, out_channels=48,
                      conv_class=GatedGraphConv, heads=4)
        kw = dict(top_forecasts_val=torch.tensor(top_val, dtype=torch.float32),
                  top_forecasts_test=torch.tensor(top_test, dtype=torch.float32), top_level_model=None) if top \
            else dict(top_level_model="xgb")
        return Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=32,
                       model_kwargs={"lr": 1e-3, "num_epochs": 300}, reconcile=True, lam_reg=0, **kw)

    base = "./checkpoints/GatedGraphConv_dtw/batch32_hidden64_layers2_epochs300"
    t = trainer(top=True)
    rec = t.train(force_training=False, saving_directory=base, patience=30, plot_loss=False)[0].numpy().sum(0)
    base_test, _ = per_node(t, te, "test")
    base_val, y_val_nodes = per_node(t, va, "val")
    assert np.max(np.abs(base_test - archive)) < 1.0, "checkpoint differs from the archived forecast"
    # The gradient-boosting top level of the paper's reconciliation paragraph, from the same checkpoint.
    rec_xgb = trainer(top=False).train(force_training=False, saving_directory=base, patience=30,
                                       plot_loss=False)[0].numpy().sum(0)
    print(f"load GatedGraphConv, gradient-boosting top level: {rmse(rec_xgb, y_test):.1f} MW", flush=True)
    assert np.max(np.abs(y_val_nodes.sum(0) - y_val_tam)) < 1e-3, "2018 targets of TAM and of the GNN differ"
    out = score("load GatedGraphConv", base_val, y_val_nodes, base_test, top_val, top_test, y_test,
                ref=rmse(rec, y_test))
    out["trainer_wls_xgb_top"] = rmse(rec_xgb, y_test)
    FORECASTS["load GatedGraphConv__trainer_wls_xgb_top"] = rec_xgb
    out["dm_p_trainer_xgb_top_vs_unreconciled"] = float(
        diebold_mariano(rec_xgb, base_test.sum(0), y_test, loss="squared", h=H).pvalue)
    return out


def netload_task():
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(EXAMPLES, "netload"))
    cwd = os.getcwd()
    os.chdir(os.path.join(EXAMPLES, "netload"))
    import netload_pipeline as P
    import decomp_sweep_netload as D
    from netload_data_v030 import direct_datasets, lead_cols
    from graphtoolbox.data.dataset import DataClass, GraphDataset
    ck = os.path.join(HERE, "checkpoints_netload_v030", "leads_seed42")

    def predict(dsets, tag):
        tr, va, te = dsets
        for ds in dsets:
            ds._set_adj_matrix(adj_matrix="dtw")
        seed_everything(42)
        model = myGNN(in_channels=tr.num_node_features, num_layers=1, hidden_channels=64,
                      out_channels=P.OUT_CHANNELS, conv_class=FastRGCNConv, conv_kwargs={})
        t = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=16,
                    model_kwargs={"lr": 1e-3, "num_epochs": 300}, reconcile=False, lam_reg=0)
        t.train(force_training=False, save=False, patience=40, plot_loss=False, saving_directory=os.path.join(ck, tag))
        return per_node(t, va, "val"), per_node(t, te, "test"), va

    (pv, yv), (pt, yt), va = predict(direct_datasets(True), "FastRGCNConv")
    archived = torch.load(os.path.join(HERE, "netload_v030", "leads_seed42", "FastRGCNConv.pt"), weights_only=False).numpy()
    assert np.max(np.abs(pt - archived)) < 1.0, "checkpoint differs from the archived forecast"
    days = sorted(pd.to_datetime(va.dataframe["date"], utc=True).dt.tz_localize(None).dt.normalize().unique())
    offset = int((days[0] - np.datetime64("2018-01-01")) / np.timedelta64(1, "D"))
    y_test = yt.sum(0)
    top_val, y_val_tam, top_test = (tam("netload", len(va), offset, len(y_test)) if TOP == "tam"
                                    else {"xgb": xgb_pooled, "mlp": mlp_regional}[TOP]("netload", *direct_datasets(True)))
    assert np.max(np.abs(yv.sum(0) - y_val_tam)) < 1e-3, "2018 targets of TAM and of the GNN differ"
    out = {"direct FastRGCNConv": score("netload direct FastRGCNConv", pv, yv, pt, top_val, top_test, y_test)}

    dkw = dict(D.data_kwargs)
    dkw["features_to_lag"] = {**D.data_kwargs["features_to_lag"], **{v: (-47, -1) for v in
                              ["wind", "wind_by_wind_power_weights", "nebulosity", "nebulosity_by_solar_power_weights"]}}
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=dkw, folder_config=str(P.HERE))
    extra = {"Load": lead_cols({"wind", "cloud"}), "Wind_power": lead_cols({"wind"}), "Solar_power": lead_cols({"cloud"})}
    sign = {"Load": 1.0, "Wind_power": -1.0, "Solar_power": -1.0}
    pv_d = pt_d = 0.0
    for comp, feats in D.COMPONENTS.items():
        dk = {"batch_size": 32, "adj_matrix": "dtw", "features_base": feats + extra[comp], "target_base": comp}
        common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
        tr = GraphDataset(data=data, period="train", **common)
        dsets = (tr, GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common),
                 GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common))
        (cv, _), (ct, _), _ = predict(dsets, f"FastRGCNConv__{comp}")
        pv_d, pt_d = pv_d + sign[comp] * cv, pt_d + sign[comp] * ct
    out["decomposed FastRGCNConv"] = score("netload decomposed FastRGCNConv", pv_d, yv, pt_d, top_val, top_test, y_test)
    os.chdir(cwd)
    return out


if __name__ == "__main__":
    os.chdir(os.path.join(EXAMPLES, "load"))
    tasks = [a for a in sys.argv[1:] if a in ("load", "netload")] or ["load", "netload"]
    path = os.path.join(HERE, f"mint_{TOP}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if "load" in tasks:
        res["load"] = load_task()
        json.dump(res, open(path, "w"), indent=1)
    if "netload" in tasks:
        res["netload"] = netload_task()
        json.dump(res, open(path, "w"), indent=1)
    fpath = os.path.join(HERE, f"mint_{TOP}_forecasts.npz")
    old = dict(np.load(fpath)) if os.path.exists(fpath) else {}
    np.savez_compressed(fpath, **(old | FORECASTS))
