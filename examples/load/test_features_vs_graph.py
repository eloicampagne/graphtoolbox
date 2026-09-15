"""Compare graph and graph-free models with two feature configurations."""
import os
import sys
import gc
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("EXPL_ADJ", "space")
import run_gnnexplainer_load as R
import config as C
from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device
from torch_geometric.nn.conv import SAGEConv

ADJ = os.environ.get("EXPL_ADJ", "space")
SEEDS = [int(s) for s in os.environ.get("SEED_LIST", "42,0,1").split(",")]
HIDDEN = int(os.environ.get("FV_HIDDEN", 256))
EPOCHS = int(os.environ.get("ABL_EPOCHS", 300))


def datasets_poor():
    """The example config: no temperature lags, no computed nonlinear terms."""
    data = DataClass(path_train=os.path.join(HERE, "train.csv"),
                     path_test=os.path.join(HERE, "test.csv"),
                     data_kwargs=C.data_kwargs, folder_config=HERE)
    dk = dict(C.dataset_kwargs); dk["adj_matrix"] = ADJ
    common = dict(graph_folder=os.path.join(os.path.dirname(HERE), "graph_representations"),
                  dataset_kwargs=dk, out_channels=48)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix=ADJ)
    return tr, va, te


def stack(ds):
    return (np.stack([d.x.numpy() for d in ds]), np.stack([d.y.numpy() for d in ds]))


def run_gnn(dsets, seed):
    tr, va, te = dsets
    set_device("cpu"); torch.manual_seed(seed); gc.collect()
    m = myGNN(in_channels=tr.num_node_features, num_layers=2, hidden_channels=HIDDEN,
              out_channels=48, conv_class=SAGEConv, conv_kwargs={}, heads=1)
    t = Trainer(model=m, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=16,
                model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS}, reconcile=True,
                top_level_model="xgb", lam_reg=0)
    pred, target, *_ = t.train(plot_loss=False, force_training=True, save=False, patience=75)
    p, y = pred.sum(0).cpu().numpy(), target.sum(0).cpu().numpy()
    return float(np.sqrt(np.mean((p - y) ** 2)))


def run_mlp(dsets, seed):
    tr, va, te = dsets
    Xtr, Ytr = stack(tr); Xva, Yva = stack(va); Xte, Yte = stack(te)
    torch.manual_seed(seed); np.random.seed(seed)
    pred = np.zeros_like(Yte)
    for i in range(Xtr.shape[1]):
        A, B, Cc = Xtr[:, i, :], Xva[:, i, :], Xte[:, i, :]
        mu, sg = A.mean(0), A.std(0) + 1e-6
        a, b, c = [torch.tensor((z - mu) / sg, dtype=torch.float32) for z in (A, B, Cc)]
        ym, ys = Ytr[:, i, :].mean(), Ytr[:, i, :].std() + 1e-6
        yt = torch.tensor((Ytr[:, i, :] - ym) / ys, dtype=torch.float32)
        yv = torch.tensor((Yva[:, i, :] - ym) / ys, dtype=torch.float32)
        m = nn.Sequential(nn.Linear(a.shape[1], 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 48))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.MSELoss()
        best = (1e18, None); bad = 0
        for ep in range(EPOCHS):
            m.train(); perm = torch.randperm(len(a))
            for k in range(0, len(a), 64):
                idx = perm[k:k + 64]; opt.zero_grad(); lf(m(a[idx]), yt[idx]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = float(lf(m(b), yv))
            if vl < best[0] - 1e-6:
                best = (vl, {k: v.clone() for k, v in m.state_dict().items()}); bad = 0
            else:
                bad += 1
                if bad >= 75:
                    break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad():
            pred[:, i, :] = m(c).numpy() * ys + ym
    pn, yn = pred.sum(1).ravel(), Yte.sum(1).ravel()
    return float(np.sqrt(np.mean((pn - yn) ** 2)))


def main():
    out = {}
    for fname, builder in (("poor", datasets_poor), ("rich", R.datasets)):
        ds = builder()
        F = ds[0].num_node_features
        print(f"\n##### {fname} features: {F} per node #####", flush=True)
        for mname, fn in (("GNN(SAGE)", run_gnn), ("MLP/region", run_mlp)):
            v = []
            for s in SEEDS:
                try:
                    v.append(fn(ds, s))
                    print(f"[{fname:4} {mname:10} seed {s:2}] RMSE {v[-1]:.0f} MW", flush=True)
                except Exception as e:
                    print(f"[{fname:4} {mname:10} seed {s:2}] FAILED: {str(e).splitlines()[0]}", flush=True)
            if v:
                out[(fname, mname)] = np.array(v)

    print("\n===== does richer own-information devalue the graph? =====")
    for f in ("poor", "rich"):
        g, m = out.get((f, "GNN(SAGE)")), out.get((f, "MLP/region"))
        if g is None or m is None:
            continue
        print(f"{f:4}: GNN {g.mean():6.0f} +/- {g.std(ddof=1):3.0f}   "
              f"MLP {m.mean():6.0f} +/- {m.std(ddof=1):3.0f}   "
              f"graph edge {m.mean()-g.mean():+.0f} MW")
    print("\nINFORMS (earlier feature set): SAGE 998 +/- 72 vs FF/region 1318 +/- 9 "
          "-> graph edge +320 MW")


if __name__ == "__main__":
    main()
