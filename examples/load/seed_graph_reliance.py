"""Measure forecast error and graph reliance across random seeds."""
import os
import sys
import gc
import json
import numpy as np
import torch
import torch_geometric.nn.conv as pyg_conv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("EXPL_ADJ", "space")
import run_gnnexplainer_load as R
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device

CONVS = (os.environ.get("SEED_CONVS", "GATConv,GATv2Conv")).split(",")
SEEDS = [int(s) for s in os.environ.get("SEED_LIST", "42,0,1,2,3").split(",")]
ADJ = os.environ.get("EXPL_ADJ", "space")
EPOCHS = int(os.environ.get("ABL_EPOCHS", 300))
NDAYS = int(os.environ.get("DIAG_DAYS", 20))
OUT = os.path.join(HERE, "results_seed_graph_reliance")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, f"reliance_{ADJ}.json")
res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def measure(model, test):
    """Neighbor share per layer, node-spread of the conv term, attention mass on self."""
    empty = torch.empty((2, 0), dtype=torch.long)
    shares, spreads = [], []
    with torch.no_grad():
        for d in range(min(NDAYS, len(test))):
            g = test[d]
            x = model.node_encoder(g.x)
            sh, sp = [], []
            for layer in model.layers:
                hin = layer.act(layer.norm(x))
                h = layer.conv(hin, g.edge_index)
                nb = h - layer.conv(hin, empty)
                sh.append(float(torch.norm(nb) / (torch.norm(h) + 1e-12)))
                sp.append(float(h.std(0).mean() / (h.abs().mean() + 1e-12)))
                x = x + h
            shares.append(sh); spreads.append(sp)

    self_mass = None
    ad = model.layers[0].conv
    if hasattr(ad, "capture_attention"):
        ad.capture_attention = True
        with torch.no_grad():
            g = test[0]
            L0 = model.layers[0]
            L0.conv(L0.act(L0.norm(model.node_encoder(g.x))), g.edge_index)
        if getattr(ad, "last_attention", None) is not None:
            ei, aw = ad.last_attention
            a = aw.mean(-1)
            m = ei[0] == ei[1]
            self_mass = float(a[m].sum() / a.sum())
        ad.capture_attention = False
    return np.mean(shares, axis=0), np.mean(spreads, axis=0), self_mass


def run(conv_name, seed):
    tr, va, te = R.datasets()
    set_device("cpu")
    torch.manual_seed(seed); np.random.seed(seed); gc.collect()
    model = myGNN(in_channels=tr.num_node_features, num_layers=2, hidden_channels=64,
                  out_channels=48, conv_class=getattr(pyg_conv, conv_name),
                  conv_kwargs={}, heads=1)
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                      batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    pred, target, *_ = trainer.train(plot_loss=False, force_training=True, save=False, patience=75)
    p, y = pred.sum(0).cpu().numpy(), target.sum(0).cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    sh, sp, sm = measure(model, te)
    return rmse, sh.tolist(), sp.tolist(), sm


def main():
    print(f"### seed study, {ADJ} graph, h64/l2/{EPOCHS}ep, seeds {SEEDS} ###", flush=True)
    for c in CONVS:
        for s in SEEDS:
            key = f"{c}|s{s}"
            if key in res:
                continue
            try:
                rmse, sh, sp, sm = run(c, s)
                res[key] = {"rmse": rmse, "share": sh, "spread": sp, "self_mass": sm}
                json.dump(res, open(CACHE, "w"), indent=1)
                print(f"[{c:11} seed {s:2}] RMSE {rmse:6.0f} MW  share {np.round(sh,3)}  "
                      f"self-attention mass {sm:.3f}" if sm is not None else
                      f"[{c:11} seed {s:2}] RMSE {rmse:6.0f} MW  share {np.round(sh,3)}", flush=True)
            except Exception as e:
                print(f"[{c:11} seed {s:2}] FAILED: {str(e).splitlines()[0]}", flush=True)

    print("\n===== summary (mean +/- sd over seeds) =====")
    for c in CONVS:
        v = [res[f"{c}|s{s}"] for s in SEEDS if f"{c}|s{s}" in res]
        if not v:
            continue
        r = np.array([x["rmse"] for x in v])
        sh = np.array([np.mean(x["share"]) for x in v])
        sm = np.array([x["self_mass"] for x in v if x["self_mass"] is not None])
        print(f"{c:11} (n={len(v)})")
        print(f"   national RMSE      {r.mean():6.0f} +/- {r.std(ddof=1):4.0f} MW   [{r.min():.0f}, {r.max():.0f}]")
        print(f"   neighbor share     {sh.mean():.4f} +/- {sh.std(ddof=1):.4f}   [{sh.min():.3f}, {sh.max():.3f}]")
        if len(sm):
            print(f"   attention on self  {sm.mean():.3f} +/- {sm.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
