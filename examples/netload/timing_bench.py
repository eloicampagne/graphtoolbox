"""Per-epoch training time of the ten table convolutions, on load and net-load.

Times are wall-clock seconds per training epoch, averaged over a fixed number of
epochs at the basic configuration (hidden 64, one layer, DTW graph), so the
comparison isolates the operator cost from the early-stopping epoch count.
Runs on CPU (fastest on this 12-node graph). Produces a grouped bar chart and a
CSV.
"""
import os
import time
import csv
import numpy as np
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (LEConv, GatedGraphConv, GATv2Conv, ResGatedGraphConv,
                                     ARMAConv, SGConv, RGATConv, RGCNConv, GATConv, GCNConv)

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device
import netload_pipeline as P

set_device("cpu")
HIDDEN, LAYERS, ADJ, OUT = 64, 1, "dtw", 48
K = 20
HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
OUTPUT_DIR = os.environ.get(
    "TIMING_OUTPUT_DIR", os.path.join(HERE, "results_timing_netload"))

CONVS = {  # the ten load-table convolutions, basic kwargs
    "LEConv": (LEConv, {}), "GatedGraphConv": (GatedGraphConv, {}),
    "GATv2Conv": (GATv2Conv, {"heads": 2}), "ResGatedGraphConv": (ResGatedGraphConv, {}),
    "ARMAConv": (ARMAConv, {}), "SGConv": (SGConv, {}), "RGATConv": (RGATConv, {"heads": 2}),
    "RGCNConv": (RGCNConv, {}), "GATConv": (GATConv, {"heads": 2}), "GCNConv": (GCNConv, {}),
}


def build_load():
    feats = ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "JourFerie", "offset",
             "Weekend", "temp_liss_fort", "temp_liss_faible"]
    dk_data = {
        "node_var": "Region",
        "features_to_lag": {"load": (1, 48)},
        "dummies": ["Instant", "JourSemaine", "DayType", "offset"],
        "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
        "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
        "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
    }
    dk = {"batch_size": 32, "adj_matrix": ADJ,
          "features_base": feats + [f"load_l{t}" for t in range(1, 49)], "target_base": "load"}
    ld = os.path.join(EX, "load")
    data = DataClass(path_train=os.path.join(ld, "train.csv"), path_test=os.path.join(ld, "test.csv"),
                     data_kwargs=dk_data, folder_config=ld)
    common = dict(graph_folder=os.path.join(EX, "graph_representations"), dataset_kwargs=dk, out_channels=OUT)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix=ADJ)
    return tr, va, te


def time_conv(cls, kw, tr, va, te):
    seed_everything(42)
    model = myGNN(in_channels=tr.num_node_features, num_layers=LAYERS, hidden_channels=HIDDEN,
                  out_channels=OUT, conv_class=cls, conv_kwargs=kw)
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                      batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": K},
                      reconcile=False, lam_reg=0)
    t0 = time.perf_counter()
    trainer.train(plot_loss=False, force_training=True, save=False, patience=K + 1)
    return (time.perf_counter() - t0) / K


def main():
    print("Building datasets...")
    tr_l, va_l, te_l = build_load()
    tr_n, va_n, te_n, _ = P.build_datasets()
    for ds in (tr_n, va_n, te_n):
        ds._set_adj_matrix(adj_matrix=ADJ)
    nf_l, ns_l = tr_l.num_node_features, len(tr_l)
    nf_n, ns_n = tr_n.num_node_features, len(tr_n)
    print(f"load: {nf_l} features, {ns_l} days ; netload: {nf_n} features, {ns_n} days")

    rows = {}
    for name, (cls, kw) in CONVS.items():
        tl = time_conv(cls, kw, tr_l, va_l, te_l)
        tn = time_conv(cls, kw, tr_n, va_n, te_n)
        rows[name] = (tl, tn)
        print(f"  {name:<18} load {tl*1000:6.0f} ms/ep   netload {tn*1000:6.0f} ms/ep")

    order = sorted(rows, key=lambda n: rows[n][1])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "timing_per_epoch.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["conv", "load_s_per_epoch", "netload_s_per_epoch"])
        for n in order:
            w.writerow([n, f"{rows[n][0]:.4f}", f"{rows[n][1]:.4f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(order)); wd = 0.4
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.bar(x - wd/2, [rows[n][0]*1e3 for n in order], wd, color="#4C72B0",
           label=f"Load ({nf_l} features, {ns_l} days)")
    b2 = ax.bar(x + wd/2, [rows[n][1]*1e3 for n in order], wd, color="#DD8452",
                label=f"Net-load ({nf_n} features, {ns_n} days)")
    ax.set_yscale("log")
    ax.set_ylabel("Time per epoch (ms, log scale)")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Per-epoch training time (hidden 64, 1 layer, DTW graph, Apple M4 Pro CPU)")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=6)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUTPUT_DIR, f"timing_per_epoch.{ext}"), dpi=150)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
