"""Run reproducibility and stability diagnostics for load-model edge attributions."""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch_geometric.nn.conv as pyg_conv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.interpretability import (
    compute_edge_masks,
    explainer_reproducibility,
    topk_indices,
)
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device

CONV = os.environ.get("EXPL_CONV", "GATv2Conv")
ADJ = os.environ.get("EXPL_ADJ", "dtw")
EPOCHS = int(os.environ.get("EXPL_EPOCHS", 100))          # explainer optimization steps
LIMIT = int(os.environ.get("EXPL_LIMIT", 0))              # 0 = all test days
TRAIN_EPOCHS = int(os.environ.get("EXPL_TRAIN_EPOCHS", 300))
HIDDEN, LAYERS, BATCH = 64, 2, 32

OUT = os.path.join(HERE, "results_gnnexplainer")
os.makedirs(OUT, exist_ok=True)
MASKS = os.path.join(OUT, f"edge_masks_{CONV}_{ADJ}.npy")
CKPT_DIR = os.path.join(HERE, "checkpoints", f"{CONV}_{ADJ}",
                        f"batch{BATCH}_hidden{HIDDEN}_layers{LAYERS}_epochs{TRAIN_EPOCHS}")

# The checkpointed model the paper reports comes from the ICTAI run_experiments
# notebook: its feature set adds four computed columns to the example config,
# which is what takes the encoder from 183 to 187 inputs.
DATA_KWARGS = {
    "node_var": "Region",
    "features_to_lag": {"load": (1, 48), "temp": (-47, -1)},
    "dummies": ["Instant", "RegionInt", "JourSemaine", "DayType", "offset"],
    "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
    "computed_features": {
        "heat_deg": lambda df: (15 - df["temp"]).clip(lower=0),
        "cool_deg": lambda df: (df["temp"] - 15).clip(lower=0),
        "temp_sq": lambda df: df["temp"] ** 2,
        "is_summer": lambda df: (pd.to_datetime(df["date"]).dt.month == 7).astype(float)
                              + (pd.to_datetime(df["date"]).dt.month == 8).astype(float),
    },
}
DATASET_KWARGS = {
    "batch_size": BATCH,
    "adj_matrix": ADJ,
    "features_base": ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "Instant",
                      "RegionInt", "JourSemaine", "JourFerie", "offset", "DayType", "Weekend",
                      "temp_liss_fort", "temp_liss_faible",
                      "heat_deg", "cool_deg", "temp_sq", "is_summer"]
                     + [f"load_l{t}" for t in range(1, 49)]
                     + [f"temp_l{t}" for t in range(-47, 0)],
    "target_base": "load",
}


def datasets():
    data = DataClass(path_train=os.path.join(HERE, "train.csv"),
                     path_test=os.path.join(HERE, "test.csv"),
                     data_kwargs=DATA_KWARGS, folder_config=HERE)
    common = dict(graph_folder=os.path.join(os.path.dirname(HERE), "graph_representations"),
                  dataset_kwargs=DATASET_KWARGS, out_channels=48)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix=ADJ)
    return tr, va, te


def build():
    tr, va, te = datasets()
    conv_class = getattr(pyg_conv, CONV)
    ckpt = None
    if os.path.isdir(CKPT_DIR):
        found = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".params"))
        ckpt = os.path.join(CKPT_DIR, found[-1]) if found else None

    if ckpt:
        sd = torch.load(ckpt, weights_only=False, map_location="cpu")
        # `heads` is a myGNN argument, not a conv kwarg: the adapter splits the hidden
        # width across the heads, so the checkpoint's att shape reveals the head count.
        att = sd.get("layers.0.conv.conv.att", None)
        heads = int(att.shape[1]) if att is not None else 1
        model = myGNN(in_channels=tr.num_node_features, num_layers=LAYERS, hidden_channels=HIDDEN,
                      out_channels=48, conv_class=conv_class, conv_kwargs={}, heads=heads)
        model.load_state_dict(sd)
        print(f"loaded {CONV}_{ADJ}/{os.path.basename(ckpt)} (heads={heads})")
    else:
        print(f"no checkpoint under {CKPT_DIR}, training {CONV} on the {ADJ} graph "
              f"for {TRAIN_EPOCHS} epochs ...")
        set_device("cpu")
        torch.manual_seed(42)
        model = myGNN(in_channels=tr.num_node_features, num_layers=LAYERS, hidden_channels=HIDDEN,
                      out_channels=48, conv_class=conv_class, conv_kwargs={}, heads=1)
        trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                          batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": TRAIN_EPOCHS},
                          reconcile=True, top_level_model="xgb", lam_reg=0,
                          saving_directory=CKPT_DIR)
        pred, target, *_ = trainer.train(plot_loss=False, force_training=True, save=True, patience=75)
        p, y = pred.sum(0).cpu().numpy(), target.sum(0).cpu().numpy()
        print(f"trained: national RMSE {np.sqrt(np.mean((p - y) ** 2)):.0f} MW")
    model.eval()
    g = te[0]
    print(f"graph {ADJ}: {g.edge_index.shape[1]} edges over {g.num_nodes} nodes "
          f"(mean degree {g.edge_index.shape[1]/g.num_nodes:.1f}), {len(te)} test days")
    return te, model


def topk(v, K):
    return topk_indices(v, K)


def reproducibility(test, model, day=0, runs=2):
    """Do two independent explainer runs on one day agree? The weakest possible bar."""
    result = explainer_reproducibility(
        test, model, day=day, runs=runs, epochs=EPOCHS)
    c = result["correlation"]
    K = result["top_count"]
    ov = result["top_overlap"]
    chance = result["chance_overlap"]
    print("\n=== 1. Reproducibility (same day, same model, two explainer runs) ===")
    print(f"corr {c:+.3f}   top-{K} overlap {ov}/{K} (chance ~{chance:.1f})   "
          f"{'PASS' if c > 0.5 else 'FAIL: attributions are not reproducible'}")
    return c


def stability_and_season(em):
    dates = pd.date_range(DATA_KWARGS["day_inf_test"], periods=em.shape[0], freq="D")
    jan = np.where(dates.month_name() == "January")[0]
    jul = np.where(dates.month_name() == "July")[0]
    K = max(1, em.shape[1] // 10)
    chance = K * K / em.shape[1]
    rng = np.random.default_rng(0)
    wl, wo, wc = [], [], []
    for ix in (jan, jul):
        for _ in range(200):
            p = rng.permutation(ix)
            a, b = em[p[:len(p)//2]].mean(0), em[p[len(p)//2:]].mean(0)
            wl.append(np.abs(a - b).mean()); wo.append(len(topk(a, K) & topk(b, K)))
            wc.append(np.corrcoef(a, b)[0, 1])
    within = float(np.mean(wl))
    print("\n=== 2. Within-month stability (two halves of one month) ===")
    print(f"corr {np.mean(wc):+.3f}   top-{K} overlap {np.mean(wo):.1f}/{K} (chance ~{chance:.1f})   "
          f"L1 {within:.6f}")

    mj, mu = em[jan].mean(0), em[jul].mean(0)
    between = np.abs(mj - mu).mean()
    print("\n=== 3. Seasonal contrast (January vs July) ===")
    print(f"mask range over the year [{em.mean(0).min():.4f}, {em.mean(0).max():.4f}]")
    print(f"L1 {between:.6f}   top-{K} overlap {len(topk(mj, K) & topk(mu, K))}/{K}")
    print(f"ratio between/within = {between/within:.2f}  "
          f"({'seasonal signal exceeds within-month noise' if between/within > 1 else 'NO seasonal signal'})")


def main():
    print(f"### {CONV} on the {ADJ} graph ###")
    test, model = build()
    reproducibility(test, model)
    if os.path.exists(MASKS) and not os.environ.get("EXPL_FORCE"):
        em = np.load(MASKS); print(f"\nreusing cached masks {em.shape}")
    else:
        em = compute_edge_masks(
            test, model, epochs=EPOCHS, limit=LIMIT or None,
            output_path=MASKS)
        print(f"edge masks {em.shape} -> {os.path.basename(MASKS)}")
    stability_and_season(em)


if __name__ == "__main__":
    main()
