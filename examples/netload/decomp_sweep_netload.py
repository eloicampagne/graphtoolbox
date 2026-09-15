"""Decomposed net-load for the ten best load convolutions, component by component.

For each of the ten leading convolutions on gross load (Table 2 of the paper),
trains one model per physical component (Load / Wind / Solar) at a basic config
(hidden=64, 1 layer), combines NetLoad = Load - Wind - Solar nationally, and
reports the result against the direct model, the aggregation and the GAM.
Component datasets are built once and reused across convolutions.
"""
import os
import gc
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (RGATConv, GatedGraphConv, SuperGATConv, FiLMConv,
                                     PANConv, GMMConv, GCN2Conv, GATConv, FastRGCNConv,
                                     ResGatedGraphConv)

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer
import netload_pipeline as P

HIDDEN, LAYERS, EPOCHS, ADJ = 64, 1, 300, "dtw"
OUT = os.path.join(P.HERE, "results_std_decomp_netload")
os.makedirs(OUT, exist_ok=True)

# The ten convolutions of the direct net-load block, with basic kwargs.
CONVS = {
    "RGATConv": (RGATConv, {"heads": 2}), "GatedGraphConv": (GatedGraphConv, {}),
    "SuperGATConv": (SuperGATConv, {}), "FiLMConv": (FiLMConv, {}), "PANConv": (PANConv, {}),
    "GMMConv": (GMMConv, {}), "GCN2Conv": (GCN2Conv, {}), "GATConv": (GATConv, {"heads": 2}),
    "FastRGCNConv": (FastRGCNConv, {}), "ResGatedGraphConv": (ResGatedGraphConv, {}),
}

data_kwargs = {
    "node_var": "Region",
    "features_to_lag": {"Load": (1, 48), "Wind_power": (1, 48), "Solar_power": (1, 48),
                        "temperature": (-47, -1)},
    "dummies": ["tod", "day_type_week"],
    "day_inf_train": "2014-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
    "computed_features": P.COMPUTED_FEATURES,
}
feat_load = (["temperature", "temperature_lisse_990", "temperature_lisse_950", "wind", "nebulosity",
             "toy", "year", "month", "day_type_jf", "day_type_week", "period_holiday", "period_hour_changed",
             "period_holiday_zone_a", "period_holiday_zone_b", "period_holiday_zone_c",
             "period_christmas", "period_summer"] + P.ENG_LOAD)
feat_wind = ["wind", "wind_by_wind_power_weights", "toy", "year", "month"] + P.ENG_WIND
feat_solar = ["nebulosity", "nebulosity_by_solar_power_weights", "toy", "year", "month"] + P.ENG_SOLAR
COMPONENTS = {
    "Load": feat_load + [f"Load_l{t}" for t in range(1, 49)] + [f"temperature_l{t}" for t in range(-47, 0)],
    "Wind_power": feat_wind + [f"Wind_power_l{t}" for t in range(1, 49)] + [f"temperature_l{t}" for t in range(-47, 0)],
    "Solar_power": feat_solar + [f"Solar_power_l{t}" for t in range(1, 49)] + [f"temperature_l{t}" for t in range(-47, 0)],
}


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main():
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=data_kwargs, folder_config=str(P.HERE))
    # Build the three component datasets once.
    dsets, tgt_nat = {}, {}
    for name, feats in COMPONENTS.items():
        dk = {"batch_size": 32, "adj_matrix": ADJ, "features_base": feats, "target_base": name}
        common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
        tr = GraphDataset(data=data, period="train", **common)
        va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                          scalers_target=tr.scalers_target, **common)
        te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                          scalers_target=tr.scalers_target, **common)
        for ds in (tr, va, te):
            ds._set_adj_matrix(adj_matrix=ADJ)
        dsets[name] = (tr, va, te)

    y = torch.load(os.path.join(P.HERE, "results_all_convolutions_netload/target.pt"),
                   weights_only=False).numpy()
    rows = []
    for cname, (cls, kw) in CONVS.items():
        comp_pred = {}
        ok = True
        for comp, (tr, va, te) in dsets.items():
            ppath = os.path.join(OUT, f"{cname}__{comp}.pt")
            if os.path.exists(ppath):
                pred = torch.load(ppath, weights_only=False)
            else:
                print(f"\nTraining {cname} / {comp}...")
                seed_everything(42); gc.collect()
                try:
                    model = myGNN(in_channels=tr.num_node_features, num_layers=LAYERS,
                                  hidden_channels=HIDDEN, out_channels=P.OUT_CHANNELS,
                                  conv_class=cls, conv_kwargs=kw)
                    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                                      batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                                      reconcile=True, top_level_model="xgb", lam_reg=0)
                    res = trainer.train(force_training=True, save=False, patience=40)
                    pred = res[0]
                    torch.save(pred, ppath)
                    if comp not in tgt_nat:
                        tgt_nat[comp] = res[1].sum(dim=0).cpu().numpy()
                except Exception as e:
                    print(f"  {cname}/{comp} failed: {str(e).splitlines()[0]}")
                    ok = False; break
            comp_pred[comp] = pred.sum(dim=0).cpu().numpy()
        if not ok:
            continue
        net = comp_pred["Load"] - comp_pred["Wind_power"] - comp_pred["Solar_power"]
        net = net[:len(y)]
        rows.append((cname, smape(net, y), rmse(net, y)))
        print(f"  => {cname:<18} decomposed net-load  sMAPE={smape(net, y):.2f}%  RMSE={rmse(net, y):.0f} MW")

    rows.sort(key=lambda x: x[2])
    print("\n=== Decomposed net-load (one GNN per component, h64/l1), 2019 test ===")
    print(f'{"Convolution":<20}{"sMAPE":>8}{"RMSE":>10}')
    for n, s, r in rows:
        print(f"{n:<20}{s:>7.2f}%{r:>8.0f}")
    print("\nReferences: GAM 1763 ; MLpol agg (direct sweep) 1940 ; best direct conv 1983 MW")


if __name__ == "__main__":
    main()
