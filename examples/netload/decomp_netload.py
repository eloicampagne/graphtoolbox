"""Decomposed vs direct net-load with one GNN per component (Load/Wind/Solar).

Trains one convolution (LEConv, the fastest strong operator) at the fair config
(hidden=64, 1 layer) for each of the three physical components, then combines
NetLoad = Load - Wind - Solar at the national level, and compares against the
direct single-GNN net-load model and the operational GAM. Mirrors the GAM
decomposition of the INFORMS benchmark (dev.ipynb), on the GNN side.
"""
import os
import gc
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import LEConv

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer
import netload_pipeline as P

HIDDEN, LAYERS, EPOCHS, ADJ = 64, 1, 150, "dtw"
CONV, CKW = LEConv, {}
OUT = os.path.join(P.HERE, "results_decomp_netload")
os.makedirs(OUT, exist_ok=True)

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
    "Load": {"features": feat_load + [f"Load_l{t}" for t in range(1, 49)]
             + [f"temperature_l{t}" for t in range(-47, 0)], "target": "Load"},
    "Wind_power": {"features": feat_wind + [f"Wind_power_l{t}" for t in range(1, 49)]
                   + [f"temperature_l{t}" for t in range(-47, 0)], "target": "Wind_power"},
    "Solar_power": {"features": feat_solar + [f"Solar_power_l{t}" for t in range(1, 49)]
                    + [f"temperature_l{t}" for t in range(-47, 0)], "target": "Solar_power"},
}


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main():
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=data_kwargs, folder_config=str(P.HERE))
    comp_pred, comp_tgt = {}, {}
    for name, spec in COMPONENTS.items():
        dk = {"batch_size": 32, "adj_matrix": ADJ, "features_base": spec["features"], "target_base": spec["target"]}
        common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
        tr = GraphDataset(data=data, period="train", **common)
        va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                          scalers_target=tr.scalers_target, **common)
        te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                          scalers_target=tr.scalers_target, **common)
        for ds in (tr, va, te):
            ds._set_adj_matrix(adj_matrix=ADJ)
        ppath = os.path.join(OUT, f"{name}.pt")
        if os.path.exists(ppath):
            pred = torch.load(ppath, weights_only=False)
            tgt = torch.load(os.path.join(OUT, f"{name}_target.pt"), weights_only=False)
        else:
            print(f"\nTraining {name} (LEConv h{HIDDEN} l{LAYERS})...")
            seed_everything(42); gc.collect()
            model = myGNN(in_channels=tr.num_node_features, num_layers=LAYERS, hidden_channels=HIDDEN,
                          out_channels=P.OUT_CHANNELS, conv_class=CONV, conv_kwargs=CKW)
            trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                              batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                              reconcile=True, top_level_model="xgb", lam_reg=0)
            res = trainer.train(force_training=True, save=False, patience=40)
            pred, tgt = res[0], res[1]
            torch.save(pred, ppath); torch.save(tgt, os.path.join(OUT, f"{name}_target.pt"))
        comp_pred[name] = pred.sum(dim=0).cpu().numpy()
        comp_tgt[name] = tgt.sum(dim=0).cpu().numpy()
        print(f"  {name:<12} national RMSE={rmse(comp_pred[name], comp_tgt[name]):.0f} MW")

    net_pred = comp_pred["Load"] - comp_pred["Wind_power"] - comp_pred["Solar_power"]
    net_tgt = comp_tgt["Load"] - comp_tgt["Wind_power"] - comp_tgt["Solar_power"]
    # Cross-check against the direct NetLoad target.
    direct_tgt = torch.load(os.path.join(P.HERE, "results_all_convolutions_netload/target.pt"),
                            weights_only=False).numpy()
    print(f"\n[check] decomposed truth vs direct NetLoad truth: "
          f"RMSE={rmse(net_tgt[:len(direct_tgt)], direct_tgt):.1f} MW (~0 expected)")
    y = direct_tgt
    net_pred = net_pred[:len(y)]
    print("\n=== National net-load, 2019 test ===")
    print(f"{'Decomposed (LEConv per component)':<34} sMAPE={smape(net_pred, y):.2f}%  RMSE={rmse(net_pred, y):.0f} MW")
    print(f"{'Direct (LEConv, fair h64/l1)':<34} sMAPE=3.34%  RMSE=2081 MW")
    print(f"{'Direct best conv (RGATConv)':<34} sMAPE=3.19%  RMSE=1983 MW")
    print(f"{'MLpol aggregation (GNN sweep)':<34} sMAPE=3.09%  RMSE=1940 MW")
    print(f"{'GAM (operational, per tod)':<34} sMAPE=2.92%  RMSE=1763 MW")


if __name__ == "__main__":
    main()
