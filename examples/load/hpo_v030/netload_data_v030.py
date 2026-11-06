"""Net-load datasets of the v0.3.0 study, with or without the weather trajectories of the forecast day.

direct_datasets(leads) builds the direct net-load train/val/test datasets of
examples/netload/netload_pipeline.py. With leads=True, every model also receives the
wind and cloud-cover trajectories of the forecast day (wind,
wind_by_wind_power_weights, nebulosity, nebulosity_by_solar_power_weights at the 48
half-hours), as the temperature already is. Call after os.chdir into examples/netload
with that folder on sys.path.
"""
import netload_pipeline as P
from graphtoolbox.data.dataset import DataClass, GraphDataset

LEAD_VARS = {"wind": "wind", "wind_by_wind_power_weights": "wind", "nebulosity": "cloud",
             "nebulosity_by_solar_power_weights": "cloud"}


def lead_cols(kinds):
    """Names of the lead columns of the weather variables of the given kinds."""
    return [f"{v}_l{t}" for v, k in LEAD_VARS.items() if k in kinds for t in range(-47, 0)]


def direct_datasets(leads):
    if not leads:
        return P.build_datasets()[:3]
    data_kwargs = {"node_var": "Region",
                   "features_to_lag": {"NetLoad": (1, 48), "temperature": (-47, -1),
                                       **{v: (-47, -1) for v in LEAD_VARS}},
                   "dummies": ["tod", "day_type_week"],
                   "day_inf_train": "2014-01-01", "day_sup_train": "2018-01-01",
                   "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
                   "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
                   "computed_features": P.COMPUTED_FEATURES}
    dk = {"batch_size": 32, "adj_matrix": "dtw", "target_base": "NetLoad",
          "features_base": (P.FEATURES + [f"NetLoad_l{t}" for t in range(1, 49)]
                            + [f"temperature_l{t}" for t in range(-47, 0)] + lead_cols({"wind", "cloud"}))}
    data = DataClass(path_train=str(P.HERE / "train.csv"), path_test=str(P.HERE / "test.csv"),
                     data_kwargs=data_kwargs, folder_config=str(P.HERE))
    common = dict(graph_folder=P.GRAPH_FOLDER, dataset_kwargs=dk, out_channels=P.OUT_CHANNELS)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    return tr, va, te
