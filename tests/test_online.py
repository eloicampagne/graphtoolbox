"""Forecasting on a stream: Trainer.predict, OnlineForecaster and the replay engine.

The stream tests send the test rows of a synthetic panel block by block, first without
their target, then with it, and compare with the GraphDataset built by DataClass from the
same files and with RollingTrainer(update="replay") on that dataset.
"""
import copy
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GCNConv

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import OnlineForecaster, RollingTrainer, Trainer, set_device
from graphtoolbox.training.online import NodeScaler

NODES = ["A", "B", "C"]
OC = 2


def _panel(dates, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for j, node in enumerate(NODES):
        temp = 15 + 5 * np.sin(np.arange(len(dates)) * 0.2) + rng.standard_normal(len(dates))
        load = 50 + 5 * j + 0.8 * temp + rng.standard_normal(len(dates))
        for i, d in enumerate(dates):
            rows.append({"date": str(d.date()), "node": node, "dow": int(d.dayofweek),
                         "temp": float(temp[i]), "load": float(load[i])})
    return pd.DataFrame(rows)


def _heat(df):
    return (18 - df["temp"]).clip(lower=0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_device("cpu")
    seed_everything(0)
    train_raw = _panel(pd.date_range("2020-01-01", periods=60, freq="D"), 0)
    test_raw = _panel(pd.date_range("2020-03-05", periods=15, freq="D"), 1)
    train_raw.to_csv("train.csv", index=False)
    test_raw.to_csv("test.csv", index=False)
    data_kwargs = {
        "node_var": "node", "dummies": ["dow", "nodeInt"], "features_to_lag": {"load": (1, 2)},
        "computed_features": {"heat": _heat},
        "day_inf_train": "2020-01-01", "day_sup_train": "2020-02-10",
        "day_inf_val": "2020-02-11", "day_sup_val": "2020-02-29",
        "day_inf_test": "2020-03-05", "day_sup_test": "2020-03-19",
    }
    dataset_kwargs = {"adj_matrix": "eye", "target_base": "load",
                      "features_base": ["temp", "heat", "load_l1", "load_l2", "dow", "nodeInt"]}
    data = DataClass(path_train="train.csv", path_test="test.csv",
                     data_kwargs=data_kwargs, folder_config=".")
    common = dict(graph_folder="./no_graphs", dataset_kwargs=dataset_kwargs, out_channels=OC)
    train = GraphDataset(data=data, period="train", **common)
    sc = dict(scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)
    val = GraphDataset(data=data, period="val", **sc)
    test = GraphDataset(data=data, period="test", **sc)
    seed_everything(0)
    model = myGNN(in_channels=train.num_node_features, num_layers=1, hidden_channels=8,
                  out_channels=OC, conv_class=GCNConv)
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=8, model_kwargs={"lr": 1e-2, "num_epochs": 2}, reconcile=False)
    trainer.train(force_training=True, patience=2)
    stream = test_raw[test_raw["date"] < "2020-03-19"]
    return dict(train=train, val=val, test=test, trainer=trainer, train_raw=train_raw,
                stream=stream, data=data, common=common)


def _blocks(stream):
    dates = sorted(stream["date"].unique())
    for k in range(0, len(dates), OC):
        yield stream[stream["date"].isin(dates[k:k + OC])]


def test_node_scaler_matches_sklearn(setup):
    train = setup["train"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(NODES), 11, train.num_node_features)).astype(np.float32)
    fx = NodeScaler(train.scalers_feat, train.nodes)
    assert fx.fast
    ref = np.stack([train.scalers_feat[n].transform(X[i]) for i, n in enumerate(train.nodes)])
    np.testing.assert_array_equal(fx.transform(X), ref.astype(np.float32))
    fy = NodeScaler(train.scalers_target, train.nodes)
    Z = rng.normal(size=(len(NODES), 13)).astype(np.float32)
    ref = np.stack([train.scalers_target[n].inverse_transform(Z[i].reshape(-1, 1)).reshape(-1)
                    for i, n in enumerate(train.nodes)])
    np.testing.assert_array_equal(fy.inverse_transform(Z), ref.astype(np.float32))


def test_predict_matches_train_and_ignores_missing_targets(setup):
    trainer, test = setup["trainer"], setup["test"]
    _, preds, _ = trainer._predict(trainer.test_loader, split="test")
    np.testing.assert_allclose(trainer.predict(test, batch_size=3).numpy(), preds.numpy(),
                               rtol=1e-6, atol=1e-5)
    data = copy.copy(setup["data"])
    data.df_test = data.df_test.assign(load=np.nan)
    future = GraphDataset(data=data, period="test", scalers_feat=setup["train"].scalers_feat,
                          scalers_target=setup["train"].scalers_target, **setup["common"])
    out = trainer.predict(future)
    assert out.shape == preds.shape and bool(torch.isfinite(out).all())
    assert bool((out.abs() > 1).all())               # not zeroed by the target mask


def _forecaster(setup, **kw):
    return OnlineForecaster(copy.deepcopy(setup["trainer"].model), setup["train"],
                            history=setup["train_raw"], batch_size=8, **kw)


def test_stream_samples_match_graphdataset(setup):
    fc = _forecaster(setup, replay_steps=0)
    got = []
    for rows in _blocks(setup["stream"]):
        fc.forecast(rows.drop(columns="load"))
        dates = fc._ingest(rows)
        got += fc._samples(fc.prepare(dates), dates)
    ref = setup["test"].pyg_data
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        for attr in ("x", "y_scaled", "mask_y", "y", "mask_X"):
            assert torch.equal(getattr(g, attr).float(), getattr(r, attr).float()), attr


def test_stream_matches_rolling_replay(setup):
    seed = 3
    roller = RollingTrainer(
        dataset_train_0=setup["train"], dataset_val_0=setup["val"], dataset_test_full=setup["test"],
        model_class=myGNN, model_kwargs={}, window_size=OC, step_size=OC, batch_size=8,
        reconcile=False, trainer_kwargs={"seed": seed}, update="replay",
        initial_model=copy.deepcopy(setup["trainer"].model), replay_steps=3, replay_lr=1e-2,
        replay_recent=2)
    ref = roller.run()
    fc = _forecaster(setup, replay_steps=3, replay_lr=1e-2, replay_recent=2, seed=seed)
    for r, rows in zip(ref, _blocks(setup["stream"])):
        pred = fc.forecast(rows.drop(columns="load"))
        fc.update(rows[["date", "node", "load"]])
        wide = pred.pivot(index="node", columns="date", values="forecast").loc[NODES]
        np.testing.assert_array_equal(wide.to_numpy(np.float32), r["preds"].numpy())


def test_stream_rejects_bad_inputs(setup):
    fc = _forecaster(setup, replay_steps=0)
    first = next(_blocks(setup["stream"]))
    with pytest.raises(ValueError, match="whole blocks"):
        fc.forecast(first[first["date"] == first["date"].min()])
    with pytest.raises(ValueError, match="absent from training"):
        fc.forecast(first.assign(node="Z"))
    fc.forecast(first.drop(columns="load"))
    with pytest.warns(UserWarning, match="no new observed target"):
        fc.update(first.assign(load=np.nan))
    late = setup["stream"][setup["stream"]["date"] > "2020-03-14"][["date", "node", "load"]]
    with pytest.raises(ValueError, match="not in the buffer"):
        fc.update(late)
    with pytest.raises(ValueError, match="covariates"):          # missing covariate, new dates
        fc.forecast(late.drop(columns="load").assign(dow=1))
    with pytest.raises(ValueError, match="target column"):
        fc.update(first.drop(columns="load"))


def _pool_x(fc, start):
    return fc.engine._X[start:fc.engine.size].cpu()


def test_node_int_dummy_is_encoded(setup):
    assert any(c.startswith("nodeInt_") for c in setup["train"].features)
    fc = _forecaster(setup, replay_steps=0)
    rows = next(_blocks(setup["stream"]))
    prepared = fc.prepare(fc._ingest(rows))
    for i, n in enumerate(setup["train"].nodes):
        sub = prepared[prepared["node"] == n]
        for c in fc._dummy_cols["nodeInt"]:
            assert bool((sub[c] == float(float(c.split("_")[1]) == i)).all())


@pytest.mark.parametrize("blocks_per_call", [1, 3, 7])
def test_late_updates_match_graphdataset(setup, blocks_per_call):
    """Calls of several blocks, targets only in update(): the pool gets the reference samples."""
    fc = _forecaster(setup, replay_steps=0)
    start = fc.engine.size
    dates = sorted(setup["stream"]["date"].unique())
    step = blocks_per_call * OC
    for k in range(0, len(dates), step):
        rows = setup["stream"][setup["stream"]["date"].isin(dates[k:k + step])]
        fc.forecast(rows.drop(columns="load"))
        fc.update(rows[["date", "node", "load"]])
    ref = torch.stack([d.x for d in setup["test"].pyg_data])
    assert torch.equal(_pool_x(fc, start), ref)


def test_nan_in_dummy_source_keeps_other_rows(setup):
    fc = _forecaster(setup, replay_steps=0)
    start = fc.engine.size
    stream = setup["stream"].copy()
    dates = sorted(stream["date"].unique())
    stream["dow"] = stream["dow"].astype(float)
    stream.loc[(stream["date"] == dates[9]) & (stream["node"] == "A"), "dow"] = np.nan  # not a block start
    for rows in _blocks(stream):
        fc.forecast(rows.drop(columns="load"))
        fc.update(rows[["date", "node", "load"]])
    ref = torch.stack([d.x for d in setup["test"].pyg_data])
    assert torch.equal(_pool_x(fc, start), ref)


def test_missing_node_does_not_shift_lags(setup):
    fc = _forecaster(setup, replay_steps=0)
    blocks = list(_blocks(setup["stream"]))
    got = []
    for k, rows in enumerate(blocks[:4]):
        if k == 2:
            rows = rows[rows["node"] != "B"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fc.forecast(rows.drop(columns="load"))
            dates = fc._ingest(rows)
            got.append(fc._samples(fc.prepare(dates), dates)[0])
        if k in (2, 3):
            assert any("zeroed" in str(x.message) for x in w)
    ref = setup["test"].pyg_data
    b = list(setup["train"].nodes).index("B")
    for k in (2, 3):                                   # B is empty, its lags reach the gap
        assert bool((got[k].x[b] == 0).all())
        others = [i for i in range(len(NODES)) if i != b]
        assert torch.equal(got[k].x[others], ref[k].x[others])


def test_update_twice_adds_the_block_once(setup):
    fc = _forecaster(setup, replay_steps=0)
    rows = next(_blocks(setup["stream"]))
    fc.forecast(rows.drop(columns="load"))
    fc.update(rows)
    size = fc.engine.size
    with pytest.warns(UserWarning, match="already added"):
        fc.update(rows)
    assert fc.engine.size == size


def test_misaligned_block_warns(setup):
    fc = _forecaster(setup, replay_steps=0)
    dates = sorted(setup["stream"]["date"].unique())
    rows = setup["stream"][setup["stream"]["date"].isin(dates[1:1 + OC])]
    with pytest.warns(UserWarning, match="shifted"):
        fc.forecast(rows.drop(columns="load"))


def test_forecaster_keeps_the_model_device(setup):
    model = copy.deepcopy(setup["trainer"].model)
    fc = OnlineForecaster(model, setup["train"], history=setup["train_raw"], replay_steps=0)
    assert fc.engine.device == next(model.parameters()).device
    if torch.backends.mps.is_available():
        model = copy.deepcopy(setup["trainer"].model).to("mps")
        fc = OnlineForecaster(model, setup["train"], history=setup["train_raw"], replay_steps=1,
                              batch_size=8, replay_recent=2)
        assert next(model.parameters()).device.type == "mps"
        rows = next(_blocks(setup["stream"]))
        out = fc.forecast(rows.drop(columns="load"))
        fc.update(rows)
        assert bool(np.isfinite(out["forecast"]).all())


def test_forecaster_survives_save_and_load(setup, tmp_path):
    fc = _forecaster(setup, replay_steps=2, replay_recent=2, seed=0)
    blocks = list(_blocks(setup["stream"]))
    fc.forecast(blocks[0].drop(columns="load"))
    fc.update(blocks[0])
    torch.save(fc, tmp_path / "fc.pt")
    back = torch.load(tmp_path / "fc.pt", map_location="cpu", weights_only=False).to("cpu")
    for f in (fc, back):
        out = f.forecast(blocks[1].drop(columns="load"))
        f.update(blocks[1])
        f.out = f.forecast(blocks[2].drop(columns="load"))["forecast"].to_numpy()
    np.testing.assert_array_equal(fc.out, back.out)
