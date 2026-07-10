"""Direct unit tests for the forecasting metrics, on numpy and torch inputs."""
import numpy as np
import pytest
import torch

from graphtoolbox.training import MAE, NMAE, MAPE, RMSE, BIAS


@pytest.mark.parametrize("wrap", [np.asarray, torch.tensor])
def test_metric_values(wrap):
    preds = wrap([1.0, 2.0, 3.0])
    targets = wrap([1.0, 1.0, 1.0])
    # preds-targets = [0, 1, 2]; BIAS uses the target-minus-pred convention.
    assert float(MAE(preds, targets)) == pytest.approx(1.0)
    assert float(RMSE(preds, targets)) == pytest.approx(np.sqrt(5.0 / 3.0))
    assert float(BIAS(preds, targets)) == pytest.approx(-1.0)
    assert float(NMAE(preds, targets)) == pytest.approx(1.0)     # mean|e| / mean|t| = 1/1
    assert float(MAPE(preds, targets)) == pytest.approx(1.0)     # |(t-p)/t| mean = 1


def test_perfect_forecast_is_zero_error():
    y = np.array([3.0, -2.0, 5.0])
    assert float(MAE(y, y)) == pytest.approx(0.0)
    assert float(RMSE(y, y)) == pytest.approx(0.0)
    assert float(BIAS(y, y)) == pytest.approx(0.0)


def test_numpy_torch_agree():
    rng = np.random.default_rng(0)
    p, t = rng.standard_normal(50), rng.standard_normal(50)
    for metric in (MAE, RMSE, MAPE, NMAE, BIAS):
        a = float(metric(p, t))
        b = float(metric(torch.tensor(p), torch.tensor(t)))
        assert a == pytest.approx(b, rel=1e-6)
