import numpy as np
import torch

from graphtoolbox.interpretability import additive_ale
from graphtoolbox.models import AdditiveGraphModel


def _model(seed=0):
    torch.manual_seed(seed)
    return AdditiveGraphModel(feature_group_dims={"a": 1, "b": 2}, num_layers=1,
                              hidden_channels=8, out_channels=3).eval()


def test_ale_matches_shape_function_without_edges():
    # Without edges each node is independent and group "a" reads one feature, so the
    # uncentered ALE at each edge equals f(z_k) - f(z_0), with f the group contribution.
    model = _model()
    D, N = 200, 2
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.uniform(size=(D, N, 3)), dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    res = additive_ale(model, x, x[..., 0].numpy(), 0, edge_index, n_bins=10, batch_size=64)
    z = torch.as_tensor(res["edges"], dtype=torch.float32)
    probe = torch.zeros(len(z), 3)
    probe[:, 0] = z
    with torch.no_grad():
        f = model.group_output("a", probe, edge_index).numpy() / model.num_groups   # [K+1, out]
    expected = (f - f[0]).T
    got = res["ale"] - res["ale"][:, :1]
    assert res["group"] == "a"
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert res["importance"] > 0


def test_binary_feature_keeps_its_effect():
    model = _model(2)
    D, N = 120, 2
    x = torch.rand(D, N, 3)
    x[..., 0] = (x[..., 0] > 0.7).float()
    res = additive_ale(model, x, x[..., 0].numpy(), 0, torch.empty((2, 0), dtype=torch.long))
    p = float(x[..., 0].mean())
    jump = res["ale"][:, 1] - res["ale"][:, 0]
    # Centered over observations: p * ALE(1) + (1 - p) * ALE(0) = 0.
    np.testing.assert_allclose(p * res["ale"][:, 1] + (1 - p) * res["ale"][:, 0], 0.0, atol=1e-5)
    assert np.abs(jump).max() > 0 and res["importance"] > 0


def test_feature_is_attributed_to_its_group():
    model = _model(1)
    D, N = 50, 3
    x = torch.rand(D, N, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    res = additive_ale(model, x, x[..., 2].numpy(), 2, edge_index, n_bins=5)
    assert res["group"] == "b" and res["importance"] > 0
