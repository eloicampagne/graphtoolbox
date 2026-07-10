"""Forward/backward tests for the additive and gated multi-branch models."""
import torch
from torch_geometric.nn.conv import GCNConv, GATConv

from graphtoolbox.models import AdditiveGraphModel, GatedMultiConvGNN


def _graph(n=10, f=30, e=36):
    return torch.randn(n, f), torch.randint(0, n, (2, e)), torch.rand(e)


def test_additive_forward_and_group_outputs():
    x, ei, ew = _graph(f=30)
    groups = {"a": 10, "b": 12, "c": 8}          # sums to 30
    model = AdditiveGraphModel(feature_group_dims=groups, num_layers=2,
                               hidden_channels=16, out_channels=6,
                               conv_class=GCNConv, conv_kwargs={})
    out = model(x, ei, edge_weight=ew)
    assert out.shape == (10, 6)
    y_hat, group_out = model(x, ei, edge_weight=ew, return_group_outputs=True)
    assert set(group_out) == set(groups)
    assert all(v.shape == (10, 6) for v in group_out.values())
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_gated_multiconv_mixes_branches():
    x, ei, ew = _graph(f=30)
    model = GatedMultiConvGNN(in_channels=30, hidden_channels=16, out_channels=6,
                              conv_classes=[GATConv, GCNConv], num_layers=2,
                              conv_kwargs_list=[{"heads": 4}, {}])
    out = model(x, ei, edge_weight=ew)
    assert out.shape == (10, 6)
    y_hat, gate = model(x, ei, edge_weight=ew, return_attention=True)
    # per-node convex gate over the two branches
    assert gate.shape == (10, 2)
    assert torch.allclose(gate.sum(dim=-1), torch.ones(10), atol=1e-5)
