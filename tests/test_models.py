"""Smoke tests for myGNN and the convolution adapter on a synthetic graph."""
import torch
from torch_geometric.nn.conv import GCNConv, GATConv, LEConv, ChebConv

from graphtoolbox.models import myGNN


def _graph(n=12, f=20, e=40):
    x = torch.randn(n, f)
    edge_index = torch.randint(0, n, (2, e))
    edge_weight = torch.rand(e)
    return x, edge_index, edge_weight


def test_mygnn_forward_backward():
    x, edge_index, edge_weight = _graph()
    model = myGNN(20, 2, 32, 8, conv_class=GCNConv)
    out = model(x, edge_index, edge_weight=edge_weight)
    assert out.shape == (12, 8)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_heads_split_keeps_width_fixed():
    # Multi-head convs split the latent width across heads (attention-is-all-you-need
    # convention), so the parameter count does not grow with the head count.
    p1 = sum(p.numel() for p in myGNN(20, 2, 32, 8, conv_class=GATConv, heads=1).parameters())
    p4 = sum(p.numel() for p in myGNN(20, 2, 32, 8, conv_class=GATConv, heads=4).parameters())
    assert p1 == p4


def test_adapter_runs_diverse_operators():
    x, edge_index, edge_weight = _graph()
    cases = [(GCNConv, {}), (LEConv, {}), (ChebConv, {"K": 3}), (GATConv, {"heads": 2})]
    for cls, kw in cases:
        model = myGNN(20, 2, 32, 8, conv_class=cls, conv_kwargs=kw, heads=kw.get("heads", 1))
        out = model(x, edge_index, edge_weight=edge_weight)
        assert out.shape == (12, 8), cls.__name__
