"""Smoke tests for myGNN and the convolution adapter on a synthetic graph."""
import pytest
import torch
from torch_geometric.nn.conv import GCNConv, GATConv, LEConv, ChebConv, RGATConv, RGCNConv

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


def test_edge_type_reaches_relational_convolutions():
    x, edge_index, _ = _graph()
    edge_type = torch.randint(0, 3, (edge_index.size(1),))
    for cls in (RGCNConv, RGATConv):
        torch.manual_seed(0)
        typed = myGNN(20, 1, 32, 1, conv_class=cls, conv_kwargs={"num_relations": 3}).eval()
        torch.manual_seed(0)
        single = myGNN(20, 1, 32, 1, conv_class=cls).eval()
        assert typed.layers[0].conv.conv.num_relations == 3, cls.__name__
        with torch.no_grad():
            out = typed(x, edge_index, edge_type=edge_type)
            shuffled = typed(x, edge_index, edge_type=edge_type.roll(1))
        assert not torch.allclose(out, shuffled), cls.__name__
        assert single.layers[0].conv.conv.num_relations == 1
        with pytest.raises(ValueError):
            single(x, edge_index, edge_type=edge_type)


def test_edge_type_does_not_change_other_convolutions():
    x, edge_index, _ = _graph()
    model = myGNN(20, 2, 32, 1, conv_class=GCNConv).eval()
    with torch.no_grad():
        assert torch.equal(model(x, edge_index),
                           model(x, edge_index, edge_type=torch.randint(0, 3, (edge_index.size(1),))))


def test_adapter_runs_diverse_operators():
    x, edge_index, edge_weight = _graph()
    cases = [(GCNConv, {}), (LEConv, {}), (ChebConv, {"K": 3}), (GATConv, {"heads": 2})]
    for cls, kw in cases:
        model = myGNN(20, 2, 32, 8, conv_class=cls, conv_kwargs=kw, heads=kw.get("heads", 1))
        out = model(x, edge_index, edge_weight=edge_weight)
        assert out.shape == (12, 8), cls.__name__
