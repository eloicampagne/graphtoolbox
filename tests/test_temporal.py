"""Tests for TemporalGNN / ConvAdapterTemporal using fake PyG-Temporal cells.

The real recurrent cells live in the optional torch_geometric_temporal package, so
these tests fake the two calling conventions (step-recurrent and windowed) to
exercise the adapter and the lag/static split without that dependency.
"""
import re

import torch
import torch.nn as nn

from graphtoolbox.models import TemporalGNN, lag_sequence_indices


class _FakeStepCell(nn.Module):
    """Mimics GConvGRU/DCRNN/TGCN: forward(x_t, edge_index, edge_weight, H)."""
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, H=None):
        if H is None:
            H = x.new_zeros(x.size(0), self.out_channels)
        return torch.tanh(self.lin(torch.cat([x, H], dim=1)))


class _FakeWindowCell(nn.Module):
    """Mimics A3TGCN: forward(x_window[N, C, periods], edge_index, edge_weight)."""
    def __init__(self, in_channels, out_channels, periods):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels * periods, out_channels)

    def forward(self, x, edge_index, edge_weight=None, H=None):
        return torch.tanh(self.lin(x.reshape(x.size(0), -1)))


def _features():
    # 5 static features, then 8 ordered load lags
    return ["temp", "nebu", "wind", "cal1", "cal2"] + [f"load_l{t}" for t in range(1, 9)]


def _graph(n=12, e=40):
    feats = _features()
    x = torch.randn(n, len(feats))
    return feats, x, torch.randint(0, n, (2, e)), torch.rand(e)


def test_lag_sequence_indices_orders_oldest_first():
    feats = _features()
    seq, static = lag_sequence_indices(feats, target="load")
    assert len(seq) == 8 and len(static) == 5
    assert feats[seq[0]] == "load_l8" and feats[seq[-1]] == "load_l1"


def test_temporalgnn_step_cell_forward_backward():
    feats, x, ei, ew = _graph()
    model = TemporalGNN.from_features(feats, cell_class=_FakeStepCell,
                                      hidden_channels=16, out_channels=6,
                                      cell_kwargs={"K": 2})
    assert model.adapter.windowed is False
    out = model(x, ei, ew)
    assert out.shape == (12, 6)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_temporalgnn_windowed_cell_sets_periods():
    feats, x, ei, ew = _graph()
    model = TemporalGNN.from_features(feats, cell_class=_FakeWindowCell,
                                      hidden_channels=16, out_channels=6)
    assert model.adapter.windowed is True
    out = model(x, ei, ew)
    assert out.shape == (12, 6)
