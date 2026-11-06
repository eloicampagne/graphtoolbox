"""Attention matched to edges, occlusion influence and neighbourhood deviations.

GATConv removes the self-loops of its input and appends new ones after every
other edge of the batch, so batched attention must be matched to edges through
the edge index the convolution returns. These tests check that the weights of a
batch equal those of each graph run alone, that they sum to one over the edges
entering every node, and that caches written by the toolbox load back unchanged.
"""
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import GATConv, GATv2Conv

from graphtoolbox.interpretability import (
    attention_by_snapshot, dump_attention_batches, load_attention_batches,
    neighbourhood_deviation, occlusion_influence, sender_variance_share,
)
from graphtoolbox.models import myGNN

N, F, OUT = 5, 4, 3


def _graphs(count=4, seed=0):
    """Snapshots on one path graph 0-1-2-3-4 whose input carries self-loops."""
    rng = np.random.default_rng(seed)
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    edge_index = torch.tensor(pairs, dtype=torch.long).T
    return [Data(x=torch.tensor(rng.standard_normal((N, F)), dtype=torch.float32), edge_index=edge_index)
            for _ in range(count)]


def _model(conv=GATConv, layers=2):
    torch.manual_seed(0)
    return myGNN(in_channels=F, num_layers=layers, hidden_channels=8, out_channels=OUT, conv_class=conv, heads=1).eval()


@pytest.mark.parametrize("conv", [GATConv, GATv2Conv])
def test_batched_attention_matches_each_graph(conv):
    model, graphs = _model(conv), _graphs()
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        _, att = model(batch.x, batch.edge_index, return_attention=True, num_graphs=len(graphs))
        for layer, (per_graph, edges) in enumerate(zip(att["per_graph"], att["edge_index"])):
            assert per_graph.shape[:2] == (len(graphs), edges.shape[1])
            for b, g in enumerate(graphs):
                _, single = model(g.x, g.edge_index, return_attention=True)
                assert torch.equal(single["edge_index"][layer], edges)
                assert torch.allclose(single["per_graph"][layer][0], per_graph[b], atol=1e-6)
                sums = torch.zeros(N).index_add_(0, edges[1], per_graph[b, :, 0])
                assert torch.allclose(sums, torch.ones(N), atol=1e-5)
            assert torch.allclose(att["mean"][layer], per_graph.mean(dim=0))


def test_batch_size_keyword_is_accepted_for_backward_compatibility():
    model, graphs = _model(), _graphs(3)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        _, att = model(batch.x, batch.edge_index, return_attention=True, batch_size=3)
    assert att["per_graph"][0].shape[0] == 3


def test_dump_and_load_round_trip(tmp_path):
    model, graphs = _model(), _graphs(6)
    saved = dump_attention_batches(model, graphs, str(tmp_path), batch_size=3)
    assert saved == 2
    masks, edge_index = load_attention_batches(str(tmp_path))
    alpha, edges = attention_by_snapshot(model, graphs)
    assert np.array_equal(edge_index.numpy(), edges)
    assert np.allclose(masks.numpy(), alpha.mean(axis=1), atol=1e-6)


def test_legacy_cache_with_several_graphs_is_refused(tmp_path):
    edge_index = torch.zeros((2, 4), dtype=torch.long)
    torch.save({"attention_weights": [torch.rand(8, 1)], "edge_idx": edge_index, "num_graphs": 2},
               os.path.join(tmp_path, "num_batch0.pt"))
    with pytest.raises(ValueError, match="legacy cache"):
        load_attention_batches(str(tmp_path))


def test_occlusion_influence_follows_the_receptive_field():
    model, graphs = _model(layers=1), _graphs(3)
    influence = occlusion_influence(model, graphs)
    assert influence.shape == (3, N, N)
    # one layer: node 0 only hears node 1 (and itself)
    assert np.all(influence[:, 0, 2:] == 0)
    assert np.all(influence[:, 0, :2] > 0)
    scaled = occlusion_influence(model, graphs, node_scale=np.full(N, 10.0))
    assert np.allclose(scaled, 10 * influence)


def test_neighbourhood_deviation_and_sender_share():
    edge_index = np.array([[1, 2, 0, 2, 0, 1, 0], [0, 0, 1, 1, 2, 2, 0]])
    uniform = neighbourhood_deviation(np.ones(7), edge_index)
    assert np.allclose(uniform[:6], 0) and np.isnan(uniform[6])
    by_sender = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 5.0])     # weight 2 for sender 1, else 1
    dev = neighbourhood_deviation(by_sender, edge_index)
    for i in range(3):
        e = (edge_index[1] == i) & (edge_index[0] != i)
        assert np.isclose(((dev[e] + 1) / e.sum()).sum(), 1)
    sender_only = np.array([0.3, -0.2, 0.1, -0.2, 0.1, 0.3, np.nan])
    assert np.isclose(sender_variance_share(sender_only, edge_index), 1.0)
