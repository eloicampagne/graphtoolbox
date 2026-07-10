"""Unit tests for the pure utility helpers."""
import os

import numpy as np
import pytest
import torch

from graphtoolbox.utils import (
    parser_config, extract_parameters, get_geodesic_distance,
    get_exponential_similarity, build_adjacency_matrix, clean_dir,
)


def test_parser_config_types():
    d = parser_config('{"model_name": "GCN", "num_epochs": "100", "lr": "0.001"}')
    assert d == {"model_name": "GCN", "num_epochs": 100, "lr": 0.001}


def test_extract_parameters_match_and_miss():
    d = extract_parameters("batch16_hidden256_layers2_epochs300")
    assert d == {"batch_size": 16, "hidden_channels": 256, "num_layers": 2}
    assert extract_parameters("not_a_checkpoint_name") is None


def test_geodesic_distance():
    paris = (2.35, 48.85)
    london = (-0.13, 51.51)
    assert get_geodesic_distance(paris, paris) == pytest.approx(0.0, abs=1e-6)
    d = get_geodesic_distance(paris, london)
    assert 300 < d < 400          # ~344 km


def test_exponential_similarity_kernel():
    dist = np.array([0.0, 1.0, 5.0])
    sim = get_exponential_similarity(dist, bandwidth=2.0, threshold=0.1)
    assert sim[0] == pytest.approx(1.0)            # zero distance -> similarity 1
    assert sim[2] == 0.0                            # far pair thresholded to 0
    assert np.all(np.diff(sim[:2]) <= 0)            # decreasing with distance


def test_build_adjacency_matrix():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([0.5, 1.0, 2.0])
    A = build_adjacency_matrix(edge_index, edge_weight)
    assert A.shape == (3, 3)
    assert A[0, 1] == pytest.approx(0.5)
    assert A[2, 0] == pytest.approx(2.0)
    assert A[1, 0] == 0.0                            # directed, no reverse edge


def test_clean_dir(tmp_path):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("x")
    clean_dir(str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_spectral_helpers():
    from graphtoolbox.utils import (
        normalized_laplacian, spectral_embedding, cosine_similarity_matrix,
    )
    # simple 4-node ring adjacency
    A = torch.tensor([[0., 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    L = normalized_laplacian(A)
    assert L.shape == (4, 4)
    assert torch.allclose(L, L.T, atol=1e-5)          # symmetric

    emb = spectral_embedding(L, k=2)
    assert emb.shape[0] == 4 and emb.shape[1] == 2

    X = torch.randn(5, 8)
    S = cosine_similarity_matrix(X)
    assert S.shape == (5, 5)
    assert torch.allclose(torch.diagonal(S), torch.ones(5), atol=1e-5)   # self-similarity 1
    assert float(S.max()) <= 1.0 + 1e-5 and float(S.min()) >= -1.0 - 1e-5
