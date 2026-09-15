"""Compare matched MLP and GNN latent representations on the French panel.

The identity-SAGE model is the shared MLP of the thesis: with identity adjacency,
every node is transformed by the same weights and reads no other node.  The
spatial-SAGE model has the same encoder, latent width, depth and readout, and only
adds neighbourhood aggregation.  We extract the input of the final linear
readout for every (test day, region) pair.

The matched activations are saved for the projection script in the thesis.  The
two t-SNE maps are then fitted separately because independently trained latent
coordinates are not aligned; quantitative comparisons use the original spaces.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from graphtoolbox.models import myGNN
from torch_geometric.nn.conv import SAGEConv
import run_gnnexplainer_load as data_source

HIDDEN = 64
LAYERS = 2
EPOCHS = 300
BATCH = 32
REGIONS = (
    "ARA", "BFC", "BRE", "CVL", "GES", "HDF",
    "IDF", "NAQ", "NOR", "OCC", "PAC", "PDL",
)


def checkpoint(adjacency: str) -> str:
    folder = os.path.join(
        HERE, "checkpoints", f"SAGEConv_{adjacency}",
        f"batch{BATCH}_hidden{HIDDEN}_layers{LAYERS}_epochs{EPOCHS}",
    )
    files = sorted(f for f in os.listdir(folder) if f.endswith(".params"))
    if not files:
        raise FileNotFoundError(f"No checkpoint in {folder}")
    return os.path.join(folder, files[-1])


def dataset(adjacency: str):
    data_source.ADJ = adjacency
    data_source.DATASET_KWARGS["adj_matrix"] = adjacency
    _, _, test = data_source.datasets()
    return test


def model_for(test, adjacency: str):
    model = myGNN(
        in_channels=test.num_node_features,
        num_layers=LAYERS,
        hidden_channels=HIDDEN,
        out_channels=48,
        conv_class=SAGEConv,
        conv_kwargs={},
        heads=1,
    )
    model.load_state_dict(torch.load(checkpoint(adjacency), map_location="cpu", weights_only=False))
    model.eval()
    return model


def extract(adjacency: str):
    test = dataset(adjacency)
    model = model_for(test, adjacency)
    hidden = []
    targets = []

    def capture(_module, inputs):
        hidden.append(inputs[0].detach().cpu().numpy())

    handle = model.fc.register_forward_pre_hook(capture)
    with torch.no_grad():
        for graph in test:
            model(
                graph.x,
                graph.edge_index,
                edge_weight=getattr(graph, "edge_weight", None),
            )
            targets.append(graph.y.detach().cpu().numpy().mean(axis=1))
    handle.remove()

    z = np.stack(hidden)                 # days x nodes x hidden
    y = np.stack(targets)                # days x nodes
    node = np.tile(np.arange(z.shape[1]), z.shape[0])
    day = np.repeat(np.arange(z.shape[0]), z.shape[1])
    season = np.minimum(3, (4 * day // z.shape[0])).astype(int)
    return z.reshape(-1, z.shape[-1]), y.reshape(-1), node, season


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        HERE, "latent_representations_mlp_gnn.npz"))
    args = parser.parse_args()

    z_mlp, y_mlp, nodes, seasons = extract("eye")
    z_gnn, y_gnn, nodes_gnn, seasons_gnn = extract("space")
    if not (np.array_equal(nodes, nodes_gnn) and np.array_equal(seasons, seasons_gnn)):
        raise RuntimeError("The MLP and GNN observations are not aligned")
    if not np.allclose(y_mlp, y_gnn):
        raise RuntimeError("The MLP and GNN targets are not aligned")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output, shared_mlp=z_mlp, sage_gnn=z_gnn,
        target=y_mlp, node=nodes, season=seasons,
    )
    print(f"{len(nodes)} matched representations -> {args.output}")


if __name__ == "__main__":
    main()
