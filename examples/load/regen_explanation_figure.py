"""Regenerate selected monthly panels of the explanation-graph figure."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.interpretability import (
    VisualizationConfig,
    load_attention_batches,
    plot_explanation_graph,
)
import config as C

ATTN = os.path.join(HERE, "attention_matrix", "GATConv_dtw",
                    "test_batch32_hidden64_layers2_epochs100_heads1")
OUT_ROOT = os.path.join(HERE, "interpretability_regen")
MONTHS = sys.argv[1:] or ["January", "July"]


def build_test_dataset():
    data = DataClass(path_train=os.path.join(HERE, "train.csv"),
                     path_test=os.path.join(HERE, "test.csv"),
                     data_kwargs=C.data_kwargs, folder_config=HERE)
    common = dict(graph_folder=os.path.join(os.path.dirname(HERE), "graph_representations"),
                  dataset_kwargs=C.dataset_kwargs, out_channels=48)
    train = GraphDataset(data=data, period="train", **common)
    test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)
    return test


def main():
    em, edge_index = load_attention_batches(ATTN)
    test = build_test_dataset()
    ne_graph = test[0].edge_index.shape[1]
    if ne_graph != em.shape[1]:
        raise ValueError(f"cached attention has {em.shape[1]} edges but the test graph "
                         f"has {ne_graph}: the cache does not match this graph")

    explanation = types.SimpleNamespace(edge_index=edge_index,
                                        edge_mask=em.mean(dim=0))
    all_explanations = {"GATConv": {"mean": explanation, "edge_masks": em}}

    cfg = VisualizationConfig(name="load", output_root=OUT_ROOT,
                              pos_df=C.df_pos, lon_col="LONGITUDE", lat_col="LATITUDE",
                              file_ext="pdf", fontsize=13, labelsize=9,
                              node_size=180, label_fontsize=7, save_dpi=300)
    print(f"panels: {MONTHS}")
    plot_explanation_graph(all_explanations, test, C.data_kwargs,
                           vis_mode="model", months_to_plot=MONTHS,
                           edge_keep_ratio=0.10, viz_cfg=cfg)


if __name__ == "__main__":
    main()
