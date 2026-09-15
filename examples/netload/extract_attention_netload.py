"""Extract attention matrices on net-load (mirror of the load attention_matrix/).

Trains GATConv and GATv2Conv (heads=1) on the DTW graph, then runs an explicit
eval pass over the test loader and saves the per-batch attention dictionaries to
attention_matrix/{model}_dtw/test_.../num_batch{i}.pt, matching the load layout.

Attention saving is done explicitly here because the current GraphToolbox
``myGNN.forward`` returns attention but no longer writes it to disk itself.
"""
import os
import gc

from torch_geometric import seed_everything
from torch_geometric.nn.conv import GATConv, GATv2Conv

from graphtoolbox.interpretability import dump_attention_batches
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device

import netload_pipeline as P

ADJ = "dtw"
HIDDEN = 64
NUM_LAYERS = 2
NUM_EPOCHS = int(os.environ.get("ATTN_EPOCHS", "100"))
HEADS = 1
BATCH = 32

MODELS = {
    "GATConv": (GATConv, {"heads": HEADS}),
    "GATv2Conv": (GATv2Conv, {"heads": HEADS}),
}


def main():
    set_device("cpu")
    print("Building net-load datasets...")
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)

    for name, (cls, kw) in MODELS.items():
        root = os.path.join(os.path.dirname(__file__), "attention_matrix", f"{name}_{ADJ}")
        sub = os.path.join(root, f"test_batch{BATCH}_hidden{HIDDEN}_layers{NUM_LAYERS}_epochs{NUM_EPOCHS}_heads{HEADS}")
        if os.path.isdir(sub) and any(f.startswith("num_batch") for f in os.listdir(sub)):
            print(f"[cache] {name}: attention already extracted -> {sub}")
            continue
        print(f"\nTraining {name} (heads={HEADS})...")
        seed_everything(42); gc.collect()
        model = myGNN(in_channels=train.num_node_features, num_layers=NUM_LAYERS,
                      hidden_channels=HIDDEN, out_channels=P.OUT_CHANNELS,
                      conv_class=cls, conv_kwargs=kw)
        trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                          batch_size=BATCH, model_kwargs={"lr": 1e-3, "num_epochs": NUM_EPOCHS},
                          reconcile=True, top_level_model="xgb", lam_reg=0)
        res = trainer.train(force_training=False, patience=30)
        pred, target = res[0], res[1]
        m = P.metrics_dict(pred.sum(dim=0).cpu(), target.sum(dim=0).cpu())
        print(f"  national RMSE={m['rmse']:.0f} MW  MAPE={m['mape']:.2f}%")
        n = dump_attention_batches(trainer.model, test, sub, batch_size=BATCH)
        print(f"  attention maps saved: {n} batch file(s) under {sub}")


if __name__ == "__main__":
    main()
