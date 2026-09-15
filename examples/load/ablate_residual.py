"""Compare the standard DeepGCN residual block with a no-skip variant."""
import os
import sys
import gc
import numpy as np
import torch
from torch import nn
import torch_geometric.nn.conv as pyg_conv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("EXPL_ADJ", "space")
import run_gnnexplainer_load as R
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device

CONVS = (os.environ.get("ABL_CONVS", "GATConv,SAGEConv")).split(",")
ADJ = os.environ.get("EXPL_ADJ", "space")
EPOCHS = int(os.environ.get("ABL_EPOCHS", 300))


class NoSkip(nn.Module):
    """A pre-activation DeepGCN block without the skip connection."""

    def __init__(self, layer):
        super().__init__()
        self.conv, self.norm, self.act = layer.conv, layer.norm, layer.act

    def forward(self, x, *args, **kwargs):
        return self.conv(self.act(self.norm(x)), *args, **kwargs)


def neighbor_share(model, test, ndays=20):
    empty = torch.empty((2, 0), dtype=torch.long)
    out = []
    with torch.no_grad():
        for d in range(min(ndays, len(test))):
            g = test[d]
            x = model.node_encoder(g.x)
            shares = []
            for layer in model.layers:
                hin = layer.act(layer.norm(x))
                h = layer.conv(hin, g.edge_index)
                nb = h - layer.conv(hin, empty)
                shares.append(float(torch.norm(nb) / (torch.norm(h) + 1e-12)))
                x = h if isinstance(layer, NoSkip) else x + h
            out.append(shares)
    return np.mean(out, axis=0)


def run(conv_name, skip):
    tr, va, te = R.datasets()
    set_device("cpu")
    torch.manual_seed(42); gc.collect()
    model = myGNN(in_channels=tr.num_node_features, num_layers=2, hidden_channels=64,
                  out_channels=48, conv_class=getattr(pyg_conv, conv_name),
                  conv_kwargs={}, heads=1)
    if not skip:
        model.layers = nn.ModuleList([NoSkip(l) for l in model.layers])
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                      batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    pred, target, *_ = trainer.train(plot_loss=False, force_training=True, save=False, patience=75)
    p, y = pred.sum(0).cpu().numpy(), target.sum(0).cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return rmse, neighbor_share(model, te)


def main():
    print(f"### residual ablation, {ADJ} graph, {EPOCHS} epochs, seed 42 ###\n")
    rows = []
    for c in CONVS:
        for skip in (True, False):
            tag = "res+   " if skip else "no-skip"
            try:
                rmse, sh = run(c, skip)
                rows.append((c, tag, rmse, sh))
                print(f"[{c:16} {tag}]  national RMSE {rmse:6.0f} MW   "
                      f"neighbor share per layer {np.round(sh, 3)}", flush=True)
            except Exception as e:
                print(f"[{c:16} {tag}]  FAILED: {str(e).splitlines()[0]}", flush=True)
    print("\n=== summary ===")
    for c in CONVS:
        r = {t: (v, s) for cc, t, v, s in rows if cc == c}
        if len(r) == 2:
            a, b = r["res+   "], r["no-skip"]
            print(f"{c:16} RMSE {a[0]:.0f} -> {b[0]:.0f} ({b[0]-a[0]:+.0f} MW)   "
                  f"neighbor share {a[1].mean():.3f} -> {b[1].mean():.3f}")


if __name__ == "__main__":
    main()
