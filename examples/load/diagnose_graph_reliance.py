"""Decompose graph-layer updates into residual, self, and neighbor terms."""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CONV = os.environ.get("EXPL_CONV", "GATConv")
ADJ = os.environ.get("EXPL_ADJ", "space")
NDAYS = int(os.environ.get("DIAG_DAYS", 30))


def rel(a, b):
    return float(torch.norm(a) / (torch.norm(b) + 1e-12))


def main():
    os.environ.setdefault("EXPL_CONV", CONV)
    os.environ.setdefault("EXPL_ADJ", ADJ)
    import run_gnnexplainer_load as R
    test, model = R.build()
    empty = torch.empty((2, 0), dtype=torch.long)

    print(f"\n### node-update decomposition, {CONV} on {ADJ}, {NDAYS} days ###")
    print("x      = residual carried into the layer")
    print("h      = conv(act(norm(x)))            (what the layer adds)")
    print("h_self = same conv with no edges       (self-loop transform only)")
    print("h-h_self                                (the neighbors' contribution)\n")

    stats = {}
    with torch.no_grad():
        for d in range(min(NDAYS, len(test))):
            g = test[d]
            x = model.node_encoder(g.x)
            for li, layer in enumerate(model.layers):
                hin = layer.act(layer.norm(x))
                h_full = layer.conv(hin, g.edge_index)
                h_self = layer.conv(hin, empty)
                nb = h_full - h_self
                s = stats.setdefault(li, {k: [] for k in
                                          ("h/x", "nb/x", "nb/h", "h_node_std", "nb_node_std")})
                s["h/x"].append(rel(h_full, x))
                s["nb/x"].append(rel(nb, x))
                s["nb/h"].append(rel(nb, h_full))
                s["h_node_std"].append(float(h_full.std(0).mean() / (h_full.abs().mean() + 1e-12)))
                s["nb_node_std"].append(float(nb.std(0).mean() / (nb.abs().mean() + 1e-12)))
                x = x + h_full

    for li, s in stats.items():
        print(f"layer {li}:")
        print(f"   ||h||/||x||            {np.mean(s['h/x']):.4f}   "
              f"(the whole conv term, relative to the residual)")
        print(f"   ||h-h_self||/||x||     {np.mean(s['nb/x']):.4f}   "
              f"(NEIGHBORS only, relative to the residual)")
        print(f"   ||h-h_self||/||h||     {np.mean(s['nb/h']):.4f}   "
              f"(neighbors' share of the conv term)")
        print(f"   node-spread of h       {np.mean(s['h_node_std']):.3f}")
        print(f"   node-spread of h-h_self {np.mean(s['nb_node_std']):.3f}")

    ad = model.layers[0].conv
    if hasattr(ad, "capture_attention"):
        ad.capture_attention = True
        with torch.no_grad():
            g = test[0]
            hin = model.layers[0].act(model.layers[0].norm(model.node_encoder(g.x)))
            model.layers[0].conv(hin, g.edge_index)
        if getattr(ad, "last_attention", None) is not None:
            ei, aw = ad.last_attention
            a = aw.mean(-1)
            self_mask = ei[0] == ei[1]
            print(f"\nattention: {int(self_mask.sum())} self-loops, {int((~self_mask).sum())} real edges")
            print(f"   mean weight on self-loop {a[self_mask].mean():.4f}")
            print(f"   mean weight on neighbor  {a[~self_mask].mean():.4f}")
            print(f"   total mass on self  {a[self_mask].sum() / a.sum():.3f}   "
                  f"on neighbors {a[~self_mask].sum() / a.sum():.3f}")
        ad.capture_attention = False


if __name__ == "__main__":
    main()
