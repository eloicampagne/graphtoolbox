"""Convolution coverage of the paper: run every operator of torch_geometric.nn.conv in myGNN.

Each operator of `torch_geometric.nn.conv.__all__` (aliases are not listed there) is
instantiated in a 2-layer myGNN and trained for one step on a complete homogeneous graph
with random node features and edge weights. Prints the working count and the error of
each failing operator, and writes the outcome per operator to the JSON path given
(default convolution_coverage.json).

Usage: python examples/convolution_coverage.py [OUTPUT.json]
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import torch
import torch_geometric.nn.conv as C
from graphtoolbox.models import myGNN

torch.manual_seed(0)
N, F, OUT = 12, 20, 48
x = torch.randn(N, F)
ei = torch.combinations(torch.arange(N), 2).T
ei = torch.cat([ei, ei.flip(0)], 1)
ew = torch.rand(ei.size(1))
y = torch.randn(N, OUT)
res = {}
for name in [n for n in C.__all__ if n != "MessagePassing"]:
    try:
        m = myGNN(in_channels=F, num_layers=2, hidden_channels=16, out_channels=OUT,
                  conv_class=getattr(C, name), heads=2)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        out = m(x, ei, edge_weight=ew)
        out = out[0] if isinstance(out, tuple) else out
        loss = ((out - y) ** 2).mean(); loss.backward(); opt.step()
        assert torch.isfinite(loss)
        res[name] = "ok"
    except Exception as e:
        res[name] = f"{type(e).__name__}: {str(e)[:120]}"
ok = [k for k, v in res.items() if v == "ok"]
print(len(ok), "/", len(res))
for k, v in res.items():
    if v != "ok": print(f"  {k}: {v}")
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "convolution_coverage.json", "w"), indent=1)
