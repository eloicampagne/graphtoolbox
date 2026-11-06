"""MLpol mixture over the MinT-reconciled forecasts, for the old and the v0.3.0 operator sets.

Reads ./mint_all.npz and ./mint_all_ECConv.npz (written by mint_all.py). The old
set reproduces the ICTAI reconciliation paragraph; the new set replaces LEConv,
GATConv and APPNP by the configurations of their v0.3.0 searches, and the default set
(the paper's) by the common configuration of the sweep.
"""
import json
import os

import numpy as np

from graphtoolbox.aggregation import Aggregation
from graphtoolbox.evaluation import diebold_mariano

HERE = os.path.dirname(os.path.abspath(__file__))
arrays = {}
for f in ("mint_all.npz", "mint_all_ECConv.npz", "mint_all_APPNP_GATConv_LEConv.npz"):
    if not os.path.exists(os.path.join(HERE, f)):
        continue
    with np.load(os.path.join(HERE, f)) as z:
        arrays.update({k: z[k] for k in z.files})
y = arrays["target"]
# ECConv is an alias of NNConv in PyTorch Geometric and is left out of the mixture.
NEW = {"LEConv", "GATConv", "APPNP"}
same = sorted(k[len("rec__"):] for k in arrays if k.startswith("rec__") and "__" not in k[5:] and k[5:] not in NEW | {"ECConv"})
sets = {"old": same + [f"{n}__old" for n in sorted(NEW)], "new": same + sorted(NEW)}
if all(f"rec__{n}__default" in arrays for n in NEW):
    sets["default"] = same + [f"{n}__default" for n in sorted(NEW)]


def rmse(p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


out = {}
for tag, names in sets.items():
    nat = {n: arrays[f"rec__{n}"].sum(axis=0) for n in names}
    unrec = {n: arrays[f"unrec__{n}"].sum(axis=0) for n in names}
    experts = np.column_stack([nat[n] for n in sorted(nat)])
    mix = Aggregation(model="MLpol", loss="square").run(experts, y, block_size=48).prediction_
    best_rec = min(nat, key=lambda n: rmse(nat[n]))
    best_unrec = min(unrec, key=lambda n: rmse(unrec[n]))
    dm_rec = diebold_mariano(mix, nat[best_rec], y, loss="squared", h=48)
    dm_same = diebold_mariano(mix, nat[best_unrec], y, loss="squared", h=48)
    out[tag] = {"n_operators": len(names), "mixture_rec_rmse": rmse(mix),
                "best_unrec": best_unrec, "best_unrec_rmse": rmse(unrec[best_unrec]),
                "best_unrec_after_mint_rmse": rmse(nat[best_unrec]),
                "best_rec": best_rec, "best_rec_rmse": rmse(nat[best_rec]),
                "dm_p_mix_vs_best_rec": float(dm_rec.pvalue),
                "dm_p_mix_vs_best_unrec_op_after_mint": float(dm_same.pvalue)}
json.dump(out, open(os.path.join(HERE, "mint_mixture.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
