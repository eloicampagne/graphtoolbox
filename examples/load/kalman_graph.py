"""Kalman adaptation of the regional TAM effects, independent per region or coupled by the graph, French load.

The static TAM of each region (tam_baselines.py, one model per region and half-hour slot) is fitted before
the split; its effects phi_i(x_t) (one column per term of the formula, intercept included) are frozen. The
forecast of region i is theta_i^T phi_i(x_t), and the coefficients of the 12 regions follow a random walk

    theta_{t+1} = theta_t + eta_t,   eta_t ~ N(0, q Sigma_G (x) I_d),   y_t = H_t theta_t + eps_t,

with H_t = blockdiag(phi_i(x_t)^T), eps_t ~ N(0, diag(sigma_i^2)) and sigma_i^2 the in-sample residual variance
of the static model of region i. Sigma_G is the correlation matrix of (I + lam L)^{-1}, L the Laplacian of the
DTW graph without self-loops: lam = 0 gives independent regions, lam > 0 lets the innovation of region j update
the coefficients of region i through the off-diagonal blocks of the Kalman gain. The prior theta_0 = 1 (the
static model) has covariance p0 Sigma_G (x) I_d.

Day-ahead protocol of the benchmark: the 48 forecasts of day D use theta after assimilating day D-1, then the
48 observations of day D are assimilated one half-hour at a time.

    python kalman_graph.py effects   # fits TAM before 2018 and 2019, caches kalman_effects.npz (needs tam-ml)
    python kalman_graph.py grid      # selects (q, p0, lam) on 2018 and scores 2019
    python kalman_graph.py stats     # Diebold-Mariano tests and monthly RMSE on 2019
"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "kalman_effects.npz")
H = 48


def effects():
    sys.path.insert(0, EXAMPLES)
    from tam_baselines import TASKS  # first: it keeps TAM off MPS, where it falls back to float32
    import pandas as pd
    import torch
    import tam as ta
    cfg = TASKS["load"]
    target = torch.load(os.path.join(HERE, cfg["target_pt"]), weights_only=False).numpy().astype(float)
    df = pd.concat([pd.read_csv(os.path.join(HERE, "train.csv"), usecols=cfg["cols"]),
                    pd.read_csv(os.path.join(HERE, "test.csv"), usecols=cfg["cols"])], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns=cfg["rename"])
    regions = sorted(df["Region"].unique())
    out = {"regions": np.array(regions)}
    for name, split, end in (("val", "2018-01-01", "2019-01-01"), ("test", "2019-01-01", None)):
        phis, ys, sig2, cols = [], [], [], None
        for r in regions:
            t0 = time.time()
            frame = df[df["Region"] == r].sort_values("date").reset_index(drop=True)
            train = frame[(frame["date"] >= cfg["train"][0]) & (frame["date"] < split)]
            evalp = frame[frame["date"] >= split]
            evalp = evalp[evalp["date"] < end] if end else evalp.iloc[:len(target)]
            base = ta.StaticTAM(formula=cfg["formula"], group_col=cfg["slot"], date_col="date").auto_fit(train)
            dec = base.decompose_prediction(evalp.copy())
            c = sorted(k for k in dec.columns if k.startswith("effect_"))
            assert cols in (None, c), (cols, c)
            cols = c
            phis.append(dec[cols].to_numpy(float))
            ys.append(evalp[cfg["y"]].to_numpy(float))
            dtr = base.decompose_prediction(train.copy())
            sig2.append(float(np.var(train[cfg["y"]].to_numpy(float) - dtr[cols].to_numpy(float).sum(1))))
            dates = evalp["date"].to_numpy()
            if r == regions[0]:
                ref = dates
            assert (dates == ref).all(), f"dates of {r} differ"
            print(f"{name} {r:<28} {len(dates)} rows, static RMSE "
                  f"{np.sqrt(np.mean((phis[-1].sum(1) - ys[-1]) ** 2)):.0f} MW ({time.time() - t0:.0f} s)", flush=True)
        out[f"{name}_phi"] = np.stack(phis, 1)            # T, N, d
        out[f"{name}_y"] = np.stack(ys, 1)                # T, N
        out[f"{name}_sig2"] = np.array(sig2)
        out[f"{name}_dates"] = ref.astype("datetime64[m]").astype(str)
        out["effects"] = np.array(cols)
    assert np.max(np.abs(out["test_y"].sum(1) - target)) < 1e-6, "national truth differs from target.pt"
    np.savez_compressed(CACHE, **out)


def graph_corr(lam, A):
    """Correlation matrix of (I + lam L)^{-1}, L the Laplacian of the weighted adjacency A."""
    L = np.diag(A.sum(1)) - A
    S = np.linalg.inv(np.eye(len(A)) + lam * L)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def kalman(phi, y, sig2, C, q, p0):
    """Day-ahead forecasts (T, N) of the coupled Kalman filter; rows are half-hours, 48 per day."""
    T, N, d = phi.shape
    I_d = np.eye(d)
    Q = q * np.kron(C, I_d)
    P = p0 * np.kron(C, I_d)
    theta = np.ones(N * d)
    R = np.diag(sig2)
    pred = np.empty((T, N))
    rows = np.arange(N)[:, None] * d + np.arange(d)[None]   # state indices of each region
    for day in range(T // H):
        sl = slice(day * H, (day + 1) * H)
        pred[sl] = np.einsum("tnd,nd->tn", phi[sl], theta[rows])
        for t in range(day * H, (day + 1) * H):
            Ht = np.zeros((N, N * d))
            Ht[np.arange(N)[:, None], rows] = phi[t]
            PHt = P @ Ht.T
            S = Ht @ PHt + R
            K = np.linalg.solve(S, PHt.T).T
            theta = theta + K @ (y[t] - Ht @ theta)
            P = P - K @ PHt.T + Q
            P = (P + P.T) / 2
    return pred


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def grid():
    z = np.load(CACHE)
    A = np.loadtxt(os.path.join(EXAMPLES, "graph_representations", "dtw", "W.txt"))[:, :12]
    A = A - np.diag(np.diag(A))
    for s in ("val", "test"):
        assert z[f"{s}_phi"].shape[0] % H == 0
    QS, P0S, LAMS = (1e-5, 1e-4, 1e-3, 1e-2), (1e-5, 1e-4, 1e-3, 1e-2), (0.0, 0.3, 1.0, 3.0, 10.0)
    rng = np.random.default_rng(0)
    perms = [rng.permutation(12) for _ in range(3)]
    graphs = {"dtw": A} | {f"perm{k}": A[np.ix_(p, p)] for k, p in enumerate(perms)}
    res = {"static_val": rmse(z["val_phi"].sum(2).sum(1), z["val_y"].sum(1)),
           "static_test": rmse(z["test_phi"].sum(2).sum(1), z["test_y"].sum(1)), "val": {}}
    print(f"static TAM: 2018 {res['static_val']:.0f} MW, 2019 {res['static_test']:.0f} MW", flush=True)
    for g, Ag in graphs.items():
        for q, p0, lam in itertools.product(QS, P0S, LAMS):
            if lam == 0 and g != "dtw":
                continue
            t0 = time.time()
            p = kalman(z["val_phi"], z["val_y"], z["val_sig2"], graph_corr(lam, Ag), q, p0)
            r = rmse(p.sum(1), z["val_y"].sum(1))
            res["val"][f"{g}|{q}|{p0}|{lam}"] = r
            print(f"2018 {g} q={q:g} p0={p0:g} lam={lam:g}: {r:.0f} MW ({time.time() - t0:.0f} s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "kalman_graph.json"), "w"), indent=1)

    # Selection on 2018, then 2019 with the static model refitted before 2019.
    def best(pred):
        k = min((k for k in res["val"] if pred(k)), key=res["val"].get)
        g, q, p0, lam = k.split("|")
        return k, g, float(q), float(p0), float(lam)
    choices = {"independent": best(lambda k: k.endswith("|0.0")),
               "graph dtw": best(lambda k: k.startswith("dtw|") and not k.endswith("|0.0"))}
    for k in range(len(perms)):
        choices[f"graph perm{k}"] = best(lambda key, k=k: key.startswith(f"perm{k}|"))
    preds = {"static": z["test_phi"].sum(2)}
    for name, (key, g, q, p0, lam) in choices.items():
        preds[name] = kalman(z["test_phi"], z["test_y"], z["test_sig2"], graph_corr(lam, graphs[g]), q, p0)
        res[name] = {"selected": key, "val": res["val"][key], "test": rmse(preds[name].sum(1), z["test_y"].sum(1))}
        print(f"{name:<14} selected {key}: 2018 {res[name]['val']:.0f}, 2019 {res[name]['test']:.0f} MW", flush=True)
    json.dump(res, open(os.path.join(HERE, "kalman_graph.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(HERE, "kalman_graph_forecasts.npz"), **preds)


def stats():
    """Diebold-Mariano tests on 2019 (two-sided, squared loss, h = 48) and monthly RMSE."""
    sys.path.insert(0, os.path.dirname(EXAMPLES))
    from graphtoolbox.evaluation.significance import diebold_mariano
    z, f = np.load(CACHE), np.load(os.path.join(HERE, "kalman_graph_forecasts.npz"))
    res = json.load(open(os.path.join(HERE, "kalman_graph.json")))
    y = z["test_y"].sum(1)
    nat = {k: f[k].sum(1) for k in f.files}
    sweep = np.load(os.path.join(HERE, "hpo_v030", "load_sweep_v030.npz"))
    assert np.max(np.abs(sweep["target"] - y)) < 1e-3
    nat["GatedGraphConv (paper run)"] = sweep["GatedGraphConv"].sum(0)
    nat["MLpol top (paper)"] = sweep["aggregate__mlpol_top"]
    res["dm"] = {}
    for a, b in [("graph dtw", "independent"), ("independent", "static"), ("graph dtw", "static"),
                 ("graph perm0", "independent"), ("graph perm1", "independent"), ("graph perm2", "independent"),
                 ("graph dtw", "graph perm0"), ("graph dtw", "GatedGraphConv (paper run)"),
                 ("graph dtw", "MLpol top (paper)")]:
        d = diebold_mariano(nat[a], nat[b], y, loss="squared", h=48)
        res["dm"][f"{a} vs {b}"] = {"diff": rmse(nat[a], y) - rmse(nat[b], y), "p": float(d.pvalue)}
        print(f"{a} vs {b}: {rmse(nat[a], y):.0f} vs {rmse(nat[b], y):.0f} MW, p = {d.pvalue:.1e}")
    month = np.array([int(d[5:7]) for d in z["test_dates"]])
    res["monthly"] = {k: [rmse(nat[k][month == m], y[month == m]) for m in range(1, 13)]
                      for k in ("static", "independent", "graph dtw")}
    for k, v in res["monthly"].items():
        print(f"{k:<12} " + " ".join(f"{x:5.0f}" for x in v))
    json.dump(res, open(os.path.join(HERE, "kalman_graph.json"), "w"), indent=1)


if __name__ == "__main__":
    {"effects": effects, "grid": grid, "stats": stats}[sys.argv[1]]()
