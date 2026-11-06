"""Non-graph baselines of the paper: MLPs shared across or fitted per node, and XGBoost pooled over the nodes
(b-global) or fitted per node (b-local).

Copied without change from the benchmark of the companion study (Campagne et al., 2026, gnn4elf,
experiments/benchmark/benchmark_rfrance.py and ff_model.py), so that the baselines of the load and net-load tables
are refitted from this repository: national, system_metrics, mlp_fit, run_mlps and run_xgb. The day-aligned samples
are passed in by the callers (nongraph_baselines.py, netload_nongraph.py, mint_tam.py).
"""
import numpy as np
import torch

H = 48


class FF(torch.nn.Module):
    """Feedforward network: num_layers - 1 hidden layers of width hidden_channels with ReLU, then a linear head."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, **kwargs):
        super(FF, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=in_channels, out_features=hidden_channels)])
        for _ in range(num_layers - 2):
            linear = torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels)
            self.layers.append(linear)
        self.fc = torch.nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.fc(x)
        return x


def system_metrics(pred, target):
    p = np.asarray(pred).sum(axis=0) if np.ndim(pred) == 2 else np.asarray(pred)
    t = np.asarray(target).sum(axis=0) if np.ndim(target) == 2 else np.asarray(target)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mape = float(100 * np.mean(np.abs((p - t) / t)))
    return rmse, mape



def national(Y):
    """[S, N, 48] -> national half-hourly series of length S*48."""
    return Y.sum(axis=1).reshape(-1)


def mlp_fit(Xtr, Ytr, Xva, Yva, Xte, cfg, seed, per_node, epochs=150, patience=12):
    """Shared MLP on stacked nodes, or one MLP per node trained in parallel.

    Targets are standardised per node on the training samples. The per-node
    variant holds one weight set per node (batched), so nodes do not share any
    parameter; early stopping reads the summed validation loss.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    S, N, F = Xtr.shape
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    xtr, xva, xte = f(Xtr), f(Xva), f(Xte)
    ytr, yva = f((Ytr - mu) / sd), f((Yva - mu) / sd)
    Hh, L = cfg["hidden_channels"], cfg["num_layers"]
    if per_node:
        dims = [F] + [Hh] * (L - 1) + [H]
        Ws = [torch.nn.Parameter(torch.randn(N, a, b) / np.sqrt(a)) for a, b in zip(dims[:-1], dims[1:])]
        bs = [torch.nn.Parameter(torch.zeros(N, b)) for b in dims[1:]]
        params = Ws + bs
        def fwd(x):                                     # x [B, N, F]
            h = x
            for i, (W, b) in enumerate(zip(Ws, bs)):
                h = torch.einsum("bnf,nfh->bnh", h, W) + b
                if i < len(Ws) - 1:
                    h = torch.relu(h)
            return h
    else:

        net = FF(in_channels=F, hidden_channels=Hh, out_channels=H, num_layers=L)
        params = list(net.parameters())
        fwd = net
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    best, best_state, bad = np.inf, None, 0
    bsz = cfg["batch_size"]
    for ep in range(epochs):
        perm = torch.randperm(S)
        for i in range(0, S, bsz):
            idx = perm[i:i + bsz]
            loss = ((fwd(xtr[idx]) - ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((fwd(xva) - yva) ** 2).mean())
        if v < best:
            best, bad = v, 0
            best_state = [p.detach().clone() for p in params]
        else:
            bad += 1
            if bad >= patience:
                break
    with torch.no_grad():
        for p, q in zip(params, best_state):
            p.copy_(q)
        pred = fwd(xte).numpy() * sd + mu
        pva = fwd(xva).numpy() * sd + mu
    return pred, pva


def run_mlps(tr, va, te, n_seeds=10):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    out = {}
    rng = np.random.default_rng(42)
    cands = {"hidden_channels": [64, 128, 192, 256], "num_layers": [2, 3],
             "lr": [3e-4, 1e-3, 2e-3], "batch_size": [2, 4, 8]}
    for name, per_node in (("Shared MLP", False), ("Per-node MLP", True)):
        best, best_cfg = np.inf, None
        for t in range(8):
            cfg = {k: (float(rng.choice(v)) if k == "lr" else int(rng.choice(v))) for k, v in cands.items()}
            _, pva = mlp_fit(Xtr, Ytr, Xva, Yva, Xte, cfg, 42, per_node)
            r, _ = system_metrics(national(pva), national(Yva))
            if r < best:
                best, best_cfg = r, cfg
        preds = []
        for s in range(n_seeds):
            p, _ = mlp_fit(Xtr, Ytr, Xva, Yva, Xte, best_cfg, 42 + s, per_node)
            preds.append(p)
        out[name] = {"cfg": best_cfg, "preds": preds}
        print(f"{name}: cfg {best_cfg}", flush=True)
    return out


def run_xgb(tr, va, te, n_seeds=3):
    import xgboost as xgb
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    S, N, F = Xtr.shape
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    ytr_s = (Ytr - mu) / sd; yva_s = (Yva - mu) / sd
    space = lambda rng: {
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1, 0.2])),
        "max_depth": int(rng.choice([4, 6, 8, 10])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 6.0, 10.0])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2])),
        "reg_lambda": float(rng.choice([0.1, 1.0, 5.0, 10.0])),
        "n_estimators": int(rng.choice([120, 180, 260, 340]))}
    out = {}
    # b-global: one model per horizon on the stacked nodes
    Xs = Xtr.reshape(-1, F); Ys = ytr_s.reshape(-1, H)
    Xv = Xva.reshape(-1, F); Yv = yva_s.reshape(-1, H)
    rng = np.random.default_rng(42); best, best_cfg = np.inf, None
    for t in range(10):
        cfg = space(rng)
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, **cfg)
        m.fit(Xs, Ys[:, H // 2]); v = float(np.sqrt(np.mean((m.predict(Xv) - Yv[:, H // 2]) ** 2)))
        if v < best:
            best, best_cfg = v, cfg
    preds = []
    for s in range(n_seeds):
        P = np.zeros((Xte.shape[0] * N, H))
        for h in range(H):
            m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                 random_state=42 + s, **best_cfg)
            m.fit(Xs, Ys[:, h]); P[:, h] = m.predict(Xte.reshape(-1, F))
        preds.append(P.reshape(Xte.shape[0], N, H) * sd + mu)
    out["XGBoost (b-global)"] = {"cfg": best_cfg, "preds": preds}
    print(f"XGBoost b-global: cfg {best_cfg}", flush=True)
    # b-local: one model per node and horizon, defaults of the notebook
    loc_cfg = {"learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1.0, "subsample": 1.0,
               "colsample_bytree": 1.0, "n_estimators": 120}
    preds = []
    for s in range(n_seeds):
        P = np.zeros((Xte.shape[0], N, H))
        for n in range(N):
            for h in range(H):
                m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                     random_state=42 + s, n_jobs=1, **loc_cfg)
                m.fit(Xtr[:, n, :], ytr_s[:, n, h]); P[:, n, h] = m.predict(Xte[:, n, :])
        preds.append(P * sd + mu)
        print(f"XGBoost b-local seed {s} done", flush=True)
    out["XGBoost (b-local)"] = {"cfg": loc_cfg, "preds": preds}
    return out


