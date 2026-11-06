"""Stop each search after trial 99 and record its best trial from the Optuna log.

The TPE sampler is seeded and does not depend on n_trials, so the first 100
trials of the running 200-trial searches are those of a search run with
n_trials=100. The Optimizer writes its summary only when optimize() returns, so
the summary is rebuilt here from the per-trial log lines.
"""
import ast
import json
import os
import re
import signal
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 100
LAST = re.compile(rf"Trial {N_TRIALS - 1} (finished|pruned)")
DONE_LINE = re.compile(r"Trial (\d+) finished with value: ([0-9.eE+-]+) and parameters: (\{.*?\})\.")
PRUNED_LINE = re.compile(r"Trial (\d+) pruned")


def read_log(conv):
    with open(os.path.join(HERE, "logs", f"{conv}.log"), errors="ignore") as f:
        return f.read().replace("\r", "\n")


def pid_of(conv):
    out = subprocess.run(["pgrep", "-f", f"run_hpo.py {conv}$"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def summarize(conv, log):
    done = {int(n): (float(v), ast.literal_eval(p)) for n, v, p in DONE_LINE.findall(log) if int(n) < N_TRIALS}
    pruned = {int(n) for n in PRUNED_LINE.findall(log) if int(n) < N_TRIALS}
    best_n = min(done, key=lambda n: done[n][0])
    value, params = done[best_n]
    out = os.path.join(HERE, f"results_optim_{conv}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "myGNN.txt"), "w") as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(done) + len(pruned)}\n")
        f.write(f"  Number of pruned trials: {len(pruned)}\n")
        f.write(f"  Number of complete trials: {len(done)}\n\n")
        f.write(f"Best trial: {best_n}\n  Value: {value}\n\n  Params:\n")
        for k, v in params.items():
            f.write(f"    {k}: {v}\n")
    with open(os.path.join(out, "myGNN_best.json"), "w") as f:
        json.dump({"value": value, "params": params, "trial": best_n,
                   "n_trials": len(done) + len(pruned), "n_pruned": len(pruned)}, f, indent=2)
    return best_n, value, params, len(pruned)


pending = {"LEConv", "GATConv", "APPNP"}
while pending:
    for conv in sorted(pending):
        log = read_log(conv)
        if LAST.search(log):
            for pid in pid_of(conv):
                os.kill(pid, signal.SIGTERM)
            best_n, value, params, n_pruned = summarize(conv, log)
            print(f"{conv} stopped after {N_TRIALS} trials ({n_pruned} pruned): best trial {best_n}, "
                  f"val {value:.6g}, {params}", flush=True)
            pending.discard(conv)
        elif not pid_of(conv):
            print(f"{conv} exited before trial {N_TRIALS - 1}; check logs/{conv}.log", flush=True)
            pending.discard(conv)
    time.sleep(15)
