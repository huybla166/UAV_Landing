"""The TSF-baseline study end to end on the local GPU (experiments/phylstm/tsf_baselines.py), resumable: every stage
skips the jobs whose results already exist (tsf_baselines.py checks that they used the same lr and configuration).
  1 tune   cv0, every model x LRS, validation only (the cv0 test runs are not forecast)    -> results/sim/tsf_tune
           a model whose best lr lies at an end of the grid gets the next lr beyond it (factor ~3), at most 2 more;
           an lr whose tuning job failed is not retried
  2 pick   per model the lr with the lowest best-epoch validation MSE                     -> results/sim/tsf_tune/chosen_lr.json
           (a chosen_lr.json from an earlier start is kept, so a restart never changes the lr of finished splits)
  3 cv0    the chosen tuning run of each model scored on the cv0 test fold                 -> results/sim/phylstm/tsf
  4 rest   cv1..cv5, ood_Hs, ood_beta with the chosen lr, one split at a time              -> results/sim/phylstm/tsf
  5 check  every model at its chosen lr in float32 on cv0, validation only: is fp16 AMP harmless? -> results/sim/tsf_tune
Keeps Windows awake while it runs (SetThreadExecutionState). Log: results/sim/tsf_tune/orchestrate.log, with the exit
code of every stage and a summary of the failed ones at the end.

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/tsf_orchestrate.py
"""
import ctypes
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
SCRIPT = os.path.join(HERE, "tsf_baselines.py")
TUNE = os.path.join(ROOT, "results", "sim", "tsf_tune")
FINAL = os.path.join(ROOT, "results", "sim", "phylstm", "tsf")
MODELS = ["itransformer", "timexer", "patchtst"]           # fastest first
LRS = [1e-4, 3e-4, 1e-3]
LADDER = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
REST = ["cv1", "cv2", "cv3", "cv4", "cv5", "ood_Hs", "ood_beta"]
EPOCHS = 60
CHOSEN = os.path.join(TUNE, "chosen_lr.json")
os.makedirs(TUNE, exist_ok=True)
LOG = open(os.path.join(TUNE, "orchestrate.log"), "a", encoding="utf-8")
FAILED = []


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def keep_awake():
    if os.name == "nt":                                    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(args, worker):
    cmd = [PY, SCRIPT, *args, "--worker", worker, "--epochs", str(EPOCHS)]
    log("run " + " ".join(args) + f" (worker {worker})")
    t0 = time.time()
    code = subprocess.call(cmd, cwd=ROOT)
    log(f"  exit {code} after {(time.time() - t0) / 60:.1f} min")
    if code:
        FAILED.append(f"{worker}: {' '.join(args)}")
    return code


def tune_results():
    """{model: {lr: best validation MSE}} from the tuning histories (fp16 runs of EPOCHS epochs)."""
    res = {}
    for f in os.listdir(TUNE):
        if f.startswith("history_") and f.endswith(".json"):
            for key, h in json.load(open(os.path.join(TUNE, f))).items():
                if h.get("epochs") == EPOCHS and not h.get("fp32") and "_fp32" not in key and "best_val" in h:
                    res.setdefault(h["model"], {})[float(h["lr"])] = float(h["best_val"])
    return res


def lr_str(lr):
    return f"{lr:g}"


def main():
    keep_awake()
    log(f"start: models {MODELS}, lr grid {LRS}, epochs {EPOCHS}")
    if os.path.exists(CHOSEN):
        chosen = {m: float(c["lr"]) for m, c in json.load(open(CHOSEN)).items()}
        log(f"  learning rates from {CHOSEN}: {chosen}")
    else:
        for m in MODELS:                                                      # 1 tune
            tried = set(LRS)
            run(["--tune", "--out", TUNE, "--jobs", *[f"cv0:{m}:{lr_str(lr)}" for lr in LRS]], f"tune_{m}")
            for _ in range(2):                                                # extend the grid at an edge
                r = tune_results().get(m, {})
                if not r:
                    break
                best = min(r, key=r.get)
                ok = sorted(r)
                i = min(range(len(LADDER)), key=lambda j: abs(LADDER[j] - best))
                if best == ok[0] and i > 0 and LADDER[i - 1] not in tried:
                    nxt = LADDER[i - 1]
                elif best == ok[-1] and i < len(LADDER) - 1 and LADDER[i + 1] not in tried:
                    nxt = LADDER[i + 1]
                else:
                    break
                tried.add(nxt)
                log(f"  {m}: best lr {best:g} at the end of {ok}, adding {nxt:g}")
                run(["--tune", "--out", TUNE, "--jobs", f"cv0:{m}:{lr_str(nxt)}"], f"tune_{m}")
        res = tune_results()                                                  # 2 pick
        chosen = {}
        for m in MODELS:
            if m not in res:
                log(f"  {m}: no tuning result, left out")
                continue
            chosen[m] = min(res[m], key=res[m].get)
            log(f"  {m}: validation MSE by lr " + ", ".join(f"{lr:g} {v:.4f}" for lr, v in sorted(res[m].items()))
                + f" -> lr {chosen[m]:g}")
        json.dump({m: {"lr": chosen[m], "val_by_lr": {lr_str(k): v for k, v in sorted(res[m].items())}} for m in chosen},
                  open(CHOSEN, "w"), indent=1)
    run(["--score-from", TUNE, "--out", FINAL, "--jobs", *[f"cv0:{m}:{lr_str(chosen[m])}" for m in chosen]], "cv0")  # 3
    for s in REST:                                                            # 4
        run(["--out", FINAL, "--jobs", *[f"{s}:{m}:{lr_str(chosen[m])}" for m in chosen]], s)
    for m in ["patchtst", "timexer", "itransformer"]:                         # 5
        if m in chosen:
            run(["--tune", "--fp32", "--out", TUNE, "--jobs", f"cv0:{m}:{lr_str(chosen[m])}"], f"fp32_{m}")
    log("all stages done" + (f"; FAILED: {FAILED}" if FAILED else "; no failures"))


if __name__ == "__main__":
    main()
