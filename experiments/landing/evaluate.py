"""Landing go / no-go decisions from 20 s deck-motion forecasts, scored against the same decision on the true future.
Deterministic only: counts, rates of correct / incorrect decisions and timing errors (no probabilities).

    D:/tools/venv-gpu/Scripts/python.exe experiments/landing/evaluate.py                   # cv0, built-in predictors
    D:/tools/venv-gpu/Scripts/python.exe experiments/landing/evaluate.py --splits cv0 --max-runs 50
    D:/tools/venv-gpu/Scripts/python.exe experiments/landing/evaluate.py --plugin my_lstm.py:make --name lstm_wang
    D:/tools/venv-gpu/Scripts/python.exe experiments/landing/evaluate.py report            # tables from saved decisions

Decision at forecast origin o (every 1 s of a test run, sim_common.origins, protocol (a)):
  1. horizon trajectory = the 10 true samples before o (needed by the centred 4 s moving statistics; they are past,
     so known at o) followed by the 100-sample (20 s) forecast x_hat[o:o+100] of heave, roll, pitch;
  2. per-sample stable mask over the 100 horizon samples with definition1.stable_mask and a threshold set
     (the centred window shrinks at the horizon end, identically for forecast and truth);
  3. GO if some start k <= D s has the mask True on [k, k + T_req) inside the horizon (and, with heave=True, the
     Definition 1 C1 heave phase at k: deck above the 60 s past mean and moving down); the start is the first such k.
     T_req is the time the deck must stay calm (descent / touchdown time, T_c < dt in the paper's terms).
  The same three steps on the TRUE future give the reference decision ('truth'). Both use only samples up to o+100.
Scores per predictor (all origins of all test runs pooled):
  true-go (both GO), false-go (forecast GO, truth no-go: the critical safety error), missed-go (forecast no-go, truth
  GO), true-no-go; false-go rate = false-go / truth-no-go origins; missed rate = missed-go / truth-GO origins;
  unsafe GO = forecast GO whose executed interval [k_hat, k_hat + T_req) is NOT all stable in the truth (all false-go
  and the mistimed true-go); safe-landing rate = 1 - unsafe GO / forecast GO; timing error = (k_hat - k_true) / 5 s on
  true-go origins (median, median |.|, mean |.|).
Predictor interface (see README.md): a factory make(split) -> f, f(ctx) -> forecasts, ctx (B, 300, 3) float64 = the
past 60 s x[o-300:o] (heave D m positive down, roll deg, pitch deg as sim_common.motion), forecasts (B, 100, 3) same
units. Built-in: oracle (true future, consistency check), persistence (hold last value), mean_past (mean of the past
60 s), wiener_past (experiments/diagnostics/wiener.py, legitimate, past 60 s only).
Output: results/sim/landing/decisions_<split>.npz (per predictor, threshold set, config: go and start index per origin
for forecast and truth, safe flag), results/sim/landing/landing_prototype.md.
"""
import argparse
import importlib.util
import os
import sys
import time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # small batched solves (wiener_past): multithreaded BLAS is much slower
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diagnostics"))
sys.path.insert(0, HERE)
import definition1 as D1  # noqa: E402
import run_sim_lstm_cv as LCV  # noqa: E402
from sim_common import FS, HORIZON, MAX_CONTEXT, load, motion, origins  # noqa: E402

OUT = D1.OUT
H, CTX = HORIZON, MAX_CONTEXT
PAD = D1.MA_LEN // 2                    # past samples the centred window reaches back
SETS = ["wang_literal", "otter_scaled", "otter_share"]
CONFIGS = [dict(T_req=2.5, D=10.0, heave=False), dict(T_req=5.0, D=10.0, heave=False),
           dict(T_req=2.5, D=10.0, heave=True), dict(T_req=2.5, D=0.0, heave=False)]   # last: 'land now' (k = 0 only)
BUILTIN = ["oracle", "persistence", "mean_past", "wiener_past"]


def cfg_key(c):
    return f"T{c['T_req']:g}_D{c['D']:g}" + ("_heave" if c["heave"] else "")


# ---------------------------------------------------------------- predictors: make(split) -> f(ctx (B,300,3)) -> (B,100,3)
def _persistence(split):
    return lambda ctx: np.repeat(ctx[:, -1:], H, axis=1)


def _mean_past(split):
    return lambda ctx: np.repeat(ctx.mean(1, keepdims=True), H, axis=1)


def _wiener_past(split):
    """wiener.predict_past on contexts: G and mean from the 60 s context only, predictor from its last L samples."""
    import wiener as W
    c = W.CFG["wiener_past"]

    def f(ctx, chunk=96):
        Wc = np.asarray(ctx, np.float64)
        mu = Wc.mean(1, keepdims=True)
        Wc = Wc - mu
        G = W.parzen(W.acov(Wc, c["L"] + H - 1, c["est"]), c["lagwin"])
        pred = np.empty((len(Wc), H, 3))
        for i in range(0, len(Wc), chunk):
            Cpp, Cpf = W.cov_blocks(G[i:i + chunk], c["L"], c["uni"])
            P = Wc[i:i + chunk, -c["L"]:].reshape(len(Cpp), -1, 1)
            a = np.linalg.solve(W.add_ridge(Cpp, G[i:i + chunk], c["L"], c["ridge"]), P)
            pred[i:i + chunk] = (np.swapaxes(a, -1, -2) @ Cpf).reshape(len(Cpp), H, 3)
        return pred + mu
    return f


FACTORIES = {"persistence": _persistence, "mean_past": _mean_past, "wiener_past": _wiener_past}


def load_plugin(spec):
    """'path/to/file.py:factory' -> factory(split) (see README.md)."""
    path, fn = spec.rsplit(":", 1)
    m = importlib.util.spec_from_file_location("landing_plugin", path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    return getattr(mod, fn)


# ---------------------------------------------------------------- decisions
def horizon_masks(y, o, fut, thr, heave):
    """y (T, 3) run, o (O,) origins, fut (O, H, 3) future (forecast or truth) -> mask (O, H), start_ok (O, H) or None."""
    past = np.lib.stride_tricks.sliding_window_view(y, PAD, axis=0)[o - PAD].transpose(0, 2, 1)   # (O, PAD, 3)
    seg = np.concatenate([past, fut], axis=1)
    m = D1.stable_mask(seg[..., 1], seg[..., 2], thr)[:, PAD:]
    st = None
    if heave:
        ref = np.lib.stride_tricks.sliding_window_view(y[:, 0], CTX)[o - CTX].mean(-1, keepdims=True)  # past 60 s mean
        st = D1.heave_phase_ok(seg[..., 0], ref)[:, PAD:]
    return m, st


def decide_run(y, o, fut_hat, fut_true):
    """-> {(set, cfg): dict(go_t, k_t, go_p, k_p, safe)} for one run."""
    res = {}
    for s in SETS:
        thr = D1.THRESHOLDS[s]
        for c in CONFIGS:
            m_t, st_t = horizon_masks(y, o, fut_true, thr, c["heave"])
            m_p, st_p = horizon_masks(y, o, fut_hat, thr, c["heave"])
            n_req, k_max = int(round(c["T_req"] * FS)), int(round(c["D"] * FS))
            go_t, k_t = D1.first_window(m_t, n_req, k_max, st_t)
            go_p, k_p = D1.first_window(m_p, n_req, k_max, st_p)
            okw = np.lib.stride_tricks.sliding_window_view(m_t, n_req, axis=-1).all(-1)       # truth calm on [k, k+n)
            safe = go_p & okw[np.arange(len(o)), np.clip(k_p, 0, None)]
            if c["heave"]:                                         # C1 (deck above its mean, moving down) must hold
                safe &= st_t[np.arange(len(o)), np.clip(k_p, 0, None)]   # in the truth at the chosen start too
            res[(s, cfg_key(c))] = dict(go_t=go_t, k_t=k_t.astype(np.int16), go_p=go_p, k_p=k_p.astype(np.int16),
                                        safe=safe)
    return res


def evaluate(split, predictors, max_runs=None, log=print):
    """predictors: {name: factory}. Returns {name: {(set, cfg): arrays (R, O)}}, ids, Hs, timings."""
    meta, X = load()
    rows = LCV.split_rows(meta, split)["test"]
    if max_runs:
        rows = rows[:max_runs]
    T = X.shape[1]
    o = origins(T)
    out, tim = {}, {}
    for name, fac in predictors.items():
        f = None if name == "oracle" else fac(split)
        acc = {}
        t0, t_pred = time.time(), 0.0
        for r, i in enumerate(rows):
            y = motion(X[i]).astype(np.float64)
            fut_true = np.lib.stride_tricks.sliding_window_view(y, H, axis=0)[o].transpose(0, 2, 1)
            if f is None:
                fut = fut_true
            else:
                ctx = np.lib.stride_tricks.sliding_window_view(y, CTX, axis=0)[o - CTX].transpose(0, 2, 1)
                tp = time.time()
                fut = np.asarray(f(np.ascontiguousarray(ctx)), np.float64)
                t_pred += time.time() - tp
                assert fut.shape == (len(o), H, 3), fut.shape
            for k, v in decide_run(y, o, fut, fut_true).items():
                for a, arr in v.items():
                    acc.setdefault(k, {}).setdefault(a, []).append(arr)
            if (r + 1) % 100 == 0 or r + 1 == len(rows):
                log(f"  {split} {name}: {r + 1}/{len(rows)} runs, {time.time() - t0:.0f} s")
        out[name] = {k: {a: np.stack(v) for a, v in d.items()} for k, d in acc.items()}
        tim[name] = (time.time() - t0, t_pred)
    return out, meta.id.to_numpy()[rows], meta.Hs.to_numpy(float)[rows], tim


def save(split, out, ids, Hs, tim):
    path = os.path.join(OUT, f"decisions_{split}.npz")
    old = dict(np.load(path, allow_pickle=False)) if os.path.exists(path) else {}
    if old and not np.array_equal(old.get("ids"), ids):
        old = {}                                                    # different run set: start over
    d = dict(old, ids=ids, Hs=Hs)
    for name, res in out.items():
        for (s, c), v in res.items():
            for a, arr in v.items():
                d[f"{name}|{s}|{c}|{a}"] = arr
        d[f"{name}|time"] = np.array(tim[name])
    np.savez_compressed(path, **d)
    return path


# ---------------------------------------------------------------- scores and report
def scores(d, name, s, c, sel=None):
    g = lambda a: d[f"{name}|{s}|{c}|{a}"] if sel is None else d[f"{name}|{s}|{c}|{a}"][sel]   # noqa: E731
    go_t, go_p, k_t, k_p, safe = g("go_t"), g("go_p"), g("k_t"), g("k_p"), g("safe")
    tg, fg = int((go_p & go_t).sum()), int((go_p & ~go_t).sum())
    mg, tn = int((~go_p & go_t).sum()), int((~go_p & ~go_t).sum())
    e = (k_p[go_p & go_t] - k_t[go_p & go_t]) / FS
    npg = tg + fg
    return dict(n=go_t.size, base=go_t.mean(), pgo=npg, tg=tg, fg=fg, mg=mg, tn=tn,
                fgr=fg / (fg + tn) if fg + tn else np.nan, mr=mg / (tg + mg) if tg + mg else np.nan,
                unsafe=int(npg - safe.sum()), safe=safe.sum() / npg if npg else np.nan,
                e_med=np.median(e) if len(e) else np.nan, e_abs=np.median(np.abs(e)) if len(e) else np.nan,
                e_mabs=np.abs(e).mean() if len(e) else np.nan)


def fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


HDR = ("| predictor | GO said | true-go | false-go | missed-go | true-no-go | false-go rate | missed rate | unsafe GO | "
       "safe-landing rate | timing err median s | median abs s | mean abs s |")


def table(d, names, s, c, sel=None):
    L = [HDR, "|" + "---|" * 13]
    for nm in names:
        if f"{nm}|{s}|{c}|go_t" not in d:
            continue
        q = scores(d, nm, s, c, sel)
        L.append(f"| {nm} | {q['pgo']} | {q['tg']} | {q['fg']} | {q['mg']} | {q['tn']} | {fmt(q['fgr'])} | "
                 f"{fmt(q['mr'])} | {q['unsafe']} | {fmt(q['safe'])} | {fmt(q['e_med'], 2)} | {fmt(q['e_abs'], 2)} | "
                 f"{fmt(q['e_mabs'], 2)} |")
    return L


def report(split="cv0"):
    d = dict(np.load(os.path.join(OUT, f"decisions_{split}.npz")))
    names = [n for n in BUILTIN if f"{n}|time" in d] + sorted(
        {k.split("|")[0] for k in d if k.endswith("|time")} - set(BUILTIN))
    Hs = d["Hs"]
    n_runs, n_or = d["ids"].shape[0], d[f"{names[0]}|{SETS[0]}|{cfg_key(CONFIGS[0])}|go_t"].shape[1]
    L = ["# Landing-window decisions from 20 s forecasts: prototype on CPU predictors", "",
         f"Written by `experiments/landing/evaluate.py` (split {split}, {n_runs} test runs x {n_or} origins = "
         f"{n_runs * n_or} decisions per row; raw decisions `decisions_{split}.npz`). Deterministic only: counts, "
         "rates of correct / incorrect decisions, timing errors.", ""]
    notes = os.path.join(HERE, "prototype_notes.md")
    if os.path.exists(notes):
        L += [open(notes, encoding="utf-8").read().strip(), ""]
    L += ["## Threshold sets (deg; `experiments/landing/definition1.py`)", "",
          "| set | moving average roll / pitch | movstd roll / pitch | test | window |", "|---|---|---|---|---|"]
    for s in D1.THRESHOLDS:
        t = D1.THRESHOLDS[s]
        L.append(f"| {s} | {t['ma_roll']:.3f} / {t['ma_pitch']:.3f} | {t['var_roll']:.3f} / {t['var_pitch']:.3f} | "
                 f"{'abs' if t['two_sided'] else 'signed'} MA | {t['L']} samples = {t['L'] / FS:g} s |")
    L += ["", "Truth statistics of every set (stable share, windows per hour by duration, per Hs): "
          "`definition1_truth_stats.md`.", ""]
    L += ["## Decision tables", "",
          "Columns: see 'How to read' above. 'GO said' = origins where the predictor's forecast gives GO. "
          "Rates are over origins pooled across runs. oracle = decision on the true future (must give 0 false-go, "
          "0 missed-go).", ""]
    for s in SETS:
        for c in CONFIGS:
            ck = cfg_key(c)
            if f"{names[0]}|{s}|{ck}|go_t" not in d:
                continue
            q = scores(d, names[0], s, ck)
            L += [f"### {s}, T_req = {c['T_req']:g} s, start within D = {c['D']:g} s"
                  + (", with heave phase C1 at the start" if c["heave"] else ""), "",
                  f"Truth GO at {q['tg'] + q['mg']} of {q['n']} origins ({q['base']:.3f}).", ""]
            L += table(d, names, s, ck) + [""]
    # per-Hs breakdown for the main configuration
    s, ck = "otter_share", cfg_key(CONFIGS[0])
    if f"{names[0]}|{s}|{ck}|go_t" in d:
        L += [f"## Per significant wave height: {s}, {ck}", ""]
        for h in sorted(set(Hs.tolist())):
            sel = np.isclose(Hs, h)
            q = scores(d, names[0], s, ck, sel)
            L += [f"### Hs = {h:.1f} m ({sel.sum()} runs; truth GO at {q['tg'] + q['mg']} of {q['n']} origins, "
                  f"{q['base']:.3f})", ""] + table(d, names, s, ck, sel) + [""]
    L += ["## Runtime (CPU, single process, 1 BLAS thread)", "", "| predictor | total s | of which forecasting s |",
          "|---|---|---|"]
    for nm in names:
        t = d[f"{nm}|time"]
        L.append(f"| {nm} | {t[0]:.0f} | {t[1]:.0f} |")
    path = os.path.join(OUT, "landing_prototype.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("written", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("--splits", default="cv0")
    ap.add_argument("--predictors", default=",".join(BUILTIN))
    ap.add_argument("--plugin", help="path.py:factory, a factory make(split) -> f(ctx) (README.md)")
    ap.add_argument("--name", help="row name of the plugin predictor")
    ap.add_argument("--max-runs", type=int)
    a = ap.parse_args()
    splits = a.splits.split(",")
    if a.cmd == "run":
        preds = {} if a.plugin else {n: FACTORIES.get(n) for n in a.predictors.split(",")}
        if a.plugin:
            preds = {"oracle": None, a.name or "plugin": load_plugin(a.plugin)}
        for sp in splits:
            out, ids, Hs, tim = evaluate(sp, preds, a.max_runs)
            print("saved", save(sp, out, ids, Hs, tim))
    for sp in splits:
        report(sp)


if __name__ == "__main__":
    main()
