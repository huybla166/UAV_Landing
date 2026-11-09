"""Write a Kaggle script kernel that runs a given list of phylstm_kaggle.py jobs.

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/make_kernel.py [--dataset OWNER/SLUG] NAME JOB [JOB ...]

JOB = 'split:variant[:seed[:epochs]]' (phylstm_kaggle.parse_spec). Writes experiments/phylstm/kernels/<slug>/ with
  kernel-metadata.json  id 'wanghui1101/usv-phylstm-<slug>', private, GPU T4 x2, internet off, dataset
                        --dataset (default wanghui1101/usv-dataset-v2-phylstm), code_file phylstm_kaggle.py
  phylstm_kaggle.py     a copy of experiments/phylstm/phylstm_kaggle.py whose DEFAULT_ARGS line is ['--jobs', *JOBS]
and prints the estimated run time of each of the two GPU workers (jobs[i::2], as phylstm_kaggle.main). Refuses to
write a kernel whose estimate exceeds LIMIT_H: split the job list into two kernels instead. Push it yourself:
  kaggle kernels push -p experiments/phylstm/kernels/<slug>

Dataset source. The phase-2 variants (phylstm_kaggle.EXTENDED: cummins, seastate, phy_seastate, cummins_seastate)
need the extended pack: the files of data/sim/dataset_v2/kaggle_phylstm_v2/ (usv_phylstm.npz with nu/group/sea and
cummins_groups.npz, pack_kaggle.py --v2), uploaded as a new version of wanghui1101/usv-dataset-v2-phylstm (the
default source) or as their own dataset (pass --dataset). The kernel finds both files anywhere under /kaggle/input
(phylstm_kaggle.find_data prefers a usv_phylstm.npz with cummins_groups.npz beside it; groups_file) and its worker
stops before training if they are missing. The original variants run unchanged on either version (bit-identical
results, the old arrays are identical).

The kernels of phase 1 (job order balances the two workers):
  seed1     all 8 splits x {phy, data} at seed 1
  controls  cv0-cv2 x {je, phy_past, anchor, sobolev} at seed 0, cv0 data and phy at 45 epochs
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv, _argv = sys.argv[:1], sys.argv                    # phylstm_kaggle reads sys.argv at import
import phylstm_kaggle as pk  # noqa: E402

sys.argv = _argv
OWNER = "wanghui1101"
DATASET = f"{OWNER}/usv-dataset-v2-phylstm"                  # default dataset source (a new version holds the extended pack)
N_GPU = 2
LIMIT_H = 11.0                                               # phylstm_kaggle --max-hours; Kaggle stops a session at 12 h
# T4 seconds per training epoch (incl. the validation forecast). phy, data: measured in the 'cv' kernel
# (results/sim/phylstm/log_0.txt, log_1.txt: 122-135 s and 67 s). The others: data x their per-batch cost relative
# to data on the local RTX 3060 (phy_past 1.87-1.92, je 1.07-1.09, anchor 1.06-1.09, sobolev 1.05-1.13), rounded up.
EPOCH_S = {"phy": 135, "data": 67, "phy_past": 135, "je": 77, "anchor": 77, "sobolev": 77}
# Phase 2: data x the per-epoch cost relative to data measured on the local RTX 3060 (1 full epoch on cv0 each,
# 05/11/2026, each in its own process; data 36.9 s and 35.8 s): cummins, seastate, cummins_seastate 1.19 (43.1-43.3 s),
# phy_seastate 76.6 s = phy 76.5 s x 1.00 (phy 2.10 x data here, 2.01 on the T4), rounded up.
EPOCH_S.update({"cummins": 80, "seastate": 80, "cummins_seastate": 80, "phy_seastate": 136})
# Review 05/11/2026 (per-batch timing, RTX 3060): cummins x1.22, seastate x1.03 of data -> 82 s and 69 s on the T4.
EPOCH_S.update({"cummins": 82, "cummins_seastate": 82, "seastate": 69})
EPOCH_S.update({"cummins_fexc": 100, "data_fexc": 85})      # 07/11: + one FIR pass per batch for the input channels (guess)
JOB_OVERHEAD_S = 100                                         # test forecasts (best + last), job set-up: 'cv' kernel
                                                             # 6.91-6.95 h vs 6.70 h of epochs for 8 jobs per worker


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def estimate(jobs):
    """Estimated hours of each GPU worker."""
    a = type("A", (), {"seed": 0, "epochs": pk.DEFAULT_EPOCHS})()
    hours = []
    for i in range(N_GPU):
        s = 0.0
        for spec in jobs[i::N_GPU]:
            _, variant, _, epochs, _ = pk.parse_spec(spec, a)
            s += epochs * EPOCH_S[variant] + JOB_OVERHEAD_S
        hours.append(s / 3600)
    return hours


def write(name, jobs, dataset=DATASET, owner=OWNER, extra=()):
    sl = slug(name)
    if not sl:
        raise SystemExit(f"bad kernel name {name!r}")
    a = type("A", (), {"seed": 0, "epochs": pk.DEFAULT_EPOCHS})()
    keys = [(p[0], p[4]) for p in (pk.parse_spec(j, a) for j in jobs)]        # raises on a bad spec
    if len(set(keys)) < len(keys):
        raise SystemExit("two jobs write the same result keys")
    hours = estimate(jobs)
    for i, h in enumerate(hours):
        print(f"  GPU {i}: {h:.2f} h  {jobs[i::N_GPU]}")
    if max(hours) > LIMIT_H:
        raise SystemExit(f"estimated {max(hours):.2f} h on one GPU > {LIMIT_H} h: split the jobs into two kernels")
    title = f"USV PhyLSTM {sl}"
    kid = f"usv-phylstm-{sl}"
    assert slug(title) == kid and 5 <= len(title) <= 50, (title, kid)
    src = open(os.path.join(HERE, "phylstm_kaggle.py"), encoding="utf-8").read()
    pat = re.compile(r"^DEFAULT_ARGS = \[\].*$", re.M)
    if len(pat.findall(src)) != 1:
        raise SystemExit("phylstm_kaggle.py: expected exactly one 'DEFAULT_ARGS = []' line")
    line = f"DEFAULT_ARGS = {['--jobs', *jobs, *extra]!r}  # make_kernel.py {sl}"
    src = pat.sub(lambda _: line, src)
    out = os.path.join(HERE, "kernels", sl)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "phylstm_kaggle.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    meta = {"id": f"{owner}/{kid}", "title": title, "code_file": "phylstm_kaggle.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [dataset], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"{len(jobs)} jobs -> {out} (id {meta['id']}, dataset {dataset}, max {max(hours):.2f} h per GPU)")
    ext = sorted({p[1] for p in (pk.parse_spec(j, a) for j in jobs)} & set(pk.EXTENDED))
    if ext:
        print(f"  variants {ext} need the extended pack: {dataset} must hold usv_phylstm.npz (with nu, group, sea) "
              f"and cummins_groups.npz from data/sim/dataset_v2/kaggle_phylstm_v2/")


def main():
    args = sys.argv[1:]
    dataset, owner, extra = DATASET, OWNER, []
    while args[:1] in (["--dataset"], ["--owner"], ["--args"]):
        if len(args) < 2:
            raise SystemExit("--dataset OWNER/SLUG | --owner OWNER | --args '--units 200 --mirror'")
        if args[0] == "--args":
            extra = args[1].split()                          # extra phylstm_kaggle.py arguments after the jobs
        elif args[0] == "--dataset":
            if "/" not in args[1]:
                raise SystemExit("--dataset OWNER/SLUG")
            dataset = args[1]
        else:
            owner = args[1]
        args = args[2:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    write(args[0], args[1:], dataset, owner, extra)


if __name__ == "__main__":
    main()
