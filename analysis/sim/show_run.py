"""Hình minh họa một lần chạy mô phỏng, để xem nhanh và trình bày.

  python analysis/sim/show_run.py                      # tạo các hình mẫu vào Trinh_bay/
  python analysis/sim/show_run.py <tên lần chạy>       # vẽ một lần chạy bất kỳ (tên file CSV, không có .csv)

Đọc data/sim/dataset_v2/runs/<tên>.csv (22 cột, 20 Hz). Góc trong file tính bằng rad, D (heave) dương hướng xuống;
hình đổi góc sang độ và vẽ heave dương hướng LÊN cho dễ nhìn.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "data", "sim", "dataset_v2", "runs")
OUT = os.path.join(ROOT, "Trinh_bay")
os.makedirs(OUT, exist_ok=True)
T0, T1 = 100.0, 160.0          # cửa sổ 60 s để nhìn rõ từng con sóng


def load(name):
    d = pd.read_csv(os.path.join(RUNS, name + ".csv"))
    d["heave_up"] = -d.D
    d["roll_deg"], d["pitch_deg"], d["yaw_deg"] = np.degrees(d.phi), np.degrees(d.theta), np.degrees(d.psi)
    return d


def one_run(name):
    d = load(name)
    w = d[(d.t >= T0) & (d.t <= T1)]
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    rows = [("zeta", "Sóng tại tàu (m)", "#3b6ea5"), ("heave_up", "Heave, lên + (m)", "#2a9d8f"),
            ("roll_deg", "Roll (độ)", "#e76f51"), ("pitch_deg", "Pitch (độ)", "#8e5aa8")]
    for a, (c, lab, col) in zip(ax, rows):
        a.plot(w.t, w[c], color=col, lw=1.3)
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        s = d[d.t >= 30][c].std()
        a.text(0.995, 0.92, f"độ lệch chuẩn cả lần chạy: {s:.3f}", transform=a.transAxes, ha="right", va="top", fontsize=8)
    ax[-1].set_xlabel("Thời gian (s)")
    fig.suptitle(f"{name}\nĐầu vào: sóng ngẫu nhiên  →  đầu ra: heave, roll, pitch của tàu (60 s trong 600 s)")
    fig.tight_layout()
    p = os.path.join(OUT, f"1_mot_lan_chay_{name}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def headings(base="otter_55kg_Hs0.30_Tp2.35_b{b:03d}_U1_k0_r1"):
    betas = [(0, "β = 0° (sóng theo)"), (90, "β = 90° (sóng ngang)"), (180, "β = 180° (sóng ngược)")]
    fig, ax = plt.subplots(len(betas), 2, figsize=(12, 7.5), sharex=True, sharey="col")
    for i, (b, lab) in enumerate(betas):
        d = load(base.format(b=b))
        w = d[(d.t >= T0) & (d.t <= T1)]
        for j, (c, nm, col) in enumerate([("roll_deg", "Roll", "#e76f51"), ("pitch_deg", "Pitch", "#8e5aa8")]):
            a = ax[i, j]
            a.plot(w.t, w[c], color=col, lw=1.2)
            a.grid(alpha=0.3)
            a.set_title(f"{lab} — {nm}, độ lệch chuẩn {d[d.t >= 30][c].std():.2f}°", fontsize=10)
            if j == 0:
                a.set_ylabel("độ")
    ax[-1, 0].set_xlabel("Thời gian (s)"); ax[-1, 1].set_xlabel("Thời gian (s)")
    fig.suptitle("Cùng một biển (Hs 0.30 m, Tp 2.35 s, cùng seed), tàu 55 kg chạy 1 m/s, chỉ đổi hướng sóng:\n"
                 "sóng ngang làm lắc ngang (roll), sóng dọc làm chúi (pitch)")
    fig.tight_layout()
    p = os.path.join(OUT, "2_cung_bien_khac_huong_song.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(one_run(sys.argv[1]))
    else:
        print(one_run("otter_55kg_Hs0.30_Tp2.35_b135_U1_k0_r1"))
        print(headings())
