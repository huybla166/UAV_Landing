"""Plug different probabilistic base forecasters into the same risk-controlled landing-window pipeline.

For every base model: per-lead coverage of its raw 10-90 % band, and the decision metrics of
  (1) pooled Conformal Risk Control  and  (2) CRC by wave heading + ACI (recommended rule),
over all 8 threshold x window settings, at alpha = 0.05.
"""
import json
import os

import numpy as np

import run_regime_calibration as rc
import windows_lib as wl
from common import RESULTS, RUNS, targets

# Trained models need leakage-free (nested) calibration forecasts: preds_<name>_nested.npz.
# K²VAE is left out of the decision comparison: its forecasts have no skill on the real data (see summary.md).
BASES = {"tirex_ctx300": "TiRex zero-shot", "tirex_sp": "TiRex zero-shot (1 lượt)",
         "tirex_ft": "TiRex tinh chỉnh (hiệu chỉnh lồng nhau)"}
SETTINGS = [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]
available = [b for b in BASES if os.path.exists(os.path.join(RESULTS, f"preds_{b}.npz"))]

rows, out = [], {}
for b in available:
    wl.set_base(b)
    rc.tx_med, rc.tx_w = wl.tx_med, wl.tx_w
    cov = np.mean([((wl.truth[r] >= wl.tx_q[r][..., 0]) & (wl.truth[r] <= wl.tx_q[r][..., -1])).mean()
                   for r in RUNS])
    width = np.mean([(wl.tx_w[r][:, :25, 1:]).mean() for r in RUNS])
    main_pooled = wl.evaluate("tirex_crc", 1.5, 1.0, 5.0, 0.05)[0]
    main_reg = rc.evaluate("crc_heading_aci", 1.5, 1.0, 5.0)[0]
    grid_pooled = [wl.evaluate("tirex_crc", r_, p_, T, 0.05)[0] for r_, p_ in SETTINGS for T in (5.0, 10.0)]
    grid_reg = [rc.evaluate("crc_heading_aci", r_, p_, T)[0] for r_, p_ in SETTINGS for T in (5.0, 10.0)]
    out[b] = dict(raw_band_coverage=cov, mean_half_width_roll_pitch_5s=width, main_pooled=main_pooled,
                  main_heading_aci=main_reg,
                  grid_pooled_recall=float(np.mean([g["recall"] for g in grid_pooled])),
                  grid_reg_recall=float(np.mean([g["recall"] for g in grid_reg])),
                  grid_reg_worst=float(max(g["max_run_violation"] for g in grid_reg)),
                  grid_reg_mean_worst=float(np.mean([g["max_run_violation"] for g in grid_reg])))
    o = out[b]
    rows.append(f"| {BASES[b]} | {cov:.2f} | {width:.2f} | {main_pooled['fsr']:.2f} / {main_pooled['recall']:.2f} | "
                f"{main_reg['fsr']:.2f} / {main_reg['recall']:.2f} / {main_reg['max_run_violation']:.3f} | "
                f"{o['grid_reg_recall']:.2f} | {o['grid_reg_worst']:.3f} / {o['grid_reg_mean_worst']:.3f} |")
    print(rows[-1], flush=True)

L = ["# So sánh mô hình nền trong quy trình chọn cửa sổ có kiểm soát rủi ro (α = 0.05)", "",
     "| Mô hình nền | Độ phủ dải 10–90 % (danh nghĩa 0.80) | Nửa độ rộng dải roll/pitch, 5 s đầu [°] | CRC gộp: FSR / recall | "
     "CRC theo hướng + ACI: FSR / recall / rủi ro tệ nhất 1 lần chạy | 8 tổ hợp: recall TB | 8 tổ hợp: rủi ro tệ nhất / TB tệ nhất |",
     "|---|---|---|---|---|---|---|"] + rows
with open(os.path.join(RESULTS, "bases_summary.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(RESULTS, "bases_results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
print("\n".join(L))
