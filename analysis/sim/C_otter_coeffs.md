# Kiểm chứng C (bước 1.5): hệ số Otter — MSS-Capytaine so với `otter.m`

Script: `sim/verify_C_otter.m` → `analysis/sim/C_otter_coeffs.csv`. Chạy ngày 26/10/2026.

Ba nguồn được so sánh:
- **BEM (tần số phụ thuộc):** A(ω), B(ω), C từ Capytaine 3.0.0 (hull xấp xỉ của MSS-Capytaine). Tần số riêng là điểm bất động của ω = √(C_kk / (MRB_kk + A_kk(ω))), không ghép đôi; ζ = (B_kk(ω_n) + Bv_kk) / (2√(C_kk (MRB_kk + A_kk))).
- **Mô hình hằng số của bộ dữ liệu:** `build_model` (MSS `computeManeuveringModel`) tại ω₀ = 2π/1.9 = 3.31 rad/s; tần số riêng và ζ theo mode ghép đôi (`modalFromMDG`).
- **`otter.m`** (MSS, `CRAFT/USV/models/otter.m`): M do hàm trả về; G, D lấy bằng sai phân trung tâm của ν̇ tại trạng thái nghỉ; mode ghép đôi bằng `modalFromMDG`. Tải 80 kg tương ứng `mP = 25 kg` đặt tại CO (khớp CG x = 0.1375 m của config `otter_80kg`).

## 1. Otter 55 kg

| DOF | Nguồn | C_kk | A_kk / MRB_kk | ω_n [rad/s] | T_n [s] | ζ |
|---|---|---|---|---|---|---|
| heave | BEM | 6851 N/m | 0.755 | 8.42 | 0.746 | 0.095 (bức xạ) |
| heave | hằng số tại ω₀ = 3.31 | — | 1.880 | 6.68 | 0.941 | 0.154 |
| heave | `otter.m` | 7541 N/m | 1.0 (giả định) | 12.05* | 0.521* | 0.357 |
| roll | BEM | 960 Nm/rad | 1.670 | 5.92 | 1.062 | 0.274 (0.075 bức xạ + 0.198 nhớt) |
| roll | hằng số tại ω₀ | — | 1.626 | 6.06 | 1.037 | 0.246 |
| roll | `otter.m` | 1074 Nm/rad | 0.2 (giả định) | 8.88* | 0.708* | 0.214 |
| pitch | BEM | 1402 Nm/rad | 0.576 | 8.04 | 0.781 | 0.085 (bức xạ) |
| pitch | hằng số tại ω₀ | — | 1.195 | 6.59 | 0.954 | 0.077 |
| pitch | `otter.m` | 2850 Nm/rad | 0.8 (giả định) | 6.80* | 0.924* | 0.366 |

\* mode ghép đôi heave–pitch, quanh gốc CO của `otter.m` (không cùng gốc với BEM, vốn quanh CG).
Thủy tĩnh: `otter.m` GM_T ≈ 1.99 m, GM_L ≈ 5.28 m; MSS-Capytaine GM_T 1.778 m, GM_L 2.350 m, mớn 0.165 m (`otter.m`: 0.134 m).

## 2. Otter 80 kg

| DOF | BEM ω_n (ζ) | hằng số tại ω₀ ω_n (ζ) | `otter.m` ω_n (ζ) |
|---|---|---|---|
| heave | 7.88 (0.070) | 6.23 (0.155) | 11.25 (0.354)* |
| roll | 6.24 (0.275) | 6.29 (0.241) | 8.68 (0.212)* |
| pitch | 8.52 (0.085) | 7.02 (0.078) | 6.30 (0.361)* |

MSS-Capytaine GM_T 1.370 m, GM_L 2.099 m, mớn 0.195 m.

## 3. Hội tụ lưới (Otter 55 kg)

| Lưới | trạm × điểm/mặt cắt | heave ω_n | roll ω_n | pitch ω_n | A44/MRB44 | ζ bức xạ heave |
|---|---|---|---|---|---|---|
| thô | 48 × 8 | 8.428 | 5.907 | 8.045 | 1.680 | 0.097 |
| gốc (dùng cho dữ liệu) | 48 × 12 | 8.424 | 5.917 | 8.044 | 1.670 | 0.095 |
| mịn | 64 × 16 | 8.424 | 5.923 | 8.044 | 1.664 | 0.095 |

Lưới thô → gốc: ω_n lệch < 0.2 %, khối lượng nước kèm ≤ 0.6 %, ζ bức xạ ~2 %. Gốc → mịn: ω_n heave/pitch trùng tới 3 chữ số, roll lệch 0.1 %, A44 lệch 0.4 %, ζ bức xạ trùng. **Lưới gốc đã hội tụ.** Thời gian giải: thô 552 s, gốc 460 s (lúc máy rảnh), mịn 3 770 s. Số trạm không thể giảm dưới 48 vì bảng offset có 48 trạm (Capytaine từ chối).

## 4. Giải thích khác biệt
1. **Hai nguồn đều là ước lượng.** `otter.m` chỉ hiệu chỉnh cản surge/sway/yaw bằng chuỗi thời gian thực nghiệm (revision 2024-02-03, 2026-10-14). Heave/roll/pitch dùng hệ số "best practice": khối lượng nước kèm 1.0 m / 0.2 Ixx / 0.8 Iyy, cản ζ = 0.3 / 0.2 / 0.4 ghi rõ là "specified" (dòng 214–216), thủy tĩnh từ công thức pontoon (Cw = 0.75, Cb = 0.4, I_L nhân 0.8). MSS-Capytaine dùng contour giả định ("assumed contours, not measured Otter lines").
2. **Roll:** BEM cho A44 ≈ 1.67 Ixx vì với tàu hai thân, roll kéo theo heave của từng thân ở cánh tay đòn y = ±0.395 m. Giá trị 0.2 Ixx của `otter.m` là giả định kiểu tàu một thân, nên roll của `otter.m` nhanh hơn (T 0.71 s so với 1.06 s).
3. **Pitch:** độ cứng C55 của BEM bằng khoảng một nửa `otter.m` (GM_L 2.35 m so với ≈ 5.3 m), do công thức I_L của `otter.m` khác hình dạng mặt đường nước của hull Capytaine.
4. **Heave, pitch trong bộ dữ liệu:** mô hình hằng số lấy A_eq tại ω₀ = 3.31 rad/s, nơi khối lượng nước kèm heave lớn gấp 2.5 lần giá trị tại cộng hưởng (1.88 so với 0.755). Vì vậy cộng hưởng heave/pitch trong mô phỏng bị kéo xuống 6.6–6.7 rad/s thay vì 8.0–8.4 rad/s. Đây là hạn chế của cách MSS lấy ma trận hằng; kiểm chứng A2 cho thấy cùng hiện tượng.
5. **Cản heave/pitch** trong mô phỏng chỉ là cản bức xạ (ζ ≈ 0.08–0.15) vì `delta_zeta_345 = [0 0.2 0]`. `otter.m` giả định 0.3–0.4. Không có số đo để chọn; cần quyết định (xem báo cáo tổng hợp).
