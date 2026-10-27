# Kiểm chứng hàm lực sóng `sim/waveLoads.m` (26/10/2026)

Script: `sim/check_fastwave.m`. Dữ liệu: `data/sim/check_fastwave/{ref_msslib,fast_mss,fast_position,fast_pos_cut5}.csv`.
Tàu `otter_55kg` (MSS-Capytaine), Hs 0.15 m, Tp 1.9 s (ω₀ = 3.31 rad/s), sóng ngang β = 90°, U = 1 m/s, seed 7, 600 s, h = 0.05 s, phổ tới 9.50 rad/s.

## 1. Mức hàm (500 bộ (t, U, ψ, β) ngẫu nhiên)
| So sánh | Sai lệch tương đối lớn nhất |
|---|---|
| `waveLoads(...,'mss')` vs MSS `waveForceRAO` | 3.25e-13 |
| `waveLoads(...,'position')` trên quỹ đạo thẳng đều vs MSS | lực 1.86e-11, ζ 4.01e-14 m |

→ Nội suy RAO theo hướng bằng chỉ số và hai công thức pha cài đặt đúng.

## 2. Mức hệ kín, 600 s
| | Thời gian chạy |
|---|---|
| MSS `waveForceRAO` | 276 s |
| `waveLoads` pha `mss` | 118 s |
| `waveLoads` pha `position` | 119 s |

Tăng tốc ×2.3. Hai lần chạy cùng công thức pha MSS (bản MSS và bản `waveLoads('mss')`, khớp tới 1e-13 ở mức hàm) lệch nhau max|Δ|/σ = 5.4 (ζ), 6.0 (heave), 6.4 (roll), 5.6 (pitch), 5.2 (yaw): sai số làm tròn bị khuếch đại thành hai quỹ đạo khác hẳn.

## 3. Nguyên nhân: pha `Ω_e(U, ψ) · t`
MSS (`waveForceRAO.m`, Fossen 2027 eq. 10.96) dùng pha `Ω_e t + ε` với `Ω_e = |ω − (ω²/g) U cos(β − ψ + μ)|` tính lại ở mỗi bước từ U, ψ tức thời. Tần số tức thời của thành phần sóng khi đó là `Ω_e + t · dΩ_e/dt`; số hạng thứ hai tăng tuyến tính theo t khi U hoặc ψ dao động.

Tần số trung bình (theo phổ năng lượng) của ζ tại tàu:

| | 30–300 s | 300–600 s |
|---|---|---|
| MSS `waveForceRAO` | 15.44 rad/s | 28.28 rad/s |
| `waveLoads` pha `position` | 3.85 rad/s | 3.79 rad/s |

Phổ sóng có đỉnh 3.31 rad/s và bị cắt ở 9.50 rad/s, nên giá trị 15–28 rad/s là phi vật lý và tăng dần theo thời gian.

Công thức thay thế (lý thuyết sóng tuyến tính, nước sâu): `phase = ω t − k (N cos χ + E sin χ) + ε`, `k = ω²/g`, `χ = β + μ`. Tần số gặp sóng sinh ra từ quỹ đạo thật của tàu; khi U, ψ không đổi, công thức trùng với MSS (mục 1).

## 4. Phổ bị cắt ở 5 rad/s (mặc định `waveInitialization`)
Độ lệch chuẩn sau 30 s, pha `position`:

| | phổ tới 9.5 rad/s | cắt 5 rad/s | tỉ số |
|---|---|---|---|
| heave [m] | 0.0330 | 0.0327 | 1.01 |
| roll [deg] | 2.1869 | 2.0584 | 1.06 |
| pitch [deg] | 1.6269 | 1.3277 | 1.23 |
| Hs thực [m] | 0.149 | 0.142 | |

Tần số riêng heave/roll/pitch của Otter 55 kg ≈ 0.96/1.05/0.92 s (≈ 6–7 rad/s) nằm trên mức cắt 5 rad/s.

## 5. Bổ sung: pha theo vị trí *tức thời* cũng sai gần cộng hưởng → dùng hệ seakeeping
Phát hiện khi chạy kiểm chứng A (sóng đều, `sim/diag_A_drift.m`): Otter 55 kg, β = 180°, ω = 6 rad/s, U = 0, tắt điều khiển, 60 s.

| Biên độ sóng | Pha tính theo | Trôi surge u (30–60 s) | Tần số heave quan sát | Biên độ / H_const (heave, pitch) | Residual khớp sin |
|---|---|---|---|---|---|
| 1 cm | vị trí tức thời N(t) | 0.039 m/s | 6.137 rad/s | 0.445 / 0.419 | 0.68 |
| 1 cm | hệ seakeeping cố định | 0.028 m/s | 6.001 rad/s | 0.954 / 0.912 | 0.00 |
| 1 mm | vị trí tức thời | 0.0003 m/s | 6.001 rad/s | 0.994 / 0.991 | 0.01 |
| 1 mm | hệ seakeeping cố định | 0.0003 m/s | 6.001 rad/s | 0.993 / 0.990 | 0.00 |

Giải thích: các số hạng Coriolis phi tuyến của mô hình thân cứng (kiểu −m q w) cho lực trung bình ∝ a², tàu trôi chậm; với pha theo vị trí tức thời, độ trôi đó Doppler hóa tần số gặp sóng (k·u = 3.67 × 0.039 = 0.143 rad/s, khớp 0.137 quan sát) và đáp ứng gần cộng hưởng sai một nửa. Ở biên độ 1 mm cả hai cách đều cho ≈ 0.99, nên phần tích phân ODE là đúng.

Sửa: `run_one` tính lực sóng tại **hệ seakeeping** (mô hình hợp nhất của Fossen): vị trí và hướng lấy từ bộ lọc thông thấp bậc 1, T_lp = 20 s, của vận tốc NED và ψ. Dao động tần số sóng (2–9 rad/s) bị loại, chuyển động chậm vẫn được theo. Khi U, ψ không đổi, công thức trùng với MSS (mục 1). Các kết quả A, B, D tính trước khi sửa được lưu ở `analysis/sim/old_instantaneous_phase/` và không dùng.

## Quyết định cho bộ dữ liệu
- `run_one` mặc định: `waveImpl = 'fast'`, `wavePhase = 'position'`, `waveFrame = 'seakeeping'` (T_lp = 20 s), `maxFreq` = tần số RAO hữu hạn lớn nhất.
- Bài báo phải nêu: (i) thay công thức pha của MSS và lý do; (ii) phổ không bị cắt ở 5 rad/s.
- Với pha `position`, ψ của Otter dao động −4.2…3.6° (so với −13…16° khi dùng pha MSS).
