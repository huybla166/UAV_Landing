# Kiểm chứng bộ dữ liệu mô phỏng (dataset_v1 → dataset_v2)

Ngày 27/10/2026. Câu hỏi: dữ liệu sinh bằng Octave/GPU có đúng không, và có đủ để viết bài Q1 không.
Mọi con số dưới đây có file nguồn ghi kèm; script chính: `analysis/sim/verify_dataset_v1.py`
(kết quả trong `analysis/sim/V/`), `experiments/pinn/check_residual.py` (phiên khác viết), `sim/verify_E_memory.m`.

## 1. Danh sách kiểm tra

| Mã | Kiểm tra gì | Cách | Kết quả v1 |
|---|---|---|---|
| V1 | File nguyên vẹn, khớp ma trận thiết kế, động học η̇ = J(η)ν | đọc cả 3150 CSV + JSON, sai phân bậc 4 | **đạt** 3150/3150; phần dư động học ≤ 0.3 % (heave), 1.0 % (roll), 0.6 % (pitch), 2.1 % (yaw) |
| V2 | GPU = Octave | 40 lần chạy chung | **đạt**: 35/40 trùng tới 7 chữ số; 5 lần (2 m/s, sóng theo) tách ra do hỗn loạn, Octave-so-với-Octave cũng tách y hệt |
| V3 | Lực sóng và cao độ sóng trong file đúng công thức | viết lại độc lập bằng numpy từ mô hình sóng + hệ seakeeping dựng lại từ log, so từng mẫu (`V/force_reconstruction.csv`) | **đạt** 343 lần (cả Octave lẫn GPU, mọi U): max \|dựng lại − file\|/σ = 5.5e-5, trung vị 3e-6 (mức làm tròn 7 chữ số của CSV) |
| V4 | Dữ liệu thỏa phương trình đầy đủ với hệ số trong JSON | phần dư RMS / độ lệch chuẩn lực sóng, sai phân 7 điểm, bộ nhớ dựng lại từ ν | **đạt** (169 lần Octave): trung vị 0.01 / 0.16 / 0.05 / 0.66 % (heave/roll/pitch/yaw), 95 % ≤ 1.4 %, tối đa 5.8 % pitch ở 1 lần sóng ngược 2 m/s Tp 1.14 s; phần còn lại đang chạy |
| V5 | Mô hình bộ nhớ Cummins thụ động và hệ tuyến tính ổn định | phần thực K_fit(jω) trên 0–60 rad/s; trị riêng hệ [η; ν; x_r] | **KHÔNG ĐẠT** (xem mục 2) |
| V6 | Hội tụ bước thời gian; độ nhạy điều kiện đầu | chạy lại h = 0.025, 0.0125; cặp hàm động lực học khác nhau 1e-14 | **một phần** (xem mục 2) |
| V7 | Vùng vận hành và lực điều khiển | surge, yaw, τ so với giới hạn thrust Otter (otter.m) | τ trong giới hạn ở 3150/3150 (τX ∈ [−25, 40] N so với [−133, 239]; max τN 73.0 so với 73.6 N·m); 700 lần "dữ dội" (xem mục 3) |
| V8 | Độ lệch chuẩn heave/roll/pitch so với lý thuyết tuyến tính (bộ nhớ đã khớp, tần số gặp sóng) | 3150 lần | heave 0.98, roll 1.00 (κ = 0) / 0.81 (κ = 5), pitch 0.96; **pitch sóng ngược khi chạy 0.63–0.69** → cùng gốc với V5 |

## 2. Hai lỗi mô hình tìm thấy và cách sửa

### 2a. Khớp FDI của bộ nhớ chất lỏng không thụ động ngoài dải sóng (`analysis/sim/V/memory_passivity.md`)

- BEM chỉ có tới 9.5 rad/s. Phép khớp hữu tỉ (MSS `fit_siso_fresp`) tự do ngoài dải và cho Re K₅₅(jω) < 0 ở 13–30 rad/s: −9 % (55 kg) và **−40 %** (80 kg) của max B₅₅.
- Với 80 kg, cản cơ bản: hệ tuyến tính **không ổn định**, trị riêng +0.0257 ± 13.06j (2.08 Hz, gấp e mỗi 39 s); |H₅₅| có đỉnh giả 6.8× ở >9.5 rad/s.
- Trong dữ liệu v1, cản phi tuyến (cross-flow, κ) giữ mode này ở biên độ không đo được: phương sai pitch trong 11–16 rad/s chỉ 0.01–0.03 % tổng. Nhưng (i) phương trình tuyến tính trong JSON mà PINN dùng là sai ở điểm này, (ii) ở 1–2 m/s sóng ngược, tần số gặp sóng lên tới 28 rad/s nên lực sóng kích thích đúng vùng này; dự đoán tuyến tính v1 đặt 59–71 % phương sai pitch ở >9.5 rad/s, còn mô phỏng cho ít hơn nhiều (V8: 0.63–0.69).
- **Sửa (khớp v3, `sim/build_memory.m`)**: chỉ nhận bậc khớp có Re K_ii ≥ 0 trên 0–60 rad/s và |K_ij| ngoài dải không vượt 1.1× max trong dải; kiểm tra thụ động ma trận và ổn định toàn hệ sau khi ghép. Kết quả: hai tàu ổn định (Re λ max −0.083, −0.038), không còn đỉnh giả (|H_kk| ngoài dải ≤ 1.04× giá trị tại 9.5 rad/s), R² trong dải ≥ 0.98 trừ K₁₁ 80 kg (0.96) và K₂₆ (0.95). Còn lại: ma trận chưa hoàn toàn thụ động ở các cặp chéo (55 kg: −6.4 ở 10.8 rad/s, −0.2 % thang B; 80 kg: −28.7 ở 8.1 rad/s trong không gian sway–roll–yaw, −3.6 % B roll). Thử neo điểm tần số cao vào phép khớp thì hỏng R² trong dải (bị loại).

### 2b. Bước thời gian 0.05 s chưa hội tụ ở sóng ngắn gặp ở tốc độ cao (`scratchpad v6`, bảng trong mục 5)

- Trường hợp hiền (U = 0, Hs 0.3): h = 0.025 so với 0.05 lệch ≤ 4.5 % σ theo điểm, σ trùng 4 chữ số.
- Sóng ngược 2 m/s Tp 1.14 s (80 kg): h = 0.05 cho σ pitch thấp 2 % và quỹ đạo lệch 1–2 σ; h = 0.025 so với 0.0125 lệch ≤ 14 % theo điểm, σ trùng. Nguyên nhân: ω_e·h = 28 × 0.05 = 1.4 vượt vùng chính xác của RK4.
- **Sửa**: dataset_v2 tích phân với h = 0.025 s, ghi 20 Hz (`sim/run_one.m` `h_log`, `sim/batch_sim_gpu.py --h 0.025`). GPU và Octave ở h = 0.025 trùng nhau tới 7 chữ số (kiểm tra 60 s).

## 3. Vùng "cưỡi sóng" (không phải lỗi, phải ghi rõ)

Ở U = 2 m/s với β = 0°, 45° (và một phần 90°), tỉ số U/c_p = 0.4–1.1 (c_p = gTp/2π là tốc độ pha sóng): tàu chạy ngang tốc độ sóng. Surge dao động tới ±1.7 m/s (trung vị) và yaw lệch chuẩn ~10° dù bộ lái vẫn trong giới hạn thrust. Mô hình chỉ có lực sóng bậc 1 nên đây là bản tuyến tính của hiện tượng surf-riding, không phải mô phỏng đúng của nó. Quỹ đạo ở đây **hỗn loạn**: sai số 1e-14 tăng gấp e mỗi 19–22 s (ba lần chạy, `scratchpad v6`), nên hai bộ máy (hoặc hai bước thời gian) cho quỹ đạo khác nhau sau ~150 s nhưng cùng thống kê (±10 %). Với tầm dự đoán 20 s, sai số chỉ tăng 2.7× — vẫn dự đoán được về nguyên tắc.

Đề nghị cho bài báo: cột `regime` trong manifest (`violent` = yaw std > 5° hoặc |u − U| max > 1 m/s: 700/3150 lần, hầu hết U = 2, β ≤ 90°); kết quả chính trên phần còn lại, phần "dữ dội" phân tích riêng như giới hạn dự đoán nội tại.

## 4. Kết luận về độ đúng

- **Bộ máy** (Octave và GPU) tích phân đúng phương trình đã công bố: V1, V2, V4 đạt. Không có lỗi lập trình nào tìm thấy.
- **Mô hình** có hai điểm yếu (2a, 2b) ảnh hưởng chủ yếu tới pitch ở sóng ngược khi chạy 1–2 m/s và tới phương trình tuyến tính trong JSON. dataset_v1 vẫn dùng được cho heave/roll và cho mọi trường hợp U = 0, nhưng **dataset_v2** (khớp v3, h = 0.025) thay thế nó làm bộ chính.

## 5. Kết quả trên dataset_v2

*(cập nhật sau khi sinh xong)*

## 6. Có đủ cho bài Q1 không

*(xem phần cuối, viết sau khi có v2)*
