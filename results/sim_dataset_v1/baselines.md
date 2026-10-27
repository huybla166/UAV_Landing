# Baseline trên dữ liệu mô phỏng dataset_v1 (Giai đoạn 3.1)

Giao thức: `experiments/sim_common.py` (5 Hz, gốc dự đoán mỗi 1 s, ngữ cảnh ≤ 60 s, tầm 20 s, bỏ 30 s đầu). Huấn luyện trên tập `train` (1440 lần chạy), chấm điểm trên mọi gốc dự đoán của ba tập kiểm tra. nRMSE = RMSE / độ lệch chuẩn của chính lần chạy đó, rồi lấy trung bình theo lần chạy (1.0 ≈ không tốt hơn đoán giá trị trung bình). Script: `experiments/run_sim_baselines.py`.

Bậc AR chọn theo AIC (heave, roll, pitch): 60, 60, 60.

## Trong phân phối (rep 3) — 576 lần chạy

**Heave**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.003 | 0.999 | 0.999 | 1.000 | 1.000 | 1.001 | 0.0 s |
| Persistence | 1.634 | 1.379 | 1.394 | 1.387 | 1.410 | 1.408 | 0.2 s |
| AR(5) – như bài | 0.919 | 1.035 | 1.005 | 1.000 | 1.000 | 0.994 | 0.4 s |
| AR(p) theo AIC | 0.857 | 1.031 | 0.999 | 1.000 | 1.000 | 0.986 | 1.2 s |
| LSTM của bài (Wang 2026) | 0.680 | 0.952 | 1.070 | 1.036 | 1.034 | 1.004 | 1.6 s |

**Roll**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.001 | 0.999 | 0.999 | 0.999 | 0.999 | 1.001 | 0.0 s |
| Persistence | 1.623 | 1.375 | 1.407 | 1.410 | 1.408 | 1.411 | 0.2 s |
| AR(5) – như bài | 0.898 | 1.030 | 1.001 | 0.999 | 0.999 | 0.988 | 1.0 s |
| AR(p) theo AIC | 0.836 | 1.000 | 1.000 | 0.999 | 0.999 | 0.981 | 1.0 s |
| LSTM của bài (Wang 2026) | 0.703 | 0.988 | 1.064 | 1.025 | 1.020 | 0.997 | 1.4 s |

**Pitch**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.001 | 0.999 | 0.999 | 0.999 | 1.001 | 1.001 | 0.0 s |
| Persistence | 1.556 | 1.414 | 1.388 | 1.396 | 1.408 | 1.408 | 0.2 s |
| AR(5) – như bài | 0.982 | 1.011 | 0.999 | 0.999 | 1.001 | 0.996 | 0.2 s |
| AR(p) theo AIC | 0.950 | 0.993 | 0.997 | 0.999 | 1.001 | 0.990 | 0.4 s |
| LSTM của bài (Wang 2026) | 0.676 | 0.950 | 1.037 | 1.027 | 1.029 | 0.992 | 1.6 s |

## Ngoài phân phối: Hs = 0.40 m — 630 lần chạy

**Heave**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.005 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0 s |
| Persistence | 1.735 | 1.413 | 1.392 | 1.413 | 1.411 | 1.405 | 0.2 s |
| AR(5) – như bài | 0.820 | 1.027 | 1.003 | 1.000 | 1.000 | 0.980 | 1.4 s |
| AR(p) theo AIC | 0.588 | 0.919 | 0.996 | 0.999 | 1.000 | 0.959 | 1.6 s |
| LSTM của bài (Wang 2026) | 0.572 | 0.890 | 1.035 | 1.012 | 1.007 | 0.970 | 2.0 s |

**Roll**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.002 | 0.997 | 1.000 | 1.000 | 0.999 | 1.001 | 0.0 s |
| Persistence | 1.743 | 1.342 | 1.411 | 1.415 | 1.410 | 1.411 | 0.2 s |
| AR(5) – như bài | 0.853 | 1.031 | 1.001 | 1.000 | 0.999 | 0.987 | 1.0 s |
| AR(p) theo AIC | 0.693 | 0.957 | 1.000 | 0.999 | 0.999 | 0.973 | 1.2 s |
| LSTM của bài (Wang 2026) | 0.657 | 0.984 | 1.072 | 1.012 | 1.004 | 0.987 | 1.4 s |

**Pitch**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.003 | 0.999 | 1.000 | 1.001 | 1.001 | 1.002 | 0.0 s |
| Persistence | 1.651 | 1.443 | 1.377 | 1.413 | 1.415 | 1.410 | 0.2 s |
| AR(5) – như bài | 0.956 | 1.016 | 1.000 | 1.001 | 1.001 | 0.993 | 0.4 s |
| AR(p) theo AIC | 0.865 | 0.989 | 0.997 | 1.000 | 1.001 | 0.984 | 1.2 s |
| LSTM của bài (Wang 2026) | 0.644 | 0.945 | 1.027 | 1.020 | 1.016 | 0.983 | 1.6 s |

## Ngoài phân phối: β = 45° — 504 lần chạy

**Heave**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.004 | 0.998 | 1.000 | 1.000 | 1.001 | 0.999 | 0.0 s |
| Persistence | 1.696 | 1.426 | 1.373 | 1.394 | 1.415 | 1.404 | 0.2 s |
| AR(5) – như bài | 0.880 | 1.047 | 1.001 | 1.000 | 1.001 | 0.984 | 1.2 s |
| AR(p) theo AIC | 0.766 | 1.080 | 0.999 | 0.999 | 1.001 | 0.980 | 1.4 s |
| LSTM của bài (Wang 2026) | 0.703 | 1.042 | 1.093 | 1.044 | 1.066 | 1.026 | 1.6 s |

**Roll**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.002 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 | 0.0 s |
| Persistence | 1.695 | 1.380 | 1.410 | 1.416 | 1.418 | 1.410 | 0.2 s |
| AR(5) – như bài | 0.856 | 1.033 | 1.001 | 1.000 | 1.000 | 0.985 | 1.0 s |
| AR(p) theo AIC | 0.736 | 0.986 | 1.000 | 1.000 | 1.000 | 0.975 | 1.2 s |
| LSTM của bài (Wang 2026) | 0.660 | 0.986 | 1.041 | 1.014 | 1.011 | 0.986 | 1.2 s |

**Pitch**

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |
|---|---|---|---|---|---|---|---|
| Trung bình 60 s | 1.002 | 0.998 | 0.998 | 0.999 | 1.000 | 1.001 | 0.0 s |
| Persistence | 1.585 | 1.455 | 1.323 | 1.371 | 1.408 | 1.400 | 0.2 s |
| AR(5) – như bài | 0.976 | 1.017 | 0.998 | 0.999 | 1.000 | 0.992 | 0.4 s |
| AR(p) theo AIC | 0.904 | 0.982 | 0.989 | 0.997 | 0.999 | 0.983 | 0.4 s |
| LSTM của bài (Wang 2026) | 0.717 | 0.961 | 1.031 | 1.024 | 1.026 | 0.993 | 1.4 s |

## Theo điều kiện (tập trong phân phối): nRMSE trung bình 0–20 s, trung bình 3 kênh

**AR(p) theo AIC** (hàng: β, cột: U m/s)

| β \ U | 0 | 1 | 2 |
|---|---|---|---|
| 0° | 0.977 | 0.961 | 1.031 |
| 90° | 0.978 | 0.975 | 0.984 |
| 135° | 0.978 | 0.986 | 0.993 |
| 180° | 0.977 | 0.989 | 0.998 |

**LSTM của bài (Wang 2026)** (hàng: β, cột: U m/s)

| β \ U | 0 | 1 | 2 |
|---|---|---|---|
| 0° | 1.001 | 1.007 | 1.105 |
| 90° | 0.980 | 0.979 | 0.997 |
| 135° | 0.970 | 0.979 | 0.988 |
| 180° | 0.988 | 0.983 | 0.992 |


## So với Wang et al. 2026 (JFI 363:108381), cùng kiến trúc LSTM

| | Bài báo `[W p.12–14]` | Mô phỏng này, tập trong phân phối |
|---|---|---|
| Dữ liệu | 1 chuỗi 500 s, 5 Hz; sóng Hs 3.2 m, chu kỳ TB 8.7 s; tàu và phương trình không nêu | 3150 lần chạy Otter 2 m, Tp 1.1–3.5 s; 576 lần kiểm tra độc lập |
| Chia tập | 70 % đầu / 30 % cuối của cùng một chuỗi | các lần chạy (seed sóng) khác nhau |
| Tầm của RMSE | không nêu | 0.2 s (1 bước) và 1–20 s |
| LSTM, roll / pitch | 0.0371° / 0.0264° | 0.345° / 0.502° ở 0.2 s; 1.40° / 1.62° ở 1 s |
| AR(5), roll / pitch | 0.0448° / 0.0450° | 0.589° / 1.198° ở 0.2 s |
| LSTM tốt hơn AR(5) | 17 % / 41 % | 42 % / 58 % ở 0.2 s |
| Tầm hữu ích (nRMSE < 0.9) | không đo | 1.4 s (roll), 1.6 s (pitch) |

RMSE tuyệt đối ở đây là trung bình bình phương theo lần chạy của nRMSE × độ lệch chuẩn từng lần chạy
(độ lệch chuẩn trung vị: roll 1.66°, pitch 2.21°). Trên dữ liệu thật NTUA, cùng LSTM có tầm hữu ích
1.4 / 0.6 / 0.8 s (heave / roll / pitch) `[results/final_summary.md]`.
