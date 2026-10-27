# Kết quả dự đoán trên dữ liệu NTUA (leave-one-run-out, 4 lần chạy)

nRMSE = RMSE / độ lệch chuẩn của chuỗi thật (1.0 ≈ không tốt hơn đoán giá trị trung bình). Càng nhỏ càng tốt.

## Heave

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s |
|---|---|---|---|---|---|---|
| Trung bình (không làm gì) | 0.956 | 0.952 | 0.951 | 0.957 | 0.950 | 0.961 |
| Persistence | 1.661 | 1.363 | 1.293 | 1.377 | 1.330 | 1.346 |
| AR(5) – như bài | 0.793 | 1.018 | 0.946 | 0.957 | 0.950 | 0.938 |
| AR(p) chọn theo AIC | 0.417 | 0.843 | 0.932 | 0.957 | 0.961 | 0.915 |
| LSTM của bài (10 bước) | 0.785 | 1.081 | 1.058 | 0.995 | 0.994 | 0.990 |
| LSTM của bài + chuẩn hóa cửa sổ | 0.760 | 1.063 | 1.061 | 1.020 | 1.031 | 1.009 |
| TiRex zero-shot (30 s) | 0.643 | 0.890 | 0.960 | 0.987 | 0.957 | 0.936 |
| TiRex zero-shot (60 s) | 0.541 | 0.861 | 0.945 | 0.969 | 0.969 | 0.932 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | 0.541 | 0.861 | 0.945 | 0.970 | 0.971 | 0.933 |
| TiRex tinh chỉnh đầu ra | 0.424 | 0.812 | 0.906 | 0.952 | 0.952 | 0.904 |
| K²VAE (ICML 2025), cấu hình gốc | 0.978 | 0.980 | 0.962 | 1.029 | 0.981 | 0.993 |
| K²VAE thu nhỏ | 0.968 | 0.956 | 0.964 | 0.968 | 0.969 | 0.975 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | 0.964 | 0.952 | 0.972 | 0.977 | 0.974 | 0.976 |

## Roll

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s |
|---|---|---|---|---|---|---|
| Trung bình (không làm gì) | 0.932 | 0.924 | 0.934 | 0.948 | 0.990 | 0.961 |
| Persistence | 1.579 | 1.473 | 1.252 | 1.295 | 1.315 | 1.324 |
| AR(5) – như bài | 0.644 | 0.842 | 0.960 | 0.955 | 0.991 | 0.929 |
| AR(p) chọn theo AIC | 0.352 | 0.701 | 0.954 | 0.964 | 0.999 | 0.905 |
| LSTM của bài (10 bước) | 1.106 | 1.202 | 1.281 | 1.319 | 1.362 | 1.287 |
| LSTM của bài + chuẩn hóa cửa sổ | 0.651 | 0.904 | 1.016 | 0.949 | 0.988 | 0.945 |
| TiRex zero-shot (30 s) | 0.521 | 0.718 | 0.912 | 0.954 | 1.007 | 0.911 |
| TiRex zero-shot (60 s) | 0.438 | 0.692 | 0.911 | 0.963 | 1.024 | 0.916 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | 0.438 | 0.692 | 0.911 | 0.964 | 1.026 | 0.917 |
| TiRex tinh chỉnh đầu ra | 0.354 | 0.662 | 0.896 | 0.941 | 0.989 | 0.888 |
| K²VAE (ICML 2025), cấu hình gốc | 0.935 | 0.933 | 0.959 | 0.988 | 1.023 | 0.981 |
| K²VAE thu nhỏ | 0.937 | 0.937 | 0.946 | 0.957 | 1.001 | 0.974 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | 0.951 | 0.942 | 0.947 | 0.959 | 1.002 | 0.974 |

## Pitch

| Mô hình | 1 s | 2 s | 5 s | 10 s | 20 s | TB 0–20 s |
|---|---|---|---|---|---|---|
| Trung bình (không làm gì) | 1.023 | 1.019 | 1.017 | 1.021 | 1.006 | 0.993 |
| Persistence | 1.755 | 1.476 | 1.363 | 1.442 | 1.409 | 1.413 |
| AR(5) – như bài | 0.963 | 1.154 | 1.016 | 1.021 | 1.006 | 0.981 |
| AR(p) chọn theo AIC | 0.516 | 0.947 | 0.988 | 1.003 | 1.009 | 0.936 |
| LSTM của bài (10 bước) | 0.905 | 1.124 | 1.205 | 1.192 | 1.167 | 1.132 |
| LSTM của bài + chuẩn hóa cửa sổ | 0.780 | 1.082 | 1.132 | 1.104 | 1.057 | 1.043 |
| TiRex zero-shot (30 s) | 0.702 | 0.937 | 1.012 | 1.057 | 1.023 | 0.975 |
| TiRex zero-shot (60 s) | 0.599 | 0.923 | 1.001 | 1.046 | 1.029 | 0.962 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | 0.599 | 0.923 | 1.001 | 1.047 | 1.031 | 0.963 |
| TiRex tinh chỉnh đầu ra | 0.491 | 0.885 | 0.982 | 1.014 | 1.010 | 0.936 |
| K²VAE (ICML 2025), cấu hình gốc | 1.052 | 1.038 | 1.052 | 1.056 | 1.051 | 1.032 |
| K²VAE thu nhỏ | 1.059 | 1.048 | 1.037 | 1.040 | 1.040 | 1.016 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | 1.051 | 1.039 | 1.040 | 1.042 | 1.049 | 1.017 |

## Độ phủ khoảng 10–90 % của TiRex (lý tưởng = 0.80), chưa hiệu chỉnh conformal

| Mô hình | Kênh | 1 s | 2 s | 5 s | 10 s | 20 s |
|---|---|---|---|---|---|---|
| TiRex zero-shot (30 s) | Heave | 0.84 | 0.75 | 0.79 | 0.81 | 0.83 |
| TiRex zero-shot (30 s) | Roll | 0.84 | 0.77 | 0.76 | 0.77 | 0.78 |
| TiRex zero-shot (30 s) | Pitch | 0.85 | 0.76 | 0.78 | 0.81 | 0.84 |
| TiRex zero-shot (60 s) | Heave | 0.84 | 0.72 | 0.80 | 0.80 | 0.82 |
| TiRex zero-shot (60 s) | Roll | 0.86 | 0.72 | 0.76 | 0.73 | 0.72 |
| TiRex zero-shot (60 s) | Pitch | 0.85 | 0.73 | 0.78 | 0.80 | 0.80 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | Heave | 0.84 | 0.72 | 0.80 | 0.80 | 0.82 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | Roll | 0.86 | 0.72 | 0.76 | 0.73 | 0.72 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | Pitch | 0.85 | 0.73 | 0.78 | 0.80 | 0.81 |
| TiRex tinh chỉnh đầu ra | Heave | 0.76 | 0.78 | 0.79 | 0.79 | 0.81 |
| TiRex tinh chỉnh đầu ra | Roll | 0.78 | 0.79 | 0.78 | 0.78 | 0.77 |
| TiRex tinh chỉnh đầu ra | Pitch | 0.75 | 0.80 | 0.79 | 0.78 | 0.80 |
| K²VAE (ICML 2025), cấu hình gốc | Heave | 0.75 | 0.77 | 0.80 | 0.76 | 0.77 |
| K²VAE (ICML 2025), cấu hình gốc | Roll | 0.76 | 0.76 | 0.76 | 0.74 | 0.77 |
| K²VAE (ICML 2025), cấu hình gốc | Pitch | 0.80 | 0.80 | 0.81 | 0.83 | 0.80 |
| K²VAE thu nhỏ | Heave | 0.79 | 0.82 | 0.82 | 0.78 | 0.82 |
| K²VAE thu nhỏ | Roll | 0.77 | 0.78 | 0.77 | 0.78 | 0.75 |
| K²VAE thu nhỏ | Pitch | 0.80 | 0.84 | 0.82 | 0.83 | 0.83 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | Heave | 0.79 | 0.81 | 0.81 | 0.76 | 0.81 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | Roll | 0.76 | 0.78 | 0.77 | 0.77 | 0.76 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | Pitch | 0.80 | 0.83 | 0.81 | 0.82 | 0.82 |

## Tầm dự đoán hữu ích: thời điểm đầu tiên nRMSE ≥ 0.9 (mô hình không còn tốt hơn đoán trung bình 10 %)

| Mô hình | Heave | Roll | Pitch |
|---|---|---|---|
| Trung bình (không làm gì) | 0.0 s | 0.0 s | 0.0 s |
| Persistence | 0.2 s | 0.4 s | 0.2 s |
| AR(5) – như bài | 1.6 s | 2.0 s | 0.6 s |
| AR(p) chọn theo AIC | 2.6 s | 3.2 s | 1.4 s |
| LSTM của bài (10 bước) | 1.4 s | 0.6 s | 0.8 s |
| LSTM của bài + chuẩn hóa cửa sổ | 1.4 s | 1.8 s | 1.2 s |
| TiRex zero-shot (30 s) | 2.0 s | 3.8 s | 1.8 s |
| TiRex zero-shot (60 s) | 2.0 s | 4.0 s | 1.8 s |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | 2.0 s | 4.0 s | 1.8 s |
| TiRex tinh chỉnh đầu ra | 3.0 s | 4.2 s | 2.0 s |
| K²VAE (ICML 2025), cấu hình gốc | 0.0 s | 0.2 s | 0.0 s |
| K²VAE thu nhỏ | 0.0 s | 0.0 s | 0.0 s |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | 0.0 s | 0.0 s | 0.0 s |

## nRMSE trung bình 0–20 s theo từng lần chạy (trung bình 3 kênh)

| Mô hình | noWing_headQ | noWing_followQ | wing_headQ | wing_followQ |
|---|---|---|---|---|
| Trung bình (không làm gì) | 1.000 | 0.875 | 1.017 | 0.994 |
| Persistence | 1.385 | 1.234 | 1.435 | 1.391 |
| AR(5) – như bài | 0.975 | 0.845 | 1.002 | 0.975 |
| AR(p) chọn theo AIC | 0.962 | 0.828 | 0.956 | 0.930 |
| LSTM của bài (10 bước) | 1.027 | 1.275 | 1.201 | 1.042 |
| LSTM của bài + chuẩn hóa cửa sổ | 0.988 | 0.990 | 1.024 | 0.994 |
| TiRex zero-shot (30 s) | 0.976 | 0.846 | 0.972 | 0.969 |
| TiRex zero-shot (60 s) | 0.972 | 0.849 | 0.952 | 0.975 |
| TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh) | 0.972 | 0.850 | 0.953 | 0.975 |
| TiRex tinh chỉnh đầu ra | 0.946 | 0.824 | 0.930 | 0.938 |
| K²VAE (ICML 2025), cấu hình gốc | 1.023 | 0.928 | 1.048 | 1.009 |
| K²VAE thu nhỏ | 1.014 | 0.892 | 1.034 | 1.013 |
| K²VAE thu nhỏ, dừng sớm theo 5 s đầu | 1.014 | 0.892 | 1.035 | 1.014 |
