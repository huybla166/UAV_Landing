# PhyLSTM2 so với LSTM của bài gốc — dataset_v2, giao thức (a)

Cùng 6 fold theo seed sóng + 2 bài thử ngoài phân phối, cùng gốc dự đoán (mỗi 1 s, đoán 20 s), cùng cách tính RMSE và MAE (`run_sim_lstm_cv.metrics`). Không mô hình nào được biết sóng tương lai khi dự đoán; PhyLSTM2 chỉ dùng lực sóng thật trong hàm phạt vật lý lúc huấn luyện. LSTM của bài: trung bình 3 seed; PhyLSTM: 1 seed. Dòng "6 fold": trung bình ± độ lệch chuẩn giữa các fold. σ = độ lệch chuẩn chuyển động của các lần chạy kiểm tra (đoán bằng trung bình cho RMSE ≈ σ). Đơn vị: heave m, roll và pitch độ. Số liệu: `results/sim/phylstm/phylstm_metrics.npz`, `results/sim/lstm_cv_metrics.npz`; script `experiments/phylstm/report_phylstm.py`.

Hai điểm cần nhớ khi đọc: (1) PhyLSTM (cả hai biến thể) giữ epoch có sai số kiểm định thấp nhất như bài báo, còn LSTM của bài gốc lấy epoch cuối; hàng "epoch cuối" cho phép so sòng phẳng. (2) Phạt vật lý J_g trên 20 s tương lai dùng lực sóng thật, nên một phần chênh lệch PhyLSTM2 so với mạng không vật lý có thể đến từ thông tin thêm lúc huấn luyện chứ không chỉ từ ràng buộc phương trình.

## Heave (m)

| Tập | Mô hình | σ | RMSE | MAE | RMSE 1 s | RMSE 2 s | RMSE 5 s | RMSE 10 s | RMSE 20 s |
|---|---|---|---|---|---|---|---|---|---|
| **6 fold** | LSTM của bài gốc (Wang 2026) | 0.07694 | 0.07561 ± 0.00093 | 0.0526 ± 0.00067 | 0.04587 | 0.06767 | 0.08009 | 0.07813 | 0.07733 |
| **6 fold** | Mạng 1 của PhyLSTM, không vật lý | 0.07694 | 0.07159 ± 0.00094 | 0.04949 ± 0.00073 | 0.03555 | 0.05061 | 0.07146 | 0.07633 | 0.07698 |
| **6 fold** | PhyLSTM2 | 0.07694 | 0.0713 ± 0.00089 | 0.04917 ± 0.00065 | 0.03422 | 0.04929 | 0.07077 | 0.07623 | 0.07696 |
| **6 fold** | Mạng 1 không vật lý, epoch cuối | 0.07694 | 0.07161 ± 0.00093 | 0.0495 ± 0.0007 | 0.03555 | 0.05072 | 0.07152 | 0.07634 | 0.07698 |
| **6 fold** | PhyLSTM2, epoch cuối | 0.07694 | 0.07135 ± 0.00087 | 0.04923 ± 0.00064 | 0.03434 | 0.04937 | 0.07086 | 0.07627 | 0.07698 |
| Ngoài phân phối: Hs = 0,40 m | LSTM của bài gốc (Wang 2026) | 0.09398 | 0.09221 | 0.0724 | 0.05502 | 0.08295 | 0.09838 | 0.09477 | 0.09401 |
| Ngoài phân phối: Hs = 0,40 m | Mạng 1 của PhyLSTM, không vật lý | 0.09398 | 0.08733 | 0.06789 | 0.04245 | 0.06067 | 0.08711 | 0.09316 | 0.09394 |
| Ngoài phân phối: Hs = 0,40 m | PhyLSTM2 | 0.09398 | 0.08695 | 0.06745 | 0.04074 | 0.05901 | 0.08623 | 0.09301 | 0.09392 |
| Ngoài phân phối: Hs = 0,40 m | Mạng 1 không vật lý, epoch cuối | 0.09398 | 0.08734 | 0.06791 | 0.04277 | 0.06074 | 0.08714 | 0.09316 | 0.09396 |
| Ngoài phân phối: Hs = 0,40 m | PhyLSTM2, epoch cuối | 0.09398 | 0.08707 | 0.06758 | 0.04094 | 0.0593 | 0.08646 | 0.09312 | 0.09395 |
| Ngoài phân phối: β = 45° | LSTM của bài gốc (Wang 2026) | 0.07402 | 0.07685 | 0.05413 | 0.04576 | 0.07337 | 0.08252 | 0.07818 | 0.07773 |
| Ngoài phân phối: β = 45° | Mạng 1 của PhyLSTM, không vật lý | 0.07402 | 0.06811 | 0.04647 | 0.02739 | 0.04358 | 0.06729 | 0.07352 | 0.07425 |
| Ngoài phân phối: β = 45° | PhyLSTM2 | 0.07402 | 0.0679 | 0.04629 | 0.0269 | 0.04249 | 0.06697 | 0.0734 | 0.07421 |
| Ngoài phân phối: β = 45° | Mạng 1 không vật lý, epoch cuối | 0.07402 | 0.0681 | 0.04649 | 0.02769 | 0.04367 | 0.06709 | 0.07353 | 0.07428 |
| Ngoài phân phối: β = 45° | PhyLSTM2, epoch cuối | 0.07402 | 0.06792 | 0.04628 | 0.02704 | 0.04243 | 0.06693 | 0.07348 | 0.07421 |

## Roll (độ)

| Tập | Mô hình | σ | RMSE | MAE | RMSE 1 s | RMSE 2 s | RMSE 5 s | RMSE 10 s | RMSE 20 s |
|---|---|---|---|---|---|---|---|---|---|
| **6 fold** | LSTM của bài gốc (Wang 2026) | 2.256 | 2.225 ± 0.058 | 1.616 ± 0.041 | 1.488 | 2.171 | 2.329 | 2.275 | 2.262 |
| **6 fold** | Mạng 1 của PhyLSTM, không vật lý | 2.256 | 2.174 ± 0.061 | 1.575 ± 0.044 | 1.385 | 1.971 | 2.224 | 2.257 | 2.258 |
| **6 fold** | PhyLSTM2 | 2.256 | 2.17 ± 0.062 | 1.571 ± 0.044 | 1.357 | 1.948 | 2.22 | 2.257 | 2.258 |
| **6 fold** | Mạng 1 không vật lý, epoch cuối | 2.256 | 2.174 ± 0.062 | 1.575 ± 0.044 | 1.383 | 1.973 | 2.224 | 2.257 | 2.258 |
| **6 fold** | PhyLSTM2, epoch cuối | 2.256 | 2.171 ± 0.062 | 1.571 ± 0.044 | 1.358 | 1.953 | 2.221 | 2.258 | 2.258 |
| Ngoài phân phối: Hs = 0,40 m | LSTM của bài gốc (Wang 2026) | 2.68 | 2.651 | 2.015 | 1.731 | 2.57 | 2.807 | 2.706 | 2.681 |
| Ngoài phân phối: Hs = 0,40 m | Mạng 1 của PhyLSTM, không vật lý | 2.68 | 2.576 | 1.95 | 1.605 | 2.319 | 2.635 | 2.679 | 2.68 |
| Ngoài phân phối: Hs = 0,40 m | PhyLSTM2 | 2.68 | 2.571 | 1.943 | 1.575 | 2.293 | 2.628 | 2.679 | 2.68 |
| Ngoài phân phối: Hs = 0,40 m | Mạng 1 không vật lý, epoch cuối | 2.68 | 2.576 | 1.949 | 1.602 | 2.32 | 2.634 | 2.679 | 2.68 |
| Ngoài phân phối: Hs = 0,40 m | PhyLSTM2, epoch cuối | 2.68 | 2.572 | 1.944 | 1.576 | 2.296 | 2.629 | 2.679 | 2.68 |
| Ngoài phân phối: β = 45° | LSTM của bài gốc (Wang 2026) | 2.353 | 2.362 | 1.775 | 1.498 | 2.359 | 2.483 | 2.4 | 2.376 |
| Ngoài phân phối: β = 45° | Mạng 1 của PhyLSTM, không vật lý | 2.353 | 2.267 | 1.699 | 1.353 | 2.039 | 2.319 | 2.356 | 2.358 |
| Ngoài phân phối: β = 45° | PhyLSTM2 | 2.353 | 2.267 | 1.699 | 1.376 | 2.047 | 2.314 | 2.356 | 2.357 |
| Ngoài phân phối: β = 45° | Mạng 1 không vật lý, epoch cuối | 2.353 | 2.266 | 1.698 | 1.356 | 2.035 | 2.315 | 2.355 | 2.357 |
| Ngoài phân phối: β = 45° | PhyLSTM2, epoch cuối | 2.353 | 2.264 | 1.696 | 1.365 | 2.024 | 2.311 | 2.356 | 2.357 |

## Pitch (độ)

| Tập | Mô hình | σ | RMSE | MAE | RMSE 1 s | RMSE 2 s | RMSE 5 s | RMSE 10 s | RMSE 20 s |
|---|---|---|---|---|---|---|---|---|---|
| **6 fold** | LSTM của bài gốc (Wang 2026) | 2.778 | 2.736 ± 0.085 | 1.995 ± 0.051 | 1.724 | 2.591 | 2.857 | 2.815 | 2.795 |
| **6 fold** | Mạng 1 của PhyLSTM, không vật lý | 2.778 | 2.619 ± 0.087 | 1.913 ± 0.053 | 1.581 | 2.212 | 2.621 | 2.745 | 2.771 |
| **6 fold** | PhyLSTM2 | 2.778 | 2.609 ± 0.086 | 1.904 ± 0.053 | 1.548 | 2.175 | 2.604 | 2.742 | 2.77 |
| **6 fold** | Mạng 1 không vật lý, epoch cuối | 2.778 | 2.62 ± 0.087 | 1.914 ± 0.053 | 1.586 | 2.215 | 2.623 | 2.746 | 2.771 |
| **6 fold** | PhyLSTM2, epoch cuối | 2.778 | 2.611 ± 0.087 | 1.906 ± 0.054 | 1.548 | 2.179 | 2.608 | 2.745 | 2.771 |
| Ngoài phân phối: Hs = 0,40 m | LSTM của bài gốc (Wang 2026) | 3.269 | 3.215 | 2.432 | 2.01 | 3.045 | 3.362 | 3.311 | 3.279 |
| Ngoài phân phối: Hs = 0,40 m | Mạng 1 của PhyLSTM, không vật lý | 3.269 | 3.071 | 2.325 | 1.824 | 2.563 | 3.048 | 3.231 | 3.264 |
| Ngoài phân phối: Hs = 0,40 m | PhyLSTM2 | 3.269 | 3.057 | 2.312 | 1.736 | 2.506 | 3.029 | 3.226 | 3.264 |
| Ngoài phân phối: Hs = 0,40 m | Mạng 1 không vật lý, epoch cuối | 3.269 | 3.072 | 2.325 | 1.809 | 2.564 | 3.051 | 3.235 | 3.266 |
| Ngoài phân phối: Hs = 0,40 m | PhyLSTM2, epoch cuối | 3.269 | 3.061 | 2.314 | 1.735 | 2.509 | 3.031 | 3.233 | 3.266 |
| Ngoài phân phối: β = 45° | LSTM của bài gốc (Wang 2026) | 3.049 | 3.063 | 2.256 | 2.322 | 3.055 | 3.216 | 3.104 | 3.061 |
| Ngoài phân phối: β = 45° | Mạng 1 của PhyLSTM, không vật lý | 3.049 | 2.871 | 2.114 | 1.674 | 2.375 | 2.861 | 3.011 | 3.04 |
| Ngoài phân phối: β = 45° | PhyLSTM2 | 3.049 | 2.863 | 2.106 | 1.647 | 2.36 | 2.831 | 3.01 | 3.039 |
| Ngoài phân phối: β = 45° | Mạng 1 không vật lý, epoch cuối | 3.049 | 2.876 | 2.119 | 1.724 | 2.356 | 2.865 | 3.017 | 3.042 |
| Ngoài phân phối: β = 45° | PhyLSTM2, epoch cuối | 3.049 | 2.864 | 2.107 | 1.646 | 2.356 | 2.838 | 3.012 | 3.039 |
