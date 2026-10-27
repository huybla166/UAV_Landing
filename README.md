# UAV_Landing: dự đoán chuyển động boong USV để drone hạ cánh an toàn

Khi drone hạ cánh lên một tàu mặt nước không người lái (USV) nhỏ, boong tàu luôn chao lắc theo sóng. Dự án này
dự đoán **heave** (nhấp nhô), **roll** (lắc ngang) và **pitch** (lắc dọc) của boong trong vài giây tới, rồi chọn
**cửa sổ hạ cánh**: khoảng thời gian boong đủ êm để hạ, kèm mức rủi ro.

Dự án gồm ba phần:

1. **Mô phỏng**: tạo dữ liệu chuyển động USV trong sóng ngẫu nhiên bằng MSS (Fossen) với tàu Otter.
2. **Dự đoán**: so sánh các mô hình AR, LSTM, TiRex, K²VAE; đang phát triển mô hình lai vật lý (PINN).
3. **Quyết định**: hiệu chỉnh conformal để biết lúc nào hạ cánh an toàn với mức rủi ro cho trước.

> Dự án đang trong quá trình nghiên cứu nên code và kết quả còn thay đổi.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| [`sim/`](sim/) | Mô phỏng sóng → lực sóng → chuyển động 6 bậc tự do (Octave + MSS 2.0, có bản chạy GPU bằng PyTorch). Hướng dẫn chi tiết: [sim/README.md](sim/README.md) |
| [`experiments/`](experiments/) | Mô hình dự đoán, thí nghiệm cửa sổ hạ cánh. `experiments/pinn/` chứa phương trình chuyển động dạng khả vi cho PINN |
| [`analysis/`](analysis/) | Kiểm chứng bộ mô phỏng và hình phân tích |
| [`results/`](results/) | Tóm tắt kết quả (`*.md`, `*.json`) và log chạy. Bắt đầu đọc từ [results/summary.md](results/summary.md) |
| [`Trinh_bay/`](Trinh_bay/) | Hình dùng khi trình bày |

## Dữ liệu

Dữ liệu **không có trong repo** vì quá lớn (khoảng 19 GB). Code đọc dữ liệu từ thư mục `data/`:

| Bộ dữ liệu | Mô tả | Đặt ở |
|---|---|---|
| **dataset_v2** (mô phỏng) | 3150 lần chạy, mỗi lần 600 s, ghi 20 Hz. Tàu Otter 55 kg và 80 kg; Hs 0,10–0,50 m; hướng sóng 0–180°; tốc độ 0–2 m/s. Mỗi lần chạy có 1 file CSV chuyển động và 1 file JSON hệ số phương trình chuyển động | `data/sim/dataset_v2/` |
| **NTUA/UiT** (đo thật) | Thử nghiệm ngoài biển với mô hình tàu tự hành, công khai tại DataverseNO, [DOI 10.18710/VXGPII](https://doi.org/10.18710/VXGPII). Xử lý bằng `analysis/analyze_vxgpii.py` | `data/processed/` |

Cần dataset_v2 thì liên hệ tác giả. Có thể tự sinh lại bằng code trong `sim/`, nhưng mất nhiều giờ chạy.

## Cài đặt

**Mô phỏng:** GNU Octave (hoặc MATLAB) và [MSS 2.0](https://github.com/cybergalactic/MSS). Hệ số thủy động của
tàu Otter tính bằng MSS-Capytaine. Xem [sim/README.md](sim/README.md).

**Dự đoán (Python 3.11):**

```bash
pip install numpy scipy pandas matplotlib torch einops tirex-ts pytest
```

## Chạy thử

```bash
python sim/make_design_matrix.py               # tạo ma trận thí nghiệm sim/design_matrix.csv
python experiments/run_sim_baselines.py        # mô hình cơ sở (mean, AR, LSTM) trên dữ liệu mô phỏng
python experiments/run_baselines.py            # mô hình cơ sở trên dữ liệu thật NTUA
python experiments/run_conformal_windows.py    # cửa sổ hạ cánh có hiệu chỉnh conformal
pytest experiments/pinn                        # kiểm tra phương trình chuyển động của PINN
```

## Giấy phép

Chưa chọn giấy phép. Riêng `experiments/k2vae/` được chép từ
[K2VAE](https://github.com/decisionintelligence/K2VAE) theo giấy phép MIT (xem `experiments/k2vae/LICENSE`).

## Tác giả

Phạm Quang Huy
