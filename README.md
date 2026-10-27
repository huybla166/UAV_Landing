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
| [`analysis/`](analysis/) | Xử lý dữ liệu thật NTUA, kiểm chứng bộ mô phỏng, hình phân tích |
| [`results/`](results/) | Tóm tắt kết quả (`*.md`, `*.json`) và log chạy. Bắt đầu đọc từ [results/summary.md](results/summary.md) |
| [`Trinh_bay/`](Trinh_bay/) | Hình dùng khi trình bày |

## Dữ liệu

Dữ liệu **không có trong repo** vì quá lớn (gần 20 GB). Code đọc dữ liệu từ thư mục `data/`:

| Bộ dữ liệu | Mô tả | Đặt ở |
|---|---|---|
| **dataset_v2** (mô phỏng) | 3150 lần chạy, mỗi lần 600 s, ghi 20 Hz. Tàu Otter 55 kg và 80 kg; Hs 0,10–0,50 m; hướng sóng 0–180°; tốc độ 0–2 m/s. Mỗi lần chạy có 1 file CSV chuyển động và 1 file JSON hệ số phương trình chuyển động | `data/sim/dataset_v2/` |
| **dataset_v1** (mô phỏng, bản cũ) | Cùng ma trận thí nghiệm với v2, chỉ dùng cho test PINN | `data/sim/dataset_v1/` |
| **NTUA/UiT** (đo thật) | Thử nghiệm ngoài biển gần Athens (Hy Lạp) với mô hình tàu dài 10–12 m, công khai tại DataverseNO: [DOI 10.18710/VXGPII](https://doi.org/10.18710/VXGPII) | Tải về và giải nén vào `data/VXGPII/` (có hai thư mục `with_dynamic_wing/`, `without_dynamic_wing/`) |

Cần dataset_v1/v2 thì liên hệ tác giả. Có thể tự sinh lại bằng code trong `sim/` (bắt đầu từ
`sim/make_design_matrix.py`), nhưng mất nhiều giờ chạy.

## Cài đặt

**Mô phỏng:** GNU Octave (hoặc MATLAB) và [MSS 2.0](https://github.com/cybergalactic/MSS). Hệ số thủy động của
tàu Otter tính bằng MSS-Capytaine. Xem [sim/README.md](sim/README.md).

**Dự đoán (Python 3.11):**

```bash
pip install numpy scipy pandas matplotlib torch einops tirex-ts pytest
```

## Chạy thử

Chạy theo đúng thứ tự, vì bước sau dùng kết quả của bước trước. Kết quả thí nghiệm ghi vào `results/`.

**Dữ liệu thật NTUA** (cần `data/VXGPII/`):

```bash
python analysis/analyze_vxgpii.py              # xử lý dữ liệu thô -> data/processed/
python experiments/run_baselines.py            # mô hình cơ sở: mean, persistence, AR, LSTM của bài gốc
python experiments/run_tirex.py                # TiRex zero-shot (lần đầu tự tải mô hình NX-AI/TiRex)
python experiments/run_conformal_windows.py    # cửa sổ hạ cánh có hiệu chỉnh conformal
```

**Dữ liệu mô phỏng** (cần `data/sim/dataset_v2/`):

```bash
python experiments/sim_common.py               # tạo cache 5 Hz, chỉ cần chạy 1 lần
python experiments/run_sim_baselines.py        # mô hình cơ sở trên dữ liệu mô phỏng
```

**Test phương trình chuyển động của PINN** (cần `data/sim/dataset_v1/`, thiếu thì test tự bỏ qua):

```bash
pytest experiments/pinn
```

## Giấy phép

Chưa chọn giấy phép. Riêng `experiments/k2vae/` được chép từ
[K2VAE](https://github.com/decisionintelligence/K2VAE) theo giấy phép MIT (xem `experiments/k2vae/LICENSE`).

## Tác giả

Phạm Quang Huy
