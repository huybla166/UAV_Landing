# Hướng dẫn tạo lại dataset_v2 từng bước

Làm đủ 7 bước dưới đây trên một máy Windows có GPU NVIDIA sẽ tạo lại bộ dữ liệu `dataset_v2`: 3150 lần mô phỏng × 600 s, tàu Otter 55/80 kg. Tổng thời gian khoảng 1.5 giờ, chủ yếu là bước 5. Mọi lệnh, phiên bản và con số trong file này đã được chạy hoặc đo ngày 27/10/2026 trên máy gốc.

Mức "giống" đạt được (đã thử, chi tiết ở cuối file): sóng, hệ số tàu và file JSON giống hoàn toàn; chuyển động của 2608/3150 lần chạy giống tới khoảng 8 chữ số; 542 lần chạy ở vùng "cưỡi sóng" giống vài phút đầu rồi chỉ giống về thống kê. Ai cần đúng từng con số của bản gốc thì dùng file gốc.

## Cần chuẩn bị

| Thứ | Phiên bản đã dùng |
| --- | --- |
| Windows + Git Bash | Windows 11 (mọi lệnh dưới đây gõ trong Git Bash) |
| GPU NVIDIA có CUDA | RTX 3060 Laptop 6 GB, driver 595.79 |
| RAM | 16 GB |
| Ổ đĩa trống | khoảng 10 GB |
| Code và đầu vào | repo https://github.com/huybla166/UAV_Landing (đủ mọi thứ để tạo dữ liệu; không cần bộ dữ liệu gốc) |

Clone repo vào `D:/USV-landing-prediction`; đặt ở chỗ khác thì thay đường dẫn này trong mọi lệnh:

```bash
git clone https://github.com/huybla166/UAV_Landing.git D:/USV-landing-prediction
```

Thư mục `sim/reproduce_v2/` (chứa file này) có đúng hai file hệ số thủy động và bốn file bộ nhớ chất lỏng đã dùng (`inputs/`), mã băm của 3150 file dữ liệu gốc (`checksums_sha256.txt`) và mã băm của code (`code_sha256.txt`).

## Bước 1. Cài phần mềm

1. **GNU Octave 11.3.0**, bản Windows, giải nén vào `D:/tools/octave`. Bản này có sẵn hai gói `control` 4.2.1 và `signal` 1.4.7 mà code cần; kiểm tra bằng lệnh `pkg list` trong Octave.
2. **MSS** (Marine Systems Simulator), đúng commit đã dùng:

```bash
git clone https://github.com/cybergalactic/MSS D:/tools/MSS
git -C D:/tools/MSS checkout ed2ce7b
```

3. **Python 3.11** cho bước GPU, trong một môi trường riêng:

```bash
python -m venv D:/tools/venv-gpu
D:/tools/venv-gpu/Scripts/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
D:/tools/venv-gpu/Scripts/pip install numpy==2.4.6 scipy==1.17.1 pandas==3.1.0
```

4. **Python 3.11 thường** (cho bước QC và kiểm tra): cần `pandas` và `numpy`. Máy gốc dùng pandas 3.0.6, numpy 2.4.6.

Không cần MATLAB, không cần Capytaine.

## Bước 2. Kiểm tra code đúng phiên bản

```bash
cd D:/USV-landing-prediction
python sim/checksums.py code-check sim/reproduce_v2/code_sha256.txt
```

Dòng cuối phải là `all code files match`. File nào báo `FAILED` là code đó đã khác bản dùng để tạo dữ liệu. Phép kiểm tra bỏ qua kiểu xuống dòng (Windows/Linux), nên đúng với mọi cách clone.

## Bước 3. Chép đúng hệ số tàu và bộ nhớ chất lỏng

```bash
mkdir -p D:/tools/MSS/HYDRO/vessels_capytaine/otter_55kg D:/tools/MSS/HYDRO/vessels_capytaine/otter_80kg sim/cache
cp sim/reproduce_v2/inputs/hydro/otter_55kg/otter_55kg.mat D:/tools/MSS/HYDRO/vessels_capytaine/otter_55kg/
cp sim/reproduce_v2/inputs/hydro/otter_80kg/otter_80kg.mat D:/tools/MSS/HYDRO/vessels_capytaine/otter_80kg/
cp sim/reproduce_v2/inputs/memory_cache/*.mat sim/cache/
```

**Không tự chạy lại Capytaine.** Chạy lại cho hệ số lệch khoảng 10⁻⁴ so với file đã dùng (khối lượng nước kèm lệch tới 0.0103 trên thang 282, lực sóng 0.237 trên thang 2820), đủ để dữ liệu không còn giống.

## Bước 4. Ma trận thí nghiệm, sóng và file JSON (Octave, khoảng 10 phút)

```bash
OUT=D:/USV-landing-prediction/data/sim/dataset_v2_repro
mkdir -p $OUT/logs
python sim/make_design_matrix.py
cp sim/design_matrix.csv $OUT/design_matrix.csv
sha256sum $OUT/design_matrix.csv
grep "  design_matrix.csv" sim/reproduce_v2/checksums_sha256.txt
```

Hai mã băm in ra phải giống nhau. Sau đó chạy 10 tiến trình Octave song song, mỗi tiến trình 315 dòng của ma trận:

```bash
for a in 1 316 631 946 1261 1576 1891 2206 2521 2836; do
  b=$((a + 314))
  D:/tools/octave/mingw64/bin/octave-cli.exe --no-gui --quiet --eval "cd('D:/USV-landing-prediction/sim'); mss_setup; prep_gpu_inputs('D:/USV-landing-prediction/sim/design_matrix.csv', '$OUT', $a, $b, true, false)" > $OUT/logs/prep_$a.txt 2>&1 &
done
wait
tail -1 $OUT/logs/prep_*.txt
```

Mỗi file log phải kết thúc bằng dòng `prep rows ... done in ... s` (máy gốc: tối đa 588 s). Bước này ghi mô hình sóng và hằng số tàu vào `$OUT/gpu_inputs/`, và 3150 file JSON vào `$OUT/runs/`.

## Bước 5. Tích phân trên GPU (khoảng 75 phút)

Chạy **lần lượt** bốn nhóm, không chạy song song (thiếu RAM). Giữ đúng tham số `--batch` như dưới để cách chia lô giống bản gốc.

```bash
cd D:/USV-landing-prediction/sim
PY=D:/tools/venv-gpu/Scripts/python.exe
$PY batch_sim_gpu.py --out $OUT --ids missing --group otter_55kg,0   --tag g55_0   --writers 3 --h 0.025 > $OUT/logs/g55_0.txt 2>&1
$PY batch_sim_gpu.py --out $OUT --ids missing --group otter_80kg,0   --tag g80_0   --writers 3 --h 0.025 --batch 450 > $OUT/logs/g80_0.txt 2>&1
$PY batch_sim_gpu.py --out $OUT --ids missing --group otter_55kg,0.2 --tag g55_02 --writers 3 --h 0.025 > $OUT/logs/g55_02.txt 2>&1
$PY batch_sim_gpu.py --out $OUT --ids missing --group otter_80kg,0.2 --tag g80_02 --writers 3 --h 0.025 > $OUT/logs/g80_02.txt 2>&1
cd ..
tail -1 $OUT/logs/g*.txt
ls $OUT/runs/*.csv | wc -l
```

| Nhóm | Số lần chạy | Chia lô | Thời gian trên máy gốc |
| --- | --- | --- | --- |
| otter_55kg, không cản nhớt | 1350 | 1 lô 1350 | 1115 s |
| otter_80kg, không cản nhớt | 1350 | 3 lô × 450 | 1938 s |
| otter_55kg, cản nhớt | 225 | 1 lô 225 | 630 s |
| otter_80kg, cản nhớt | 225 | 1 lô 225 | 635 s |

Mỗi log phải kết thúc bằng `done: ... runs in ... s`, và lệnh cuối phải đếm được 3150 file. Nhóm đầu cần khoảng 3 GB RAM; đóng các ứng dụng nặng trước khi chạy.

## Bước 6. QC và nhãn vùng vận hành

```bash
python sim/batch_report.py data/sim/dataset_v2_repro
DATASET=dataset_v2_repro python analysis/sim/verify_dataset_v1.py integrity
DATASET=dataset_v2_repro python analysis/sim/verify_dataset_v1.py regime
DATASET=dataset_v2_repro python analysis/sim/add_regime.py
```

Bản gốc cho: 3126 lần đạt QC, 24 lần không đạt (đều vì Hs thực lệch quá 25 %, đều ở U = 0); 2608 lần `regular`, 542 lần `violent`.

## Bước 7. Kiểm tra đã ra giống bản gốc

So với mã băm của bản gốc (không cần có bộ dữ liệu gốc):

```bash
python sim/checksums.py check sim/reproduce_v2/checksums_sha256.txt data/sim/dataset_v2_repro/runs
```

Lệnh in, cho từng file CSV, `identical = True` nếu trùng từng byte với bản gốc. Ai có thêm bộ dữ liệu gốc (đặt ở `data/sim/dataset_v2/`) thì so chi tiết được:

```bash
python sim/checksums.py check sim/reproduce_v2/checksums_sha256.txt data/sim/dataset_v2_repro/runs data/sim/dataset_v2/runs
python sim/compare_gpu_octave.py data/sim/dataset_v2_repro/runs data/sim/dataset_v2/runs
```

- Lệnh thứ nhất thêm, cho file không trùng byte: độ lệch lớn nhất (tính theo độ lệch chuẩn của cột) và thời điểm quỹ đạo bắt đầu tách ra (`diverges_at_s`).
- Lệnh thứ hai so file JSON theo giá trị, bỏ qua thời điểm tạo và đường dẫn file. Trường `run.matPath` sẽ khác nếu MSS cài ở chỗ khác `D:/tools/MSS`; đó chỉ là đường dẫn.

Kết quả đúng trông như mục dưới: lần chạy `regular` lệch ở chữ số cuối hoặc trùng byte; lần chạy `violent` có thể có `diverges_at_s` từ khoảng 5 phút trở đi.

## Mức "giống" đã đo được

GPU cộng các số theo thứ tự không cố định giữa hai lần chạy, nên **không thể trùng từng byte cho mọi file, kể cả trên chính máy gốc**. Vùng "cưỡi sóng" (tàu chạy 2 m/s xuôi hoặc chéo sóng) lại hỗn loạn, nên sai khác ở chữ số cuối bị khuếch đại. Kết quả thử trên máy gốc ngày 27/10/2026:

| Thử | Kết quả |
| --- | --- |
| Bước 4 cho 20 lần chạy | Mô hình sóng và hằng số tàu trùng từng bit; 20/20 file JSON trùng giá trị |
| Bước 5, nhóm otter_55kg cản nhớt (225 lần), chia lô như bản gốc | 68 file trùng từng byte; 136 file lệch ở chữ số cuối (trung vị 1.1e-8 độ lệch chuẩn); 21 file tách quỹ đạo |
| 21 file tách quỹ đạo ở trên | Cả 21 thuộc vùng "cưỡi sóng" (U = 2 m/s, β = 0° hoặc 45°); trùng nhau ít nhất 319 s đầu (trung vị 426 s); độ lệch chuẩn heave/roll/pitch/yaw lệch trung vị 0.3/0.7/0.5/0.9 %, lớn nhất 2.0/6.5/5.4/11.4 % |
| Bước 5 với 5 lần chạy gộp lô 5 (khác bản gốc) | 3 lần bình thường lệch 7e-12 đến 3e-8; 2 lần "cưỡi sóng" tách ở 386 s và 518 s |

Kết quả phép thử lưu trên máy gốc (không có trong repo): `data/sim/repro_full/check.txt`, `data/sim/repro_full/classified.csv`, `data/sim/repro_test/`. Với cách dùng của đề tài (ngữ cảnh 60 s, dự đoán 20 s), hai bản tương đương; khi báo kết quả nên tách vùng "cưỡi sóng" theo cột `regime`.

## Những gì làm kết quả khác đi

| Thay đổi | Ảnh hưởng | Đã thử chưa |
| --- | --- | --- |
| Chạy lại Capytaine thay vì dùng `inputs/hydro/` | Hệ số lệch khoảng 10⁻⁴, mọi lần chạy khác ở mức nhỏ | Đã so hệ số; chưa chạy mô phỏng với hệ số mới |
| Xóa `sim/cache/` để khớp lại bộ nhớ chất lỏng | Chưa biết | Chưa thử |
| Dùng MATLAB thay Octave | Bộ sinh số ngẫu nhiên có thể cho dãy khác dù cùng seed, nên sóng có thể khác | Chưa thử |
| GPU khác loại, CUDA khác, hoặc chạy CPU (`--device cpu`) | Dự kiến như bảng trên: lần chạy bình thường lệch chữ số cuối, vùng "cưỡi sóng" tách quỹ đạo | Chưa thử |
| Chia lô khác bản gốc | Như trên | Đã thử với lô 5 |
| Bước tích phân khác 0.025 s | Khác hẳn khi tàu chạy nhanh ngược sóng | Đã đo khi làm v2 (`analysis/sim/verify_dataset_v1.md`, mục 2b) |
