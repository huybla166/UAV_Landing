# Bằng chứng cho các con số trong báo cáo

Sinh tự động bởi `experiments/verify_report.py`. Mỗi dòng: giá trị ghi trong báo cáo, giá trị tính lại từ dữ liệu hoặc đọc từ nguồn (làm tròn cùng số chữ số), và nơi lấy.

**Tổng: 183 mục, khớp 183, lệch 0.**

| Mục | Nội dung | Báo cáo ghi | Tính lại / nguồn ghi | Nguồn | Khớp |
|---|---|---|---|---|---|
| 2. Bối cảnh | Mô hình USV (5) là tổng các hàm sin | tổng các hàm sin | “superpositions of finite sinusoidal components” | PDF bài gốc, tr. 5 | ✓ |
| 2. Bối cảnh | Cấp biển chính | sóng 3.2 m, chu kỳ trung bình 8.7 s | “significant wave height 3.2 m, wavelength 125 m, and mean period 8.7 s” | PDF bài gốc, tr. 12 | ✓ |
| 2. Bối cảnh | Hai cấp biển thử thêm | cấp 3: 0.9 m / 6.5 s; cấp 5: 3.3 m / 9.5 s | “Sea state 3: significant wave height 0.9 m and mean period 6.5 s. � Sea state 5: significant wave height 3.3 m and mean period 9.5 s” | PDF bài gốc, tr. 14 | ✓ |
| 2. Bối cảnh | Kích thước dữ liệu | 2500 mẫu, 500 s, 5 Hz | “2500 samples collected over 500 s at a 5 Hz” | PDF bài gốc, tr. 12 | ✓ |
| 2. Bối cảnh | Đầu vào LSTM | 1 lớp, 50 đơn vị, 10 bước | “single layer with 50 hidden units, processing 10-time-step” | PDF bài gốc, tr. 12 | ✓ |
| 2. Bối cảnh | Chia dữ liệu và chỉ số | 70% đầu huấn luyện, 30% cuối kiểm tra; RMSE và MAE | “first 70% of data and evaluated on the remaining 30% using root mean square error (RMSE) and mean absolute error” | PDF bài gốc, tr. 13 | ✓ |
| 2. Bối cảnh | Mốc so sánh | AR bậc 5, Yule–Walker | “5th-order autoregressive process with parameters estimated using Yule-Walker” | PDF bài gốc, tr. 13 | ✓ |
| 2. Bối cảnh | Ngưỡng cửa sổ | roll 0.5, pitch 0.3; ngưỡng biến thiên 0.22 | “set to 0.5 for roll and 0.3 for pitch. Fig. 6(a) and (b) demonstrate the roll and pitch stability analysis, where the moving averages remain” | PDF bài gốc, tr. 13 | ✓ |
| 2. Bối cảnh | Định nghĩa 1 có điều kiện heave | boong đang đi xuống sau đỉnh (z_s > 0, ż_s < 0) | “begins its descent after reaching the peak position” | PDF bài gốc, tr. 5 | ✓ |
| 2. Bối cảnh | Cửa sổ minh họa duy nhất | 98 mẫu (388–485) = 19.6 s | “98 sample points (from sample 388 to 485)” | PDF bài gốc, tr. 13 | ✓ |
| 2. Bối cảnh | LSTM huấn luyện | Adam, tốc độ học 0.005, dropout 0.1, giảm từng đoạn | “dropout rate of 0.1 is applied for regularization. The model is trained using Adam optimizer with initial learning rate of 0.005 and piecewi” | PDF bài gốc, tr. 13 | ✓ |
| 8.2 LSTM | Bài có nêu số vòng? | không nêu → dự án tự chọn | không có từ “epoch” trong toàn bài | PDF bài gốc | ✓ |
| 8.2 LSTM | Bài có nêu batch? | không nêu → dự án tự chọn | không có từ “batch” trong toàn bài | PDF bài gốc | ✓ |
| 3. Nguồn gốc dữ liệu | Tác giả | Belibassakis (NTUA), Perera, Adnan (UiT) | “Belibassakis, Kostas authorAffiliation: National Technical University of Athens authorIdentifierScheme: ORCID authorIdentifier: 0000-0002-91” | metadata DataverseNO | ✓ |
| 3. Nguồn gốc dữ liệu | Liên hệ | prasad.perera@uit.no | “prasad.perera@uit.no” | metadata DataverseNO | ✓ |
| 3. Nguồn gốc dữ liệu | Tài trợ | European Commission, grant 857840 | “grantNumberAgency: European Commission grantNumberValue: 857840” | metadata DataverseNO | ✓ |
| 3. Nguồn gốc dữ liệu | Thời gian đo | 01/09/2022 – 30/10/2022 | “dateOfCollectionStart: 2022-09-01 dateOfCollectionEnd: 2022-10-30” | metadata DataverseNO | ✓ |
| 3. Nguồn gốc dữ liệu | Địa điểm | Lávrio, Attica, Hy Lạp; 24.03–24.12°E, 37.65–37.75°N | “Lávrio geographicUnit: region westLongitude: 24.028577849966577 eastLongitude: 24.12335345485694 northLatitude: 37.74797868826538 southLatit” | metadata DataverseNO | ✓ |
| 3. Nguồn gốc dữ liệu | Công bố, giấy phép | 11/10/2023, CC0 1.0 | “CC0 1.0 uri: http://creativecommons.org/publicdomain/zero/1.0 iconUri: https://licensebuttons.net/p/zero/1.0/88x31.png rightsIdentifier: CC0” | metadata DataverseNO | ✓ |
| 3. Nguồn gốc dữ liệu | Tàu | 10–12 m, tự đẩy, có người trên tàu | “10-12m length. The tested model is equipped with an autonomous propulsion system and was deployed in coastal area near Athens, Greece. The m” | 00_README.txt | ✓ |
| 3. Nguồn gốc dữ liệu | Phao sóng | phao định hướng di động: Hs, Tp, phổ, hướng sóng | “Significant wave height, Peak period, frequency spectrum, mean wave direction” | 00_README.txt | ✓ |
| 3. Nguồn gốc dữ liệu | Gia tốc kế | 7 gia tốc kế chữ thập; FR24, 1.6 m trước giữa tàu | “seven accelerometers in a cross arrangement was used to measure the 6-dof responses of the ship model travelling at sea. Details concerning ” | 00_README.txt | ✓ |
| 3. Nguồn gốc dữ liệu | Động cơ | 2 động cơ Torqeedo 4 kW, cảm biến lực đẩy | “2 electric Torquedo outboard motors of nominal power 4kW” | 00_README.txt | ✓ |
| 3. Nguồn gốc dữ liệu | Định vị: RTK | hệ định vị thời gian thực (RTK) | “monitored via a real-time kinematics system” | 00_README.txt | ✓ |
| 3. Nguồn gốc dữ liệu | Định vị: GPS | GPS ghi quỹ đạo và vận tốc | “vessel track data and ship speed recording through GPS” | 00_README.txt | ✓ |
| 3. Nguồn gốc dữ liệu | Cánh | load cell lực nâng/đẩy + con quay 3 trục | “lift and thrust by means of load cells, and additional data concerning roll, pitch, yaw at the location of the wing using an installed 3D-gy” | 00_README.txt | ✓ |
| 5. Xử lý | MD5 khớp kho gốc | 19/19 | 19 / 19 | data/VXGPII/_manifest.tsv | ✓ |
| 4. Cấu trúc file | Số file | 19 | 19 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Tổng dung lượng | 9.2 MB | 9.2 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Kích thước file loại 00_README | 17.6 KB | 17.6 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Kích thước file loại 1 | 69.7 KB | 69.7 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Kích thước file loại 2 | 396 KB | 396 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Kích thước file loại 3 | 1.02 MB | 1.02 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Kích thước file loại 4 | 33.9 KB | 33.9 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Kích thước file loại 5 | 1.56 MB | 1.56 | data/VXGPII/ | ✓ |
| 4. Cấu trúc file | Tần số lấy mẫu chuyển động | 20 Hz | 20 | file loại 2 | ✓ |
| 4. Cấu trúc file | Số mẫu / thời lượng mỗi lần chạy | 6000 mẫu = 300 s | 6000 / 300 | file loại 2 | ✓ |
| 4. Cấu trúc file | Giá trị thiếu | 0 | 0 | file loại 2 | ✓ |
| 4. Cấu trúc file | Độ phân giải roll/pitch | 0.0057° | 0.0057 | file loại 2 (bước nhỏ nhất giữa hai giá trị) | ✓ |
| 4. Cấu trúc file | Độ phân giải heave | 0.1 mm | 0.1 | file loại 2 | ✓ |
| 4. Cấu trúc file | Phổ sóng: bước tần số, tần số lớn nhất | 0.024 Hz, 50 Hz | 0.024 / 50 | file loại 1 | ✓ |
| 4. Cấu trúc file | noWing_headQ: sóng Hs, Tp | 0.10 m, 1.86 s | 0.10 / 1.86 | file loại 1–2, noWing_headQ | ✓ |
| 4. Cấu trúc file | noWing_headQ: heave std (min … max) | 2.6 cm (−9.5 … +8.9) | 2.6 / -9.5 / 8.9 | file loại 1–2, noWing_headQ | ✓ |
| 4. Cấu trúc file | noWing_headQ: roll TB; std (min … max) | −0.41°; 0.88° (−4.7 … +3.2) | -0.41 / 0.88 / -4.7 / 3.2 | file loại 1–2, noWing_headQ | ✓ |
| 4. Cấu trúc file | noWing_headQ: pitch std (min … max) | 0.75° (−2.0 … +2.7) | 0.75 / -2.0 / 2.7 | file loại 1–2, noWing_headQ | ✓ |
| 4. Cấu trúc file | noWing_headQ: chu kỳ chính heave/roll/pitch | 1.8 / 2.9 / 1.8 s | 1.8 / 2.9 / 1.8 | file loại 1–2, noWing_headQ | ✓ |
| 4. Cấu trúc file | noWing_followQ: sóng Hs, Tp | 0.10 m, 1.86 s | 0.10 / 1.86 | file loại 1–2, noWing_followQ | ✓ |
| 4. Cấu trúc file | noWing_followQ: heave std (min … max) | 1.4 cm (−5.4 … +5.7) | 1.4 / -5.4 / 5.7 | file loại 1–2, noWing_followQ | ✓ |
| 4. Cấu trúc file | noWing_followQ: roll TB; std (min … max) | −0.75°; 0.85° (−5.2 … +3.4) | -0.75 / 0.85 / -5.2 / 3.4 | file loại 1–2, noWing_followQ | ✓ |
| 4. Cấu trúc file | noWing_followQ: pitch std (min … max) | 0.24° (−0.6 … +1.0) | 0.24 / -0.6 / 1.0 | file loại 1–2, noWing_followQ | ✓ |
| 4. Cấu trúc file | noWing_followQ: chu kỳ chính heave/roll/pitch | 2.9 / 2.8 / 2.2 s | 2.9 / 2.8 / 2.2 | file loại 1–2, noWing_followQ | ✓ |
| 4. Cấu trúc file | wing_headQ: sóng Hs, Tp | 0.20 m, 1.37 s | 0.20 / 1.37 | file loại 1–2, wing_headQ | ✓ |
| 4. Cấu trúc file | wing_headQ: heave std (min … max) | 2.2 cm (−8.3 … +9.0) | 2.2 / -8.3 / 8.9 | file loại 1–2, wing_headQ | ✓ |
| 4. Cấu trúc file | wing_headQ: roll TB; std (min … max) | +0.83°; 1.02° (−2.2 … +4.3) | 0.83 / 1.02 / -2.2 / 4.3 | file loại 1–2, wing_headQ | ✓ |
| 4. Cấu trúc file | wing_headQ: pitch std (min … max) | 0.43° (−1.0 … +2.2) | 0.43 / -1.0 / 2.2 | file loại 1–2, wing_headQ | ✓ |
| 4. Cấu trúc file | wing_headQ: chu kỳ chính heave/roll/pitch | 2.4 / 3.3 / 4.7 s | 2.4 / 3.3 / 4.7 | file loại 1–2, wing_headQ | ✓ |
| 4. Cấu trúc file | wing_followQ: sóng Hs, Tp | 0.20 m, 1.37 s | 0.20 / 1.37 | file loại 1–2, wing_followQ | ✓ |
| 4. Cấu trúc file | wing_followQ: heave std (min … max) | 2.7 cm (−12.1 … +9.7) | 2.7 / -12.1 / 9.7 | file loại 1–2, wing_followQ | ✓ |
| 4. Cấu trúc file | wing_followQ: roll TB; std (min … max) | +0.25°; 0.69° (−1.9 … +2.6) | 0.25 / 0.69 / -1.9 / 2.6 | file loại 1–2, wing_followQ | ✓ |
| 4. Cấu trúc file | wing_followQ: pitch std (min … max) | 0.33° (−0.7 … +1.4) | 0.33 / -0.7 / 1.4 | file loại 1–2, wing_followQ | ✓ |
| 4. Cấu trúc file | wing_followQ: chu kỳ chính heave/roll/pitch | 3.9 / 3.9 / 4.3 s | 3.9 / 3.9 / 4.3 | file loại 1–2, wing_followQ | ✓ |
| 4. Cấu trúc file | Roll lớn nhất (trị tuyệt đối) | 5.2° | 5.2 | file loại 2 | ✓ |
| 4. Cấu trúc file | Chu kỳ chính của roll (4 lần chạy) | 2.8–3.9 s | 2.8 / 3.9 | file loại 2 (Welch) | ✓ |
| 4. Cấu trúc file | Tần số chính (mọi kênh, mọi lần chạy) | 0.21–0.55 Hz | 0.21 / 0.55 | file loại 2 (Welch) | ✓ |
| 4. Cấu trúc file | Đợt lắc roll mạnh, noWing_headQ | 267–283 s, 287–299 s | 267–283 s, 287–299 s | file loại 2: std trượt 5 s > 1.5 × std cả chuỗi | ✓ |
| 4. Cấu trúc file | Đợt lắc roll mạnh, noWing_followQ | 17–40 s | 17–40 s | file loại 2: std trượt 5 s > 1.5 × std cả chuỗi | ✓ |
| 5. Xử lý | Năng lượng trên 2.5 Hz (lớn nhất) | 0.024% | 0.024 | Welch, file loại 2 | ✓ |
| 5. Xử lý | Năng lượng dưới 0.15 Hz (lớn nhất) | 0.005% | 0.005 | Welch, file loại 2 | ✓ |
| 5. Xử lý | noWing_headQ: biên độ 1 s đầu / độ lệch chuẩn (lớn nhất 3 kênh) | ≤ 0.6 | 0.28 | file loại 2 | ✓ |
| 5. Xử lý | noWing_headQ: roll, pitch ở mẫu đầu và mẫu cuối | −0.61, 0.40 và −0.60, 0.40 | -0.61 / 0.40 / -0.60 / 0.40 | file loại 2 | ✓ |
| 5. Xử lý | noWing_followQ: biên độ 1 s đầu / độ lệch chuẩn (lớn nhất 3 kênh) | ≤ 0.6 | 0.547 | file loại 2 | ✓ |
| 5. Xử lý | noWing_followQ: roll, pitch ở mẫu đầu và mẫu cuối | −0.70, 0.39 và −0.69, 0.39 | -0.70 / 0.39 / -0.69 / 0.39 | file loại 2 | ✓ |
| 5. Xử lý | Độ nghiêng tĩnh roll | −0.75° đến +0.83° | -0.75 / 0.83 | file loại 2 | ✓ |
| 5. Xử lý | Phổ phao: tần số đầu tiên có S > 0.1% đỉnh | 0.46–0.49 Hz | 0.46 / 0.49 | file loại 1 | ✓ |
| 5. Xử lý | Phổ phao: S lớn nhất dưới 0.46 Hz so với đỉnh | 0.01% | 0.01 | file loại 1 | ✓ |
| 5. Xử lý | Hai lần chạy cùng điều kiện dùng chung phổ | MD5 trùng | trùng | _manifest.tsv | ✓ |
| 5. Xử lý | Hs tính từ phổ | 0.10 / 0.20 m | 0.10 / 0.20 | file loại 1 | ✓ |
| 5. Xử lý | Tự tương quan cắt 0 sau | 0.45–0.8 s | 0.45 / 0.8 | file loại 2 | ✓ |
| 5. Xử lý | Định nghĩa 1 (0.5/0.3/0.22), roll thô: số cửa sổ ≥ 5 s; dài nhất | 0; 1.3 s | 0 / 1.3 | file loại 2, 20 Hz | ✓ |
| 5. Xử lý | Định nghĩa 1, roll/pitch đã trừ trung bình: số cửa sổ ≥ 5 s; dài nhất | 0; 1.8 s | 0 / 1.8 | file loại 2, 20 Hz | ✓ |
| 6. Tìm dữ liệu | MSS Otter USV | dài 2 m, chỉ có mô hình | “The length of the USV is L = 2.0 m” | github.com/cybergalactic/MSS (otter.m) | ✓ (trang nguồn) |
| 6. Tìm dữ liệu | ACE, tàu Akademik Tryoshnikov | IMU 1 Hz, 12/2016–3/2017 | “from December 2016 to March 2017 … IMU at a sampling rate of 1 Hz” | zenodo.org/records/4541564; data.aad.gov.au AAS_4434 | ✓ (trang nguồn) |
| 6. Tìm dữ liệu | IEEE DataPort “ship motion” | bước 0.176 s, hơn 1000 s, cần đăng nhập | “sampling time exceeding 1000s, collection interval of 0.176s … login credentials needed” | ieee-dataport.org/documents/ship-motion (DOI 10.21227/sk98-8j20) | ✓ (trang nguồn) |
| 6. Tìm dữ liệu | ship-movement-dataset | tàu trong cảng, gộp theo giờ | “Outer Port of Punta Langosteira … variables … have a period of 1 hour” | github.com/aalvarell/ship-movement-dataset | ✓ (trang nguồn) |
| 6. Tìm dữ liệu | Jinghai VII | USV 8.2 m, 10 Hz, hướng/pitch/roll, không có heave, không công bố dữ liệu | “8.2 m length … Sampling rate: 10 Hz … Heading, pitch, and roll … no explicit data availability statement” | pmc.ncbi.nlm.nih.gov/articles/PMC7582249 | ✓ (trang nguồn) |
| 6. Tìm dữ liệu | SPOWTT | 14 tàu CTV 15–27 m, tư thế 40 Hz | “14 different crew transfer vessels … lengths from 15 to 27 meters … attitude … at 40 Hz” | doi.org/10.1002/we.2647 | ✓ (trang nguồn) |
| 6. Tìm dữ liệu | TU Delft green water | mô hình đứng yên trong bể, 38.9 GB | “the model is stationary and water and waves flow past the ship … 38.9 GB” | data.4tu.nl/…/21031981 | ✓ (trang nguồn) |
| 7. Quy trình | Số mẫu mỗi lần chạy sau cắt (5 Hz) | 1450 | 1450 | data/processed/*_5Hz.csv, common.py | ✓ |
| 7. Quy trình | Thời điểm dự đoán: mỗi lần chạy, tổng | 211, 844 | 211 / 844 | common.origins | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±1.0°/±0.75°, T = 5 s | 20% | 20 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±1.0°/±0.75°, T = 10 s | 10% | 10 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±1.5°/±1.0°, T = 5 s | 47% | 47 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±1.5°/±1.0°, T = 10 s | 32% | 32 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±2.0°/±1.25°, T = 5 s | 73% | 73 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±2.0°/±1.25°, T = 10 s | 61% | 61 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±2.5°/±1.5°, T = 5 s | 86% | 86 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | Tỉ lệ an toàn thật ±2.5°/±1.5°, T = 10 s | 77% | 77 | windows_lib.safe_truth | ✓ |
| 7. Quy trình | K²VAE, CRC gộp, hiệu chỉnh KHÔNG lồng nhau: P(báo an toàn & nguy hiểm) | 14.7% | 14.7 | windows_lib, preds_k2vae.npz | ✓ |
| 8. Thuật toán | Bậc AR chọn theo AIC (nhỏ nhất – lớn nhất) | p = 39–60 | 39 / 60 | run_baselines.run_ar('aic') | ✓ |
| 8. Thuật toán | Ridge: lịch sử chọn theo kiểm định | L = 120–200 mẫu (24–40 s) | 120 / 200 / 24 / 40 | online_elf.fit_ridge | ✓ |
| 8. Thuật toán | Ridge: hệ số phạt chọn | α = 10⁻⁶ | 1e-06 | online_elf.fit_ridge | ✓ |
| 8. Thuật toán | FreDF: pinball kiểm định (có FreDF, không FreDF) | 0.3420 so với 0.3421 | 0.3420 / 0.3421 | results/optimize_report.json | ✓ |
| 8. Thuật toán | Dải conformal chung 90%: recall | 1% | 1 | results/windows_results.json (tirex_cp_aci) | ✓ |
| 8. Thuật toán | CRC cục bộ: recall | 0.11 | 0.11 | results/regime_results.json (crc_local) | ✓ |
| 11.3 Phát hiện | K²VAE trên dữ liệu tổng các hàm sin: nRMSE 1–20 s | 0.10–0.17 | 0.10 / 0.17 | results/k2vae_synthetic.json (k2vae_synthetic_check.py) | ✓ |
| 8. Thuật toán | K²VAE thu nhỏ: số tham số | 0.52 triệu | 0.52 | results/k2vae_synthetic.json | ✓ |
| 8. Thuật toán | K²VAE cấu hình gốc: số tham số | 8.38 triệu | 8.38 | run_k2vae.Cfg | ✓ |
| 8. Thuật toán | TiRex: số tham số; số khối sLSTM | 35.3 triệu; 12 | 35.3 / 12 | NX-AI/TiRex | ✓ |
| 8. Thuật toán | TiRex: đoạn đầu ra; số phân vị | 32 mẫu; 9 | 32 / 9 | NX-AI/TiRex config | ✓ |
| 8. Thuật toán | TiRex: vector ẩn; đoạn đầu vào | 512 chiều; 32 mẫu | 512 / 32 | NX-AI/TiRex config | ✓ |
| 10. TiRex | Phần đầu ra được học lại: số tham số, tỉ lệ | 1.79 triệu, khoảng 5% | 1.79 / 5 | finetune_tirex.Head | ✓ |
| 10. TiRex | Tính đặc trưng một lần: thời gian, dung lượng | 24 phút, 96 MB | 24 / 96 | results/log_tirex_cache.txt, tirex_feature_cache.npz | ✓ |
| 11.1 Dự đoán | LSTM của bài: nRMSE 1 s | 0.79 / 1.11 / 0.90 | 0.79 / 1.11 / 0.90 | results/preds_lstm_paper.npz | ✓ |
| 11.1 Dự đoán | LSTM của bài: nRMSE 2 s | 1.08 / 1.20 / 1.12 | 1.08 / 1.20 / 1.12 | results/preds_lstm_paper.npz | ✓ |
| 11.1 Dự đoán | LSTM của bài: nRMSE 5 s | 1.06 / 1.28 / 1.20 | 1.06 / 1.28 / 1.20 | results/preds_lstm_paper.npz | ✓ |
| 11.1 Dự đoán | LSTM của bài: tầm hữu ích (s) | 1.4 / 0.6 / 0.8 | 1.4 / 0.6 / 0.8 | results/preds_lstm_paper.npz | ✓ |
| 11.1 Dự đoán | AR(p) theo AIC: nRMSE 1 s | 0.42 / 0.35 / 0.52 | 0.42 / 0.35 / 0.52 | results/preds_ar_aic.npz | ✓ |
| 11.1 Dự đoán | AR(p) theo AIC: nRMSE 2 s | 0.84 / 0.70 / 0.95 | 0.84 / 0.70 / 0.95 | results/preds_ar_aic.npz | ✓ |
| 11.1 Dự đoán | AR(p) theo AIC: nRMSE 5 s | 0.93 / 0.95 / 0.99 | 0.93 / 0.95 / 0.99 | results/preds_ar_aic.npz | ✓ |
| 11.1 Dự đoán | AR(p) theo AIC: tầm hữu ích (s) | 2.6 / 3.2 / 1.4 | 2.6 / 3.2 / 1.4 | results/preds_ar_aic.npz | ✓ |
| 11.1 Dự đoán | TiRex zero-shot: nRMSE 1 s | 0.54 / 0.44 / 0.60 | 0.54 / 0.44 / 0.60 | results/preds_tirex_ctx300.npz | ✓ |
| 11.1 Dự đoán | TiRex zero-shot: nRMSE 2 s | 0.86 / 0.69 / 0.92 | 0.86 / 0.69 / 0.92 | results/preds_tirex_ctx300.npz | ✓ |
| 11.1 Dự đoán | TiRex zero-shot: nRMSE 5 s | 0.94 / 0.91 / 1.00 | 0.94 / 0.91 / 1.00 | results/preds_tirex_ctx300.npz | ✓ |
| 11.1 Dự đoán | TiRex zero-shot: tầm hữu ích (s) | 2.0 / 4.0 / 1.8 | 2.0 / 4.0 / 1.8 | results/preds_tirex_ctx300.npz | ✓ |
| 11.1 Dự đoán | TiRex zero-shot: độ phủ dải 10–90% | 0.77 | 0.77 | results/preds_tirex_ctx300.npz | ✓ |
| 11.1 Dự đoán | TiRex tinh chỉnh: nRMSE 1 s | 0.42 / 0.35 / 0.49 | 0.42 / 0.35 / 0.49 | results/preds_tirex_ft.npz | ✓ |
| 11.1 Dự đoán | TiRex tinh chỉnh: nRMSE 2 s | 0.81 / 0.66 / 0.88 | 0.81 / 0.66 / 0.88 | results/preds_tirex_ft.npz | ✓ |
| 11.1 Dự đoán | TiRex tinh chỉnh: nRMSE 5 s | 0.91 / 0.90 / 0.98 | 0.91 / 0.90 / 0.98 | results/preds_tirex_ft.npz | ✓ |
| 11.1 Dự đoán | TiRex tinh chỉnh: tầm hữu ích (s) | 3.0 / 4.2 / 2.0 | 3.0 / 4.2 / 2.0 | results/preds_tirex_ft.npz | ✓ |
| 11.1 Dự đoán | TiRex tinh chỉnh: pinball 0–5 s | 0.227 / 0.202 / 0.243 | 0.227 / 0.202 / 0.243 | results/preds_tirex_ft.npz | ✓ |
| 11.1 Dự đoán | TiRex tinh chỉnh: độ phủ dải 10–90% | 0.78 / 0.77 / 0.80 | 0.78 / 0.77 / 0.80 | results/preds_tirex_ft.npz | ✓ |
| 11.1 Dự đoán | Tuyến tính trực tiếp: nRMSE 1 s | 0.30 / 0.21 / 0.40 | 0.30 / 0.21 / 0.40 | results/preds_ridge_direct.npz | ✓ |
| 11.1 Dự đoán | Tuyến tính trực tiếp: nRMSE 2 s | 0.64 / 0.39 / 0.82 | 0.64 / 0.39 / 0.82 | results/preds_ridge_direct.npz | ✓ |
| 11.1 Dự đoán | Tuyến tính trực tiếp: nRMSE 5 s | 0.91 / 0.83 / 0.95 | 0.91 / 0.83 / 0.95 | results/preds_ridge_direct.npz | ✓ |
| 11.1 Dự đoán | Tuyến tính trực tiếp: tầm hữu ích (s) | 3.8 / 5.2 / 2.0 | 3.8 / 5.2 / 2.0 | results/preds_ridge_direct.npz | ✓ |
| 11.1 Dự đoán | Mô hình cuối: nRMSE 1 s | 0.30 / 0.21 / 0.40 | 0.30 / 0.21 / 0.40 | results/preds_final.npz | ✓ |
| 11.1 Dự đoán | Mô hình cuối: nRMSE 2 s | 0.64 / 0.39 / 0.82 | 0.64 / 0.39 / 0.82 | results/preds_final.npz | ✓ |
| 11.1 Dự đoán | Mô hình cuối: nRMSE 5 s | 0.90 / 0.82 / 0.93 | 0.90 / 0.82 / 0.93 | results/preds_final.npz | ✓ |
| 11.1 Dự đoán | Mô hình cuối: tầm hữu ích (s) | 4.0 / 5.2 / 2.0 | 4.0 / 5.2 / 2.0 | results/preds_final.npz | ✓ |
| 11.1 Dự đoán | Mô hình cuối: pinball 0–5 s | 0.202 / 0.161 / 0.227 | 0.202 / 0.161 / 0.227 | results/preds_final.npz | ✓ |
| 11.1 Dự đoán | Mô hình cuối: độ phủ dải 10–90% | 0.80 / 0.79 / 0.81 | 0.80 / 0.79 / 0.81 | results/preds_final.npz | ✓ |
| 11.1 Dự đoán | K²VAE (3 bản): nRMSE 1 s | 0.94–1.06 | 0.94 / 1.06 | results/preds_k2vae*.npz | ✓ |
| 11.1 Dự đoán | K²VAE (3 bản): nRMSE 2 s | 0.93–1.05 | 0.93 / 1.05 | results/preds_k2vae*.npz | ✓ |
| 11.1 Dự đoán | K²VAE (3 bản): nRMSE 5 s | 0.95–1.05 | 0.95 / 1.05 | results/preds_k2vae*.npz | ✓ |
| 11.1 Dự đoán | K²VAE: tầm hữu ích | 0.0–0.2 | 0.0 / 0.2 | results/preds_k2vae*.npz | ✓ |
| 11.1 Dự đoán | K²VAE: độ phủ (TB 0–20 s, theo kênh) | 0.76–0.82 | 0.76 / 0.82 | results/preds_k2vae*.npz | ✓ |
| 11.1 Dự đoán | Diebold–Mariano, mô hình cuối so với lstm_paper: giảm MSE 0–5 s (%) | 46 / 77 / 48 | 46 / 77 / 48 | dm trên results/preds_*.npz | ✓ |
| 11.1 Dự đoán | Diebold–Mariano so với lstm_paper: p | p < 10⁻¹⁰ | 0.0e+00, 0.0e+00, 0.0e+00 | dm trên results/preds_*.npz | ✓ |
| 11.1 Dự đoán | Diebold–Mariano, mô hình cuối so với tirex_ft: giảm MSE 0–5 s (%) | 17 / 34 / 11 | 17 / 34 / 11 | dm trên results/preds_*.npz | ✓ |
| 11.1 Dự đoán | Diebold–Mariano so với tirex_ft: p | p < 10⁻¹⁰ | 0.0e+00, 0.0e+00, 9.6e-11 | dm trên results/preds_*.npz | ✓ |
| 11.1 Dự đoán | Diebold–Mariano, mô hình cuối so với ridge_direct: giảm MSE 0–5 s (%) | 3 / 0 / 6 | 3 / 0 / 6 | dm trên results/preds_*.npz | ✓ |
| 11.1 Dự đoán | Diebold–Mariano so với ridge_direct: p | 0.095 / 0.86 / 0.00027 | 0.095 / 0.86 / 0.00027 | dm trên results/preds_*.npz | ✓ |
| 11.2 Quyết định | LSTM của bài (điểm + ngưỡng): FSR / recall / P(báo an toàn & nguy hiểm) / rủi ro tệ nhất | 0.42 / 0.81 / 0.277 / 0.536 | 0.42 / 0.81 / 0.277 / 0.536 | tính lại: windows_lib + run_regime_calibration, preds_lstmq | ✓ |
| 11.2 Quyết định | LSTM của bài (điểm + ngưỡng): 8 tổ hợp, recall TB / rủi ro tệ nhất | 0.75 / 0.687 | 0.75 / 0.687 | như trên | ✓ |
| 11.2 Quyết định | LSTM + lớp quyết định đề xuất: FSR / recall / P(báo an toàn & nguy hiểm) / rủi ro tệ nhất | 0.22 / 0.27 / 0.036 / 0.062 | 0.22 / 0.27 / 0.036 / 0.062 | tính lại: windows_lib + run_regime_calibration, preds_lstmq | ✓ |
| 11.2 Quyết định | Tuyến tính + CRC: FSR / recall / P(báo an toàn & nguy hiểm) / rủi ro tệ nhất | 0.15 / 0.57 / 0.049 / 0.062 | 0.15 / 0.57 / 0.049 / 0.062 | tính lại: windows_lib + run_regime_calibration, preds_ridgeq | ✓ |
| 11.2 Quyết định | Tuyến tính + CRC: 8 tổ hợp, recall TB / rủi ro tệ nhất | 0.54 / 0.081 | 0.54 / 0.081 | như trên | ✓ |
| 11.2 Quyết định | TiRex tinh chỉnh + CRC: FSR / recall / P(báo an toàn & nguy hiểm) / rủi ro tệ nhất | 0.17 / 0.50 / 0.047 / 0.052 | 0.17 / 0.50 / 0.047 / 0.052 | tính lại: windows_lib + run_regime_calibration, preds_tirex_ft | ✓ |
| 11.2 Quyết định | TiRex tinh chỉnh + CRC: 8 tổ hợp, recall TB / rủi ro tệ nhất | 0.53 / 0.066 | 0.53 / 0.066 | như trên | ✓ |
| 11.2 Quyết định | Mô hình cuối + CRC: FSR / recall / P(báo an toàn & nguy hiểm) / rủi ro tệ nhất | 0.14 / 0.59 / 0.046 / 0.066 | 0.14 / 0.59 / 0.046 / 0.066 | tính lại: windows_lib + run_regime_calibration, preds_final | ✓ |
| 11.2 Quyết định | Mô hình cuối + CRC: 8 tổ hợp, recall TB / rủi ro tệ nhất | 0.54 / 0.076 | 0.54 / 0.076 | như trên | ✓ |
| 11. Kết quả | LSTM gốc: rủi ro tệ nhất theo cấu hình | 34–69% | 34 / 69 | như trên | ✓ |
| 11.2 Quyết định | Bootstrap C so với A: ΔFSR [KTC 95%] | −0.25 [−0.32, −0.18] | -0.25 / -0.32 / -0.18 | results/lstm_vs_proposed.json | ✓ |
| 11.2 Quyết định | Bootstrap C so với A: ΔP [KTC 95%] | −0.23 [−0.27, −0.20] | -0.23 / -0.27 / -0.20 | results/lstm_vs_proposed.json | ✓ |
| 11.2 Quyết định | Bootstrap C so với A: Δrecall [KTC 95%] | −0.31 [−0.41, −0.20] | -0.31 / -0.41 / -0.20 | results/lstm_vs_proposed.json | ✓ |
| 11.2 Quyết định | Tách đóng góp: chỉ đổi lớp quyết định, ΔFSR | −0.21 | -0.21 | results/lstm_vs_proposed.json (B_vs_A) | ✓ |
| 11.2 Quyết định | Tách đóng góp: chỉ đổi mô hình dự đoán, Δrecall [KTC 95%] | +0.24 [+0.14, +0.34] | 0.24 / 0.14 / 0.34 | results/lstm_vs_proposed.json (C_vs_B) | ✓ |
| 12. Hạn chế | noWing_headQ: tỉ lệ thời điểm an toàn thật | 18% | 18 | windows_lib.safe_truth | ✓ |
| 12. Hạn chế | noWing_headQ: số lần báo nhầm, mô hình cuối / LSTM gốc | 12 / 113 | 12 / 113 | tính lại như mục 11.2 | ✓ |
| 12. Hạn chế | noWing_headQ: số lần báo nhầm, TiRex tinh chỉnh + CRC | 10 | 10 | tính lại như mục 11.2 | ✓ |
| 11.3 Phát hiện | CRC gộp (TiRex zero-shot): rủi ro TB; lần chạy tệ nhất | 4.7%; 10% | 4.7 / 10 | results/regime_results.json | ✓ |
| 11.3 Phát hiện | CRC gộp, 8 tổ hợp: lần chạy tệ nhất | 29.4% | 29.4 | results/regime_results.json | ✓ |
| 11.3 Phát hiện | CRC theo hướng + ACI, 8 tổ hợp: lần chạy tệ nhất | 8.1% | 8.1 | results/regime_results.json | ✓ |
| 11.3 Phát hiện | CRC gộp + ACI, 8 tổ hợp: lần chạy tệ nhất | 8.1% | 8.1 | results/regime_results.json | ✓ |
| 11.3 Phát hiện | Chỉ theo hướng sóng, không ACI: lần chạy tệ nhất | 13.3% | 13.3 | results/regime_results.json | ✓ |
| 11.3 Phát hiện | Dải TiRex hẹp đột ngột tại các tầm (bước) | mỗi 32 bước = 6.4 s | 33 (6.6 s), 65 (13.0 s), 97 (19.4 s) | results/preds_tirex_ctx300.npz | ✓ |
| 11.3 Phát hiện | Conformal theo từng tầm: độ phủ (nhỏ nhất – lớn nhất) | 0.90–0.91 | 0.90 / 0.91 | results/windows_summary.md | ✓ |
| 13. Phụ lục | Phiên bản numpy | 2.4.6 | 2.4.6 | importlib.metadata | ✓ |
| 13. Phụ lục | Phiên bản scipy | 1.17.1 | 1.17.1 | importlib.metadata | ✓ |
| 13. Phụ lục | Phiên bản matplotlib | 3.11.2 | 3.11.2 | importlib.metadata | ✓ |
| 13. Phụ lục | Phiên bản torch | 2.14.1 | 2.14.1+cpu | importlib.metadata | ✓ |
| 13. Phụ lục | Phiên bản tirex-ts | 1.4.2 | 1.4.2 | importlib.metadata | ✓ |
| 13. Phụ lục | Phiên bản einops | 0.8.2 | 0.8.2 | importlib.metadata | ✓ |
