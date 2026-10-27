# So sánh mô hình nền trong quy trình chọn cửa sổ có kiểm soát rủi ro (α = 0.05)

| Mô hình nền | Độ phủ dải 10–90 % (danh nghĩa 0.80) | Nửa độ rộng dải roll/pitch, 5 s đầu [°] | CRC gộp: FSR / recall | CRC theo hướng + ACI: FSR / recall / rủi ro tệ nhất 1 lần chạy | 8 tổ hợp: recall TB | 8 tổ hợp: rủi ro tệ nhất / TB tệ nhất |
|---|---|---|---|---|---|---|
| TiRex zero-shot | 0.77 | 0.55 | 0.13 / 0.68 | 0.20 / 0.42 / 0.057 | 0.53 | 0.081 / 0.059 |
| TiRex zero-shot (1 lượt) | 0.77 | 0.55 | 0.13 / 0.68 | 0.20 / 0.42 / 0.057 | 0.53 | 0.076 / 0.059 |
| TiRex tinh chỉnh (hiệu chỉnh lồng nhau) | 0.78 | 0.56 | 0.13 / 0.64 | 0.17 / 0.50 / 0.052 | 0.53 | 0.066 / 0.056 |
