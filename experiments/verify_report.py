"""Bằng chứng cho các con số trong báo cáo dự án.

Mỗi con số trong báo cáo được tính lại từ dữ liệu gốc / file dự đoán, hoặc đọc từ file kết quả, hoặc tìm
nguyên văn trong nguồn (PDF bài gốc, README và metadata DataverseNO), rồi so với giá trị ghi trong báo cáo.
Kết quả: results/report_evidence.md (bảng: mục | nội dung | báo cáo ghi | tính lại / nguồn ghi | nguồn | khớp).

Chạy: python verify_report.py   (khoảng 3–5 phút trên CPU; cần pdftotext để kiểm tra PDF bài gốc)
"""
import glob
import hashlib
import importlib.metadata as md
import json
import os
import re
import subprocess

import numpy as np
from scipy import signal
from scipy.stats import norm

import online_elf as oe
import run_baselines as rb
import run_regime_calibration as rc
import windows_lib as wl
from common import FS, HORIZON, RESULTS, ROOT, RUNS, load_all, load_preds, origins, targets

ROWS = []
NUM = re.compile(r"[-−+]?\d+(?:\.\d+)?")


def _nums(s):
    return [(float(m.replace("−", "-")), len(m.split(".")[1]) if "." in m else 0) for m in NUM.findall(s)]


def check(sec, item, reported, values, source):
    """Compare every number written in `reported` with `values` rounded to the same number of decimals."""
    values = list(np.atleast_1d(values).astype(float))
    rep = _nums(reported)
    ok = len(rep) == len(values) and all(abs(v - r) <= 0.5 * 10 ** -d + 1e-9 for (r, d), v in zip(rep, values))
    shown = " / ".join(f"{v:.{d}f}" for (r, d), v in zip(rep, values)) if len(rep) == len(values) else str(values)
    ROWS.append((sec, item, reported, shown, source, "✓" if ok else "✗ LỆCH"))


def within(sec, item, reported, values, lo, hi, source):
    """Claim states a range/bound: every computed value must lie inside [lo, hi]."""
    values = list(np.atleast_1d(values).astype(float))
    ok = all(lo <= v <= hi for v in values)
    ROWS.append((sec, item, reported, " / ".join(f"{v:.3g}" for v in values), source, "✓" if ok else "✗ LỆCH"))


def check_text(sec, item, reported, haystack, needle, source):
    """Evidence by quotation: `needle` (regex) must occur in the source text."""
    m = re.search(needle, haystack, flags=re.I | re.S)
    quote = re.sub(r"\s+", " ", m.group(0))[:140] if m else "không tìm thấy"
    ROWS.append((sec, item, reported, f"“{quote}”", source, "✓" if m else "✗ KHÔNG CÓ"))


def note(sec, item, reported, value, source, ok=True):
    ROWS.append((sec, item, reported, value, source, "✓" if ok else "✗ LỆCH"))


# =====================================================================================================
# A. Bài gốc (mục 2, 8.2): trích nguyên văn từ PDF
# =====================================================================================================
PDF = os.path.join(ROOT, "references", "Wang2026_JFI_108381.pdf")
pages = []
try:
    txt = subprocess.run(["pdftotext", "-layout", PDF, "-"], capture_output=True).stdout.decode("utf-8", "replace")
    pages = txt.split("\f")
except FileNotFoundError:
    pass


def paper(sec, item, reported, needle):
    for i, p in enumerate(pages):
        if re.search(needle, p, flags=re.I | re.S):
            return check_text(sec, item, reported, p, needle, f"PDF bài gốc, tr. {i + 1}")
    ROWS.append((sec, item, reported, "không tìm thấy", "PDF bài gốc", "✗ KHÔNG CÓ"))


if pages:
    S = "2. Bối cảnh"
    paper(S, "Mô hình USV (5) là tổng các hàm sin", "tổng các hàm sin", r"superpositions of finite sinusoidal components")
    paper(S, "Cấp biển chính", "sóng 3.2 m, chu kỳ trung bình 8.7 s", r"significant wave height 3\.2 m.{0,40}mean period 8\.7 s")
    paper(S, "Hai cấp biển thử thêm", "cấp 3: 0.9 m / 6.5 s; cấp 5: 3.3 m / 9.5 s",
          r"Sea state 3: significant wave height 0\.9 m and mean period 6\.5 s.{0,80}Sea state 5: significant wave height 3\.3 m and mean period 9\.5 s")
    paper(S, "Kích thước dữ liệu", "2500 mẫu, 500 s, 5 Hz", r"2500 samples collected over 500 s at a 5 Hz")
    paper(S, "Đầu vào LSTM", "1 lớp, 50 đơn vị, 10 bước", r"single layer with 50 hidden units, processing 10-time-step")
    paper(S, "Chia dữ liệu và chỉ số", "70% đầu huấn luyện, 30% cuối kiểm tra; RMSE và MAE",
          r"first 70% of data and evaluated on the remaining 30% using root mean square error \(RMSE\)\s+and mean absolute error")
    paper(S, "Mốc so sánh", "AR bậc 5, Yule–Walker", r"5th-order autoregressive process with parameters estimated using Yule-Walker")
    paper(S, "Ngưỡng cửa sổ", "roll 0.5, pitch 0.3; ngưỡng biến thiên 0.22", r"set to 0\.5\s+for roll and 0\.3 for pitch.{0,400}0\.22")
    paper(S, "Định nghĩa 1 có điều kiện heave", "boong đang đi xuống sau đỉnh (z_s > 0, ż_s < 0)",
          r"begins its descent after reaching the peak position")
    paper(S, "Cửa sổ minh họa duy nhất", "98 mẫu (388–485) = 19.6 s", r"98 sample points \(from sample 388 to 485\)")
    paper(S, "LSTM huấn luyện", "Adam, tốc độ học 0.005, dropout 0.1, giảm từng đoạn",
          r"dropout rate of 0\.1.{0,120}Adam optimizer with initial learning rate of 0\.005 and\s+piecewise decay")
    for kw, what in (("epoch", "số vòng"), ("batch", "batch")):
        hit = any(re.search(kw, p, re.I) for p in pages)
        note("8.2 LSTM", f"Bài có nêu {what}?", "không nêu → dự án tự chọn",
             "có" if hit else "không có từ “%s” trong toàn bài" % kw, "PDF bài gốc", ok=not hit)

# =====================================================================================================
# B. Dữ liệu: metadata, README, file gốc
# =====================================================================================================
VX = os.path.join(ROOT, "data", "VXGPII")
meta = json.load(open(os.path.join(VX, "_dataverse_metadata.json"), encoding="utf-8"))


def _flat(o, key=""):
    """DataverseNO metadata as 'field: value' lines."""
    if isinstance(o, dict):
        if "typeName" in o and "value" in o:
            return _flat(o["value"], o["typeName"])
        return "\n".join(_flat(v, k) for k, v in o.items())
    if isinstance(o, list):
        return "\n".join(_flat(v, key) for v in o)
    return f"{key}: {o}"


mtxt = _flat(meta["data"]["latestVersion"]) + f"\npublicationDate: {meta['data'].get('publicationDate')}"
readme = open(os.path.join(VX, "00_README.txt"), encoding="utf-8", errors="replace").read()
S = "3. Nguồn gốc dữ liệu"
check_text(S, "Tác giả", "Belibassakis (NTUA), Perera, Adnan (UiT)", mtxt,
           r"Belibassakis, Kostas\s+authorAffiliation: National Technical University of Athens.{0,200}Perera, Lokukaluge Prasad\s+authorAffiliation: UiT.{0,200}Adnan, Muhammad\s+authorAffiliation: UiT", "metadata DataverseNO")
check_text(S, "Liên hệ", "prasad.perera@uit.no", mtxt, r"prasad\.perera@uit\.no", "metadata DataverseNO")
check_text(S, "Tài trợ", "European Commission, grant 857840", mtxt, r"grantNumberAgency: European Commission\s+grantNumberValue: 857840", "metadata DataverseNO")
check_text(S, "Thời gian đo", "01/09/2022 – 30/10/2022", mtxt, r"dateOfCollectionStart: 2022-09-01\s+dateOfCollectionEnd: 2022-10-30", "metadata DataverseNO")
check_text(S, "Địa điểm", "Lávrio, Attica, Hy Lạp; 24.03–24.12°E, 37.65–37.75°N", mtxt,
           r"Lávrio.{0,300}westLongitude: 24\.028\d*\s+eastLongitude: 24\.123\d*\s+northLatitude: 37\.747\d*\s+southLatitude: 37\.654", "metadata DataverseNO")
check_text(S, "Công bố, giấy phép", "11/10/2023, CC0 1.0", mtxt, r"CC0 1\.0.*publicationDate: 2023-10-11", "metadata DataverseNO")
check_text(S, "Tàu", "10–12 m, tự đẩy, có người trên tàu", readme, r"10-12m length.{0,200}The model is manned", "00_README.txt")
check_text(S, "Phao sóng", "phao định hướng di động: Hs, Tp, phổ, hướng sóng", readme,
           r"Significant wave height, Peak period, frequency spectrum, mean\s+wave direction", "00_README.txt")
check_text(S, "Gia tốc kế", "7 gia tốc kế chữ thập; FR24, 1.6 m trước giữa tàu", readme,
           r"seven accelerometers in a cross arrangement.{0,500}FR24 of the model, at a distance of 1\.6m forward the midship", "00_README.txt")
check_text(S, "Động cơ", "2 động cơ Torqeedo 4 kW, cảm biến lực đẩy", readme, r"2 electric Torquedo outboard motors of\s+nominal power 4kW", "00_README.txt")
check_text(S, "Định vị: RTK", "hệ định vị thời gian thực (RTK)", readme, r"monitored via a real-time kinematics system", "00_README.txt")
check_text(S, "Định vị: GPS", "GPS ghi quỹ đạo và vận tốc", readme, r"vessel track data and ship speed recording through\s+GPS", "00_README.txt")
check_text(S, "Cánh", "load cell lực nâng/đẩy + con quay 3 trục", readme, r"lift and thrust\s+by means of load cells.{0,120}3D-gyro", "00_README.txt")

S = "4. Cấu trúc file"
manifest = [l.split("\t") for l in open(os.path.join(VX, "_manifest.tsv"), encoding="utf-8").read().splitlines() if l.strip()]
files = {}
for _, folder, name, md5_ in manifest:
    p = "\\\\?\\" + os.path.join(VX, *folder.split("/"), name) if folder else os.path.join(VX, name)
    files[name] = (p, md5_.strip())
ok = sum(hashlib.md5(open(p, "rb").read()).hexdigest() == m for p, m in files.values())
check("5. Xử lý", "MD5 khớp kho gốc", "19/19", [ok, len(files)], "data/VXGPII/_manifest.tsv")
size = {n: os.path.getsize(p) for n, (p, _) in files.items()}
check(S, "Số file", "19", len(files), "data/VXGPII/")
check(S, "Tổng dung lượng", "9.2 MB", sum(size.values()) / 1e6, "data/VXGPII/")
for tag, rep in (("00_README", "17.6 KB"), ("_1_", "69.7 KB"), ("_2_", "396 KB"), ("_3_", "1.02 MB"), ("_4_", "33.9 KB"), ("_5_", "1.56 MB")):
    vals = [s for n, s in size.items() if tag in n]
    v = np.mean(vals) / (1e6 if "MB" in rep else 1e3)
    check(S, f"Kích thước file loại {tag.strip('_')}", rep, v, "data/VXGPII/")

RAW = {"noWing_headQ": "without_Aset", "noWing_followQ": "without_Bset", "wing_headQ": "with_Aset", "wing_followQ": "with_Bset"}


def raw(run, kind):
    n = [k for k in files if k.startswith(RAW[run] + f"_{kind}_")][0]
    return np.loadtxt(files[n][0], skiprows=1), files[n][1]


resp = {r: raw(r, 2)[0] for r in RUNS}
wave = {r: raw(r, 1) for r in RUNS}
t0 = resp[RUNS[0]][:, 0]
check(S, "Tần số lấy mẫu chuyển động", "20 Hz", 1 / np.median(np.diff(t0)), "file loại 2")
check(S, "Số mẫu / thời lượng mỗi lần chạy", "6000 mẫu = 300 s", [len(t0), len(t0) / 20], "file loại 2")
check(S, "Giá trị thiếu", "0", sum(int(np.isnan(x).sum()) for x in resp.values()), "file loại 2")
res_roll = min(np.diff(np.unique(np.round(x[:, c], 9))).min() for x in resp.values() for c in (2, 3))
res_heave = min(np.diff(np.unique(np.round(x[:, 1], 9))).min() for x in resp.values())
check(S, "Độ phân giải roll/pitch", "0.0057°", res_roll, "file loại 2 (bước nhỏ nhất giữa hai giá trị)")
check(S, "Độ phân giải heave", "0.1 mm", res_heave * 1000, "file loại 2")
f0 = wave[RUNS[0]][0][:, 0]
check(S, "Phổ sóng: bước tần số, tần số lớn nhất", "0.024 Hz, 50 Hz", [np.median(np.diff(f0)), f0.max()], "file loại 1")


def stats(x):
    f, p = signal.welch(signal.detrend(x), fs=20, nperseg=2048)
    k = np.argmax(p[1:]) + 1
    return dict(mean=x.mean(), std=np.std(signal.detrend(x)), mn=x.min(), mx=x.max(), per=1 / f[k],
                lo=p[(f > 0) & (f < 0.15)].sum() / p.sum(), hi=p[f > 2.5].sum() / p.sum())


def decor(x):
    x = signal.detrend(x); ac = np.correlate(x, x, "full")[len(x) - 1:]; return np.argmax(ac / ac[0] < 0) / 20


TABLE = {  # giá trị ghi trong bảng mục 4
    "noWing_headQ": ("0.10 m, 1.86 s", "2.6 cm (−9.5 … +8.9)", "−0.41°; 0.88° (−4.7 … +3.2)", "0.75° (−2.0 … +2.7)", "1.8 / 2.9 / 1.8 s"),
    "noWing_followQ": ("0.10 m, 1.86 s", "1.4 cm (−5.4 … +5.7)", "−0.75°; 0.85° (−5.2 … +3.4)", "0.24° (−0.6 … +1.0)", "2.9 / 2.8 / 2.2 s"),
    "wing_headQ": ("0.20 m, 1.37 s", "2.2 cm (−8.3 … +9.0)", "+0.83°; 1.02° (−2.2 … +4.3)", "0.43° (−1.0 … +2.2)", "2.4 / 3.3 / 4.7 s"),
    "wing_followQ": ("0.20 m, 1.37 s", "2.7 cm (−12.1 … +9.7)", "+0.25°; 0.69° (−1.9 … +2.6)", "0.33° (−0.7 … +1.4)", "3.9 / 3.9 / 4.3 s"),
}
ST, HS = {}, {}
for r in RUNS:
    x = resp[r]
    st = [stats(x[:, c]) for c in (1, 2, 3)]
    ST[r] = st
    w = wave[r][0]
    df = np.gradient(w[:, 0]); m0 = np.sum(w[:, 1] * df)
    HS[r] = (4 * np.sqrt(m0), 1 / w[np.argmax(w[:, 1]), 0])
    rep = TABLE[r]
    src = f"file loại 1–2, {r}"
    check(S, f"{r}: sóng Hs, Tp", rep[0], HS[r], src)
    h = st[0]; check(S, f"{r}: heave std (min … max)", rep[1], [h["std"] * 100, h["mn"] * 100, h["mx"] * 100], src)
    o = st[1]; check(S, f"{r}: roll TB; std (min … max)", rep[2], [o["mean"], o["std"], o["mn"], o["mx"]], src)
    p = st[2]; check(S, f"{r}: pitch std (min … max)", rep[3], [p["std"], p["mn"], p["mx"]], src)
    check(S, f"{r}: chu kỳ chính heave/roll/pitch", rep[4], [s["per"] for s in st], src)
check(S, "Roll lớn nhất (trị tuyệt đối)", "5.2°", max(abs(ST[r][1][k]) for r in RUNS for k in ("mn", "mx")), "file loại 2")
check(S, "Chu kỳ chính của roll (4 lần chạy)", "2.8–3.9 s", [min(ST[r][1]["per"] for r in RUNS), max(ST[r][1]["per"] for r in RUNS)], "file loại 2 (Welch)")
check(S, "Tần số chính (mọi kênh, mọi lần chạy)", "0.21–0.55 Hz",
      [min(1 / s["per"] for r in RUNS for s in ST[r]), max(1 / s["per"] for r in RUNS for s in ST[r])], "file loại 2 (Welch)")

# đợt lắc mạnh: độ lệch chuẩn roll trượt 5 s lớn hơn 1.5 lần độ lệch chuẩn cả chuỗi
for r in ("noWing_headQ", "noWing_followQ"):
    roll = resp[r][:, 2] - resp[r][:, 2].mean()
    rs = np.array([roll[max(0, i - 50):i + 50].std() for i in range(len(roll))])
    hot = rs > 1.5 * roll.std()
    segs, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            if j - i >= 40:
                segs.append(f"{t0[i]:.0f}–{t0[j]:.0f} s")
            i = j + 1
        else:
            i += 1
    want = {"noWing_headQ": "267–283 s, 287–299 s", "noWing_followQ": "17–40 s"}[r]
    note(S, f"Đợt lắc roll mạnh, {r}", want, ", ".join(segs) or "không có",
         "file loại 2: std trượt 5 s > 1.5 × std cả chuỗi", ok=", ".join(segs) == want)

S = "5. Xử lý"
check(S, "Năng lượng trên 2.5 Hz (lớn nhất)", "0.024%", 100 * max(s["hi"] for r in RUNS for s in ST[r]), "Welch, file loại 2")
check(S, "Năng lượng dưới 0.15 Hz (lớn nhất)", "0.005%", 100 * max(s["lo"] for r in RUNS for s in ST[r]), "Welch, file loại 2")
for r in ("noWing_headQ", "noWing_followQ"):
    x = resp[r][:, 1:] - resp[r][:, 1:].mean(0)
    edge = (np.abs(x[:20]).max(0) / x.std(0)).max()
    within(S, f"{r}: biên độ 1 s đầu / độ lệch chuẩn (lớn nhất 3 kênh)", "≤ 0.6", edge, 0, 0.6, "file loại 2")
    check(S, f"{r}: roll, pitch ở mẫu đầu và mẫu cuối", "−0.61, 0.40 và −0.60, 0.40" if r == "noWing_headQ" else "−0.70, 0.39 và −0.69, 0.39",
          [resp[r][0, 2], resp[r][0, 3], resp[r][-1, 2], resp[r][-1, 3]], "file loại 2")
check(S, "Độ nghiêng tĩnh roll", "−0.75° đến +0.83°", [min(ST[r][1]["mean"] for r in RUNS), max(ST[r][1]["mean"] for r in RUNS)], "file loại 2")
first = {r: wave[r][0][np.argmax(wave[r][0][:, 1] > 1e-3 * wave[r][0][:, 1].max()), 0] for r in RUNS}
check(S, "Phổ phao: tần số đầu tiên có S > 0.1% đỉnh", "0.46–0.49 Hz", [min(first.values()), max(first.values())], "file loại 1")
check(S, "Phổ phao: S lớn nhất dưới 0.46 Hz so với đỉnh", "0.01%",
      100 * max(wave[r][0][wave[r][0][:, 0] < 0.46, 1].max() / wave[r][0][:, 1].max() for r in RUNS), "file loại 1")
same = wave["noWing_headQ"][1] == wave["noWing_followQ"][1] and wave["wing_headQ"][1] == wave["wing_followQ"][1]
note(S, "Hai lần chạy cùng điều kiện dùng chung phổ", "MD5 trùng", "trùng" if same else "khác", "_manifest.tsv", ok=same)
check(S, "Hs tính từ phổ", "0.10 / 0.20 m", [HS["noWing_headQ"][0], HS["wing_headQ"][0]], "file loại 1")
check(S, "Tự tương quan cắt 0 sau", "0.45–0.8 s",
      [min(decor(resp[r][:, c]) for r in RUNS for c in (1, 2, 3)), max(decor(resp[r][:, c]) for r in RUNS for c in (1, 2, 3))], "file loại 2")


def def1_windows(t, z, phi, theta, pt, tt, var=None):
    dz = np.gradient(z, t[1] - t[0]); okk = (np.abs(phi) <= pt) & (np.abs(theta) <= tt); out = []; i = 0
    while i < len(t):
        if z[i] > 0 and dz[i] < 0 and okk[i]:
            j = i; a = b = phi[i]; c_ = d_ = theta[i]
            while j + 1 < len(t) and okk[j + 1]:
                if var is not None:
                    a, b, c_, d_ = min(a, phi[j + 1]), max(b, phi[j + 1]), min(c_, theta[j + 1]), max(d_, theta[j + 1])
                    if b - a > var or d_ - c_ > var:
                        break
                j += 1
            out.append(t[j] - t[i]); i = j + 1
        else:
            i += 1
    return out


wr = [d for r in RUNS for d in def1_windows(resp[r][:, 0], resp[r][:, 1], resp[r][:, 2], resp[r][:, 3], 0.5, 0.3, 0.22)]
wd = [d for r in RUNS for d in def1_windows(resp[r][:, 0], resp[r][:, 1], resp[r][:, 2] - resp[r][:, 2].mean(),
                                            resp[r][:, 3] - resp[r][:, 3].mean(), 0.5, 0.3, 0.22)]
check(S, "Định nghĩa 1 (0.5/0.3/0.22), roll thô: số cửa sổ ≥ 5 s; dài nhất", "0; 1.3 s", [sum(d >= 5 for d in wr), max(wr)], "file loại 2, 20 Hz")
check(S, "Định nghĩa 1, roll/pitch đã trừ trung bình: số cửa sổ ≥ 5 s; dài nhất", "0; 1.8 s", [sum(d >= 5 for d in wd), max(wd)], "file loại 2, 20 Hz")

# =====================================================================================================
# B2. Nguồn dữ liệu khác đã xét (mục 6): trích từ trang nguồn đã đọc khi tìm dữ liệu (không tải về)
# =====================================================================================================
S = "6. Tìm dữ liệu"
for item, rep, quote, url in (
    ("MSS Otter USV", "dài 2 m, chỉ có mô hình", "The length of the USV is L = 2.0 m", "github.com/cybergalactic/MSS (otter.m)"),
    ("ACE, tàu Akademik Tryoshnikov", "IMU 1 Hz, 12/2016–3/2017", "from December 2016 to March 2017 … IMU at a sampling rate of 1 Hz", "zenodo.org/records/4541564; data.aad.gov.au AAS_4434"),
    ("IEEE DataPort “ship motion”", "bước 0.176 s, hơn 1000 s, cần đăng nhập", "sampling time exceeding 1000s, collection interval of 0.176s … login credentials needed", "ieee-dataport.org/documents/ship-motion (DOI 10.21227/sk98-8j20)"),
    ("ship-movement-dataset", "tàu trong cảng, gộp theo giờ", "Outer Port of Punta Langosteira … variables … have a period of 1 hour", "github.com/aalvarell/ship-movement-dataset"),
    ("Jinghai VII", "USV 8.2 m, 10 Hz, hướng/pitch/roll, không có heave, không công bố dữ liệu", "8.2 m length … Sampling rate: 10 Hz … Heading, pitch, and roll … no explicit data availability statement", "pmc.ncbi.nlm.nih.gov/articles/PMC7582249"),
    ("SPOWTT", "14 tàu CTV 15–27 m, tư thế 40 Hz", "14 different crew transfer vessels … lengths from 15 to 27 meters … attitude … at 40 Hz", "doi.org/10.1002/we.2647"),
    ("TU Delft green water", "mô hình đứng yên trong bể, 38.9 GB", "the model is stationary and water and waves flow past the ship … 38.9 GB", "data.4tu.nl/…/21031981"),
):
    ROWS.append((S, item, rep, f"“{quote}”", url, "✓ (trang nguồn)"))

# =====================================================================================================
# C. Quy trình đánh giá
# =====================================================================================================
S = "7. Quy trình"
data = load_all()
check(S, "Số mẫu mỗi lần chạy sau cắt (5 Hz)", "1450", len(data[RUNS[0]]), "data/processed/*_5Hz.csv, common.py")
n_o = [len(origins(len(data[r]))) for r in RUNS]
check(S, "Thời điểm dự đoán: mỗi lần chạy, tổng", "211, 844", [n_o[0], sum(n_o)], "common.origins")
for (rt, pt), reps in zip([(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)], [("20%", "10%"), ("47%", "32%"), ("73%", "61%"), ("86%", "77%")]):
    for T, rep in zip((5, 10), reps):
        base = np.mean(np.concatenate([wl.safe_truth(r, int(T * FS), rt, pt) for r in RUNS]))
        check(S, f"Tỉ lệ an toàn thật ±{rt}°/±{pt}°, T = {T} s", rep, 100 * base, "windows_lib.safe_truth")

# rò rỉ khi hiệu chỉnh không lồng nhau: K²VAE với hiệu chỉnh trên các lần chạy mà mô hình đã học
wl.set_base("k2vae")
assert wl.NESTED is None
leak = np.concatenate([wl.decide("tirex_crc", r, 25, 1.5, 1.0, 0.05)[0] & ~wl.safe_truth(r, 25, 1.5, 1.0) for r in RUNS]).mean()
check(S, "K²VAE, CRC gộp, hiệu chỉnh KHÔNG lồng nhau: P(báo an toàn & nguy hiểm)", "14.7%", 100 * leak, "windows_lib, preds_k2vae.npz")

# =====================================================================================================
# D. Thuật toán
# =====================================================================================================
S = "8. Thuật toán"
_, chosen = rb.run_ar("aic")
check(S, "Bậc AR chọn theo AIC (nhỏ nhất – lớn nhất)", "p = 39–60", [min(chosen.values()), max(chosen.values())], "run_baselines.run_ar('aic')")
sel = [oe.fit_ridge([r for r in RUNS if r != t], c)[1] for t in RUNS for c in range(3)]
check(S, "Ridge: lịch sử chọn theo kiểm định", "L = 120–200 mẫu (24–40 s)",
      [min(s[0] for s in sel), max(s[0] for s in sel), min(s[0] for s in sel) / FS, max(s[0] for s in sel) / FS], "online_elf.fit_ridge")
note(S, "Ridge: hệ số phạt chọn", "α = 10⁻⁶", ", ".join(sorted({f"{s[1]:.0e}" for s in sel})), "online_elf.fit_ridge",
     ok={s[1] for s in sel} == {1e-6})
opt = json.load(open(os.path.join(RESULTS, "optimize_report.json"), encoding="utf-8"))
check(S, "FreDF: pinball kiểm định (có FreDF, không FreDF)", "0.3420 so với 0.3421", [opt["ft_fredf"]["val_pinball"], opt["ft"]["val_pinball"]], "results/optimize_report.json")
W = json.load(open(os.path.join(RESULTS, "windows_results.json"), encoding="utf-8"))
check(S, "Dải conformal chung 90%: recall", "1%", 100 * W["methods"]["tirex_cp_aci"]["pooled"]["recall"], "results/windows_results.json (tirex_cp_aci)")
R = json.load(open(os.path.join(RESULTS, "regime_results.json"), encoding="utf-8"))
check(S, "CRC cục bộ: recall", "0.11", R["methods"]["crc_local"]["pooled"]["recall"], "results/regime_results.json (crc_local)")
k2s = json.load(open(os.path.join(RESULTS, "k2vae_synthetic.json"), encoding="utf-8"))
allv = [v for vs in k2s["nrmse_by_lead"].values() for v in vs]
check("11.3 Phát hiện", "K²VAE trên dữ liệu tổng các hàm sin: nRMSE 1–20 s", "0.10–0.17", [min(allv), max(allv)], "results/k2vae_synthetic.json (k2vae_synthetic_check.py)")
check(S, "K²VAE thu nhỏ: số tham số", "0.52 triệu", k2s["n_params"] / 1e6, "results/k2vae_synthetic.json")
try:
    import torch
    import run_k2vae as rk
    rk.Cfg.d_model, rk.Cfg.d_ff, rk.Cfg.n_heads, rk.Cfg.e_layers = 128, 256, 8, 2
    rk.Cfg.dynamic_dim, rk.Cfg.hidden_dim, rk.Cfg.hidden_layers = 128, 256, 2
    check(S, "K²VAE cấu hình gốc: số tham số", "8.38 triệu", sum(p.numel() for p in rk.k2VAE(rk.Cfg()).parameters()) / 1e6, "run_k2vae.Cfg")
    from tirex import load_model
    from finetune_tirex import Head
    tx = load_model("NX-AI/TiRex")
    n_all = sum(p.numel() for p in tx.parameters()); n_head = sum(p.numel() for p in Head(tx).parameters())
    check(S, "TiRex: số tham số; số khối sLSTM", "35.3 triệu; 12", [n_all / 1e6, len(tx.blocks)], "NX-AI/TiRex")
    check(S, "TiRex: đoạn đầu ra; số phân vị", "32 mẫu; 9", [tx.config.output_patch_size, len(tx.config.quantiles)], "NX-AI/TiRex config")
    check(S, "TiRex: vector ẩn; đoạn đầu vào", "512 chiều; 32 mẫu", [tx.config.block_kwargs["embedding_dim"], tx.config.input_patch_size], "NX-AI/TiRex config")
    check("10. TiRex", "Phần đầu ra được học lại: số tham số, tỉ lệ", "1.79 triệu, khoảng 5%", [n_head / 1e6, 100 * n_head / n_all], "finetune_tirex.Head")
except Exception as e:  # noqa: BLE001
    note(S, "Đếm tham số TiRex / K²VAE", "", f"bỏ qua: {e}", "", ok=False)
log = open(os.path.join(RESULTS, "log_tirex_cache.txt"), encoding="utf-8", errors="replace").read()
secs = max(float(s) for s in re.findall(r"\((\d+)s\)", log))
check("10. TiRex", "Tính đặc trưng một lần: thời gian, dung lượng", "24 phút, 96 MB",
      [secs / 60, os.path.getsize(os.path.join(RESULTS, "tirex_feature_cache.npz")) / 1e6], "results/log_tirex_cache.txt, tirex_feature_cache.npz")

# =====================================================================================================
# E. Kết quả dự đoán (mục 1, 11.1)
# =====================================================================================================
truth = {r: targets(x) for r, x in data.items()}
scale = {r: data[r].std(0) for r in RUNS}
LV = np.round(np.arange(0.1, 1.0, 0.1), 1)


def fc(name):
    p, q, lv = load_preds(name)
    nr = np.mean([np.sqrt(np.mean((p[r] - truth[r]) ** 2, 0)) / scale[r] for r in RUNS], 0)
    useful = [(np.nonzero(nr[:, c] >= 0.9)[0][:1] / FS).tolist() or [HORIZON / FS] for c in range(3)]
    out = dict(nr=nr, useful=[u[0] for u in useful], pin=None, cov=None)
    if q is not None and not np.allclose(q[RUNS[0]][..., 0], q[RUNS[0]][..., -1]):
        pins, covs = [], []
        for r in RUNS:
            e = (truth[r][..., None] - q[r]) / scale[r][None, None, :, None]
            pins.append(np.mean(np.maximum(LV * e, (LV - 1) * e)[:, :25], axis=(0, 1, 3)))
            covs.append(np.mean((truth[r] >= q[r][..., 0]) & (truth[r] <= q[r][..., -1]), axis=(0, 1)))
        out["pin"], out["cov"] = np.mean(pins, 0), np.mean(covs, 0)
    return out


S = "11.1 Dự đoán"
ROWS_FC = {
    "lstm_paper": ("LSTM của bài", "0.79 / 1.11 / 0.90", "1.08 / 1.20 / 1.12", "1.06 / 1.28 / 1.20", "1.4 / 0.6 / 0.8", None, None),
    "ar_aic": ("AR(p) theo AIC", "0.42 / 0.35 / 0.52", "0.84 / 0.70 / 0.95", "0.93 / 0.95 / 0.99", "2.6 / 3.2 / 1.4", None, None),
    "tirex_ctx300": ("TiRex zero-shot", "0.54 / 0.44 / 0.60", "0.86 / 0.69 / 0.92", "0.94 / 0.91 / 1.00", "2.0 / 4.0 / 1.8", None, "0.77"),
    "tirex_ft": ("TiRex tinh chỉnh", "0.42 / 0.35 / 0.49", "0.81 / 0.66 / 0.88", "0.91 / 0.90 / 0.98", "3.0 / 4.2 / 2.0", "0.227 / 0.202 / 0.243", "0.78 / 0.77 / 0.80"),
    "ridge_direct": ("Tuyến tính trực tiếp", "0.30 / 0.21 / 0.40", "0.64 / 0.39 / 0.82", "0.91 / 0.83 / 0.95", "3.8 / 5.2 / 2.0", None, None),
    "final": ("Mô hình cuối", "0.30 / 0.21 / 0.40", "0.64 / 0.39 / 0.82", "0.90 / 0.82 / 0.93", "4.0 / 5.2 / 2.0", "0.202 / 0.161 / 0.227", "0.80 / 0.79 / 0.81"),
}
FC = {}
for m, (lab, a, b, c, u, pin, cov) in ROWS_FC.items():
    f = FC[m] = fc(m)
    src = f"results/preds_{m}.npz"
    check(S, f"{lab}: nRMSE 1 s", a, f["nr"][4], src)
    check(S, f"{lab}: nRMSE 2 s", b, f["nr"][9], src)
    check(S, f"{lab}: nRMSE 5 s", c, f["nr"][24], src)
    check(S, f"{lab}: tầm hữu ích (s)", u, f["useful"], src)
    if pin:
        check(S, f"{lab}: pinball 0–5 s", pin, f["pin"], src)
    if cov:
        check(S, f"{lab}: độ phủ dải 10–90%", cov, f["cov"] if len(_nums(cov)) == 3 else f["cov"].mean(), src)
K2 = [fc(m) for m in ("k2vae", "k2vae_small", "k2vae_small_short")]
for lead, k, rep in ((1, 4, "0.94–1.06"), (2, 9, "0.93–1.05"), (5, 24, "0.95–1.05")):
    v = np.array([f["nr"][k] for f in K2])
    check(S, f"K²VAE (3 bản): nRMSE {lead} s", rep, [v.min(), v.max()], "results/preds_k2vae*.npz")
check(S, "K²VAE: tầm hữu ích", "0.0–0.2", [min(min(f["useful"]) for f in K2), max(max(f["useful"]) for f in K2)], "results/preds_k2vae*.npz")
check(S, "K²VAE: độ phủ (TB 0–20 s, theo kênh)", "0.76–0.82", [min(f["cov"].min() for f in K2), max(f["cov"].max() for f in K2)], "results/preds_k2vae*.npz")


def dm(a, b, c, lag=5):
    pa, pb = load_preds(a)[0], load_preds(b)[0]
    ea = np.concatenate([np.mean(((pa[r][:, :25, c] - truth[r][:, :25, c]) / scale[r][c]) ** 2, 1) for r in RUNS])
    eb = np.concatenate([np.mean(((pb[r][:, :25, c] - truth[r][:, :25, c]) / scale[r][c]) ** 2, 1) for r in RUNS])
    d = ea - eb; n = len(d); dc = d - d.mean()
    g = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))
    z = d.mean() / np.sqrt(var / n)
    return 2 * (1 - norm.cdf(abs(z))), 100 * (1 - eb.mean() / ea.mean())


for a, rep, prep in (("lstm_paper", "46 / 77 / 48", None), ("tirex_ft", "17 / 34 / 11", None), ("ridge_direct", "3 / 0 / 6", "0.095 / 0.86 / 0.00027")):
    r_ = [dm(a, "final", c) for c in range(3)]
    check(S, f"Diebold–Mariano, mô hình cuối so với {a}: giảm MSE 0–5 s (%)", rep, [abs(x[1]) if abs(x[1]) < 0.5 else x[1] for x in r_], "dm trên results/preds_*.npz")
    if prep:
        check(S, f"Diebold–Mariano so với {a}: p", prep, [x[0] for x in r_], "dm trên results/preds_*.npz")
    else:
        note(S, f"Diebold–Mariano so với {a}: p", "p < 10⁻¹⁰", ", ".join(f"{x[0]:.1e}" for x in r_), "dm trên results/preds_*.npz",
             ok=all(x[0] < 1e-10 for x in r_))

# =====================================================================================================
# F. Quyết định (mục 1, 11.2, 12)
# =====================================================================================================
S = "11.2 Quyết định"
SETTINGS = [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]


def decide_all(base, method, rth, pth, T):
    wl.set_base(base); rc.tx_med, rc.tx_w = wl.tx_med, wl.tx_w
    H = int(round(T * FS)); out = {}
    for r in RUNS:
        if method == "point":
            out[r] = (wl.decide("lstm_point", r, H, rth, pth)[0], ~wl.safe_truth(r, H, rth, pth))
        else:
            out[r] = rc.decide("crc_heading_aci", r, H, rth, pth)
    return out


def metr(dec):
    D = np.concatenate([dec[r][0] for r in RUNS]); U = np.concatenate([dec[r][1] for r in RUNS])
    return dict(fsr=(D & U).sum() / D.sum(), recall=(D & ~U).sum() / (~U).sum(), viol=(D & U).mean(),
                worst=max((dec[r][0] & dec[r][1]).mean() for r in RUNS))


VAR = {"A": ("lstmq", "point", "LSTM của bài (điểm + ngưỡng)", "0.42 / 0.81 / 0.277 / 0.536", "0.75 / 0.687"),
       "B": ("lstmq", "crc", "LSTM + lớp quyết định đề xuất", "0.22 / 0.27 / 0.036 / 0.062", None),
       "R": ("ridgeq", "crc", "Tuyến tính + CRC", "0.15 / 0.57 / 0.049 / 0.062", "0.54 / 0.081"),
       "C": ("tirex_ft", "crc", "TiRex tinh chỉnh + CRC", "0.17 / 0.50 / 0.047 / 0.052", "0.53 / 0.066"),
       "F": ("final", "crc", "Mô hình cuối + CRC", "0.14 / 0.59 / 0.046 / 0.066", "0.54 / 0.076")}
DEC = {}
for k, (base, meth, lab, main, grid) in VAR.items():
    for rth, pth in SETTINGS:
        for T in (5.0, 10.0):
            if grid or (rth, T) == (1.5, 5.0):
                DEC[(k, rth, T)] = decide_all(base, meth, rth, pth, T)
    m = metr(DEC[(k, 1.5, 5.0)])
    check(S, f"{lab}: FSR / recall / P(báo an toàn & nguy hiểm) / rủi ro tệ nhất", main, [m["fsr"], m["recall"], m["viol"], m["worst"]],
          f"tính lại: windows_lib + run_regime_calibration, preds_{base}")
    if grid:
        ms = [metr(DEC[(k, r_, T)]) for r_, _ in SETTINGS for T in (5.0, 10.0)]
        check(S, f"{lab}: 8 tổ hợp, recall TB / rủi ro tệ nhất", grid, [np.mean([x["recall"] for x in ms]), max(x["worst"] for x in ms)], "như trên")
mA = [metr(DEC[("A", r_, T)])["worst"] for r_, _ in SETTINGS for T in (5.0, 10.0)]
check("11. Kết quả", "LSTM gốc: rủi ro tệ nhất theo cấu hình", "34–69%", [100 * min(mA), 100 * max(mA)], "như trên")

boot = json.load(open(os.path.join(RESULTS, "lstm_vs_proposed.json"), encoding="utf-8"))["bootstrap_main"]
b = boot["C_vs_A"]
check(S, "Bootstrap C so với A: ΔFSR [KTC 95%]", "−0.25 [−0.32, −0.18]", b["fsr"], "results/lstm_vs_proposed.json")
check(S, "Bootstrap C so với A: ΔP [KTC 95%]", "−0.23 [−0.27, −0.20]", b["violation"], "results/lstm_vs_proposed.json")
check(S, "Bootstrap C so với A: Δrecall [KTC 95%]", "−0.31 [−0.41, −0.20]", b["recall"], "results/lstm_vs_proposed.json")
check(S, "Tách đóng góp: chỉ đổi lớp quyết định, ΔFSR", "−0.21", boot["B_vs_A"]["fsr"][0], "results/lstm_vs_proposed.json (B_vs_A)")
check(S, "Tách đóng góp: chỉ đổi mô hình dự đoán, Δrecall [KTC 95%]", "+0.24 [+0.14, +0.34]", boot["C_vs_B"]["recall"], "results/lstm_vs_proposed.json (C_vs_B)")

S = "12. Hạn chế"
r = "noWing_headQ"
U = DEC[("F", 1.5, 5.0)][r][1]
check(S, f"{r}: tỉ lệ thời điểm an toàn thật", "18%", 100 * (~U).mean(), "windows_lib.safe_truth")
check(S, f"{r}: số lần báo nhầm, mô hình cuối / LSTM gốc", "12 / 113",
      [int((DEC[("F", 1.5, 5.0)][r][0] & U).sum()), int((DEC[("A", 1.5, 5.0)][r][0] & U).sum())], "tính lại như mục 11.2")
check(S, f"{r}: số lần báo nhầm, TiRex tinh chỉnh + CRC", "10", int((DEC[("C", 1.5, 5.0)][r][0] & U).sum()), "tính lại như mục 11.2")

S = "11.3 Phát hiện"
grid_max = lambda m: max(g["max_run_violation"] for g in R["grid"] if g["method"] == m)
check(S, "CRC gộp (TiRex zero-shot): rủi ro TB; lần chạy tệ nhất", "4.7%; 10%",
      [100 * R["methods"]["crc_pooled"]["pooled"]["violation"], 100 * R["methods"]["crc_pooled"]["pooled"]["max_run_violation"]], "results/regime_results.json")
check(S, "CRC gộp, 8 tổ hợp: lần chạy tệ nhất", "29.4%", 100 * grid_max("crc_pooled"), "results/regime_results.json")
check(S, "CRC theo hướng + ACI, 8 tổ hợp: lần chạy tệ nhất", "8.1%", 100 * grid_max("crc_heading_aci"), "results/regime_results.json")
check(S, "CRC gộp + ACI, 8 tổ hợp: lần chạy tệ nhất", "8.1%", 100 * grid_max("crc_pooled_aci"), "results/regime_results.json")
check(S, "Chỉ theo hướng sóng, không ACI: lần chạy tệ nhất", "13.3%", 100 * R["methods"]["crc_heading"]["pooled"]["max_run_violation"], "results/regime_results.json")
_, q300, _ = load_preds("tirex_ctx300")
wid = np.mean([np.mean(q300[r][..., -1] - q300[r][..., 0], axis=(0, 2)) for r in RUNS], 0)
drops = [h + 1 for h in range(1, HORIZON) if wid[h] < 0.9 * wid[h - 1]]
note(S, "Dải TiRex hẹp đột ngột tại các tầm (bước)", "mỗi 32 bước = 6.4 s", ", ".join(f"{h} ({h / FS:.1f} s)" for h in drops),
     "results/preds_tirex_ctx300.npz", ok=all(h % 32 in (0, 1) for h in drops) and len(drops) > 0)
wsum = open(os.path.join(RESULTS, "windows_summary.md"), encoding="utf-8").read()
covs = [float(v) for line in wsum.splitlines() if "TiRex + conformal (mục tiêu 0.90)" in line for v in re.findall(r"\| (0\.\d\d)", line)]
check(S, "Conformal theo từng tầm: độ phủ (nhỏ nhất – lớn nhất)", "0.90–0.91", [min(covs), max(covs)], "results/windows_summary.md")

# =====================================================================================================
# G. Môi trường (mục 13)
# =====================================================================================================
S = "13. Phụ lục"
for pkg, rep in (("numpy", "2.4.6"), ("scipy", "1.17.1"), ("matplotlib", "3.11.2"), ("torch", "2.14.1"), ("tirex-ts", "1.4.2"), ("einops", "0.8.2")):
    try:
        v = md.version(pkg)
    except md.PackageNotFoundError:
        v = "không cài"
    note(S, f"Phiên bản {pkg}", rep, v, "importlib.metadata", ok=v.split("+")[0] == rep)

# =====================================================================================================
lines = ["# Bằng chứng cho các con số trong báo cáo", "",
         "Sinh tự động bởi `experiments/verify_report.py`. Mỗi dòng: giá trị ghi trong báo cáo, giá trị tính lại từ dữ liệu "
         "hoặc đọc từ nguồn (làm tròn cùng số chữ số), và nơi lấy.", "",
         f"**Tổng: {len(ROWS)} mục, khớp {sum(r[5].startswith('✓') for r in ROWS)}, lệch {sum(not r[5].startswith('✓') for r in ROWS)}.**", "",
         "| Mục | Nội dung | Báo cáo ghi | Tính lại / nguồn ghi | Nguồn | Khớp |", "|---|---|---|---|---|---|"]
lines += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in ROWS]
with open(os.path.join(RESULTS, "report_evidence.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
bad = [r for r in ROWS if not r[5].startswith("✓")]
print(f"{len(ROWS)} checks, {len(bad)} mismatches")
for r in bad:
    print(" ✗", r[0], "|", r[1], "| report:", r[2], "| computed:", r[3])
