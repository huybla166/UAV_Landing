# sim/ — tạo dữ liệu sóng → lực → chuyển động 6 DOF bằng MSS (Octave/MATLAB)

**Tạo lại bộ dữ liệu dataset_v2 từng bước: [reproduce_v2/REPRODUCE.md](reproduce_v2/REPRODUCE.md).** Phần dưới đây là cách chạy thử một lần mô phỏng.

Không dùng GUI. Chuỗi gọi giống `SIMhydroVessel.m` của MSS 2.0:
`hydroVesselConfig → load vessel → waveInitialization → computeManeuveringModel → [headingAutopilot + waveForceRAO + rk4(@hydroVessel)]`.

## Cài đặt
- GNU Octave (portable zip) tại `D:\tools\octave`, MSS tại `D:\tools\MSS` (hoặc đặt biến môi trường `MSS_ROOT`).
- File tàu: `testShip.mat` có sẵn trong `MSS/HYDRO/vessels_capytaine/testShip/`. Otter: chạy MSS-Capytaine rồi copy `otter_55kg.mat` vào `MSS/HYDRO/vessels_capytaine/otter_55kg/`.

## Chạy một lần (smoke test, 120 s)
```bash
octave-cli --eval "cd('D:/USV-landing-prediction/sim'); mss_setup; run_one('testShip.mat', 0.5, 3.0, 90, 0, 1, 120, 0.05, '../data/sim/smoke')"
```
Tham số: `run_one(matFile, Hs, Tp, beta_deg, U_ref, seed, T_final, h, outDir)`; β = 0 sóng xuôi, 180 sóng ngược.

## Đầu ra
- `<name>.csv` 22 cột: `t, zeta, tau_w1..6, N, E, D, phi, theta, psi, u, v, w, p, q, r, tauX, tauN` (m, rad, N, Nm; 20 Hz khi h = 0.05).
- `<name>_coef.json`: `M, MRB, MA, D, G, kappa, main, natural_period_s, environment, run` — "đáp án" cho PINN.

## Kiểm tra nhanh
`out.Hs_realized = 4·std(zeta)` phải gần `Hs`; `std_roll_deg` tăng khi β → 90°; không có cảnh báo phân kỳ.
