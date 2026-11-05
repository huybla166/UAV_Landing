Independent checks of results/sim/phylstm_cv.md, 04/11/2026 (workflow agents). Scripts: experiments/phylstm/checks/ (paired.py, skeptic_phy.py, rerun.py, compare_rerun.py, rng_check.py, lead0.py); the second-seed reruns of cv0 and their logs: results/sim/phylstm/rerun/. Paths below that point to the session scratchpad refer to those copies.

# Paired statistics

## Paired statistics: PhyLSTM2 vs no-physics network vs paper LSTM (dataset_v2, protocol (a))

The physics effect holds up in direction and consistency, but it is small: about 0.2–0.4% on overall RMSE and 2–4% at 1–2 s leads. Because each variant was trained with one seed only, I can't yet rule out that it comes from that one seed. The gap between the big network and the paper LSTM (5–25%) is clearly real.

Values are in percent and use 3 significant digits. A negative value means the first model has lower error. "LSTM" is the mean of the paper LSTM's 3 seeds. The `t` column is the paired t over the 6 folds (df = 5). The `t_NB` column is the same t with the Nadeau–Bengio correction for overlapping training sets (training = 4 folds, test = 1, factor 0.632).

### 1–2. (phy − data)/data, per split (%)

| ch | metric | cv0 | cv1 | cv2 | cv3 | cv4 | cv5 | ood_Hs | ood_beta | CV phy<data | CV mean | CV std | t (p) | t_NB (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heave | RMSE | -0.378 | -0.179 | -0.394 | -0.643 | -0.552 | -0.328 | -0.426 | -0.308 | 6/6 | -0.412 | 0.165 | -6.13 (0.0017) | -3.88 (0.012) |
| heave | RMSE 1 s | -4.13 | -1.60 | -2.44 | -5.16 | -5.51 | -3.59 | -4.02 | -1.76 | 6/6 | -3.74 | 1.52 | -6.02 (0.0018) | -3.81 (0.013) |
| heave | RMSE 2 s | -2.09 | -0.661 | -1.81 | -4.25 | -4.94 | -1.78 | -2.74 | -2.50 | 6/6 | -2.59 | 1.64 | -3.86 (0.012) | -2.44 (0.059) |
| heave | RMSE 5 s | -0.816 | -0.483 | -0.975 | -1.78 | -1.40 | -0.353 | -1.01 | -0.482 | 6/6 | -0.967 | 0.544 | -4.36 (0.0073) | -2.76 (0.040) |
| heave | MAE | -0.548 | -0.239 | -0.586 | -0.990 | -0.892 | -0.511 | -0.640 | -0.388 | 6/6 | -0.628 | 0.273 | -5.63 (0.0024) | -3.56 (0.016) |
| roll | RMSE | -0.199 | -0.0445 | -0.196 | -0.310 | -0.148 | -0.158 | -0.200 | -0.0320 | 6/6 | -0.176 | 0.0865 | -4.98 (0.0042) | -3.15 (0.025) |
| roll | RMSE 1 s | -2.35 | -0.576 | -1.58 | -3.44 | -2.11 | -2.05 | -1.92 | **+1.72** | 6/6 | -2.02 | 0.940 | -5.25 (0.0033) | -3.32 (0.021) |
| roll | RMSE 2 s | -1.46 | -0.298 | -1.20 | -1.89 | -1.14 | -0.838 | -1.12 | **+0.398** | 6/6 | -1.14 | 0.543 | -5.14 (0.0036) | -3.25 (0.023) |
| roll | RMSE 5 s | -0.137 | +0.026 | -0.269 | -0.430 | -0.110 | -0.208 | -0.244 | -0.185 | 5/6 | -0.188 | 0.155 | -2.97 (0.031) | -1.88 (0.12) |
| roll | MAE | -0.349 | -0.0848 | -0.273 | -0.459 | -0.200 | -0.254 | -0.322 | -0.00684 | 6/6 | -0.270 | 0.128 | -5.18 (0.0035) | -3.28 (0.022) |
| pitch | RMSE | -0.434 | -0.120 | -0.354 | -0.735 | -0.261 | -0.212 | -0.452 | -0.306 | 6/6 | -0.353 | 0.217 | -3.99 (0.010) | -2.52 (0.053) |
| pitch | RMSE 1 s | -2.83 | -1.08 | +0.218 | -4.56 | -2.46 | -1.68 | -4.81 | -1.64 | 5/6 | -2.06 | 1.63 | -3.10 (0.027) | -1.96 (0.11) |
| pitch | RMSE 2 s | -2.09 | -0.622 | -1.54 | -3.05 | -1.35 | -1.33 | -2.23 | -0.628 | 6/6 | -1.66 | 0.827 | -4.93 (0.0044) | -3.12 (0.026) |
| pitch | RMSE 5 s | -0.865 | -0.329 | -0.754 | -1.43 | -0.312 | -0.280 | -0.614 | -1.04 | 6/6 | -0.662 | 0.452 | -3.59 (0.016) | -2.27 (0.072) |
| pitch | MAE | -0.528 | -0.130 | -0.464 | -0.883 | -0.337 | -0.292 | -0.577 | -0.402 | 6/6 | -0.439 | 0.258 | -4.17 (0.0087) | -2.64 (0.046) |

- **6 of 6 folds in the same direction:** a two-sided sign test gives p = 0.031, which is the smallest p it can reach with 6 folds.
- **Last epoch instead of best epoch:** comparing `phy_last` with `data_last` gives the same picture. All 15 rows have physics better in 6/6 folds, except roll 5 s at 5/6. The CV means are -0.365% (heave RMSE), -0.153% (roll RMSE), -0.315% (pitch RMSE), and -3.38 / -1.77 / -2.40% at 1 s, with t from -2.6 to -6.5.

### (data − LSTM)/LSTM and (phy − LSTM)/LSTM: CV mean ± std, t, and OOD (%)

| ch | metric | data−LSTM CV | t | ood_Hs | ood_beta | phy−LSTM CV | t | ood_Hs | ood_beta |
|---|---|---|---|---|---|---|---|---|---|
| heave | RMSE | -5.32 ± 0.418 | -31.2 | -5.29 | -11.4 | -5.71 ± 0.370 | -37.8 | -5.70 | -11.6 |
| heave | RMSE 1 s | -22.5 ± 2.15 | -25.6 | -22.8 | -40.2 | -25.4 ± 2.11 | -29.5 | -25.9 | -41.2 |
| heave | RMSE 2 s | -25.2 ± 1.64 | -37.7 | -26.9 | -40.6 | -27.1 ± 1.74 | -38.3 | -28.9 | -42.1 |
| heave | RMSE 5 s | -10.8 ± 0.682 | -38.7 | -11.5 | -18.4 | -11.6 ± 0.525 | -54.4 | -12.4 | -18.8 |
| heave | MAE | -5.91 ± 0.516 | -28.1 | -6.23 | -14.2 | -6.51 ± 0.336 | -47.5 | -6.83 | -14.5 |
| roll | RMSE | -2.29 ± 0.208 | -27.1 | -2.82 | -4.00 | -2.47 ± 0.246 | -24.5 | -3.01 | -4.03 |
| roll | RMSE 1 s | -6.93 ± 0.682 | -24.9 | -7.26 | -9.64 | -8.81 ± 0.923 | -23.4 | -9.04 | -8.09 |
| roll | RMSE 2 s | -9.23 ± 0.625 | -36.1 | -9.76 | -13.6 | -10.3 ± 0.786 | -32.0 | -10.8 | -13.2 |
| roll | RMSE 5 s | -4.52 ± 0.583 | -19.0 | -6.15 | -6.63 | -4.70 ± 0.676 | -17.0 | -6.38 | -6.80 |
| roll | MAE | -2.57 ± 0.236 | -26.7 | -3.26 | -4.28 | -2.84 ± 0.287 | -24.2 | -3.57 | -4.29 |
| pitch | RMSE | -4.28 ± 0.358 | -29.3 | -4.48 | -6.26 | -4.62 ± 0.257 | -44.1 | -4.92 | -6.55 |
| pitch | RMSE 1 s | -8.32 ± 1.53 | -13.3 | -9.26 | -27.9 | -10.2 ± 0.844 | -29.7 | -13.6 | -29.1 |
| pitch | RMSE 2 s | -14.6 ± 0.601 | -59.6 | -15.8 | -22.3 | -16.0 ± 0.633 | -62.1 | -17.7 | -22.7 |
| pitch | RMSE 5 s | -8.25 ± 0.587 | -34.5 | -9.35 | -11.0 | -8.86 ± 0.716 | -30.3 | -9.91 | -12.0 |
| pitch | MAE | -4.12 ± 0.378 | -26.7 | -4.40 | -6.28 | -4.54 ± 0.228 | -48.8 | -4.95 | -6.66 |

Both comparisons are 6/6 folds on every row. With the correction, t is still between -8.4 and -38 (p < 4e-4).

### 3. Phy vs data on individual test runs (per-run MSE)

Runs that share a wave seed are averaged first, so each wave seed counts once. "frac" is the share of wave seeds where phy is better; p is the Wilcoxon signed-rank test on those averages. Each CV fold has 37–38 wave seeds and 516–528 runs; each OOD split has 45 seeds and about 625 runs.

| split | heave all / 1 s / 2 s: frac (p) | roll all / 1 s / 2 s | pitch all / 1 s / 2 s |
|---|---|---|---|
| cv0 | 0.838 (2e-7) / 1.00 (1e-11) / 0.784 (1e-7) | 0.946 (1e-7) / 0.973 (3e-11) / 0.946 (6e-10) | 0.865 (7e-7) / 0.892 (2e-8) / 0.865 (1e-8) |
| cv1 | 0.703 (0.0011) / 0.865 (7e-8) / 0.703 (9e-4) | 0.649 (0.11) / 0.541 (0.12) / 0.703 (0.016) | 0.514 (0.43) / 0.595 (0.094) / 0.595 (0.055) |
| cv2 | 0.946 (1e-10) / 0.838 (1e-6) / 0.919 (4e-9) | 0.865 (3e-8) / 0.838 (1e-7) / 0.838 (4e-8) | 0.865 (3e-8) / 0.486 (0.93) / 0.838 (2e-7) |
| cv3 | 0.974 (7e-11) / 1.00 (7e-12) / 0.921 (2e-10) | 1.00 / 1.00 / 1.00 (all 7e-12) | 0.921 (2e-9) / 0.763 (3e-5) / 0.816 (3e-6) |
| cv4 | 0.974 / 0.947 / 0.974 (all ≤4e-11) | 0.737 (1e-4) / 0.789 (2e-5) / 0.763 (2e-4) | 0.895 (4e-8) / 0.947 (1e-10) / 0.921 (6e-8) |
| cv5 | 0.895 (1e-8) / 0.921 (5e-10) / 0.868 (1e-8) | 0.868 (1e-8) / 0.868 (6e-8) / 0.816 (2e-6) | 0.921 (5e-7) / 0.737 (1e-4) / 0.789 (2e-5) |
| ood_Hs | 1.00 / 1.00 / 1.00 (all 6e-14) | 0.933 (1e-12) / 0.911 (6e-12) / 0.867 (1e-9) | 1.00 / 1.00 / 1.00 (all 6e-14) |
| ood_beta | 0.711 (3e-5) / 0.822 (5e-7) / 0.667 (1e-5) | 0.444 (0.96) / **0.0222 (1e-13)** / **0.200 (0.0019)** | 0.689 (8e-5) / 0.756 (2e-5) / 0.578 (0.071) |
| 6 CV pooled (225 wave seeds, relative difference per seed) | 0.889 (6e-32) / 0.929 (1e-32) / 0.862 (3e-28) | 0.844 (2e-22) / 0.836 (2e-24) / 0.844 (3e-22) | 0.831 (2e-22) / 0.738 (2e-16) / 0.804 (9e-21) |

- **Run-level fractions without grouping** are 0.47–0.95 in CV, and up to 0.95 in ood_Hs. The run-level p-values are much smaller still but overstated, because runs sharing a wave seed are not independent.
- **Physics is worse in one place:** roll at 1–2 s in ood_beta (seed-level 1/45 and 9/45 better).
- **What these tests cannot show:** they only say this one phy model beats this one data model on this test data. They say nothing about training-seed variance.

### 4. Paper LSTM spread across its 3 seeds vs the phy−data gap

Spread is (max − min)/mean in %; the gap is |phy − data|/data in %.

| split | heave RMSE: spread / gap | heave 1 s: spread / gap | roll RMSE | roll 1 s | pitch RMSE | pitch 1 s |
|---|---|---|---|---|---|---|
| cv0 | 1.02 / 0.378 | 1.67 / 4.13 | 0.319 / 0.199 | 0.257 / 2.35 | 0.699 / 0.434 | 1.84 / 2.83 |
| cv1 | 2.51 / 0.179 | 1.55 / 1.60 | 0.755 / 0.0445 | 0.147 / 0.576 | 3.07 / 0.120 | 0.259 / 1.08 |
| cv2 | 0.864 / 0.394 | 1.58 / 2.44 | 0.254 / 0.196 | 0.612 / 1.58 | 1.37 / 0.354 | 1.35 / 0.218 |
| cv3 | 1.28 / 0.643 | 3.38 / 5.16 | 0.378 / 0.310 | 0.442 / 3.44 | 0.790 / 0.735 | 3.31 / 4.56 |
| cv4 | 1.87 / 0.552 | 1.96 / 5.51 | 0.408 / 0.148 | 0.863 / 2.11 | 1.11 / 0.261 | 0.675 / 2.46 |
| cv5 | 0.513 / 0.328 | 2.16 / 3.59 | 0.297 / 0.158 | 0.679 / 2.05 | 1.14 / 0.212 | 4.36 / 1.68 |
| ood_Hs | 0.635 / 0.426 | 1.22 / 4.02 | 0.346 / 0.200 | 0.811 / 1.92 | 0.453 / 0.452 | 2.47 / 4.81 |
| ood_beta | 11.4 / 0.308 | 2.56 / 1.76 | 1.27 / 0.0320 | 0.865 / 1.72 | 1.32 / 0.306 | 1.14 / 1.64 |

The table below averages over the 6 CV folds:
- **Single-seed std:** estimated from the LSTM's 3 seeds.
- **Expected std of a one-seed difference:** that std × √2, assuming independent seeds.
- **Its std over 6 folds:** the expected std divided by √6.
- **Observed phy−data:** the CV mean from section 1.

| ch | metric | LSTM spread | single-seed std | expected std of a one-seed difference | its std over 6 folds | observed phy−data | observed data−LSTM |
|---|---|---|---|---|---|---|---|
| heave | RMSE | 1.34 | 0.778 | 1.10 | 0.449 | -0.412 | -5.32 |
| heave | RMSE 1 s | 2.05 | 1.17 | 1.66 | 0.678 | -3.74 | -22.5 |
| roll | RMSE | 0.402 | 0.236 | 0.334 | 0.136 | -0.176 | -2.29 |
| roll | RMSE 1 s | 0.500 | 0.291 | 0.411 | 0.168 | -2.02 | -6.93 |
| pitch | RMSE | 1.36 | 0.802 | 1.13 | 0.463 | -0.353 | -4.28 |
| pitch | RMSE 1 s | 1.96 | 1.33 | 1.88 | 0.767 | -2.06 | -8.32 |

Two things to know when reading this table:
- **The variants do not train to the end.** Best epochs are 26–30 out of 30 for both phy and data, so the networks are still improving at epoch 30. "Best epoch" and "last epoch" therefore give nearly the same results.
- **The noise yardstick may not fit.** The phy−data differences vary much less from fold to fold than LSTM seed noise would predict (std 0.165% observed vs 1.10% expected for heave RMSE). Either the larger network varies less between seeds, or phy and data share seed 0 so their noise partly cancels. That makes the LSTM spread an uncertain yardstick for the physics effect.

### Conclusion
1. **Physics effect on overall RMSE and MAE:** it is consistent (6/6 folds, best and last epoch, 83–89% of 225 wave seeds) but small (-0.18 to -0.41%). On a single fold it is below the LSTM single-seed noise (0.24–0.80%) and below the LSTM's 3-seed spread in every split.
2. **At 1–2 s leads:** the gain is -1.1 to -3.7%. It exceeds the 1 s 3-seed noise for roll in every split and for heave in most folds. The corrected t is -1.96 to -3.8 (p = 0.013–0.11): suggestive, not conclusive. It reverses for roll in ood_beta (+1.7% at 1 s, 1/45 seeds better).
3. **What the tests can't settle:** the run-level and seed-level tests are extremely significant, but they measure test-set sampling noise for one fixed pair of trained models. They cannot rule out that the gain comes from the single training seed. Also, the physics loss uses the true future wave force during training, so part of the gain may be extra information rather than the physics constraint itself.
4. **The big network vs the paper LSTM is real:** -5.3% (heave), -2.3% (roll), -4.3% (pitch) on overall RMSE and -7 to -25% at 1–2 s. That is 6/6 folds with t from -13 to -60 (corrected p < 4e-4), larger in both OOD splits, and 3 to 15 times the LSTM seed spread. This gap comes from the network and training (both phy and data have it), not from physics.
5. **Next step:** to claim a physics effect, train at least 3 seeds per variant per fold, with separate init seeds for phy and data. Then test phy−data against the measured seed variance of this network rather than the LSTM's.

The script is `C:/Users/huybl/AppData/Local/Temp/claude/D--USV-landing-prediction/712c7cfe-cce9-4431-b9c6-f794ff7cbf8c/scratchpad/results_check/paired.py`. It reads `results/sim/phylstm/phylstm_metrics.npz`, `results/sim/lstm_cv_metrics.npz`, `results/sim/phylstm/phylstm_history.json` and `data/sim/dataset_v2/cache_5Hz/meta.csv`. No repository files were modified.

# Skeptic (attempted refutation, incl. a second-seed rerun of cv0)

**Verdict: not refuted, with two caveats.** The physics loss does make the forecast better than the same network trained without it, and this is not training noise. But the gain is small and almost all of it is in the first 1–2 s. Whether it comes from the physics equation or from extra information is **inconclusive**. No repository files were changed.

### 1. Training noise (seed / sampling stream)

- **The sampling streams were identical.** The premise that building Network 2 changes the window sampling is wrong on GPU. `torch.randint` and `torch.randperm` on CUDA use the CUDA generator, and the weight initialisation only consumes the CPU generator. I checked this with `rng_check.py`: on CUDA the start weights and the windows were identical with and without Network 2; on CPU the windows differ. So in the Kaggle runs (Tesla T4) the two variants differ only in the loss.
- **The paper-LSTM's 3-seed spread is the wrong noise yardstick.** It is about 10× too pessimistic. It averages 2.5 / 0.50 / 1.24 % (heave/roll/pitch) over the 20 s, and the phy gain is below it in 0/8 splits. I measured Network 1's own noise directly by retraining cv0 for the full 30 epochs on the local RTX 3060:

| cv0, RMSE over 20 s (heave m, roll °, pitch °) | heave | roll | pitch |
|---|---|---|---|
| phy seed 0 (Kaggle) | 0.070496 | 2.1808 | 2.6340 |
| phy seed 1 (local) | 0.070534 | 2.1811 | 2.6360 |
| data seed 0 (Kaggle) | 0.070763 | 2.1851 | 2.6455 |
| data seed 0 (local; same start weights, different GPU sampling stream) | 0.070766 | 2.1845 | 2.6470 |
| data seed 1 (local) | 0.070858 | 2.1856 | 2.6479 |
| **phy gain, seed 0 / seed 1** | 0.38 / 0.46 % | 0.20 / 0.21 % | 0.43 / 0.45 % |
| noise: other sampling stream / other seed | 0.00 / 0.13 % | 0.03 / 0.05 % | 0.06 / 0.04 % |

- Both phy runs beat all three data runs in every channel. The gain over 20 s is about 3–10× Network 1's own seed noise.
- The gain is positive in all 8 splits × 3 channels, for both the best and the last epoch. Run by run, phy wins 63–93 % of heave runs.
- **Single-seed numbers at short leads are noisy.** At 0.2 s, changing the stream or seed moves heave RMSE by 9–16 %, and at 1 s by up to 2.8 %. The phy gain there is 17–23 % at 0.2 s and 1.7–4.4 % at 1 s, so it still holds, but those columns need more seeds.
- **The run-level paired z-test is not valid here.** LSTM seed0 vs seed1 alone gives |z| up to 16. The unit to replicate is a training run, not a test run.
- **Learning curves:** validation MSE for phy is below data at 29–30 of 30 epochs in every split, from epoch 1 onward. In the seed-1 rerun it is 30/30.

### 2. Was training still improving?

Barely. In the last 5 epochs the validation slope is between −0.0006 and +0.0003 per epoch. That is no larger than the epoch-to-epoch noise (sd 0.0002–0.0014). The learning rate is already 1e-4 from epoch 21, so best epochs of 26–30 just mean a flat, noisy plateau.

The gap between the variants does narrow over training, though. The validation gap averages 0.0063–0.0101 at epochs 1–5 and 0.0027–0.0091 at epochs 26–30. In cv1 and cv5 it shrank to about a third. So part of the gain is faster convergence, and longer training could shrink it further. A rank flip at learning rate 1e-4 looks unlikely, but it is not ruled out.

### 3. Where the gain sits

**By lead** (6 folds pooled, change in RMSE, heave / roll / pitch):

| Lead | Gain |
|---|---|
| 0.2 s | 17.6 / 16.1 / 6.0 % |
| 1 s | 3.8 / 2.0 / 2.1 % |
| 2 s | 2.6 / 1.1 / 1.7 % |
| 5 s | 1.0 / 0.2 / 0.7 % |
| 10 s | 0.14 / 0.02 / 0.10 % |
| 20 s | 0.02 / 0.00 / 0.03 % |

So the 20-s gain is essentially a 0–3 s effect. In absolute terms it is small: heave at 1 s goes from 35.6 to 34.2 mm.

At 0.2 s, the no-physics network (8.3 mm heave) is worse than the paper LSTM (7.3 mm), while PhyLSTM2 is better (6.8 mm). This points to a fix at the junction between past and forecast.

**Condition table.** Each cell is "gain over 20 s % / gain at leads ≤2 s % / share of runs where phy wins". Gain = 100·(1 − ΣMSE_phy / ΣMSE_data). "LSTM noise" is the paper-LSTM seed1-vs-seed0 difference over 20 s, heave only.

| Group | n | Heave | Roll | Pitch | LSTM noise, heave |
|---|---|---|---|---|---|
| cv0 | 518 | +0.75 / +5.8 / 76 % | +0.40 / +4.8 / 82 % | +0.87 / +5.5 / 67 % | −0.74 |
| cv1 | 522 | +0.36 / +2.6 / 63 % | +0.09 / +1.6 / 65 % | +0.24 / +2.5 / 47 % | −1.28 |
| cv2 | 517 | +0.79 / +5.0 / 86 % | +0.39 / +3.3 / 80 % | +0.71 / +2.7 / 69 % | −1.74 |
| cv3 | 516 | +1.28 / +10.2 / 91 % | +0.62 / +5.9 / 91 % | +1.46 / +8.8 / 86 % | +1.83 |
| cv4 | 525 | +1.10 / +11.0 / 92 % | +0.29 / +4.0 / 67 % | +0.52 / +4.4 / 75 % | +3.66 |
| cv5 | 528 | +0.66 / +6.2 / 82 % | +0.32 / +3.7 / 78 % | +0.42 / +3.7 / 73 % | −0.29 |
| ood_Hs | 628 | +0.85 / +7.1 / 93 % | +0.40 / +4.7 / 90 % | +0.90 / +8.1 / 87 % | −0.40 |
| ood_beta | 625 | +0.62 / +4.7 / 65 % | +0.06 / −1.7 / 46 % | +0.61 / +1.9 / 67 % | −2.94 |
| U = 0 (all folds) | 1026 | +0.89 / +13.2 / 81 % | +0.40 / +5.2 / 74 % | +0.57 / +4.5 / 71 % | |
| U = 1 | 1050 | +0.87 / +7.4 / 84 % | +0.41 / +4.4 / 80 % | +0.93 / +6.2 / 72 % | |
| U = 2 | 1050 | +0.72 / +4.4 / 81 % | +0.23 / +2.7 / 78 % | +0.65 / +3.6 / 65 % | |
| β = 0° | 626 | +1.78 / +17.9 / 86 % | +0.82 / +9.1 / 76 % | +1.11 / +8.7 / 81 % | |
| β = 45° | 625 | +1.02 / +9.1 / 83 % | +0.36 / +3.2 / 81 % | +0.82 / +4.6 / 75 % | |
| β = 90° | 626 | +0.66 / +6.8 / 84 % | +0.31 / +3.6 / 89 % | +0.41 / +3.3 / 73 % | |
| β = 135° | 624 | +0.41 / +3.9 / 78 % | +0.14 / +2.0 / 68 % | +0.13 / +1.7 / 52 % | |
| β = 180° | 625 | +0.59 / +5.6 / 79 % | +0.49 / +4.9 / 72 % | +0.46 / +3.6 / 66 % | |
| regular | 2584 | +0.76 / +7.0 / 81 % | +0.36 / +4.0 / 76 % | +0.67 / +4.5 / 68 % | |
| violent | 542 | +1.20 / +6.5 / 84 % | +0.31 / +3.5 / 83 % | +0.80 / +5.0 / 74 % | |
| 55 kg / 80 kg | 1562 / 1564 | +0.84 / +0.81 | +0.35 / +0.35 | +0.73 / +0.67 | |
| Hs 0.1 → 0.5 m | | +1.53, +0.65, +0.65, +0.78, +0.93 | | | |

- The gain is broad: every fold, both OOD tests, both vessels, every Hs, U and regime.
- It is largest at β = 0° (about 4× the gain at 135°), and largest of all at U = 1–2 with β = 0°: heave ≤2 s +26 % at U = 1.
- It is weakest in ood_beta for roll, where it reverses at U = 2: −0.20 / −3.1 / 23 %.
- Pitch is roughly a tie in some cells: U = 1, β = 135° and U = 2, β = 180°.

### 4. Physics equation or extra information?

The existing runs cannot answer this. Three things argue against crediting the equation itself:

- **One of the two extra loss terms needs no physics at all.** J_e only says the predicted velocity must match the time derivative of the predicted motion. It has no physics and no extra information, but its stencil crosses the past/future boundary. That fits the gain sitting at 0.2–1 s and fixing the no-physics network's 0.2 s weakness.
- **The other term carries almost no equation.** J_g is the only term with wave excitation f, and it dominates the extra loss (about 0.22 at epoch 30, against J_e about 0.002). But the restoring force g in it is a free LSTM that takes the run parameters as input, so it is close to an auxiliary task: predict the true wave forcing f, past and future, from Network 1's latent state.
- **The pattern doesn't tell the two explanations apart.** The gain dies after 5 s, so the true future f did not teach any lasting wave forecast. The gain is largest in following seas (β = 0°), where encounter periods are long. Both fit a short-horizon regularisation effect as well as a physics effect.

Separating them needs three ablations: J_e only, J_g with f replaced by zero or shuffled across runs in the future part, and J_g with a fixed linear g.

My scripts and outputs are in `C:\Users\huybl\AppData\Local\Temp\claude\D--USV-landing-prediction\712c7cfe-cce9-4431-b9c6-f794ff7cbf8c\scratchpad\results_check\`:
- `skeptic_phy.py` and its output `out.txt`
- `per_run.csv`
- `lead0.py`
- `rng_check.py`
- `rerun.py` with `rerun_cv0_{data_s0,data_s1,phy_s1}.npz/.json` and `logF_*.txt`
- `compare_rerun.py`