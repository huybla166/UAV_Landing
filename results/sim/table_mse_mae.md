Table. Comparison of MSE and MAE for different models and prediction horizons (dataset_v2, 6-fold CV, z-score normalised, mean over heave, roll and pitch)

| Model | 2s Horizon MSE | 2s Horizon MAE | 5s Horizon MSE | 5s Horizon MAE | 10s Horizon MSE | 10s Horizon MAE | 20s Horizon MSE | 20s Horizon MAE |
|---|---|---|---|---|---|---|---|---|
| LSTM [Wang 2026] | 0.4508 | 0.4496 | 0.8111 | 0.6266 | 0.9319 | 0.6850 | 0.9771 | 0.7086 |
| PhyLSTM w/o physics | 0.3309 | 0.3883 | 0.6233 | 0.5496 | 0.7981 | 0.6325 | 0.9015 | 0.6790 |
| PhyLSTM2 | 0.3148 | 0.3747 | 0.6076 | 0.5401 | 0.7877 | 0.6268 | 0.8959 | 0.6761 |
