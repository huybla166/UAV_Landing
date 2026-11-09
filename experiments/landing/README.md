# Landing-window decisions (Wang et al. 2026, Definition 1)

Files:

- `digitize_wang_fig6.py`: reads Fig. 6 of `references/Wang2026_JFI_108381.pdf` from its vector paths and identifies the rule behind it (centred `movmean(x,20)` and `movstd(x,20)`, longest run where both are under the thresholds). Output goes to `results/sim/landing/wang_fig6_*`.
- `definition1.py`: the vectorised detector (per-sample stable mask, windows, the start decision) and the threshold sets `wang_literal`, `otter_scaled`, `otter_scaled_abs` and `otter_share`. `main()` checks that the detector reproduces Fig. 6 (samples 388..485) and writes truth statistics to `results/sim/landing/definition1_truth_stats.md`.
- `evaluate.py`: the go / no-go decision from each 20 s forecast, scored against the same decision on the true future. `report` rebuilds `results/sim/landing/landing_prototype.md` from `decisions_<split>.npz`. `prototype_notes.md` is inserted into that report.

Everything here is deterministic and runs on the CPU. The full cv0 run (518 runs, 4 built-in predictors) takes about 4 min.

## Plugging in a trained forecaster

`evaluate.py --plugin path/to/plugin.py:make --name <row name> --splits cv0,cv1,...` imports `make` from the plugin file and calls it once per split:

```python
def make(split: str):            # split in run_sim_lstm_cv.SPLITS: 'cv0'..'cv5', 'ood_Hs', 'ood_beta'
    ...                          # load the weights + normalisation saved for THIS split's training job
    def f(ctx):                  # ctx: np.ndarray (B, 300, 3) float64 = x[o-300:o], the past 60 s at 5 Hz
        ...                      #   channels as sim_common.motion: heave D (m, + down), roll (deg), pitch (deg)
        return pred              # np.ndarray (B, 100, 3): x_hat[o:o+100], same channels and units
    return f
```

`f` only ever sees the past 60 s, so the protocol cannot leak future samples. B is the 486 origins of one test run. Example for the paper LSTM, which uses the last 10 samples and is normalised with the train mean/std:

```python
import numpy as np, torch
from run_sim_lstm_cv import PaperLSTM
def make(split, seed=0):
    ck = torch.load(f"results/sim/weights/lstm_{split}_seed{seed}.pt", map_location="cpu")
    net = PaperLSTM(); net.load_state_dict(ck["state_dict"]); net.eval()
    mu, sd = ck["mu"], ck["sd"]                                    # numpy (3,), train statistics
    def f(ctx):
        h = torch.tensor((ctx[:, -10:] - mu) / sd, dtype=torch.float32); out = []
        with torch.no_grad():
            for _ in range(100):
                n = net(h); out.append(n); h = torch.cat([h[:, 1:], n[:, None]], 1)
        return torch.stack(out, 1).numpy() * sd + mu
    return f
```

The decision tables then get one row for the plugin next to `oracle`. `decisions_<split>.npz` keeps the rows saved earlier for the same runs, so the built-in rows stay in the report.

## What the training scripts must save to enable this

None of the current scripts writes reusable weights. `run_sim_lstm_cv.py` keeps `model, mu, sd` in memory only. `phylstm/phylstm_kaggle.py` holds the Network-1 `state_dict` in memory, at lines ~406-445. Per job (split x variant x seed), the author should add:

1. `torch.save({"state_dict": net1.state_dict(), "mu": mu, "sd": sd, "config": {...}}, path)` after the best-validation weights are restored. The config should hold the architecture arguments, the context length, the variant and the epochs. The path should be one file per job, for example `results/sim/weights/<model>_<split>_<variant>_s<seed>.pt`, and it must also be fetched from Kaggle with the other outputs.
2. The normalisation statistics exactly as the model's forward pass uses them. For PhyLSTM, that includes any physical-unit conversion (rad and deg: `to_report_units`) and the state packing (`job.windows` / `job.unpack`). The plugin's `f` must return heave (m) and roll / pitch (deg).
3. Optionally, a small `forecast_ctx(net1, ctx, stats)` helper in the training module, so the plugin and the training-time scoring share one code path. A plugin that re-implements the forward pass can drift from the scored model.

There is a fallback if saving weights is not possible. A job can dump its forecasts for the test runs instead, as `<split>|ids` (R,) plus `<split>|pred` (R, 486, 100, 3) float16, about 150 MB per split. This is **not implemented yet**: `f(ctx)` does not receive the run id, so `evaluate.py` would need a `--forecasts file.npz` option that bypasses `f`. The weights route is preferred.
