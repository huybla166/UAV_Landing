"""TiRex (NeurIPS 2025) zero-shot forecasts on the NTUA data: no training, univariate per channel.
Context lengths: 150 samples (30 s) and 300 samples (60 s). Saves mean forecasts and the 9 quantiles
(0.1 ... 0.9) so the interval coverage can be checked before any conformal calibration.
"""
import sys
import time

import numpy as np
import torch
from tirex import load_model

from common import HORIZON, RUNS, load_all, origins, save_preds

LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 1)


def main(context_lengths):
    torch.set_num_threads(4)
    data = load_all()
    model = load_model("NX-AI/TiRex")
    for L in context_lengths:
        t0 = time.time()
        preds, quants = {}, {}
        for r, x in data.items():
            o = origins(len(x))
            pred = np.empty((len(o), HORIZON, 3))
            q = np.empty((len(o), HORIZON, 3, len(LEVELS)))
            for c in range(3):
                ctx = torch.tensor(np.stack([x[i - L:i, c] for i in o]), dtype=torch.float32)
                qq, mean = model.forecast(context=ctx, prediction_length=HORIZON, batch_size=64)
                pred[:, :, c] = mean.numpy()
                q[:, :, c, :] = qq.numpy()
            preds[r], quants[r] = pred, q
            print(f"  tirex ctx={L} run {r} done ({time.time() - t0:.0f}s)", flush=True)
        save_preds(f"tirex_ctx{L}", preds, quants, LEVELS)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [150, 300])
