"""Leakage checks of evaluation protocol (a) on dataset_v2: grouped 6-fold CV + two OOD tests.

  python -m pytest experiments/test_sim_splits.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_common as S  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(S.CACHE, "meta.csv"))
                                or not os.path.exists(os.path.join(S.SIM, "folds.csv")), reason="dataset_v2 cache or folds.csv missing")


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(os.path.join(S.CACHE, "meta.csv"))


def seeds(meta, idx):
    return set(meta.seed.to_numpy()[idx])


def test_cv_folds_are_disjoint_and_cover_everything(meta):
    tested = []
    for k in range(S.K_FOLDS):
        sp = S.cv_split(meta, k)
        rows = [set(sp[p]) for p in ("train", "val", "test")]
        assert not (rows[0] & rows[1]) and not (rows[0] & rows[2]) and not (rows[1] & rows[2])
        assert len(rows[0] | rows[1] | rows[2]) == len(meta)
        s = [seeds(meta, sp[p]) for p in ("train", "val", "test")]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2]), "a sea on two sides"
        tested += list(sp["test"])
    assert sorted(tested) == list(range(len(meta))), "every run must be tested exactly once"


def test_every_test_fold_has_every_condition(meta):
    for k in range(S.K_FOLDS):
        t = meta.iloc[S.cv_split(meta, k)["test"]]
        assert t.Hs.nunique() == 5 and t.beta_deg.nunique() == 5 and t.U.nunique() == 3 and t.vessel.nunique() == 2


@pytest.mark.parametrize("which", ["Hs", "beta"])
def test_ood_level_is_unseen(meta, which):
    col, level = S.OOD[which]
    sp = S.ood_split(meta, which)
    for part in ("train", "val"):
        assert not np.isclose(meta[col].to_numpy(float)[sp[part]], level).any()
    assert np.isclose(meta[col].to_numpy(float)[sp["test"]], level).all()
    assert len(sp["test"]) == 630
    assert not (seeds(meta, sp["train"]) & seeds(meta, sp["test"]))
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == len(meta)
