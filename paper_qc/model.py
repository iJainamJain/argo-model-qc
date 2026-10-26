"""The two predictive models behind the QC scores, trained once and saved.

    DOXY  from PRES, TEMP, PSAL, position, season   (log target)
    PSAL  from PRES, TEMP, position, season         (a T-S consistency model)

Both are heteroscedastic deep ensembles (paper_qc.lib.nets), trained on the split's training
floats (paper_qc.split) and validated on its validation floats. The members are saved, not
only their predictions: the point here is to score data the models have never seen -
including data we corrupt afterwards.

    python -m paper_qc.model --target DOXY --out runs_qc/doxy
    python -m paper_qc.model --target PSAL --out runs_qc/psal
    python -m paper_qc.model --target PSAL --out <scratch> --members 1 --epochs 2 --max-train 50000
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from paper_qc.lib import data as odata
from paper_qc.lib import nets as models
from paper_qc import split

LOG_OFFSET = 5.0   # log(O2 + 5) keeps near-zero and slightly negative oxygen finite
INPUTS = {
    "DOXY": ["PRES", "TEMP", "PSAL", "LATITUDE", "LONGITUDE", "sin_doy", "cos_doy"],
    "PSAL": ["PRES", "TEMP", "LATITUDE", "LONGITUDE", "sin_doy", "cos_doy"],
}


def design(df, target):
    """Input matrix for `target` from a level table (TIME as datetime64 or epoch seconds)."""
    t = df.TIME
    if not np.issubdtype(t.dtype, np.datetime64):
        t = pd.to_datetime(t, unit="s")
    doy = pd.DatetimeIndex(t).dayofyear.to_numpy()
    cols = {"sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25)}
    return np.column_stack([cols[c] if c in cols else df[c].to_numpy("float64")
                            for c in INPUTS[target]]).astype("float32")


def forward(y, target):
    return np.log(np.maximum(y + LOG_OFFSET, 1e-3)) if target == "DOXY" else y


def inverse(mu, sd, target):
    """Back to physical units; delta method for sigma under the log transform."""
    if target == "DOXY":
        mu = np.exp(mu) - LOG_OFFSET
        return mu, sd * (mu + LOG_OFFSET)
    return mu, sd


def train(target, out, members=5, epochs=60, max_train=1_500_000, seed=1,
          train_floats=None, val_floats=None, df=None):
    """Fit and save one ensemble. Floats default to the QC split's train / val groups."""
    models.configure_gpu()
    if df is None:
        df = odata.load(verbose=False)
        df["PLATFORM"] = df.PLATFORM.astype(str).str.strip()
    s = split.load()
    tr = df[df.PLATFORM.isin(train_floats if train_floats is not None else s["train"])]
    va = df[df.PLATFORM.isin(val_floats if val_floats is not None else s["val"])]
    if max_train and len(tr) > max_train:
        tr = tr.sample(max_train, random_state=seed)
    Xtr, Xva = design(tr, target), design(va, target)
    ytr, yva = forward(tr[target].to_numpy("float64"), target), forward(va[target].to_numpy("float64"), target)
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0)
    x_sd[x_sd == 0] = 1.0
    y_mu, y_sd = float(ytr.mean()), float(ytr.std())
    print(f"{target}: training on {len(tr):,} levels of {tr.PLATFORM.nunique()} floats, "
          f"validating on {len(va):,} of {va.PLATFORM.nunique()}", flush=True)

    t0 = time.time()
    nets, n_epochs = models.fit_ensemble(
        ((Xtr - x_mu) / x_sd).astype("float32"), ((ytr - y_mu) / y_sd).astype("float32"),
        ((Xva - x_mu) / x_sd).astype("float32"), ((yva - y_mu) / y_sd).astype("float32"),
        n_members=members, base_seed=seed * 100, epochs=epochs)
    os.makedirs(out, exist_ok=True)
    for i, net in enumerate(nets):
        net.save(os.path.join(out, f"member_{i}.keras"))
    meta = {"target": target, "inputs": INPUTS[target], "x_mu": x_mu.tolist(),
            "x_sd": x_sd.tolist(), "y_mu": y_mu, "y_sd": y_sd, "members": members,
            "epochs": n_epochs, "max_train": max_train, "seed": seed,
            "split_seed": s["seed"], "n_train": int(len(tr)), "n_val": int(len(va)),
            "seconds": round(time.time() - t0)}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    mu, sd = Ensemble(out).predict(va)
    err = va[target].to_numpy("float64") - mu
    z = err / sd
    print(f"  epochs {n_epochs}, {meta['seconds']}s | validation MAE {np.mean(np.abs(err)):.4f}, "
          f"median sigma {np.median(sd):.4f}, |z|<1.645: {np.mean(np.abs(z) < 1.645):.3f}",
          flush=True)
    return meta


class Ensemble:
    """A saved ensemble; predict(df) returns (mu, sigma) in physical units."""

    def __init__(self, path):
        import tensorflow as tf
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.target = self.meta["target"]
        self.nets = [tf.keras.models.load_model(os.path.join(path, f"member_{i}.keras"),
                                                compile=False)
                     for i in range(self.meta["members"])]
        self.x_mu = np.asarray(self.meta["x_mu"], "float32")
        self.x_sd = np.asarray(self.meta["x_sd"], "float32")

    def predict(self, df):
        X = design(df, self.target)
        ok = np.isfinite(X).all(axis=1)
        mu = np.full(len(df), np.nan)
        sd = np.full(len(df), np.nan)
        if ok.any():
            m, s, _ = models.predict_ensemble(self.nets, (X[ok] - self.x_mu) / self.x_sd)
            m = m * self.meta["y_sd"] + self.meta["y_mu"]
            s = s * self.meta["y_sd"]
            mu[ok], sd[ok] = inverse(m, s, self.target)
        return mu, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--target", choices=list(INPUTS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-train", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    train(a.target, a.out, a.members, a.epochs, a.max_train, a.seed)


if __name__ == "__main__":
    main()
