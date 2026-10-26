"""Cross-fitted calibration: out-of-sample scores for every non-test float.

Calibrating the float-level threshold on the 57 validation floats alone put the 5 % false-
alarm quantile on about three floats (validation 3/54 = 5.6 %, 95 % CI 1.2-15.4 %; clean test
8/73 = 11.0 %, CI 4.9-20.5 %). And a float's chance of a false alarm grows with the length
of its record, which differs between groups. So the pool is enlarged: the 299 training and
validation floats are split into K folds; for each fold both ensembles are retrained on the
other folds (same recipe as paper_qc.model, with an inner float-disjoint validation set for
early stopping) and score the held-out fold. Every non-test float then has out-of-sample
scores, and the thresholds come from all of them.

The test floats are still scored by the final models (paper_qc.model, trained on the train
group) and are never used here.

    python -m paper_qc.crossfit train      # K x 2 ensembles into runs_qc/cv/ (~3 h on CPU)
    python -m paper_qc.crossfit score      # out-of-fold scores -> data/qc_oof_scores.parquet
    python -m paper_qc.crossfit calibrate  # thresholds -> tables/score_thresholds_cv.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from paper_qc.lib import data as odata
from paper_qc import model, score, split

K = 5
SEED = 7
ROOT = os.path.join("runs_qc", "cv")
OOF = os.path.join("data", "qc_oof_scores.parquet")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CAL = os.path.join(HERE, "tables", "score_thresholds_cv.json")


def folds():
    s = split.load()
    pool = np.array(sorted(s["train"] + s["val"]))
    rng = np.random.default_rng(SEED)
    rng.shuffle(pool)
    return [sorted(pool[k::K].tolist()) for k in range(K)], pool


def train_all():
    fl, pool = folds()
    df = odata.load(verbose=False)
    df["PLATFORM"] = df.PLATFORM.astype(str).str.strip()
    rng = np.random.default_rng(SEED + 1)
    for k, held in enumerate(fl):
        rest = np.array([f for f in pool if f not in set(held)])
        rng.shuffle(rest)
        n_val = int(round(0.15 * len(rest)))
        inner_val, inner_train = sorted(rest[:n_val]), sorted(rest[n_val:])
        for target in ("DOXY", "PSAL"):
            out = os.path.join(ROOT, f"fold{k}", target.lower())
            if os.path.exists(os.path.join(out, "meta.json")):
                print(f"fold {k} {target}: exists, skipped", flush=True)
                continue
            print(f"fold {k} {target}: {len(held)} held out, {len(inner_train)} train, "
                  f"{len(inner_val)} inner val", flush=True)
            meta = model.train(target, out, train_floats=inner_train, val_floats=inner_val,
                               df=df)
            meta["held_out"] = held
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1)


def score_all():
    fl, _ = folds()
    prof = pd.read_parquet(score.profiles.OUT)
    prof = prof[prof.group.isin(["train", "val"])]
    parts = []
    for k, held in enumerate(fl):
        t0 = time.time()
        ens = {"D": model.Ensemble(os.path.join(ROOT, f"fold{k}", "doxy")),
               "S": model.Ensemble(os.path.join(ROOT, f"fold{k}", "psal"))}
        d = prof[prof.PLATFORM.isin(held)].reset_index(drop=True)
        zD, zS, _ = score.level_scores(d, score.predict(ens, d))
        parts.append(pd.DataFrame({"PLATFORM": d.PLATFORM, "CYCLE": d.CYCLE, "PRES": d.PRES,
                                   "group": d.group, "fold": k,
                                   "zD": zD.astype("float32"), "zS": zS.astype("float32")}))
        print(f"fold {k}: {d.PLATFORM.nunique()} floats, {len(d):,} levels, "
              f"{time.time() - t0:.0f}s", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    oof.to_parquet(OOF, index=False)
    print(f"wrote {OOF}")


def calibrate_all(method="window"):
    oof = pd.read_parquet(OOF)
    cal = score.calibrate_from_z(oof, oof.zD.to_numpy("float64"), oof.zS.to_numpy("float64"),
                                 method)
    cal["source"] = f"out-of-fold scores, {K}-fold cross-fitting over train+val floats"
    with open(OUT_CAL, "w") as f:
        json.dump(cal, f, indent=1)
    print(json.dumps({k: v for k, v in cal.items() if "threshold" in k or k.endswith("floats")},
                     indent=1))
    print(f"wrote {OUT_CAL}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train_all, "score": score_all, "calibrate": calibrate_all}[step]()
