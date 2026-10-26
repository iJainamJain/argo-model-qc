"""The benchmark with repeats: every scenario planted with several seeds, per-float records kept.

Each repeat draws a different corrupted half of the clean test floats, different onsets and
signs (inject.inject with seed = crc32(scenario#rep)), and both detectors see the identical
corruption. Instead of one summary row per scenario, this keeps

  profiles  one row per (scenario, rep, profile): its level counts (bad / flagged), whether
            the profile is bad and flagged, and its distance to the nearest training float
  floats    one row per (scenario, rep, float): corrupted?, float alarm, delay

so paper_qc.ci can put float-bootstrap confidence intervals on every rate, and detection can
be broken down by distance to training data without another run.

    python -m paper_qc.bench model --reps 5    # needs tables/score_thresholds_cv.json
    python -m paper_qc.bench rtqc --reps 5
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from paper_qc import inject, rtqc, score
from paper_qc.model import Ensemble

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
EARTH_R = 6371.0
COLS = ["PLATFORM", "CYCLE", "TIME", "PRES", "TEMP", "PSAL", "DOXY", "LATITUDE", "LONGITUDE"]


def seed_for(s, rep):
    return zlib.crc32(f"{s.name}#{rep}".encode()) % (2 ** 31)


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def distance_to_training(df):
    """Great-circle km from each profile of df to the nearest training-float profile."""
    tr = pd.read_parquet(score.profiles.OUT, columns=["PLATFORM", "CYCLE", "LATITUDE", "LONGITUDE"],
                         filters=[("group", "==", "train")])
    tr = tr.groupby(["PLATFORM", "CYCLE"])[["LATITUDE", "LONGITUDE"]].first()
    tree = cKDTree(_unit(tr.LATITUDE.to_numpy(), tr.LONGITUDE.to_numpy()))
    p = df.groupby(["PLATFORM", "CYCLE"])[["LATITUDE", "LONGITUDE"]].first()
    chord, _ = tree.query(_unit(p.LATITUDE.to_numpy(), p.LONGITUDE.to_numpy()))
    return pd.Series(2 * EARTH_R * np.arcsin(np.clip(chord / 2, 0, 1)), index=p.index,
                     name="dist_km")


def records(df, truth, floats, target, level_flag, level_valid, prof_flag, float_alarm, meta):
    """Per-profile and per-float records for one (scenario, rep, detector).

    level_flag / level_valid: bool arrays over levels (valid = the detector produced a verdict)
    prof_flag: bool Series indexed by (PLATFORM, CYCLE), or None
    float_alarm: bool Series indexed by (PLATFORM, CYCLE) - alarm state at each profile
    """
    bad = truth[f"bad_{target}"].to_numpy()
    key = [df.PLATFORM.to_numpy(), df.CYCLE.to_numpy()]
    lv = pd.DataFrame({"bad": bad & level_valid, "bad_flag": bad & level_flag & level_valid,
                       "clean": ~bad & level_valid, "clean_flag": ~bad & level_flag & level_valid})
    p = lv.groupby(key).sum()
    p.index.names = ["PLATFORM", "CYCLE"]
    p.columns = ["lv_bad", "lv_bad_flag", "lv_clean", "lv_clean_flag"]
    p["prof_bad"] = pd.Series(bad).groupby(key).any().to_numpy()
    p["prof_flag"] = (prof_flag.reindex(p.index, fill_value=False).to_numpy()
                      if prof_flag is not None else False)
    p = p.reset_index()
    for k, v in meta.items():
        p[k] = v

    rows = []
    corrupted = dict(zip(floats.platform, floats.corrupted))
    onset = dict(zip(floats.platform, floats.onset_cycle))
    for f, a in float_alarm.groupby(level="PLATFORM"):
        a = a.droplevel(0).sort_index()
        cycles = a.index.to_numpy()
        on = onset.get(f, -1)
        start = on if (corrupted.get(f) and on >= 0) else cycles.min()
        after = a[cycles >= start]
        hit = bool(after.any())
        delay = np.nan
        if hit and corrupted.get(f):
            first = after.index[after.to_numpy()][0]
            delay = int(np.searchsorted(cycles, first) - np.searchsorted(cycles, start))
        # profiles at risk before the first alarm (after the reference profiles), for the
        # false-alarm rate per 100 profiles on clean floats
        x = a.to_numpy()[score.REF_N:]
        first = np.flatnonzero(x)
        at_risk = int(first[0]) + 1 if len(first) else len(x)
        rows.append({"PLATFORM": f, "corrupted": bool(corrupted.get(f)), "alarm": hit,
                     "delay": delay, "profiles": len(a), "at_risk": at_risk,
                     "first_alarm": bool(len(first)), **meta})
    return p, pd.DataFrame(rows)


def run_model(df, reps, scenarios, cal):
    ens = {"D": Ensemble(os.path.join("runs_qc", "doxy")),
           "S": Ensemble(os.path.join("runs_qc", "psal"))}
    clean_pred = score.predict(ens, df)
    P, F = [], []
    for s in scenarios:
        target = inject.TARGET_OF.get(s.kind, s.var)
        for rep in range(reps):
            t0 = time.time()
            out, truth, floats = inject.inject(df, s, seed=seed_for(s, rep))
            sel = out.PLATFORM.isin(floats.platform[floats.corrupted]).to_numpy()
            pred = {k: (v[0].copy(), v[1].copy()) for k, v in clean_pred.items()}
            if sel.any():
                p2 = score.predict(ens, out[sel].reset_index(drop=True))
                for k in pred:
                    pred[k][0][sel], pred[k][1][sel] = p2[k]
            zD, zS, both = score.level_scores(out, pred)
            ps = score.profile_score(score.profile_means(out, zD, zS), cal)
            deep = score.profile_score(score.profile_means(out, zD, zS, min_pres=score.DEEP),
                                       cal, "deep")
            fa = score.float_stat(deep, cal) > cal["cusum_threshold"]
            # every profile gets an alarm state, deep window or not
            idx = out.groupby(["PLATFORM", "CYCLE"]).size().index
            fa = fa.reindex(idx, fill_value=False)
            meta = {"scenario": s.name, "kind": s.kind, "variable": s.var,
                    "magnitude": s.magnitude, "rep": rep, "detector": "model"}
            p, f = records(out, truth, floats, target, both > cal["level_threshold"],
                           np.isfinite(both), ps.score > cal["profile_threshold"], fa, meta)
            P.append(p)
            F.append(f)
        print(f"  {s.name:<22} x{reps}  {time.time() - t0:.0f}s/rep", flush=True)
    return pd.concat(P, ignore_index=True), pd.concat(F, ignore_index=True)


def run_rtqc(df, reps, scenarios):
    clean = rtqc.run(df)
    P, F = [], []
    for s in scenarios:
        target = inject.TARGET_OF.get(s.kind, s.var)
        for rep in range(reps):
            t0 = time.time()
            out, truth, floats = inject.inject(df, s, seed=seed_for(s, rep))
            res = clean.copy()
            sel = out.PLATFORM.isin(floats.platform[floats.corrupted]).to_numpy()
            res.loc[sel] = rtqc.run(out[sel]).to_numpy()
            if target == "PRES":
                flag = res[[c for c in res if c.startswith("fail_")]].any(axis=1).to_numpy()
                before = clean[[c for c in clean if c.startswith("fail_")]].any(axis=1).to_numpy()
            else:
                flag, before = res[f"fail_{target}"].to_numpy(), clean[f"fail_{target}"].to_numpy()
            new = flag & ~before          # flags the planted error caused
            key = [out.PLATFORM.to_numpy(), out.CYCLE.to_numpy()]
            pf = pd.Series(new).groupby(key).any()
            pf.index.names = ["PLATFORM", "CYCLE"]
            meta = {"scenario": s.name, "kind": s.kind, "variable": s.var,
                    "magnitude": s.magnitude, "rep": rep, "detector": "rtqc"}
            valid = np.ones(len(out), bool) if target == "PRES" else out[target].notna().to_numpy()
            # level_flag counts caused flags on bad levels, raw flags on clean ones (false alarms)
            lvl = np.where(truth[f"bad_{target}"].to_numpy(), new, flag)
            p, f = records(out, truth, floats, target, lvl, valid, pf, pf, meta)
            P.append(p)
            F.append(f)
        print(f"  {s.name:<22} x{reps}  {time.time() - t0:.0f}s/rep", flush=True)
    return pd.concat(P, ignore_index=True), pd.concat(F, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detector", choices=["model", "rtqc"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cal", default=os.path.join(TABLES, "score_thresholds_cv.json"))
    ap.add_argument("--floats", type=int, default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    df = score.load_group("test_clean")
    if a.floats:
        df = df[df.PLATFORM.isin(sorted(df.PLATFORM.unique())[: a.floats])]
    df = df[COLS].reset_index(drop=True)
    wanted = [s for s in inject.SCENARIOS if not a.scenarios or s.name in a.scenarios]
    print(f"{a.detector}: {len(wanted)} scenarios x {a.reps} reps on "
          f"{df.PLATFORM.nunique()} floats", flush=True)
    if a.detector == "model":
        with open(a.cal) as f:
            cal = json.load(f)
        P, F = run_model(df, a.reps, wanted, cal)
    else:
        P, F = run_rtqc(df, a.reps, wanted)
    dist = distance_to_training(df).reset_index()
    P = P.merge(dist, on=["PLATFORM", "CYCLE"], how="left")
    tag = a.tag or ("" if not (a.floats or a.scenarios) else "_check")
    P.to_parquet(os.path.join(TABLES, f"records_{a.detector}_profiles{tag}.parquet"), index=False)
    F.to_parquet(os.path.join(TABLES, f"records_{a.detector}_floats{tag}.parquet"), index=False)
    print(f"wrote records_{a.detector}_*{tag}.parquet: {len(P):,} profile rows, {len(F):,} float rows")


if __name__ == "__main__":
    main()
