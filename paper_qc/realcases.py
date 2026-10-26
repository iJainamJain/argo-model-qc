"""Real errors, scored in the real-time view (raw values, as real time saw them).

A  Greylisted salinity. For each exclusion-list window on a known-bad float: the deep
   (>= 1000 dbar) salinity score before and after the window's start date, and when the
   float-level CUSUM and the per-profile score first alarm relative to that date. The start
   date is the operator's later judgement of when the problem began, so an alarm before it
   is either early warning or a false alarm - both are reported, not interpreted away.
B  Float 2902209 (salinity 14.5 at 2000 dbar under flags 1-2; not on the greylist).
C  Optode gain. For every profile with raw oxygen, the model's real-time gain estimate
   median(predicted / raw) against the gain delayed-mode processing later applied
   (tables/adjustments.csv.gz). Baseline: assume the array median gain for every profile.
   Judge it with robust statistics: one float with impossible salinity (2902209) gets a
   gain estimate of ~15 and alone drags a Pearson correlation to ~0.

Thresholds are those calibrated on the validation floats by score.py
(tables/score_thresholds.json); nothing here is tuned on these floats.

    python -m paper_qc.realcases --doxy runs_qc/doxy --psal runs_qc/psal
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from paper_qc import profiles, rtqc
from paper_qc.model import Ensemble
from paper_qc.score import (DEEP, OUT_CAL, float_cusum, profile_means, profile_score)

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
GREYLIST = os.path.join("data", "argo_greylist", "ar_greylist.txt")
ADJ = os.path.join(TABLES, "adjustments.csv.gz")
BAD_QC = {"4"}


def raw_view(group=None):
    df = pd.read_parquet(profiles.OUT_RAW)
    if group:
        df = df[df.group == group]
    # levels real time would have passed on: T and S not flagged bad in the raw flags
    ok = ~df.TEMP_QC.isin(BAD_QC) & ~df.PRES_QC.isin(BAD_QC)
    return df[ok].reset_index(drop=True)


def zs_scores(df, ens_s, ens_d=None):
    mu, sd = ens_s.predict(df)
    zS = (df.PSAL.to_numpy("float64") - mu) / sd
    zD = np.full(len(df), np.nan)
    if ens_d is not None:
        has = df.DOXY.notna().to_numpy()
        if has.any():
            m, s = ens_d.predict(df[has])
            zD[has] = (df.DOXY.to_numpy("float64")[has] - m) / s
    return zD, zS


def part_a(df, zD, zS, cal):
    g = pd.read_csv(GREYLIST, dtype=str)
    g = g[(g.PARAMETER_NAME == "PSAL") & g.PLATFORM_CODE.isin(set(df.PLATFORM))]
    g["start"] = pd.to_datetime(g.START_DATE, format="%Y%m%d")

    # salinity only: raw oxygen carries the optode gain and would confound the float score
    none = np.full(len(df), np.nan)
    ps = profile_score(profile_means(df, none, zS), cal)
    deep = profile_score(profile_means(df, none, zS, min_pres=DEEP), cal, "deep")
    c = float_cusum(deep, cal)
    when = df.groupby(["PLATFORM", "CYCLE"]).TIME.first()

    rows = []
    for _, e in g.iterrows():
        f = e.PLATFORM_CODE
        if f not in deep.index.get_level_values(0):
            continue
        t = when.loc[f]
        before, after = (t < e.start).to_numpy(), (t >= e.start).to_numpy()
        # not every profile reaches the deep window, so align everything on cycle
        d = deep.loc[f].s.reindex(t.index)
        prof = (ps.loc[f].score > cal["profile_threshold"]).reindex(t.index, fill_value=False)
        cf = (c.loc[f] > cal["cusum_threshold"]).reindex(t.index, fill_value=False)

        def first(mask):
            hit = t[mask.to_numpy(bool)]
            return hit.min() if len(hit) else pd.NaT

        fc, fp = first(cf), first(prof)
        rows.append({
            "float": f, "code": e.QUALITY_CODE, "start": e.START_DATE, "comment": e.COMMENT,
            "profiles_before": int(before.sum()), "profiles_after": int(after.sum()),
            "deep_score_before": float(d[before].mean()) if before.any() else np.nan,
            "deep_score_after": float(d[after].mean()) if after.any() else np.nan,
            "profile_alarms_before": float(prof[before].mean()) if before.any() else np.nan,
            "profile_alarms_after": float(prof[after].mean()) if after.any() else np.nan,
            "first_cusum_alarm": fc, "cusum_lead_days": (e.start - fc).days if pd.notna(fc) else np.nan,
            "first_profile_alarm": fp,
            "profile_lead_days": (e.start - fp).days if pd.notna(fp) else np.nan,
        })
    return pd.DataFrame(rows)


def part_b(df, zS, cal):
    f = df.PLATFORM == "2902209"
    if not f.any():
        return {}
    sub = df[f]
    z = np.abs(zS[f.to_numpy()])
    fresh = (sub.PSAL < 30).to_numpy()
    res = rtqc.run(sub[["PLATFORM", "CYCLE", "PRES", "TEMP", "PSAL", "DOXY", "LATITUDE",
                        "LONGITUDE"]])
    return {
        "levels": int(len(sub)), "levels_psal_below_30": int(fresh.sum()),
        "raw_flags_on_those": sub.PSAL_QC[fresh].value_counts().to_dict(),
        "median_abs_zS_on_those": float(np.nanmedian(z[fresh])) if fresh.any() else None,
        "median_abs_zS_elsewhere": float(np.nanmedian(z[~fresh])),
        "model_flags_on_those": float(np.mean(z[fresh] > cal["level_threshold"])) if fresh.any() else None,
        "rtqc_flags_on_those": float(res.fail_PSAL.to_numpy()[fresh].mean()) if fresh.any() else None,
        "rtqc_tests_on_those": res.test_PSAL[fresh].value_counts().to_dict(),
    }


def part_c(df, ens_d):
    has = df.DOXY.notna().to_numpy() & (df.DOXY.to_numpy() > 5) & ~df.PSAL_QC.isin(BAD_QC).to_numpy()
    sub = df[has].reset_index(drop=True)
    mu, _ = ens_d.predict(sub)
    sub["ratio"] = mu / sub.DOXY.to_numpy("float64")
    est = sub.groupby(["PLATFORM", "CYCLE"]).ratio.median().rename("model_gain").reset_index()
    adj = pd.read_csv(ADJ, dtype={"platform": str})
    adj = adj[adj.mode_DOXY == "D"][["platform", "cycle", "ratio_DOXY"]].dropna()
    adj.columns = ["PLATFORM", "CYCLE", "dm_gain"]
    m = est.merge(adj, on=["PLATFORM", "CYCLE"])
    array_median = float(adj.dm_gain.median())
    err_model = m.model_gain - m.dm_gain
    err_const = array_median - m.dm_gain
    return m, {
        "profiles": int(len(m)), "floats": int(m.PLATFORM.nunique()),
        "array_median_gain": array_median,
        "corr_model_vs_dm": float(np.corrcoef(m.model_gain, m.dm_gain)[0, 1]),
        "mae_model": float(err_model.abs().mean()), "mae_array_median": float(err_const.abs().mean()),
        # robust per-float view (a mean or Pearson r is decided by a single float fed
        # impossible salinity, whose gain estimate is meaningless)
        "spearman_per_float": float(m.groupby("PLATFORM")[["model_gain", "dm_gain"]].median()
                                    .corr(method="spearman").iloc[0, 1]),
        "median_abs_error_model": float(err_model.abs().median()),
        "median_abs_error_array_median": float(err_const.abs().median()),
        "within_2pct_model": float((err_model.abs() < 0.02).mean()),
        "within_2pct_array_median": float((err_const.abs() < 0.02).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doxy", default=os.path.join("runs_qc", "doxy"))
    ap.add_argument("--psal", default=os.path.join("runs_qc", "psal"))
    ap.add_argument("--cal", default=OUT_CAL)
    ap.add_argument("--dry", action="store_true", help="smoke models: print, write nothing")
    a = ap.parse_args()
    with open(a.cal) as f:
        cal = json.load(f)
    ens_s, ens_d = Ensemble(a.psal), Ensemble(a.doxy)

    bad = raw_view("test_known_bad")
    zD, zS = zs_scores(bad, ens_s)
    ta = part_a(bad, zD, zS, cal)
    print("A  greylisted salinity windows:")
    print(ta.drop(columns=["comment"]).to_string(index=False))
    b = part_b(bad, zS, cal)
    print("\nB  float 2902209:", json.dumps(b, indent=1, default=str))

    both = pd.concat([bad, raw_view("test_clean")], ignore_index=True)
    m, c = part_c(both, ens_d)
    print("\nC  optode gain, model vs delayed mode:", json.dumps(c, indent=1))
    if not a.dry:
        ta.to_csv(os.path.join(TABLES, "real_greylist.csv"), index=False)
        m.to_csv(os.path.join(TABLES, "real_gain.csv.gz"), index=False)
        with open(os.path.join(TABLES, "real_summary.json"), "w") as f:
            json.dump({"float_2902209": b, "gain": c}, f, indent=1, default=str)
        print("wrote tables/real_greylist.csv, real_gain.csv.gz, real_summary.json")


if __name__ == "__main__":
    main()
