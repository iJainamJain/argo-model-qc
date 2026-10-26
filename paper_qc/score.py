"""Model-based QC scores, their thresholds, and the injection benchmark.

Scores, from the two saved ensembles (paper_qc.model):
    level    z = (observed - predicted) / sigma, for oxygen (zD) and for salinity (zS);
             one combined score max(|zD|, |zS|). A corrupted T or S moves zS directly and zD
             through the oxygen model's inputs; a corrupted oxygen moves zD.
    profile  the mean signed zD and zS over the profile's levels, in units of their spread
             on validation floats; score = the larger magnitude. Averaging turns a small
             offset shared by hundreds of levels into a large profile signal.
    float    a two-sided CUSUM (Page 1954) over each float's deep (>= 1000 dbar) profile
             means, relative to that float's own first profiles, which accumulates small persistent shifts - the
             drift case - into an alarm.

Every threshold is calibrated on the validation floats, never on test data: the level and
profile thresholds at their 99th percentile, the CUSUM limit so that 5 % of validation floats
alarm at some point in their life. The same thresholds are then applied to the clean test
floats (false alarms) and to every injected scenario (detection).

    python -m paper_qc.score --doxy runs_qc/doxy --psal runs_qc/psal
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

from paper_qc import inject, profiles
from paper_qc.model import Ensemble

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "score_benchmark.csv")
OUT_CAL = os.path.join(HERE, "tables", "score_thresholds.json")
LEVEL_Q, PROFILE_Q, FLOAT_FA = 0.99, 0.99, 0.05
FLOAT_HAZARD = 0.0005  # first false alarms per at-risk profile: ~5 % chance per 100 profiles
GROSS_Z, GROSS_FRAC = 10.0, 0.10  # evidently failed float: >10 % of profiles beyond 10 robust sd
CUSUM_K = 0.5
REF_N = 10           # profiles that define each float's own normal
WINDOW = 5           # profiles in the moving-average float statistic
DEEP = 1000.0        # dbar; the float-level statistic uses deep water only, as delayed-mode
                     # salinity correction does (Owens and Wong 2009): stable water, small bias


def load_group(group):
    return pd.read_parquet(profiles.OUT, filters=[("group", "==", group)]).reset_index(drop=True)


def predict(ens, df):
    """{'D': (mu, sd), 'S': (mu, sd)}; oxygen only where oxygen was observed."""
    out = {}
    has_o2 = df.DOXY.notna().to_numpy()
    mu = np.full(len(df), np.nan)
    sd = np.full(len(df), np.nan)
    if has_o2.any():
        m, s = ens["D"].predict(df[has_o2])
        mu[has_o2], sd[has_o2] = m, s
    out["D"] = (mu, sd)
    out["S"] = ens["S"].predict(df)
    return out


def level_scores(df, pred):
    zD = (df.DOXY.to_numpy("float64") - pred["D"][0]) / pred["D"][1]
    zS = (df.PSAL.to_numpy("float64") - pred["S"][0]) / pred["S"][1]
    both = np.fmax(np.abs(zD), np.abs(zS))       # fmax ignores the NaN where no oxygen
    return zD, zS, both


def profile_means(df, zD, zS, min_pres=None):
    """Mean zD and zS per profile; with min_pres, over levels at or below that pressure only."""
    g = pd.DataFrame({"PLATFORM": df.PLATFORM.to_numpy(), "CYCLE": df.CYCLE.to_numpy(),
                      "zD": zD, "zS": zS})
    if min_pres is not None:
        g = g[df.PRES.to_numpy() >= min_pres]
    return g.groupby(["PLATFORM", "CYCLE"], sort=True)[["zD", "zS"]].mean()


def profile_score(pm, cal, prefix="pm"):
    d = (pm.zD - cal[f"{prefix}_mu_D"]) / cal[f"{prefix}_sd_D"]
    s = (pm.zS - cal[f"{prefix}_mu_S"]) / cal[f"{prefix}_sd_S"]
    return pd.DataFrame({"d": d, "s": s, "score": np.fmax(d.abs(), s.abs())})


def cusum(x, k=CUSUM_K):
    """Largest two-sided CUSUM statistic reached at each step (NaN steps skipped)."""
    hi = lo = 0.0
    out = np.zeros(len(x))
    for i, v in enumerate(x):
        if np.isfinite(v):
            hi = max(0.0, hi + v - k)
            lo = max(0.0, lo - v - k)
        out[i] = max(hi, lo)
    return out


def float_stat(ps, cal, method=None):
    """Per profile, a running float-level statistic of change from the float's own start.

    Every float carries its own model bias, and it wanders slowly as the float crosses water
    masses: on validation floats the deep profile means have a lag-1 autocorrelation of
    0.74-0.85. So the spread of a float's first few profiles understates its variability and
    must not be the yardstick. Each float is referenced to the mean of its first REF_N
    profiles (floats generally start healthy; no planted error begins before 20 % of a float's
    life), and the change is measured in units of the typical within-float spread on the
    validation floats (cal["within_d"], cal["within_s"]).

        window  |mean of the last WINDOW profiles - reference|, a moving-average control chart
        cusum   two-sided CUSUM of the change, allowance CUSUM_K (Page 1954)

    Which one is used is decided on errors planted in validation floats (--select), never on
    test floats. Errors present from deployment are the per-profile score's job.
    """
    method = method or cal.get("float_method", "window")
    k = cal.get("cusum_k", CUSUM_K)
    w = int(cal.get("window", WINDOW))
    res = []
    for _, g in ps.groupby(level="PLATFORM", sort=False):
        out = np.zeros(len(g))
        for col, spread in (("d", cal["within_d"]), ("s", cal["within_s"])):
            x = g[col].to_numpy()
            ref = x[:REF_N][np.isfinite(x[:REF_N])]
            if len(ref) < 3:
                continue
            z = (x - ref.mean()) / spread
            if method == "cusum":
                stat = cusum(z, k)
            else:
                stat = np.abs(pd.Series(z).rolling(w, min_periods=w).mean().to_numpy())
                stat[:REF_N] = 0.0          # no alarms inside the reference itself
            out = np.fmax(out, np.nan_to_num(stat))
        res.append(pd.Series(out, index=g.index))
    return pd.concat(res)


def float_cusum(ps, cal=None, method=None):
    """Backwards-compatible name used by realcases.py."""
    return float_stat(ps, cal, method)


def calibrate(val, ens, method="window", **params):
    pred = predict(ens, val)
    zD, zS, _ = level_scores(val, pred)
    return calibrate_from_z(val, zD, zS, method, **params)


def _robust(x):
    """Median and MAD-based scale (1.4826 x MAD): a few failed floats cannot move them."""
    x = np.asarray(x, "float64")
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def first_alarm_index(c, threshold):
    """Per float: (profiles at risk before the first alarm, alarmed?).

    At-risk profiles start after the REF_N reference profiles, which cannot alarm.
    """
    out = {}
    for f, g in c.groupby(level="PLATFORM", sort=False):
        x = g.to_numpy()[REF_N:]
        hit = np.flatnonzero(x > threshold)
        out[f] = (int(hit[0]) + 1, True) if len(hit) else (len(x), False)
    return out


def hazard_threshold(c, target=FLOAT_HAZARD):
    """Smallest threshold whose rate of first false alarms per at-risk profile <= target.

    A threshold on each float's whole-life maximum makes the false-alarm probability depend
    on how long the float lives - unknown in real time, and different between float groups.
    Controlling the rate per profile (a control chart's in-control run length) does not.
    """
    grid = np.quantile(c[c > 0], np.linspace(0.5, 0.99999, 400)) if (c > 0).any() else [1.0]
    for h in np.unique(grid):
        fa = first_alarm_index(c, h)
        risk = sum(v[0] for v in fa.values())
        alarms = sum(v[1] for v in fa.values())
        if risk and alarms / risk <= target:
            return float(h), alarms / risk, alarms / len(fa)
    return float(np.max(c)), 0.0, 0.0


def calibrate_from_z(val, zD, zS, method="window", window=WINDOW, cusum_k=CUSUM_K):
    """Thresholds from scores already computed (e.g. out-of-fold, paper_qc.crossfit).

    Robust throughout: calibration pools are real archive data and contain failed floats
    the flags missed (the cross-fitted pool held two with salinity profile means ~250
    standard deviations out, inflating a plain standard deviation 7.5-fold). Floats whose
    profiles are grossly off on more than GROSS_FRAC of their record are excluded from the
    thresholds and listed in cal["excluded_floats"].
    """
    zD, zS = np.asarray(zD, "float64"), np.asarray(zS, "float64")
    pm = profile_means(val, zD, zS)
    cal = {}
    cal["pm_mu_D"], cal["pm_sd_D"] = _robust(pm.zD)
    cal["pm_mu_S"], cal["pm_sd_S"] = _robust(pm.zS)
    ps = profile_score(pm, cal)
    gross = (ps.score > GROSS_Z).groupby(level="PLATFORM").mean()
    excluded = sorted(gross.index[gross > GROSS_FRAC])
    cal["excluded_floats"] = {f: round(float(gross[f]), 3) for f in excluded}
    keep_lv = ~val.PLATFORM.isin(excluded).to_numpy()
    keep_pr = ~ps.index.get_level_values("PLATFORM").isin(excluded)

    both = np.fmax(np.abs(zD), np.abs(zS))
    cal["level_threshold"] = float(np.nanquantile(both[keep_lv], LEVEL_Q))
    cal["profile_threshold"] = float(np.nanquantile(ps.score[keep_pr], PROFILE_Q))

    deep = profile_means(val, zD, zS, min_pres=DEEP)
    deep = deep[~deep.index.get_level_values("PLATFORM").isin(excluded)]
    cal["deep_mu_D"], cal["deep_sd_D"] = _robust(deep.zD)
    cal["deep_mu_S"], cal["deep_sd_S"] = _robust(deep.zS)
    dps = profile_score(deep, cal, "deep")
    cal["within_d"] = float(dps.d.groupby(level=0).std().median())
    cal["within_s"] = float(dps.s.groupby(level=0).std().median())
    cal["float_method"] = method
    cal["window"], cal["cusum_k"] = int(window), float(cusum_k)
    c = float_stat(dps, cal)
    h, hz, frac = hazard_threshold(c)
    cal["cusum_threshold"] = h
    cal["float_hazard_per_100_profiles"] = 100 * hz
    cal["float_false_alarm_fraction_pool"] = frac
    cal["val_floats"] = int(c.index.get_level_values("PLATFORM").nunique())
    cal["val_levels"] = int(keep_lv.sum())
    return cal


def evaluate(df, truth, floats, zD, zS, both, cal, target):
    """Detection at the calibrated thresholds, for levels, profiles and floats."""
    bad = truth[f"bad_{target}"].to_numpy()
    flag = both > cal["level_threshold"]
    row = {"levels_bad": int(bad.sum()),
           "level_detection": float(np.mean(flag[bad])) if bad.any() else np.nan,
           "level_false_alarm": float(np.mean(flag[~bad & np.isfinite(both)]))}
    try:
        from sklearn.metrics import roc_auc_score
        ok = np.isfinite(both)
        row["level_auc"] = float(roc_auc_score(bad[ok], both[ok])) if bad[ok].any() else np.nan
    except ValueError:
        row["level_auc"] = np.nan

    pm = profile_means(df, zD, zS)
    ps = profile_score(pm, cal)
    prof_bad = pd.Series(bad).groupby([df.PLATFORM.to_numpy(), df.CYCLE.to_numpy()]).any()
    prof_bad = prof_bad.reindex(ps.index).fillna(False).to_numpy(bool)
    pflag = (ps.score > cal["profile_threshold"]).to_numpy()
    row["profile_detection"] = float(pflag[prof_bad].mean()) if prof_bad.any() else np.nan
    row["profile_false_alarm"] = float(pflag[~prof_bad].mean())

    c = float_stat(profile_score(profile_means(df, zD, zS, min_pres=DEEP), cal, "deep"), cal)
    alarm = c > cal["cusum_threshold"]
    hits, delays, fa = 0, [], 0
    clean_f = floats[~floats.corrupted].platform
    for f in clean_f:
        if f in alarm.index.get_level_values(0) and alarm.loc[f].any():
            fa += 1
    bad_f = floats[floats.corrupted]
    for _, r in bad_f.iterrows():
        if r.platform not in alarm.index.get_level_values(0):
            continue
        a = alarm.loc[r.platform]
        cycles = a.index.to_numpy()
        onset = r.onset_cycle if r.onset_cycle >= 0 else cycles.min()
        after = a[cycles >= onset]
        if after.any():
            hits += 1
            first = after.index[after.to_numpy()][0]
            delays.append(int(np.searchsorted(cycles, first) - np.searchsorted(cycles, onset)))
    row["floats_corrupted"] = int(len(bad_f))
    row["float_detection"] = hits / len(bad_f) if len(bad_f) else np.nan
    row["float_false_alarm"] = fa / len(clean_f) if len(clean_f) else np.nan
    row["median_delay_cycles"] = float(np.median(delays)) if delays else np.nan
    return row


def scenario_seed(s):
    # identical to baseline_rtqc, so both see exactly the same corruption
    return zlib.crc32(s.name.encode()) % (2 ** 31)


def run_scenarios(df, ens, cal, scenarios, clean_pred=None, quiet=False):
    """Plant each scenario in df, score, and evaluate at the thresholds in cal."""
    clean_pred = clean_pred or predict(ens, df)
    rows = []
    for s in scenarios:
        t1 = time.time()
        out, truth, floats = inject.inject(df, s, seed=scenario_seed(s))
        sel = out.PLATFORM.isin(floats.platform[floats.corrupted]).to_numpy()
        pred = {k: (v[0].copy(), v[1].copy()) for k, v in clean_pred.items()}
        if sel.any():
            p2 = predict(ens, out[sel].reset_index(drop=True))
            for k in pred:
                pred[k][0][sel], pred[k][1][sel] = p2[k]
        zD, zS, both = level_scores(out, pred)
        target = inject.TARGET_OF.get(s.kind, s.var)
        row = {"scenario": s.name, "kind": s.kind, "variable": s.var, "magnitude": s.magnitude}
        row.update(evaluate(out, truth, floats, zD, zS, both, cal, target))
        rows.append(row)
        if not quiet:
            print(f"  {s.name:<22} AUC {row['level_auc']:.3f}  level {row['level_detection']:.3f}  "
                  f"profile {row['profile_detection']:.3f}  float {row['float_detection']:.2f} "
                  f"(FA {row['float_false_alarm']:.2f}, delay {row['median_delay_cycles']})  "
                  f"{time.time() - t1:.0f}s", flush=True)
    return pd.DataFrame(rows)


FLOAT_CONFIGS = [("window", {"window": 3}), ("window", {"window": 5}), ("window", {"window": 10}),
                 ("cusum", {"cusum_k": 0.25}), ("cusum", {"cusum_k": 0.5}), ("cusum", {"cusum_k": 1.0})]


def select_float_method(ens):
    """Choose the float-level statistic and its parameter on validation floats only.

    Half the validation floats set every threshold (with the same robust, per-profile
    calibration used for the paper); errors are planted in the other half and each
    configuration's float-level detection is compared at those thresholds, over the persistent
    scenarios - what the float statistic is for. Each configuration gets its own threshold for
    the same false-alarm target, so the comparison is at equal false alarms. The CUSUM decision
    interval is that threshold; its allowance k is varied.
    """
    val = load_group("val")
    floats = sorted(val.PLATFORM.unique())
    half_a, half_b = floats[0::2], floats[1::2]
    va = val[val.PLATFORM.isin(half_a)].reset_index(drop=True)
    vb = val[val.PLATFORM.isin(half_b)].reset_index(drop=True)
    vb = vb[["PLATFORM", "CYCLE", "TIME", "PRES", "TEMP", "PSAL", "DOXY", "LATITUDE", "LONGITUDE"]]
    za = level_scores(va, predict(ens, va))
    cals = {f"{m} {list(p.values())[0]:g}": calibrate_from_z(va, za[0], za[1], m, **p)
            for m, p in FLOAT_CONFIGS}
    persistent = [s for s in inject.SCENARIOS if s.kind in inject.PERSISTENT]
    clean_b = predict(ens, vb)
    rows = []
    for s in persistent:
        out, truth, floats_ = inject.inject(vb, s, seed=scenario_seed(s))
        sel = out.PLATFORM.isin(floats_.platform[floats_.corrupted]).to_numpy()
        pred = {k: (v[0].copy(), v[1].copy()) for k, v in clean_b.items()}
        p2 = predict(ens, out[sel].reset_index(drop=True))
        for k in pred:
            pred[k][0][sel], pred[k][1][sel] = p2[k]
        zD, zS, both = level_scores(out, pred)
        target = inject.TARGET_OF.get(s.kind, s.var)
        for name, cal in cals.items():
            r = evaluate(out, truth, floats_, zD, zS, both, cal, target)
            rows.append({"config": name, "scenario": s.name, **r})
    r = pd.DataFrame(rows)
    sel = (r.groupby("config").agg(mean_float_detection=("float_detection", "mean"),
                                   median_delay_cycles=("median_delay_cycles", "median"),
                                   float_false_alarm=("float_false_alarm", "mean"))
           .join(pd.Series({n: c["cusum_threshold"] for n, c in cals.items()}, name="threshold"))
           .join(pd.Series({n: c["float_hazard_per_100_profiles"] for n, c in cals.items()},
                           name="calibration_rate_per_100")))
    sel = sel.sort_values("mean_float_detection", ascending=False)
    sel.to_csv(os.path.join(HERE, "tables", "float_method_selection.csv"))
    r.to_csv(os.path.join(HERE, "tables", "float_method_selection_detail.csv"), index=False)
    print(sel.round(3).to_string())
    print(f"selected: {sel.index[0]}")
    return sel.index[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doxy", default=os.path.join("runs_qc", "doxy"))
    ap.add_argument("--psal", default=os.path.join("runs_qc", "psal"))
    ap.add_argument("--floats", type=int, default=None, help="quick check on n test floats")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--float-method", choices=["window", "cusum"], default="window")
    ap.add_argument("--select", action="store_true",
                    help="choose the float statistic on validation floats and stop")
    args = ap.parse_args()

    ens = {"D": Ensemble(args.doxy), "S": Ensemble(args.psal)}
    if args.select:
        select_float_method(ens)
        return
    t0 = time.time()
    val = load_group("val")
    if args.floats:
        val = val[val.PLATFORM.isin(sorted(val.PLATFORM.unique())[: args.floats])]
    cal = calibrate(val, ens, args.float_method)
    print(f"calibrated on {cal['val_floats']} validation floats ({time.time() - t0:.0f}s), "
          f"float method {args.float_method}: "
          + ", ".join(f"{k} {v:.3f}" for k, v in cal.items() if "threshold" in k), flush=True)

    df = load_group("test_clean")
    if args.floats:
        df = df[df.PLATFORM.isin(sorted(df.PLATFORM.unique())[: args.floats])].reset_index(drop=True)
    cols = ["PLATFORM", "CYCLE", "TIME", "PRES", "TEMP", "PSAL", "DOXY", "LATITUDE", "LONGITUDE"]
    df = df[cols]
    wanted = [s for s in inject.SCENARIOS if not args.scenarios or s.name in args.scenarios]
    rows = run_scenarios(df, ens, cal, wanted).to_dict("records")

    if not args.floats and not args.scenarios:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        with open(OUT_CAL, "w") as f:
            json.dump(cal, f, indent=1)
        print(f"wrote {OUT} and {OUT_CAL}")


if __name__ == "__main__":
    main()
