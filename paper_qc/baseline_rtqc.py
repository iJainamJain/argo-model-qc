"""What the real-time Argo tests catch: the baseline every model-based score is compared with.

1. False alarms: the tests run on the clean test floats as delivered, at full
   resolution (paper_qc.profiles), which is what the tests were designed for.
2. Detection: for every scenario in inject.SCENARIOS, errors are planted in half of the clean
   test floats and the tests rerun on those floats (the other half are unchanged, so their
   results are reused from step 1).

Per scenario it records the fraction of corrupted levels flagged, the fraction of corrupted
floats with any flagged level after onset, and the delay in cycles from onset to the first
flag. Writes tables/baseline_rtqc.csv and tables/rtqc_false_alarms.csv.

    python -m paper_qc.baseline_rtqc              # all scenarios
    python -m paper_qc.baseline_rtqc --floats 10  # quick check on 10 floats
"""
import argparse
import os
import time
import zlib

import numpy as np
import pandas as pd

from paper_qc import inject, profiles, rtqc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "baseline_rtqc.csv")
OUT_FA = os.path.join(HERE, "tables", "rtqc_false_alarms.csv")
COLS = ["PLATFORM", "CYCLE", "TIME", "PRES", "TEMP", "PSAL", "DOXY", "LATITUDE", "LONGITUDE"]


def scenario_seed(s):
    return zlib.crc32(s.name.encode()) % (2 ** 31)


def test_floats(n=None, group="test_clean"):
    """Full-resolution profiles of the clean test floats (paper_qc.profiles).

    Not the oxygen dataset: that keeps only levels where oxygen and measured T/S coincide
    (median 70 levels per profile on these floats against 488 in the full profiles), and the
    real-time tests were designed for the full CTD profile.
    """
    df = pd.read_parquet(profiles.OUT)
    df = df[df.group == group]
    if n:
        df = df[df.PLATFORM.isin(sorted(df.PLATFORM.unique())[:n])]
    return df[COLS].reset_index(drop=True)


def any_fail(res):
    return res[[c for c in res if c.startswith("fail_")]].any(axis=1).to_numpy()


def score(df, truth, floats, res, target, clean):
    """Detection metrics for one scenario.

    The tests flag some clean data too (on these floats, oxygen levels of 31 of 75 floats),
    so a flag on a corrupted level is not necessarily caused by the corruption. Alongside the
    raw rates, the *_new columns count only flags absent from the clean run of the same data:
    detection attributable to the error.
    """
    bad = truth[f"bad_{target}"].to_numpy()
    if target == "PRES":
        flagged, before = any_fail(res), any_fail(clean)
    else:
        flagged, before = res[f"fail_{target}"].to_numpy(), clean[f"fail_{target}"].to_numpy()
    new = flagged & ~before
    row = {"levels_bad": int(bad.sum()),
           "level_detection": float(flagged[bad].mean()) if bad.any() else np.nan,
           "level_detection_new": float(new[bad].mean()) if bad.any() else np.nan,
           "level_false_alarm": float(flagged[~bad].mean())}
    bad_f = floats[floats.corrupted]
    hits, hits_new, delays = 0, 0, []
    for _, r in bad_f.iterrows():
        sel = (df.PLATFORM == r.platform).to_numpy() & bad
        if not sel.any():
            continue
        hits_new += bool((sel & new).any())
        cyc = df.CYCLE.to_numpy()[sel & flagged]
        if len(cyc):
            hits += 1
            if r.onset_cycle >= 0:
                cycles = np.sort(df.CYCLE[(df.PLATFORM == r.platform)].unique())
                delays.append(int(np.searchsorted(cycles, cyc.min())
                                  - np.searchsorted(cycles, r.onset_cycle)))
    row["floats_corrupted"] = int(len(bad_f))
    row["float_detection"] = hits / len(bad_f) if len(bad_f) else np.nan
    row["float_detection_new"] = hits_new / len(bad_f) if len(bad_f) else np.nan
    row["median_delay_cycles"] = float(np.median(delays)) if delays else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floats", type=int, default=None)
    args = ap.parse_args()

    df = test_floats(args.floats)
    print(f"{df.PLATFORM.nunique()} clean test floats, "
          f"{df.groupby(['PLATFORM', 'CYCLE']).ngroups:,} profiles, {len(df):,} levels", flush=True)

    t0 = time.time()
    clean = rtqc.run(df)
    print(f"clean run {time.time() - t0:.0f}s", flush=True)
    fa = []
    for v in rtqc.VARS:
        f = clean[f"fail_{v}"]
        fa.append({"variable": v, "test": "any", "levels": int(f.sum()), "rate": float(f.mean()),
                   "floats": int(df.PLATFORM[f].nunique())})
        for name, n in clean[f"test_{v}"][f].value_counts().items():
            fa.append({"variable": v, "test": name, "levels": int(n), "rate": n / len(df),
                       "floats": int(df.PLATFORM[clean[f"test_{v}"] == name].nunique())})
    fa = pd.DataFrame(fa)
    if not args.floats:
        fa.to_csv(OUT_FA, index=False)
    print(fa.to_string(index=False), flush=True)

    rows = []
    for s in inject.SCENARIOS:
        t1 = time.time()
        out, truth, floats = inject.inject(df, s, seed=scenario_seed(s))
        res = clean.copy()
        sel = out.PLATFORM.isin(floats.platform[floats.corrupted]).to_numpy()
        res.loc[sel] = rtqc.run(out[sel]).to_numpy()
        target = inject.TARGET_OF.get(s.kind, s.var)
        row = {"scenario": s.name, "kind": s.kind, "variable": s.var, "magnitude": s.magnitude}
        row.update(score(df, truth, floats, res, target, clean))
        # which test did the catching
        if target != "PRES":
            caught = res[f"test_{target}"][truth[f"bad_{target}"] & res[f"fail_{target}"]]
            row["tests"] = "; ".join(f"{k}:{v}" for k, v in caught.value_counts().items())
        rows.append(row)
        print(f"  {s.name:<22} level {row['level_detection']:.3f} "
              f"(new {row['level_detection_new']:.3f})  float {row['float_detection']:.2f} "
              f"(new {row['float_detection_new']:.2f})  delay {row['median_delay_cycles']}  "
              f"({time.time() - t1:.0f}s)", flush=True)
    r = pd.DataFrame(rows)
    if not args.floats:
        r.to_csv(OUT, index=False)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
