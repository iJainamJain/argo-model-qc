"""Which residual raises the alarm: telling a CTD fault from an optode fault.

The oxygen model takes salinity as an input, and a real optode's oxygen is computed with the
CTD salinity, so a salinity fault can move the oxygen residual too. An operator needs to know
which sensor to suspect. For planted salinity, temperature and oxygen errors (5 repeats each,
clean test floats, cross-fitted thresholds), every alarmed corrupted profile is classified by
which standardized profile component exceeds the profile threshold:

    salinity only   |s| > threshold, |d| <= threshold
    oxygen only     |d| > threshold, |s| <= threshold
    both

Oxygen-only alarms under a CTD fault are further split into those already present on the same
profiles before corruption (the oxygen component's background false alarms) and those the
fault created.

The rule we test is: suspect the CTD whenever the salinity component exceeds threshold; suspect
the optode only when the oxygen component exceeds it alone. The salinity model does not see
oxygen, so an optode fault cannot move the salinity component - the rule's only possible error
is to blame the optode for a CTD fault, and this measures how often that happens.

Writes tables/attribution.csv.

    python -m paper_qc.attribution
"""
import json
import os

import numpy as np
import pandas as pd

from paper_qc import bench, inject, score
from paper_qc.model import Ensemble

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "attribution.csv")
SCENARIOS = ["offset_PSAL_0.05", "offset_PSAL_0.1", "drift_PSAL_0.05", "offset_TEMP_0.5",
             "gain_DOXY_0.1", "gain_DOXY_0.2", "offset_DOXY_20", "drift_DOXY_0.05"]
REPS = 5


def main():
    cal = json.load(open(os.path.join(HERE, "tables", "score_thresholds_cv.json")))
    thr = cal["profile_threshold"]
    ens = {"D": Ensemble(os.path.join("runs_qc", "doxy")), "S": Ensemble(os.path.join("runs_qc", "psal"))}
    df = score.load_group("test_clean")[bench.COLS].reset_index(drop=True)
    clean = score.predict(ens, df)
    # the same profiles uncorrupted: which oxygen-only alarms were there before any error?
    zD0, zS0, _ = score.level_scores(df, clean)
    ps0 = score.profile_score(score.profile_means(df, zD0, zS0), cal)
    background = float(((ps0.d.abs() > thr) & (ps0.s.abs() <= thr)).mean())
    print(f"clean data: oxygen-only profile alarms on {100 * background:.2f}% of profiles")
    by = {s.name: s for s in inject.SCENARIOS}
    rows = []
    for name in SCENARIOS:
        s = by[name]
        target = inject.TARGET_OF.get(s.kind, s.var)
        counts = {"salinity only": 0, "oxygen only": 0, "both": 0, "flagged": 0, "bad": 0,
                  "oxygen only, already on clean data": 0}
        for rep in range(REPS):
            out, truth, floats = inject.inject(df, s, seed=bench.seed_for(s, rep))
            sel = out.PLATFORM.isin(floats.platform[floats.corrupted]).to_numpy()
            pred = {k: (v[0].copy(), v[1].copy()) for k, v in clean.items()}
            p2 = score.predict(ens, out[sel].reset_index(drop=True))
            for k in pred:
                pred[k][0][sel], pred[k][1][sel] = p2[k]
            zD, zS, _ = score.level_scores(out, pred)
            ps = score.profile_score(score.profile_means(out, zD, zS), cal)
            bad = pd.Series(truth[f"bad_{target}"].to_numpy()).groupby(
                [out.PLATFORM.to_numpy(), out.CYCLE.to_numpy()]).any()
            bad.index.names = ["PLATFORM", "CYCLE"]
            p = ps[bad.reindex(ps.index, fill_value=False).to_numpy()]
            s_hi, d_hi = p.s.abs() > thr, p.d.abs() > thr
            counts["bad"] += len(p)
            counts["flagged"] += int((s_hi | d_hi).sum())
            counts["salinity only"] += int((s_hi & ~d_hi).sum())
            counts["oxygen only"] += int((d_hi & ~s_hi).sum())
            was = (ps0.d.abs().reindex(p.index) > thr)
            counts["oxygen only, already on clean data"] += int((d_hi & ~s_hi & was).sum())
            counts["both"] += int((s_hi & d_hi).sum())
        f = max(counts["flagged"], 1)
        rows.append({"scenario": name, "fault": {"PSAL": "CTD", "TEMP": "CTD"}.get(s.var, "optode"),
                     "profiles_bad": counts["bad"], "profiles_flagged": counts["flagged"],
                     "salinity_only": counts["salinity only"] / f,
                     "oxygen_only": counts["oxygen only"] / f, "both": counts["both"] / f,
                     "oxygen_only_preexisting": (counts["oxygen only, already on clean data"]
                                                 / max(counts["oxygen only"], 1)),
                     "background_oxygen_only": background})
        r = rows[-1]
        print(f"  {name:<18} ({r['fault']}) flagged {r['profiles_flagged']:>6,} of {r['profiles_bad']:>6,}: "
              f"salinity only {r['salinity_only']:.3f}, oxygen only {r['oxygen_only']:.3f}, "
              f"both {r['both']:.3f}", flush=True)
    t = pd.DataFrame(rows)
    # the rule's error: a CTD fault whose alarm came from the oxygen component alone
    ctd = t[t.fault == "CTD"]
    opt = t[t.fault == "optode"]
    print(f"\nCTD faults blamed on the optode (oxygen-only alarms): "
          f"{(ctd.oxygen_only * ctd.profiles_flagged).sum() / ctd.profiles_flagged.sum():.3f}")
    print(f"optode faults with a salinity alarm: "
          f"{((opt.salinity_only + opt.both) * opt.profiles_flagged).sum() / opt.profiles_flagged.sum():.3f}")
    t.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
