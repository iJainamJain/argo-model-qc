"""How much delayed-mode (and real-time adjusted) processing changed each profile.

For every synthetic-profile file on disk, one row per profile: the data mode of PRES, TEMP,
PSAL and DOXY, and the median change adjustment made to each - additive for the physical
variables, and both additive and as a ratio for oxygen, whose optode errors are mostly a gain
(sensitivity loss, often 5-20 %). This is the continuous label "how wrong was the unadjusted
value", which is what a model-based check has to anticipate before the adjustment exists.

Writes tables/adjustments.csv.gz.  Run from the project root:
    python -m paper_qc.adjustments
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from paper_qc import argofile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "adjustments.csv.gz")
PARAMS = ("PRES", "TEMP", "PSAL", "DOXY")
GOOD = np.array(["1", "2"])


def summarise(path):
    r = argofile.read(path, PARAMS)
    row = {"file": os.path.basename(path), "platform": r["platform"], "cycle": r["cycle"],
           "juld": r["juld"], "lat": r["lat"], "lon": r["lon"]}
    for p in PARAMS:
        row[f"mode_{p}"] = r["mode"].get(p, "")
        raw, adj = r.get(p), r.get(p + "_ADJUSTED")
        if raw is None or adj is None:
            continue
        ok = np.isfinite(raw) & np.isfinite(adj)
        if p + "_ADJUSTED_QC" in r:
            ok &= np.isin(r[p + "_ADJUSTED_QC"][: len(ok)], GOOD)
        row[f"n_{p}"] = int(ok.sum())
        if ok.any():
            d = adj[ok] - raw[ok]
            row[f"dmed_{p}"] = float(np.median(d))
            row[f"dmax_{p}"] = float(d[np.argmax(np.abs(d))])
            if p == "DOXY":
                pos = ok & (raw > 5)
                if pos.any():
                    row["ratio_DOXY"] = float(np.median(adj[pos] / raw[pos]))
    return row


def main(limit=None):
    files = sorted(glob.glob(os.path.join(argofile.RAW_ROOT, "*", "*", "profiles", "S*.nc")))
    if limit:
        files = files[:limit]
    print(f"{len(files):,} synthetic-profile files", flush=True)
    rows, failed, t0 = [], 0, time.time()
    for i, f in enumerate(files):
        try:
            rows.append(summarise(f))
        except Exception:
            failed += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,} files, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} profiles ({failed} unreadable) to {OUT} in {time.time() - t0:.0f}s")
    for p in PARAMS:
        if f"mode_{p}" in df:
            print(f"  {p} modes:", df[f"mode_{p}"].value_counts().to_dict())
    if "ratio_DOXY" in df:
        dm = df[df.mode_DOXY == "D"].ratio_DOXY.dropna()
        print(f"  DOXY delayed-mode gain: median {dm.median():.3f}, "
              f"5-95% {dm.quantile(0.05):.3f}-{dm.quantile(0.95):.3f} over {len(dm):,} profiles")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
