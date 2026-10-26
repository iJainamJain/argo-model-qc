"""Where the Argo greylist meets our dataset, and what the data mode was there.

The greylist (data/argo_greylist/ar_greylist.txt, US GDAC) names a float, a parameter, a date
window and a quality code. Real-time QC is meant to apply it; delayed-mode review supersedes it.
So a level inside a code 3/4 window is only a real-time miss if it is still real-time (R) or
real-time adjusted (A) data carrying flags 1-2. Delayed-mode (D) levels were kept by an operator
after the greylisting and are correct as they stand.

Writes tables/greylist_overlap.csv.  Run from the project root:
    python paper_qc/greylist.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_qc.lib import data as odata  # noqa: E402
from paper_qc import argofile  # noqa: E402

GREYLIST = os.path.join("data", "argo_greylist", "ar_greylist.txt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "greylist_overlap.csv")
PARAMS = ["TEMP", "PSAL", "PRES", "DOXY"]


def load_greylist():
    g = pd.read_csv(GREYLIST, dtype=str)
    g["start"] = pd.to_datetime(g.START_DATE, format="%Y%m%d")
    g["end"] = pd.to_datetime(g.END_DATE, format="%Y%m%d", errors="coerce")
    return g


def parameter_mode(wmo, cycle, parameter):
    """PARAMETER_DATA_MODE of one parameter in one synthetic profile file, or '' if absent."""
    path = argofile.s_file(wmo, cycle)
    return argofile.read(path, parameters=())["mode"].get(parameter, "") if path else ""


def main():
    df = odata.load(verbose=False)
    df["PLATFORM"] = df.PLATFORM.astype(str).str.strip()
    t = pd.to_datetime(df.TIME)
    df["t"] = t.dt.tz_localize(None) if t.dt.tz is not None else t

    g = load_greylist()
    print(f"greylist: {len(g):,} entries on {g.PLATFORM_CODE.nunique():,} floats")
    g = g[g.PARAMETER_NAME.isin(PARAMS) & g.PLATFORM_CODE.isin(set(df.PLATFORM))]
    print(f"  {len(g)} T/S/P/DOXY entries on floats in our dataset")

    rows = []
    for _, e in g.iterrows():
        sel = (df.PLATFORM == e.PLATFORM_CODE) & (df.t >= e.start)
        if pd.notna(e.end):
            sel &= df.t < e.end
        if not sel.any():
            continue
        # data mode is per profile, so checking every cycle in the window is exact
        modes = {}
        for c in sorted(set(df[sel].CYCLE.astype(int))):
            modes[c] = parameter_mode(e.PLATFORM_CODE, c, e.PARAMETER_NAME)
        lev_mode = df[sel].CYCLE.astype(int).map(modes)
        rows.append({
            "float": e.PLATFORM_CODE, "parameter": e.PARAMETER_NAME,
            "quality_code": e.QUALITY_CODE, "start": e.START_DATE, "end": e.END_DATE,
            "comment": e.COMMENT, "profiles": len(modes), "levels": int(sel.sum()),
            "levels_D": int((lev_mode == "D").sum()), "levels_A": int((lev_mode == "A").sum()),
            "levels_R": int((lev_mode == "R").sum()),
        })
        print(f"  {e.PLATFORM_CODE} {e.PARAMETER_NAME} code {e.QUALITY_CODE}: "
              f"{rows[-1]['levels']:,} levels, modes {''.join(sorted(set(modes.values())))}", flush=True)

    r = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r.to_csv(OUT, index=False)
    bad = r[r.quality_code.isin(["3", "4"])]
    misses = bad.levels_A.sum() + bad.levels_R.sum()
    print(f"\ninside code 3/4 windows: {bad.levels.sum():,} levels "
          f"({100 * bad.levels.sum() / len(df):.2f}% of {len(df):,}) on {bad.float.nunique()} floats")
    print(f"  delayed-mode (operator kept them): {bad.levels_D.sum():,}")
    print(f"  real-time or adjusted with flags 1-2 (real-time misses): {misses:,} "
          f"({100 * misses / len(df):.4f}%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
