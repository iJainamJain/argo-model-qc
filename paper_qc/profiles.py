"""Full-resolution profiles of the test floats, as the real-time tests would see them.

The training table (paper_qc.lib.build) keeps only levels where oxygen and measured T/S
coincide, which thins many profiles to an irregular subset of the CTD levels. The real-time tests were
designed for the full CTD profile: on the thinned one, "digit rollover" fires when 28 C at the
surface is followed by 12 C at the next retained level, and the spike and gradient tests see
neighbours hundreds of dbar apart. Running them there would handicap the baseline.

So this reads the same S-files under the same rules - adjusted values and flags wherever the
parameter's data mode is A or D, flags 1-2 only, time and position QC good, ascending profiles,
delayed mode preferred per cycle - but keeps every level with good PRES, measured TEMP and
measured PSAL (flag 8, interpolated, excluded). Oxygen is attached where it is good and inside
paper_qc.lib.data.O2_RANGE, NaN elsewhere.

Writes data/qc_profiles.parquet (every float, labelled by its split group).  Run from the project root:
    python -m paper_qc.profiles          # adjusted, QC-filtered (the default view)
    python -m paper_qc.profiles --raw    # real-time view: raw values and flags, test floats
"""
import glob
import os
import time

import netCDF4
import numpy as np
import pandas as pd

from paper_qc.lib.build import DAYS_1950_TO_1970, GOOD_QC, NAME, _chars, _num, _text
from paper_qc.lib.data import O2_RANGE
from paper_qc import argofile, split

OUT = os.path.join("data", "qc_profiles.parquet")


def float_files(wmo):
    """{cycle: path} of ascending S-files for one float, delayed mode preferred."""
    best = {}
    for path in glob.glob(os.path.join(argofile.RAW_ROOT, "*", str(wmo), "profiles", "S*.nc")):
        m = NAME.match(os.path.basename(path))
        if not m or m.group(4):
            continue
        cycle = int(m.group(3))
        if m.group(1) == "SD" or cycle not in best:
            best[cycle] = path
    return best


def read_full(path):
    """All good CTD levels of one S-file, with oxygen where good; None if unusable."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_chartostring(False)
        prof = 0
        if (_chars(ds, "JULD_QC")[prof] not in GOOD_QC
                or _chars(ds, "POSITION_QC")[prof] not in GOOD_QC):
            return None
        params = [_text(r) for r in _chars(ds, "STATION_PARAMETERS")[prof]]
        modes = _text(_chars(ds, "PARAMETER_DATA_MODE")[prof])
        vals, ok, mode_of = {}, {}, {}
        for name in ("PRES", "TEMP", "PSAL", "DOXY"):
            if name not in params:
                if name == "DOXY":
                    continue
                return None
            i = params.index(name)
            adjusted = i < len(modes) and modes[i] in "AD"
            var = f"{name}_ADJUSTED" if adjusted else name
            if var not in ds.variables:
                if name == "DOXY":
                    continue
                return None
            vals[name] = _num(ds, var)[prof]
            ok[name] = np.isfinite(vals[name]) & np.isin(_chars(ds, f"{var}_QC")[prof], GOOD_QC)
            mode_of[name] = modes[i] if i < len(modes) else " "
        keep = ok["PRES"] & ok["TEMP"] & ok["PSAL"]
        if not keep.any():
            return None
        days = float(np.ma.filled(ds["JULD"][prof], np.nan))
        lat = float(np.ma.filled(ds["LATITUDE"][prof], np.nan))
        lon = float(np.ma.filled(ds["LONGITUDE"][prof], np.nan))
        if not (np.isfinite(days) and np.isfinite(lat) and np.isfinite(lon)):
            return None
        n = int(keep.sum())
        doxy = np.full(n, np.nan)
        if "DOXY" in vals:
            d = vals["DOXY"][keep]
            good = ok["DOXY"][keep] & (d >= O2_RANGE[0]) & (d <= O2_RANGE[1])
            doxy[good] = d[good]
        return {
            "PRES": vals["PRES"][keep], "TEMP": vals["TEMP"][keep],
            "PSAL": vals["PSAL"][keep], "DOXY": doxy,
            "LATITUDE": np.full(n, lat), "LONGITUDE": np.full(n, lon),
            "TIME": np.full(n, (days - DAYS_1950_TO_1970) * 86400.0),
            "MODE_PSAL": np.full(n, mode_of["PSAL"]),
            "MODE_DOXY": np.full(n, mode_of.get("DOXY", " ")),
        }



OUT_RAW = os.path.join("data", "qc_profiles_raw.parquet")


def read_raw(path):
    """The real-time view of one S-file: unadjusted values and their own flags, every level.

    Delayed-mode processing corrects or discards exactly the data a QC method should catch
    (greylisted salinity is adjusted or flagged 4, oxygen gain is corrected), so the real
    cases are scored on what real time saw: raw PRES/TEMP/PSAL/DOXY, no QC filtering, the raw
    flags kept alongside so any subset can be chosen afterwards.
    """
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_chartostring(False)
        prof = 0
        if (_chars(ds, "JULD_QC")[prof] not in GOOD_QC
                or _chars(ds, "POSITION_QC")[prof] not in GOOD_QC):
            return None
        cols = {}
        for name in ("PRES", "TEMP", "PSAL", "DOXY"):
            if name in ds.variables:
                cols[name] = _num(ds, name)[prof]
                cols[name + "_QC"] = np.array(
                    [c.decode() if isinstance(c, bytes) else str(c)
                     for c in _chars(ds, name + "_QC")[prof]])
            elif name == "DOXY":
                n = len(cols["PRES"])
                cols["DOXY"], cols["DOXY_QC"] = np.full(n, np.nan), np.full(n, " ")
            else:
                return None
        keep = np.isfinite(cols["PRES"]) & np.isfinite(cols["TEMP"]) & np.isfinite(cols["PSAL"])
        if not keep.any():
            return None
        days = float(np.ma.filled(ds["JULD"][prof], np.nan))
        lat = float(np.ma.filled(ds["LATITUDE"][prof], np.nan))
        lon = float(np.ma.filled(ds["LONGITUDE"][prof], np.nan))
        if not (np.isfinite(days) and np.isfinite(lat) and np.isfinite(lon)):
            return None
        n = int(keep.sum())
        out = {k: v[keep] for k, v in cols.items()}
        out.update({"LATITUDE": np.full(n, lat), "LONGITUDE": np.full(n, lon),
                    "TIME": np.full(n, (days - DAYS_1950_TO_1970) * 86400.0)})
        return out


def main_raw():
    s = split.load()
    groups = {f: "test_clean" for f in s["test_clean"]}
    groups.update({f: "test_known_bad" for f in s["test_known_bad"]})
    parts, t0 = [], time.time()
    for wmo, group in sorted(groups.items()):
        for cycle, path in sorted(float_files(wmo).items()):
            r = read_raw(path)
            if r is None:
                continue
            part = pd.DataFrame(r)
            part.insert(0, "CYCLE", cycle)
            part.insert(0, "PLATFORM", str(wmo))
            part["group"] = group
            part["FILE_MODE"] = os.path.basename(path)[1]
            parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["TIME"] = pd.to_datetime(df.TIME, unit="s")
    for c in ("PRES", "TEMP", "PSAL", "DOXY", "LATITUDE", "LONGITUDE"):
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT_RAW, index=False)
    print(f"raw view: {df.groupby(['PLATFORM', 'CYCLE']).ngroups:,} profiles, {len(df):,} levels, "
          f"{time.time() - t0:.0f}s -> {OUT_RAW}")
    print(df.groupby("group").agg(floats=("PLATFORM", "nunique"), levels=("PRES", "size")))


def main():
    s = split.load()
    # training and validation floats too: detection thresholds are calibrated on them (with
    # out-of-fold models for the training floats, paper_qc.crossfit), never on test data
    groups = {f: "train" for f in s["train"]}
    groups.update({f: "val" for f in s["val"]})
    groups.update({f: "test_clean" for f in s["test_clean"]})
    groups.update({f: "test_known_bad" for f in s["test_known_bad"]})
    parts, t0, n_files, n_skip = [], time.time(), 0, 0
    for wmo, group in sorted(groups.items()):
        for cycle, path in sorted(float_files(wmo).items()):
            n_files += 1
            r = read_full(path)
            if r is None:
                n_skip += 1
                continue
            part = pd.DataFrame(r)
            part.insert(0, "CYCLE", cycle)
            part.insert(0, "PLATFORM", str(wmo))
            part["group"] = group
            parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["TIME"] = pd.to_datetime(df.TIME, unit="s")
    for c in ("PRES", "TEMP", "PSAL", "DOXY", "LATITUDE", "LONGITUDE"):
        df[c] = df[c].astype("float32")
    df.to_parquet(OUT, index=False)
    prof = df.groupby(["PLATFORM", "CYCLE"])
    print(f"{n_files:,} files ({n_skip:,} unusable) -> {prof.ngroups:,} profiles, "
          f"{len(df):,} levels, {df.DOXY.notna().mean():.0%} with oxygen, "
          f"{time.time() - t0:.0f}s")
    print(df.groupby("group").agg(floats=("PLATFORM", "nunique"), levels=("PRES", "size")))
    print(f"median levels per profile: {prof.size().median():.0f} "
          f"(training table: see paper_qc.lib.data.load)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    import sys
    main_raw() if "--raw" in sys.argv else main()
