"""
Build the level table from Argo synthetic (S) profile files.

    python -m paper_qc.lib.build --raw-dir data/argo_synthetic --out data/qc_levels.nc

Input : <raw-dir>/**/{SR,SD}<wmo>_<cycle>.nc   (ascending profiles only)
Output: flat NetCDF with one row per retained level.

A synthetic profile reports temperature and salinity on the biogeochemical levels too, but
values it interpolated there carry QC flag 8. Levels where temperature, salinity and pressure
have flags 1-2 are therefore exactly the levels with real CTD measurements; interpolated ones
are kept but marked (TS_INTERP_FLAG).

QC follows the Argo convention: per parameter, if its PARAMETER_DATA_MODE is A or D use the
*_ADJUSTED values and flags exclusively, otherwise the raw ones; keep flags 1-2 only; drop
profiles failing JULD_QC or POSITION_QC.
"""

import argparse
import glob
import os
import re
import time

import netCDF4
import numpy as np
import pandas as pd

NAME = re.compile(r"^(SR|SD)(\d+)_(\d+)(D?)\.nc$")
GOOD_QC = (b"1", b"2")
NEEDED = ["PRES", "TEMP", "PSAL", "DOXY"]
DAYS_1950_TO_1970 = 7305

COLUMNS = [
    ("PRES", "f4", "decibar", "Sea water pressure"),
    ("TEMP", "f4", "degree_Celsius", "Sea temperature in-situ ITS-90 scale"),
    ("PSAL", "f4", "psu", "Practical salinity"),
    ("DOXY", "f4", "umol kg-1", "Dissolved oxygen"),
    ("LATITUDE", "f4", "degree_north", "Latitude of the profile"),
    ("LONGITUDE", "f4", "degree_east", "Longitude of the profile"),
    ("TIME", "f8", "seconds since 1970-01-01T00:00:00Z", "Time of the profile"),
    ("PLATFORM", "i4", "1", "WMO identifier of the float"),
    ("CYCLE", "i4", "1", "Cycle number"),
    ("DOXY_ADJUSTED_FLAG", "i1", "1", "1 if DOXY came from delayed/adjusted mode, else 0"),
    ("TS_INTERP_FLAG", "i1", "1",
     "1 if TEMP/PSAL at this level were interpolated by the synthetic-profile procedure "
     "(QC flag 8) rather than measured; default analyses use flag 0 only"),
]


def _chars(ds, name):
    a = ds[name][:]
    return np.asarray(np.ma.filled(a, b" ") if np.ma.isMaskedArray(a) else a)


def _text(chars):
    return b"".join(np.ravel(chars).tolist()).decode(errors="replace").strip()


def _num(ds, name):
    return np.ma.filled(ds[name][:].astype("f8"), np.nan)


def read_sprofile(path):
    """Retained levels of one synthetic profile, or a string saying why it was skipped."""
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_chartostring(False)
        if "DOXY" not in ds.variables:
            return "no DOXY"
        prof = 0  # synthetic profiles carry a single merged profile per cycle

        if (_chars(ds, "JULD_QC")[prof] not in GOOD_QC
                or _chars(ds, "POSITION_QC")[prof] not in GOOD_QC):
            return "bad time/position QC"

        params = [_text(r) for r in _chars(ds, "STATION_PARAMETERS")[prof]]
        modes = _text(_chars(ds, "PARAMETER_DATA_MODE")[prof])

        values, flags, adjusted_doxy = {}, {}, False
        for name in NEEDED:
            if name not in params:
                return f"no {name}"
            i = params.index(name)
            adjusted = i < len(modes) and modes[i] in "AD"
            var = f"{name}_ADJUSTED" if adjusted else name
            if var not in ds.variables:
                return f"no {var}"
            values[name] = _num(ds, var)[prof]
            flags[name] = _chars(ds, f"{var}_QC")[prof]
            if name == "DOXY":
                adjusted_doxy = adjusted

        ok = {name: np.isfinite(values[name]) & np.isin(flags[name], GOOD_QC) for name in NEEDED}
        # Levels where the synthetic file interpolated T/S (flag 8) are kept but marked, so
        # the default analysis can exclude them and a sensitivity test can include them.
        # Without them ~20% of profiles - those sampling oxygen only between CTD levels -
        # would be dropped entirely.
        ts_interp = {name: np.isfinite(values[name]) & (flags[name] == b"8")
                     for name in ("TEMP", "PSAL")}
        ts_usable = (ok["TEMP"] | ts_interp["TEMP"]) & (ok["PSAL"] | ts_interp["PSAL"])
        keep = ok["PRES"] & ok["DOXY"] & ts_usable
        interp_flag = keep & ~(ok["TEMP"] & ok["PSAL"])
        if not keep.any():
            # Separate the usual causes: real-time oxygen is flagged 3 until adjusted, and
            # oxygen measured only between CTD levels has interpolated T/S (flag 8).
            if not ok["DOXY"].any():
                return f"no good DOXY levels (mode {'AD' if adjusted_doxy else 'R'})"
            if not (ok["TEMP"] & ok["PSAL"]).any():
                return "no good T/S levels"
            return "no level with good T/S and DOXY together"

        juld = ds["JULD"]
        assert "1950-01-01" in juld.units, juld.units
        days = float(np.ma.filled(juld[prof], np.nan))
        lat = float(np.ma.filled(ds["LATITUDE"][prof], np.nan))
        lon = float(np.ma.filled(ds["LONGITUDE"][prof], np.nan))
        if not (np.isfinite(days) and np.isfinite(lat) and np.isfinite(lon)):
            return "bad time/position QC"

        n = int(keep.sum())
        return {
            "PRES": values["PRES"][keep], "TEMP": values["TEMP"][keep],
            "PSAL": values["PSAL"][keep], "DOXY": values["DOXY"][keep],
            "LATITUDE": np.full(n, lat), "LONGITUDE": np.full(n, lon),
            "TIME": np.full(n, (days - DAYS_1950_TO_1970) * 86400.0),
            "DOXY_ADJUSTED_FLAG": np.full(n, int(adjusted_doxy)),
            "TS_INTERP_FLAG": interp_flag[keep].astype("i1"),
        }


def index_files(raw_dir):
    """[(wmo, cycle, path)] for ascending synthetic profiles; delayed mode wins per cycle."""
    best = {}
    for path in glob.glob(os.path.join(raw_dir, "**", "*.nc"), recursive=True):
        m = NAME.match(os.path.basename(path))
        if not m or m.group(4):  # skip core files and descending profiles
            continue
        key = (int(m.group(2)), int(m.group(3)))
        if m.group(1) == "SD" or key not in best:
            best[key] = path
    return [(wmo, cycle, path) for (wmo, cycle), path in sorted(best.items())]


def write(cols, out_path, source):
    data = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((data["PRES"], data["TIME"], data["PLATFORM"]))
    data = {k: v[order] for k, v in data.items()}
    n = data["PRES"].size

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as out:
        out.createDimension("OBS", n)
        for name, dtype, units, long_name in COLUMNS:
            var = out.createVariable(name, dtype, ("OBS",), zlib=True, complevel=4)
            var.units, var.long_name = units, long_name
            if name == "TIME":
                var.calendar = "standard"
            var[:] = data[name]
        out.title = "Level table for model-based quality control of Argo salinity and oxygen"
        out.source_files = source
        out.qc_policy = ("PRES/DOXY QC flags 1-2 only; TEMP/PSAL flags 1-2, or 8 (interpolated) "
                         "with TS_INTERP_FLAG set; ADJUSTED values and flags used exclusively "
                         "where PARAMETER_DATA_MODE is A or D")
    return data, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="process only the first N profiles (testing)")
    args = parser.parse_args()

    files = index_files(args.raw_dir)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"Error: no synthetic profile files (S*.nc) under {args.raw_dir}/")
        return 1
    print(f"{len(files):,} ascending synthetic profiles in {args.raw_dir}/")

    cols = {name: [] for name, *_ in COLUMNS}
    skipped, n_profiles, started = {}, 0, time.time()
    for i, (wmo, cycle, path) in enumerate(files, 1):
        prof = read_sprofile(path)
        if isinstance(prof, str):
            skipped[prof] = skipped.get(prof, 0) + 1
            continue
        n = len(prof["PRES"])
        for name, dtype, *_ in COLUMNS:
            if name == "PLATFORM":
                cols[name].append(np.full(n, wmo, dtype="i4"))
            elif name == "CYCLE":
                cols[name].append(np.full(n, cycle, dtype="i4"))
            else:
                cols[name].append(prof[name].astype(dtype))
        n_profiles += 1
        if i % 5000 == 0:
            print(f"  ...{i:,}/{len(files):,} ({time.time() - started:.0f}s)")

    for reason, count in sorted(skipped.items()):
        print(f"skipped {count:,} profiles: {reason}")
    if not n_profiles:
        print("Error: nothing retained.")
        return 1

    data, n = write(cols, args.out, f"{n_profiles:,} synthetic profiles from {args.raw_dir}/")
    when = pd.to_datetime(data["TIME"], unit="s")
    print(f"\n{n:,} oxygen levels from {n_profiles:,} profiles, "
          f"{len(np.unique(data['PLATFORM']))} floats")
    print(f"time span {when.min():%Y-%m} to {when.max():%Y-%m}; "
          f"lat {data['LATITUDE'].min():.1f} to {data['LATITUDE'].max():.1f}; "
          f"lon {data['LONGITUDE'].min():.1f} to {data['LONGITUDE'].max():.1f}")
    print(f"delayed/adjusted oxygen: {data['DOXY_ADJUSTED_FLAG'].mean():.0%} of levels")
    interp = data["TS_INTERP_FLAG"].astype(bool)
    print(f"levels with measured T/S: {(~interp).sum():,} ({(~interp).mean():.0%}); "
          f"with interpolated T/S (flagged): {interp.sum():,}")
    print(f"oxygen range {np.nanmin(data['DOXY']):.1f} to {np.nanmax(data['DOXY']):.1f} umol/kg; "
          f"median levels per profile {n / n_profiles:.0f}")
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) "
          f"in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
