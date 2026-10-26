"""Fast reads of Argo synthetic-profile (S-) files.

xarray spends ~3 s decoding each of these small files on this machine; netCDF4 directly takes
~0.015 s, which is the difference between days and minutes over the 50,214 delayed-mode files.
Only the first profile (N_PROF = 0, the primary sampling scheme) is read.
"""
import glob
import os

import netCDF4
import numpy as np

RAW_ROOT = os.path.join("data", "argo_synthetic")


def s_file(wmo, cycle, root=RAW_ROOT):
    """Path of the S-file for one float cycle (delayed or real-time), or None."""
    hits = glob.glob(os.path.join(root, "*", str(wmo), "profiles", f"S?{wmo}_{int(cycle):03d}.nc"))
    return hits[0] if hits else None


def _strings(var):
    """Fixed-width char array (..., STRLEN) -> list of stripped strings, first profile only."""
    a = np.ma.filled(var[:], b" ")
    a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
    return ["".join(c.decode(errors="ignore") if isinstance(c, bytes) else str(c) for c in row).strip()
            for row in a]


def _chars(arr):
    a = np.ma.filled(np.atleast_1d(arr), b" ")
    return np.array([c.decode() if isinstance(c, bytes) else str(c) for c in a.ravel()])


def read(path, parameters=("PRES", "TEMP", "PSAL", "DOXY")):
    """One profile: metadata, per-parameter data mode, and raw/adjusted values and flags.

    Returns a dict with 'platform', 'cycle', 'juld', 'lat', 'lon', 'direction', 'mode'
    ({parameter: 'R'|'A'|'D'}), and for each parameter P present: P, P_ADJUSTED (float arrays,
    NaN where missing) and P_QC, P_ADJUSTED_QC (single-character string arrays).
    """
    with netCDF4.Dataset(path) as d:
        d.set_auto_chartostring(False)
        names = _strings(d["STATION_PARAMETERS"])[: d.dimensions["N_PARAM"].size]
        modes = _chars(d["PARAMETER_DATA_MODE"][0]) if "PARAMETER_DATA_MODE" in d.variables else []
        out = {
            "platform": _strings(d["PLATFORM_NUMBER"])[0],
            "cycle": int(np.ma.filled(d["CYCLE_NUMBER"][0], -1)),
            "juld": float(np.ma.filled(d["JULD"][0], np.nan)),
            "lat": float(np.ma.filled(d["LATITUDE"][0], np.nan)),
            "lon": float(np.ma.filled(d["LONGITUDE"][0], np.nan)),
            "mode": {n: str(m) for n, m in zip(names, modes)},
        }
        for p in parameters:
            for suffix in ("", "_ADJUSTED"):
                name = p + suffix
                if name in d.variables:
                    out[name] = np.ma.filled(d[name][0].astype("float64"), np.nan)
                if name + "_QC" in d.variables:
                    out[name + "_QC"] = _chars(d[name + "_QC"][0])
    return out


if __name__ == "__main__":
    # smoke test on a few delayed-mode files
    import random
    import time
    files = glob.glob(os.path.join(RAW_ROOT, "*", "*", "profiles", "SD*.nc"))
    random.seed(3)
    t0 = time.time()
    for f in random.sample(files, 20):
        r = read(f)
    print(f"20 files in {time.time() - t0:.2f}s; last: float {r['platform']} cycle {r['cycle']}, "
          f"modes {r['mode']}, {np.isfinite(r.get('DOXY_ADJUSTED', np.array([]))).sum()} "
          f"adjusted DOXY levels")
