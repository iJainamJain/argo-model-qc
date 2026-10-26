"""The Argo real-time QC tests that can see T, S and oxygen values: the baseline to beat.

Thresholds are those of the Argo Quality Control Manual for CTD and Trajectory Data v3.9
(20 Feb 2025, doi:10.13155/33951) and the Argo Quality Control Manual for Dissolved Oxygen
Concentration v2.2 (15 Oct 2025, doi:10.13155/46542). Tests on position, date, platform and
the trajectory are out of scope - the injected errors never touch them - and so is the
exclusion list (test 15), which is a record of past human decisions, not a test of the data.

    per profile  6 global range, 7 regional range (Red Sea), 8 pressure increasing, 9 spike,
                 11 gradient (oxygen only; obsolete for T/S since 2019), 12 digit rollover,
                 13 stuck value, 14 density inversion
    per float    16 gross T/S drift against the previous good profile, 18 frozen profile

`run(df)` takes a level table (PLATFORM, CYCLE, PRES, TEMP, PSAL, DOXY, LATITUDE, LONGITUDE)
and returns boolean columns fail_<VAR> plus the test that fired first for each variable.
Oxygen's blanket real-time flag of 3 (policy since 2020) is deliberately not reproduced: it
flags everything, so it detects nothing.
"""
import numpy as np
import pandas as pd

from paper_qc.eos import potential_density

VARS = ("TEMP", "PSAL", "DOXY")
GLOBAL = {"TEMP": (-2.5, 40.0), "PSAL": (2.0, 41.0), "DOXY": (-5.0, 600.0)}
RED_SEA_POLY = np.array([(40.0, 10.0), (50.0, 20.0), (30.0, 30.0), (40.0, 10.0)])  # lon, lat
RED_SEA = {"TEMP": (21.0, 40.0), "PSAL": (2.0, 41.0)}
SPIKE = {"TEMP": (6.0, 2.0), "PSAL": (0.9, 0.3), "DOXY": (50.0, 25.0)}  # (<500, >=500 dbar)
GRADIENT = {"DOXY": (50.0, 25.0)}
ROLLOVER = {"TEMP": 10.0, "PSAL": 5.0}
DENSITY_INVERSION = 0.03
GROSS_DRIFT = {"TEMP": 1.0, "PSAL": 0.5}
FROZEN = {"TEMP": (0.3, 0.001, 0.02), "PSAL": (0.3, 0.001, 0.004)}  # max, min, mean
PRES_REVERSAL = 20.0
SLAB_EDGES = np.arange(0.0, 6000.0, 50.0)


def _in_polygon(lon, lat, poly):
    """Ray casting; poly is (n, 2) lon/lat with the first vertex repeated at the end."""
    inside = False
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _three_point(v, p, limits, subtract_slope):
    """Spike (test 9) or gradient (test 11) on consecutive valid values."""
    out = np.zeros(len(v), bool)
    if len(v) < 3:
        return out
    v1, v2, v3 = v[:-2], v[1:-1], v[2:]
    tv = np.abs(v2 - (v3 + v1) / 2)
    if subtract_slope:
        tv = tv - np.abs((v3 - v1) / 2)
    lim = np.where(p[1:-1] < 500.0, limits[0], limits[1])
    out[1:-1] = tv > lim
    return out


def _pressure_increasing(p):
    """Test 8, run from the middle of the profile outwards."""
    n = len(p)
    bad = np.zeros(n, bool)
    if n <= 2:
        return bad
    mid = n // 2
    hi = p[mid]
    for i in range(mid + 1, n):          # middle -> deep
        if p[i] <= hi - PRES_REVERSAL:
            bad[i] = True
        hi = max(hi, p[i])
    lo = p[mid]
    for i in range(mid - 1, -1, -1):     # middle -> shallow
        if p[i] >= lo + PRES_REVERSAL:
            bad[i] = True
        lo = min(lo, p[i])
    return bad


def profile_tests(p, values, lat, lon):
    """Per-profile tests. `values` maps VAR -> array aligned with pressure `p` (NaN = absent).

    Returns {VAR: (fail bool array, first-test name array)}.
    """
    n = len(p)
    fails = {v: np.zeros(n, bool) for v in values}
    first = {v: np.full(n, "", dtype=object) for v in values}

    def mark(var, mask, name):
        new = mask & ~fails[var]
        first[var][new] = name
        fails[var] |= mask

    red_sea = _in_polygon(lon, lat, RED_SEA_POLY)
    bad_p = _pressure_increasing(p)

    for var, v in values.items():
        ok = np.isfinite(v)
        lo_, hi_ = GLOBAL[var]
        mark(var, ok & ((v < lo_) | (v > hi_)), "6 global range")
        if red_sea and var in RED_SEA:
            lo_, hi_ = RED_SEA[var]
            mark(var, ok & ((v < lo_) | (v > hi_)), "7 regional range")
        mark(var, bad_p, "8 pressure increasing")
        idx = np.flatnonzero(ok)
        vv, pp = v[idx], p[idx]
        if var in SPIKE:
            m = np.zeros(n, bool)
            m[idx] = _three_point(vv, pp, SPIKE[var], True)
            mark(var, m, "9 spike")
        if var in GRADIENT:
            m = np.zeros(n, bool)
            m[idx] = _three_point(vv, pp, GRADIENT[var], False)
            mark(var, m, "11 gradient")
        if var in ROLLOVER and len(idx) > 1:
            jump = np.abs(np.diff(vv)) > ROLLOVER[var]
            m = np.zeros(n, bool)
            m[idx[1:][jump]] = True
            m[idx[:-1][jump]] = True
            mark(var, m, "12 digit rollover")
        if len(idx) > 1 and np.all(vv == vv[0]):
            m = np.zeros(n, bool)
            m[idx] = True
            mark(var, m, "13 stuck value")

    # test 14: density inversion, both directions, referenced to each pair's mid-point
    if "TEMP" in values and "PSAL" in values:
        t, s = values["TEMP"], values["PSAL"]
        idx = np.flatnonzero(np.isfinite(t) & np.isfinite(s))
        if len(idx) > 1:
            a, b = idx[:-1], idx[1:]
            pref = (p[a] + p[b]) / 2
            ra = potential_density(s[a], t[a], p[a], pref)
            rb = potential_density(s[b], t[b], p[b], pref)
            inv = ra - rb > DENSITY_INVERSION        # deeper level lighter than shallower
            m = np.zeros(n, bool)
            m[a[inv]] = True
            m[b[inv]] = True
            mark("TEMP", m, "14 density inversion")
            mark("PSAL", m, "14 density inversion")
    return {v: (fails[v], first[v]) for v in values}


def _slab_means(p, v):
    ok = np.isfinite(v)
    k = np.digitize(p[ok], SLAB_EDGES)
    return pd.Series(v[ok]).groupby(k).mean()


def _window_mean(p, v, top, bottom):
    sel = (p >= top) & (p <= bottom)
    return float(np.mean(v[sel])) if sel.any() else np.nan


def _float_tests(p, vals, res, prev_good):
    """Tests 16 and 18 on T and S against the previous good profile; mutates `res`.

    Test 16 compares "the deepest 100 dbar" of the two profiles. Taken literally that compares
    different water whenever consecutive profiles end at different depths - a float profiling
    to 200-350 dbar sees a 2 C "drift" from 50 dbar of thermocline. The manual's intent is a
    sensor drift, so both means are taken over the same window: the deepest 100 dbar of the
    shallower profile. Returns this profile's good values, to become `prev_good`.
    """
    cur = {}
    for v in ("TEMP", "PSAL"):
        if v not in vals:
            continue
        good = ~res[v][0] & np.isfinite(vals[v])
        cur[v] = (p[good], vals[v][good], _slab_means(p[good], vals[v][good]))
    for v in cur:
        fail, first = res[v]
        if prev_good is None or v not in prev_good:
            continue
        pp, pv, pslab = prev_good[v]
        cp, cv, cslab = cur[v]
        if len(cp) and len(pp):
            bottom = min(cp.max(), pp.max())
            a = _window_mean(cp, cv, bottom - 100.0, bottom)
            b = _window_mean(pp, pv, bottom - 100.0, bottom)
            if np.isfinite(a) and np.isfinite(b) and abs(a - b) > GROSS_DRIFT[v]:
                first[~fail] = "16 gross drift"
                fail[:] = True
        common = cslab.index.intersection(pslab.index)
        if len(common) >= 3:
            d = np.abs(cslab[common].to_numpy() - pslab[common].to_numpy())
            mx, mn, me = FROZEN[v]
            if d.max() < mx and d.min() < mn and d.mean() < me:
                first[~fail] = "18 frozen profile"
                fail[:] = True
    return cur


def run(df, variables=VARS):
    """All tests on a level table. Returns a frame aligned with df.index."""
    fail = {v: np.zeros(len(df), bool) for v in variables}
    test = {v: np.full(len(df), "", dtype=object) for v in variables}
    pos = pd.Series(np.arange(len(df)), index=df.index)

    for _, fl in df.groupby("PLATFORM", sort=False):
        prev_good = None
        for _, g in fl.groupby("CYCLE", sort=True):
            g = g.sort_values("PRES", kind="stable")
            p = g.PRES.to_numpy("float64")
            vals = {v: g[v].to_numpy("float64") for v in variables if v in g}
            res = profile_tests(p, vals, float(g.LATITUDE.iloc[0]), float(g.LONGITUDE.iloc[0]))
            cur = _float_tests(p, vals, res, prev_good)
            # a profile whose T or S failed everywhere is not a usable reference
            if cur and all(not res[v][0].all() for v in cur):
                prev_good = cur
            rows = pos[g.index].to_numpy()
            for v in vals:
                fail[v][rows] = res[v][0]
                test[v][rows] = res[v][1]

    out = pd.DataFrame(index=df.index)
    for v in variables:
        out[f"fail_{v}"] = fail[v]
        out[f"test_{v}"] = test[v]
    return out


def _self_test():
    p = np.array([10.0, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500])
    t = np.array([28.0, 27.5, 24.0, 18.0, 12.0, 10.0, 8.0, 6.5, 5.5, 4.5])
    s = np.array([35.0, 35.1, 35.2, 35.1, 35.0, 34.9, 34.85, 34.8, 34.78, 34.75])
    o = np.array([200.0, 198, 150, 80, 20, 10, 15, 30, 60, 90])
    checks = 0

    clean = profile_tests(p, {"TEMP": t, "PSAL": s, "DOXY": o}, lat=-20.0, lon=80.0)
    assert not any(f.any() for f, _ in clean.values()), "clean profile flagged"
    checks += 1

    spiky = o.copy()
    spiky[6] += 60.0                            # 60 umol/kg spike at 800 dbar (limit 25)
    r = profile_tests(p, {"DOXY": spiky}, -20.0, 80.0)["DOXY"]
    assert r[0][6] and r[1][6] == "9 spike", r
    checks += 1

    salty = s.copy()
    salty[7] += 0.5                             # 0.5 psu spike at 1000 dbar (limit 0.3)
    r = profile_tests(p, {"TEMP": t, "PSAL": salty}, -20.0, 80.0)["PSAL"]
    assert r[0][7] and r[1][7] == "9 spike", r
    checks += 1

    fresh = s.copy()
    fresh[8] -= 0.25                            # passes spike (0.25 < 0.3), inverts density
    r = profile_tests(p, {"TEMP": t, "PSAL": fresh}, -20.0, 80.0)["PSAL"]
    assert r[0][8] and r[1][8] == "14 density inversion", r
    checks += 1

    r = profile_tests(p, {"TEMP": t, "PSAL": s + 0.2}, -20.0, 80.0)["PSAL"]
    assert not r[0].any(), "a constant offset should pass every profile test"
    checks += 1

    red = profile_tests(p, {"TEMP": t}, lat=20.0, lon=38.0)["TEMP"]   # Red Sea: T >= 21
    assert red[0][3] and red[1][3] == "7 regional range"
    assert not profile_tests(p, {"TEMP": t}, lat=20.0, lon=60.0)["TEMP"][0].any()
    checks += 1

    # float level: a 0.6 psu jump in the deep mean between cycles triggers test 16
    rows = []
    for c, ds in ((1, 0.0), (2, 0.6)):
        for pi, ti, si in zip(p, t, s):
            rows.append(dict(PLATFORM="X", CYCLE=c, PRES=pi, TEMP=ti + 0.05 * c, PSAL=si + ds,
                             DOXY=np.nan, LATITUDE=-20.0, LONGITUDE=80.0))
    res = run(pd.DataFrame(rows), variables=("TEMP", "PSAL"))
    second = res.iloc[len(p):]
    assert second.fail_PSAL.all() and (second.test_PSAL == "16 gross drift").all()
    assert not res.iloc[:len(p)].fail_PSAL.any()
    checks += 1

    # identical consecutive profiles -> frozen (temperature too, since it repeats exactly)
    rows2 = [dict(r, TEMP=r["TEMP"] - 0.05 * r["CYCLE"],
                  PSAL=r["PSAL"] - (0.6 if r["CYCLE"] == 2 else 0.0)) for r in rows]
    res = run(pd.DataFrame(rows2), variables=("TEMP", "PSAL"))
    assert (res.iloc[len(p):].test_PSAL == "18 frozen profile").all()
    assert (res.iloc[len(p):].test_TEMP == "18 frozen profile").all()
    checks += 1

    # stuck
    r = profile_tests(p, {"DOXY": np.full(len(p), 123.0)}, -20.0, 80.0)["DOXY"]
    assert r[0].all() and (r[1] == "13 stuck value").all()
    checks += 1
    print(f"rtqc self-test: {checks} checks ok")


if __name__ == "__main__":
    _self_test()
