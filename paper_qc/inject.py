"""Plant known errors in held-out floats, so detection can be scored against the truth.

Each scenario corrupts a random half of the floats it is given and leaves the rest clean, so
one evaluation yields both detection rates and false alarms. Errors that persist (offset,
gain, drift, pressure offset) begin at a random onset cycle between 20 % and 60 % of the
float's life - floats usually start well and fail later, and the clean cycles before onset
are what a per-float detector has to learn "normal" from. Transient errors (spike, stuck) hit
a tenth of the profiles of a corrupted float.

    kind       variable            magnitude means
    offset     TEMP, PSAL, DOXY    added from onset (sign random per float)
    gain       DOXY                multiplied by (1 - m) from onset: optode sensitivity loss
    drift      PSAL                m per year since onset, additive (sign random)
    drift      DOXY                gain loss of m per year since onset
    spike      TEMP, PSAL, DOXY    one level per hit profile, +/- m
    stuck      TEMP, PSAL, DOXY    a run of up to 10 levels repeats the value above it
    pressure   PRES                added from onset: every variable appears at the wrong depth

Salinity errors are carried into oxygen, as they would be in a real float: optode oxygen is
computed with the CTD salinity, so a salinity error changes the reported oxygen by the ratio
of solubilities at the wrong and right salinity (Garcia and Gordon 1992, the same salinity
dependence paper_qc.lib.data.oxygen_solubility implements). For small errors this is tiny (0.06 %
per 0.1 psu); for a failed conductivity cell it is not.

Every call returns the corrupted table, per-level truth (bad_<VAR>, delta_<VAR>) and a per-
float record of what was done to it.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paper_qc.lib.data import oxygen_solubility

SECONDS_PER_YEAR = 365.25 * 86400.0
PERSISTENT = {"offset", "gain", "drift", "pressure"}
TRANSIENT = {"spike", "stuck"}
TARGET_OF = {"pressure": "PRES"}


@dataclass(frozen=True)
class Scenario:
    kind: str
    var: str
    magnitude: float

    @property
    def name(self):
        return f"{self.kind}_{self.var}_{self.magnitude:g}"


# The benchmark grid. Magnitudes span "below any plausible detection" to "gross", and bracket
# the real-time thresholds: e.g. PSAL spikes at 0.3 (the >=500 dbar spike limit), deep-mean
# salinity jumps at 0.5 (test 16), oxygen gains around the ~7.6 % median optode storage loss.
SCENARIOS = (
    [Scenario("offset", "PSAL", m) for m in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)]
    + [Scenario("drift", "PSAL", m) for m in (0.01, 0.02, 0.05, 0.1)]
    + [Scenario("offset", "TEMP", m) for m in (0.01, 0.05, 0.1, 0.5)]
    + [Scenario("offset", "DOXY", m) for m in (2.0, 5.0, 10.0, 20.0)]
    + [Scenario("gain", "DOXY", m) for m in (0.01, 0.02, 0.05, 0.1, 0.2)]
    + [Scenario("drift", "DOXY", m) for m in (0.005, 0.01, 0.02, 0.05)]
    + [Scenario("spike", "PSAL", m) for m in (0.05, 0.1, 0.3, 1.0)]
    + [Scenario("spike", "TEMP", m) for m in (0.1, 0.5, 2.0)]
    + [Scenario("spike", "DOXY", m) for m in (5.0, 10.0, 25.0, 50.0)]
    + [Scenario("stuck", v, 10) for v in ("TEMP", "PSAL", "DOXY")]
    + [Scenario("pressure", "PRES", m) for m in (5.0, 10.0, 20.0, 50.0)]
)


def _onsets(fl_cycles, rng):
    """Onset cycle per float, 20-60 % of the way through its own cycles."""
    out = {}
    for f, cycles in fl_cycles.items():
        c = np.sort(cycles)
        k = int(rng.uniform(0.2, 0.6) * len(c))
        out[f] = c[min(k, len(c) - 1)]
    return out


def _seconds(time):
    """Epoch seconds from either datetime64 (paper_qc.lib.data.load) or numeric seconds."""
    time = pd.Series(time)
    if np.issubdtype(time.dtype, np.datetime64):
        return time.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
    return time.to_numpy("float64")


def _salinity_ratio(s_true, s_bad, t):
    """Factor by which a salinity error changes optode oxygen (solubility ratio)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = oxygen_solubility(s_bad, t) / oxygen_solubility(s_true, t)
    return np.where(np.isfinite(r), r, 1.0)


def inject(df, scenario, seed=0, corrupt_fraction=0.5, hit_fraction=0.1,
           propagate_salinity=True):
    """Corrupt a copy of `df` (a level table of held-out floats) under one scenario.

    Returns (corrupted, truth, floats):
      corrupted  same columns as df, values changed where the error applies
      truth      bad_<VAR> (bool) and delta_<VAR> (corrupted - original) for PRES, TEMP,
                 PSAL, DOXY, aligned with df.index
      floats     one row per float: corrupted?, onset cycle, sign
    """
    rng = np.random.default_rng(seed)
    kind, var, m = scenario.kind, scenario.var, float(scenario.magnitude)
    target = TARGET_OF.get(kind, var)

    out = df.copy()
    truth = pd.DataFrame(index=df.index)
    for v in ("PRES", "TEMP", "PSAL", "DOXY"):
        truth[f"bad_{v}"] = False

    platforms = df.PLATFORM.unique()
    n_bad = int(round(corrupt_fraction * len(platforms)))
    bad_floats = set(rng.choice(platforms, size=n_bad, replace=False)) if n_bad else set()
    cycles = df.groupby("PLATFORM").CYCLE.unique().to_dict()
    onset = _onsets(cycles, rng)
    sign = {f: rng.choice([-1.0, 1.0]) for f in platforms}

    on_bad = df.PLATFORM.isin(bad_floats).to_numpy()
    if kind in PERSISTENT:
        onset_cycle = df.PLATFORM.map(onset).to_numpy()
        hit = on_bad & (df.CYCLE.to_numpy() >= onset_cycle)
        sgn = df.PLATFORM.map(sign).to_numpy()
        x = df[target].to_numpy("float64")
        if kind == "offset":
            new = x + sgn * m
        elif kind == "pressure":
            new = x + m
        elif kind == "gain":
            new = x * (1.0 - m)
        elif kind == "drift":
            # time since the onset profile of the same float
            secs = _seconds(df.TIME)
            t_on = (pd.DataFrame({"PLATFORM": df.PLATFORM.to_numpy(), "s": secs,
                                  "on": df.CYCLE.to_numpy() == onset_cycle})
                    .query("on").groupby("PLATFORM").s.min())
            years = (secs - df.PLATFORM.map(t_on).to_numpy("float64")) / SECONDS_PER_YEAR
            years = np.clip(years, 0.0, None)
            new = x * (1.0 - m * years) if var == "DOXY" else x + sgn * m * years
        else:
            raise ValueError(kind)
        vals = np.where(hit, new, x)
        out[target] = vals
        truth[f"bad_{target}"] = hit

    elif kind in TRANSIENT:
        hit = np.zeros(len(df), bool)
        x = df[target].to_numpy("float64")
        vals = x.copy()
        prof = df[on_bad].groupby(["PLATFORM", "CYCLE"]).indices
        pos_of = np.flatnonzero(on_bad)
        keys = list(prof)
        chosen = rng.choice(len(keys), size=int(round(hit_fraction * len(keys))),
                            replace=False) if keys else []
        for k in chosen:
            rows = pos_of[prof[keys[k]]]
            rows = rows[np.argsort(df.PRES.to_numpy()[rows], kind="stable")]
            rows = rows[np.isfinite(x[rows])]
            if len(rows) < 3:
                continue
            if kind == "spike":
                i = rows[rng.integers(1, len(rows) - 1)]     # never the end levels
                vals[i] = x[i] + rng.choice([-1.0, 1.0]) * m
                hit[i] = True
            else:  # stuck: a run repeats the value of the level above it
                start = rng.integers(0, len(rows) - 2)
                run = rows[start + 1: start + 1 + int(m)]
                vals[run] = x[rows[start]]
                hit[run] = vals[run] != x[run]
        out[target] = vals
        truth[f"bad_{target}"] = hit
    else:
        raise ValueError(kind)

    if propagate_salinity and target == "PSAL" and "DOXY" in df:
        ratio = _salinity_ratio(df.PSAL.to_numpy("float64"), out.PSAL.to_numpy("float64"),
                                df.TEMP.to_numpy("float64"))
        out["DOXY"] = df.DOXY.to_numpy("float64") * ratio

    for v in ("PRES", "TEMP", "PSAL", "DOXY"):
        if v in df:
            truth[f"delta_{v}"] = (out[v].to_numpy("float64") - df[v].to_numpy("float64"))

    floats = pd.DataFrame({
        "platform": platforms,
        "corrupted": [f in bad_floats for f in platforms],
        "onset_cycle": [onset[f] if kind in PERSISTENT else -1 for f in platforms],
        "sign": [sign[f] for f in platforms],
    })
    floats["scenario"] = scenario.name
    return out, truth, floats


def _self_test():
    rng = np.random.default_rng(1)
    rows = []
    for f in range(10):
        for c in range(1, 21):
            for p in np.linspace(10, 1500, 25):
                rows.append(dict(PLATFORM=f"F{f}", CYCLE=c, TIME=c * 10 * 86400.0, PRES=p,
                                 TEMP=20 - p / 100, PSAL=35.0 + rng.normal(0, 0.001),
                                 DOXY=200 - p / 10))
    df = pd.DataFrame(rows)
    n_checks = 0

    out, truth, floats = inject(df, Scenario("offset", "PSAL", 0.1), seed=3)
    assert floats.corrupted.sum() == 5
    bad = truth.bad_PSAL.to_numpy()
    assert np.allclose(np.abs(truth.delta_PSAL[bad]), 0.1)
    assert np.all(truth.delta_PSAL[~bad] == 0)
    # clean floats untouched everywhere, including oxygen
    clean = df.PLATFORM.isin(floats.platform[~floats.corrupted]).to_numpy()
    assert (truth.loc[clean, ["delta_PSAL", "delta_DOXY"]].abs().to_numpy() == 0).all()
    # salinity error carried into oxygen, small and of the right sign (fresher -> more O2)
    d_o2 = truth.delta_DOXY[bad].to_numpy() / df.DOXY[bad].to_numpy()
    d_s = truth.delta_PSAL[bad].to_numpy()
    assert np.all(np.abs(d_o2) < 1e-3) and np.all(np.sign(d_o2) == -np.sign(d_s))
    # onset: nothing before it, everything after it
    for _, r in floats[floats.corrupted].iterrows():
        sel = (df.PLATFORM == r.platform).to_numpy()
        assert not bad[sel & (df.CYCLE < r.onset_cycle).to_numpy()].any()
        assert bad[sel & (df.CYCLE >= r.onset_cycle).to_numpy()].all()
    n_checks += 5

    out, truth, _ = inject(df, Scenario("drift", "DOXY", 0.05), seed=4)
    rel = -truth.delta_DOXY / df.DOXY
    assert rel.max() <= 0.05 * (20 * 10 / 365.25) + 1e-9 and rel.min() >= 0
    # the same with datetime64 time, as paper_qc.lib.data.load delivers it
    dt = df.assign(TIME=pd.to_datetime(df.TIME, unit="s"))
    _, truth2, _ = inject(dt, Scenario("drift", "DOXY", 0.05), seed=4)
    assert np.allclose(truth2.delta_DOXY, truth.delta_DOXY)
    n_checks += 1

    out, truth, _ = inject(df, Scenario("spike", "DOXY", 25.0), seed=5)
    per_prof = truth[truth.bad_DOXY].groupby([df.PLATFORM, df.CYCLE]).size()
    assert (per_prof == 1).all() and np.allclose(truth.delta_DOXY[truth.bad_DOXY].abs(), 25)
    n_checks += 1

    out, truth, _ = inject(df, Scenario("stuck", "PSAL", 10), seed=6)
    assert truth.bad_PSAL.any()
    n_checks += 1

    out, truth, _ = inject(df, Scenario("pressure", "PRES", 20.0), seed=7)
    assert np.allclose(truth.delta_PRES[truth.bad_PRES], 20.0)
    n_checks += 1

    for s in SCENARIOS:               # every scenario runs and is reproducible
        a = inject(df, s, seed=11)[1]
        b = inject(df, s, seed=11)[1]
        assert a.equals(b), s
    n_checks += 1
    print(f"inject self-test: {n_checks} checks ok; {len(SCENARIOS)} scenarios")


if __name__ == "__main__":
    _self_test()
