"""Rates with float-bootstrap confidence intervals, and their dependence on distance and length.

Reads the records of paper_qc.bench. Profiles and levels of one float are not independent,
so intervals come from resampling floats with replacement (a cluster bootstrap, 2,000
draws), pooling all repeats of a scenario. Every rate is a ratio of sums, so a draw is a
float-weight vector applied to per-float numerators and denominators.

Writes tables/ci_summary.csv, tables/ci_distance.csv, tables/ci_length.csv and
figures/detection_ci.pdf, figures/distance.pdf.

    python -m paper_qc.ci
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES, FIGS = os.path.join(HERE, "tables"), os.path.join(HERE, "figures")
B = 2000
SEED = 11
BANDS = [0, 50, 100, 200, 400, 800, 20000]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# (name, table, numerator, denominator, row filter)
RATES = [
    ("level_detection", "P", "lv_bad_flag", "lv_bad", None),
    ("level_false_alarm", "P", "lv_clean_flag", "lv_clean", None),
    ("profile_detection", "P", "prof_flag", "one", "prof_bad"),
    ("profile_false_alarm", "P", "prof_flag", "one", "~prof_bad"),
    ("float_detection", "F", "alarm", "one", "corrupted"),
    ("float_false_alarm", "F", "alarm", "one", "~corrupted"),
    # first false alarms per at-risk profile on clean floats (x100 below): the calibrated rate
    ("float_false_alarm_rate", "F", "first_alarm", "at_risk", "~corrupted"),
]


def load(detector, tag=""):
    P = pd.read_parquet(os.path.join(TABLES, f"records_{detector}_profiles{tag}.parquet"))
    F = pd.read_parquet(os.path.join(TABLES, f"records_{detector}_floats{tag}.parquet"))
    P["one"] = 1
    F["one"] = 1
    return P, F


def _filtered(t, filt):
    if filt is None:
        return t
    return t[~t[filt[1:]]] if filt.startswith("~") else t[t[filt]]


def boot(per_float_num, per_float_den, rng):
    """Point estimate and 95 % interval of sum(num)/sum(den) under float resampling."""
    num, den = np.asarray(per_float_num, float), np.asarray(per_float_den, float)
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    n = len(num)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ num) / (w @ den)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return num.sum() / den.sum(), lo, hi


def summarise(P, F, group_cols, rng):
    rows = []
    keys = P[group_cols].drop_duplicates().itertuples(index=False)
    tables = {"P": P, "F": F}
    for key in keys:
        key = dict(zip(group_cols, key))
        row = dict(key)
        for name, tab, num, den, filt in RATES:
            t = tables[tab]
            for c, v in key.items():
                if c in t:
                    t = t[t[c] == v]
            t = _filtered(t, filt)
            if not len(t):
                continue
            g = t.groupby("PLATFORM")[[num, den]].sum()
            est, lo, hi = boot(g[num], g[den], rng)
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = est, lo, hi
        f = F
        for c, v in key.items():
            if c in f:
                f = f[f[c] == v]
        for k in ("float_false_alarm_rate", "float_false_alarm_rate_lo", "float_false_alarm_rate_hi"):
            if k in row:
                row[k.replace("_rate", "_per_100")] = 100 * row.pop(k)
        d = f[f.corrupted & f.alarm].delay
        row["median_delay"] = float(d.median()) if len(d) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def by_distance(P, scenarios, rng):
    """Profile detection (named scenarios) and false alarms (all scenarios) per distance band."""
    P = P.copy()
    P["band"] = pd.cut(P.dist_km, BANDS, right=False,
                       labels=[f"{a}-{b}" for a, b in zip(BANDS[:-1], BANDS[1:])])
    rows = []
    for band, t in P.groupby("band", observed=True):
        clean = t[~t.prof_bad]
        g = clean.groupby("PLATFORM")[["prof_flag", "one", "lv_clean_flag", "lv_clean"]].sum()
        fa, fa_lo, fa_hi = boot(g.prof_flag, g.one, rng)
        lfa, lfa_lo, lfa_hi = boot(g.lv_clean_flag, g.lv_clean, rng)
        row = {"band": band, "profiles_clean": int(len(clean)), "floats": int(t.PLATFORM.nunique()),
               "profile_false_alarm": fa, "profile_false_alarm_lo": fa_lo,
               "profile_false_alarm_hi": fa_hi, "level_false_alarm": lfa,
               "level_false_alarm_lo": lfa_lo, "level_false_alarm_hi": lfa_hi}
        for s in scenarios:
            b = t[(t.scenario == s) & t.prof_bad]
            if len(b):
                g = b.groupby("PLATFORM")[["prof_flag", "one"]].sum()
                row[s], row[f"{s}_lo"], row[f"{s}_hi"] = boot(g.prof_flag, g.one, rng)
                row[f"{s}_n"] = int(len(b))
        rows.append(row)
    return pd.DataFrame(rows)


def by_length(F, rng):
    """Float false alarms on clean floats by record length (profiles per float), terciles."""
    c = F[~F.corrupted].copy()
    c["length"] = pd.qcut(c.profiles, 3, duplicates="drop")
    rows = []
    for (det, L), t in c.groupby(["detector", "length"], observed=True):
        g = t.groupby("PLATFORM")[["alarm", "one"]].sum()
        est, lo, hi = boot(g.alarm, g.one, rng)
        rows.append({"detector": det, "profiles_per_float": str(L), "floats": len(g),
                     "float_false_alarm": est, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.edgecolor": AXIS,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
        "grid.linewidth": 0.5, "savefig.dpi": 300, "legend.frameon": True,
        "legend.framealpha": 1.0, "legend.facecolor": "white", "legend.edgecolor": GRIDC,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


PANELS = [("offset", "PSAL", "salinity offset (psu)"), ("drift", "PSAL", "salinity drift (psu / yr)"),
          ("offset", "TEMP", "temperature offset (C)"), ("pressure", "PRES", "pressure offset (dbar)"),
          ("offset", "DOXY", "oxygen offset (umol/kg)"), ("gain", "DOXY", "oxygen gain loss (fraction)"),
          ("drift", "DOXY", "oxygen gain drift (fraction / yr)"), ("spike", "DOXY", "oxygen spike (umol/kg)")]


def figure_detection(S):
    style()
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.7), layout="constrained", sharey=True)
    series = [("rtqc", "level_detection", "real-time tests, levels", MUTED, "s", "--"),
              ("rtqc", "float_detection", "real-time tests, floats", MUTED, "x", ":"),
              ("model", "level_detection", "model, levels", BLUE, "o", "-"),
              ("model", "profile_detection", "model, profiles", AQUA, "^", "-"),
              ("model", "float_detection", "model, floats", ORANGE, "D", "-")]
    for ax, (kind, var, xl), letter in zip(axes.ravel(), PANELS, "abcdefgh"):
        for det, col, label, color, marker, ls in series:
            d = S[(S.detector == det) & (S.kind == kind) & (S.variable == var)].sort_values("magnitude")
            if not len(d) or col not in d or d[col].isna().all():
                continue
            ax.fill_between(d.magnitude, d[f"{col}_lo"], d[f"{col}_hi"], color=color, alpha=0.15,
                            lw=0)
            ax.plot(d.magnitude, d[col], color=color, marker=marker, ms=2.8, lw=1.0, ls=ls,
                    label=label)
        d = S[(S.kind == kind) & (S.variable == var)].magnitude.drop_duplicates().sort_values()
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(d.to_numpy())
        ax.set_xticklabels([f"{m:g}" for m in d], fontsize=5.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(xl)
        ax.set_title(f"({letter})", loc="left", fontsize=7, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction detected")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=5)
    out = os.path.join(FIGS, "detection_ci.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return out


def figure_distance(D, scenarios, labels):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), layout="constrained")
    x = np.arange(len(D))
    for s, lab, color, marker in zip(scenarios, labels, (BLUE, ORANGE, AQUA), "o^D"):
        if s not in D:
            continue
        axes[0].fill_between(x, D[f"{s}_lo"], D[f"{s}_hi"], color=color, alpha=0.15, lw=0)
        axes[0].plot(x, D[s], color=color, marker=marker, ms=3, lw=1.1, label=lab)
    axes[0].set_ylabel("profiles detected")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(loc="lower left")
    for col, lab, color, marker in (("profile_false_alarm", "profiles", AQUA, "^"),
                                    ("level_false_alarm", "levels", BLUE, "o")):
        axes[1].fill_between(x, D[f"{col}_lo"], D[f"{col}_hi"], color=color, alpha=0.15, lw=0)
        axes[1].plot(x, D[col], color=color, marker=marker, ms=3, lw=1.1, label=lab)
    axes[1].set_ylabel("false-alarm rate (clean data)")
    axes[1].legend(loc="upper left")
    for ax, letter in zip(axes, "ab"):
        ax.set_xticks(x)
        # float counts under each band: the far bands rest on few floats
        ax.set_xticklabels([f"{b}\n({n} floats)" for b, n in zip(D.band.astype(str), D.floats)],
                           fontsize=5.5)
        ax.set_xlabel("distance to nearest training-float profile (km)")
        ax.set_title(f"({letter})", loc="left", fontsize=7, color=INK)
    out = os.path.join(FIGS, "distance.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return out


DIST_SCENARIOS = ["offset_PSAL_0.1", "gain_DOXY_0.1", "drift_PSAL_0.05"]
DIST_LABELS = ["salinity offset 0.1 psu", "oxygen gain loss 10 %", "salinity drift 0.05 psu/yr"]


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    rng = np.random.default_rng(SEED)
    os.makedirs(FIGS, exist_ok=True)
    Pm, Fm = load("model", tag)
    Pr, Fr = load("rtqc", tag)
    cols = ["detector", "scenario", "kind", "variable", "magnitude"]
    S = pd.concat([summarise(Pm, Fm, cols, rng), summarise(Pr, Fr, cols, rng)], ignore_index=True)
    S.to_csv(os.path.join(TABLES, "ci_summary.csv"), index=False)
    D = by_distance(Pm, DIST_SCENARIOS, rng)
    D.to_csv(os.path.join(TABLES, "ci_distance.csv"), index=False)
    L = by_length(pd.concat([Fm, Fr], ignore_index=True), rng)
    L.to_csv(os.path.join(TABLES, "ci_length.csv"), index=False)

    show = ["detector", "scenario", "level_detection", "level_detection_lo", "level_detection_hi",
            "float_detection", "float_detection_lo", "float_detection_hi",
            "float_false_alarm", "float_false_alarm_lo", "float_false_alarm_hi",
            "float_false_alarm_per_100", "float_false_alarm_per_100_lo",
            "float_false_alarm_per_100_hi", "median_delay"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(S[show].round(3).to_string(index=False))
        print("\nby distance (model):")
        print(D.round(3).to_string(index=False))
        print("\nfloat false alarms by record length:")
        print(L.round(3).to_string(index=False))
    print("wrote", figure_detection(S), figure_distance(D, DIST_SCENARIOS, DIST_LABELS))


if __name__ == "__main__":
    main()
