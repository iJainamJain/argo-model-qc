"""Real-time tests against the model-based scores, scenario by scenario.

Reads tables/baseline_rtqc.csv (paper_qc.baseline_rtqc) and tables/score_benchmark.csv
(paper_qc.score), which planted identical errors (same scenario seeds) in the same clean test
floats. For the real-time tests only flags caused by the planted error count (the *_new
columns); their false alarms on clean data are reported separately, as are the model's.

Writes tables/comparison.csv and figures/detection.pdf.

    python -m paper_qc.compare
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES, FIGS = os.path.join(HERE, "tables"), os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

PANELS = [  # (kind, variable, x label)
    ("offset", "PSAL", "salinity offset (psu)"),
    ("drift", "PSAL", "salinity drift (psu / yr)"),
    ("offset", "TEMP", "temperature offset (C)"),
    ("pressure", "PRES", "pressure offset (dbar)"),
    ("offset", "DOXY", "oxygen offset (umol/kg)"),
    ("gain", "DOXY", "oxygen gain loss (fraction)"),
    ("drift", "DOXY", "oxygen gain drift (fraction / yr)"),
    ("spike", "DOXY", "oxygen spike (umol/kg)"),
]


def load():
    rt = pd.read_csv(os.path.join(TABLES, "baseline_rtqc.csv"))
    md = pd.read_csv(os.path.join(TABLES, "score_benchmark.csv"))
    keep_rt = ["scenario", "kind", "variable", "magnitude", "levels_bad",
               "level_detection_new", "float_detection_new"]
    rt = rt[keep_rt].rename(columns={"level_detection_new": "rt_level",
                                     "float_detection_new": "rt_float"})
    md = md.rename(columns={c: f"model_{c}" for c in md.columns
                            if c not in ("scenario", "kind", "variable", "magnitude")})
    return rt.merge(md, on=["scenario", "kind", "variable", "magnitude"])


def false_alarms():
    fa = pd.read_csv(os.path.join(TABLES, "rtqc_false_alarms.csv"))
    fa = fa[fa.test == "any"].set_index("variable")
    with open(os.path.join(TABLES, "score_thresholds.json")) as f:
        cal = json.load(f)
    return fa, cal


def figure(c):
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.edgecolor": AXIS,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
        "grid.linewidth": 0.5, "savefig.dpi": 300, "legend.frameon": True,
        "legend.framealpha": 1.0, "legend.facecolor": "white", "legend.edgecolor": GRIDC,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6), layout="constrained", sharey=True)
    series = [("rt_level", "real-time tests (levels)", MUTED, "s", "--"),
              ("model_level_detection", "model, levels", BLUE, "o", "-"),
              ("model_profile_detection", "model, profiles", AQUA, "^", "-"),
              ("model_float_detection", "model, floats (moving window)", ORANGE, "D", "-")]
    for ax, (kind, var, xl), letter in zip(axes.ravel(), PANELS, "abcdefgh"):
        d = c[(c.kind == kind) & (c.variable == var)].sort_values("magnitude")
        for col, label, color, marker, ls in series:
            if col in d and d[col].notna().any():
                ax.plot(d.magnitude, d[col], color=color, marker=marker, ms=3, lw=1.1, ls=ls,
                        label=label)
        ax.set_xscale("log")
        # label exactly the tested magnitudes; default log labels collide at this size
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(d.magnitude.to_numpy())
        ax.set_xticklabels([f"{m:g}" for m in d.magnitude], fontsize=5.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(xl)
        ax.set_title(f"({letter})", loc="left", fontsize=7, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction detected")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "detection.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    return out


def main():
    c = load()
    c.to_csv(os.path.join(TABLES, "comparison.csv"), index=False)
    fa, cal = false_alarms()
    cols = ["scenario", "rt_level", "model_level_detection", "model_level_auc",
            "model_profile_detection", "rt_float", "model_float_detection",
            "model_float_false_alarm", "model_median_delay_cycles"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(c[cols].round(3).to_string(index=False))
    print("\nfalse alarms on clean test floats")
    for v in fa.index:
        print(f"  real-time tests, {v}: {fa.loc[v, 'rate']:.4f} of levels, "
              f"{fa.loc[v, 'floats']} floats")
    m = pd.read_csv(os.path.join(TABLES, "score_benchmark.csv"))
    print(f"  model: level {m.level_false_alarm.median():.4f} (threshold set at 1% on val), "
          f"profile {m.profile_false_alarm.median():.4f}, float {m.float_false_alarm.median():.3f} "
          f"(5% on val)")
    print("wrote", figure(c))


if __name__ == "__main__":
    main()
