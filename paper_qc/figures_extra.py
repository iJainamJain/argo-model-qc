"""Three figures of the QC manuscript: the float split, two real failures and the optode gain.

    python -m paper_qc.figures_extra

map.pdf   profile positions by split group (training, validation, clean test, known-bad).
gain.pdf  delayed-mode optode gain against the model's estimate, per profile and per float.
cases.pdf (a) float 2902175 (greylisted "salty drift and wreckage", code 4 from 2016-11-16):
          its deep salinity score per profile against the profile threshold, raw real-time
          view; (b) one profile of float 2902209: raw salinity with its real-time flags against
          the model's prediction and 99 % interval.
Thresholds are the cross-fitted ones (tables/score_thresholds_cv.json).
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_qc import profiles, score
from paper_qc.model import Ensemble

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUPS = [("train", "training (242 floats)", "#b9b8b0", 0.25),
          ("val", "validation (57)", AQUA, 0.45),
          ("test_clean", "clean test (75)", BLUE, 0.45),
          ("test_known_bad", "known-bad test (20)", ORANGE, 0.9)]


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


def figure_map():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    prof = pd.read_parquet(profiles.OUT, columns=["PLATFORM", "CYCLE", "LATITUDE", "LONGITUDE",
                                                   "group"])
    prof = prof.groupby(["PLATFORM", "CYCLE"]).first().reset_index()
    fig = plt.figure(figsize=(4.6, 3.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([20, 120, -60, 30])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efeeea", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=MUTED, linewidth=0.4)
    for g, label, color, alpha in GROUPS:
        p = prof[prof.group == g]
        ax.scatter(p.LONGITUDE, p.LATITUDE, s=0.8 if g != "test_known_bad" else 1.5,
                   color=color, alpha=alpha, linewidths=0, transform=ccrs.PlateCarree(),
                   rasterized=True, label=f"{label}")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=GRIDC)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6, "color": INK2}
    leg = ax.legend(loc="lower left", markerscale=6)
    for h in leg.legend_handles:
        h.set_alpha(1)
    out = os.path.join(FIGS, "map.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return out, int(len(prof))


def figure_cases():
    cal = json.load(open(os.path.join(HERE, "tables", "score_thresholds_cv.json")))
    ens_s = Ensemble(os.path.join("runs_qc", "psal"))
    raw = pd.read_parquet(profiles.OUT_RAW, filters=[("group", "==", "test_known_bad")])
    raw = raw[~raw.TEMP_QC.isin(["4"]) & ~raw.PRES_QC.isin(["4"])].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1.7, 1]})
    # (a) 2902175 through its greylisting
    f = raw[raw.PLATFORM == "2902175"].reset_index(drop=True)
    mu, sd = ens_s.predict(f)
    zS = (f.PSAL.to_numpy("float64") - mu) / sd
    none = np.full(len(f), np.nan)
    ps = score.profile_score(score.profile_means(f, none, zS), cal)
    when = f.groupby(["PLATFORM", "CYCLE"]).TIME.first()
    t = when.reindex(ps.index).to_numpy()
    s = ps.score.to_numpy()
    shown = np.clip(s, None, 60)
    ax = axes[0]
    ax.plot(t, shown, color=BLUE, lw=0.8, marker="o", ms=1.8)
    ax.axhline(cal["profile_threshold"], color=ORANGE, lw=0.9, ls="--",
               label=f"profile threshold ({cal['profile_threshold']:.1f})")
    start = pd.Timestamp("2016-11-16")
    ax.axvline(start, color=INK2, lw=0.8, ls=":")
    ax.text(start, 52, " greylisted from\n 2016-11-16 (code 4)", fontsize=6, color=INK2,
            va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_ylim(0, 62)
    ax.set_ylabel("salinity profile score (clipped at 60)")
    ax.set_title("(a) float 2902175, raw real-time salinity", loc="left", fontsize=7, color=INK)
    ax.legend(loc="upper left")

    # (b) one 2902209 profile with impossible salinity
    g = raw[raw.PLATFORM == "2902209"]
    fresh = g[g.PSAL < 30].groupby("CYCLE").size()
    cyc = int(fresh[fresh.index.isin(g[(g.PSAL < 30) & (g.PSAL_QC == "1")].CYCLE)].idxmax())
    p = g[g.CYCLE == cyc].sort_values("PRES").reset_index(drop=True)
    m, sdev = ens_s.predict(p)
    ax = axes[1]
    ax.fill_betweenx(p.PRES, m - 2.576 * sdev, m + 2.576 * sdev, color=BLUE, alpha=0.2, lw=0,
                     label="model 99 % interval")
    ax.plot(m, p.PRES, color=BLUE, lw=1.0, label="model prediction")
    for flag, color, label in (("1", ORANGE, "raw, flag 1 (good)"),
                               ("3", MUTED, "raw, flag 3"), ("4", INK, "raw, flag 4 (bad)")):
        q = p[p.PSAL_QC == flag]
        if len(q):
            ax.scatter(q.PSAL, q.PRES, s=3, color=color, linewidths=0, label=label, zorder=3)
    ax.set_ylim(p.PRES.max() * 1.02, 0)
    ax.set_xlabel("practical salinity")
    ax.set_ylabel("pressure (dbar)")
    ax.set_title(f"(b) float 2902209, cycle {cyc}", loc="left", fontsize=7, color=INK)
    ax.legend(loc="lower left")
    out = os.path.join(FIGS, "cases.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    n1 = int(((p.PSAL < 30) & (p.PSAL_QC == "1")).sum())
    return out, cyc, n1, int((p.PSAL < 30).sum())


def figure_gain(lo=0.8, hi=1.4):
    """Delayed-mode optode gain against the model's real-time estimate (the negative result)."""
    m = pd.read_csv(os.path.join(HERE, "tables", "real_gain.csv.gz"), dtype={"PLATFORM": str})
    f = m.groupby("PLATFORM")[["model_gain", "dm_gain"]].median()
    med = float(m.dm_gain.median())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), layout="constrained", sharey=True)
    for ax, d, title, s, a in ((axes[0], m, f"(a) {len(m):,} profiles", 2, 0.25),
                               (axes[1], f, f"(b) {len(f)} floats (medians)", 12, 0.9)):
        inside = d.model_gain.between(lo, hi)
        ax.scatter(d.model_gain[inside], d.dm_gain[inside], s=s, color=BLUE, alpha=a, linewidths=0,
                   rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--", label="1:1")
        ax.axhline(med, color=ORANGE, lw=0.9, label=f"array median ({med:.3f})")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("model gain estimate (predicted / raw oxygen)")
        n_out = int((~inside).sum())
        ax.set_title(f"{title}; {n_out} outside the x range", loc="left", fontsize=7, color=INK)
    axes[0].set_ylabel("delayed-mode gain (adjusted / raw)")
    axes[0].set_ylim(0.84, 1.22)
    axes[0].legend(loc="upper left")
    out = os.path.join(FIGS, "gain.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return out


def main():
    import sys
    style()
    os.makedirs(FIGS, exist_ok=True)
    if "--gain" in sys.argv:
        print("wrote", figure_gain())
        return
    print("wrote", figure_map())
    print("wrote", figure_cases())
    print("wrote", figure_gain())


if __name__ == "__main__":
    main()
