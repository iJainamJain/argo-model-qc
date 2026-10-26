"""Every number in the QC manuscript, computed from the result tables - none typed by hand.

Writes manuscript/numbers.tex, a table of values the text reads with \\V{key}: e.g.
\\V{model.offset_PSAL_0.1.float_detection.pct} -> 83 and \\V{...ci} -> 74--91. A key that
does not exist prints as a bold ??key in the PDF, and `python -m paper_qc.check_manuscript`
lists every such key. Also writes the LaTeX tables in manuscript/tables/.

    python -m paper_qc.make_numbers
"""
import json
import os
import zlib

import numpy as np
import pandas as pd

from paper_qc import inject, split

HERE = os.path.dirname(os.path.abspath(__file__))
BS = chr(92)  # backslash, for LaTeX built in f-strings
T = os.path.join(HERE, "tables")
MS = os.path.join(HERE, "manuscript")
V = {}


def put(key, value):
    V[key] = value


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def fmt_int(n):
    return f"{int(n):,}".replace(",", "{,}")


def rates():
    S = pd.read_csv(os.path.join(T, "ci_summary.csv"))
    for _, r in S.iterrows():
        base = f"{r.detector}.{r.scenario}"
        for m in ("level_detection", "profile_detection", "float_detection",
                  "level_false_alarm", "profile_false_alarm", "float_false_alarm"):
            if pd.notna(r.get(m)):
                nd = 1 if "level_false" in m else 0
                put(f"{base}.{m}.pct", pct(r[m], nd))
                put(f"{base}.{m}.ci", f"{pct(r[m + '_lo'], nd)}--{pct(r[m + '_hi'], nd)}")
        if pd.notna(r.get("median_delay")):
            put(f"{base}.delay", f"{r.median_delay:g}")
    return S


def overall_false_alarms():
    """Clean-float false alarms pooled over every scenario and repeat, with float bootstrap."""
    F = pd.read_parquet(os.path.join(T, "records_model_floats.parquet"))
    c = F[~F.corrupted]
    g = c.groupby("PLATFORM").agg(alarm=("first_alarm", "sum"), risk=("at_risk", "sum"),
                                  n=("first_alarm", "size"), anyalarm=("alarm", "sum"))
    rng = np.random.default_rng(5)
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=2000)
    rate = (w @ g.alarm.to_numpy()) / (w @ g.risk.to_numpy()) * 100
    frac = (w @ g.anyalarm.to_numpy()) / (w @ g.n.to_numpy())
    put("fa.per100", f"{g.alarm.sum() / g.risk.sum() * 100:.2f}")
    put("fa.per100.ci", f"{np.percentile(rate, 2.5):.2f}--{np.percentile(rate, 97.5):.2f}")
    put("fa.floats.pct", pct(g.anyalarm.sum() / g.n.sum()))
    put("fa.floats.ci", f"{pct(np.percentile(frac, 2.5))}--{pct(np.percentile(frac, 97.5))}")
    L = pd.read_csv(os.path.join(T, "ci_length.csv"))
    L = L[L.detector == "model"].reset_index(drop=True)
    for i, name in enumerate(("short", "mid", "long")):
        put(f"fa.len.{name}.pct", pct(L.float_false_alarm[i]))
        put(f"fa.len.{name}.range", L.profiles_per_float[i].strip("(]").replace(", ", "--")
            .replace(".0", "").replace("1.999", "2"))


def calibration():
    c = json.load(open(os.path.join(T, "score_thresholds_cv.json")))
    v = json.load(open(os.path.join(T, "score_thresholds.json")))
    put("cal.pool.floats", str(c["val_floats"]))
    put("cal.pool.levels", f"{c['val_levels'] / 1e6:.1f}")
    put("cal.level", f"{c['level_threshold']:.2f}")
    put("cal.profile", f"{c['profile_threshold']:.2f}")
    put("cal.float", f"{c['cusum_threshold']:.2f}")
    put("cal.hazard", f"{c['float_hazard_per_100_profiles']:.3f}")
    put("cal.poolfa.pct", pct(c["float_false_alarm_fraction_pool"], 1))
    put("cal.excluded", ", ".join(c["excluded_floats"]))
    put("cal.within.s", f"{c['within_s']:.2f}")
    put("val.floats", str(v["val_floats"]))
    sel = pd.read_csv(os.path.join(T, "float_method_selection.csv")).set_index("config")
    best_cusum = sel[sel.index.str.startswith("cusum")].mean_float_detection.idxmax()
    for key, cfg in (("window", sel.index[0]), ("cusum", best_cusum)):
        put(f"sel.{key}.name", cfg)
        put(f"sel.{key}.det.pct", pct(sel.loc[cfg, "mean_float_detection"], 1))
        put(f"sel.{key}.delay", f"{sel.loc[cfg, 'median_delay_cycles']:.0f}")
    cus = sel[sel.index.str.startswith("cusum")]
    put("sel.cusum.det.range", f"{pct(cus.mean_float_detection.min(), 1)}--{pct(cus.mean_float_detection.max(), 1)}")
    put("sel.cusum.delay.range", f"{cus.median_delay_cycles.min():.0f}--{cus.median_delay_cycles.max():.0f}")
    put("sel.cusum.thr.min", f"{cus.threshold.min():.0f}")
    oof = pd.read_parquet(os.path.join("data", "qc_oof_scores.parquet"), columns=["PLATFORM", "CYCLE", "PRES", "zS"])
    return oof


def contamination(oof):
    """The inflation of a plain standard deviation by failed floats in the pool."""
    from paper_qc import score
    pm = score.profile_means(oof, np.full(len(oof), np.nan), oof.zS.to_numpy("float64"))
    x = pm.zS.dropna()
    med = np.median(x)
    rob = 1.4826 * np.median(np.abs(x - med))
    put("pool.sd.plain", f"{x.std():.2f}")
    put("pool.sd.robust", f"{rob:.2f}")
    put("pool.sd.ratio", f"{x.std() / rob:.1f}")
    worst = x.abs().groupby(level=0).max()
    for f in ("2902199", "2902205"):
        put(f"pool.{f}.z", f"{worst.get(f, np.nan):.0f}")


def data_numbers():
    s = split.load()
    for k in ("train", "val", "test_clean", "test_known_bad"):
        put(f"split.{k}", str(len(s[k])))
    put("split.total", str(sum(len(s[k]) for k in ("train", "val", "test_clean", "test_known_bad"))))
    prof = pd.read_parquet(os.path.join("data", "qc_profiles.parquet"), columns=["PLATFORM", "CYCLE", "group"])
    put("prof.levels", f"{len(prof) / 1e6:.1f}")
    per = prof.groupby(["group", "PLATFORM", "CYCLE"]).size()
    put("prof.profiles", fmt_int(len(per)))
    put("prof.median.test", f"{per.loc['test_clean'].median():.0f}")
    deep = pd.read_parquet(os.path.join("data", "qc_profiles.parquet"), columns=["PLATFORM", "CYCLE", "PRES"],
                           filters=[("group", "==", "test_clean")])
    pmax = deep.groupby(["PLATFORM", "CYCLE"]).PRES.max()
    put("prof.shallow.pct", pct((pmax < 1000).mean()))
    fmax = pmax.groupby(level=0).max()
    put("prof.shallow.floats", str(int((fmax < 1000).sum())))
    from paper_qc.lib import data as odata
    df = odata.load(verbose=False)
    df["PLATFORM"] = df.PLATFORM.astype(str).str.strip()
    thin = df[df.PLATFORM.isin(s["test_clean"])].groupby(["PLATFORM", "CYCLE"]).size()
    put("prof.median.thin", f"{thin.median():.0f}")
    put("o2.levels", f"{len(df) / 1e6:.2f}")
    put("o2.floats", str(df.PLATFORM.nunique()))
    raw = pd.read_parquet(os.path.join("data", "qc_profiles_raw.parquet"), columns=["PLATFORM", "CYCLE"])
    put("raw.levels", f"{len(raw) / 1e6:.1f}")
    put("raw.profiles", fmt_int(raw.groupby(["PLATFORM", "CYCLE"]).ngroups))


def adjustments():
    a = pd.read_csv(os.path.join(T, "adjustments.csv.gz"))
    put("adj.files", fmt_int(len(a)))
    d = a[a.mode_DOXY == "D"].ratio_DOXY.dropna()
    put("adj.gain.median", f"{d.median():.3f}")
    put("adj.gain.pct", f"{100 * (d.median() - 1):.1f}")
    put("adj.gain.p5", f"{d.quantile(.05):.3f}")
    put("adj.gain.p95", f"{d.quantile(.95):.3f}")
    put("adj.gain.n", fmt_int(len(d)))


def greylist():
    g = pd.read_csv(os.path.join("data", "argo_greylist", "ar_greylist.txt"), dtype=str)
    put("gl.entries", fmt_int(len(g)))
    put("gl.floats", fmt_int(g.PLATFORM_CODE.nunique()))
    o = pd.read_csv(os.path.join(T, "greylist_overlap.csv"), dtype={"float": str})
    bad = o[o.quality_code.astype(str).isin(["3", "4"])]
    put("gl.bad.levels", fmt_int(bad.levels.sum()))
    put("gl.bad.floats", str(bad.float.nunique()))
    put("gl.bad.D", fmt_int(bad.levels_D.sum()))
    put("gl.bad.missed", fmt_int(bad.levels_A.sum() + bad.levels_R.sum()))
    r = pd.read_csv(os.path.join(T, "real_greylist.csv"), dtype={"float": str, "start": str})
    sev = r[r.code.astype(str).isin(["3", "4"])]
    hit = sev[sev.profile_alarms_after > 0.1]
    put("real.severe", str(len(sev)))
    put("real.severe.hit", str(len(hit)))
    put("real.severe.range", f"{pct(hit.profile_alarms_after.min())}--{pct(hit.profile_alarms_after.max())}")
    pre = sev[sev.profiles_before > 0]
    put("real.pre.floats", str(pre.float.nunique()))
    put("real.pre.alarms", pct(pre.profile_alarms_before.max()))
    c2 = r[r.code.astype(str) == "2"]
    put("real.code2", str(len(c2)))
    put("real.code2.max", pct(c2.profile_alarms_after.max()))
    miss = sev[sev.profile_alarms_after <= 0.1].float.tolist()
    put("real.missed", " and ".join(miss))
    return r


def real_summary():
    j = json.load(open(os.path.join(T, "real_summary.json")))
    b = j["float_2902209"]
    put("f09.levels", fmt_int(b["levels_psal_below_30"]))
    put("f09.flag1", fmt_int(b["raw_flags_on_those"].get("1", 0)))
    put("f09.flag4", fmt_int(b["raw_flags_on_those"].get("4", 0)))
    put("f09.model.pct", pct(b["model_flags_on_those"]))
    put("f09.rtqc.pct", pct(b["rtqc_flags_on_those"]))
    put("f09.z", f"{b['median_abs_zS_on_those']:.0f}")
    put("f09.z.else", f"{b['median_abs_zS_elsewhere']:.2f}")
    g = j["gain"]
    m = pd.read_csv(os.path.join(T, "real_gain.csv.gz"), dtype={"PLATFORM": str})
    f = m.groupby("PLATFORM").agg(model=("model_gain", "median"), dm=("dm_gain", "median"))
    put("gain.floats", str(len(f)))
    put("gain.profiles", fmt_int(g["profiles"]))
    # Robust statistics: a mean or a Pearson correlation is decided by the one float fed
    # impossible salinity (2902209, model gain ~15), which the salinity alarm already flags.
    from scipy.stats import spearmanr
    med = f.dm.median()
    em, ec = (f.model - f.dm).abs(), (med - f.dm).abs()
    put("gain.corr", f"{np.corrcoef(f.model, f.dm)[0, 1]:.2f}")
    put("gain.spearman", f"{spearmanr(f.model, f.dm)[0]:.2f}")
    put("gain.spearman.profiles", f"{spearmanr(m.model_gain, m.dm_gain)[0]:.2f}")
    put("gain.medae.model", f"{em.median():.3f}")
    put("gain.medae.const", f"{ec.median():.3f}")
    put("gain.within1.model", pct(np.mean(em < 0.01)))
    put("gain.within1.const", pct(np.mean(ec < 0.01)))
    put("gain.within2.model", pct(np.mean(em < 0.02)))
    put("gain.within2.const", pct(np.mean(ec < 0.02)))
    # is the model's advantage over the array median real? float bootstrap, 4000 draws
    rng = np.random.default_rng(3)
    a, c = em.to_numpy(), ec.to_numpy()
    red, w2 = [], []
    for _ in range(4000):
        i = rng.integers(0, len(a), len(a))
        red.append(1 - np.median(a[i]) / np.median(c[i]))
        w2.append(np.mean(a[i] < 0.02) - np.mean(c[i] < 0.02))
    put("gain.reduction.pct", pct(1 - em.median() / ec.median()))
    put("gain.reduction.ci", "{}--{}".format(*[pct(x) for x in np.percentile(red, [2.5, 97.5])]).replace("-", "$-$", 1)
        if np.percentile(red, 2.5) < 0 else "{}--{}".format(*[pct(x) for x in np.percentile(red, [2.5, 97.5])]))
    put("gain.within2.diff.ci", "{}--{}".format(*[pct(x) for x in np.percentile(w2, [2.5, 97.5])]).replace("-", "$-$", 1)
        if np.percentile(w2, 2.5) < 0 else "{}--{}".format(*[pct(x) for x in np.percentile(w2, [2.5, 97.5])]))
    put("gain.closer.pct", pct(np.mean(a < c)))
    put("gain.median.model", f"{f.model.median():.3f}")
    put("gain.median.dm", f"{med:.3f}")
    out = f[(f.model < 0.8) | (f.model > 1.4)]
    inl = f.drop(out.index)
    put("gain.outliers", " and ".join(out.index))
    put("gain.outliers.n", str(len(out)))
    put("gain.f09.model", f"{f.loc['2902209', 'model']:.0f}" if "2902209" in f.index else "--")
    put("gain.corr.inliers", f"{np.corrcoef(inl.model, inl.dm)[0, 1]:.2f}")


def rtqc_clean():
    fa = pd.read_csv(os.path.join(T, "rtqc_false_alarms.csv"))
    for v in ("TEMP", "PSAL", "DOXY"):
        r = fa[(fa.variable == v) & (fa.test == "any")].iloc[0]
        put(f"rt.clean.{v}.floats", str(int(r.floats)))
    g = fa[(fa.variable == "DOXY") & (fa.test == "11 gradient")].iloc[0]
    put("rt.clean.gradient.floats", str(int(g.floats)))


def distance():
    D = pd.read_csv(os.path.join(T, "ci_distance.csv"))
    D = D.set_index("band")
    for band in D.index:
        b = band.replace("-", "to")
        put(f"dist.{b}.floats", str(int(D.loc[band, "floats"])))
        put(f"dist.{b}.lfa", pct(D.loc[band, "level_false_alarm"], 1))
        for s in ("offset_PSAL_0.1", "gain_DOXY_0.1", "drift_PSAL_0.05"):
            put(f"dist.{b}.{s}", pct(D.loc[band, s]))
            put(f"dist.{b}.{s}.ci", f"{pct(D.loc[band, s + '_lo'])}--{pct(D.loc[band, s + '_hi'])}")


def attribution():
    a = pd.read_csv(os.path.join(T, "attribution.csv"))
    ctd, opt = a[a.fault == "CTD"], a[a.fault == "optode"]
    w = ctd.profiles_flagged
    put("attr.ctd.oxyonly.pct", pct((ctd.oxygen_only * w).sum() / w.sum(), 1))
    put("attr.ctd.sal.range", f"{pct((ctd.salinity_only + ctd.both).min())}--{pct((ctd.salinity_only + ctd.both).max())}")
    w = opt.profiles_flagged
    put("attr.opt.sal.pct", pct(((opt.salinity_only + opt.both) * w).sum() / w.sum(), 1))
    pre = ctd.oxygen_only_preexisting
    put("attr.ctd.pre.range", f"{pct(pre.min())}--{pct(pre.max())}")
    n_oxo = ctd.oxygen_only * ctd.profiles_flagged
    put("attr.ctd.pre.pct", pct((pre * n_oxo).sum() / n_oxo.sum()))
    put("attr.background.pct", pct(a.background_oxygen_only.iloc[0], 1))
    for _, r in a.iterrows():
        put(f"attr.{r.scenario}.oxyonly.pct", pct(r.oxygen_only, 1))
        put(f"attr.{r.scenario}.sal.pct", pct(r.salinity_only + r.both, 1))


def table_attribution():
    a = pd.read_csv(os.path.join(T, "attribution.csv"))
    lines = [f"{label(r.scenario)} & {r.fault} & {fmt_int(r.profiles_flagged)} & "
             f"{pct(r.salinity_only, 1)} & {pct(r.oxygen_only, 1)} & {pct(r.both, 1)} {BS}{BS}"
             for _, r in a.iterrows()]
    return ("\\begin{tabular}{llrrrr}\n\\toprule\n"
            "Planted error & Sensor & Alarmed profiles & salinity only & oxygen only & both \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_selection():
    s = pd.read_csv(os.path.join(T, "float_method_selection.csv"))
    def name(c):
        m, v = c.split()
        return f"moving window, {v} profiles" if m == "window" else f"CUSUM, $k={v}$"
    lines = [f"{name(r.config)} & {pct(r.mean_float_detection, 1)} & {r.median_delay_cycles:.0f} & "
             f"{r.threshold:.1f} {BS}{BS}" for _, r in s.iterrows()]
    return ("\\begin{tabular}{lrrr}\n\\toprule\n"
            "Float statistic & Mean detection (\\%) & Median delay (cycles) & Threshold \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def models():
    """Validation skill of the two final ensembles, recomputed (the training log only prints it)."""
    from paper_qc.lib import data as odata
    from paper_qc.model import Ensemble
    df = odata.load(verbose=False)
    df["PLATFORM"] = df.PLATFORM.astype(str).str.strip()
    va = df[df.PLATFORM.isin(split.load()["val"])]
    for t, var in (("doxy", "DOXY"), ("psal", "PSAL")):
        m = json.load(open(os.path.join("runs_qc", t, "meta.json")))
        put(f"model.{t}.epochs", "--".join(str(x) for x in (min(m["epochs"]), max(m["epochs"]))))
        put(f"model.{t}.ntrain", fmt_int(m["n_train"]))
        mu, sd = Ensemble(os.path.join("runs_qc", t)).predict(va)
        err = va[var].to_numpy("float64") - mu
        put(f"model.{t}.mae", f"{np.mean(np.abs(err)):.{2 if t == 'doxy' else 3}f}")
        put(f"model.{t}.cov90", pct(np.mean(np.abs(err / sd) < 1.645), 1))


# --------------------------------------------------------------------------- tables
def table_scenarios():
    rows = {}
    for s in inject.SCENARIOS:
        rows.setdefault((s.kind, s.var), []).append(s.magnitude)
    unit = {"PSAL": "psu", "TEMP": "$^\\circ$C", "DOXY": "\\umol{}", "PRES": "dbar"}
    what = {"offset": "added from onset", "drift": "grows linearly from onset",
            "gain": "oxygen multiplied by $1-m$ from onset", "spike": "one level per hit profile",
            "stuck": "run of up to 10 levels repeats the level above", "pressure": "added from onset"}
    lines = []
    for (k, v), mags in rows.items():
        u = unit[v]
        if k == "gain":
            u = "fraction"
        if k == "drift":
            u = "psu\\,yr$^{-1}$" if v == "PSAL" else "fraction\\,yr$^{-1}$"
        if k == "stuck":
            ms = "10 levels"
        else:
            ms = ", ".join(f"{m:g}" for m in mags) + f" {u}"
        lines.append(f"{k} & {v} & {ms} & {what[k]} \\\\")
    body = "\n".join(lines)
    return ("\\begin{tabular}{llp{4.6cm}p{5.2cm}}\n\\toprule\nError & Variable & Magnitudes $m$ & "
            "Form \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


HEADLINE = ["offset_PSAL_0.05", "offset_PSAL_0.1", "offset_PSAL_0.2", "drift_PSAL_0.02",
            "drift_PSAL_0.05", "drift_PSAL_0.1", "offset_TEMP_0.5", "offset_DOXY_10",
            "offset_DOXY_20", "gain_DOXY_0.05", "gain_DOXY_0.1", "gain_DOXY_0.2",
            "drift_DOXY_0.05", "spike_PSAL_0.1", "spike_TEMP_0.5", "spike_DOXY_25",
            "stuck_DOXY_10", "pressure_PRES_20"]


def label(s):
    k, v, m = s.split("_")
    names = {"PSAL": "salinity", "TEMP": "temperature", "DOXY": "oxygen", "PRES": "pressure"}
    base = {"PSAL": "psu", "TEMP": r"$^\circ$C", "DOXY": r"\umol{}", "PRES": "dbar"}[v]
    if k == "stuck":
        u = "levels"
    elif k == "gain":
        u = ""
    elif k == "drift":
        u = r"psu\,yr$^{-1}$" if v == "PSAL" else "yr$^{-1}$"
    else:
        u = base
    mm = f"{float(m) * 100:g}\\%" if k == "gain" or (k == "drift" and v == "DOXY") else f"{float(m):g}"
    kind = {"gain": "gain loss", "pressure": "offset"}.get(k, k)
    return f"{names[v]} {kind} {mm} {u}".strip()


def table_headline(S, scenarios=HEADLINE, name="headline"):
    def cell(det, s, m):
        r = S[(S.detector == det) & (S.scenario == s)]
        if not len(r) or pd.isna(r.iloc[0].get(m)):
            return "--"
        r = r.iloc[0]
        return f"{pct(r[m])} {{\\scriptsize({pct(r[m + '_lo'])}--{pct(r[m + '_hi'])})}}"
    lines = []
    for s in scenarios:
        d = S[(S.detector == "model") & (S.scenario == s)].iloc[0].median_delay
        if s.split("_")[0] in inject.TRANSIENT:
            d = np.nan  # the float statistic ignores isolated errors: its "delay" is background alarms
        lines.append(" & ".join([label(s), cell("rtqc", s, "level_detection"),
                                 cell("rtqc", s, "float_detection"),
                                 cell("model", s, "level_detection"),
                                 cell("model", s, "profile_detection"),
                                 cell("model", s, "float_detection"),
                                 f"{d:g}" if pd.notna(d) else "--"]) + " \\\\")
    return ("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Real-time tests} & "
            "\\multicolumn{4}{c}{Model} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
            "Planted error & levels & floats & levels & profiles & floats & delay \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_real(r):
    sev = r[r.code.astype(str).isin(["3", "4"])].copy()
    g = pd.read_csv(os.path.join("data", "argo_greylist", "ar_greylist.txt"), dtype=str)
    lines = []
    for _, e in sev.iterrows():
        comment = str(e.comment).replace("&", "and").replace("_", " ").replace("%", "")
        b = "--" if e.profiles_before == 0 else f"{pct(e.profile_alarms_before)} ({int(e.profiles_before)})"
        lines.append(f"{e.float} & {e.code} & {e.start[:4]}-{e.start[4:6]}-{e.start[6:]} & "
                     f"{comment} & {b} & {pct(e.profile_alarms_after)} ({int(e.profiles_after)}) \\\\")
    return ("\\begin{tabular}{lclp{4.5cm}rr}\n\\toprule\nFloat & Code & Start & Reason (exclusion list) & "
            "before & after \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
    S = rates()
    overall_false_alarms()
    oof = calibration()
    contamination(oof)
    data_numbers()
    adjustments()
    r = greylist()
    real_summary()
    rtqc_clean()
    distance()
    models()
    attribution()
    put("inject.n", str(len(inject.SCENARIOS)))
    put("inject.kinds", str(len({(s.kind, s.var) for s in inject.SCENARIOS})))
    with open(os.path.join(MS, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by paper_qc/make_numbers.py - do not edit\n")
        for k in sorted(V):
            f.write(f"\\expandafter\\def\\csname V@{k}\\endcsname{{{V[k]}}}\n")
    with open(os.path.join(MS, "tables", "scenarios.tex"), "w", encoding="utf-8") as f:
        f.write(table_scenarios())
    with open(os.path.join(MS, "tables", "headline.tex"), "w", encoding="utf-8") as f:
        f.write(table_headline(S))
    with open(os.path.join(MS, "tables", "all_scenarios.tex"), "w", encoding="utf-8") as f:
        f.write(table_headline(S, [s.name for s in inject.SCENARIOS]))
    with open(os.path.join(MS, "tables", "real.tex"), "w", encoding="utf-8") as f:
        f.write(table_real(r))
    with open(os.path.join(MS, "tables", "attribution.tex"), "w", encoding="utf-8") as f:
        f.write(table_attribution())
    with open(os.path.join(MS, "tables", "selection.tex"), "w", encoding="utf-8") as f:
        f.write(table_selection())
    print(f"wrote {len(V)} values to manuscript/numbers.tex and 4 tables")


if __name__ == "__main__":
    main()
