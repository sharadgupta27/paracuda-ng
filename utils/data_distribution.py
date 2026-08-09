"""
data_distribution.py
====================
Distribution inspection for target properties, shared by every PARACUDA-NG
front end (Tkinter desktop, Qt/QGIS panel and the Data Converter).

The point is to let a user judge a dataset **before** committing to conversion
or model training.  A property that is strongly skewed, near-constant, riddled
with outliers or effectively bimodal will not model well no matter which
algorithm is chosen, and it is far cheaper to discover that here than after a
long batch run.

This module is deliberately toolkit-agnostic: it computes statistics with
numpy/pandas and draws onto a bare matplotlib ``Figure``, so the same numbers
and the same plot appear in all three versions.

@author: Sharad Kumar Gupta
"""
import numpy as np
import pandas as pd

from utils.lazy_imports import LazyModule, LazyCallable

Figure = LazyCallable('matplotlib.figure', 'Figure')
stats = LazyModule('scipy.stats')

__all__ = [
    "summarize_distribution", "describe_column", "interpret_distribution",
    "create_distribution_figure", "create_property_figure",
    "distribution_report_text", "property_report_text",
    "overall_severity", "suggest_numeric_columns",
]

# Skewness bands.  |skew| < 0.5 is generally treated as near-symmetric, 0.5-1.0
# as moderate and > 1.0 as strong (Bulmer, 1979).
_SKEW_MODERATE = 0.5
_SKEW_STRONG = 1.0
# Share of IQR-rule outliers above which a column is worth flagging.
_OUTLIER_WARN_FRAC = 0.05
# Coefficient of variation below which a property carries too little spread to
# be worth modelling (as a fraction, i.e. 2%).
_LOW_CV = 0.02


def suggest_numeric_columns(df, wavelength_cols=None, max_cols=None):
    """Return the numeric, non-wavelength columns worth inspecting.

    Wavelength columns are excluded because a spectrum is not a property
    distribution - inspecting 2000 band columns individually is meaningless.
    """
    if df is None or getattr(df, "empty", True):
        return []
    wl = {str(c) for c in (wavelength_cols or [])}
    cols = []
    for c in df.columns:
        if str(c) in wl:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # Keep a column only if it is mostly numeric and actually varies.
        if s.notna().sum() >= 3 and s.nunique(dropna=True) > 1:
            cols.append(c)
    if max_cols is not None:
        cols = cols[:max_cols]
    return cols


def describe_column(series, name=None):
    """Full distribution summary for one column.

    Returns a dict of plain Python floats/ints (JSON- and Excel-friendly).
    ``skewness``/``kurtosis`` fall back to ``nan`` when SciPy is unavailable or
    the column is degenerate, rather than raising.
    """
    name = name if name is not None else getattr(series, "name", "value")
    raw = pd.to_numeric(pd.Series(series), errors="coerce")
    n_total = int(len(raw))
    values = raw.dropna().to_numpy(dtype=float)
    n = int(values.size)

    out = {
        "column": str(name),
        "n_total": n_total,
        "n_valid": n,
        "n_missing": n_total - n,
        "missing_pct": (100.0 * (n_total - n) / n_total) if n_total else float("nan"),
    }
    if n == 0:
        out.update({k: float("nan") for k in
                    ("mean", "median", "std", "min", "max", "range", "q1", "q3",
                     "iqr", "cv", "skewness", "kurtosis", "outlier_pct")})
        out["n_outliers"] = 0
        out["n_unique"] = 0
        return out

    q1, med, q3 = (float(np.percentile(values, p)) for p in (25, 50, 75))
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.count_nonzero((values < lo) | (values > hi))) if iqr > 0 else 0

    try:
        skew = float(stats.skew(values, bias=False)) if n > 2 else float("nan")
        kurt = float(stats.kurtosis(values, bias=False)) if n > 3 else float("nan")
    except Exception:  # noqa: BLE001 - SciPy missing or degenerate input
        skew = kurt = float("nan")

    out.update({
        "mean": mean,
        "median": med,
        "std": sd,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        # Coefficient of variation is undefined around a zero mean.
        "cv": (sd / abs(mean)) if abs(mean) > 1e-12 else float("nan"),
        "skewness": skew,
        "kurtosis": kurt,
        "n_outliers": n_out,
        "outlier_pct": 100.0 * n_out / n,
        "n_unique": int(np.unique(values).size),
    })
    return out


def interpret_distribution(summary):
    """Turn one :func:`describe_column` result into plain-language findings.

    Returns ``(severity, [messages])`` where severity is ``"ok"``, ``"warn"`` or
    ``"problem"`` - the caller colours its display accordingly.
    """
    msgs, severity = [], "ok"

    def bump(level):
        nonlocal severity
        order = {"ok": 0, "warn": 1, "problem": 2}
        if order[level] > order[severity]:
            severity = level

    n = summary.get("n_valid", 0)
    if n == 0:
        return "problem", ["No numeric values at all - this column cannot be modelled."]

    if n < 20:
        msgs.append(f"Only {n} valid samples; results will be unstable "
                    f"(30+ is a practical minimum, 50+ is comfortable).")
        bump("problem" if n < 10 else "warn")

    miss = summary.get("missing_pct", 0.0)
    if np.isfinite(miss) and miss > 0:
        msgs.append(f"{miss:.1f}% of values are missing or non-numeric.")
        bump("problem" if miss > 20 else "warn")

    skew = summary.get("skewness", float("nan"))
    if np.isfinite(skew) and abs(skew) >= _SKEW_MODERATE:
        direction = "right (a long tail of high values)" if skew > 0 else \
                    "left (a long tail of low values)"
        if abs(skew) >= _SKEW_STRONG:
            msgs.append(f"Strongly skewed to the {direction}: skew = {skew:+.2f}. "
                        f"A log or square-root transform of this property usually "
                        f"helps, and RMSE will be dominated by the tail.")
            bump("problem")
        else:
            msgs.append(f"Moderately skewed to the {direction}: skew = {skew:+.2f}.")
            bump("warn")

    kurt = summary.get("kurtosis", float("nan"))
    if np.isfinite(kurt) and kurt > 3.0:
        msgs.append(f"Heavy tails (excess kurtosis = {kurt:+.2f}); a few extreme "
                    f"samples will dominate the fit.")
        bump("warn")

    out_pct = summary.get("outlier_pct", 0.0)
    if np.isfinite(out_pct) and out_pct > 100 * _OUTLIER_WARN_FRAC:
        msgs.append(f"{summary['n_outliers']} samples ({out_pct:.1f}%) lie outside "
                    f"the 1.5xIQR whiskers - check them before training, or enable "
                    f"Target Variable Outlier Removal.")
        bump("warn")

    cv = summary.get("cv", float("nan"))
    if np.isfinite(cv) and cv < _LOW_CV:
        msgs.append(f"Very little spread (CV = {100 * cv:.2f}%). With a nearly "
                    f"constant target, R² is meaningless and RPD will be poor.")
        bump("problem")

    if summary.get("n_unique", 0) <= 5 and n > 10:
        msgs.append(f"Only {summary['n_unique']} distinct values - this looks "
                    f"categorical rather than continuous; regression may be the "
                    f"wrong tool.")
        bump("warn")

    if not msgs:
        msgs.append("Distribution looks well behaved: roughly symmetric, "
                    "adequate spread, no dominant outliers.")
    return severity, msgs


def summarize_distribution(df, columns=None, wavelength_cols=None):
    """Summarise every requested column.

    Returns a list of dicts, each being :func:`describe_column` plus
    ``severity`` and ``findings`` from :func:`interpret_distribution`.
    """
    if columns is None:
        columns = suggest_numeric_columns(df, wavelength_cols)
    results = []
    for c in columns:
        if c not in df.columns:
            continue
        s = describe_column(df[c], name=c)
        s["severity"], s["findings"] = interpret_distribution(s)
        results.append(s)
    return results


def distribution_report_text(summaries):
    """Render :func:`summarize_distribution` output as plain text."""
    icons = {"ok": "OK", "warn": "!", "problem": "!!"}
    lines = []
    for s in summaries:
        lines.append("=" * 72)
        lines.append(f"[{icons.get(s['severity'], '?')}] {s['column']}")
        lines.append("=" * 72)
        lines.append(
            f"  n = {s['n_valid']} valid / {s['n_total']} total"
            f"   missing = {s['n_missing']} ({s['missing_pct']:.1f}%)")
        lines.append(
            f"  mean = {s['mean']:.4g}   median = {s['median']:.4g}   "
            f"std = {s['std']:.4g}")
        lines.append(
            f"  min = {s['min']:.4g}   Q1 = {s['q1']:.4g}   "
            f"Q3 = {s['q3']:.4g}   max = {s['max']:.4g}")
        cv_txt = f"{100 * s['cv']:.2f}%" if np.isfinite(s.get('cv', np.nan)) else "n/a"
        lines.append(
            f"  skewness = {s['skewness']:+.3f}   kurtosis = {s['kurtosis']:+.3f}   "
            f"CV = {cv_txt}")
        lines.append(
            f"  IQR outliers = {s['n_outliers']} ({s['outlier_pct']:.1f}%)")
        lines.append("")
        for m in s["findings"]:
            lines.append(f"  - {m}")
        lines.append("")
    if not lines:
        lines.append("No numeric property columns found to summarise.")
    return "\n".join(lines)


def property_report_text(summary):
    """Render one :func:`summarize_distribution` entry as plain text."""
    return distribution_report_text([summary])


def overall_severity(summaries):
    """Worst severity across ``summaries`` - 'ok', 'warn' or 'problem'."""
    worst = "ok"
    for s in summaries:
        if s.get("severity") == "problem":
            return "problem"
        if s.get("severity") == "warn":
            worst = "warn"
    return worst


def _severity_colour(severity, fallback="#222222"):
    return {"ok": "#1a9850", "warn": "#f0a202",
            "problem": "#d73027"}.get(severity, fallback)


def create_property_figure(df, column, wavelength_cols=None, palette=None):
    """Build a four-panel diagnostic figure for a **single** property.

    Showing every property at once produced a wall of thumbnails in which no
    individual distribution could actually be read.  With one property selected
    there is room for the diagnostics that matter:

    ``histogram``  values against a same-mean/same-sd normal curve, with the
                   mean and median marked - their separation *is* the skew;
    ``boxplot``    the 1.5xIQR whiskers and every outlier, on the histogram's
                   own x axis so the two read together;
    ``Q-Q plot``   sample quantiles against normal quantiles - a straight line
                   means normal, a bend names the departure (skew vs tails);
    ``ECDF``       the cumulative share of samples with the quartiles marked,
                   which is where "how much of my data sits below X" is read.

    Returns ``(figure, summary)`` where ``summary`` is the
    :func:`describe_column` dict plus ``severity``/``findings``.
    """
    summary = describe_column(df[column], name=column)
    summary["severity"], summary["findings"] = interpret_distribution(summary)

    fg = (palette or {}).get("TXT") or "#222222"
    bg = (palette or {}).get("WIN") or "#FFFFFF"
    accent = (palette or {}).get("ACC") or "#2c7fb8"

    fig = Figure(figsize=(10.0, 6.4), constrained_layout=True)
    fig.patch.set_facecolor(bg)

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, f"'{column}' has no numeric values to plot",
                ha="center", va="center", fontsize=12, color=fg)
        ax.set_axis_off()
        return fig, summary

    # Histogram + boxplot on the left (boxplot is the short strip underneath),
    # Q-Q and ECDF on the right.
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 2, 2])
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[1, 0], sharex=ax_hist)
    ax_qq = fig.add_subplot(gs[0:2, 1])
    ax_ecdf = fig.add_subplot(gs[2:4, 0])
    ax_txt = fig.add_subplot(gs[2:4, 1])

    bins = int(np.clip(np.sqrt(values.size) * 1.5, 8, 60))
    ax_hist.hist(values, bins=bins, color=accent, alpha=0.75,
                 edgecolor="white", linewidth=0.5)
    ax_hist.axvline(summary["mean"], color="#d73027", lw=1.8,
                    label=f"mean {summary['mean']:.4g}")
    ax_hist.axvline(summary["median"], color="#1a9850", lw=1.8, ls="--",
                    label=f"median {summary['median']:.4g}")
    if summary["std"] > 0 and values.size > 5:
        xs = np.linspace(values.min(), values.max(), 300)
        pdf = np.exp(-0.5 * ((xs - summary["mean"]) / summary["std"]) ** 2)
        pdf /= pdf.max() or 1.0
        counts, _ = np.histogram(values, bins=bins)
        ax_hist.plot(xs, pdf * counts.max(), color=fg, lw=1.3, alpha=0.55,
                     label="normal reference")
    ax_hist.legend(fontsize=8, framealpha=0.85)
    ax_hist.set_ylabel("Count", fontsize=9, color=fg)
    ax_hist.tick_params(labelsize=8, colors=fg, labelbottom=False)
    ax_hist.set_title(f"Distribution of {column}", fontsize=11,
                      fontweight="bold", color=_severity_colour(
                          summary["severity"], fg))

    ax_box.boxplot(values, vert=False, widths=0.6,
                   flierprops=dict(marker="o", markersize=3.5,
                                   markerfacecolor="#d73027",
                                   markeredgecolor="none", alpha=0.6))
    ax_box.set_yticks([])
    ax_box.tick_params(labelsize=8, colors=fg)
    ax_box.set_xlabel(str(column), fontsize=9, color=fg)

    # Q-Q plot against a normal with the sample's own mean and sd.
    n = values.size
    ordered = np.sort(values)
    probs = (np.arange(1, n + 1) - 0.5) / n
    try:
        theo = stats.norm.ppf(probs, loc=summary["mean"], scale=summary["std"] or 1.0)
    except Exception:  # noqa: BLE001 - SciPy unavailable
        theo = None
    if theo is not None:
        ax_qq.scatter(theo, ordered, s=10, color=accent, alpha=0.7,
                      edgecolors="none")
        lo = float(min(theo.min(), ordered.min()))
        hi = float(max(theo.max(), ordered.max()))
        ax_qq.plot([lo, hi], [lo, hi], color="#d73027", lw=1.2, ls="--",
                   label="perfectly normal")
        ax_qq.legend(fontsize=8, framealpha=0.85)
        ax_qq.set_xlabel("Theoretical (normal) quantile", fontsize=9, color=fg)
    else:
        ax_qq.text(0.5, 0.5, "Q-Q plot needs SciPy", ha="center", va="center",
                   fontsize=9, color=fg)
    ax_qq.set_ylabel("Sample quantile", fontsize=9, color=fg)
    ax_qq.set_title("Normal Q-Q - bends mean non-normal", fontsize=10,
                    fontweight="bold", color=fg)
    ax_qq.tick_params(labelsize=8, colors=fg)
    ax_qq.grid(True, alpha=0.25)

    ax_ecdf.step(ordered, np.arange(1, n + 1) / n * 100.0, where="post",
                 color=accent, lw=1.6)
    for q, lbl, col in ((summary["q1"], "Q1", "#888888"),
                        (summary["median"], "median", "#1a9850"),
                        (summary["q3"], "Q3", "#888888")):
        ax_ecdf.axvline(q, color=col, lw=1.0, ls=":")
        # Labels ride along the top: near a skewed property's quartiles they sit
        # close together, and stacking them at the bottom made them collide.
        ax_ecdf.annotate(lbl, xy=(q, 98), fontsize=7, color=col,
                         rotation=90, ha="right", va="top")
    ax_ecdf.set_ylim(0, 100)
    ax_ecdf.set_xlabel(str(column), fontsize=9, color=fg)
    ax_ecdf.set_ylabel("% of samples below", fontsize=9, color=fg)
    ax_ecdf.set_title("Cumulative distribution", fontsize=10,
                      fontweight="bold", color=fg)
    ax_ecdf.tick_params(labelsize=8, colors=fg)
    ax_ecdf.grid(True, alpha=0.25)

    # The numbers, next to the plots that produced them.
    ax_txt.set_axis_off()
    cv_txt = (f"{100 * summary['cv']:.2f}%"
              if np.isfinite(summary.get("cv", np.nan)) else "n/a")
    lines = [
        f"n valid          {summary['n_valid']} of {summary['n_total']}",
        f"missing          {summary['n_missing']} ({summary['missing_pct']:.1f}%)",
        f"mean / median    {summary['mean']:.4g}  /  {summary['median']:.4g}",
        f"std / CV         {summary['std']:.4g}  /  {cv_txt}",
        f"min / max        {summary['min']:.4g}  /  {summary['max']:.4g}",
        f"Q1 / Q3 / IQR    {summary['q1']:.4g} / {summary['q3']:.4g} / {summary['iqr']:.4g}",
        f"skewness         {summary['skewness']:+.3f}",
        f"excess kurtosis  {summary['kurtosis']:+.3f}",
        f"IQR outliers     {summary['n_outliers']} ({summary['outlier_pct']:.1f}%)",
    ]
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                fontsize=8.5, family="monospace", color=fg,
                transform=ax_txt.transAxes)

    for a in (ax_hist, ax_box, ax_qq, ax_ecdf, ax_txt):
        a.set_facecolor(bg)
        for spine in a.spines.values():
            spine.set_color(fg)
            spine.set_alpha(0.35)

    return fig, summary


def create_distribution_figure(df, columns=None, wavelength_cols=None,
                               palette=None, max_cols=6):
    """Build a matplotlib Figure: histogram + boxplot per column.

    Each column gets a histogram (with mean/median markers and a normal curve
    for reference) above a horizontal boxplot sharing the same x axis, so skew
    and outliers are visible at a glance.

    Returns ``(figure, summaries)``.
    """
    if columns is None:
        columns = suggest_numeric_columns(df, wavelength_cols)
    columns = list(columns)[:max_cols]
    summaries = summarize_distribution(df, columns, wavelength_cols)

    n = max(1, len(summaries))
    ncols = 1 if n == 1 else (2 if n <= 4 else 3)
    nrows = int(np.ceil(n / ncols))

    fg = (palette or {}).get("TXT") or "#222222"
    bg = (palette or {}).get("WIN") or "#FFFFFF"
    accent = (palette or {}).get("ACC") or "#2c7fb8"

    fig = Figure(figsize=(5.2 * ncols, 4.0 * nrows), constrained_layout=True)
    fig.patch.set_facecolor(bg)

    if not summaries:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No numeric property columns to plot",
                ha="center", va="center", fontsize=12, color=fg)
        ax.set_axis_off()
        return fig, summaries

    # One gridspec for the whole figure.  Building it inside the loop created a
    # fresh (and differently sized) grid per column, which is what left the
    # panels overlapping each other's axis labels.
    gs = fig.add_gridspec(nrows * 4, ncols)

    for i, s in enumerate(summaries):
        values = pd.to_numeric(df[s["column"]], errors="coerce").dropna().to_numpy(float)
        # Histogram on top, boxplot underneath, sharing the value axis.
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r * 4:r * 4 + 3, c])
        ax_box = fig.add_subplot(gs[r * 4 + 3, c], sharex=ax)

        if values.size:
            bins = int(np.clip(np.sqrt(values.size) * 1.5, 8, 40))
            ax.hist(values, bins=bins, color=accent, alpha=0.75,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(s["mean"], color="#d73027", lw=1.6, label=f"mean {s['mean']:.3g}")
            ax.axvline(s["median"], color="#1a9850", lw=1.6, ls="--",
                       label=f"median {s['median']:.3g}")
            # A normal curve with the same mean/sd makes skew obvious.
            if s["std"] > 0 and values.size > 5:
                xs = np.linspace(values.min(), values.max(), 200)
                pdf = np.exp(-0.5 * ((xs - s["mean"]) / s["std"]) ** 2)
                pdf /= pdf.max() or 1.0
                counts, _ = np.histogram(values, bins=bins)
                ax.plot(xs, pdf * counts.max(), color=fg, lw=1.1, alpha=0.55,
                        label="normal reference")
            ax.legend(fontsize=7, framealpha=0.85)
            ax_box.boxplot(values, vert=False, widths=0.6,
                           flierprops=dict(marker="o", markersize=3,
                                           markerfacecolor="#d73027",
                                           markeredgecolor="none", alpha=0.6))

        colour = _severity_colour(s["severity"], fg)
        ax.set_title(f"{s['column']}   (skew {s['skewness']:+.2f}, "
                     f"n={s['n_valid']})",
                     fontsize=10, fontweight="bold", color=colour)
        ax.set_ylabel("Count", fontsize=9, color=fg)
        ax.tick_params(labelsize=8, colors=fg, labelbottom=False)
        ax_box.set_yticks([])
        ax_box.tick_params(labelsize=8, colors=fg)
        ax_box.set_xlabel(str(s["column"]), fontsize=9, color=fg)
        for a in (ax, ax_box):
            a.set_facecolor(bg)
            for spine in a.spines.values():
                spine.set_color(fg)
                spine.set_alpha(0.35)

    return fig, summaries
