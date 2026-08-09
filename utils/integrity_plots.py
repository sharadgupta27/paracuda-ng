"""
integrity_plots.py
==================
Figures for the Check Spectral Integrity tools, shared by every PARACUDA-NG
front end (Tkinter desktop, Qt/QGIS panel and the PyPI package).

Like ``utils/data_distribution``, this module is deliberately toolkit-agnostic:
it draws onto a bare matplotlib ``Figure``, so all three versions show the same
picture and only the widget that hosts it differs.

@author: Sharad Kumar Gupta
"""
import numpy as np

from utils.lazy_imports import LazyCallable

Figure = LazyCallable('matplotlib.figure', 'Figure')

__all__ = ["create_mixing_figure", "label_reassignment_rows"]


def label_reassignment_rows(y_true, y_assigned, changed_idx, limit=24):
    """Compact ``"#idx: true -> assigned"`` strings for the relabelled samples.

    Returns ``(rows, n_hidden)``.  The caller lays the rows out in however many
    columns its toolkit makes convenient - a single tall column wasted most of
    the dialog and pushed the plot off screen.
    """
    idx = np.asarray(changed_idx, dtype=int)
    shown = idx[:limit]
    rows = [f"#{int(i):>4}: {float(y_true[i]):.4g} → {float(y_assigned[i]):.4g}"
            for i in shown]
    return rows, int(max(0, idx.size - shown.size))


def create_mixing_figure(wavelengths, X, y_true, y_assigned, changed_idx,
                         prop="property", unit="nm", mix_fraction=None,
                         max_spectra=8, palette=None):
    """Visualise what a spectral-mixing integrity check actually did.

    The mixing check does *not* alter a single spectrum - it swaps labels
    between samples.  Drawing the affected spectra twice, once per label set,
    therefore produced two identical curve plots whose only difference was the
    legend, which told the user nothing.

    This figure shows the two things that did change instead:

    ``left``   the affected spectra drawn **once**, coloured by their TRUE
               property value on a continuous scale with a colour bar.  This is
               what the model sees, and it makes the spectral ordering of the
               property (or its absence) visible.
    ``right``  a slope chart - one line per relabelled sample connecting its
               TRUE value to the value mixing ASSIGNED it, coloured by the same
               scale.  Steep, crossing lines are the scrambling itself; the
               annotation reports the mean absolute displacement, i.e. how hard
               the check actually pushed.

    Returns the figure.
    """
    fg = (palette or {}).get("TXT") or "#222222"
    bg = (palette or {}).get("WIN") or "#FFFFFF"

    idx = np.asarray(changed_idx, dtype=int)
    y_true = np.asarray(y_true, dtype=float)
    y_assigned = np.asarray(y_assigned, dtype=float)

    fig = Figure(figsize=(9.6, 4.0), constrained_layout=True)
    fig.patch.set_facecolor(bg)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_slope = fig.add_subplot(gs[0, 1])
    for a in (ax_spec, ax_slope):
        a.set_facecolor(bg)

    if idx.size == 0:
        for a, msg in ((ax_spec, "No samples were relabelled."),
                       (ax_slope, "")):
            a.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
                   color=fg)
            a.set_axis_off()
        return fig

    wl = np.asarray([float(w) for w in wavelengths], dtype=float)
    order = np.argsort(wl)
    wl_sorted = wl[order]

    # Colour every affected sample by its TRUE value, on one shared scale used
    # by both panels so a curve on the left and its line on the right match.
    affected_true = y_true[idx]
    vmin, vmax = float(affected_true.min()), float(affected_true.max())
    if vmax <= vmin:
        vmax = vmin + 1e-9
    import matplotlib
    cmap = matplotlib.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    sel = idx[:max_spectra]
    for i in sel:
        ax_spec.plot(wl_sorted, np.asarray(X[i], dtype=float)[order],
                     color=cmap(norm(y_true[i])), lw=1.2, alpha=0.9)
    ax_spec.set_xlabel(f"Wavelength ({unit})", fontsize=9, color=fg)
    ax_spec.set_ylabel("Reflectance", fontsize=9, color=fg)
    ax_spec.set_title(f"Affected spectra, coloured by TRUE {prop}",
                      fontsize=10, fontweight="bold", color=fg)
    ax_spec.grid(True, alpha=0.25)
    ax_spec.tick_params(labelsize=8, colors=fg)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_spec,
        fraction=0.046, pad=0.02)
    cbar.set_label(f"TRUE {prop}", fontsize=8, color=fg)
    cbar.ax.tick_params(labelsize=7, colors=fg)
    if sel.size < idx.size:
        ax_spec.text(0.99, 0.02, f"showing {sel.size} of {idx.size}",
                     transform=ax_spec.transAxes, ha="right", va="bottom",
                     fontsize=7.5, color=fg, alpha=0.8)

    # Slope chart: TRUE on the left axis, ASSIGNED on the right.
    for i in idx:
        ax_slope.plot([0, 1], [y_true[i], y_assigned[i]],
                      color=cmap(norm(y_true[i])), lw=1.0, alpha=0.65,
                      marker="o", markersize=2.5)
    ax_slope.set_xlim(-0.12, 1.12)
    ax_slope.set_xticks([0, 1])
    ax_slope.set_xticklabels(["TRUE", "ASSIGNED"], fontsize=9,
                             fontweight="bold")
    ax_slope.get_xticklabels()[1].set_color("tomato")
    ax_slope.set_ylabel(prop, fontsize=9, color=fg)
    ax_slope.set_title("Where each label was moved to", fontsize=10,
                       fontweight="bold", color=fg)
    ax_slope.grid(True, axis="y", alpha=0.25)
    ax_slope.tick_params(labelsize=8, colors=fg)

    shift = float(np.mean(np.abs(y_assigned[idx] - y_true[idx])))
    spread = float(np.std(y_true)) or 1.0
    ax_slope.annotate(
        f"{idx.size} samples relabelled\n"
        f"mean |shift| = {shift:.4g}  ({shift / spread:.2f} sd)",
        xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=8, color=fg,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor="#cccccc",
                  alpha=0.9))

    if mix_fraction is not None:
        fig.suptitle(f"Mixing applied to {prop} - {mix_fraction * 100:.0f}% "
                     f"of samples requested", fontsize=9, color=fg)

    for a in (ax_spec, ax_slope):
        for spine in a.spines.values():
            spine.set_color(fg)
            spine.set_alpha(0.35)
    return fig
