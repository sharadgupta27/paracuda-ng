"""
Dynamic model-development process-flow chart (⑦ Apply tab).

Renders the exact pipeline the user configured - data → spectral → resampling →
preprocessing → model → validation → apply - as a publication-quality figure that
can be exported at 300 DPI.  The chart is *dynamic*: every node reads the live GUI
selection (and the trained/loaded-model metadata once a model exists), so it
doubles as a record of how a given model was built.

@author: Sharad Kumar Gupta
"""
import contextlib
import os
import textwrap

from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


class FlowChartMixin:

    # ─────────────────────────────────────────────────────────────────────────
    # Stage collection - read live GUI state defensively
    # ─────────────────────────────────────────────────────────────────────────
    def _flow_get(self, attr, default=""):
        """Return a tk-var's value (or plain attribute) safely."""
        obj = getattr(self, attr, None)
        if obj is None:
            return default
        try:
            return obj.get()
        except Exception:
            return obj if obj is not None else default

    def _gather_flow_stages(self):
        """Build the ordered list of ``(number, title, value, active)`` stages
        from the current configuration.  ``active`` greys-out unset stages."""
        stages = []
        # Steps the user has actually engaged with (populated by the live-refresh
        # traces).  Stages that always carry a default value - spectral domain,
        # model, validation - light up only after they are touched (or once a
        # model is trained), so a fresh / reset session shows every step faded.
        touched = getattr(self, '_flow_active_steps', None) or set()
        trained = getattr(self, 'trained_model', None) is not None

        # ① Data
        fname = getattr(self, 'input_filename', None)
        data_val = os.path.basename(fname) if fname else "No file loaded"
        stages.append(("1", "Data", data_val, bool(fname)))

        # ② Target property / properties
        props = getattr(self, 'selected_properties', None) or []
        single_prop = getattr(self, 'selected_property', None)
        if props and len(props) > 1:
            prop_val = f"{len(props)} properties: " + ", ".join(map(str, props[:4]))
            if len(props) > 4:
                prop_val += " …"
        elif single_prop:
            prop_val = str(single_prop)
        elif props:
            prop_val = str(props[0])
        else:
            prop_val = "Not selected"
        stages.append(("2", "Property", prop_val, bool(props or single_prop)))

        # ③ Spectral domain + range
        domain = self._flow_get('spectral_domain_var', 'VSWIR')
        unit = getattr(self, 'wavelength_unit', 'nm')
        lo = self._flow_get('min_wave_var', '')
        hi = self._flow_get('max_wave_var', '')
        rng = f"{lo}–{hi} {unit}" if (lo and hi) else "full range"
        stages.append(("3a", "Spectral domain", f"{domain}\n{rng}",
                       ("3a" in touched) or trained))

        # ③b Exclude ranges (optional)
        excl = str(self._flow_get('exclude_ranges_var', '')).strip()
        stages.append(("3b", "Excluded bands", excl if excl else "none",
                       bool(excl)))

        # ④ Resampling
        resampling_on = self._flow_get('resampling_var', 'No') == "Yes"
        if resampling_on:
            method = self._flow_get('resample_method_var', 'Linear Interpolation')
            sensor = self._flow_get('sensor_var', 'Custom')
            extra = (f"{self._flow_get('spacing_var', '')} nm grid"
                     if sensor == "Custom" else f"{sensor} bands")
            binning = (f", bin 1/{self._flow_get('bin_size_var', '')}"
                       if self._flow_get('binning_var', 'No') == "Yes" else "")
            resamp_val = f"{method}\n{extra}{binning}"
        else:
            resamp_val = "Off (native bands)"
        stages.append(("4", "Resampling", resamp_val, resampling_on))

        # ⑤ Preprocessing (+ missing-data)
        if trained and getattr(self, 'model_preprocessing', None):
            pre = self.model_preprocessing
        elif self._flow_get('find_best_preprocess_var', False):
            pre = "Auto (find best)"
        else:
            pre = self._flow_get('preprocess_var', 'No Preprocessing')
        missing = self._flow_get('missing_data_var', '')
        pre_val = pre + (f"\nMissing: {missing}" if missing else "")
        stages.append(("5", "Preprocessing", pre_val,
                       pre not in ("", "No Preprocessing")))

        # ⑥ Model
        batch = self._flow_get('model_mode_var', 'Single') == "Batch"
        if batch:
            sel = getattr(self, 'selected_models', None) or []
            algo = (", ".join(map(str, sel)) if sel else "not selected")
            mode_line = f"Batch: {algo}"
            if self._flow_get('auto_select_best_var', False):
                mode_line += "\nauto-select best"
        else:
            mode_line = f"Single: {self._flow_get('model_var', 'PLS-R')}"
        if self._flow_get('tune_hyperparams_var', False):
            mode_line += f"\nOptuna ×{self._flow_get('tune_trials_var', '')} tuning"
        stages.append(("6", "Model", mode_line, ("6" in touched) or trained))

        # ⑦ Validation
        test_size = self._flow_get('test_size_var', '0.2')
        cv = self._flow_get('cv_strategy_var', 'None')
        cores = self._flow_get('cores_var', '1')
        val_val = f"Test size {test_size}\nCV: {cv}  |  {cores} core(s)"
        stages.append(("7", "Validation", val_val, ("7" in touched) or trained))

        # ⑧ Apply target
        targets = []
        if self._flow_get('apply_tabular_var', False):
            targets.append("Tabular data")
        if self._flow_get('apply_models_var', False):
            targets.append("Spectral image")
        apply_val = " + ".join(targets) if targets else "Not configured"
        stages.append(("8", "Apply", apply_val, bool(targets)))

        return stages

    # ─────────────────────────────────────────────────────────────────────────
    # Figure builder
    # ─────────────────────────────────────────────────────────────────────────
    def build_process_flow_figure(self, figsize=None, dpi=100, compact=False):
        """Return a Matplotlib Figure of the current model-development pipeline.

        ``compact=True`` renders a short horizontal strip of title-only chips for
        the docked overview; the default is the detailed vertical column used by
        the "Large" view and the 300-DPI export."""
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        pal = getattr(self, '_palette', None) or {}
        c_card = pal.get('CARD', '#FFFFFF')
        c_win  = pal.get('WIN',  '#EFF4FC')
        c_line = pal.get('LINE', '#C8D8EE')
        c_bnr  = pal.get('BNR',  '#1B3A6B')
        c_pri  = pal.get('PRI',  '#1A56DB')
        c_mut  = pal.get('MUT',  '#6B7280')
        c_acc  = pal.get('ACCENT', c_pri)
        c_ok   = pal.get('OK',   '#15803D')
        font   = pal.get('FONT', 'DejaVu Sans')

        stages = self._gather_flow_stages()
        n = len(stages)

        if compact:
            return self._build_flow_compact(
                stages, dict(card=c_card, win=c_win, line=c_line, bnr=c_bnr,
                             pri=c_pri, mut=c_mut, acc=c_acc, ok=c_ok, font=font),
                figsize=figsize, dpi=dpi)

        # Geometry: single vertical column of rounded cards with down-arrows.
        row_h = 1.0                    # per-stage vertical slot (data units)
        box_h = 0.66
        box_w = 8.4
        x_left = 0.8
        xc = x_left + box_w / 2.0
        top = n * row_h                # header sits above this
        header_h = 0.9

        if figsize is None:
            figsize = (5.6, max(4.5, 0.92 * n + 1.1))
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(c_win)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, box_w + 2 * x_left)
        ax.set_ylim(0, top + header_h + 0.4)
        ax.axis('off')

        # ── Header banner ────────────────────────────────────────────────────
        trained = getattr(self, 'trained_model', None) is not None
        title = "Model Development Flow"
        subtitle = "Model trained" if trained else "configuration preview"
        hb = FancyBboxPatch((x_left, top + 0.15), box_w, header_h,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=0, facecolor=c_bnr, zorder=2)
        ax.add_patch(hb)
        ax.text(xc, top + 0.15 + header_h * 0.62, title,
                ha='center', va='center', color='white',
                fontsize=12, fontweight='bold', family=font, zorder=3)
        ax.text(xc, top + 0.15 + header_h * 0.24, subtitle,
                ha='center', va='center',
                color=(c_ok if trained else '#B9C7DE'),
                fontsize=8.5, fontstyle='italic', family=font, zorder=3)

        # ── Stage cards + connecting arrows ──────────────────────────────────
        centers = []
        for i, (num, stitle, value, active) in enumerate(stages):
            yc = top - (i + 0.5) * row_h
            centers.append(yc)
            fill = c_card if active else c_win
            edge = c_pri if active else c_line
            tcol = c_bnr if active else c_mut
            box = FancyBboxPatch((x_left, yc - box_h / 2.0), box_w, box_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.10",
                                 linewidth=1.4, edgecolor=edge, facecolor=fill,
                                 linestyle=('solid' if active else 'dashed'),
                                 zorder=2)
            ax.add_patch(box)
            # left accent stripe
            ax.add_patch(FancyBboxPatch(
                (x_left, yc - box_h / 2.0), 0.16, box_h,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                linewidth=0, facecolor=(c_acc if active else c_line), zorder=3))
            # number chip
            ax.text(x_left + 0.42, yc + box_h * 0.22, num,
                    ha='center', va='center', color=c_acc if active else c_mut,
                    fontsize=9, fontweight='bold', family=font, zorder=3)
            # title
            ax.text(x_left + 0.75, yc + box_h * 0.22, stitle,
                    ha='left', va='center', color=tcol,
                    fontsize=9.5, fontweight='bold', family=font, zorder=3)
            # value (wrapped)
            wrapped = "\n".join(
                textwrap.fill(line, 46) for line in str(value).split("\n"))
            ax.text(x_left + 0.75, yc - box_h * 0.20, wrapped,
                    ha='left', va='center', color=tcol,
                    fontsize=8, family=font, zorder=3)

        # arrows between consecutive cards
        for i in range(n - 1):
            y0 = centers[i] - box_h / 2.0
            y1 = centers[i + 1] + box_h / 2.0
            ax.add_patch(FancyArrowPatch(
                (xc, y0), (xc, y1), arrowstyle='-|>', mutation_scale=12,
                linewidth=1.4, color=c_mut, zorder=1))
        # header → first card
        ax.add_patch(FancyArrowPatch(
            (xc, top + 0.15), (xc, centers[0] + box_h / 2.0),
            arrowstyle='-|>', mutation_scale=12, linewidth=1.4,
            color=c_mut, zorder=1))

        fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
        return fig

    def _build_flow_compact(self, stages, c, figsize=None, dpi=100):
        """Docked overview: a short horizontal strip of title-only chips linked
        by arrows.  ``c`` is a dict of palette colours; ``stages`` the gathered
        ``(num, title, value, active)`` tuples."""
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        font = c['font']
        n = len(stages)

        # Horizontal geometry (data units): one slot per stage.
        card_w, gap, card_h = 1.0, 0.42, 1.15
        slot = card_w + gap
        total_w = n * slot - gap
        yc = 0.0

        if figsize is None:
            figsize = (max(7.0, 1.15 * n), 1.7)
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(c['win'])
        ax = fig.add_subplot(111)
        ax.set_xlim(-0.2, total_w + 0.2)
        ax.set_ylim(-card_h / 2.0 - 0.15, card_h / 2.0 + 0.15)
        ax.axis('off')

        # Short labels keep the strip legible when squeezed.
        short = {"Spectral domain": "Spectral", "Excluded bands": "Excluded",
                 "Preprocessing": "Preprocess", "Property": "Property"}

        centers = []
        for i, (num, title, value, active) in enumerate(stages):
            x0 = i * slot
            xcen = x0 + card_w / 2.0
            centers.append((x0, xcen))
            fill = c['card'] if active else c['win']
            edge = c['pri'] if active else c['line']
            tcol = c['bnr'] if active else c['mut']
            ax.add_patch(FancyBboxPatch(
                (x0, yc - card_h / 2.0), card_w, card_h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.3, edgecolor=edge, facecolor=fill,
                linestyle=('solid' if active else 'dashed'), zorder=2))
            # number chip
            ax.text(xcen, yc + card_h * 0.30, num, ha='center', va='center',
                    color=(c['acc'] if active else c['mut']),
                    fontsize=9, fontweight='bold', family=font, zorder=3)
            # broad title (wrapped)
            lbl = short.get(title, title)
            wrapped = textwrap.fill(lbl, 10)
            ax.text(xcen, yc - card_h * 0.08, wrapped, ha='center', va='center',
                    color=tcol, fontsize=7.5, fontweight='bold',
                    family=font, zorder=3, linespacing=0.95)

        # right-pointing arrows between chips
        for i in range(n - 1):
            x_end = centers[i][0] + card_w
            x_nxt = centers[i + 1][0]
            ax.add_patch(FancyArrowPatch(
                (x_end, yc), (x_nxt, yc), arrowstyle='-|>', mutation_scale=9,
                linewidth=1.2, color=c['mut'], zorder=1))

        fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Live-refresh plumbing - keep the always-visible preview in sync with the
    # wizard as parameters change on *any* step (not just the Apply tab).
    # ─────────────────────────────────────────────────────────────────────────

    # stage-number → tk-var attributes whose changes mark that step "engaged"
    # (lighting its card) and trigger a live redraw of the flow chart.
    _FLOW_STAGE_VARS = {
        "3a":  ["spectral_domain_var", "min_wave_var", "max_wave_var"],
        "3b": ["exclude_ranges_var"],
        "4":  ["resampling_var", "resample_method_var", "sensor_var",
               "spacing_var", "binning_var", "bin_size_var"],
        "5":  ["preprocess_var", "find_best_preprocess_var", "missing_data_var"],
        "6":  ["model_mode_var", "model_var", "auto_select_best_var",
               "tune_hyperparams_var", "tune_trials_var"],
        "7":  ["test_size_var", "cv_strategy_var", "cores_var"],
        "8":  ["apply_tabular_var", "apply_models_var"],
    }

    def _wire_flow_live_refresh(self):
        """Attach write-traces to every configuration var and start a light
        state poll so the flow chart tracks the live GUI in real time."""
        if getattr(self, '_flow_live_wired', False):
            return
        self._flow_active_steps = set()
        self._flow_suspend_trace = False
        self._flow_last_signature = None
        self._flow_refresh_job = None
        for stage, names in self._FLOW_STAGE_VARS.items():
            for name in names:
                var = getattr(self, name, None)
                if var is None or not hasattr(var, 'trace_add'):
                    continue
                def _cb(*_a, _s=stage):
                    if not getattr(self, '_flow_suspend_trace', False):
                        self._flow_active_steps.add(_s)
                    self._schedule_flow_refresh()
                with contextlib.suppress(Exception):
                    var.trace_add("write", _cb)
        self._flow_live_wired = True
        # Reflow the compact strip when its docked pane is resized.
        holder = getattr(self, 'flow_preview_holder', None)
        if holder is not None:
            holder.bind("<Configure>", self._on_flow_holder_resize)
        # Poll for non-var state changes (data load, property pick, model trained).
        self._flow_poll()
        # Initial draw so the (faded) chart is visible immediately.
        self.refresh_flow_preview()

    def _flow_signature(self):
        """Cheap fingerprint of the rendered state so the poll only redraws on
        an actual change."""
        try:
            stages = self._gather_flow_stages()
        except Exception:
            return None
        trained = getattr(self, 'trained_model', None) is not None
        return repr([(n, v, a) for n, _t, v, a in stages]) + f"|{trained}"

    def _schedule_flow_refresh(self):
        """Debounce rapid var changes into a single redraw (~150 ms)."""
        if getattr(self, '_closing', False):
            return
        job = getattr(self, '_flow_refresh_job', None)
        if job:
            with contextlib.suppress(Exception):
                self.after_cancel(job)
        try:
            self._flow_refresh_job = self.after(400, self._do_flow_refresh)
        except Exception:
            self._do_flow_refresh()

    def _do_flow_refresh(self):
        self._flow_refresh_job = None
        self._flow_last_signature = self._flow_signature()
        self.refresh_flow_preview()

    def _flow_poll(self):
        """Redraw when the gathered state changes from something a trace can't
        see (file loaded, property selected, model trained)."""
        # Stop rescheduling once the window is closing, otherwise a pending
        # after-job fires against a destroyed interpreter ("invalid command
        # name ..._flow_poll") and spams the console at shutdown.
        if getattr(self, '_closing', False):
            return
        with contextlib.suppress(Exception):
            sig = self._flow_signature()
            if sig is not None and sig != getattr(self, '_flow_last_signature', None):
                self._flow_last_signature = sig
                self.refresh_flow_preview()
        with contextlib.suppress(Exception):
            self._flow_poll_job = self.after(700, self._flow_poll)

    def cancel_flow_jobs(self):
        """Cancel the pending flow-chart after-jobs (called on window close)."""
        for attr in ('_flow_poll_job', '_flow_refresh_job'):
            job = getattr(self, attr, None)
            if job:
                with contextlib.suppress(Exception):
                    self.after_cancel(job)
                setattr(self, attr, None)

    # ─────────────────────────────────────────────────────────────────────────
    # Embedded preview + actions (widgets built in layout_mixin)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_flow_holder_resize(self, event):
        """Reflow the compact strip when the docked pane is resized (debounced)."""
        last = getattr(self, '_flow_last_draw_size', (0, 0))
        if abs(event.width - last[0]) < 12 and abs(event.height - last[1]) < 12:
            return
        self._schedule_flow_refresh()

    def refresh_flow_preview(self, event=None):
        """(Re)draw the always-visible, docked compact flow overview, sized to
        fill its holder so all stage chips stay visible."""
        holder = getattr(self, 'flow_preview_holder', None)
        if holder is None:
            return
        w = holder.winfo_width() or 700
        h = holder.winfo_height() or 200
        self._flow_last_draw_size = (w, h)
        figsize = (max(4.0, w / 96.0), max(1.3, h / 96.0))
        try:
            fig = self.build_process_flow_figure(compact=True, figsize=figsize, dpi=96)
        except Exception as e:
            # Never let a drawing error break the tab.
            if hasattr(self, 'status_text'):
                self.status_text.insert(tk.END, f"Flow chart error: {e}\n")
            return
        old_fig = getattr(self, 'flow_preview_fig', None)
        canvas = getattr(self, 'flow_preview_canvas', None)
        self.flow_preview_fig = fig
        if canvas is not None:
            # Reuse the existing embedded widget -- destroying and recreating
            # the Tk canvas on every keystroke (min/max wave, etc.) is what
            # was stalling the event loop and eating typed characters.
            try:
                canvas.figure = fig
                canvas.draw()
                if old_fig is not None:
                    plt.close(old_fig)
                return
            except Exception:
                with contextlib.suppress(Exception):
                    canvas.get_tk_widget().destroy()
                self.flow_preview_canvas = None
        self.flow_preview_canvas = FigureCanvasTkAgg(fig, holder)
        self.flow_preview_canvas.draw()
        self.flow_preview_canvas.get_tk_widget().pack(fill="both", expand=True)

    def view_flow_large(self):
        """Open a large, scrollable view of the process-flow chart."""
        win = tk.Toplevel(self)
        win.title("Model Development Flow")
        win.geometry("640x900")
        with contextlib.suppress(Exception):
            win.configure(bg=self._c('WIN'))
        fig = self.build_process_flow_figure(figsize=(6.2, 9.2), dpi=110)
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        bar = ttk.Frame(win)
        bar.pack(fill="x")
        ttk.Button(bar, text="💾 Save 300 DPI",
                   command=lambda f=fig: self._save_flow_fig(f)).pack(
            side="right", padx=6, pady=6)

    def save_flow_figure(self):
        """Save the current process-flow chart to a publication-ready file."""
        fig = self.build_process_flow_figure(figsize=(6.2, 9.2), dpi=300)
        self._save_flow_fig(fig)

    def _save_flow_fig(self, fig):
        path = filedialog.asksaveasfilename(
            title="Save process-flow chart (300 DPI)",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"),
                       ("SVG vector", "*.svg")],
            initialfile="paracuda_model_flow.png")
        if not path:
            return
        try:
            fig.savefig(path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            messagebox.showinfo(
                "Saved", f"Process-flow chart saved at 300 DPI:\n"
                         f"{os.path.basename(path)}")
            if hasattr(self, 'status_text'):
                self.status_text.insert(
                    tk.END, f"Saved process-flow chart: {path}\n")
                self.status_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
