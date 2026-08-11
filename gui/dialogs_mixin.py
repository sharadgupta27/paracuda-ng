"""
Data Distribution, Check Spectral Integrity and Spectral Harmonization dialogs.

@author: Sharad Kumar Gupta
"""
import contextlib
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


class DialogsMixin:

    # ─────────────────────────────────────────────────────────────────────────
    # Data Distribution viewer
    # ─────────────────────────────────────────────────────────────────────────
    def show_data_distribution(self):
        """Show the distribution of every property column in the loaded data.

        Deliberately available straight after loading, before any configuration:
        a strongly skewed, near-constant or mostly-missing target will not model
        well whatever is chosen downstream, and seeing that here saves the user
        from a pointless run.
        """
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        wl_cols = list(self.wavelengths) if self.wavelengths is not None else []
        # Prefer the properties the user selected; otherwise show every candidate.
        columns = (list(self.selected_properties)
                   if getattr(self, 'selected_properties', None)
                   else suggest_numeric_columns(self.df, wl_cols))
        if not columns:
            messagebox.showinfo(
                "Data Distribution",
                "No numeric property columns were found to summarise.\n\n"
                "The wavelength columns are excluded on purpose - this view is "
                "about the properties you want to predict.")
            return

        win = tk.Toplevel(self)
        win.title("Data Distribution")
        win.geometry("1060x780")
        win.resizable(True, True)
        with contextlib.suppress(Exception):
            win.transient(self)

        header = ttk.Frame(win, padding=(10, 8, 10, 0))
        header.pack(fill="x")
        ttk.Label(header, text="Distribution of the target properties",
                  font=('Helvetica', 12, 'bold')).pack(anchor="w")
        ttk.Label(
            header,
            text=("Check for skew, outliers and missing values BEFORE training. "
                  "A strongly skewed or near-constant property is a data problem, "
                  "not a modelling one."),
            wraplength=1020, foreground="#666666",
            font=('Helvetica', 9, 'italic')).pack(anchor="w", pady=(2, 6))

        # Every property is summarised up front (cheap - it is only statistics),
        # but only the selected one is plotted.  Drawing all of them at once
        # produced a grid of thumbnails too small to read.
        summaries = summarize_distribution(self.df, columns, wl_cols)
        palette = getattr(self, '_palette', None)

        picker = ttk.Frame(win, padding=(10, 0, 10, 4))
        picker.pack(fill="x")
        ttk.Label(picker, text="Property:",
                  font=('Helvetica', 9, 'bold')).pack(side="left")
        # Flag the problem properties in the list itself, so a bad one is
        # obvious without stepping through every entry.
        _marks = {'ok': '✓', 'warn': '!', 'problem': '!!'}
        choice_to_col, choices = {}, []
        for s in summaries:
            label = f"{_marks.get(s['severity'], '')} {s['column']}".strip()
            choice_to_col[label] = s['column']
            choices.append(label)
        prop_var = tk.StringVar(value=choices[0] if choices else "")
        prop_combo = ttk.Combobox(picker, textvariable=prop_var, values=choices,
                                  state='readonly', width=32)
        prop_combo.pack(side="left", padx=(6, 12))
        counts = {k: sum(1 for s in summaries if s['severity'] == k)
                  for k in ('ok', 'warn', 'problem')}
        ttk.Label(picker,
                  text=(f"{len(summaries)} properties  -  {counts['ok']} ok, "
                        f"{counts['warn']} to check, {counts['problem']} problem"),
                  foreground="#666666",
                  font=('Helvetica', 9)).pack(side="left")

        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        plot_tab = ttk.Frame(nb, padding=4)
        text_tab = ttk.Frame(nb, padding=4)
        nb.add(plot_tab, text="  📈 Plots  ")
        nb.add(text_tab, text="  📋 Statistics & Findings  ")

        state = {'fig': None, 'column': None}
        canvas_holder = ttk.Frame(plot_tab)
        canvas_holder.pack(fill="both", expand=True)

        report = tk.Text(text_tab, wrap="word", font=('Consolas', 9))
        scroll = ttk.Scrollbar(text_tab, orient="vertical", command=report.yview)
        report.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        report.pack(side="left", fill="both", expand=True)
        with contextlib.suppress(Exception):
            report.configure(bg=self._c('BNR'), fg=self._c('TXT'),
                             insertbackground=self._c('TXT'))

        verdict_lbl = ttk.Label(win, text="", font=('Helvetica', 10, 'bold'))
        verdict_lbl.pack(anchor="w", padx=12, pady=(0, 4))

        _verdicts = {
            'ok': ("✓ %s looks usable.", "#1a9850"),
            'warn': ("! %s needs a look - see Statistics & Findings.", "#b8860b"),
            'problem': ("!! %s has a serious distribution problem - fix the "
                        "data before training.", "#d73027"),
        }

        def show_property(*_args):
            column = choice_to_col.get(prop_var.get())
            if column is None or column == state['column']:
                return
            state['column'] = column
            for child in canvas_holder.winfo_children():
                child.destroy()
            fig, summary = create_property_figure(
                self.df, column, wavelength_cols=wl_cols, palette=palette)
            state['fig'] = fig
            canvas = FigureCanvasTkAgg(fig, canvas_holder)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

            report.configure(state="normal")
            report.delete("1.0", tk.END)
            report.insert("1.0", property_report_text(summary))
            report.configure(state="disabled")

            text, colour = _verdicts[summary['severity']]
            verdict_lbl.config(text=text % column, foreground=colour)

        prop_combo.bind("<<ComboboxSelected>>", show_property)
        show_property()

        btns = ttk.Frame(win, padding=(10, 0, 10, 10))
        btns.pack(fill="x")

        def save_plot():
            path = filedialog.asksaveasfilename(
                parent=win, defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")],
                initialfile=(f"{getattr(self, 'input_filename', 'data')}_"
                             f"{state['column']}_distribution.png"),
                title=f"Save the {state['column']} plot as")
            if path:
                state['fig'].savefig(path, dpi=300, bbox_inches="tight")
                messagebox.showinfo("Saved", f"Plot saved to:\n{path}", parent=win)

        def save_all_plots():
            """One overview sheet with every property, for a report appendix."""
            path = filedialog.asksaveasfilename(
                parent=win, defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")],
                initialfile=f"{getattr(self, 'input_filename', 'data')}_distribution.png",
                title="Save an overview of all properties as")
            if not path:
                return
            fig_all, _ = create_distribution_figure(
                self.df, columns, wavelength_cols=wl_cols, palette=palette,
                max_cols=len(columns))
            fig_all.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Overview saved to:\n{path}", parent=win)

        def save_stats():
            path = filedialog.asksaveasfilename(
                parent=win, defaultextension=".xlsx",
                filetypes=[("Excel workbook", "*.xlsx"), ("Text file", "*.txt")],
                initialfile=f"{getattr(self, 'input_filename', 'data')}_distribution.xlsx",
                title="Save distribution statistics as")
            if not path:
                return
            if path.lower().endswith(".txt"):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(distribution_report_text(summaries))
            else:
                rows = [{k: v for k, v in s.items() if k != 'findings'}
                        for s in summaries]
                for row, s in zip(rows, summaries):
                    row['findings'] = " | ".join(s['findings'])
                pd.DataFrame(rows).to_excel(path, index=False)
            messagebox.showinfo("Saved", f"Statistics saved to:\n{path}", parent=win)

        ttk.Button(btns, text="💾 Save This Plot (300 DPI)",
                   command=save_plot).pack(side="left", padx=2)
        ttk.Button(btns, text="💾 Save All Properties",
                   command=save_all_plots).pack(side="left", padx=2)
        ttk.Button(btns, text="💾 Save Statistics",
                   command=save_stats).pack(side="left", padx=2)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right", padx=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Check Spectral Integrity dialog (label permutation + mixing)
    # ─────────────────────────────────────────────────────────────────────────
    def show_data_randomization(self):
        """Open a dialog for label permutation test and spectral mixing integrity check."""
        if self.df is None or self.wavelengths is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        if not self.soil_properties:
            messagebox.showwarning("Warning", "No soil properties found in the dataset.")
            return

        win = tk.Toplevel(self)
        win.title("Check Spectral Integrity - Label Permutation & Mixing Tests")
        win.geometry("880x720")
        win.resizable(True, True)

        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True, padx=10, pady=8)

        # ── shared state for save actions ──────────────────────────────────
        _perm_state = {}   # filled after run_perm_test
        _mix_state  = {}   # filled after run_mix_test

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 : Label Permutation Test
        # ══════════════════════════════════════════════════════════════════
        tab1 = ttk.Frame(nb, padding=10)
        nb.add(tab1, text="  Label Permutation Test  ")

        ctrl1 = ttk.Frame(tab1)
        ctrl1.pack(fill="x")

        ttk.Label(ctrl1, text="Property:").grid(row=0, column=0, sticky="w")
        prop_var = tk.StringVar(value=self.soil_properties[0])
        ttk.Combobox(ctrl1, textvariable=prop_var, values=self.soil_properties,
                     state='readonly', width=28).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(ctrl1, text="Permutations:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        nperm_var = tk.StringVar(value="200")
        ttk.Entry(ctrl1, textvariable=nperm_var, width=7).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(ctrl1, text="CV folds:").grid(row=0, column=4, sticky="w", padx=(12, 0))
        cv_var1 = tk.StringVar(value="5")
        ttk.Entry(ctrl1, textvariable=cv_var1, width=5).grid(row=0, column=5, sticky="w", padx=4)

        btn_row1 = ttk.Frame(tab1)
        btn_row1.pack(fill="x", pady=4)

        result_lbl1 = ttk.Label(tab1, text="", wraplength=640,
                                 font=('Helvetica', 9, 'italic'))
        result_lbl1.pack(anchor="w", pady=2)

        fig1 = Figure(figsize=(6.2, 3.2), tight_layout=True)
        ax1  = fig1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(fig1, tab1)
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        def run_perm_test():
            try:
                prop   = prop_var.get()
                n_perm = max(10, int(nperm_var.get()))
                cv_k   = max(2, int(cv_var1.get()))
                X = self.df[self.wavelengths].values
                y = self.df[prop].values
                X, y, _ = self._apply_missing_data_handling(X, y)

                from sklearn.cross_decomposition import PLSRegression
                from sklearn.model_selection import cross_val_score
                n_comp = min(5, X.shape[1], len(y) - 1)
                cv_k   = min(cv_k, len(y))
                # Use the SAME cv folds for the null distribution so the observed
                # R² is directly comparable to the permuted scores.
                perm_scores = randomize_label_test(X, y, n_permutations=n_perm, cv=cv_k)
                obs_r2 = float(np.mean(cross_val_score(
                    PLSRegression(n_components=n_comp), X, y,
                    cv=cv_k, scoring='r2')))

                p_val = float(np.mean(np.array(perm_scores) >= obs_r2))

                ax1.clear()
                ax1.hist(perm_scores, bins=min(30, n_perm // 5 + 5),
                         color='steelblue', alpha=0.75, edgecolor='white',
                         label=f'Permuted R² (n={n_perm})')
                ax1.axvline(obs_r2, color='crimson', linewidth=2, linestyle='--',
                            label=f'Observed R²={obs_r2:.3f}')
                ax1.axvline(float(np.mean(perm_scores)), color='orange',
                            linewidth=1.5, linestyle=':',
                            label=f'Permuted mean={np.mean(perm_scores):.3f}')
                ax1.set_xlabel("Cross-validated R²", fontsize=10)
                ax1.set_ylabel("Frequency", fontsize=10)
                ax1.set_title(f"Label Permutation Test - {prop}", fontsize=11, fontweight='bold')
                ax1.legend(fontsize=9)
                canvas1.draw()

                interp = ("✔ Model is statistically significant (p<0.05)."
                          if p_val < 0.05 else
                          "✘ Model is NOT significant (p≥0.05) - labels may be uninformative.")
                result_lbl1.config(
                    text=(f"Observed R²={obs_r2:.4f}  |  "
                          f"Permuted mean={np.mean(perm_scores):.4f}  |  "
                          f"p-value={p_val:.4f}  ({n_perm} permutations, {cv_k}-fold CV)\n"
                          f"{interp}"))

                _perm_state.update({
                    'prop': prop, 'obs_r2': obs_r2,
                    'perm_scores': np.array(perm_scores),
                    'p_val': p_val, 'n_perm': n_perm, 'cv_k': cv_k, 'fig': fig1
                })
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=win)

        def save_perm_results():
            if not _perm_state:
                messagebox.showwarning("No results", "Run the permutation test first.", parent=win)
                return
            out = filedialog.asksaveasfilename(
                title="Save permutation test results",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=f"permutation_test_{_perm_state['prop']}.xlsx",
                parent=win)
            if not out:
                return
            try:
                scores = _perm_state['perm_scores']
                df_out = pd.DataFrame({
                    'Permuted_R2': scores,
                    'Rank': pd.Series(scores).rank(ascending=False).astype(int)
                })
                summary = pd.DataFrame({
                    'Metric': ['Property', 'Observed_R2', 'Permuted_Mean_R2',
                               'Permuted_Std_R2', 'p_value', 'n_permutations', 'CV_folds'],
                    'Value': [_perm_state['prop'], f"{_perm_state['obs_r2']:.6f}",
                              f"{scores.mean():.6f}", f"{scores.std():.6f}",
                              f"{_perm_state['p_val']:.6f}",
                              _perm_state['n_perm'], _perm_state['cv_k']]
                })
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    df_out.to_excel(writer, sheet_name='Permuted_Scores', index=False)
                # Save plot alongside
                plot_out = out.replace('.xlsx', '_plot.png')
                _perm_state['fig'].savefig(plot_out, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Saved",
                    f"Results: {os.path.basename(out)}\nPlot: {os.path.basename(plot_out)}",
                    parent=win)
            except Exception as e:
                messagebox.showerror("Save Error", str(e), parent=win)

        ttk.Button(btn_row1, text="▶ Run Permutation Test",
                   command=run_perm_test).pack(side="left", padx=4)
        ttk.Button(btn_row1, text="💾 Save Results",
                   command=save_perm_results).pack(side="left", padx=4)

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 : Spectral Mixing Integrity
        # ══════════════════════════════════════════════════════════════════
        tab2 = ttk.Frame(nb, padding=10)
        nb.add(tab2, text="  Spectral Mixing Integrity  ")

        ctrl2 = ttk.Frame(tab2)
        ctrl2.pack(fill="x")

        ttk.Label(ctrl2, text="Property:").grid(row=0, column=0, sticky="w")
        prop_var2 = tk.StringVar(value=self.soil_properties[0])
        ttk.Combobox(ctrl2, textvariable=prop_var2, values=self.soil_properties,
                     state='readonly', width=28).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(ctrl2, text="Mix fraction (0–1):").grid(row=0, column=2, sticky="w", padx=(12, 0))
        mix_var = tk.StringVar(value="0.20")
        ttk.Entry(ctrl2, textvariable=mix_var, width=7).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(ctrl2, text="CV folds:").grid(row=0, column=4, sticky="w", padx=(12, 0))
        cv_var2 = tk.StringVar(value="5")
        ttk.Entry(ctrl2, textvariable=cv_var2, width=5).grid(row=0, column=5, sticky="w", padx=4)

        btn_row2 = ttk.Frame(tab2)
        btn_row2.pack(fill="x", pady=4)

        result_lbl2 = ttk.Label(tab2, text="", wraplength=820,
                                  font=('Helvetica', 9, 'italic'))
        result_lbl2.pack(anchor="w", pady=2)

        # Two views: the R² impact bar chart and a visualization of exactly how
        # the mixing was applied.
        mix_nb = ttk.Notebook(tab2)
        mix_nb.pack(fill="both", expand=True, pady=(4, 0))

        mix_tab_r2   = ttk.Frame(mix_nb, padding=4)
        mix_tab_spec = ttk.Frame(mix_nb, padding=4)
        mix_nb.add(mix_tab_r2,   text="  R² Impact  ")
        mix_nb.add(mix_tab_spec, text="  Mixing Applied (spectra)  ")

        fig2 = Figure(figsize=(6.2, 3.2), tight_layout=True)
        ax2  = fig2.add_subplot(111)
        canvas2 = FigureCanvasTkAgg(fig2, mix_tab_r2)
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        # The mixing figure is rebuilt per run (see utils/integrity_plots) and
        # swapped into this holder.
        spec_holder = ttk.Frame(mix_tab_spec)
        spec_holder.pack(fill="both", expand=True)
        ttk.Label(spec_holder,
                  text="Run the check to view how mixing is applied.",
                  foreground="#888888",
                  font=('Helvetica', 9, 'italic')).pack(pady=20)
        _spec_canvas = {'canvas': None}

        # Before→after label mapping, laid out across columns: a single tall
        # column of a dozen rows squeezed the plot above it off the dialog.
        map_frame = ttk.LabelFrame(mix_tab_spec,
                                   text="Label reassignment (true → assigned)",
                                   padding=(6, 2))
        map_frame.pack(fill="x", pady=(4, 0))
        ttk.Label(map_frame, text="(runs after the check)", foreground="#888888",
                  font=('Helvetica', 8, 'italic')).pack(anchor="w")

        def _fill_map_grid(rows, n_hidden):
            """Lay the reassignment rows out in as many columns as fit."""
            for child in map_frame.winfo_children():
                child.destroy()
            if not rows:
                return
            ncols = 4 if len(rows) > 12 else (3 if len(rows) > 6 else 2)
            nrows = int(math.ceil(len(rows) / ncols))
            for k, text in enumerate(rows):
                r, c = k % nrows, k // nrows
                ttk.Label(map_frame, text=text, font=('Consolas', 8),
                          foreground="#444444").grid(row=r, column=c,
                                                     sticky="w", padx=(0, 14))
            for c in range(ncols):
                map_frame.grid_columnconfigure(c, weight=1)
            if n_hidden:
                ttk.Label(map_frame, text=f"… (+{n_hidden} more relabelled)",
                          font=('Helvetica', 8, 'italic'),
                          foreground="#888888").grid(
                    row=nrows, column=0, columnspan=ncols, sticky="w",
                    pady=(2, 0))

        def run_mix_test():
            try:
                prop = prop_var2.get()
                frac = float(mix_var.get())
                cv_k = max(2, int(cv_var2.get()))
                if not (0.0 < frac < 1.0):
                    messagebox.showerror("Input Error",
                        "Mix fraction must be between 0 and 1 (exclusive).", parent=win)
                    return
                X = self.df[self.wavelengths].values
                y = self.df[prop].values
                X, y, _ = self._apply_missing_data_handling(X, y)
                X_mix, y_mix, changed_idx = mix_spectra_integrity_check(X, y, mix_fraction=frac)
                n_mix = int(len(changed_idx))
                if n_mix == 0:
                    messagebox.showwarning(
                        "No change",
                        "Mixing produced no label changes (too few samples or "
                        "identical values). Try a larger fraction or dataset.",
                        parent=win)
                    return

                from sklearn.cross_decomposition import PLSRegression
                from sklearn.model_selection import cross_val_score
                n_comp = min(5, X.shape[1], len(y) - 1)
                cv_k   = min(cv_k, len(y))
                r2_orig = float(np.mean(cross_val_score(
                    PLSRegression(n_components=n_comp), X, y, cv=cv_k, scoring='r2')))
                r2_mix  = float(np.mean(cross_val_score(
                    PLSRegression(n_components=n_comp), X_mix, y_mix, cv=cv_k, scoring='r2')))
                delta = r2_orig - r2_mix

                ax2.clear()
                bars = ax2.bar(['Original data', f'After {frac*100:.0f}% mixing'],
                               [r2_orig, r2_mix],
                               color=['steelblue', 'tomato'], width=0.45,
                               edgecolor='white', linewidth=1.2)
                for bar, val in zip(bars, [r2_orig, r2_mix]):
                    ax2.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.01, f'{val:.3f}',
                             ha='center', va='bottom', fontsize=11, fontweight='bold')
                ax2.set_ylabel("Cross-validated R²", fontsize=10)
                ax2.set_title(f"Spectral Mixing Integrity - {prop}", fontsize=11, fontweight='bold')
                ax2.set_ylim(bottom=min(0, r2_mix - 0.15),
                             top=max(1.0, r2_orig + 0.15))
                ax2.axhline(0, color='grey', linewidth=0.8, linestyle='--')
                canvas2.draw()

                # ── Visualise how the mixing is applied ──────────────────────
                # Mixing swaps labels, it never touches a spectrum - so drawing
                # the same curves twice (once per label set) showed nothing.
                # The shared figure instead colours the affected spectra by
                # their TRUE value and charts where each label was moved to.
                fig_spec = create_mixing_figure(
                    self.wavelengths, X_mix, y, y_mix, changed_idx,
                    prop=prop, unit=getattr(self, 'wavelength_unit', 'nm'),
                    mix_fraction=frac,
                    palette=getattr(self, '_palette', None))
                for child in spec_holder.winfo_children():
                    child.destroy()
                _spec_canvas['canvas'] = FigureCanvasTkAgg(fig_spec, spec_holder)
                _spec_canvas['canvas'].get_tk_widget().pack(fill="both", expand=True)
                _spec_canvas['canvas'].draw()

                # before → after mapping, across columns
                _fill_map_grid(*label_reassignment_rows(y, y_mix, changed_idx))

                interp = (f"✔ Label integrity confirmed - R² dropped by {delta:.4f} after mixing."
                          if delta > 0.02 else
                          f"⚠ Small R² change ({delta:.4f}) - labels may not carry strong signal.")
                result_lbl2.config(
                    text=(f"Original R²={r2_orig:.4f}  |  "
                          f"Mixed R²={r2_mix:.4f}  |  "
                          f"Δ={delta:.4f}  |  "
                          f"{n_mix} samples relabelled ({frac*100:.0f}% requested),  {cv_k}-fold CV\n"
                          f"{interp}"))

                _mix_state.update({
                    'prop': prop, 'frac': frac, 'n_mix': n_mix,
                    'r2_orig': r2_orig, 'r2_mix': r2_mix, 'delta': delta,
                    'cv_k': cv_k, 'fig': fig2, 'fig_spec': fig_spec,
                    'X': X, 'y': y, 'X_mix': X_mix, 'y_mix': y_mix,
                    'changed_idx': np.asarray(changed_idx)
                })
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=win)

        def save_mix_results():
            if not _mix_state:
                messagebox.showwarning("No results", "Run the mixing test first.", parent=win)
                return
            out = filedialog.asksaveasfilename(
                title="Save mixing integrity results",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=f"mixing_integrity_{_mix_state['prop']}.xlsx",
                parent=win)
            if not out:
                return
            try:
                summary = pd.DataFrame({
                    'Metric': ['Property', 'Mix_Fraction', 'N_Mixed_Samples',
                               'Original_R2 (CV)', 'Mixed_R2 (CV)', 'Delta_R2', 'CV_folds'],
                    'Value': [_mix_state['prop'], _mix_state['frac'],
                              _mix_state['n_mix'],
                              f"{_mix_state['r2_orig']:.6f}",
                              f"{_mix_state['r2_mix']:.6f}",
                              f"{_mix_state['delta']:.6f}",
                              _mix_state['cv_k']]
                })
                # Mixed dataset tab
                wl_names = list(self.wavelengths)
                df_mixed = pd.DataFrame(_mix_state['X_mix'], columns=wl_names)
                df_mixed.insert(0, _mix_state['prop'], _mix_state['y_mix'])
                # Before → after relabelling map (only the samples truly changed)
                cidx = _mix_state.get('changed_idx', np.array([], dtype=int))
                df_map = pd.DataFrame({
                    'Sample_Index': cidx.astype(int),
                    'True_Value': _mix_state['y'][cidx],
                    'Assigned_Value': _mix_state['y_mix'][cidx],
                })
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    df_map.to_excel(writer, sheet_name='Relabelled_Map', index=False)
                    df_mixed.to_excel(writer, sheet_name='Mixed_Dataset', index=False)
                plot_out = out.replace('.xlsx', '_plot.png')
                _mix_state['fig'].savefig(plot_out, dpi=300, bbox_inches='tight')
                # Also save the side-by-side spectra visualization if present.
                spec_out = None
                if _mix_state.get('fig_spec') is not None:
                    spec_out = out.replace('.xlsx', '_mixing_spectra.png')
                    _mix_state['fig_spec'].savefig(spec_out, dpi=300, bbox_inches='tight')
                msg = f"Results: {os.path.basename(out)}\nPlot: {os.path.basename(plot_out)}"
                if spec_out:
                    msg += f"\nSpectra: {os.path.basename(spec_out)}"
                messagebox.showinfo("Saved", msg, parent=win)
            except Exception as e:
                messagebox.showerror("Save Error", str(e), parent=win)

        ttk.Button(btn_row2, text="▶ Run Mixing Integrity Check",
                   command=run_mix_test).pack(side="left", padx=4)
        ttk.Button(btn_row2, text="💾 Save Results",
                   command=save_mix_results).pack(side="left", padx=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Spectral Harmonization dialog
    # ─────────────────────────────────────────────────────────────────────────
    def show_spectral_harmonization(self):
        """Open a dialog for cross-sensor spectral transfer functions."""
        win = tk.Toplevel(self)
        win.title("Spectral Harmonization - Transfer Function")
        win.geometry("740x680")
        win.resizable(True, True)

        nb_harm = ttk.Notebook(win)
        nb_harm.pack(fill="both", expand=True, padx=8, pady=8)

        # shared state
        _tf_state = {}   # filled after compute_tf

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 : Compute Transfer Function
        # ══════════════════════════════════════════════════════════════════
        tab_calc = ttk.Frame(nb_harm, padding=10)
        nb_harm.add(tab_calc, text="  Compute Transfer Function  ")

        frame = ttk.Frame(tab_calc)
        frame.pack(fill="x")

        src_path_var = tk.StringVar()
        tgt_path_var = tk.StringVar()

        def browse(var, label_w):
            p = filedialog.askopenfilename(
                filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
            if p:
                var.set(p)
                label_w.config(text=os.path.basename(p))

        ttk.Label(frame, text="Source spectra (field instrument):",
                  font=('Helvetica', 9, 'bold')).grid(
                      row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        src_lbl = ttk.Label(frame, text="No file selected", foreground="grey")
        src_lbl.grid(row=1, column=1, sticky="w")
        ttk.Button(frame, text="Browse",
                   command=lambda: browse(src_path_var, src_lbl)).grid(row=1, column=0, padx=4)

        ttk.Label(frame, text="Target spectra (satellite / reference):",
                  font=('Helvetica', 9, 'bold')).grid(
                      row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))
        tgt_lbl = ttk.Label(frame, text="No file selected", foreground="grey")
        tgt_lbl.grid(row=3, column=1, sticky="w")
        ttk.Button(frame, text="Browse",
                   command=lambda: browse(tgt_path_var, tgt_lbl)).grid(row=3, column=0, padx=4)

        ttk.Label(frame, text="PLS components:").grid(row=4, column=0, sticky="w", pady=(8, 2))
        ncomp_var = tk.StringVar(value="5")
        ttk.Entry(frame, textvariable=ncomp_var, width=8).grid(row=4, column=1, sticky="w")

        btn_row_tf = ttk.Frame(tab_calc)
        btn_row_tf.pack(fill="x", pady=4)

        result_lbl = ttk.Label(tab_calc, text="", wraplength=700,
                                font=('Helvetica', 9, 'italic'))
        result_lbl.pack(anchor="w", pady=2)

        fig_tf = Figure(figsize=(6.5, 3.2), tight_layout=True)
        ax_tf  = fig_tf.add_subplot(111)
        canvas_tf = FigureCanvasTkAgg(fig_tf, tab_calc)
        canvas_tf.get_tk_widget().pack(fill="both", expand=True)

        def _load_file(path):
            if path.lower().endswith('.csv'):
                return pd.read_csv(path)
            return pd.read_excel(path)

        def _extract_spectra(df):
            wl_cols = []
            for col in df.columns:
                with contextlib.suppress((ValueError, TypeError)):
                    val = float(col)
                    if 0.3 <= val <= 20000:
                        wl_cols.append(col)
            return df[wl_cols].values.astype(float), [float(c) for c in wl_cols]

        def _is_numeric_col(col, lo, hi):
            try:
                val = float(col)
                return lo <= val <= hi
            except (ValueError, TypeError):
                return False

        def compute_tf():
            try:
                if not src_path_var.get() or not tgt_path_var.get():
                    messagebox.showwarning("Warning",
                        "Please select both source and target files.", parent=win)
                    return
                n_comp = max(1, int(ncomp_var.get()))
                df_src = _load_file(src_path_var.get())
                df_tgt = _load_file(tgt_path_var.get())
                X_src, wl_src = _extract_spectra(df_src)
                X_tgt, wl_tgt = _extract_spectra(df_tgt)

                unit_s, wl_nm_s = detect_wavelength_unit(wl_src)
                unit_t, wl_nm_t = detect_wavelength_unit(wl_tgt)

                if not np.allclose(wl_nm_s, wl_nm_t, atol=2.0):
                    X_src = safe_interpolate_spectra(X_src, wl_nm_s, wl_nm_t)
                    wl_nm_s = wl_nm_t

                if X_src.shape[0] != X_tgt.shape[0]:
                    messagebox.showerror(
                        "Row mismatch",
                        f"Source ({X_src.shape[0]} rows) and target ({X_tgt.shape[0]} rows) "
                        "must have the same number of paired measurements.",
                        parent=win)
                    return

                pls, scaler_s, scaler_t, r2_per_band = spectral_transfer_function(
                    X_src, X_tgt, n_components=n_comp)
                mean_r2 = float(np.mean(r2_per_band))

                # Plot per-band R²
                ax_tf.clear()
                ax_tf.plot(wl_nm_t, r2_per_band, color='steelblue', linewidth=1.5, label='Per-band R²')
                ax_tf.axhline(mean_r2, color='crimson', linestyle='--', linewidth=1.2,
                              label=f'Mean R²={mean_r2:.3f}')
                ax_tf.fill_between(wl_nm_t, r2_per_band, alpha=0.15, color='steelblue')
                ax_tf.set_xlabel("Wavelength (nm)", fontsize=10)
                ax_tf.set_ylabel("R²", fontsize=10)
                ax_tf.set_title("Transfer Function Quality - Per-band R²", fontsize=11, fontweight='bold')
                ax_tf.set_ylim(0, 1.05)
                ax_tf.legend(fontsize=9)
                canvas_tf.draw()

                result_lbl.config(
                    text=(f"Transfer function computed.  Mean band R²={mean_r2:.4f}  "
                          f"(min={np.min(r2_per_band):.4f}, max={np.max(r2_per_band):.4f})"))

                _tf_state.update({
                    'pls': pls, 'scaler_s': scaler_s, 'scaler_t': scaler_t,
                    'r2_per_band': np.array(r2_per_band),
                    'wl_nm_s': wl_nm_s, 'wl_nm_t': wl_nm_t,
                    'X_src': X_src, 'X_tgt': X_tgt,
                    'df_src': df_src, 'df_tgt': df_tgt,
                    'mean_r2': mean_r2, 'fig': fig_tf
                })
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=win)

        def save_tf():
            if not _tf_state:
                messagebox.showwarning("No results",
                    "Compute the transfer function first.", parent=win)
                return
            out = filedialog.asksaveasfilename(
                title="Save transfer function",
                defaultextension=".joblib",
                filetypes=[("Joblib files", "*.joblib")],
                parent=win)
            if out:
                save_transfer_function(
                    out,
                    _tf_state['pls'], _tf_state['scaler_s'], _tf_state['scaler_t'],
                    _tf_state['r2_per_band'].tolist(),
                    list(_tf_state['wl_nm_s']), list(_tf_state['wl_nm_t']))
                result_lbl.config(text=result_lbl.cget("text") + f"\n✔ Saved → {os.path.basename(out)}")

        def save_r2_report():
            if not _tf_state:
                messagebox.showwarning("No results",
                    "Compute the transfer function first.", parent=win)
                return
            out = filedialog.asksaveasfilename(
                title="Save R² report",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile="transfer_function_r2_report.xlsx",
                parent=win)
            if not out:
                return
            try:
                r2 = _tf_state['r2_per_band']
                wl  = _tf_state['wl_nm_t']
                df_r2 = pd.DataFrame({'Wavelength_nm': wl, 'R2_per_band': r2})
                summary = pd.DataFrame({
                    'Metric': ['Mean_R2', 'Min_R2', 'Max_R2', 'Std_R2',
                               'N_bands', 'PLS_components'],
                    'Value': [f"{r2.mean():.6f}", f"{r2.min():.6f}",
                              f"{r2.max():.6f}", f"{r2.std():.6f}",
                              len(r2), int(ncomp_var.get())]
                })
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    df_r2.to_excel(writer, sheet_name='Per_Band_R2', index=False)
                plot_out = out.replace('.xlsx', '_plot.png')
                _tf_state['fig'].savefig(plot_out, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Saved",
                    f"Report: {os.path.basename(out)}\nPlot: {os.path.basename(plot_out)}",
                    parent=win)
            except Exception as e:
                messagebox.showerror("Save Error", str(e), parent=win)

        ttk.Button(btn_row_tf, text="▶ Compute Transfer Function",
                   command=compute_tf).pack(side="left", padx=4)
        ttk.Button(btn_row_tf, text="💾 Save TF (.joblib)",
                   command=save_tf).pack(side="left", padx=4)
        ttk.Button(btn_row_tf, text="📊 Save R² Report",
                   command=save_r2_report).pack(side="left", padx=4)

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 : Apply Transfer Function
        # ══════════════════════════════════════════════════════════════════
        tab_apply = ttk.Frame(nb_harm, padding=10)
        nb_harm.add(tab_apply, text="  Apply Transfer Function  ")

        tf_file_var = tk.StringVar()
        input_file_var = tk.StringVar()

        frm_a = ttk.Frame(tab_apply)
        frm_a.pack(fill="x")

        ttk.Label(frm_a, text="Transfer function (.joblib):",
                  font=('Helvetica', 9, 'bold')).grid(
                      row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        tf_file_lbl = ttk.Label(frm_a, text="No file selected", foreground="grey")
        tf_file_lbl.grid(row=1, column=1, sticky="w")

        def browse_tf():
            p = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib")])
            if p:
                tf_file_var.set(p)
                tf_file_lbl.config(text=os.path.basename(p))

        ttk.Button(frm_a, text="Browse", command=browse_tf).grid(row=1, column=0, padx=4)

        ttk.Label(frm_a, text="Spectra to harmonize (Excel/CSV):",
                  font=('Helvetica', 9, 'bold')).grid(
                      row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))
        inp_file_lbl = ttk.Label(frm_a, text="No file selected", foreground="grey")
        inp_file_lbl.grid(row=3, column=1, sticky="w")

        def browse_input():
            p = filedialog.askopenfilename(
                filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
            if p:
                input_file_var.set(p)
                inp_file_lbl.config(text=os.path.basename(p))

        ttk.Button(frm_a, text="Browse", command=browse_input).grid(row=3, column=0, padx=4)

        apply_result_lbl = ttk.Label(tab_apply, text="", wraplength=700,
                                      font=('Helvetica', 9, 'italic'))
        apply_result_lbl.pack(anchor="w", pady=4)

        fig_ap = Figure(figsize=(6.5, 3.2), tight_layout=True)
        ax_ap  = fig_ap.add_subplot(111)
        canvas_ap = FigureCanvasTkAgg(fig_ap, tab_apply)
        canvas_ap.get_tk_widget().pack(fill="both", expand=True)

        _apply_state = {}

        def apply_tf():
            try:
                if not tf_file_var.get():
                    messagebox.showwarning("Missing",
                        "Please select a transfer function file.", parent=win)
                    return
                if not input_file_var.get():
                    messagebox.showwarning("Missing",
                        "Please select the spectra file to harmonize.", parent=win)
                    return
                tf_data = load_transfer_function(tf_file_var.get())
                pls_m   = tf_data['pls']
                sc_s    = tf_data['scaler_source']
                sc_t    = tf_data['scaler_target']
                wl_s    = np.array(tf_data['source_wavelengths'])
                wl_t    = np.array(tf_data['target_wavelengths'])

                df_in = _load_file(input_file_var.get())
                X_in, wl_in = _extract_spectra(df_in)
                _, wl_in_nm = detect_wavelength_unit(wl_in)

                # Interpolate input to source wavelength grid if needed
                if not np.allclose(wl_in_nm, wl_s, atol=2.0):
                    X_in = safe_interpolate_spectra(X_in, wl_in_nm, wl_s)

                X_scaled  = sc_s.transform(X_in)
                X_harm    = sc_t.inverse_transform(pls_m.predict(X_scaled))
                X_harm    = np.clip(X_harm, 0, None)  # reflectance ≥ 0

                # Plot mean spectra before/after.  X_in is on the source grid
                # (wl_s); map its mean onto the target axis (wl_t) for display.
                ax_ap.clear()
                orig_mean = np.interp(wl_t, wl_s, X_in.mean(axis=0))
                ax_ap.plot(wl_t, orig_mean,
                           color='steelblue', alpha=0.8, label='Original (mean)', linewidth=1.5)
                ax_ap.plot(wl_t, X_harm.mean(axis=0),
                           color='crimson', alpha=0.8, label='Harmonized (mean)', linewidth=1.5)
                ax_ap.set_xlabel("Wavelength (nm)", fontsize=10)
                ax_ap.set_ylabel("Reflectance", fontsize=10)
                ax_ap.set_title("Harmonized Spectra - Mean Comparison", fontsize=11, fontweight='bold')
                ax_ap.legend(fontsize=9)
                canvas_ap.draw()

                apply_result_lbl.config(
                    text=f"✔ Harmonized {X_harm.shape[0]} spectra → "
                         f"{X_harm.shape[1]} bands on target grid ({wl_t[0]:.0f}–{wl_t[-1]:.0f} nm).")
                _apply_state.update({
                    'X_harm': X_harm, 'wl_t': wl_t,
                    'df_in': df_in, 'fig': fig_ap
                })
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=win)

        def save_harmonized():
            if not _apply_state:
                messagebox.showwarning("No results", "Apply a transfer function first.", parent=win)
                return
            out = filedialog.asksaveasfilename(
                title="Save harmonized spectra",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
                initialfile="harmonized_spectra.xlsx",
                parent=win)
            if not out:
                return
            try:
                X_h  = _apply_state['X_harm']
                wl_t = _apply_state['wl_t']
                df_harm = pd.DataFrame(X_h, columns=[str(w) for w in wl_t])

                # Re-attach non-spectral columns (e.g. sample IDs, soil properties)
                df_orig = _apply_state['df_in']
                meta_cols = [c for c in df_orig.columns
                             if c not in [str(w) for w in wl_t] and
                             not _is_numeric_col(c, 0.3, 20000)]
                for col in meta_cols:
                    if len(df_orig) == len(df_harm):
                        df_harm.insert(0, col, df_orig[col].values)

                if out.lower().endswith('.csv'):
                    df_harm.to_csv(out, index=False)
                else:
                    df_harm.to_excel(out, index=False)

                plot_out = out.rsplit('.', 1)[0] + '_plot.png'
                _apply_state['fig'].savefig(plot_out, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Saved",
                    f"Harmonized data: {os.path.basename(out)}\nPlot: {os.path.basename(plot_out)}",
                    parent=win)
            except Exception as e:
                messagebox.showerror("Save Error", str(e), parent=win)

        btn_row_ap = ttk.Frame(tab_apply)
        btn_row_ap.pack(fill="x", pady=6)
        ttk.Button(btn_row_ap, text="▶ Apply Transfer Function",
                   command=apply_tf).pack(side="left", padx=4)
        ttk.Button(btn_row_ap, text="💾 Save Harmonized Spectra",
                   command=save_harmonized).pack(side="left", padx=4)
