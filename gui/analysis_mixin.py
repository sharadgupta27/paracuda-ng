"""
Model training, batch analysis and best-preprocessing search.

@author: Sharad Kumar Gupta
"""
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


class AnalysisMixin:

    def _maybe_tune_hyperparameters(self, model_name, base_params, X_train_scaled,
                                    y_train_scaled, n_cores):
        """If the Tune-Hyperparameters checkbox is on, run Optuna for this model.

        Returns (tuned_params_dict, best_rmse) where tuned_params_dict contains
        ONLY the optimized keys (empty when tuning is off, unavailable, or fails).
        Progress and results are echoed to the status log.
        """
        if not getattr(self, 'tune_hyperparams_var', None) or not self.tune_hyperparams_var.get():
            return {}, None
        try:
            n_trials = int(float(self.tune_trials_var.get()))
        except (ValueError, tk.TclError):
            n_trials = 30
        try:
            cv_folds = int(float(self.tune_cv_var.get()))
        except (ValueError, tk.TclError):
            cv_folds = 3
        n_trials = max(1, n_trials)
        cv_folds = max(2, cv_folds)

        self.append_status(
            f"  ⚙ Tuning {model_name} hyperparameters "
            f"(Optuna, {n_trials} trials, {cv_folds}-fold CV)...\n")

        # Report the search space Optuna will explore so the run is reproducible.
        n_samples, n_features = np.asarray(X_train_scaled).shape
        self.append_status(
            f"    Search space: {format_search_space(model_name, n_features, n_samples)}\n")

        def _progress(done, total):
            # Optuna's callback fires on the worker thread - marshal the label
            # update back onto the UI thread.
            self._run_on_ui(
                lambda: self.progress_label.config(text=f"Tuning {done}/{total}"))

        # Offload the (long, Tk-free) Optuna study so the GUI stays responsive.
        best_params, best_rmse, _study, msg = self._run_offloaded(
            tune_hyperparameters,
            model_name, base_params, X_train_scaled, y_train_scaled,
            n_trials=n_trials, n_cores=max(1, n_cores), cv_folds=cv_folds,
            progress_cb=_progress)

        if best_params:
            self.append_status(
                f"  ✔ Best hyperparameters [{format_params(best_params)}] "
                f"(inner-CV RMSE={best_rmse:.4f})\n")
        else:
            self.append_status(f"  ↳ Tuning skipped: {msg}\n")
        return best_params, best_rmse

    # ------------------------------------------------------------------
    # Find Best Preprocessing
    # ------------------------------------------------------------------
    PREPROCESSING_METHODS = [
        "No Preprocessing", "Smoothing", "Spectral Outlier Removal",
        "Continuum Removal", "Baseline Correction",
        "First Derivative", "Second Derivative", "Absorbance"
    ]

    def _report_resampling_target(self):
        """Print the resampling target (band count + method) to the status log so
        the user can see what was applied.  Shared by the single-run, batch and
        best-preprocessing paths; a no-op when resampling is off
        (``self.new_wavelengths is None``)."""
        # Note any excluded ranges (independent of whether resampling is on).
        excl = self._get_exclude_ranges()
        if excl:
            excl_txt = ", ".join(f"{int(lo)}-{int(hi)}" for lo, hi in excl)
            self.status_text.insert(tk.END, f"Excluded ranges: {excl_txt} nm.\n")

        if self.new_wavelengths is None:
            # Warn when binning was requested but silently skipped (grid too small).
            if self._binning_kwargs().get('apply_binning'):
                self.status_text.insert(
                    tk.END,
                    "Binning requested but skipped - the grid has too few bands "
                    "(binning only thins grids with > bin_size×3 bands).\n")
            self.status_text.see(tk.END)
            return
        sensor_sel = self.sensor_var.get()
        target = (f"{sensor_sel} sensor bands" if sensor_sel != "Custom"
                  else f"{self.spacing_var.get()} nm uniform grid")
        self.status_text.insert(
            tk.END,
            f"Resampling {len(self.filtered_wavelengths)} → "
            f"{len(self.new_wavelengths)} bands ({target}, "
            f"{self.resample_method}).\n")
        self.status_text.see(tk.END)

    def _verify_resampling_feasible(self):
        """Fail-fast check that the configured resampling can populate every
        target band from the current input grid.

        Resampling depends only on the wavelength grids, method and FWHMs - not
        on the spectral values - so a single dummy-row convolution surfaces a
        "grid too coarse for SRF" failure once, up front, instead of having it
        recur identically for every model × preprocessing combination in the
        search (or fail mid-run for a plain single/batch run).

        Returns ``True`` when resampling is off or feasible; otherwise prints a
        clear message (naming the offending bands), pops an error dialog and
        returns ``False`` so the caller can abort before any heavy work.
        """
        if self.new_wavelengths is None:
            return True
        try:
            probe = np.ones((1, len(self.filtered_wavelengths)), dtype=float)
            resample_spectra(probe, self.filtered_wavelengths, self.new_wavelengths,
                             method=self.resample_method, fwhms=self.new_fwhms,
                             srf_table=self.srf_table)
            return True
        except Exception as e:
            self.status_text.insert(
                tk.END,
                "\n✖ Resampling not feasible for this input grid - aborting.\n"
                f"{e}\n"
                f"\nThe '{self.resample_method}' method cannot fill every target "
                f"band from a {len(self.filtered_wavelengths)}-band input grid. "
                "Switch the resampling Method to 'Linear Interpolation', choose a "
                "coarser target sensor, or use a finer input grid.\n")
            self.status_text.see(tk.END)
            messagebox.showerror("Resampling not feasible", str(e))
            return False

    def run_best_preprocessing_search(self, property_name, models_to_run):
        """Evaluate every preprocessing method (× every model in models_to_run) for a
        single property and return the best (preprocessing, model) combination ranked by R².

        Returns:
            best_preprocess (str): Name of the best preprocessing method.
            best_model (str): Name of the best model for that preprocessing.
            all_results (list[dict]): Full comparison table sorted by Test R² descending.
        """
        self.status_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.status_text.insert(tk.END, "FIND BEST PREPROCESSING SEARCH\n")
        self.status_text.insert(tk.END, f"Property : {property_name}\n")
        self.status_text.insert(tk.END, f"Models   : {', '.join(models_to_run)}\n")
        self.status_text.insert(tk.END, "=" * 60 + "\n")
        self.status_text.see(tk.END)

        all_results = []
        n_combos = len(self.PREPROCESSING_METHODS) * len(models_to_run)
        combo_idx = 0

        for pp_method in self.PREPROCESSING_METHODS:
            pp_kwargs = self._get_default_preprocess_kwargs(pp_method)

            for model_name in models_to_run:
                combo_idx += 1
                progress_base = (combo_idx - 1) / n_combos * 80  # reserve last 20 % for reporting
                self.status_text.insert(
                    tk.END,
                    f"  [{combo_idx}/{n_combos}] {pp_method} + {model_name} ... "
                )
                self.status_text.see(tk.END)
                self.update()  # keep GUI responsive

                try:
                    result = self.run_single_model_analysis(
                        property_name, model_name,
                        progress_base, 80 / n_combos,
                        override_preprocess=pp_method,
                        override_preprocess_kwargs=pp_kwargs
                    )
                    if result:
                        r2   = result.get('test_r2',   float('-inf'))
                        rmse = result.get('test_rmse', float('inf'))
                        unreliable = result.get('overfitting_flag', False)
                        unreliable_tag = "  ⚠ Results Unreliable" if unreliable else ""
                        self.status_text.insert(
                            tk.END,
                            f"R²={r2:.4f}  RMSE={rmse:.4f}{unreliable_tag}\n"
                        )
                        all_results.append({
                            'Preprocessing': pp_method,
                            'Model':         model_name,
                            'Test_R2':       r2,
                            'Test_RMSE':     rmse,
                            'Test_MAE':      result.get('test_mae', float('inf')),
                            'unreliable':    unreliable,
                        })
                    else:
                        self.status_text.insert(tk.END, "failed\n")
                        all_results.append({
                            'Preprocessing': pp_method,
                            'Model':         model_name,
                            'Test_R2':       float('-inf'),
                            'Test_RMSE':     float('inf'),
                            'Test_MAE':      float('inf'),
                            'unreliable':    True,
                        })
                except Exception as exc:
                    self.status_text.insert(tk.END, f"error ({exc})\n")
                    all_results.append({
                        'Preprocessing': pp_method,
                        'Model':         model_name,
                        'Test_R2':       float('-inf'),
                        'Test_RMSE':     float('inf'),
                        'Test_MAE':      float('inf'),
                        'unreliable':    True,
                    })

                self.status_text.see(tk.END)

        if not all_results:
            self.status_text.insert(tk.END, "No successful results - aborting best-preprocessing search.\n")
            return None, None, []

        # Sort by R² descending, RMSE ascending as tiebreaker
        # Reliable results come before unreliable ones
        all_results.sort(key=lambda r: (r['unreliable'], -r['Test_R2'], r['Test_RMSE']))

        # Find the best reliable result; fall back to first entry if all are unreliable
        reliable_results = [r for r in all_results if not r['unreliable'] and r['Test_R2'] > float('-inf')]
        best = reliable_results[0] if reliable_results else all_results[0]
        best_preprocess = best['Preprocessing']
        best_model      = best['Model']

        # Print ranked table (skip entries with no valid R²)
        displayable = [r for r in all_results if r['Test_R2'] > float('-inf')]
        self.status_text.insert(tk.END, "\n" + "─" * 72 + "\n")
        self.status_text.insert(tk.END, "RANKING (by Test R²):\n")
        self.status_text.insert(tk.END,
            f"{'Rank':<5} {'Preprocessing':<28} {'Model':<24} {'R2':>7}  {'RMSE':>10}\n"
        )
        self.status_text.insert(tk.END, "─" * 72 + "\n")
        for rank, row in enumerate(displayable, 1):
            marker = " ★" if rank == 1 and not row['unreliable'] else ""
            unreliable_tag = "  ⚠ Results Unreliable" if row['unreliable'] else ""
            self.status_text.insert(
                tk.END,
                f"{rank:<5} {row['Preprocessing']:<28} {row['Model']:<24} "
                f"{row['Test_R2']:>7.4f}  {row['Test_RMSE']:>10.4f}{marker}{unreliable_tag}\n"
            )
        self.status_text.insert(tk.END, "─" * 72 + "\n")
        self.status_text.insert(
            tk.END,
            f"\n★ Best combo : {best_preprocess}  +  {best_model}\n"
            f"   Test R² = {best['Test_R2']:.4f} | RMSE = {best['Test_RMSE']:.4f}\n\n"
        )
        self.status_text.see(tk.END)

        # Auto-select the winning preprocessing + model in the GUI so a
        # subsequent (optionally tuned) single run uses the best combination.
        self.preprocess_var.set(best_preprocess)
        # Trigger the on_preprocess_change to refresh param widgets
        self.on_preprocess_change(None)
        if best_model in self.model_params:
            self.model_var.set(best_model)
            self.on_model_change(None)

        return best_preprocess, best_model, all_results

    def _export_best_combo_results(self, property_name, preprocess_name, model_name, result):
        """Export the winning preprocessing + model combination (after optional
        Optuna tuning) to Excel plus a PDF of its scatter and feature-importance
        plots.  The Excel includes the hyperparameter search space when tuned.
        """
        try:
            # One shared timestamp for this run's Excel, PDF and (later) the saved
            # model, so all three artefacts of the winning combo match.
            self._last_result_timestamp = make_timestamp()
            default_filename = generate_default_filename(
                self.input_filename, property_name, model_name,
                timestamp=self._last_result_timestamp)
            file_path = filedialog.asksaveasfilename(
                title="Save best preprocessing+model results",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=default_filename)
            if not file_path:
                self.append_status("  ↳ Best-combo export cancelled.\n")
                return

            # Map the run_single_model_analysis result into the shape
            # save_results_to_excel expects.
            results_data = {
                'y_test':          result['y_test'],
                'y_pred':          result['y_pred'],
                'test_r2':         result['test_r2'],
                'test_rmse':       result['test_rmse'],
                'test_mae':        result.get('test_mae'),
                'train_r2':        result['train_r2'],
                'train_rmse':      result['train_rmse'],
                'test_size':       float(self.test_size_var.get()),
                'selected_property': property_name,
                'preprocessing':   preprocess_name,
                'n_cores':         int(self.cores_var.get()),
                'model_type':      model_name,
                'filtered_wavelengths': result['wavelengths'],
                'correlations':    result['correlations'],
                'params':          result['params'],
                'tuned_params':    result.get('tuned_params'),
                'hyperparameter_search_space': result.get('hyperparameter_search_space'),
                'cv_results':      result.get('cv_results'),
                'confidence_intervals': result.get('confidence_intervals'),
            }
            save_results_to_excel(file_path, results_data)
            self.append_status(
                f"  ✔ Best-combo results saved: {os.path.basename(file_path)}\n")

            # Scatter + feature-importance figures for the winning combo.
            combo_label = f"{model_name} [{preprocess_name}]"
            fig_scatter, _ = create_scatter_plot(
                result['y_test'], result['y_pred'], combo_label, property_name,
                {'r2': result['test_r2'], 'rmse': result['test_rmse']},
                overfitting_flag=result.get('overfitting_flag', False))
            self.store_analysis_plot(combo_label, property_name, fig_scatter)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            create_feature_importance_plot(
                result.get('model'), model_name, result['wavelengths'],
                property_name, result.get('correlations'), ax_imp)

            pdf_path = os.path.splitext(file_path)[0] + "_plots.pdf"
            with PdfPages(pdf_path) as pdf:
                refl = getattr(self, '_reflectance_spectra_fig', None)
                if refl is not None:
                    pdf.savefig(refl, bbox_inches='tight')
                pdf.savefig(fig_scatter, bbox_inches='tight')
                pdf.savefig(fig_imp, bbox_inches='tight')
            plt.close(fig_imp)
            self.append_status(
                f"  ✔ Best-combo plots saved: {os.path.basename(pdf_path)}\n")
        except Exception as exc:
            self.append_status(f"  ⚠ Best-combo export failed: {exc}\n")

    def _print_batch_summary_table(self, property_name, comparison_df):
        """Print a ranked summary table (Train/Test R², RMSE, CV, overfitting) for a
        property's batch run to the status log, best-first.

        comparison_df is the DataFrame returned by suggest_best_model (already sorted
        by Score descending)."""
        def _fmt(v, width=8, prec=4):
            try:
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return f"{'-':>{width}}"
                return f"{float(v):>{width}.{prec}f}"
            except (TypeError, ValueError):
                return f"{'-':>{width}}"

        self.append_status("\n" + "─" * 88 + "\n")
        self.append_status(f"MODEL SUMMARY - {property_name} (ranked by Score):\n")
        self.append_status(
            f"{'Rank':<5} {'Model':<22} {'Train R2':>9} {'Test R2':>9} "
            f"{'Test RMSE':>10} {'CV R2':>8}  {'Overfit':<8}\n")
        self.append_status("─" * 88 + "\n")

        for rank, (_, row) in enumerate(comparison_df.iterrows(), 1):
            overfit = row.get('Overfitting_Flag', False)
            severity = row.get('Overfitting_Severity', 'none')
            overfit_tag = f"⚠ {severity}" if overfit else "ok"
            cv_val = row.get('CV_R2_Mean', None)
            self.append_status(
                f"{rank:<5} {str(row['Model'])[:22]:<22} "
                f"{_fmt(row.get('Train_R2'), 9)} {_fmt(row.get('Test_R2'), 9)} "
                f"{_fmt(row.get('Test_RMSE'), 10)} {_fmt(cv_val, 8)}  {overfit_tag:<8}\n")
        self.append_status("─" * 88 + "\n\n")

    def run_batch_analysis(self):
        """Public entry: run batch analysis with the window pinned and busy-guarded."""
        if getattr(self, '_batch_running', False):
            return
        self.set_busy_state(True)
        self._pin_geometry()
        try:
            self._run_batch_analysis_impl()
        finally:
            self._unpin_geometry()
            self.set_busy_state(False)

    def _promote_result_to_saveable(self, result, property_name, model_label=None):
        """Copy a ``run_single_model_analysis`` result into the ``self.*`` state
        that :meth:`save_model` reads, so a best/auto-selected model from a batch
        or best-preprocessing run can be saved exactly like a plain single run.

        ``model_label`` is the display key of the winning model (may carry a
        ``" (Tuned)"`` suffix); the base model type is derived from it (falling
        back to the result's ``model_type``) and, together with the winning
        hyper-parameters, is recorded as an explicit save override so the saved
        model reflects the best combination rather than the current GUI widgets.
        Enables the Save button when a single property is selected.
        """
        if not result:
            return
        self.compositional_model = None
        self.trained_model = result.get('model')
        self.pca_component = result.get('pca_component')
        self.scaler_X = result.get('scaler_X')
        self.scaler_y = result.get('scaler_y')
        self.model_preprocessing = result.get('preprocessing')
        self.model_preprocessing_kwargs = dict(result.get('preprocessing_kwargs') or {})
        self.selected_property = property_name

        base_type = result.get('model_type')
        if model_label:
            base_type = str(model_label).replace(' (Tuned)', '').strip() or base_type
        # Explicit overrides consumed by save_model (model_type + hyper-parameters
        # of the winning model), so the saved file matches the best combination.
        self._save_model_type_override = base_type
        params = result.get('params') or {}
        self._save_model_params_override = {k: str(v) for k, v in params.items()}

        if getattr(self, 'single_property_mode', False) and hasattr(self, 'save_model_btn'):
            self.save_model_btn.config(state='normal')

    def _clear_save_overrides(self):
        """Drop any best-model save overrides so a subsequent plain single run
        saves from the live GUI widgets, not a stale batch winner."""
        self._save_model_type_override = None
        self._save_model_params_override = None

    def _run_batch_analysis_impl(self):
        """Run analysis for multiple properties and/or models"""
        try:
            # Drop any stale best-model save overrides from a previous run; the
            # single-property paths below re-set them for the new winner.
            self._clear_save_overrides()
            # Check batch mode
            if self.batch_mode_var.get() and not self.selected_models:
                messagebox.showwarning("Warning", "Please select models for batch processing")
                return
            
            if not hasattr(self, 'selected_properties') or not self.selected_properties:
                messagebox.showwarning("Warning", "Please select soil properties first")
                return
            
            if self.df is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            # Validate test size
            try:
                test_size = float(self.test_size_var.get())
                if not 0 < test_size < 1:
                    messagebox.showerror("Error", "Test size must be between 0 and 1")
                    return
            except ValueError:
                messagebox.showerror("Error", "Invalid test size value")
                return
            
            # Filter wavelengths based on spectral configuration (CRITICAL FIX)
            try:
                self.resample_method = normalize_resample_method(
                    self.resample_method_var.get()
                    if self.resampling_var.get() == "Yes"
                    else "Linear Interpolation")
                self.filtered_wavelengths, self.new_wavelengths, self.new_fwhms = filter_wavelengths(
                    self.wavelengths, self.min_wave_var.get(), self.max_wave_var.get(),
                    self.resampling_var.get(), self.spacing_var.get(),
                    sensor=self.sensor_var.get(), resample_method=self.resample_method,
                    exclude_ranges=self._get_exclude_ranges(),
                    custom_fwhm=self.custom_fwhm_table,
                    **self._binning_kwargs()
                )
            except Exception as e:
                messagebox.showerror("Wavelength Filtering Error", str(e))
                return

            # Report the resampling target so the user can see what was applied
            # (batch and best-preprocessing runs otherwise resample silently).
            self._report_resampling_target()

            # Fail fast: verify the resampling can actually populate every band
            # before running any model, so an incompatible grid aborts once here
            # instead of failing identically for every combination below.
            if not self._verify_resampling_feasible():
                return

            # Determine which models to run
            if self.batch_mode_var.get():
                models_to_run = self.selected_models
            else:
                models_to_run = [self.model_var.get()]
            
            properties_to_run = self.selected_properties

            # ── Find Best Preprocessing (single property only) ──────────────
            if (getattr(self, 'find_best_preprocess_var', tk.BooleanVar()).get()
                    and len(properties_to_run) == 1):
                best_pp, best_mdl, _search_results = self.run_best_preprocessing_search(
                    properties_to_run[0], models_to_run
                )
                if best_pp is None:
                    return  # search failed - message already shown
                # Re-run the winning combo once more to obtain the full result
                # (predictions, correlations, trained model) needed for export.
                # Tune it when the Tune-Hyperparameters checkbox is on.
                do_tune = bool(best_mdl and getattr(self, 'tune_hyperparams_var', None)
                               and self.tune_hyperparams_var.get())
                if do_tune:
                    self.status_text.insert(
                        tk.END, f"\n⚙ Tuning best combo: {best_pp} + {best_mdl}\n")
                    self.status_text.see(tk.END)
                    self.update()
                best_result = self.run_single_model_analysis(
                    properties_to_run[0], best_mdl, 90, 5,
                    override_preprocess=best_pp,
                    override_preprocess_kwargs=self._get_default_preprocess_kwargs(best_pp),
                    tune=do_tune)
                # Export the winning combo's Excel + figures for the record.
                if best_result:
                    self._export_best_combo_results(
                        properties_to_run[0], best_pp, best_mdl, best_result)
                    # Promote the winning (preprocessing + model, tuned if enabled)
                    # combination to saveable state so it can be saved like a plain
                    # single run — including the winning preprocessing method and
                    # hyper-parameters, regardless of Auto-Select settings.
                    self._promote_result_to_saveable(
                        best_result, properties_to_run[0], model_label=best_mdl)
                    self.status_text.insert(
                        tk.END, f"\n💾 Best combo ready to save: {best_pp} + {best_mdl} "
                                f"for {properties_to_run[0]} (use the Save button).\n")
                    self.status_text.see(tk.END)
                else:
                    self.save_model_btn.config(state='disabled')
                # The search already ran all preprocessing × model combinations and
                # printed the ranking table. No need to re-run batch analysis.
                self.update_progress(100, "Best preprocessing search complete!")
                return
            # ────────────────────────────────────────────────────────────────

            self.status_text.insert(tk.END, "\nBATCH PROCESSING:\n")
            self.status_text.insert(tk.END, f"Properties: {', '.join(properties_to_run)}\n")
            self.status_text.insert(tk.END, f"Models: {', '.join(models_to_run)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
            
            # Store all results
            batch_results = {}
            all_results_for_comparison = {}
            
            total_combinations = len(properties_to_run) * len(models_to_run)
            current_combination = 0
            
            for property_name in properties_to_run:
                batch_results[property_name] = {}
                
                for model_name in models_to_run:
                    current_combination += 1
                    progress_base = (current_combination - 1) / total_combinations * 100
                    
                    self.status_text.insert(tk.END, f"\n[{current_combination}/{total_combinations}] Running {model_name} for {property_name}...\n")
                    self.status_text.see(tk.END)
                    self.update()
                    
                    try:
                        # Run single model analysis
                        result = self.run_single_model_analysis(
                            property_name, model_name, progress_base, 100 / total_combinations
                        )
                        
                        if result:
                            batch_results[property_name][model_name] = result
                            key = f"{property_name}_{model_name}"
                            all_results_for_comparison[key] = result
                            
                            # Store plot for visualization (for all batch processing scenarios)
                            fig, ax = create_scatter_plot(
                                result['y_test'], result['y_pred'], model_name, property_name,
                                {'r2': result['test_r2'], 'rmse': result['test_rmse']},
                                overfitting_flag=result.get('overfitting_flag', False)
                            )
                            self.store_analysis_plot(model_name, property_name, fig)
                            
                            # Display with confidence intervals if available
                            if result.get('confidence_intervals'):
                                ci = result['confidence_intervals']
                                self.status_text.insert(tk.END, 
                                    f"✓ {model_name}: R²={result['test_r2']:.4f} [{ci['r2_ci'][0]:.4f}, {ci['r2_ci'][1]:.4f}], " +
                                    f"RMSE={result['test_rmse']:.4f}, MAE={result['test_mae']:.4f}\n")
                            else:
                                self.status_text.insert(tk.END, 
                                    f"✓ {model_name}: R²={result['test_r2']:.4f}, RMSE={result['test_rmse']:.4f}, MAE={result['test_mae']:.4f}\n")
                        else:
                            self.status_text.insert(tk.END, f"✗ {model_name}: Failed\n")
                    
                    except Exception as e:
                        self.status_text.insert(tk.END, f"✗ {model_name}: Error - {str(e)}\n")
                        continue
                    
                    self.status_text.see(tk.END)
                    self.update()
            
            # Suggest best model for each property (only if auto-select is enabled)
            auto_select = self.auto_select_best_var.get()
            
            if auto_select:
                self.status_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.status_text.insert(tk.END, "BEST MODEL RECOMMENDATIONS:\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
            
            best_models_summary = []
            
            for property_name, models_results in batch_results.items():
                if models_results:
                    best_model, best_scores, comparison_df = suggest_best_model(models_results)

                    # Always print a ranked summary table for the property's models,
                    # regardless of the Auto-Select checkbox.
                    self._print_batch_summary_table(property_name, comparison_df)

                    if auto_select:
                        self.status_text.insert(tk.END, 
                            f"\n{property_name}: Best Model = {best_model}\n")
                        self.status_text.insert(tk.END, 
                            f"  R² = {best_scores['test_r2']:.3f}, RMSE = {best_scores['test_rmse']:.3f}, MAE = {best_scores['test_mae']:.3f}\n")
                        
                        best_models_summary.append({
                            'Property': property_name,
                            'Best_Model': best_model,
                            'Test_R2': best_scores['test_r2'],
                            'Test_RMSE': best_scores['test_rmse'],
                            'Test_MAE': best_scores['test_mae']
                        })
                    
                    # Store comparison df and best model for this property
                    batch_results[property_name]['_comparison'] = comparison_df
                    batch_results[property_name]['_best_model'] = best_model if auto_select else None
                    # The top-ranked model is recorded unconditionally (even when
                    # Auto-Select is off) so a single-property run can still promote
                    # and save it.  Upgraded to the tuned key below when tuning runs.
                    batch_results[property_name]['_winner_key'] = best_model

                    # ── Optuna tuning of the winning model (manual checkbox) ──
                    if (getattr(self, 'tune_hyperparams_var', None)
                            and self.tune_hyperparams_var.get()):
                        self.status_text.insert(
                            tk.END,
                            f"\n⚙ Tuning best model for {property_name}: {best_model}\n")
                        self.status_text.see(tk.END)
                        self.update()
                        tuned_result = self.run_single_model_analysis(
                            property_name, best_model, 88, 2, tune=True)
                        if tuned_result and tuned_result.get('tuned_params'):
                            tuned_key = f"{best_model} (Tuned)"
                            batch_results[property_name][tuned_key] = tuned_result
                            all_results_for_comparison[f"{property_name}_{tuned_key}"] = tuned_result
                            # Prefer the tuned model as the recommended one.
                            batch_results[property_name]['_best_model'] = tuned_key
                            batch_results[property_name]['_winner_key'] = tuned_key
                            fig, ax = create_scatter_plot(
                                tuned_result['y_test'], tuned_result['y_pred'],
                                tuned_key, property_name,
                                {'r2': tuned_result['test_r2'], 'rmse': tuned_result['test_rmse']},
                                overfitting_flag=tuned_result.get('overfitting_flag', False))
                            self.store_analysis_plot(tuned_key, property_name, fig)
                            self.status_text.insert(
                                tk.END,
                                f"  ✔ {tuned_key}: R²={tuned_result['test_r2']:.4f} "
                                f"(default {best_scores['test_r2']:.4f}), "
                                f"RMSE={tuned_result['test_rmse']:.4f} "
                                f"(default {best_scores['test_rmse']:.4f})\n")
                            self.status_text.see(tk.END)

            # Save results per property
            self.update_progress(90, "Saving results...")

            # Ask for output folder
            output_folder = filedialog.askdirectory(title="Select folder to save results")

            # One shared timestamp for this run's Excel, PDF and (later) the saved
            # model, so all three artefacts of the same analysis match.
            self._last_result_timestamp = make_timestamp()

            saved_files = []

            for property_name, models_results in batch_results.items():
                # Generate filename for this property
                property_filename = generate_property_filename(
                    self.input_filename, property_name,
                    timestamp=self._last_result_timestamp)
                file_path = os.path.join(output_folder, property_filename)
                
                # Determine which model(s) to save
                best_model = models_results.get('_best_model')
                model_results = {k: v for k, v in models_results.items() if not k.startswith('_')}
                
                # Save Excel - always save all models
                save_property_batch_results(
                    file_path, property_name, model_results, 
                    auto_select_best=False, best_model=best_model
                )
                
                saved_files.append(file_path)
                self.status_text.insert(tk.END, f"\nResults saved: {os.path.basename(file_path)}\n")
                
                # Generate PDF for this property
                pdf_filename = os.path.splitext(property_filename)[0] + "_plots.pdf"
                pdf_path = os.path.join(output_folder, pdf_filename)
                
                with PdfPages(pdf_path) as pdf:
                    # Lead with the reflectance-spectra overview when available.
                    refl = getattr(self, '_reflectance_spectra_fig', None)
                    if refl is not None:
                        pdf.savefig(refl, bbox_inches='tight')

                    # Always plot all models
                    models_to_plot = list(model_results.keys())
                    
                    # Create comparison plot for this property (only if batch mode with multiple models)
                    if '_comparison' in models_results and self.batch_mode_var.get():
                        fig = create_comparison_plots(models_results['_comparison'], property_name)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                    
                    # Create scatter plots and feature importance for each model
                    for model_name in models_to_plot:
                        if model_name not in model_results:
                            continue
                            
                        results = model_results[model_name]
                        
                        # Scatter plot
                        fig, ax = create_scatter_plot(
                            results['y_test'],
                            results['y_pred'],
                            model_name,
                            property_name,
                            {'r2': results['test_r2'], 'rmse': results['test_rmse']},
                            overfitting_flag=results.get('overfitting_flag', False)
                        )
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        
                        # Feature importance plot
                        if 'model' in results and 'wavelengths' in results:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            create_feature_importance_plot(
                                results['model'],
                                model_name,
                                results['wavelengths'],
                                property_name,
                                results.get('correlations'),
                                ax
                            )
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                
                self.status_text.insert(tk.END, f"Plots saved: {os.path.basename(pdf_path)}\n")
            
            self.update_progress(100, "Process complete!")
            self.status_text.insert(tk.END, "\n" + "="*50 + "\n")
            self.status_text.insert(tk.END, "BATCH ANALYSIS COMPLETE!\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
            
            # Save-model availability: a single-property batch (even with
            # Auto-Select Best / batch models / hyper-parameter tuning) still has
            # ONE winning model, so promote it to saveable state and enable Save —
            # it saves exactly like a plain single run (best model + its params +
            # preprocessing + resampling config).  Multi-property runs have no
            # single model to save, so the button stays disabled.
            if len(properties_to_run) == 1:
                prop = properties_to_run[0]
                winner_key = batch_results.get(prop, {}).get('_winner_key')
                winner_result = (batch_results.get(prop, {}).get(winner_key)
                                 if winner_key else None)
                if winner_result:
                    self._promote_result_to_saveable(winner_result, prop,
                                                     model_label=winner_key)
                    self.status_text.insert(
                        tk.END, f"\n💾 Best model ready to save: {winner_key} "
                                f"for {prop} (use the Save button).\n")
                    self.status_text.see(tk.END)
                else:
                    self.save_model_btn.config(state='disabled')
            else:
                self.save_model_btn.config(state='disabled')

            messagebox.showinfo("Success",
                f"Batch analysis complete!\n\nSaved {len(saved_files)} property result files to:\n{output_folder}")
        
        except Exception as e:
            error_msg = f"Batch analysis failed: {str(e)}\\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.status_text.insert(tk.END, f"Error: {str(e)}\\n")
            self.status_text.see(tk.END)

    def run_single_model_analysis(self, property_name, model_name, progress_base, progress_range,
                                   override_preprocess=None, override_preprocess_kwargs=None,
                                   tune=False):
        """Run analysis for a single property-model combination.

        Args:
            override_preprocess: When provided, use this preprocessing method name instead of
                the GUI selection.  Used by run_best_preprocessing_search().
            override_preprocess_kwargs: When provided, use these kwargs for preprocessing instead
                of reading from GUI vars.
        """
        try:
            test_size = float(self.test_size_var.get())
            n_cores = int(self.cores_var.get())
            
            # Prepare data
            X = self.df[self.wavelengths].values
            y = self.df[property_name].values
            
            # Apply wavelength filtering
            wavelength_indices = [i for i, w in enumerate(self.wavelengths)
                                if float(w) >= min(self.filtered_wavelengths)
                                and float(w) <= max(self.filtered_wavelengths)]
            X = X[:, wavelength_indices]

            # Handle missing/empty values (NaN/Inf) before resampling/preprocessing.
            try:
                X, y, _ = self._apply_missing_data_handling(X, y)
            except ValueError as miss_err:
                self.status_text.insert(
                    tk.END, f"  ↳ Skipped [{property_name}]: {miss_err}\n")
                self.status_text.see(tk.END)
                return None

            # Apply resampling if needed
            if self.new_wavelengths is not None:
                try:
                    X = resample_spectra(X, self.filtered_wavelengths, self.new_wavelengths,
                                         method=self.resample_method, fwhms=self.new_fwhms,
                                         srf_table=self.srf_table)
                    current_wavelengths = self.new_wavelengths
                except Exception as interp_err:
                    pp_label = override_preprocess if override_preprocess is not None else self.preprocess_var.get()
                    self.status_text.insert(
                        tk.END,
                        f"  ↳ Skipped [{pp_label}]: interpolation incompatible - {interp_err}\n"
                    )
                    self.status_text.see(tk.END)
                    return None
            else:
                current_wavelengths = self.filtered_wavelengths
            
            self.update_progress(progress_base + progress_range * 0.2, "Preprocessing...")

            # Resolve which preprocessing method / kwargs to use
            active_preprocess = override_preprocess if override_preprocess is not None else self.preprocess_var.get()
            if override_preprocess_kwargs is not None:
                preprocess_kwargs = override_preprocess_kwargs
            else:
                # Get preprocessing parameters from GUI vars
                preprocess_kwargs = {}
                if active_preprocess == "Smoothing":
                    preprocess_kwargs['window_length'] = int(self.smooth_window_var.get())
                    preprocess_kwargs['polyorder'] = int(self.smooth_poly_var.get())
                elif active_preprocess == "Spectral Outlier Removal":
                    preprocess_kwargs['outlier_method'] = self.outlier_method_var.get()
                    preprocess_kwargs['threshold'] = float(self.outlier_threshold_var.get())
                elif active_preprocess == "Baseline Correction":
                    if hasattr(self, 'baseline_method_var'):
                        preprocess_kwargs['baseline_method'] = self.baseline_method_var.get()
                    if hasattr(self, 'baseline_degree_var'):
                        preprocess_kwargs['degree'] = int(self.baseline_degree_var.get())
            
            # Preprocess spectra
            if active_preprocess == "Spectral Outlier Removal":
                X, outlier_mask = preprocess_spectra(X, active_preprocess, **preprocess_kwargs)
                # Apply mask to y as well to keep samples aligned
                y = y[outlier_mask]
            else:
                X = preprocess_spectra(X, active_preprocess, **preprocess_kwargs)

            # Guard: replace any NaN/Inf produced by preprocessing
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            scaler_X.fit(X_train)
            # Clip near-zero scale factors (same guard as in single-model training)
            scaler_X.scale_ = np.maximum(scaler_X.scale_, 1e-6)
            scaler_X.var_   = scaler_X.scale_ ** 2

            X_train_scaled = scaler_X.transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            self.update_progress(progress_base + progress_range * 0.4, f"Training {model_name}...")
            
            # Get parameters for the selected model
            params = {}
            if model_name == self.model_var.get():
                # Use current GUI parameters
                for param_name, param_var in self.param_vars.items():
                    param_value = param_var.get()
                    params[param_name] = parse_parameter_value(param_name, param_value, param_name)
            else:
                # Use default parameters
                for param_name, param_info in self.model_params[model_name]["params"].items():
                    param_value = param_info.get("default", "")
                    params[param_name] = parse_parameter_value(param_name, param_value, param_name)
            
            # Component optimization for PLS-R whenever the Optimize Components box
            # is checked.  In batch mode the single-model selector is disabled, so
            # this is NOT gated on ``model_name == self.model_var.get()`` — every
            # PLS-R model in the run is optimized when the box is on.
            if (self.optimize_components_var.get() and model_name in ["PLS-R"]):
                max_components = min(50, X_train.shape[1], X_train.shape[0] - 1)
                optimal_components, _, _, _ = self._run_offloaded(
                    optimize_components_parallel,
                    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                    model_name, max_components, scaler_y, n_cores
                )
                if optimal_components is not None:
                    params['n_components'] = optimal_components

            # Optional Optuna hyperparameter tuning (only for the explicitly
            # tuned best-model pass - the normal batch models keep defaults).
            tuned_params = {}
            search_space = None
            if tune:
                tuned_params, _rmse = self._maybe_tune_hyperparameters(
                    model_name, params, X_train_scaled, y_train_scaled, n_cores)
                params.update(tuned_params)
                n_s, n_f = X_train_scaled.shape
                search_space = search_space_table(model_name, n_f, n_s)

            # Create and train model
            trained_model = create_model(model_name, params, n_cores)
            # Guard PLS/PCA against a component count larger than the (possibly
            # resampled/binned) feature grid or the training-set size.
            trained_model = clamp_n_components(
                trained_model, X_train_scaled.shape[0], X_train_scaled.shape[1])

            # Handle PCA separately.  Model fitting is the heavy, Tk-free step, so
            # it is offloaded to keep the GUI responsive.
            pca_component = None
            if model_name == "PCA":
                pca = trained_model
                X_train_pca = self._run_offloaded(pca.fit_transform, X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                trained_model = LinearRegression()
                trained_model.fit(X_train_pca, y_train_scaled)
                y_pred_scaled = trained_model.predict(X_test_pca)
                y_train_pred_scaled = trained_model.predict(X_train_pca)
                pca_component = pca  # keep so a saved/best PCA model predicts correctly
            else:
                self._run_offloaded(trained_model.fit, X_train_scaled, y_train_scaled)
                y_pred_scaled = trained_model.predict(X_test_scaled)
                y_train_pred_scaled = trained_model.predict(X_train_scaled)
            
            self.update_progress(progress_base + progress_range * 0.7, "Calculating metrics...")
            
            # Convert predictions back to original scale
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            # Calculate 95% confidence intervals
            confidence_intervals = calculate_confidence_interval(y_test, y_pred, confidence=0.95)
            
            # Calculate model-specific feature importance/correlations
            correlations = None
            if model_name in ['PLS-R', 'Ridge', 'Lasso', 'Elastic Net']:
                # For linear models, use coefficients as feature importance
                if hasattr(trained_model, 'coef_'):
                    if len(trained_model.coef_.shape) > 1:
                        # For multi-output models like PLS-R, take first component
                        correlations = trained_model.coef_[0, :]
                    else:
                        correlations = trained_model.coef_
                    # Normalize to [-1, 1] range for visualization
                    max_abs = np.max(np.abs(correlations))
                    if max_abs > 0:
                        correlations = correlations / max_abs
            
            if correlations is None:
                # For non-linear models or as fallback, calculate correlation between 
                # each feature and the model's predictions on full dataset
                # This shows which wavelengths the model relies on most
                full_pred = trained_model.predict(scaler_X.transform(X))
                # Ensure full_pred is 1D
                if len(full_pred.shape) > 1:
                    full_pred = full_pred.ravel()
                full_pred_unscaled = scaler_y.inverse_transform(full_pred.reshape(-1, 1)).ravel()
                std_X = X.std(axis=0)
                valid_cols = (std_X >= 1e-10) & np.all(np.isfinite(X), axis=0)
                y_std = full_pred_unscaled.std()
                if y_std < 1e-10:
                    correlations = [0.0] * X.shape[1]
                else:
                    X_c = X - X.mean(axis=0)
                    y_c = full_pred_unscaled - full_pred_unscaled.mean()
                    # Divide only on valid (non-constant, finite) bands so
                    # constant bands don't trigger a divide-by-zero warning; they
                    # get 0 correlation, matching the mask applied next.
                    numer = (X_c * y_c[:, None]).mean(axis=0)
                    corr = np.divide(numer, std_X * y_std,
                                     out=np.zeros(X.shape[1]), where=valid_cols)
                    corr = np.where(valid_cols & np.isfinite(corr), corr, 0.0)
                    correlations = corr.tolist()

            # Cross-validation if requested
            cv_results = None
            if self.cv_strategy_var.get() != "None":
                cv_params = {}
                if self.cv_strategy_var.get() == "K-Fold":
                    cv_params['k_folds'] = int(self.k_folds_var.get())
                    cv_params['shuffle'] = self.shuffle_var.get().lower() == 'true'
                elif self.cv_strategy_var.get() == "Leave-One-Out":
                    # LOOCV requires no parameters
                    pass
                elif self.cv_strategy_var.get() == "Leave-P-Out":
                    cv_params['p_out'] = int(self.p_out_var.get())
                
                cv_rmse_scores, cv_r2_scores, cv_rmse_mean, cv_r2_mean, cv_rmse_std, cv_r2_std = self._run_offloaded(
                    perform_cross_validation,
                    X_train_scaled, y_train_scaled, trained_model,
                    self.cv_strategy_var.get(), cv_params, scaler_y, n_cores
                )

                cv_results = {
                    'strategy': self.cv_strategy_var.get(),
                    'cv_rmse_scores': cv_rmse_scores,
                    'cv_r2_scores': cv_r2_scores,
                    'rmse_mean': cv_rmse_mean,
                    'rmse_std': cv_rmse_std,
                    'r2_mean': cv_r2_mean,
                    'r2_std': cv_r2_std,
                    'parameters': cv_params
                }
            
            self.update_progress(progress_base + progress_range, "Complete")

            # Overfitting detection - check several measures, not just the R² gap
            assessment = assess_overfitting(
                train_r2, test_r2, train_rmse, test_rmse,
                cv_results['r2_mean'] if cv_results else None)
            overfitting_gap = assessment['gap']
            overfitting_flag = assessment['flag']
            overfitting_reasons = assessment['reasons']
            if overfitting_flag:
                self.append_status(
                    f"⚠ WARNING: Overfitting detected for {model_name} "
                    f"[{assessment['severity']}] - {'; '.join(overfitting_reasons)}\n")

            # Return results
            return {
                'y_test': y_test,
                'y_pred': y_pred,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'confidence_intervals': confidence_intervals,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'overfitting_gap': overfitting_gap,
                'overfitting_flag': overfitting_flag,
                'overfitting_severity': assessment['severity'],
                'overfitting_reasons': overfitting_reasons,
                'model': trained_model,
                'pca_component': pca_component,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'wavelengths': current_wavelengths,
                'correlations': correlations,
                'params': params,
                'tuned_params': tuned_params,
                'hyperparameter_search_space': search_space,
                'cv_results': cv_results,
                'preprocessing': active_preprocess,
                'preprocessing_kwargs': preprocess_kwargs,
                'model_type': model_name,
                'property_name': property_name
            }

        except Exception as e:
            self.status_text.insert(tk.END, f"Error in {model_name}: {str(e)}\\n")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Compositional (log-ratio) modelling - ALR / CLR / ILR
    # ─────────────────────────────────────────────────────────────────────────
    def _current_preprocess_kwargs(self):
        """Read the active preprocessing parameters from the GUI (mirrors the
        inline logic used by the single/batch trainers)."""
        pk = {}
        m = self.preprocess_var.get()
        if m == "Smoothing":
            pk['window_length'] = int(self.smooth_window_var.get())
            pk['polyorder'] = int(self.smooth_poly_var.get())
        elif m == "Spectral Outlier Removal":
            pk['outlier_method'] = self.outlier_method_var.get()
            pk['threshold'] = float(self.outlier_threshold_var.get())
        elif m == "Baseline Correction":
            if hasattr(self, 'baseline_method_var'):
                pk['baseline_method'] = self.baseline_method_var.get()
            if hasattr(self, 'baseline_degree_var'):
                pk['degree'] = int(self.baseline_degree_var.get())
        return pk

    def run_compositional_analysis(self, transform):
        """Train sum-constrained ("compositional") properties in log-ratio space.

        The selected parts (e.g. sand/silt/clay) are closed to 100 %, mapped to
        ALR/CLR/ILR coordinates, and one model (the current model type) is fit per
        coordinate on the SAME train/test split.  Test predictions are back-
        transformed to a closed composition (parts sum to 100 %), scored per part,
        written to Excel and saved as a compositional model bundle usable by
        Load Model + Predict Unknown."""
        try:
            parts = list(self.selected_properties)
            if self.df is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            missing_cols = [p for p in parts if p not in self.df.columns]
            if missing_cols:
                messagebox.showerror("Error", f"Property column(s) not found: {missing_cols}")
                return
            try:
                test_size = float(self.test_size_var.get())
                if not 0 < test_size < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Test size must be between 0 and 1")
                return
            n_cores = int(self.cores_var.get())
            model_type = self.model_var.get()
            total = 100.0

            self.update_progress(0, f"Compositional ({transform}) analysis…")
            self.status_text.insert(
                tk.END, f"\n=== Compositional modelling ({transform}) - parts: "
                        f"{', '.join(parts)} ===\n")
            self.status_text.see(tk.END)

            # Resampling grid (same as the normal path)
            self.resample_method = normalize_resample_method(
                self.resample_method_var.get()
                if self.resampling_var.get() == "Yes" else "Linear Interpolation")
            self.filtered_wavelengths, self.new_wavelengths, self.new_fwhms = filter_wavelengths(
                self.wavelengths, self.min_wave_var.get(), self.max_wave_var.get(),
                self.resampling_var.get(), self.spacing_var.get(),
                sensor=self.sensor_var.get(), resample_method=self.resample_method,
                exclude_ranges=self._get_exclude_ranges(),
                custom_fwhm=self.custom_fwhm_table, **self._binning_kwargs())
            self._report_resampling_target()

            # Build spectra X and composition Y
            X = self.df[self.wavelengths].values.astype(float)
            Y = self.df[parts].values.astype(float)
            wl_idx = [i for i, w in enumerate(self.wavelengths)
                      if min(self.filtered_wavelengths) <= float(w) <= max(self.filtered_wavelengths)]
            X = X[:, wl_idx]

            # Joint clean: keep rows finite in X and all parts, with a positive sum.
            good = (np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
                    & np.all(Y >= 0, axis=1) & (Y.sum(axis=1) > 0))
            X, Y = X[good], Y[good]
            if len(X) < 10:
                messagebox.showerror(
                    "Compositional Analysis",
                    "Too few complete samples with all parts present (need ≥ 10).")
                return

            # Resample onto the model grid, then preprocess (training config)
            if self.new_wavelengths is not None:
                X = resample_spectra(X, self.filtered_wavelengths, self.new_wavelengths,
                                     method=self.resample_method, fwhms=self.new_fwhms,
                                     srf_table=self.srf_table)
            pp_method = self.preprocess_var.get()
            pp_kwargs = self._current_preprocess_kwargs()
            if pp_method == "Spectral Outlier Removal":
                X, omask = preprocess_spectra(X, pp_method, **pp_kwargs)
                Y = Y[omask]
            else:
                X = preprocess_spectra(X, pp_method, **pp_kwargs)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Close composition and map to log-ratio coordinates
            comp = comp_close(Y, total)
            Z = comp_forward(comp, transform)
            k = Z.shape[1]

            # Single shared split (indices tracked to align composition rows)
            idx = np.arange(len(X))
            Xtr, Xte, Ztr, Zte, itr, ite = train_test_split(
                X, Z, idx, test_size=test_size, random_state=42)
            comp_te = comp[ite]

            scaler_X = StandardScaler().fit(Xtr)
            scaler_X.scale_ = np.maximum(scaler_X.scale_, 1e-6)
            scaler_X.var_ = scaler_X.scale_ ** 2
            Xtr_s, Xte_s = scaler_X.transform(Xtr), scaler_X.transform(Xte)

            params = {name: parse_parameter_value(name, var.get(), name)
                      for name, var in self.param_vars.items()}

            # One regression per log-ratio coordinate (shares X + split)
            coord_models, coord_scalers = [], []
            Zte_pred = np.zeros_like(Zte)
            for j in range(k):
                self.update_progress(10 + 70 * j / max(k, 1),
                                     f"Training coordinate {j + 1}/{k}…")
                sy = StandardScaler().fit(Ztr[:, j:j + 1])
                mdl = create_model(model_type, params, n_cores)
                mdl.fit(Xtr_s, sy.transform(Ztr[:, j:j + 1]).ravel())
                zp = np.asarray(mdl.predict(Xte_s)).ravel()
                Zte_pred[:, j] = sy.inverse_transform(zp.reshape(-1, 1)).ravel()
                coord_models.append(mdl)
                coord_scalers.append(sy)

            # Back-transform → closed composition (parts sum to 100 %)
            comp_pred = comp_inverse(Zte_pred, transform, total=total)

            # Per-part metrics on the original (percentage) scale
            self.update_progress(85, "Scoring composition…")
            metric_rows = []
            for d, p in enumerate(parts):
                r2 = r2_score(comp_te[:, d], comp_pred[:, d])
                rmse = float(np.sqrt(mean_squared_error(comp_te[:, d], comp_pred[:, d])))
                metric_rows.append({'Property': p, 'R2': r2, 'RMSE': rmse})
                self.status_text.insert(
                    tk.END, f"  {p:>10}: R² = {r2:.3f}   RMSE = {rmse:.3f}\n")
            self.status_text.insert(
                tk.END, f"  (all test predictions sum to {total:.0f}% by construction)\n")
            self.status_text.see(tk.END)

            # Compositional model bundle (usable by Load Model + Predict Unknown)
            bundle = {
                'compositional': True, 'transform': transform, 'parts': parts,
                'total': total, 'model_type': model_type,
                'coord_models': coord_models, 'coord_scalers_y': coord_scalers,
                'scaler_X': scaler_X, 'pca_component': None,
                'wavelengths': self.wavelengths,
                'filtered_wavelengths': self.filtered_wavelengths,
                'new_wavelengths': self.new_wavelengths, 'new_fwhms': self.new_fwhms,
                'resample_method': self.resample_method, 'srf_table': self.srf_table,
                'custom_fwhm_table': self.custom_fwhm_table,
                'exclude_ranges': self._get_exclude_ranges(),
                'preprocessing': pp_method, 'preprocessing_kwargs': pp_kwargs,
                'model_parameters': {name: var.get() for name, var in self.param_vars.items()},
            }
            self.compositional_model = bundle
            self.trained_model = None
            self.scaler_X = scaler_X

            # Save the report (Excel) + bundle (.joblib), sharing one timestamp
            self._last_result_timestamp = make_timestamp()
            safe_parts = "-".join(p.replace(' ', '') for p in parts)[:40]
            default_name = (f"{self.input_filename}_composition_{transform}_"
                            f"{safe_parts}_{self._last_result_timestamp}.xlsx")
            out_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")],
                initialfile=default_name)
            if out_path:
                pred_df = pd.DataFrame(
                    {f"{p}_obs": comp_te[:, d] for d, p in enumerate(parts)})
                for d, p in enumerate(parts):
                    pred_df[f"{p}_pred"] = comp_pred[:, d]
                pred_df["pred_sum"] = comp_pred.sum(axis=1)
                with pd.ExcelWriter(out_path, engine='openpyxl') as w:
                    pd.DataFrame(metric_rows).to_excel(w, sheet_name='Metrics', index=False)
                    pred_df.to_excel(w, sheet_name='Test Predictions', index=False)
                    pd.DataFrame({'Setting': ['Transform', 'Parts', 'Model',
                                              'Preprocessing', 'Test size', 'Total'],
                                  'Value': [transform, ', '.join(parts), model_type,
                                            pp_method, test_size, total]}
                                 ).to_excel(w, sheet_name='Config', index=False)
                bundle_path = os.path.splitext(out_path)[0] + "_model.joblib"
                joblib.dump(bundle, bundle_path)
                self.status_text.insert(
                    tk.END, f"Compositional results saved: {out_path}\n"
                            f"Compositional model saved: {bundle_path}\n" + "=" * 50 + "\n")
                self.status_text.see(tk.END)

            if hasattr(self, 'save_model_btn'):
                self.save_model_btn.config(state='normal')
            self.update_progress(100, "Compositional analysis complete")

        except Exception as e:
            messagebox.showerror(
                "Compositional Analysis Error",
                f"{str(e)}\n\n{traceback.format_exc()}")
            self.update_progress(0, "Compositional analysis failed")

    def predict_composition(self, X_scaled):
        """Predict a closed composition (parts sum to the bundle's total) from an
        already-scaled feature matrix, using the loaded compositional bundle."""
        b = self.compositional_model
        Z = np.column_stack([
            sy.inverse_transform(
                np.asarray(m.predict(X_scaled)).reshape(-1, 1)).ravel()
            for m, sy in zip(b['coord_models'], b['coord_scalers_y'])])
        return comp_inverse(Z, b['transform'], total=b.get('total', 100.0))

    def start_analysis(self):
        """Public entry: run single-model analysis with the window pinned and busy-guarded."""
        if getattr(self, '_batch_running', False):
            return
        self.set_busy_state(True)
        self._pin_geometry()
        try:
            self._start_analysis_impl()
        finally:
            self._unpin_geometry()
            self.set_busy_state(False)

    def _start_analysis_impl(self):
        try:
            # A plain single run saves from the live GUI widgets, so drop any
            # best-model override left by a previous batch/best-preprocessing run.
            self._clear_save_overrides()
            # Compositional (log-ratio) modelling takes priority: when a transform
            # is chosen and 2+ parts are selected, models are trained on ALR/CLR/
            # ILR coordinates and predictions are back-transformed to sum to 100 %.
            comp_var = getattr(self, 'comp_transform_var', None)
            if (comp_var is not None and comp_var.get() != "None"
                    and getattr(self, 'selected_properties', None)
                    and len(self.selected_properties) >= 2):
                self.run_compositional_analysis(comp_var.get())
                return

            # Check if batch mode is enabled or multiple properties selected
            if (self.batch_mode_var.get() and self.selected_models) or \
               (hasattr(self, 'selected_properties') and len(self.selected_properties) > 1):
                # Route to batch analysis (already inside the busy/pinned context)
                self._run_batch_analysis_impl()
                return

            # Find Best Preprocessing in single-run mode
            if getattr(self, 'find_best_preprocess_var', tk.BooleanVar()).get():
                if not hasattr(self, 'selected_property') or self.selected_property is None:
                    messagebox.showwarning("Warning", "Please select a soil property first")
                    return
                if self.df is None:
                    messagebox.showwarning("Warning", "Please load an Excel file first")
                    return
                try:
                    self.resample_method = (self.resample_method_var.get()
                                        if self.resampling_var.get() == "Yes"
                                        else "Interpolation")
                    self.filtered_wavelengths, self.new_wavelengths, self.new_fwhms = filter_wavelengths(
                        self.wavelengths, self.min_wave_var.get(), self.max_wave_var.get(),
                        self.resampling_var.get(), self.spacing_var.get(),
                        sensor=self.sensor_var.get(), resample_method=self.resample_method,
                        **self._binning_kwargs()
                    )
                except Exception as e:
                    messagebox.showerror("Wavelength Filtering Error", str(e))
                    return
                self._report_resampling_target()
                # Fail fast before the 88-combination search if the configured
                # resampling cannot populate every target band.
                if not self._verify_resampling_feasible():
                    return
                best_pp, _bm, _res = self.run_best_preprocessing_search(
                    self.selected_property, [self.model_var.get()]
                )
                if best_pp is None:
                    return
                # Preprocessing is now set; fall through to run a normal single analysis
            
            if not hasattr(self, 'selected_property') or self.selected_property is None:
                messagebox.showwarning("Warning", "Please select a soil property first")
                return
            
            if self.df is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            # Validate test size
            try:
                test_size = float(self.test_size_var.get())
                if not 0 < test_size < 1:
                    messagebox.showerror("Error", "Test size must be between 0 and 1")
                    return
            except ValueError:
                messagebox.showerror("Error", "Invalid test size value")
                return
            
            n_cores = int(self.cores_var.get())
            self.status_text.insert(tk.END, f"Using {n_cores} cores for model training\n")
            
            self.update_progress(0, "Starting analysis...")
            
            # Filter wavelengths based on spectral configuration
            try:
                self.resample_method = normalize_resample_method(
                    self.resample_method_var.get()
                    if self.resampling_var.get() == "Yes"
                    else "Linear Interpolation")
                self.filtered_wavelengths, self.new_wavelengths, self.new_fwhms = filter_wavelengths(
                    self.wavelengths, self.min_wave_var.get(), self.max_wave_var.get(),
                    self.resampling_var.get(), self.spacing_var.get(),
                    sensor=self.sensor_var.get(), resample_method=self.resample_method,
                    exclude_ranges=self._get_exclude_ranges(),
                    custom_fwhm=self.custom_fwhm_table,
                    **self._binning_kwargs()
                )
            except Exception as e:
                messagebox.showerror("Wavelength Filtering Error", str(e))
                return

            # Report the resampling target so the user can see what was applied.
            self._report_resampling_target()

            # Prepare data
            X = self.df[self.wavelengths].values
            y = self.df[self.selected_property].values
            
            # Apply wavelength filtering
            wavelength_indices = [i for i, w in enumerate(self.wavelengths)
                                if float(w) >= min(self.filtered_wavelengths)
                                and float(w) <= max(self.filtered_wavelengths)]
            X = X[:, wavelength_indices]

            # Handle missing/empty values (NaN/Inf) before any resampling or
            # preprocessing, using the user-selected strategy.
            try:
                X, y, _ = self._apply_missing_data_handling(X, y)
            except ValueError as miss_err:
                messagebox.showerror("Missing Data", str(miss_err))
                return

            # Apply resampling if needed. IMPORTANT: keep self.filtered_wavelengths
            # as the SOURCE grid (original wavelengths within range) - the model
            # save and the tabular/image prediction paths resample FROM it TO
            # self.new_wavelengths.  The wavelengths of the final feature matrix X
            # are tracked separately in `current_wavelengths` (used for the
            # correlogram / results, which must match `correlations` in length).
            if self.new_wavelengths is not None:
                try:
                    X = resample_spectra(X, self.filtered_wavelengths, self.new_wavelengths,
                                         method=self.resample_method, fwhms=self.new_fwhms,
                                         srf_table=self.srf_table)
                    current_wavelengths = self.new_wavelengths
                except Exception as e:
                    messagebox.showerror("Resampling Error", str(e))
                    return
            else:
                current_wavelengths = self.filtered_wavelengths

            self.update_progress(10, "Calculating statistics...")
            
            # Calculate data statistics
            data_stats = {
                'Input Data Statistics': calculate_statistics(y),
                'Spectral Statistics': {
                    'Mean Reflectance': calculate_statistics(np.mean(X, axis=1)),
                    'Min Reflectance': calculate_statistics(np.min(X, axis=1)),
                    'Max Reflectance': calculate_statistics(np.max(X, axis=1))
                }
            }
            
            self.update_progress(20, "Preprocessing spectra...")
            
            # Get preprocessing parameters
            preprocess_kwargs = {}
            if self.preprocess_var.get() == "Smoothing":
                preprocess_kwargs['window_length'] = int(self.smooth_window_var.get())
                preprocess_kwargs['polyorder'] = int(self.smooth_poly_var.get())
            elif self.preprocess_var.get() == "Spectral Outlier Removal":
                preprocess_kwargs['outlier_method'] = self.outlier_method_var.get()
                preprocess_kwargs['threshold'] = float(self.outlier_threshold_var.get())
            elif self.preprocess_var.get() == "Baseline Correction":
                if hasattr(self, 'baseline_method_var'):
                    preprocess_kwargs['baseline_method'] = self.baseline_method_var.get()
                if hasattr(self, 'baseline_degree_var'):
                    preprocess_kwargs['degree'] = int(self.baseline_degree_var.get())
            
            # Preprocess spectra
            if self.preprocess_var.get() == "Spectral Outlier Removal":
                X, outlier_mask = preprocess_spectra(X, self.preprocess_var.get(), **preprocess_kwargs)
                # Apply mask to y as well to keep samples aligned
                y = y[outlier_mask]
                self.status_text.insert(tk.END, f"Removed {(~outlier_mask).sum()} outlier samples\n")
            else:
                X = preprocess_spectra(X, self.preprocess_var.get(), **preprocess_kwargs)

            # Guard: replace any NaN/Inf produced by preprocessing
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Warn if preprocessing collapsed feature variance (would cause overflow in scaling)
            feat_std = X.std(axis=0)
            n_zero = int((feat_std < 1e-10).sum())
            if n_zero > 0:
                self.status_text.insert(tk.END,
                    f"⚠ Warning: {n_zero}/{X.shape[1]} wavelength band(s) have near-zero "
                    f"variance after preprocessing. Consider a different preprocessing method.\n")
                self.status_text.see(tk.END)

            self.update_progress(30, "Splitting data...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            self.update_progress(40, "Scaling data...")
            
            # Scale data
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

            self.scaler_X.fit(X_train)
            # Clip near-zero scale factors to prevent numerical explosion when bands
            # have near-constant values after preprocessing (e.g. baseline correction)
            self.scaler_X.scale_ = np.maximum(self.scaler_X.scale_, 1e-6)
            self.scaler_X.var_   = self.scaler_X.scale_ ** 2

            X_train_scaled = self.scaler_X.transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            self.update_progress(50, "Training model...")
            
            # Get parameters for the selected model
            model_type = self.model_var.get()
            params = {}
            for param_name, param_var in self.param_vars.items():
                param_value = param_var.get()
                params[param_name] = parse_parameter_value(param_name, param_value, param_name)
            
            # Component optimization for PCA and PLSR
            component_optimization_results = None
            if self.optimize_components_var.get() and model_type in ["PLS-R"]:
                max_components = min(50, X_train.shape[1], X_train.shape[0] - 1)
                optimal_components, components, rmse_values, r2_values = self._run_offloaded(
                    optimize_components_parallel,
                    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                    model_type, max_components, self.scaler_y, n_cores
                )
                
                if optimal_components is not None:
                    params['n_components'] = optimal_components
                    component_optimization_results = {
                        'Components': components,
                        'RMSE': rmse_values,
                        'R2_Score': r2_values,
                        'optimal_components': optimal_components
                    }
                    self.status_text.insert(tk.END, f"Optimal components: {optimal_components}\n")

            # Optional Optuna hyperparameter tuning of the selected model.
            tuned_params, _tuned_rmse = self._maybe_tune_hyperparameters(
                model_type, params, X_train_scaled, y_train_scaled, n_cores)
            params.update(tuned_params)
            # Reflect tuned values in the GUI widgets so the user sees them and
            # they are persisted when the model is saved.
            for _k, _v in tuned_params.items():
                if _k in self.param_vars:
                    self.param_vars[_k].set(str(_v))

            # Create and train model
            self.trained_model = create_model(model_type, params, n_cores)
            # Guard PLS/PCA against a component count larger than the (possibly
            # resampled/binned) feature grid or the training-set size.
            self.trained_model = clamp_n_components(
                self.trained_model, X_train_scaled.shape[0], X_train_scaled.shape[1])

            # Handle PCA separately.  Model fitting is the heavy, Tk-free step, so
            # it is offloaded to keep the GUI responsive.
            if model_type == "PCA":
                pca = self.trained_model
                X_train_pca = self._run_offloaded(pca.fit_transform, X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                self.trained_model = LinearRegression()
                self.trained_model.fit(X_train_pca, y_train_scaled)
                y_pred_scaled = self.trained_model.predict(X_test_pca)
                y_train_pred_scaled = self.trained_model.predict(X_train_pca)
                self.pca_component = pca  # store for prediction paths
            else:
                self._run_offloaded(self.trained_model.fit, X_train_scaled, y_train_scaled)
                y_pred_scaled = self.trained_model.predict(X_test_scaled)
                y_train_pred_scaled = self.trained_model.predict(X_train_scaled)
                self.pca_component = None  # clear any previous PCA component

            # Store preprocessing metadata so predict_unknown_csv always uses training config
            self.model_preprocessing = self.preprocess_var.get()
            self.model_preprocessing_kwargs = dict(preprocess_kwargs)

            self.update_progress(70, "Making predictions...")
            
            # Convert predictions back to original scale
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics for both train and test sets
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            
            # Cross-validation if requested
            cv_results = None
            if self.cv_strategy_var.get() != "None":
                self.update_progress(85, "Performing cross-validation...")
                
                # Prepare CV parameters
                cv_params = {}
                if self.cv_strategy_var.get() == "K-Fold":
                    cv_params['k_folds'] = int(self.k_folds_var.get())
                    cv_params['shuffle'] = self.shuffle_var.get().lower() == 'true'
                elif self.cv_strategy_var.get() == "Leave-One-Out":
                    # LOOCV requires no parameters
                    pass
                elif self.cv_strategy_var.get() == "Leave-P-Out":
                    cv_params['p_out'] = int(self.p_out_var.get())
                
                # Perform cross-validation (offloaded - heavy and Tk-free)
                cv_rmse_scores, cv_r2_scores, cv_rmse_mean, cv_r2_mean, cv_rmse_std, cv_r2_std = self._run_offloaded(
                    perform_cross_validation,
                    X_train_scaled, y_train_scaled, self.trained_model,
                    self.cv_strategy_var.get(), cv_params, self.scaler_y, n_cores
                )
                
                cv_results = {
                    'strategy': self.cv_strategy_var.get(),
                    'cv_rmse_scores': cv_rmse_scores,
                    'cv_r2_scores': cv_r2_scores,
                    'rmse_mean': cv_rmse_mean,
                    'rmse_std': cv_rmse_std,
                    'r2_mean': cv_r2_mean,
                    'r2_std': cv_r2_std,
                    'parameters': cv_params
                }
            
            self.update_progress(90, "Calculating correlations and confidence intervals...")
            
            # Calculate model-specific feature importance/correlations
            correlations = None
            if model_type in ['PLS-R', 'Ridge', 'Lasso', 'Elastic Net']:
                # For linear models, use coefficients as feature importance
                if hasattr(self.trained_model, 'coef_'):
                    if len(self.trained_model.coef_.shape) > 1:
                        # For multi-output models like PLS-R, take first component
                        correlations = self.trained_model.coef_[0, :]
                    else:
                        correlations = self.trained_model.coef_
                    # Normalize to [-1, 1] range for visualization
                    max_abs = np.max(np.abs(correlations))
                    if max_abs > 0:
                        correlations = correlations / max_abs
            
            if correlations is None:
                # For non-linear models or as fallback, calculate correlation between 
                # each feature and the model's predictions on full dataset
                full_pred = self.trained_model.predict(self.scaler_X.transform(X))
                # Ensure full_pred is 1D
                if len(full_pred.shape) > 1:
                    full_pred = full_pred.ravel()
                full_pred_unscaled = self.scaler_y.inverse_transform(full_pred.reshape(-1, 1)).ravel()
                std_X = X.std(axis=0)
                valid_cols = (std_X >= 1e-10) & np.all(np.isfinite(X), axis=0)
                y_std = full_pred_unscaled.std()
                if y_std < 1e-10:
                    correlations = [0.0] * X.shape[1]
                else:
                    X_c = X - X.mean(axis=0)
                    y_c = full_pred_unscaled - full_pred_unscaled.mean()
                    # Divide only on valid (non-constant, finite) bands so
                    # constant bands don't trigger a divide-by-zero warning; they
                    # get 0 correlation, matching the mask applied next.
                    numer = (X_c * y_c[:, None]).mean(axis=0)
                    corr = np.divide(numer, std_X * y_std,
                                     out=np.zeros(X.shape[1]), where=valid_cols)
                    corr = np.where(valid_cols & np.isfinite(corr), corr, 0.0)
                    correlations = corr.tolist()

            # Calculate MAE
            test_mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate 95% confidence intervals
            confidence_intervals = calculate_confidence_interval(y_test, y_pred, confidence=0.95)
            
            # Ask user for output filename.  Stamp the results, PDF and (later)
            # the saved model with ONE shared timestamp for this analysis run.
            self._last_result_timestamp = make_timestamp()
            default_filename = generate_default_filename(
                self.input_filename, self.selected_property, model_type,
                timestamp=self._last_result_timestamp)

            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=default_filename
            )
            
            if not file_path:
                return
            
            self.output_filename = file_path
            
            # Prepare results data
            results_data = {
                'y_test': y_test,
                'y_pred': y_pred,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'confidence_intervals': confidence_intervals,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'test_size': test_size,
                'selected_property': self.selected_property,
                'preprocessing': self.preprocess_var.get(),
                'n_cores': n_cores,
                'model_type': model_type,
                # Wavelengths of the final feature matrix X (resampled grid when
                # resampling is on) - must align 1:1 with `correlations`.
                'filtered_wavelengths': current_wavelengths,
                'correlations': correlations,
                'params': params,
                'tuned_params': tuned_params,
                'cv_results': cv_results,
                'component_optimization_results': component_optimization_results,
                'export_stats': self.export_stats_var.get(),
                'data_stats': data_stats if self.export_stats_var.get() else None,
                'model': self.trained_model,  # Include trained model for feature importance
                'wavelengths_used': current_wavelengths
            }
            
            # Save results to Excel
            save_results_to_excel(self.output_filename, results_data)
            
            # Overfitting detection - check several measures, not just the R² gap
            overfit = assess_overfitting(
                train_r2, test_r2, train_rmse, test_rmse,
                cv_results['r2_mean'] if cv_results else None)

            # Create and store scatter plot for visualization
            fig, ax = create_scatter_plot(
                y_test, y_pred, model_type, self.selected_property,
                {'r2': test_r2, 'rmse': test_rmse},
                overfitting_flag=overfit['flag']
            )
            self.store_analysis_plot(model_type, self.selected_property, fig)

            # Export the figures to a PDF next to the Excel results so a plain
            # single-model run produces the same artifacts as batch / best-combo
            # runs: reflectance spectra (if available) → scatter → feature importance.
            try:
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                create_feature_importance_plot(
                    self.trained_model, model_type, current_wavelengths,
                    self.selected_property, correlations, ax_imp)
                pdf_path = os.path.splitext(self.output_filename)[0] + "_plots.pdf"
                with PdfPages(pdf_path) as pdf:
                    refl = getattr(self, '_reflectance_spectra_fig', None)
                    if refl is not None:
                        pdf.savefig(refl, bbox_inches='tight')
                    pdf.savefig(fig, bbox_inches='tight')
                    pdf.savefig(fig_imp, bbox_inches='tight')
                plt.close(fig_imp)
                self.status_text.insert(tk.END, f"Plots saved to {pdf_path}\n")
            except Exception as pdf_exc:
                self.status_text.insert(
                    tk.END, f"  ⚠ PDF plot export failed: {pdf_exc}\n")

            # Display results with confidence intervals
            if confidence_intervals:
                r2_ci = confidence_intervals['r2_ci']
                rmse_ci = confidence_intervals['rmse_ci']
                mae_ci = confidence_intervals['mae_ci']
                
                self.status_text.insert(tk.END, 
                    f"Test R² = {test_r2:.4f} [95% CI: {r2_ci[0]:.4f}, {r2_ci[1]:.4f}]\n")
                self.status_text.insert(tk.END, 
                    f"Test RMSE = {test_rmse:.4f} [95% CI: {rmse_ci[0]:.4f}, {rmse_ci[1]:.4f}]\n")
                self.status_text.insert(tk.END, 
                    f"Test MAE = {test_mae:.4f} [95% CI: {mae_ci[0]:.4f}, {mae_ci[1]:.4f}]\n")
            else:
                self.status_text.insert(tk.END, f"Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:.4f}\n")
                self.status_text.insert(tk.END, f"Test MAE = {test_mae:.4f}\n")
            
            self.status_text.insert(tk.END, f"Train R² = {train_r2:.4f}, Train RMSE = {train_rmse:.4f}\n")

            # Overfitting detection - report only when multiple measures agree
            if overfit['flag']:
                self.status_text.insert(tk.END,
                    f"⚠ WARNING: Overfitting detected! [{overfit['severity']}] - "
                    f"{'; '.join(overfit['reasons'])}. "
                    f"Consider regularization, more data, or fewer features.\n")
            
            if cv_results:
                self.status_text.insert(tk.END, f"CV R² = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}\n")
                self.status_text.insert(tk.END, f"CV RMSE = {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}\n")
            
            self.status_text.insert(tk.END, f"Results saved to {self.output_filename}\n")
            
            self.update_progress(100, f"Analysis complete for {model_type}!")
            
            # Enable save model button only if single property is selected
            if hasattr(self, 'single_property_mode') and self.single_property_mode:
                self.save_model_btn.config(state='normal')
            else:
                self.save_model_btn.config(state='disabled')
            
            # Enable predict button if image is loaded and apply on image is checked
            if self.apply_models_var.get() and hasattr(self, 'image_data') and self.image_data is not None:
                self.predict_image_btn.config(state='normal')
            
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.status_text.insert(tk.END, f"Error: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
