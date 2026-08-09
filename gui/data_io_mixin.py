"""
Loading/saving Excel, models, images, predictions and missing-data handling.

@author: Sharad Kumar Gupta
"""
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


class DataIOMixin:

    # ─────────────────────────────────────────────────────────────────────────
    # Reading a data file
    # ─────────────────────────────────────────────────────────────────────────
    def _load_dataframe_with_progress(self, file_path):
        """Read a CSV/Excel file into a DataFrame, driving the progress bar.

        Uses the shared :func:`utils.data_converter.load_file`, which picks the
        fastest reader available (python-calamine, else a streaming .xlsx reader
        for plain grids, else pandas) and reports how far it has got.  A large
        spreadsheet takes roughly half as long as a plain ``pd.read_excel`` and,
        more usefully, now says so while it works.

        The read runs on a worker thread, which must never touch Tk: it only
        writes the latest ``(fraction, message)`` into ``latest``, and the poll
        callback - which ``_run_offloaded`` invokes on the main thread - is what
        moves the widgets.
        """
        # load_file only claims the extensions it knows; anything else keeps the
        # previous behaviour of handing the file to pandas as a spreadsheet.
        known = ('.csv',) + _data_converter._EXCEL_EXTS if _HAS_CONVERTER else ()
        if not file_path.lower().endswith(known):
            reader = (pd.read_csv if file_path.lower().endswith('.csv')
                      else pd.read_excel)
            return self._run_offloaded(reader, file_path)

        latest = {'state': (None, 'Opening %s…'
                            % os.path.basename(file_path)), 'shown': None}

        def report(fraction, message):
            """Worker thread - no Tk calls here."""
            latest['state'] = (fraction, message)

        def poll():
            """Main thread - safe to touch widgets."""
            state = latest['state']
            if state is latest['shown']:
                return
            latest['shown'] = state
            fraction, message = state
            if fraction is None:
                # No percentage available (a spreadsheet mid-parse): leave the
                # bar where it is rather than jumping it back to zero.
                self.progress_label.config(text=message or "Loading…")
            else:
                self.progress_var.set(max(0.0, min(1.0, fraction)) * 100.0)
                self.progress_label.config(text=message or "Loading…")

        self.progress_var.set(0)
        self.progress_label.config(text="Loading…")
        return self._run_offloaded(
            _data_converter.load_file, file_path, progress=report, _on_poll=poll)

    def _reset_load_progress(self):
        """Put the progress bar back to its idle 0% state after a load."""
        try:
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
        except Exception:  # noqa: BLE001 - cosmetic only
            pass

    def load_excel(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
            
            if file_path:
                # Reset GUI
                self.reset_gui()

                # Reading a large workbook can take seconds - offload it and pin the
                # window so the GUI stays responsive and does not auto-resize.
                self.set_busy_state(True)
                self._pin_geometry()
                try:
                    self.df = self._load_dataframe_with_progress(file_path)
                finally:
                    self._reset_load_progress()
                    self._unpin_geometry()
                    self.set_busy_state(False)

                # Store input filename
                self.input_filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # Identify wavelength columns (numeric column names).
                # First pass: collect every purely-numeric column name.
                numeric_cols = [col for col in self.df.columns
                                if str(col).replace('.', '', 1).replace('-', '', 1).isdigit()]

                if numeric_cols:
                    raw_wl = [float(c) for c in numeric_cols]
                    # Auto-detect nm vs μm
                    unit, wl_nm = detect_wavelength_unit(raw_wl)
                    self.wavelength_unit = unit

                    if unit == "μm":
                        # Rename columns to nm for internal consistency
                        rename_map = {col: str(w_nm) for col, w_nm in zip(numeric_cols, wl_nm)}
                        self.df.rename(columns=rename_map, inplace=True)
                        numeric_cols = list(rename_map.values())

                    # Filter to sensible spectral range 350–14500 nm
                    self.wavelengths = [col for col in numeric_cols
                                        if 350 <= float(col) <= 14500]
                else:
                    self.wavelengths = []
                    self.wavelength_unit = "nm"

                # Auto-detect spectral domain
                if self.wavelengths:
                    wl_floats = [float(w) for w in self.wavelengths]
                    self.spectral_domain = infer_spectral_domain(wl_floats)
                    # Sync domain selector to detected domain
                    domain_map = {"VSWIR": "VSWIR (350–2500 nm)",
                                  "LWIR": "LWIR (7500–14000 nm)",
                                  "VSWIR+LWIR": "VSWIR+LWIR (350–14000 nm)"}
                    self.spectral_domain_var.set(domain_map.get(self.spectral_domain, "VSWIR (350–2500 nm)"))

                # Update wavelength-unit display label
                if hasattr(self, 'wl_unit_label'):
                    self.wl_unit_label.config(text=self.wavelength_unit)
                if hasattr(self, '_min_wave_lbl'):
                    self._min_wave_lbl.config(
                        text=f"Min Wave ({self.wavelength_unit}):", width=15)
                    self._max_wave_lbl.config(
                        text=f"Max Wave ({self.wavelength_unit}):", width=12)
                if hasattr(self, 'lwir_note_label'):
                    if self.spectral_domain in ("LWIR", "VSWIR+LWIR"):
                        self.lwir_note_label.pack(anchor="w", pady=(2, 0))
                    else:
                        self.lwir_note_label.pack_forget()
                
                # Set default min/max wavelengths
                self.min_wave_var.set(min(float(w) for w in self.wavelengths))
                self.max_wave_var.set(max(float(w) for w in self.wavelengths))
                
                # Identify soil property columns
                self.soil_properties = [col for col in self.df.columns 
                                      if not str(col).replace('.', '').isdigit()]
                
                # Remove 'Names' if it appears in soil properties
                self.soil_properties = [prop for prop in self.soil_properties if prop != 'Names']

                # Data is loaded, so the distribution view has something to show.
                if hasattr(self, 'distribution_btn'):
                    self.distribution_btn.config(state='normal')

                # Coerce the spectral (wavelength) columns to numeric so that
                # blank cells / stray text become proper NaN and are handled by
                # the missing-data strategy rather than silently breaking training.
                if self.wavelengths:
                    self.df[self.wavelengths] = self.df[self.wavelengths].apply(
                        pd.to_numeric, errors='coerce')

                # Detect missing / empty cells and report to the user.
                self.missing_data_summary = analyze_missing_data(
                    self.df, self.wavelengths, self.soil_properties)

                # Set default number of components
                for model_type in ["PLS-R"]:
                    if model_type in self.model_params:
                        self.model_params[model_type]["params"]["n_components"]["default"] = str(len(self.wavelengths))
                    
                # Update default number of components
                if "n_components" in self.param_vars:
                    self.param_vars["n_components"].set(str(len(self.wavelengths)))
                
                self.status_text.insert(tk.END, f"Loaded file: {self.input_filename}\n")
                self.status_text.insert(tk.END, f"Found {len(self.wavelengths)} wavelengths and {len(self.soil_properties)} properties\n")

                # Report missing / empty cells and update the sidebar hint.
                self._report_missing_data(self.missing_data_summary)

                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
                # Display data in Data View tab
                self.data_display_text.config(state='normal')
                self.data_display_text.delete('1.0', tk.END)
                
                # Show first 100 rows, first 20 columns only (for speed)
                MAX_DISPLAY_COLS = 20
                display_df = self.df.head(100).iloc[:, :MAX_DISPLAY_COLS]
                total_cols = len(self.df.columns)
                col_note = (f" (showing first {MAX_DISPLAY_COLS} of {total_cols} columns)"
                            if total_cols > MAX_DISPLAY_COLS else "")
                # Wavelength column names are strings, so min()/max() on them would
                # compare lexicographically ("1000" < "999"). Compare numerically.
                if self.wavelengths:
                    wl_floats = [float(w) for w in self.wavelengths]
                    wl_min, wl_max = min(wl_floats), max(wl_floats)
                    wl_range = f"{wl_min:g} - {wl_max:g} nm"
                else:
                    wl_range = "n/a"
                self.data_display_text.insert('1.0',
                    f"Dataset: {self.input_filename}\n"
                    f"Samples: {len(self.df)}\n"
                    f"Wavelengths: {len(self.wavelengths)} ({wl_range}) "
                    f"[detected unit: {self.wavelength_unit}]\n"
                    f"Spectral domain: {self.spectral_domain}\n"
                    f"Properties: {', '.join(self.soil_properties)}\n"
                    f"{self._format_missing_data_line(self.missing_data_summary)}\n"
                    f"First 100 rows{col_note}:\n{'='*80}\n{display_df.to_string()}\n")
                self.data_display_text.config(state='disabled')

                # Build the reflectance-spectra preview (Visualization tab) now
                # that wavelengths/samples are known.
                self._build_reflectance_spectra_plot()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.status_text.insert(tk.END, f"Error loading file: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
    
    def _get_spectra_selection(self, n_rows):
        """Resolve the Visualization spectra controls into (n_samples, row_pool, note).

        ``row_pool`` is the set of row indices the random draw may pick from:
        ``None`` (every sample) in "Random" mode, or just the user's 1-based
        From/To sample range in "Within sample range" mode.  Bad or empty entries
        fall back to the full dataset rather than raising.
        """
        try:
            n_samples = int(float(self.spectra_count_var.get()))
        except (AttributeError, ValueError, tk.TclError):
            n_samples = 10
        n_samples = max(1, min(n_samples, n_rows))

        mode_var = getattr(self, 'spectra_mode_var', None)
        if mode_var is None or mode_var.get() != "Range":
            return n_samples, None, f"{n_samples} random of {n_rows} samples"

        def _read(var, default):
            try:
                return int(float(var.get()))
            except (ValueError, tk.TclError):
                return default

        lo = min(max(_read(self.spectra_from_var, 1), 1), n_rows)
        hi = min(max(_read(self.spectra_to_var, n_rows), 1), n_rows)
        if lo > hi:
            lo, hi = hi, lo

        pool = np.arange(lo - 1, hi)  # 1-based inclusive -> 0-based indices
        n_samples = min(n_samples, pool.size)
        note = f"{n_samples} of samples {lo}-{hi}"
        return n_samples, pool, note

    def refresh_reflectance_plot(self):
        """Redraw the Reflectance Spectra figure from the Visualization controls.

        Each press re-seeds the RNG, so "Random" genuinely draws a different
        subset every time.  Purely a viewing action - nothing here touches the
        data used for modelling.
        """
        if self.df is None or not self.wavelengths:
            messagebox.showinfo(
                "Reflectance Spectra",
                "Load a dataset with spectral columns first.")
            return
        # Fresh seed so a repeated "Random" press shows a new subset. The cached
        # figure is the same object the PDF export reuses, so the two stay in sync.
        self._spectra_seed = int(np.random.SeedSequence().generate_state(1)[0])
        self._build_reflectance_spectra_plot()

        key = "Reflectance Spectra"
        if key in getattr(self, 'analysis_plots', {}):
            self.viz_model_var.set(key)
            self.display_selected_plot()

    def _build_reflectance_spectra_plot(self):
        """Build a publication-quality reflectance-spectra plot, cache it on
        ``self._reflectance_spectra_fig`` (so PDF exports reuse the exact same
        figure) and show it in the Visualization tab.

        How many spectra are drawn - and whether they are picked from the whole
        dataset or from a specific sample range - comes from the Visualization
        tab's spectra controls (defaults: 10, random).

        Uses the raw loaded reflectance (before any preprocessing/resampling).
        A no-op when there is no spectral data.  Never raises into the caller.
        """
        self._reflectance_spectra_fig = None
        try:
            if self.df is None or not self.wavelengths:
                return
            X = self.df[self.wavelengths].apply(
                pd.to_numeric, errors='coerce').values.astype(float)
            # Drop all-NaN rows so blank samples don't produce empty lines.
            valid = ~np.all(np.isnan(X), axis=1)
            X = X[valid]
            if X.shape[0] == 0:
                return
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            names = None
            if 'Names' in self.df.columns:
                names = list(self.df['Names'].values[valid])

            n_rows = X.shape[0]
            n_samples, row_pool, note = self._get_spectra_selection(n_rows)

            # Keep the range controls honest about how many samples actually exist.
            if hasattr(self, 'spectra_range_hint'):
                self.spectra_range_hint.config(text=f"(1–{n_rows} available)")
            if getattr(self, 'spectra_to_var', None) is not None and not self.spectra_to_var.get():
                self.spectra_to_var.set(str(n_rows))

            fig = create_reflectance_spectra_plot(
                X, self.wavelengths, sample_names=names, n_samples=n_samples,
                wavelength_unit=self.wavelength_unit,
                seed=getattr(self, '_spectra_seed', 42),
                row_pool=row_pool,
                title=f"Reflectance Spectra - {self.input_filename}\n{note}")
            self._reflectance_spectra_fig = fig
            self.store_analysis_plot("Reflectance Spectra", "Dataset", fig,
                                     item_type="spectra")
        except Exception as e:
            self.status_text.insert(
                tk.END, f"Warning: could not build reflectance spectra plot: {e}\n")
            self.status_text.see(tk.END)

    def check_excel(self):
        try:
            if self.df is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            # Wavelength column names are strings - compare numerically, not
            # lexicographically (otherwise "1000" < "999" gives a bogus range).
            if self.wavelengths:
                _wl_floats = [float(w) for w in self.wavelengths]
                _wl_range = f"{min(_wl_floats):g} - {max(_wl_floats):g}"
            else:
                _wl_range = "n/a"
            info = "Dataset Info:\n"
            info += f"Number of samples: {len(self.df)}\n"
            info += f"Wavelength range: {_wl_range} nm\n"
            info += f"Wavelength unit (detected): {self.wavelength_unit}\n"
            info += f"Spectral domain (detected): {self.spectral_domain}\n"
            info += f"Available soil properties: {', '.join(self.soil_properties)}\n"
            
            messagebox.showinfo("Excel File Info", info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check Excel: {str(e)}")
            
    def view_model(self):
        try:
            if self.trained_model is None:
                messagebox.showwarning("Warning", "No model loaded")
                return

            # Real classes needed for the isinstance checks below (the module-level
            # names are lazy proxies). A model exists here, so sklearn is loaded.
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.decomposition import PCA
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.svm import SVR
            
            # The model's actual input-feature count is the resampled grid when
            # resampling was applied, else the filtered grid, else the raw bands.
            if self.new_wavelengths is not None:
                n_features = len(self.new_wavelengths)
            elif self.filtered_wavelengths is not None:
                n_features = len(self.filtered_wavelengths)
            elif self.wavelengths:
                n_features = len(self.wavelengths)
            else:
                n_features = 'Unknown'

            model_info = "Model Properties:\n\n"
            model_info += f"Model Type: {self.model_var.get()}\n"
            model_info += f"Preprocessing: {self.preprocess_var.get()}\n"
            model_info += f"Number of Input Features: {n_features}\n"
            if self.new_wavelengths is not None:
                model_info += f"Resampling: {self.resample_method}\n"
            
            # Add model-specific properties
            if isinstance(self.trained_model, PLSRegression):
                model_info += f"Number of Components: {self.trained_model.n_components}\n"
            elif isinstance(self.trained_model, PCA):
                model_info += f"Number of Components: {self.trained_model.n_components}\n"
            elif isinstance(self.trained_model, RandomForestRegressor):
                model_info += f"Number of Trees: {self.trained_model.n_estimators}\n"
                model_info += f"Max Depth: {self.trained_model.max_depth}\n"
                model_info += f"Min Samples Split: {self.trained_model.min_samples_split}\n"
                model_info += f"Min Samples Leaf: {self.trained_model.min_samples_leaf}\n"
            elif isinstance(self.trained_model, (Ridge, Lasso)):
                model_info += f"Alpha: {self.trained_model.alpha}\n"
            elif isinstance(self.trained_model, SVR):
                model_info += f"Kernel: {self.trained_model.kernel}\n"
                model_info += f"C: {self.trained_model.C}\n"
                model_info += f"Epsilon: {self.trained_model.epsilon}\n"
                if self.trained_model.kernel == 'poly':
                    model_info += f"Degree: {self.trained_model.degree}\n"
            
            messagebox.showinfo("Model Properties", model_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view model properties: {str(e)}")
            self.status_text.insert(tk.END, f"Error viewing model properties: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
    
    def save_model(self):
        try:
            # Compositional (ALR/CLR/ILR) bundle - save the whole bundle as-is.
            if getattr(self, 'compositional_model', None) is not None:
                b = self.compositional_model
                safe_parts = "-".join(p.replace(' ', '') for p in b.get('parts', []))[:40]
                default = (f"{self.input_filename}_composition_{b.get('transform')}_"
                           f"{safe_parts}_"
                           f"{getattr(self, '_last_result_timestamp', '') or make_timestamp()}"
                           f"_model.joblib")
                path = filedialog.asksaveasfilename(
                    defaultextension=".joblib",
                    filetypes=[("Joblib files", "*.joblib")], initialfile=default)
                if path:
                    joblib.dump(b, path)
                    self.status_text.insert(
                        tk.END, f"Compositional model saved to: {path}\n" + "=" * 50 + "\n")
                    self.status_text.see(tk.END)
                return

            if self.trained_model is None:
                messagebox.showwarning("Warning", "Please train a model first")
                return

            # Model type + hyper-parameters to record.  A best/auto-selected model
            # from a batch or best-preprocessing run sets explicit overrides so the
            # saved file describes the winning combination rather than the current
            # GUI widgets (which may show a different model); a plain single run
            # leaves the overrides unset and falls back to the live widgets.
            _mtype_override = getattr(self, '_save_model_type_override', None)
            _mparams_override = getattr(self, '_save_model_params_override', None)
            save_model_type = _mtype_override or self.model_var.get()
            save_model_parameters = (dict(_mparams_override) if _mparams_override is not None
                                     else {name: var.get() for name, var in self.param_vars.items()})

            # Generate default filename - reuse the timestamp of this run's
            # results/PDF so the model file matches them.
            default_filename = generate_model_filename(
                self.input_filename, self.selected_property, save_model_type,
                timestamp=getattr(self, '_last_result_timestamp', None))
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Joblib files", "*.joblib")],
                initialfile=default_filename
            )
            
            if file_path:
                # Use TRAINING preprocessing config (not current GUI state)
                _pp_method = self.model_preprocessing
                _pp_kwargs = dict(self.model_preprocessing_kwargs)

                # Save model and scalers with parameters
                model_data = {
                    'model': self.trained_model,
                    'pca_component': self.pca_component,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'wavelengths': self.wavelengths,
                    'filtered_wavelengths': self.filtered_wavelengths,
                    'new_wavelengths': self.new_wavelengths,
                    'new_fwhms': self.new_fwhms,
                    # The model's actual input band grid (what the scaler expects):
                    # the resampled grid when resampling is on, else the training
                    # (Excel) wavelengths.  Recorded so unknown data / images can be
                    # resampled onto exactly this grid at predict time.
                    'training_input_wavelengths': (
                        list(self.new_wavelengths) if self.new_wavelengths is not None
                        else list(self.filtered_wavelengths) if self.filtered_wavelengths is not None
                        else None),
                    'training_input_fwhms': (
                        list(self.new_fwhms) if self.new_fwhms is not None
                        else (list(estimate_fwhms_from_grid(self.filtered_wavelengths))
                              if self.filtered_wavelengths is not None else None)),
                    'resample_method': self.resample_method,
                    # Full resampling configuration so the exact same spectral
                    # resampling can be re-applied to unknown data / images later.
                    'resampling_enabled': self.new_wavelengths is not None,
                    'resample_sensor': self.sensor_var.get(),
                    'resample_spacing': self.spacing_var.get(),
                    'srf_table': self.srf_table,
                    'custom_fwhm_table': self.custom_fwhm_table,
                    'exclude_ranges': self._get_exclude_ranges(),
                    'model_type': save_model_type,
                    'preprocessing': _pp_method,
                    'preprocessing_kwargs': _pp_kwargs,
                    'model_parameters': save_model_parameters,
                    'selected_property': self.selected_property
                }
                joblib.dump(model_data, file_path)

                # Remember the saved model's base name for prediction-output naming
                # (<input>_predicted_<model file name>.<ext>).
                self.model_basename = os.path.splitext(os.path.basename(file_path))[0]

                self.status_text.insert(tk.END, f"Model saved to: {file_path}\n")
                # Record the resampling configuration in the log so it is clear
                # what will be re-applied to unknown data / images at predict time.
                if self.new_wavelengths is not None:
                    sensor_sel = self.sensor_var.get()
                    target = (f"{sensor_sel} sensor bands" if sensor_sel != "Custom"
                              else f"{self.spacing_var.get()} nm uniform grid")
                    self.status_text.insert(
                        tk.END,
                        f"  ↳ Resampling saved: {self.resample_method}, "
                        f"{len(self.filtered_wavelengths)} → "
                        f"{len(self.new_wavelengths)} bands ({target}).\n")
                else:
                    self.status_text.insert(tk.END, "  ↳ Resampling saved: none.\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Joblib files", "*.joblib")])
            
            if file_path:
                # Load model and scalers
                model_data = joblib.load(file_path)

                # Compositional (ALR/CLR/ILR) bundle - different shape, handled
                # separately from single-property models.
                if model_data.get('compositional'):
                    self._load_compositional_model(model_data, file_path)
                    return

                self.compositional_model = None
                self.trained_model = model_data['model']
                self.pca_component = model_data.get('pca_component', None)
                self.scaler_X = model_data['scaler_X']
                self.scaler_y = model_data['scaler_y']
                self.wavelengths = model_data['wavelengths']
                self.filtered_wavelengths = model_data.get('filtered_wavelengths')
                self.new_wavelengths = model_data.get('new_wavelengths')
                self.new_fwhms = model_data.get('new_fwhms')
                # The model's input band grid (falls back for older models that
                # predate this key: resampled grid, else filtered training grid).
                self.training_input_wavelengths = model_data.get(
                    'training_input_wavelengths',
                    self.new_wavelengths if self.new_wavelengths is not None
                    else self.filtered_wavelengths)
                self.training_input_fwhms = model_data.get(
                    'training_input_fwhms', self.new_fwhms)
                # Normalize legacy method labels ("Interpolation" / "SRF Convolution").
                self.resample_method = normalize_resample_method(
                    model_data.get('resample_method', 'Linear Interpolation'))
                self.srf_table = model_data.get('srf_table')
                self.custom_fwhm_table = model_data.get('custom_fwhm_table')

                # Restore the resampling configuration so the GUI reflects it and
                # the same resampling can be offered for unknown-data prediction.
                if 'resample_sensor' in model_data:
                    self.sensor_var.set(model_data['resample_sensor'])
                if 'resample_spacing' in model_data:
                    self.spacing_var.set(model_data['resample_spacing'])
                if hasattr(self, 'exclude_ranges_var') and model_data.get('exclude_ranges'):
                    self.exclude_ranges_var.set(", ".join(
                        f"{int(lo)}-{int(hi)}" for lo, hi in model_data['exclude_ranges']))
                if hasattr(self, 'resampling_var'):
                    self.resampling_var.set(
                        "Yes" if self.new_wavelengths is not None else "No")
                if hasattr(self, 'resample_method_var'):
                    self.resample_method_var.set(self.resample_method)

                # Restore preprocessing metadata from saved model
                self.model_preprocessing = model_data.get('preprocessing', 'No Preprocessing')
                self.model_preprocessing_kwargs = model_data.get('preprocessing_kwargs', {})

                # Update GUI to match loaded model
                self.model_var.set(model_data['model_type'])
                self.preprocess_var.set(model_data['preprocessing'])
                
                if 'selected_property' in model_data:
                    self.selected_property = model_data['selected_property']
                
                # Load model parameters if available
                if 'model_parameters' in model_data:
                    self.on_model_change(None)  # Create parameter widgets first
                    for param_name, param_value in model_data['model_parameters'].items():
                        if param_name in self.param_vars:
                            self.param_vars[param_name].set(param_value)
                
                # Remember the model file's base name for prediction-output naming
                # (<input>_predicted_<model file name>.<ext>).
                self.model_basename = os.path.splitext(os.path.basename(file_path))[0]

                self.status_text.insert(tk.END, f"Model loaded from: {file_path}\n")
                if self.new_wavelengths is not None:
                    self.status_text.insert(
                        tk.END,
                        f"  ↳ Model resampling: {self.resample_method}, "
                        f"{len(self.new_wavelengths)} target bands "
                        f"(will be offered for unknown-data prediction).\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
                # Enable view model button after loading
                self.view_model_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def _load_compositional_model(self, bundle, file_path):
        """Restore a compositional (ALR/CLR/ILR) model bundle and the shared
        resampling/preprocessing config used at prediction time."""
        self.compositional_model = bundle
        self.model_basename = os.path.splitext(os.path.basename(file_path))[0]
        self.trained_model = None
        self.pca_component = None
        self.scaler_X = bundle['scaler_X']
        self.scaler_y = None
        self.wavelengths = bundle['wavelengths']
        self.filtered_wavelengths = bundle.get('filtered_wavelengths')
        self.new_wavelengths = bundle.get('new_wavelengths')
        self.new_fwhms = bundle.get('new_fwhms')
        self.resample_method = normalize_resample_method(
            bundle.get('resample_method', 'Linear Interpolation'))
        self.srf_table = bundle.get('srf_table')
        self.custom_fwhm_table = bundle.get('custom_fwhm_table')
        self.model_preprocessing = bundle.get('preprocessing', 'No Preprocessing')
        self.model_preprocessing_kwargs = bundle.get('preprocessing_kwargs', {})
        # A representative property label for output filenames.
        self.selected_property = " + ".join(bundle.get('parts', ['composition']))
        if hasattr(self, 'model_var'):
            self.model_var.set(bundle.get('model_type', self.model_var.get()))
        if hasattr(self, 'preprocess_var'):
            self.preprocess_var.set(self.model_preprocessing)
        if hasattr(self, 'resampling_var'):
            self.resampling_var.set("Yes" if self.new_wavelengths is not None else "No")

        parts = bundle.get('parts', [])
        self.status_text.insert(
            tk.END, f"Compositional model loaded from: {file_path}\n"
                    f"  ↳ Transform: {bundle.get('transform')} | Parts: "
                    f"{', '.join(parts)} (predictions sum to "
                    f"{bundle.get('total', 100.0):.0f}%)\n"
                    f"  ↳ Preprocessing: {self.model_preprocessing} | "
                    f"Model: {bundle.get('model_type')}\n" + "=" * 50 + "\n")
        self.status_text.see(tk.END)
        if hasattr(self, 'view_model_btn'):
            self.view_model_btn.config(state='normal')

    def load_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Load geospatial image",
                filetypes=open_image_filetypes())

            if file_path:
                # Reading a spectral image cube can be slow - offload it so the GUI
                # stays responsive, keeping all Tk access on the main thread.  The
                # reader auto-detects the format and recovers per-band wavelengths
                # / FWHM from the header (ENVI/.hdr etc.) when present.
                self.set_busy_state(True)
                self._pin_geometry()
                try:
                    (self.image_data, self.image_meta, self.image_wavelengths,
                     self.image_fwhms, image_info) = self._run_offloaded(
                        read_geospatial_image, file_path)
                finally:
                    self._unpin_geometry()
                    self.set_busy_state(False)

                self.image_path = file_path
                self.image_driver = image_info.get('driver')
                self.image_interleave = image_info.get('interleave')

                self.status_text.insert(tk.END, f"Loaded image: {file_path}\n")
                self.status_text.insert(
                    tk.END, f"Format: {self.image_driver}   "
                            f"Shape (bands, rows, cols): {self.image_data.shape}\n")
                if self.image_wavelengths:
                    units = image_info.get('wavelength_units') or 'nm'
                    wl = self.image_wavelengths
                    self.status_text.insert(
                        tk.END, f"Header wavelengths: {len(wl)} bands "
                                f"({wl[0]:.1f}–{wl[-1]:.1f} {units})"
                                f"{'  + FWHM' if self.image_fwhms else ''}\n")
                else:
                    self.status_text.insert(
                        tk.END, "No wavelengths in header - will assume the model's "
                                "band grid at prediction time.\n")
                if image_info.get('gain_applied'):
                    g = image_info.get('gain') or []
                    g0 = g[0] if g else None
                    self.status_text.insert(
                        tk.END, f"Radiometric gain/offset applied from header "
                                f"(gain={g0}{'  (per-band)' if len(set(g)) > 1 else ''}) "
                                f"- values are now in physical/reflectance units.\n")
                if image_info.get('nodata') is not None:
                    self.status_text.insert(
                        tk.END, f"No-data value {image_info['nodata']} converted to NaN.\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)

                # Enable predict button if model is trained
                if self.trained_model is not None:
                    self.predict_image_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _image_matches_model_grid(self, img_wl, model_wl):
        """True when the image's per-band wavelengths already coincide with the
        model's resampled grid, so resampling would be a no-op and must be
        skipped (e.g. an EnMAP image fed to an EnMAP-resampled model).

        Requires the same band count and every centre within a small tolerance
        (10% of the model band spacing, min 0.5 nm) to absorb sub-nanometre
        header-vs-preset rounding."""
        try:
            if not img_wl or model_wl is None or len(img_wl) != len(model_wl):
                return False
            a = np.sort(np.asarray([float(w) for w in img_wl], dtype=float))
            b = np.sort(np.asarray([float(w) for w in model_wl], dtype=float))
            tol = max(0.5, 0.1 * float(np.min(np.diff(b)))) if b.size >= 2 else 0.5
            return bool(np.all(np.abs(a - b) <= tol))
        except Exception:
            return False

    def _confirm_apply_resampling(self, context):
        """Ask whether the model's spectral resampling should be re-applied to the
        unknown ``context`` (e.g. "unknown data" / "image") before prediction.

        Returns True to resample, False to skip.  Only meaningful when the loaded
        model carries a resampling configuration (``self.new_wavelengths`` set)."""
        sensor_sel = self.sensor_var.get() if hasattr(self, 'sensor_var') else "Custom"
        if sensor_sel != "Custom":
            target = f"{sensor_sel} sensor bands"
        elif hasattr(self, 'spacing_var'):
            target = f"{self.spacing_var.get()} nm uniform grid"
        else:
            target = "target grid"
        n_bands = len(self.new_wavelengths)
        return messagebox.askyesno(
            "Apply Spectral Resampling?",
            f"This model was trained with spectral resampling:\n"
            f"    • Method: {self.resample_method}\n"
            f"    • Target: {n_bands} bands ({target})\n\n"
            f"Apply the SAME resampling to the {context} before prediction?\n\n"
            f"Choose 'Yes' (recommended) so the {context} is aligned to the "
            f"model's input grid. Choose 'No' only if it is already on the "
            f"model's {n_bands}-band grid.")

    def save_prediction(self):
        try:
            if not hasattr(self, 'image_data') or self.image_data is None:
                messagebox.showwarning("Warning", "Please load an image first")
                return
            
            if self.trained_model is None:
                messagebox.showwarning("Warning", "Please load a model first")
                return
            
            # Resolve the image's per-band wavelengths.  Prefer the real values
            # recovered from the image header - this lets the processor select the
            # overlapping bands and resample onto the model grid *correctly* even
            # when the ranges differ.  Only fall back to the training grid (or an
            # assumed-same-range interpolation) when the header carries none.
            image_bands = self.image_data.shape[0]
            training_bands = len(self.wavelengths)
            header_wl = getattr(self, 'image_wavelengths', None)
            image_wavelengths = None
            if header_wl and len(header_wl) == image_bands:
                image_wavelengths = list(header_wl)
                self.status_text.insert(
                    tk.END, f"  ↳ Using {image_bands} header wavelengths "
                            f"({header_wl[0]:.1f}–{header_wl[-1]:.1f} nm) from the image.\n")
                self.status_text.see(tk.END)
            elif image_bands != training_bands:
                try:
                    min_wl = min(float(w) for w in self.wavelengths)
                    max_wl = max(float(w) for w in self.wavelengths)
                except Exception:
                    min_wl, max_wl = 0, image_bands - 1
                proceed = messagebox.askyesno(
                    "Band Count Mismatch",
                    f"The image has {image_bands} bands, but the model was trained on "
                    f"{training_bands} bands (wavelengths {min_wl:.1f}–{max_wl:.1f} nm).\n\n"
                    f"The image header carries no wavelengths, so PARACUDA-NG will assume its "
                    f"bands are evenly spaced across the training range "
                    f"({min_wl:.1f}–{max_wl:.1f} nm) and resample them onto the model's grid "
                    f"using '{self.resample_method}'.\n\n"
                    f"If the image actually covers a different spectral range, predictions "
                    f"will be unreliable. Proceed?"
                )
                if not proceed:
                    return
                # Build linearly-spaced image wavelengths spanning training range
                image_wavelengths = list(np.linspace(min_wl, max_wl, image_bands))

            self.update_progress(0, "Processing image for prediction...")

            # Use TRAINING preprocessing config (not current GUI state) - same as predict_unknown_csv
            preprocess_kwargs = dict(self.model_preprocessing_kwargs)

            # ── Resampling: same confirm dialog as tabular prediction ─────────
            # If the model carries a spectral-resampling config, ask whether to
            # re-apply it to the image (Yes = align to the model grid; No = the
            # image is asserted to already be on that grid, so skip resampling).
            img_resample = True
            if self.new_wavelengths is not None and len(self.new_wavelengths) > 0:
                if self._image_matches_model_grid(image_wavelengths, self.new_wavelengths):
                    # The image is already on the model's band grid (e.g. an EnMAP
                    # image predicted with an EnMAP-resampled model).  Re-applying
                    # resampling would resample the image onto essentially itself,
                    # so skip it automatically - no need to prompt the user.
                    img_resample = False
                    self.status_text.insert(
                        tk.END, f"  ↳ Resampling skipped - image already matches the "
                                f"model's {len(self.new_wavelengths)}-band grid "
                                f"(no resampling needed).\n")
                elif self._confirm_apply_resampling("image"):
                    self.status_text.insert(
                        tk.END, f"  ↳ Applied resampling: {self.resample_method} → "
                                f"{len(self.new_wavelengths)} bands.\n")
                else:
                    img_resample = False
                    self.status_text.insert(
                        tk.END, "  ↳ Resampling skipped (user choice) - assuming the "
                                "image already matches the model's band grid.\n")
            else:
                self.status_text.insert(
                    tk.END, f"  ↳ Aligning image to the {len(self.filtered_wavelengths)}-band "
                            f"training grid ({self.resample_method}).\n")

            # Make it explicit that the model's TRAINING preprocessing (saved in
            # the model), not the current GUI setting, is applied to the image.
            self.status_text.insert(
                tk.END, f"  ↳ Preprocessing applied (from model): {self.model_preprocessing}"
                        + (f"  {preprocess_kwargs}" if preprocess_kwargs else "") + "\n")
            self.status_text.see(tk.END)

            mask_bg = self.mask_background_var.get() if hasattr(self, 'mask_background_var') else True
            want_cube = (hasattr(self, 'export_resampled_var')
                         and self.export_resampled_var.get())

            # Process image with same preprocessing as training.  self.image_data
            # is already in physical units with no-data converted to NaN by
            # read_geospatial_image, so no raw nodata value needs to be passed here.
            # The validation cube costs a full-size array - only built if the user
            # asked to export it (large scenes otherwise run out of memory).
            img_wl = image_wavelengths if image_wavelengths is not None else self.wavelengths

            # Heavy, Tk-free work (resample + preprocess + model.predict over the
            # whole scene) - offload to a worker thread and pin the window so the
            # GUI stays responsive and does not shrink/auto-resize ("Not Responding").
            # The event loop keeps pumping during the offload, so disable the
            # Predict/View buttons to prevent a re-entrant second prediction.
            self.set_busy_state(True)
            self._pin_geometry()
            self.predict_image_btn.config(state='disabled')
            self.view_image_btn.config(state='disabled')

            # Predict per pixel-chunk INSIDE the image processor so the full
            # (n_px, features) scaled matrix is never materialised - for a large
            # scene that matrix alone can be tens of GiB (e.g. 13.4M px × 1787
            # bands ≈ 89 GiB).  This closure maps a scaled block to final-unit
            # predictions (PCA → model.predict → inverse-transform).
            _pca = self.pca_component
            _model = self.trained_model
            _scaler_y = self.scaler_y

            def _predict_chunk(scaled_block):
                Xb = _pca.transform(scaled_block) if _pca is not None else scaled_block
                ps = np.asarray(_model.predict(Xb)).reshape(-1, 1)
                return _scaler_y.inverse_transform(ps).ravel()

            try:
                (predictions, original_shape, valid_mask, resampled_cube,
                 target_wl) = self._run_offloaded(
                    process_image_for_prediction,
                    self.image_data, img_wl, self.model_preprocessing,
                    self.scaler_X, self.filtered_wavelengths, self.new_wavelengths,
                    preprocess_kwargs=preprocess_kwargs,
                    resample_method=self.resample_method, new_fwhms=self.new_fwhms,
                    srf_table=self.srf_table, resample=img_resample,
                    mask_background=mask_bg, return_cube=want_cube,
                    predict_fn=_predict_chunk
                )
                n_bg = int((~valid_mask).sum())
                if n_bg:
                    self.status_text.insert(
                        tk.END, f"  ↳ Masked {n_bg} background/no-data pixel(s) "
                                f"(left unpredicted).\n")
                    self.status_text.see(tk.END)

                self.update_progress(40, "Making predictions...")
            finally:
                self._unpin_geometry()
                self.set_busy_state(False)
                # Re-enable Predict so the user can run again; View is enabled
                # below only once a prediction image actually exists.
                self.predict_image_btn.config(state='normal')

            self.update_progress(80, "Saving results...")
            
            # Save prediction - default to the SAME format/extension as the input
            # image so the output header matches the source (ENVI→.hdr, .img, …).
            in_ext = os.path.splitext(getattr(self, 'image_path', '') or '')[1].lower() or ".tif"
            prop_name = self.selected_property if hasattr(self, 'selected_property') else "prediction"
            # <image file name>_predicted_<model file name>.<input format>
            img_base = os.path.splitext(
                os.path.basename(getattr(self, 'image_path', '') or 'image'))[0]
            file_path = filedialog.asksaveasfilename(
                title="Save prediction map",
                defaultextension=in_ext,
                initialfile=f"{self._predicted_output_basename(img_base)}{in_ext}",
                filetypes=[(f"Same as input ({in_ext})", f"*{in_ext}"),
                           ("GeoTIFF", "*.tif *.tiff"),
                           ("ENVI", "*.dat *.bil *.bsq"),
                           ("ERDAS Imagine", "*.img"),
                           ("All files", "*.*")])

            if file_path:
                self.predicted_image = save_prediction_image(
                    predictions, original_shape, self.image_meta, file_path,
                    band_name=f"Predicted {prop_name}", valid_mask=valid_mask)

                # Optionally export the resampled reflectance cube so the user can
                # validate that resampling produced sensible spectra.  Named after
                # the INPUT image (<input>_resampled.<input format>) and written in
                # the same format/interleave, beside the chosen prediction output.
                if hasattr(self, 'export_resampled_var') and self.export_resampled_var.get():
                    in_base = os.path.splitext(os.path.basename(self.image_path))[0]
                    in_ext = os.path.splitext(self.image_path)[1] or '.tif'
                    cube_path = os.path.join(os.path.dirname(file_path),
                                             f"{in_base}_resampled{in_ext}")
                    try:
                        save_cube_image(
                            resampled_cube, original_shape, target_wl,
                            self.training_input_fwhms or self.new_fwhms,
                            self.image_meta, cube_path,
                            interleave=getattr(self, 'image_interleave', None),
                            src_image_path=self.image_path)
                        self.status_text.insert(
                            tk.END, f"  ↳ Resampled cube exported: {cube_path} "
                                    f"({len(target_wl)} bands).\n")
                        self.status_text.see(tk.END)
                    except Exception as ce:
                        self.status_text.insert(
                            tk.END, f"  ↳ Could not export resampled cube: {ce}\n")
                        self.status_text.see(tk.END)
                
                # Store predicted image in visualization as matplotlib figure
                if self.predicted_image is not None:
                    property_name = self.selected_property if hasattr(self, 'selected_property') else 'Soil Property'

                    # Build the figure with the current colormap and store it.  The
                    # same builder is reused when the colormap is changed later so the
                    # visualization updates live without re-running the prediction.
                    fig = self._build_predicted_image_figure(self.predicted_image, property_name)
                    self.store_analysis_plot('Image Output', property_name, fig, item_type='image')

                    # A prediction image now exists - enable the View button so the
                    # user can return to it after navigating away.
                    self.view_image_btn.config(state='normal')

                    # Switch to visualization tab and select the image
                    self.display_notebook.select(2)  # Select Visualization tab (index 2)
                    
                self.update_progress(100, f"Prediction saved to: {file_path}")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save prediction: {str(e)}")
            self.status_text.insert(tk.END, f"Error saving prediction: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)

    def _predicted_output_basename(self, input_base):
        """Return ``<input_base>_predicted_<model file name>`` for naming a
        prediction output (image or Excel).

        ``model file name`` is the base name of the loaded/saved model file when
        available, otherwise the selected algorithm name.  All parts are
        filesystem-sanitised.
        """
        def _safe(s):
            return str(s).replace(' ', '_').replace('/', '_').replace('\\', '_')
        model_base = getattr(self, 'model_basename', None) or self.model_var.get()
        return f"{_safe(input_base)}_predicted_{_safe(model_base)}"

    def _build_predicted_image_figure(self, image, property_name):
        """Build the predicted-image matplotlib Figure using the currently
        selected colormap (values <= 0 masked out)."""
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        masked_image = np.ma.masked_where(image <= 0, image)
        im = ax.imshow(masked_image, cmap=self.colormap_var.get())
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Predicted {property_name}')
        return fig

    def _on_colormap_change(self, event=None):
        """Re-render the stored predicted image with the newly selected colormap
        and refresh the Visualization tab live.

        This must NOT re-run the prediction or change the Predict/View button
        states - only the displayed colormap updates.  A no-op when no prediction
        has been made yet.
        """
        if getattr(self, 'predicted_image', None) is None:
            return
        property_name = (self.selected_property
                         if getattr(self, 'selected_property', None) else 'Soil Property')
        fig = self._build_predicted_image_figure(self.predicted_image, property_name)
        self.store_analysis_plot('Image Output', property_name, fig, item_type='image')
        # If the predicted image is the plot on screen, redraw it in place.
        key = f"Image Output - {property_name}"
        if hasattr(self, 'viz_model_var') and key in getattr(self, 'analysis_plots', {}):
            self.viz_model_var.set(key)
            self.display_selected_plot()

    def select_soil_property(self):
        try:
            if self.soil_properties is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            # Check if the dialog already exists and is open
            if hasattr(self, 'selection_window') and self.selection_window.winfo_exists():
                self.selection_window.lift()
                return
            
            self.selection_window = tk.Toplevel(self)
            self.selection_window.title("Select Soil Properties")
            self.selection_window.geometry("400x350")
            
            # Instructions
            ttk.Label(self.selection_window, 
                     text="Select one or more soil properties:",
                     font=('Helvetica', 10, 'bold')).pack(pady=5)
            
            # Frame for listbox and scrollbar
            list_frame = ttk.Frame(self.selection_window)
            list_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Listbox with MULTIPLE selection
            listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
            for prop in self.soil_properties:
                listbox.insert(tk.END, prop)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)
            
            # Buttons frame
            btn_frame = ttk.Frame(self.selection_window)
            btn_frame.pack(pady=5)
            
            def on_select():
                selections = listbox.curselection()
                if selections:
                    self.selected_properties = [self.soil_properties[i] for i in selections]
                    self.selected_property = self.selected_properties[0]  # Keep for backward compatibility
                    prop_str = ", ".join(self.selected_properties)
                    self.status_text.insert(tk.END, f"Selected properties: {prop_str}\n")
                    self.status_text.insert(tk.END, "="*50 + "\n")
                    self.status_text.see(tk.END)
                    
                    # Enable/disable save model button based on number of properties
                    # Save model is only enabled when one property is selected
                    if len(self.selected_properties) == 1:
                        # Will be enabled after training
                        self.single_property_mode = True
                    else:
                        self.save_model_btn.config(state='disabled')
                        self.single_property_mode = False
                    
                    # Update Find Best Preprocessing checkbox state
                    self._update_best_preprocess_state()

                    self.selection_window.destroy()
            
            def select_all():
                listbox.select_set(0, tk.END)
            
            def clear_all():
                listbox.selection_clear(0, tk.END)
            
            ttk.Button(btn_frame, text="Select All", command=select_all).grid(row=0, column=0, padx=5)
            ttk.Button(btn_frame, text="Clear All", command=clear_all).grid(row=0, column=1, padx=5)
            ttk.Button(btn_frame, text="Confirm Selection", command=on_select).grid(row=0, column=2, padx=5)

            def on_close():
                self.selection_window.destroy()
                del self.selection_window
            
            self.selection_window.protocol("WM_DELETE_WINDOW", on_close)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select property: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tabular prediction: load unknown data without disturbing tool state
    # ─────────────────────────────────────────────────────────────────────────
    def load_tabular_excel(self):
        """Load an unknown-data file for tabular prediction without resetting the tool."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select unknown data file (CSV or Excel)",
                filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
            if not file_path:
                return

            # Offload the (potentially slow) read to keep the GUI responsive.
            self.set_busy_state(True)
            self._pin_geometry()
            try:
                df = self._load_dataframe_with_progress(file_path)
            finally:
                self._reset_load_progress()
                self._unpin_geometry()
                self.set_busy_state(False)

            # Detect wavelength columns
            raw_wl_cols = [col for col in df.columns
                           if str(col).replace('.', '', 1).replace('-', '', 1).isdigit()]
            if not raw_wl_cols:
                messagebox.showerror("Error", "No numeric wavelength columns found in the file.")
                return

            raw_wl_floats = [float(c) for c in raw_wl_cols]
            unit, wl_nm = detect_wavelength_unit(raw_wl_floats)

            if unit == "μm":
                rename_map = {col: str(w) for col, w in zip(raw_wl_cols, wl_nm)}
                df.rename(columns=rename_map, inplace=True)
                wl_col_names = list(rename_map.values())  # now string repr of nm floats
            else:
                wl_col_names = list(raw_wl_cols)  # original column names (may be int or str)

            # Store separately - do NOT touch self.df or any training state
            self.tabular_df = df
            self.tabular_wl_nm = wl_nm
            self.tabular_wl_cols = wl_col_names   # exact column names as they exist in df
            self.tabular_file_path = file_path

            n_samples = len(df)
            wl_min = min(wl_nm)
            wl_max = max(wl_nm)
            self.status_text.insert(tk.END,
                f"[Tabular] Loaded unknown file: {os.path.basename(file_path)}\n"
                f"[Tabular] Samples: {n_samples}, Wavelengths: {len(wl_nm)} "
                f"({wl_min:.1f}–{wl_max:.1f} nm, detected unit: {unit})\n"
                + "=" * 50 + "\n")
            self.status_text.see(tk.END)

            # Activate Predict button now that data is loaded
            self.predict_tabular_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file: {str(e)}")

    def check_tabular_excel(self):
        """Show information about the loaded unknown tabular data."""
        if self.tabular_df is None:
            messagebox.showwarning("Warning", "Please load an unknown data file first.")
            return
        wl_nm = self.tabular_wl_nm or []
        info = (
            f"Unknown data file: {os.path.basename(self.tabular_file_path)}\n"
            f"Samples: {len(self.tabular_df)}\n"
            f"Wavelengths: {len(wl_nm)}"
            + (f" ({min(wl_nm):.1f}–{max(wl_nm):.1f} nm)" if wl_nm else "") + "\n"
            f"Columns: {', '.join(str(c) for c in self.tabular_df.columns[:10])}"
            + (" ..." if len(self.tabular_df.columns) > 10 else "") + "\n\n"
            f"Model loaded: {'Yes' if self.trained_model is not None else 'No'}\n"
            f"Model preprocessing: {self.model_preprocessing}\n"
            f"Preprocessing params: {self.model_preprocessing_kwargs or 'none'}"
        )
        messagebox.showinfo("Unknown Data Info", info)

    # ─────────────────────────────────────────────────────────────────────────
    # Predict CSV/Excel with trained model
    # ─────────────────────────────────────────────────────────────────────────
    def predict_unknown_csv(self):
        """Predict soil properties for pre-loaded unknown tabular data using the trained model."""
        try:
            is_comp = getattr(self, 'compositional_model', None) is not None
            if self.trained_model is None and not is_comp:
                messagebox.showwarning("Warning",
                    "Please load or train a model first, then use this button.")
                return

            if self.tabular_df is None:
                messagebox.showwarning("Warning",
                    "Please load an unknown data file first (Load Excel button).")
                return

            unknown_df = self.tabular_df
            wl_nm_new = list(self.tabular_wl_nm)
            wl_col_names = list(self.tabular_wl_cols)
            file_path = self.tabular_file_path

            # Build spectral matrix - use the stored column names (exact match in df)
            missing = [c for c in wl_col_names if c not in unknown_df.columns]
            if len(missing) == len(wl_col_names):
                messagebox.showerror("Error",
                    "Wavelength columns not found in the loaded file. Reload the file.")
                return
            # Filter to only columns actually present (should be all of them)
            wl_col_names = [c for c in wl_col_names if c in unknown_df.columns]
            wl_nm_new = [w for c, w in zip(
                list(self.tabular_wl_cols), list(self.tabular_wl_nm)) if c in wl_col_names]

            X_unknown = unknown_df[wl_col_names].values.astype(float)

            # ── Align to model wavelength space ──────────────────────────────
            model_wl = [float(w) for w in self.wavelengths]
            if len(wl_nm_new) != len(model_wl) or not np.allclose(
                    sorted(wl_nm_new), sorted(model_wl), atol=2.0):
                # Interpolate onto the model grid, but only over the span the
                # unknown data actually covers - safe_interpolate_spectra clips
                # to that span internally, so we must track the SAME grid here to
                # keep `current_wl` aligned 1:1 with the resulting columns.
                model_wl_arr = np.asarray(model_wl, dtype=float)
                lo, hi = min(wl_nm_new), max(wl_nm_new)
                target_wl = model_wl_arr[(model_wl_arr >= lo) & (model_wl_arr <= hi)]
                if len(target_wl) == 0:
                    messagebox.showerror(
                        "Prediction Error",
                        "The unknown data's wavelength range does not overlap the "
                        "model's training wavelengths. Cannot align spectra.")
                    return
                X_unknown = safe_interpolate_spectra(X_unknown, wl_nm_new, target_wl)
                current_wl = list(target_wl)
            else:
                current_wl = wl_nm_new

            # ── Wavelength filtering (same as training) ───────────────────────
            # Apply BOTH halves of the training filter: the min/max range AND the
            # excluded regions (water bands, noisy edges).  Dropping only on the
            # range would keep the excluded interior bands, so with resampling off
            # the model would be handed more features than it was trained on.
            # (`exclude_ranges` is saved with the model and restored on load, so
            # this matches training whether or not the model came from disk.)
            if self.filtered_wavelengths is not None and len(self.filtered_wavelengths) > 0:
                lo = min(self.filtered_wavelengths)
                hi = max(self.filtered_wavelengths)
                excl = parse_exclude_ranges(self._get_exclude_ranges())

                def _kept(w):
                    if not (lo <= w <= hi):
                        return False
                    return not any(a <= w <= b for a, b in excl)

                wl_idx = [i for i, w in enumerate(current_wl) if _kept(w)]
                X_unknown = X_unknown[:, wl_idx]
                # Actual wavelengths of the kept columns - the true SOURCE grid to
                # resample FROM (guaranteed to match X_unknown's column count).
                filt_wl = [current_wl[i] for i in wl_idx]
            else:
                filt_wl = current_wl

            export_wl = filt_wl
            if self.new_wavelengths is not None and len(self.new_wavelengths) > 0:
                if self._image_matches_model_grid(filt_wl, self.new_wavelengths):
                    # Data already on the model's grid - resampling is a no-op, skip
                    # it silently rather than prompting.
                    self.status_text.insert(
                        tk.END,
                        f"  ↳ Resampling skipped - data already matches the model's "
                        f"{len(self.new_wavelengths)}-band grid (no resampling needed).\n")
                elif self._confirm_apply_resampling("unknown data"):
                    X_unknown = resample_spectra(X_unknown, filt_wl, self.new_wavelengths,
                                                 method=self.resample_method, fwhms=self.new_fwhms,
                                                 srf_table=self.srf_table)
                    export_wl = self.new_wavelengths
                    self.status_text.insert(
                        tk.END,
                        f"  ↳ Applied resampling: {self.resample_method} → "
                        f"{len(self.new_wavelengths)} bands.\n")
                else:
                    self.status_text.insert(
                        tk.END,
                        "  ↳ Resampling skipped (user choice) - assuming the "
                        "unknown data already matches the model's band grid.\n")
                self.status_text.see(tk.END)

            # Snapshot the spectra AFTER resampling but BEFORE preprocessing -
            # this is what "resampling validation" should show: the same grid
            # the model consumes, in its original (not derivative/transformed) units.
            X_resampled_for_export = X_unknown.copy()

            # ── Preprocessing - use TRAINING config, not current GUI ──────────
            method = self.model_preprocessing
            pp_kwargs = dict(self.model_preprocessing_kwargs)
            self.status_text.insert(
                tk.END, f"  ↳ Preprocessing applied (from model): {method}"
                        + (f"  {pp_kwargs}" if pp_kwargs else "") + "\n")
            self.status_text.see(tk.END)
            if method not in ("No Preprocessing", "Spectral Outlier Removal", ""):
                X_unknown = preprocess_spectra(X_unknown, method, **pp_kwargs)

            # ── Scale and predict ─────────────────────────────────────────────
            X_scaled = self.scaler_X.transform(X_unknown)
            # Apply PCA transform if this is a PCA model
            if self.pca_component is not None:
                X_scaled = self.pca_component.transform(X_scaled)

            # Compositional model → predict a closed composition (parts sum to
            # the bundle total, e.g. 100 %) and write one column per part.
            if is_comp:
                b = self.compositional_model
                comp_pred = self.predict_composition(X_scaled)   # (n, D)
                parts = b['parts']
                out_df = unknown_df.copy()
                for d, p in enumerate(parts):
                    out_df[f"Predicted_{p}"] = comp_pred[:, d]
                out_df["Predicted_sum"] = comp_pred.sum(axis=1)
                self.status_text.insert(
                    tk.END, f"[Compositional prediction] {b['transform']} | parts: "
                            f"{', '.join(parts)} | n={len(comp_pred)} | "
                            f"each row sums to {b.get('total', 100.0):.0f}%\n"
                            f"  Preprocessing used: {method} {pp_kwargs or ''}\n")
                self.status_text.see(tk.END)
                data_base = os.path.splitext(os.path.basename(file_path))[0]
                default_fname = f"{self._predicted_output_basename(data_base)}.xlsx"
                out_path = filedialog.asksaveasfilename(
                    title="Save compositional predictions as",
                    defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")],
                    initialfile=default_fname)
                if out_path:
                    out_df.to_excel(out_path, index=False)
                    self.status_text.insert(tk.END, f"Predictions saved: {out_path}\n")
                    self.status_text.see(tk.END)
                    messagebox.showinfo("Done", f"Compositional predictions saved to:\n{out_path}")
                return

            y_pred_scaled = self.trained_model.predict(X_scaled)
            predictions = self.scaler_y.inverse_transform(
                np.array(y_pred_scaled).ravel().reshape(-1, 1)).ravel()

            self.status_text.insert(tk.END,
                f"[Tabular prediction] File: {os.path.basename(file_path)} | "
                f"n={len(predictions)}, mean={np.mean(predictions):.4f}, "
                f"std={np.std(predictions):.4f}, "
                f"min={np.min(predictions):.4f}, max={np.max(predictions):.4f}\n"
                f"  Preprocessing used: {method} {pp_kwargs or ''}\n")
            self.status_text.see(tk.END)

            prop = self.selected_property or "property"
            model_type = self.model_var.get()
            # <data file name>_predicted_<model file name>.xlsx
            data_base = os.path.splitext(os.path.basename(file_path))[0]
            default_fname = f"{self._predicted_output_basename(data_base)}.xlsx"
            out_path = filedialog.asksaveasfilename(
                title="Save predictions as",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=default_fname)
            if out_path:
                save_unknown_predictions_to_excel(
                    out_path, unknown_df, predictions, prop, model_type, self.wavelength_unit)
                self.status_text.insert(tk.END, f"Predictions saved: {out_path}\n")
                self.status_text.see(tk.END)

                # Optionally export the resampled spectra so the user can
                # validate resampling - named after the INPUT file, beside
                # the chosen prediction output (same pattern as image export).
                if (hasattr(self, 'export_resampled_tabular_var')
                        and self.export_resampled_tabular_var.get()):
                    in_base = os.path.splitext(os.path.basename(file_path))[0]
                    resampled_path = os.path.join(
                        os.path.dirname(out_path), f"{in_base}_resampled.xlsx")
                    try:
                        id_cols = [c for c in unknown_df.columns if c not in wl_col_names]
                        save_resampled_tabular_to_excel(
                            resampled_path, unknown_df[id_cols], X_resampled_for_export,
                            export_wl, self.wavelength_unit)
                        self.status_text.insert(
                            tk.END, f"Resampled data exported: {resampled_path} "
                                    f"({len(export_wl)} bands).\n")
                        self.status_text.see(tk.END)
                    except Exception as ce:
                        self.status_text.insert(
                            tk.END, f"Could not export resampled data: {ce}\n")
                        self.status_text.see(tk.END)

                messagebox.showinfo("Done", f"Predictions saved to:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Prediction Error",
                f"Failed to predict: {str(e)}\n{traceback.format_exc()}")
    
    # ------------------------------------------------------------------
    # Missing-data reporting helpers
    # ------------------------------------------------------------------
    def _format_missing_data_line(self, summary):
        """One-line missing-data summary for the Data View header."""
        if not summary or not summary.get('has_missing'):
            return "Missing data: none detected ✓"
        return (f"⚠ Missing data: {summary['total_missing']} empty/invalid cell(s) "
                f"in {summary['rows_with_missing']}/{summary['n_rows']} row(s) "
                f"- handled via 'Missing data' option before training")

    def _report_missing_data(self, summary):
        """Write missing-data findings to the status log, sidebar hint, and
        (when present) pop a warning dialog so the user picks a strategy."""
        if not summary or not summary.get('has_missing'):
            if hasattr(self, 'missing_data_label'):
                self.missing_data_label.config(
                    text="No missing values detected ✓", foreground="#006000")
            self.status_text.insert(tk.END, "Missing data: none detected.\n")
            return

        prop_bits = ", ".join(f"{k} ({v})" for k, v in
                              summary['property_missing'].items())
        self.status_text.insert(
            tk.END,
            f"⚠ Missing/empty cells detected: {summary['total_missing']} cell(s) "
            f"across {summary['rows_with_missing']}/{summary['n_rows']} row(s).\n")
        if summary['spectral_missing']:
            self.status_text.insert(
                tk.END, f"    • {summary['spectral_missing']} in spectral (wavelength) columns\n")
        if prop_bits:
            self.status_text.insert(
                tk.END, f"    • property columns: {prop_bits}\n")
        self.status_text.insert(
            tk.END,
            f"    → Will be handled with '{self.missing_data_var.get()}' "
            f"(change via the 'Missing data' option) at train time.\n")

        if hasattr(self, 'missing_data_label'):
            self.missing_data_label.config(
                text=(f"⚠ {summary['total_missing']} missing cell(s) in "
                      f"{summary['rows_with_missing']} row(s) - choose a strategy"),
                foreground="#b06000")

        messagebox.showwarning(
            "Missing Data Detected",
            f"The loaded dataset contains {summary['total_missing']} empty or "
            f"invalid cell(s) in {summary['rows_with_missing']} of "
            f"{summary['n_rows']} rows.\n\n"
            f"Choose how to handle them with the 'Missing data' dropdown in the "
            f"Preprocessing panel. Current setting: "
            f"'{self.missing_data_var.get()}'.\n\n"
            f"Note: rows whose target property is missing are always dropped.")

    def _apply_missing_data_handling(self, X, y):
        """Clean NaN/Inf in (X, y) using the user-selected strategy.

        Returns (X_clean, y_clean, info_dict). Logs a summary to the status box
        when anything was changed. Raises ValueError when no usable data remains.
        """
        method = (self.missing_data_var.get()
                  if hasattr(self, 'missing_data_var') else MISSING_DATA_METHODS[0])
        X_clean, y_clean, info = handle_missing_data(X, y, method)
        if info.get('message'):
            self.status_text.insert(tk.END, f"  ↳ {info['message']}\n")
            self.status_text.see(tk.END)
        return X_clean, y_clean, info

    def _apply_target_outlier_removal(self, X, y, property_name=None):
        """Drop samples whose target value ``y`` is an outlier, if the
        'Target Variable Outlier Removal' checkbox is enabled.

        The selected property (e.g. clay %) is the target variable: samples whose
        value is a z-score / IQR outlier are removed together with their paired
        spectra (``X``).  Returns the possibly-reduced ``(X, y)`` unchanged when
        the option is off or no samples qualify.  Logs a one-line summary.
        """
        if not getattr(self, 'target_outlier_var', None) or not self.target_outlier_var.get():
            return X, y

        method = (self.target_outlier_method_var.get()
                  if hasattr(self, 'target_outlier_method_var')
                  else TARGET_OUTLIER_METHODS[0])
        try:
            threshold = float(self.target_outlier_threshold_var.get())
        except (ValueError, tk.TclError):
            threshold = 2.5

        # The mask depends only on y + method/threshold, not on which model is
        # about to be trained on it. A batch run calls this once per model for
        # the same property with an unchanged y (find-best-preprocessing runs
        # legitimately vary y per combo, so those still recompute), so cache
        # on the actual target values and only log/apply it once instead of
        # redoing the detection and re-printing the summary for every model.
        cache = self.__dict__.setdefault('_target_outlier_cache', {})
        cache_key = (method, threshold, y.tobytes())
        if cache_key in cache:
            keep = cache[cache_key]
            return (X[keep], y[keep]) if keep is not None else (X, y)

        keep = remove_target_outliers(y, method=method, threshold=threshold)
        n_removed = int((~keep).sum())
        if n_removed and int(keep.sum()) >= 2:
            label = property_name or getattr(self, 'selected_property', None) or "target"
            self.status_text.insert(
                tk.END,
                f"  ↳ Target outlier removal [{label}]: removed {n_removed} "
                f"sample(s) ({method}, threshold={threshold:g}). "
                f"{int(keep.sum())} remaining.\n")
            self.status_text.see(tk.END)
            cache[cache_key] = keep
            return X[keep], y[keep]
        if n_removed:
            # Would leave <2 samples - skip rather than break training.
            self.status_text.insert(
                tk.END,
                f"  ↳ Target outlier removal skipped: would leave too few "
                f"samples ({int(keep.sum())}).\n")
            self.status_text.see(tk.END)
        cache[cache_key] = None
        return X, y
