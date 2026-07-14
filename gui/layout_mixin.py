"""
Main window layout, parameter widgets and their change callbacks.

@author: Sharad Kumar Gupta
"""
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


class LayoutMixin:

    # Fallback colours used when no shared theme is active (preserve the old look).
    _C_FALLBACK = {
        'WIN': '#f0f0f0', 'CARD': '#ffffff', 'BNR': '#2c3e50', 'MUT': '#7f8c8d',
        'CONSOLE_BG': '#2c3e50', 'CONSOLE_FG': '#ecf0f1',
    }

    def _c(self, key, default=None):
        """Return a palette colour by key, falling back to the legacy value."""
        pal = getattr(self, '_palette', None)
        if pal and key in pal:
            return pal[key]
        return default if default is not None else self._C_FALLBACK.get(key, '#f0f0f0')

    def create_gui(self):
        # Set window size and make it resizable
        self.geometry("1400x900")
        self.minsize(1200, 700)
        # Give the root window the theme background so that regions briefly
        # exposed when restoring from a minimized/maximized state repaint in the
        # theme colour instead of flashing black (a Windows Tk redraw artifact).
        self.configure(bg=self._c('WIN'))
        # Start maximized
        self.state('zoomed')
        
        # Main container with PanedWindow for resizable panels
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # ========== LEFT PANEL: STEP-BY-STEP WIZARD ==========
        control_panel = ttk.Frame(main_paned, width=780)
        control_panel.pack_propagate(False)
        main_paned.add(control_panel, weight=0)

        # Set sash after window is fully rendered (fires once, then unbinds)
        def _set_sash(event, _fired=[False]):
            if not _fired[0]:
                _fired[0] = True
                try:
                    main_paned.sashpos(0, 780)
                except Exception:
                    pass
        main_paned.bind("<Configure>", _set_sash)

        # Title / clock header above the wizard
        title_label = ttk.Label(control_panel, text="Paracuda III",
                               font=('Helvetica', 16, 'bold'), foreground=self._c('BNR'))
        title_label.pack(pady=(8, 2))
        self.datetime_label = ttk.Label(control_panel, text="",
                                       font=('Helvetica', 9), foreground=self._c('MUT'))
        self.datetime_label.pack(pady=(0, 4))

        # Back / Next navigation bar (packed at the bottom so it is always visible)
        nav_bar = ttk.Frame(control_panel)
        nav_bar.pack(fill="x", side="bottom", padx=6, pady=(2, 6))
        self._wizard_back_btn = ttk.Button(nav_bar, text="◀ Back",
                                           command=lambda: self._wizard_step(-1), width=10)
        self._wizard_back_btn.pack(side="left")
        self._wizard_next_btn = ttk.Button(nav_bar, text="Next ▶",
                                           command=lambda: self._wizard_step(1), width=10)
        self._wizard_next_btn.pack(side="right")

        # Split the left panel vertically: the step wizard on top, a compact
        # Model-Development-Flow overview docked directly below it (details on
        # demand via "Large"/"300 DPI").  A draggable sash lets the user trade
        # space between them.
        left_split = ttk.PanedWindow(control_panel, orient=tk.VERTICAL)
        left_split.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        self._left_split = left_split

        wizard_holder = ttk.Frame(left_split)
        left_split.add(wizard_holder, weight=3)

        # The wizard notebook — one scrollable step per tab.  Each step frame is a
        # scrollable inner frame; the existing control groups are packed into the
        # step that matches the natural processing order.
        self.wizard_nb = ttk.Notebook(wizard_holder)
        self.wizard_nb.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        # Track the per-step scroll canvases so a live theme switch can repaint
        # their (plain tk.Canvas) backgrounds.
        self._scroll_canvases = []

        step1 = self._make_scroll_tab("① Data")
        step2 = self._make_scroll_tab("② Configuration")
        step3 = self._make_scroll_tab("③ Preprocess")
        step4 = self._make_scroll_tab("④ Model")
        step5 = self._make_scroll_tab("⑤ Validate")
        step6 = self._make_scroll_tab("⑥ Execution")
        step7 = self._make_scroll_tab("⑦ Apply")
        self.wizard_nb.bind("<<NotebookTabChanged>>", self._on_wizard_tab_changed)

        # Data Loading  →  Step ① Data
        data_frame = ttk.LabelFrame(step1, text="Data", padding="10")
        data_frame.pack(fill="x", padx=10, pady=5)
        
        # Load Excel and Check Data side by side
        data_row = ttk.Frame(data_frame)
        data_row.pack(fill="x", pady=2)
        
        ttk.Button(data_row, text="📁 Load Excel", 
                  command=self.load_excel, width=18).pack(side="left", padx=2, expand=True, fill="x")
        ttk.Button(data_row, text="📊 Check Data", 
                  command=self.check_excel, width=18).pack(side="left", padx=2, expand=True, fill="x")
        
        # Export Statistics option
        self.export_stats_var = tk.BooleanVar()
        ttk.Checkbutton(data_frame, text="Export Statistics", 
                       variable=self.export_stats_var).pack(anchor="w", pady=2)
        
        # Property Selection  →  Step ① Data
        property_frame = ttk.LabelFrame(step1, text="Property Selection", padding="10")
        property_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Button(property_frame, text="Select Properties",
                  command=self.select_soil_property, width=25).pack(pady=2, fill="x", expand=True)

        # Compositional (log-ratio) modelling for sum-constrained properties
        # (e.g. sand + silt + clay = 100 %).  When set to CLR/ALR/ILR and 2+
        # properties are selected, models are trained on log-ratio coordinates
        # and predictions are back-transformed to a closed composition.
        comp_row = ttk.Frame(property_frame)
        comp_row.pack(fill="x", pady=(6, 0))
        ttk.Label(comp_row, text="Compositional (log-ratio):").pack(side="left")
        self.comp_transform_var = tk.StringVar(value="None")
        ttk.Combobox(comp_row, textvariable=self.comp_transform_var,
                     values=["None", "CLR", "ALR", "ILR"], state="readonly",
                     width=8).pack(side="left", padx=4)
        ttk.Label(property_frame,
                  text="Use for parts that sum to 100% (sand/silt/clay). Select the "
                       "parts above, then Run - predictions will sum to 100%.",
                  wraplength=380, foreground="#666666",
                  font=('Helvetica', 8, 'italic')).pack(anchor="w", pady=(2, 0))

        # Preprocessing  →  Step ③ Preprocess
        preprocess_frame = ttk.LabelFrame(step3, text="Preprocessing", padding="10")
        preprocess_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Label(preprocess_frame, text="Method:").pack(anchor="w")
        self.preprocess_var = tk.StringVar(value="No Preprocessing")
        self.preprocess_combo = ttk.Combobox(preprocess_frame, textvariable=self.preprocess_var,
                                       values=["No Preprocessing", "Smoothing", "Spectral Outlier Removal",
                                              "Continuum Removal", "Baseline Correction",
                                              "First Derivative", "Second Derivative", "Absorbance"],
                                       state='readonly', width=23)
        self.preprocess_combo.pack(fill="x", pady=2)
        self.preprocess_combo.bind("<<ComboboxSelected>>", self.on_preprocess_change)
        
        self.preprocess_params_frame = ttk.Frame(preprocess_frame)
        self.preprocess_params_frame.pack(fill="both", expand=True, pady=5)

        # Find Best Preprocessing option (searches each selected property)
        self.find_best_preprocess_var = tk.BooleanVar(value=False)
        self.find_best_preprocess_checkbox = ttk.Checkbutton(
            preprocess_frame,
            text="Find Best Preprocessing",
            variable=self.find_best_preprocess_var,
            command=self._sync_preprocess_combo_state,
            state='disabled'
        )
        self.find_best_preprocess_checkbox.pack(anchor="w", pady=(2, 0))

        # ── Missing-data handling ─────────────────────────────────────────────
        # Empty cells / non-numeric text become NaN and would otherwise break
        # training with confusing downstream errors.  This strategy is applied to
        # the feature matrix before resampling/preprocessing; rows with a missing
        # target value are always dropped.
        ttk.Label(preprocess_frame, text="Missing data:").pack(anchor="w", pady=(6, 0))
        self.missing_data_var = tk.StringVar(value=MISSING_DATA_METHODS[0])
        missing_combo = ttk.Combobox(preprocess_frame, textvariable=self.missing_data_var,
                                     values=MISSING_DATA_METHODS,
                                     state='readonly', width=23)
        missing_combo.pack(fill="x", pady=2)
        self.missing_data_label = ttk.Label(
            preprocess_frame, text="", wraplength=230,
            font=('Helvetica', 8), foreground="#b06000")
        self.missing_data_label.pack(anchor="w", pady=(0, 2))

        # ── Target-variable outlier removal ───────────────────────────────────
        # Drop samples whose SELECTED PROPERTY value (e.g. clay %) is a
        # statistical outlier.  This is distinct from the "Spectral Outlier
        # Removal" preprocessing method above (which flags anomalous *spectra*):
        # here the outlier test runs on the target y and its paired spectra are
        # removed together, just before the train/test split.
        ttk.Separator(preprocess_frame, orient='horizontal').pack(fill='x', pady=(8, 4))
        self.target_outlier_var = tk.BooleanVar(value=False)
        self.target_outlier_method_var = tk.StringVar(value=TARGET_OUTLIER_METHODS[0])
        self.target_outlier_threshold_var = tk.StringVar(value="2.5")
        ttk.Checkbutton(
            preprocess_frame,
            text="Target Variable Outlier Removal",
            variable=self.target_outlier_var,
            command=self._sync_target_outlier_state,
        ).pack(anchor="w", pady=(2, 0))
        self.target_outlier_row = ttk.Frame(preprocess_frame)
        self.target_outlier_row.pack(fill="x", pady=2)
        ttk.Label(self.target_outlier_row, text="Method:").pack(side="left")
        self.target_outlier_method_combo = ttk.Combobox(
            self.target_outlier_row, textvariable=self.target_outlier_method_var,
            values=TARGET_OUTLIER_METHODS, state="disabled", width=8)
        self.target_outlier_method_combo.pack(side="left", padx=2)
        ttk.Label(self.target_outlier_row, text="Threshold:").pack(side="left", padx=(6, 0))
        self.target_outlier_threshold_entry = ttk.Entry(
            self.target_outlier_row, textvariable=self.target_outlier_threshold_var,
            width=6, state="disabled")
        self.target_outlier_threshold_entry.pack(side="left", padx=2)
        ttk.Label(preprocess_frame,
                  text="Removes samples whose selected property value is an "
                       "outlier (z-score: std-devs from mean; IQR: Tukey-fence "
                       "multiplier).",
                  wraplength=230, foreground="#666666",
                  font=('Helvetica', 8, 'italic')).pack(anchor="w", pady=(0, 2))

        # ===== Model selection group — a Single/Batch mode chooser =====

        self.model_frame = ttk.LabelFrame(step4, text="Model Selection", padding="10")
        self.model_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Mode selector: Single vs Batch (mutually exclusive radios) ---
        ttk.Label(self.model_frame, text="Mode:",
                  font=('Helvetica', 9, 'bold')).pack(anchor="w")
        mode_row = ttk.Frame(self.model_frame)
        mode_row.pack(fill="x", expand=True, pady=2)

        # `model_mode_var` drives the UI; `batch_mode_var` (kept for every existing
        # read in analysis_mixin / uiflow_mixin) is derived from it in _on_mode_change.
        self.model_mode_var = tk.StringVar(value="Single")
        self.batch_mode_var = tk.BooleanVar(value=False)
        ttk.Radiobutton(mode_row, text="Single", value="Single",
                        variable=self.model_mode_var,
                        command=self._on_mode_change).pack(side="left", padx=2)
        ttk.Radiobutton(mode_row, text="Batch", value="Batch",
                        variable=self.model_mode_var,
                        command=self._on_mode_change).pack(side="left", padx=2)

        # --- Batch-only controls (enabled only in Batch mode) ---
        self.auto_select_best_var = tk.BooleanVar()
        self.auto_select_checkbox = ttk.Checkbutton(self.model_frame, text="Auto-Select Best",
                       variable=self.auto_select_best_var, state='disabled')
        self.auto_select_checkbox.pack(anchor="w", pady=2)

        self.model_selection_btn = ttk.Button(self.model_frame, text="Select Algorithms",
                                             command=self.select_models, state='disabled', width=25)
        self.model_selection_btn.pack(pady=2, fill="x", expand=True)

        ttk.Separator(self.model_frame, orient='horizontal').pack(fill='x', pady=(8, 4))

        # --- Single-mode model configuration (enabled only in Single mode) ---
        ttk.Label(self.model_frame, text="Select Algorithm:").pack(anchor="w")
        self.model_var = tk.StringVar(value="PLS-R")
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, 
                                           values=list(self.model_params.keys()), 
                                           state='readonly', width=23)
        self.model_combobox.pack(fill="x", pady=2)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Model parameters will be created by create_scrollable_params_frame()
        self.create_scrollable_params_frame()

        # ── Hyperparameter tuning (Optuna) ────────────────────────────────────
        # Models are first evaluated on their default parameters; when this is
        # enabled the winning / selected model is re-fit with Optuna-optimized
        # hyperparameters and the best settings are reported.
        tune_frame = ttk.Frame(self.model_frame)
        tune_frame.pack(fill="x", pady=(6, 0))
        self.tune_hyperparams_var = tk.BooleanVar(value=False)
        self.tune_checkbox = ttk.Checkbutton(
            tune_frame, text="⚙ Tune Hyperparameters (Optuna)",
            variable=self.tune_hyperparams_var,
            command=self._on_tune_toggle)
        self.tune_checkbox.pack(anchor="w")

        tune_opts = ttk.Frame(self.model_frame)
        tune_opts.pack(fill="x")
        ttk.Label(tune_opts, text="Trials:", width=8).pack(side="left")
        self.tune_trials_var = tk.StringVar(value="30")
        self.tune_trials_entry = ttk.Entry(tune_opts, textvariable=self.tune_trials_var,
                                           width=6, state='disabled')
        self.tune_trials_entry.pack(side="left", padx=2)
        ttk.Label(tune_opts, text="CV folds:", width=9).pack(side="left", padx=(8, 0))
        self.tune_cv_var = tk.StringVar(value="3")
        self.tune_cv_entry = ttk.Entry(tune_opts, textvariable=self.tune_cv_var,
                                       width=5, state='disabled')
        self.tune_cv_entry.pack(side="left", padx=2)

        # Validation  →  Step ⑤ Validate
        validation_frame = ttk.LabelFrame(step5, text="Validation", padding="10")
        validation_frame.pack(fill="x", padx=10, pady=5)
        
        # Test Size and CV Strategy side by side
        row1 = ttk.Frame(validation_frame)
        row1.pack(fill="x", pady=2)
        
        ttk.Label(row1, text="Test Size:", width=15).pack(side="left")
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(row1, textvariable=self.test_size_var, width=8).pack(side="left", padx=2)
        
        ttk.Label(row1, text="CV Method:", width=12).pack(side="left", padx=(10,0))
        self.cv_strategy_var = tk.StringVar(value="None")
        cv_combo = ttk.Combobox(row1, textvariable=self.cv_strategy_var, 
                               values=["None", "K-Fold", "Leave-One-Out", "Leave-P-Out"],
                               state='readonly', width=10)
        cv_combo.pack(side="left", padx=2)
        cv_combo.bind("<<ComboboxSelected>>", self.on_cv_strategy_change)
        
        self.cv_params_frame = ttk.Frame(validation_frame)
        self.cv_params_frame.pack(fill="x", pady=2)
        
        # Resource Allocation  →  Step ⑤ Validate
        resource_frame = ttk.LabelFrame(step5, text="Resource Allocation", padding="10")
        resource_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(resource_frame, text="CPU Cores:").pack(anchor="w")
        self.cores_var = tk.StringVar(value="1")
        ttk.Combobox(resource_frame, textvariable=self.cores_var,
                    values=[str(i) for i in range(1, multiprocessing.cpu_count() + 1)], 
                    width=23, state='readonly').pack(fill="x", pady=2)
        
        # Spectral Configuration — laid out as a logical top-to-bottom sequence:
        #   ① Spectral domain  →  ② Wavelength range  →  ③ Resampling (reveals
        #   its sensor / method / spacing options only when turned on).
        spectral_frame = ttk.LabelFrame(step2, text="Spectral Configuration", padding="10")
        spectral_frame.pack(fill="x", padx=10, pady=5)

        _step_font = ('Helvetica', 9, 'bold')

        # ── ① Spectral domain ────────────────────────────────────────────────
        ttk.Label(spectral_frame, text="① Spectral domain", font=_step_font,
                  foreground="#0066cc").pack(anchor="w")
        domain_row = ttk.Frame(spectral_frame)
        domain_row.pack(fill="x", pady=(0, 2))
        ttk.Label(domain_row, text="Domain:", width=15).pack(side="left")
        self.spectral_domain_var = tk.StringVar(value="VSWIR")
        domain_combo = ttk.Combobox(domain_row,
                                    textvariable=self.spectral_domain_var,
                                    values=["VSWIR (350–2500 nm)",
                                            "VSWIR+LWIR (350–14000 nm)",
                                            "LWIR (7500–14000 nm)"],
                                    width=22, state='readonly')
        domain_combo.pack(side="left", padx=2)
        domain_combo.bind("<<ComboboxSelected>>", self._on_domain_change)

        # Wavelength unit display
        unit_row = ttk.Frame(spectral_frame)
        unit_row.pack(fill="x", pady=(0, 2))
        ttk.Label(unit_row, text="Wavelength unit:", width=15).pack(side="left")
        self.wl_unit_label = ttk.Label(unit_row, text="nm", foreground="#0066cc",
                                       font=('Helvetica', 9, 'bold'))
        self.wl_unit_label.pack(side="left")

        # LWIR note (emissivity) — kept next to the domain it refers to.
        note_holder = ttk.Frame(spectral_frame)
        note_holder.pack(fill="x")
        self.lwir_note_label = ttk.Label(note_holder,
                                          text="ⓘ LWIR measures emissivity, not reflectance.",
                                          foreground="#888888", font=('Helvetica', 8, 'italic'),
                                          wraplength=260)
        # Only visible when LWIR domain selected (toggled in _on_domain_change)

        # ── ② Wavelength range ───────────────────────────────────────────────
        ttk.Label(spectral_frame, text="② Wavelength range", font=_step_font,
                  foreground="#0066cc").pack(anchor="w", pady=(6, 0))
        row2 = ttk.Frame(spectral_frame)
        row2.pack(fill="x", pady=(0, 2))
        self._min_wave_lbl = ttk.Label(row2, text="Min Wave (nm):", width=15)
        self._min_wave_lbl.pack(side="left")
        self.min_wave_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.min_wave_var, width=8).pack(side="left", padx=2)

        self._max_wave_lbl = ttk.Label(row2, text="Max Wave (nm):", width=12)
        self._max_wave_lbl.pack(side="left", padx=(10, 0))
        self.max_wave_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.max_wave_var, width=8).pack(side="left", padx=2)

        # ── ②b Exclude ranges (optional) ─────────────────────────────────────
        ttk.Label(spectral_frame, text="② b Exclude ranges (optional)", font=_step_font,
                  foreground="#0066cc").pack(anchor="w", pady=(6, 0))
        excl_row = ttk.Frame(spectral_frame)
        excl_row.pack(fill="x", pady=(0, 2))
        ttk.Label(excl_row, text="Omit (nm):", width=15).pack(side="left")
        self.exclude_ranges_var = tk.StringVar(value="")
        ttk.Entry(excl_row, textvariable=self.exclude_ranges_var, width=22).pack(
            side="left", padx=2, fill="x", expand=True)
        excl_btn_row = ttk.Frame(spectral_frame)
        excl_btn_row.pack(fill="x", pady=(0, 2))
        ttk.Button(excl_btn_row, text="+ H₂O bands", width=13,
                   command=lambda: self._append_exclude_preset(WATER_ABSORPTION_RANGES)
                   ).pack(side="left", padx=2)
        ttk.Button(excl_btn_row, text="+ Noisy edges", width=13,
                   command=lambda: self._append_exclude_preset(NOISY_EDGE_RANGES)
                   ).pack(side="left", padx=2)
        ttk.Button(excl_btn_row, text="Clear", width=6,
                   command=lambda: self.exclude_ranges_var.set("")
                   ).pack(side="left", padx=2)
        ttk.Label(spectral_frame,
                  text="e.g. 1350-1450, 1800-1960  (comma-separated lo-hi pairs)",
                  foreground="#888888", font=('Helvetica', 8, 'italic')).pack(anchor="w")

        # ── ③ Resampling ─────────────────────────────────────────────────────
        ttk.Label(spectral_frame, text="③ Resampling", font=_step_font,
                  foreground="#0066cc").pack(anchor="w", pady=(6, 0))
        row1 = ttk.Frame(spectral_frame)
        row1.pack(fill="x", pady=(0, 2))
        ttk.Label(row1, text="Resample bands:", width=15).pack(side="left")
        self.resampling_var = tk.StringVar(value="No")
        resampling_combo = ttk.Combobox(row1, textvariable=self.resampling_var,
                    values=["Yes", "No"], width=8, state='readonly')
        resampling_combo.pack(side="left", padx=2)
        # Always-visible reminder that binning is active — the binning controls
        # themselves are hidden with the resample options when Resample = No, so
        # this keeps the state discoverable.
        self.binning_status_label = ttk.Label(
            row1, text="", foreground="#0066cc",
            font=('Helvetica', 8, 'bold'))
        self.binning_status_label.pack(side="left", padx=(8, 0))

        # Sub-frame holding every resampling option; shown only when enabled so
        # the panel stays uncluttered when resampling is off.
        self.resample_options_frame = ttk.Frame(spectral_frame)

        # Target sensor (or "Custom" uniform grid)
        row1b = ttk.Frame(self.resample_options_frame)
        row1b.pack(fill="x", pady=(2, 2))
        ttk.Label(row1b, text="Target sensor:", width=15).pack(side="left")
        self.sensor_var = tk.StringVar(value="Custom")
        self.sensor_combo = ttk.Combobox(row1b, textvariable=self.sensor_var,
                    values=SENSOR_OPTIONS, width=14, state='readonly')
        self.sensor_combo.pack(side="left", padx=2)
        self.sensor_hint_label = ttk.Label(
            row1b, text="", foreground="#888888",
            font=('Helvetica', 8, 'italic'))
        self.sensor_hint_label.pack(side="left", padx=(6, 0))

        # Resampling method — six options: three interpolation orders and three
        # bandwidth-aware (FWHM-driven) methods.  See RESAMPLE_METHODS.
        row1c = ttk.Frame(self.resample_options_frame)
        row1c.pack(fill="x", pady=(2, 2))
        ttk.Label(row1c, text="Method:", width=15).pack(side="left")
        self.resample_method_var = tk.StringVar(value="Linear Interpolation")
        self.resample_method_combo = ttk.Combobox(
            row1c, textvariable=self.resample_method_var,
            values=RESAMPLE_METHODS, width=22, state='readonly')
        self.resample_method_combo.pack(side="left", padx=2)

        # FWHM / SRF upload — only shown for a Custom sensor with a bandwidth-aware
        # method (predefined sensors carry their own FWHM globally).
        self.fwhm_upload_frame = ttk.Frame(self.resample_options_frame)
        fwhm_row = ttk.Frame(self.fwhm_upload_frame)
        fwhm_row.pack(fill="x", pady=(2, 0))
        ttk.Button(fwhm_row, text="📄 Load FWHM CSV", width=16,
                   command=self.load_custom_fwhm).pack(side="left", padx=2)
        self.fwhm_status_label = ttk.Label(
            fwhm_row, text="using spacing as FWHM", foreground="#888888",
            font=('Helvetica', 8, 'italic'))
        self.fwhm_status_label.pack(side="left", padx=(6, 0))
        self.srf_upload_row = ttk.Frame(self.fwhm_upload_frame)
        self.srf_upload_row.pack(fill="x", pady=(2, 0))
        ttk.Button(self.srf_upload_row, text="📄 Load SRF CSV", width=16,
                   command=self.load_srf_table).pack(side="left", padx=2)
        self.srf_status_label = ttk.Label(
            self.srf_upload_row, text="no curve → Gaussian fallback",
            foreground="#888888", font=('Helvetica', 8, 'italic'))
        self.srf_status_label.pack(side="left", padx=(6, 0))

        # Spacing — only meaningful for the "Custom" uniform grid.
        row1d = ttk.Frame(self.resample_options_frame)
        row1d.pack(fill="x", pady=(2, 2))
        self._spacing_lbl = ttk.Label(row1d, text="Spacing (nm):", width=15)
        self._spacing_lbl.pack(side="left")
        self.spacing_var = tk.StringVar(value="10")
        self.spacing_entry = ttk.Entry(row1d, textvariable=self.spacing_var, width=8)
        self.spacing_entry.pack(side="left", padx=2)
        ttk.Label(row1d, text="(Custom grid only)", foreground="#888888",
                  font=('Helvetica', 8, 'italic')).pack(side="left", padx=(6, 0))

        # Binning — thin the resampled grid by keeping the centre band of every
        # `bin_size` group (resample → bin).  Only affects dense grids.
        row1e = ttk.Frame(self.resample_options_frame)
        row1e.pack(fill="x", pady=(2, 2))
        ttk.Label(row1e, text="Binning:", width=15).pack(side="left")
        self.binning_var = tk.StringVar(value="No")
        self.binning_combo = ttk.Combobox(row1e, textvariable=self.binning_var,
                    values=["Yes", "No"], width=8, state='readonly')
        self.binning_combo.pack(side="left", padx=2)
        self._bin_size_lbl = ttk.Label(row1e, text="Bin size:")
        self._bin_size_lbl.pack(side="left", padx=(8, 2))
        self.bin_size_var = tk.StringVar(value="30")
        self.bin_size_entry = ttk.Entry(row1e, textvariable=self.bin_size_var, width=6)
        self.bin_size_entry.pack(side="left", padx=2)
        self.binning_hint_label = ttk.Label(
            row1e, text="", foreground="#888888",
            font=('Helvetica', 8, 'italic'))
        self.binning_hint_label.pack(side="left", padx=(6, 0))

        # Methods that need a per-band FWHM (bandwidth-aware family).
        _fwhm_methods = {"Gaussian SRF", "Empirical SRF", "Band Averaging"}

        def _on_resampling_change(*_):
            resampling_on = self.resampling_var.get() == "Yes"
            sensor_sel = self.sensor_var.get()
            method_sel = self.resample_method_var.get()
            needs_fwhm = method_sel in _fwhm_methods
            # Reveal / hide the whole options block with the Yes/No toggle.
            if resampling_on:
                self.resample_options_frame.pack(fill="x", pady=(0, 2))
            else:
                self.resample_options_frame.pack_forget()
            # Spacing applies only to the uniform "Custom" grid.
            use_spacing = resampling_on and sensor_sel == "Custom"
            self.spacing_entry.config(state='normal' if use_spacing else 'disabled')
            if sensor_sel == "Custom":
                self.sensor_hint_label.config(
                    text=f"uniform {self.spacing_var.get()} nm grid")
            else:
                self.sensor_hint_label.config(text=f"{sensor_sel} band centres")
            # FWHM / SRF upload row: only for a Custom sensor with a FWHM method.
            show_fwhm = resampling_on and sensor_sel == "Custom" and needs_fwhm
            if show_fwhm:
                self.fwhm_upload_frame.pack(fill="x", pady=(0, 2))
            else:
                self.fwhm_upload_frame.pack_forget()
            # The SRF-curve upload only matters for Empirical SRF.
            if method_sel == "Empirical SRF":
                self.srf_upload_row.pack(fill="x", pady=(2, 0))
            else:
                self.srf_upload_row.pack_forget()
            # Reflect current FWHM source in the status label.
            if self.custom_fwhm_table is not None:
                n = len(self.custom_fwhm_table[0])
                self.fwhm_status_label.config(text=f"FWHM: {n} custom bands loaded",
                                              foreground="#0066cc")
            else:
                self.fwhm_status_label.config(text="using spacing as FWHM",
                                              foreground="#888888")
            # Bin size only editable when binning is on.
            binning_on = self.binning_var.get() == "Yes"
            self.bin_size_entry.config(state='normal' if binning_on else 'disabled')
            if binning_on:
                self.binning_hint_label.config(
                    text=f"keep 1 of every {self.bin_size_var.get()} bands")
            else:
                self.binning_hint_label.config(text="")
            # Always-visible status (binning is independent of the resample toggle).
            self.binning_status_label.config(
                text=(f"⤵ Binning ON (1/{self.bin_size_var.get()})"
                      if binning_on else ""))
        self._on_resampling_change = _on_resampling_change
        self.resampling_var.trace_add("write", _on_resampling_change)
        self.sensor_var.trace_add("write", _on_resampling_change)
        self.resample_method_var.trace_add("write", _on_resampling_change)
        self.spacing_var.trace_add("write", _on_resampling_change)
        self.binning_var.trace_add("write", _on_resampling_change)
        self.bin_size_var.trace_add("write", _on_resampling_change)
        _on_resampling_change()  # apply initial state

        # Action Buttons  →  Step ⑥ Run
        action_frame = ttk.LabelFrame(step6, text="Analysis", padding="10")
        action_frame.pack(fill="x", padx=10, pady=5)
        
        # All action buttons side by side
        action_row = ttk.Frame(action_frame)
        action_row.pack(fill="x", pady=2)
        
        self.run_analysis_btn = ttk.Button(action_row, text="▶ Run", 
                  command=self.start_analysis, width=10)
        self.run_analysis_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        self.batch_run_btn = ttk.Button(action_row, text="⚡ Batch", 
                  command=self.run_batch_analysis, width=10, state='disabled')
        self.batch_run_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        ttk.Button(action_row, text="🔄 Reset", 
                  command=self.reset_gui, width=10).pack(side="left", padx=2, expand=True, fill="x")

        # Data Tools section  →  Step ⑥ Run
        tools_frame = ttk.LabelFrame(step6, text="Data Tools", padding="10")
        tools_frame.pack(fill="x", padx=10, pady=5)

        tools_row = ttk.Frame(tools_frame)
        tools_row.pack(fill="x", pady=2)
        ttk.Button(tools_row, text="🎲 Randomize",
                   command=self.show_data_randomization, width=12
                   ).pack(side="left", padx=2, expand=True, fill="x")
        ttk.Button(tools_row, text="🔬 Harmonize",
                   command=self.show_spectral_harmonization, width=12
                   ).pack(side="left", padx=2, expand=True, fill="x")
        
        # ── Step ⑦ Apply — how a trained model reaches new data ───────────────
        ttk.Label(step7,
                  text="A trained model + its resampling is re-applied here to a "
                       "hyperspectral image or an unknown tabular dataset. Save or "
                       "load a model, then predict.",
                  wraplength=420, foreground="#555555",
                  font=('Helvetica', 8, 'italic')).pack(anchor="w", padx=10, pady=(8, 2))

        # Model Management  →  Step ⑦ Apply
        model_mgmt_frame = ttk.LabelFrame(step7, text="Model Management", padding="10")
        model_mgmt_frame.pack(fill="x", padx=10, pady=5)
        
        # All model management buttons side by side
        mgmt_row = ttk.Frame(model_mgmt_frame)
        mgmt_row.pack(fill="x", pady=2)
        
        self.save_model_btn = ttk.Button(mgmt_row, text="💾 Save", 
                                        command=self.save_model, state='disabled', width=10)
        self.save_model_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        self.load_model_btn = ttk.Button(mgmt_row, text="📂 Load", 
                                        command=self.load_model, width=10)
        self.load_model_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        self.view_model_btn = ttk.Button(mgmt_row, text="ℹ View", 
                                        command=self.view_model, state='disabled', width=10)
        self.view_model_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        # ── Tabular Prediction  →  Step ⑦ Apply ─────────────────────────────
        tabular_frame = ttk.LabelFrame(step7, text="Tabular Prediction", padding="10")
        tabular_frame.pack(fill="x", padx=10, pady=5)

        self.apply_tabular_var = tk.BooleanVar()
        ttk.Checkbutton(tabular_frame, text="Apply on Tabular Data",
                        variable=self.apply_tabular_var,
                        command=self.toggle_tabular_options).pack(anchor="w", pady=2)

        self.export_resampled_tabular_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tabular_frame,
            text="Export resampled data (to validate resampling)",
            variable=self.export_resampled_tabular_var).pack(anchor="w", pady=1)

        tabular_row = ttk.Frame(tabular_frame)
        tabular_row.pack(fill="x", pady=2)

        self.load_tabular_btn = ttk.Button(tabular_row, text="📂 Load Excel",
                                           command=self.load_tabular_excel, state='disabled', width=12)
        self.load_tabular_btn.pack(side="left", padx=2, expand=True, fill="x")

        self.check_tabular_btn = ttk.Button(tabular_row, text="🔍 Check Excel",
                                            command=self.check_tabular_excel, state='disabled', width=12)
        self.check_tabular_btn.pack(side="left", padx=2, expand=True, fill="x")

        self.predict_tabular_btn = ttk.Button(tabular_row, text="📊 Predict",
                                              command=self.predict_unknown_csv, state='disabled', width=10)
        self.predict_tabular_btn.pack(side="left", padx=2, expand=True, fill="x")

        # ── Image Processing  →  Step ⑦ Apply ───────────────────────────────
        image_frame = ttk.LabelFrame(step7, text="Image Processing", padding="10")
        image_frame.pack(fill="x", padx=10, pady=5)
        
        self.apply_models_var = tk.BooleanVar()
        ttk.Checkbutton(image_frame, text="Apply on Image",
                       variable=self.apply_models_var,
                       command=self.toggle_image_options).pack(anchor="w", pady=2)

        # Prediction quality / validation options.
        self.mask_background_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            image_frame,
            text="Mask background / no-data pixels (skip NaN, zero & extreme)",
            variable=self.mask_background_var).pack(anchor="w", pady=1)
        self.export_resampled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            image_frame,
            text="Export resampled cube (to validate resampling)",
            variable=self.export_resampled_var).pack(anchor="w", pady=1)

        # Image buttons side by side
        image_row = ttk.Frame(image_frame)
        image_row.pack(fill="x", pady=2)
        
        self.load_image_btn = ttk.Button(image_row, text="📁 Load", 
                                       command=self.load_image, state='disabled', width=10)
        self.load_image_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        self.predict_image_btn = ttk.Button(image_row, text="🔮 Predict", 
                                          command=self.save_prediction, state='disabled', width=10)
        self.predict_image_btn.pack(side="left", padx=2, expand=True, fill="x")
        
        self.view_image_btn = ttk.Button(image_row, text="👁 View", 
                                        command=self.view_predicted_image, state='disabled', width=10)
        self.view_image_btn.pack(side="left", padx=2, expand=True, fill="x")

        ttk.Label(image_frame, text="Colormap:").pack(anchor="w", pady=(5,0))
        self.colormap_var = tk.StringVar(value="viridis")
        self.colormap_combo = ttk.Combobox(image_frame, textvariable=self.colormap_var,
                                          values=['viridis', 'plasma', 'inferno', 'magma',
                                                 'cividis', 'RdYlGn', 'Spectral'],
                                          state='readonly', width=23)
        self.colormap_combo.pack(fill="x", pady=2)
        # Changing the colormap re-renders the existing predicted image live in the
        # Visualization tab — it must NOT re-run the prediction.
        self.colormap_combo.bind("<<ComboboxSelected>>", self._on_colormap_change)

        # Initialise the Back/Next button states for the first tab.
        self._on_wizard_tab_changed()

        # ========== MODEL-DEVELOPMENT-FLOW OVERVIEW (docked below wizard) ======
        # A compact, always-visible strip summarising the configured pipeline for
        # every step (not just ⑦ Apply).  It updates live as parameters change;
        # on a fresh / reset session every stage is faded and lights up as set.
        # Full per-stage detail is available via "Large" / "300 DPI".
        flow_lf = ttk.LabelFrame(left_split, text="Model Development Flow", padding="4")
        left_split.add(flow_lf, weight=1)

        flow_btn_row = ttk.Frame(flow_lf)
        flow_btn_row.pack(fill="x", pady=(0, 3))
        ttk.Button(flow_btn_row, text="🔄 Refresh",
                   command=self.refresh_flow_preview
                   ).pack(side="left", padx=1, expand=True, fill="x")
        ttk.Button(flow_btn_row, text="🔍 Enlarge",
                   command=self.view_flow_large
                   ).pack(side="left", padx=1, expand=True, fill="x")
        ttk.Button(flow_btn_row, text="💾 Save Model Flow",
                   command=self.save_flow_figure
                   ).pack(side="left", padx=1, expand=True, fill="x")

        # Holder for the embedded matplotlib preview (populated on first draw).
        self.flow_preview_holder = ttk.Frame(flow_lf)
        self.flow_preview_holder.pack(fill="both", expand=True)
        self.flow_preview_canvas = None
        self.flow_preview_fig = None

        # Give the wizard the lion's share of the vertical space initially,
        # leaving a short strip for the flow overview (one-shot after render).
        def _set_vsash(event, _fired=[False]):
            if not _fired[0] and left_split.winfo_height() > 300:
                _fired[0] = True
                try:
                    left_split.sashpos(0, left_split.winfo_height() - 210)
                except Exception:
                    pass
        left_split.bind("<Configure>", _set_vsash)

        # ========== RIGHT PANEL: DISPLAY AREA ==========
        display_panel = ttk.Frame(main_paned)
        main_paned.add(display_panel, weight=1)
        
        # Progress Bar at bottom (pack first with side='bottom')
        progress_frame = ttk.Frame(display_panel)
        progress_frame.pack(fill="x", pady=5, side="bottom")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side="right", padx=5)
        
        # Display tabs (pack after progress bar to fill remaining space)
        self.display_notebook = ttk.Notebook(display_panel)
        self.display_notebook.pack(fill="both", expand=True)
        
        # Initialize plot storage
        self.analysis_plots = {}
        
        # Tab 1: Data View
        data_tab = ttk.Frame(self.display_notebook)
        self.display_notebook.add(data_tab, text="📊 Data View")
        
        # Create text widget for data display
        data_text_frame = ttk.Frame(data_tab)
        data_text_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        data_scrollbar_y = ttk.Scrollbar(data_text_frame)
        data_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        data_scrollbar_x = ttk.Scrollbar(data_text_frame, orient=tk.HORIZONTAL)
        data_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.data_display_text = tk.Text(data_text_frame, wrap=tk.NONE, font=('Courier', 10),
                                         yscrollcommand=data_scrollbar_y.set,
                                         xscrollcommand=data_scrollbar_x.set)
        self.data_display_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar_y.config(command=self.data_display_text.yview)
        data_scrollbar_x.config(command=self.data_display_text.xview)
        
        self.data_display_text.insert('1.0', 
            "Welcome to Paracuda III!\n\n"
            "To get started:\n"
            "1. Load an Excel file (File → Load Excel File)\n"
            "2. Select soil properties\n"
            "3. Configure model settings\n"
            "4. Run analysis\n\n"
            "Loaded data will be displayed here.")
        self.data_display_text.config(state='disabled')
        
        # Tab 2: Status/Log
        # Kept on self so a starting run can raise this tab by widget reference
        # (see _show_status_tab) rather than a brittle hard-coded tab index.
        status_tab = ttk.Frame(self.display_notebook)
        self.status_tab = status_tab
        self.display_notebook.add(status_tab, text="📝 Status & Log")
        
        status_frame = ttk.Frame(status_tab)
        status_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        status_scrollbar = ttk.Scrollbar(status_frame)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log console: keep the classic dark terminal look (banner colour is
        # always a dark shade across all palettes, so a light foreground fits).
        self.status_text = tk.Text(status_frame, wrap=tk.WORD, font=('Consolas', 9),
                                   yscrollcommand=status_scrollbar.set,
                                   bg=self._c('BNR', self._C_FALLBACK['CONSOLE_BG']),
                                   fg=self._C_FALLBACK['CONSOLE_FG'])
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.config(command=self.status_text.yview)
        
        self.status_text.insert('1.0', "System ready. Waiting for input...\n")
        
        # Tab 3: Visualization
        viz_tab = ttk.Frame(self.display_notebook)
        self.display_notebook.add(viz_tab, text="📈 Visualization")
        
        # Visualization controls
        viz_control_frame = ttk.Frame(viz_tab)
        viz_control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(viz_control_frame, text="Select Model/Image:").pack(side="left", padx=5)
        self.viz_model_var = tk.StringVar()
        self.viz_model_combo = ttk.Combobox(viz_control_frame, textvariable=self.viz_model_var, 
                                            state='readonly', width=40)
        self.viz_model_combo.pack(side="left", padx=5)
        self.viz_model_combo.bind("<<ComboboxSelected>>", self.display_selected_plot)

        # ── Reflectance-spectra view options ──────────────────────────────────
        # Purely a viewing control: choosing how many spectra to draw (and from
        # which samples) only redraws the Reflectance Spectra figure - it never
        # affects the data used for modelling.  The row is shown only while the
        # "Reflectance Spectra" plot is selected (see display_selected_plot).
        self.spectra_opts_frame = ttk.Frame(viz_tab)

        # Pack the action button FIRST, anchored right: Tk gives a side="right"
        # child its space before the left-hand children, so a narrow window
        # shrinks the option rows instead of clipping the button off-screen.
        ttk.Button(self.spectra_opts_frame, text="Update Plot",
                   command=self.refresh_reflectance_plot).pack(side="right", padx=(8, 5))

        # The options themselves stack on two short rows so the whole strip stays
        # narrow enough for small screens.
        opts = ttk.Frame(self.spectra_opts_frame)
        opts.pack(side="left", fill="x", expand=True)

        # Row 1 — how many spectra, and how they are picked.
        row_a = ttk.Frame(opts)
        row_a.pack(fill="x")
        ttk.Label(row_a, text="Spectra to show:").pack(side="left", padx=(5, 2))
        self.spectra_count_var = tk.StringVar(value="10")
        ttk.Entry(row_a, textvariable=self.spectra_count_var, width=5).pack(side="left")

        self.spectra_mode_var = tk.StringVar(value="Random")
        ttk.Radiobutton(row_a, text="Random (all samples)", value="Random",
                        variable=self.spectra_mode_var,
                        command=self._sync_spectra_range_state).pack(side="left", padx=(10, 2))
        ttk.Radiobutton(row_a, text="Within sample range", value="Range",
                        variable=self.spectra_mode_var,
                        command=self._sync_spectra_range_state).pack(side="left", padx=2)

        # Row 2 — the sample range (active only in "Within sample range" mode).
        row_b = ttk.Frame(opts)
        row_b.pack(fill="x", pady=(3, 0))
        ttk.Label(row_b, text="From:").pack(side="left", padx=(5, 2))
        self.spectra_from_var = tk.StringVar(value="1")
        self.spectra_from_entry = ttk.Entry(
            row_b, textvariable=self.spectra_from_var, width=6, state='disabled')
        self.spectra_from_entry.pack(side="left")
        ttk.Label(row_b, text="To:").pack(side="left", padx=(6, 2))
        self.spectra_to_var = tk.StringVar(value="")
        self.spectra_to_entry = ttk.Entry(
            row_b, textvariable=self.spectra_to_var, width=6, state='disabled')
        self.spectra_to_entry.pack(side="left")
        self.spectra_range_hint = ttk.Label(
            row_b, text="", foreground="#666666", font=('Helvetica', 8, 'italic'))
        self.spectra_range_hint.pack(side="left", padx=6)

        # Visualization display area
        self.viz_display_frame = ttk.Frame(viz_tab)
        self.viz_display_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initialize variables not already set
        self.optimize_components_var = tk.BooleanVar()
        
        # Start datetime update
        self.update_datetime()
        
        # Initialize preprocessing parameter variables
        self.smooth_window_var = tk.StringVar(value="11")
        self.smooth_poly_var = tk.StringVar(value="2")
        self.outlier_method_var = tk.StringVar(value="zscore")
        self.outlier_threshold_var = tk.StringVar(value="3.0")
        
        # Initialize parameter entries for default model
        self.on_model_change(None)
        self.on_cv_strategy_change(None)

        # Wire the always-visible flow chart to live GUI state (traces + poll)
        # and draw the initial (all-faded) preview.
        if hasattr(self, '_wire_flow_live_refresh'):
            self._wire_flow_live_refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Wizard helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _make_scroll_tab(self, title):
        """Add a scrollable tab to the wizard notebook and return its inner frame.

        Reproduces the scrollable-canvas pattern used for the old control panel so
        each step scrolls independently.  The mouse wheel is bound only while the
        pointer is over that tab's canvas.
        """
        tab = ttk.Frame(self.wizard_nb)
        self.wizard_nb.add(tab, text=title)

        canvas = tk.Canvas(tab, bg=self._c('WIN'), highlightthickness=0)
        if hasattr(self, '_scroll_canvases'):
            self._scroll_canvases.append(canvas)
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind('<Enter>', lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _wizard_step(self, delta):
        """Move the wizard selection by ``delta`` tabs, clamped to the ends."""
        tabs = self.wizard_nb.tabs()
        if not tabs:
            return
        cur = tabs.index(self.wizard_nb.select())
        nxt = min(max(cur + delta, 0), len(tabs) - 1)
        self.wizard_nb.select(nxt)

    def _on_wizard_tab_changed(self, event=None):
        """Enable/disable the Back/Next buttons at the wizard ends."""
        tabs = self.wizard_nb.tabs()
        if not tabs:
            return
        cur = tabs.index(self.wizard_nb.select())
        self._wizard_back_btn.config(state='normal' if cur > 0 else 'disabled')
        self._wizard_next_btn.config(
            state='normal' if cur < len(tabs) - 1 else 'disabled')
        # The flow chart is now a persistent pane visible for every step, so
        # refresh it on any tab change to reflect the latest configuration.
        if hasattr(self, 'refresh_flow_preview'):
            self.refresh_flow_preview()

    def _on_mode_change(self):
        """Sync the derived ``batch_mode_var`` from the Single/Batch radio, then
        apply the existing enable/disable logic."""
        self.batch_mode_var.set(self.model_mode_var.get() == "Batch")
        self.toggle_batch_mode()

    # ─────────────────────────────────────────────────────────────────────────
    # Resampling: exclude-range + custom FWHM/SRF helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _get_exclude_ranges(self):
        """Parse the exclude-ranges entry into a list of ``(lo, hi)`` tuples."""
        spec = self.exclude_ranges_var.get() if hasattr(self, 'exclude_ranges_var') else ""
        self.exclude_ranges = parse_exclude_ranges(spec)
        return self.exclude_ranges

    def _append_exclude_preset(self, ranges):
        """Append a preset list of ranges to the exclude-ranges entry."""
        existing = self.exclude_ranges_var.get().strip().rstrip(",").strip()
        addition = ", ".join(f"{int(lo)}-{int(hi)}" for lo, hi in ranges)
        self.exclude_ranges_var.set(f"{existing}, {addition}" if existing else addition)

    def load_custom_fwhm(self):
        """Load a custom sensor's (centre, FWHM) table from a CSV for the Custom
        sensor's bandwidth-aware resampling."""
        path = filedialog.askopenfilename(
            title="Load FWHM CSV (columns: centre_nm, fwhm_nm)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.custom_fwhm_table = load_fwhm_csv(path)
        except Exception as e:
            messagebox.showerror("FWHM CSV Error", str(e))
            return
        n = len(self.custom_fwhm_table[0])
        self.status_text.insert(
            tk.END, f"Loaded custom FWHM: {n} bands from {os.path.basename(path)}\n")
        self.status_text.see(tk.END)
        if hasattr(self, '_on_resampling_change'):
            self._on_resampling_change()

    def load_srf_table(self):
        """Load per-band empirical spectral-response curves from a CSV for the
        'Empirical SRF' resampling method."""
        path = filedialog.askopenfilename(
            title="Load SRF CSV (columns: center, wavelength, response)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.srf_table = load_srf_csv(path)
        except Exception as e:
            messagebox.showerror("SRF CSV Error", str(e))
            return
        n = len(self.srf_table)
        self.status_text.insert(
            tk.END, f"Loaded empirical SRF: {n} band curves from {os.path.basename(path)}\n")
        self.status_text.see(tk.END)
        if hasattr(self, 'srf_status_label'):
            self.srf_status_label.config(text=f"SRF: {n} curves loaded",
                                         foreground="#0066cc")

    def create_scrollable_params_frame(self):
        # Remove old frame if exists
        if hasattr(self, 'params_canvas'):
            self.params_canvas.destroy()
        if hasattr(self, 'params_frame'):
            self.params_frame.destroy()
        if hasattr(self, 'params_scrollbar'):
            self.params_scrollbar.destroy()
        if hasattr(self, 'params_container'):
            self.params_container.destroy()

        # Create frame container for canvas and scrollbar
        self.params_container = ttk.Frame(self.model_frame)
        self.params_container.pack(fill="both", expand=True, pady=(4, 0))

        # Canvas for scrollability - dynamic height based on content
        # bg set to the theme colour so exposed areas don't flash black on restore.
        self.params_canvas = tk.Canvas(self.params_container, highlightthickness=0, bd=0, bg=self._c('WIN'))
        self.params_canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        self.params_scrollbar = ttk.Scrollbar(self.params_container, orient="vertical", command=self.params_canvas.yview)
        self.params_scrollbar.pack(side="right", fill="y")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        
        # Frame inside canvas for parameters
        self.params_frame = ttk.Frame(self.params_canvas)
        self.canvas_window = self.params_canvas.create_window((0, 0), window=self.params_frame, anchor='nw')

        # Populate parameters
        self.param_vars.clear()
        self.param_widgets.clear()
        self.param_widgets_dict = {}  # Track widgets for batch mode disable
        model = self.model_var.get()
        params = self.model_params[model]["params"]
        
        # Calculate layout - 2 columns of parameter pairs
        row = 0
        col = 0
        max_params_per_row = 2
        
        for pname, pinfo in params.items():
            # Label
            ttk.Label(self.params_frame, text=pinfo.get("label", pname), width=13).grid(
                row=row, column=col*2, sticky=tk.W, padx=(2, 1), pady=2)
            
            # Input widget
            if pinfo["type"] == "entry":
                var = tk.StringVar(value=pinfo.get("default", ""))
                widget = ttk.Entry(self.params_frame, textvariable=var, width=8)
            elif pinfo["type"] == "combobox":
                var = tk.StringVar(value=pinfo.get("default", ""))
                widget = ttk.Combobox(self.params_frame, textvariable=var, 
                                    values=pinfo.get("values", []), width=8, state='readonly')
            
            widget.grid(row=row, column=col*2+1, sticky=tk.W, padx=(1, 6), pady=2)
            self.param_vars[pname] = var
            self.param_widgets[pname] = widget
            self.param_widgets_dict[pname] = {'widget': widget, 'type': pinfo["type"]}
            
            # Move to next position
            col += 1
            if col >= max_params_per_row:
                col = 0
                row += 1

        # Add optimize components checkbox for PCA/PLSR
        if model in ["PLS-R", "PCA"]:
            # Add some spacing
            if col > 0:  # If we're not at the start of a new row
                row += 1
                col = 0
            
            ttk.Separator(self.params_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
            row += 1
            
            # Reset the variable to ensure it's properly bound
            if not hasattr(self, 'optimize_components_var'):
                self.optimize_components_var = tk.BooleanVar()
            
            self.optimize_components_checkbox = ttk.Checkbutton(self.params_frame, text="Optimize Components", 
                           variable=self.optimize_components_var)
            self.optimize_components_checkbox.grid(
                row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        else:
            # Reset to False for non-PCA/PLSR models
            if hasattr(self, 'optimize_components_var'):
                self.optimize_components_var.set(False)
            if hasattr(self, 'optimize_components_checkbox'):
                self.optimize_components_checkbox = None

        # Configure scrolling
        def configure_scroll_region(event=None):
            self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
            
            # Calculate required height (max 200px, min 100px)
            bbox = self.params_canvas.bbox("all")
            if bbox:
                content_height = bbox[3] - bbox[1]
                canvas_height = min(max(content_height + 10, 100), 200)
                self.params_canvas.configure(height=canvas_height)
                
                # Show/hide scrollbar based on content
                if content_height > canvas_height - 15:
                    if not self.params_scrollbar.winfo_manager():
                        self.params_scrollbar.pack(side="right", fill="y")
                else:
                    self.params_scrollbar.pack_forget()

        def configure_canvas_width(event):
            # Make the frame width match the canvas width
            canvas_width = event.width
            self.params_canvas.itemconfig(self.canvas_window, width=canvas_width)

        # Bind events
        self.params_frame.bind("<Configure>", configure_scroll_region)
        self.params_canvas.bind("<Configure>", configure_canvas_width)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            self.params_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            self.params_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            self.params_canvas.unbind_all("<MouseWheel>")
        
        self.params_canvas.bind('<Enter>', bind_mousewheel)
        self.params_canvas.bind('<Leave>', unbind_mousewheel)
        
        # Initial configuration
        self.params_frame.update_idletasks()
        configure_scroll_region()
    
    def on_cv_strategy_change(self, event=None):
        # Clear existing CV parameter widgets
        for widget in self.cv_params_frame.winfo_children():
            widget.destroy()
        
        cv_strategy = self.cv_strategy_var.get()
        
        if cv_strategy == "K-Fold":
            ttk.Label(self.cv_params_frame, text="K (folds):").grid(row=0, column=0, sticky=tk.W, padx=2)
            self.k_folds_var = tk.StringVar(value="5")
            ttk.Entry(self.cv_params_frame, textvariable=self.k_folds_var, width=8).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)
            
            ttk.Label(self.cv_params_frame, text="Shuffle:").grid(row=0, column=2, sticky=tk.W, padx=2)
            self.shuffle_var = tk.StringVar(value="True")
            ttk.Combobox(self.cv_params_frame, textvariable=self.shuffle_var, 
                        values=["True", "False"], state='readonly', width=8).grid(row=0, column=3, sticky=(tk.W, tk.E), padx=2)
        
        elif cv_strategy == "Leave-One-Out":
            # LOOCV requires no parameters
            ttk.Label(self.cv_params_frame, text="(No parameters required)").grid(row=0, column=0, sticky=tk.W, padx=2)
        
        elif cv_strategy == "Leave-P-Out":
            ttk.Label(self.cv_params_frame, text="P (samples):").grid(row=0, column=0, sticky=tk.W, padx=2)
            self.p_out_var = tk.StringVar(value="2")
            ttk.Entry(self.cv_params_frame, textvariable=self.p_out_var, width=8).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)
    
    def on_model_change(self, event=None):
        self.create_scrollable_params_frame()

        # Hide/show optimize components checkbox in validation frame based on model
        model = self.model_var.get()
        if hasattr(self, 'optimize_components_checkbox') and self.optimize_components_checkbox is not None:
            if model in ["PLS-R", "PCA"]:
                self.optimize_components_checkbox.grid()
            else:
                self.optimize_components_checkbox.grid_remove()

    def _on_tune_toggle(self):
        """Enable/disable the Optuna trials + CV-folds entries with the checkbox."""
        state = 'normal' if self.tune_hyperparams_var.get() else 'disabled'
        self.tune_trials_entry.config(state=state)
        self.tune_cv_entry.config(state=state)

    def _get_gui_model_params(self, model_name):
        """Build the complete default parameter dict for a model from its GUI
        definitions (used as the base that Optuna overrides selected keys of)."""
        params = {}
        for pname, pinfo in self.model_params.get(model_name, {}).get("params", {}).items():
            params[pname] = pinfo.get("default", "")
        return params

    def on_preprocess_change(self, event=None):
        """Update preprocessing parameter widgets based on selected method"""
        # Clear existing widgets
        for widget in self.preprocess_params_frame.winfo_children():
            widget.destroy()
        
        method = self.preprocess_var.get()
        
        if method == "Smoothing":
            # Window Length and Poly Order side by side
            row1 = ttk.Frame(self.preprocess_params_frame)
            row1.pack(fill="x", pady=2)
            
            ttk.Label(row1, text="Window Length:", width=15).pack(side="left")
            ttk.Entry(row1, textvariable=self.smooth_window_var, width=8).pack(side="left", padx=2)
            
            ttk.Label(row1, text="Poly Order:", width=12).pack(side="left", padx=(10,0))
            ttk.Entry(row1, textvariable=self.smooth_poly_var, width=8).pack(side="left", padx=2)
        
        elif method == "Spectral Outlier Removal":
            # Method and Threshold side by side
            row1 = ttk.Frame(self.preprocess_params_frame)
            row1.pack(fill="x", pady=2)
            
            ttk.Label(row1, text="Method:", width=15).pack(side="left")
            ttk.Combobox(row1, textvariable=self.outlier_method_var, 
                        values=["zscore", "iqr"], width=8, state='readonly').pack(side="left", padx=2)
            
            ttk.Label(row1, text="Threshold:", width=12).pack(side="left", padx=(10,0))
            ttk.Entry(row1, textvariable=self.outlier_threshold_var, width=8).pack(side="left", padx=2)

        elif method == "Baseline Correction":
            row1 = ttk.Frame(self.preprocess_params_frame)
            row1.pack(fill="x", pady=2)
            ttk.Label(row1, text="Method:", width=15).pack(side="left")
            if not hasattr(self, 'baseline_method_var'):
                self.baseline_method_var = tk.StringVar(value="linear")
            ttk.Combobox(row1, textvariable=self.baseline_method_var,
                         values=["linear", "polynomial"], width=10,
                         state='readonly').pack(side="left", padx=2)
            row2 = ttk.Frame(self.preprocess_params_frame)
            row2.pack(fill="x", pady=2)
            ttk.Label(row2, text="Degree (poly):", width=15).pack(side="left")
            if not hasattr(self, 'baseline_degree_var'):
                self.baseline_degree_var = tk.StringVar(value="3")
            ttk.Entry(row2, textvariable=self.baseline_degree_var, width=8).pack(side="left", padx=2)
    
    def toggle_tabular_options(self):
        state = 'normal' if self.apply_tabular_var.get() else 'disabled'
        self.load_tabular_btn.config(state=state)
        self.check_tabular_btn.config(state=state)
        # Predict only becomes active once data is loaded
        if not self.apply_tabular_var.get():
            self.predict_tabular_btn.config(state='disabled')
        elif self.tabular_df is not None:
            self.predict_tabular_btn.config(state='normal')
        else:
            self.predict_tabular_btn.config(state='disabled')

    def toggle_image_options(self):
        state = 'normal' if self.apply_models_var.get() else 'disabled'
        self.load_image_btn.config(state=state)
        # The colormap picker is a fixed-choice list — keep it read-only (never
        # freely editable) when enabled so users can only pick valid colormaps.
        self.colormap_combo.config(state='readonly' if self.apply_models_var.get() else 'disabled')
        # Predict and View buttons remain disabled until image is loaded and model is trained
        if not self.apply_models_var.get():
            self.predict_image_btn.config(state='disabled')
            self.view_image_btn.config(state='disabled')

    def _get_current_preprocess_kwargs(self):
        preprocess_kwargs = {}
        active_preprocess = self.preprocess_var.get()
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
        return preprocess_kwargs

    # ─────────────────────────────────────────────────────────────────────────
    # Spectral domain selector helper
    # ─────────────────────────────────────────────────────────────────────────
    def _on_domain_change(self, event=None):
        """Set min/max wavelength fields (clamped to loaded data) and update LWIR note."""
        sel = self.spectral_domain_var.get()
        if "VSWIR+LWIR" in sel:
            domain_min, domain_max = 350.0, 14000.0
            self.spectral_domain = "VSWIR+LWIR"
        elif "LWIR" in sel:
            domain_min, domain_max = 7500.0, 14000.0
            self.spectral_domain = "LWIR"
        else:
            domain_min, domain_max = 350.0, 2500.0
            self.spectral_domain = "VSWIR"

        # Clamp to actual data wavelengths when a file is loaded
        if self.wavelengths:
            data_wl = [float(w) for w in self.wavelengths]
            in_domain = [w for w in data_wl if domain_min <= w <= domain_max]
            if in_domain:
                self.min_wave_var.set(min(in_domain))
                self.max_wave_var.set(max(in_domain))
            else:
                # No data in selected domain -- leave fields blank to signal
                # that training cannot proceed with this domain selection
                self.min_wave_var.set("")
                self.max_wave_var.set("")
        else:
            # No file loaded yet -- also leave blank (no defaults to show)
            self.min_wave_var.set("")
            self.max_wave_var.set("")

        if hasattr(self, 'lwir_note_label'):
            if self.spectral_domain in ("LWIR", "VSWIR+LWIR"):
                self.lwir_note_label.pack(anchor="w", pady=(2, 0))
            else:
                self.lwir_note_label.pack_forget()

    def toggle_batch_mode(self):
        """Toggle batch mode controls"""
        state = 'normal' if self.batch_mode_var.get() else 'disabled'
        self.model_selection_btn.config(state=state)
        
        # Enable/disable Auto-Select checkbox based on batch mode
        self.auto_select_checkbox.config(state=state)
        if not self.batch_mode_var.get():
            self.auto_select_best_var.set(False)
        
        # Enable/disable Run Analysis button (opposite of batch mode)
        run_state = 'disabled' if self.batch_mode_var.get() else 'normal'
        self.run_analysis_btn.config(state=run_state)
        
        # Enable/disable Batch Run button (same as batch mode)
        self.batch_run_btn.config(state=state)
        
        # Keep Optimize Components (PLS-R/PCA) available in batch mode so component
        # optimization can be applied to PLS-R during a batch run.  Previously the
        # checkbox was force-disabled and its value reset whenever batch mode was
        # on, which silently prevented PLS-R component optimization in batch.
        if hasattr(self, 'optimize_components_checkbox') and self.optimize_components_checkbox is not None:
            self.optimize_components_checkbox.config(state='normal')
            self.optimize_components_checkbox.grid()
        
        # When batch mode is enabled, disable single model selection
        if self.batch_mode_var.get():
            self.model_combobox.config(state='disabled')
            # Disable all parameter widgets in scrollable params frame
            if hasattr(self, 'param_widgets_dict'):
                for param_name, widget_info in self.param_widgets_dict.items():
                    if 'widget' in widget_info:
                        widget_info['widget'].config(state='disabled')
        else:
            self.model_combobox.config(state='readonly')
            # Re-enable parameter widgets
            if hasattr(self, 'param_widgets_dict'):
                for param_name, widget_info in self.param_widgets_dict.items():
                    if 'widget' in widget_info:
                        if widget_info.get('type') == 'combobox':
                            widget_info['widget'].config(state='readonly')
                        else:
                            widget_info['widget'].config(state='normal')

        # Re-evaluate Find Best Preprocessing state (needs >= 1 property selected)
        self._update_best_preprocess_state()
    
    def select_models(self):
        """Allow selection of multiple models for batch processing"""
        try:
            # Check if the dialog already exists and is open
            if hasattr(self, 'model_selection_window') and self.model_selection_window.winfo_exists():
                self.model_selection_window.lift()
                return
            
            self.model_selection_window = tk.Toplevel(self)
            self.model_selection_window.title("Select Models")
            self.model_selection_window.geometry("400x350")
            
            # Instructions
            ttk.Label(self.model_selection_window, 
                     text="Select one or more models to run:",
                     font=('Helvetica', 10, 'bold')).pack(pady=5)
            
            # Frame for listbox and scrollbar
            list_frame = ttk.Frame(self.model_selection_window)
            list_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Listbox with MULTIPLE selection
            model_options = list(self.model_params.keys())
            listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
            for model in model_options:
                listbox.insert(tk.END, model)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)
            
            # Buttons frame
            btn_frame = ttk.Frame(self.model_selection_window)
            btn_frame.pack(pady=5)
            
            def on_select():
                selections = listbox.curselection()
                if selections:
                    self.selected_models = [model_options[i] for i in selections]
                    model_str = ", ".join(self.selected_models)
                    self.status_text.insert(tk.END, f"Selected models: {model_str}\n")
                    self.status_text.insert(tk.END, "="*50 + "\n")
                    self.status_text.see(tk.END)
                    self.model_selection_window.destroy()
            
            def select_all():
                listbox.select_set(0, tk.END)
            
            def clear_all():
                listbox.selection_clear(0, tk.END)
            
            ttk.Button(btn_frame, text="Select All", command=select_all).grid(row=0, column=0, padx=5)
            ttk.Button(btn_frame, text="Clear All", command=clear_all).grid(row=0, column=1, padx=5)
            ttk.Button(btn_frame, text="Confirm Selection", command=on_select).grid(row=0, column=2, padx=5)

            def on_close():
                self.model_selection_window.destroy()
                if hasattr(self, 'model_selection_window'):
                    del self.model_selection_window
            
            self.model_selection_window.protocol("WM_DELETE_WINDOW", on_close)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select models: {str(e)}")
    
    def _update_best_preprocess_state(self):
        """Enable 'Find Best Preprocessing' whenever at least one property is selected.

        The search runs per property, so several properties can be searched in one
        go - each gets its own winning preprocessing + model combination.
        """
        if not hasattr(self, 'find_best_preprocess_checkbox'):
            return
        have_props = bool(getattr(self, 'selected_properties', None))
        state = 'normal' if have_props else 'disabled'
        self.find_best_preprocess_checkbox.config(state=state)
        if not have_props:
            self.find_best_preprocess_var.set(False)
        self._sync_preprocess_combo_state()

    def _sync_preprocess_combo_state(self):
        """Disable the preprocessing-method dropdown while 'Find Best Preprocessing'
        is enabled, since the search chooses the method automatically."""
        if not hasattr(self, 'preprocess_combo'):
            return
        disabled = bool(self.find_best_preprocess_var.get())
        self.preprocess_combo.config(state='disabled' if disabled else 'readonly')

    def _sync_spectra_range_state(self):
        """Enable the From/To sample-range entries only in 'Within sample range' mode."""
        if not hasattr(self, 'spectra_from_entry'):
            return
        on = (self.spectra_mode_var.get() == "Range")
        state = 'normal' if on else 'disabled'
        self.spectra_from_entry.config(state=state)
        self.spectra_to_entry.config(state=state)

    def _sync_target_outlier_state(self):
        """Enable the target-outlier method/threshold controls only while the
        'Target Variable Outlier Removal' checkbox is ticked."""
        if not hasattr(self, 'target_outlier_method_combo'):
            return
        on = bool(self.target_outlier_var.get())
        self.target_outlier_method_combo.config(state='readonly' if on else 'disabled')
        self.target_outlier_threshold_entry.config(state='normal' if on else 'disabled')

    def _get_default_preprocess_kwargs(self, method):
        """Return safe default kwargs for a preprocessing method (no GUI dependency)."""
        if method == "Smoothing":
            return {'window_length': 11, 'polyorder': 2}
        if method == "Spectral Outlier Removal":
            return {'outlier_method': 'iqr', 'threshold': 3.0}
        if method == "Baseline Correction":
            return {'baseline_method': 'linear', 'degree': 3}
        return {}
    
    def _binning_kwargs(self):
        """Read the Resampling→Binning UI options as filter_wavelengths kwargs.

        Binning is independent of the resample toggle: when it is on it applies
        whether or not resampling is enabled (the controls live in the resampling
        options block for layout, but the setting persists and still thins the
        raw grid when Resample = No).  Returns a safe default bin_size on bad input.
        """
        apply_binning = (getattr(self, 'binning_var', None) is not None
                         and self.binning_var.get() == "Yes")
        try:
            bin_size = int(float(self.bin_size_var.get()))
        except (ValueError, AttributeError):
            bin_size = 30
        if bin_size < 1:
            bin_size = 30
        return {'apply_binning': apply_binning, 'bin_size': bin_size}
