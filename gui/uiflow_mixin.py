"""
Threading/UI-thread helpers, status/progress, plot list handling.

@author: Sharad Kumar Gupta
"""
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


class UIFlowMixin:
    
    def _on_ui_thread(self):
        return threading.get_ident() == getattr(self, '_main_thread_id', None)

    def _run_on_ui(self, callback, *args, **kwargs):
        """Run a small UI callback safely from worker threads."""
        if self._on_ui_thread():
            callback(*args, **kwargs)
        else:
            self.after(0, lambda: callback(*args, **kwargs))

    def _run_offloaded(self, fn, *args, **kwargs) -> Any:
        """Run a heavy, Tk-free compute call on a worker thread while keeping the Tk
        event loop pumping, so the window never shows "(Not Responding)".

        Tkinter is not thread-safe — only the main thread may touch Tk objects — so
        the *orchestration* code (which reads Tk vars, prints status, opens dialogs)
        stays on the main thread; only the numeric work (model.fit, Optuna tuning,
        cross-validation, pandas reads) is offloaded here.  Any callback passed into
        ``fn`` that touches the GUI must marshal via ``_run_on_ui``.
        """
        result = {}
        done = threading.Event()

        def _work():
            try:
                result['value'] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on caller side
                result['error'] = exc
            finally:
                done.set()

        threading.Thread(target=_work, daemon=True).start()

        # Pump the event loop so the GUI stays responsive (draggable, repainting)
        # while the worker runs.  Run/Batch buttons are disabled via set_busy_state,
        # so re-entrancy from this update() is prevented.
        while not done.is_set():
            try:
                self.update()
            except tk.TclError:
                break
            done.wait(0.02)

        if 'error' in result:
            raise result['error']
        return result.get('value')

    def _pin_geometry(self):
        """Freeze the window size for the duration of a run so embedding plots or
        adding widgets cannot auto-resize the window."""
        try:
            self._locked_geometry = self.geometry()
        except tk.TclError:
            self._locked_geometry = None

    def _unpin_geometry(self):
        """Re-assert and release the pinned window size."""
        locked = getattr(self, '_locked_geometry', None)
        self._locked_geometry = None
        if locked:
            try:
                self.geometry(locked)
            except tk.TclError:
                pass

    def append_status(self, message, see=True):
        def _append():
            self.status_text.insert(tk.END, message)
            if see:
                self.status_text.see(tk.END)
        self._run_on_ui(_append)

    def set_busy_state(self, busy):
        def _set():
            self._batch_running = busy
            state = 'disabled' if busy else 'normal'
            if hasattr(self, 'batch_run_btn'):
                self.batch_run_btn.config(state=state if self.batch_mode_var.get() else 'disabled')
            if hasattr(self, 'run_analysis_btn'):
                self.run_analysis_btn.config(state='disabled' if busy else ('disabled' if self.batch_mode_var.get() else 'normal'))
        self._run_on_ui(_set)

    def update_progress(self, value, message=""):
        if not self._on_ui_thread():
            self._run_on_ui(self.update_progress, value, message)
            return
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")
        if message:
            self.status_text.insert(tk.END, f"{message}\n")
            self.status_text.see(tk.END)
        # Repaint the progress widgets without re-entering the full event loop
        # (compute now runs on a worker thread; the main loop redraws on its own).
        self.update_idletasks()

    def reset_gui(self):
        """Return the whole tool to its first-launch state.

        Every user-facing control is set back to the exact default it holds when
        the app starts, all loaded data / trained-model state is cleared, and the
        dependent widget enable/disable logic is re-run through the normal
        handlers so the panels look identical to a fresh launch.
        """
        try:
            # Suspend flow-chart traces so resetting vars below does not
            # re-light stages we are about to clear.
            self._flow_suspend_trace = True

            # ── Step ① Data ─────────────────────────────────────────────
            self.export_stats_var.set(False)
            self.comp_transform_var.set("None")

            # ── Step ② Spectral configuration ───────────────────────────
            self.spectral_domain_var.set("VSWIR")
            self.min_wave_var.set("")
            self.max_wave_var.set("")
            self.exclude_ranges_var.set("")
            self.resampling_var.set("No")
            self.sensor_var.set("Custom")
            self.resample_method_var.set("Linear Interpolation")
            self.spacing_var.set("10")
            self.binning_var.set("No")
            self.bin_size_var.set("30")

            # ── Step ③ Preprocessing ────────────────────────────────────
            self.preprocess_var.set("No Preprocessing")
            self.find_best_preprocess_var.set(False)
            self.missing_data_var.set(MISSING_DATA_METHODS[0])
            self.smooth_window_var.set("11")
            self.smooth_poly_var.set("2")
            self.outlier_method_var.set("zscore")
            self.outlier_threshold_var.set("3.0")
            if hasattr(self, 'target_outlier_var'):
                self.target_outlier_var.set(False)
                self.target_outlier_method_var.set(TARGET_OUTLIER_METHODS[0])
                self.target_outlier_threshold_var.set("2.5")
                self._sync_target_outlier_state()

            # ── Step ④ Model selection ──────────────────────────────────
            self.model_mode_var.set("Single")
            self.batch_mode_var.set(False)
            self.auto_select_best_var.set(False)
            self.model_var.set("PLS-R")
            if hasattr(self, 'tune_hyperparams_var'):
                self.tune_hyperparams_var.set(False)
                self.tune_trials_var.set("30")
                self.tune_cv_var.set("3")

            # ── Step ⑤ Validation / resources ───────────────────────────
            self.test_size_var.set("0.2")
            self.cv_strategy_var.set("None")
            self.cores_var.set("1")
            self.optimize_components_var.set(False)

            # ── Step ⑦ Apply (tabular / image) ──────────────────────────
            self.apply_tabular_var.set(False)
            self.export_resampled_tabular_var.set(False)
            self.apply_models_var.set(False)
            self.mask_background_var.set(True)
            self.export_resampled_var.set(False)
            self.colormap_var.set("viridis")

            # ── Clear loaded data / trained-model state (fresh start) ───
            self.df = None
            self.wavelengths = None
            self.soil_properties = None
            self.input_filename = None
            self.output_filename = None
            self.trained_model = None
            self.model_basename = None
            self.scaler_X = None
            self.scaler_y = None
            self.image_data = None
            self.image_meta = None
            self.image_wavelengths = None
            self.image_fwhms = None
            self.image_path = None
            self.image_driver = None
            self.image_interleave = None
            self.training_input_wavelengths = None
            self.training_input_fwhms = None
            self.compositional_model = None
            self._last_result_timestamp = None
            self.predicted_image = None
            self.image_canvas = None
            self.selected_property = None
            self.selected_properties = []
            self.selected_models = []
            self.single_property_mode = False
            self.filtered_wavelengths = None
            self.new_wavelengths = None
            self.new_fwhms = None
            self.resample_method = "Linear Interpolation"
            self.custom_fwhm_table = None
            self.srf_table = None
            self.exclude_ranges = []
            self.wavelength_unit = "nm"
            self.spectral_domain = "VSWIR"
            self.model_preprocessing = "No Preprocessing"
            self.model_preprocessing_kwargs = {}
            self.pca_component = None
            self.tabular_df = None
            self.tabular_wl_nm = None
            self.tabular_wl_cols = None
            self.tabular_file_path = None

            # ── Progress / status ───────────────────────────────────────
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
            self.status_text.delete(1.0, tk.END)

            # ── Re-sync dependent widget states via the normal handlers ─
            if hasattr(self, 'wl_unit_label'):
                self.wl_unit_label.config(text="nm")
            if hasattr(self, 'missing_data_label'):
                self.missing_data_label.config(text="")
            self._on_mode_change()          # Single/Batch enable logic
            self._on_domain_change()        # domain → wavelength fields / LWIR note
            self.on_preprocess_change()     # rebuild preprocessing param frame (empty)
            self._sync_preprocess_combo_state()
            if hasattr(self, '_on_tune_toggle'):
                self._on_tune_toggle()
            if hasattr(self, '_on_resampling_change'):
                self._on_resampling_change()
            self.toggle_image_options()
            self.toggle_tabular_options()
            self.on_model_change(None)
            self.on_cv_strategy_change(None)

            # Model-management buttons back to first-launch state.
            self.save_model_btn.config(state='disabled')
            self.load_model_btn.config(state='normal')
            self.view_model_btn.config(state='disabled')

            # Clear visualization plots
            self.analysis_plots = {}
            self._reflectance_spectra_fig = None
            self.refresh_plot_list()
            # Clear visualization display
            if hasattr(self, 'viz_display_frame'):
                for widget in self.viz_display_frame.winfo_children():
                    widget.destroy()

            # Reflectance-spectra view controls back to their defaults.
            if hasattr(self, 'spectra_opts_frame'):
                self.spectra_count_var.set("10")
                self.spectra_mode_var.set("Random")
                self.spectra_from_var.set("1")
                self.spectra_to_var.set("")
                self.spectra_range_hint.config(text="")
                self._sync_spectra_range_state()
                self.spectra_opts_frame.pack_forget()
            self._spectra_seed = 42

            # Fade every flow-chart stage again and force a redraw.
            if hasattr(self, '_flow_active_steps'):
                self._flow_active_steps.clear()
            self._flow_last_signature = None
            if hasattr(self, 'refresh_flow_preview'):
                self.refresh_flow_preview()
        except Exception as e:
            self.status_text.insert(tk.END, f"Warning during GUI reset: {str(e)}\n")
            self.status_text.see(tk.END)
        finally:
            self._flow_suspend_trace = False
    
    def refresh_plot_list(self):
        """Refresh the list of available plots"""
        # Safety check - ensure viz_model_combo exists
        if not hasattr(self, 'viz_model_combo'):
            return
            
        if hasattr(self, 'analysis_plots') and self.analysis_plots:
            model_list = list(self.analysis_plots.keys())
            self.viz_model_combo['values'] = model_list
            if model_list and not self.viz_model_var.get():
                self.viz_model_var.set(model_list[0])
                self.display_selected_plot(None)
        else:
            self.viz_model_combo['values'] = []
            self.viz_model_var.set('')
    
    def display_selected_plot(self, event=None):
        """Display the selected model's plot or image output"""
        try:
            selected_model = self.viz_model_var.get()
            if not selected_model or selected_model not in self.analysis_plots:
                return

            # The spectra view controls (count / random / sample range) only make
            # sense for the Reflectance Spectra figure, so show them just for it.
            if hasattr(self, 'spectra_opts_frame'):
                if selected_model == "Reflectance Spectra":
                    self.spectra_opts_frame.pack(fill="x", padx=10, pady=(0, 5),
                                                 before=self.viz_display_frame)
                else:
                    self.spectra_opts_frame.pack_forget()

            # Clear existing plot
            for widget in self.viz_display_frame.winfo_children():
                widget.destroy()
            
            # Get the figure or image
            plot_item = self.analysis_plots[selected_model]
            
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from PIL import Image, ImageTk
            
            # Check if it's a matplotlib figure or PIL image
            if hasattr(plot_item, 'canvas'):  # matplotlib Figure
                canvas = FigureCanvasTkAgg(plot_item, master=self.viz_display_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                # Add toolbar
                toolbar = NavigationToolbar2Tk(canvas, self.viz_display_frame)
                toolbar.update()
            elif hasattr(plot_item, 'size'):  # PIL Image
                # Display PIL image
                # Resize image to fit display if needed
                display_img = plot_item.copy()
                max_width, max_height = 800, 600
                display_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(display_img)
                label = ttk.Label(self.viz_display_frame, image=photo)
                label.image = photo  # Keep reference
                label.pack(fill="both", expand=True)
        except Exception as e:
            self.status_text.insert(tk.END, f"Error displaying plot: {str(e)}\n")
            self.status_text.see(tk.END)
    
    def store_analysis_plot(self, model_name, property_name, fig_or_img, item_type="model"):
        """Store a plot or image for later display in visualization tab"""
        try:
            def _store():
                if not hasattr(self, 'analysis_plots'):
                    self.analysis_plots = {}
                if item_type == "image":
                    key = f"Image Output - {property_name}"
                elif item_type == "spectra":
                    key = "Reflectance Spectra"
                else:
                    key = f"{property_name} - {model_name}"
                self.analysis_plots[key] = fig_or_img
                self.refresh_plot_list()
                # Keep the window from auto-resizing when a plot is embedded mid-run.
                locked = getattr(self, '_locked_geometry', None)
                if locked:
                    try:
                        self.geometry(locked)
                    except tk.TclError:
                        pass
            self._run_on_ui(_store)
        except Exception as e:
            self.append_status(f"Error storing plot: {str(e)}\n")
    
    def view_predicted_image(self):
        """Switch to visualization tab to view the predicted image"""
        try:
            if not hasattr(self, 'analysis_plots') or not self.analysis_plots:
                messagebox.showwarning("Warning", "No visualization available. Please run a prediction first.")
                return
            
            # Switch to visualization tab
            self.display_notebook.select(2)  # Select Visualization tab (index 2)
            
            # Try to select the image output if it exists
            if hasattr(self, 'viz_model_combo'):
                for key in self.analysis_plots.keys():
                    if 'Image Output' in key:
                        self.viz_model_var.set(key)
                        self.display_selected_plot()
                        break
                        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display visualization: {str(e)}")
            self.status_text.insert(tk.END, f"Error switching to visualization: {str(e)}\n")
            self.status_text.see(tk.END)
