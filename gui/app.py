"""
Paracuda III main application window.

This file keeps the constructor and the window-lifecycle helpers for the
``SpectralAnalyzer`` GUI class; the remaining methods live in the other
``gui/*_mixin.py`` modules.

@author: Sharad Kumar Gupta
"""
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace
from gui.menu_help_mixin import MenuHelpMixin
from gui.layout_mixin import LayoutMixin
from gui.uiflow_mixin import UIFlowMixin
from gui.data_io_mixin import DataIOMixin
from gui.analysis_mixin import AnalysisMixin
from gui.dialogs_mixin import DialogsMixin
from gui.flowchart_mixin import FlowChartMixin

# Shared theme — bring the main window up to the Data Converter's card look and
# honour the user's Theme-menu choice.  Optional so the app still starts if the
# module is missing.
try:
    from paracuda_theme import (get_palette, load_theme_name, save_theme_name,
                                list_palettes, apply_ttk_theme)
    _HAS_THEME = True
except Exception:
    _HAS_THEME = False


class SpectralAnalyzer(MenuHelpMixin, LayoutMixin, UIFlowMixin,
                       DataIOMixin, AnalysisMixin, DialogsMixin,
                       FlowChartMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.iconbitmap('icon.ico')
        self.title("Paracuda III")
        # self.geometry("850x880")
        self.resizable(True, True)

        # Apply the shared visual theme before any widgets are built so every
        # ttk widget picks up the palette.  self._palette is consumed by the
        # layout (canvas/text backgrounds) and the process-flow chart.
        self.theme_name = load_theme_name() if _HAS_THEME else "Ocean"
        if _HAS_THEME:
            self._palette = apply_ttk_theme(self, get_palette(self.theme_name))
        else:
            self._palette = None
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Initialize datetime callback ID
        self._datetime_after_id = None
        self._main_thread_id = threading.get_ident()
        self._batch_running = False
        
        # Set up proper window close handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create menu bar
        self.create_menu()
        
        # Initialize optimize components variable
        self.optimize_components_var = tk.BooleanVar()
        
        # Data storage
        self.df = None
        self.wavelengths = None
        self.soil_properties = None
        self.input_filename = None
        self.output_filename = None
        self.trained_model = None
        self.model_basename = None  # base name of the loaded/saved model file (for output naming)
        self.scaler_X = None
        self.scaler_y = None
        self.image_data = None
        self.image_meta = None
        self.image_wavelengths = None   # per-band wavelengths recovered from header
        self.image_fwhms = None
        self.image_path = None
        self.image_driver = None
        self.image_interleave = None    # source band interleave (bil/bip/bsq)
        self.training_input_wavelengths = None  # model's input band grid (from saved model)
        self.training_input_fwhms = None
        self.compositional_model = None  # bundle of per-coordinate models (ALR/CLR/ILR)
        self._last_result_timestamp = None
        self.predicted_image = None
        self.image_canvas = None
        self.selected_property = None
        self.selected_properties = []  # For batch processing
        self.selected_models = []  # For batch model processing
        self.single_property_mode = False  # Track if single property is selected for save model
        # Explicit save overrides set when a best/auto-selected model (from a batch
        # or best-preprocessing run) becomes the saveable model; None → save from
        # the live GUI widgets (plain single run).
        self._save_model_type_override = None
        self._save_model_params_override = None
        self.filtered_wavelengths = None
        self.new_wavelengths = None
        self.new_fwhms = None            # per-band FWHM for SRF resampling
        self.resample_method = "Linear Interpolation"  # one of RESAMPLE_METHODS
        self.custom_fwhm_table = None    # (centers, fwhms) from an uploaded FWHM CSV
        self.srf_table = None            # {center: (wl, response)} from an SRF CSV
        self.exclude_ranges = []         # [(lo, hi), ...] nm ranges to omit
        self._reflectance_spectra_fig = None  # cached reflectance plot (for PDF export)
        self._spectra_seed = 42               # RNG seed for the spectra subset (re-rolled on Update Plot)

        # Wavelength unit and spectral domain (set during load_excel)
        self.wavelength_unit = "nm"          # "nm" or "μm"
        self.spectral_domain = "VSWIR"       # "VSWIR", "LWIR", "VSWIR+LWIR"

        # Preprocessing metadata stored with model (set after train / load_model)
        self.model_preprocessing = "No Preprocessing"
        self.model_preprocessing_kwargs = {}
        self.pca_component = None  # PCA object for two-step PCA predict pipeline

        # Tabular-prediction state (separate from training data)
        self.tabular_df = None
        self.tabular_wl_nm = None
        self.tabular_wl_cols = None   # exact column names as they exist in the loaded df
        self.tabular_file_path = None
        
        # Available colormaps
        self.colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                         'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        
        # Initialize help assistant
        self.help_assistant = HelpAssistant()
        
        # Model parameters configuration with XGBoost
        self.model_params = {
            "PLS-R": {
                "params": {
                    "n_components": {
                        "label": "Components", 
                        "default": "32", 
                        "type": "entry",
                        "help": "Number of components to keep"
                    }
                }
            },
            "SVM": {
                "params": {
                    "kernel": {
                        "label": "Kernel", 
                        "default": "rbf", 
                        "type": "combobox",
                        "values": ["linear", "poly", "rbf", "sigmoid"],
                        "help": "Specifies the kernel type to be used"
                    },
                    "C": {
                        "label": "Regularization (C)", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Regularization parameter"
                    },
                    "degree": {
                        "label": "Degree", 
                        "default": "3", 
                        "type": "entry",
                        "help": "Degree of polynomial kernel (ignored by other kernels)"
                    },
                    "gamma": {
                        "label": "Gamma", 
                        "default": "scale", 
                        "type": "combobox",
                        "values": ["scale", "auto"],
                        "help": "Kernel coefficient for rbf, poly and sigmoid"
                    },
                    "epsilon": {
                        "label": "Epsilon", 
                        "default": "0.1", 
                        "type": "entry",
                        "help": "Epsilon in the epsilon-SVR model"
                    }
                }
            },
            "Ridge": {
                "params": {
                    "alpha": {
                        "label": "Alpha", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Regularization strength"
                    },
                    "solver": {
                        "label": "Solver", 
                        "default": "auto", 
                        "type": "combobox",
                        "values": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                        "help": "Solver to use in the computational routines"
                    },
                    "max_iter": {
                        "label": "Max Iterations", 
                        "default": "5000", 
                        "type": "entry",
                        "help": "Maximum number of iterations (increase if convergence warnings appear)"
                    }
                }
            },
            "Lasso": {
                "params": {
                    "alpha": {
                        "label": "Alpha", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Constant that multiplies the L1 term"
                    },
                    "max_iter": {
                        "label": "Max Iterations", 
                        "default": "5000", 
                        "type": "entry",
                        "help": "Maximum number of iterations (increase if convergence warnings appear)"
                    },
                    "selection": {
                        "label": "Selection", 
                        "default": "cyclic", 
                        "type": "combobox",
                        "values": ["cyclic", "random"],
                        "help": "Selection method for updating coefficients"
                    },
                    "tol": {
                        "label": "Tolerance", 
                        "default": "1e-4", 
                        "type": "entry",
                        "help": "Tolerance for the optimization"
                    }
                }
            },
            "Multiple Linear Regression": {
                "params": {
                    "fit_intercept": {
                        "label": "Fit Intercept", 
                        "default": "True", 
                        "type": "combobox",
                        "values": ["True", "False"],
                        "help": "Whether to calculate the intercept"
                    }
                }
            },
            "Elastic Net": {
                "params": {
                    "alpha": {
                        "label": "Alpha", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Constant that multiplies the penalty terms"
                    },
                    "l1_ratio": {
                        "label": "L1 Ratio", 
                        "default": "0.5", 
                        "type": "entry",
                        "help": "The ElasticNet mixing parameter (0 <= l1_ratio <= 1)"
                    },
                    "max_iter": {
                        "label": "Max Iterations", 
                        "default": "1000", 
                        "type": "entry",
                        "help": "Maximum number of iterations"
                    },
                    "tol": {
                        "label": "Tolerance", 
                        "default": "1e-4", 
                        "type": "entry",
                        "help": "Tolerance for the optimization"
                    }
                }
            },
            "Huber Regressor": {
                "params": {
                    "epsilon": {
                        "label": "Epsilon", 
                        "default": "1.35", 
                        "type": "entry",
                        "help": "Controls the number of samples that should be classified as outliers"
                    },
                    "max_iter": {
                        "label": "Max Iterations", 
                        "default": "2000", 
                        "type": "entry",
                        "help": "Maximum number of iterations (increase if convergence warnings appear)"
                    },
                    "alpha": {
                        "label": "Alpha", 
                        "default": "0.0001", 
                        "type": "entry",
                        "help": "Regularization parameter"
                    }
                }
            },
            "Gradient Boosting": {
                "params": {
                    "n_estimators": {
                        "label": "N Estimators", 
                        "default": "100", 
                        "type": "entry",
                        "help": "Number of boosting stages"
                    },
                    "learning_rate": {
                        "label": "Learning Rate", 
                        "default": "0.1", 
                        "type": "entry",
                        "help": "Learning rate shrinks the contribution of each tree"
                    },
                    "max_depth": {
                        "label": "Max Depth", 
                        "default": "3", 
                        "type": "entry",
                        "help": "Maximum depth of the individual regression estimators"
                    },
                    "min_samples_split": {
                        "label": "Min Samples Split", 
                        "default": "2", 
                        "type": "entry",
                        "help": "Minimum samples required to split an internal node"
                    },
                    "min_samples_leaf": {
                        "label": "Min Samples Leaf", 
                        "default": "1", 
                        "type": "entry",
                        "help": "Minimum samples required to be at a leaf node"
                    },
                    "subsample": {
                        "label": "Subsample", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Fraction of samples used for fitting individual base learners"
                    }
                }
            },
            "Gaussian Process": {
                "params": {
                    "length_scale": {
                        "label": "Length Scale", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Length scale of the RBF kernel"
                    },
                    "alpha": {
                        "label": "Alpha", 
                        "default": "1e-10", 
                        "type": "entry",
                        "help": "Value added to the diagonal of the kernel matrix"
                    },
                    "n_restarts_optimizer": {
                        "label": "N Restarts", 
                        "default": "0", 
                        "type": "entry",
                        "help": "Number of restarts of the optimizer"
                    }
                }
            },
            "Random Forest": {
                "params": {
                    "n_estimators": {
                        "label": "Number of Trees", 
                        "default": "100", 
                        "type": "entry",
                        "help": "Number of trees in the forest"
                    },
                    "max_depth": {
                        "label": "Max Depth", 
                        "default": "5", 
                        "type": "entry",
                        "help": "Maximum depth of the tree"
                    },
                    "min_samples_split": {
                        "label": "Min Samples Split", 
                        "default": "2", 
                        "type": "entry",
                        "help": "Minimum samples required to split an internal node"
                    },
                    "min_samples_leaf": {
                        "label": "Min Samples Leaf", 
                        "default": "1",
                        "type": "entry",
                        "help": "Minimum samples required to be at a leaf node"
                    },
                    "max_features": {
                        "label": "Max Features", 
                        "default": "sqrt",
                        "type": "combobox",
                        "values": ["sqrt", "log2", "None"],
                        "help": "Number of features to consider when looking for best split"
                    },
                    "bootstrap": {
                        "label": "Bootstrap", 
                        "default": "True", 
                        "type": "combobox",
                        "values": ["True", "False"],
                        "help": "Whether bootstrap samples are used when building trees"
                    }
                }
            },
            "XGBoost": {
                "params": {
                    "n_estimators": {
                        "label": "N Estimators", 
                        "default": "100", 
                        "type": "entry",
                        "help": "Number of boosting rounds"
                    },
                    "max_depth": {
                        "label": "Max Depth", 
                        "default": "6", 
                        "type": "entry",
                        "help": "Maximum depth of trees"
                    },
                    "learning_rate": {
                        "label": "Learning Rate", 
                        "default": "0.3", 
                        "type": "entry",
                        "help": "Step size shrinkage used to prevent overfitting"
                    },
                    "subsample": {
                        "label": "Subsample", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Subsample ratio of training instances"
                    },
                    "colsample_bytree": {
                        "label": "Col Sample Tree", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Subsample ratio of columns when constructing each tree"
                    },
                    "reg_alpha": {
                        "label": "Reg Alpha", 
                        "default": "0.0", 
                        "type": "entry",
                        "help": "L1 regularization term on weights"
                    },
                    "reg_lambda": {
                        "label": "Reg Lambda", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "L2 regularization term on weights"
                    }
                }
            }

        }
        
        self.param_vars = {}
        self.param_widgets = {}
        self.cv_param_vars = {}
        self.cv_param_widgets = {}
        
        self.create_gui()
    
    def on_closing(self):
        """Proper cleanup when closing the window"""
        # Flag closing FIRST so self-rescheduling polls/refreshes stop looping.
        self._closing = True
        try:
            # Cancel the datetime update callback if it exists
            if hasattr(self, '_datetime_after_id') and self._datetime_after_id:
                self.after_cancel(self._datetime_after_id)
                self._datetime_after_id = None
        except Exception:
            pass
        # Cancel the flow-chart poll/refresh after-jobs (avoids "invalid command
        # name ..._flow_poll" errors when a pending job fires post-destroy).
        try:
            if hasattr(self, 'cancel_flow_jobs'):
                self.cancel_flow_jobs()
        except Exception:
            pass
        # Shut the joblib/loky worker pool down cleanly before the interpreter
        # exits, so its temp memmap folders are removed in-process instead of
        # racing the resource_tracker at shutdown (a benign but noisy Windows
        # "FileNotFoundError" warning otherwise).
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except Exception:
            pass

        # Close any matplotlib figures while the Tk interpreter is still alive, so
        # their canvases/PhotoImages are released now instead of by the GC after
        # teardown (which prints "main thread is not in main loop").  Only if
        # pyplot was actually imported — don't force the heavy import on exit.
        try:
            import sys
            if 'matplotlib.pyplot' in sys.modules:
                sys.modules['matplotlib.pyplot'].close('all')
        except Exception:
            pass

        # Destroy the window
        self.destroy()
    
    def update_datetime(self):
        try:
            if not self.winfo_exists():
                return
            current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            computer_name = socket.gethostname()
            self.datetime_label.config(text=f"{current_time} | {computer_name}")
            # Store the after callback ID so we can cancel it later
            self._datetime_after_id = self.after(1000, self.update_datetime)
        except tk.TclError:
            # Widget was destroyed, stop the callback
            pass

    def _open_data_converter(self):
        """Open the Paracuda Data Converter in a separate window."""
        if not _HAS_CONVERTER:
            tk.messagebox.showerror(
                "Not available",
                "data_converter.py was not found.\n"
                "Make sure it is in the same folder as paracuda.py."
            )
            return
        # Run in a new Toplevel-compatible process so it has its own mainloop
        import subprocess, sys
        # data_converter.py lives in the utils/ package, one level up from gui/.
        repo_root = os.path.dirname(os.path.dirname(__file__))
        subprocess.Popen(
            [sys.executable, os.path.join(repo_root, "utils", "data_converter.py")]
        )
