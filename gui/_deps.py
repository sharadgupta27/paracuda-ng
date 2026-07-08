"""
Shared imports for the Paracuda GUI package.

This module provides the shared module-level namespace for the
``SpectralAnalyzer`` GUI.  Every ``gui`` module does ``from gui._deps import *``
so that each group of methods sees precisely the same names — no per-file import
guesswork and no risk of a missed symbol.

@author: Sharad Kumar Gupta
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import multiprocessing
import threading
from typing import Any
from datetime import datetime
import os
import traceback
import joblib
import socket
import webbrowser

# --- Heavy scientific libraries: imported lazily so the GUI window appears fast.
# Each proxy resolves to the real object on first use; see utils/lazy_imports.py.
from utils.lazy_imports import LazyModule, LazyCallable

StandardScaler = LazyCallable('sklearn.preprocessing', 'StandardScaler')
PLSRegression = LazyCallable('sklearn.cross_decomposition', 'PLSRegression')
SVR = LazyCallable('sklearn.svm', 'SVR')
Ridge = LazyCallable('sklearn.linear_model', 'Ridge')
Lasso = LazyCallable('sklearn.linear_model', 'Lasso')
LinearRegression = LazyCallable('sklearn.linear_model', 'LinearRegression')
PCA = LazyCallable('sklearn.decomposition', 'PCA')
RandomForestRegressor = LazyCallable('sklearn.ensemble', 'RandomForestRegressor')
train_test_split = LazyCallable('sklearn.model_selection', 'train_test_split')
mean_squared_error = LazyCallable('sklearn.metrics', 'mean_squared_error')
r2_score = LazyCallable('sklearn.metrics', 'r2_score')
mean_absolute_error = LazyCallable('sklearn.metrics', 'mean_absolute_error')
# Force the non-interactive Agg backend BEFORE matplotlib.pyplot is ever imported.
# Every figure in Paracuda is either embedded in Tk via FigureCanvasTkAgg or written
# to a PDF — none are shown with plt.show() — so the interactive TkAgg backend is not
# needed.  TkAgg's pyplot-managed figure managers register Tcl async handlers which,
# torn down at interpreter exit, cause the fatal "Tcl_AsyncDelete: async handler
# deleted by the wrong thread" crash (and the noisy "main thread is not in main loop"
# messages) when the app closes.  Agg avoids all of that.
import matplotlib as _matplotlib
_matplotlib.use('Agg')
plt = LazyModule('matplotlib.pyplot')
Figure = LazyCallable('matplotlib.figure', 'Figure')
PdfPages = LazyCallable('matplotlib.backends.backend_pdf', 'PdfPages')
FigureCanvasTkAgg = LazyCallable('matplotlib.backends.backend_tkagg', 'FigureCanvasTkAgg')
rasterio = LazyModule('rasterio')

# Import our modular components
try:
    import utils.data_converter as _data_converter
    _HAS_CONVERTER = True
except ImportError:
    _HAS_CONVERTER = False

from preprocessing.data_processing import (safe_interpolate_spectra, preprocess_spectra,
                              calculate_statistics, filter_wavelengths,
                              calculate_confidence_interval,
                              detect_wavelength_unit, infer_spectral_domain,
                              randomize_label_test, mix_spectra_integrity_check,
                              spectral_transfer_function, SENSOR_OPTIONS,
                              resample_spectra, analyze_missing_data,
                              handle_missing_data, MISSING_DATA_METHODS,
                              remove_target_outliers, TARGET_OUTLIER_METHODS,
                              RESAMPLE_METHODS, normalize_resample_method,
                              estimate_fwhms_from_grid,
                              get_sensor_bands, load_fwhm_csv, load_srf_csv,
                              # (compositional imported separately below)
                              parse_exclude_ranges, WATER_ABSORPTION_RANGES,
                              NOISY_EDGE_RANGES)
from preprocessing.compositional import (forward as comp_forward,
                              inverse as comp_inverse, close as comp_close,
                              n_coords as comp_n_coords, TRANSFORMS as COMP_TRANSFORMS)
from models.model_training import (create_model, optimize_components_parallel,
                            parse_parameter_value, clamp_n_components)
from models.hyperparameter_tuning import (tune_hyperparameters, format_params,
                                    format_search_space, search_space_table)
from validation.cross_validation import perform_cross_validation
from utils.image_processing import (process_image_for_prediction, save_prediction_image,
                              save_cube_image, read_geospatial_image,
                              open_image_filetypes, driver_for_path)
from utils.file_operations import (save_results_to_excel, generate_default_filename,
                              generate_model_filename, make_timestamp,
                              save_property_batch_results,
                              generate_property_filename,
                              save_unknown_predictions_to_excel,
                              generate_unknown_prediction_filename,
                              save_resampled_tabular_to_excel,
                              save_transfer_function, load_transfer_function)
from models.batch_processing import (suggest_best_model, create_scatter_plot,
                              create_feature_importance_plot, create_comparison_plots,
                              create_reflectance_spectra_plot, assess_overfitting)
from utils.help_assistant import HelpAssistant

# Export the full namespace (including the single-underscore converter names) so
# ``from gui._deps import *`` reproduces the original module scope exactly.
__all__ = [name for name in dir() if not name.startswith('__')]
