"""
Batch processing utilities for running multiple models and properties

@author: Sharad Kumar Gupta
"""
import contextlib
import numpy as np
import pandas as pd

# matplotlib imported lazily so importing this module at GUI startup is cheap;
# it loads on first plot. See utils/lazy_imports.py.
from utils.lazy_imports import LazyModule, LazyCallable

plt = LazyModule('matplotlib.pyplot')
cm = LazyModule('matplotlib.cm')
Figure = LazyCallable('matplotlib.figure', 'Figure')
PdfPages = LazyCallable('matplotlib.backends.backend_pdf', 'PdfPages')


# ── Accuracy metrics ─────────────────────────────────────────────────────────
# RPD and normalized RMSEP are reported alongside R2 / RMSE / MAE everywhere a
# model is scored, so single runs, batch runs, cross-validation and the exported
# workbooks all speak the same language.

def compute_rpd(y_true, y_pred=None, rmse=None):
    """Ratio of Performance to Deviation: SD(observed) / RMSE.

    The standard chemometric measure of how useful a calibration is relative to
    the natural spread of the property.  Pass either predictions or a
    precomputed RMSE.  Uses the sample standard deviation (ddof=1) of the
    OBSERVED values, which is the usual convention.

    Returns ``nan`` when it cannot be computed (fewer than 2 samples, or a
    perfect fit where RMSE is 0 and the ratio is unbounded).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    if y_true.size < 2:
        return float("nan")
    if rmse is None:
        if y_pred is None:
            return float("nan")
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmse = float(rmse)
    if not np.isfinite(rmse) or rmse <= 1e-12:
        return float("nan")
    sd = float(np.std(y_true, ddof=1))
    if not np.isfinite(sd):
        return float("nan")
    return sd / rmse


def compute_nrmsep(y_true, y_pred=None, rmse=None, norm="range"):
    """Normalized RMSEP as a PERCENTAGE, so error is comparable across properties.

    ``norm='range'`` (default) divides by max-min of the observed values;
    ``norm='mean'`` divides by the observed mean.  Range normalization is the
    more common reading of "normalized RMSE" and is well defined for properties
    whose mean sits near zero, so it is what the GUI reports.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    if y_true.size == 0:
        return float("nan")
    if rmse is None:
        if y_pred is None:
            return float("nan")
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rmse = float(rmse)
    if not np.isfinite(rmse):
        return float("nan")
    if norm == "mean":
        denom = abs(float(np.mean(y_true)))
    else:
        denom = float(np.max(y_true) - np.min(y_true))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return 100.0 * rmse / denom


def compute_metrics(y_true, y_pred):
    """Return the full metric set for one observed/predicted pair.

    Keys: ``r2``, ``rmse``, ``mae``, ``rpd``, ``nrmsep`` (percent of observed
    range) and ``nrmsep_mean`` (percent of observed mean).  ``rmsep`` is an alias
    of ``rmse`` kept so test-split results can be referred to by the name used in
    the chemometrics literature.
    """
    from sklearn.metrics import r2_score, mean_absolute_error

    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")
    try:
        r2 = float(r2_score(yt, yp)) if yt.size >= 2 else float("nan")
    except Exception:  # noqa: BLE001 - degenerate splits should not abort a run
        r2 = float("nan")
    try:
        mae = float(mean_absolute_error(yt, yp)) if yt.size else float("nan")
    except Exception:  # noqa: BLE001
        mae = float("nan")
    return {
        "r2": r2,
        "rmse": rmse,
        "rmsep": rmse,
        "mae": mae,
        "rpd": compute_rpd(yt, rmse=rmse),
        "nrmsep": compute_nrmsep(yt, rmse=rmse, norm="range"),
        "nrmsep_mean": compute_nrmsep(yt, rmse=rmse, norm="mean"),
    }


def rpd_quality(rpd):
    """Plain-language reading of an RPD value (Viscarra Rossel et al., 2006)."""
    if rpd is None or not np.isfinite(rpd):
        return "n/a"
    if rpd < 1.0:
        return "very poor"
    if rpd < 1.4:
        return "poor"
    if rpd < 1.8:
        return "fair"
    if rpd < 2.0:
        return "good"
    if rpd < 2.5:
        return "very good"
    return "excellent"


def assess_overfitting(train_r2, test_r2, train_rmse=None, test_rmse=None, cv_r2_mean=None):
    """Assess overfitting using several measures rather than a single R² gap.

    Overfitting means the model fit the *training* data well but generalises poorly.
    A model that fits training badly (low/negative train R²) is underfitting, not
    overfitting, so it must never be flagged here - that was the old false-positive.

    Args:
        train_r2, test_r2:     R² on train / test splits.
        train_rmse, test_rmse: RMSE on train / test splits (optional, corroborating).
        cv_r2_mean:            Mean cross-validated R² (optional, corroborating).

    Returns:
        dict with:
            flag     (bool):  True only when overfitting is genuinely indicated.
            severity (str):   'none' | 'mild' | 'strong'.
            gap      (float): train_r2 - test_r2 (kept for display / back-compat).
            reasons  (list):  human-readable measures that fired.
    """
    try:
        train_r2 = float(train_r2)
        test_r2 = float(test_r2)
    except (TypeError, ValueError):
        return {'flag': False, 'severity': 'none', 'gap': 0.0, 'reasons': []}

    gap = train_r2 - test_r2
    rel_gap = gap / max(abs(train_r2), 1e-6)
    reasons = []

    # Primary measures - ALL must hold for overfitting to be reported.
    learned_train = train_r2 >= 0.6           # model actually fit the training data
    abs_gap_large = gap > 0.15                 # sizeable absolute drop train → test
    rel_gap_large = rel_gap > 0.25             # test lost >25% of the training skill

    flag = learned_train and abs_gap_large and rel_gap_large
    if not flag:
        return {'flag': False, 'severity': 'none', 'gap': gap, 'reasons': []}

    reasons.append(f"train R²={train_r2:.3f} (model fit training well)")
    reasons.append(f"train−test R² gap={gap:.3f} (>0.15)")
    reasons.append(f"relative drop={rel_gap*100:.0f}% of train skill (>25%)")

    severity = 'mild'

    # Corroborating measures raise severity to 'strong'.
    if cv_r2_mean is not None:
        with contextlib.suppress((TypeError, ValueError)):
            cv_gap = train_r2 - float(cv_r2_mean)
            if cv_gap > 0.15:
                reasons.append(f"train−CV R² gap={cv_gap:.3f} (>0.15)")
                severity = 'strong'

    if train_rmse is not None and test_rmse is not None:
        with contextlib.suppress((TypeError, ValueError)):
            train_rmse = float(train_rmse)
            test_rmse = float(test_rmse)
            if train_rmse > 1e-12 and test_rmse > 1.3 * train_rmse:
                reasons.append(f"test RMSE={test_rmse:.3f} > 1.3× train RMSE={train_rmse:.3f}")
                severity = 'strong'

    return {'flag': True, 'severity': severity, 'gap': gap, 'reasons': reasons}


def suggest_best_model(results_dict):
    """
    Suggest the best model based on multiple criteria
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
                     Each result should have 'test_r2', 'test_rmse', 'cv_r2_mean' (optional)
    
    Returns:
        best_model: Name of the best model
        best_scores: Dictionary of scores for the best model
        comparison_df: DataFrame comparing all models
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {
            'Model': model_name,
            'Test_R2': results.get('test_r2', 0),
            'Test_RMSE': results.get('test_rmse', float('inf')),
            'Train_R2': results.get('train_r2', 0),
            'Train_RMSE': results.get('train_rmse', float('inf'))
        }

        # RMSEP / RPD / normalized RMSEP.  Prefer the values the caller already
        # computed; fall back to deriving them from the stored observations so
        # older result dicts still populate the comparison table.
        row['Test_RMSEP'] = results.get('test_rmsep', row['Test_RMSE'])
        rpd = results.get('test_rpd')
        nrmsep = results.get('test_nrmsep')
        if rpd is None or nrmsep is None:
            y_true = results.get('y_test')
            if y_true is not None and len(np.asarray(y_true).ravel()) >= 2:
                if rpd is None:
                    rpd = compute_rpd(y_true, rmse=row['Test_RMSE'])
                if nrmsep is None:
                    nrmsep = compute_nrmsep(y_true, rmse=row['Test_RMSE'])
        row['Test_RPD'] = rpd if rpd is not None else float('nan')
        row['Test_nRMSEP'] = nrmsep if nrmsep is not None else float('nan')
        row['RPD_Quality'] = rpd_quality(row['Test_RPD'])
        
        # Add cross-validation metrics if available
        if results.get('cv_results'):
            row['CV_R2_Mean'] = results['cv_results'].get('r2_mean', 0)
            row['CV_RMSE_Mean'] = results['cv_results'].get('rmse_mean', float('inf'))
        else:
            row['CV_R2_Mean'] = None
            row['CV_RMSE_Mean'] = None
        
        # Overfitting indicator - multi-measure assessment (not just the R² gap)
        assessment = assess_overfitting(
            results.get('train_r2'), results.get('test_r2'),
            results.get('train_rmse'), results.get('test_rmse'),
            (results.get('cv_results') or {}).get('r2_mean'),
        )
        row['Overfitting_Gap'] = assessment['gap']
        row['Overfitting_Flag'] = assessment['flag']
        row['Overfitting_Severity'] = assessment['severity']

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Score calculation: prioritize test R2, penalize *genuine* overfitting
    comparison_df['Score'] = comparison_df['Test_R2'] * 100

    # Penalize based on the multi-measure flag/severity, not the raw gap, so that
    # a model that merely underfits (low train R²) is not wrongly punished.
    if 'Overfitting_Severity' in comparison_df.columns:
        penalty = comparison_df['Overfitting_Severity'].map(
            {'strong': 15.0, 'mild': 7.0}).fillna(0.0)
        comparison_df['Score'] -= penalty
    
    # Bonus for low RMSE (normalized)
    if comparison_df['Test_RMSE'].max() > 0:
        rmse_normalized = 1 - (comparison_df['Test_RMSE'] / comparison_df['Test_RMSE'].max())
        comparison_df['Score'] += rmse_normalized * 10
    
    # Sort by score
    comparison_df = comparison_df.sort_values('Score', ascending=False)
    
    best_model = comparison_df.iloc[0]['Model']
    best_scores = results_dict[best_model]
    
    return best_model, best_scores, comparison_df


def create_scatter_plot(y_true, y_pred, model_name, property_name, metrics, ax=None,
                        overfitting_flag=False, dataset_label="Test Data"):
    """
    Create an observed-vs-predicted scatter plot for ONE split (test or train).

    There is no figure title (the split + model are shown in the GUI selector and
    the compact top-left legend).  The legend carries the split name, the 1:1 line,
    the least-squares fit equation, and the split's R² / RMSE.  Layout is
    constrained so the axis labels never clip when the canvas is resized.

    Args:
        y_true, y_pred:            observed / predicted values for this split.
        model_name, property_name: labels (property_name is used on the axes).
        metrics:                   dict with 'r2' and 'rmse'.
        ax:                        optional axis to draw into.
        overfitting_flag:          outline the legend in red when True.
        dataset_label:             "Test Data" (default) or "Training Data".

    Returns:
        (fig, ax) when *ax* is None, else None.
    """
    from matplotlib.lines import Line2D

    if ax is None:
        # Standalone Figure (off pyplot's registry); constrained layout re-fits
        # margins on every draw so labels survive GUI resizing.
        fig = Figure(figsize=(6.4, 6.4), constrained_layout=True)
        ax = fig.add_subplot(111)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    is_train = dataset_label == "Training Data"
    series = "Train" if is_train else "Test"
    color = '#f0a202' if is_train else '#2c7fb8'
    marker = '^' if is_train else 'o'
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=42, c=color,
               marker=marker, label=series, zorder=3)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.03 * ((hi - lo) or 1.0)
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.6, label='1:1 line', zorder=1)
    with contextlib.suppress(Exception):  # noqa: BLE001
        a, b = np.polyfit(y_true, y_pred, 1)
        xs = np.array([lo, hi])
        ax.plot(xs, a * xs + b, color='#1a9850', lw=1.6,
                label=f"Fit: y = {a:.3f}x {b:+.3f}", zorder=2)

    metric_rows = [f"R² = {metrics['r2']:.3f}", f"RMSE = {metrics['rmse']:.3f}"]
    text_handles = [Line2D([], [], linestyle='none', marker='', label=t)
                    for t in metric_rows]

    ax.set_xlabel(f'Observed {property_name}', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'Predicted {property_name}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    handles += text_handles
    labels += [h.get_label() for h in text_handles]
    leg = ax.legend(handles, labels, loc='upper left', fontsize=8, framealpha=0.9,
                    handlelength=1.4, labelspacing=0.3, borderpad=0.4)
    if overfitting_flag and leg is not None:
        leg.get_frame().set_edgecolor('red')

    if return_fig:
        return fig, ax
    return None


# Above this many spectra the per-sample legend stops being readable and starts
# covering the curves, so it is omitted.
LEGEND_MAX_SPECTRA = 20


def create_reflectance_spectra_plot(X, wavelengths, sample_names=None, n_samples=10,
                                    wavelength_unit="nm", seed=42, title=None,
                                    row_pool=None):
    """Publication-quality reflectance-spectra plot for a random subset of samples.

    Draws ``n_samples`` spectra (or all rows when fewer are available) as thin
    lines over the wavelength axis, using discrete qualitative colours so
    individual spectra stay distinguishable.  The per-sample legend is omitted
    once more than ``LEGEND_MAX_SPECTRA`` lines are drawn, since it would then
    obscure the curves.

    Args:
        X:              2-D array (n_rows, n_bands) of reflectance values.
        wavelengths:    sequence of band-centre wavelengths (len == n_bands).
        sample_names:   optional per-row labels used for the legend.
        n_samples:      number of rows to draw at random (capped at the pool size).
        wavelength_unit: axis unit label ("nm" or "μm").
        seed:           RNG seed so the GUI figure and the exported PDF show the
                        SAME random samples.
        title:          optional title override.
        row_pool:       optional row indices the random draw may pick from.  ``None``
                        (default) means every row is eligible; pass a subset to
                        restrict the draw to a specific range of samples.

    Returns:
        matplotlib Figure.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Reflectance plot needs a non-empty 2-D spectra matrix.")

    wl = np.asarray([float(w) for w in wavelengths], dtype=float)
    # Plot in ascending-wavelength order regardless of column order.
    order = np.argsort(wl)
    wl = wl[order]
    X = X[:, order]

    n_rows = X.shape[0]
    if row_pool is None:
        pool = np.arange(n_rows)
    else:
        pool = np.asarray([int(i) for i in np.asarray(row_pool).ravel()
                           if 0 <= int(i) < n_rows], dtype=int)
        if pool.size == 0:
            raise ValueError("The selected sample range contains no valid samples.")

    k = int(min(max(int(n_samples), 1), pool.size))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(pool, size=k, replace=False))

    # constrained_layout re-fits margins on each draw, so the title / axis labels
    # stay visible when the canvas is resized inside the GUI.
    fig = Figure(figsize=(9, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111)

    # Discrete qualitative colours - a continuous gradient (viridis) makes
    # neighbouring spectra blend into one another, so cycle through high-contrast
    # categorical palettes instead (30 distinct hues before repeating).
    discrete = list(plt.get_cmap('tab10').colors) + list(plt.get_cmap('tab20b').colors)
    colors = [discrete[j % len(discrete)] for j in range(k)]

    for j, row in enumerate(idx):
        if sample_names is not None and row < len(sample_names):
            label = str(sample_names[row])
        else:
            label = f"Sample {int(row) + 1}"
        ax.plot(wl, X[row], color=colors[j], lw=1.4, alpha=0.9, label=label)

    ax.set_xlabel(f"Wavelength ({wavelength_unit})", fontsize=9)
    ax.set_ylabel("Reflectance", fontsize=9)
    ax.grid(True, which='major', alpha=0.25, linewidth=0.6)
    ax.margins(x=0.01)
    ax.tick_params(labelsize=8)

    # Beyond ~20 lines the legend crowds out the plot itself, so drop it.
    if k <= LEGEND_MAX_SPECTRA:
        ax.legend(fontsize=6, ncol=2, frameon=False, loc='best',
                  handlelength=1.2, labelspacing=0.3, columnspacing=1.0)

    return fig


def get_feature_importance(model, model_type, wavelengths, correlations=None):
    """
    Extract feature importance values for any model type
    
    Args:
        model: Trained model
        model_type: Type of model
        wavelengths: List of wavelengths
        correlations: Correlation values (for non-tree models)
    
    Returns:
        numpy array of feature importance values
    """
    if model_type in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
    elif model_type in ['Ridge', 'Lasso', 'ElasticNet', 'Linear Regression']:
        # Coefficient-based importance for linear models
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_)
    elif correlations is not None:
        # Use correlations for other models
        if not isinstance(correlations, np.ndarray):
            correlations = np.array(correlations)
        return correlations
    
    return None


def create_feature_importance_plot(model, model_type, wavelengths, property_name, 
                                   correlations=None, ax=None, top_n=20):
    """
    Create feature importance plot for tree-based models or correlation plot for others
    
    Args:
        model: Trained model
        model_type: Type of model
        wavelengths: List of wavelengths
        property_name: Name of the soil property
        correlations: Correlation values (for non-tree models)
        ax: Matplotlib axis (optional)
        top_n: Number of top features to show
    
    Returns:
        fig, ax if ax is None, otherwise None
    """
    if ax is None:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        return_fig = True
    else:
        return_fig = False

    if model_type in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        # Feature importance for tree-based models - plot as line with wavelength on x-axis
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Calculate 95th percentile
            percentile_95 = np.percentile(importances, 95)
            
            # Plot importance vs wavelength
            ax.plot(wavelengths, importances, linewidth=2, color='darkblue', label='Importance')
            
            # Add 95th percentile line
            ax.axhline(y=percentile_95, color='red', linestyle='--', linewidth=2, 
                      label='95th percentile')
            
            ax.set_xlabel('Wavelength [nm]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_type} - Feature Importance\n{property_name}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim([min(wavelengths), max(wavelengths)])
    else:
        # Feature importance plot for other models (using correlations or coefficients)
        if correlations is not None:
            # Convert to numpy array if needed
            if not isinstance(correlations, np.ndarray):
                correlations = np.array(correlations)
            
            # Calculate 95th percentile for threshold line
            percentile_95 = np.percentile(np.abs(correlations), 95)
            
            # Plot correlation/coefficient vs wavelength
            ax.plot(wavelengths, correlations, linewidth=2, color='darkblue', label='Feature Weight')
            
            # Add 95th percentile lines (positive and negative)
            ax.axhline(y=percentile_95, color='red', linestyle='--', linewidth=2, 
                      label='95th percentile', alpha=0.7)
            ax.axhline(y=-percentile_95, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            ax.set_xlabel('Wavelength [nm]', fontsize=12, fontweight='bold')
            if model_type in ['PLSR', 'Ridge', 'Lasso', 'ElasticNet']:
                ax.set_ylabel('Normalized Coefficient', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_type} - Feature Coefficients\n{property_name}',
                            fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel('Correlation with Predictions', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_type} - Feature-Prediction Correlation\n{property_name}',
                            fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim([min(wavelengths), max(wavelengths)])

    if return_fig:
        fig.tight_layout()
        return fig, ax
    return None


def save_batch_plots(results_dict, output_folder, property_name):
    """
    Save all plots for batch processing results
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        output_folder: Folder to save plots
        property_name: Name of the soil property
    """
    import os
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create PDF with all plots
    pdf_path = os.path.join(output_folder, f'{property_name}_batch_results.pdf')
    
    with PdfPages(pdf_path) as pdf:
        for model_name, results in results_dict.items():
            # Training scatter (when the result carries train predictions)
            if results.get('y_train') is not None and results.get('y_train_pred') is not None:
                fig, ax = create_scatter_plot(
                    results['y_train'],
                    results['y_train_pred'],
                    model_name,
                    property_name,
                    {'r2': results['train_r2'], 'rmse': results['train_rmse']},
                    dataset_label="Training Data"
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            # Test scatter plot
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
    
    return pdf_path


def create_comparison_plots(comparison_df, property_name):
    """
    Create comparison plots for all models
    
    Args:
        comparison_df: DataFrame with model comparison
        property_name: Name of the soil property
    
    Returns:
        fig: Matplotlib figure
    """
    # Built with the OO Figure API rather than pyplot: this figure is only
    # ever saved into a PDF, and a pyplot-managed figure stays in pyplot's
    # global registry until explicitly closed - a leak in the long-running
    # QGIS panel, whose caller has no pyplot handle to close it with.
    fig = Figure(figsize=(14, 10))
    axes = fig.subplots(2, 2)
    fig.suptitle(f'Model Comparison - {property_name}', fontsize=16, fontweight='bold')
    
    # R² comparison
    ax = axes[0, 0]
    models = comparison_df['Model']
    r2_values = comparison_df['Test_R2']
    colors = cm.viridis(np.linspace(0.3, 0.9, len(models)))
    ax.barh(models, r2_values, color=colors)
    ax.set_xlabel('Test R²', fontsize=11, fontweight='bold')
    ax.set_title('Test R² Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # RMSE comparison
    ax = axes[0, 1]
    rmse_values = comparison_df['Test_RMSE']
    ax.barh(models, rmse_values, color=colors)
    ax.set_xlabel('Test RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Test RMSE Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Train vs Test R²
    ax = axes[1, 0]
    if 'Train_R2' in comparison_df.columns:
        x = np.arange(len(models))
        width = 0.35
        train_bars = ax.bar(x - width/2, comparison_df['Train_R2'], width, label='Train', alpha=0.8)
        test_bars  = ax.bar(x + width/2, comparison_df['Test_R2'],  width, label='Test',  alpha=0.8)
        # Red hatch on bars flagged as overfitting (multi-measure flag when present,
        # else fall back to the raw R² gap for older callers).
        has_flag = 'Overfitting_Flag' in comparison_df.columns
        if has_flag or 'Overfitting_Gap' in comparison_df.columns:
            for i, (tb, ob) in enumerate(zip(train_bars, test_bars)):
                if has_flag:
                    is_overfit = bool(comparison_df.iloc[i].get('Overfitting_Flag', False))
                else:
                    is_overfit = (comparison_df.iloc[i].get('Overfitting_Gap', 0) or 0) > 0.15
                if is_overfit:
                    for bar in (tb, ob):
                        bar.set_edgecolor('red')
                        bar.set_linewidth(2)
                        bar.set_hatch('/')
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title('Train vs Test R²\n(red hatch = overfitting)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Overall Score
    ax = axes[1, 1]
    if 'Score' in comparison_df.columns:
        scores = comparison_df['Score']
        ax.barh(models, scores, color=colors)
        ax.set_xlabel('Overall Score', fontsize=11, fontweight='bold')
        ax.set_title('Overall Model Score', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight best model
        best_idx = comparison_df['Score'].idxmax()
        ax.get_children()[best_idx].set_color('gold')
        ax.get_children()[best_idx].set_edgecolor('darkgoldenrod')
        ax.get_children()[best_idx].set_linewidth(2)
    
    fig.tight_layout()
    return fig
