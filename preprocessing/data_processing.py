"""
Data processing utilities for spectral analysis

@author: Sharad Kumar Gupta
"""
import numpy as np
import pandas as pd

# Heavy libs imported lazily (see utils/lazy_imports.py) so importing this module
# at GUI startup stays cheap; SciPy / scikit-learn load on first preprocessing call.
from utils.lazy_imports import LazyModule, LazyCallable

interpolate = LazyModule('scipy.interpolate')
stats = LazyModule('scipy.stats')
savgol_filter = LazyCallable('scipy.signal', 'savgol_filter')
scipy_skew = LazyCallable('scipy.stats', 'skew')
scipy_kurtosis = LazyCallable('scipy.stats', 'kurtosis')
StandardScaler = LazyCallable('sklearn.preprocessing', 'StandardScaler')
PLSRegression = LazyCallable('sklearn.cross_decomposition', 'PLSRegression')
cross_val_score = LazyCallable('sklearn.model_selection', 'cross_val_score')
r2_score = LazyCallable('sklearn.metrics', 'r2_score')
mean_squared_error = LazyCallable('sklearn.metrics', 'mean_squared_error')
mean_absolute_error = LazyCallable('sklearn.metrics', 'mean_absolute_error')


# ---------------------------------------------------------------------------
# Wavelength unit detection
# ---------------------------------------------------------------------------

def detect_wavelength_unit(wavelengths):
    """
    Auto-detect whether wavelengths are in nanometres (nm) or micrometres (μm).

    Decision heuristic
    ------------------
    • max ≤ 25  → assume μm  (VSWIR 0.4–2.5 μm, LWIR 7.5–14 μm)
    • max >  25 → assume nm

    Returns
    -------
    unit : str        "nm" or "μm"
    wl_nm : list      wavelengths expressed in nm (float)
    """
    wl = [float(w) for w in wavelengths]
    if max(wl) <= 25.0:
        return "μm", [w * 1000.0 for w in wl]
    return "nm", wl


def infer_spectral_domain(wl_nm):
    """
    Infer the spectral domain from wavelengths already in nm.

    Returns one of: "VSWIR", "LWIR", "VSWIR+LWIR"

    Boundaries (nm):
      VSWIR : 350 – 2500
      LWIR  : 7000 – 14500
      Anything that spans both, or lies in MIR (2500–7000),
      is classified as VSWIR+LWIR.
    """
    mn, mx = min(wl_nm), max(wl_nm)
    vswir_end  = 2500.0
    lwir_start = 7000.0
    if mx <= vswir_end:
        return "VSWIR"
    if mn >= lwir_start:
        return "LWIR"
    return "VSWIR+LWIR"

def safe_interpolate_spectra(X, original_wavelengths, new_wavelengths, kind="linear"):
    """
    Safely interpolate spectra with comprehensive error handling.

    ``kind`` selects the ``scipy.interpolate.interp1d`` interpolation order:
    "linear" (default), "quadratic" (needs ≥ 3 source points) or "cubic" (needs
    ≥ 4).  A clear error is raised when there are too few source points for the
    requested order instead of letting scipy raise a cryptic message.
    """
    try:
        # Validate inputs
        if len(original_wavelengths) != X.shape[1]:
            raise ValueError(f"Wavelength count ({len(original_wavelengths)}) doesn't match spectral data columns ({X.shape[1]})")

        # Normalise to ndarrays so downstream boolean indexing works regardless
        # of whether callers pass Python lists or numpy arrays.
        original_wavelengths = np.asarray(original_wavelengths, dtype=float)
        new_wavelengths = np.asarray(new_wavelengths, dtype=float)

        # Check for valid wavelength ranges
        orig_min, orig_max = original_wavelengths.min(), original_wavelengths.max()
        new_min, new_max = new_wavelengths.min(), new_wavelengths.max()

        if new_min < orig_min or new_max > orig_max:
            # Silently clip new_wavelengths to the original data bounds rather
            # than raising – the filter step already tried to prevent this, but
            # floating-point drift can still produce tiny overshoots.
            new_wavelengths = new_wavelengths[
                (new_wavelengths >= orig_min) & (new_wavelengths <= orig_max)
            ]
            if len(new_wavelengths) == 0:
                raise ValueError(f"Interpolation range ({new_min:.1f}-{new_max:.1f} nm) exceeds original data range ({orig_min:.1f}-{orig_max:.1f} nm)")
            new_min, new_max = min(new_wavelengths), max(new_wavelengths)
        
        # Check for sufficient data points (order-dependent: nearest ≥ 1,
        # linear ≥ 2, quadratic ≥ 3, cubic ≥ 4).
        min_pts = {"nearest": 1, "linear": 2, "quadratic": 3, "cubic": 4}.get(kind, 2)
        if len(original_wavelengths) < min_pts:
            raise ValueError(
                f"'{kind}' interpolation needs at least {min_pts} wavelength "
                f"points, but only {len(original_wavelengths)} are available. "
                f"Widen the wavelength range or choose a lower-order method "
                f"(e.g. Linear Interpolation).")

        # Check for duplicate wavelengths
        if len(set(original_wavelengths)) != len(original_wavelengths):
            raise ValueError("Duplicate wavelengths found in original data")
        
        # Sort wavelengths if not already sorted
        sort_indices = np.argsort(original_wavelengths)
        sorted_wavelengths = np.array(original_wavelengths)[sort_indices]
        
        # Sort all spectra in one shot (vectorized column reorder)
        sorted_X = X[:, sort_indices]  # shape (n_samples, n_wavelengths)

        # Bulk NaN/Inf check — much faster than per-row validation
        if np.any(~np.isfinite(sorted_X)):
            bad_rows = np.where(np.any(~np.isfinite(sorted_X), axis=1))[0]
            raise ValueError(
                f"Invalid values (NaN/Inf) found in spectra at rows {bad_rows.tolist()}"
            )

        # Create ONE interpolator for ALL rows at once.
        # interp1d with axis=1 interpolates each row independently,
        # avoiding the per-row Python loop entirely.
        f = interpolate.interp1d(sorted_wavelengths, sorted_X,
                                 kind=kind, axis=1, bounds_error=True)
        interpolated_X = f(new_wavelengths)

        return interpolated_X
        
    except Exception as e:
        error_msg = f"Spectral interpolation failed: {str(e)}\n\n"
        error_msg += "Possible solutions:\n"
        error_msg += "1. Reduce the wavelength range (Min/Max Wave)\n"
        error_msg += "2. Increase spacing value\n"
        error_msg += "3. Check for missing or invalid spectral data\n"
        error_msg += "4. Disable resampling if not needed"
        raise ValueError(error_msg)

# ---------------------------------------------------------------------------
# Missing-data detection & handling
# ---------------------------------------------------------------------------
#
# Excel/CSV data frequently contains empty cells or non-numeric text in the
# spectral / property columns.  Left unhandled these become NaN (or Inf) and
# silently propagate into model training, producing confusing downstream errors
# ("Input contains NaN", overflow in scaling, etc.).  The helpers below let the
# GUI (a) report exactly how much data is missing right after loading and
# (b) apply a user-chosen strategy to clean it before training.

# User-selectable strategies (order = GUI dropdown order; first is the default).
MISSING_DATA_METHODS = [
    "Drop rows with missing",
    "Mean imputation",
    "Median imputation",
    "Spectral interpolation",
    "Fill with zero",
]


def analyze_missing_data(df, wavelength_cols=None, property_cols=None):
    """Summarise missing (NaN / empty / non-numeric) values in a loaded DataFrame.

    A cell counts as missing when it is NaN/None, or — for columns expected to
    be numeric (the wavelength columns) — when it cannot be coerced to a finite
    float (blank strings, text, Inf).

    Parameters
    ----------
    df : pandas.DataFrame
    wavelength_cols : list[str] | None   spectral (numeric) column names
    property_cols   : list[str] | None   target/property column names

    Returns
    -------
    dict with keys:
        total_missing       : int   total missing cells across the inspected columns
        rows_with_missing   : int   number of rows containing >= 1 missing cell
        n_rows              : int
        spectral_missing    : int   missing cells within wavelength columns
        property_missing    : dict  {property_name: missing_count} (only > 0)
        has_missing         : bool
    """
    wavelength_cols = list(wavelength_cols or [])
    property_cols = list(property_cols or [])
    inspected = [c for c in (wavelength_cols + property_cols) if c in df.columns]

    n_rows = len(df)
    if not inspected:
        return {'total_missing': 0, 'rows_with_missing': 0, 'n_rows': n_rows,
                'spectral_missing': 0, 'property_missing': {}, 'has_missing': False}

    # Numeric (wavelength) columns: coerce so blanks / text / Inf count as missing.
    spectral_cols = {}
    for c in wavelength_cols:
        if c in df.columns:
            num = pd.to_numeric(df[c], errors='coerce')
            spectral_cols[c] = ~np.isfinite(num.to_numpy(dtype=float))
    spectral_mask = pd.DataFrame(spectral_cols, index=df.index)

    property_missing = {}
    prop_cols = {}
    for c in property_cols:
        if c in df.columns:
            miss = df[c].isna()
            prop_cols[c] = miss
            cnt = int(miss.sum())
            if cnt:
                property_missing[c] = cnt
    prop_mask = pd.DataFrame(prop_cols, index=df.index)

    spectral_missing = int(spectral_mask.to_numpy().sum()) if len(spectral_mask.columns) else 0
    combined = pd.concat([spectral_mask, prop_mask], axis=1)
    rows_with_missing = int(combined.any(axis=1).sum()) if len(combined.columns) else 0
    total_missing = int(combined.to_numpy().sum()) if len(combined.columns) else 0

    return {
        'total_missing': total_missing,
        'rows_with_missing': rows_with_missing,
        'n_rows': n_rows,
        'spectral_missing': spectral_missing,
        'property_missing': property_missing,
        'has_missing': total_missing > 0,
    }


def handle_missing_data(X, y, method="Drop rows with missing"):
    """Clean missing (NaN/Inf) values in a feature matrix and target vector.

    Rows whose target ``y`` is missing can never be used for supervised training
    (the label is unknown) and are therefore ALWAYS dropped, regardless of the
    chosen ``method``.  The ``method`` only governs how missing values inside the
    spectral matrix ``X`` are treated:

      * "Drop rows with missing" – drop any sample that still has a missing
        band value.
      * "Mean imputation"        – replace each missing value with that band's
        (column) mean over the finite samples.
      * "Median imputation"      – as above, using the column median.
      * "Spectral interpolation" – linearly interpolate missing values along the
        wavelength axis within each spectrum (edges/all-NaN fall back to the
        band mean, then 0).
      * "Fill with zero"         – replace every missing value with 0.

    Parameters
    ----------
    X : array-like (n_samples, n_bands)
    y : array-like (n_samples,)
    method : str

    Returns
    -------
    (X_clean, y_clean, info) where ``info`` is a dict:
        rows_dropped_target  : int
        rows_dropped_missing : int
        cells_filled         : int
        method               : str
        message              : str  human-readable summary
        n_remaining          : int

    Raises
    ------
    ValueError if no usable samples remain.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_start = X.shape[0]

    # 1) Always drop rows with a missing / non-finite target.
    target_ok = np.isfinite(y)
    rows_dropped_target = int((~target_ok).sum())
    X = X[target_ok]
    y = y[target_ok]

    if X.shape[0] == 0:
        raise ValueError(
            "Every sample has a missing target value — nothing left to train on. "
            "Check the selected property column.")

    rows_dropped_missing = 0
    cells_filled = 0
    missing = ~np.isfinite(X)

    if missing.any():
        if method == "Drop rows with missing":
            keep = ~missing.any(axis=1)
            rows_dropped_missing = int((~keep).sum())
            X = X[keep]
            y = y[keep]
            missing = missing[keep]
        elif method in ("Mean imputation", "Median imputation"):
            cells_filled = int(missing.sum())
            Xm = np.where(missing, np.nan, X)
            with np.errstate(all='ignore'):
                if method == "Mean imputation":
                    fill = np.nanmean(Xm, axis=0)
                else:
                    fill = np.nanmedian(Xm, axis=0)
            # Bands that are entirely missing have no statistic — use 0.
            fill = np.where(np.isfinite(fill), fill, 0.0)
            X = np.where(missing, fill[None, :], X)
        elif method == "Spectral interpolation":
            cells_filled = int(missing.sum())
            n_bands = X.shape[1]
            xp = np.arange(n_bands, dtype=float)
            # Per-band means as a fallback for edges / fully-missing bands.
            with np.errstate(all='ignore'):
                band_mean = np.nanmean(np.where(missing, np.nan, X), axis=0)
            band_mean = np.where(np.isfinite(band_mean), band_mean, 0.0)
            for i in range(X.shape[0]):
                row_missing = missing[i]
                if not row_missing.any():
                    continue
                good = ~row_missing
                if good.sum() >= 2:
                    X[i, row_missing] = np.interp(
                        xp[row_missing], xp[good], X[i, good])
                elif good.sum() == 1:
                    X[i, row_missing] = X[i, good][0]
                else:
                    X[i, row_missing] = band_mean[row_missing]
        elif method == "Fill with zero":
            cells_filled = int(missing.sum())
            X = np.where(missing, 0.0, X)
        else:
            raise ValueError(f"Unknown missing-data method: {method!r}")

    # Final safety net (e.g. an all-NaN band under interpolation).
    residual = ~np.isfinite(X)
    if residual.any():
        cells_filled += int(residual.sum())
        X = np.where(residual, 0.0, X)

    if X.shape[0] == 0:
        raise ValueError(
            "No samples remain after dropping rows with missing values. "
            "Try an imputation strategy instead of 'Drop rows with missing'.")

    parts = []
    if rows_dropped_target:
        parts.append(f"dropped {rows_dropped_target} row(s) with a missing target")
    if rows_dropped_missing:
        parts.append(f"dropped {rows_dropped_missing} row(s) with missing bands")
    if cells_filled:
        parts.append(f"filled {cells_filled} missing cell(s) via '{method}'")
    if parts:
        message = ("Missing-data handling: " + "; ".join(parts) +
                   f". {X.shape[0]}/{n_start} sample(s) used.")
    else:
        message = ""

    info = {
        'rows_dropped_target': rows_dropped_target,
        'rows_dropped_missing': rows_dropped_missing,
        'cells_filled': cells_filled,
        'method': method,
        'message': message,
        'n_remaining': int(X.shape[0]),
    }
    return X, y, info


def preprocess_spectra(spectra, method, **kwargs):
    """
    Apply preprocessing to spectral data
    
    Args:
        spectra: Spectral data array
        method: Preprocessing method name
        **kwargs: Additional parameters for specific methods
            - window_length, polyorder for Smoothing
            - outlier_method, threshold for Outlier Removal
    """
    try:
        if method == "Smoothing":
            window_length = kwargs.get('window_length', 11)
            polyorder = kwargs.get('polyorder', 2)
            
            # Ensure window_length is odd and valid
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(polyorder + 1, window_length)
            window_length = min(window_length, spectra.shape[1])
            
            # Apply Savitzky-Golay filter to the entire 2D array at once (axis=1)
            processed = savgol_filter(spectra, window_length, polyorder, axis=1)
            return processed
        
        elif method == "Spectral Outlier Removal":
            outlier_method = kwargs.get('outlier_method', 'zscore')
            threshold = kwargs.get('threshold', 3.0)
            
            # Identify outlier samples
            if outlier_method == 'zscore':
                # Calculate z-scores for each sample's mean spectrum value
                sample_means = np.mean(spectra, axis=1)
                z_scores = np.abs(stats.zscore(sample_means))
                mask = z_scores < threshold
            elif outlier_method == 'iqr':
                # IQR method
                sample_means = np.mean(spectra, axis=1)
                Q1 = np.percentile(sample_means, 25)
                Q3 = np.percentile(sample_means, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (sample_means >= lower_bound) & (sample_means <= upper_bound)
            else:
                return spectra
            
            # Return both filtered data and mask
            return spectra[mask, :], mask
        
        elif method == "Continuum Removal":
            # Vectorized continuum removal across all samples at once
            hull = np.maximum.accumulate(spectra, axis=1)
            hull = np.maximum.accumulate(hull[:, ::-1], axis=1)[:, ::-1]
            processed = spectra / (hull + 1e-10)  # Avoid division by zero
            return processed
            
        elif method == "First Derivative":
            return np.gradient(spectra, axis=1)
            
        elif method == "Second Derivative":
            return np.gradient(np.gradient(spectra, axis=1), axis=1)
            
        elif method == "Absorbance":
            # Ensure no zero or negative values
            spectra_safe = np.maximum(spectra, 1e-10)
            return -np.log10(spectra_safe)

        elif method == "Baseline Correction":
            # Proper baseline correction using the lower-envelope (rubber-band) approach.
            # The baseline is estimated from the spectral MINIMA, not through peaks,
            # so that subtracting it enhances absorption features without collapsing
            # peak values to near-zero (which would destroy feature variance).
            baseline_method = kwargs.get('baseline_method', 'linear')
            processed = np.zeros_like(spectra, dtype=float)
            n = spectra.shape[1]
            x = np.arange(n, dtype=float)

            def _lower_envelope_anchors(sp, n_anchors):
                """Return (ax, ay) anchor points at the local minima of each segment."""
                anchors_idx = []
                seg_size = max(1, n // n_anchors)
                for seg in range(n_anchors):
                    start = seg * seg_size
                    end = min(n, (seg + 1) * seg_size)
                    if start < end:
                        local_min = start + int(np.argmin(sp[start:end]))
                        anchors_idx.append(local_min)
                # Always include true endpoints
                anchors_idx = sorted(set([0, n - 1] + anchors_idx))
                return np.array(anchors_idx, dtype=float), sp[anchors_idx]

            if baseline_method == 'linear':
                # Vectorised rubber-band baseline — no Python loop over samples.
                # Use the minimum of the first/last 5% of bands as endpoint anchors
                # so a strong peak at band 0 or n-1 cannot inflate the baseline.
                if n == 1:
                    return np.zeros_like(spectra, dtype=float)
                edge = max(1, n // 20)
                left_vals = np.min(spectra[:, :edge], axis=1)              # (n_samples,)
                right_vals = np.min(spectra[:, max(0, n - edge):], axis=1)  # (n_samples,)
                t = x / float(n - 1)                                        # (n,)  in [0, 1]
                baselines = (left_vals[:, None] * (1.0 - t)
                             + right_vals[:, None] * t)                     # (n_samples, n)
                # Clamp to lower envelope and subtract
                baselines = np.minimum(baselines, spectra)
                return spectra - baselines
            else:  # polynomial — anchor positions differ per spectrum, keep loop
                for i in range(spectra.shape[0]):
                    sp = spectra[i, :].astype(float)
                    degree = int(kwargs.get('degree', 3))
                    # Clamp degree so it cannot exceed (n_anchors - 1)
                    n_anchors = max(degree + 3, 6)
                    ax, ay = _lower_envelope_anchors(sp, n_anchors)
                    # Fit through the lower-envelope anchors (not through peaks)
                    d = min(degree, len(ax) - 1)
                    with np.errstate(all='ignore'):
                        coeffs = np.polyfit(ax, ay, d)
                    baseline = np.polyval(coeffs, x)
                    # Ensure baseline never exceeds the actual spectrum (clamp to lower envelope)
                    baseline = np.minimum(baseline, sp)
                    corrected = sp - baseline
                    # corrected is already >= 0 because baseline <= sp everywhere;
                    # do NOT apply an additional global shift (which was inflating variances).
                    processed[i, :] = corrected
                return processed

        return spectra

    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")

def calculate_statistics(data):
    """
    Calculate comprehensive statistics for data
    """
    try:
        stats = {
            'Count': len(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data),
            'Min': np.min(data),
            'Q1': np.percentile(data, 25),
            'Median': np.median(data),
            'Q3': np.percentile(data, 75),
            'Max': np.max(data),
            'Skewness': float(scipy_skew(data, bias=False)),
            'Kurtosis': float(scipy_kurtosis(data, fisher=True, bias=False))
        }
        return stats
        
    except Exception as e:
        raise Exception(f"Statistics calculation failed: {str(e)}")

# ---------------------------------------------------------------------------
# Satellite / airborne sensor band definitions (nominal band centres, nm)
# ---------------------------------------------------------------------------
#
# Each entry maps a sensor label (as shown in the GUI dropdown) to:
#   "reflective"      : band centres in the VIS-NIR-SWIR (~350-2500 nm) range
#   "reflective_fwhm" : (optional) matching Full-Width-at-Half-Maximum (nm) of
#                       each reflective band, used by SRF-convolution resampling
#   "thermal"         : Thermal-Infrared (TIR/LWIR) band centres, if any
#   "thermal_fwhm"    : (optional) matching FWHM of the thermal bands
#
# Where "*_fwhm" is omitted (the hyperspectral sensors), the FWHM of each band
# is estimated from the local band spacing — a good approximation for the
# Nyquist-sampled hyperspectral instruments (EnMAP / EMIT / PRISMA).
#
# When the loaded data lies in the VIS-NIR-SWIR range only, the thermal bands
# are NOT used (they would fall far outside the data and only cause
# extrapolation errors) — this satisfies the requirement to exclude the
# Thermal-Infrared range of Sentinel-2 / Landsat-8 for VIS-NIR-SWIR data.
#
# Sentinel-2 (MSI) and Landsat-8 (OLI/TIRS) use the published multispectral
# band centres.  EnMAP and EMIT use the instruments' nominal hyperspectral
# band centres.  PRISMA is represented by its nominal ~10 nm sampling grid
# across its 400-2500 nm range (its exact centres vary slightly per scene).
# ---------------------------------------------------------------------------

# Sentinel-2A MSI reflective band centres (B1-B9, B11, B12) and their published
# FWHM bandwidths (nm). Sentinel-2 has no thermal channels.
_S2_REFLECTIVE = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
_S2_FWHM       = [ 20,  65,  35,  30,  15,  15,  20, 115,  20,  20,   90,  180]

# Landsat-8 OLI reflective band centres (coastal, blue, green, red, NIR,
# SWIR-1, SWIR-2) with FWHM, and TIRS thermal band centres (B10, B11) with FWHM.
_L8_REFLECTIVE = [443, 482, 561, 655, 865, 1609, 2201]
_L8_FWHM       = [ 16,  60,  57,  37,  28,   85,  187]
_L8_THERMAL      = [10895, 12005]
_L8_THERMAL_FWHM = [  590,  1010]

# EnMAP nominal band centres (VNIR + SWIR).
_ENMAP_REFLECTIVE = [
    418, 424, 429, 435, 440, 445, 450, 454, 459, 464, 468, 473, 478, 482, 487,
    492, 496, 501, 506, 511, 516, 521, 525, 530, 535, 540, 546, 551, 556, 561,
    566, 572, 577, 583, 588, 594, 599, 605, 611, 617, 623, 629, 635, 641, 648,
    654, 660, 667, 673, 680, 686, 693, 700, 707, 714, 721, 728, 735, 742, 749,
    756, 764, 771, 779, 786, 794, 801, 809, 817, 824, 832, 840, 848, 856, 864,
    872, 880, 888, 896, 902, 904, 912, 920, 921, 928, 931, 936, 941, 944, 951,
    953, 961, 962, 969, 972, 977, 983, 985, 993, 1004, 1015, 1026, 1037, 1048,
    1059, 1070, 1082, 1093, 1105, 1116, 1128, 1140, 1152, 1163, 1175, 1187,
    1199, 1211, 1223, 1235, 1247, 1259, 1271, 1283, 1295, 1307, 1319, 1331,
    1343, 1355, 1367, 1379, 1390, 1449, 1461, 1473, 1484, 1496, 1507, 1519,
    1530, 1542, 1553, 1564, 1576, 1587, 1598, 1609, 1620, 1631, 1642, 1653,
    1664, 1675, 1685, 1696, 1707, 1718, 1728, 1739, 1749, 1760, 1770, 1780,
    1968, 1977, 1986, 1996, 2005, 2014, 2024, 2033, 2042, 2051, 2060, 2069,
    2078, 2087, 2096, 2105, 2113, 2122, 2131, 2140, 2148, 2157, 2165, 2174,
    2183, 2191, 2199, 2208, 2216, 2225, 2233, 2241, 2249, 2258, 2266, 2274,
    2282, 2290, 2298, 2306, 2314, 2322, 2330, 2338, 2346, 2354, 2361, 2369,
    2377, 2385, 2392, 2400, 2408, 2415, 2423, 2430, 2438, 2445,
]

# EMIT nominal band centres.
_EMIT_REFLECTIVE = [
    381, 388, 396, 403, 411, 418, 426, 433, 440, 448, 455, 463, 470, 478, 485,
    493, 500, 507, 515, 522, 530, 537, 545, 552, 559, 567, 574, 582, 589, 597,
    604, 612, 619, 626, 634, 641, 649, 656, 664, 671, 678, 686, 693, 701, 708,
    716, 723, 731, 738, 745, 753, 760, 768, 775, 783, 790, 797, 805, 812, 820,
    827, 835, 842, 850, 857, 864, 872, 879, 887, 894, 902, 909, 916, 924, 931,
    939, 946, 954, 961, 968, 976, 983, 991, 998, 1006, 1013, 1021, 1028, 1035,
    1043, 1050, 1058, 1065, 1073, 1080, 1087, 1095, 1102, 1110, 1117, 1125,
    1132, 1140, 1147, 1154, 1162, 1169, 1177, 1184, 1192, 1199, 1206, 1214,
    1221, 1229, 1236, 1244, 1251, 1259, 1266, 1273, 1281, 1288, 1296, 1303,
    1311, 1318, 1325, 1333, 1340, 1348, 1355, 1363, 1370, 1378, 1385, 1392,
    1400, 1407, 1415, 1422, 1430, 1437, 1444, 1452, 1459, 1467, 1474, 1482,
    1489, 1496, 1504, 1511, 1519, 1526, 1534, 1541, 1549, 1556, 1563, 1571,
    1578, 1586, 1593, 1601, 1608, 1615, 1623, 1630, 1638, 1645, 1653, 1660,
    1668, 1675, 1682, 1690, 1697, 1705, 1712, 1720, 1727, 1734, 1742, 1749,
    1757, 1764, 1772, 1779, 1787, 1794, 1801, 1809, 1816, 1824, 1831, 1839,
    1846, 1853, 1861, 1868, 1876, 1883, 1891, 1898, 1906, 1913, 1920, 1928,
    1935, 1943, 1950, 1958, 1965, 1972, 1980, 1987, 1995, 2002, 2010, 2017,
    2024, 2032, 2039, 2047, 2054, 2062, 2069, 2077, 2084, 2091, 2099, 2106,
    2114, 2121, 2129, 2136, 2143, 2151, 2158, 2166, 2173, 2181, 2188, 2196,
    2203, 2210, 2218, 2225, 2233, 2240, 2248, 2255, 2262, 2270, 2277, 2285,
    2292, 2300, 2307, 2315, 2322, 2329, 2337, 2344, 2352, 2359, 2367, 2374,
    2381, 2389, 2396, 2404, 2411, 2419, 2426, 2434, 2441, 2448, 2456, 2463,
    2471, 2478, 2486, 2493,
]

# PRISMA nominal band centres — ~10 nm sampling across its 400-2500 nm range
# (239 bands). Exact centres are scene-dependent; this nominal grid is used as
# the resampling target.
_PRISMA_REFLECTIVE = [round(float(w)) for w in np.linspace(402, 2497, 239)]

# Landsat-9 OLI-2 — spectrally identical to Landsat-8 OLI (same band centres and
# FWHM), including the two TIRS-2 thermal bands.
_L9_REFLECTIVE = [443, 482, 561, 655, 865, 1609, 2201]
_L9_FWHM       = [ 16,  60,  57,  37,  28,   85,  187]
_L9_THERMAL      = [10895, 12005]
_L9_THERMAL_FWHM = [  590,  1010]

# Landsat-7 ETM+ reflective bands (B1-B5, B7) with FWHM, and thermal band B6
# (~11.45 µm).
_L7_REFLECTIVE = [483, 560, 662, 835, 1648, 2206]
_L7_FWHM       = [ 70,  80,  60, 130,  200,  260]
_L7_THERMAL      = [11450]
_L7_THERMAL_FWHM = [ 2100]

# Landsat-5 TM reflective bands (B1-B5, B7) with FWHM, and thermal band B6
# (~11.45 µm).
_L5_REFLECTIVE = [485, 560, 660, 830, 1650, 2215]
_L5_FWHM       = [ 70,  80,  60, 140,  200,  270]
_L5_THERMAL      = [11450]
_L5_THERMAL_FWHM = [ 2100]

# VENµS (VENUS) — 12 narrow VNIR super-spectral bands (user-provided centres).
_VENUS_REFLECTIVE = [424, 447, 492, 555, 620, 621, 666, 702, 741, 782, 861, 909]
_VENUS_FWHM       = [ 40,  40,  40,  40,  40,  40,  30,  25,  16,  20,  40,  20]

# PlanetScope (SuperDove, PSB.SD) — 8 VNIR bands (coastal-blue, blue, green-I,
# green, yellow, red, red-edge, NIR) with nominal FWHM.
_PLANETSCOPE_REFLECTIVE = [443, 490, 531, 565, 610, 665, 705, 865]
_PLANETSCOPE_FWHM       = [ 20,  50,  36,  36,  20,  31,  15,  40]

# DESIS (DLR Earth Sensing Imaging Spectrometer) — hyperspectral VNIR, 235 bands
# over ~401-1000 nm at ~2.55 nm sampling (FWHM ~3.5 nm, derived from spacing).
_DESIS_REFLECTIVE = [round(float(w), 1) for w in np.linspace(402.0, 999.0, 235)]

SATELLITE_SENSORS = {
    "Sentinel-2": {"reflective": _S2_REFLECTIVE, "reflective_fwhm": _S2_FWHM,
                   "thermal": [], "thermal_fwhm": []},
    "Landsat-5": {"reflective": _L5_REFLECTIVE, "reflective_fwhm": _L5_FWHM,
                  "thermal": _L5_THERMAL, "thermal_fwhm": _L5_THERMAL_FWHM},
    "Landsat-7": {"reflective": _L7_REFLECTIVE, "reflective_fwhm": _L7_FWHM,
                  "thermal": _L7_THERMAL, "thermal_fwhm": _L7_THERMAL_FWHM},
    "Landsat-8": {"reflective": _L8_REFLECTIVE, "reflective_fwhm": _L8_FWHM,
                  "thermal": _L8_THERMAL, "thermal_fwhm": _L8_THERMAL_FWHM},
    "Landsat-9": {"reflective": _L9_REFLECTIVE, "reflective_fwhm": _L9_FWHM,
                  "thermal": _L9_THERMAL, "thermal_fwhm": _L9_THERMAL_FWHM},
    "VENUS": {"reflective": _VENUS_REFLECTIVE, "reflective_fwhm": _VENUS_FWHM,
              "thermal": [], "thermal_fwhm": []},
    "PlanetScope": {"reflective": _PLANETSCOPE_REFLECTIVE,
                    "reflective_fwhm": _PLANETSCOPE_FWHM,
                    "thermal": [], "thermal_fwhm": []},
    "DESIS": {"reflective": _DESIS_REFLECTIVE, "thermal": []},
    "EnMAP": {"reflective": _ENMAP_REFLECTIVE, "thermal": []},
    "EMIT": {"reflective": _EMIT_REFLECTIVE, "thermal": []},
    "PRISMA": {"reflective": _PRISMA_REFLECTIVE, "thermal": []},
}

# Sensor dropdown options (the first entry keeps the original uniform-spacing
# behaviour).
SENSOR_OPTIONS = ["Custom"] + list(SATELLITE_SENSORS.keys())

# Wavelength (nm) above which a band is considered Thermal-Infrared. Used to
# decide whether the loaded data actually reaches the thermal range.
_THERMAL_THRESHOLD_NM = 3000.0

# ---------------------------------------------------------------------------
# Resampling method registry
# ---------------------------------------------------------------------------
#
# Six user-selectable resampling methods (GUI dropdown order):
#   * Interpolation family — reproject onto the target grid without any notion
#     of instrument bandwidth:
#       "Linear Interpolation"    — piecewise-linear (interp1d kind='linear')
#       "Quadratic Interpolation" — 2nd-order spline (kind='quadratic', ≥3 pts)
#       "Cubic Spline"            — natural cubic spline (kind='cubic', ≥4 pts)
#   * Bandwidth-aware family — integrate the source spectrum under each target
#     band's spectral-response function; every one of these needs a per-band
#     FWHM (or an explicit response curve):
#       "Gaussian SRF"  — analytic Gaussian response (σ = FWHM / 2.3548, ±3σ)
#       "Empirical SRF" — integrate against an uploaded per-band response curve;
#                         falls back to the Gaussian response where no curve is
#                         supplied
#       "Band Averaging"— flat (tophat) response over each band's FWHM window
# ---------------------------------------------------------------------------
RESAMPLE_METHODS = [
    "Linear Interpolation",
    "Nearest Neighbour Interpolation",
    "Quadratic Interpolation",
    "Cubic Spline",
    "Gaussian SRF",
    "Empirical SRF",
    "Band Averaging",
]

# Methods that consume a per-band FWHM (the bandwidth-aware family).
_FWHM_METHODS = {"Gaussian SRF", "Empirical SRF", "Band Averaging"}

# Map the interpolation methods to their scipy ``interp1d`` ``kind``.  Nearest-
# neighbour picks each target band's closest source band unchanged — useful when
# linear interpolation would smear sharp absorption features across a coarse grid.
_INTERP_KINDS = {
    "Linear Interpolation": "linear",
    "Nearest Neighbour Interpolation": "nearest",
    "Quadratic Interpolation": "quadratic",
    "Cubic Spline": "cubic",
}


def estimate_fwhms_from_grid(wavelengths):
    """Estimate a per-band FWHM (nm) from band spacing for a wavelength grid that
    carries no explicit bandwidths (e.g. tabular training wavelengths).  Uses the
    nearest-neighbour gap so bands bordering a spectral gap aren't over-widened."""
    wl = np.asarray(wavelengths, dtype=float)
    if len(wl) < 2:
        return np.ones_like(wl)
    order = np.argsort(wl)
    ws = wl[order]
    diffs = np.diff(ws)
    left = np.concatenate(([np.inf], diffs))
    right = np.concatenate((diffs, [np.inf]))
    local = np.minimum(left, right)
    fwhms = np.where(np.isfinite(local) & (local > 0), local, 1.0)
    # Restore original band order.
    out = np.empty_like(fwhms)
    out[order] = fwhms
    return out


def normalize_resample_method(method):
    """Return the canonical :data:`RESAMPLE_METHODS` label for ``method``.

    Accepts the current labels as-is and maps the legacy labels that older saved
    models / configs may carry ("Interpolation" → "Linear Interpolation",
    "SRF" / "SRF Convolution" → "Gaussian SRF").  Unknown / empty values default
    to "Linear Interpolation" so a bad string never crashes prediction.
    """
    if not method:
        return "Linear Interpolation"
    m = str(method).strip()
    if m in RESAMPLE_METHODS:
        return m
    upper = m.upper()
    if upper.startswith("SRF"):
        return "Gaussian SRF"
    if "INTERP" in upper:
        return "Linear Interpolation"
    return "Linear Interpolation"


def _needs_fwhm(method):
    """True when ``method`` is a bandwidth-aware method that requires FWHMs."""
    return normalize_resample_method(method) in _FWHM_METHODS


def get_sensor_bands(sensor, include_thermal=False):
    """Return the nominal band centres and FWHMs (nm) for a sensor.

    Args:
        sensor: Sensor label (must be a key of ``SATELLITE_SENSORS``).
        include_thermal: When True, the sensor's Thermal-Infrared bands are
            appended.  For VIS-NIR-SWIR data this should stay False so the
            thermal bands of Sentinel-2 / Landsat-8 are excluded.

    Returns:
        (centers, fwhms): two float ndarrays of equal length, sorted by centre
        and de-duplicated.  Where a sensor provides no explicit FWHM, each band's
        FWHM is estimated from the local band spacing.

    Raises:
        ValueError: If ``sensor`` is unknown.
    """
    if sensor not in SATELLITE_SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. "
                         f"Valid options: {', '.join(SATELLITE_SENSORS)}")
    spec = SATELLITE_SENSORS[sensor]

    def _pairs(centers_key, fwhm_key):
        centers = list(spec.get(centers_key, []))
        fwhms = list(spec.get(fwhm_key, []))
        if fwhms and len(fwhms) != len(centers):
            raise ValueError(f"Sensor '{sensor}': {fwhm_key} length "
                             f"({len(fwhms)}) does not match {centers_key} "
                             f"({len(centers)}).")
        # None marks "no explicit FWHM" — filled from spacing later.
        return [(float(c), (float(fwhms[i]) if fwhms else None))
                for i, c in enumerate(centers)]

    pairs = _pairs("reflective", "reflective_fwhm")
    if include_thermal:
        pairs += _pairs("thermal", "thermal_fwhm")

    # De-duplicate by centre (keep first FWHM seen) and sort.
    by_center = {}
    for c, f in pairs:
        by_center.setdefault(c, f)
    centers = np.array(sorted(by_center), dtype=float)
    fwhms = np.array([by_center[c] if by_center[c] is not None else np.nan
                      for c in centers], dtype=float)

    # Fill any missing FWHMs from the local band spacing.  Use the NEAREST
    # neighbour gap (min of the left/right spacing) rather than the average, so
    # bands bordering a spectral gap (e.g. EnMAP's water-absorption gaps around
    # 1390-1449 / 1780-1968 nm) are not assigned an inflated FWHM.  For a
    # Nyquist-sampled hyperspectral sensor FWHM ≈ the sampling interval.
    if np.any(np.isnan(fwhms)) and len(centers) >= 2:
        diffs = np.diff(centers)
        left = np.concatenate(([np.inf], diffs))      # gap to previous band
        right = np.concatenate((diffs, [np.inf]))     # gap to next band
        local = np.minimum(left, right)               # endpoints use their one gap
        fwhms = np.where(np.isnan(fwhms), local, fwhms)
    # Guard against any residual non-positive / non-finite width.
    fwhms = np.where(np.isfinite(fwhms) & (fwhms > 0), fwhms, 1.0)
    return centers, fwhms


def srf_convolve_spectra(X, original_wavelengths, centers, fwhms):
    """Resample spectra onto sensor bands by Gaussian spectral-response
    convolution (physically-weighted band averaging).

    Each output band ``b`` is the response-weighted average of the input
    spectrum::

        R_b = Σ_i w_i S_i / Σ_i w_i ,   w_i = g_b(λ_i) · Δλ_i

    where ``g_b`` is a Gaussian Spectral Response Function centred on the band
    centre with the band's FWHM (σ = FWHM / 2.3548), truncated at ±3σ, and
    ``Δλ_i`` is the local sampling interval of the input grid (so non-uniform
    input grids integrate correctly).

    Args:
        X: (n_samples, n_bands) spectra.
        original_wavelengths: length-n_bands wavelengths (nm) of ``X``.
        centers: target band centres (nm).
        fwhms: matching FWHMs (nm), same length as ``centers``.

    Returns:
        (n_samples, n_centers) resampled spectra.
    """
    try:
        owl = np.asarray(original_wavelengths, dtype=float)
        centers = np.asarray(centers, dtype=float)
        fwhms = np.asarray(fwhms, dtype=float)

        if len(owl) != X.shape[1]:
            raise ValueError(f"Wavelength count ({len(owl)}) doesn't match "
                             f"spectral data columns ({X.shape[1]})")
        if len(centers) != len(fwhms):
            raise ValueError("centers and fwhms must have the same length")
        if len(owl) < 2:
            raise ValueError("Need at least 2 wavelength points for convolution")

        # Sort the input grid and reorder the spectra columns to match.
        order = np.argsort(owl)
        owl = owl[order]
        Xs = X[:, order]

        if np.any(~np.isfinite(Xs)):
            bad = np.where(np.any(~np.isfinite(Xs), axis=1))[0]
            raise ValueError(f"Invalid values (NaN/Inf) found in spectra at "
                             f"rows {bad.tolist()}")

        # Local sampling interval (nm) for trapezoid-style integration weights.
        dlambda = np.abs(np.gradient(owl))

        sigma = fwhms / 2.3548200450309493  # FWHM -> Gaussian sigma
        # Gaussian response of every target band evaluated on the input grid.
        diff = owl[None, :] - centers[:, None]           # (n_centers, n_bands)
        resp = np.exp(-0.5 * (diff / sigma[:, None]) ** 2)
        resp[np.abs(diff) > 3.0 * sigma[:, None]] = 0.0  # truncate tails
        weights = resp * dlambda[None, :]                # integration weights

        norm = weights.sum(axis=1)                       # (n_centers,)
        zero = norm <= 0
        if np.any(zero):
            bad_c = centers[zero]
            raise ValueError(
                "No input samples fall under the spectral response of band(s) "
                f"centred at {np.round(bad_c, 1).tolist()} nm. The input grid "
                "is too coarse for SRF convolution at these bands."
            )
        weights /= norm[:, None]

        return Xs @ weights.T                            # (n_samples, n_centers)

    except Exception as e:
        error_msg = f"SRF convolution failed: {str(e)}\n\n"
        error_msg += "Possible solutions:\n"
        error_msg += "1. Use 'Interpolation' resampling instead of 'SRF Convolution'\n"
        error_msg += "2. Reduce the wavelength range (Min/Max Wave)\n"
        error_msg += "3. Check for missing or invalid spectral data"
        raise ValueError(error_msg)


def band_average_spectra(X, original_wavelengths, centers, fwhms):
    """Resample spectra by flat (tophat) band averaging.

    Each output band ``b`` is the plain, Δλ-weighted mean of the input samples
    that fall inside the band's FWHM window ``[c_b - FWHM_b/2, c_b + FWHM_b/2]``::

        R_b = Σ_i (S_i · Δλ_i) / Σ_i Δλ_i ,  for λ_i in the window

    where ``Δλ_i`` is the local sampling interval of the input grid so
    non-uniform grids integrate correctly.  This is the box-response counterpart
    to :func:`srf_convolve_spectra` (which uses a Gaussian response).

    Args:
        X: (n_samples, n_bands) spectra.
        original_wavelengths: length-n_bands wavelengths (nm) of ``X``.
        centers: target band centres (nm).
        fwhms: matching FWHMs (nm), same length as ``centers``.

    Returns:
        (n_samples, n_centers) resampled spectra.
    """
    try:
        owl = np.asarray(original_wavelengths, dtype=float)
        centers = np.asarray(centers, dtype=float)
        fwhms = np.asarray(fwhms, dtype=float)

        if len(owl) != X.shape[1]:
            raise ValueError(f"Wavelength count ({len(owl)}) doesn't match "
                             f"spectral data columns ({X.shape[1]})")
        if len(centers) != len(fwhms):
            raise ValueError("centers and fwhms must have the same length")
        if len(owl) < 1:
            raise ValueError("Need at least 1 wavelength point for band averaging")

        order = np.argsort(owl)
        owl = owl[order]
        Xs = X[:, order]

        if np.any(~np.isfinite(Xs)):
            bad = np.where(np.any(~np.isfinite(Xs), axis=1))[0]
            raise ValueError(f"Invalid values (NaN/Inf) found in spectra at "
                             f"rows {bad.tolist()}")

        dlambda = np.abs(np.gradient(owl)) if len(owl) > 1 else np.ones_like(owl)

        half = fwhms / 2.0
        # Boolean membership of each input band in each target window.
        inside = ((owl[None, :] >= (centers - half)[:, None]) &
                  (owl[None, :] <= (centers + half)[:, None]))   # (n_centers, n_bands)
        weights = inside * dlambda[None, :]

        norm = weights.sum(axis=1)
        empty = norm <= 0
        if np.any(empty):
            bad_c = centers[empty]
            raise ValueError(
                "No input samples fall inside the FWHM window of band(s) centred "
                f"at {np.round(bad_c, 1).tolist()} nm. The input grid is too "
                "coarse (or the FWHM too narrow) for band averaging at these "
                "bands. Widen the FWHM, reduce the wavelength range, or use an "
                "interpolation method instead.")
        weights = weights / norm[:, None]
        return Xs @ weights.T

    except Exception as e:
        raise ValueError(f"Band averaging failed: {str(e)}")


def empirical_srf_integrate(X, original_wavelengths, centers, fwhms, srf_table=None):
    """Resample spectra by integrating against an empirical spectral response.

    For each target band the input spectrum is integrated under a per-band
    response curve supplied in ``srf_table`` (as loaded from an SRF CSV)::

        R_b = Σ_i (S_i · r_b(λ_i) · Δλ_i) / Σ_i (r_b(λ_i) · Δλ_i)

    where ``r_b`` is the band's response interpolated onto the input grid.  Any
    band **without** a usable curve — or when ``srf_table`` is ``None`` — falls
    back to the analytic Gaussian response of :func:`srf_convolve_spectra`, so
    this method always produces a full output even with a partial SRF file.

    Args:
        X: (n_samples, n_bands) spectra.
        original_wavelengths: length-n_bands wavelengths (nm) of ``X``.
        centers: target band centres (nm).
        fwhms: matching FWHMs (nm) — used for the Gaussian fallback.
        srf_table: optional ``{center_nm: (wl_array, response_array)}`` mapping.

    Returns:
        (n_samples, n_centers) resampled spectra.
    """
    try:
        owl = np.asarray(original_wavelengths, dtype=float)
        centers = np.asarray(centers, dtype=float)
        fwhms = np.asarray(fwhms, dtype=float)

        if len(owl) != X.shape[1]:
            raise ValueError(f"Wavelength count ({len(owl)}) doesn't match "
                             f"spectral data columns ({X.shape[1]})")
        if len(centers) != len(fwhms):
            raise ValueError("centers and fwhms must have the same length")

        # No table at all → the method degenerates to the Gaussian SRF.
        if not srf_table:
            return srf_convolve_spectra(X, owl, centers, fwhms)

        order = np.argsort(owl)
        owl_s = owl[order]
        Xs = X[:, order]
        if np.any(~np.isfinite(Xs)):
            bad = np.where(np.any(~np.isfinite(Xs), axis=1))[0]
            raise ValueError(f"Invalid values (NaN/Inf) found in spectra at "
                             f"rows {bad.tolist()}")
        dlambda = np.abs(np.gradient(owl_s)) if len(owl_s) > 1 else np.ones_like(owl_s)

        # Match each target centre to the nearest key in the SRF table (within a
        # small tolerance) so minor rounding between the sensor grid and the CSV
        # does not drop a curve.
        table_keys = np.array(sorted(srf_table.keys()), dtype=float) if srf_table else np.array([])

        def _curve_for(center):
            if len(table_keys) == 0:
                return None
            j = int(np.argmin(np.abs(table_keys - center)))
            if abs(table_keys[j] - center) > max(1.0, 0.05 * center):
                return None
            return srf_table[table_keys[j]]

        n_samples = Xs.shape[0]
        out = np.empty((n_samples, len(centers)), dtype=float)
        # Bands that must use the Gaussian fallback (collected then done in bulk).
        fallback_idx = []
        for b, c in enumerate(centers):
            curve = _curve_for(c)
            if curve is None:
                fallback_idx.append(b)
                continue
            wl_r, resp_r = np.asarray(curve[0], dtype=float), np.asarray(curve[1], dtype=float)
            # Interpolate the response onto the input grid (0 outside its span).
            resp = np.interp(owl_s, wl_r, resp_r, left=0.0, right=0.0)
            w = resp * dlambda
            denom = w.sum()
            if denom <= 0:
                fallback_idx.append(b)
                continue
            out[:, b] = (Xs @ w) / denom

        if fallback_idx:
            fb = np.array(fallback_idx, dtype=int)
            gauss = srf_convolve_spectra(X, owl, centers[fb], fwhms[fb])
            out[:, fb] = gauss
        return out

    except Exception as e:
        raise ValueError(f"Empirical SRF integration failed: {str(e)}")


def resample_spectra(X, original_wavelengths, new_wavelengths,
                     method="Linear Interpolation", fwhms=None, srf_table=None):
    """Dispatch spectral resampling to the chosen method.

    Args:
        X: (n_samples, n_bands) spectra.
        original_wavelengths: wavelengths (nm) of ``X``.
        new_wavelengths: target band centres (nm).
        method: one of :data:`RESAMPLE_METHODS` (legacy labels accepted via
            :func:`normalize_resample_method`).
        fwhms: required for the bandwidth-aware methods (Gaussian / Empirical SRF
            and Band Averaging) — per-band FWHM (nm) matching ``new_wavelengths``.
        srf_table: optional per-band response curves for "Empirical SRF".

    Returns:
        (n_samples, n_new) resampled spectra.
    """
    canonical = normalize_resample_method(method)

    if canonical in _INTERP_KINDS:
        return safe_interpolate_spectra(X, original_wavelengths, new_wavelengths,
                                        kind=_INTERP_KINDS[canonical])

    # Bandwidth-aware family — all require per-band FWHMs.
    if fwhms is None:
        raise ValueError(f"'{canonical}' resampling requires per-band FWHM values.")
    if canonical == "Gaussian SRF":
        return srf_convolve_spectra(X, original_wavelengths, new_wavelengths, fwhms)
    if canonical == "Empirical SRF":
        return empirical_srf_integrate(X, original_wavelengths, new_wavelengths,
                                       fwhms, srf_table=srf_table)
    if canonical == "Band Averaging":
        return band_average_spectra(X, original_wavelengths, new_wavelengths, fwhms)
    # Should be unreachable (normalize guarantees a known label).
    return safe_interpolate_spectra(X, original_wavelengths, new_wavelengths)


def apply_spectral_binning(bands, bin_size, fwhms=None):
    """Reduce a band grid by decimation binning (after Ori_Code_TAU
    ``get_resample_bands`` Step 3).

    One representative band is kept per consecutive group of ``bin_size`` bands —
    the band at the centre of each group (index ``i`` where
    ``i % bin_size == bin_size // 2``).  As a safety guard (mirroring the
    reference), binning is skipped unless the grid has more than ``bin_size * 3``
    bands, so it never collapses a grid to fewer than ~3 bands.

    Args:
        bands: 1-D array-like of wavelengths (nm).
        bin_size: number of bands per bin (int ≥ 1).  ``None``/<1 disables binning.
        fwhms: optional per-band FWHMs, subset by the same indices as ``bands``.

    Returns:
        (binned_bands, binned_fwhms) as ndarrays.  ``binned_fwhms`` is ``None``
        when ``fwhms`` is ``None``.  When binning is skipped the inputs are
        returned unchanged (as ndarrays).
    """
    bands = np.asarray(bands, dtype=float)
    fwhms_arr = None if fwhms is None else np.asarray(fwhms, dtype=float)

    if bin_size is None:
        return bands, fwhms_arr
    bs = int(bin_size)
    if bs < 1:
        raise ValueError("bin_size must be a positive integer.")
    if bs == 1 or len(bands) <= bs * 3:
        return bands, fwhms_arr

    idx = [i for i in range(len(bands)) if i % bs == bs // 2]
    binned = bands[idx]
    binned_fwhms = None if fwhms_arr is None else fwhms_arr[idx]
    return binned, binned_fwhms


# Standard atmospheric water-vapour / noisy regions offered as quick presets in
# the GUI's "exclude ranges" control (nm).
WATER_ABSORPTION_RANGES = [(1350.0, 1450.0), (1800.0, 1960.0)]
NOISY_EDGE_RANGES = [(350.0, 400.0), (2450.0, 2500.0)]


def parse_exclude_ranges(spec):
    """Normalise an exclude-ranges specification into a sorted list of tuples.

    Accepts either a string like ``"1350-1450, 1800-1960"`` (comma/semicolon
    separated ``lo-hi`` pairs) or an iterable of ``(lo, hi)`` pairs.  Malformed
    entries (non-numeric, ``lo >= hi``) are silently skipped so a stray keystroke
    never crashes filtering.  Returns ``[]`` when nothing valid is found.
    """
    if not spec:
        return []
    pairs = []
    if isinstance(spec, str):
        chunks = [c for c in spec.replace(";", ",").split(",") if c.strip()]
        for chunk in chunks:
            # Support "1350-1450" and "1350 to 1450".
            token = chunk.lower().replace("to", "-")
            parts = [p for p in token.split("-") if p.strip()]
            if len(parts) != 2:
                continue
            try:
                lo, hi = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if lo < hi:
                pairs.append((lo, hi))
    else:
        for item in spec:
            try:
                lo, hi = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if lo < hi:
                pairs.append((lo, hi))
    return sorted(pairs)


def _centers_keep_mask(centers, fwhms, exclude_ranges):
    """Boolean mask of target ``centers`` that do NOT overlap any excluded range.

    A band is dropped when its response window overlaps an excluded range.  When
    ``fwhms`` is given the window is ``[c - FWHM/2, c + FWHM/2]``; otherwise the
    centre itself must simply lie outside every excluded range.
    """
    centers = np.asarray(centers, dtype=float)
    keep = np.ones(len(centers), dtype=bool)
    if not exclude_ranges:
        return keep
    if fwhms is not None:
        half = np.asarray(fwhms, dtype=float) / 2.0
        lo_edge, hi_edge = centers - half, centers + half
    else:
        lo_edge, hi_edge = centers, centers
    for lo, hi in exclude_ranges:
        # Overlap test between [lo_edge, hi_edge] and [lo, hi].
        keep &= ~((hi_edge >= lo) & (lo_edge <= hi))
    return keep


def load_fwhm_csv(path):
    """Load a custom sensor's band centres + FWHMs from a CSV.

    The file must have two numeric columns; a header row is auto-detected.  The
    first column is the band centre (nm), the second the FWHM (nm).  Returns
    ``(centers, fwhms)`` as float ndarrays sorted by centre.

    Raises:
        ValueError: on an unreadable / malformed file or non-positive FWHM.
    """
    try:
        df = pd.read_csv(path)
        # If the first row parsed as data but isn't numeric, retry headerless.
        if df.shape[1] < 2 or not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            df = pd.read_csv(path, header=None)
        arr = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna().values
        if arr.shape[0] < 2:
            raise ValueError("Need at least 2 (centre, FWHM) rows.")
        centers, fwhms = arr[:, 0].astype(float), arr[:, 1].astype(float)
        if np.any(fwhms <= 0):
            raise ValueError("All FWHM values must be positive.")
        order = np.argsort(centers)
        return centers[order], fwhms[order]
    except Exception as e:
        raise ValueError(f"Could not read FWHM CSV '{path}': {e}")


def load_srf_csv(path):
    """Load per-band empirical spectral-response curves from a CSV.

    Expected long format with a header: columns ``center`` (or ``band``),
    ``wavelength`` and ``response``.  Rows are grouped by ``center`` into a
    ``{center_nm: (wavelength_array, response_array)}`` mapping used by
    :func:`empirical_srf_integrate`.

    Raises:
        ValueError: when the required columns are missing or no valid curve is
        found.
    """
    try:
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        center_col = cols.get("center") or cols.get("centre") or cols.get("band")
        wl_col = cols.get("wavelength") or cols.get("wl") or cols.get("nm")
        resp_col = cols.get("response") or cols.get("srf") or cols.get("weight")
        if not (center_col and wl_col and resp_col):
            raise ValueError(
                "SRF CSV must have 'center', 'wavelength' and 'response' columns.")
        table = {}
        for center, grp in df.groupby(center_col):
            wl = pd.to_numeric(grp[wl_col], errors="coerce").values.astype(float)
            resp = pd.to_numeric(grp[resp_col], errors="coerce").values.astype(float)
            good = np.isfinite(wl) & np.isfinite(resp)
            if good.sum() < 2:
                continue
            wl, resp = wl[good], resp[good]
            order = np.argsort(wl)
            table[float(center)] = (wl[order], resp[order])
        if not table:
            raise ValueError("No usable response curves found in the SRF CSV.")
        return table
    except Exception as e:
        raise ValueError(f"Could not read SRF CSV '{path}': {e}")


def filter_wavelengths(wavelengths, min_wave, max_wave, resampling, spacing,
                       sensor=None, resample_method="Linear Interpolation",
                       apply_binning=False, bin_size=30,
                       exclude_ranges=None, custom_fwhm=None):
    """
    Filter and optionally resample wavelengths.

    Returns a 3-tuple ``(filtered_wavelengths, new_wavelengths, new_fwhms)``.
    ``new_wavelengths`` / ``new_fwhms`` are ``None`` when no resampling applies.
    ``new_fwhms`` is only populated for the bandwidth-aware methods (Gaussian /
    Empirical SRF, Band Averaging).

    Resampling modes (only when ``resampling == "Yes"``):
      * ``sensor`` is None / "Custom" — uniform grid at ``spacing`` nm.
      * ``sensor`` is a satellite key  — resample onto that sensor's nominal
        band centres.  The Thermal-Infrared bands of Sentinel-2 / Landsat-8 are
        only used when the data itself reaches the thermal range.

    Method (``resample_method``, one of :data:`RESAMPLE_METHODS`):
      * Interpolation family — the target centre must lie within the data range
        so interpolation never extrapolates.
      * Bandwidth-aware family — a band is kept only when its full FWHM window
        lies inside the data range, so every response function is fully sampled
        (no biased partial-coverage bands).  On a Custom sensor these methods use
        ``custom_fwhm`` if given, else the uniform ``spacing`` as the FWHM.

    ``exclude_ranges`` (list of ``(lo, hi)`` nm, or a ``"lo-hi, lo-hi"`` string):
    wavelengths inside any excluded range are dropped from the source grid, and
    target bands whose window overlaps an excluded range are dropped too.

    Binning (``apply_binning`` / ``bin_size``): independent of ``resampling``.
    When resampling is on it thins the resampled grid (resample → bin); when
    resampling is off it decimates the filtered grid directly (returned as
    ``new_wavelengths`` so the caller selects those columns by interpolation).
    Keeps the centre band of every ``bin_size`` group via
    :func:`apply_spectral_binning`, and only takes effect when the grid has more
    than ``bin_size * 3`` bands, so sparse sensor grids are left untouched while
    dense/hyperspectral grids are thinned.
    """
    try:
        if not wavelengths:
            return None, None, None

        needs_fwhm = _needs_fwhm(resample_method)
        excl = parse_exclude_ranges(exclude_ranges)

        wavelengths_array = np.array([float(w) for w in wavelengths])
        data_lo_all = float(wavelengths_array.min())
        data_hi_all = float(wavelengths_array.max())
        min_wave = float(min_wave or data_lo_all)
        max_wave = float(max_wave or data_hi_all)
        # Clamp the requested range symmetrically to the actual data extent so a
        # too-low Min / too-high Max never produces out-of-range grid points.
        min_wave = max(min_wave, data_lo_all)
        max_wave = min(max_wave, data_hi_all)
        if min_wave >= max_wave:
            raise Exception(
                f"The selected wavelength range ({min_wave:.1f}-{max_wave:.1f} nm) "
                f"does not overlap the data ({data_lo_all:.1f}-{data_hi_all:.1f} nm).")

        mask = (wavelengths_array >= min_wave) & (wavelengths_array <= max_wave)
        # Drop excluded regions from the source grid too.
        for lo, hi in excl:
            mask &= ~((wavelengths_array >= lo) & (wavelengths_array <= hi))
        filtered_wavelengths = wavelengths_array[mask]

        if len(filtered_wavelengths) == 0:
            raise Exception(
                f"No wavelengths fall within the selected range "
                f"({min_wave:.1f}-{max_wave:.1f} nm) after applying exclusions."
            )

        if resampling == "Yes":
            data_min = float(filtered_wavelengths.min())
            data_max = float(filtered_wavelengths.max())

            # ---- Predefined sensor OR custom uploaded FWHM grid --------------
            centers = None
            if sensor and sensor != "Custom":
                # Only pull in thermal bands when the data actually extends into
                # the thermal range; otherwise VIS-NIR-SWIR data would trigger
                # extrapolation against far-away thermal centres.
                include_thermal = data_max >= _THERMAL_THRESHOLD_NM
                centers, fwhms = get_sensor_bands(sensor, include_thermal=include_thermal)
            elif needs_fwhm and custom_fwhm is not None:
                # Custom sensor with an uploaded (centres, FWHMs) table.
                centers = np.asarray(custom_fwhm[0], dtype=float)
                fwhms = np.asarray(custom_fwhm[1], dtype=float)

            if centers is not None:
                if needs_fwhm:
                    # Keep a band only when its full FWHM window is covered by the
                    # data — guarantees a fully-sampled response function.
                    keep = ((centers - fwhms / 2.0 >= data_min) &
                            (centers + fwhms / 2.0 <= data_max))
                else:
                    # Interpolation: centre must lie within the data range.
                    keep = (centers >= data_min) & (centers <= data_max)
                keep &= _centers_keep_mask(centers, fwhms if needs_fwhm else None, excl)

                new_wavelengths = centers[keep]
                new_fwhms = fwhms[keep] if needs_fwhm else None
                if len(new_wavelengths) < 2:
                    window = ("FWHM windows" if needs_fwhm else "band centres")
                    label = sensor if (sensor and sensor != "Custom") else "the custom FWHM table"
                    raise Exception(
                        f"{label} has fewer than 2 {window} within the "
                        f"data range ({data_min:.0f}-{data_max:.0f} nm). "
                        f"Cannot resample — widen the Min/Max wavelength range, "
                        f"reduce exclusions, switch to an interpolation method, "
                        f"or choose a sensor that overlaps the data."
                    )
                if apply_binning:
                    new_wavelengths, new_fwhms = apply_spectral_binning(
                        new_wavelengths, bin_size, new_fwhms)
                return filtered_wavelengths, new_wavelengths, new_fwhms

            # ---- Uniform-spacing "Custom" grid ------------------------------
            spacing = float(spacing or 10)
            if spacing <= 0:
                raise Exception("Spacing must be a positive number.")
            new_wavelengths = np.arange(min_wave, max_wave + spacing, spacing)
            # Clip against the ACTUAL filtered data extent on BOTH sides so no
            # interpolation point falls outside the original data range.
            new_wavelengths = new_wavelengths[
                (new_wavelengths >= data_min) & (new_wavelengths <= data_max)]
            if len(new_wavelengths) == 0 or new_wavelengths[-1] < data_max:
                new_wavelengths = np.append(new_wavelengths, data_max)
            # For a bandwidth-aware method on the uniform grid there is no
            # instrument FWHM, so use the grid spacing as the response width.
            new_fwhms = (np.full(len(new_wavelengths), spacing, dtype=float)
                         if needs_fwhm else None)
            if needs_fwhm:
                # Drop edge bands whose FWHM window spills past the data range so
                # every response function is fully sampled (no partial coverage).
                keep = ((new_wavelengths - new_fwhms / 2.0 >= data_min) &
                        (new_wavelengths + new_fwhms / 2.0 <= data_max))
                new_wavelengths, new_fwhms = new_wavelengths[keep], new_fwhms[keep]
            # Drop grid points overlapping an excluded range.
            if excl:
                keep = _centers_keep_mask(new_wavelengths, new_fwhms, excl)
                new_wavelengths = new_wavelengths[keep]
                new_fwhms = new_fwhms[keep] if new_fwhms is not None else None
            if len(new_wavelengths) < 2:
                raise Exception(
                    f"The {spacing:.0f} nm uniform grid has fewer than 2 bands "
                    f"within the data range after coverage/exclusion filtering. "
                    f"Reduce the spacing or the excluded ranges.")
            if apply_binning:
                new_wavelengths, new_fwhms = apply_spectral_binning(
                    new_wavelengths, bin_size, new_fwhms)
            return filtered_wavelengths, new_wavelengths, new_fwhms

        # ---- Binning without resampling --------------------------------------
        # Binning is independent of the resample toggle: when resampling is off
        # but binning is on, decimate the filtered grid directly.  The binned
        # wavelengths are an exact subset of the originals, so the caller's
        # interpolation onto them is a loss-less column selection (no FWHM →
        # new_fwhms stays None; the caller uses Interpolation in this case).
        if apply_binning:
            binned, _ = apply_spectral_binning(filtered_wavelengths, bin_size)
            if len(binned) != len(filtered_wavelengths):
                return filtered_wavelengths, binned, None

        return filtered_wavelengths, None, None

    except Exception as e:
        raise Exception(f"Wavelength filtering failed: {str(e)}")


def calculate_confidence_interval(y_true, y_pred, confidence=0.95):
    """
    Calculate confidence interval for regression metrics using bootstrap
    
    Args:
        y_true: True values
        y_pred: Predicted values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Dictionary with CI for R², RMSE, and MAE
    """
    try:
        n = len(y_true)
        n_bootstraps = 1000

        rng = np.random.RandomState(42)
        # Generate all bootstrap indices at once — avoids 1000-iteration Python loop
        all_indices = rng.randint(0, n, (n_bootstraps, n))  # shape (1000, n)

        y_true_boots = y_true[all_indices]  # shape (1000, n)
        y_pred_boots = y_pred[all_indices]  # shape (1000, n)

        # Vectorized RMSE and MAE
        residuals = y_true_boots - y_pred_boots
        rmse_scores = np.sqrt(np.mean(residuals ** 2, axis=1))
        mae_scores = np.mean(np.abs(residuals), axis=1)

        # Vectorized R²: 1 - SS_res / SS_tot
        y_true_means = y_true_boots.mean(axis=1, keepdims=True)
        ss_tot = np.sum((y_true_boots - y_true_means) ** 2, axis=1)
        ss_res = np.sum(residuals ** 2, axis=1)
        r2_scores = 1.0 - ss_res / (ss_tot + 1e-12)

        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        r2_ci = (np.percentile(r2_scores, lower_percentile), 
                 np.percentile(r2_scores, upper_percentile))
        rmse_ci = (np.percentile(rmse_scores, lower_percentile), 
                   np.percentile(rmse_scores, upper_percentile))
        mae_ci = (np.percentile(mae_scores, lower_percentile), 
                  np.percentile(mae_scores, upper_percentile))
        
        return {
            'r2_ci': r2_ci,
            'rmse_ci': rmse_ci,
            'mae_ci': mae_ci
        }
        
    except Exception:
        # If bootstrap fails, return None
        return None


# ---------------------------------------------------------------------------
# Data-integrity / randomisation utilities
# ---------------------------------------------------------------------------

def randomize_label_test(X, y, n_permutations=100, n_components=5, cv=5):
    """
    Permutation / label-randomisation test to expose human-labelling errors
    and data leakage.

    Trains a PLS-R model with *shuffled* chemical labels ``n_permutations``
    times and returns the distribution of cross-validated R² values.  A real
    model's R² should sit far above this null distribution.

    Parameters
    ----------
    X : array (n_samples, n_features)
    y : array (n_samples,)
    n_permutations : int
    n_components : int   – latent variables for PLS-R
    cv : int             – number of CV folds (the caller should pass the SAME
                           value used for the observed R² so the null
                           distribution is directly comparable)

    Returns
    -------
    perm_r2 : list of float   – permuted CV-R² scores
    """
    n_comp = min(n_components, X.shape[0] - 2, X.shape[1])
    model = PLSRegression(n_components=max(1, n_comp))
    rng = np.random.RandomState(42)
    cv_k = max(2, min(cv, len(y)))
    perm_r2 = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        scores = cross_val_score(model, X, y_perm,
                                 cv=cv_k,
                                 scoring='r2')
        perm_r2.append(float(np.mean(scores)))
    return perm_r2


def _random_derangement(m, rng):
    """Return a length-*m* permutation with no fixed point (a derangement).

    Guarantees that every selected position is reassigned a *different*
    position's value.  Tries random permutations first (fast for the common
    case) and falls back to a single cyclic rotation, which is always a valid
    derangement for ``m >= 2``.
    """
    if m < 2:
        return np.arange(m)
    for _ in range(20):
        perm = rng.permutation(m)
        if not np.any(perm == np.arange(m)):
            return perm
    # Deterministic fallback: rotate by one (no element maps to itself).
    return np.roll(np.arange(m), 1)


def mix_spectra_integrity_check(X, y, mix_fraction=0.2, random_seed=42):
    """
    Mix a fraction of spectra-chemical pairs to simulate data integrity issues
    (e.g., wrong sample labelling or cross-contamination).

    The spectra stay in place; only the chemical labels of the selected samples
    are reassigned among themselves using a **derangement** (no sample keeps its
    own label), so the perturbation is genuine.  This models a data-entry error
    where the right spectrum is paired with the wrong chemistry.

    Returns
    -------
    X_mixed     : array  – spectral matrix (unchanged copy of X)
    y_mixed     : array  – chemistry vector with the selected labels reassigned
    changed_idx : ndarray(int) – indices whose label VALUE actually changed
                  (``len(changed_idx)`` is the true number of affected samples;
                  duplicate y-values can leave a reassigned position unchanged,
                  and this is reported honestly rather than overstated)
    """
    rng = np.random.RandomState(random_seed)
    y = np.asarray(y)
    n = len(y)
    X_mixed = X.copy()
    y_mixed = y.copy()
    if n < 2:
        return X_mixed, y_mixed, np.array([], dtype=int)

    n_mix = max(2, int(round(mix_fraction * n)))
    n_mix = min(n_mix, n)
    idx = rng.choice(n, n_mix, replace=False)

    # Reassign the selected labels among themselves with no fixed point.
    perm = _random_derangement(len(idx), rng)
    y_mixed[idx] = y[idx[perm]]

    # Report only the samples whose label value truly differs.
    changed_idx = idx[y_mixed[idx] != y[idx]]
    return X_mixed, y_mixed, np.asarray(changed_idx, dtype=int)


# ---------------------------------------------------------------------------
# Spectral transfer function (cross-sensor harmonisation)
# ---------------------------------------------------------------------------

def spectral_transfer_function(X_source, X_target, n_components=5):
    """
    Estimate a PLS-based spectral transfer function that maps spectra from
    *source* instrument/library to *target* instrument/satellite.

    Both X_source and X_target must have the same number of samples (paired
    measurements).

    Returns
    -------
    pls      : fitted PLSRegression  (source → target)
    scaler_s : StandardScaler for source
    scaler_t : StandardScaler for target
    r2_per_band : array  – per-band R² of the mapping
    """
    scaler_s = StandardScaler()
    scaler_t = StandardScaler()
    X_s = scaler_s.fit_transform(X_source)
    X_t = scaler_t.fit_transform(X_target)

    n_comp = min(n_components, X_s.shape[0] - 2, X_s.shape[1], X_t.shape[1])
    pls = PLSRegression(n_components=max(1, n_comp))
    pls.fit(X_s, X_t)

    X_t_pred = pls.predict(X_s)
    # Vectorized R² across all bands — eliminates per-band Python loop
    ss_res = np.sum((X_t - X_t_pred) ** 2, axis=0)
    ss_tot = np.sum((X_t - X_t.mean(axis=0)) ** 2, axis=0)
    r2_per_band = 1.0 - ss_res / (ss_tot + 1e-12)
    return pls, scaler_s, scaler_t, r2_per_band


def apply_transfer_function(X_new, pls, scaler_s, scaler_t):
    """
    Apply a fitted spectral transfer function to new (unseen) source spectra.

    Returns harmonised spectra in the target space (original scale).
    """
    X_s_scaled = scaler_s.transform(X_new)
    X_t_scaled = pls.predict(X_s_scaled)
    return scaler_t.inverse_transform(X_t_scaled)


# ---------------------------------------------------------------------------
# Per-row / per-pixel prediction validation helpers
# ---------------------------------------------------------------------------

def compute_prediction_statistics_per_row(predictions_df):
    """
    Given a DataFrame whose columns are prediction arrays from different
    models (or bootstrap replicates), compute per-row mean, std, and CV (%).

    Returns a DataFrame with columns: mean, std, cv_pct.
    """
    means = predictions_df.mean(axis=1)
    stds  = predictions_df.std(axis=1)
    cv    = (stds / (means.abs() + 1e-12)) * 100.0
    return pd.DataFrame({'mean': means, 'std': stds, 'cv_pct': cv})


def compute_image_prediction_statistics(predictions_2d):
    """
    Compute per-pixel statistics for a 2-D prediction array
    (rows × cols after spatial reshaping).

    Returns a dict with mean, std, cv_pct (all 2-D arrays of the same shape).
    """
    # predictions_2d: shape (H, W)
    mean_val = np.nanmean(predictions_2d)
    std_val  = np.nanstd(predictions_2d)
    cv_val   = (std_val / (abs(mean_val) + 1e-12)) * 100.0
    return {
        'mean': float(mean_val),
        'std':  float(std_val),
        'cv_pct': float(cv_val),
        'min':  float(np.nanmin(predictions_2d)),
        'max':  float(np.nanmax(predictions_2d)),
        'median': float(np.nanmedian(predictions_2d)),
    }