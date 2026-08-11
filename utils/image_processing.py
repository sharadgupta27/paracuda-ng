"""
Image processing utilities for spectral analysis.

Reads/writes any GDAL-supported geospatial raster (GeoTIFF/TIFF, ENVI + .hdr,
ERDAS Imagine .img, ESRI/generic .bil/.bip/.bsq/.dat with header, NITF, PCIDSK,
…) - everything except ordinary JPEG/PNG snapshots.
Spectral metadata (per-band wavelength / FWHM) is recovered from the header when
present, and predictions are written back in the *same* driver with a correct,
spectral-free header derived from the input.

@author: Sharad Kumar Gupta
"""
import contextlib
import os
import re

import numpy as np
from preprocessing.data_processing import (preprocess_spectra, resample_spectra,
                                           estimate_fwhms_from_grid)

# Resampling methods that need per-band FWHMs (bandwidth-aware family).
_FWHM_RESAMPLE = {"Gaussian SRF", "Empirical SRF", "Band Averaging"}

# rasterio (and its heavy GDAL DLLs) imported lazily so importing this module at
# GUI startup is cheap; it loads on first image read. See utils/lazy_imports.py.
from utils.lazy_imports import LazyModule

rasterio = LazyModule('rasterio')

# ── Format handling ─────────────────────────────────────────────────────────
# Extension → GDAL driver for *writing* the prediction back in the input format.
# (Reading auto-detects the driver, so this table only guides output.)
EXT_TO_DRIVER = {
    '.tif': 'GTiff', '.tiff': 'GTiff',
    '.img': 'HFA',
    '.dat': 'ENVI', '.bil': 'ENVI', '.bip': 'ENVI', '.bsq': 'ENVI',
    '.bin': 'ENVI', '.raw': 'ENVI', '.hdr': 'ENVI',
    '.ntf': 'NITF', '.nitf': 'NITF',
    '.pix': 'PCIDSK',
}

# Raster containers a user may open (GDAL picks the actual driver on open).
# Ordinary photos (jpg/png/etc.) are intentionally excluded.  JP2 is omitted:
# this GDAL build ships no JP2 driver.
OPEN_EXTENSIONS = ['.tif', '.tiff', '.img', '.dat', '.bil', '.bip', '.bsq',
                   '.bin', '.raw', '.hdr', '.ntf', '.nitf', '.pix', '.vrt']
_REJECT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}


def open_image_filetypes():
    """tk ``filetypes`` list for the load-image dialog (all raster formats)."""
    patterns = " ".join(f"*{e}" for e in OPEN_EXTENSIONS)
    return [
        ("Geospatial rasters", patterns),
        ("GeoTIFF / TIFF", "*.tif *.tiff"),
        ("ENVI / band-interleaved", "*.dat *.bil *.bip *.bsq *.bin *.raw *.hdr"),
        ("ERDAS Imagine", "*.img"),
        ("NITF", "*.ntf *.nitf"),
        ("All files", "*.*"),
    ]


def driver_for_path(path, fallback='GTiff'):
    """Return the GDAL write-driver implied by ``path``'s extension."""
    return EXT_TO_DRIVER.get(os.path.splitext(path)[1].lower(), fallback)


# ── ENVI header parsing (wavelength / FWHM recovery) ────────────────────────
def _num_list(raw):
    """Parse an ENVI ``{a, b, c}`` (or bare comma list) into floats."""
    if raw is None:
        return None
    s = str(raw).strip().strip('{}')
    out = []
    for tok in s.replace('\n', ' ').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return None
    return out or None


def parse_envi_header(hdr_path):
    """Parse an ENVI ``.hdr`` text file into a ``{lower_key: value}`` dict.

    Handles multi-line ``key = { ... }`` blocks (wavelength, fwhm, band names,
    map info, …) as in a standard ENVI header."""
    with open(hdr_path, 'r', errors='ignore') as f:
        text = f.read()
    hdr = {}
    # key = value, where value is either a { ... } block (possibly multi-line)
    # or the remainder of the line.
    for m in re.finditer(r'([A-Za-z][\w ]*?)\s*=\s*(\{[^}]*\}|[^\n]*)', text):
        hdr[m.group(1).strip().lower()] = m.group(2).strip()
    return hdr


def _find_sidecar_hdr(path):
    """Locate the ENVI header that accompanies a data file, if any."""
    root, _ = os.path.splitext(path)
    for cand in (root + '.hdr', path + '.hdr'):
        if os.path.exists(cand):
            return cand
    return None


def _spectral_from_dataset(src):
    """Recover (wavelengths, fwhms, units) from GDAL/rasterio tags."""
    for ns in ('ENVI', None):
        try:
            tags = src.tags(ns=ns) if ns else src.tags()
        except Exception:
            tags = None
        if not tags:
            continue
        wl = tags.get('wavelength') or tags.get('Wavelength')
        if wl:
            units = tags.get('wavelength_units') or tags.get('wavelength units')
            return _num_list(wl), _num_list(tags.get('fwhm')), units
    # Fall back to per-band 'wavelength' items.
    wls = []
    try:
        for b in range(1, src.count + 1):
            w = src.tags(b).get('wavelength')
            if w is None:
                break
            wls.append(float(str(w).split()[0]))
    except Exception:
        wls = []
    if wls and len(wls) == src.count:
        return wls, None, None
    return None, None, None


def _gain_offset_from_dataset(src):
    """Recover a per-band (gain, offset) to convert stored DN to physical units.

    Prefers GDAL's native per-band Scale/Offset (standard TIFF tags etc.), which
    rasterio applies nowhere automatically - reading always returns the raw
    stored numbers.  Falls back to ENVI's ``data gain values`` / ``data offset
    values`` header fields, which are NOT part of GDAL's ENVI metadata mapping
    (unlike ``wavelength``/``fwhm``) and so must be parsed by hand from the tag
    text.  Returns ``(None, None)`` when the dataset carries no scaling (gain 1,
    offset 0 everywhere) - the overwhelmingly common case for GeoTIFF.
    """
    with contextlib.suppress(Exception):
        scales, offsets = list(src.scales), list(src.offsets)
        if any(s != 1.0 for s in scales) or any(o != 0.0 for o in offsets):
            return scales, offsets
    for ns in ('ENVI', None):
        try:
            tags = src.tags(ns=ns) if ns else src.tags()
        except Exception:
            tags = None
        if not tags:
            continue
        gain = _num_list(tags.get('data gain values') or tags.get('data_gain_values'))
        if gain:
            offset = _num_list(tags.get('data offset values') or tags.get('data_offset_values'))
            return gain, offset
    return None, None


def read_geospatial_image(path):
    """Open any GDAL-supported raster and return PHYSICAL (not raw-DN) values.

    Many multispectral/hyperspectral products (e.g. ENVI L2A reflectance, scaled
    GeoTIFF) store integer digital numbers with a header-defined gain/offset
    (e.g. ``data gain values = 0.0001`` so a raw DN of 3000 means reflectance
    0.3). Feeding raw DN straight into a model trained on 0-1 reflectance
    produces wildly out-of-range scaled features and garbage/negative
    predictions that then look like an empty (all-background) output. This
    reader applies that gain/offset - from GDAL's native per-band Scale/Offset
    or ENVI's custom header fields - so callers always get physical units. The
    header's no-data sentinel is also converted to NaN here (before scaling),
    so callers never need to special-case the raw sentinel value.

    Returns ``(data, profile, wavelengths, fwhms, info)`` where ``data`` is a
    ``(bands, rows, cols)`` float array in physical units with NaN at no-data
    pixels, ``profile`` the rasterio profile (for writing the prediction
    back), and ``wavelengths``/``fwhms`` per-band lists recovered from the
    header (or ``None`` when the format carries no spectral metadata)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _REJECT_EXTENSIONS:
        raise ValueError(
            f"'{ext}' photos are not geospatial rasters. Load a multispectral "
            f"or hyperspectral cube (GeoTIFF, ENVI/.hdr, .img, .bil/.bip/.bsq, "
            f".dat, …).")
    with rasterio.open(path) as src:
        raw = src.read()
        profile = dict(src.profile)
        driver = src.driver
        wl, fwhm, units = _spectral_from_dataset(src)
        gain, offset = _gain_offset_from_dataset(src)
        nodata_raw = profile.get('nodata')
        # ENVI band interleave (bil/bip/bsq) - rasterio's profile normalises this
        # away, so read it from the ENVI tags to preserve it on write.
        try:
            interleave = src.tags(ns='ENVI').get('interleave')
        except Exception:
            interleave = None
    if not interleave:
        hdr = _find_sidecar_hdr(path)
        if hdr:
            interleave = parse_envi_header(hdr).get('interleave')
    if wl is None:
        hdr = _find_sidecar_hdr(path)
        if hdr:
            h = parse_envi_header(hdr)
            wl = _num_list(h.get('wavelength'))
            fwhm = _num_list(h.get('fwhm'))
            units = h.get('wavelength units')
    if gain is None:
        hdr = _find_sidecar_hdr(path)
        if hdr:
            h = parse_envi_header(hdr)
            gain = _num_list(h.get('data gain values'))
            offset = _num_list(h.get('data offset values'))

    # Sanity: only trust spectral metadata whose length matches the band count
    # (guards against a stray same-basename .hdr belonging to another dataset).
    n_bands = raw.shape[0]
    if wl is not None and len(wl) != n_bands:
        wl, fwhm, units = None, None, None
    if fwhm is not None and len(fwhm) != n_bands:
        fwhm = None

    # ── No-data → NaN (on the RAW sentinel, before any scaling) ───────────
    # float32 keeps a full image cube at half the footprint of float64
    # (a 1195×1150×224 EnMAP cube is 1.2 GiB vs 2.4 GiB) - ample precision for
    # reflectance; the prediction pipeline upcasts per-chunk where needed.
    data = raw.astype('float32', copy=False)
    del raw
    if nodata_raw is not None:
        data[data == np.float32(nodata_raw)] = np.nan

    # ── Apply per-band gain/offset to recover physical units ──────────────
    gain_applied = False
    if gain:
        g = np.asarray(gain, dtype='float32')
        if len(g) == 1:
            g = np.full(n_bands, g[0], dtype='float32')
        o = (np.zeros(n_bands, dtype='float32') if not offset
             else np.asarray(offset, dtype='float32'))
        if len(o) == 1:
            o = np.full(n_bands, o[0], dtype='float32')
        if len(g) == n_bands and len(o) == n_bands and (
                np.any(g != 1.0) or np.any(o != 0.0)):
            # In-place per-band scaling avoids a second full-size copy.
            data *= g[:, None, None]
            data += o[:, None, None]
            gain_applied = True

    info = {'driver': driver, 'wavelength_units': units, 'bands': n_bands,
            'interleave': (str(interleave).lower() if interleave else None),
            'nodata': nodata_raw, 'gain_applied': gain_applied,
            'gain': (list(g) if gain_applied else None)}
    return data, profile, wl, fwhm, info

def process_image_for_prediction(image_data, wavelengths, preprocessing_method,
                                scaler_X, filtered_wavelengths, new_wavelengths=None,
                                preprocess_kwargs=None, resample_method="Linear Interpolation",
                                new_fwhms=None, srf_table=None, nodata=None,
                                mask_background=True, saturation=1.0e6,
                                return_cube=True, chunk_pixels=100_000,
                                min_band_valid_frac=0.5, min_pixel_valid_frac=0.5,
                                resample=True, predict_fn=None):
    """Process an image cube into the model's scaled input, aligned to the model's
    band grid, and flag background / no-data pixels so they are not predicted.

    The source bands within the training range are always resampled onto the
    model's input grid (``new_wavelengths`` when the model was resampled, else
    ``filtered_wavelengths`` - i.e. the training-Excel wavelengths), so an image
    on any band grid can be predicted.

    Memory: a full scene (e.g. EnMAP 1195×1150×224) is far too large to push
    through resample→preprocess→scale as one float64 matrix (each full-size
    copy is ~2 GiB and the old single-shot pipeline needed several at once).
    The spectra are therefore processed in blocks of ``chunk_pixels`` pixels -
    only the small per-chunk temporaries are float64; the persistent outputs
    are float32.

    Args:
        image_data: (bands, rows, cols) cube.
        wavelengths: per-band wavelengths of ``image_data``.
        preprocessing_method / preprocess_kwargs: training preprocessing.
        scaler_X: StandardScaler fitted on training data.
        filtered_wavelengths: training wavelengths after range/exclude filtering.
        new_wavelengths / new_fwhms: model's resampled grid (or None).
        resample_method / srf_table: resampling configuration.
        nodata: input no-data value (from the header) to treat as background.
        mask_background: also flag all-zero/near-zero and saturated pixels.
        saturation: pixels with |value| ≥ this on any band are background.
        return_cube: build the (pixels, bands) validation cube.  Costs one extra
            full-size float32 array - pass False unless the user asked to
            export the resampled cube.
        chunk_pixels: pixels per processing block.
        predict_fn: optional callable mapping a scaled ``(n_chunk, features)``
            block to a ``(n_chunk,)`` prediction vector (in the final output
            units - the caller wraps PCA/​model.predict/​inverse-transform).
            When given, predictions are computed *inside* the chunk loop and only
            a single ``(n_px,)`` float32 vector is kept, instead of the full
            ``(n_px, features)`` scaled matrix.  This is essential for large
            scenes: a 13.4 M-pixel × 1787-band scaled matrix alone is ~89 GiB,
            whereas the prediction vector is ~54 MB.  The first element of the
            returned tuple is then the prediction vector rather than the scaled
            matrix.

    Returns:
        (scaled_data_or_predictions, original_shape, valid_mask, resampled_cube,
        target_wl).  The first element is the ``(n_px,)`` prediction vector when
        ``predict_fn`` is supplied, otherwise the ``(n_px, features)`` scaled
        matrix.  ``valid_mask`` is a per-pixel bool (flattened row-major) marking
        real pixels, and ``resampled_cube`` is the (pixels, target_bands)
        reflectance on the model grid (NaN at background pixels) for validation
        - or ``None`` when ``return_cube`` is False.
    """
    try:
        if preprocess_kwargs is None:
            preprocess_kwargs = {}

        original_shape = image_data.shape

        # ── Resolve the model's input grid (what the scaler expects) ─────────
        if new_wavelengths is not None and len(new_wavelengths) > 0:
            target_wl = np.asarray(new_wavelengths, dtype=float)
            target_fwhms = new_fwhms
        else:
            target_wl = np.asarray(filtered_wavelengths, dtype=float)
            target_fwhms = None
        if resample_method in _FWHM_RESAMPLE and target_fwhms is None:
            target_fwhms = estimate_fwhms_from_grid(target_wl)

        wl_arr = np.array([float(w) for w in wavelengths])
        tgt_lo, tgt_hi = float(target_wl.min()), float(target_wl.max())
        flat = image_data.reshape(original_shape[0], -1)     # (B, n_px) view
        B, n_px = flat.shape

        # ── Identify background pixels and BAD BANDS (band-wise, memory-light) ─
        # Real image scenes carry (a) no-data/border pixels that are fill
        # in *every* band, and (b) "bad bands" - whole bands that are fill/NaN in
        # *every* real pixel (e.g. EnMAP's water-vapour gaps).  Requiring every
        # band finite would wrongly mark the entire scene background whenever one
        # bad band lies in range, so we treat these separately: a pixel is real
        # if it has any finite band; a band is usable only if it is finite for a
        # sufficient fraction of the *real* pixels (so a bad band is dropped, not
        # the whole scene).  If ``nodata`` slips through un-NaN'd, honour it here.
        finite_count = np.zeros(n_px, dtype=np.int32)
        nd = None if nodata is None else np.float32(nodata)
        for b in range(B):
            row = flat[b]
            ok = np.isfinite(row)
            if nd is not None:
                ok &= (row != nd)
            finite_count += ok
        nonbg = finite_count > 0
        n_real = int(nonbg.sum())
        if n_real == 0:
            raise Exception("The image contains no valid (non-background) pixels.")

        band_good_frac = np.empty(B)
        for b in range(B):
            row = flat[b]
            ok = np.isfinite(row)
            if nd is not None:
                ok &= (row != nd)
            band_good_frac[b] = ok[nonbg].mean()
        good_band = band_good_frac >= min_band_valid_frac

        # ── Select GOOD source bands that BRACKET the target range ───────────
        # Bracketing among good bands ensures the target endpoints stay covered
        # even after dropping interior bad bands, so the scaler always gets the
        # full feature count.
        good_idx = np.where(good_band)[0]
        if good_idx.size < 2:
            raise Exception("Too few usable bands in the image's spectral range "
                            "(most bands are no-data/bad within the model range).")
        gwl = wl_arr[good_idx]
        sel = list(good_idx[(gwl >= tgt_lo) & (gwl <= tgt_hi)])
        below = good_idx[gwl < tgt_lo]
        above = good_idx[gwl > tgt_hi]
        if below.size:
            sel.append(int(below[wl_arr[below].argmax()]))   # nearest good below
        if above.size:
            sel.append(int(above[wl_arr[above].argmin()]))   # nearest good above
        wavelength_indices = sorted(set(int(i) for i in sel))
        if len(wavelength_indices) < 2:
            raise Exception("The image has too few usable bands within the "
                            "model's training wavelength range.")
        source_wavelengths = [float(wavelengths[i]) for i in wavelength_indices]

        # (pixels, kept-good-bands) as the single full-size float32 working copy.
        src = np.ascontiguousarray(flat[wavelength_indices].T, dtype='float32')
        if nd is not None:
            src[src == nd] = np.nan

        # ── Per-pixel validity on the kept bands ─────────────────────────────
        finite = np.isfinite(src)
        finite_frac = finite.mean(axis=1)
        valid = finite_frac >= min_pixel_valid_frac
        if mask_background:
            # positive-signal & saturation tested on finite values only
            tmp = np.where(finite, src, 0.0)
            valid &= np.any(tmp > 0, axis=1)                    # drop all-zero/negative
            valid &= np.all(np.abs(tmp) < saturation, axis=1)   # drop saturated/extreme

        # ── Fill residual sparse NaNs in kept bands so resampling never breaks ─
        # After dropping whole bad bands, any leftover NaN is a rare per-pixel
        # dropout; fill it with that pixel's own mean reflectance (cheap,
        # vectorised, and harmless since invalid pixels are zeroed next).
        nan_mask = ~finite
        if nan_mask.any():
            # Per-pixel mean of the finite bands (0 where a pixel has none),
            # computed without nanmean to avoid its empty-slice warning.
            finite_n = finite.sum(axis=1)
            row_mean = (np.where(finite, src, 0.0).sum(axis=1)
                        / np.maximum(finite_n, 1))
            src[nan_mask] = np.repeat(row_mean, nan_mask.sum(axis=1))
        src[~valid] = 0.0

        # ── Resample→preprocess→scale in pixel chunks ─────────────────────────
        # ``resample=False`` (user asserts the image is already on the model's
        # grid) skips the resample step; the selected source bands are fed
        # straight through and the scaler validates the feature count.
        #
        # Even when ``resample=True``, if the selected source bands already sit on
        # the model's target grid the resample step is a no-op, so we skip it: an
        # EnMAP image predicted with an EnMAP-resampled model must NOT be
        # resampled onto (essentially) itself.  The tolerance is a small fraction
        # of the target band spacing (min 0.5 nm) rather than 1e-6, because real
        # header wavelengths differ from a sensor preset by sub-nanometre rounding.
        same_grid = False
        if len(source_wavelengths) == len(target_wl):
            sw = np.sort(np.asarray(source_wavelengths, dtype=float))
            tw = np.sort(np.asarray(target_wl, dtype=float))
            grid_tol = (max(0.5, 0.1 * float(np.min(np.diff(tw))))
                        if tw.size >= 2 else 0.5)
            same_grid = bool(np.all(np.abs(sw - tw) <= grid_tol))
        do_resample = bool(resample) and not same_grid
        out_wl = np.asarray(target_wl if do_resample else source_wavelengths,
                            dtype=float)
        n_out = len(out_wl)
        cube = np.empty((n_px, n_out), dtype='float32') if return_cube else None
        # With ``predict_fn`` we keep only the (n_px,) prediction vector; without
        # it we accumulate the full (n_px, features) scaled matrix (its width is
        # fixed by the first chunk).
        scaled_out = None
        predictions = None

        step = max(1, int(chunk_pixels))
        for s in range(0, n_px, step):
            e = min(s + step, n_px)
            block = src[s:e].astype('float64')
            if do_resample:
                block = resample_spectra(block, source_wavelengths, target_wl,
                                         method=resample_method, fwhms=target_fwhms,
                                         srf_table=srf_table)
            if cube is not None:
                cube[s:e] = block
            if preprocessing_method == "Spectral Outlier Removal":
                block, _ = preprocess_spectra(block, preprocessing_method, **preprocess_kwargs)
            else:
                block = preprocess_spectra(block, preprocessing_method, **preprocess_kwargs)
            block = scaler_X.transform(block)
            if predict_fn is not None:
                pr = np.asarray(predict_fn(block), dtype='float32').ravel()
                if predictions is None:
                    predictions = np.empty(n_px, dtype='float32')
                predictions[s:e] = pr
            else:
                if scaled_out is None:
                    scaled_out = np.empty((n_px, block.shape[1]), dtype='float32')
                scaled_out[s:e] = block

        # Reflectance on the model grid for validation export (NaN at background).
        if cube is not None:
            cube[~valid] = np.nan

        result = predictions if predict_fn is not None else scaled_out
        return result, original_shape, valid, cube, out_wl

    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

def _spatial_profile(image_meta, driver, count, rows, cols, dtype, nodata):
    """Minimal, format-agnostic write profile that keeps georeferencing and drops
    all band/spectral-specific baggage from the source header."""
    profile = {
        'driver': driver, 'height': int(rows), 'width': int(cols),
        'count': int(count), 'dtype': dtype, 'nodata': nodata,
    }
    for key in ('crs', 'transform'):
        if image_meta.get(key) is not None:
            profile[key] = image_meta[key]
    return profile


def save_prediction_image(predictions, original_shape, image_meta, file_path,
                          driver=None, band_name=None, nodata=-9999.0,
                          valid_mask=None):
    """Write the single-band prediction back in a geospatial format.

    The output uses the driver implied by ``file_path`` (or ``driver`` if given),
    defaulting to the input driver, and a *clean* header derived from the input:
    georeferencing (crs / transform) is preserved, while multi-band spectral
    metadata (wavelength / fwhm / band names) is dropped so the header correctly
    describes the derived one-band product.  GDAL emits the driver's native
    header (e.g. an ENVI ``.hdr`` for ENVI output) automatically.

    ``valid_mask`` (flattened per-pixel bool) marks real pixels; background /
    no-data pixels are written as ``nodata`` and never carry a prediction.

    Returns the prediction as a float32 array with NaN where masked.
    """
    try:
        rows, cols = original_shape[1], original_shape[2]
        preds = np.asarray(predictions, dtype='float32').ravel()

        # Background / no-data pixels get no prediction.
        if valid_mask is not None:
            preds = np.where(np.asarray(valid_mask).ravel(), preds, np.nan)

        prediction_image = preds.reshape(rows, cols)
        # Drop non-finite / non-positive predictions (extreme = background).
        prediction_image = np.where(
            np.isfinite(prediction_image) & (prediction_image > 0),
            prediction_image, np.nan)
        raster = np.where(np.isnan(prediction_image), nodata,
                          prediction_image).astype('float32')

        out_driver = (driver or driver_for_path(file_path,
                      fallback=image_meta.get('driver', 'GTiff')))
        profile = _spatial_profile(image_meta, out_driver, 1, rows, cols,
                                   'float32', nodata)

        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(raster, 1)
            if band_name:
                with contextlib.suppress(Exception):
                    dst.set_band_description(1, str(band_name))

        return prediction_image

    except Exception as e:
        raise Exception(f"Failed to save prediction image: {str(e)}")


# ENVI-style interleave tokens (bil/bip/bsq) accepted from either the ENVI header
# or GDAL's generic names, mapped to the ENVI ``INTERLEAVE`` creation option.
_ENVI_INTERLEAVE = {'bil': 'BIL', 'bip': 'BIP', 'bsq': 'BSQ',
                    'line': 'BIL', 'pixel': 'BIP', 'band': 'BSQ'}


def _envi_georef(image_meta, src_image_path):
    """Return ``(map_info_list, coord_sys_string)`` for an ENVI output.

    Prefers an *exact copy* of the input ENVI header - spectral resampling does
    not change the spatial grid, so the input ``map info`` /
    ``coordinate system string`` apply verbatim.  Falls back to reconstructing
    ``map info`` from the rasterio profile's crs/transform."""
    if src_image_path:
        hdr = _find_sidecar_hdr(src_image_path)
        if hdr:
            h = parse_envi_header(hdr)
            mi, cs = h.get('map info'), h.get('coordinate system string')
            if mi:
                map_info = [t.strip() for t in mi.strip().strip('{}').split(',') if t.strip()]
                coord = cs.strip().strip('{}').strip() if cs else None
                return map_info, coord
    crs = image_meta.get('crs')
    transform = image_meta.get('transform')
    if crs is None or transform is None:
        return None, None
    try:
        px, py = abs(transform.a), abs(transform.e)
        east, north = transform.c, transform.f
        d = crs.to_dict() if hasattr(crs, 'to_dict') else {}
        cs = crs.to_wkt() if hasattr(crs, 'to_wkt') else None
        if d.get('proj') == 'utm':
            hemi = 'South' if d.get('south') else 'North'
            mi = ['UTM', 1.0, 1.0, east, north, px, py,
                  int(d.get('zone')), hemi, 'WGS-84', 'units=Meters']
        elif getattr(crs, 'is_geographic', False):
            mi = ['Geographic Lat/Lon', 1.0, 1.0, east, north, px, py,
                  'WGS-84', 'units=Degrees']
        else:
            mi = ['Arbitrary', 1.0, 1.0, east, north, px, py, 'units=Meters']
        return mi, cs
    except Exception:
        return None, None


def save_cube_image(cube, original_shape, wavelengths, fwhms, image_meta,
                    file_path, driver=None, nodata=-9999.0, interleave=None,
                    src_image_path=None):
    """Write the resampled reflectance cube (bands on the model grid) so the user
    can validate resampling.

    ENVI output is written with the **spectral** library (SPy), which produces a
    correct, complete ``.hdr`` - wavelength, FWHM, band names, ``data ignore
    value``, the preserved band ``interleave`` (bil/bip/bsq) and the input's
    georeferencing (``map info`` / ``coordinate system string``).  Other drivers
    (GeoTIFF, …) are written via rasterio with wavelengths in band descriptions.

    Args:
        cube: (pixels, bands) reflectance (row-major pixels; NaN at background).
        original_shape: source (bands, rows, cols) - only rows/cols are used.
        wavelengths / fwhms: per-band centres and (optional) widths of ``cube``.
        interleave: source band interleave to reproduce (ENVI output only).
        src_image_path: input image path, used to copy its ENVI georeferencing.
    """
    try:
        rows, cols = original_shape[1], original_shape[2]
        n_bands = cube.shape[1]

        wl = [float(w) for w in np.asarray(wavelengths).ravel()]
        wl = wl if len(wl) == n_bands else None      # only trust matching lengths
        fw = None if fwhms is None else [float(f) for f in np.asarray(fwhms).ravel()]
        fw = fw if (fw is not None and len(fw) == n_bands) else None

        out_driver = (driver or driver_for_path(file_path,
                      fallback=image_meta.get('driver', 'GTiff')))

        if out_driver == 'ENVI':
            return _save_cube_envi_spy(cube, rows, cols, n_bands, wl, fw,
                                       image_meta, file_path, nodata,
                                       interleave, src_image_path)

        # ── Non-ENVI (GeoTIFF, …) via rasterio ───────────────────────────────
        stack = cube.T.reshape(n_bands, rows, cols).astype('float32')   # (bands,r,c)
        stack = np.where(np.isfinite(stack), stack, nodata).astype('float32')
        profile = _spatial_profile(image_meta, out_driver, n_bands, rows, cols,
                                   'float32', nodata)
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(stack)
            if wl is not None:
                for b in range(n_bands):
                    with contextlib.suppress(Exception):
                        dst.set_band_description(b + 1, f"{wl[b]:.3f} nm")
        return stack

    except Exception as e:
        raise Exception(f"Failed to save resampled cube: {str(e)}")


def _save_cube_envi_spy(cube, rows, cols, n_bands, wl, fw, image_meta, file_path,
                        nodata, interleave, src_image_path):
    """Write a multi-band ENVI cube with a complete header via the SPy library."""
    import spectral.io.envi as envi

    # (pixels, bands) → (rows, cols, bands) - SPy's native layout - NaN → nodata.
    img = cube.reshape(rows, cols, n_bands).astype('float32')
    img = np.where(np.isfinite(img), img, np.float32(nodata))

    il = str(interleave).lower() if interleave else 'bil'
    if il not in ('bil', 'bip', 'bsq'):
        il = _ENVI_INTERLEAVE.get(il, 'BIL').lower()

    metadata = {'wavelength units': 'Nanometers', 'data ignore value': float(nodata)}
    if wl is not None:
        metadata['wavelength'] = wl
        metadata['band names'] = [f"{w:.3f} nm" for w in wl]
    if fw is not None:
        metadata['fwhm'] = fw
    map_info, coord = _envi_georef(image_meta, src_image_path)
    if map_info:
        metadata['map info'] = map_info
    if coord:
        metadata['coordinate system string'] = coord

    base, ext = os.path.splitext(file_path)
    hdr_path = base + '.hdr'
    envi.save_image(hdr_path, img, dtype='float32', force=True,
                    ext=(ext or '.img'), interleave=il, metadata=metadata)
    return img