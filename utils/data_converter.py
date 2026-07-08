"""
data_converter.py
=================
Paracuda Data Converter -- three-tab Tkinter wizard that auto-detects the
layout of an arbitrary Excel / CSV spectral file and converts it to the
Paracuda-compatible format:

    Names | Prop1 | ... | PropN | WL1 | WL2 | ... | WLM

Supported input layouts
-----------------------
1. Row-wise  -- samples in rows, wavelength values as column headers
   (e.g. Names | Sand | Silt | 425 | 480 | 545 | ...)
2. Col-wise  -- samples in columns, wavelengths as first-column values
   (transposed; common for hyperspectral instruments)
   May contain soil-property rows BEFORE the wavelength rows:
   e.g.  Row_label | S1 | S2
         Sand      | 45 | 32
         Silt      | 30 | 40
         425.0     | .45| .52
         480.0     | .48| .55

Wavelength units
----------------
* nm  : 350 - 14100
* um  : > 1.0 and <= 14.5  (values <= 1.0 are treated as reflectance /
        emissivity, never as wavelengths)

@author: Sharad Kumar Gupta
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

# Make the repo root importable — this file is launched as its own process from
# the utils/ folder, so the parent directory (where paracuda_theme.py lives) is
# not automatically on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Shared theme (optional): keeps the converter visually in sync with the main
# Paracuda window and honours the user's Theme-menu choice.  If the module is
# unavailable (e.g. this file copied out on its own) we fall back to the
# original hardcoded navy look further below.
try:
    from paracuda_theme import (get_palette, load_theme_name,
                                 apply_ttk_theme as _shared_apply_theme)
    _HAS_THEME = True
except Exception:
    _HAS_THEME = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NM_MIN, _NM_MAX = 350, 14100
_UM_MIN, _UM_MAX = 1.0, 25.0   # strictly > 1.0 (reflectance guard); up to ~25 µm for full FTIR range

_NAME_KEYWORDS = [
    'name', 'names', 'sample', 'samples', 'sample_id', 'sampleid',
    'sample_name', 'samplename', 'sample_no', 'sampleno', 'id',
    'label', 'labels', 'site', 'site_id', 'siteid',
    # soil science aliases
    'soil', 'soil_id', 'soilid', 'soil id',
    'soil_name', 'soil name', 'soil_no', 'soil no', 'soil number',
    'profile', 'profile_id', 'pedon', 'pedon_id',
]

_WL_LABEL_KEYWORDS = [
    'wavelength', 'wavelengths', 'wl', 'wave', 'nm', 'um',
    # wavenumber synonyms (Bruker FTIR / OPUS exports)
    'wavenumber', 'wavenumbers', 'wavenum', 'wavenumber [cm-1]',
    'cm-1', 'cm^-1', 'cm\u207b\u00b9',
]

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _is_wavelength_value(val):
    """Return True if *val* looks like a wavelength (nm or um range)."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return False
    # Reflectance / emissivity guard -- values <= 1 are never wavelengths
    if v <= 1.0:
        return False
    if _NM_MIN <= v <= _NM_MAX:
        return True
    if _UM_MIN < v <= _UM_MAX:
        return True
    return False


def _count_wavelength_like(series):
    """Return fraction [0..1] of non-null values in *series* that pass _is_wavelength_value."""
    vals = series.dropna()
    if len(vals) == 0:
        return 0.0
    hits = sum(1 for v in vals if _is_wavelength_value(v))
    return hits / len(vals)


def _is_micrometer_series(series):
    """
    Return True if all numeric values are in the um range (1.0 < v <= 14.5)
    and none are in the nm range (>= 350).
    """
    nums = pd.to_numeric(series, errors='coerce').dropna()
    if len(nums) == 0:
        return False
    return (
        bool((nums > 1.0).all())
        and bool((nums <= _UM_MAX).all())
        and not bool((nums >= _NM_MIN).any())
    )


def _to_numeric_safe(v):
    """Return float(v) or NaN."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return float('nan')


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def detect_wavelength_columns(df):
    """Return list of column names whose HEADER value looks like a wavelength."""
    result = []
    for col in df.columns:
        try:
            v = float(str(col))
            if _is_wavelength_value(v):
                result.append(col)
        except (ValueError, TypeError):
            pass
    return result


def detect_orientation(df):
    """
    Return (orientation, wl_cols, notes).

    orientation : 'row-wise' | 'col-wise'
    wl_cols     : list of wavelength column names (populated for row-wise only)
    notes       : list of human-readable detection messages

    Logic
    -----
    Row-wise : >= 5 numeric column headers that look like wavelengths.
    Col-wise : the first column (or a column named wavelength/wl/um/etc.)
               contains wavelength-like values -- EITHER overall >= 70 % OR
               the *tail* (last 30 values) >= 85 % wavelength-like.
               The tail check handles mixed files where soil-property rows
               precede spectral rows, reducing the overall ratio.
    """
    notes = []

    # ---- Try row-wise first -------------------------------------------------
    wl_cols = detect_wavelength_columns(df)
    if len(wl_cols) >= 5:
        wl_vals = []
        for c in wl_cols:
            try:
                wl_vals.append(float(str(c)))
            except (ValueError, TypeError):
                pass
        notes.append(
            "Row-wise layout detected: %d wavelength columns found (%.0f - %.0f)."
            % (len(wl_cols), min(wl_vals), max(wl_vals))
        )
        all_in_um = (
            len(wl_vals) >= 5
            and all(_UM_MIN < v <= _UM_MAX for v in wl_vals)
        )
        if all_in_um:
            notes.append(
                "Wavelength column headers appear to be in um "
                "(%.3f - %.3f um). They will be converted to nm (x1000)."
                % (min(wl_vals), max(wl_vals))
            )
        return 'row-wise', wl_cols, notes

    # ---- Try col-wise -------------------------------------------------------
    wl_col_candidates = [
        c for c in df.columns
        if str(c).lower() in _WL_LABEL_KEYWORDS
    ]
    check_col = wl_col_candidates[0] if wl_col_candidates else df.columns[0]

    series_vals = df[check_col].dropna().reset_index(drop=True)
    ratio       = _count_wavelength_like(series_vals)

    # Tail check: wavelengths come AFTER property rows in a mixed file.
    # Use the LAST half of the column (capped at 20) so that a small number of
    # leading property rows doesn't dilute the tail-wavelength fraction.
    tail_n     = min(20, max(5, len(series_vals) // 2))
    tail_ratio = _count_wavelength_like(series_vals.iloc[-tail_n:])

    if ratio >= 0.70 or tail_ratio >= 0.85:
        # Use only the numeric (wavelength-like) portion for um detection
        numeric_wl_vals = pd.to_numeric(
            series_vals[series_vals.apply(_is_wavelength_value)], errors='coerce'
        ).dropna()
        unit_note = ""
        if _is_micrometer_series(numeric_wl_vals):
            unit_note = " Wavelengths are in um and will be converted to nm (x1000)."
        notes.append(
            "Col-wise (transposed) layout detected: column '%s' contains "
            "wavelength values (overall=%.0f%%, tail=%.0f%%).%s"
            % (check_col, ratio * 100, tail_ratio * 100, unit_note)
        )
        return 'col-wise', [], notes

    # ---- Fallback -----------------------------------------------------------
    notes.append(
        "Could not confidently detect orientation -- defaulting to row-wise. "
        "Please verify in the Configure tab."
    )
    return 'row-wise', wl_cols, notes


def detect_name_column(df, wl_cols):
    """
    Return the column most likely to contain sample/soil names.

    Priority:
      1. Column header matches a known keyword (name, sample, soil, etc.)
      2. First column that is mostly strings and not a wavelength column
    """
    wl_set = set(str(c) for c in wl_cols)
    for col in df.columns:
        if str(col).lower().strip() in _NAME_KEYWORDS:
            return col
    # fallback: first mostly-string, non-wavelength column
    for col in df.columns:
        if str(col) in wl_set:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        n_num = sum(
            1 for v in vals
            if pd.to_numeric(str(v), errors='coerce') is not None
            and str(pd.to_numeric(str(v), errors='coerce')) != 'nan'
        )
        n_str = len(vals) - n_num
        if (n_str / len(vals)) >= 0.5:
            return col
    return None


def detect_property_columns(df, wl_cols, name_col):
    """
    Return columns that are neither the name column nor wavelength columns,
    but contain mostly numeric data (Sand, Silt, Clay, pH, OC, ...).
    """
    wl_set = set(str(c) for c in wl_cols)
    skip   = {str(name_col)} | wl_set if name_col else wl_set
    result = []
    for col in df.columns:
        if str(col) in skip:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        n_num = sum(
            1 for v in vals
            if str(pd.to_numeric(str(v), errors='coerce')) != 'nan'
        )
        if (n_num / len(vals)) > 0.5:
            result.append(col)
    return result


def is_already_paracuda(df):
    """Return True if *df* already matches the Paracuda format."""
    if len(df.columns) < 6:
        return False
    if str(df.columns[0]).lower().strip() != 'names':
        return False
    return len(detect_wavelength_columns(df)) >= 5


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_to_paracuda(df, orientation, wl_cols, name_col,
                        property_cols, auto_name_prefix='S',
                        wl_unit='auto'):
    # wl_unit: 'auto' | 'nm' | 'um'  (overrides heuristic when set)
    """
    Convert *df* to Paracuda format.

    Returns (out_df, warnings_list).

    Col-wise handling (transposed input)
    -------------------------------------
    The function splits the label column (first column) into:
      - spectral rows   : rows whose label is a wavelength-like number
      - property rows   : rows whose label is a soil/text identifier
                          (e.g. Sand, Silt, Clay)
    Spectral rows are transposed to form the spectral matrix.
    Property rows are transposed to form additional property columns.
    Both are then merged on the sample-name column.
    """
    warn = []

    if is_already_paracuda(df):
        warn.append("File is already in Paracuda format -- no conversion needed.")
        return df, warn

    # =========================================================================
    # COL-WISE (transposed)
    # =========================================================================
    if orientation == 'col-wise':
        wl_col_cands = [
            c for c in df.columns
            if str(c).lower() in _WL_LABEL_KEYWORDS
        ]
        idx_col = wl_col_cands[0] if wl_col_cands else df.columns[0]

        try:
            # Split rows into spectral and property groups
            is_wl_row = df[idx_col].apply(_is_wavelength_value)
            df_wl     = df[is_wl_row].copy().reset_index(drop=True)
            df_prop   = df[~is_wl_row].copy().reset_index(drop=True)

            if df_wl.empty:
                warn.append(
                    "WARNING: No wavelength rows found in transposed file -- "
                    "using entire dataframe as spectral data."
                )
                df_wl   = df.copy()
                df_prop = pd.DataFrame()

            # um detection on wavelength portion only (honour user override)
            if wl_unit == 'um':
                need_um = True
            elif wl_unit == 'nm':
                need_um = False
            else:
                need_um = _is_micrometer_series(
                    pd.to_numeric(df_wl[idx_col], errors='coerce').dropna()
                )

            # ---- Transpose spectral block -----------------------------------
            df_spec = df_wl.set_index(idx_col).T.reset_index()
            df_spec = df_spec.rename(columns={'index': 'Names'})
            df_spec.columns.name = None

            if need_um:
                def _col_to_nm(c):
                    try:
                        return round(float(c) * 1000, 1)
                    except (ValueError, TypeError):
                        return c
                df_spec.columns = [
                    _col_to_nm(c) if c != 'Names' else c
                    for c in df_spec.columns
                ]
                warn.append(
                    "INFO: Wavelengths were in micrometres (um) -- converted to "
                    "nanometres (nm) by multiplying by 1000."
                )

            # ---- Transpose property block (if any) -------------------------
            if not df_prop.empty:
                df_prop_clean = df_prop[
                    df_prop[idx_col].notna()
                    & (df_prop[idx_col].astype(str).str.strip() != '')
                ].copy()

                # If the caller supplied property names (from GUI selection),
                # keep only those property rows that the user selected.
                if property_cols:
                    pc_set = set(str(p) for p in property_cols)
                    df_prop_clean = df_prop_clean[
                        df_prop_clean[idx_col].astype(str).isin(pc_set)
                    ]

                if not df_prop_clean.empty:
                    df_props_t = df_prop_clean.set_index(idx_col).T.reset_index()
                    df_props_t = df_props_t.rename(columns={'index': 'Names'})
                    df_props_t.columns.name = None
                    prop_names_found = [c for c in df_props_t.columns if c != 'Names']

                    # Coerce property values to numeric
                    for pc in prop_names_found:
                        df_props_t[pc] = pd.to_numeric(df_props_t[pc], errors='coerce')

                    df_combined   = df_spec.merge(df_props_t, on='Names', how='left')
                    property_cols = [c for c in prop_names_found if c in df_combined.columns]

                    if prop_names_found:
                        warn.append(
                            "INFO: Found %d property row(s) in transposed file: %s%s"
                            % (
                                len(prop_names_found),
                                ', '.join(str(p) for p in prop_names_found[:8]),
                                ' ...' if len(prop_names_found) > 8 else '',
                            )
                        )
                else:
                    df_combined   = df_spec
                    property_cols = []
            else:
                df_combined   = df_spec
                property_cols = []

            # Re-detect wavelength columns after um->nm rename
            wl_cols  = detect_wavelength_columns(df_combined)
            name_col = 'Names'
            df       = df_combined

        except Exception as exc:
            warn.append("ERROR during col-wise transpose: %s" % exc)

    # =========================================================================
    # Build Names column
    # =========================================================================
    if name_col and name_col in df.columns:
        names = df[name_col].astype(str).str.strip().tolist()
    else:
        names = ["%s%03d" % (auto_name_prefix, i + 1) for i in range(len(df))]
        warn.append(
            "No sample-name column found; auto-generated names (S001, S002, ...)."
        )

    # =========================================================================
    # Row-wise um -> nm header conversion
    # =========================================================================
    wl_floats = []
    for c in wl_cols:
        try:
            wl_floats.append(float(str(c)))
        except (ValueError, TypeError):
            pass

    row_wise_um = (
        len(wl_floats) >= 5
        and all(_UM_MIN < v <= _UM_MAX for v in wl_floats)
    )
    if row_wise_um:
        col_rename = {}
        for old_c in df.columns:
            try:
                v = float(str(old_c))
                if _UM_MIN < v <= _UM_MAX:
                    col_rename[old_c] = round(v * 1000, 1)
            except (ValueError, TypeError):
                pass
        if col_rename:
            df      = df.rename(columns=col_rename)
            wl_cols = [col_rename.get(c, c) for c in wl_cols]
            warn.append(
                "INFO: Row-wise wavelength column headers converted from um to nm (x1000)."
            )

    # =========================================================================
    # Assemble output dataframe
    # =========================================================================
    wl_present   = [c for c in wl_cols if c in df.columns]
    prop_present = [
        c for c in (property_cols or [])
        if c in df.columns and c != name_col and c not in wl_present
    ]

    # Sort wavelength columns numerically
    def _wl_key(c):
        try:
            return float(str(c))
        except (ValueError, TypeError):
            return 0.0

    wl_sorted = sorted(wl_present, key=_wl_key)

    # Any column whose name is a purely-numeric value (e.g. a wavelength header
    # like 1000 or 1000.0) is written out as a *string* header. Numeric headers
    # otherwise round-trip through Excel/pandas as int/float, which breaks
    # downstream detection and produces lexicographic-vs-numeric mismatches.
    def _colname_to_str(c):
        s = str(c).strip()
        try:
            f = float(s)
        except (ValueError, TypeError):
            return c  # not fully numeric -> leave label untouched
        return str(int(f)) if f == int(f) else s

    # Build output using pd.concat to avoid DataFrame fragmentation warnings
    cols = {'Names': names}
    for pc in prop_present:
        cols[_colname_to_str(pc)] = pd.to_numeric(df[pc], errors='coerce').values
    for wc in wl_sorted:
        cols[_colname_to_str(wc)] = pd.to_numeric(df[wc], errors='coerce').values

    out_df = pd.DataFrame(cols)

    return out_df, warn


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def load_file(path, sheet=None):
    """Load an Excel or CSV file and return a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls', '.xlsm', '.ods'):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, header=0)
    if ext == '.csv':
        try:
            return pd.read_csv(path, encoding='utf-8', header=0)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='cp1252', header=0)
    raise ValueError("Unsupported file type: %s" % ext)


def get_sheet_names(path):
    """Return list of sheet names (Excel) or ['Sheet1'] for CSV."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls', '.xlsm', '.ods'):
        return pd.ExcelFile(path).sheet_names
    return ['Sheet1']


# ---------------------------------------------------------------------------
# GUI – style constants  (used throughout the whole application)
# ---------------------------------------------------------------------------

_FONT   = 'Segoe UI'          # single font family for all widgets
_FN     = (_FONT, 10)          # normal
_FB     = (_FONT, 10, 'bold')  # bold
_FH     = (_FONT, 11, 'bold')  # section heading
_FT     = (_FONT, 14, 'bold')  # window title
_FS     = (_FONT, 9)           # small / caption

# Colour palette — sourced from the shared theme (honours the user's Theme
# choice) with a fall-back to the original hardcoded navy values so this file
# still runs stand-alone if paracuda_theme.py is missing.
if _HAS_THEME:
    _PAL = get_palette(load_theme_name())
    _FONT = _PAL.get('FONT', _FONT)
    _FN = (_FONT, 10); _FB = (_FONT, 10, 'bold')
    _FH = (_FONT, 11, 'bold'); _FT = (_FONT, 14, 'bold'); _FS = (_FONT, 9)
    _C_WIN  = _PAL['WIN'];  _C_CARD = _PAL['CARD']; _C_LINE = _PAL['LINE']
    _C_BNR  = _PAL['BNR'];  _C_PRI  = _PAL['PRI'];  _C_PRI2 = _PAL['PRI2']
    _C_ACT  = _PAL['ACT'];  _C_ACT2 = _PAL['ACT2']; _C_GHO  = _PAL['GHO']
    _C_OK   = _PAL['OK'];   _C_WRN  = _PAL['WRN'];  _C_ERR  = _PAL['ERR']
    _C_INF  = _PAL['INF'];  _C_MUT  = _PAL['MUT']
    _C_ODD  = _PAL['ODD'];  _C_EVEN = _PAL['EVEN']; _C_SEL  = _PAL['SEL']
    _C_THDR = _PAL['THDR']; _C_TAB  = _PAL['TAB'];  _C_TABA = _PAL['TABA']
    _C_ACCENT = _PAL.get('ACCENT', '#3B82F6')
else:
    _PAL = None
    _C_WIN  = '#EFF4FC'   # outer window background
    _C_CARD = '#FFFFFF'   # card / panel fill
    _C_LINE = '#C8D8EE'   # separator / border
    _C_BNR  = '#1B3A6B'   # deep-navy banner
    _C_PRI  = '#1A56DB'   # primary action button
    _C_PRI2 = '#1446C0'   # primary hover
    _C_ACT  = '#E8EDF8'   # action / secondary button fill
    _C_ACT2 = '#D0DCF2'   # action hover
    _C_GHO  = _C_WIN      # ghost button fill
    _C_OK   = '#15803D'   # success green
    _C_WRN  = '#B45309'   # amber
    _C_ERR  = '#B91C1C'   # red
    _C_INF  = '#1E40AF'   # info blue
    _C_MUT  = '#6B7280'   # muted grey text
    _C_ODD  = '#F5F8FD'   # treeview odd-row tint
    _C_EVEN = '#FFFFFF'   # treeview even-row
    _C_SEL  = '#BFDBFE'   # treeview selected
    _C_THDR = '#DDE8F5'   # treeview heading background
    _C_TAB  = '#D6E4F5'   # inactive notebook tab
    _C_TABA = '#FFFFFF'   # active notebook tab
    _C_ACCENT = '#3B82F6'


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class ParacudaConverter(tk.Tk):
    """Three-tab Tkinter wizard for converting spectral data to Paracuda format."""

    def __init__(self):
        super().__init__()
        self.title("Paracuda Data Converter")
        self.geometry("1040x760")
        self.minsize(820, 600)
        self.resizable(True, True)
        self.configure(background=_C_WIN)
        self._set_icon()
        self._apply_styles()

        # ---- state ----
        self._df      = None
        self._df2     = None
        self._out_df  = None
        self._wl_unit = tk.StringVar(value='auto')   # 'auto' | 'nm' | 'um'

        self._path        = tk.StringVar()
        self._path2       = tk.StringVar()
        self._sheet       = tk.StringVar()
        self._sheet2      = tk.StringVar()
        self._orientation = tk.StringVar(value='auto')
        self._name_col    = tk.StringVar()
        self._wl_start    = tk.StringVar()
        self._wl_end      = tk.StringVar()
        self._out_path    = tk.StringVar()
        self._out_fmt     = tk.StringVar(value='xlsx')
        self._merge_mode  = tk.BooleanVar(value=False)
        self._status_var  = tk.StringVar(value='Ready — open a spectral file to begin.')

        self._wl_cols_detected   = []
        self._name_col_detected  = None
        self._prop_cols_detected = []

        self._build_ui()

    # ------------------------------------------------------------------
    def _set_icon(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for name in ('paracuda.ico', 'icon.ico'):
                p = os.path.join(script_dir, name)
                if os.path.exists(p):
                    self.iconbitmap(p)
                    break
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _apply_styles(self):
        # Prefer the shared theme routine so the converter matches the main
        # Paracuda window and the user's Theme-menu choice.
        if _HAS_THEME and _PAL is not None:
            _shared_apply_theme(self, _PAL)
            self.option_add('*background', _C_WIN)
            return

        st = ttk.Style(self)
        st.theme_use('clam')

        # propagate default font to all non-ttk widgets
        self.option_add('*Font', _FN)
        self.option_add('*background', _C_WIN)

        # TFrame
        st.configure('TFrame',        background=_C_WIN)
        st.configure('Card.TFrame',   background=_C_CARD)

        # TLabel
        st.configure('TLabel',        background=_C_WIN,  font=_FN)
        st.configure('Card.TLabel',   background=_C_CARD, font=_FN)
        st.configure('Head.TLabel',   background=_C_CARD, font=_FH,
                     foreground=_C_BNR)
        st.configure('Small.TLabel',  background=_C_WIN,  font=_FS,
                     foreground=_C_MUT)
        st.configure('Stat.TLabel',   background=_C_THDR, font=_FS,
                     foreground=_C_INF, padding=(6, 2))

        # TButton – consistent font everywhere
        st.configure('TButton',       font=_FN, padding=(10, 5))

        st.configure('Primary.TButton', font=_FB, padding=(18, 9),
                     background=_C_PRI, foreground='white',
                     bordercolor=_C_PRI, focusthickness=0)
        st.map('Primary.TButton',
               background=[('active', _C_PRI2), ('pressed', '#1039A8'),
                            ('disabled', '#9BB0D8')],
               foreground=[('disabled', 'white')])

        st.configure('Action.TButton', font=_FB, padding=(12, 7),
                     background=_C_ACT, foreground=_C_BNR,
                     bordercolor=_C_LINE, focusthickness=0)
        st.map('Action.TButton',
               background=[('active', _C_ACT2), ('pressed', '#B8C8E8')])

        st.configure('Ghost.TButton', font=_FN, padding=(6, 4),
                     background=_C_WIN, foreground=_C_MUT,
                     bordercolor=_C_WIN, focusthickness=0)
        st.map('Ghost.TButton',
               background=[('active', _C_LINE)])

        # TNotebook
        st.configure('TNotebook',     background=_C_WIN, borderwidth=0,
                     tabmargins=[2, 4, 0, 0])
        st.configure('TNotebook.Tab', font=_FB, padding=(20, 10),
                     background=_C_TAB, foreground=_C_MUT, borderwidth=0)
        st.map('TNotebook.Tab',
               background=[('selected', _C_TABA)],
               foreground=[('selected', _C_BNR)],
               expand=[('selected', [1, 1, 1, 0])])

        # TLabelframe
        st.configure('TLabelframe',       background=_C_CARD,
                     bordercolor=_C_LINE, relief='groove', borderwidth=1)
        st.configure('TLabelframe.Label', background=_C_CARD, font=_FB,
                     foreground=_C_BNR, padding=(4, 0))

        # Treeview
        st.configure('Treeview',          font=_FN, rowheight=26,
                     background=_C_CARD, fieldbackground=_C_CARD,
                     borderwidth=0, relief='flat')
        st.configure('Treeview.Heading',  font=_FB,
                     background=_C_THDR, foreground=_C_BNR,
                     relief='flat', padding=(4, 6))
        st.map('Treeview',
               background=[('selected', _C_SEL)],
               foreground=[('selected', _C_BNR)])

        # TEntry / TCombobox
        st.configure('TEntry',    font=_FN, padding=(5, 4),
                     fieldbackground=_C_CARD, bordercolor=_C_LINE)
        st.map('TEntry',    bordercolor=[('focus', _C_PRI)])

        st.configure('TCombobox', font=_FN, padding=(5, 4),
                     fieldbackground=_C_CARD, bordercolor=_C_LINE)
        st.map('TCombobox', bordercolor=[('focus', _C_PRI)])

        # TCheckbutton / TRadiobutton
        st.configure('TCheckbutton',       background=_C_CARD, font=_FN)
        st.configure('TRadiobutton',       background=_C_CARD, font=_FN)
        st.configure('Card.TCheckbutton',  background=_C_CARD, font=_FN)
        st.configure('Card.TRadiobutton',  background=_C_CARD, font=_FN)

        # TScrollbar
        st.configure('TScrollbar',   background=_C_WIN, troughcolor=_C_WIN,
                     arrowcolor=_C_MUT, borderwidth=0, relief='flat')
        st.map('TScrollbar', background=[('active', _C_LINE)])

        # TSeparator
        st.configure('TSeparator',   background=_C_LINE)

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)   # notebook row expands

        self._build_header()

        # Notebook (main content)
        self._nb = ttk.Notebook(self)
        self._nb.grid(row=1, column=0, sticky='nsew')
        self._nb.bind('<<NotebookTabChanged>>', self._on_tab_change)

        self._tab1 = ttk.Frame(self._nb, padding=16)
        self._tab2 = ttk.Frame(self._nb, padding=16)
        self._tab3 = ttk.Frame(self._nb, padding=16)

        self._nb.add(self._tab1, text='  Step 1 · Load Data  ')
        self._nb.add(self._tab2, text='  Step 2 · Configure  ')
        self._nb.add(self._tab3, text='  Step 3 · Preview & Export  ')

        self._build_tab1()
        self._build_tab2()
        self._build_tab3()

        self._build_nav()
        self._build_statusbar()
        self._update_nav(0)

    # ------------------------------------------------------------------
    def _build_header(self):
        hf = tk.Frame(self, background=_C_BNR, height=62)
        hf.grid(row=0, column=0, sticky='ew')
        hf.grid_propagate(False)
        hf.columnconfigure(1, weight=1)

        # left accent stripe
        tk.Frame(hf, background='#3B82F6', width=6).grid(
            row=0, column=0, sticky='ns')

        # title + subtitle
        tf = tk.Frame(hf, background=_C_BNR)
        tf.grid(row=0, column=1, padx=(18, 0), sticky='w')
        tk.Label(tf, text='Paracuda  Data Converter',
                 font=(_FONT, 15, 'bold'),
                 background=_C_BNR, foreground='#FFFFFF').pack(anchor='w',
                                                               pady=(10, 0))
        tk.Label(tf, text='Convert spectral / soil data files to Paracuda format',
                 font=_FS, background=_C_BNR,
                 foreground='#93B4D8').pack(anchor='w')

        # right version tag
        tk.Label(hf, text='v2.0', font=_FS,
                 background=_C_BNR, foreground='#93B4D8').grid(
            row=0, column=2, padx=18, sticky='e')

    # ------------------------------------------------------------------
    def _build_nav(self):
        """Shared bottom navigation bar with Back / Next buttons."""
        tk.Frame(self, background=_C_LINE, height=1).grid(
            row=2, column=0, sticky='ew')

        nf = tk.Frame(self, background=_C_CARD, height=56)
        nf.grid(row=3, column=0, sticky='ew')
        nf.grid_propagate(False)
        nf.columnconfigure(0, weight=1)

        self._btn_back = ttk.Button(nf, text='← Back',
                                    style='Action.TButton',
                                    command=self._go_back)
        self._btn_back.grid(row=0, column=0, padx=18, pady=10, sticky='w')

        self._btn_next = ttk.Button(nf, text='Next: Configure  →',
                                    style='Primary.TButton',
                                    command=self._go_next)
        self._btn_next.grid(row=0, column=1, padx=18, pady=10, sticky='e')

    # ------------------------------------------------------------------
    def _build_statusbar(self):
        sf = tk.Frame(self, background='#E4EDF8', height=24)
        sf.grid(row=4, column=0, sticky='ew')
        sf.grid_propagate(False)
        sf.columnconfigure(1, weight=1)

        tk.Frame(sf, background=_C_PRI, width=4).grid(
            row=0, column=0, sticky='ns')
        tk.Label(sf, textvariable=self._status_var,
                 font=_FS, background='#E4EDF8',
                 foreground=_C_MUT, anchor='w').grid(
            row=0, column=1, padx=(8, 0), sticky='w')

    # ------------------------------------------------------------------
    def _on_tab_change(self, _event=None):
        idx = self._nb.index(self._nb.select())
        self._update_nav(idx)

    def _update_nav(self, tab_idx):
        if tab_idx == 0:
            self._btn_back.configure(state='disabled', text='← Back')
            self._btn_next.configure(state='normal',
                                     text='Next: Configure  →',
                                     style='Primary.TButton')
        elif tab_idx == 1:
            self._btn_back.configure(state='normal', text='← Back')
            self._btn_next.configure(state='normal',
                                     text='Next: Preview Output  →',
                                     style='Primary.TButton')
        else:
            self._btn_back.configure(state='normal', text='← Back')
            self._btn_next.configure(state='normal',
                                     text='  Export File  ',
                                     style='Primary.TButton')

    def _go_back(self):
        idx = self._nb.index(self._nb.select())
        if idx > 0:
            self._nb.select(idx - 1)

    def _go_next(self):
        idx = self._nb.index(self._nb.select())
        if idx == 0:
            self._load_and_preview()
            if self._df is not None:
                self._nb.select(1)
        elif idx == 1:
            self._apply_and_goto_tab3()
        else:
            self._export()

    def _set_status(self, msg):
        self._status_var.set(msg)
        self.update_idletasks()

    # ------------------------------------------------------------------
    @staticmethod
    def _make_stat_badge(parent, label, value, bg=_C_THDR, fg=_C_INF):
        """Return a small info badge frame."""
        f = tk.Frame(parent, background=bg)
        tk.Label(f, text=label + ': ', font=_FS,
                 background=bg, foreground=_C_MUT).pack(side='left',
                                                        padx=(6, 0))
        tk.Label(f, text=str(value), font=(_FONT, 9, 'bold'),
                 background=bg, foreground=fg).pack(side='left',
                                                    padx=(0, 6))
        return f

    def _refresh_badges(self, frame, badge_data):
        """Repopulate a stats-badge row from list of (label, value) pairs."""
        for w in frame.winfo_children():
            w.destroy()
        for lbl, val in badge_data:
            b = self._make_stat_badge(frame, lbl, val)
            b.pack(side='left', padx=(0, 6), pady=2)

    # ==================================================================
    # Tab 1 -- Load Data
    # ==================================================================
    def _build_tab1(self):
        f = self._tab1
        f.columnconfigure(0, weight=1)
        f.rowconfigure(3, weight=1)

        # ---- Primary file card ----
        c1 = ttk.LabelFrame(f, text=' Spectral / Data File ', padding=12)
        c1.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        c1.columnconfigure(0, weight=1)

        pr = tk.Frame(c1, background=_C_CARD)
        pr.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        pr.columnconfigure(0, weight=1)
        ttk.Entry(pr, textvariable=self._path).grid(
            row=0, column=0, sticky='ew', ipady=3)
        ttk.Button(pr, text='Browse…', style='Action.TButton',
                   command=self._browse_file1).grid(
            row=0, column=1, padx=(8, 0))

        sr = tk.Frame(c1, background=_C_CARD)
        sr.grid(row=1, column=0, sticky='w')
        tk.Label(sr, text='Sheet:', background=_C_CARD,
                 font=_FN).pack(side='left')
        self._sheet_cb = ttk.Combobox(sr, textvariable=self._sheet,
                                       width=36, state='readonly')
        self._sheet_cb.pack(side='left', padx=(8, 0))

        # ---- Merge card ----
        c2 = ttk.LabelFrame(f, text=' Optional: Merge Soil-Properties File ',
                             padding=12)
        c2.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        c2.columnconfigure(0, weight=1)

        ck_row = tk.Frame(c2, background=_C_CARD)
        ck_row.grid(row=0, column=0, sticky='w', pady=(0, 4))
        ttk.Checkbutton(ck_row,
                        text='Enable merge with a separate properties file',
                        style='Card.TCheckbutton',
                        variable=self._merge_mode,
                        command=self._toggle_merge).pack(side='left')

        self._merge_inner = tk.Frame(c2, background=_C_CARD)
        self._merge_inner.grid(row=1, column=0, sticky='ew')
        self._merge_inner.columnconfigure(0, weight=1)
        self._merge_inner.grid_remove()

        pr2 = tk.Frame(self._merge_inner, background=_C_CARD)
        pr2.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        pr2.columnconfigure(0, weight=1)
        ttk.Entry(pr2, textvariable=self._path2).grid(
            row=0, column=0, sticky='ew', ipady=3)
        ttk.Button(pr2, text='Browse…', style='Action.TButton',
                   command=self._browse_file2).grid(
            row=0, column=1, padx=(8, 0))

        sr2 = tk.Frame(self._merge_inner, background=_C_CARD)
        sr2.grid(row=1, column=0, sticky='w')
        tk.Label(sr2, text='Sheet:', background=_C_CARD,
                 font=_FN).pack(side='left')
        self._sheet2_cb = ttk.Combobox(sr2, textvariable=self._sheet2,
                                        width=36, state='readonly')
        self._sheet2_cb.pack(side='left', padx=(8, 0))

        # ---- Load button + stats badges ----
        lr = tk.Frame(f, background=_C_WIN)
        lr.grid(row=2, column=0, sticky='ew', pady=(0, 8))
        lr.columnconfigure(1, weight=1)
        ttk.Button(lr, text='Load & Preview', style='Action.TButton',
                   command=self._load_and_preview).grid(
            row=0, column=0, sticky='w')
        self._load_stats_row = tk.Frame(lr, background=_C_WIN)
        self._load_stats_row.grid(row=0, column=1, padx=(14, 0), sticky='w')

        # ---- Preview card ----
        cp = ttk.LabelFrame(f, text=' Data Preview (first 8 rows) ', padding=4)
        cp.grid(row=3, column=0, sticky='nsew')
        cp.rowconfigure(0, weight=1)
        cp.columnconfigure(0, weight=1)

        self._prev_tv = ttk.Treeview(cp, show='headings', height=8)
        vsb = ttk.Scrollbar(cp, orient='vertical',   command=self._prev_tv.yview)
        hsb = ttk.Scrollbar(cp, orient='horizontal', command=self._prev_tv.xview)
        self._prev_tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._prev_tv.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self._prev_tv.tag_configure('odd',  background=_C_ODD)
        self._prev_tv.tag_configure('even', background=_C_EVEN)

    # ==================================================================
    # Tab 2 -- Configure
    # ==================================================================
    def _build_tab2(self):
        f = self._tab2
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        # ---- Detection notes card ----
        cn = ttk.LabelFrame(f, text=' Auto-Detection Notes ', padding=10)
        cn.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        cn.columnconfigure(0, weight=1)

        self._notes_text = tk.Text(
            cn, height=5, wrap='word', state='disabled',
            background=_C_CARD, relief='flat', font=_FN,
            borderwidth=0, highlightthickness=0, foreground='#374151')
        self._notes_text.grid(row=0, column=0, sticky='ew')
        # colour tags for different message kinds
        self._notes_text.tag_configure('ok',   foreground=_C_OK)
        self._notes_text.tag_configure('warn', foreground=_C_WRN)
        self._notes_text.tag_configure('err',  foreground=_C_ERR)
        self._notes_text.tag_configure('info', foreground=_C_INF)
        self._notes_text.tag_configure('sep',  foreground=_C_MUT)

        # ---- Settings card ----
        cs = ttk.LabelFrame(f, text=' Conversion Settings ', padding=12)
        cs.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        cs.columnconfigure(1, weight=1)

        W = 22   # label column width (chars)

        tk.Label(cs, text='Orientation:', background=_C_CARD,
                 font=_FN, width=W, anchor='w').grid(
            row=0, column=0, sticky='w', pady=5)
        ttk.Combobox(cs, textvariable=self._orientation, width=22,
                     state='readonly',
                     values=['auto', 'row-wise', 'col-wise']).grid(
            row=0, column=1, sticky='w', pady=5)

        tk.Label(cs, text='Sample-name column:', background=_C_CARD,
                 font=_FN, width=W, anchor='w').grid(
            row=1, column=0, sticky='w', pady=5)
        self._name_cb = ttk.Combobox(cs, textvariable=self._name_col,
                                      width=32, state='readonly')
        self._name_cb.grid(row=1, column=1, sticky='w', pady=5)

        tk.Label(cs, text='Wavelength range:', background=_C_CARD,
                 font=_FN, width=W, anchor='w').grid(
            row=2, column=0, sticky='w', pady=5)
        rng = tk.Frame(cs, background=_C_CARD)
        rng.grid(row=2, column=1, sticky='w', pady=5)
        self._wl_start_cb = ttk.Combobox(rng, textvariable=self._wl_start,
                                          width=10)
        self._wl_end_cb   = ttk.Combobox(rng, textvariable=self._wl_end,
                                          width=10)
        self._wl_start_cb.pack(side='left')
        tk.Label(rng, text='  to  ', background=_C_CARD,
                 font=_FN).pack(side='left')
        self._wl_end_cb.pack(side='left')

        ttk.Button(cs, text='Run Auto-Detect', style='Action.TButton',
                   command=self._run_autodetect).grid(
            row=3, column=0, columnspan=2, sticky='w', pady=(10, 0))

        # ---- Property columns card ----
        cp = ttk.LabelFrame(
            f, text=' Property Columns  (Ctrl+click = multi-select) ',
            padding=10)
        cp.grid(row=2, column=0, sticky='nsew')
        cp.rowconfigure(0, weight=1)
        cp.columnconfigure(0, weight=1)

        pf = tk.Frame(cp, background=_C_CARD)
        pf.grid(row=0, column=0, sticky='nsew')
        pf.rowconfigure(0, weight=1)
        pf.columnconfigure(0, weight=1)

        self._prop_lb = tk.Listbox(
            pf, selectmode='extended', height=6,
            exportselection=False, font=_FN,
            background=_C_CARD, relief='flat',
            selectbackground=_C_SEL, selectforeground=_C_BNR,
            borderwidth=0, highlightthickness=1,
            highlightcolor=_C_PRI, highlightbackground=_C_LINE)
        prop_vsb = ttk.Scrollbar(pf, orient='vertical',
                                  command=self._prop_lb.yview)
        self._prop_lb.configure(yscrollcommand=prop_vsb.set)
        self._prop_lb.grid(row=0, column=0, sticky='nsew')
        prop_vsb.grid(row=0, column=1, sticky='ns')

        sf = tk.Frame(cp, background=_C_CARD)
        sf.grid(row=1, column=0, sticky='w', pady=(6, 0))
        ttk.Button(sf, text='Select All', style='Ghost.TButton',
                   command=lambda: self._prop_lb.select_set(0, 'end')).pack(
            side='left')
        ttk.Button(sf, text='Select None', style='Ghost.TButton',
                   command=lambda: self._prop_lb.selection_clear(0, 'end')).pack(
            side='left', padx=(4, 0))

    # ==================================================================
    # Tab 3 -- Preview & Export
    # ==================================================================
    def _build_tab3(self):
        f = self._tab3
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        # ---- Output preview card ----
        cp = ttk.LabelFrame(f, text=' Converted Output Preview ', padding=4)
        cp.grid(row=0, column=0, sticky='nsew', pady=(0, 8))
        cp.rowconfigure(0, weight=1)
        cp.columnconfigure(0, weight=1)

        self._out_tv = ttk.Treeview(cp, show='headings', height=10)
        vsb = ttk.Scrollbar(cp, orient='vertical',   command=self._out_tv.yview)
        hsb = ttk.Scrollbar(cp, orient='horizontal', command=self._out_tv.xview)
        self._out_tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._out_tv.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self._out_tv.tag_configure('odd',  background=_C_ODD)
        self._out_tv.tag_configure('even', background=_C_EVEN)

        # output stats badges row
        self._out_stats_row = tk.Frame(f, background=_C_WIN)
        self._out_stats_row.grid(row=1, column=0, sticky='w', pady=(0, 8))

        # ---- Export settings card ----
        ce = ttk.LabelFrame(f, text=' Export Settings ', padding=12)
        ce.grid(row=2, column=0, sticky='ew')
        ce.columnconfigure(1, weight=1)

        tk.Label(ce, text='Save to:', background=_C_CARD,
                 font=_FN, width=12, anchor='w').grid(
            row=0, column=0, sticky='w', pady=5)
        or_ = tk.Frame(ce, background=_C_CARD)
        or_.grid(row=0, column=1, sticky='ew', pady=5)
        or_.columnconfigure(0, weight=1)
        ttk.Entry(or_, textvariable=self._out_path).grid(
            row=0, column=0, sticky='ew', ipady=3)
        ttk.Button(or_, text='Browse…', style='Action.TButton',
                   command=self._browse_out).grid(
            row=0, column=1, padx=(8, 0))

        tk.Label(ce, text='Format:', background=_C_CARD,
                 font=_FN, width=12, anchor='w').grid(
            row=1, column=0, sticky='w', pady=5)
        ff = tk.Frame(ce, background=_C_CARD)
        ff.grid(row=1, column=1, sticky='w', pady=5)
        ttk.Radiobutton(ff, text='Excel (.xlsx)', style='Card.TRadiobutton',
                        variable=self._out_fmt, value='xlsx').pack(side='left')
        ttk.Radiobutton(ff, text='CSV (.csv)', style='Card.TRadiobutton',
                        variable=self._out_fmt, value='csv').pack(
            side='left', padx=(20, 0))

    # ==================================================================
    # Event handlers
    # ==================================================================

    def _browse_file1(self):
        path = filedialog.askopenfilename(
            title="Select spectral / data file",
            filetypes=[
                ("Spreadsheets", "*.xlsx *.xls *.xlsm *.ods *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path.set(path)
            sheets = get_sheet_names(path)
            self._sheet_cb['values'] = sheets
            self._sheet.set(sheets[0])

    def _browse_file2(self):
        path = filedialog.askopenfilename(
            title="Select properties file",
            filetypes=[
                ("Spreadsheets", "*.xlsx *.xls *.xlsm *.ods *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path2.set(path)
            sheets = get_sheet_names(path)
            self._sheet2_cb['values'] = sheets
            self._sheet2.set(sheets[0])

    def _toggle_merge(self):
        if self._merge_mode.get():
            self._merge_inner.grid()
        else:
            self._merge_inner.grid_remove()

    def _load_and_preview(self):
        path = self._path.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please select a file first.")
            return
        self._set_status('Loading file…')
        try:
            sheet = self._sheet.get() or 0
            self._df = load_file(path, sheet=sheet)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._set_status('Load failed.')
            return

        if self._merge_mode.get() and self._path2.get().strip():
            try:
                sheet2 = self._sheet2.get() or 0
                self._df2 = load_file(self._path2.get().strip(), sheet=sheet2)
            except Exception as exc:
                messagebox.showwarning("Merge file warning", str(exc))
                self._df2 = None

        self._populate_treeview(self._prev_tv, self._df.head(8))

        # show stats badges
        r, c = self._df.shape
        wl_q  = len(detect_wavelength_columns(self._df))
        self._refresh_badges(self._load_stats_row,
                             [('Rows', r), ('Columns', c),
                              ('Spectral bands detected', wl_q)])

        self._run_autodetect()
        self._set_status(
            'Loaded: %s  (%d rows × %d columns)' % (
                os.path.basename(path), r, c))

    def _populate_treeview(self, tv, df):
        tv.delete(*tv.get_children())
        # Limit to first 20 columns to keep the preview fast
        MAX_PREVIEW_COLS = 20
        total_cols = len(df.columns)
        if total_cols > MAX_PREVIEW_COLS:
            df = df.iloc[:, :MAX_PREVIEW_COLS].copy()
            df.insert(len(df.columns), f'… (+{total_cols - MAX_PREVIEW_COLS} more)', '…')
        cols = [str(c) for c in df.columns]
        tv['columns'] = cols
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=max(70, min(160, len(c) * 9)),
                      anchor='center', minwidth=55)
        for i, (_, row) in enumerate(df.iterrows()):
            tag = 'odd' if i % 2 == 0 else 'even'
            tv.insert('', 'end', values=[str(v) for v in row], tags=(tag,))

    def _insert_note(self, text):
        """Append a line to the notes widget, coloured by content."""
        tl = text.lower()
        if 'warning' in tl:
            tag = 'warn'
        elif 'error' in tl:
            tag = 'err'
        elif tl.startswith('info') or 'info:' in tl:
            tag = 'info'
        elif 'detected' in tl or 'found' in tl or 'layout' in tl:
            tag = 'ok'
        elif '---' in text:
            tag = 'sep'
        else:
            tag = ''
        self._notes_text.insert('end', '  ' + text + '\n', tag)

    def _ask_wl_unit_dialog(self, min_wl, max_wl):
        """Popup asking the user whether wavelengths are in nm or µm."""
        dlg = tk.Toplevel(self)
        dlg.title("Wavelength Units")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.configure(background=_C_WIN)

        unit_var = tk.StringVar(value='um')

        ttk.Label(dlg, text="Confirm Wavelength Units",
                  style='Head.TLabel', padding=(20, 16, 20, 4)).pack(fill='x')

        msg = (
            "Values like %.4g \u2013 %.4g were detected in the wavelength column.\n"
            "Please confirm the units so Paracuda Converter can apply the\n"
            "correct nm conversion."
        ) % (min_wl, max_wl)
        ttk.Label(dlg, text=msg, padding=(20, 4, 20, 8)).pack(fill='x')

        frame = ttk.Frame(dlg, padding=(20, 4, 20, 8))
        frame.pack(fill='x')
        for label, val in [
            ('Micrometres (µm)  →  will be converted to nm', 'um'),
            ('Nanometres (nm)  →  no unit conversion applied', 'nm'),
            ('Auto-detect (use heuristic)', 'auto'),
        ]:
            ttk.Radiobutton(frame, text=label,
                            variable=unit_var, value=val).pack(anchor='w', pady=2)

        def _ok():
            self._wl_unit.set(unit_var.get())
            dlg.destroy()

        btn_frame = ttk.Frame(dlg, padding=(20, 4, 20, 16))
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text='OK', style='Primary.TButton',
                   command=_ok, width=12).pack(side='right')

        self.wait_window(dlg)

    def _run_autodetect(self):
        if self._df is None:
            messagebox.showinfo("No data", "Load a file first (Tab 1).")
            return

        df       = self._df
        ori_pref = self._orientation.get()

        if ori_pref == 'auto':
            orientation, wl_cols, notes = detect_orientation(df)
        else:
            orientation = ori_pref
            wl_cols     = detect_wavelength_columns(df) if orientation == 'row-wise' else []
            notes       = ["Orientation manually set to '%s'." % orientation]

        self._wl_cols_detected   = wl_cols
        self._name_col_detected  = detect_name_column(df, wl_cols)

        if orientation == 'col-wise':
            # For col-wise (transposed) files the soil property names are the
            # string values in the first column (rows that are NOT wavelengths).
            idx_col = df.columns[0]
            self._prop_cols_detected = [
                v for v in df[idx_col]
                if pd.notna(v) and str(v).strip() and not _is_wavelength_value(v)
            ]
        else:
            self._prop_cols_detected = detect_property_columns(
                df, wl_cols, self._name_col_detected)

        # If col-wise with µm wavelengths, ask user to confirm units once
        if orientation == 'col-wise' and self._wl_unit.get() == 'auto':
            idx_col = df.columns[0]
            col_vals = pd.to_numeric(df[idx_col], errors='coerce').dropna()
            if _is_micrometer_series(col_vals):
                self._ask_wl_unit_dialog(float(col_vals.min()), float(col_vals.max()))

        # Notes panel
        self._notes_text.configure(state='normal')
        self._notes_text.delete('1.0', 'end')
        for n in notes:
            self._insert_note('* ' + n)
        self._notes_text.configure(state='disabled')

        # Name-column dropdown
        non_wl = [str(c) for c in df.columns if c not in self._wl_cols_detected]
        self._name_cb['values'] = non_wl
        nc = str(self._name_col_detected) if self._name_col_detected else ''
        if nc in non_wl:
            self._name_col.set(nc)
        elif non_wl:
            self._name_col.set(non_wl[0])

        # Wavelength range combos
        wl_numeric = []
        for c in wl_cols:
            try:
                wl_numeric.append(float(str(c)))
            except (ValueError, TypeError):
                pass
        if wl_numeric:
            wl_strs = [str(int(v)) if v == int(v) else str(v)
                       for v in sorted(wl_numeric)]
            self._wl_start_cb['values'] = wl_strs
            self._wl_end_cb['values']   = wl_strs
            self._wl_start.set(wl_strs[0])
            self._wl_end.set(wl_strs[-1])

        # Property listbox
        self._prop_lb.delete(0, 'end')
        for pc in self._prop_cols_detected:
            self._prop_lb.insert('end', str(pc))
        self._prop_lb.select_set(0, 'end')

    def _get_selected_properties(self):
        return [self._prop_lb.get(i) for i in self._prop_lb.curselection()]

    def _apply_and_goto_tab3(self):
        if self._df is None:
            messagebox.showinfo("No data", "Load a file first (Tab 1).")
            return

        self._set_status('Converting…')
        df = self._df.copy()

        # Wavelength range filter
        try:
            wl_start = float(self._wl_start.get()) if self._wl_start.get() else None
            wl_end   = float(self._wl_end.get())   if self._wl_end.get()   else None
        except ValueError:
            wl_start = wl_end = None

        wl_cols = self._wl_cols_detected[:]
        if wl_start is not None and wl_end is not None:
            def _in_range(c):
                try:
                    return wl_start <= float(str(c)) <= wl_end
                except (ValueError, TypeError):
                    return True
            wl_cols = [c for c in wl_cols if _in_range(c)]

        ori = self._orientation.get()
        if ori == 'auto':
            ori = detect_orientation(df)[0]

        name_col      = self._name_col.get() or self._name_col_detected
        property_cols = self._get_selected_properties()

        out_df, warn_list = convert_to_paracuda(
            df, ori, wl_cols, name_col, property_cols,
            wl_unit=self._wl_unit.get()
        )

        # Merge second file if requested
        if self._merge_mode.get() and self._df2 is not None:
            try:
                df2  = self._df2.copy()
                wl2  = detect_wavelength_columns(df2)
                nc2  = detect_name_column(df2, wl2)
                if nc2:
                    df2 = df2.rename(columns={nc2: 'Names'})
                    merge_props = [c for c in df2.columns
                                   if c != 'Names' and c not in wl2]
                    out_df = out_df.merge(
                        df2[['Names'] + merge_props], on='Names', how='left')
                    warn_list.append(
                        "INFO: Merged %d property column(s) from second file: %s"
                        % (len(merge_props),
                           ', '.join(str(p) for p in merge_props[:6])))
            except Exception as exc:
                warn_list.append("WARNING: Merge failed: %s" % exc)

        self._out_df = out_df

        # Append conversion messages to notes
        if warn_list:
            self._notes_text.configure(state='normal')
            self._insert_note('--- Conversion messages ---')
            for w in warn_list:
                self._insert_note(w)
            self._notes_text.configure(state='disabled')

        self._populate_treeview(self._out_tv, out_df.head(10))

        # Output stats badges
        n_samples = len(out_df)
        n_wl      = len(detect_wavelength_columns(out_df))
        n_props   = len(out_df.columns) - n_wl - 1   # minus Names column
        self._refresh_badges(self._out_stats_row,
                             [('Samples', n_samples),
                              ('Property columns', max(n_props, 0)),
                              ('Spectral bands', n_wl)])

        # Suggest output path
        in_path = self._path.get()
        if in_path and not self._out_path.get():
            base, _ = os.path.splitext(in_path)
            self._out_path.set(base + "_paracuda.xlsx")

        self._set_status(
            'Conversion complete — %d samples, %d spectral bands, %d properties.'
            % (n_samples, n_wl, max(n_props, 0)))
        self._nb.select(2)

    def _browse_out(self):
        fmt    = self._out_fmt.get()
        ftypes = [("Excel", "*.xlsx")] if fmt == 'xlsx' else [("CSV", "*.csv")]
        path   = filedialog.asksaveasfilename(
            title="Save converted file",
            filetypes=ftypes,
            defaultextension="." + fmt,
        )
        if path:
            self._out_path.set(path)

    def _export(self):
        if self._out_df is None:
            messagebox.showinfo("Nothing to export",
                                "Run conversion first (Step 2 → Next).")
            return
        path = self._out_path.get().strip()
        if not path:
            messagebox.showwarning("No path",
                                   "Please specify an output file path.")
            return
        self._set_status('Saving…')
        try:
            fmt = self._out_fmt.get()
            if fmt == 'xlsx':
                self._out_df.to_excel(path, index=False)
            else:
                self._out_df.to_csv(path, index=False)
            self._set_status(
                'Saved: %s  (%d rows × %d columns)' % (
                    os.path.basename(path),
                    len(self._out_df), len(self._out_df.columns)))
            messagebox.showinfo(
                "Exported",
                "Saved %d rows \u00d7 %d columns to:\n%s"
                % (len(self._out_df), len(self._out_df.columns), path),
            )
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self._set_status('Export failed.')


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def launch():
    """Create and run the converter GUI."""
    app = ParacudaConverter()
    app.mainloop()


def main():
    launch()


if __name__ == '__main__':
    main()
