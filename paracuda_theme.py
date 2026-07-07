"""
paracuda_theme.py
=================
Shared visual theme for the whole Paracuda software — both the main
``Paracuda III`` window (``gui/``) and the standalone Data Converter
(``utils/data_converter.py``).

The Data Converter already shipped a polished navy/blue "card" look driven by a
set of ``_C_*`` colour constants plus a ``_apply_styles`` method.  This module
generalises that into

  * a dictionary of named **palettes** (Ocean / Slate / Forest / Light / Dark),
  * ``apply_ttk_theme(root, palette)`` — the single styling routine both tools
    call (lifted from the converter's ``_apply_styles``), and
  * a tiny JSON settings file so the two *separate processes* agree on the
    active theme.

Everything degrades gracefully: if this module cannot be imported (e.g. the
converter is copied out on its own) the caller keeps its old hardcoded look.

@author: Sharad Kumar Gupta
"""

import json
import os
from tkinter import ttk

# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------
# Every palette carries the full key set the converter used, so a palette dict
# is a drop-in replacement for the old module-level ``_C_*`` constants.
#   WIN  outer window bg      CARD panel fill        LINE separator/border
#   BNR  banner / heading     PRI  primary button    PRI2 primary hover
#   ACT  secondary button     ACT2 secondary hover   GHO  ghost button fill
#   OK/WRN/ERR/INF semantic text    MUT muted text
#   ODD/EVEN treeview rows     SEL  selection         THDR treeview heading
#   TAB inactive tab           TABA active tab        FONT ui font family

_ACCENTS = {
    "OK":  "#15803D",
    "WRN": "#B45309",
    "ERR": "#B91C1C",
}

PALETTES = {
    # The current Data-Converter look — the default.
    "Ocean": {
        "FONT": "Segoe UI",
        "WIN":  "#EFF4FC", "CARD": "#FFFFFF", "LINE": "#C8D8EE",
        "BNR":  "#1B3A6B", "PRI":  "#1A56DB", "PRI2": "#1446C0",
        "ACT":  "#E8EDF8", "ACT2": "#D0DCF2", "GHO":  "#EFF4FC",
        "INF":  "#1E40AF", "MUT":  "#6B7280",
        "ODD":  "#F5F8FD", "EVEN": "#FFFFFF", "SEL":  "#BFDBFE",
        "THDR": "#DDE8F5", "TAB":  "#D6E4F5", "TABA": "#FFFFFF",
        "ACCENT": "#3B82F6", **_ACCENTS,
    },
    # Neutral graphite / steel.
    "Slate": {
        "FONT": "Segoe UI",
        "WIN":  "#EEF1F4", "CARD": "#FFFFFF", "LINE": "#CBD3DB",
        "BNR":  "#2C3E50", "PRI":  "#3B6EA5", "PRI2": "#335F8F",
        "ACT":  "#E6EBF0", "ACT2": "#D3DBE4", "GHO":  "#EEF1F4",
        "INF":  "#2C5282", "MUT":  "#6B7280",
        "ODD":  "#F4F6F8", "EVEN": "#FFFFFF", "SEL":  "#C6D6E6",
        "THDR": "#DCE3EA", "TAB":  "#D4DCE4", "TABA": "#FFFFFF",
        "ACCENT": "#5B8DC0", **_ACCENTS,
    },
    # Earthy green — fitting for soil spectroscopy.
    "Forest": {
        "FONT": "Segoe UI",
        "WIN":  "#EEF5EE", "CARD": "#FFFFFF", "LINE": "#C4DBC4",
        "BNR":  "#1E4620", "PRI":  "#2E7D32", "PRI2": "#276B2A",
        "ACT":  "#E4F0E4", "ACT2": "#D0E4D0", "GHO":  "#EEF5EE",
        "INF":  "#2E5E2E", "MUT":  "#6B7280",
        "ODD":  "#F4F9F4", "EVEN": "#FFFFFF", "SEL":  "#BFE0C0",
        "THDR": "#D9EAD9", "TAB":  "#D2E6D2", "TABA": "#FFFFFF",
        "ACCENT": "#4CAF50", **_ACCENTS,
    },
    # High-contrast light — maximum readability.
    "Light-Contrast": {
        "FONT": "Segoe UI",
        "WIN":  "#F5F5F5", "CARD": "#FFFFFF", "LINE": "#9AA0A6",
        "BNR":  "#111827", "PRI":  "#1D4ED8", "PRI2": "#163FB0",
        "ACT":  "#E5E7EB", "ACT2": "#D1D5DB", "GHO":  "#F5F5F5",
        "INF":  "#1E3A8A", "MUT":  "#374151",
        "ODD":  "#F3F4F6", "EVEN": "#FFFFFF", "SEL":  "#BFDBFE",
        "THDR": "#E5E7EB", "TAB":  "#D1D5DB", "TABA": "#FFFFFF",
        "ACCENT": "#2563EB", **_ACCENTS,
    },
    # Dark mode.
    "Dark": {
        "FONT": "Segoe UI",
        "WIN":  "#1E232B", "CARD": "#2A303B", "LINE": "#3C4551",
        "BNR":  "#0F1116", "PRI":  "#3B82F6", "PRI2": "#2F6FE0",
        "ACT":  "#333B47", "ACT2": "#3E4855", "GHO":  "#1E232B",
        "INF":  "#93C5FD", "MUT":  "#9AA4B2",
        "ODD":  "#2A303B", "EVEN": "#242A33", "SEL":  "#3B5480",
        "THDR": "#333B47", "TAB":  "#2A303B", "TABA": "#3B82F6",
        "ACCENT": "#60A5FA",
        "OK": "#4ADE80", "WRN": "#FBBF24", "ERR": "#F87171",
    },
}

DEFAULT_THEME = "Ocean"


def list_palettes():
    """Return the ordered list of available palette names."""
    return list(PALETTES.keys())


def get_palette(name=None):
    """Return the palette dict for *name* (falls back to the default)."""
    if name is None:
        name = load_theme_name()
    return dict(PALETTES.get(name, PALETTES[DEFAULT_THEME]))


# ---------------------------------------------------------------------------
# Persistence — a tiny JSON both processes read.
# ---------------------------------------------------------------------------

def _settings_path():
    return os.path.join(os.path.expanduser("~"), ".paracuda", "settings.json")


def load_theme_name():
    """Return the persisted theme name, or the default if none/invalid."""
    try:
        with open(_settings_path(), "r", encoding="utf-8") as fh:
            name = json.load(fh).get("theme")
        if name in PALETTES:
            return name
    except Exception:
        pass
    return DEFAULT_THEME


def save_theme_name(name):
    """Persist *name* as the active theme (best-effort, never raises)."""
    if name not in PALETTES:
        return
    try:
        path = _settings_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
            except Exception:
                data = {}
        data["theme"] = name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The single styling routine (generalised from the converter's _apply_styles)
# ---------------------------------------------------------------------------

def apply_ttk_theme(root, palette=None):
    """Configure all ttk styles on *root* from *palette*.

    This is a direct generalisation of ``ParacudaConverter._apply_styles`` — the
    same style names and mappings, with the hardcoded ``_C_*`` colours replaced
    by palette lookups.  Both the main app and the converter call it so the two
    windows share one visual language.
    """
    p = palette or get_palette()
    C = p.get                      # shorthand: C('WIN'), C('CARD'), ...
    FONT = p.get("FONT", "Segoe UI")
    FN = (FONT, 10)
    FB = (FONT, 10, "bold")
    FH = (FONT, 11, "bold")
    FS = (FONT, 9)

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except Exception:
        pass

    # Propagate default font / bg to plain (non-ttk) widgets.
    root.option_add("*Font", FN)

    # TFrame
    st.configure("TFrame",      background=C("WIN"))
    st.configure("Card.TFrame", background=C("CARD"))

    # TLabel
    st.configure("TLabel",       background=C("WIN"),  font=FN, foreground=C("BNR"))
    st.configure("Card.TLabel",  background=C("CARD"), font=FN, foreground=C("BNR"))
    st.configure("Head.TLabel",  background=C("CARD"), font=FH, foreground=C("BNR"))
    st.configure("Small.TLabel", background=C("WIN"),  font=FS, foreground=C("MUT"))
    st.configure("Stat.TLabel",  background=C("THDR"), font=FS, foreground=C("INF"),
                 padding=(6, 2))

    # TButton
    st.configure("TButton", font=FN, padding=(10, 5))

    st.configure("Primary.TButton", font=FB, padding=(18, 9),
                 background=C("PRI"), foreground="white",
                 bordercolor=C("PRI"), focusthickness=0)
    st.map("Primary.TButton",
           background=[("active", C("PRI2")), ("pressed", C("PRI2")),
                       ("disabled", C("LINE"))],
           foreground=[("disabled", C("MUT"))])

    st.configure("Action.TButton", font=FB, padding=(12, 7),
                 background=C("ACT"), foreground=C("BNR"),
                 bordercolor=C("LINE"), focusthickness=0)
    st.map("Action.TButton",
           background=[("active", C("ACT2")), ("pressed", C("ACT2"))])

    st.configure("Ghost.TButton", font=FN, padding=(6, 4),
                 background=C("GHO"), foreground=C("MUT"),
                 bordercolor=C("GHO"), focusthickness=0)
    st.map("Ghost.TButton", background=[("active", C("LINE"))])

    # TNotebook
    st.configure("TNotebook", background=C("WIN"), borderwidth=0,
                 tabmargins=[2, 4, 0, 0])
    st.configure("TNotebook.Tab", font=FB, padding=(16, 8),
                 background=C("TAB"), foreground=C("MUT"), borderwidth=0)
    st.map("TNotebook.Tab",
           background=[("selected", C("TABA"))],
           foreground=[("selected", C("BNR"))],
           expand=[("selected", [1, 1, 1, 0])])

    # TLabelframe
    st.configure("TLabelframe", background=C("CARD"),
                 bordercolor=C("LINE"), relief="groove", borderwidth=1)
    st.configure("TLabelframe.Label", background=C("CARD"), font=FB,
                 foreground=C("BNR"), padding=(4, 0))

    # Treeview
    st.configure("Treeview", font=FN, rowheight=26,
                 background=C("CARD"), fieldbackground=C("CARD"),
                 foreground=C("BNR"), borderwidth=0, relief="flat")
    st.configure("Treeview.Heading", font=FB,
                 background=C("THDR"), foreground=C("BNR"),
                 relief="flat", padding=(4, 6))
    st.map("Treeview",
           background=[("selected", C("SEL"))],
           foreground=[("selected", C("BNR"))])

    # TEntry / TCombobox
    st.configure("TEntry", font=FN, padding=(5, 4),
                 fieldbackground=C("CARD"), foreground=C("BNR"),
                 bordercolor=C("LINE"))
    st.map("TEntry", bordercolor=[("focus", C("PRI"))])

    st.configure("TCombobox", font=FN, padding=(5, 4),
                 fieldbackground=C("CARD"), foreground=C("BNR"),
                 bordercolor=C("LINE"))
    st.map("TCombobox",
           fieldbackground=[("readonly", C("CARD"))],
           bordercolor=[("focus", C("PRI"))])

    # TCheckbutton / TRadiobutton
    st.configure("TCheckbutton",      background=C("WIN"),  font=FN, foreground=C("BNR"))
    st.configure("TRadiobutton",      background=C("WIN"),  font=FN, foreground=C("BNR"))
    st.configure("Card.TCheckbutton", background=C("CARD"), font=FN, foreground=C("BNR"))
    st.configure("Card.TRadiobutton", background=C("CARD"), font=FN, foreground=C("BNR"))

    # TScrollbar
    st.configure("TScrollbar", background=C("WIN"), troughcolor=C("WIN"),
                 arrowcolor=C("MUT"), borderwidth=0, relief="flat")
    st.map("TScrollbar", background=[("active", C("LINE"))])

    # TSeparator
    st.configure("TSeparator", background=C("LINE"))

    # TProgressbar
    st.configure("TProgressbar", background=C("PRI"), troughcolor=C("ACT"),
                 bordercolor=C("LINE"))

    # TPanedwindow
    st.configure("TPanedwindow", background=C("WIN"))

    return p
