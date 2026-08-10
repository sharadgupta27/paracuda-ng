#!/usr/bin/env bash
# =====================================================================
#  PARACUDA-NG - One-Click Launcher (Linux / macOS)
# =====================================================================
#  The counterpart of run_paracuda.bat.  It gets PARACUDA-NG running on
#  a machine that has nothing installed:
#
#    1. If Conda (Miniforge / Miniconda / Anaconda) is present, it
#       creates/uses the 'paracuda' Conda environment.
#    2. Otherwise it looks for any usable system Python 3.10+ that has
#       tkinter and venv, and builds a local .venv.
#    3. If there is no usable Python at all, it downloads Miniforge and
#       installs it into the home directory - no root, no sudo, nothing
#       outside $HOME is touched.
#    4. It then installs requirements.txt and launches the tool.
#
#  Usage:  ./run_paracuda.sh          (chmod +x run_paracuda.sh first)
# =====================================================================
set -uo pipefail

echo
echo "========================================"
echo "  PARACUDA-NG - Launcher"
echo "========================================"
echo

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ---- Configuration --------------------------------------------------
# Both setup paths target the same Python series, 3.14, exactly as the
# Windows launcher does, so every machine runs on the same interpreter.
# The supported floor is 3.10 - older releases are rejected because
# several dependencies (greenlet, via optuna -> SQLAlchemy) no longer
# publish 3.9 wheels and would have to be compiled from source.
ENV_NAME="paracuda"
PY_VERSION="3.14"
PY_MIN_MAJOR=3
PY_MIN_MINOR=10

VENV_DIR="$SCRIPT_DIR/.venv"
# Where Miniforge is installed when the machine has no usable Python.
CONDA_TARGET="$HOME/miniforge3"
# ---------------------------------------------------------------------

PY_EXE=""

# Pause before exiting so a double-click from a file manager does not
# close the window before the message can be read.
die() {
    echo
    echo "ERROR: $*" >&2
    echo
    if [ -t 0 ]; then
        printf 'Press Enter to exit...'
        read -r _ || true
    fi
    exit 1
}

# ---------------------------------------------------------------------
# validate_py <path-to-python>
# Succeeds when the interpreter is PY_MIN+ and can import tkinter.
# ---------------------------------------------------------------------
validate_py() {
    [ -n "${1:-}" ] || return 1
    command -v "$1" >/dev/null 2>&1 || [ -x "$1" ] || return 1
    "$1" - "$PY_MIN_MAJOR" "$PY_MIN_MINOR" >/dev/null 2>&1 <<'PYEOF'
import sys
import tkinter  # noqa: F401  - the GUI toolkit, must be importable
major, minor = int(sys.argv[1]), int(sys.argv[2])
sys.exit(0 if sys.version_info >= (major, minor) else 1)
PYEOF
}

# Same check without tkinter, so we can tell "too old" apart from
# "modern enough but missing the python3-tk package".
has_min_version() {
    [ -n "${1:-}" ] || return 1
    "$1" - "$PY_MIN_MAJOR" "$PY_MIN_MINOR" >/dev/null 2>&1 <<'PYEOF'
import sys
major, minor = int(sys.argv[1]), int(sys.argv[2])
sys.exit(0 if sys.version_info >= (major, minor) else 1)
PYEOF
}

echo "[1/4] Locating a Python installation..."

# ---------------------------------------------------------------------
# PATH A - Conda is available
# ---------------------------------------------------------------------
CONDA_ROOT=""
for candidate in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
                 "$HOME/anaconda3" "/opt/miniforge3" "/opt/miniconda3" \
                 "/opt/anaconda3" "/opt/conda"; do
    if [ -x "$candidate/bin/conda" ]; then
        CONDA_ROOT="$candidate"
        echo "Found: Conda in $CONDA_ROOT"
        break
    fi
done

# A conda that is only on PATH (or installed somewhere unusual) still counts.
if [ -z "$CONDA_ROOT" ] && command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
    [ -x "${CONDA_ROOT:-/nonexistent}/bin/conda" ] || CONDA_ROOT=""
    [ -n "$CONDA_ROOT" ] && echo "Found: Conda in $CONDA_ROOT"
fi

setup_conda_env() {
    echo
    echo "[2/4] Preparing '$ENV_NAME' Conda environment..."
    local env_dir="$CONDA_ROOT/envs/$ENV_NAME"

    # The environment's python is run directly, exactly as on Windows -
    # no 'conda activate', which needs an initialised shell.
    if [ ! -x "$env_dir/bin/python" ]; then
        echo "'$ENV_NAME' environment not found - creating it now."
        echo "This is a one-time setup and may take several minutes..."
        echo
        "$CONDA_ROOT/bin/conda" create -n "$ENV_NAME" -c conda-forge \
            "python=$PY_VERSION" -y \
            || die "Failed to create the '$ENV_NAME' environment! Please check your internet connection and Conda installation."
    else
        echo "Found existing '$ENV_NAME' environment."
    fi

    PY_EXE="$env_dir/bin/python"
}

# ---------------------------------------------------------------------
# PATH B - No Conda: use / install a plain Python and build a .venv
# ---------------------------------------------------------------------
find_system_python() {
    local candidate
    # Newest first, so a machine with several Pythons uses the best one.
    for candidate in python3.15 python3.14 python3.13 python3.12 python3.11 \
                     python3.10 python3 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if validate_py "$candidate"; then
            SYS_PY="$(command -v "$candidate")"
            return 0
        fi
        # Modern enough, but tkinter is missing: on Debian/Ubuntu that is
        # just the python3-tk package, so say so instead of silently
        # falling through to a 400 MB Miniforge download.
        if [ -z "${TK_HINT:-}" ] && has_min_version "$candidate"; then
            TK_HINT="$(command -v "$candidate")"
        fi
    done
    return 1
}

install_miniforge() {
    local os arch url installer
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os" in
        Linux)  os="Linux" ;;
        Darwin) os="MacOSX" ;;
        *) die "Unsupported operating system: $os. Please install Python $PY_MIN_MAJOR.$PY_MIN_MINOR or newer manually." ;;
    esac
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${os}-${arch}.sh"
    installer="${TMPDIR:-/tmp}/Miniforge3-paracuda-$$.sh"

    echo
    echo "---------------------------------------------------------------"
    echo "  PARACUDA-NG needs Python $PY_MIN_MAJOR.$PY_MIN_MINOR or newer."
    echo "  Miniforge (Conda) will be downloaded and installed for the"
    echo "  current user only, into:"
    echo "    $CONDA_TARGET"
    echo "  No root access is needed and nothing outside your home"
    echo "  directory is changed."
    echo "---------------------------------------------------------------"
    echo

    if [ -t 0 ]; then
        printf 'Download and install Miniforge now? [Y/n] '
        read -r reply || reply="Y"
        case "${reply:-Y}" in
            [Nn]*) die "Installation cancelled by the user." ;;
        esac
    fi

    echo
    echo "Downloading $url"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "$installer" "$url" \
            || die "Could not download Miniforge. Please check your internet connection or proxy settings."
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$installer" "$url" \
            || die "Could not download Miniforge. Please check your internet connection or proxy settings."
    else
        die "Neither curl nor wget is available, so Miniforge cannot be downloaded. Install one of them, or install Python $PY_MIN_MAJOR.$PY_MIN_MINOR+ yourself."
    fi

    echo
    echo "Installing Miniforge - please wait, this takes a few minutes..."
    bash "$installer" -b -p "$CONDA_TARGET" || {
        rm -f "$installer"
        die "The Miniforge installer failed."
    }
    rm -f "$installer"

    [ -x "$CONDA_TARGET/bin/conda" ] \
        || die "Miniforge did not install correctly."
    CONDA_ROOT="$CONDA_TARGET"
    echo "Miniforge installed successfully."
}

setup_venv() {
    echo
    echo "[2/4] Preparing a local virtual environment..."

    # Re-use the .venv from a previous run, but only when it was built on
    # a Python we still support - an old .venv may sit on 3.9, where pip
    # has to compile packages from source.
    if [ -x "$VENV_DIR/bin/python" ]; then
        if validate_py "$VENV_DIR/bin/python"; then
            echo "Found existing virtual environment: $VENV_DIR"
            PY_EXE="$VENV_DIR/bin/python"
            return 0
        fi
        echo "The existing virtual environment uses an unsupported Python - rebuilding it."
        rm -rf "$VENV_DIR"
    fi

    echo "Using Python: $SYS_PY"
    echo "Creating virtual environment in $VENV_DIR"
    echo "This is a one-time setup and may take a minute..."
    echo
    if ! "$SYS_PY" -m venv "$VENV_DIR"; then
        # Debian and Ubuntu ship venv in a separate package.
        die "Failed to create the virtual environment!
On Debian or Ubuntu install the venv package first:
    sudo apt install python3-venv
then run this launcher again."
    fi
    [ -x "$VENV_DIR/bin/python" ] \
        || die "The virtual environment was not created correctly."
    PY_EXE="$VENV_DIR/bin/python"
}

if [ -n "$CONDA_ROOT" ]; then
    setup_conda_env
else
    echo "No Conda installation found - looking for a standard Python setup."
    SYS_PY=""
    TK_HINT=""
    if find_system_python; then
        setup_venv
    else
        if [ -n "$TK_HINT" ]; then
            echo
            echo "Found $TK_HINT, but it cannot import tkinter, which"
            echo "PARACUDA-NG's interface is built on.  On Debian or Ubuntu the"
            echo "quickest fix is:"
            echo
            echo "    sudo apt install python3-tk"
            echo
            echo "Then run this launcher again.  Alternatively, continue below"
            echo "and Miniforge will be installed into your home directory with"
            echo "its own copy of Python and tkinter."
        else
            echo "No suitable Python $PY_MIN_MAJOR.$PY_MIN_MINOR+ installation was found on this computer."
        fi
        install_miniforge
        setup_conda_env
    fi
fi

[ -n "$PY_EXE" ] && [ -x "$PY_EXE" ] \
    || die "No usable Python interpreter could be prepared."

# =====================================================================
#  Shared - make sure the packages are there, then launch
# =====================================================================

# Verify the required packages are actually installed (this also repairs
# an environment that was created but never finished installing).
if ! "$PY_EXE" -c "import numpy, pandas, sklearn, rasterio, xgboost" >/dev/null 2>&1; then
    echo
    echo "Installing required packages from requirements.txt..."
    echo "This may take several minutes..."
    echo
    "$PY_EXE" -m pip install --upgrade pip
    # --prefer-binary: take a slightly older release that ships a wheel
    # rather than compile the newest one from source, which would need a
    # C/C++ toolchain the user is not expected to have.
    if ! "$PY_EXE" -m pip install --prefer-binary -r requirements.txt; then
        die "Failed to install required packages!
Please check your internet connection and try again.

If the messages above mention a missing compiler, Python.h or gcc, pip had
to build a package from source because no ready-made wheel matched this
Python.  Delete the .venv folder next to this file and run the launcher
again - it will rebuild on a supported Python version."
    fi
    echo
    echo "Setup complete - the environment is ready."
else
    echo "Required packages are already installed."
fi

echo "Environment ready: $PY_EXE"
echo
echo "[3/4] Setting working directory..."
echo "Working directory: $SCRIPT_DIR"
echo
echo "[4/4] Launching PARACUDA-NG..."
echo
echo "========================================"
echo "  Starting PARACUDA-NG..."
echo "========================================"
echo

"$PY_EXE" paracuda.py
STATUS=$?

echo
if [ "$STATUS" -ne 0 ]; then
    echo "========================================"
    echo "  PARACUDA-NG closed with an error"
    echo "========================================"
    echo
    if [ -t 0 ]; then
        printf 'Press Enter to close this window...'
        read -r _ || true
    fi
    exit "$STATUS"
fi

echo "========================================"
echo "  PARACUDA-NG closed successfully"
echo "========================================"
echo
exit 0
