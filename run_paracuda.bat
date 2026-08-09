@echo off
REM Plain setlocal (NOT enabledelayedexpansion): nothing here needs !var!
REM expansion, and delayed expansion would swallow the '!' in messages.
setlocal
REM ====================================================================
REM PARACUDA-NG - One-Click Launcher
REM ====================================================================
REM This batch file gets PARACUDA-NG running on a machine that has
REM nothing installed:
REM
REM   1. If Conda (Miniforge / Miniconda / Anaconda) is present, it
REM      creates/uses the 'paracuda' Conda environment.
REM   2. Otherwise it looks for any usable system Python 3.9+.
REM   3. If there is no Python at all, it downloads the official
REM      python.org installer, installs it per-user (no admin rights),
REM      and continues.
REM   4. It then builds a local .venv, installs requirements.txt into
REM      it, and launches the tool.
REM ====================================================================

title PARACUDA-NG Launcher
echo.
echo ========================================
echo   PARACUDA-NG - Launcher
echo ========================================
echo.

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ---- Configuration --------------------------------------------------
REM Name of the Conda environment to create / use, and the Python version
REM used when creating it.  Change these two lines if you want a different
REM environment name or Python version.
set "ENV_NAME=paracuda"
set "PY_VERSION=3.14"

REM Used only when NO Conda is available: the local virtual environment
REM folder, plus the Python version/location installed when the machine
REM has no Python at all.
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PY_INSTALL_VER=3.12.10"
set "PY_TARGET=%LOCALAPPDATA%\Programs\PARACUDA-Python"
REM ---------------------------------------------------------------------

echo [1/4] Locating a Python installation...

REM Search for conda in common locations
set "CONDA_BAT="

REM Check for Miniforge (most common in your setup)
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniforge3\Scripts\activate.bat"
    echo Found: Miniforge in %USERPROFILE%\miniforge3
    goto :found_conda
)

REM Check for Miniconda in user profile
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
    echo Found: Miniconda in %USERPROFILE%\miniconda3
    goto :found_conda
)

REM Check for Anaconda in user profile
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
    echo Found: Anaconda in %USERPROFILE%\anaconda3
    goto :found_conda
)

REM Check for Anaconda in ProgramData
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=C:\ProgramData\anaconda3\Scripts\activate.bat"
    echo Found: Anaconda in C:\ProgramData\anaconda3
    goto :found_conda
)

REM Check for Miniconda in ProgramData
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=C:\ProgramData\miniconda3\Scripts\activate.bat"
    echo Found: Miniconda in C:\ProgramData\miniconda3
    goto :found_conda
)

REM Check in AppData\Local
if exist "%LOCALAPPDATA%\miniforge3\Scripts\activate.bat" (
    set "CONDA_BAT=%LOCALAPPDATA%\miniforge3\Scripts\activate.bat"
    echo Found: Miniforge in %LOCALAPPDATA%\miniforge3
    goto :found_conda
)

if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
    echo Found: Miniconda in %LOCALAPPDATA%\miniconda3
    goto :found_conda
)

if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"
    echo Found: Anaconda in %LOCALAPPDATA%\anaconda3
    goto :found_conda
)

REM No Conda anywhere - fall back to a plain Python + virtual environment.
echo No Conda installation found - falling back to a standard Python setup.
goto :no_conda


REM =====================================================================
REM  PATH A - Conda is available
REM =====================================================================
:found_conda
echo.
echo [2/4] Preparing '%ENV_NAME%' Conda environment...

REM Derive the Conda root from the activate.bat path (<root>\Scripts\activate.bat)
REM and build the path to this environment's folder.
set "CONDA_ROOT=%CONDA_BAT:\Scripts\activate.bat=%"
set "ENV_DIR=%CONDA_ROOT%\envs\%ENV_NAME%"

REM Locate a conda launcher we can call WITHOUT initializing the shell.  We must
REM NOT call Scripts\activate.bat here: on current Miniforge/mamba it routes
REM through libmamba and aborts with "critical libmamba Shell not initialized".
REM 'conda' is only needed for the one-time 'conda create' below; the app itself
REM runs the environment's python.exe directly (DLL folders are added to PATH
REM later), which needs no activation.  Prefer the real conda.exe; fall back to
REM condabin\conda.bat.
set "CONDA_EXE=%CONDA_ROOT%\Scripts\conda.exe"
if not exist "%CONDA_EXE%" set "CONDA_EXE=%CONDA_ROOT%\condabin\conda.bat"
if not exist "%CONDA_EXE%" (
    echo.
    echo ERROR: Could not find the 'conda' executable under %CONDA_ROOT%.
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

REM Create the environment if its python.exe is not present yet.
if not exist "%ENV_DIR%\python.exe" (
    echo '%ENV_NAME%' environment not found - creating it now.
    echo This is a one-time setup and may take several minutes...
    echo.
    call "%CONDA_EXE%" create -n %ENV_NAME% -c conda-forge python=%PY_VERSION% -y
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create the '%ENV_NAME%' environment!
        echo Please check your internet connection and Conda installation.
        echo.
        echo Press any key to exit...
        pause > nul
        exit /b 1
    )
) else (
    echo Found existing '%ENV_NAME%' environment.
)

REM Put the environment and its MKL / GDAL DLL directories on PATH so LAPACK
REM (numpy/scikit-learn) and rasterio load correctly without full activation.
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Scripts;%PATH%"
set "GDAL_DATA=%ENV_DIR%\Library\share\gdal"
set "PROJ_LIB=%ENV_DIR%\Library\share\proj"

set "PY_EXE=%ENV_DIR%\python.exe"
goto :env_ready


REM =====================================================================
REM  PATH B - No Conda: use / install a plain Python and build a .venv
REM =====================================================================
:no_conda
echo.
echo [2/4] Preparing a local virtual environment...

REM Re-use the .venv from a previous run if it is already there.
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Found existing virtual environment: %VENV_DIR%
    goto :venv_ready
)

REM Find any usable Python 3.9+ that also has tkinter.
call :find_system_python
if not defined SYS_PY (
    echo No suitable Python 3.9+ installation was found on this computer.
    call :install_python
    if not defined SYS_PY (
        echo.
        echo ERROR: Python could not be installed automatically.
        echo Please install Python 3.9 or newer from https://www.python.org/downloads/
        echo and make sure to tick "tcl/tk and IDLE" during the installation.
        echo.
        echo Press any key to exit...
        pause > nul
        exit /b 1
    )
)

echo Using Python: %SYS_PY%
echo Creating virtual environment in %VENV_DIR%
echo This is a one-time setup and may take a minute...
echo.
"%SYS_PY%" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create the virtual environment!
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ERROR: The virtual environment was not created correctly.
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:venv_ready
set "PATH=%VENV_DIR%\Scripts;%VENV_DIR%;%PATH%"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
goto :env_ready


REM =====================================================================
REM  Shared - make sure the packages are there, then launch
REM =====================================================================
:env_ready

REM Verify the required packages are actually installed (this also repairs an
REM environment that was created but never finished installing).  If the import
REM check fails, (re)install everything from requirements.txt.
"%PY_EXE%" -c "import numpy, pandas, sklearn, rasterio, xgboost" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing required packages from requirements.txt...
    echo This may take several minutes...
    echo.
    "%PY_EXE%" -m pip install --upgrade pip
    "%PY_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install required packages!
        echo Please check your internet connection and try again.
        echo.
        echo Press any key to exit...
        pause > nul
        exit /b 1
    )
    echo.
    echo Setup complete - the environment is ready.
) else (
    echo Required packages are already installed.
)

echo Environment ready: %PY_EXE%
echo.

echo [3/4] Setting working directory...
echo Working directory: %SCRIPT_DIR%
echo.

echo [4/4] Launching PARACUDA-NG...
echo.
echo ========================================
echo   Starting PARACUDA-NG...
echo ========================================
echo.

REM Run the app with this environment's interpreter directly.
"%PY_EXE%" paracuda.py

REM Check if the script ran successfully
if errorlevel 1 (
    echo.
    echo ========================================
    echo   PARACUDA-NG closed with an error
    echo ========================================
    echo.
    echo Press any key to close this window...
    pause > nul
    exit /b 1
)

echo.
echo ========================================
echo   PARACUDA-NG closed successfully
echo ========================================
echo.
exit /b 0


REM =====================================================================
REM  Subroutines
REM =====================================================================

REM ---------------------------------------------------------------------
REM :validate_py <path-to-python.exe>
REM Sets PY_OK=1 when the interpreter exists, is 3.9+ and has tkinter.
REM ---------------------------------------------------------------------
:validate_py
set "PY_OK="
if "%~1"=="" goto :eof
if not exist "%~1" goto :eof
REM Skip the Microsoft Store "app execution alias" stub - it is not a real Python.
echo %~1 | find /i "WindowsApps" >nul && goto :eof
"%~1" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PY_OK=1"
goto :eof

REM ---------------------------------------------------------------------
REM :find_system_python
REM Sets SYS_PY to the first usable interpreter found, or leaves it empty.
REM ---------------------------------------------------------------------
:find_system_python
set "SYS_PY="

REM 1) A Python this launcher installed on an earlier run.
call :validate_py "%PY_TARGET%\python.exe"
if defined PY_OK (
    set "SYS_PY=%PY_TARGET%\python.exe"
    goto :eof
)

REM 2) The 'py' launcher - it knows about every registered installation.
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if not defined SYS_PY (
            call :validate_py "%%i"
            if defined PY_OK set "SYS_PY=%%i"
        )
    )
)
if defined SYS_PY goto :eof

REM 3) Anything called python / python3 on PATH.
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined SYS_PY (
        call :validate_py "%%i"
        if defined PY_OK set "SYS_PY=%%i"
    )
)
if defined SYS_PY goto :eof

for /f "delims=" %%i in ('where python3 2^>nul') do (
    if not defined SYS_PY (
        call :validate_py "%%i"
        if defined PY_OK set "SYS_PY=%%i"
    )
)
if defined SYS_PY goto :eof

REM 4) The standard python.org install locations, per-user and machine-wide.
call :scan_python_dir "%LOCALAPPDATA%\Programs\Python\Python3*"
if defined SYS_PY goto :eof
call :scan_python_dir "%ProgramFiles%\Python3*"
if defined SYS_PY goto :eof
call :scan_python_dir "%ProgramFiles(x86)%\Python3*"
if defined SYS_PY goto :eof
call :scan_python_dir "C:\Python3*"
goto :eof

REM ---------------------------------------------------------------------
REM :scan_python_dir <folder-glob>
REM Looks for <folder>\python.exe in every directory matching the glob.
REM ---------------------------------------------------------------------
:scan_python_dir
if "%~1"=="" goto :eof
for /d %%d in ("%~1") do (
    if not defined SYS_PY (
        call :validate_py "%%~fd\python.exe"
        if defined PY_OK set "SYS_PY=%%~fd\python.exe"
    )
)
goto :eof

REM ---------------------------------------------------------------------
REM :install_python
REM Downloads the official python.org installer and installs it for the
REM current user only - no administrator rights needed.  Sets SYS_PY.
REM ---------------------------------------------------------------------
:install_python
set "SYS_PY="

set "PY_ARCH=amd64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "PY_ARCH=arm64"
if /i "%PROCESSOR_ARCHITECTURE%"=="x86" if not defined PROCESSOR_ARCHITEW6432 set "PY_ARCH=win32"

set "PY_URL=https://www.python.org/ftp/python/%PY_INSTALL_VER%/python-%PY_INSTALL_VER%-%PY_ARCH%.exe"
if /i "%PY_ARCH%"=="win32" set "PY_URL=https://www.python.org/ftp/python/%PY_INSTALL_VER%/python-%PY_INSTALL_VER%.exe"
set "PY_INSTALLER=%TEMP%\python-%PY_INSTALL_VER%-%PY_ARCH%-paracuda.exe"

echo.
echo ---------------------------------------------------------------
echo   PARACUDA-NG needs Python %PY_INSTALL_VER% to run.
echo   It will be downloaded from python.org and installed for the
echo   current user only, into:
echo     %PY_TARGET%
echo   No administrator rights are needed and nothing else on this
echo   computer is changed.
echo ---------------------------------------------------------------
echo.

set "DO_INSTALL=Y"
where choice >nul 2>&1
if not errorlevel 1 (
    choice /C YN /N /T 20 /D Y /M "Download and install Python now? [Y/n] "
    if errorlevel 2 set "DO_INSTALL=N"
)
if /i "%DO_INSTALL%"=="N" (
    echo Installation cancelled by the user.
    goto :eof
)

echo.
echo Downloading %PY_URL%
if exist "%PY_INSTALLER%" del /f /q "%PY_INSTALLER%" >nul 2>&1

where curl >nul 2>&1
if not errorlevel 1 (
    curl -L --fail --progress-bar -o "%PY_INSTALLER%" "%PY_URL%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'"
)

if not exist "%PY_INSTALLER%" (
    echo.
    echo ERROR: Could not download the Python installer.
    echo Please check your internet connection or proxy settings.
    goto :eof
)

echo.
echo Installing Python %PY_INSTALL_VER% - please wait, this takes a few minutes...
start /wait "" "%PY_INSTALLER%" /passive InstallAllUsers=0 PrependPath=0 Include_launcher=1 InstallLauncherAllUsers=0 Include_tcltk=1 Include_pip=1 Include_test=0 Include_doc=0 SimpleInstall=1 TargetDir="%PY_TARGET%"
if errorlevel 1 (
    echo.
    echo WARNING: The Python installer reported an error - checking anyway...
)

del /f /q "%PY_INSTALLER%" >nul 2>&1

call :validate_py "%PY_TARGET%\python.exe"
if defined PY_OK (
    set "SYS_PY=%PY_TARGET%\python.exe"
    echo Python installed successfully.
    goto :eof
)

REM The installer may have landed somewhere else - do one more sweep.
call :find_system_python
goto :eof
