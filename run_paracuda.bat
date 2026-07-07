@echo off
REM ====================================================================
REM Paracuda III - One-Click Launcher
REM ====================================================================
REM This batch file automatically finds and activates the conda 
REM environment, then runs Paracuda III spectral analysis tool.
REM ====================================================================

title Paracuda III Launcher
echo.
echo ========================================
echo   Paracuda III - Launcher
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
REM ---------------------------------------------------------------------

echo [1/4] Locating Conda installation...

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

REM If not found, show error
echo.
echo ERROR: Could not find Conda installation!
echo.
echo Please make sure Anaconda, Miniconda, or Miniforge is installed.
echo Common installation locations checked:
echo   - %USERPROFILE%\miniforge3
echo   - %USERPROFILE%\miniconda3
echo   - %USERPROFILE%\anaconda3
echo   - C:\ProgramData\anaconda3
echo   - C:\ProgramData\miniconda3
echo   - %LOCALAPPDATA%\miniforge3
echo   - %LOCALAPPDATA%\miniconda3
echo   - %LOCALAPPDATA%\anaconda3
echo.
echo Press any key to exit...
pause > nul
exit /b 1

:found_conda
echo.
echo [2/4] Preparing '%ENV_NAME%' environment...

REM Derive the Conda root from the activate.bat path (<root>\Scripts\activate.bat)
REM and build the path to this environment's folder.
set "CONDA_ROOT=%CONDA_BAT:\Scripts\activate.bat=%"
set "ENV_DIR=%CONDA_ROOT%\envs\%ENV_NAME%"

REM Make the 'conda' command available (activate base only).  We deliberately do
REM NOT use 'conda activate <env>' / 'activate.bat <env>' below: under Miniforge
REM that routes through mamba, which fails unless the shell was 'conda init'-ed.
REM Instead we run the environment's python.exe directly and put its DLL folders
REM on PATH, which works without any shell initialization.
call "%CONDA_BAT%"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to initialize Conda!
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
    call conda create -n %ENV_NAME% -c conda-forge python=%PY_VERSION% -y
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

REM Verify the required packages are actually installed (this also repairs an
REM environment that was created but never finished installing).  If the import
REM check fails, (re)install everything from requirements.txt.
"%ENV_DIR%\python.exe" -c "import numpy, pandas, sklearn, rasterio, xgboost" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing required packages from requirements.txt...
    echo This may take several minutes...
    echo.
    "%ENV_DIR%\python.exe" -m pip install --upgrade pip
    "%ENV_DIR%\python.exe" -m pip install -r requirements.txt
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
    echo Setup complete - '%ENV_NAME%' environment is ready.
) else (
    echo Required packages are already installed.
)

REM Put the environment and its MKL / GDAL DLL directories on PATH so LAPACK
REM (numpy/scikit-learn) and rasterio load correctly without full activation.
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Scripts;%PATH%"
set "GDAL_DATA=%ENV_DIR%\Library\share\gdal"
set "PROJ_LIB=%ENV_DIR%\Library\share\proj"

echo Environment ready: %ENV_DIR%
echo.

echo [3/4] Setting working directory...
echo Working directory: %SCRIPT_DIR%
echo.

echo [4/4] Launching Paracuda III...
echo.
echo ========================================
echo   Starting Paracuda III...
echo ========================================
echo.

REM Run the app with this environment's interpreter directly.
"%ENV_DIR%\python.exe" paracuda.py

REM Check if the script ran successfully
if errorlevel 1 (
    echo.
    echo ========================================
    echo   Paracuda III closed with an error
    echo ========================================
    echo.
    echo Press any key to close this window...
    pause > nul
    exit /b 1
)

echo.
echo ========================================
echo   Paracuda III closed successfully
echo ========================================
echo.
exit /b 0
