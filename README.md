# PARACUDA-NG

*PARAmetric CUbe Data Analysis, Next Generation*

[![PyPI](https://img.shields.io/pypi/v/paracuda-ng.svg?color=0073b7&label=pypi)](https://pypi.org/project/paracuda-ng/)
[![Downloads](https://img.shields.io/pepy/dt/paracuda-ng?label=downloads&color=1f9d55)](https://pepy.tech/project/paracuda-ng)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/paracuda-ng.svg?color=fb8c00&label=conda-forge)](https://anaconda.org/conda-forge/paracuda-ng)
[![conda downloads](<https://img.shields.io/conda/dn/conda-forge/paracuda-ng.svg?label=conda%20downloads>)](https://anaconda.org/conda-forge/paracuda-ng)
[![QGIS plugin](https://img.shields.io/badge/QGIS-plugin-589632?logo=qgis&logoColor=white)](https://plugins.qgis.org/plugins/paracuda_ng/)
[![Python](https://img.shields.io/pypi/pyversions/paracuda-ng.svg?label=python)](https://pypi.org/project/paracuda-ng/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://paracuda-ng.github.io/)
[![Stars](https://img.shields.io/github/stars/sharadgupta27/paracuda-ng?style=flat&color=blueviolet)](https://github.com/sharadgupta27/paracuda-ng/stargazers)

**Documentation:** [https://paracuda-ng.github.io/](https://paracuda-ng.github.io/)

PARACUDA-NG is an open-source tool for analyzing spectral data and building, validating, and applying machine-learning prediction models - designed for spectroscopy and remote sensing, but usable for any tabular spectral dataset.

The interface is organized as a **7-step wizard** (Data → Configuration → Preprocess → Model → Validate → Execution → Apply) with a live *Model Development Flow* diagram, so a complete workflow - from loading spectra to applying a trained model on a multispectral or hyperspectral image - can be run without writing any code.

![PARACUDA-NG main window](screenshot_paracuda.png)

## Features

### Data Handling

- **Flexible input**: load spectral data from Excel (`.xlsx`, `.xls`, `.ods`) or CSV
- **Data Converter**: a built-in wizard that auto-detects arbitrary instrument layouts
  (row-wise or transposed/column-wise), handles `nm`/`µm` units, optionally merges a
  separate target properties (soil/vegetation) file, and exports to the PARACUDA-NG-compatible format
- **Data checks & statistics**: inspect loaded data and export summary statistics
- **Multi-property selection**: analyze several properties in a single run

### Spectral Processing

- **Preprocessing**: Continuum Removal, First Derivative, Second Derivative, Absorbance
  (plus optional PCA dimensionality reduction in the model pipeline)
- **Band resampling (7 methods)**: Linear, Nearest-Neighbour, Quadratic and Cubic-Spline
  interpolation, Gaussian SRF convolution, Empirical SRF integration, and Band Averaging -
  for harmonizing spectra to a target sensor's band configuration
- **Wavelength exclusion**: drop noisy regions with custom ranges or one-click presets
  (water-absorption bands, noisy detector edges)
- **Custom band definitions**: upload per-band FWHM or full Spectral Response Function
  (SRF) tables for bandwidth-aware resampling

### Modeling

- **Regression models**: PLS-R, SVM, Ridge, Lasso, Multiple Linear Regression, Elastic Net,
  Huber Regressor, Gradient Boosting, Gaussian Process, Random Forest, and XGBoost
- **Compositional modeling**: ALR / CLR / ILR log-ratio transforms so predictions of
  sum-constrained parts (e.g. sand + silt + clay) add up to 100%
- **Hyperparameter tuning**: automated optimization (Optuna) for supported models
- **Cross-validation**: K-Fold, Leave-One-Out, and Leave-P-Out strategies

### Batch Mode

- **Multi-model comparison**: train and compare multiple models automatically
- **Best-fit recommendation**: suggests the best model per property from R², RMSE, and
  cross-validation metrics
- **Automated plots**: observed-vs-predicted scatter plots, reflectance-spectra plots, and
  feature-importance charts (importance bars for tree models; wavelength-correlation plots
  for others)
- **Comprehensive reporting**: Excel workbooks with per-model comparison sheets plus
  multi-page PDF plot reports (model file, Excel report, and PDF share one run timestamp)

### Apply to Images

- **Multispectral & hyperspectral image prediction**: apply a trained model across a full image cube
- **Multi-format geospatial I/O**: GeoTIFF/TIFF, ENVI (`.dat`/`.bil`/`.bip`/`.bsq` + `.hdr`),
  ERDAS Imagine (`.img`), NITF, and PCIDSK - wavelengths/FWHM are recovered from ENVI headers,
  and predictions are written back in the same format with georeferencing preserved
- **Robust handling**: automatic gain/offset scaling, background/no-data and bad-band masking,
  and memory-safe chunked processing for large scenes
- **Resampled-cube export**: optionally export the resampled image (or tabular spectra) to
  verify the harmonization step

### Interface

- **Themes**: Ocean, Slate, Forest, Light-Contrast, and Dark (View → Theme)
- **Model Development Flow**: a live, themed pipeline diagram that reflects your current
  settings and can be exported at 300 DPI
- **Built-in Help Assistant**: searchable in-app guidance (see `HELP_SYSTEM_README.md`)

## Data Converter

Not every instrument exports data in the layout PARACUDA-NG expects. The **PARACUDA-NG Data
Converter** (Tools → Data Converter, or run `utils/data_converter.py`) is a three-step
wizard - *Load Data → Configure → Preview & Export* - that auto-detects the layout of an
arbitrary Excel/CSV file and converts it to the standard
`Names | Prop1 … PropN | WL1 … WLM` format.

![PARACUDA-NG Data Converter](screenshot_data_converter.png)

## Project Structure

PARACUDA-NG is organized into focused packages:

| Path                  | Contents                                                     |
| --------------------- | ------------------------------------------------------------ |
| `paracuda.py`       | Launcher entry point                                         |
| `gui/`              | Tkinter application, split into composable mixins            |
| `preprocessing/`    | Spectral preprocessing, resampling, compositional transforms |
| `models/`           | Model training, hyperparameter tuning, batch processing      |
| `validation/`       | Cross-validation                                             |
| `utils/`            | File I/O, data converter, image processing, help assistant   |
| `paracuda_theme.py` | Shared color themes                                          |

## Quick Start (One-Click Launcher)

If you're not comfortable installing packages or running scripts from a terminal, just
use the included launcher - **double-click `run_paracuda.bat`**. It automatically:

1. Finds your Conda installation (Miniforge, Miniconda, or Anaconda) and creates the
   `paracuda` environment in it
2. If there is no Conda, falls back to any Python 3.9+ already on the computer and builds
   a local `.venv` next to the launcher instead
3. If there is no Python at all, downloads the official python.org installer and installs
   it for the current user only (no administrator rights, into
   `%LOCALAPPDATA%\Programs\PARACUDA-Python`), then builds the `.venv`
4. Installs everything from `requirements.txt` the first time you run it (this one-time
   setup takes a few minutes)
5. Launches PARACUDA-NG

There is no prerequisite - a machine with nothing installed works. The only thing needed
for the first run is an internet connection. If setup fails, the launcher prints clear
guidance and stays open so you can read the message.

## Installation (Manual)

The tool uses several Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage (Manual)

Run the tool from the repository root:

```bash
python paracuda.py
```

A typical batch workflow:

1. **Data** - load spectral data and select one or more properties
2. **Configuration / Preprocess** - choose resampling, exclusions, and preprocessing
3. **Model / Validate** - pick single or multiple models and a cross-validation strategy
4. **Execution** - run the analysis to get an Excel report, PDF plots, and best-model
   recommendations, then save the trained model
5. **Apply** - load a multispectral or hyperspectral image, or a new dataset, and predict with the saved model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```
@software{ParacudaNG2026,
author = {Sharad Kumar Gupta},
title = {PARACUDA-NG: An Open-Source Machine Learning tool for Spectroscopic Analysis},
year = {2026},
url = {https://github.com/sharadgupta27/paracuda-ng}
}
```

## Acknowledgement

PARACUDA-NG takes its inspiration from the PARACUDA tool (based on Excel)
developed by Schwartz et al. (2013) and PARACUDA II (MATLAB) by Carmon and
Ben-Dor (2017).

1. Schwartz, G. (2013). *Reflectance Spectroscopy as a Rapid Tool for
   Quantitative Mapping of Hydrocarbons Soil Contamination* (PhD dissertation).
   The Porter School of Environmental Studies, Tel Aviv University. Supervised
   by Prof. Eyal Ben-Dor and Dr. Gil Eshel.
2. Carmon, N., and Ben-Dor, E. (2017). An advanced analytical approach for
   spectral-based modelling of soil properties. *International Journal of
   Emerging Technologies and Advanced Engineering*, 7, 90-97.

## Commercial Use

The code is MIT-licensed, so commercial use is permitted - keeping the copyright
notice is all that is required. For collaboration or support enquiries, contact:
sharadgupta27@gmail.comFor commercial licensing inquiries, please contact: sharadgupta27@gmail.c
