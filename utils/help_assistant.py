"""
Paracuda III - Intelligent Help Assistant
A context-aware help system that provides user-friendly guidance based on tool functionality.

@author: Sharad Kumar Gupta
"""

import re
from difflib import SequenceMatcher

class HelpAssistant:
    """Intelligent help system for Paracuda III"""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self):
        """Build comprehensive knowledge base from tool functionality"""
        return {
            # Getting Started
            "getting_started": {
                "keywords": ["start", "begin", "first time", "new", "how to use", "tutorial", "basics"],
                "title": "Getting Started with Paracuda III",
                "content": """
**Welcome to Paracuda III!** Here's how to get started:

1. **Load Data**: Click 'Browse' under Input File to select your spectral data (Excel or CSV)
   - Data should have wavelengths as columns and samples as rows
   - Include soil property columns (e.g., Clay, Sand, SOC)

2. **Select Property**: Choose the soil property you want to predict from the dropdown

3. **Choose Model**: Select a regression model (PLS-R, SVM, Random Forest, etc.)

4. **Configure**: Set preprocessing options and model parameters as needed

5. **Train**: Click 'Train Model' to start analysis

6. **Review Results**: Check R², RMSE, and cross-validation scores

7. **Save**: Save your results or trained model for later use
"""
            },
            
            # Data Loading
            "data_loading": {
                "keywords": ["load data", "import", "file", "excel", "csv", "input", "browse", "data format"],
                "title": "Loading Spectral Data",
                "content": """
**How to Load Your Data:**

**Supported Formats:**
- Excel files (.xlsx, .xls)
- CSV files (.csv)

**Data Structure Required:**
- Columns: Wavelengths (e.g., 350, 351, 352...) and soil properties (e.g., Clay, Sand, SOC)
- Rows: Individual samples/observations
- Wavelengths should be numeric column headers
- Property columns should have descriptive names

**Steps:**
1. Click 'Browse' button next to 'Input File'
2. Navigate to your data file
3. Select the file and click 'Open'
4. Tool will automatically detect wavelengths and properties
5. Available properties will appear in the dropdown

**Tips:**
- Ensure no missing values in wavelength columns
- Property names should be clear (e.g., "Clay_percent", "SOC")
- Remove any non-numeric data from wavelength columns
"""
            },
            
            # Model Selection
            "models": {
                "keywords": ["model", "algorithm", "pls", "svm", "random forest", "xgboost", "ridge", "lasso", "which model", "choose model", "huber", "gradient boosting", "gaussian process", "elastic net"],
                "title": "Choosing the Right Model",
                "content": """
**Available Models and When to Use Them:**

**PLS-R (Partial Least Squares Regression)**
- Best for: High-dimensional spectral data with multicollinearity
- Use when: You have many wavelengths and correlated features
- Parameters: Number of components (typically 5-30)

**SVM (Support Vector Machine)**
- Best for: Non-linear relationships, robust to outliers
- Use when: Data has complex patterns
- Parameters: Kernel (rbf, linear, poly), C, gamma

**Random Forest**
- Best for: Robust predictions, feature importance
- Use when: You want reliable results without much tuning
- Parameters: Number of trees, max depth

**XGBoost**
- Best for: Highest accuracy, competitive modeling
- Use when: Performance is critical
- Parameters: Learning rate, max depth, n_estimators

**Gradient Boosting**
- Best for: Strong predictive performance with sequential learning
- Use when: You want boosting without XGBoost library
- Parameters: Learning rate, n_estimators, max depth

**Ridge Regression**
- Best for: Linear relationships with regularization
- Use when: Preventing overfitting is important

**Lasso Regression**
- Best for: Feature selection, sparse models
- Use when: You want to identify key wavelengths

**Elastic Net**
- Best for: Combines Ridge and Lasso benefits
- Use when: You need both regularization and feature selection
- Parameters: Alpha (strength), l1_ratio (balance)

**Huber Regressor**
- Best for: Data with outliers
- Use when: Your dataset contains anomalous measurements
- Parameters: Epsilon (outlier threshold), alpha

**Gaussian Process**
- Best for: Small datasets with uncertainty estimates
- Use when: You need prediction confidence intervals
- Parameters: Kernel, alpha (noise level)

**Multiple Linear Regression**
- Best for: Simple baseline models
- Use when: Testing basic linear relationships

**Recommendation:** Start with PLS-R for spectral data, then try Random Forest or XGBoost for comparison.
"""
            },
            
            # Preprocessing
            "preprocessing": {
                "keywords": ["preprocess", "transform", "continuum removal", "derivative", "absorbance", "normalization", "filter wavelength"],
                "title": "Data Preprocessing Options",
                "content": """
**Preprocessing Techniques:**

**Continuum Removal**
- Removes baseline variations
- Enhances absorption features
- Use for: Reflectance spectroscopy data
- Good for: Isolating absorption bands

**First Derivative**
- Removes baseline shifts
- Enhances spectral features
- Use for: Reducing scatter effects
- Good for: Highlighting slope changes

**Second Derivative**
- Removes linear baselines
- Sharpens peaks
- Use for: Overlapping peaks
- Good for: Fine feature detection

**Absorbance Transformation**
- Converts reflectance to absorbance
- Use for: Direct comparison with lab spectra
- Formula: -log10(reflectance)

**Wavelength Filtering**
- Select specific wavelength ranges (Step ② Spectral)
- Use for: Removing noisy regions
- Example: 400-2500 nm for VIS-NIR-SWIR
- Optional "Excluded bands" lets you cut out specific sub-ranges (e.g. water
  vapour 1350-1450, 1800-1960 nm) without discarding the whole spectrum —
  see the **Resampling & Sensor Bands** help topic

**Missing-Data Handling (Step ① Data)**
When the loaded Excel/CSV has empty or non-numeric spectral cells, choose a
strategy — see the dedicated **Missing Data Handling** help topic for details
on each of the 5 available methods.

**Spectral Resampling (Step ④ Resampling)**
Reprojecting spectra onto a different band grid (uniform spacing or a named
satellite sensor) is now a full engine with 7 methods and physically-weighted
options — see the dedicated **Resampling & Sensor Bands** help topic for
details on which method to pick and why linear can be wrong.

**Best Practices:**
- Try without preprocessing first (baseline)
- Test different combinations
- Check if preprocessing improves model performance
"""
            },
            
            # Cross-Validation
            "cross_validation": {
                "keywords": ["cross validation", "k-fold", "leave one out", "loo", "validate", "test", "evaluation"],
                "title": "Cross-Validation Methods",
                "content": """
**Understanding Cross-Validation:**

Cross-validation tests how well your model generalizes to unseen data.

**K-Fold Cross-Validation**
- Splits data into K equal folds
- Trains on K-1 folds, tests on remaining fold
- Repeats K times
- **Use when:** You have moderate to large datasets (>50 samples)
- **Recommended K:** 5 or 10

**Leave-One-Out (LOO)**
- Leaves one sample out for testing
- Trains on all other samples
- Repeats for each sample
- **Use when:** Small datasets (<50 samples)
- **Warning:** Computationally expensive for large datasets

**Leave-P-Out**
- Leaves P samples out for testing
- More thorough than LOO
- **Use when:** You want rigorous validation with small datasets
- **Warning:** Very computationally intensive

**Interpreting Results:**
- **R² (CV):** Should be close to training R²
- **RMSE (CV):** Should be similar to training RMSE
- Large difference indicates overfitting

**Best Practices:**
- Always perform cross-validation before trusting your model
- Use same preprocessing in each fold
- Report both training and CV metrics
"""
            },
            
            # Batch Processing
            "batch_processing": {
                "keywords": ["batch", "multiple properties", "multiple models", "compare models", "batch mode", "run multiple"],
                "title": "Batch Processing - Multiple Properties & Models",
                "content": """
**Batch Processing Features:**

**Multiple Properties:**
1. Hold Ctrl while clicking to select multiple soil properties
2. Tool will analyze each property separately
3. Results saved in one comprehensive Excel file

**Multiple Models:**
1. Enable "Run Multiple Models" checkbox
2. Select which models to run (Ctrl+Click for multiple)
3. Tool compares all models for each property
4. Automatically recommends best model

**What You Get:**
- **Excel File** with sheets for each property-model combination
- **Model Comparison Sheet** ranking models by performance
- **Scatter Plots** (Observed vs Predicted) for each combination
- **Feature Importance Plots** showing key wavelengths/features
- **Best Model Recommendations** based on R², RMSE, and CV scores

**Workflow:**
1. Load your spectral data
2. Select multiple properties (Ctrl+Click)
3. Check "Run Multiple Models"
4. Select models to compare
5. Set preprocessing options (applied to all)
6. Click "Start Batch Analysis"
7. Wait for completion
8. Review comprehensive results

**Tips:**
- More models = longer processing time
- Results are saved automatically
- Check comparison sheet for best model per property
"""
            },
            
            # Parameters
            "parameters": {
                "keywords": ["parameter", "settings", "tune", "optimize", "components", "configuration"],
                "title": "Model Parameters and Tuning",
                "content": """
**Understanding Model Parameters:**

**PLS-R Parameters:**
- **Components:** Number of latent variables (5-30 typical)
  - Too few: Underfitting
  - Too many: Overfitting
  - Enable "Optimize Components" for automatic selection

**SVM Parameters:**
- **Kernel:** rbf (default), linear, poly, sigmoid
- **C:** Regularization (0.1-100, default 1.0)
- **Gamma:** Kernel coefficient (0.001-1.0, default 'scale')
- **Epsilon:** Tolerance (0.01-1.0, default 0.1)

**Random Forest Parameters:**
- **N_estimators:** Number of trees (100-500)
- **Max_depth:** Tree depth (None for unlimited)
- **Min_samples_split:** Min samples to split (2-10)

**XGBoost Parameters:**
- **Learning_rate:** Step size (0.01-0.3)
- **Max_depth:** Tree depth (3-10)
- **N_estimators:** Number of boosting rounds (100-1000)

**Ridge/Lasso Parameters:**
- **Alpha:** Regularization strength (0.01-100)
  - Higher = more regularization

**Optimization Tips:**
- Start with default values
- Use "Optimize Components" for PLS-R
- Adjust parameters if underfitting/overfitting
- Cross-validation helps identify optimal settings
"""
            },
            
            # Image Processing
            "image_processing": {
                "keywords": ["image", "raster", "tiff", "geotiff", "hyperspectral", "predict image", "map",
                             "envi", "bil", "bip", "bsq", "erdas", "img format", "nitf", "background mask",
                             "no-data", "nodata", "resampled cube", "export resampled"],
                "title": "Hyperspectral Image Processing",
                "content": """
**Applying Models to Images (Step ⑦ Apply → Image Processing):**

**Requirements:**
1. Trained model must be loaded or created
2. Image must be a multi-band geospatial raster
3. Bands do NOT need to match the training grid exactly — the image is
   automatically resampled onto the model's input grid before prediction
   (see 'What Happens' below)

**Supported Formats (anything except plain photos):**
- GeoTIFF / TIFF (.tif, .tiff)
- ENVI with a .hdr header (.dat, .bil, .bip, .bsq, .bin, .raw)
- ERDAS Imagine (.img)
- NITF (.ntf, .nitf), PCIDSK (.pix)
- Ordinary photos (.jpg, .png, etc.) are rejected — they carry no spectral bands

**Steps:**
1. Train a model or load a saved model
2. Check "Apply on Image", click "Load" and select your image
3. (Optional) tick "Export resampled cube" to save the exact spectra the
   model saw — useful to sanity-check resampling before trusting predictions
4. Click "Predict" — a progress bar reports resampling / prediction / saving
5. Click "View" to preview the predicted map, or open the saved file in GIS software

**What Happens:**
- The image's own per-band wavelengths are read from its header (ENVI tags or
  sidecar .hdr) when present; otherwise a range is assumed and you're warned
- Bands are resampled onto the model's input grid: the model's resampled grid
  if it was trained with resampling, otherwise the training Excel's wavelengths
  — using the SAME resampling method saved with the model
- Same preprocessing as training is applied automatically, then scaled and predicted

**Background / No-Data Masking:**
- "Mask background / no-data pixels" (on by default) skips pixels that are
  the header's no-data value, all-zero/negative, NaN/Inf, or extreme
  (saturated) — these are written as no-data in the output, never predicted
- This prevents nonsense values (huge/negative/NaN) showing up over image
  borders, clouds, or sensor gaps

**Validating Resampling:**
- "Export resampled cube" writes `<input-name>_resampled.<input-format>`
  next to the prediction — same driver, band interleave (BIL/BIP/BSQ) and a
  correct header with the model's band wavelengths, so you can open it in
  GIS/spectral-viewer software and confirm the spectra look right

**Output:**
- Predicted property map, written in the SAME format as the input image
  (e.g. ENVI in → ENVI + .hdr out), with georeferencing preserved
- A clean, single-band header (no leftover multi-band spectral metadata)

**Tips:**
- Use models with good CV scores
- Consider computational time for large images
- Visualize with different colormaps
"""
            },
            
            # Saving Results
            "saving": {
                "keywords": ["save", "export", "output", "results", "model file", "save model"],
                "title": "Saving Your Work",
                "content": """
**Save Options:**

**1. Save Results (Excel)**
- Training and test statistics
- Cross-validation scores
- Predicted vs observed values
- Wavelength importance/coefficients
- Automatically generated filename with timestamp

**2. Save Trained Model**
- Saves model, scalers, and metadata
- Can be reloaded later
- Use for: Applying to new data or images
- File format: .joblib

**3. Save Batch Results**
- Comprehensive Excel with multiple sheets
- Comparison charts
- Feature importance
- Scatter plots saved as PDF

**4. Save Predicted Image**
- GeoTIFF format
- Georeferenced output
- Ready for GIS software

**Loading Saved Models:**
1. Click "Load Model" button
2. Select .joblib file
3. Model is ready to use
4. Apply to new spectra or images

**File Naming:**
- Results: property_model_YYYYMMDD_HHMMSS.xlsx
- Models: model_property_YYYYMMDD_HHMMSS.joblib
- Images: predicted_property_YYYYMMDD_HHMMSS.tif

**Best Practices:**
- Save models with good CV scores
- Keep original data with results
- Document preprocessing settings used
"""
            },
            
            # Troubleshooting
            "troubleshooting": {
                "keywords": ["error", "problem", "not working", "fail", "crash", "bug", "issue", "help"],
                "title": "Troubleshooting Common Issues",
                "content": """
**Common Issues and Solutions:**

**Data Loading Errors:**
- Check file format (Excel or CSV)
- Ensure wavelengths are numeric column headers
- Remove special characters from column names
- Check for missing values

**Model Training Fails:**
- Verify data has enough samples (at least 10)
- Check for missing values in property column
- Ensure wavelengths are numeric
- Try simpler model (Linear instead of XGBoost)

**Poor Model Performance:**
- Try different preprocessing methods
- Increase cross-validation folds
- Check for outliers in data
- Try different models
- Optimize parameters

**Image Prediction Issues:**
- Ensure image bands match training wavelengths
- Check image format (must be multi-band raster)
- Verify model is trained/loaded
- Check memory for large images

**Slow Performance:**
- Reduce cross-validation folds
- Use fewer models in batch mode
- Close other programs
- Consider smaller datasets for testing

**General Tips:**
- Check console for detailed error messages
- Ensure all dependencies are installed
- Verify data quality before analysis
- Start with simple models and build complexity
"""
            },
            
            # Interpretation
            "interpretation": {
                "keywords": ["interpret", "understand", "meaning", "r2", "rmse", "results", "metrics", "scores"],
                "title": "Interpreting Results",
                "content": """
**Understanding Your Results:**

**R² (R-Squared)**
- Range: 0 to 1 (higher is better)
- 0.9-1.0: Excellent prediction
- 0.7-0.9: Good prediction
- 0.5-0.7: Moderate prediction
- <0.5: Poor prediction (reconsider model/data)

**RMSE (Root Mean Square Error)**
- Lower is better
- Same units as your property
- Compare to property's range
- Examples:
  - Clay %: RMSE < 5% is good
  - SOC %: RMSE < 0.5% is good

**Training vs Cross-Validation:**
- **Similar scores:** Model generalizes well
- **Training >> CV:** Overfitting (model memorizes data)
- **Solution:** Simplify model or get more data

**Feature Importance:**
- **Tree models:** Direct feature importance scores
- **Linear models:** Wavelength coefficients
- **High importance:** Key wavelengths for prediction
- Use to: Understand which parts of spectrum matter

**Scatter Plots:**
- Points near 1:1 line: Good predictions
- Scattered points: Poor model fit
- Patterns in residuals: Consider non-linear model

**Best Practices:**
- Report both training and CV metrics
- Check residual plots for patterns
- Compare multiple models
- Validate on independent dataset when possible
"""
            },
            
            # Workflow
            "workflow": {
                "keywords": ["workflow", "process", "steps", "procedure", "how do i", "complete analysis"],
                "title": "Complete Analysis Workflow",
                "content": """
**Recommended Analysis Workflow:**

**Step 1: Data Preparation**
- Prepare spectral data (Excel/CSV)
- Ensure proper format (wavelengths as columns)
- Clean data (remove missing values)

**Step 2: Load and Explore**
- Load data into Paracuda
- Select property of interest
- Check data statistics

**Step 3: Preprocessing (Optional)**
- Try baseline (no preprocessing) first
- Test continuum removal
- Try derivatives if needed
- Filter wavelengths if necessary

**Step 4: Model Selection**
- Start with PLS-R (good for spectral data)
- Set cross-validation (k-fold with k=10)
- Train initial model
- Check R² and RMSE

**Step 5: Model Comparison**
- Enable "Run Multiple Models"
- Select 3-5 models to compare
- Run batch analysis
- Review comparison sheet

**Step 6: Optimization**
- For best model, optimize parameters
- Use "Optimize Components" for PLS-R
- Adjust hyperparameters if needed
- Retrain with optimal settings

**Step 7: Validation**
- Check cross-validation scores
- Examine scatter plots
- Review feature importance
- Test on independent data if available

**Step 8: Save Everything**
- Save final model
- Export results to Excel
- Document preprocessing used
- Save settings for reproducibility

**Step 9: Application (Optional)**
- Load trained model
- Apply to hyperspectral images
- Generate prediction maps
"""
            },
            
            # Best Practices
            "best_practices": {
                "keywords": ["best practice", "recommendation", "tips", "advice", "should i", "optimal"],
                "title": "Best Practices for Spectral Modeling",
                "content": """
**General Best Practices:**

**Data Quality:**
✓ Ensure consistent measurement conditions
✓ Calibrate instruments properly
✓ Remove obvious outliers
✓ Have adequate sample size (50+ preferred)
✓ Balance training data across property range

**Preprocessing:**
✓ Always keep an unprocessed baseline
✓ Test multiple preprocessing methods
✓ Use same preprocessing for training and prediction
✓ Document which preprocessing works best

**Model Selection:**
✓ Start simple (Linear, PLS-R)
✓ Progress to complex (Random Forest, XGBoost)
✓ Compare multiple models
✓ Don't overfit - simpler is often better

**Cross-Validation:**
✓ Always perform CV before trusting results
✓ Use k-fold (k=5 or 10) for most cases
✓ Use LOO only for small datasets
✓ Report both training and CV metrics

**Parameter Tuning:**
✓ Start with defaults
✓ Use automatic optimization when available
✓ Avoid extreme parameter values
✓ Validate after tuning

**Reporting Results:**
✓ Report R², RMSE, and CV scores
✓ Include scatter plots
✓ Show feature importance
✓ Document preprocessing and parameters
✓ Compare to literature values

**Model Application:**
✓ Only use models with good CV scores
✓ Test on independent validation set
✓ Be cautious of extrapolation
✓ Document model limitations
"""
            },

            # ── NEW FEATURES ─────────────────────────────────────────────

            # Tabular Prediction
            "tabular_prediction": {
                "keywords": ["tabular", "predict unknown", "unknown data", "excel prediction",
                             "csv prediction", "new samples", "apply model", "predict new",
                             "unknown excel", "unknown csv"],
                "title": "Tabular Prediction — Predicting Unknown Samples",
                "content": """
**What is Tabular Prediction?**
Apply a trained model to new spectral data (Excel or CSV) to predict soil properties
for samples that do not have lab-measured values.

**Workflow:**
1. Train a model (or load a saved one with 'Load Model')
2. Open the **Tabular Prediction** section in the control panel
3. Click **'Load Excel/CSV'** — select your unknown samples file
4. Click **'Check Data Info'** to verify wavelengths were detected correctly
5. Click **'Predict Unknown'** — results are saved to Excel automatically

**Input File Requirements:**
- Same format as training data (wavelengths as column headers)
- Wavelength columns must overlap with the model's training wavelengths
- Wavelength units (nm or μm) are detected automatically
- Non-wavelength columns (sample IDs, etc.) are preserved in output

**Important Notes:**
- The preprocessing applied during training is automatically re-applied to the unknown data
- No need to manually match preprocessing settings — they are saved inside the model
- Loading unknown data does NOT reset your trained model or any current results
- The predict button is only enabled after unknown data has been loaded

**Output:**
- Excel file with sample IDs + predicted values
- Auto-generated filename with property name and timestamp
"""
            },

            # Overfitting Detection
            "overfitting": {
                "keywords": ["overfit", "overfitting", "red bar", "overfitting flag",
                             "training vs cv", "gap", "memorize", "generalise", "generalize"],
                "title": "Overfitting Detection",
                "content": """
**What is Overfitting?**
A model that memorises training data but fails to generalise to new samples.
Signs: high training R² but much lower CV R².

**How Paracuda Detects It:**
Paracuda checks several measures together and only flags overfitting when the
model genuinely learned the training data AND generalises materially worse.
ALL of these must hold:
- Training R² ≥ 0.60 — the model actually fit the training set. (A model with a
  low/negative Training R² is *underfitting*, not overfitting, and is not flagged.)
- Training R² − Test R² > 0.15 — a sizeable absolute drop from train to test.
- The drop is > 25% of the training skill — a large *relative* degradation.

Two extra measures raise the severity from *mild* to *strong* when present:
- Training R² − CV R² > 0.15 (cross-validation confirms the gap).
- Test RMSE > 1.3 × Train RMSE.

**Visual Cues in Batch Reports:**
- Normal bars: blue/green fill
- Overfitting bars: **red hatched fill** — easy to spot in comparison charts
- The batch summary table shows an `Overfit` column (ok / mild / strong)
- A warning is written to the status log listing exactly which measures fired

**What To Do:**
- Reduce model complexity (fewer components, shallower trees)
- Try stronger regularisation (higher alpha for Ridge/Lasso)
- Increase dataset size if possible
- Use simpler preprocessing
- For PLS-R: reduce the number of components or enable 'Optimize Components'

**Note:**
These are heuristics — always inspect scatter plots and residuals as well.
"""
            },

            # Baseline Correction
            "baseline_correction": {
                "keywords": ["baseline", "baseline correction", "lower envelope", "polynomial baseline",
                             "linear baseline", "scatter correction", "background removal"],
                "title": "Baseline Correction Preprocessing",
                "content": """
**What is Baseline Correction?**
Removes slowly-varying background (scatter, fluorescence, instrument drift) from spectra,
isolating the true absorption features.

**Method Options:**

**Linear Baseline**
- Uses the minimum reflectance in the first and last 5% of bands as anchor points
- Fits a straight line between them
- Subtracts line from the spectrum
- Best for: simple tilt correction

**Polynomial Baseline**
- Detects lower-envelope anchor points (local minima along the spectrum)
- Fits a polynomial through those points
- Baseline is clamped to never exceed the spectrum
- Subtracts polynomial from the spectrum
- Best for: curved backgrounds, fluorescence, broad scatter

**Why Is This Better Than Simple Subtraction?**
Paracuda uses lower-envelope (valley) anchors — not peak anchors.
This avoids the "variance destruction" problem where baseline overshoots the spectrum
and causes near-zero variance → StandardScaler overflow → R² = −1e27.

**When to Use:**
- When spectra show systematic baseline shifts between samples
- When instrument calibration drift is present
- Before derivative transforms (removes linear/polynomial trends)
"""
            },

            # Wavelength Unit Auto-Detection
            "wavelength_units": {
                "keywords": ["wavelength unit", "nanometer", "micrometer", "nm", "um", "µm",
                             "auto detect", "wavelength detection", "micron", "unit conversion"],
                "title": "Wavelength Unit Auto-Detection",
                "content": """
**Automatic Wavelength Unit Handling:**
Paracuda automatically detects whether your column headers are in nanometers (nm)
or micrometers (μm) and converts everything to nm internally.

**Detection Logic:**
- Values in range 350–2500 → assumed nm (standard VIS-NIR-SWIR)
- Values in range 0.35–2.5 → assumed μm → multiplied by 1000 to convert to nm
- Values in range 1000–20000 → assumed nm (MIR range)

**When Does This Matter?**
- Loading training data with μm column headers
- Loading unknown prediction data with different units than training data
- Spectral harmonization between sensors using different units

**No Manual Action Needed:**
Units are handled transparently. The status panel shows:
'Wavelength range detected: X–Y nm (unit: nm)' or '(unit: μm → converted)'

**Spectral Domain Selection:**
After loading data you can filter to a specific spectral domain:
- VIS-NIR (350–1000 nm)
- NIR-SWIR (1000–2500 nm)
- Full range (all bands)
- Custom range (manually specify min/max nm)
"""
            },

            # Data Randomization & Integrity Tests
            "data_randomization": {
                "keywords": ["randomization", "permutation test", "label permutation",
                             "data integrity", "mixing test", "spectral mixing",
                             "statistical significance", "p-value", "shuffle labels"],
                "title": "Data Integrity / Randomization Tests",
                "content": """
**Why Test Data Integrity?**
Before trusting a model, verify that spectral patterns genuinely predict your target property
and aren't an artefact of chance, data leakage, or random correlation.

**Available Tests (Tools → Data Randomization):**

**1. Label Permutation Test**
- Randomly shuffles the target property labels many times (e.g. 200 permutations)
- Trains PLS-R model on each shuffled version and records CV R²
- Compares the observed (real) R² against the distribution of shuffled R² values
- p-value = proportion of permuted R² values ≥ observed R²
- p < 0.05 → model is statistically significant
- Result shown as histogram with observed R² marked in red
- **Save**: saves permuted scores Excel + plot PNG

**2. Spectral Mixing Integrity**
- Takes a fraction of samples (e.g. 20%) and swaps their labels with random samples
- Measures how much R² drops after mixing
- Large drop → labels are meaningful and not random
- Small/no drop → model may not be learning true soil-spectra relationships
- Result shown as a bar chart (Original vs Mixed R²)
- **Save**: saves summary + mixed dataset Excel + plot PNG

**Interpretation:**
- Both tests passing → high confidence in model validity
- Label permutation p-value < 0.05 is the primary criterion
"""
            },

            # Spectral Harmonization
            "spectral_harmonization": {
                "keywords": ["harmonization", "harmonise", "harmonize", "transfer function",
                             "cross-sensor", "sensor calibration", "band matching",
                             "apply transfer function", "pls transfer"],
                "title": "Spectral Harmonization — Transfer Functions",
                "content": """
**What is Spectral Harmonization?**
Converting spectra from one sensor/instrument to match another sensor's response,
enabling models trained on one sensor to be applied to data from a different sensor.

**Example Use Cases:**
- Field spectrometer (ASD) → Satellite (EMIT, EnMAP, Sentinel-2)
- Lab scanner → Airborne hyperspectral
- Old instrument → New instrument after calibration drift

**Two-Tab Workflow (Tools → Spectral Harmonization):**

**Tab 1: Compute Transfer Function**
1. Load paired source and target spectra (same samples, different instruments)
2. Set number of PLS components (default 5)
3. Click 'Compute Transfer Function'
4. Per-band R² plot shows quality across the spectrum
5. Save the transfer function (.joblib) for reuse
6. Save R² report (Excel + plot) for documentation

**Tab 2: Apply Transfer Function**
1. Load a previously saved transfer function (.joblib)
2. Load the source spectra you want to harmonize
3. Click 'Apply Transfer Function'
4. Mean spectrum comparison plot shows before/after
5. Save harmonized spectra (Excel or CSV)

**Quality Assessment:**
- Mean band R² > 0.90 → excellent harmonization
- Mean band R² 0.70–0.90 → acceptable
- Mean band R² < 0.70 → consider more paired samples or different n_components

**File Formats:**
- Transfer function: .joblib (portable, versioned)
- Harmonized output: .xlsx or .csv with wavelength nm as column headers
"""
            },

            # Preprocessing kwargs / model portability
            "model_portability": {
                "keywords": ["save model", "load model", "preprocessing saved", "portable model",
                             "joblib", "model file", "reuse model", "preprocessing settings",
                             "model metadata", "training input wavelengths", "resampling saved"],
                "title": "Model Portability — Saving and Loading Models",
                "content": """
**What's Stored in a Saved Model (.joblib):**
- The fitted regression model + PCA component (if used)
- Feature scalers (StandardScaler for X and y)
- Training metadata: property name, model type, training wavelengths
- Preprocessing method and all its parameters
- **Full resampling configuration**: whether resampling was on, the method,
  target sensor/spacing, per-band FWHMs, custom FWHM/SRF tables, exclude ranges
- **The model's exact input band grid** (`training_input_wavelengths` /
  `..._fwhms`) — the resampled grid if resampling was used, otherwise the
  training Excel's wavelengths — so unknown data or images can always be
  resampled onto precisely what the model expects, even on a different
  original band grid

**Why This Matters:**
When you load a model and apply it to new data (tabular prediction or image
prediction), the exact same preprocessing AND resampling used during training
is automatically re-applied. You don't need to manually remember or
re-configure any settings — even if the new data's own wavelength grid
differs from the model's.

**Loading a Model:**
1. Click 'Load Model' button
2. Select the .joblib file
3. Model is ready to use — preprocessing and resampling auto-configured
4. Status panel confirms what was restored (preprocessing method, resampling
   method and target band count)

**Applying to New Data:**
- Tabular Prediction: loads Excel/CSV, resamples + preprocesses, returns predictions
- Image Prediction: reads any supported geospatial raster, resamples every
  pixel's spectrum onto the model's grid, applies preprocessing, predicts

**Best Practice:**
Save models immediately after training with good CV scores.
Include the property name and date in the filename for traceability.
"""
            },

            # Spectral Resampling
            "resampling": {
                "keywords": ["resample", "resampling", "sensor bands", "band matching",
                             "nearest neighbour", "nearest neighbor interpolation",
                             "linear interpolation", "cubic spline", "quadratic interpolation",
                             "gaussian srf", "empirical srf", "band averaging", "fwhm",
                             "spectral response function", "srf", "custom fwhm", "spectral binning",
                             "exclude bands", "exclude ranges", "water absorption"],
                "title": "Resampling & Sensor Bands",
                "content": """
**What is Spectral Resampling?**
Reprojecting spectra from their native wavelength grid onto a different band
grid — either a uniform spacing or a named satellite sensor (e.g. Sentinel-2,
Landsat-8, EnMAP) — so a model can be trained on, or applied to, data from a
different instrument. Configured in Step ④ Resampling.

**The 7 Methods (dropdown order):**

*Interpolation family* — reproject the curve onto new points, no notion of
instrument bandwidth:
- **Linear Interpolation** — straight line between neighbouring points.
  Simple and safe, but can smear sharp absorption features on a coarse grid.
- **Nearest Neighbour Interpolation** — copies the closest source band
  unchanged. Preserves the true measured value at each point rather than
  averaging/blending it — often better than linear when the target grid is
  coarser than the source, or when you want to avoid inventing intermediate
  values that were never actually measured.
- **Quadratic Interpolation** — smoother curve through 3+ points.
- **Cubic Spline** — smoothest curve through 4+ points; can overshoot near
  sharp peaks.

*Bandwidth-aware family* — integrate the source spectrum under each target
band's actual spectral response, matching how a real sensor measures light
(needs a per-band FWHM):
- **Gaussian SRF** — analytic Gaussian response shaped by each band's FWHM.
  The physically realistic default for satellite/airborne sensors.
- **Empirical SRF** — integrates against an uploaded real per-band response
  curve (CSV), falling back to Gaussian where no curve is supplied. Most
  accurate when you have the sensor's actual spectral response function.
- **Band Averaging** — flat (tophat) average over each band's FWHM window.

**Which Should I Pick?**
- Matching a named sensor → Gaussian SRF (or Empirical SRF if you have the
  real response curves) — this is what the sensor physically measures
- Simple uniform down/up-sampling, want exact measured values → Nearest
  Neighbour Interpolation
- Simple uniform resampling, want a smooth curve → Linear or Cubic Spline
- Linear interpolation is NOT always correct: it assumes reflectance varies
  smoothly between points, which can blur real absorption features — try
  Nearest Neighbour or a bandwidth-aware method if results look flattened

**Sensor Bands & Custom Grids:**
- Choose a named sensor (its band centres + FWHMs are built in), or "Custom"
  with a uniform spacing (nm)
- **Custom FWHM CSV** — load your own (centre, FWHM) table instead of the
  sensor default
- **Empirical SRF CSV** — load real per-band response curves for the most
  physically accurate integration

**Excluded Bands (Step ② Spectral, "Excluded bands"):**
Cut out specific sub-ranges without discarding the whole spectrum — quick
presets are offered for common water-vapour absorption windows
(1350-1450 nm, 1800-1960 nm) and noisy sensor edges.

**Spectral Binning:**
Reduces band count by keeping one representative band per group (decimation),
applied after resampling — useful to shrink a very dense hyperspectral grid.

**Where This Applies:**
- Training: resampling is applied to the loaded Excel data before model fitting
- Prediction: the SAME method + target grid saved with the model is
  automatically re-applied to unknown tabular data and to images (see
  **Model Portability** and **Hyperspectral Image Processing** topics)
"""
            },

            # Missing Data Handling
            "missing_data": {
                "keywords": ["missing data", "imputation", "mean imputation", "median imputation",
                             "drop rows with missing", "fill with zero", "nan values", "empty cells",
                             "missing values", "spectral interpolation missing", "impute"],
                "title": "Missing Data Handling",
                "content": """
**Why This Matters:**
Excel/CSV spectral data often contains empty cells or non-numeric text.
Left unhandled, these become NaN and silently break scaling and model
training with confusing errors. Right after loading, Paracuda reports exactly
how much data is missing; you then choose a strategy (Step ① Data).

**The 5 Strategies (dropdown order):**
- **Drop rows with missing** (default) — removes any sample with at least
  one missing value. Safest choice; use when you can afford to lose a few rows.
- **Mean imputation** — fills each missing cell with that wavelength
  column's mean across all samples.
- **Median imputation** — same idea, using the median (more robust to
  outliers than the mean).
- **Spectral interpolation** — fills a gap using the neighbouring
  wavelengths in the SAME row (linear interpolation along the spectrum);
  falls back to the column mean if a row has too few valid points to
  interpolate from.
- **Fill with zero** — last resort; can distort the spectral shape and bias
  the model, only use if you understand the consequence.

**Which Should I Pick?**
- Few missing rows, plenty of samples → Drop rows with missing
- Missing scattered across many samples/columns → Mean or Median imputation
- Missing is a short gap within an otherwise good spectrum → Spectral
  interpolation (preserves the row's own spectral shape best)
- Avoid Fill with zero unless you have a specific reason

**Where to Check:**
The status log after loading data reports total missing cells, how many rows
are affected, and which strategy will be applied.
"""
            },

            # Hyperparameter Tuning
            "hyperparameter_tuning": {
                "keywords": ["hyperparameter", "optuna", "tune hyperparameters", "trials",
                             "auto tune", "tpe sampler", "tuning cv"],
                "title": "Hyperparameter Tuning with Optuna",
                "content": """
**What is Automatic Hyperparameter Tuning?**
Instead of manually guessing model parameters, Paracuda can search for good
values automatically using Optuna (Bayesian/TPE search), in Step ④ Model.

**How to Use:**
1. Check "⚙ Tune Hyperparameters (Optuna)"
2. Set **Trials** — how many parameter combinations to try (default 30;
   more trials = better search but longer runtime)
3. Set **CV folds** — how many folds each trial is validated on internally
   (default 3) before scoring it
4. Train as normal — the winning/selected model is re-fit using the best
   parameters Optuna found

**What Happens:**
- Each trial samples a candidate set of parameters for the current model
- The candidate is scored by internal cross-validation (not your main CV)
- After all trials, the best-scoring parameters are used to fit the final model
- Works for both Single-model and Batch (multi-model) mode — each model in a
  batch gets its own independent tuning search

**Tips:**
- Start with the default 30 trials; increase for a more thorough search if
  you have time (runtime scales roughly linearly with trials)
- Tuning searches a sensible parameter range per model — for PLS-R this
  includes number of components, for tree models depth/estimators, etc.
- Combine with cross-validation to confirm the tuned model generalises well
"""
            },

            # Model Development Flow
            "model_development_flow": {
                "keywords": ["model development flow", "flow chart", "pipeline overview",
                             "process flow", "configuration preview", "flow diagram",
                             "development flow pane"],
                "title": "Model Development Flow Chart",
                "content": """
**What is the Model Development Flow?**
A live diagram — docked below the step wizard — that shows the exact pipeline
you've configured: Data → Property → Spectral domain → Excluded bands →
Resampling → Preprocessing → Model → Validation → Apply.

**How It Works:**
- Visible for every wizard step, not just Apply — it updates in real time as
  you change any setting
- On a fresh session or after "🔄 Reset", every stage is faded (dashed
  border) — cards light up (solid blue border) only once you actually
  configure that step, or after a model is trained
- The compact docked view shows short titles; click "🔍 Large" for full
  per-stage detail (exact values, methods, parameters) in a pop-up window
- "💾 300 DPI" exports the detailed chart as a publication-ready PNG/PDF/SVG
  — useful for documenting exactly how a model was built in a report or paper

**Why Use It:**
- Quickly see, at a glance, what's configured vs. still default/unset
- Confirm the resampling/preprocessing/model settings before running a
  long analysis
- Attach the exported chart to a report as a record of the model's provenance
"""
            },
        }

    def find_best_match(self, query):
        """Find the best matching help topic based on user query"""
        if not query or len(query.strip()) < 2:
            return self._get_default_help()
        
        query = query.lower().strip()
        best_match = None
        best_score = 0
        
        # Check for exact keyword matches first
        for topic_id, topic in self.knowledge_base.items():
            for keyword in topic["keywords"]:
                if keyword in query or query in keyword:
                    score = len(keyword) / len(query) if len(query) > len(keyword) else len(query) / len(keyword)
                    if score > best_score:
                        best_score = score
                        best_match = topic_id
        
        # If no good keyword match, try fuzzy matching on title and keywords
        if best_score < 0.5:
            for topic_id, topic in self.knowledge_base.items():
                # Check similarity with title
                title_similarity = SequenceMatcher(None, query, topic["title"].lower()).ratio()
                
                # Check similarity with keywords
                keyword_similarities = [
                    SequenceMatcher(None, query, keyword).ratio()
                    for keyword in topic["keywords"]
                ]
                max_keyword_similarity = max(keyword_similarities) if keyword_similarities else 0
                
                # Use the better of the two
                score = max(title_similarity, max_keyword_similarity)
                
                if score > best_score:
                    best_score = score
                    best_match = topic_id
        
        # If we found a reasonable match, return it
        if best_match and best_score > 0.3:
            return self._format_response(self.knowledge_base[best_match])
        
        # Otherwise, provide helpful suggestions
        return self._get_suggestions(query)
    
    def _format_response(self, topic):
        """Format a help topic into a readable response"""
        return f"{topic['title']}\n{'=' * len(topic['title'])}\n\n{topic['content'].strip()}"
    
    def _get_default_help(self):
        """Return default help when no query is provided"""
        return """Welcome to Paracuda III Help Assistant!

I can help you with:

🔰 GETTING STARTED & CORE WORKFLOW
• Getting Started — Learn the basics
• Workflow — Complete analysis procedure
• Loading Data — How to import your spectral data
• Best Practices — Tips for optimal results

📈 MODELS & TRAINING
• Choosing Models — Which model to use and when
• Preprocessing — Data transformation options
• Missing Data Handling — Dealing with empty/invalid spectral cells
• Parameters — Model settings
• Hyperparameter Tuning — Automatic search with Optuna
• Overfitting Detection — Recognise and fix overfitting

✅ VALIDATION & ANALYSIS
• Cross-Validation — Understanding validation methods
• Batch Processing — Running multiple properties/models
• Data Randomization — Label permutation and mixing integrity tests
• Model Development Flow — Live pipeline overview & 300-DPI export

🔗 ADVANCED / MULTI-SENSOR
• Tabular Prediction — Apply model to new unknown samples (Excel/CSV)
• Resampling & Sensor Bands — 7 methods incl. Nearest Neighbour & Gaussian SRF
• Spectral Harmonization — Transfer functions across sensors
• Wavelength Units — nm / μm auto-detection and conversion
• Baseline Correction — Background removal from spectra
• Image Processing — Any geospatial raster (GeoTIFF, ENVI, ERDAS, BIL/BIP/BSQ…)

💾 SAVING & LOADING
• Model Portability — Save/load models with preprocessing & resampling
• Saving Results — Exporting your work

🛠 TROUBLESHOOTING
• Troubleshooting — Solving common issues
• Interpreting Results — Understanding R², RMSE, and metrics

💡 Tip: In the help content, click on blue "📘 More about..." links
next to model names, validation methods, or metrics to get detailed
information with links to documentation!

Just type your question and I'll provide relevant guidance.

Examples:
- "How do I predict unknown samples?"
- "What resampling method should I use?"
- "How do I fix overfitting?"
- "How does baseline correction work?"
- "Can I load ENVI or BIL images?"
- "What does R² mean?"
"""
    
    def _get_suggestions(self, query):
        """Provide suggestions when no good match is found"""
        # Extract key terms from query
        terms = re.findall(r'\w+', query.lower())
        
        # Find topics that match any term
        related_topics = []
        for topic_id, topic in self.knowledge_base.items():
            for term in terms:
                if any(term in keyword for keyword in topic["keywords"]):
                    if topic["title"] not in related_topics:
                        related_topics.append(topic["title"])
                    break
        
        response = f"I couldn't find a specific answer for: '{query}'\n\n"
        
        if related_topics:
            response += "However, these topics might help:\n\n"
            for title in related_topics[:5]:
                response += f"• {title}\n"
            response += "\nTry asking about one of these topics!"
        else:
            response += """Here are some things I can help with:

• Getting Started with Paracuda
• Loading and Preparing Data
• Missing Data Handling
• Selecting and Comparing Models
• Preprocessing Techniques
• Resampling & Sensor Bands (Nearest Neighbour, Gaussian/Empirical SRF, …)
• Hyperparameter Tuning (Optuna)
• Cross-Validation Methods
• Batch Processing Multiple Properties
• Understanding Parameters
• Tabular Prediction (predicting unknown samples)
• Spectral Harmonization / Transfer Functions
• Data Randomization & Integrity Tests
• Baseline Correction
• Wavelength Unit Auto-Detection
• Overfitting Detection
• Processing Hyperspectral Images (GeoTIFF, ENVI, ERDAS, BIL/BIP/BSQ, …)
• Model Development Flow Chart
• Interpreting Results and Metrics
• Model Portability (save/load with preprocessing & resampling)
• Best Practices and Tips

Please rephrase your question or ask about a specific topic!"""
        
        return response
    
    def get_all_topics(self):
        """Return a list of all available help topics"""
        return [topic["title"] for topic in self.knowledge_base.values()]
