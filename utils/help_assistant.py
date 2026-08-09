"""
PARACUDA-NG - Intelligent Help Assistant
A context-aware help system that provides user-friendly guidance based on tool functionality.

@author: Sharad Kumar Gupta
"""

import re
from difflib import SequenceMatcher

class HelpAssistant:
    """Intelligent help system for PARACUDA-NG"""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self):
        """Build comprehensive knowledge base from tool functionality"""
        return {
            # Getting Started
            "getting_started": {
                "keywords": ["start", "begin", "first time", "new", "how to use", "tutorial", "basics"],
                "title": "Getting Started with PARACUDA-NG",
                "content": """
**Welcome to PARACUDA-NG!** Here's how to get started:

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
                "keywords": ["model", "algorithm", "pls", "svm", "random forest", "xgboost", "ridge", "lasso",
                             "which model", "choose model", "huber", "gradient boosting", "gaussian process",
                             "elastic net", "pca", "principal component", "linear regression", "svr",
                             "n_estimators", "n_components", "learning_rate", "learning rate", "max_depth",
                             "min_samples_split", "min_samples_leaf", "subsample", "colsample_bytree",
                             "reg_alpha", "reg_lambda", "l1_ratio", "max_features", "bootstrap",
                             "svd_solver", "whiten", "length_scale", "n_restarts_optimizer",
                             "fit_intercept", "kernel", "gamma", "epsilon", "solver", "max_iter",
                             "ann", "mlp", "neural network", "artificial neural network",
                             "multi-layer perceptron", "multilayer perceptron", "deep learning",
                             "hidden layers", "hidden_layer_sizes", "hidden layers mlp",
                             "activation", "relu", "tanh", "early stopping", "learning_rate_init"],
                "title": "Choosing the Right Model",
                "content": """
**PARACUDA-NG ships 13 regression algorithms.**  Every parameter listed below
is exposed in Step ④ Model and is what the estimator actually receives.
Tree/boosting models, the neural network and Gaussian Process use
`random_state=42`, so repeated runs on the same data are reproducible.

════════════════════════════════════════════════════════════
**LATENT-VARIABLE / PROJECTION**
════════════════════════════════════════════════════════════
**PLS-R (Partial Least Squares Regression)** - the spectroscopy workhorse
- Best for: high-dimensional spectra with severe multicollinearity.
- Use when: bands ≫ samples (the normal case in spectroscopy).
- Parameters: `n_components` - clamped to 1…50, and further capped to
  min(n_samples, n_features) so it can never exceed the data's rank.
- Tick "Optimize components" to scan component counts in parallel and keep
  the best-scoring value.

**PCA (Principal Component Regression)**
- Best for: decorrelating bands before a linear fit.
- Parameters: `n_components` (1…50), `svd_solver` ('auto'/'full'/'arpack'/
  'randomized'), `whiten` (True/False).
- Note: PCA components are chosen to explain *spectral* variance, not target
  variance - PLS-R usually beats it for prediction.

════════════════════════════════════════════════════════════
**REGULARISED LINEAR**
════════════════════════════════════════════════════════════
**Ridge Regression** (L2)
- Best for: linear relationships where all bands contribute a little.
- Parameters: `alpha` (regularisation strength), `solver` ('auto', 'svd',
  'cholesky', 'lsqr', 'sag', 'saga'), `max_iter`.
- Shrinks coefficients but never to exactly zero.

**Lasso Regression** (L1)
- Best for: feature selection - identifying the few key wavelengths.
- Parameters: `alpha`, `max_iter`, `selection` ('cyclic' or 'random'), `tol`.
- Drives most coefficients to exactly zero - the surviving bands are your
  "important" wavelengths.

**Elastic Net** (L1 + L2)
- Best for: correlated groups of bands where pure Lasso picks one arbitrarily.
- Parameters: `alpha` (overall strength), `l1_ratio` (0 = Ridge, 1 = Lasso),
  `max_iter`, `tol`.

**Multiple Linear Regression** (OLS)
- Best for: a simple, honest baseline.
- Parameters: `fit_intercept` (True/False).
- Warning: with more bands than samples this is unstable - use it as a
  reference point, not a production model.

**Huber Regressor** (robust linear)
- Best for: data containing a few anomalous target values.
- Parameters: `epsilon` (outlier cut-off; smaller = more robust, typical
  1.1-2.0), `max_iter`, `alpha` (L2 strength).
- Squared loss inside `epsilon`, linear loss outside - so extreme samples
  cannot dominate the fit.

════════════════════════════════════════════════════════════
**KERNEL / NON-LINEAR**
════════════════════════════════════════════════════════════
**SVM (Support Vector Regression)**
- Best for: complex non-linear relationships; robust to outliers.
- Parameters: `kernel` ('rbf', 'linear', 'poly', 'sigmoid'), `C` (error
  penalty), `degree` (poly only), `gamma` ('scale', 'auto', or a float),
  `epsilon` (width of the no-penalty tube).
- `C` and `gamma` interact strongly - this is the model that benefits most
  from **Hyperparameter Tuning**.

**Gaussian Process**
- Best for: small datasets where you want uncertainty estimates.
- Kernel: constant C(1.0, 1e-3…1e3) × RBF(`length_scale`, 1e-2…1e2).
- Parameters: `length_scale`, `alpha` (assumed noise level),
  `n_restarts_optimizer`.
- Cost grows as O(n³) - impractical beyond a few thousand samples.

════════════════════════════════════════════════════════════
**ENSEMBLE / TREE-BASED**
════════════════════════════════════════════════════════════
**Random Forest**
- Best for: reliable results with little tuning; gives feature importance.
- Parameters: `n_estimators`, `max_depth` ("None" = unlimited),
  `min_samples_split`, `min_samples_leaf`, `max_features`
  ('sqrt', 'log2', 'None', an int, or a float), `bootstrap`.
- Runs on `n_cores` threads (set in the Execution step).

**Gradient Boosting**
- Best for: strong sequential learning without the XGBoost dependency.
- Parameters: `n_estimators`, `learning_rate`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `subsample`.
- Lower `learning_rate` + more `n_estimators` generally wins, but costs time.

**XGBoost**
- Best for: top accuracy on larger tabular/spectral datasets.
- Parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
  `colsample_bytree`, `reg_alpha` (L1), `reg_lambda` (L2).
- Multi-threaded via `n_jobs`.

════════════════════════════════════════════════════════════
**NEURAL NETWORK**
════════════════════════════════════════════════════════════
**Artificial Neural Network (multi-layer perceptron)**
- Best for: genuinely non-linear property/spectrum relationships when you have
  enough samples to support them.
- Parameters:
  • `hidden_layer_sizes` - neurons per hidden layer as free text.  "64, 32"
    means two hidden layers of 64 and 32 neurons; "100" means one layer of 100.
    Commas, spaces, hyphens and brackets are all accepted, and an unparseable
    entry falls back to the default rather than aborting a batch run.
  • `activation` - relu (default), tanh, logistic or identity.
  • `solver` - adam (default), lbfgs or sgd.
    **lbfgs is often better on small datasets** (a few hundred samples);
    adam suits larger ones.
  • `alpha` - L2 penalty.  This is the main overfitting control; raise it
    (1e-3, 1e-2) if train R² far exceeds test R².
  • `learning_rate_init` - initial step size for adam / sgd.
  • `max_iter` - maximum training epochs (default 2000).
  • `early_stopping` - holds out `validation_fraction` of the training data and
    stops after `n_iter_no_change` epochs without improvement.  On by default;
    turn it OFF for very small datasets, where the holdout costs more than the
    early stop saves.
- **Always scale the inputs** - PARACUDA-NG already standardises X and y, so
  this is handled for you.
- Caveats worth knowing before you pick it:
  • Neural networks are the most data-hungry option here.  Below ~100 samples
    PLS-R will usually beat an ANN, and the ANN result will vary a lot between
    runs.
  • Convergence warnings mean it hit `max_iter` without settling: raise
    `max_iter`, raise `alpha`, or simplify the architecture.
  • It gives no interpretable coefficients, so the feature-importance plot falls
    back to prediction/band correlations rather than true weights.
- Tuning searches the architecture as whole strings ("32", "64", "64, 32",
  "128, 64", "128, 64, 32") plus activation, alpha and learning rate, because
  layer count and width interact too strongly to sample independently.

════════════════════════════════════════════════════════════
**HOW TO CHOOSE**
════════════════════════════════════════════════════════════
1. Start with **PLS-R** - it is designed for exactly this data shape.
2. Add **Random Forest** and **XGBoost** as non-linear comparisons.
3. Add **Ridge** or **Lasso** if you need an interpretable linear model or
   want to know which wavelengths matter.
4. Use **Batch Processing** to run several models × properties in one go and
   read the ranked comparison table.
5. Use **Hyperparameter Tuning** (Optuna) once you have a shortlist.

**Watch out for:**
- Tree models cannot extrapolate beyond the training target range.
- Very high training R² with poor test R² means overfitting - see
  **Overfitting Detection**.
- With < ~50 samples prefer PLS-R / Ridge / Gaussian Process over the
  boosting models and the neural network, which need data to be stable.
- Judge models on RPD and nRMSEP as well as R² - see **Interpreting Results**.
"""
            },
            
            # Compositional (log-ratio) modelling
            "compositional": {
                "keywords": ["compositional", "composition", "log-ratio", "log ratio", "logratio",
                             "clr", "alr", "ilr", "clr alr ilr", "alr clr ilr", "clr/alr/ilr",
                             "log-ratio transform", "log-ratio modelling", "log-ratio modeling",
                             "compositional modelling", "compositional modeling",
                             "centred log-ratio", "centered log-ratio",
                             "additive log-ratio", "isometric log-ratio", "aitchison", "simplex",
                             "closure", "constant sum", "sum to 100", "sums to 100", "percentages",
                             "parts", "texture", "sand silt clay", "grain size", "mineralogy",
                             "constrained", "closed data", "proportions"],
                "title": "Compositional (Log-Ratio) Modelling - CLR / ALR / ILR",
                "content": """
**Use this when your targets are PARTS OF A WHOLE** - sand + silt + clay = 100 %,
mineral percentages, grain-size fractions, or any set of proportions that must
sum to a constant.

────────────────────────────────────────────────────────────
**WHY A NORMAL REGRESSION IS WRONG HERE**
────────────────────────────────────────────────────────────
Compositional data live on the *simplex*, not in ordinary real space.  They
carry only **relative** information and are bound by a constant-sum (closure)
constraint.  If you train one independent model per part:

- the predictions need not sum to 100 % - you can get 45 + 38 + 25 = 108
- a part can be predicted negative, which is meaningless
- the covariance matrix is singular by construction, so the correlation
  structure is mis-specified and standard errors are not trustworthy

Aitchison's log-ratio transforms map a D-part composition off the simplex into
real space, where ordinary regression IS valid.  Predictions are mapped back
with the exact inverse, which is **closed to the total by construction** - so
sand + silt + clay always sums to 100 % again, and no part can go negative.

────────────────────────────────────────────────────────────
**THE THREE TRANSFORMS**
────────────────────────────────────────────────────────────
**CLR - Centred Log-Ratio**

    clr(x)_i = ln( x_i / g(x) )        g(x) = geometric mean of all parts

- Produces **D** coordinates (one per part), and every row sums to zero.
- Each coordinate maps back to one named part, so it is the most directly
  interpretable of the three.
- The zero-sum constraint makes the basis **singular** - fine as regression
  targets here (each coordinate is modelled separately), but not suitable for
  methods that need a full-rank covariance matrix.

**ALR - Additive Log-Ratio**

    alr(x)_i = ln( x_i / x_ref )       for the D-1 non-reference parts

- Produces **D-1** coordinates; one part is chosen as the reference
  (PARACUDA-NG uses the **last** part by default, `ref=-1`).
- Simplest and easy to read: each coordinate is "log of this part relative to
  the reference part".
- The geometry is **oblique** - distances and angles are distorted, and your
  results depend on which part you made the reference.

**ILR - Isometric Log-Ratio**  *(statistically preferred default)*

    ilr(x) = clr(x) @ V                V = Helmert orthonormal contrast basis

- Produces **D-1** coordinates via an orthonormal rotation of the CLR.
- **Isometric**: preserves distances and angles, and the coordinates are
  orthogonal with a full-rank covariance - the cleanest statistical choice.
- Trade-off: individual coordinates are *balances* between groups of parts, so
  they have no one-to-one meaning as single parts.  You interpret the
  back-transformed percentages, not the coordinates.

**Which to choose:**
- **ILR** - the safe default; use it unless you need per-coordinate meaning.
- **CLR** - when you want each coordinate to correspond to a named part.
- **ALR** - when one part is a natural, well-measured reference denominator.
- All three give closed predictions summing to 100 %; they usually score very
  similarly, so pick on interpretability, not on chasing R².

────────────────────────────────────────────────────────────
**HANDLING ZEROS**
────────────────────────────────────────────────────────────
Log-ratios are undefined at zero, so any part measured as exactly 0 is fixed by
**multiplicative zero replacement** (Martín-Fernández et al., 2003) before the
logs are taken:

- each zero becomes a small `delta`, defaulting to
  **65 % of the smallest positive proportion** in the data (a standard
  rule of thumb)
- the non-zero parts are scaled down so the row still sums to 1, which
  **preserves the ratios between the observed parts**

This is automatic - you do not configure it.  But a dataset with many true
zeros is weak evidence for a compositional model; consider merging rare parts
into an "other" category first.

────────────────────────────────────────────────────────────
**HOW TO RUN IT**
────────────────────────────────────────────────────────────
1. Load your data and select **2 or more** properties that are parts of the
   same whole (Step ① Data → Select Properties).
2. Set the **"Compositional (log-ratio)"** drop-down (just under the property
   selector) to **CLR**, **ALR** or **ILR**.  Leaving it on "None" runs the
   ordinary independent-model path instead.
3. Configure spectral range, resampling, preprocessing and model type exactly
   as normal - the compositional path uses the same settings.
4. Click **Run**.

**What PARACUDA-NG does internally:**
- Keeps only rows where the spectra and every part are finite, no part is
  negative, and the parts sum to more than zero (needs **≥ 10** such samples).
- Closes the selected parts to 100 %, then maps them to log-ratio coordinates.
- Fits **one model per coordinate** - your chosen model type, your chosen
  parameters - all sharing the SAME train/test split and the same fitted
  `scaler_X`, so the coordinates stay mutually consistent.  Each coordinate
  gets its own target scaler.
- Back-transforms the test predictions to a closed composition.
- Scores **per part on the original percentage scale** (R² and RMSE), so the
  numbers are directly comparable with a non-compositional run.
- Logs "all test predictions sum to 100 % by construction" as confirmation.

────────────────────────────────────────────────────────────
**OUTPUT & MODEL PORTABILITY**
────────────────────────────────────────────────────────────
- An Excel report named `<input>_composition_<transform>_<parts>_<timestamp>`
  with per-part observed vs predicted values and metrics.
- A **compositional model bundle** (`.joblib`) flagged `compositional: True`,
  storing every coordinate model, the coordinate scalers, `scaler_X`, the
  transform name, the part names and total, plus the full preprocessing and
  resampling configuration.
- Load it later with **Load Model** and use **Predict Unknown** on new samples;
  PARACUDA-NG detects the bundle type automatically and returns predictions
  that already sum to 100 %.

────────────────────────────────────────────────────────────
**TIPS & CAVEATS**
────────────────────────────────────────────────────────────
- Only group parts that genuinely belong to one whole.  Sand/silt/clay: yes.
  Sand + pH + organic carbon: no - those are not a composition.
- Your parts do not have to be pre-normalised; they are closed to 100 %
  automatically.  But they must all be non-negative.
- Per-part R² can look slightly lower than an unconstrained per-part model.
  That is expected and honest - the unconstrained model was buying accuracy by
  ignoring a constraint the data actually obey.
- With only 2 parts, ALR and ILR reduce to a single coordinate (a single
  log-ratio), which is a perfectly valid and very stable model.
- Cross-validation, batch processing and hyperparameter tuning run on the
  ordinary (non-compositional) path - set the drop-down back to "None" for those.
"""
            },

            # Preprocessing
            "preprocessing": {
                "keywords": ["preprocess", "transform", "continuum removal", "derivative", "absorbance",
                             "normalization", "filter wavelength", "smoothing", "savitzky", "golay",
                             "savgol", "window length", "polyorder", "spectral outlier", "outlier removal",
                             "none", "preprocessing method", "which preprocessing",
                             "snv", "msc", "what is snv", "what is msc", "snv or msc",
                             "standard normal variate", "multiplicative scatter correction",
                             "scatter correction", "scatter", "msc reference", "snv vs msc",
                             "difference between snv and msc"],
                "title": "Data Preprocessing Options",
                "content": """
**PARACUDA-NG implements exactly 9 preprocessing methods**, selected in
Step ③ Preprocess.  "No Preprocessing" (raw reflectance) is always the 10th
choice and is the correct baseline to compare everything else against.

Every method is applied to the *whole* spectral matrix, and whichever method
you choose is stored inside the saved model, so image and tabular predictions
reproduce it automatically (see **Model Portability**).

────────────────────────────────────────────────────────────
**1. Smoothing (Savitzky-Golay)**
────────────────────────────────────────────────────────────
Fits a low-order polynomial in a sliding window and replaces each point with
the fitted value - suppressing noise while preserving peak height and width
far better than a moving average.

- Parameters:
  • `window_length` (default 11) - number of bands in the window.
    Forced to be ODD; if you enter an even number it is increased by 1.
    Also clamped to at least `polyorder + 1` and at most the band count.
  • `polyorder` (default 2) - polynomial degree inside the window.
- Use for: noisy field/handheld spectrometer data, low-albedo dark targets.
- Rule of thumb: keep `window_length` below the width of your narrowest real
  absorption feature, or you will smooth the signal away along with the noise.
- Applied along the band axis (axis=1) for all samples at once.

────────────────────────────────────────────────────────────
**2. Spectral Outlier Removal**
────────────────────────────────────────────────────────────
Removes whole *samples* (rows), not bands.  Each sample is reduced to its mean
reflectance and the outliers among those means are dropped.

- Parameters:
  • `outlier_method` - 'zscore' (default) or 'iqr'
  • `threshold` (default 3.0)
    - zscore: keep samples with |z| < threshold
    - iqr:    keep samples inside Q1 − threshold·IQR … Q3 + threshold·IQR
- Returns both the filtered spectra AND the keep-mask, so the target vector
  stays aligned with the retained samples.
- Use for: datasets with a few badly-measured spectra (saturation, mis-aimed
  probe, contaminated sample cup).
- Caution: this changes your sample count.  Check how many rows were removed
  before trusting a big jump in R².

────────────────────────────────────────────────────────────
**3. Continuum Removal**
────────────────────────────────────────────────────────────
Divides each spectrum by its upper convex hull, normalising absorption depth
onto a 0-1 scale so features can be compared between samples of different
overall brightness.

- Implementation: a running forward/backward maximum builds the hull, then
  `spectrum / (hull + 1e-10)` (the epsilon prevents division by zero).
- Use for: reflectance spectroscopy where you care about *absorption depth*
  (clay OH features near 2200 nm, carbonate near 2340 nm).
- Good for: isolating absorption bands; removing brightness/albedo effects.

────────────────────────────────────────────────────────────
**4. First Derivative**
────────────────────────────────────────────────────────────
`np.gradient` along the band axis.

- Removes additive baseline shifts (constant offsets vanish).
- Enhances slope changes and shoulder features.
- Use for: reducing multiplicative/additive scatter effects.
- Caution: amplifies noise - pair it with Smoothing if your data is noisy.

────────────────────────────────────────────────────────────
**5. Second Derivative**
────────────────────────────────────────────────────────────
`np.gradient` applied twice.

- Removes linear baselines (both offset and slope vanish).
- Sharpens and separates overlapping peaks; absorption minima become clear.
- Use for: resolving overlapping features, fine feature detection.
- Caution: noise amplification is much stronger than the first derivative.
  Smoothing first is strongly recommended.

────────────────────────────────────────────────────────────
**6. Absorbance**
────────────────────────────────────────────────────────────
Converts reflectance to apparent absorbance: **A = −log10(R)**

- Values are floored at 1e-10 before the log, so zero or negative reflectance
  cannot produce −inf / NaN.
- Use for: direct comparison with laboratory transmission spectra, and to
  linearise the relationship between concentration and signal
  (Beer-Lambert-like behaviour).

────────────────────────────────────────────────────────────
**7. Baseline Correction**
────────────────────────────────────────────────────────────
Subtracts an estimated background using **lower-envelope (rubber-band)**
anchors - deliberately NOT peak anchors, so absorption features are enhanced
rather than flattened.

- Parameters:
  • `baseline_method` - 'linear' (default) or 'polynomial'
  • `degree` (polynomial only, default 3)
- linear: endpoints are the minimum of the first and last 5% of bands, so a
  strong peak at the very first or last band cannot inflate the baseline.
  Fully vectorised across samples.
- polynomial: fits a degree-`d` polynomial through per-segment local minima
  (at least 6 anchors); `d` is clamped to `n_anchors − 1`.
- In both cases the baseline is clamped so it never exceeds the spectrum, and
  no extra global shift is applied (which would inflate variance).
- See the dedicated **Baseline Correction** topic for worked detail.

────────────────────────────────────────────────────────────
**8. Standard Normal Variate**
────────────────────────────────────────────────────────────
The classic scatter correction (Barnes, Dhanoa & Lister, 1989), often written
"SNV" in the literature.  Each spectrum is centred on its own mean and divided
by its own standard deviation:

    (x - mean(x)) / sd(x)

- Removes **additive** baseline offsets and **multiplicative** scatter in one
  step - exactly the effects that particle size and surface roughness impose on
  diffuse reflectance.
- Row-wise and **stateless**: every spectrum is corrected using only itself, so
  nothing is fitted and new samples behave identically to training samples.
  That makes it the safest scatter correction for model portability.
- Afterwards every spectrum has mean 0 and sd 1, so absolute reflectance level
  is gone by design - only band-to-band *shape* remains.
- sd is the sample standard deviation (ddof = 1); a perfectly flat spectrum is
  left centred rather than divided by zero.
- Use for: soil and powder spectroscopy, handheld/contact probe data, anything
  where sample presentation varies between measurements.

────────────────────────────────────────────────────────────
**9. Multiplicative Scatter Correction**
────────────────────────────────────────────────────────────
Geladi, MacDougall & Martens (1985), often written "MSC" in the literature.
Each spectrum is least-squares fitted against a **reference** spectrum and then
rescaled onto it:

    x = a * ref + b        ->        (x - b) / a

- Corrects the same additive + multiplicative scatter as Standard Normal
  Variate, but relative to a common reference rather than to each spectrum's own
  statistics, so the corrected spectra keep the reference's physical scale.
- **This is a fitted transform.**  PARACUDA-NG uses the mean of the TRAINING
  spectra as the reference and stores it inside the saved model, so predictions
  on new samples or on an image are corrected against the same baseline the
  model was fitted on.  This is the difference that matters: correcting new data
  against *its own* mean would silently shift it relative to the calibration.
- If a model is applied to data on a different band grid it refuses rather
  than guessing - the stored reference has a fixed band count.
- Spectra whose fitted slope is ~0 (no reference shape at all) are left
  untouched instead of being divided by a near-zero number.

**Standard Normal Variate or Multiplicative Scatter Correction?**
- They usually perform within noise of each other on the same data.
- **Standard Normal Variate** needs no reference, so it is simpler and more
  portable; prefer it unless you have a reason not to.
- **Multiplicative Scatter Correction** keeps results on the reference's scale,
  which some workflows want for interpretability, and can be slightly better
  when the calibration set is a good representation of everything the model will
  ever see.
- Both are alternatives to Continuum Removal, not additions to it - pick one
  scatter correction, then compare with **Find Best Preprocessing**.

════════════════════════════════════════════════════════════
**Recommended ordering / combinations**
════════════════════════════════════════════════════════════
PARACUDA-NG applies ONE method per run.  To compare them systematically use
**Find Best Preprocessing** (Step ⑥ Execution), which trains the same model
under every method and ranks the results.

Typical picks:
- Soil organic carbon, clay (VIS-NIR-SWIR) → Standard Normal Variate,
                                             Continuum Removal or 1st Derivative
- Variable sample presentation / packing   → Standard Normal Variate, or
                                             Multiplicative Scatter Correction
- Noisy handheld data                      → Smoothing, then 1st Derivative
- Overlapping mineral features             → 2nd Derivative
- Lab-style quantitative work              → Absorbance
- Strong instrument drift / sloping background → Baseline Correction

════════════════════════════════════════════════════════════
**Related steps that are NOT preprocessing methods**
════════════════════════════════════════════════════════════
**Wavelength Filtering (Step ② Spectral)**
- Restrict the modelled range, e.g. 400-2500 nm for VIS-NIR-SWIR.
- "Excluded bands" cuts specific sub-ranges without discarding the rest.
  Built-in suggestions: water vapour 1350-1450 and 1800-1960 nm; noisy
  detector edges 350-400 and 2450-2500 nm.

**Missing-Data Handling (Step ① Data)** - 5 strategies;
see **Missing Data Handling**.

**Spectral Resampling (Step ④ Resampling)** - 7 methods onto a uniform grid
or a named satellite sensor; see **Resampling & Sensor Bands**.

**Best Practices:**
- Always run "None" first - it is your honest baseline.
- Change one thing at a time; preprocessing interacts strongly with the model.
- A method that helps PLS-R may hurt Random Forest - re-check per model.
- If preprocessing improves training R² but not test R², you are fitting
  noise, not signal (see **Overfitting Detection**).
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
                "keywords": ["image", "raster", "tiff", "geotiff", "multispectral", "hyperspectral",
                             "predict image", "map",
                             "envi", "bil", "bip", "bsq", "erdas", "img format", "nitf", "background mask",
                             "no-data", "nodata", "resampled cube", "export resampled",
                             "chunk", "chunk_pixels", "memory", "large image", "large scene", "out of memory",
                             "gain", "offset", "reflectance scale factor", "hdr", "sidecar", "header",
                             "min_band_valid_frac", "min_pixel_valid_frac", "saturation", "georeferencing"],
                "title": "Multispectral & Hyperspectral Image Processing",
                "content": """
**Applying Models to Images (Step ⑦ Apply → Image Processing):**

**Requirements:**
1. Trained model must be loaded or created
2. Image must be a multi-band geospatial raster
3. Bands do NOT need to match the training grid exactly - the image is
   automatically resampled onto the model's input grid before prediction
   (see 'What Happens' below)

**Supported Formats (anything except plain photos):**
- GeoTIFF / TIFF (.tif, .tiff)
- ENVI with a .hdr header (.dat, .bil, .bip, .bsq, .bin, .raw)
- ERDAS Imagine (.img)
- NITF (.ntf, .nitf), PCIDSK (.pix)
- Ordinary photos (.jpg, .png, etc.) are rejected - they carry no spectral bands

**Steps:**
1. Train a model or load a saved model
2. Check "Apply on Image", click "Load" and select your image
3. (Optional) tick "Export resampled cube" to save the exact spectra the
   model saw - useful to sanity-check resampling before trusting predictions
4. Click "Predict" - a progress bar reports resampling / prediction / saving
5. Click "View" to preview the predicted map, or open the saved file in GIS software

**What Happens:**
- The image's own per-band wavelengths are read from its header (ENVI tags or
  sidecar .hdr) when present; otherwise a range is assumed and you're warned
- Bands are resampled onto the model's input grid: the model's resampled grid
  if it was trained with resampling, otherwise the training Excel's wavelengths
  - using the SAME resampling method saved with the model
- Same preprocessing as training is applied automatically, then scaled and predicted

**Background / No-Data Masking:**
- "Mask background / no-data pixels" (on by default) skips pixels that are
  the header's no-data value, all-zero/negative, NaN/Inf, or extreme
  (saturated) - these are written as no-data in the output, never predicted
- This prevents nonsense values (huge/negative/NaN) showing up over image
  borders, clouds, or sensor gaps

**Validating Resampling:**
- "Export resampled cube" writes `<input-name>_resampled.<input-format>`
  next to the prediction - same driver, band interleave (BIL/BIP/BSQ) and a
  correct header with the model's band wavelengths, so you can open it in
  GIS/spectral-viewer software and confirm the spectra look right

**Reading Wavelengths & Radiometry:**
- Wavelengths are taken, in order of preference, from: the dataset's own band
  metadata, an ENVI `wavelength` tag, or a sidecar `.hdr` next to the image
- `wavelength units` is honoured, so a header in micrometres is converted to
  nanometres automatically (see **Wavelength Units**)
- `fwhm` is read when present and reused for SRF-based resampling
- Per-band `gain`/`offset` (ENVI `reflectance scale factor`, `data gain values`,
  `data offset values`) are applied so integer-stored cubes become real
  reflectance before preprocessing

**Memory Handling (large scenes):**
- A full scene (e.g. EnMAP 1195×1150×224) cannot be pushed through
  resample → preprocess → scale as one float64 matrix
- Spectra are processed in blocks of `chunk_pixels` (default 100 000 pixels);
  only the small per-chunk temporaries are float64, persistent arrays are
  float32
- This keeps peak memory roughly constant regardless of scene size

**Validity Thresholds:**
- `min_band_valid_frac` (default 0.5) - a band must be valid across at least
  this fraction of pixels to be used
- `min_pixel_valid_frac` (default 0.5) - a pixel must have at least this
  fraction of valid bands to be predicted; otherwise it is written as no-data

**Output:**
- Predicted property map, written in the SAME format as the input image
  (e.g. ENVI in → ENVI + .hdr out), with georeferencing preserved
- A clean, single-band header (no leftover multi-band spectral metadata)
- Optional resampled cube keeps the input's interleave (BIL/BIP/BSQ) and
  carries the model's band centres and FWHMs in its header
- Prediction statistics (min / max / mean / std over valid pixels) are
  reported so you can sanity-check the value range immediately

**Tips:**
- Use models with good CV scores - a weak model produces a confident-looking
  but meaningless map
- The image's spectral range should overlap the training range; bands outside
  it are dropped, and if too few remain the prediction is refused
- Check that image and training data are the same quantity (both reflectance,
  both scaled the same way) - this is the most common source of nonsense maps
- Consider computational time for large images; the progress bar reports the
  resampling / prediction / saving phases separately
- Visualize with different colormaps, and always inspect the histogram
"""
            },
            
            # Data Distribution
            "data_distribution": {
                "keywords": ["distribution", "data distribution", "histogram", "boxplot", "box plot",
                             "skew", "skewed", "skewness", "kurtosis", "normality", "normal distribution",
                             "spread", "outliers in data", "iqr", "quartile", "coefficient of variation",
                             "cv%", "bimodal", "check data quality", "is my data ok",
                             "should i convert", "before training", "explore data"],
                "title": "Data Distribution - Checking Your Data Before You Commit",
                "content": """
**Where:** Step ① Data → **📈 Show Data Distribution** (enabled as soon as a file
is loaded), and the same button in the **Data Converter** right after Load.

**Why it exists:** a property that is strongly skewed, almost constant, mostly
missing or effectively categorical will not model well no matter which
algorithm, preprocessing or resampling you choose afterwards.  Finding that out
takes seconds here; finding it out after a batch run takes an hour.  In the
Data Converter it comes even earlier - if the data is unusable you may not want
to convert it at all.

════════════════════════════════════════════════════════════
**WHAT YOU SEE**
════════════════════════════════════════════════════════════
Two tabs, for every numeric property column:

**Plots** - per property, a histogram with
- the **mean** (red line) and **median** (green dashed line): when these separate
  clearly, the distribution is skewed;
- a **normal reference curve** scaled to the data, so departures from symmetry
  are obvious at a glance;
- a **boxplot** underneath sharing the same axis, showing the quartiles and every
  1.5xIQR outlier as a dot.

The panel title is colour coded: green = fine, amber = worth a look,
red = a real problem.

**Statistics & Findings** - the numbers behind the picture:
n valid / total, missing count and %, mean, median, std, min, Q1, Q3, max,
skewness, kurtosis, coefficient of variation, and the IQR-outlier count -
followed by plain-language findings.

Wavelength columns are deliberately excluded: a spectrum is not a property
distribution, and summarising 2000 band columns individually is meaningless.

════════════════════════════════════════════════════════════
**WHAT IT FLAGS, AND WHAT TO DO**
════════════════════════════════════════════════════════════
**Strong skew (|skewness| >= 1.0)** - red
A long tail on one side.  RMSE will be dominated by the tail and most models
will underpredict the extremes.  *Fix:* log or square-root transform the
property before modelling, or model the transformed value and back-transform.

**Moderate skew (0.5 <= |skewness| < 1.0)** - amber
Usually tolerable; worth noting when you report results.

**Heavy tails (excess kurtosis > 3)** - amber
A handful of extreme samples will dominate the fit.  *Fix:* check whether they
are real or measurement errors; consider the Huber Regressor, which is robust
to them.

**Many IQR outliers (> 5% of samples)** - amber
*Fix:* inspect them; enable **Target Variable Outlier Removal** in Step ③ if
they are genuinely bad measurements.

**Very little spread (CV < 2%)** - red
An almost constant property.  R² becomes meaningless and RPD will be poor no
matter what you do, because there is nothing to predict.  *Fix:* get a more
varied sample set.

**Few distinct values (<= 5 levels)** - amber
The property looks categorical rather than continuous; regression is probably
the wrong tool.

**Missing values** - amber above 0%, red above 20%
*Fix:* choose a strategy in **Missing Data Handling** (Step ① Data).

**Too few samples (< 20 valid, red below 10)**
30+ is a practical minimum and 50+ is comfortable; below that, every metric is
unstable and cross-validation is the only result worth quoting.

════════════════════════════════════════════════════════════
**EXPORTING**
════════════════════════════════════════════════════════════
- **Save Plot (300 DPI)** - PNG or PDF, publication ready.
- **Save Statistics** - Excel (one row per property, all statistics plus the
  findings text) or plain text.

**Tip:** run this before *and* after choosing your properties.  Selecting
properties narrows the view to just those columns; with none selected it shows
every numeric candidate in the file.
"""
            },

            # Data Converter
            "data_converter": {
                "keywords": ["data converter", "converter", "convert", "convert file", "reformat",
                             "wrong format", "transpose", "transposed", "col-wise", "row-wise",
                             "column-wise", "orientation", "instrument export", "asd", "opus",
                             "ftir", "wavenumber", "sheet", "worksheet", "csv format",
                             "names column", "property columns", "prepare data", "my file won't load"],
                "title": "Data Converter - Reformatting Arbitrary Instrument Files",
                "content": """
**What it is:** a standalone three-tab wizard that takes an arbitrary Excel /
CSV export from any spectrometer and rewrites it into the layout PARACUDA-NG
expects.  Open it from **Tools → Data Converter** (desktop / QGIS panel) or
run it on its own:

    paracuda-converter                         # installed console script
    python -m paracuda_ng.utils.data_converter # PyPI package
    python -m utils.data_converter             # source checkout (repo root)

════════════════════════════════════════════════════════════
**THE TARGET FORMAT**
════════════════════════════════════════════════════════════
    Names | Prop1 | ... | PropN | WL1 | WL2 | ... | WLM

- Column 1 is the sample identifier, literally named `Names`
- Then zero or more property (target) columns - Sand, Silt, SOC, …
- Then the spectral columns, whose HEADERS are the wavelengths themselves
- A file is recognised as already-converted when it has ≥ 6 columns, the first
  column is called `Names`, and ≥ 5 wavelength-like headers are present.  In
  that case the converter tells you no conversion is needed.

════════════════════════════════════════════════════════════
**SUPPORTED INPUT LAYOUTS**
════════════════════════════════════════════════════════════
**1. Row-wise** - samples in rows, wavelengths as column headers

    Names | Sand | Silt | 425 | 480 | 545 | ...
    S1    |  45  |  30  | .45 | .48 | .51 | ...

**2. Col-wise (transposed)** - samples in columns, wavelengths down the first
column.  Common for spectral instrument exports.  Property ROWS may
appear before the wavelength rows:

    Row_label | S1  | S2
    Sand      |  45 |  32
    Silt      |  30 |  40
    425.0     | .45 | .52
    480.0     | .48 | .55

The converter splits label rows into "wavelength-like" and "property-like",
transposes the spectral block, transposes the property block, and merges the
two on the sample-name column.

════════════════════════════════════════════════════════════
**AUTO-DETECTION (all of it is overridable)**
════════════════════════════════════════════════════════════
**Orientation** - decided by which axis carries more wavelength-like numbers.

**Wavelength values** - a number counts as a wavelength when it is
> 1.0 and falls in the nm range (350-14100) or the µm range (≤ 14.5).
Values ≤ 1.0 are treated as reflectance / emissivity and never as wavelengths,
which is what stops a reflectance column being mistaken for a band list.

**Wavelength label column** - matched against `wavelength`, `wavelengths`,
`wl`, `wave`, `nm`, `um`, plus FTIR wavenumber synonyms (`wavenumber`,
`wavenum`, `cm-1`, `cm^-1`, `cm⁻¹`) for Bruker/OPUS exports.

**Name column** - matched against `name(s)`, `sample(s)`, `sample_id`, `id`,
`label(s)`, `site`, `site_id`, and soil-science aliases `soil`, `soil_id`,
`soil_name`, `soil_no`, `profile`, `profile_id`, `pedon`, `pedon_id`.
If nothing matches, names are generated as `S1, S2, S3, …` (the prefix is
configurable via `auto_name_prefix`).

**Property columns** - whatever is left over: not the name column, not a
wavelength column.

════════════════════════════════════════════════════════════
**WAVELENGTH UNITS**
════════════════════════════════════════════════════════════
- `auto` (default) - a series is treated as micrometres only when every value
  is > 1.0, ≤ 14.5, and none reaches 350; otherwise nanometres
- `nm` / `um` - force the interpretation when the heuristic guesses wrong
- The wizard asks you to confirm the detected unit before writing, because a
  wrong unit silently ruins every later resampling step

════════════════════════════════════════════════════════════
**FILE SUPPORT**
════════════════════════════════════════════════════════════
- Excel: `.xlsx`, `.xls`, `.xlsm`, and OpenDocument `.ods`
- CSV: `.csv` - read as UTF-8, automatically retried as cp1252 (Windows-1252)
  if that fails, so European-encoded exports load without manual fixing
- Multi-sheet workbooks: pick the worksheet from a drop-down before converting

════════════════════════════════════════════════════════════
**USING IT (three tabs)**
════════════════════════════════════════════════════════════
1. **Load** - choose the file (and sheet).  A preview appears with the detected
   orientation, name column, property columns and band count.
   Then click **📈 Show Data Distribution** to check the property distributions
   *before* converting - if a property is strongly skewed, near-constant or
   mostly missing, you may want to fix the source data rather than convert it.
   See the **Data Distribution** topic.  (For a transposed/col-wise file the
   view automatically summarises the property ROWS instead of the columns.)
2. **Configure** - override anything the detector got wrong: orientation,
   which column holds the names, which columns are properties, and the
   wavelength unit.
3. **Convert & Save** - review the warnings, then write
   `<original-name>_paracuda.xlsx`.  Load that file back in Step ① Data.

════════════════════════════════════════════════════════════
**WARNINGS YOU MAY SEE**
════════════════════════════════════════════════════════════
- "File is already in PARACUDA-NG format" - nothing to do
- "No wavelength rows found in transposed file" - the whole frame was treated
  as spectral data; check the label column
- Duplicate sample names, non-numeric spectral cells, or bands that fall
  outside the valid range are reported so you can fix the source file

**Tip:** the converter only reshapes data - it never resamples, smooths or
rescales.  All of that happens later in PARACUDA-NG itself, so the converted
file stays a faithful copy of your measurements.
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
                "keywords": ["interpret", "understand", "meaning", "r2", "rmse", "results", "metrics",
                             "scores", "rpd", "rmsep", "nrmsep", "normalized rmse", "normalised rmse",
                             "ratio of performance to deviation", "root mean square error of prediction",
                             "which metric", "how good is my model", "quality bands",
                             "what does rpd mean", "what is rpd", "rpd value", "good rpd",
                             "what does rmsep mean", "what is rmsep", "normalized rmsep",
                             "error metrics", "accuracy metrics", "model performance"],
                "title": "Interpreting Results",
                "content": """
**Understanding Your Results:**

PARACUDA-NG reports five accuracy measures for every model, on both the training
and the held-out (test) split, plus cross-validation when enabled.  R² alone is
not enough: it depends on how much the property varies in YOUR dataset, so the
same model can look excellent on a diverse set and poor on a narrow one.

**R² (R-Squared)**
- Range: 0 to 1 (higher is better)
- 0.9-1.0: Excellent prediction
- 0.7-0.9: Good prediction
- 0.5-0.7: Moderate prediction
- <0.5: Poor prediction (reconsider model/data)
- Can go NEGATIVE, which means the model is worse than predicting the mean.

**RMSE (Root Mean Square Error)**
- Lower is better
- Same units as your property
- Compare to property's range
- Examples:
  - Clay %: RMSE < 5% is good
  - SOC %: RMSE < 0.5% is good

**RMSEP (Root Mean Square Error of Prediction)**
- The RMSE computed on the **held-out test split** - i.e. on samples the model
  never saw.  It is reported under its own name because that is the number the
  chemometrics literature quotes and the one you should put in a paper.
- Train RMSE << RMSEP is the classic overfitting signature.

**nRMSEP (Normalized RMSEP), shown as a percentage**
- nRMSEP = 100 x RMSEP / (max(observed) - min(observed))
- Expresses the error as a share of the observed **range**, which makes it
  comparable between properties measured in different units - you can finally
  say whether clay is predicted better than pH.
- Rough reading: < 10% excellent, 10-20% good, 20-30% fair, > 30% poor.
- Range normalization is used (rather than dividing by the mean) because it
  stays well defined for properties whose mean sits near zero.  The
  mean-normalized variant is also computed and stored in the exported workbook.

**RPD (Ratio of Performance to Deviation)**
- RPD = SD(observed) / RMSE - how big the natural spread of the property is
  compared with the model's error.
- The standard chemometric verdict on whether a calibration is *useful*:
  - RPD < 1.0  very poor  - the model is worse than just guessing the mean
  - 1.0 - 1.4  poor       - not usable
  - 1.4 - 1.8  fair       - rough screening only
  - 1.8 - 2.0  good       - usable for approximate quantification
  - 2.0 - 2.5  very good  - quantitative prediction
  - > 2.5      excellent  - analytical quality
- PARACUDA-NG prints the band name next to the number, e.g. `RPD = 2.31 (very good)`.
- RPD is undefined (shown as `-` / nan) when RMSE is 0 or there are fewer than
  two samples.
- **Why RPD matters:** a high R² on a dataset with a huge spread can still mean a
  useless model in absolute terms, and a modest R² on a narrow dataset can still
  be a genuinely precise one. RPD normalises exactly that.

**Reading them together**
- High R², high RPD, low nRMSEP  -> a genuinely good model.
- High R² but RPD < 1.8          -> your samples are just very spread out;
                                    the model is not as good as R² suggests.
- Low R² but RPD ~ 2             -> a narrow property range; the model may
                                    still be practically useful.
- Good test scores, bad CV scores -> the single split was lucky; trust the CV.

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
- Load data into PARACUDA-NG
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
- Apply to multispectral / hyperspectral images
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
                "title": "Tabular Prediction - Predicting Unknown Samples",
                "content": """
**What is Tabular Prediction?**
Apply a trained model to new spectral data (Excel or CSV) to predict soil properties
for samples that do not have lab-measured values.

**Workflow:**
1. Train a model (or load a saved one with 'Load Model')
2. Open the **Tabular Prediction** section in the control panel
3. Click **'Load Excel/CSV'** - select your unknown samples file
4. Click **'Check Data Info'** to verify wavelengths were detected correctly
5. Click **'Predict Unknown'** - results are saved to Excel automatically

**Input File Requirements:**
- Same format as training data (wavelengths as column headers)
- Wavelength columns must overlap with the model's training wavelengths
- Wavelength units (nm or μm) are detected automatically
- Non-wavelength columns (sample IDs, etc.) are preserved in output

**Important Notes:**
- The preprocessing applied during training is automatically re-applied to the unknown data
- No need to manually match preprocessing settings - they are saved inside the model
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

**How PARACUDA-NG Detects It:**
PARACUDA-NG checks several measures together and only flags overfitting when the
model genuinely learned the training data AND generalises materially worse.
ALL of these must hold:
- Training R² ≥ 0.60 - the model actually fit the training set. (A model with a
  low/negative Training R² is *underfitting*, not overfitting, and is not flagged.)
- Training R² − Test R² > 0.15 - a sizeable absolute drop from train to test.
- The drop is > 25% of the training skill - a large *relative* degradation.

Two extra measures raise the severity from *mild* to *strong* when present:
- Training R² − CV R² > 0.15 (cross-validation confirms the gap).
- Test RMSE > 1.3 × Train RMSE.

**Visual Cues in Batch Reports:**
- Normal bars: blue/green fill
- Overfitting bars: **red hatched fill** - easy to spot in comparison charts
- The batch summary table shows an `Overfit` column (ok / mild / strong)
- A warning is written to the status log listing exactly which measures fired

**What To Do:**
- Reduce model complexity (fewer components, shallower trees)
- Try stronger regularisation (higher alpha for Ridge/Lasso)
- Increase dataset size if possible
- Use simpler preprocessing
- For PLS-R: reduce the number of components or enable 'Optimize Components'

**Note:**
These are heuristics - always inspect scatter plots and residuals as well.
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
PARACUDA-NG uses lower-envelope (valley) anchors - not peak anchors.
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
PARACUDA-NG automatically detects whether your column headers are in nanometers (nm)
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

            # Check Spectral Integrity (permutation + mixing tests)
            "data_randomization": {
                "keywords": ["randomization", "permutation test", "label permutation",
                             "data integrity", "mixing test", "spectral mixing",
                             "statistical significance", "p-value", "shuffle labels"],
                "title": "Check Spectral Integrity - Label Permutation & Mixing Tests",
                "content": """
**Why Test Data Integrity?**
Before trusting a model, verify that spectral patterns genuinely predict your target property
and aren't an artefact of chance, data leakage, or random correlation.

**Available Tests (Data Tools → Check Spectral Integrity):**

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
                "title": "Spectral Harmonization - Transfer Functions",
                "content": """
**What is Spectral Harmonization?**
Converting spectra from one sensor/instrument to match another sensor's response,
enabling models trained on one sensor to be applied to data from a different sensor.

**Example Use Cases:**
- Field spectrometer (ASD) → Satellite (EMIT, EnMAP, Sentinel-2)
- Lab scanner → Airborne hyperspectral
- Old instrument → New instrument after calibration drift

**Two-Tab Workflow (Data Tools → Spectral Harmonization):**

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
                "title": "Model Portability - Saving and Loading Models",
                "content": """
**What's Stored in a Saved Model (.joblib):**
- The fitted regression model + PCA component (if used)
- Feature scalers (StandardScaler for X and y)
- Training metadata: property name, model type, training wavelengths
- Preprocessing method and all its parameters
- **Full resampling configuration**: whether resampling was on, the method,
  target sensor/spacing, per-band FWHMs, custom FWHM/SRF tables, exclude ranges
- **The model's exact input band grid** (`training_input_wavelengths` /
  `..._fwhms`) - the resampled grid if resampling was used, otherwise the
  training Excel's wavelengths - so unknown data or images can always be
  resampled onto precisely what the model expects, even on a different
  original band grid

**Why This Matters:**
When you load a model and apply it to new data (tabular prediction or image
prediction), the exact same preprocessing AND resampling used during training
is automatically re-applied. You don't need to manually remember or
re-configure any settings - even if the new data's own wavelength grid
differs from the model's.

**Loading a Model:**
1. Click 'Load Model' button
2. Select the .joblib file
3. Model is ready to use - preprocessing and resampling auto-configured
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
grid - either a uniform spacing or a named satellite sensor (e.g. Sentinel-2,
Landsat-8, EnMAP) - so a model can be trained on, or applied to, data from a
different instrument. Configured in Step ④ Resampling.

**The 7 Methods (dropdown order):**

*Interpolation family* - reproject the curve onto new points, no notion of
instrument bandwidth:
- **Linear Interpolation** - straight line between neighbouring points.
  Simple and safe, but can smear sharp absorption features on a coarse grid.
- **Nearest Neighbour Interpolation** - copies the closest source band
  unchanged. Preserves the true measured value at each point rather than
  averaging/blending it - often better than linear when the target grid is
  coarser than the source, or when you want to avoid inventing intermediate
  values that were never actually measured.
- **Quadratic Interpolation** - smoother curve through 3+ points.
- **Cubic Spline** - smoothest curve through 4+ points; can overshoot near
  sharp peaks.

*Bandwidth-aware family* - integrate the source spectrum under each target
band's actual spectral response, matching how a real sensor measures light
(needs a per-band FWHM):
- **Gaussian SRF** - analytic Gaussian response shaped by each band's FWHM.
  The physically realistic default for satellite/airborne sensors.
- **Empirical SRF** - integrates against an uploaded real per-band response
  curve (CSV), falling back to Gaussian where no curve is supplied. Most
  accurate when you have the sensor's actual spectral response function.
- **Band Averaging** - flat (tophat) average over each band's FWHM window.

**Which Should I Pick?**
- Matching a named sensor → Gaussian SRF (or Empirical SRF if you have the
  real response curves) - this is what the sensor physically measures
- Simple uniform down/up-sampling, want exact measured values → Nearest
  Neighbour Interpolation
- Simple uniform resampling, want a smooth curve → Linear or Cubic Spline
- Linear interpolation is NOT always correct: it assumes reflectance varies
  smoothly between points, which can blur real absorption features - try
  Nearest Neighbour or a bandwidth-aware method if results look flattened

**Sensor Bands & Custom Grids:**
- Choose a named sensor (its band centres + FWHMs are built in), or "Custom"
  with a uniform spacing (nm)
- **Custom FWHM CSV** - load your own (centre, FWHM) table instead of the
  sensor default
- **Empirical SRF CSV** - load real per-band response curves for the most
  physically accurate integration

**Excluded Bands (Step ② Spectral, "Excluded bands"):**
Cut out specific sub-ranges without discarding the whole spectrum - quick
presets are offered for common water-vapour absorption windows
(1350-1450 nm, 1800-1960 nm) and noisy sensor edges.

**Spectral Binning:**
Reduces band count by keeping one representative band per group (decimation),
applied after resampling - useful to shrink a very dense hyperspectral grid.

**Where This Applies:**
- Training: resampling is applied to the loaded Excel data before model fitting
- Prediction: the SAME method + target grid saved with the model is
  automatically re-applied to unknown tabular data and to images (see
  **Model Portability** and **Multispectral & Hyperspectral Image Processing** topics)
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
training with confusing errors. Right after loading, PARACUDA-NG reports exactly
how much data is missing; you then choose a strategy (Step ① Data).

**The 5 Strategies (dropdown order):**
- **Drop rows with missing** (default) - removes any sample with at least
  one missing value. Safest choice; use when you can afford to lose a few rows.
- **Mean imputation** - fills each missing cell with that wavelength
  column's mean across all samples.
- **Median imputation** - same idea, using the median (more robust to
  outliers than the mean).
- **Spectral interpolation** - fills a gap using the neighbouring
  wavelengths in the SAME row (linear interpolation along the spectrum);
  falls back to the column mean if a row has too few valid points to
  interpolate from.
- **Fill with zero** - last resort; can distort the spectral shape and bias
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
Instead of manually guessing model parameters, PARACUDA-NG can search for good
values automatically using Optuna (Bayesian/TPE search), in Step ④ Model.

**How to Use:**
1. Check "⚙ Tune Hyperparameters (Optuna)"
2. Set **Trials** - how many parameter combinations to try (default 30;
   more trials = better search but longer runtime)
3. Set **CV folds** - how many folds each trial is validated on internally
   (default 3) before scoring it
4. Train as normal - the winning/selected model is re-fit using the best
   parameters Optuna found

**What Happens:**
- Each trial samples a candidate set of parameters for the current model
- The candidate is scored by internal cross-validation (not your main CV)
- After all trials, the best-scoring parameters are used to fit the final model
- Works for both Single-model and Batch (multi-model) mode - each model in a
  batch gets its own independent tuning search

**Tips:**
- Start with the default 30 trials; increase for a more thorough search if
  you have time (runtime scales roughly linearly with trials)
- Tuning searches a sensible parameter range per model - for PLS-R this
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
A live diagram - docked below the step wizard - that shows the exact pipeline
you've configured: Data → Property → Spectral domain → Excluded bands →
Resampling → Preprocessing → Model → Validation → Apply.

**How It Works:**
- Visible for every wizard step, not just Apply - it updates in real time as
  you change any setting
- On a fresh session or after "🔄 Reset", every stage is faded (dashed
  border) - cards light up (solid blue border) only once you actually
  configure that step, or after a model is trained
- The compact docked view shows short titles; click "🔍 Large" for full
  per-stage detail (exact values, methods, parameters) in a pop-up window
- "💾 300 DPI" exports the detailed chart as a publication-ready PNG/PDF/SVG
  - useful for documenting exactly how a model was built in a report or paper

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
        return """Welcome to PARACUDA-NG Help Assistant!

I can help you with:

🔰 GETTING STARTED & CORE WORKFLOW
• Getting Started - Learn the basics
• Workflow - Complete analysis procedure
• Loading Data - How to import your spectral data
• Data Distribution - Check skew/outliers BEFORE you commit
• Data Converter - Reformat any instrument export to PARACUDA-NG layout
• Best Practices - Tips for optimal results

📈 MODELS & TRAINING
• Choosing Models - 13 algorithms (incl. neural network) and every parameter
• Compositional Modelling - CLR/ALR/ILR for parts that sum to 100%
• Preprocessing - All 9 methods, incl. the two scatter corrections
• Missing Data Handling - Dealing with empty/invalid spectral cells
• Parameters - Model settings
• Hyperparameter Tuning - Automatic search with Optuna
• Overfitting Detection - Recognise and fix overfitting

✅ VALIDATION & ANALYSIS
• Cross-Validation - Understanding validation methods
• Batch Processing - Running multiple properties/models
• Check Spectral Integrity - Label permutation and mixing integrity tests
• Model Development Flow - Live pipeline overview & 300-DPI export

🔗 ADVANCED / MULTI-SENSOR
• Tabular Prediction - Apply model to new unknown samples (Excel/CSV)
• Resampling & Sensor Bands - 7 methods incl. Nearest Neighbour & Gaussian SRF
• Spectral Harmonization - Transfer functions across sensors
• Wavelength Units - nm / μm auto-detection and conversion
• Baseline Correction - Background removal from spectra
• Image Processing - Any geospatial raster (GeoTIFF, ENVI, ERDAS, BIL/BIP/BSQ…)

💾 SAVING & LOADING
• Model Portability - Save/load models with preprocessing & resampling
• Saving Results - Exporting your work

🛠 TROUBLESHOOTING
• Troubleshooting - Solving common issues
• Interpreting Results - R², RMSE, RMSEP, nRMSEP and RPD explained

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
- "My file is transposed - how do I convert it?"
- "What does the smoothing window length do?"
- "How do I make sand, silt and clay sum to 100%?"
- "Standard Normal Variate or Multiplicative Scatter Correction?"
- "What does RPD mean and what is a good value?"
- "Is my data too skewed to model?"
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

• Getting Started with PARACUDA-NG
• Loading and Preparing Data
• Data Distribution (skew, outliers, missing values)
• Data Converter (reformatting arbitrary instrument files)
• Missing Data Handling
• Selecting and Comparing Models (incl. Artificial Neural Network)
• Compositional (Log-Ratio) Modelling - CLR / ALR / ILR
• Preprocessing Techniques (incl. Standard Normal Variate and
  Multiplicative Scatter Correction)
• Resampling & Sensor Bands (Nearest Neighbour, Gaussian/Empirical SRF, …)
• Hyperparameter Tuning (Optuna)
• Cross-Validation Methods
• Batch Processing Multiple Properties
• Understanding Parameters
• Tabular Prediction (predicting unknown samples)
• Spectral Harmonization / Transfer Functions
• Check Spectral Integrity (label permutation & mixing tests)
• Baseline Correction
• Wavelength Unit Auto-Detection
• Overfitting Detection
• Processing Multispectral & Hyperspectral Images (GeoTIFF, ENVI, ERDAS, BIL/BIP/BSQ, …)
• Model Development Flow Chart
• Interpreting Results and Metrics
• Model Portability (save/load with preprocessing & resampling)
• Best Practices and Tips

Please rephrase your question or ask about a specific topic!"""
        
        return response
    
    def get_all_topics(self):
        """Return a list of all available help topics"""
        return [topic["title"] for topic in self.knowledge_base.values()]


# ---------------------------------------------------------------------------
# Rendering the knowledge base
# ---------------------------------------------------------------------------
#
# The topic bodies are written in a small, consistent markup: ``**bold**`` and
# `` `code` `` spans, "- " / "• " bullets, "1. " numbered steps, indented
# continuation lines, and rules of box-drawing characters.  Every front end
# needs the same structure out of it, so the parsing lives here rather than in
# any one toolkit's renderer.
#
# The Tkinter renderer used to guess at structure line by line - "no leading
# space, no colon in the first 50 characters and under 60 characters long" was
# read as a heading - which promoted ordinary sentences to titles, left the
# ``**`` and backticks on screen as literal characters, and gave indented
# continuation lines no hanging indent, so every wrapped line fell back to the
# left margin.  Parsing once, properly, fixes all of that in all three versions.

_BULLET_RE = re.compile(r'^(\s*)([-•▸*])\s+(.*)$')
_NUMBER_RE = re.compile(r'^(\s*)(\d+[.)])\s+(.*)$')
_MARK_RE = re.compile(r'^(\s*)([✓✔✗✘⚠❌🔴🟡🟢💡ⓘ])\s+(.*)$')
_RULE_RE = re.compile(r'^\s*([═=─―—_*#]{8,})\s*$')
_INLINE_RE = re.compile(r'(\*\*.+?\*\*|`[^`]+`)')

__all__ = ["HelpAssistant", "parse_help_blocks", "help_blocks_to_html"]


def parse_inline(text):
    """Split a line into ``(text, style)`` spans; style is '', 'bold' or 'code'.

    The markers themselves are dropped - they are formatting instructions, not
    content, and showing them raw was half of what made the help unreadable.
    """
    spans = []
    for piece in _INLINE_RE.split(text):
        if not piece:
            continue
        if piece.startswith('**') and piece.endswith('**') and len(piece) > 4:
            spans.append((piece[2:-2], 'bold'))
        elif piece.startswith('`') and piece.endswith('`') and len(piece) > 2:
            spans.append((piece[1:-1], 'code'))
        else:
            spans.append((piece, ''))
    return spans or [('', '')]


def parse_help_blocks(content):
    """Parse a help response into renderable blocks.

    Returns a list of dicts with:

    ``kind``    'title', 'section', 'rule', 'bullet', 'numbered', 'body'
                or 'blank'
    ``indent``  the source indentation in spaces (a renderer turns this into a
                left margin so wrapped lines line up under their first line)
    ``marker``  the bullet/number/symbol that introduced the line, or ''
    ``spans``   ``[(text, style), ...]`` for the line's content
    """
    lines = (content or "").split('\n')
    blocks = []
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            blocks.append({'kind': 'blank', 'indent': 0, 'marker': '',
                           'spans': []})
            continue

        # A line of rule characters: either the underline of the title above it
        # (already emitted, so drop it) or a section divider in its own right.
        if _RULE_RE.match(line):
            if blocks and blocks[-1]['kind'] in ('title', 'section'):
                continue
            blocks.append({'kind': 'rule', 'indent': 0, 'marker': '',
                           'spans': []})
            continue

        # A heading is a line underlined by rule characters on the next line -
        # structure, not a guess from the line's length.
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        if _RULE_RE.match(nxt):
            kind = 'title' if not blocks else 'section'
            blocks.append({'kind': kind, 'indent': 0, 'marker': '',
                           'spans': parse_inline(line.strip())})
            continue

        stripped = line.strip()
        # A line that is entirely bold is a section heading.
        if (stripped.startswith('**') and stripped.endswith('**')
                and stripped.count('**') == 2):
            blocks.append({'kind': 'section', 'indent': len(line) - len(line.lstrip()),
                           'marker': '', 'spans': [(stripped[2:-2], '')]})
            continue

        for regex, kind in ((_BULLET_RE, 'bullet'), (_NUMBER_RE, 'numbered'),
                            (_MARK_RE, 'bullet')):
            m = regex.match(line)
            if m:
                blocks.append({'kind': kind, 'indent': len(m.group(1)),
                               'marker': m.group(2),
                               'spans': parse_inline(m.group(3))})
                break
        else:
            blocks.append({'kind': 'body',
                           'indent': len(line) - len(line.lstrip()),
                           'marker': '', 'spans': parse_inline(stripped)})
    return blocks


def _escape(text):
    return (text.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;'))


def help_blocks_to_html(content):
    """Render a help response as HTML, for the Qt front end's text browser."""
    out = ['<div style="font-family:Segoe UI,sans-serif; font-size:10pt;">']
    for b in parse_help_blocks(content):
        kind = b['kind']
        if kind == 'blank':
            out.append('<div style="height:6px"></div>')
            continue
        if kind == 'rule':
            out.append('<hr style="border:0; border-top:1px solid #c8d0de;">')
            continue
        body = ''.join(
            f'<b>{_escape(t)}</b>' if s == 'bold' else
            (f'<code style="background:#eef2f8;">{_escape(t)}</code>'
             if s == 'code' else _escape(t))
            for t, s in b['spans'])
        pad = 12 * (b['indent'] // 2)
        if kind == 'title':
            out.append(f'<div style="font-size:13pt; font-weight:bold; '
                       f'color:#1a3a5c; margin:6px 0 4px;">{body}</div>')
        elif kind == 'section':
            out.append(f'<div style="font-size:11pt; font-weight:bold; '
                       f'color:#1a4a7a; margin:8px 0 2px;">{body}</div>')
        elif kind in ('bullet', 'numbered'):
            # Hanging indent: the wrapped remainder lines up under the text,
            # not under the bullet.
            out.append(
                f'<table style="margin-left:{pad + 10}px;" cellpadding="0" '
                f'cellspacing="0"><tr>'
                f'<td style="width:22px; vertical-align:top;">'
                f'{_escape(b["marker"])}</td>'
                f'<td style="vertical-align:top;">{body}</td></tr></table>')
        else:
            out.append(f'<div style="margin-left:{pad}px;">{body}</div>')
    out.append('</div>')
    return '\n'.join(out)
