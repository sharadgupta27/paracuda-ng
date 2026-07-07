"""
Optuna-based hyperparameter tuning for the regression models used by Paracuda.

The GUI first ranks models on their DEFAULT parameters; the user can then tune
the selected / best model here.  Search spaces are adapted from the reference
``Ori_Code_TAU`` ModelFactory but keyed by the exact model names used in the
Paracuda GUI, and every candidate model is built through the shared
``model_training.create_model`` so there is a single source of truth for how a
model is constructed.

@author: Sharad Kumar Gupta
"""
import numpy as np

from models.model_training import create_model

# Heavy libs imported lazily (see utils/lazy_imports.py); Optuna / scikit-learn
# load on the first tuning run rather than at GUI startup.
from utils.lazy_imports import LazyModule, LazyCallable

optuna = LazyModule('optuna')
TPESampler = LazyCallable('optuna.samplers', 'TPESampler')
KFold = LazyCallable('sklearn.model_selection', 'KFold')
cross_val_score = LazyCallable('sklearn.model_selection', 'cross_val_score')

_optuna_quiet = False


def _ensure_optuna_quiet():
    """Silence Optuna logging on first use (deferred so importing this module does
    not import optuna). Progress is reported through the GUI instead."""
    global _optuna_quiet
    if not _optuna_quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _optuna_quiet = True

# Models the GUI exposes but that have no meaningful hyperparameters to tune.
NON_TUNABLE_MODELS = {"Multiple Linear Regression", "PCA"}


def suggest_from_distribution(trial, name, dist):
    """Translate a stored Optuna distribution into a ``trial.suggest_*`` call.

    Mirrors ``Ori_Code_TAU/utils/optuna_utils.suggest_from_distribution``.
    """
    if isinstance(dist, optuna.distributions.FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, log=dist.log, step=dist.step)
    elif isinstance(dist, optuna.distributions.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, log=dist.log, step=dist.step)
    elif isinstance(dist, optuna.distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    raise ValueError(f"Unsupported distribution for {name}")


def get_search_space(model_type, n_features, n_samples):
    """Return the Optuna search space (name -> distribution) for a GUI model.

    Only the tunable hyperparameters are returned; any remaining constructor
    arguments come from the model's current GUI defaults.  ``n_features`` and
    ``n_samples`` bound data-dependent ranges (e.g. PLS components).
    """
    F = optuna.distributions.FloatDistribution
    I = optuna.distributions.IntDistribution
    C = optuna.distributions.CategoricalDistribution

    if model_type == "PLS-R":
        # n_components cannot exceed the number of features or samples.
        hi = int(min(30, n_features, max(2, n_samples - 1)))
        if hi < 2:
            return {}
        return {"n_components": I(2, hi)}

    if model_type == "Ridge":
        return {"alpha": F(1e-2, 1e3, log=True)}

    if model_type == "Lasso":
        return {"alpha": F(1e-3, 1e2, log=True)}

    if model_type == "Elastic Net":
        return {"alpha": F(1e-3, 1e2, log=True),
                "l1_ratio": F(0.05, 1.0, log=False)}

    if model_type == "Huber Regressor":
        return {"alpha": F(1e-5, 1e1, log=True),
                "epsilon": F(1.05, 3.0, log=False)}

    if model_type == "Gradient Boosting":
        return {"learning_rate": F(0.02, 0.3, log=True),
                "n_estimators": I(50, 300, step=25),
                "max_depth": I(2, 6),
                "min_samples_split": I(2, 20),
                "min_samples_leaf": I(1, 20),
                "subsample": F(0.5, 1.0, step=0.1)}

    if model_type == "Random Forest":
        return {"n_estimators": I(100, 400, step=50),
                "max_depth": I(3, 20),
                "min_samples_split": I(2, 10),
                "min_samples_leaf": I(1, 6),
                "max_features": C([None, "sqrt", "log2"]),
                "bootstrap": C([True, False])}

    if model_type == "Gaussian Process":
        return {"alpha": F(1e-8, 1e-1, log=True),
                "length_scale": F(1e-1, 1e2, log=True),
                "n_restarts_optimizer": I(0, 5)}

    if model_type == "SVM":
        return {"C": F(1e-2, 1e2, log=True),
                "epsilon": F(1e-3, 0.5, log=False),
                "gamma": C(["scale", "auto"])}

    if model_type == "XGBoost":
        return {"n_estimators": I(50, 300, step=25),
                "max_depth": I(2, 6),
                "learning_rate": F(0.02, 0.3, log=True),
                "subsample": F(0.6, 1.0, step=0.1),
                "colsample_bytree": F(0.6, 1.0, step=0.1),
                "reg_alpha": F(0.0, 1.0, step=0.1),
                "reg_lambda": F(0.5, 2.0, step=0.1)}

    return {}


def tune_hyperparameters(model_type, base_params, X_train, y_train,
                         n_trials=30, n_cores=1, cv_folds=3, random_state=42,
                         progress_cb=None):
    """Search for the best hyperparameters of ``model_type`` via Optuna.

    Args:
        model_type: GUI model name (e.g. "Ridge", "Gradient Boosting").
        base_params: The model's current (default/GUI) parameter dict.  Tuned
            keys override entries here; untouched keys (e.g. Ridge ``solver``)
            are preserved so ``create_model`` always receives a complete dict.
        X_train, y_train: SCALED training arrays the final model is fit on.
        n_trials: Number of Optuna trials.
        n_cores: Parallel jobs used for the inner cross-validation of each trial
            (trial order stays deterministic; only the CV folds run in parallel).
        cv_folds: Inner K-Fold splits (clamped to the sample count).
        random_state: Seed for the TPE sampler and the CV shuffling.
        progress_cb: Optional callable ``progress_cb(done, total)`` invoked after
            each trial for GUI progress reporting.

    Returns:
        (best_params, best_rmse, study, message)
        * best_params: dict of only the tuned hyperparameters (may be empty).
        * best_rmse: best cross-validated RMSE (None if nothing was tuned).
        * study: the Optuna study (None if nothing was tuned).
        * message: human-readable status / reason when tuning was skipped.
    """
    _ensure_optuna_quiet()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    n_samples, n_features = X_train.shape

    if model_type in NON_TUNABLE_MODELS:
        return {}, None, None, f"{model_type} has no tunable hyperparameters."

    space = get_search_space(model_type, n_features, n_samples)
    if not space:
        return {}, None, None, f"No tunable hyperparameters available for {model_type}."

    if n_samples < 6:
        return {}, None, None, "Too few samples for reliable hyperparameter tuning."

    n_splits = int(max(2, min(cv_folds, n_samples)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        sampled = {name: suggest_from_distribution(trial, name, dist)
                   for name, dist in space.items()}
        params = dict(base_params)
        params.update(sampled)
        try:
            model = create_model(model_type, params, n_cores=1)
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="neg_root_mean_squared_error",
                                     n_jobs=n_cores)
            rmse = -float(np.mean(scores))
            if not np.isfinite(rmse):
                return float("inf")
            return rmse
        except Exception:
            # A bad hyperparameter combination should not abort the whole search.
            return float("inf")

    callbacks = []
    if progress_cb is not None:
        def _cb(study, trial):
            try:
                progress_cb(trial.number + 1, n_trials)
            except Exception:
                pass
        callbacks.append(_cb)

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=callbacks)

    if study.best_value is None or not np.isfinite(study.best_value):
        return {}, None, study, "Tuning did not find any valid hyperparameters."

    return dict(study.best_params), float(study.best_value), study, "ok"


def _describe_distribution(dist):
    """Return (kind, range_str, scale_str) describing an Optuna distribution."""
    if isinstance(dist, optuna.distributions.FloatDistribution):
        extra = []
        if dist.log:
            extra.append("log")
        if dist.step:
            extra.append(f"step {dist.step:g}")
        return "float", f"[{dist.low:g}, {dist.high:g}]", ", ".join(extra)
    if isinstance(dist, optuna.distributions.IntDistribution):
        extra = []
        if dist.log:
            extra.append("log")
        if dist.step and dist.step != 1:
            extra.append(f"step {dist.step}")
        return "int", f"[{dist.low}, {dist.high}]", ", ".join(extra)
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return "categorical", "{" + ", ".join(str(c) for c in dist.choices) + "}", ""
    return "?", str(dist), ""


def search_space_table(model_type, n_features, n_samples):
    """Return the Optuna search space as a list of row dicts (for Excel export)."""
    space = get_search_space(model_type, n_features, n_samples)
    rows = []
    for name, dist in space.items():
        kind, rng, scale = _describe_distribution(dist)
        rows.append({'Hyperparameter': name, 'Type': kind,
                     'Range': rng, 'Scale': scale or '—'})
    return rows


def format_search_space(model_type, n_features, n_samples):
    """Compact one-line search-space string for the status log."""
    space = get_search_space(model_type, n_features, n_samples)
    if not space:
        return "(no tunable hyperparameters)"
    parts = []
    for name, dist in space.items():
        _kind, rng, scale = _describe_distribution(dist)
        suffix = f" {scale}" if scale else ""
        parts.append(f"{name} {rng}{suffix}")
    return " | ".join(parts)


def format_params(params):
    """Compact, human-readable string of a hyperparameter dict for the log."""
    if not params:
        return "(none)"
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
