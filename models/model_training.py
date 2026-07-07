"""
Model training utilities for spectral analysis

@author: Sharad Kumar Gupta
"""
import numpy as np
from joblib import Parallel, delayed

# Heavy libs imported lazily so importing this module (done at GUI startup to
# expose create_model etc.) does not pull in scikit-learn / XGBoost until a model
# is actually built. Proxies resolve to the real objects on first call.
from utils.lazy_imports import LazyModule, LazyCallable

PLSRegression = LazyCallable('sklearn.cross_decomposition', 'PLSRegression')
SVR = LazyCallable('sklearn.svm', 'SVR')
Ridge = LazyCallable('sklearn.linear_model', 'Ridge')
Lasso = LazyCallable('sklearn.linear_model', 'Lasso')
LinearRegression = LazyCallable('sklearn.linear_model', 'LinearRegression')
ElasticNet = LazyCallable('sklearn.linear_model', 'ElasticNet')
HuberRegressor = LazyCallable('sklearn.linear_model', 'HuberRegressor')
PCA = LazyCallable('sklearn.decomposition', 'PCA')
RandomForestRegressor = LazyCallable('sklearn.ensemble', 'RandomForestRegressor')
GradientBoostingRegressor = LazyCallable('sklearn.ensemble', 'GradientBoostingRegressor')
GaussianProcessRegressor = LazyCallable('sklearn.gaussian_process', 'GaussianProcessRegressor')
RBF = LazyCallable('sklearn.gaussian_process.kernels', 'RBF')
C = LazyCallable('sklearn.gaussian_process.kernels', 'ConstantKernel')
StandardScaler = LazyCallable('sklearn.preprocessing', 'StandardScaler')
mean_squared_error = LazyCallable('sklearn.metrics', 'mean_squared_error')
r2_score = LazyCallable('sklearn.metrics', 'r2_score')
clone = LazyCallable('sklearn.base', 'clone')
xgb = LazyModule('xgboost')


def clamp_n_components(model, n_samples, n_features):
    """Clamp a decomposition estimator's ``n_components`` to what the data allows.

    PLS / PCA require ``n_components <= min(n_samples, n_features)``.  A user-set
    or default count (e.g. 50, or ``len(wavelengths)``) can exceed that bound once
    the training set is small or the feature grid has been thinned (resampling /
    binning), making ``fit`` raise ``ValueError: n_components upper bound is ...``.

    Returns the model unchanged when it has no ``n_components`` or the value is
    already valid; otherwise returns a clone with a clamped ``n_components`` (>= 1)
    so the caller's estimator is never mutated.
    """
    try:
        params = model.get_params()
    except Exception:
        return model
    if 'n_components' not in params or params['n_components'] is None:
        return model
    upper = max(1, min(int(n_samples), int(n_features)))
    try:
        requested = int(params['n_components'])
    except (TypeError, ValueError):
        return model
    if requested <= upper:
        return model
    clamped = clone(model)
    clamped.set_params(n_components=upper)
    return clamped


def create_model(model_type, params, n_cores=1):
    """
    Create a model instance based on type and parameters
    """
    try:
        if model_type == "PLS-R":
            n_components = min(int(params['n_components']), 50)  # Reasonable limit
            return PLSRegression(n_components=n_components)
        
        elif model_type == "PCA":
            n_components = min(int(params['n_components']), 50)
            return PCA(n_components=n_components, 
                      svd_solver=params['svd_solver'],
                      whiten=params['whiten'])
        
        elif model_type == "Ridge":
            return Ridge(
                alpha=float(params['alpha']),
                solver=params['solver'],
                max_iter=int(params['max_iter'])
            )
        
        elif model_type == "Lasso":
            return Lasso(
                alpha=float(params['alpha']),
                max_iter=int(params['max_iter']),
                selection=params['selection'],
                tol=float(params['tol'])
            )
        
        elif model_type == "Multiple Linear Regression":
            return LinearRegression(fit_intercept=params['fit_intercept'])
        
        elif model_type == "Elastic Net":
            return ElasticNet(
                alpha=float(params['alpha']),
                l1_ratio=float(params['l1_ratio']),
                max_iter=int(params['max_iter']),
                tol=float(params['tol'])
            )
        
        elif model_type == "Huber Regressor":
            return HuberRegressor(
                epsilon=float(params['epsilon']),
                max_iter=int(params['max_iter']),
                alpha=float(params['alpha'])
            )
        
        elif model_type == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=int(params['n_estimators']),
                learning_rate=float(params['learning_rate']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                subsample=float(params['subsample']),
                random_state=42
            )
        
        elif model_type == "Gaussian Process":
            # Create RBF kernel with length scale
            kernel = C(1.0, (1e-3, 1e3)) * RBF(params['length_scale'], (1e-2, 1e2))
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=float(params['alpha']),
                n_restarts_optimizer=int(params['n_restarts_optimizer']),
                random_state=42
            )
        
        elif model_type == "Random Forest":
            max_depth = None if str(params['max_depth']).lower() == "none" else int(params['max_depth'])
            
            # Parse max_features safely
            max_feat_raw = str(params['max_features']).strip().lower()
            if max_feat_raw == 'none':
                max_features = None
            elif max_feat_raw in ['sqrt', 'log2']:
                max_features = max_feat_raw
            else:
                try:
                    max_features = int(max_feat_raw)
                except ValueError:
                    try:
                        max_features = float(max_feat_raw)
                    except ValueError:
                        max_features = 'sqrt'
            
            return RandomForestRegressor(
                n_estimators=int(params['n_estimators']),
                max_depth=max_depth,
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                max_features=max_features,
                bootstrap=bool(params['bootstrap']),
                n_jobs=n_cores,
                random_state=42
            )
        
        elif model_type == "XGBoost":
            return xgb.XGBRegressor(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                learning_rate=float(params['learning_rate']),
                subsample=float(params['subsample']),
                colsample_bytree=float(params['colsample_bytree']),
                reg_alpha=float(params['reg_alpha']),
                reg_lambda=float(params['reg_lambda']),
                n_jobs=n_cores,
                random_state=42
            )
        
        else:  # SVM
            gamma_val = params['gamma'] if params['gamma'] in ['scale', 'auto'] else float(params['gamma'])
            return SVR(
                kernel=params['kernel'],
                C=float(params['C']),
                degree=int(params['degree']),
                gamma=gamma_val,
                epsilon=float(params['epsilon'])
            )
            
    except Exception as e:
        raise Exception(f"Model creation failed: {str(e)}")

def evaluate_component_count(n_comp, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, model_type, scaler_y):
    """Standalone function to evaluate a specific number of components"""
    try:
        if model_type == "PLS-R":
            model = PLSRegression(n_components=n_comp)
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            
        elif model_type == "PCA":
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            model = LinearRegression()
            model.fit(X_train_pca, y_train_scaled)
            y_pred_scaled = model.predict(X_test_pca)
        else:
            return n_comp, float('inf'), 0.0
        
        # Convert back to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return n_comp, rmse, r2
    except Exception:
        return n_comp, float('inf'), 0.0

def optimize_components_parallel(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
                               model_type, max_components, scaler_y, n_jobs):
    """Optimize number of components using parallel processing"""
    try:
        # Create a copy of scaler_y that can be pickled
        scaler_y_copy = StandardScaler()
        scaler_y_copy.mean_ = scaler_y.mean_.copy()
        scaler_y_copy.scale_ = scaler_y.scale_.copy()
        scaler_y_copy.var_ = scaler_y.var_.copy()
        scaler_y_copy.n_features_in_ = scaler_y.n_features_in_
        scaler_y_copy.n_samples_seen_ = scaler_y.n_samples_seen_
        
        # Test different numbers of components
        component_range = range(1, min(max_components + 1, X_train_scaled.shape[1] + 1, X_train_scaled.shape[0]))
        
        # Use parallel processing to evaluate different component counts.
        # max_nbytes=None disables loky's array memmapping, which on Windows can
        # leave a temp memmap folder whose cleanup races the resource_tracker and
        # spams "FileNotFoundError" warnings at shutdown.
        results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(evaluate_component_count)(
                n_comp, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, model_type, scaler_y_copy
            ) for n_comp in component_range
        )
        
        # Extract results
        components = [r[0] for r in results]
        rmse_values = [r[1] for r in results]
        r2_values = [r[2] for r in results]
        
        # Find optimal number of components (minimum RMSE)
        optimal_idx = np.argmin(rmse_values)
        optimal_components = components[optimal_idx]
        
        return optimal_components, components, rmse_values, r2_values
        
    except Exception:
        return None, [], [], []

def parse_parameter_value(param_name, param_value, param_type):
    """Parse parameter value based on its expected type"""
    try:
        if param_value.lower() == "none":
            return None
        elif param_value.lower() in ["true", "false"]:
            return param_value.lower() == "true"
        elif param_type in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                           "n_components", "degree", "max_iter", "n_restarts_optimizer"]:
            return int(float(param_value)) if param_value.lower() != "none" else None
        elif param_type in ["alpha", "C", "epsilon", "tol", "learning_rate", "subsample",
                           "colsample_bytree", "reg_alpha", "reg_lambda", "l1_ratio",
                           "length_scale"]:
            return float(param_value)
        else:
            return param_value
    except:
        return param_value
