"""
File operations for spectral analysis

@author: Sharad Kumar Gupta
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from models.batch_processing import (assess_overfitting, compute_rpd,
                                     compute_nrmsep, rpd_quality)

def save_results_to_excel(output_filename, results_data):
    """
    Save all analysis results to Excel file with multiple sheets
    """
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Main results
            results_df = pd.DataFrame({
                'Actual': results_data['y_test'],
                'Predicted': results_data['y_pred'],
                'Absolute Error': np.abs(results_data['y_test'] - results_data['y_pred'])
            })
            results_df.to_excel(writer, sheet_name=f'Predictions_{results_data["model_type"]}', index=False)
            
            # Correlogram data
            correlogram_df = pd.DataFrame({
                'Wavelength': results_data['filtered_wavelengths'],
                'Correlation': results_data['correlations']
            })
            correlogram_df.to_excel(writer, sheet_name=f'Correlogram_{results_data["model_type"]}', index=False)
            
            # Performance metrics.  RMSEP is the held-out RMSE under its
            # chemometric name; nRMSEP is a percentage of the observed range and
            # RPD is SD/RMSE, so errors stay comparable between properties.
            _test_rpd = results_data.get('test_rpd')
            _test_nrmsep = results_data.get('test_nrmsep')
            if _test_rpd is None and results_data.get('y_test') is not None:
                _test_rpd = compute_rpd(results_data['y_test'],
                                        rmse=results_data['test_rmse'])
            if _test_nrmsep is None and results_data.get('y_test') is not None:
                _test_nrmsep = compute_nrmsep(results_data['y_test'],
                                              rmse=results_data['test_rmse'])
            performance_data = {
                'Metric': ['Test R²', 'Test RMSE', 'Test RMSEP',
                           'Test nRMSEP (% of range)', 'Test RPD', 'RPD Quality',
                           'Train R²', 'Train RMSE', 'Train RPD',
                           'Train nRMSEP (% of range)', 'Test Size',
                           'Soil Property', 'Preprocessing', 'Number of Cores'],
                'Value': [results_data['test_r2'], results_data['test_rmse'],
                          results_data.get('test_rmsep', results_data['test_rmse']),
                          _test_nrmsep, _test_rpd, rpd_quality(_test_rpd),
                          results_data['train_r2'], results_data['train_rmse'],
                          results_data.get('train_rpd'),
                          results_data.get('train_nrmsep'),
                          results_data['test_size'], results_data['selected_property'],
                          results_data['preprocessing'], results_data['n_cores']]
            }
            
            # Add cross-validation results if available
            if results_data.get('cv_results'):
                cv_results = results_data['cv_results']
                performance_data['Metric'].extend([
                    'CV Strategy', 'CV R² Mean', 'CV R² Std', 'CV RMSE Mean', 'CV RMSE Std',
                    'CV RPD', 'CV nRMSEP (% of range)'
                ])
                performance_data['Value'].extend([
                    cv_results['strategy'], cv_results['r2_mean'], cv_results['r2_std'],
                    cv_results['rmse_mean'], cv_results['rmse_std'],
                    cv_results.get('cv_rpd'), cv_results.get('cv_nrmsep')
                ])
                
                # Add K-fold specific information
                if cv_results['strategy'] == 'K-Fold' and 'parameters' in cv_results:
                    performance_data['Metric'].append('Number of Folds')
                    performance_data['Value'].append(cv_results['parameters']['k_folds'])
            
            stats_df = pd.DataFrame(performance_data)
            stats_df.to_excel(writer, sheet_name=f'Performance_{results_data["model_type"]}', index=False)
            
            # Model parameters
            params_df = pd.DataFrame({
                'Parameter': list(results_data['params'].keys()),
                'Value': [str(v) for v in results_data['params'].values()]
            })
            params_df.to_excel(writer, sheet_name=f'Parameters_{results_data["model_type"]}', index=False)

            # Hyperparameter search space (only present when the model was tuned)
            if results_data.get('hyperparameter_search_space'):
                search_df = pd.DataFrame(results_data['hyperparameter_search_space'])
                search_df.to_excel(
                    writer, sheet_name=f'SearchSpace_{results_data["model_type"]}', index=False)

            # Component optimization results if available
            if results_data.get('component_optimization_results'):
                comp_opt = results_data['component_optimization_results']
                comp_opt_df = pd.DataFrame({
                    'Components': comp_opt['Components'],
                    'RMSE': comp_opt['RMSE'],
                    'R2_Score': comp_opt['R2_Score']
                })
                comp_opt_df.to_excel(writer, sheet_name=f'ComponentOpt_{results_data["model_type"]}', index=False)
            
            # Cross-validation detailed results if available
            if results_data.get('cv_results') and results_data['cv_results']['cv_rmse_scores']:
                cv_results = results_data['cv_results']
                cv_detailed_df = pd.DataFrame({
                    'Fold': range(1, len(cv_results['cv_rmse_scores']) + 1),
                    'RMSE': cv_results['cv_rmse_scores'],
                    'R2': cv_results['cv_r2_scores']
                })
                cv_detailed_df.to_excel(writer, sheet_name=f'CrossValidation_{results_data["model_type"]}', index=False)
            
            # Data statistics if requested
            if results_data.get('export_stats') and results_data.get('data_stats'):
                data_stats = results_data['data_stats']
                
                # Input statistics
                input_stats_df = pd.DataFrame.from_dict(
                    data_stats['Input Data Statistics'], orient='index', columns=['Value'])
                input_stats_df.to_excel(writer, sheet_name=f'Input Statistics_{results_data["model_type"]}')
                
                # Spectral statistics
                for stat_type, stat_data in data_stats['Spectral Statistics'].items():
                    stat_df = pd.DataFrame.from_dict(stat_data, orient='index', columns=['Value'])
                    sheet_name = f'Spectral {stat_type}_{results_data["model_type"]}'
                    stat_df.to_excel(writer, sheet_name=sheet_name)
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to save results to Excel: {str(e)}")

def make_timestamp():
    """Return a filename timestamp (YYYYMMDD_HHMMSS) for the current moment.

    Generate one per analysis run and pass it to the results / PDF / model
    filename builders so a run's artefacts share the exact same timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_default_filename(input_filename, selected_property, model_type,
                              timestamp=None):
    """
    Generate default filename for results
    """
    timestamp = timestamp or make_timestamp()
    return f"{input_filename}_{selected_property}_{model_type}_results_{timestamp}.xlsx"

def generate_model_filename(input_filename, selected_property, model_type,
                            timestamp=None):
    """
    Generate default filename for model saving.

    When ``timestamp`` is supplied (the timestamp of the run's results/PDF) it is
    embedded so the model file matches the Excel/PDF of the same analysis.
    """
    suffix = f"_{timestamp}" if timestamp else ""
    return f"{input_filename}_{selected_property}_{model_type}_model{suffix}.joblib"


def save_batch_results_to_excel(output_filename, batch_results, comparison_df):
    """
    Save batch processing results with multiple properties and models
    
    Args:
        output_filename: Output Excel file path
        batch_results: Dictionary of results organized by property and model
        comparison_df: DataFrame with model comparison
    """
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Save comparison summary
            if comparison_df is not None:
                comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Save results for each property and model
            for property_name, models_results in batch_results.items():
                for model_name, results in models_results.items():
                    sheet_prefix = f"{property_name[:15]}_{model_name[:10]}"
                    
                    # Predictions
                    if 'y_test' in results and 'y_pred' in results:
                        pred_df = pd.DataFrame({
                            'Actual': results['y_test'],
                            'Predicted': results['y_pred'],
                            'Absolute_Error': np.abs(results['y_test'] - results['y_pred'])
                        })
                        pred_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Pred', index=False)
                    
                    # Metrics
                    metrics_data = {
                        'Metric': ['Test_R2', 'Test_RMSE', 'Train_R2', 'Train_RMSE'],
                        'Value': [
                            results.get('test_r2', 'N/A'),
                            results.get('test_rmse', 'N/A'),
                            results.get('train_r2', 'N/A'),
                            results.get('train_rmse', 'N/A')
                        ]
                    }
                    
                    # Add CV metrics if available
                    if results.get('cv_results'):
                        cv_res = results['cv_results']
                        metrics_data['Metric'].extend(['CV_R2_Mean', 'CV_R2_Std', 'CV_RMSE_Mean', 'CV_RMSE_Std'])
                        metrics_data['Value'].extend([
                            cv_res.get('r2_mean', 'N/A'),
                            cv_res.get('r2_std', 'N/A'),
                            cv_res.get('rmse_mean', 'N/A'),
                            cv_res.get('rmse_std', 'N/A')
                        ])
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Metrics', index=False)
                    
                    # Correlations (if available)
                    if 'correlations' in results and 'wavelengths' in results:
                        corr_df = pd.DataFrame({
                            'Wavelength': results['wavelengths'],
                            'Correlation': results['correlations']
                        })
                        corr_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Corr', index=False)
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to save batch results: {str(e)}")


def generate_batch_filename(input_filename, properties, models):
    """
    Generate filename for batch processing results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prop_str = "_".join([p[:5] for p in properties[:2]])  # First 2 properties, truncated
    if len(properties) > 2:
        prop_str += "_etc"
    model_str = "_".join([m[:3] for m in models[:2]])  # First 2 models, abbreviated
    if len(models) > 2:
        model_str += "_etc"
    
    return f"{input_filename}_batch_{prop_str}_{model_str}_{timestamp}.xlsx"


def generate_property_filename(input_filename, property_name, timestamp=None):
    """
    Generate filename for single property batch results.

    When ``timestamp`` is supplied it is used verbatim so the results Excel, its
    PDF, and a later-saved model of the same run all share one timestamp.
    """
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean property name for filename
    safe_prop = property_name.replace(' ', '_').replace('/', '_')
    return f"{input_filename}_{safe_prop}_{timestamp}.xlsx"


def save_property_batch_results(output_filename, property_name, results_dict, auto_select_best=False, best_model=None):
    """
    Save batch results for a single property to Excel
    
    Args:
        output_filename: Output Excel filename
        property_name: Name of soil property
        results_dict: Dictionary of {model_name: results}
        auto_select_best: If True, only save detailed results for best model
        best_model: Name of best model (if auto_select_best is True)
    """
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Model Comparison Sheet
            comparison_data = []
            for model_name, results in results_dict.items():
                # Multi-measure overfitting assessment for this model
                assessment = assess_overfitting(
                    results.get('train_r2'), results.get('test_r2'),
                    results.get('train_rmse'), results.get('test_rmse'),
                    (results.get('cv_results') or {}).get('r2_mean'),
                )

                row = {
                    'Model': model_name,
                    'Best_Model': '✓' if model_name == best_model else '',
                    'Train_R2': results.get('train_r2', 'N/A'),
                    'Test_R2': results.get('test_r2', 'N/A'),
                    'Train_RMSE': results.get('train_rmse', 'N/A'),
                    'Test_RMSE': results.get('test_rmse', 'N/A'),
                    'Train_MAE': results.get('train_mae', 'N/A'),
                    'Test_MAE': results.get('test_mae', 'N/A'),
                    'CV_R2': (results.get('cv_results') or {}).get('r2_mean', 'N/A'),
                    'Overfitting': assessment['severity'] if assessment['flag'] else 'ok',
                }

                # Add confidence intervals if available
                if results.get('confidence_intervals'):
                    ci = results['confidence_intervals']
                    row['R2_CI_Lower'] = ci['r2_ci'][0]
                    row['R2_CI_Upper'] = ci['r2_ci'][1]
                    row['RMSE_CI_Lower'] = ci['rmse_ci'][0]
                    row['RMSE_CI_Upper'] = ci['rmse_ci'][1]
                    row['MAE_CI_Lower'] = ci['mae_ci'][0]
                    row['MAE_CI_Upper'] = ci['mae_ci'][1]
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Detailed results - either all models or just best model
            models_to_save = [best_model] if auto_select_best and best_model else list(results_dict.keys())
            
            for model_name in models_to_save:
                if model_name not in results_dict:
                    continue
                    
                results = results_dict[model_name]
                sheet_prefix = f"{model_name[:20]}"  # Limit sheet name length
                
                # Predictions
                if 'y_test' in results and 'y_pred' in results:
                    pred_df = pd.DataFrame({
                        'Actual': results['y_test'],
                        'Predicted': results['y_pred'],
                        'Absolute_Error': np.abs(results['y_test'] - results['y_pred'])
                    })
                    pred_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Pred', index=False)
                
                # Metrics
                metrics_data = {
                    'Metric': ['Test_R2', 'Test_RMSE', 'Test_MAE', 'Train_R2', 'Train_RMSE', 'Train_MAE'],
                    'Value': [
                        results.get('test_r2', 'N/A'),
                        results.get('test_rmse', 'N/A'),
                        results.get('test_mae', 'N/A'),
                        results.get('train_r2', 'N/A'),
                        results.get('train_rmse', 'N/A'),
                        results.get('train_mae', 'N/A')
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Metrics', index=False)
                
                # Correlations
                if 'correlations' in results and 'wavelengths' in results:
                    corr_df = pd.DataFrame({
                        'Wavelength': results['wavelengths'],
                        'Correlation': results['correlations']
                    })
                    corr_df.to_excel(writer, sheet_name=f'{sheet_prefix}_Corr', index=False)
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to save property batch results: {str(e)}")


# ---------------------------------------------------------------------------
# Unknown / unseen data prediction export
# ---------------------------------------------------------------------------

def save_unknown_predictions_to_excel(output_filename, input_df, predictions,
                                       property_name, model_type,
                                       wavelength_unit="nm"):
    """
    Save predictions made on *unseen* (unknown label) tabular spectral data.

    The output Excel has two sheets:
      • Predictions - original input columns + a new column with the
        model's predicted values.
      • Summary     - row-level statistics (mean, std, CV%) of the
        prediction column.

    Parameters
    ----------
    output_filename : str
    input_df        : DataFrame  - the original loaded DataFrame
    predictions     : array-like - model predictions (one per row)
    property_name   : str
    model_type      : str
    wavelength_unit : str        - "nm" or "μm" (for metadata)
    """
    try:
        out_df = input_df.copy()
        out_df[f'Predicted_{property_name}'] = predictions

        summary_data = {
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'CV (%)',
                       'Model', 'Property', 'Wavelength unit'],
            'Value': [
                len(predictions),
                float(np.mean(predictions)),
                float(np.std(predictions)),
                float(np.min(predictions)),
                float(np.max(predictions)),
                float(np.std(predictions) / (abs(np.mean(predictions)) + 1e-12) * 100),
                model_type,
                property_name,
                wavelength_unit,
            ]
        }
        summary_df = pd.DataFrame(summary_data)

        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            out_df.to_excel(writer, sheet_name='Predictions', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        return True
    except Exception as e:
        raise Exception(f"Failed to save unknown predictions: {str(e)}")


def save_resampled_tabular_to_excel(output_filename, id_df, X_resampled, wavelengths,
                                    wavelength_unit="nm"):
    """Save a resampled spectral matrix so the user can validate resampling -
    column headers are the actual target wavelengths used for prediction.

    Parameters
    ----------
    output_filename : str
    id_df           : DataFrame  - non-wavelength columns to preserve (sample
                      IDs etc.), same row order/count as ``X_resampled``
    X_resampled     : array-like (n_samples, n_wavelengths) - spectra AFTER
                      resampling but BEFORE preprocessing, on ``wavelengths``
    wavelengths     : the resampled band centres, same order as the columns
                      of ``X_resampled``
    wavelength_unit : str
    """
    try:
        cols = [f"{float(w):.2f}" for w in wavelengths]
        spec_df = pd.DataFrame(np.asarray(X_resampled), columns=cols)
        out_df = pd.concat([id_df.reset_index(drop=True), spec_df], axis=1)

        info_df = pd.DataFrame({
            'Metric': ['Samples', 'Bands', 'Wavelength unit', 'Min', 'Max'],
            'Value': [len(spec_df), len(cols), wavelength_unit,
                      float(min(wavelengths)), float(max(wavelengths))],
        })

        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            out_df.to_excel(writer, sheet_name='Resampled Spectra', index=False)
            info_df.to_excel(writer, sheet_name='Info', index=False)

        return True
    except Exception as e:
        raise Exception(f"Failed to save resampled tabular data: {str(e)}")


def generate_unknown_prediction_filename(input_filename, property_name, model_type):
    """Generate a timestamped filename for unknown-data prediction results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prop = property_name.replace(' ', '_').replace('/', '_')
    return f"{input_filename}_{safe_prop}_{model_type}_unknown_predictions_{timestamp}.xlsx"


def save_transfer_function(output_filename, pls, scaler_s, scaler_t,
                           r2_per_band, source_wavelengths, target_wavelengths):
    """
    Persist a fitted spectral transfer function (PLS model + scalers).

    Uses joblib so the same dict keys are compatible with load_model logic.
    """
    tf_data = {
        'pls': pls,
        'scaler_source': scaler_s,
        'scaler_target': scaler_t,
        'r2_per_band': r2_per_band,
        'source_wavelengths': source_wavelengths,
        'target_wavelengths': target_wavelengths,
    }
    joblib.dump(tf_data, output_filename)
    return True


def load_transfer_function(filename):
    """Load a transfer function saved by save_transfer_function."""
    return joblib.load(filename)