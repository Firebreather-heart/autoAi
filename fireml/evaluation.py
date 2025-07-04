"""
Model evaluation utilities for FireAutoML.

This module provides functions to evaluate machine learning models,
create performance reports, and generate visualizations.
"""
import logging
import time
from typing import Dict, List, Union, Any, Tuple, Optional
import pandas as pd
import numpy as np
import os
from fireml.settings import Settings

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for comprehensive model evaluation and comparison."""
    
    def __init__(self, task_type: str = 'classification', report_dir:Optional[Any] = 'default'):
        """
        Initialize the model evaluator.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type.lower()
        self.results = {}
        self.settings = Settings()
        if report_dir and report_dir != 'default':
            self.report_dir = report_dir 
        else:
            self.report_dir = os.path.join(self.settings.output_directory, 'reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize visualization if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_viz = True
        except ImportError:
            self.has_viz = False
            logger.warning("Matplotlib not available. Visualizations will be skipped.")
    

    def evaluate_model(self, 
                      model_name: str,
                      model: Any, 
                      X_test: pd.DataFrame, 
                      y_test: Union[pd.Series, np.ndarray],
                      X_train: Optional[pd.DataFrame] = None,
                      y_train: Optional[Union[pd.Series, np.ndarray]] = None,
                      prediction_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate a model and store the results.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            X_test: Test features
            y_test: True test labels/values
            X_train: Training features (optional, for additional metrics)
            y_train: Training labels/values (optional)
            prediction_time: Time taken for prediction (in seconds)
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        from sklearn import metrics
        
        start_time = time.time()
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            logger.error(f"Model {model_name} does not have a predict method")
            return {}
        
        if prediction_time is None:
            prediction_time = time.time() - start_time
        
        # Calculate metrics based on task type
        result = {
            'model_name': model_name,
            'prediction_time_seconds': prediction_time,
            'n_test_samples': len(X_test)
        }
        
        # Add model parameters if available
        try:
            result['model_params'] = model.get_params()
        except:
            result['model_params'] = "Not available"
            
        # Calculate metrics
        if self.task_type == 'classification':
            # Basic classification metrics
            result['accuracy'] = metrics.accuracy_score(y_test, y_pred)
            
            try:
                # Only for binary classification or with multi_class option
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)
                    if y_prob.shape[1] == 2:  # Binary case
                        result['roc_auc'] = metrics.roc_auc_score(y_test, y_prob[:, 1])
                        precision, recall, _ = metrics.precision_recall_curve(y_test, y_prob[:, 1])
                        result['pr_auc'] = metrics.auc(recall, precision)
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR AUC: {e}")
            
            # Classification report (precision, recall, f1)
            try:
                class_report = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                result['classification_report'] = class_report
                
                # Extract key metrics from report for easier access
                if isinstance(class_report, dict):
                    result['macro_avg_f1'] = class_report['macro avg']['f1-score']
                    result['weighted_avg_f1'] = class_report['weighted avg']['f1-score']
                
                # Confusion matrix
                result['confusion_matrix'] = metrics.confusion_matrix(y_test, y_pred).tolist()
                
            except Exception as e:
                logger.warning(f"Error generating classification report: {e}")
                
            # For multi-class problems
            if len(np.unique(y_test)) > 2:
                try:
                    result['cohen_kappa'] = metrics.cohen_kappa_score(y_test, y_pred)
                    result['balanced_accuracy'] = metrics.balanced_accuracy_score(y_test, y_pred)
                except Exception as e:
                    logger.warning(f"Error calculating multi-class metrics: {e}")
                    
        elif self.task_type == 'regression':
            # Regression metrics
            result['mean_absolute_error'] = metrics.mean_absolute_error(y_test, y_pred)
            result['mean_squared_error'] = metrics.mean_squared_error(y_test, y_pred)
            result['root_mean_squared_error'] = np.sqrt(result['mean_squared_error'])
            result['r2_score'] = metrics.r2_score(y_test, y_pred)
            
            # Calculate median absolute error (more robust to outliers)
            result['median_absolute_error'] = metrics.median_absolute_error(y_test, y_pred)
            
            # Calculate explained variance
            result['explained_variance'] = metrics.explained_variance_score(y_test, y_pred)
            
            # Calculate mean absolute percentage error (MAPE)
            # Avoid division by zero by masking zeros in y_test
            mask = y_test != 0
            if np.any(mask):
                result['mean_absolute_percentage_error'] = np.mean(
                    np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                result['mean_absolute_percentage_error'] = np.nan
        
        # Check for overfitting if training data provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            
            if self.task_type == 'classification':
                train_acc = metrics.accuracy_score(y_train, y_train_pred)
                result['train_accuracy'] = train_acc
                result['accuracy_delta'] = train_acc - result['accuracy']
                
                if result['accuracy_delta'] > 0.2:
                    result['overfitting_warning'] = True
                    result['overfitting_message'] = (f"Possible overfitting detected: "
                                                    f"train accuracy {train_acc:.4f} vs "
                                                    f"test accuracy {result['accuracy']:.4f}")
                else:
                    result['overfitting_warning'] = False
                    
            elif self.task_type == 'regression':
                train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
                result['train_rmse'] = train_rmse
                result['rmse_ratio'] = result['root_mean_squared_error'] / train_rmse if train_rmse > 0 else float('inf')
                
                if result['rmse_ratio'] > 1.3:
                    result['overfitting_warning'] = True
                    result['overfitting_message'] = (f"Possible overfitting detected: "
                                                    f"train RMSE {train_rmse:.4f} vs "
                                                    f"test RMSE {result['root_mean_squared_error']:.4f}")
                else:
                    result['overfitting_warning'] = False
        
        result['model_obj'] = model
        # Store the results
        self.results[model_name] = result
        
        # Generate visualizations if available
        if self.has_viz:
            self._generate_model_visualizations(model_name, model, X_test, y_test, y_pred)
        
        logger.info(f"Evaluated model: {model_name}")
        return result
    
    def _generate_model_visualizations(self, model_name: str, model: Any, 
                                     X_test: pd.DataFrame, y_test: Any, y_pred: Any) -> None:
        """Generate visualizations for model evaluation."""
        if not self.has_viz:
            return
            
        # Create directory for visualizations
        viz_dir = os.path.join(self.report_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        plt = self.plt
        
        try:
            from sklearn import metrics
            if self.task_type == 'classification':
                # Confusion matrix
                plt.figure(figsize=(8, 6))
                cm = metrics.confusion_matrix(y_test, y_pred)
                metrics.ConfusionMatrixDisplay(cm).plot(cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.savefig(os.path.join(viz_dir, f"{model_name}_confusion_matrix.png"))
                plt.close()
                
                # ROC curve for binary classification
                if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    plt.figure(figsize=(8, 6))
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics.RocCurveDisplay.from_predictions(y_test, y_prob).plot()
                    plt.title(f'ROC Curve - {model_name}')
                    plt.savefig(os.path.join(viz_dir, f"{model_name}_roc_curve.png"))
                    plt.close()
                    
                    # Precision-Recall curve
                    plt.figure(figsize=(8, 6))
                    metrics.PrecisionRecallDisplay.from_predictions(y_test, y_prob).plot()
                    plt.title(f'Precision-Recall Curve - {model_name}')
                    plt.savefig(os.path.join(viz_dir, f"{model_name}_pr_curve.png"))
                    plt.close()
            else:
                # Scatter plot of predicted vs actual
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title(f'Predicted vs Actual - {model_name}')
                plt.savefig(os.path.join(viz_dir, f"{model_name}_pred_vs_actual.png"))
                plt.close()
                
                # Residual plot
                plt.figure(figsize=(8, 6))
                residuals = y_test - y_pred
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='k', linestyles='dashed')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                plt.title(f'Residual Plot - {model_name}')
                plt.savefig(os.path.join(viz_dir, f"{model_name}_residuals.png"))
                plt.close()
                
                # Histogram of residuals
                plt.figure(figsize=(8, 6))
                plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                plt.axvline(residuals.mean(), color='k', linestyle='dashed', linewidth=1)
                plt.xlabel('Residual Value')
                plt.ylabel('Frequency')
                plt.title(f'Residual Distribution - {model_name}')
                plt.savefig(os.path.join(viz_dir, f"{model_name}_residual_dist.png"))
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from model if available.
        
        Args:
            model: Trained model object
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importances = {}
        
        try:
            # Try different methods of getting feature importance
            if hasattr(model, "feature_importances_"):
                importances = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, "coef_"):
                if len(model.coef_.shape) == 1:
                    importances = dict(zip(feature_names, np.abs(model.coef_)))
                else:
                    # For multi-class, take average of absolute coefficients across classes
                    importances = dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
            elif hasattr(model, "named_steps"):
                # Check if it's a pipeline with a model that has feature importance
                for _, step in model.named_steps.items():
                    if hasattr(step, "feature_importances_"):
                        importances = dict(zip(feature_names, step.feature_importances_))
                        break
                    elif hasattr(step, "coef_"):
                        if len(step.coef_.shape) == 1:
                            importances = dict(zip(feature_names, np.abs(step.coef_)))
                        else:
                            importances = dict(zip(feature_names, np.mean(np.abs(step.coef_), axis=0)))
                        break
            
            # Sort by importance
            if importances:
                importances = {k: float(v) for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            
        return importances
    
    def permutation_importance(self, model: Any, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                            n_repeats: int = 10, random_state: int = 42) -> Dict[str, Dict[str, float]]:
        """
        Calculate permutation feature importance.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            Dictionary with permutation importance results
        """
        try:
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance
            r = permutation_importance(
                model, X, y, 
                n_repeats=n_repeats,
                random_state=random_state
            )
            
            # Format results
            feature_names = X.columns.tolist()
            importance_dict = {}
            
            for i, feature in enumerate(feature_names):
                importance_dict[feature] = {
                    'mean_importance': float(r['importances_mean'][i]),
                    'std_importance': float(r['importances_std'][i]),
                }
            
            # Sort by mean importance
            importance_dict = {k: v for k, v in sorted(
                importance_dict.items(), 
                key=lambda x: x[1]['mean_importance'], 
                reverse=True
            )}
            
            return importance_dict
        except Exception as e:
            logger.warning(f"Could not calculate permutation importance: {e}")
            return {}
    
    def plot_feature_importance(self, importances: Dict[str, float], title: str = "Feature Importance") -> str:
        """
        Plot feature importance and save the visualization.
        
        Args:
            importances: Dictionary mapping feature names to importance scores
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        if not self.has_viz or not importances:
            return ""
            
        # Create sorted feature importance dataframe
        features = list(importances.keys())
        importance_values = list(importances.values())
        
        # Limit to top 20 features if there are many
        if len(features) > 20:
            sorted_indices = np.argsort(importance_values)[-20:]
            features = [features[i] for i in sorted_indices]
            importance_values = [importance_values[i] for i in sorted_indices]
        
        # Create plot
        plt = self.plt
        plt.figure(figsize=(10, max(6, len(features) * 0.3)))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importance_values, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(title)
        
        # Save plot
        viz_dir = os.path.join(self.report_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_path = os.path.join(viz_dir, f"{title.lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all evaluated models and return a comparison report.
        
        Returns:
            Dictionary with model comparison results
        """
        if not self.results:
            return {"error": "No models have been evaluated"}
            
        comparison = {
            "best_model": None,
            "metrics_compared": {},
            "all_models": list(self.results.keys()),
        }
        
        # Determine key metric based on task type
        if self.task_type == 'classification':
            # For classification, higher values are better
            key_metrics = ['accuracy', 'macro_avg_f1', 'weighted_avg_f1', 'roc_auc']
            compare_func = max
            reverse_sort = True  # Higher is better
            
            # Find best model based on weighted_avg_f1 or accuracy
            if all('weighted_avg_f1' in self.results[model] for model in self.results):
                primary_metric = 'weighted_avg_f1'
            else:
                primary_metric = 'accuracy'
                
        else:  # regression
            # For regression, lower errors are better
            key_metrics = ['mean_absolute_error', 'root_mean_squared_error', 'median_absolute_error']
            compare_func = min
            reverse_sort = False  # Lower is better
            primary_metric = 'root_mean_squared_error'
        
        # Compare models across key metrics
        for metric in key_metrics:
            if all(metric in self.results[model] for model in self.results):
                model_metrics = {model: self.results[model][metric] for model in self.results}
                comparison["metrics_compared"][metric] = model_metrics
                
                # Sort models by this metric
                sorted_models = sorted(
                    model_metrics.items(), 
                    key=lambda x: x[1], 
                    reverse=reverse_sort
                )
                comparison[f"models_by_{metric}"] = [model for model, _ in sorted_models]
        
        # Determine the best model based on primary metric
        if primary_metric in comparison["metrics_compared"]:
            metric_values = comparison["metrics_compared"][primary_metric]
            best_model = compare_func(metric_values.items(), key=lambda x: x[1])[0]
            comparison["best_model"] = {
                "name": best_model,
                "primary_metric": primary_metric,
                "value": metric_values[best_model],
                "all_metrics": self.results[best_model]
            }
            
            # Check for overfitting in best model
            if "overfitting_warning" in self.results[best_model] and self.results[best_model]["overfitting_warning"]:
                comparison["best_model"]["warning"] = self.results[best_model]["overfitting_message"]
        
        # Add prediction time comparison
        if all("prediction_time_seconds" in self.results[model] for model in self.results):
            time_comparison = {model: self.results[model]["prediction_time_seconds"] for model in self.results}
            comparison["prediction_times"] = time_comparison
            
            # Sort models by prediction time
            sorted_by_time = sorted(time_comparison.items(), key=lambda x: x[1])
            comparison["models_by_speed"] = [model for model, _ in sorted_by_time]
            
            # Flag if fastest model is significantly faster than best model
            if comparison["best_model"] and comparison["models_by_speed"][0] != comparison["best_model"]["name"]:
                fastest_model = comparison["models_by_speed"][0]
                best_model_time = time_comparison[comparison["best_model"]["name"]]
                fastest_time = time_comparison[fastest_model]
                
                if fastest_time < best_model_time * 0.5:  # If fastest is at least 2x faster
                    comparison["speed_recommendation"] = {
                        "message": f"{fastest_model} is significantly faster than {comparison['best_model']['name']} "
                                  f"({fastest_time:.4f}s vs {best_model_time:.4f}s) "
                                  f"and might be preferred for real-time applications.",
                        "fastest_model": fastest_model,
                        "best_model": comparison["best_model"]["name"],
                        "time_difference_factor": best_model_time / fastest_time
                    }
        
        return comparison
    
    def generate_report(self, output_format: str = 'json', include_visualizations: bool = True,
                        preprocessing_steps: Optional[list] = None,
                        data_summary: Optional[dict] = None,
                        class_distribution: Optional[dict] = None,
                        feature_names: Optional[list] = None,
                        model_paths: Optional[dict] = None,
                        extras:Optional[dict] = None) -> Union[Dict, str]:
        """
        Generate a comprehensive evaluation report.

        Args:
            output_format: Format for report ('json', 'html', 'markdown')
            include_visualizations: Whether to include visualizations
            preprocessing_steps: List of preprocessing steps applied
            data_summary: Dict with data summary info
            class_distribution: Dict with class distribution info
            feature_names: List of feature names
            model_paths: Dict mapping model names to serialized file paths

        Returns:
            Report in requested format
        """
        # Compare models to get best one
        comparison = self.compare_models()

        # Initialize report structure
        report = {
            "summary": comparison,
            "detailed_metrics": self.results,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_type": self.task_type,
            "preprocessing_steps": preprocessing_steps or [],
            "data_summary": data_summary or {},
            "class_distribution": class_distribution or {},
            "model_paths": model_paths or {},
        }

        if extras:
            report.update(extras)

        # Add feature importances for each model if available
        if feature_names:
            report["feature_importances"] = {}
            for model_name, metrics in self.results.items():
                model = metrics.get('model_obj')
                if model:
                    importances = self.feature_importance(model, feature_names)
                    if importances:
                        report["feature_importances"][model_name] = importances

        # Add recommendations
        report["recommendations"] = self._generate_recommendations(comparison)

        # Add visualization paths if requested
        if include_visualizations and self.has_viz:
            viz_dir = os.path.join(self.report_dir, 'visualizations')
            if os.path.exists(viz_dir):
                report["visualizations"] = [os.path.join(viz_dir, f) for f in os.listdir(viz_dir) if f.endswith('.png')]

        # Convert to requested format
        if output_format == 'json':
            return report
        elif output_format == 'html':
            return self._convert_to_html(report)
        elif output_format == 'markdown':
            return self._convert_to_markdown(report)
        else:
            return report
        
    def _generate_recommendations(self, comparison: Dict) -> List[Dict[str, str]]:
        """Generate model recommendations based on comparison results."""
        recommendations = []
        
        # Recommend based on best model
        if "best_model" in comparison and comparison["best_model"]:
            best_model = comparison["best_model"]
            
            recommendations.append({
                "type": "best_model",
                "message": f"The best performing model is {best_model['name']} with "
                          f"{best_model['primary_metric']} of {best_model['value']:.4f}."
            })
            
            # Check for warnings in best model
            if "warning" in best_model:
                recommendations.append({
                    "type": "overfitting_warning",
                    "message": best_model["warning"],
                    "action": "Consider increasing regularization, reducing model complexity, "
                             "or gathering more training data to mitigate overfitting."
                })
        
        # Speed recommendation
        if "speed_recommendation" in comparison:
            recommendations.append({
                "type": "speed_recommendation",
                "message": comparison["speed_recommendation"]["message"],
                "action": "Consider using the faster model for applications where inference speed is critical."
            })
        
        # Task-specific recommendations
        if self.task_type == 'classification':
            # Check for class imbalance
            for model_name, metrics in self.results.items():
                if "classification_report" in metrics:
                    report = metrics["classification_report"]
                    if "macro avg" in report and "weighted avg" in report:
                        macro_f1 = report["macro avg"]["f1-score"]
                        weighted_f1 = report["weighted avg"]["f1-score"]
                        
                        if weighted_f1 > macro_f1 * 1.2:  # Significant difference suggests imbalance
                            recommendations.append({
                                "type": "class_imbalance",
                                "message": f"Possible class imbalance detected in model {model_name} "
                                         f"(weighted F1: {weighted_f1:.4f} vs macro F1: {macro_f1:.4f}).",
                                "action": "Consider techniques such as SMOTE, class weights, or stratified sampling "
                                         "to address class imbalance."
                            })
                            break
        else:  # regression
            # Check for high errors
            for model_name, metrics in self.results.items():
                if "r2_score" in metrics and metrics["r2_score"] < 0.5:
                    recommendations.append({
                        "type": "low_r2",
                        "message": f"Model {model_name} has a low R² score of {metrics['r2_score']:.4f}.",
                        "action": "Consider feature engineering, additional features, or non-linear models "
                                 "to improve regression performance."
                    })
                    break
        
        return recommendations
    
    def _convert_to_html(self, report: Dict) -> str:
        """Convert report dict to HTML format."""
        try:
            from jinja2 import Template
            
            # Simple HTML template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>FireAutoML Model Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333; }
                    .recommendation { margin: 10px 0; padding: 10px; border-left: 4px solid #2196F3; background-color: #E3F2FD; }
                    .issues { margin: 10px 0; padding: 10px; border-left: 4px solid red; background-color: rgb(247, 128, 128); }
                    .ds-info{padding:10px; }
                    .warning { border-left-color: #FFC107; background-color: #FFF8E1; }
                    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .viz-container { display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }
                    .viz-item { flex: 1; min-width: 300px; max-width: 500px; margin-bottom: 10px; }
                    img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <h1>FireAutoML Model Evaluation Report</h1>
                <p><strong>Generated at:</strong> {{ report.generated_at }}</p>
                <p><strong>Task type:</strong> {{ report.task_type }}</p>

                <div class="ds-info">
                    <h2>Dataset Info </h2>
                    {% if report.dataset_validation_results %}
                    <div class="recommendation">
                    <h2>Summary</h2>
                    <p>
                        <strong>Shape</strong>: {{report.dataset_validation_results.summary.shape}}
                    </p>
                    <p>
                        <strong>Memory Usage:</strong>: {{report.dataset_validation_results.summary.memory_usage}} Mb
                    </p>
                    <p>
                        <strong>Duplicate Rows:</strong>: {{report.dataset_validation_results.summary.duplicate_rows}}
                    </p>
                    <p>
                        <strong>Column Types:</strong>: {{report.dataset_validation_results.summary.column_types}}
                    </p>
                    <p>
                        <strong>Column Counts:</strong>: {{report.dataset_validation_results.summary.column_counts}}
                    </p>
                    <p>
                        <strong>Missing Values:</strong>: {{report.dataset_validation_results.summary.missing_values}}
                    </p>
                    </div>
                    <div class="issues">
                    <h2>Issues</h2>
                    <ul>
                        {% for item in report.dataset_validation_results.issues %}
                            <li>
                            {{item}}
                            </li>
                        {% endfor %}
                    </ul>
                    </div>
                    <div class="warning">
                    <h2>Warnings</h2>
                    <ul>
                        {% for item in report.dataset_validation_results.warnings %}
                            <li>
                            {{item}}
                            </li>
                        {% endfor %}
                    </ul>
                    </div>
                    <div class="recommendation">
                    <h2>Recommendations</h2>
                    <ul>
                        {% for item in report.dataset_validation_results.recommendations %}
                            <li>
                            {{item}}
                            </li>
                        {% endfor %}
                    </ul>
                    </div>
                    {% endif %}
                <div>
                
                <h2>Model Comparison Summary</h2>
                {% if report.summary.best_model %}
                <div class="recommendation">
                    <strong>Best model:</strong> {{ report.summary.best_model.name }} with 
                    {{ report.summary.best_model.primary_metric }} of 
                    {{ "%.4f"|format(report.summary.best_model.value) }}
                </div>
                {% endif %}
                
                <h2>Recommendations</h2>
                {% for rec in report.recommendations %}
                <div class="recommendation {% if rec.type == 'overfitting_warning' %}warning{% endif %}">
                    <p>{{ rec.message }}</p>
                    {% if rec.action %}<p><strong>Suggested action:</strong> {{ rec.action }}</p>{% endif %}
                </div>
                {% endfor %}
                
                <h2>Model Metrics Comparison</h2>
                {% for metric_name, metric_values in report.summary.metrics_compared.items() %}
                <h3>{{ metric_name }}</h3>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Value</th>
                    </tr>
                    {% for model_name, value in metric_values.items() %}
                    <tr>
                        <td>{{ model_name }}</td>
                        <td>{{ "%.4f"|format(value) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endfor %}
                
                {% if report.visualizations %}
                <h2>Visualizations</h2>
                <div class="viz-container">
                    {% for viz_path in report.visualizations %}
                    <div class="viz-item">
                        <img src="{{ viz_path }}" alt="Model visualization">
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <h2>Detailed Model Metrics</h2>
                {% for model_name, metrics in report.detailed_metrics.items() %}
                <h3>{{ model_name }}</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for metric_name, value in metrics.items() %}
                    {% if metric_name not in ['classification_report', 'confusion_matrix', 'model_params'] %}
                    <tr>
                        <td>{{ metric_name }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endif %}
                    {% endfor %}
                </table>
                {% endfor %}
                {% if report.model_paths %}
                <h2>Download Trained Models</h2>
                <ul>
                {% for model_name, path in report.model_paths.items() %}
                    <li><a href="{{ path }}">{{ model_name }}</a></li>
                {% endfor %}
                </ul>
                {% endif %}
            </body>
            </html>
            """
            
            template = Template(template_str)
            return template.render(report=report)
        except ImportError:
            logger.warning("Jinja2 not installed. Returning JSON report instead.")
            import json
            return "<pre>" + json.dumps(report, indent=2) + "</pre>"
    
    def _convert_to_markdown(self, report: Dict) -> str:
        """Convert report dict to Markdown format."""
        md = "# FireAutoML Model Evaluation Report\n\n"
        md += f"**Generated at:** {report['generated_at']}\n"
        md += f"**Task type:** {report['task_type']}\n\n"
        
        md += "## Model Comparison Summary\n\n"
        if "best_model" in report["summary"] and report["summary"]["best_model"]:
            best = report["summary"]["best_model"]
            md += f"**Best model:** {best['name']} with {best['primary_metric']} of {best['value']:.4f}\n\n"
        
        md += "## Recommendations\n\n"
        for rec in report["recommendations"]:
            md += f"- **{rec['message']}**\n"
            if "action" in rec:
                md += f"  - Suggested action: {rec['action']}\n"
        md += "\n"
        
        md += "## Model Metrics Comparison\n\n"
        for metric_name, metric_values in report["summary"]["metrics_compared"].items():
            md += f"### {metric_name}\n\n"
            md += "| Model | Value |\n"
            md += "| --- | --- |\n"
            for model_name, value in metric_values.items():
                md += f"| {model_name} | {value:.4f} |\n"
            md += "\n"
        
        if "visualizations" in report:
            md += "## Visualizations\n\n"
            for viz_path in report["visualizations"]:
                md += f"![Model visualization]({viz_path})\n\n"
        
        md += "## Detailed Model Metrics\n\n"
        for model_name, metrics in report["detailed_metrics"].items():
            md += f"### {model_name}\n\n"
            md += "| Metric | Value |\n"
            md += "| --- | --- |\n"
            for metric_name, value in metrics.items():
                if metric_name not in ['classification_report', 'confusion_matrix', 'model_params']:
                    if isinstance(value, float):
                        md += f"| {metric_name} | {value:.4f} |\n"
                    else:
                        md += f"| {metric_name} | {value} |\n"
            md += "\n"
        
        return md
