import logging
from functools import partial
from fireml.utils.helpers import serialize

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Union, Literal
from threading import Thread
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             RandomForestRegressor, AdaBoostRegressor,
                             VotingClassifier, VotingRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import f1_score, mean_absolute_error, classification_report
from sklearn.base import BaseEstimator
from xgboost import (XGBClassifier, XGBRegressor) #type:ignore

logger = logging.getLogger(__name__)


def train_model_thread(model_tuple: Tuple[BaseEstimator, str], 
                      data: pd.DataFrame, 
                      target: Union[pd.Series, np.ndarray], 
                      test_data: pd.DataFrame,
                      test_target: Union[pd.Series, np.ndarray],
                      results: Dict,
                      metric_func: Callable) -> None:
    """
    Train a model in a separate thread and store results.
    
    Args:
        model_tuple: Tuple of (model, name)
        data: Training data
        target: Training targets
        test_data: Test data
        test_target: Test targets
        results: Dictionary to store results
        metric_func: Function to calculate performance metric
    """
    model_obj, name = model_tuple
    
    try:
        logger.info(f"Training {name} model...")
        model = model_obj.fit(data, target) #type:ignore
        
        # Make predictions
        pred = model.predict(test_data)
        
        # Calculate metrics
        if hasattr(model, "predict_proba") and metric_func == f1_score:
            # For classifiers
            logger.info(f"{name} model evaluation:")
            report = classification_report(test_target, pred)
            logger.info(f"\n{report}")
            metric_value = metric_func(test_target, pred)
        else:
            # For regressors
            metric_value = metric_func(test_target, pred)
            logger.info(f"{name} model error: {metric_value}")
        
        # Store results
        results["trained_models"].append((name, model))
        results["predictions"].append((name, pred))
        results["metrics"][(name, model)] = metric_value
        
        logger.info(f"Finished training {name} model")
    except Exception as e:
        logger.error(f"Error training {name} model: {e}")


def create_ensemble(model_results: Dict, 
                   data: pd.DataFrame, 
                   target: Union[pd.Series, np.ndarray],
                   test_data: pd.DataFrame,
                   test_target: Union[pd.Series, np.ndarray],
                   task_type: str = "classification",
                   voting: Literal['hard', 'soft'] = "soft",
                   n_jobs: int = -1) -> Tuple[BaseEstimator, float]:
    """
    Create an ensemble model from the best trained models.
    
    Args:
        model_results: Dictionary with training results
        data: Training data
        target: Training targets
        test_data: Test data
        test_target: Test targets
        task_type: Type of task ("classification" or "regression")
        voting: Voting strategy for classifier
        n_jobs: Number of jobs to run in parallel
    
    Returns:
        Tuple of (ensemble model, metric score)
    """
    # Sort models by performance
    metrics_items = sorted(
        model_results["metrics"].items(), 
        key=lambda x: x[1], 
        reverse=(task_type == "classification")  # Higher is better for classification
    )
    
    # Select top 3 models
    top_models = []

    if len(metrics_items) == 0:
        logger.error("No successful models to ensemble.")
        raise ValueError("No successful models to ensemble.")
    elif len(metrics_items) <= 3:
        top_models = [item[0] for item in metrics_items]
    else:
        top_models = [metrics_items.pop(0)[0] for _ in range(3)]

    # Create ensemble
    if task_type == "classification":
        ensemble = VotingClassifier(estimators=top_models, voting=voting, n_jobs=n_jobs)
        metric_func = get_f1_metric(target)
    else:
        ensemble = VotingRegressor(estimators=top_models, n_jobs=n_jobs)
        metric_func = mean_absolute_error
    
    # Train ensemble
    try:
        logger.info("Training ensemble model...")
        ensemble.fit(data, target)
        
        # Evaluate ensemble
        ensemble_pred = ensemble.predict(test_data)
        ensemble_score = metric_func(test_target, ensemble_pred,)
        
        if task_type == "classification":
            logger.info("Ensemble model evaluation:")
            report = classification_report(test_target, ensemble_pred)
            logger.info(f"\n{report}")
            logger.info(f"Ensemble f1 score: {ensemble_score:.4f}")
        else:
            logger.info(f"Ensemble MAE: {ensemble_score:.4f}")
        
        return ensemble, ensemble_score #type:ignore
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}")
        return top_models[0][1], metrics_items[0][1]


def get_f1_metric(target):
    n_classes = len(np.unique(target))
    if n_classes == 2:
        return partial(f1_score, average='binary')
    else:
        return partial(f1_score, average='weighted')


def train_models(data: pd.DataFrame, 
                target: Union[pd.Series, np.ndarray],
                test_data: pd.DataFrame,
                test_target: Union[pd.Series, np.ndarray],
                task_type: str = "classification",
                voting: Literal['hard', 'soft'] = "soft") -> Tuple[List, List, Dict]:
    """
    Train multiple models and create an ensemble.
    
    Args:
        data: Training data
        target: Training targets
        test_data: Test data
        test_target: Test targets
        task_type: Type of task ("classification" or "regression")
        voting: Voting strategy for classifier ensemble
    
    Returns:
        Tuple of (trained models, predictions, metrics dictionary)
    """
        # Common parameters
    random_state = 2000
    n_jobs = -1
    learning_rate = 0.001

    #encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    target = le.fit_transform(target) #type:ignore
    test_target = le.transform(test_target) #type:ignore
    
    # Select models based on task type
    if task_type == "classification":
        models = [
            (MLPClassifier(early_stopping=True, max_iter=100, verbose=True, 
                          random_state=random_state, learning_rate='adaptive',
                          n_iter_no_change=10, hidden_layer_sizes=(10,),
                          warm_start=True), 'MLP'),
            (RandomForestClassifier(n_estimators=500, n_jobs=n_jobs, 
                                   random_state=random_state, warm_start=True), 'RFC'),
            (XGBClassifier(n_estimators=500, n_jobs=n_jobs, use_label_encoder=False), 'XGB'),
            (AdaBoostClassifier(n_estimators=500, learning_rate=learning_rate, 
                               random_state=random_state), 'ABC'),
            (SVC(probability=True), 'SVC'),
            (GaussianNB(), 'GNB'),
            (KNeighborsClassifier(algorithm='brute', n_jobs=n_jobs, leaf_size=60), 'KNC')
        ]
        metric_func = get_f1_metric(target)
    else:  # regression
        models = [
            (MLPRegressor(early_stopping=False, max_iter=1000, verbose=True, 
                         random_state=random_state, learning_rate='adaptive',
                         n_iter_no_change=10, hidden_layer_sizes=(100,),
                         warm_start=True), 'MLP'),
            (RandomForestRegressor(n_estimators=500, n_jobs=n_jobs, 
                                  random_state=random_state, warm_start=True), 'RFR'),
            (XGBRegressor(n_estimators=500, n_jobs=n_jobs), 'XGB'), 
            (AdaBoostRegressor(n_estimators=500, learning_rate=learning_rate, 
                              random_state=random_state), 'ABR'),
            (SVR(), 'SVR'),
            (KNeighborsRegressor(algorithm='brute', n_jobs=n_jobs, leaf_size=60), 'KNR')
        ]
        metric_func = mean_absolute_error
        # Results container
    results = {
        "trained_models": [],
        "predictions": [],
        "metrics": {}
    }
    
    # Train models in parallel
    threads = []
    for model_tuple in models:
        thread = Thread(
            target=train_model_thread,
            args=(model_tuple, data, target, test_data, test_target, results, metric_func)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Create ensemble
    ensemble, ensemble_score = create_ensemble(
        results, data, target, test_data, test_target, task_type, voting, n_jobs
    )
    
    # Add ensemble to results
    ensemble_name = "VotingClassifier" if task_type == "classification" else "VotingRegressor"
    results["trained_models"].append((ensemble_name, ensemble))
    ensemble_pred = ensemble.predict(test_data) #type:ignore
    results["predictions"].append((ensemble_name, ensemble_pred))
    results["metrics"][(ensemble_name, ensemble)] = ensemble_score
    
    # Save top models
    top_models = sorted(
        results["metrics"].items(), 
        key=lambda x: x[1], 
        reverse=(task_type == "classification")
    )[:3]
    
    for (name, model), _ in top_models:
        serialize(model, f"{name}_model")
    
    # Save ensemble
    serialize(ensemble, f"{ensemble_name}")
    
    # Format results
    sorted_metrics = sorted(
        results["metrics"].items(), 
        key=lambda x: x[1], 
        reverse=(task_type == "classification")
    )
    best_model_name = sorted_metrics[0][0][0]
    best_score = sorted_metrics[0][1]
    
    if task_type == "classification":
        logger.info(f"{best_model_name} appears to be the best performer with f1 score of {best_score:.4f}")
    else:
        logger.info(f"{best_model_name} appears to be the best performer with error of {best_score:.4f}")
    
    return results["trained_models"], results["predictions"], dict(sorted_metrics)

