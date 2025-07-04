"""
Main pipeline for FireAutoML.
"""
import os
import logging
import pandas as pd
from typing import Dict, Any, Optional

from fireml.utils import (
    validate_dataframe,
    detect_dataset_type,
    is_classification_task,
    detect_target_column,
)
from fireml.preprocessing import (
    dataNorm,
    encoding,
    feature_selector,
    filterRedundantObject,
    manual_missing_NonObject_fix,
    manual_object_fix,
    balance_classes,
)
from fireml.models import train_models
from fireml.evaluation import ModelEvaluator
from fireml.settings import Settings

logger = logging.getLogger(__name__)

def run_full_pipeline(
    df: pd.DataFrame, 
    target_column: Optional[str] = None, 
    task_type: str = 'auto',
    output_dir: Optional[str] = None,
    run_deep_learning: bool = False
) -> Dict[str, Any]:
    """
    Execute the full FireAutoML pipeline on a given DataFrame.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        task_type: 'classification', 'regression', or 'auto'.
        output_dir: Directory to save results.
        run_deep_learning: Whether to include deep learning models.
        
    Returns:
        A dictionary containing the final evaluation report.
    """
    settings = Settings()
    if output_dir is None:
        output_dir = settings.output_directory
    os.makedirs(output_dir, exist_ok=True)

    # Target and Task Detection
    if not target_column:
        target_column = detect_target_column(df)
        if target_column is None:
            raise ValueError("No target column detected. Please specify a target column.")
        logger.info(f"Auto-detected target column: {target_column}")
    
    if task_type == 'auto':
        is_class, _ = is_classification_task(df[target_column])
        task_type = 'classification' if is_class else 'regression'
        logger.info(f"Auto-detected task type: {task_type}")

    # 2. Preprocessing
    # The functions for fixing missing values can drop rows, so we process the whole DataFrame
    # to keep features and target aligned.
    logger.info("Starting data preprocessing...")
    
    # Handle missing values
    df_clean = manual_missing_NonObject_fix(df, target_column=target_column, aggresive=True)
    df_clean = manual_object_fix(df_clean, target_column=target_column)
    
    # Re-split features and target from the cleaned dataframe
    target_clean = df_clean[target_column]
    features_clean = df_clean.drop(columns=[target_column])
    
    features_filtered = filterRedundantObject(features_clean)
    
    # 3. Balancing (for classification)
    if task_type == 'classification':
        logger.info("Balancing classes...")
        balanced_df = balance_classes(
            pd.concat([features_filtered, target_clean], axis=1),
            target_column=target_column
        )
        features_final = balanced_df.drop(columns=[target_column])
        target_final = balanced_df[target_column]
    else:
        features_final = features_filtered
        target_final = target_clean

    # 4. Feature Engineering
    normalized_dfs = dataNorm(features_final)
    selected_features, _ = feature_selector(normalized_dfs[0], target_final)
    encoded_features = encoding(selected_features)

    # Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features, target_final, test_size=0.2, random_state=42,
        stratify=target_final if task_type == 'classification' else None
    )

    #  Model Training
    logger.info(f"Training {task_type} models...")
    trained_models, _, _ = train_models(
        X_train, y_train, X_test, y_test, task_type=task_type
    )

    # Evaluation
    logger.info("Evaluating models...")
    evaluator = ModelEvaluator(task_type=task_type)
    for name, model in trained_models:
        evaluator.evaluate_model(name, model, X_test, y_test, X_train, y_train)

    # Reporting
    report = evaluator.generate_report(output_format='html')
    report_path = os.path.join(output_dir, "evaluation_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report) #type:ignore
    logger.info(f"Evaluation report saved to {report_path}")

    json_report = evaluator.generate_report(output_format='json')
    return json_report #type:ignore
