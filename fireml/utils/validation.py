import logging
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    results = {
        "issues": [],
        "warnings": [],
        "recommendations": [],
        "summary": {},
    }

    results["summary"]["shape"] = df.shape
    results["summary"]["memory_usage"] = df.memory_usage(deep=True).sum() / (1024 * 1024)

    duplicate_count = df.duplicated().sum()
    results["summary"]["duplicate_rows"] = duplicate_count
    if duplicate_count > 0:
        results["issues"].append(f"Found {duplicate_count} duplicate rows")
        results["recommendations"].append("Consider removing duplicate rows with df.drop_duplicates()")

    results["summary"]["column_types"] = df.dtypes.astype(str).to_dict()
    results["summary"]["column_counts"] = {
        "numeric": len(df.select_dtypes(include=['number']).columns),
        "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
        "datetime": len([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]),
        "boolean": len(df.select_dtypes(include=['bool']).columns),
        "other": len(df.select_dtypes(exclude=['number', 'object', 'category', 'datetime64[ns]', 'bool']).columns)
    }

    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_info = {
        col: {"count": int(count), "percentage": float(percentage)}
        for col, count, percentage in zip(missing_values.index, missing_values, missing_percentages)
        if count > 0
    }
    results["summary"]["missing_values"] = missing_info

    if missing_info:
        for col, info in missing_info.items():
            if info["percentage"] > 30:
                results["issues"].append(f"Column '{col}' has {info['percentage']:.1f}% missing values")
                results["recommendations"].append(f"Consider dropping column '{col}' or using advanced imputation")
            elif info["percentage"] > 5:
                results["warnings"].append(f"Column '{col}' has {info['percentage']:.1f}% missing values")
                results["recommendations"].append(f"Consider imputing missing values in '{col}'")

    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count == 1:
            results["issues"].append(f"Column '{col}' has only one unique value")
            results["recommendations"].append(f"Consider dropping column '{col}'")
        elif unique_count == 2 and df[col].dtype in ['float64', 'int64'] and set(df[col].unique()) == {0, 1}:
            results["warnings"].append(f"Column '{col}' appears to be binary but is numeric type")
            results["recommendations"].append(f"Consider converting '{col}' to boolean type")

    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique()
        if unique_count > 100:
            results["warnings"].append(f"Column '{col}' has high cardinality ({unique_count} unique values)")
            results["recommendations"].append(f"Consider using frequency encoding or embedding for '{col}'")
        elif unique_count > len(df) * 0.9:
            results["issues"].append(f"Column '{col}' has too many unique values ({unique_count}), possibly an ID")
            results["recommendations"].append(f"Consider dropping '{col}' or using it only for grouping")

    if len(df.columns) > 1:
        results["warnings"].append("Review your features for potential data leakage before modeling")

    for col in df.select_dtypes(include=['number']).columns:
        if df[col].count() > 0:
            skew = df[col].skew()
            if isinstance(skew, (int, float)) and not np.isnan(skew) and abs(skew) > 3:
                results["warnings"].append(f"Column '{col}' is highly skewed (skew={skew:.2f})")
                results["recommendations"].append(f"Consider applying log or other transformation to '{col}'")

    if results["issues"] or results["warnings"]:
        results["recommendations"].append("Run df.describe() and df.info() for more detailed statistics")

    return results

def detect_dataset_type(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    result = {
        "primary_type": "tabular",
        "subtypes": [],
        "confidence": 1.0,
        "recommendations": []
    }

    text_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        sample_pool = df[col].dropna()
        if sample_pool.empty:
            continue
        sample = sample_pool.sample(min(100, len(sample_pool)), random_state=42).astype(str).reset_index(drop=True)
        avg_len = sample.apply(len).mean()
        max_len = sample.apply(len).max()
        if avg_len > 50 or max_len > 200:
            text_cols.append(col)

    if text_cols:
        result["subtypes"].append("text")
        result["text_columns"] = text_cols
        result["recommendations"].append("Consider using NLP techniques for text columns")

    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    date_like_cols = []

    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(100).astype(str)
        if sample.empty:
            continue
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
        for pattern in date_patterns:
            if sample.str.match(pattern).mean() > 0.7:
                date_like_cols.append(col)
                break

    if datetime_cols or date_like_cols:
        result["subtypes"].append("time_series")
        result["datetime_columns"] = datetime_cols + date_like_cols
        result["recommendations"].append("Consider time series analysis techniques")

    image_cols = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(100).astype(str)
        if sample.empty:
            continue
        if any(sample.str.lower().str.endswith(tuple(image_extensions))):
            image_cols.append(col)

    if image_cols:
        result["subtypes"].append("image_paths")
        result["image_columns"] = image_cols
        result["recommendations"].append("Consider using computer vision techniques for image columns")

    if "text" in result["subtypes"] and len(text_cols) > len(df.columns) / 2:
        result["primary_type"] = "text"
        result["confidence"] = 0.8
    elif "time_series" in result["subtypes"]:
        result["primary_type"] = "time_series"
        result["confidence"] = 0.7
    elif "image_paths" in result["subtypes"]:
        result["primary_type"] = "image"
        result["confidence"] = 0.6

    return result

def is_classification_task(target_series: pd.Series) -> Tuple[bool, float]:
    unique_count = target_series.nunique()
    total_count = len(target_series)

    if unique_count <= 10:
        if target_series.dtype in ['object', 'category', 'bool']:
            return True, 0.95
        elif set(target_series.unique()).issubset({0, 1}) or set(target_series.unique()).issubset({-1, 0, 1}):
            return True, 0.98
        else:
            return True, 0.90
    elif unique_count <= 20:
        if target_series.dtype in ['object', 'category']:
            return True, 0.85
        elif target_series.dropna().apply(lambda x: float(x).is_integer()).all():
            return True, 0.75
        else:
            return False, 0.60
    else:
        if target_series.dtype in ['object', 'category']:
            return True, 0.70
        else:
            return False, 0.90

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    common_names = ["target", "label", "class", "y", "output", "result", "response", "outcome", "price", "sales", "revenue", "profit", "loss", "score", "rating", "category"]
    common_names = [name.lower() for name in common_names]
    for name in common_names:
        for col in df.columns:
            if col.lower() == name:
                return col
    if df.columns.size > 0:
        logger.warning("Using fallback target column: %s", df.columns[-1])
        return df.columns[-1]
    return None