"""
The `fireml.preprocessing` package contains modules for data cleaning,
feature engineering, and data augmentation.
"""

from .autoImputer import impute
from .data_augmentation import (
    balance_classes,
    augment_tabular_data,
    augment_text_data,
    augment_image_data
)
from .feature_engineering import (
    check_corr,
    dataNorm,
    encoding,
    feature_selector,
    filterRedundantObject,
    manual_missing_NonObject_fix,
    manual_object_fix
)

__all__ = [
    'dataNorm', 
    'encoding', 
    'feature_selector',
    'filterRedundantObject', 
    'manual_missing_NonObject_fix', 
    'manual_object_fix',
    'impute',
    'augment_tabular_data',
    'augment_text_data',
    'augment_image_data',
    'balance_classes'
]
