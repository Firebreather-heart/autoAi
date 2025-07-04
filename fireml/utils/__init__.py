"""
The `fireml.utils` package contains helper functions for validation,
logging, and other miscellaneous tasks.
"""
from fireml.utils.helpers import split_by_sign, fill_to_threshold, serialize
from fireml.utils.logging_config import setup_logging
from fireml.utils.mlmath import approximator, fastForward, breakDown
from fireml.utils.validation import (
    validate_dataframe, 
    detect_dataset_type,
    is_classification_task,
    # detect_data_problems,
    detect_target_column
)

__all__ = [
    'split_by_sign', 
    'fill_to_threshold',
    'approximator', 
    'fastForward', 
    'breakDown',
    'validate_dataframe', 
    'detect_dataset_type',
    'is_classification_task',
    # 'detect_data_problems',
    'detect_target_column',
    'serialize',
    'setup_logging'
]
