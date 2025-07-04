from fireml.models import train_models

import numpy as np
import pandas as pd
from typing import Literal, Union, Tuple, List, Dict

def makeRegressors(data: pd.DataFrame, 
                  target: Union[pd.Series, np.ndarray],
                  test_data: pd.DataFrame,
                  test_target: Union[pd.Series, np.ndarray]) -> Tuple[List, List, Dict]:
    """
    Train regression models.
    
    Args:
        data: Training data
        target: Training targets
        test_data: Test data
        test_target: Test targets
    
    Returns:
        Tuple of (trained models, predictions, metrics dictionary)
    """
    return train_models(data, target, test_data, test_target, "regression")