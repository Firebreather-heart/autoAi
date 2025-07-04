from fireml.models import train_models

import numpy as np
import pandas as pd
from typing import Literal, Union, Tuple, List, Dict

def makeClassifiers(data: pd.DataFrame, 
                   target: Union[pd.Series, np.ndarray],
                   test_data: pd.DataFrame,
                   test_target: Union[pd.Series, np.ndarray],
                   voting: Literal['soft', 'hard'] = "soft") -> Tuple[List, List, Dict]:
    """
    Train classification models.
    
    Args:
        data: Training data
        target: Training targets
        test_data: Test data
        test_target: Test targets
        voting: Voting strategy for ensemble
    
    Returns:
        Tuple of (trained models, predictions, metrics dictionary)
    """
    return train_models(data, target, test_data, test_target, "classification", voting)

   