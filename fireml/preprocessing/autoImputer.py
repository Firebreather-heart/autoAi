from typing import Literal
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

def impute(data: pd.DataFrame, method: Literal['mean', 'median', 'most_frequent'] = 'mean', fill_value=None) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame using scikit-learn's IterativeImputer.

    Args:
        data: DataFrame with missing values.
        method: The imputation strategy.
        fill_value: Not used by IterativeImputer, kept for API consistency.

    Returns:
        DataFrame with missing values imputed.
    """
    # Ensure the index is reset so that row alignment is preserved after imputation
    original_index = data.index
    imr = IterativeImputer(initial_strategy=method, random_state=0)
    imputed_data = imr.fit_transform(data)
    # Return DataFrame with the same index and columns as the input
    return pd.DataFrame(imputed_data, columns=data.columns, index=original_index)

