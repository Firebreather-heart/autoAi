"""
Feature engineering and data preprocessing utilities for fireAutoML.
"""
import logging
import sys
from typing import List, Tuple, Dict, Optional, Union, Literal
import pandas as pd
import numpy as np



logging.basicConfig(level=logging.INFO,)
logger = logging.getLogger(__name__)

try:
    from fireml import confirmInstallLib
    from fireml.preprocessing import autoImputer
    confirmInstallLib('sklearn')
except ImportError:
    logger.error("Failed to import required local modules. Check your installation.")
    sys.exit(1)


def check_corr(main: pd.Series, scd:pd.Series)->float:
    """Calculate correlation between two series."""
    if not isinstance(main, pd.Series):
        raise TypeError("main must be a pandas Series")
    if not isinstance(scd, pd.Series):
        raise TypeError("scd must be a pandas Series")
    return main.corr(scd)


def dataNorm(data: pd.DataFrame, ):
    """
    Normalize data using different scalers.
    
    Returns a list of pandas dataframe objects containing three categories of data:
    standard scaled data, MinMax scaled and normalized data. This allows
    flexibility in choosing models trained on the three data forms.
    
    Args:
        data: DataFrame to normalize
        
    Returns:
        List of normalized DataFrames [standard_scaled, minmax_scaled, normalized]
    """
    # normalises the given data
    logger.info('Normalising data..............\n')
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

    # Extract only numeric columns for scaling
    data_to_normalize = data.select_dtypes(exclude='object')
    
    if data_to_normalize.empty:
        logger.warning("No numeric data to normalize")
        return [data.copy(), data.copy(), data.copy()]

    stand = StandardScaler()
    minmax = MinMaxScaler(feature_range=(0, 1))
    norm = Normalizer()

    transformed_arrays = [
        stand.fit_transform(data_to_normalize),
        minmax.fit_transform(data_to_normalize),
        norm.fit_transform(data_to_normalize)
    ]

    # Convert back to DataFrames with original column names
    numeric_columns = data_to_normalize.columns
    normalized_dfs = [pd.DataFrame(arr, columns=numeric_columns) for arr in transformed_arrays]
    
    # If there were non-numeric columns, we need to keep them in the output
    categorical_data = data.select_dtypes(include='object')
    if not categorical_data.empty:
        for df in normalized_dfs:
            for col in categorical_data.columns:
                df[col] = data[col].values
    
    return normalized_dfs


def encoding(data: pd.DataFrame,):
    """
    Encode categorical variables using one-hot encoding.
    
    CAUTION: Use after feature selection for best results.
    
    Args:
        data: DataFrame to encode
        
    Returns:
        Encoded DataFrame
    """
    encoded_data = pd.get_dummies(data)
    logger.info(f"Encoded data shape: {encoded_data.shape}")
    return data


def feature_selector(data: pd.DataFrame, target: pd.Series, style='corr_check'):
    '''
        This will select the best features from the given
        dataset.
        Params:
            data is any pandas Dataframe Object
            target is a pandas series object
            style: defaults to corr_check
            the corr_check style will select the features with high
            negative or positive correlation.
            the other option is the svr_model
    '''
    logger.info('Selecting best features...')
    from fireml.utils.helpers import split_by_sign, fill_to_threshold
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    dat = data.copy()
    dat = pd.get_dummies(dat)
    y = target
    X = dat

    if len(data.columns) < 11:
        return data, target
    
    if style == 'svr_model':
        model = SVR(kernel='linear')
        rfe = RFE(model, n_features_to_select=max(1, int(
            len(data.columns)*0.5)), step=1)
        selections = rfe.fit(X, y)
        selected_features = selections.get_feature_names_out()
        return X[selected_features], y

    elif style == 'corr_check':
        buc = dict()
        for i in data.columns:
            a = check_corr(y, dat[i])
            buc[a] = i
        bucp, bucn = split_by_sign(list(buc.keys()))
        whole = fill_to_threshold(
            bucp, bucn, threshold=int(len(data.columns)*0.5))
        
        selected_columns = [buc.get(val) for val in whole if val in buc]
        
        if not selected_columns:
            logger.warning("No features selected by correlation, returning original data")
            return data, target
            
        return X[selected_columns], y


def filterRedundantObject(data: pd.DataFrame,):
    """
    Remove columns with too many unique categorical values.
    
    Columns with more than 50 unique values are considered potentially irrelevant
    and are removed.
    
    Args:
        data: DataFrame to filter
        
    Returns:
        Filtered DataFrame
    """
    result = data.copy()
    
    for column in result.select_dtypes(include='object').columns:
        unique_count = len(set(result[column].values))
        if unique_count > 50:
            logger.info(f"Dropping column '{column}' with {unique_count} unique values")
            result = result.drop(column, axis='columns')
            
    return result


def manual_missing_NonObject_fix(data: pd.DataFrame, target=None, aggresive: bool = True, imputation_method: Literal['mean'] = "mean") -> pd.DataFrame:
    """
    Handle missing values in numerical columns.
    
    If missing values exceed one-tenth the column length and aggressive=True,
    the column is dropped. Otherwise, values are imputed.
    
    Args:
        data: DataFrame to process
        target: Target column name (not used but kept for API compatibility)
        aggresive: Whether to automatically drop columns with many missing values
        imputation_method: What method to be used to fill missing values [mean, median,
        most_frequent, constant], for a constant a string of the constant should be passed
        
    Returns:
        DataFrame with missing values handled
    """
    result = data.drop_duplicates()
    
    # Check missing values in numeric columns
    for column in result.select_dtypes(exclude='object').columns:
        missing_count = result[column].isnull().sum()
        
        # If there are missing values
        if missing_count > 0:
            # If more than 10% are missing
            if missing_count > 0.1 * len(result):
                if aggresive:
                    logger.info(f"Dropping column '{column}' with {missing_count} missing values")
                    result = result.drop(column, axis='columns')
                else:
                    if imputation_method == 'constant':
                        raise ValueError("Cannot use constant imputation")
                    logger.info(f"Imputing column '{column}' with {imputation_method} values")
                    result[column] = autoImputer.impute(pd.DataFrame(result[column]), method='mean')
            else:
                result = result.dropna(axis=0)
                
    return result


def manual_object_fix(data: pd.DataFrame, target=None) -> pd.DataFrame:
    """
    Handle missing values in categorical columns.
    
    Drops columns with >10% missing values, otherwise drops rows with missing values.
    
    Args:
        data: DataFrame to process
        target: Target column name (not used but kept for API compatibility)
        
    Returns:
        DataFrame with missing categorical values handled
    """
    logger.info('Fixing missing categorical values...')
    
    # Make a copy and remove duplicates
    result = data.drop_duplicates()
    
    # Process each categorical column
    for column in result.select_dtypes(include='object').columns:
        missing_count = result[column].isnull().sum()
        
        # If there are missing values
        if missing_count > 0:
            if missing_count > 0.1 * len(result):
                logger.info(f"Dropping column '{column}' with {missing_count} missing values")
                result = result.drop(column, axis='columns')
            else:
                # Drop rows with missing values
                result = result.dropna(axis=0)
                
    logger.info('Missing values fixed')
    return result
