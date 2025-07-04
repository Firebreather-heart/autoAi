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
    from . import autoImputer
except ImportError:
    logger.error("Failed to import required local modules. Check your installation.")


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
    numeric_data = data.select_dtypes(include=np.number)
    
    if numeric_data.empty:
        logger.warning("No numeric data to normalize")
        return [data.copy(), data.copy(), data.copy()]

    stand = StandardScaler()
    minmax = MinMaxScaler(feature_range=(0, 1))
    norm = Normalizer()

    transformed_arrays = [
        stand.fit_transform(numeric_data),
        minmax.fit_transform(numeric_data),
        norm.fit_transform(numeric_data)
    ]

    # Convert back to DataFrames with original column names
    numeric_columns = numeric_data.columns
    normalized_dfs = [pd.DataFrame(arr, columns=numeric_columns, index=data.index) for arr in transformed_arrays]
    
    # If there were non-numeric columns, we need to add them back
    object_data = data.select_dtypes(exclude=np.number)
    if not object_data.empty:
        for i in range(len(normalized_dfs)):
            normalized_dfs[i] = pd.concat([normalized_dfs[i], object_data], axis=1)
    
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
    return encoded_data


def encode_cat(data:pd.DataFrame)->pd.DataFrame:
    logger.info("encoding catergorical features")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    categorical_data = data.select_dtypes(include='object').columns
    for col in categorical_data:
        data[col] = le.fit_transform(data[col])
    return data


def feature_selector(data: pd.DataFrame, 
                     target: pd.Series, 
                     style='corr_check')-> Tuple[pd.DataFrame, pd.Series]:
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
        for i in dat.columns:
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
    return data, target


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


def manual_missing_NonObject_fix(data: pd.DataFrame, target_column: Optional[str] = None, aggresive: bool = True, imputation_method: Literal['mean', 'median', 'most_frequent'] = "mean") -> pd.DataFrame:
    """
    Handle missing values in numerical columns.
    Processes the full DataFrame (features + target) to keep alignment.
    """
    result = data.drop_duplicates().copy()
    numeric_cols = result.select_dtypes(include=np.number).columns
    if target_column and target_column in numeric_cols:
        numeric_cols = numeric_cols.drop(target_column)

    cols_to_impute = []
    for column in numeric_cols:
        missing_count = result[column].isnull().sum()
        if missing_count > 0:
            if missing_count > 0.1 * len(result) and aggresive:
                logger.info(f"Dropping column '{column}' with {missing_count} missing values ({missing_count/len(result):.2%})")
                result = result.drop(column, axis='columns')
            else:
                cols_to_impute.append(column)
    if cols_to_impute:
        logger.info(f"Imputing columns {cols_to_impute} with '{imputation_method}' strategy.")
        result[cols_to_impute] = autoImputer.impute(result[cols_to_impute], method=imputation_method)
    return result


def manual_object_fix(data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Handle missing values in categorical columns.
    Processes the full DataFrame (features + target) to keep alignment.
    """
    logger.info('Fixing missing categorical values...')
    result = data.drop_duplicates().copy()
    object_cols = result.select_dtypes(include='object').columns
    if target_column and target_column in object_cols:
        object_cols = object_cols.drop(target_column)
    for column in object_cols:
        missing_count = result[column].isnull().sum()
        if missing_count > 0:
            if missing_count > 0.1 * len(result):
                logger.info(f"Dropping column '{column}' with {missing_count} missing values")
                result = result.drop(column, axis='columns')
            else:
                result = result.dropna(axis=0)
    logger.info('Missing values fixed')
    return result
