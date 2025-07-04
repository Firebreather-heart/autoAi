"""
Data loading utilities for FireAutoML.

This module provides functions to load data from various sources
including CSV, Excel, JSON, databases, and more.
"""
import os
import logging
from typing import  Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(source: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load data from various sources based on file extension or source type.
    
    Args:
        source: Path to file or connection string
        **kwargs: Additional arguments for the specific loader
        
    Returns:
        Tuple of (DataFrame, metadata dictionary)
    """
    metadata = {
        "source": source,
        "loader_used": None,
        "original_shape": None,
        "warnings": [],
        "errors": []
    }
    
    try:
        # Determine source type
        if isinstance(source, str):
            if os.path.exists(source):
                # It's a file path
                _, ext = os.path.splitext(source.lower())
                
                if ext == '.csv':
                    df = load_csv(source, **kwargs)
                    metadata["loader_used"] = "csv"
                elif ext in ['.xls', '.xlsx', '.xlsm']:
                    df = load_excel(source, **kwargs)
                    metadata["loader_used"] = "excel"
                elif ext in ['.json', '.jsn']:
                    df = load_json(source, **kwargs)
                    metadata["loader_used"] = "json"
                elif ext in ['.parquet', '.pqt']:
                    df = load_parquet(source, **kwargs)
                    metadata["loader_used"] = "parquet"
                elif ext in ['.pickle', '.pkl']:
                    df = pd.read_pickle(source)
                    metadata["loader_used"] = "pickle"
                elif ext in ['.h5', '.hdf5']:
                    df = pd.read_hdf(source, **kwargs)
                    metadata["loader_used"] = "hdf"
                elif ext in ['.txt', '.dat']:
                    df = load_text(source, **kwargs)
                    metadata["loader_used"] = "text"
                elif ext in ['.db', '.sqlite', '.sqlite3']:
                    if 'query' in kwargs:
                        df = load_sql_database(source, kwargs.pop('query'), **kwargs)
                        metadata["loader_used"] = "sqlite"
                    else:
                        raise ValueError("Query must be provided for database sources")
                else:
                    raise ValueError(f"Unsupported file extension: {ext}")
            elif source.startswith(('mysql://', 'postgresql://', 'sqlite://', 'mssql://')):
                # SQLAlchemy connection string
                if 'query' in kwargs:
                    df = load_sql_database(source, kwargs.pop('query'), **kwargs)
                    metadata["loader_used"] = "sql_alchemy"
                else:
                    raise ValueError("Query must be provided for database sources")
            else:
                raise ValueError(f"Source not found or not supported: {source}")
        else:
            raise TypeError("Source must be a string path or connection string")
        
        metadata["original_shape"] = df.shape
        logger.info(f"Loaded data from {source} with shape {df.shape}")
        
        return df, metadata
        
    except Exception as e:
        logger.error(f"Error loading data from {source}: {str(e)}")
        metadata["errors"].append(str(e))
        # Return empty DataFrame if loading fails
        return pd.DataFrame(), metadata


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file with intelligent type inference.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments for pandas.read_csv
        
    Returns:
        Pandas DataFrame
    """
    # Set sensible defaults
    default_kwargs = {
        'low_memory': False,
        'encoding': 'utf-8',
        'encoding_errors': 'ignore'
    }
    
    # Update with user-provided kwargs
    default_kwargs.update(kwargs)
    
    # Try to infer separator if not provided
    if 'sep' not in default_kwargs and 'delimiter' not in default_kwargs:
        with open(path, 'r', encoding=default_kwargs['encoding'], errors=default_kwargs['encoding_errors']) as f:
            sample = f.readline()
            
        if sample.count(';') > sample.count(','):
            default_kwargs['sep'] = ';'
        elif sample.count('\t') > sample.count(','):
            default_kwargs['sep'] = '\t'
            
    # Try to detect date columns
    try:
        df = pd.read_csv(path, **default_kwargs)
        
        # Try to convert string columns to datetime if they look like dates
        for col in df.select_dtypes(include=['object']).columns:
            try:
                date_series = pd.to_datetime(df[col], errors='coerce')
                # If most values converted successfully, use the datetime
                if date_series.notna().mean() > 0.7:
                    df[col] = date_series
                    logger.info(f"Converted column {col} to datetime")
            except:
                pass
                
        return df
    except Exception as e:
        # If normal reading fails, try with different encoding
        try:
            default_kwargs['encoding'] = 'latin1'
            return pd.read_csv(path, **default_kwargs)
        except:
            # Re-raise original exception if all attempts fail
            raise e


def load_excel(path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Args:
        path: Path to Excel file
        **kwargs: Additional arguments for pandas.read_excel
        
    Returns:
        Pandas DataFrame
    """
    # Check for sheet name
    if 'sheet_name' not in kwargs:
        # If sheet not specified, try to get a list of sheets first
        import openpyxl
        try:
            wb = openpyxl.load_workbook(path, read_only=True)
            sheets = wb.sheetnames
            if len(sheets) > 1:
                logger.info(f"Multiple sheets found: {sheets}. Using first sheet by default.")
                kwargs['sheet_name'] = sheets[0]
        except:
            pass
    
    return pd.read_excel(path, **kwargs)


def load_json(path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        **kwargs: Additional arguments for pandas.read_json
        
    Returns:
        Pandas DataFrame
    """
    try:
        return pd.read_json(path, **kwargs)
    except ValueError:
        # Handle case where JSON is not well-formed for pandas
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Try to normalize the JSON data into a DataFrame
        if isinstance(data, list):
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                return pd.json_normalize(data['data'])
            else:
                return pd.json_normalize([data])
        else:
            raise ValueError("JSON data structure not supported")


def load_parquet(path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from Parquet file.
    
    Args:
        path: Path to Parquet file
        **kwargs: Additional arguments for pandas.read_parquet
        
    Returns:
        Pandas DataFrame
    """
    return pd.read_parquet(path, **kwargs)


def load_text(path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from text file with delimiter inference.
    
    Args:
        path: Path to text file
        **kwargs: Additional arguments for pandas.read_csv
        
    Returns:
        Pandas DataFrame
    """
    # Try to infer delimiter
    with open(path, 'r', encoding=kwargs.get('encoding', 'utf-8'), errors='ignore') as f:
        sample_lines = [f.readline().strip() for _ in range(5)]
    
    # Count potential delimiters
    delimiters = [',', '\t', ';', '|', ' ']
    delimiter_counts = {}
    
    for delimiter in delimiters:
        # Count how consistent the number of fields is with this delimiter
        field_counts = [line.count(delimiter) + 1 for line in sample_lines if line]
        if field_counts:
            # Use the most common count
            from collections import Counter
            most_common_count = Counter(field_counts).most_common(1)[0][0]
            matches = sum(1 for count in field_counts if count == most_common_count)
            delimiter_counts[delimiter] = matches
    
    # Use the delimiter with the most consistent field count
    best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
    
    default_kwargs = {
        'sep': best_delimiter,
        'encoding': 'utf-8',
        'encoding_errors': 'ignore'
    }
    default_kwargs.update(kwargs)
    
    return pd.read_csv(path, **default_kwargs)


def load_sql_database(connection_string: str, query: str, **kwargs) -> pd.DataFrame:
    """
    Load data from SQL database.
    
    Args:
        connection_string: Database connection string or path to SQLite file
        query: SQL query to execute
        **kwargs: Additional arguments for pandas.read_sql
        
    Returns:
        Pandas DataFrame
    """
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine, **kwargs)
    except ImportError:
        logger.warning("SQLAlchemy not installed. Please install with 'pip install sqlalchemy' or 'pip install fireml[full]'. Using direct sqlite3 connection for now.")
        import sqlite3
        conn = sqlite3.connect(connection_string)
        return pd.read_sql(query, conn, **kwargs)


def save_data(df: pd.DataFrame, path: str, format: str = None, **kwargs) -> bool: #type:ignore
    """
    Save DataFrame to various formats.
    
    Args:
        df: DataFrame to save
        path: Path to save file
        format: Format to save in (csv, excel, parquet, etc.)
        **kwargs: Additional arguments for the specific saver function
        
    Returns:
        True if saved successfully
    """
    try:
        # Determine format from path if not provided
        if format is None:
            _, ext = os.path.splitext(path.lower())
            format = ext.lstrip('.')
        
        # Normalize format
        format = format.lower()
        
        if format in ['csv', '.csv']:
            df.to_csv(path, index=kwargs.pop('index', False), **kwargs)
        elif format in ['excel', 'xlsx', '.xlsx', 'xls', '.xls']:
            df.to_excel(path, index=kwargs.pop('index', False), **kwargs)
        elif format in ['parquet', '.parquet', 'pqt', '.pqt']:
            df.to_parquet(path, index=kwargs.pop('index', None), **kwargs)
        elif format in ['pickle', '.pickle', 'pkl', '.pkl']:
            df.to_pickle(path, **kwargs)
        elif format in ['json', '.json']:
            df.to_json(path, **kwargs)
        elif format in ['hdf', 'h5', '.h5', 'hdf5', '.hdf5']:
            df.to_hdf(path, key=kwargs.pop('key', 'data'), **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {format}")
        
        logger.info(f"Saved data to {path} with format {format}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to {path}: {str(e)}")
        return False
