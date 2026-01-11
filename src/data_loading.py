"""
Data Loading Utilities

This module provides functions for loading and validating health data
from various sources including CSV and Excel files.

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed
    """
    logger.info(f"Loading CSV data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise


def load_excel_data(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Args:
        filepath: Path to Excel file
        sheet_name: Name of sheet to load (optional)
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading Excel data from {filepath}")
    try:
        if sheet_name:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath)
        logger.info(f"Successfully loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise


def load_health_data(csv_path: str, xlsx_path: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Load health data from available source.
    
    Attempts to load from CSV first, then Excel if CSV fails.
    
    Args:
        csv_path: Path to CSV file
        xlsx_path: Path to Excel file (optional)
        
    Returns:
        Tuple of (DataFrame, source_path)
    """
    # Try CSV first
    try:
        df = load_csv_data(csv_path)
        return df, csv_path
    except Exception as csv_error:
        logger.warning(f"CSV loading failed: {csv_error}")
        
        # Try Excel if available
        if xlsx_path:
            try:
                df = load_excel_data(xlsx_path)
                return df, xlsx_path
            except Exception as xlsx_error:
                logger.error(f"Excel loading failed: {xlsx_error}")
                raise ValueError("Could not load data from any source")
        else:
            raise


def validate_data(df: pd.DataFrame, expected_columns: list = None) -> dict:
    """
    Validate loaded data for basic quality checks.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names (optional)
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_match': True
    }
    
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        validation_results['missing_columns'] = list(missing_cols)
        validation_results['extra_columns'] = list(extra_cols)
        validation_results['column_match'] = len(missing_cols) == 0 and len(extra_cols) == 0
    
    logger.info(f"Validation complete: {validation_results}")
    return validation_results


def get_data_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with column-level information
    """
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str).values,
        'Non-Null Count': df.count().values,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Unique Values': df.nunique().values,
        'Memory (MB)': [df[col].memory_usage(deep=True) / 1024**2 for col in df.columns]
    })
    
    return info_df
