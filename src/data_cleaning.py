"""
Data Cleaning Utilities

This module provides functions for cleaning and preprocessing health data,
including handling missing values, outliers, and special BRFSS codes.

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Special value mappings for BRFSS data
SPECIAL_VALUE_MAP = {
    # 88 means "None" or "Zero days" - convert to 0
    'PHYSHLTH': {88: 0},
    'MENTHLTH': {88: 0},
    'POORHLTH': {88: 0},
    
    # 77, 99 mean "Refused" or "Don't know" - convert to NaN
    'GENHLTH': {77: np.nan, 99: np.nan},
    'HLTHPLN1': {7: np.nan, 9: np.nan},
    'PERSDOC2': {7: np.nan, 9: np.nan},
    'MEDCOST': {7: np.nan, 9: np.nan},
    'CHECKUP1': {7: np.nan, 9: np.nan},
    'BPHIGH4': {7: np.nan, 9: np.nan},
    'BPMEDS': {7: np.nan, 9: np.nan},
    'SEX': {7: np.nan, 9: np.nan},
    'MARITAL': {7: np.nan, 9: np.nan},
    'EDUCA': {7: np.nan, 9: np.nan},
    'RENTHOM1': {7: np.nan, 9: np.nan},
    'VETERAN3': {7: np.nan, 9: np.nan},
    'EMPLOY1': {7: np.nan, 9: np.nan},
    'DIFFWALK': {7: np.nan, 9: np.nan},
    'SMOKE100': {7: np.nan, 9: np.nan},
    'SMOKDAY2': {7: np.nan, 9: np.nan},
    'EXERANY2': {7: np.nan, 9: np.nan},
    
    # Large special values for weight/height - convert to NaN
    'WEIGHT2': {7777: np.nan, 9999: np.nan},
    'HEIGHT3': {7777: np.nan, 9999: np.nan}
}


# Reasonable value ranges for outlier detection
REASONABLE_RANGES = {
    'PHYSHLTH': (0, 30),
    'MENTHLTH': (0, 30),
    'POORHLTH': (0, 30),
    'WEIGHT2': (50, 700),
    'HEIGHT3': (36, 96),
    'NUMADULT': (1, 20),
    'GENHLTH': (1, 5),
    'EDUCA': (1, 6)
}


def handle_special_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert BRFSS special values to appropriate representations.
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        DataFrame with special values converted
    """
    df_clean = df.copy()
    total_converted = 0
    
    for col, value_map in SPECIAL_VALUE_MAP.items():
        if col in df_clean.columns:
            before_missing = df_clean[col].isnull().sum()
            for old_value, new_value in value_map.items():
                mask = df_clean[col] == old_value
                df_clean.loc[mask, col] = new_value
            after_missing = df_clean[col].isnull().sum()
            converted = after_missing - before_missing
            if converted > 0:
                total_converted += converted
                logger.info(f"  {col}: {converted:,} special values converted")
    
    logger.info(f"Total special values converted: {total_converted:,}")
    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate imputation strategies.
    
    Args:
        df: DataFrame with data
        
    Returns:
        DataFrame with missing values treated
    """
    df_clean = df.copy()
    
    # Define treatment strategies
    treatment_strategies = {
        # Categorical/Binary: Use mode
        'SEX': 'mode',
        'HLTHPLN1': 'mode',
        'PERSDOC2': 'mode',
        'MEDCOST': 'mode',
        'CHECKUP1': 'mode',
        'BPHIGH4': 'mode',
        'BPMEDS': 'mode',
        'MARITAL': 'mode',
        'EDUCA': 'mode',
        'RENTHOM1': 'mode',
        'VETERAN3': 'mode',
        'EMPLOY1': 'mode',
        'DIFFWALK': 'mode',
        'SMOKE100': 'mode',
        'SMOKDAY2': 'mode',
        'EXERANY2': 'mode',
        
        # Ordinal: Use median
        'GENHLTH': 'median',
        
        # Numeric: Use median
        'PHYSHLTH': 'median',
        'MENTHLTH': 'median',
        'POORHLTH': 'median',
        'WEIGHT2': 'median',
        'HEIGHT3': 'median',
        
        # Special: Drop rows
        'NUMADULT': 'drop'
    }
    
    rows_before = len(df_clean)
    
    for col, treatment in treatment_strategies.items():
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            if treatment == 'mode':
                mode_value = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_value, inplace=True)
                logger.info(f"  {col}: Imputed with mode = {mode_value}")
                
            elif treatment == 'median':
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
                logger.info(f"  {col}: Imputed with median = {median_value}")
                
            elif treatment == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                logger.info(f"  {col}: Dropped {len(df_clean) - rows_before} rows")
    
    rows_after = len(df_clean)
    logger.info(f"Rows before: {rows_before:,} → Rows after: {rows_after:,}")
    
    return df_clean


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and cap outliers to reasonable ranges.
    
    Args:
        df: DataFrame with data
        
    Returns:
        DataFrame with outliers treated
    """
    df_clean = df.copy()
    
    for col, (min_val, max_val) in REASONABLE_RANGES.items():
        if col in df_clean.columns:
            below_min = (df_clean[col] < min_val).sum()
            above_max = (df_clean[col] > max_val).sum()
            total_outliers = below_min + above_max
            
            if total_outliers > 0:
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
                logger.info(f"  {col}: Capped {total_outliers} outliers to [{min_val}, {max_val}]")
    
    return df_clean


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types for memory efficiency.
    
    Args:
        df: DataFrame with data
        
    Returns:
        DataFrame with optimized data types
    """
    df_clean = df.copy()
    
    type_conversions = {
        '_STATE': 'int32',
        'SEX': 'int8',
        'GENHLTH': 'int8',
        'HLTHPLN1': 'int8',
        'PERSDOC2': 'int8',
        'MEDCOST': 'int8',
        'CHECKUP1': 'int8',
        'BPHIGH4': 'int8',
        'BPMEDS': 'int8',
        'MARITAL': 'int8',
        'EDUCA': 'int8',
        'RENTHOM1': 'int8',
        'VETERAN3': 'int8',
        'EMPLOY1': 'int8',
        'DIFFWALK': 'int8',
        'SMOKE100': 'int8',
        'SMOKDAY2': 'int8',
        'EXERANY2': 'int8',
        'NUMADULT': 'int8',
        'PHYSHLTH': 'int8',
        'MENTHLTH': 'int8',
        'POORHLTH': 'int8',
        'WEIGHT2': 'int16',
        'HEIGHT3': 'int16'
    }
    
    memory_before = df_clean.memory_usage(deep=True).sum()
    
    for col, dtype in type_conversions.items():
        if col in df_clean.columns:
            try:
                df_clean[col] = df_clean[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert {col} to {dtype}: {e}")
    
    memory_after = df_clean.memory_usage(deep=True).sum()
    logger.info(f"Memory: {memory_before/1024**2:.2f} MB → {memory_after/1024**2:.2f} MB")
    
    return df_clean


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning pipeline...")
    
    # Step 1: Handle special values
    df = handle_special_values(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Handle outliers
    df = handle_outliers(df)
    
    # Step 4: Convert data types
    df = convert_data_types(df)
    
    # Step 5: Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed duplicates. Final dataset: {len(df):,} rows")
    
    logger.info("Data cleaning pipeline complete!")
    return df


def get_cleaning_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Generate summary of cleaning operations.
    
    Args:
        df_before: DataFrame before cleaning
        df_after: DataFrame after cleaning
        
    Returns:
        Dictionary with cleaning summary
    """
    return {
        'records_before': len(df_before),
        'records_after': len(df_after),
        'records_removed': len(df_before) - len(df_after),
        'columns_before': len(df_before.columns),
        'columns_after': len(df_after.columns),
        'memory_before_mb': df_before.memory_usage(deep=True).sum() / 1024**2,
        'memory_after_mb': df_after.memory_usage(deep=True).sum() / 1024**2
    }
