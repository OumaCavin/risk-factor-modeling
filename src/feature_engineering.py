"""
Feature Engineering Utilities

This module provides functions for creating derived features from health data,
including BMI calculation, health scores, and risk factor composites.

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_bmi(weight_lbs: pd.Series, height_inches: pd.Series) -> pd.Series:
    """
    Calculate Body Mass Index from weight and height.
    
    Args:
        weight_lbs: Weight in pounds
        height_inches: Height in inches
        
    Returns:
        BMI values
    """
    # BMI = (weight_kg) / (height_m)^2
    # weight_kg = weight_lbs * 0.453592
    # height_m = height_inches * 0.0254
    bmi = (weight_lbs * 0.453592) / ((height_inches * 0.0254) ** 2)
    return bmi


def categorize_bmi(bmi: pd.Series) -> pd.Series:
    """
    Categorize BMI into standard health categories.
    
    Args:
        bmi: BMI values
        
    Returns:
        Category codes:
        1 = Underweight (BMI < 18.5)
        2 = Normal (18.5 <= BMI < 25)
        3 = Overweight (25 <= BMI < 30)
        4 = Obese (BMI >= 30)
    """
    def categorize(value):
        if value < 18.5:
            return 1  # Underweight
        elif value < 25:
            return 2  # Normal
        elif value < 30:
            return 3  # Overweight
        else:
            return 4  # Obese
    
    return bmi.apply(categorize)


def create_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Create composite health score from multiple indicators.
    
    Higher scores indicate better health.
    
    Args:
        df: DataFrame with health variables
        
    Returns:
        Health score (range approximately 0-10)
    """
    health_score = (
        # Invert health rating (1=Excellent → higher score)
        5 - df['GENHLTH'] +
        
        # Physical health days (inverse, normalized)
        (30 - df['PHYSHLTH']) / 30 * 2 +
        
        # Mental health days (inverse, normalized)
        (30 - df['MENTHLTH']) / 30 * 2 +
        
        # Exercise (1 point if exercises)
        (df['EXERANY2'] == 1).astype(int) +
        
        # Health coverage (1 point if has coverage)
        (df['HLTHPLN1'] == 1).astype(int)
    )
    
    return health_score


def create_risk_count(df: pd.DataFrame) -> pd.Series:
    """
    Count number of cardiovascular risk factors.
    
    Args:
        df: DataFrame with risk factor variables
        
    Returns:
        Count of risk factors (0-5)
    """
    risk_count = (
        # Current smoker
        (df['SMOKE100'] == 1).astype(int) +
        
        # No exercise
        (df['EXERANY2'] == 2).astype(int) +
        
        # High blood pressure
        (df['BPHIGH4'] == 1).astype(int) +
        
        # Mobility issues
        (df['DIFFWALK'] == 1).astype(int) +
        
        # Obesity
        ((df['WEIGHT2'] / (df['HEIGHT3'] ** 2) * 703) >= 30).astype(int)
    )
    
    return risk_count


def create_healthcare_access_score(df: pd.DataFrame) -> pd.Series:
    """
    Create composite healthcare access score.
    
    Args:
        df: DataFrame with healthcare access variables
        
    Returns:
        Access score (0-4)
    """
    access_score = (
        # Has health insurance
        (df['HLTHPLN1'] == 1).astype(int) +
        
        # Has personal doctor
        (df['PERSDOC2'] == 1).astype(int) +
        
        # No cost barriers
        (df['MEDCOST'] == 2).astype(int) +
        
        # Recent checkup
        (df['CHECKUP1'] == 1).astype(int)
    )
    
    return access_score


def create_mental_physical_gap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate absolute difference between mental and physical health days.
    
    Args:
        df: DataFrame with health variables
        
    Returns:
        Absolute difference in health days
    """
    return abs(df['PHYSHLTH'] - df['MENTHLTH'])


def create_demographic_flags(df: pd.DataFrame) -> dict:
    """
    Create binary demographic flags.
    
    Args:
        df: DataFrame with demographic variables
        
    Returns:
        Dictionary of demographic flags
    """
    return {
        'HIGH_EDUCATION': (df['EDUCA'] >= 5).astype(int),
        'EMPLOYED': df['EMPLOY1'].isin([1, 2]).astype(int),
        'MARRIED': (df['MARITAL'] == 1).astype(int)
    }


def create_target_variables(df: pd.DataFrame) -> dict:
    """
    Create binary target variables for modeling.
    
    Args:
        df: DataFrame with health variables
        
    Returns:
        Dictionary of target variables
    """
    return {
        'POOR_HEALTH': (df['GENHLTH'] >= 4).astype(int),
        'HIGH_RISK': (create_risk_count(df) >= 3).astype(int)
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering operations.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with new features
    """
    df_features = df.copy()
    
    logger.info("Starting feature engineering...")
    
    # Calculate BMI
    if 'WEIGHT2' in df_features.columns and 'HEIGHT3' in df_features.columns:
        df_features['BMI'] = calculate_bmi(df_features['WEIGHT2'], df_features['HEIGHT3'])
        logger.info("  Created BMI feature")
    
    # Categorize BMI
    if 'BMI' in df_features.columns:
        df_features['BMI_CAT'] = categorize_bmi(df_features['BMI'])
        logger.info("  Created BMI_CAT feature")
    
    # Create health score
    df_features['health_score'] = create_health_score(df_features)
    logger.info("  Created health_score feature")
    
    # Create risk count
    df_features['RISK_COUNT'] = create_risk_count(df_features)
    logger.info("  Created RISK_COUNT feature")
    
    # Create healthcare access score
    df_features['ACCESS_SCORE'] = create_healthcare_access_score(df_features)
    logger.info("  Created ACCESS_SCORE feature")
    
    # Create mental-physical gap
    df_features['HLTH_GAP'] = create_mental_physical_gap(df_features)
    logger.info("  Created HLTH_GAP feature")
    
    # Create demographic flags
    demo_flags = create_demographic_flags(df_features)
    for name, flag in demo_flags.items():
        df_features[name] = flag
        logger.info(f"  Created {name} feature")
    
    # Create target variables
    targets = create_target_variables(df_features)
    for name, target in targets.items():
        df_features[name] = target
        logger.info(f"  Created {name} target variable")
    
    logger.info(f"Feature engineering complete! Total features: {len(df_features.columns)}")
    
    return df_features


def select_features_for_modeling(df: pd.DataFrame, target_col: str = 'POOR_HEALTH',
                                  correlation_threshold: float = 0.05) -> list:
    """
    Select features based on correlation with target.
    
    Args:
        df: DataFrame with all features
        target_col: Name of target variable
        correlation_threshold: Minimum absolute correlation to include
        
    Returns:
        List of selected feature names
    """
    # Define feature groups
    demographic_features = ['SEX', 'EDUCA', 'MARITAL', 'EMPLOY1', 'RENTHOM1', 'VETERAN3']
    health_behaviors = ['EXERANY2', 'SMOKE100', 'SMOKDAY2']
    health_conditions = ['BPHIGH4', 'BPMEDS', 'DIFFWALK', 'PHYSHLTH', 'MENTHLTH']
    healthcare_access = ['HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1']
    derived_features = ['BMI', 'BMI_CAT', 'RISK_COUNT', 'ACCESS_SCORE', 
                       'HIGH_EDUCATION', 'EMPLOYED', 'MARRIED']
    
    # Combine all features
    all_features = (demographic_features + health_behaviors + health_conditions + 
                   healthcare_access + derived_features)
    
    # Remove duplicates and ensure features exist
    model_features = list(dict.fromkeys([f for f in all_features if f in df.columns]))
    
    # Calculate correlations with target
    if target_col in df.columns:
        correlations = df[model_features + [target_col]].corr()[target_col].drop(target_col)
        correlations = correlations.abs().sort_values(ascending=False)
        
        # Select features above threshold
        selected = correlations[correlations > correlation_threshold].index.tolist()
        
        logger.info(f"Selected {len(selected)} features with correlation > {correlation_threshold}")
        return selected
    
    return model_features
