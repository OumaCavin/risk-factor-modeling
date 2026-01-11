"""
Model Training Utilities

This module provides functions for training and tuning machine learning models
for health outcome prediction.

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str,
                 test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for modeling by splitting and scaling.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Testing set: {len(X_test):,} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train a baseline logistic regression model.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        
    Returns:
        Trained logistic regression model
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    logger.info("Baseline logistic regression model trained")
    
    return model


def train_multiple_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train and compare multiple machine learning models.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        
    Returns:
        Dictionary of trained models
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight='balanced', max_depth=10
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced',
            n_jobs=-1, max_depth=10
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=5,
            learning_rate=0.1
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"  {name} trained successfully")
    
    return trained_models


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, Dict]:
    """
    Tune Random Forest hyperparameters using grid search.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        
    Returns:
        Tuple of (best_model, best_params)
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 10, 12],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    
    logger.info("Starting hyperparameter tuning with GridSearchCV...")
    
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate multiple models and return performance metrics.
    
    Args:
        models: Dictionary of trained models
        X_test: Scaled test features
        y_test: Test labels
        
    Returns:
        DataFrame with performance metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    results = []
    
    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        logger.info(f"{name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    
    return pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
