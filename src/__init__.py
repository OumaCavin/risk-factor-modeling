"""
Risk Factor Modeling Project

A comprehensive data science project for analyzing behavioral risk factors
and predicting health outcomes using machine learning.

Author: Cavin Otieno
"""

from .data_loading import load_health_data, validate_data, get_data_info
from .data_cleaning import clean_data, handle_special_values
from .feature_engineering import engineer_features, select_features_for_modeling
from .model_training import prepare_data, train_multiple_models, tune_random_forest
from .model_evaluation import evaluate_model, generate_evaluation_report

__version__ = '1.0.0'
__author__ = 'Cavin Otieno'
