"""
Model Evaluation Utilities

This module provides functions for evaluating machine learning models,
including performance metrics, visualizations, and feature importance analysis.

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 target_names: List[str] = None) -> str:
    """
    Generate and print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for target classes
        
    Returns:
        String representation of report
    """
    if target_names is None:
        target_names = ['Good Health', 'Poor Health']
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    return report


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Good Health', 'Poor Health'],
                yticklabels=['Good Health', 'Poor Health'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   save_path: str = None) -> Tuple[plt.Figure, float]:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure (optional)
        
    Returns:
        Tuple of (figure, roc_auc_score)
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    return fig, roc_auc


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                                 save_path: str = None) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2)
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Precision-recall curve saved to {save_path}")
    
    return fig


def plot_feature_importance(model, feature_names: List[str],
                            save_path: str = None) -> plt.Figure:
    """
    Plot feature importance from Random Forest or similar model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['Feature'], importance_df['Importance'], 
            color=sns.color_palette("viridis", len(importance_df)))
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Predicting Poor Health')
    ax.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def get_top_features(model, feature_names: List[str], n: int = 10) -> pd.DataFrame:
    """
    Get top N most important features.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        n: Number of top features to return
        
    Returns:
        DataFrame with top features
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df.head(n)


def generate_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray, model, 
                               feature_names: List[str],
                               output_dir: str = None) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model: Trained model
        feature_names: List of feature names
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with all metrics and figures
    """
    report = {}
    
    # Calculate metrics
    metrics = evaluate_model(y_true, y_pred, y_prob)
    report['metrics'] = metrics
    
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print_classification_report(y_true, y_pred)
    
    # Plot confusion matrix
    if output_dir:
        cm_path = f"{output_dir}/confusion_matrix.png"
    else:
        cm_path = None
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Plot ROC curve
    if output_dir:
        roc_path = f"{output_dir}/roc_curve.png"
    else:
        roc_path = None
    fig, roc_auc = plot_roc_curve(y_true, y_prob, roc_path)
    report['roc_curve'] = fig
    report['roc_auc'] = roc_auc
    
    # Plot feature importance
    if output_dir:
        fi_path = f"{output_dir}/feature_importance.png"
    else:
        fi_path = None
    plot_feature_importance(model, feature_names, fi_path)
    
    # Get top features
    top_features = get_top_features(model, feature_names)
    report['top_features'] = top_features
    
    logger.info("\nTop 10 Most Important Features:")
    for i, row in top_features.iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return report
