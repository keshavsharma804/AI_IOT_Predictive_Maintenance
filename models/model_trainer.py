"""
Model Trainer
Handles training, evaluation, and saving of models
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from ..utils.logger import LoggerMixin
from ..config import get_config


class ModelTrainer(LoggerMixin):
    """
    Orchestrates model training and evaluation pipeline
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.logger.info("ModelTrainer initialized")
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training
        
        Args:
            features_df: DataFrame with features and labels
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info("Preparing data...")
        
        # Separate features and labels
        feature_cols = [col for col in features_df.columns 
                       if col not in ['window_id', 'window_start', 'window_end',
                                     'machine_id', 'is_anomaly', 'failure_type', 'severity']]
        
        X = features_df[feature_cols]
        y = features_df['is_anomaly'] if 'is_anomaly' in features_df.columns else None
        
        if y is None:
            raise ValueError("No labels found in dataset")
        
        # Remove features with zero variance
        variance = X.var()
        zero_var_cols = variance[variance == 0].index.tolist()
        
        if zero_var_cols:
            self.logger.warning(f"Removing {len(zero_var_cols)} zero-variance features")
            X = X.drop(columns=zero_var_cols)
        
        # Handle missing values
        if X.isnull().any().any():
            self.logger.warning("Found missing values, filling with median")
            X = X.fillna(X.median())
        
        # Split data (time-series aware)
        train_val_size = 1 - test_size
        train_size_adjusted = 1 - (val_size / train_val_size)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data split:")
        self.logger.info(f"  Train: {len(X_train)} samples ({y_train.sum()} anomalies, {y_train.mean()*100:.2f}%)")
        self.logger.info(f"  Val:   {len(X_val)} samples ({y_val.sum()} anomalies, {y_val.mean()*100:.2f}%)")
        self.logger.info(f"  Test:  {len(X_test)} samples ({y_test.sum()} anomalies, {y_test.mean()*100:.2f}%)")
        self.logger.info(f"  Features: {len(feature_cols)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved ROC curve to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved PR curve to {save_path}")
        
        return fig


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.config import get_config
    from src.features.feature_engineering import FeatureEngineer
    
    config = get_config()
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    
    # Load features
    features_df = engineer.load_features('machine_001_features.csv')
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features_df)
    
    print("✓ Data preparation test complete")