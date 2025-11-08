"""
Isolation Forest Model
Unsupervised anomaly detection using Isolation Forest algorithm
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseModel


class IsolationForestModel(BaseModel):
    """
    Isolation Forest for anomaly detection
    
    Features:
    - Unsupervised learning (no labels needed for training)
    - Fast training and prediction
    - Anomaly score calculation
    - Threshold optimization
    - Feature importance analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Isolation Forest model
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get model parameters from config
        model_config = self.config.get('models.isolation_forest', {})
        
        self.n_estimators = model_config.get('n_estimators', 200)
        self.contamination = model_config.get('contamination', 0.02)
        self.max_samples = model_config.get('max_samples', 256)
        self.random_state = model_config.get('random_state', 42)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Initialize model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        self.logger.info(f"Isolation Forest initialized: "
                        f"n_estimators={self.n_estimators}, "
                        f"contamination={self.contamination}")
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: Optional[pd.Series] = None
    ) -> None:
        """
        Train Isolation Forest model
        
        Args:
            X_train: Training features
            y_train: Optional labels (not used in unsupervised learning)
        """
        self.logger.info(f"Training Isolation Forest on {len(X_train)} samples...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_scaled)
        
        self.is_trained = True
        self.logger.info("✓ Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (1 = normal, -1 = anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Convert to 0/1 (0 = normal, 1 = anomaly)
        predictions = (predictions == -1).astype(int)
        
        return predictions
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)
        
        # Invert scores (higher = more anomalous)
        scores = -scores
        
        return scores
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels (0 = normal, 1 = anomaly)
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Evaluating model...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_scores = self.predict_scores(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_scores),
            'average_precision': average_precision_score(y_test, y_scores)
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        self.metrics.update({
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        self.logger.info("✓ Evaluation complete")
        
        return self.metrics
    
    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_metric: str = 'f1'
    ) -> float:
        """
        Optimize decision threshold for best performance
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            target_metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        self.logger.info(f"Optimizing threshold for {target_metric}...")
        
        # Get anomaly scores
        scores = self.predict_scores(X_val)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val, scores)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        if target_metric == 'f1':
            best_idx = np.argmax(f1_scores)
        elif target_metric == 'precision':
            best_idx = np.argmax(precision)
        elif target_metric == 'recall':
            best_idx = np.argmax(recall)
        else:
            raise ValueError(f"Unknown metric: {target_metric}")
        
        optimal_threshold = thresholds[best_idx]
        
        self.logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        self.logger.info(f"  Precision: {precision[best_idx]:.4f}")
        self.logger.info(f"  Recall: {recall[best_idx]:.4f}")
        self.logger.info(f"  F1-Score: {f1_scores[best_idx]:.4f}")
        
        return optimal_threshold
    
    def get_feature_importance(self, X: pd.DataFrame, n_top: int = 20) -> pd.DataFrame:
        """
        Calculate feature importance based on anomaly scores
        
        Args:
            X: Feature matrix
            n_top: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.logger.info("Calculating feature importance...")
        
        # Get baseline anomaly scores
        baseline_scores = self.predict_scores(X)
        baseline_mean = np.mean(baseline_scores)
        
        importance = []
        
        for i, feature in enumerate(self.feature_names):
            # Shuffle this feature
            X_shuffled = X.copy()
            X_shuffled.iloc[:, i] = np.random.permutation(X_shuffled.iloc[:, i].values)
            
            # Get scores with shuffled feature
            shuffled_scores = self.predict_scores(X_shuffled)
            shuffled_mean = np.mean(shuffled_scores)
            
            # Importance = change in anomaly score
            importance.append({
                'feature': feature,
                'importance': abs(shuffled_mean - baseline_mean)
            })
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.logger.info(f"✓ Feature importance calculated")
        
        return importance_df.head(n_top)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.config import get_config
    from src.features.feature_engineering import FeatureEngineer
    from src.data.data_loader import DataLoader
    
    print("\n" + "="*80)
    print("ISOLATION FOREST MODEL TEST")
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Load features (assuming they exist)
    engineer = FeatureEngineer(config)
    
    try:
        features_df = engineer.load_features('machine_001_features.csv')
        print(f"\nLoaded features: {features_df.shape}")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['window_id', 'window_start', 'window_end',
                                     'machine_id', 'is_anomaly', 'failure_type', 'severity']]
        
        X = features_df[feature_cols]
        y = features_df['is_anomaly'] if 'is_anomaly' in features_df.columns else None
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        
        if y is not None:
            print(f"Anomalies: {y.sum()} ({y.mean()*100:.2f}%)")
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        
        if y is not None:
            y_train = y.iloc[:train_size]
            y_val = y.iloc[train_size:train_size+val_size]
            y_test = y.iloc[train_size+val_size:]
        
        print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Initialize and train model
        print("\n" + "-"*80)
        print("Training Isolation Forest...")
        print("-"*80)
        
        model = IsolationForestModel(config)
        model.train(X_train)
        
        # Evaluate
        if y is not None:
            print("\n" + "-"*80)
            print("Evaluating...")
            print("-"*80)
            
            metrics = model.evaluate(X_test, y_test)
            model.print_metrics()
            
            # Optimize threshold
            print("\n" + "-"*80)
            print("Optimizing threshold...")
            print("-"*80)
            
            optimal_threshold = model.optimize_threshold(X_val, y_val, target_metric='f1')
        
        # Feature importance
        print("\n" + "-"*80)
        print("Feature Importance (Top 10):")
        print("-"*80)
        
        importance = model.get_feature_importance(X_test.iloc[:100], n_top=10)
        print(importance.to_string(index=False))
        
        # Save model
        model_path = Path(config.get('paths.models', 'models/saved_models')) / 'isolation_forest.pkl'
        model.save(str(model_path))
        print(f"\n✓ Model saved to: {model_path}")
        
        print("\n" + "="*80)
        print("✓ Test complete!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run feature extraction first:")
        print("  python scripts/extract_features.py --machine machine_001")