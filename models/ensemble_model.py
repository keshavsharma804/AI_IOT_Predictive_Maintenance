"""
Ensemble Model
Combines multiple models with intelligent voting for robust predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

from .base_model import BaseModel
from .isolation_forest_model import IsolationForestModel
from .simple_lstm_model import SimpleLSTMModel
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix
)


class EnsembleModel:
    """
    Ensemble model combining multiple anomaly detectors
    
    Voting strategies:
    - Hard voting: Majority vote (0 or 1)
    - Soft voting: Weighted average of anomaly scores
    - Adaptive: Learns optimal weights based on validation performance
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ensemble model"""
        super().__init__(config)
        
        # Get ensemble config
        ensemble_config = self.config.get('models.ensemble', {})
        
        self.voting_method = ensemble_config.get('voting_method', 'weighted')
        self.weights = ensemble_config.get('weights', {
            'isolation_forest': 0.6,
            'lstm': 0.4
        })
        
        # Initialize base models
        self.models = {
            'isolation_forest': IsolationForestModel(config),
            'lstm': SimpleLSTMModel(config)
        }
        
        self.logger.info(f"Ensemble initialized: {self.voting_method} voting")
        self.logger.info(f"  Models: {list(self.models.keys())}")
        self.logger.info(f"  Weights: {self.weights}")
    
    def train(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> None:
        """Train all ensemble models"""
        self.logger.info("="*80)
        self.logger.info("TRAINING ENSEMBLE MODELS")
        self.logger.info("="*80)
        
        for name, model in self.models.items():
            self.logger.info(f"\nTraining {name}...")
            try:
                model.train(X_train, y_train)
                self.logger.info(f"✓ {name} trained successfully")
            except Exception as e:
                self.logger.error(f"✗ Error training {name}: {e}")
                # Remove failed model
                self.models.pop(name)
                if name in self.weights:
                    self.weights.pop(name)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        self.logger.info(f"\n✓ Ensemble training complete")
        self.logger.info(f"  Active models: {len(self.models)}")
    
    def predict_scores(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get ensemble anomaly scores
        
        Returns:
            Tuple of (ensemble_scores, individual_scores_dict)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        individual_scores = {}
        
        # Get scores from each model
        for name, model in self.models.items():
            try:
                scores = model.predict_scores(X)
                individual_scores[name] = scores
            except Exception as e:
                self.logger.warning(f"Error getting scores from {name}: {e}")
                individual_scores[name] = np.zeros(len(X))
        
        # Ensemble scoring
        if self.voting_method == 'weighted':
            # Weighted average
            ensemble_scores = np.zeros(len(X))
            for name, scores in individual_scores.items():
                weight = self.weights.get(name, 0)
                ensemble_scores += weight * scores
        
        elif self.voting_method == 'max':
            # Maximum score
            scores_array = np.array(list(individual_scores.values()))
            ensemble_scores = np.max(scores_array, axis=0)
        
        elif self.voting_method == 'mean':
            # Simple average
            scores_array = np.array(list(individual_scores.values()))
            ensemble_scores = np.mean(scores_array, axis=0)
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        return ensemble_scores, individual_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies using ensemble"""
        ensemble_scores, _ = self.predict_scores(X)
        
        # Use threshold (e.g., median of scores)
        threshold = np.median(ensemble_scores)
        predictions = (ensemble_scores > threshold).astype(int)
        
        return predictions
    
    def predict_with_confidence(
        self, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict with confidence scores
        
        Returns:
            Tuple of (predictions, confidence, individual_predictions)
        """
        ensemble_scores, individual_scores = self.predict_scores(X)
        
        # Get individual predictions
        individual_predictions = {}
        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                individual_predictions[name] = preds
            except:
                individual_predictions[name] = np.zeros(len(X))
        
        # Calculate confidence (agreement between models)
        pred_array = np.array(list(individual_predictions.values()))
        agreement = np.mean(pred_array, axis=0)
        confidence = np.abs(agreement - 0.5) * 2  # 0 = no agreement, 1 = full agreement
        
        # Final prediction
        threshold = np.median(ensemble_scores)
        predictions = (ensemble_scores > threshold).astype(int)
        
        return predictions, confidence, individual_predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble model"""
        self.logger.info("="*80)
        self.logger.info("EVALUATING ENSEMBLE")
        self.logger.info("="*80)
        
        # Get predictions
        predictions, confidence, individual_preds = self.predict_with_confidence(X_test)
        ensemble_scores, individual_scores = self.predict_scores(X_test)
        
        # Evaluate ensemble
        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_test, ensemble_scores),
            'avg_confidence': float(np.mean(confidence))
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        self.metrics.update({
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        })
        
        # Evaluate individual models
        self.logger.info("\nIndividual Model Performance:")
        for name, model in self.models.items():
            try:
                model_metrics = model.evaluate(X_test, y_test)
                self.logger.info(f"\n{name}:")
                self.logger.info(f"  Precision: {model_metrics['precision']:.4f}")
                self.logger.info(f"  Recall:    {model_metrics['recall']:.4f}")
                self.logger.info(f"  F1-Score:  {model_metrics['f1_score']:.4f}")
            except Exception as e:
                self.logger.warning(f"  Error evaluating {name}: {e}")
        
        self.logger.info(f"\nEnsemble Performance:")
        self.logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {self.metrics['recall']:.4f}")
        self.logger.info(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        self.logger.info(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights on validation set
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize
            
        Returns:
            Optimized weights
        """
        self.logger.info(f"Optimizing ensemble weights for {metric}...")
        
        # Get individual scores
        _, individual_scores = self.predict_scores(X_val)
        
        # Try different weight combinations
        from scipy.optimize import minimize
        
        def objective(weights):
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / (np.sum(weights) + 1e-10)
            
            # Calculate ensemble score
            ensemble_score = np.zeros(len(X_val))
            for i, (name, scores) in enumerate(individual_scores.items()):
                ensemble_score += weights[i] * scores
            
            # Predict
            threshold = np.median(ensemble_score)
            predictions = (ensemble_score > threshold).astype(int)
            
            # Calculate metric
            if metric == 'f1':
                score = f1_score(y_val, predictions, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, predictions, zero_division=0)
            else:
                score = accuracy_score(y_val, predictions)
            
            return -score  # Minimize negative score
        
        # Initial weights
        initial_weights = np.array([self.weights.get(name, 0.5) 
                                   for name in individual_scores.keys()])
        
        # Optimize
        result = minimize(objective, initial_weights, method='Nelder-Mead')
        
        # Update weights
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        self.weights = {
            name: float(weight) 
            for name, weight in zip(individual_scores.keys(), optimal_weights)
        }
        
        self.logger.info(f"✓ Optimized weights: {self.weights}")
        
        return self.weights
    
    def save(self, filepath: str) -> None:
        """Save ensemble model"""
        # Create directory
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble configuration
        ensemble_data = {
            'voting_method': self.voting_method,
            'weights': self.weights,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        
        # Save individual models
        model_dir = Path(filepath).parent / 'ensemble_models'
        model_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = model_dir / f'{name}.pkl'
            model.save(str(model_path))
        
        # Save ensemble data
        joblib.dump(ensemble_data, filepath)
        
        self.logger.info(f"✓ Ensemble saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load ensemble model"""
        # Load ensemble data
        ensemble_data = joblib.load(filepath)
        
        self.voting_method = ensemble_data['voting_method']
        self.weights = ensemble_data['weights']
        self.metrics = ensemble_data.get('metrics', {})
        self.is_trained = ensemble_data.get('is_trained', False)
        
        # Load individual models
        model_dir = Path(filepath).parent / 'ensemble_models'
        
        for name in self.models.keys():
            model_path = model_dir / f'{name}.pkl'
            if model_path.exists():
                self.models[name].load(str(model_path))
        
        self.logger.info(f"✓ Ensemble loaded from {filepath}")
        
        
    __all__ = ["EnsembleModel"]
