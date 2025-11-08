"""
Base Model Class
Abstract base class for all ML models
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path

from ..utils.logger import LoggerMixin
from ..config import get_config


class BaseModel(ABC, LoggerMixin):
    """
    Abstract base class for all ML models
    
    All models must implement:
    - train()
    - predict()
    - evaluate()
    - save()
    - load()
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.model = None
        self.is_trained = False
        self.model_name = self.__class__.__name__
        self.metrics = {}
        
        self.logger.info(f"{self.model_name} initialized")
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> None:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (optional for unsupervised)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            self.logger.warning("Saving untrained model")
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump({
            'model': self.model,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'config': self.config
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to model file
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model
        data = joblib.load(filepath)
        
        self.model = data['model']
        self.metrics = data.get('metrics', {})
        self.is_trained = data.get('is_trained', False)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model metrics"""
        return self.metrics
    
    def print_metrics(self) -> None:
        """Print model metrics"""
        if not self.metrics:
            self.logger.warning("No metrics available")
            return
        
        print("\n" + "="*60)
        print(f"{self.model_name} - Performance Metrics".center(60))
        print("="*60)
        
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("="*60 + "\n")