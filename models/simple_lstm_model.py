"""
Simple LSTM Autoencoder Model
Lightweight sequence-based anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseModel

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        accuracy_score, roc_auc_score
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will be disabled.")


class SimpleLSTMModel(BaseModel):
    """
    Simple LSTM Autoencoder for anomaly detection
    
    Architecture:
    - Encoder: LSTM layers that compress input
    - Decoder: LSTM layers that reconstruct input
    - Anomaly = high reconstruction error
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize LSTM model"""
        super().__init__(config)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        # Get parameters
        model_config = self.config.get('models.lstm_autoencoder', {})
        
        self.sequence_length = model_config.get('sequence_length', 10)
        self.encoding_dim = model_config.get('encoding_dim', 16)
        self.epochs = model_config.get('epochs', 20)
        self.batch_size = model_config.get('batch_size', 32)
        
        self.scaler = StandardScaler()
        self.threshold = None
        
        self.logger.info(f"LSTM Autoencoder initialized (TensorFlow {tf.__version__})")
    
    def _build_model(self, n_features: int):
        """Build LSTM autoencoder architecture"""
        # Encoder
        inputs = keras.Input(shape=(self.sequence_length, n_features))
        encoded = layers.LSTM(32, activation='relu', return_sequences=True)(inputs)
        encoded = layers.LSTM(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(self.encoding_dim, activation='relu', return_sequences=True)(decoded)
        decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(n_features))(decoded)
        
        # Create model
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _prepare_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """Convert data to sequences for LSTM"""
        X_scaled = self.scaler.transform(X)
        
        sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            sequences.append(X_scaled[i:i+self.sequence_length])
        
        return np.array(sequences)
    
    def train(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> None:
        """Train LSTM autoencoder"""
        self.logger.info(f"Training LSTM Autoencoder...")
        
        # Scale features
        self.scaler.fit(X_train)
        
        # Prepare sequences
        X_seq = self._prepare_sequences(X_train)
        
        self.logger.info(f"  Sequences: {X_seq.shape}")
        
        # Build model
        self.model = self._build_model(X_train.shape[1])
        
        # Train (only on normal data if labels available)
        if y_train is not None:
            # Get indices of normal data
            normal_indices = y_train[y_train == 0].index
            normal_mask = X_train.index.isin(normal_indices)
            
            # Prepare normal sequences
            X_normal = X_train[normal_mask]
            X_seq_normal = self._prepare_sequences(X_normal)
            
            self.logger.info(f"  Training on {len(X_seq_normal)} normal sequences")
            
            self.model.fit(
                X_seq_normal, X_seq_normal,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0
            )
        else:
            self.model.fit(
                X_seq, X_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0
            )
        
        # Calculate reconstruction errors on training data to set threshold
        reconstructions = self.model.predict(X_seq, verbose=0)
        mse = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))
        self.threshold = np.percentile(mse, 95)
        
        self.is_trained = True
        self.logger.info(f"✓ Training complete (threshold: {self.threshold:.6f})")
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get reconstruction error scores"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_seq = self._prepare_sequences(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_seq, verbose=0)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))
        
        # Pad to match original length
        scores = np.zeros(len(X))
        scores[:len(mse)] = mse
        
        return scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies"""
        scores = self.predict_scores(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model"""
        self.logger.info("Evaluating LSTM model...")
        
        y_pred = self.predict(X_test)
        y_scores = self.predict_scores(X_test)
        
        # Align lengths (due to sequence padding)
        min_len = min(len(y_test), len(y_pred))
        y_test = y_test.iloc[:min_len]
        y_pred = y_pred[:min_len]
        y_scores = y_scores[:min_len]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_scores)
        }
        
        return self.metrics


# Fallback dummy model if TensorFlow not available
class DummyLSTMModel(BaseModel):
    """Dummy LSTM model when TensorFlow is not available"""
    
    def train(self, X_train, y_train=None):
        self.is_trained = True
        self.mean_score = 0.5
        self.logger.warning("Using dummy LSTM model (TensorFlow not available)")
    
    def predict(self, X):
        return np.zeros(len(X))
    
    def predict_scores(self, X):
        return np.ones(len(X)) * self.mean_score
    
    def evaluate(self, X_test, y_test):
        return {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 
                'f1_score': 0.5, 'roc_auc': 0.5}


# Use the appropriate model based on availability
if not TENSORFLOW_AVAILABLE:
    SimpleLSTMModel = DummyLSTMModel