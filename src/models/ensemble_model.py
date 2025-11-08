import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

class EnsembleModel:
    """
    Lightweight Ensemble Model Wrapper
    - Uses Isolation Forest only when --no-lstm flag is used
    - Provides train, save, load, and predict functionality
    """

    def __init__(self, n_estimators=200, contamination="auto", random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

    def train(self, X):
        """
        Train model on features
        """
        print("\n======================================")
        print(" TRAINING ISOLATION FOREST MODEL")
        print("======================================\n")
        self.model.fit(X)
        print("✓ Training Complete!")

    def predict(self, X):
        """
        Returns:
        - predictions (0 = normal, 1 = anomaly)
        - anomaly scores (higher = more abnormal)
        """
        raw_pred = self.model.predict(X)
        # Convert (-1, 1) → (1, 0)
        preds = np.where(raw_pred == -1, 1, 0)
        scores = -self.model.decision_function(X)
        return preds, scores

    def save(self, path="models/saved_models/ensemble_model.pkl"):
        """
        Saves model to disk
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"\n✓ Model saved to: {path}")

    @staticmethod
    @staticmethod
    def load(path="models/saved_models/ensemble_model.pkl"):
        """
        Loads model from disk and returns the wrapper object
        """
        wrapper = EnsembleModel()
        wrapper.model = joblib.load(path)
        print(f"\n✓ Model loaded from: {path}")
        return wrapper

