import os, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class HybridEnsemble:
    """
    Hybrid ensemble = Isolation Forest (features) + LSTM Autoencoder (sequence)
    """
    def __init__(self, if_estimators=300, if_contamination='auto',
                 seq_len=128, lstm_units=64, lstm_epochs=10, lstm_batch=256,
                 fusion_weight_if=0.5, fusion_weight_lstm=0.5, random_state=42):
        self.if_model = IsolationForest(n_estimators=if_estimators,
                                        contamination=if_contamination,
                                        random_state=random_state)
        self.scaler = StandardScaler()
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.lstm_epochs = lstm_epochs
        self.lstm_batch = lstm_batch
        self.fw_if = fusion_weight_if
        self.fw_lstm = fusion_weight_lstm
        self.lstm = None
        self.recon_threshold = None
        self.if_score_threshold = None

    def _build_lstm_autoencoder(self, timesteps):
        inp = keras.Input(shape=(timesteps, 1))
        x = layers.LSTM(self.lstm_units, return_sequences=False)(inp)
        x = layers.RepeatVector(timesteps)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(1))(x)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mae")
        return model

    def _make_sequences(self, arr, seq_len):
        n = len(arr)
        if n < seq_len:
            return np.empty((0, seq_len, 1))
        stride = seq_len // 2
        idxs = range(0, n - seq_len + 1, stride)
        out = np.stack([arr[i:i+seq_len] for i in idxs], axis=0)
        return out[..., None]

    def fit(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, signal_col="vibration_rms"):
        # Drop non-numeric columns
        features_df = features_df.select_dtypes(include=['float32','float64','int32','int64'])
        if features_df.shape[1] == 0:
            raise ValueError("No numeric features available!")

        Xs = self.scaler.fit_transform(features_df)
        self.if_model.fit(Xs)

        if_scores = -self.if_model.decision_function(Xs)
        self.if_score_threshold = float(np.percentile(if_scores, 95))

        if signal_col not in raw_df.columns:
            raise ValueError(f"Signal column '{signal_col}' not found.")

        signal = raw_df[signal_col].astype(float).values
        seqs = self._make_sequences(signal, self.seq_len)
        if len(seqs) == 0:
            raise ValueError("Not enough raw samples for LSTM. Reduce seq_len.")

        self.lstm = self._build_lstm_autoencoder(self.seq_len)
        self.lstm.fit(seqs, seqs,
                      epochs=self.lstm_epochs,
                      batch_size=self.lstm_batch,
                      validation_split=0.1, verbose=1)

        preds = self.lstm.predict(seqs, verbose=0)
        errors = np.mean(np.abs(preds - seqs), axis=(1,2))
        self.recon_threshold = float(np.percentile(errors, 95))

        print("✅ Hybrid Model Training Complete")

    # -------- Scoring --------
    def score_features(self, features_df: pd.DataFrame):
        features_df = features_df.select_dtypes(include=['float32','float64','int32','int64'])
        Xs = self.scaler.transform(features_df)
        return -self.if_model.decision_function(Xs)

    def score_sequences(self, raw_df: pd.DataFrame, signal_col="vibration_rms"):
        signal = raw_df[signal_col].astype(float).values
        seqs = self._make_sequences(signal, self.seq_len)
        if len(seqs) == 0:
            return np.array([])
        pred = self.lstm.predict(seqs, verbose=0)
        recon_err = np.mean(np.abs(pred - seqs), axis=(1,2))
        return recon_err

    def combine_scores(self, if_scores, lstm_scores):
        def norm(x):
            if len(x) == 0:
                return x
            p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
            return np.clip((x - p1) / (p99 - p1 + 1e-8), 0, 1)

        a = norm(if_scores)
        b = norm(lstm_scores) if len(lstm_scores) else np.zeros_like(a)

        m = min(len(a), len(b)) if len(b) else len(a)
        a, b = a[:m], (b[:m] if len(b) else np.zeros(m))

        fused = self.fw_if * a + self.fw_lstm * b
        return fused

    def decision(self, fused_scores, fusion_threshold=0.6):
        return (fused_scores >= fusion_threshold).astype(int)
