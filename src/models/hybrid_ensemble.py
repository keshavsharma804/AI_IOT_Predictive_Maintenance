# src/models/hybrid_ensemble.py
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
    - IF trains on feature vectors (engineered features csv)
    - LSTM-AE trains on sequences of a chosen raw signal (e.g., vibration_rms)
    - Final score = w_if * if_score + w_lstm * norm_recon_error
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

    # -------- LSTM Autoencoder definition --------
    def _build_lstm_autoencoder(self, timesteps):
        inp = keras.Input(shape=(timesteps, 1))
        x = layers.LSTM(self.lstm_units, return_sequences=False)(inp)
        x = layers.RepeatVector(timesteps)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(1))(x)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mae")
        return model

    # -------- Sequence windowing on a 1-D signal column --------
    def _make_sequences(self, arr, seq_len):
        n = len(arr)
        if n < seq_len:
            return np.empty((0, seq_len, 1))
        # 50% overlap by default
        stride = seq_len // 2
        idxs = range(0, n - seq_len + 1, stride)
        out = np.stack([arr[i:i+seq_len] for i in idxs], axis=0)
        return out[..., None]  # (N, seq_len, 1)

    # -------- Fit models --------
     def fit(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, signal_col: str = "vibration_rms"):
        print("\n---- HYBRID TRAINING START ----")
    
        # ============================================================
        # 1) CLEAN AND SELECT NUMERIC FEATURES ONLY
        # ============================================================
        bad_cols = [
            c for c in features_df.columns
            if any(key in c.lower() for key in ["timestamp", "window", "machine", "failure", "severity", "label", "id"])
        ]
        features_df = features_df.drop(columns=bad_cols, errors="ignore")
    
        # Keep **numeric only**
        features_df = features_df.select_dtypes(include=['float32','float64','int32','int64'])
        
        if features_df.shape[1] == 0:
            raise ValueError("❌ No usable numeric feature columns remain!")
    
        print(f"✅ Using {features_df.shape[1]} numeric features for Isolation Forest")
    
        # Scale + train IF
        Xs = self.scaler.fit_transform(features_df)
        self.if_model.fit(Xs)
    
        if_scores = -self.if_model.decision_function(Xs)
        self.if_score_threshold = float(np.percentile(if_scores, 95))
    
        # ============================================================
        # 2) PREP RAW SIGNAL FOR LSTM AUTOENCODER
        # ============================================================
        if signal_col not in raw_df.columns:
            raise ValueError(f"❌ Raw data does not contain signal column: {signal_col}")
    
        signal = raw_df[signal_col].astype(float).values
        seqs = self._make_sequences(signal, self.seq_len)
    
        if len(seqs) == 0:
            raise ValueError("❌ Not enough raw data to form LSTM sequences. Reduce seq_len.")
    
        print(f"✅ Created {seqs.shape[0]} sequences for LSTM-AE training")
    
        # Build + Train LSTM Autoencoder
        self.lstm = self._build_lstm_autoencoder(self.seq_len)
        self.lstm.fit(
            seqs, seqs,
            epochs=self.lstm_epochs,
            batch_size=self.lstm_batch,
            verbose=1
        )
    
        print("✅ HYBRID MODEL TRAINING COMPLETE\n")




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
        # expand to per-window mapping; align lengths later in predict()
        return recon_err

    def combine_scores(self, if_scores, lstm_scores):
        # Normalize each to [0,1] by robust scaling
        def norm(x):
            if len(x) == 0:
                return x
            p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
            return np.clip((x - p1) / (p99 - p1 + 1e-8), 0, 1)
        a = norm(if_scores)
        b = norm(lstm_scores) if len(lstm_scores) else np.zeros_like(a)
        # Align lengths (use min length)
        m = min(len(a), len(b)) if len(b) else len(a)
        a, b = a[:m], (b[:m] if len(b) else np.zeros(m))
        fused = self.fw_if * a + self.fw_lstm * b
        return fused

    def decision(self, fused_scores, fusion_threshold=0.6):
        return (fused_scores >= fusion_threshold).astype(int)

    # -------- Save / Load --------
    def save(self, dirpath="models/saved_models/hybrid"):
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.if_model, os.path.join(dirpath, "if_model.pkl"))
        joblib.dump(self.scaler, os.path.join(dirpath, "scaler.pkl"))
        self.lstm.save(os.path.join(dirpath, "lstm_ae.keras"))
        meta = {
            "seq_len": self.seq_len,
            "lstm_units": self.lstm_units,
            "lstm_epochs": self.lstm_epochs,
            "lstm_batch": self.lstm_batch,
            "fw_if": self.fw_if,
            "fw_lstm": self.fw_lstm,
            "recon_threshold": self.recon_threshold,
            "if_score_threshold": self.if_score_threshold,
        }
        with open(os.path.join(dirpath, "meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"✓ Hybrid ensemble saved to: {dirpath}")

    @staticmethod
    def load(dirpath="models/saved_models/hybrid"):
        h = HybridEnsemble()
        h.if_model = joblib.load(os.path.join(dirpath, "if_model.pkl"))
        h.scaler = joblib.load(os.path.join(dirpath, "scaler.pkl"))
        h.lstm = keras.models.load_model(os.path.join(dirpath, "lstm_ae.keras"))
        with open(os.path.join(dirpath, "meta.json"), "r") as f:
            meta = json.load(f)
        h.seq_len = meta["seq_len"]
        h.lstm_units = meta["lstm_units"]
        h.lstm_epochs = meta["lstm_epochs"]
        h.lstm_batch = meta["lstm_batch"]
        h.fw_if = meta["fw_if"]
        h.fw_lstm = meta["fw_lstm"]
        h.recon_threshold = meta["recon_threshold"]
        h.if_score_threshold = meta["if_score_threshold"]
        print(f"✓ Hybrid ensemble loaded from: {dirpath}")
        return h
