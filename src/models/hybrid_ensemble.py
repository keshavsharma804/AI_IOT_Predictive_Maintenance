import os, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

class HybridEnsemble:
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
        if len(arr) < seq_len:
            return np.empty((0, seq_len, 1))
        stride = seq_len // 2
        return np.stack([arr[i:i+seq_len] for i in range(0, len(arr)-seq_len+1, stride)], axis=0)[..., None]

    def fit(self, features_df, raw_df, signal_col="vibration_rms"):
        features_df = features_df.select_dtypes(include=['float32','float64','int32','int64'])
        Xs = self.scaler.fit_transform(features_df)
        self.if_model.fit(Xs)
        if_scores = -self.if_model.decision_function(Xs)
        self.if_score_threshold = float(np.percentile(if_scores, 95))

        signal = raw_df[signal_col].astype(float).values
        seqs = self._make_sequences(signal, self.seq_len)
        self.lstm = self._build_lstm_autoencoder(self.seq_len)
        self.lstm.fit(seqs, seqs, epochs=self.lstm_epochs, batch_size=self.lstm_batch,
                      validation_split=0.1, verbose=1)
        preds = self.lstm.predict(seqs, verbose=0)
        errors = np.mean(np.abs(preds - seqs), axis=(1,2))
        self.recon_threshold = float(np.percentile(errors, 95))
        print("✅ Hybrid Model Training Finished")

    def score_features(self, features_df):
        features_df = features_df.select_dtypes(include=['float32','float64','int32','int64'])
        Xs = self.scaler.transform(features_df)
        return -self.if_model.decision_function(Xs)

    def score_sequences(self, raw_df, signal_col="vibration_rms"):
        signal = raw_df[signal_col].astype(float).values
        seqs = self._make_sequences(signal, self.seq_len)
        pred = self.lstm.predict(seqs, verbose=0)
        return np.mean(np.abs(pred - seqs), axis=(1,2))

    def combine_scores(self, if_scores, lstm_scores):
        def norm(x):
            p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
            return np.clip((x - p1) / (p99 - p1 + 1e-8), 0, 1)

        a = norm(if_scores)
        b = norm(lstm_scores)

        m = min(len(a), len(b))
        fused = self.fw_if * a[:m] + self.fw_lstm * b[:m]
        return fused

    def decision(self, fused, threshold=0.6):
        return (fused >= threshold).astype(int)

    def save(self, dirpath="models/saved_models/hybrid"):
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.if_model, os.path.join(dirpath, "if_model.pkl"))
        joblib.dump(self.scaler, os.path.join(dirpath, "scaler.pkl"))
        self.lstm.save(os.path.join(dirpath, "lstm_ae.keras"))
        json.dump({
            "seq_len": self.seq_len,
            "lstm_units": self.lstm_units,
            "lstm_epochs": self.lstm_epochs,
            "lstm_batch": self.lstm_batch,
            "fw_if": self.fw_if,
            "fw_lstm": self.fw_lstm,
            "recon_threshold": self.recon_threshold,
            "if_score_threshold": self.if_score_threshold
        }, open(os.path.join(dirpath, "meta.json"), "w"))
        print(f"💾 Model saved to {dirpath}")

    @staticmethod
    def load(dirpath="models/saved_models/hybrid"):
        h = HybridEnsemble()
        h.if_model = joblib.load(os.path.join(dirpath, "if_model.pkl"))
        h.scaler = joblib.load(os.path.join(dirpath, "scaler.pkl"))
        h.lstm = keras.models.load_model(os.path.join(dirpath, "lstm_ae.keras"))
        meta = json.load(open(os.path.join(dirpath, "meta.json"), "r"))
        h.__dict__.update(meta)
        print(f"✅ Loaded model from {dirpath}")
        return h
