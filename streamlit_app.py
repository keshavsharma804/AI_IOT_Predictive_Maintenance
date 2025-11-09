import streamlit as st
import pandas as pd
import numpy as np
import time
from src.models.hybrid_ensemble import HybridEnsemble

# Load saved hybrid model
model = HybridEnsemble.load("./models/saved_models/hybrid")

st.title("🟢 Real-Time Machine Health Dashboard")

# Load latest features and raw data
features = pd.read_csv("data/features/machine_001_features.csv")
raw = pd.read_csv("data/synthetic/machine_001_data.csv")

# Get predictions
if_scores = model.score_features(features)
lstm_scores = model.score_sequences(raw, "vibration_rms")
fused = model.combine_scores(if_scores, lstm_scores)

threshold = np.percentile(fused[:2000], 99)
labels = (fused >= threshold).astype(int)

# Summary Box
normal = (labels == 0).sum()
fault = (labels == 1).sum()

st.metric("Normal Windows", normal)
st.metric("Fault Windows", fault)

# Live Timeline Chart
chart = st.line_chart()

for i in range(len(fused)):
    chart.add_rows({"anomaly_score": fused[i]})
    time.sleep(0.01)

