# streamlit_app.py
import json, time, math, queue
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional imports
try:
    from scipy.signal import butter, filtfilt
    from scipy.stats import kurtosis, skew
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

MQTT_OK = True
try:
    import paho.mqtt.client as mqtt
except Exception:
    MQTT_OK = False

# ===== If you have these, keep them; otherwise comment them out =====
try:
    from src.models.hybrid_ensemble import HybridEnsemble
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False

# ────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

# ────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────
def ensure_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

@st.cache_resource
def load_model():
    if not HAS_MODEL:
        return None
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.warning("Model files missing; running without live anomaly scoring.")
        return None
    return HybridEnsemble.load(MODEL_DIR.as_posix())

@st.cache_data
def load_demo_dataframe() -> pd.DataFrame:
    if ensure_exists(DEMO_CSV):
        return pd.read_csv(DEMO_CSV)
    # tiny synthetic fallback
    n = 5000
    t = np.arange(n) / 200.0
    x = 0.5 + 0.05*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
    y = 0.5 + 0.04*np.sin(2*np.pi*3.2*t + 0.3) + 0.02*np.random.randn(n)
    z = 0.5 + 0.06*np.sin(2*np.pi*2.8*t - 0.2) + 0.02*np.random.randn(n)
    z[2000:2100] += 0.25*np.sin(2*np.pi*15*t[2000:2100])
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df["vibration_rms"] = np.sqrt((df["x"]**2 + df["y"]**2 + df["z"]**2)/3.0)
    return df

def lowpass(x, cutoff=50, fs=1000, order=4):
    if not SCIPY_OK: return x
    b, a = butter(order, cutoff/(0.5*fs), btype="low")
    return filtfilt(b, a, x)

def fuse_scores(model, if_scores: np.ndarray, lstm_scores: np.ndarray) -> np.ndarray:
    try:
        return model.combine_scores(if_scores, lstm_scores)
    except Exception:
        m = min(len(if_scores), len(lstm_scores))
        return 0.5 * if_scores[:m] + 0.5 * lstm_scores[:m]

def score_offline(model, df: pd.DataFrame) -> dict:
    if model is None:
        fused = np.zeros(len(df))
        return {"lstm_scores": fused, "if_scores": fused, "fused": fused, "decisions": np.zeros(len(fused)), "threshold": 0.0}
    lstm_scores = model.score_sequences(df, signal_col="vibration_rms")
    if_scores = np.zeros_like(lstm_scores)
    fused = fuse_scores(model, if_scores, lstm_scores)
    base = min(2000, len(fused))
    thr = np.percentile(fused[:base], 99)
    decisions = (fused >= thr).astype(int)
    return {"lstm_scores": lstm_scores, "if_scores": if_scores, "fused": fused, "decisions": decisions, "threshold": float(thr)}

def compute_features(sig: np.ndarray) -> pd.DataFrame:
    rms = float(np.sqrt(np.mean(sig**2)))
    peak = float(np.max(np.abs(sig)))
    krt = float(kurtosis(sig)) if SCIPY_OK else float("nan")
    skw = float(skew(sig)) if SCIPY_OK else float("nan")
    return pd.DataFrame({"RMS":[rms], "Peak":[peak], "Kurtosis":[krt], "Skewness":[skw]})

def score_live_window(arr: np.ndarray, model):
    if model is None or arr.size < 32:
        return np.array([]), np.array([]), 0.0
    df = pd.DataFrame({"vibration_rms": arr})
    try:
        lstm = model.score_sequences(df, "vibration_rms")
        ifs = np.zeros_like(lstm)
        fused = fuse_scores(model, ifs, lstm)
        thr = float(np.percentile(fused[:min(200, len(fused))], 99))
        dec = (fused >= thr).astype(int)
        return fused, dec, thr
    except Exception:
        # fail-safe
        return np.array([]), np.array([]), 0.0

# ────────────────────────────────────────────────────────────────────
# Session State (init first!)
# ────────────────────────────────────────────────────────────────────
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=6000)  # RMS

for k in ("vibration_x","vibration_y","vibration_z","live_rpm","live_temp"):
    if k not in st.session_state:
        st.session_state[k] = deque(maxlen=6000)

for k in ("live_running","mqtt_connected","mqtt_last_err"):
    if k not in st.session_state:
        st.session_state[k] = False if k != "mqtt_last_err" else ""

if "asset_name" not in st.session_state:
    st.session_state.asset_name = "Motor-001"

# ────────────────────────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────────────────────────
model = load_model()

# ────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Data Source")
    source = st.radio("Choose input", ["Upload / CSV", "Demo (static)", "Simulated Stream", "MQTT Live"], index=1)
    st.text_input("Asset Name", value=st.session_state.asset_name, key="asset_name")

    st.divider()
    st.header("⚙️ Visualization")
    max_points = st.slider("Max chart points (downsample)", 200, 4000, 800, 100)
    update_interval = st.slider("Live update (ms)", 100, 1500, 400, 50)

# ────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────
tab_overview, tab_signals, tab_freq, tab_features, tab_anom, tab_live, tab_admin = st.tabs(
    ["Overview", "Signals", "Filters & FFT", "Features", "Anomalies", "Live", "Admin"]
)

# ────────────────────────────────────────────────────────────────────
# Offline modes (keep your structure)
# ────────────────────────────────────────────────────────────────────
if source in ["Upload / CSV", "Demo (static)"]:
    with tab_overview:
        st.title("🛠️ AI-Based Predictive Maintenance Dashboard")
        st.write("**Hybrid Ensemble:** LSTM Autoencoder (sequence) + Isolation Forest (feature)")

        if source == "Upload / CSV":
            up = st.file_uploader("Upload CSV containing `vibration_rms` or axes x,y,z", type=["csv"])
            if up is None: st.stop()
            df = pd.read_csv(up)
        else:
            df = load_demo_dataframe()

        if "vibration_rms" not in df.columns:
            if {"x","y","z"}.issubset(df.columns):
                df["vibration_rms"] = np.sqrt((df["x"]**2 + df["y"]**2 + df["z"]**2)/3.0)
            else:
                st.error("❌ Provide `vibration_rms` or columns `x,y,z`."); st.stop()

        out = score_offline(model, df)
        c1,c2,c3 = st.columns(3)
        c1.metric("Samples", len(df))
        c2.metric("Windows Scored", int(len(out["fused"])))
        c3.metric("Fault Windows", int((out["decisions"]==1).sum()))

        step = max(1, len(out["fused"]) // max_points)
        fused_view = out["fused"][::step]
        st.subheader("📈 Anomaly Score Timeline")
        st.line_chart(pd.DataFrame({"Anomaly Score": fused_view}))

        st.caption(f"Decision threshold (99th pct on baseline): {out['threshold']:.4f}")

    with tab_signals:
        st.subheader("📊 Raw Signals")
        cols = [c for c in df.columns if c not in ["timestamp"]]
        sel = st.multiselect("Select signals to plot", cols, default=[c for c in ["x","y","z","vibration_rms"] if c in cols])
        if sel:
            view = df[sel].iloc[::max(1, len(df)//max_points)]
            st.line_chart(view)

    with tab_freq:
        st.subheader("🔧 Filtered Signal (Low-Pass)")
        if SCIPY_OK:
            filt = lowpass(df["vibration_rms"].values)
            view = pd.DataFrame({"vibration_rms": df["vibration_rms"].iloc[::10].values[:len(filt[::10])],
                                 "filtered": filt[::10]})
            st.line_chart(view)
        else:
            st.info("Install `scipy` to enable filtering.")

        st.subheader("⚡ Frequency Spectrum (FFT)")
        sig = df["vibration_rms"].values
        freq = np.fft.rfftfreq(len(sig), 1/1000)
        amp = np.abs(np.fft.rfft(sig))
        step_f = max(1, len(amp)//max_points)
        fft_df = pd.DataFrame({"Amplitude": amp[::step_f]}, index=freq[::step_f])
        st.line_chart(fft_df)

    with tab_features:
        st.subheader("📐 Extracted Features")
        feats = compute_features(df["vibration_rms"].values)
        st.table(feats)

    with tab_anom:
        st.subheader("🔍 Anomaly Details")
        out = score_offline(model, df)
        fused = out["fused"]; decisions = out["decisions"]
        fault_idx = np.where(decisions == 1)[0]
        if len(fault_idx)==0:
            st.success("✅ No anomalies detected.")
        else:
            st.error(f"⚠️ {len(fault_idx)} fault windows detected.")
            st.write(f"First 50 fault windows: {fault_idx[:50].tolist()}")

# ────────────────────────────────────────────────────────────────────
# LIVE: Simulated Stream + MQTT (queue-based, smooth, zoomable)
# ────────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("🟢 Live Monitoring (Real-Time)")
    mode = st.radio("Live Mode:", ["Simulated Stream", "MQTT Live"], horizontal=True)

    # ---- Process queued messages (MQTT / Sim Stream use same code) ----
    def process_message_queue():
        processed = 0
        q = st.session_state.message_queue
        while not q.empty():
            msg = q.get_nowait()
            x = float(msg.get("x", 0.0))
            y = float(msg.get("y", 0.0))
            z = float(msg.get("z", 0.0))
            rpm = float(msg.get("rpm", 1500))
            temp = float(msg.get("temp", 65))

            rms = math.sqrt((x*x + y*y + z*z)/3.0)

            st.session_state.vibration_x.append(x)
            st.session_state.vibration_y.append(y)
            st.session_state.vibration_z.append(z)
            st.session_state.live_buffer.append(rms)
            st.session_state.live_rpm.append(rpm)
            st.session_state.live_temp.append(temp)
            processed += 1
        return processed

    # ---- UI Controls ----
    left, right = st.columns([2,1])
    with left:
        series_choice = st.selectbox("Signal", ["RMS", "X", "Y", "Z", "RPM", "Temp"])
    with right:
        display_points = st.slider("Display Points", 200, 2000, max_points, 100)

    chart_placeholder = st.empty()
    k1, k2, k3, k4 = st.columns(4)

    def get_series(choice):
        return np.array({
            "RMS": st.session_state.live_buffer,
            "X":   st.session_state.vibration_x,
            "Y":   st.session_state.vibration_y,
            "Z":   st.session_state.vibration_z,
            "RPM": st.session_state.live_rpm,
            "Temp":st.session_state.live_temp
        }[choice])

    def draw_chart(series, title):
        series = series[-display_points:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, mode="lines", name=title, line=dict(width=2)))
        fig.update_layout(title=title + " — Live", template="plotly_white", height=380)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # ---- MODE A: Simulated Stream ----
    if mode == "Simulated Stream":
        sim_rate = st.slider("Samples per tick", 1, 50, 20)
        c1, c2 = st.columns(2)
        if c1.button("▶️ Start"): st.session_state.live_running = True
        if c2.button("⏸️ Stop"): st.session_state.live_running = False

        demo = load_demo_dataframe()
        xs, ys, zs = demo["x"].values, demo["y"].values, demo["z"].values

        if st.session_state.live_running:
            if "sim_idx" not in st.session_state: st.session_state.sim_idx = 0
            for _ in range(sim_rate):
                j = st.session_state.sim_idx % len(xs)
                st.session_state.message_queue.put({
                    "x": xs[j], "y": ys[j], "z": zs[j],
                    "rpm": 1500 + 10*np.sin(j/200),
                    "temp": 65 + 1.0*np.sin(j/350)
                })
                st.session_state.sim_idx += 1

    # ---- MODE B: MQTT Live ----
    else:
        st.caption("🌍 Broker: broker.hivemq.com  |  Topic: machine/vibration/data")

        def create_client():
            client = mqtt.Client()
            client.on_message = lambda c,u,m: st.session_state.message_queue.put(json.loads(m.payload))
            client.connect("broker.hivemq.com", 1883, 60)
            client.subscribe("machine/vibration/data")
            client.loop_start()
            return client

        if not st.session_state.mqtt_connected and st.button("🔌 Connect"):
            st.session_state.mqtt_client = create_client()
            st.session_state.mqtt_connected = True

        if st.session_state.mqtt_connected:
            st.success("✅ Connected (Receiving)")

    # ---- Refresh UI ----
    processed = process_message_queue()
    series = get_series(series_choice)

    if series.size:
        k1.metric("Samples", int(series.size))
        draw_chart(series, series_choice)
    else:
        st.info("Waiting for data...")

    time.sleep(update_interval/1000)
    st.rerun()


# ────────────────────────────────────────────────────────────────────
# Admin
# ────────────────────────────────────────────────────────────────────
with tab_admin:
    st.subheader("🏷️ Asset Health Table")
    latest_obs = float(st.session_state.live_buffer[-1]) if len(st.session_state.live_buffer) else np.nan
    threshold_hint = 0.6
    status = "Alert" if (not np.isnan(latest_obs) and latest_obs > threshold_hint) else "OK"
    priority = "High" if status == "Alert" else "Low"
    created_on = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    asset = st.session_state.asset_name
    table = pd.DataFrame([{
        "Asset Name": asset, "Signal": "vibration_rms", "Priority": priority,
        "Status": status, "Created On": created_on, "Threshold": threshold_hint, "Observed": latest_obs
    }])
    st.dataframe(table, use_container_width=True)
    st.download_button("⬇️ Export Asset Table (CSV)", data=table.to_csv(index=False).encode("utf-8"),
                       file_name="asset_health_table.csv", mime="text/csv")
