import json, time, math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional imports (graceful fallback if not present)
try:
    from scipy.signal import butter, filtfilt
    from scipy.stats import kurtosis, skew
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# MQTT optional
MQTT_OK = True
try:
    import paho.mqtt.client as mqtt
except Exception:
    MQTT_OK = False

from src.models.hybrid_ensemble import HybridEnsemble

# ────────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def ensure_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

@st.cache_resource
def load_model():
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.error(
            "❌ Trained model files missing in `models/saved_models/hybrid/`:\n" +
            "\n".join(f"- {f}" for f in required)
        )
        st.stop()
    return HybridEnsemble.load(MODEL_DIR.as_posix())

@st.cache_data
def load_demo_dataframe() -> pd.DataFrame:
    if ensure_exists(DEMO_CSV):
        return pd.read_csv(DEMO_CSV)
    # Tiny synthetic fallback (always available)
    n = 5000
    t = np.arange(n) / 200.0
    # synthetic 3-axis with subtle differences
    x = 0.5 + 0.05*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
    y = 0.5 + 0.04*np.sin(2*np.pi*3.2*t + 0.3) + 0.02*np.random.randn(n)
    z = 0.5 + 0.06*np.sin(2*np.pi*2.8*t - 0.2) + 0.02*np.random.randn(n)
    # inject anomaly burst
    z[2000:2100] += 0.25*np.sin(2*np.pi*15*t[2000:2100])
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df["vibration_rms"] = np.sqrt((df["x"]**2 + df["y"]**2 + df["z"]**2)/3.0)
    return df

def lowpass(x, cutoff=50, fs=1000, order=4):
    if not SCIPY_OK:
        return x  # fallback: no filtering
    b, a = butter(order, cutoff/(0.5*fs), btype="low")
    return filtfilt(b, a, x)

def fuse_scores(model, if_scores: np.ndarray, lstm_scores: np.ndarray) -> np.ndarray:
    try:
        return model.combine_scores(if_scores, lstm_scores)
    except Exception:
        m = min(len(if_scores), len(lstm_scores))
        return 0.5 * if_scores[:m] + 0.5 * lstm_scores[:m]

def make_decisions(fused: np.ndarray, baseline: int = 2000, pctl: float = 99.0) -> np.ndarray:
    if len(fused) == 0:
        return np.array([], dtype=int)
    thr = np.percentile(fused[:min(baseline, len(fused))], pctl)
    return (fused >= thr).astype(int)

@st.cache_data(
    hash_funcs={
        pd.DataFrame: lambda _: None,
        HybridEnsemble: lambda _: None,  # CRITICAL FIX
    }
)
def score_offline(model: HybridEnsemble, df: pd.DataFrame) -> dict:
    lstm_scores = model.score_sequences(df, signal_col="vibration_rms")
    if_scores = np.zeros_like(lstm_scores)
    fused = fuse_scores(model, if_scores, lstm_scores)

    base = min(2000, len(fused))
    thr = np.percentile(fused[:base], 99)
    decisions = (fused >= thr).astype(int)

    return {
        "lstm_scores": lstm_scores,
        "if_scores": if_scores,
        "fused": fused,
        "decisions": decisions,
        "threshold": float(thr)
    }

def compute_features(sig: np.ndarray) -> pd.DataFrame:
    rms = float(np.sqrt(np.mean(sig**2)))
    peak = float(np.max(np.abs(sig)))
    krt = float(kurtosis(sig)) if SCIPY_OK else float("nan")
    skw = float(skew(sig)) if SCIPY_OK else float("nan")
    return pd.DataFrame({"RMS":[rms], "Peak":[peak], "Kurtosis":[krt], "Skewness":[skw]})

# ────────────────────────────────────────────────────────────────────────────────
# Session state for Live modes
# ────────────────────────────────────────────────────────────────────────────────
if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=6000)   # vibration RMS

if "live_rpm" not in st.session_state:
    st.session_state.live_rpm = deque(maxlen=6000)      # RPM / speed

if "live_temp" not in st.session_state:
    st.session_state.live_temp = deque(maxlen=6000)     # Temperature

if "live_running" not in st.session_state:
    st.session_state.live_running = False

if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "mqtt_last_err" not in st.session_state:
    st.session_state.mqtt_last_err = ""

if "asset_name" not in st.session_state:
    st.session_state.asset_name = "Motor-001"


# ────────────────────────────────────────────────────────────────────────────────
# Load model & data source selection
# ────────────────────────────────────────────────────────────────────────────────
model = load_model()

with st.sidebar:
    st.header("📦 Data Source")
    source = st.radio(
        "Choose input",
        ["Upload / CSV", "Demo (static)", "Simulated Stream (A2)", "MQTT Live"],
        index=1,
        help="Switches the dashboard between offline analysis and live modes."
    )
    st.text_input("Asset Name", value=st.session_state.asset_name, key="asset_name")

    st.divider()
    st.header("⚙️ Visualization")
    max_points = st.slider("Max chart points (downsample)", 200, 4000, 800, 100)
    update_interval = st.slider("Live update (ms)", 100, 1500, 400, 50)

    st.divider()
    if not SCIPY_OK:
        st.warning("`scipy` not available → filtering/feature kurtosis/skew use fallbacks.", icon="⚠️")
    if not MQTT_OK:
        st.info("Install `paho-mqtt` to enable MQTT live mode.", icon="ℹ️")

# ────────────────────────────────────────────────────────────────────────────────
# Tabs keep your earlier sections but compute on demand (faster)
# ────────────────────────────────────────────────────────────────────────────────
tab_overview, tab_signals, tab_freq, tab_features, tab_anom, tab_live, tab_admin = st.tabs(
    ["Overview", "Signals", "Filters & FFT", "Features", "Anomalies", "Live", "Admin"]
)

# ────────────────────────────────────────────────────────────────────────────────
# Source: Upload / Demo (offline analysis)
# ────────────────────────────────────────────────────────────────────────────────
if source in ["Upload / CSV", "Demo (static)"]:
    with tab_overview:
        st.title("🛠️ AI-Based Predictive Maintenance Dashboard")
        st.write("""
**Hybrid Ensemble:** LSTM Autoencoder (sequence) + Isolation Forest (feature)  
Detects **early faults** in rotating machinery from vibration signals.
""")

        if source == "Upload / CSV":
            up = st.file_uploader("Upload CSV containing `vibration_rms` or axes x,y,z", type=["csv"])
            if up is None:
                st.stop()
            df = pd.read_csv(up)
        else:
            df = load_demo_dataframe()

        # Accept x,y,z too → compute vibration_rms
        if "vibration_rms" not in df.columns:
            axes = {"x","y","z"}
            if axes.issubset(df.columns):
                df["vibration_rms"] = np.sqrt((df["x"]**2 + df["y"]**2 + df["z"]**2)/3.0)
            else:
                st.error("❌ Provide `vibration_rms` or columns `x,y,z`.")
                st.stop()

        with st.spinner("Scoring with LSTM…"):
            out = score_offline(model, df)

        c1,c2,c3 = st.columns(3)
        c1.metric("Samples", len(df))
        c2.metric("Windows Scored", int(len(out["fused"])))
        c3.metric("Fault Windows", int((out["decisions"]==1).sum()))

        # Anomaly trend (downsampled)
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

        # Heatmap (lightweight)
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.imshow(fused[np.newaxis, :], aspect="auto", cmap="coolwarm")
        ax.set_yticks([])
        ax.set_xlabel("Window Index")
        ax.set_title("Fault Heatmap")
        st.pyplot(fig)

        # Download
        res = pd.DataFrame({
            "index": np.arange(len(fused)),
            "lstm_score": out["lstm_scores"][:len(fused)],
            "if_score": out["if_scores"][:len(fused)],
            "fused_score": fused,
            "label": decisions
        })
        st.download_button("⬇️ Download Predictions CSV", res.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

# ────────────────────────────────────────────────────────────────────────────────
# LIVE tab: Simulated Stream (A2) and MQTT  (REPLACE THIS WHOLE BLOCK)
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# LIVE tab: Simulated Stream (A2) and MQTT
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# LIVE tab: Simulated Stream (A2) and MQTT
# ────────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("🟢 Live Monitoring (Real-Time)")

    mode = st.radio("Select Live Mode:", ["Simulated Stream (A2)", "MQTT Live"], horizontal=True)

    # KPI + charts placeholders
    k1, k2, k3, k4 = st.columns(4)
    chart_area = st.empty()
    status_area = st.empty()

    # Quick scorer for live windows
    def score_live_window(arr):
        # Need enough samples for the LSTM windowing (very important)
        MIN_SAMPLES = 300   # try 200 if you want faster trigger
    
        if arr is None or len(arr) < MIN_SAMPLES:
            # Not enough data yet → return empty signals, no scoring
            return np.array([]), np.array([]), 0.0
    
        df = pd.DataFrame({"vibration_rms": arr})
    
        try:
            lstm_scores = model.score_sequences(df, "vibration_rms")
        except Exception:
            # If model crashes due to sequence edge conditions, skip scoring this cycle
            return np.array([]), np.array([]), 0.0
    
        if_scores = np.zeros_like(lstm_scores)
    
        fused = fuse_scores(model, if_scores, lstm_scores)
    
        thr = float(np.percentile(fused[:min(2000, len(fused))], 99))
        decisions = (fused >= thr).astype(int)
    
        return fused, decisions, thr


    # Push samples to buffer
    def push_sample_data(x, y, z, rpm=None, temp=None):
        rms = math.sqrt((x*x + y*y + z*z) / 3.0)
        st.session_state.live_buffer.append(rms)
    
        if rpm is not None:
            st.session_state.live_rpm.append(float(rpm))
    
        if temp is not None:
            st.session_state.live_temp.append(float(temp))


    # ─────────────────────────────
    # MODE A2 - Simulated Stream
    # ─────────────────────────────
    if mode == "Simulated Stream (A2)":
        st.write("📡 Streaming demo data live...")

        sim_rate = st.slider("Samples per update", 1, 100, 20)
        demo = load_demo_dataframe()

        if {"x","y","z"}.issubset(demo.columns):
            xs, ys, zs = demo["x"].values, demo["y"].values, demo["z"].values
        else:
            rms = demo["vibration_rms"].values
            xs, ys, zs = rms*0.97, rms*1.01, rms*1.03

        if st.button("Start Simulation"):
            st.session_state.live_running = True

        if st.session_state.live_running:
            if "sim_idx" not in st.session_state: st.session_state.sim_idx = 0
            for i in range(st.session_state.sim_idx, st.session_state.sim_idx + sim_rate):
                j = i % len(xs)
                push_sample_data(xs[j], ys[j], zs[j],
                                 rpm=1500 + 20*np.sin(j/200),
                                 temp=65 + 1.5*np.sin(j/350))
            st.session_state.sim_idx += sim_rate

        recent = list(st.session_state.live_buffer)[-max_points:]
        recent_rpm = list(st.session_state.live_rpm)[-max_points:]
        recent_temp = list(st.session_state.live_temp)[-max_points:]

        if recent:
            fused, dec, thr = score_live_window(np.array(recent))
            faults = (dec == 1).sum() if len(dec) else 0

            k1.metric("Health", f"{100 - faults:.0f}%")
            k2.metric("RPM", f"{recent_rpm[-1]:.0f}" if recent_rpm else "—")
            k3.metric("Temp (°C)", f"{recent_temp[-1]:.1f}" if recent_temp else "—")
            k4.metric("Fault Windows", faults)

            df_plot = pd.DataFrame({"RMS": recent})
            if recent_rpm: df_plot["RPM"] = recent_rpm
            if recent_temp: df_plot["Temp"] = recent_temp
            chart_area.line_chart(df_plot)

        time.sleep(update_interval/1000)
        st.rerun()

    # ─────────────────────────────
    # MODE MQTT
    # ─────────────────────────────
        # ─────────────────────────────
    # MODE MQTT  (TCP :1883, no websockets)
    # ─────────────────────────────
    if mode == "MQTT Live":
        st.write("🌍 MQTT Live Telemetry Mode (TCP 1883)")

        broker = "broker.hivemq.com"
        port = 1883
        topic = "machine/vibration/data"

        # IMPORTANT: don't call Streamlit UI from callbacks except via session_state
        def on_connect(client, userdata, flags, rc, props=None):
            if rc == 0:
                st.session_state.mqtt_connected = True
                st.session_state.mqtt_last_err = ""
                client.subscribe(topic)
            else:
                st.session_state.mqtt_connected = False
                st.session_state.mqtt_last_err = f"Connect failed (rc={rc})"

        def on_message(client, userdata, msg):
            try:
                j = json.loads(msg.payload.decode("utf-8"))
                x = float(j.get("x", 0.0))
                y = float(j.get("y", 0.0))
                z = float(j.get("z", 0.0))
                rpm = j.get("rpm", None)
                temp = j.get("temp", None)
                push_sample_data(x, y, z, rpm=rpm, temp=temp)
            except Exception:
                # ignore malformed payloads
                pass

        colA, colB = st.columns(2)
        if colA.button("🔌 Connect (TCP 1883)", type="primary", disabled=st.session_state.mqtt_connected):
            try:
                # Plain TCP client (no websockets)
                try:
                    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                except:
                    client = mqtt.Client()
                client.on_connect = on_connect
                client.on_message = on_message
                client.connect(broker, port, 60)
                client.loop_start()
                st.session_state.mqtt_client = client
                time.sleep(0.2)
                st.rerun()
            except Exception as e:
                st.error(f"MQTT error: {e}")

        if colB.button("🔕 Disconnect", disabled=not st.session_state.mqtt_connected):
            c = st.session_state.get("mqtt_client")
            if c:
                c.loop_stop()
                c.disconnect()
            st.session_state.mqtt_connected = False
            st.session_state.mqtt_last_err = "Disconnected"
            st.rerun()

        if st.session_state.mqtt_connected:
            st.success("✅ Connected (TCP 1883)")
        if st.session_state.mqtt_last_err:
            st.warning(st.session_state.mqtt_last_err)

        # Draw latest buffer
        recent = list(st.session_state.live_buffer)[-max_points:]
        recent_rpm = list(st.session_state.live_rpm)[-max_points:]
        recent_temp = list(st.session_state.live_temp)[-max_points:]

        if recent:
            fused, dec, thr = score_live_window(np.array(recent))
            faults = (dec == 1).sum() if len(dec) else 0

            k1.metric("Health", f"{100 - faults:.0f}%")
            k2.metric("RPM", f"{recent_rpm[-1]:.0f}" if recent_rpm else "—")
            k3.metric("Temp (°C)", f"{recent_temp[-1]:.1f}" if recent_temp else "—")
            k4.metric("Fault Windows", faults)

            df_plot = pd.DataFrame({"RMS": recent})
            if recent_rpm: df_plot["RPM"] = recent_rpm
            if recent_temp: df_plot["Temp"] = recent_temp
            chart_area.line_chart(df_plot, use_container_width=True)

        # Keep the app ticking for live redraws
        time.sleep(update_interval/1000.0)
        st.rerun()



# ────────────────────────────────────────────────────────────────────────────────
# Admin: Asset table & thresholds
# ────────────────────────────────────────────────────────────────────────────────
with tab_admin:
    st.subheader("🏷️ Asset Health Table")

    # Build a lightweight snapshot row from latest buffer / last offline result if any
    latest_obs = None
    if len(st.session_state.live_buffer) > 0:
        latest_obs = float(st.session_state.live_buffer[-1])

    threshold_hint = 0.6  # display-only default; your model threshold is percentile-based
    status = "OK"
    priority = "Low"
    if latest_obs is not None and latest_obs > threshold_hint:
        status = "Alert"
        priority = "High"

    created_on = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    asset = st.session_state.asset_name

    table = pd.DataFrame([{
        "Asset Name": asset,
        "Signal": "vibration_rms",
        "Priority": priority,
        "Status": status,
        "Created On": created_on,
        "Threshold": threshold_hint,
        "Observed": latest_obs if latest_obs is not None else np.nan
    }])

    st.dataframe(table, use_container_width=True)

    st.download_button(
        "⬇️ Export Asset Table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="asset_health_table.csv",
        mime="text/csv"
    )
