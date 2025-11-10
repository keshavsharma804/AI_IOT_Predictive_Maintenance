import json, time, math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.utils.telegram_alert import send_alert


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
# Session state for Live modes (multi-sensor + per-axis)
# ────────────────────────────────────────────────────────────────────────────────
from collections import deque

# Main live series (RMS from axes)
if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=6000)  # vibration RMS

# Extra live series
if "live_rpm" not in st.session_state:
    st.session_state.live_rpm = deque(maxlen=6000)     # RPM / speed
if "live_temp" not in st.session_state:
    st.session_state.live_temp = deque(maxlen=6000)    # Temperature
if "live_acoustic" not in st.session_state:
    st.session_state.live_acoustic = deque(maxlen=6000)  # Acoustics
if "live_magnetic" not in st.session_state:
    st.session_state.live_magnetic = deque(maxlen=6000)  # Magnetic Flux
if "live_current" not in st.session_state:
    st.session_state.live_current = deque(maxlen=6000)    # Current

# Per-axis raw vibration (to compute RMS or view individually)
for axis in ("axial", "horizontal", "vertical"):
    key = f"vibration_{axis}"
    if key not in st.session_state:
        st.session_state[key] = deque(maxlen=6000)

# Control & status
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
# LIVE tab: Simulated Stream (A2) and MQTT (sensor/axis/time-window/threshold UI)
# ────────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("🟢 Live Monitoring")

    # ===== Controls row =========================================================
    left, mid, right = st.columns([2, 2, 1.4])

    with left:
        mode = st.radio("Live Mode", ["Simulated Stream (A2)", "MQTT Live"], horizontal=True)

    with mid:
        sensor_choice = st.selectbox(
            "Signal",
            ["Vibration", "Temperature", "Acoustics", "Magnetic Flux", "Current",
             "Time Domain Features", "Frequency Domain Features"]
        )

    with right:
        time_range = st.radio("Time Range", ["1 Day", "1 Week", "1 Month"], horizontal=False)

    # Axis selector only when viewing vibration
    axis_choice = None
    if sensor_choice == "Vibration":
        axis_choice = st.radio("Axis", ["All Axis", "Axial", "Horizontal", "Vertical"], horizontal=True)

    # Threshold slider (affects visualization/KPIs only)
    threshold = st.slider("Set Threshold", 0.0, 10.0, 2.0)

    # Derived helper: how many recent samples for selected time window
    def samples_for_time_window(update_ms: int, label: str) -> int:
        # Approximate samples per second from UI update interval; cap by 6000
        sps = max(1, int(1000 / max(update_ms, 1)))
        if label == "1 Day":
            return min(6000, 60 * 60 * 24 * sps)
        if label == "1 Week":
            return min(6000, 60 * 60 * 24 * 7 * sps)
        if label == "1 Month":
            return min(6000, 60 * 60 * 24 * 30 * sps)
        return 2000

    window_n = min(max_points, samples_for_time_window(update_interval, time_range))

    # Small scorer for the live buffer (no cache)
    def score_live_window(arr: np.ndarray):
        if arr.size < 32:
            return np.array([]), np.array([]), 0.0
        df = pd.DataFrame({"vibration_rms": arr})
        lstm = model.score_sequences(df, "vibration_rms")
        ifs = np.zeros_like(lstm)
        fused = fuse_scores(model, ifs, lstm)
        thr = float(np.percentile(fused[:min(200, len(fused))], 99))
        dec = (fused >= thr).astype(int)
        return fused, dec, thr

    # Add sample(s) into buffers (used by simulator and MQTT callback)
    def push_sample_data(x, y, z, rpm=None, temp=None, acoustic=None, magnetic=None, current=None):
        # Per-axis
        st.session_state.vibration_axial.append(float(x))
        st.session_state.vibration_horizontal.append(float(y))
        st.session_state.vibration_vertical.append(float(z))
        # RMS
        rms = math.sqrt((x*x + y*y + z*z) / 3.0)
        st.session_state.live_buffer.append(rms)
        # Extra channels
        if rpm is not None:      st.session_state.live_rpm.append(float(rpm))
        if temp is not None:     st.session_state.live_temp.append(float(temp))
        if acoustic is not None: st.session_state.live_acoustic.append(float(acoustic))
        if magnetic is not None: st.session_state.live_magnetic.append(float(magnetic))
        if current is not None:  st.session_state.live_current.append(float(current))

    # ======== MODE A2: Simulated streaming ======================================
    if mode == "Simulated Stream (A2)":
        st.caption("📡 Streaming demo data as live feed.")
        sim_rate = st.slider("Samples per update", 1, 100, 20, 1)

        c1, c2, c3 = st.columns(3)
        if c1.button("▶️ Start"):
            st.session_state.live_running = True
        if c2.button("⏸️ Pause"):
            st.session_state.live_running = False
        if c3.button("🛑 Reset"):
            st.session_state.live_running = False
            for key in ("live_buffer","live_rpm","live_temp","live_acoustic","live_magnetic","live_current",
                        "vibration_axial","vibration_horizontal","vibration_vertical"):
                st.session_state[key].clear()
            st.session_state.pop("sim_idx", None)

        demo = load_demo_dataframe()
        if {"x","y","z"}.issubset(demo.columns):
            xs, ys, zs = demo["x"].values, demo["y"].values, demo["z"].values
        else:
            rms = demo["vibration_rms"].values
            xs, ys, zs = rms*0.97, rms*1.01, rms*1.03

        if st.session_state.live_running:
            if "sim_idx" not in st.session_state:
                st.session_state.sim_idx = 0
            i0, i1 = st.session_state.sim_idx, st.session_state.sim_idx + sim_rate
            for i in range(i0, i1):
                j = i % len(xs)
                # synthesize extra channels for demo
                rpm = 1500 + 25*np.sin(j/180)
                temp = 65 + 1.5*np.sin(j/360)
                acoustic = 0.2 + 0.02*np.sin(j/140)
                magnetic = 0.1 + 0.01*np.sin(j/200)
                current = 3.0 + 0.1*np.sin(j/220)
                push_sample_data(xs[j], ys[j], zs[j], rpm=rpm, temp=temp,
                                 acoustic=acoustic, magnetic=magnetic, current=current)
            st.session_state.sim_idx = i1
            time.sleep(update_interval/1000.0)
            st.rerun()

    # ======== MODE MQTT: subscribe & buffer only =================================
    if mode == "MQTT Live":
        st.caption("🌍 MQTT broker: broker.hivemq.com  •  Topic: `machine/vibration/data`")
        st.caption("Payload example: {\"axial\":0.51,\"horizontal\":0.49,\"vertical\":0.55,"
                   "\"rpm\":1480,\"temp\":67.4,\"acoustic\":0.23,\"mag\":0.12,\"current\":3.2}")

        if not MQTT_OK:
            st.error("Install MQTT client:  pip install paho-mqtt")
        else:
            use_tls = st.toggle("Use secure WebSocket (wss)", value=True,
                                help="Enable on Streamlit Cloud; local dev can use ws://")
            broker = "broker.hivemq.com"
            port   = 8884 if use_tls else 8000
            ws_path = "/mqtt"
            topic  = "machine/vibration/data"

            def on_connect(client, userdata, flags, rc, properties=None):
                if rc == 0:
                    st.session_state.mqtt_connected = True
                    client.subscribe(topic, qos=0)
                else:
                    st.session_state.mqtt_last_err = f"MQTT connect failed (rc={rc})"

            # No Streamlit UI calls in this callback—only push to buffers
            def on_message(client, userdata, msg):
                try:
                    j = json.loads(msg.payload.decode("utf-8"))
                    ax = float(j.get("axial", j.get("x", 0.0)))
                    hz = float(j.get("horizontal", j.get("y", 0.0)))
                    vt = float(j.get("vertical", j.get("z", 0.0)))
                    rpm = j.get("rpm")
                    temp = j.get("temp")
                    acoustic = j.get("acoustic")
                    magnetic = j.get("mag")
                    current = j.get("current")
                    push_sample_data(ax, hz, vt, rpm=rpm, temp=temp,
                                     acoustic=acoustic, magnetic=magnetic, current=current)
                except Exception:
                    pass

            ca, cb, cc = st.columns(3)
            if ca.button("🔌 Connect"):
                try:
                    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets")
                    client.ws_set_options(path=ws_path)
                    if use_tls:
                        import ssl
                        client.tls_set(cert_reqs=ssl.CERT_NONE)
                        client.tls_insecure_set(True)
                    client.on_connect = on_connect
                    client.on_message = on_message
                    client.connect(broker, port, 60)
                    client.loop_start()
                    st.session_state.mqtt_client = client
                except Exception as e:
                    st.error(f"MQTT error: {e}")

            if cb.button("🔕 Disconnect"):
                c = st.session_state.get("mqtt_client")
                if c:
                    c.loop_stop()
                    c.disconnect()
                st.session_state.mqtt_connected = False

            if cc.button("🧹 Clear Buffers"):
                for key in ("live_buffer","live_rpm","live_temp","live_acoustic","live_magnetic","live_current",
                            "vibration_axial","vibration_horizontal","vibration_vertical"):
                    st.session_state[key].clear()

            if st.session_state.mqtt_connected:
                st.success("✅ Connected")
            if st.session_state.mqtt_last_err:
                st.warning(st.session_state.mqtt_last_err)

    # ======== Chart & KPIs for the selected signal ===============================
    # Select the correct series to visualize
    def pick_series():
        if sensor_choice == "Vibration":
            if axis_choice == "Axial":
                data = list(st.session_state.vibration_axial)
            elif axis_choice == "Horizontal":
                data = list(st.session_state.vibration_horizontal)
            elif axis_choice == "Vertical":
                data = list(st.session_state.vibration_vertical)
            else:
                data = list(st.session_state.live_buffer)  # RMS
        elif sensor_choice == "Temperature":
            data = list(st.session_state.live_temp)
        elif sensor_choice == "Acoustics":
            data = list(st.session_state.live_acoustic)
        elif sensor_choice == "Magnetic Flux":
            data = list(st.session_state.live_magnetic)
        elif sensor_choice == "Current":
            data = list(st.session_state.live_current)
        else:
            data = list(st.session_state.live_buffer)  # default for feature tabs
        return np.array(data[-window_n:])

    series = pick_series()
# KPIs and plots
    k1, k2, k3, k4 = st.columns(4)
    if series.size:
        # Defaults
        faults = 0
        thr = float("nan")
    
        if sensor_choice == "Vibration":
            fused, dec, thr = score_live_window(series)
            faults = int((dec == 1).sum()) if dec.size else 0
    
            # ✅ send Telegram alert only when we actually have faults
            if faults > 0:
                msg = (
                    f"⚠️ Fault Detected!\n"
                    f"Asset: {st.session_state.asset_name}\n"
                    f"Fault Windows (buffer): {faults}"
                )
                send_alert(msg)
    
            k1.metric("Fault windows (buffer)", faults)
            k2.metric("Decision thr (live)", f"{thr:.4f}" if not np.isnan(thr) else "—")
        else:
            k1.metric("Samples", int(series.size))
            k2.metric("Decision thr (live)", "—")
    
        k3.metric("Last value", f"{series[-1]:.4f}")
        k4.metric("Threshold", threshold)
    
        # Draw chart (line + horizontal threshold marker via matplotlib)
        import plotly.graph_objects as go

        fig = go.Figure()
        
        # Plot main signal (blue)
        fig.add_trace(go.Scatter(
            y=series,
            mode="lines",
            name=sensor_choice,
            line=dict(color="steelblue", width=2)
        ))
        
        # Add threshold reference line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            opacity=0.7
        )
        
        # If vibration — highlight anomalies
        if sensor_choice == "Vibration" and faults > 0 and dec.size:
            fault_idx = np.where(dec == 1)[0]
            fault_vals = series[fault_idx]
        
            fig.add_trace(go.Scatter(
                x=fault_idx,
                y=fault_vals,
                mode="markers",
                marker=dict(color="red", size=8),
                name="Fault"
            ))
        
        fig.update_layout(
            title=f"{sensor_choice} — Live Streaming (Zoom & Pan Enabled)",
            xaxis_title="Time (newest → right)",
            yaxis_title="Value",
            template="plotly_white",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)



        # Feature tabs without retraining
        # ---- Feature Computation Block ----
        if sensor_choice == "Time Domain Features":
            if SCIPY_OK:
                from scipy.stats import kurtosis as _kurt, skew as _skew
                krt = float(_kurt(series))
                skw = float(_skew(series))
            else:
                krt = float("nan")
                skw = float("nan")
        
            rms = float(np.sqrt(np.mean(series**2)))
            peak = float(np.max(np.abs(series)))
        
            st.table(pd.DataFrame({
                "RMS": [rms],
                "Peak": [peak],
                "Kurtosis": [krt],
                "Skewness": [skw]
            }))


        if sensor_choice == "Frequency Domain Features":
            fft = np.abs(np.fft.rfft(series))
            freq = np.fft.rfftfreq(series.size, d=1.0)
            df_fft = pd.DataFrame({"Amplitude": fft}, index=freq)
            st.line_chart(df_fft.head(min(len(df_fft), 2000)))

    else:
        st.info("Waiting for data…")


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
