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
    st.session_state.live_buffer = deque(maxlen=6000)   # keep ~ last 6k samples
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
with tab_live:
    st.subheader("🟢 Live Monitoring (Real-Time)")

    mode = st.radio("Select Live Mode:", ["Simulated Stream (A2)", "MQTT Live"], horizontal=True)

    # Placeholders
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    live_placeholder = st.empty()
    meta_placeholder = st.empty()
    debug_placeholder = st.empty()

    # ---- small live scorer (no cache; used only on short buffers) ----
    def score_live_window(model: HybridEnsemble, arr: np.ndarray):
        if len(arr) < 32:
            return np.array([]), np.array([]), 0.0
        df = pd.DataFrame({"vibration_rms": arr})
        lstm = model.score_sequences(df, "vibration_rms")
        ifs = np.zeros_like(lstm)
        fused = fuse_scores(model, ifs, lstm)
        base = min(200, len(fused))
        thr = float(np.percentile(fused[:base], 99))
        dec = (fused >= thr).astype(int)
        return fused, dec, thr

    # ---- append sample(s) into session buffers (no UI in callbacks) ----
    def push_sample_xyz(x: float, y: float, z: float, rpm: float | None = None, temp: float | None = None):
        rms = float(math.sqrt((x*x + y*y + z*z) / 3.0))
        st.session_state.live_buffer.append(rms)
        if rpm is not None:
            st.session_state.live_rpm.append(float(rpm))
        if temp is not None:
            st.session_state.live_temp.append(float(temp))

    # ─────────────────────────────
    # Mode A2 — simulated streaming
    # ─────────────────────────────
    if mode == "Simulated Stream (A2)":
        st.write("📡 Streaming from demo data as live feed")

        # controls
        sim_rate = st.slider("Samples per frame", 1, 100, 20)
        base_rpm = st.slider("Base RPM (sim)", 600, 3600, 1500, 100)
        base_temp = st.slider("Base Temp °C (sim)", 30, 100, 65, 1)

        cstart, cpause, creset = st.columns(3)
        if cstart.button("▶️ Start"):
            st.session_state.live_running = True
        if cpause.button("⏸️ Pause"):
            st.session_state.live_running = False
        if creset.button("🛑 Reset"):
            st.session_state.live_running = False
            st.session_state.live_buffer.clear()
            st.session_state.live_rpm.clear()
            st.session_state.live_temp.clear()
            st.session_state.pop("sim_idx", None)

        demo = load_demo_dataframe()
        if {"x","y","z"}.issubset(demo.columns):
            xs, ys, zs = demo["x"].values, demo["y"].values, demo["z"].values
        else:
            rms = demo["vibration_rms"].values
            xs, ys, zs = rms*0.97, rms*1.01, rms*1.03

        # emit a few samples per frame when running
        if st.session_state.live_running:
            if "sim_idx" not in st.session_state:
                st.session_state.sim_idx = 0
            i0, i1 = st.session_state.sim_idx, st.session_state.sim_idx + sim_rate
            for i in range(i0, i1):
                j = i % len(xs)
                # synthesize rpm/temp with slight drift + noise
                rpm = base_rpm + 30*np.sin(2*np.pi*(j/500.0)) + np.random.randn()*5
                temp = base_temp + 2.0*np.sin(2*np.pi*(j/900.0)) + np.random.randn()*0.3
                push_sample_xyz(float(xs[j]), float(ys[j]), float(zs[j]), rpm=rpm, temp=temp)
            st.session_state.sim_idx = i1

        # draw UI from latest buffer
        recent = np.array(list(st.session_state.live_buffer)[-max_points:])
        recent_rpm = np.array(list(st.session_state.live_rpm)[-max_points:]) if len(st.session_state.live_rpm) else None
        recent_tmp = np.array(list(st.session_state.live_temp)[-max_points:]) if len(st.session_state.live_temp) else None

        # KPIs (computed on short buffer)
        if recent.size:
            fused, dec, thr = score_live_window(model, recent)
            faults = int((dec == 1).sum()) if dec.size else 0
            last_rms = float(recent[-1])
            last_rpm = float(recent_rpm[-1]) if recent_rpm is not None and recent_rpm.size else float("nan")
            last_tmp = float(recent_tmp[-1]) if recent_tmp is not None and recent_tmp.size else float("nan")
            last_fused = float(fused[-1]) if fused.size else 0.0
            health = max(0.0, 100.0*(1.0 - (last_fused / (thr + 1e-9))))
            health = float(np.clip(health, 0, 100))

            kpi_col1.metric("Health Score", f"{health:.0f} %")
            kpi_col2.metric("RPM", f"{last_rpm:.0f}" if not math.isnan(last_rpm) else "—")
            kpi_col3.metric("Temp (°C)", f"{last_tmp:.1f}" if not math.isnan(last_tmp) else "—")
            kpi_col4.metric("Fault windows (buffer)", faults)

            plot_df = pd.DataFrame({"RMS": recent})
            if recent_rpm is not None:  plot_df["RPM"] = recent_rpm
            if recent_tmp is not None:  plot_df["Temp"] = recent_tmp
            live_placeholder.line_chart(plot_df)

            meta_placeholder.caption(f"Δt={update_interval}ms • buffer={len(st.session_state.live_buffer)} • thr={thr:.4f} • last RMS={last_rms:.4f}")

        time.sleep(update_interval/1000.0)
        st.rerun()

    # ─────────────────────────────
    # Mode: MQTT Live (x,y,z,rpm,temp)
    # ─────────────────────────────
    if mode == "MQTT Live":
        st.write("🌍 Connect to public MQTT broker (HiveMQ WebSocket).")
        st.caption("Topic: `machine/vibration/data`  •  Payload: `{ \"x\":0.54, \"y\":0.49, \"z\":0.61, \"rpm\":1480, \"temp\":67.4 }`")

        if not MQTT_OK:
            st.error("Install MQTT:  `pip install paho-mqtt`")
        else:
            use_tls = st.toggle("Use secure WebSocket (wss)", value=True,
                                help="Enable for Streamlit Cloud")
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

            # ⚠️ No Streamlit calls here
            def on_message(client, userdata, msg):
                try:
                    j = json.loads(msg.payload.decode("utf-8"))
                    x = float(j.get("x", 0))
                    y = float(j.get("y", 0))
                    z = float(j.get("z", 0))
                    rpm = float(j.get("rpm")) if "rpm" in j else None
                    temp = float(j.get("temp")) if "temp" in j else None
                    push_sample_xyz(x, y, z, rpm=rpm, temp=temp)
                except Exception:
                    # ignore malformed payloads
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
                st.session_state.live_buffer.clear()
                st.session_state.live_rpm.clear()
                st.session_state.live_temp.clear()

            # connection status
            if st.session_state.mqtt_connected:
                kpi_col1.success("Connected")
            if st.session_state.mqtt_last_err:
                st.warning(st.session_state.mqtt_last_err)

        # UI driven by buffer
        recent = np.array(list(st.session_state.live_buffer)[-max_points:])
        recent_rpm = np.array(list(st.session_state.live_rpm)[-max_points:]) if len(st.session_state.live_rpm) else None
        recent_tmp = np.array(list(st.session_state.live_temp)[-max_points:]) if len(st.session_state.live_temp) else None

        if recent.size:
            fused, dec, thr = score_live_window(model, recent)
            faults = int((dec == 1).sum()) if dec.size else 0
            last_rpm = float(recent_rpm[-1]) if recent_rpm is not None and recent_rpm.size else float("nan")
            last_tmp = float(recent_tmp[-1]) if recent_tmp is not None and recent_tmp.size else float("nan")
            last_fused = float(fused[-1]) if fused.size else 0.0
            health = max(0.0, 100.0*(1.0 - (last_fused / (thr + 1e-9))))
            health = float(np.clip(health, 0, 100))

            kpi_col1.metric("Health Score", f"{health:.0f} %")
            kpi_col2.metric("RPM", f"{last_rpm:.0f}" if not math.isnan(last_rpm) else "—")
            kpi_col3.metric("Temp (°C)", f"{last_tmp:.1f}" if not math.isnan(last_tmp) else "—")
            kpi_col4.metric("Fault windows (buffer)", faults)

            plot_df = pd.DataFrame({"RMS": recent})
            if recent_rpm is not None:  plot_df["RPM"] = recent_rpm
            if recent_tmp is not None:  plot_df["Temp"] = recent_tmp
            live_placeholder.line_chart(plot_df)

            meta_placeholder.caption(
                f"Δt={update_interval}ms • buffer={len(st.session_state.live_buffer)} • thr={thr:.4f}"
            )

        # throttle reruns so Cloud doesn’t freeze
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
