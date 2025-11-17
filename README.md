🚀 AI-Driven Predictive Maintenance Platform (Real-Time + Anomaly Detection + MQTT + Telegram Alerts)

A Hybrid ML + IoT Simulation System for Industrial Machine Health Monitoring

⸻

📌 Overview

This project is a complete end-to-end Predictive Maintenance System combining:
	•	IoT-style real-time data streaming (via MQTT)
	•	AI/ML anomaly detection
	•	Live dashboards and signal visualization (Streamlit)
	•	Physics-aware vibration analytics (RMS, FFT, x/y/z accelerometer signals)
	•	Telegram alerting for equipment faults
	•	Hybrid ML model using LSTM Autoencoder + Isolation Forest
	•	Dynamic, zoomable live monitoring charts (Plotly)

The entire system simulates a real industrial vibration-monitoring pipeline used in motors, pumps, compressors, turbines, and rotating machinery.

Professional predictive maintenance companies (e.g., Presage Insights, Augury, ABB Ability) use similar end-to-end architectures.

⸻

🎯 Key Features

🟢 Real-Time Monitoring
	•	Live vibration (x, y, z), RMS, temperature, RPM
	•	MQTT-based streaming at 5–10 Hz
	•	High-resolution moving waveform charts (Plotly)
	•	Auto-updating dashboards

⚙ Smart Anomaly Detection (Hybrid ML System)

Component	Purpose	Technology
🧠 LSTM Autoencoder	Learns normal vibration patterns	Deep learning sequence model
🌲 Isolation Forest	Detects statistical outliers	Unsupervised anomaly detection
⚡ Hybrid Ensemble Fusion	Combines both scores	Weighted fusion

This design mimics industrial-grade analytics systems.

📨 Telegram Fault Notifications

Automatic alerts when RMS exceeds safety thresholds:
	•	“High Vibration Alert”
	•	“Vibration normalized”
	•	Alert cooldown + visual history table

📊 Signal Analysis Tools
	•	Raw Accelerometer Plots
	•	RMS envelope
	•	Low-pass filtering
	•	FFT spectrum
	•	Feature engineering: RMS, Peak, Kurtosis, Skewness

🎛 Administrative Panel
	•	Machine health table
	•	Thresholds
	•	Real-time observations
	•	Downloadable CSV report

⸻

🏗 System Architecture

                    ┌──────────────────────┐
                    │   Synthetic Sensor   │
                    │  (Python Publisher)  │
                    └──────────┬───────────┘
                               │ MQTT
                               ▼
                     mqtt://broker.hivemq.com
                               │
                               ▼
         ┌─────────────────────────────────────────────┐
         │          Streamlit Real-Time Dashboard      │
         │  - Live Charts                              │
         │  - RMS Analytics                            │
         │  - FFT + Features                           │
         │  - ML Hybrid Model                          │
         │  - Alerts + History                         │
         └──────────┬──────────────────────────────────┘
                    │ Telegram Bot API
                    ▼
              📱 Fault Notifications


⸻

📂 Repository Structure

project/
│
├── streamlit_app.py              # Main dashboard
├── publisher.py                  # IoT data simulator (MQTT publisher)
├── README.md                     # Documentation
│
└── src/
    ├── models/
    │   └── hybrid_ensemble.py    # LSTM AE + IF hybrid model
    │
    └── utils/
        ├── telegram_alert.py     # Telegram send function
        └── preprocessing.py      # RMS, filters, feature functions

models/
└── saved_models/
    ├── lstm_ae.keras
    ├── if_model.pkl
    ├── scaler.pkl
    └── meta.json


⸻

🧠 Machine Learning Models Used

🔹 1. LSTM Autoencoder

Unsupervised sequence-learning model.
	•	Input: vibration_rms time windows
	•	Learns “normal machine behavior”
	•	Reconstruction error → anomaly score
	•	High error = abnormal vibration pattern

🔹 2. Isolation Forest
	•	Tree-based unsupervised outlier detector
	•	Computes anomaly score using feature vectors
	•	Used when features deviate from baseline behavior

🔹 3. Hybrid Ensemble Scoring

Final Score = 0.5 * LSTM_AE_Score + 0.5 * IF_Score
Threshold = 99th percentile baseline fused score


⸻

📊 Signal Processing Used

Technique	Purpose
RMS (Root Mean Square)	Industry-standard vibration health metric
FFT	Frequency fault detection
Low-pass Filter	Remove high-frequency noise
XYZ Vector → RMS fusion	Convert raw accelerometer signals

Features computed:
	•	RMS
	•	Peak
	•	Kurtosis
	•	Skewness

⸻

🚀 How to Run

1. Clone the repository

git clone https://github.com/yourusername/predictive-maintenance-dashboard.git
cd predictive-maintenance-dashboard


⸻

2. Install dependencies

pip install -r requirements.txt


⸻

3. Start the Streamlit Dashboard

streamlit run streamlit_app.py

Open:

http://localhost:8501


⸻

4. Start the MQTT Publisher (in Colab or locally)

!python publisher.py

This generates live simulated IoT data.

⸻

🛰 Telegram Alerts Setup

Step 1 — Create bot

@BotFather → /newbot

Step 2 — Add bot token

BOT_TOKEN = "YOUR_TOKEN"
CHAT_ID   = "YOUR_CHAT_ID"

Step 3 — Test

python test_bot.py


⸻

💡 What Makes This Project Industry-Level?

✓ Real IoT architecture

MQTT → processing → ML → UI → alerts.

✓ Hybrid ML architecture

Deep learning + traditional ML combined.

✓ Real-time Plotly visualizations

Smooth, zoomable, industry-grade.

✓ Notification system

Instant alerting like industrial control systems.

✓ Modular, production-like code structure

Mirrors real predictive maintenance platforms.

✓ Expandable

Can easily integrate:
	•	Cloud IoT (AWS IoT Core / Azure IoT Hub)
	•	Real sensors
	•	Edge devices
	•	More advanced ML

⸻

🚀 Future Enhancements

🔹 Fault Classification (bearing wear, imbalance, misalignment)

Add ML classifier for individual faults.

🔹 RUL (Remaining Useful Life)

Predict machine failure timeline.

🔹 Digital Twin Simulation

Physics-based modeling of rotating systems.

🔹 Cloud Integration

Store time series in:
	•	InfluxDB
	•	TimescaleDB
	•	DynamoDB

🔹 Mobile App / Dashboard Cloud Deployment

⸻

📚 References
	1.	S. Hochreiter, “LSTM Networks,” Neural Computation, 1997.
	2.	Scikit-Learn Isolation Forest Documentation
	3.	MQTT v3.1 Standard
	4.	Engineering Vibration Textbooks (RMS, frequency analysis)
	5.	Predictive Maintenance Industry Standards (ISO 13373-2)

⸻

🏁 Conclusion

This project demonstrates a full-stack AI + IoT predictive maintenance system built from scratch, integrating:
	•	Real-time data ingestion
	•	Deep learning
	•	Signal processing
	•	Visual analytics
	•	Fault alerting



