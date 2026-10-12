# Digital Twin for Industrial Equipment Health Monitoring

> Real-time predictive maintenance system for motor-pump assets using physics-based simulation, MQTT telemetry, and ML-driven anomaly detection.

Built as part of HCLTech's Industrial IoT initiative (Project P-222), this project implements a digital twin of an industrial pump that generates realistic sensor telemetry, detects anomalies using Isolation Forest, classifies fault types with Random Forest, and visualizes everything through both Grafana dashboards and an interactive Streamlit web app.

---

## Architecture

```
                    PUMP DIGITAL TWIN (pump01)
                              │
                    Python Physics Simulator
                    (6 sensors @ 1 Hz + faults)
                              │
                     Sensor Telemetry (JSON)
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
     ┌─────────────────┐         ┌─────────────────┐
     │   Mosquitto     │         │  Streamlit App   │
     │  MQTT Broker    │         │  (standalone)    │
     └────────┬────────┘         └─────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
telemetry topic     truth topic
    └─────────┬──────────┘
              ▼
     ┌─────────────────┐
     │  ML Analytics   │
     │  30-sample win  │
     │  Isolation Forest│  → Health Index
     │  Random Forest   │  → Fault Classification
     │  RUL Estimation  │  → Remaining Useful Life
     └────────┬────────┘
              │
    ┌─────────┴────────┐
    ▼                  ▼
┌──────────────┐ ┌──────────────┐
│  InfluxDB    │ │  Grafana     │
│  time-series │ │  dashboard   │
└──────────────┘ └──────────────┘
```

Two deployment modes are supported:
- **Full stack** (Docker): Mosquitto + InfluxDB + Grafana for production-like monitoring
- **Streamlit** (standalone): Everything runs in-process, no external services needed

---

## Project Structure

```
├── docker-compose.yml          # Mosquitto + InfluxDB + Grafana
├── streamlit_app.py            # Standalone web dashboard (no Docker needed)
├── requirements.txt
├── mosquitto/
│   └── mosquitto.conf
├── grafana/
│   ├── provisioning/           # Auto-provisioned datasource + dashboard loader
│   └── dashboards/
│       └── pump01_twin.json    # 5-row Grafana dashboard
├── src/
│   ├── config.py               # Centralized constants (topics, credentials, paths)
│   ├── asset_model.py          # PumpTwin physics model with 4 fault types
│   ├── features.py             # Feature engineering (14-dim sliding window)
│   ├── feature_engineering.py  # Extended feature set (40+ features for research)
│   ├── generate_dataset.py     # Synthetic training data generator
│   ├── train.py                # Offline ML training (IF + RF)
│   ├── real_data_train.py      # Real-world dataset experiments (SKAB, AI4I)
│   ├── simulator.py            # Live MQTT publisher (1 Hz telemetry)
│   ├── analytics.py            # ML inference service + InfluxDB writer
│   ├── control.py              # Fault injection CLI
│   └── datasets/               # Real-world dataset loaders
│       ├── base_loader.py
│       ├── skab_loader.py
│       ├── ai4i_loader.py
│       ├── nasa_ims_loader.py
│       └── paderborn_loader.py
├── models/                     # Trained model artifacts (.joblib)
├── data/                       # Generated training CSV
├── datasets/                   # Raw/processed real-world data
├── results/                    # Experiment outputs and plots
└── tests/
    ├── test_feature_engineering.py
    ├── test_inference_compat.py
    ├── test_leakage.py
    └── test_skab_loader.py
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (only for the full-stack mode)

### Installation

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option A: Streamlit Dashboard (quick start)

No Docker needed. The simulator and ML pipeline run entirely in Python:

```powershell
python -m src.train          # train models (~30 seconds, run once)
streamlit run streamlit_app.py
```

Open http://localhost:8501 — use the sidebar to inject faults and observe the ML response in real time.

### Option B: Full Stack with Docker

```powershell
# 1. Start infrastructure
docker compose up -d

# 2. Train models (run once)
python -m src.train

# 3. Start the analytics service (Terminal 1)
python -m src.analytics

# 4. Start the simulator (Terminal 2)
python -m src.simulator

# 5. Inject faults during demo (Terminal 3)
python -m src.control inject bearing_wear 1
python -m src.control repair
python -m src.control inject cavitation 1.5
```

| Service   | URL                   | Credentials       |
|-----------|-----------------------|--------------------|
| Grafana   | http://localhost:3000  | admin / admin      |
| InfluxDB  | http://localhost:8086  | admin / adminpassword |
| Mosquitto | localhost:1883        | anonymous          |

---

## ML Methodology

### Feature Engineering

Each 30-second sliding window produces 14 features from 6 sensor signals:
- Per-signal mean and standard deviation (12 features)
- Linear trend slopes for vibration and temperature (2 features)

An extended 40+ feature set (including RMS, kurtosis, cross-sensor correlations) is available in `feature_engineering.py` for offline research experiments.

### Anomaly Detection — Isolation Forest

Trained exclusively on healthy operating data. The decision function score is calibrated between two empirically derived anchors (healthy median vs. severely faulty median) to produce a 0–100 health index. EMA smoothing (α = 0.2) is applied for temporal stability.

### Fault Classification — Random Forest

Trained on all five classes (healthy + four fault types) across 90 simulated episodes with randomized progression rates. An episode-based test split prevents data leakage between adjacent time windows.

### RUL Estimation

Linear regression on the last 90 health index samples, extrapolated to a failure threshold of 30. Returns "Stable" when no significant downward trend is detected.

### Alert Levels

| Health Index | Level    |
|-------------|----------|
| ≥ 70        | OK       |
| 40–69       | WARNING  |
| < 40        | CRITICAL |

---

## Fault Physics Model

The `PumpTwin` class implements physics-inspired sensor responses for four fault types:

| Fault Type        | Primary Sensor Effects               | Severity Rate |
|-------------------|--------------------------------------|--------------|
| Bearing wear      | Vibration ↑, Temperature ↑            | 0.004/s      |
| Imbalance         | Vibration ↑, RPM noise ↑              | 0.005/s      |
| Cavitation        | Flow ↓, Pressure ↓, Current noise ↑   | 0.005/s      |
| Cooling failure   | Temperature ↑↑                        | 0.004/s      |

Temperature uses a first-order thermal lag (τ ≈ 40s) to simulate realistic thermal inertia. Fault severities accumulate continuously from 0 to 1 over time.

---

## MQTT Topic Structure

| Topic                              | Direction         | Payload                        |
|------------------------------------|-------------------|--------------------------------|
| `factory/line1/pump01/telemetry`   | sim → analytics   | 6 sensor readings @ 1 Hz       |
| `factory/line1/pump01/truth`       | sim → analytics   | Ground truth fault & severity   |
| `factory/line1/pump01/health`      | analytics → all   | Health index, fault class, RUL  |
| `factory/line1/pump01/cmd`         | control → sim     | inject / repair commands        |
| `factory/line1/pump01/status`      | sim (LWT)         | online / offline (retained)     |

---

## Real-World Dataset Experiments

Beyond synthetic simulation, the project includes loaders and training pipelines for real industrial datasets:

- **SKAB** — Real industrial pump telemetry with labelled anomalies. Used to calibrate the Isolation Forest's anomaly score boundaries against actual sensor measurements.
- **AI4I 2020** — Synthetic predictive maintenance benchmark for multi-class fault classification evaluation.
- **NASA IMS** (optional) — Bearing degradation run-to-failure data.
- **Paderborn** (optional) — Bearing vibration data with artificial and real damage.

Run experiments with:
```powershell
python -m src.real_data_train --experiment all
```

Results (metrics, confusion matrices, distribution plots) are saved to `results/`.

---

## Grafana Dashboard

Five-row layout with full-width panels:

1. **Overall Condition** — Health gauge, alert status, detected fault class, confidence, RUL
2. **Health Intelligence** — ML health index vs. ground truth trend, fault state timeline
3. **Condition Indicators** — Vibration and temperature time series
4. **Process Variables** — RPM, current, flow, pressure
5. **All Sensors Combined** — Multi-signal overlay with legend

---

## Testing

```powershell
python -m pytest tests/ -v
```

Test coverage includes feature engineering correctness, model inference compatibility, data leakage prevention, and dataset loader validation.

---

## Deployment on Streamlit Cloud

To deploy the Streamlit dashboard publicly:

1. Push the repo to GitHub (ensure `models/twin_models.joblib` is committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select the repo, set main file to `streamlit_app.py`, and click Deploy

The app runs entirely in-process — no Docker, MQTT, or InfluxDB needed on the cloud.

---

## Technology Stack

| Layer              | Technology               |
|--------------------|--------------------------|
| Simulation         | Python, NumPy            |
| IoT Messaging      | MQTT (Paho 2.1.0)        |
| Broker             | Eclipse Mosquitto 2       |
| Anomaly Detection  | Isolation Forest (sklearn)|
| Fault Classification| Random Forest (sklearn)  |
| Time Series DB     | InfluxDB 2.7             |
| Visualization      | Grafana 11.2, Streamlit   |
| Containerization   | Docker Compose            |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope Process Bypass` |
| `ModuleNotFoundError: src` | Run from project root with `python -m src.xxx` |
| Connection refused on 1883 | Start Docker Desktop, then `docker compose up -d` |
| Paho callback signature error | `pip install paho-mqtt==2.1.0` |
| Grafana shows "no data" | Set time range to "Last 5m"; check InfluxDB has data |
| InfluxDB 401 | Token must be `twin-super-token`; recreate volumes if needed |
| Health stays ~100 during fault | Retrain: `python -m src.train` |
| Analytics crashes on startup | Train models first: `python -m src.train` |

---

## License

This project was developed as part of HCLTech's Industrial IoT & Digital Twin initiative.
