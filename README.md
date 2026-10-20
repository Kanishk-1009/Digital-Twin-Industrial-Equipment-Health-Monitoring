# Digital Twin for Industrial Equipment Health Monitoring

> An ML-powered industrial digital twin for equipment health monitoring, anomaly detection, fault classification, and predictive maintenance.

**🚀 Live Demo:** [Launch the Streamlit Dashboard](https://digital-twin-industrial-equipment-health-monitoring-bthmf3oxh6.streamlit.app/)


## Overview

This project implements a digital twin of an industrial motor-pump system to demonstrate predictive maintenance using simulated sensor telemetry and machine learning. The system simulates equipment behavior, injects faults, detects anomalies, classifies fault types, estimates Remaining Useful Life (RUL), and visualizes equipment health.

The project supports two deployment modes:

- **Standalone Streamlit Dashboard:** Interactive pump simulation and ML inference without requiring Docker, MQTT, or InfluxDB.
- **Full-Stack Monitoring:** MQTT-based telemetry, machine-learning analytics, InfluxDB time-series storage, and Grafana dashboards using Docker Compose.

The ML evaluation workflow also uses two datasets: **SKAB** and **AI4I 2020**.

## Live Demo

Try the deployed application:

**[Digital Twin — Industrial Equipment Health Monitoring](https://digital-twin-industrial-equipment-health-monitoring-bthmf3oxh6.streamlit.app/)**

The hosted Streamlit dashboard runs its simulation and ML inference in-process. It is separate from the MQTT, InfluxDB, and Grafana pipeline unless an explicit integration is configured.

## Key Features

- Physics-inspired digital twin of an industrial pump (`pump01`).
- Simulated sensor telemetry for equipment monitoring.
- Four fault scenarios: bearing wear, imbalance, cavitation, and cooling failure.
- Isolation Forest-based anomaly detection.
- Random Forest-based fault classification.
- Health Index estimation on a 0–100 scale.
- Trend-based Remaining Useful Life estimation.
- Interactive Streamlit dashboard with fault injection and sensor visualizations.
- Grafana dashboards for time-series and equipment-health monitoring.
- Offline ML evaluation using SKAB and AI4I 2020 datasets.
- Automated tests for feature engineering, inference compatibility, data leakage, and dataset loaders.
- Docker Compose-based infrastructure for the full-stack monitoring mode.

## System Architecture

### Standalone Streamlit Mode

```text
Streamlit Dashboard
        |
        v
PumpTwin Simulation
        |
        v
Feature Engineering
        |
        v
Isolation Forest + Random Forest
        |
        v
Health Index, Fault Classification, RUL
        |
        v
Interactive Dashboard and Sensor Charts
```

### Full-Stack Monitoring Mode

```text
Python Digital Twin Simulator
        |
        | MQTT Telemetry + Ground Truth
        v
Eclipse Mosquitto
        |
        v
ML Analytics Engine
        |
        +---------------------+
        |                     |
        v                     v
     InfluxDB            MQTT Health Topic
        |
        v
     Grafana
```

The two modes serve different purposes. Streamlit provides a convenient standalone demonstration, while the full-stack deployment demonstrates MQTT communication, time-series persistence, and Grafana-based monitoring.

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python |
| Simulation and data processing | NumPy, Pandas |
| IoT communication | MQTT, Paho MQTT |
| MQTT broker | Eclipse Mosquitto |
| Anomaly detection | Scikit-learn, Isolation Forest |
| Fault classification | Scikit-learn, Random Forest |
| Time-series database | InfluxDB 2.x |
| Visualization | Grafana, Streamlit |
| Containerization | Docker, Docker Compose |
| Model persistence | Joblib |
| Testing | Pytest |
| Dataset evaluation | SKAB, AI4I 2020 |
| Version control | Git, GitHub |

## Fault Simulation Model

The digital twin simulates four equipment fault types.

| Fault Type | Simulated Sensor Effects |
|---|---|
| Bearing Wear | Increased vibration and temperature |
| Imbalance | Increased vibration and RPM variation |
| Cavitation | Reduced flow and pressure, increased current noise |
| Cooling Failure | Increased temperature |

The `PumpTwin` model implements physics-inspired relationships between fault severity and sensor behavior. Temperature includes a simplified thermal-lag effect to represent delayed temperature response.

The model is intended for simulation and experimentation rather than as a validated physical model of a specific commercial pump.

## Machine Learning Methodology

### 1. Feature Engineering

The core inference pipeline uses a sliding window of sensor observations. The features include:

- Mean and standard deviation for sensor signals.
- Vibration trend slope.
- Temperature trend slope.

An extended feature-engineering module supports additional features for offline research experiments.

### 2. Anomaly Detection — Isolation Forest

Isolation Forest is trained on healthy simulated operating data. Its anomaly scores are calibrated using healthy and faulty score distributions to derive a Health Index between 0 and 100.

Exponential moving average smoothing helps reduce short-term fluctuations in the health trend.

### 3. Fault Classification — Random Forest

The Random Forest classifier identifies five operating classes:

- Healthy
- Bearing Wear
- Imbalance
- Cavitation
- Cooling Failure

The training workflow uses episode-based splitting to reduce data leakage between adjacent time windows.

### 4. Remaining Useful Life Estimation

RUL is estimated by extrapolating the recent Health Index trend toward a configured failure threshold.

This is a simplified trend-based estimate and should not be interpreted as a guaranteed prediction of actual equipment lifetime.

### 5. Health Status

| Health Index | Status |
|---|---|
| 70–100 | OK |
| 40–69 | WARNING |
| Below 40 | CRITICAL |

These thresholds are configured for the project and may require recalibration for real industrial equipment.

## Repository Structure

```text
Digital-Twin-Industrial-Equipment-Health-Monitoring/
├── docker-compose.yml
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── mosquitto/
│   └── mosquitto.conf
├── grafana/
│   ├── provisioning/
│   └── dashboards/
├── src/
│   ├── config.py
│   ├── asset_model.py
│   ├── features.py
│   ├── feature_engineering.py
│   ├── generate_dataset.py
│   ├── train.py
│   ├── real_data_train.py
│   ├── simulator.py
│   ├── analytics.py
│   ├── control.py
│   └── datasets/
├── models/
├── data/
├── datasets/
├── results/
└── tests/
```

The exact contents of `data/`, `datasets/`, `models/`, and `results/` depend on the files included in the repository. Credentials, confidential material, and files that cannot legally be redistributed should not be committed.

## Getting Started

### Prerequisites

- Python 3.11 or newer
- Git
- Docker Desktop, required only for the full-stack monitoring configuration

### Installation

Clone the repository:

```powershell
git clone https://github.com/Kanishk-1009/Digital-Twin-Industrial-Equipment-Health-Monitoring.git
cd Digital-Twin-Industrial-Equipment-Health-Monitoring
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Streamlit Dashboard

Start the standalone dashboard:

```powershell
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Use the dashboard controls to run the simulation, inject faults, and observe equipment-health indicators and sensor charts.

If the application requires a model artifact that is missing, train the model using the project's training command before launching the dashboard.

## Running the Full Stack with Docker

Start Docker Desktop, then run:

```powershell
docker compose up -d
```

Train or regenerate the machine-learning models if required:

```powershell
python -m src.train
```

Start the analytics service in one terminal:

```powershell
python -m src.analytics
```

Start the simulator in another terminal:

```powershell
python -m src.simulator
```

Use a third terminal to inject and repair faults:

```powershell
python -m src.control inject bearing_wear 1
python -m src.control repair
python -m src.control inject cavitation 1.5
```

<<<<<<< HEAD
| Service   | URL                   | Credentials       |
|-----------|-----------------------|--------------------|
| Grafana   | http://localhost:3000  | admin / admin      |
| InfluxDB  | http://localhost:8086  | admin / adminpassword |
| Mosquitto | http://localhost:1883  | anonymous          |
=======
### Service Endpoints
>>>>>>> 66d12f7 (Upadated website design)

| Service | Local Address |
|---|---|
| Grafana | [http://localhost:3000](http://localhost:3000) |
| InfluxDB | [http://localhost:8086](http://localhost:8086) |
| MQTT Broker | `localhost:1883` |

Use the credentials and tokens configured in your local environment. Change default credentials before exposing services outside your machine, and never commit real secrets to GitHub.

## MQTT Topic Structure

| Topic | Direction | Purpose |
|---|---|---|
| `factory/line1/pump01/telemetry` | Simulator → Analytics | Sensor telemetry |
| `factory/line1/pump01/truth` | Simulator → Analytics | Simulated fault ground truth |
| `factory/line1/pump01/health` | Analytics → Subscribers | Health Index, fault class, and RUL |
| `factory/line1/pump01/cmd` | Controller → Simulator | Fault injection and repair commands |
| `factory/line1/pump01/status` | Simulator | Online/offline status |

## Dataset Experiments

The project uses two datasets for offline machine-learning experiments.

### 1. SKAB — Skoltech Anomaly Benchmark

SKAB contains industrial sensor time-series data with anomaly labels. It is used in the project's anomaly-detection experiments to evaluate how the pipeline handles industrial sensor patterns and anomalous observations.

### 2. AI4I 2020 Predictive Maintenance Dataset

AI4I 2020 is a predictive-maintenance benchmark used for fault-related classification experiments and evaluation.

### Run the Experiments

Execute the experiment pipeline with:

```powershell
python -m src.real_data_train --experiment all
```

Generated metrics, plots, and other experiment outputs are saved under `results/` when produced by the experiment scripts.

**Important distinction:** SKAB and AI4I are used for offline dataset experiments. They are not live physical sensor readings from the simulated pump. The standalone Streamlit simulation and the MQTT-based monitoring pipeline are separate from these offline experiments unless an explicit integration is implemented.

Review each dataset's license and usage conditions before redistributing raw data or processed derivatives.

## Grafana Dashboard

The full-stack Grafana dashboard is designed to display:

1. **Overall Condition:** Health Index, alert status, fault classification, confidence, and RUL.
2. **Health Intelligence:** ML health trends and simulated ground-truth fault timelines.
3. **Condition Indicators:** Vibration and temperature.
4. **Process Variables:** RPM, current, flow, and pressure.
5. **Combined Sensor Trends:** Multiple sensor signals for comparison.

For the panels to populate, Docker services must be running and the simulator and analytics service must be publishing and consuming telemetry.

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest tests/ -v
```

The tests cover selected feature-engineering behavior, model inference compatibility, data-leakage checks, and dataset-loader functionality.

## Deployment on Streamlit Community Cloud

The dashboard is deployed on Streamlit Community Cloud.

**Live application:** [Open the deployed dashboard](https://digital-twin-industrial-equipment-health-monitoring-bthmf3oxh6.streamlit.app/)

To deploy your own version:

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and create a new app.
4. Select this repository and the `main` branch.
5. Set the main file path to `streamlit_app.py`.
6. Deploy the application.

Ensure `requirements.txt` contains the packages imported by the Streamlit app and that all required model artifacts and application files are available in the repository.

The hosted standalone app does not require a locally running Docker, MQTT broker, or InfluxDB service.

## Troubleshooting

| Issue | Suggested Solution |
|---|---|
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process Bypass` in the current session. |
| `ModuleNotFoundError` for `src` | Run commands from the repository root using `python -m src.module_name`. |
| MQTT connection refused | Start Docker Desktop and run `docker compose up -d`. |
| Grafana shows no data | Verify the simulator and analytics service are running and select a recent dashboard time range. |
| InfluxDB authentication error | Check the local Docker Compose configuration and configured token. |
| Streamlit deployment fails | Review Streamlit Cloud logs, verify `requirements.txt`, and ensure required files are committed. |

## Limitations

- The pump is simulated; the system is not connected to physical industrial sensors by default.
- Fault behavior and degradation rates are simplified.
- Health Index calibration and RUL estimates depend on the data, model, and configured thresholds.
- The Streamlit dashboard and MQTT/InfluxDB/Grafana stack are separate deployment modes unless explicitly integrated.
- SKAB and AI4I experiments provide dataset-based evaluation but do not, by themselves, establish deployment readiness for real industrial equipment.

## Future Scope

Potential extensions include:

- Multi-asset fleet monitoring.
- MQTT authentication and TLS.
- Improved RUL prediction models.
- REST API integration.
- Advanced alerting and notifications.
- Validation using additional industrial datasets and physical sensor measurements.

## License and Project Use

No open-source license is specified in this repository. Add a license only after confirming that you have the right to publish the project and selecting the appropriate terms.

If this project was developed as part of an organizational or academic initiative, verify code ownership, disclosure requirements, and dataset redistribution permissions before public release.
