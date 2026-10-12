# MQTT & InfluxDB
ASSET_ID = "pump01"
BASE = "factory/line1/pump01"
TOPIC_TELEMETRY = f"{BASE}/telemetry"
TOPIC_TRUTH     = f"{BASE}/truth"
TOPIC_HEALTH    = f"{BASE}/health"
TOPIC_CMD       = f"{BASE}/cmd"
TOPIC_STATUS    = f"{BASE}/status"

MQTT_HOST, MQTT_PORT = "localhost", 1883
INFLUX_URL, INFLUX_TOKEN = "http://localhost:8086", "twin-super-token"
INFLUX_ORG, INFLUX_BUCKET = "hcl", "twin"

# Sensor signals and fault classes
SIGNALS = ["rpm", "current_a", "flow_m3h", "pressure_bar", "vibration_mms", "temp_c"]
FAULTS  = ["healthy", "bearing_wear", "imbalance", "cavitation", "cooling_failure"]

# ML sliding window
WINDOW  = 30           # samples per feature window (seconds at 1 Hz)
FAIL_THRESHOLD = 30    # health index considered "failed" for RUL

# operational thresholds (project-defined, not industry standards)
HEALTH_NORMAL   = 80   # >= 80 : normal
HEALTH_WARNING  = 60   # 60-80 : warning
HEALTH_DEGRADED = 40   # 40-60 : degraded
# < 40 : critical

MODEL_PATH = "models/twin_models.joblib"

# Real-world dataset paths (relative to project root)
DATA_ROOT          = "datasets"
RAW_ROOT           = f"{DATA_ROOT}/raw"
PROCESSED_ROOT     = f"{DATA_ROOT}/processed"
METADATA_ROOT      = f"{DATA_ROOT}/metadata"

SKAB_RAW_DIR       = f"{RAW_ROOT}/skab"
AI4I_RAW_DIR       = f"{RAW_ROOT}/ai4i"
NASA_IMS_RAW_DIR   = f"{RAW_ROOT}/nasa_ims"    # optional, large
PADERBORN_RAW_DIR  = f"{RAW_ROOT}/paderborn"   # optional, large

SKAB_FEATURES      = f"{PROCESSED_ROOT}/skab_features.csv"
AI4I_FEATURES      = f"{PROCESSED_ROOT}/ai4i_features.csv"
NASA_IMS_FEATURES  = f"{PROCESSED_ROOT}/nasa_ims_features.csv"
PADERBORN_FEATURES = f"{PROCESSED_ROOT}/paderborn_features.csv"

RESULTS_ROOT       = "results"

# Rolling window sizes (based on each dataset's sampling rate)
SKAB_WINDOW        = 30     # ~1 Hz, 30-sample window = 30 s
AI4I_WINDOW        = 1      # already aggregated per operation cycle
NASA_IMS_WINDOW    = 2048   # ~20 kHz raw data, 2048 samples ≈ 0.1 s

# Optional large datasets (set True when data is downloaded)
NASA_IMS_AVAILABLE  = False
PADERBORN_AVAILABLE = False
