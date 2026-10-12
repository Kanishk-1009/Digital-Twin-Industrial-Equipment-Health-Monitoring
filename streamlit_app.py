"""
Digital Twin — Industrial Equipment Health Monitoring
=====================================================
Streamlit dashboard for real-time pump health monitoring.

Runs the PumpTwin physics simulator and ML inference engine entirely
in-process — no MQTT broker, InfluxDB, or Docker required.

Usage
-----
    streamlit run streamlit_app.py
"""

import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from collections import deque
from pathlib import Path

from src.asset_model import PumpTwin
from src.features import window_features
from src.config import SIGNALS, WINDOW, FAIL_THRESHOLD

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin — Pump Health Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.1));
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
        border: 1px solid rgba(100,116,139,0.2);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Alert badges */
    .alert-ok {
        color: #34d399;
        text-shadow: 0 0 20px rgba(52,211,153,0.3);
    }
    .alert-warning {
        color: #fbbf24;
        text-shadow: 0 0 20px rgba(251,191,36,0.3);
    }
    .alert-critical {
        color: #f87171;
        text-shadow: 0 0 20px rgba(248,113,113,0.3);
        animation: pulse-critical 1.5s ease-in-out infinite;
    }
    @keyframes pulse-critical {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Health gauge */
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
    }

    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(100,116,139,0.2);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(100,116,139,0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #f1f5f9;
        font-size: 1.3rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #cbd5e1;
        font-size: 1rem;
    }

    /* Charts */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* History table */
    .event-log {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(100,116,139,0.15);
        border-radius: 10px;
        padding: 12px 16px;
        max-height: 200px;
        overflow-y: auto;
        font-family: 'Inter', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Streamlit metric styling override */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
        border: 1px solid rgba(100,116,139,0.2);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 0.72rem !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load ML models (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load trained ML models from disk."""
    model_path = Path("models/twin_models.joblib")
    if not model_path.exists():
        return None
    m = joblib.load(model_path)
    return m


M = load_models()
if M is None:
    st.error("⚠️ Models not found. Please run `python -m src.train` first to train the ML models.")
    st.stop()

scaler, iso, clf = M["scaler"], M["iso"], M["clf"]
s_good, s_bad = M["s_good"], M["s_bad"]

# ── Session state initialisation ──────────────────────────────────────────────
if "twin" not in st.session_state:
    st.session_state.twin = PumpTwin(seed=42)
    st.session_state.buf = deque(maxlen=WINDOW)
    st.session_state.ema = None
    st.session_state.step_count = 0
    st.session_state.health_history = []
    st.session_state.health_true_history = []
    st.session_state.sensor_history = {s: [] for s in SIGNALS}
    st.session_state.ts_history = []
    st.session_state.fault_history = []
    st.session_state.alert_history = []
    st.session_state.rul_history = []
    st.session_state.event_log = []
    st.session_state.running = False
    st.session_state.active_fault = "healthy"
    st.session_state.speed = 1.0

MAX_HISTORY = 500  # keep last 500 data points for charts

twin = st.session_state.twin


# ── Helper functions ──────────────────────────────────────────────────────────

def alert_level(h: float) -> str:
    if h >= 70:
        return "OK"
    elif h >= 40:
        return "WARNING"
    else:
        return "CRITICAL"


def alert_color(level: str) -> str:
    return {"OK": "#34d399", "WARNING": "#fbbf24", "CRITICAL": "#f87171"}.get(level, "#94a3b8")


def alert_emoji(level: str) -> str:
    return {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level, "❓")


def health_color(h: float) -> str:
    if h >= 70:
        return "#34d399"
    elif h >= 40:
        return "#fbbf24"
    return "#f87171"


def estimate_rul(h_hist) -> float:
    if len(h_hist) < 45:
        return -1.0
    y = np.array(h_hist)
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    if slope > -0.05:
        return -1.0
    return float(max(0.0, (y[-10:].mean() - FAIL_THRESHOLD) / -slope))


def run_simulation_step():
    """Execute one step of the physics simulation + ML inference."""
    tel, truth = twin.step()
    st.session_state.step_count += 1

    # Update sensor buffer
    buf = st.session_state.buf
    buf.append([tel[s] for s in SIGNALS])

    # Store sensor history
    for s in SIGNALS:
        st.session_state.sensor_history[s].append(tel[s])
        if len(st.session_state.sensor_history[s]) > MAX_HISTORY:
            st.session_state.sensor_history[s] = st.session_state.sensor_history[s][-MAX_HISTORY:]

    st.session_state.ts_history.append(st.session_state.step_count)
    if len(st.session_state.ts_history) > MAX_HISTORY:
        st.session_state.ts_history = st.session_state.ts_history[-MAX_HISTORY:]

    # Store ground truth health
    st.session_state.health_true_history.append(truth["health_true"])
    if len(st.session_state.health_true_history) > MAX_HISTORY:
        st.session_state.health_true_history = st.session_state.health_true_history[-MAX_HISTORY:]

    # ML inference (only when window is full)
    if len(buf) >= WINDOW:
        x = window_features(buf).reshape(1, -1)

        # Health index from Isolation Forest
        score = float(iso.decision_function(scaler.transform(x))[0])
        raw = float(np.clip((score - s_bad) / (s_good - s_bad), 0, 1) * 100)
        st.session_state.ema = raw if st.session_state.ema is None else 0.8 * st.session_state.ema + 0.2 * raw
        health = st.session_state.ema

        # Fault classification from Random Forest
        proba = clf.predict_proba(x)[0]
        k = int(proba.argmax())
        fault_pred = str(clf.classes_[k])
        conf = float(proba[k])

        # RUL estimation
        st.session_state.health_history.append(health)
        if len(st.session_state.health_history) > MAX_HISTORY:
            st.session_state.health_history = st.session_state.health_history[-MAX_HISTORY:]

        rul = estimate_rul(st.session_state.health_history)

        # Alert level
        level = alert_level(health)

        # Store fault / alert / RUL history
        st.session_state.fault_history.append(fault_pred)
        st.session_state.alert_history.append(level)
        st.session_state.rul_history.append(rul)
        for lst in [st.session_state.fault_history, st.session_state.alert_history, st.session_state.rul_history]:
            if len(lst) > MAX_HISTORY:
                del lst[:-MAX_HISTORY]

        return {
            "health": health,
            "health_true": truth["health_true"],
            "score": score,
            "fault_pred": fault_pred,
            "confidence": conf,
            "rul": rul,
            "alert": level,
            "sensors": tel,
            "truth": truth,
        }
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏭 Digital Twin")
    st.markdown("**Pump01 — Motor-Pump Asset**")
    st.markdown("---")

    st.markdown("### ⚡ Simulation Control")
    sim_speed = st.select_slider(
        "Simulation Speed",
        options=[0.5, 1.0, 2.0, 5.0, 10.0],
        value=1.0,
        format_func=lambda x: f"{x}×",
    )
    st.session_state.speed = sim_speed

    steps_per_refresh = st.slider("Steps per Refresh", 1, 50, 5,
                                  help="Number of simulation steps between each UI update")

    col_start, col_stop = st.columns(2)
    with col_start:
        start = st.button("▶️ Start", use_container_width=True, type="primary")
    with col_stop:
        stop = st.button("⏹ Stop", use_container_width=True)

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    st.markdown("---")
    st.markdown("### 🔧 Fault Injection")

    fault_type = st.selectbox(
        "Fault Type",
        ["healthy", "bearing_wear", "imbalance", "cavitation", "cooling_failure"],
        format_func=lambda x: {
            "healthy": "🟢 Healthy (no fault)",
            "bearing_wear": "🔴 Bearing Wear",
            "imbalance": "🟠 Imbalance",
            "cavitation": "🟡 Cavitation",
            "cooling_failure": "🔵 Cooling Failure",
        }.get(x, x),
    )

    rate_mult = st.slider("Progression Rate", 0.5, 3.0, 1.0, 0.1,
                          help="Multiplier for fault severity accumulation speed")

    col_inject, col_repair = st.columns(2)
    with col_inject:
        inject = st.button("💉 Inject", use_container_width=True,
                          disabled=(fault_type == "healthy"), type="primary")
    with col_repair:
        repair = st.button("🔧 Repair", use_container_width=True)

    if inject and fault_type != "healthy":
        twin.set_fault(fault_type, rate_mult)
        st.session_state.active_fault = fault_type
        st.session_state.event_log.append(
            f"⚡ Injected **{fault_type}** (rate={rate_mult:.1f}×) at step {st.session_state.step_count}"
        )
        st.toast(f"Fault injected: {fault_type}", icon="💉")

    if repair:
        twin.repair()
        st.session_state.active_fault = "healthy"
        st.session_state.event_log.append(
            f"🔧 Repaired at step {st.session_state.step_count}"
        )
        st.toast("System repaired!", icon="🔧")

    # Show current fault state
    active = st.session_state.active_fault
    if active == "healthy":
        st.success("System: 🟢 Healthy")
    else:
        st.warning(f"Active Fault: 🔴 {active}")

    st.markdown("---")
    st.markdown("### 📊 Simulation Info")
    st.markdown(f"**Steps run:** {st.session_state.step_count}")
    st.markdown(f"**Window size:** {WINDOW}")
    st.markdown(f"**Buffer fill:** {len(st.session_state.buf)}/{WINDOW}")

    # Event log
    if st.session_state.event_log:
        st.markdown("---")
        st.markdown("### 📝 Event Log")
        for event in reversed(st.session_state.event_log[-10:]):
            st.markdown(f"- {event}")


# ── Main dashboard ────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏭 Digital Twin — Pump Health Monitor</h1>
    <p>Real-time ML-driven predictive maintenance · Isolation Forest + Random Forest · Physics-based simulation</p>
</div>
""", unsafe_allow_html=True)

# Run simulation steps
latest = None
if st.session_state.running:
    for _ in range(steps_per_refresh):
        result = run_simulation_step()
        if result is not None:
            latest = result

# Get latest state for display
if latest is None and st.session_state.health_history:
    health = st.session_state.health_history[-1]
    health_true = st.session_state.health_true_history[-1] if st.session_state.health_true_history else 100
    fault_pred = st.session_state.fault_history[-1] if st.session_state.fault_history else "healthy"
    alert = st.session_state.alert_history[-1] if st.session_state.alert_history else "OK"
    rul = st.session_state.rul_history[-1] if st.session_state.rul_history else -1.0
    latest = {
        "health": health,
        "health_true": health_true,
        "fault_pred": fault_pred,
        "confidence": 0.0,
        "rul": rul,
        "alert": alert,
    }

# ── Row 1: Key metrics ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Overall Condition</div>', unsafe_allow_html=True)

if latest:
    h = latest["health"]
    hc = health_color(h)
    ac = alert_color(latest["alert"])
    ae = alert_emoji(latest["alert"])
    rul_display = f'{latest["rul"]:.0f}s' if latest["rul"] >= 0 else "Stable"

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Health Index</div>
            <div class="metric-value" style="color: {hc};">{h:.1f}</div>
            <div class="metric-sub">out of 100</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Alert Status</div>
            <div class="metric-value alert-{latest['alert'].lower()}">{ae} {latest['alert']}</div>
            <div class="metric-sub">threshold: 70 / 40</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        fault_emoji = {"healthy": "🟢", "bearing_wear": "🔴", "imbalance": "🟠",
                       "cavitation": "🟡", "cooling_failure": "🔵"}.get(latest["fault_pred"], "❓")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detected Fault</div>
            <div class="metric-value" style="color: #e2e8f0; font-size: 1.3rem;">{fault_emoji} {latest['fault_pred'].replace('_', ' ').title()}</div>
            <div class="metric-sub">ML classification</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        conf = latest.get("confidence", 0)
        conf_color = "#34d399" if conf >= 0.8 else ("#fbbf24" if conf >= 0.5 else "#f87171")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value" style="color: {conf_color};">{conf:.1%}</div>
            <div class="metric-sub">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        rul_color = "#34d399" if latest["rul"] < 0 else ("#fbbf24" if latest["rul"] > 60 else "#f87171")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RUL (Remaining)</div>
            <div class="metric-value" style="color: {rul_color};">{rul_display}</div>
            <div class="metric-sub">linear extrapolation</div>
        </div>
        """, unsafe_allow_html=True)

    # Second row: ground truth comparison
    col_gt1, col_gt2, col_gt3 = st.columns(3)
    with col_gt1:
        ht = latest.get("health_true", 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ground Truth Health</div>
            <div class="metric-value" style="color: {health_color(ht)};">{ht:.1f}</div>
            <div class="metric-sub">physics model truth</div>
        </div>
        """, unsafe_allow_html=True)
    with col_gt2:
        err = abs(h - ht)
        err_color = "#34d399" if err < 10 else ("#fbbf24" if err < 25 else "#f87171")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimation Error</div>
            <div class="metric-value" style="color: {err_color};">{err:.1f}</div>
            <div class="metric-sub">|ML − truth|</div>
        </div>
        """, unsafe_allow_html=True)
    with col_gt3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulation Step</div>
            <div class="metric-value" style="color: #60a5fa;">{st.session_state.step_count}</div>
            <div class="metric-sub">@ {st.session_state.speed}× speed</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👆 Click **Start** in the sidebar to begin the simulation.")

# ── Row 2: Health trend chart ─────────────────────────────────────────────────
if st.session_state.health_history:
    st.markdown('<div class="section-header">📈 Health Intelligence</div>', unsafe_allow_html=True)

    col_trend, col_fault = st.columns([3, 2])

    with col_trend:
        n_pts = len(st.session_state.health_history)
        ts_start = len(st.session_state.ts_history) - n_pts
        ts_for_health = st.session_state.ts_history[ts_start:] if ts_start >= 0 else list(range(n_pts))

        health_df = pd.DataFrame({
            "Step": ts_for_health[:n_pts],
            "ML Health Index": st.session_state.health_history[:n_pts],
            "Ground Truth": st.session_state.health_true_history[-n_pts:] if st.session_state.health_true_history else [100]*n_pts,
        })
        st.line_chart(
            health_df.set_index("Step"),
            color=["#60a5fa", "#34d399"],
            use_container_width=True,
            height=300,
        )

    with col_fault:
        if st.session_state.fault_history:
            # Show fault distribution as a simple summary
            fault_counts = pd.Series(st.session_state.fault_history[-100:]).value_counts()
            st.markdown("**Fault Classification (Last 100 steps)**")
            st.bar_chart(fault_counts, color="#a78bfa", height=260)

# ── Row 3: Condition Indicators ───────────────────────────────────────────────
if st.session_state.sensor_history["vibration_mms"]:
    st.markdown('<div class="section-header">🔍 Condition Indicators</div>', unsafe_allow_html=True)

    col_vib, col_temp = st.columns(2)

    with col_vib:
        n_vib = len(st.session_state.sensor_history["vibration_mms"])
        vib_df = pd.DataFrame({
            "Step": st.session_state.ts_history[-n_vib:],
            "Vibration (mm/s)": st.session_state.sensor_history["vibration_mms"][-n_vib:],
        })
        st.line_chart(vib_df.set_index("Step"), color=["#f97316"], height=220, use_container_width=True)

    with col_temp:
        n_temp = len(st.session_state.sensor_history["temp_c"])
        temp_df = pd.DataFrame({
            "Step": st.session_state.ts_history[-n_temp:],
            "Temperature (°C)": st.session_state.sensor_history["temp_c"][-n_temp:],
        })
        st.line_chart(temp_df.set_index("Step"), color=["#ef4444"], height=220, use_container_width=True)

# ── Row 4: Process Variables ──────────────────────────────────────────────────
if st.session_state.sensor_history["rpm"]:
    st.markdown('<div class="section-header">⚙️ Process Variables</div>', unsafe_allow_html=True)

    col_rpm, col_cur, col_flow, col_press = st.columns(4)

    process_signals = [
        ("rpm", "RPM", "#3b82f6", col_rpm),
        ("current_a", "Current (A)", "#8b5cf6", col_cur),
        ("flow_m3h", "Flow (m³/h)", "#06b6d4", col_flow),
        ("pressure_bar", "Pressure (bar)", "#10b981", col_press),
    ]

    for sig, label, color, col in process_signals:
        with col:
            n_sig = len(st.session_state.sensor_history[sig])
            sig_df = pd.DataFrame({
                "Step": st.session_state.ts_history[-n_sig:],
                label: st.session_state.sensor_history[sig][-n_sig:],
            })
            st.line_chart(sig_df.set_index("Step"), color=[color], height=180, use_container_width=True)

# ── Row 5: All sensors combined ───────────────────────────────────────────────
if st.session_state.sensor_history["rpm"] and len(st.session_state.sensor_history["rpm"]) > 10:
    st.markdown('<div class="section-header">📡 All Sensors Combined</div>', unsafe_allow_html=True)

    # Normalise all signals to 0-1 range for combined view
    combined = {}
    min_len = min(len(st.session_state.sensor_history[s]) for s in SIGNALS)
    for s in SIGNALS:
        vals = np.array(st.session_state.sensor_history[s][-min_len:])
        v_min, v_max = vals.min(), vals.max()
        span = v_max - v_min if v_max > v_min else 1.0
        combined[s] = ((vals - v_min) / span).tolist()

    combined["Step"] = st.session_state.ts_history[-min_len:]
    combined_df = pd.DataFrame(combined).set_index("Step")
    st.line_chart(combined_df, height=300, use_container_width=True)


# ── Auto-refresh for continuous simulation ────────────────────────────────────
if st.session_state.running:
    delay = max(0.05, 1.0 / st.session_state.speed / 10)
    time.sleep(delay)
    st.rerun()
