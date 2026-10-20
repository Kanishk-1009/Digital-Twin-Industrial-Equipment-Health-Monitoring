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

# Page config
st.set_page_config(
    page_title="Digital Twin — Pump01 Health Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design System CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Reset & Tokens */
    :root {
        --bg-base:        #0d1117;
        --bg-panel:       #161b22;
        --bg-panel-alt:   #1c2128;
        --bg-hover:       #21262d;
        --border-default: #30363d;
        --border-muted:   #21262d;
        --border-strong:  #484f58;
        --text-primary:   #e6edf3;
        --text-secondary: #8b949e;
        --text-muted:     #6e7681;
        --status-ok:       #3fb950;
        --status-ok-dim:   rgba(63, 185, 80, 0.12);
        --status-warn:     #d29922;
        --status-warn-dim: rgba(210, 153, 34, 0.12);
        --status-crit:     #f85149;
        --status-crit-dim: rgba(248, 81, 73, 0.12);
        --accent:          #388bfd;
        --accent-dim:      rgba(56, 139, 253, 0.12);
        --font-ui:   'IBM Plex Sans', system-ui, sans-serif;
        --font-data: 'JetBrains Mono', 'Cascadia Code', monospace;
        --r-sm:   4px;
        --r-base: 6px;
        --r-lg:   8px;
    }

    html, body, .stApp {
        font-family: var(--font-ui);
        background-color: var(--bg-base) !important;
        color: var(--text-primary);
    }
    #MainMenu, footer, header { visibility: hidden; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

    .block-container {
        padding: 1.25rem 1.75rem 2rem !important;
        max-width: 100% !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-default) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    .sidebar-brand {
        padding: 0 1rem 1rem;
        border-bottom: 1px solid var(--border-muted);
        margin-bottom: 1rem;
    }
    .sidebar-brand .asset-id {
        font-family: var(--font-data);
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .sidebar-brand .asset-name {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-top: 6px;
        padding: 3px 8px;
        border-radius: var(--r-sm);
        font-family: var(--font-data);
        font-size: 0.68rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .status-pill.ok   { background: var(--status-ok-dim);   color: var(--status-ok);   border: 1px solid rgba(63,185,80,0.25); }
    .status-pill.warn { background: var(--status-warn-dim); color: var(--status-warn); border: 1px solid rgba(210,153,34,0.25); }
    .status-pill.crit { background: var(--status-crit-dim); color: var(--status-crit); border: 1px solid rgba(248,81,73,0.25); }
    .status-pill.idle { background: var(--accent-dim);      color: var(--accent);      border: 1px solid rgba(56,139,253,0.25); }

    .sidebar-section-label {
        font-size: 0.63rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        padding: 0 1rem 0.35rem;
        margin-top: 0.75rem;
    }

    /* Status bar */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.55rem 1rem;
        background: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--r-base);
        margin-bottom: 1.25rem;
        flex-wrap: wrap;
    }
    .status-bar-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.76rem;
        color: var(--text-secondary);
    }
    .status-bar-item strong {
        font-family: var(--font-data);
        font-weight: 500;
        color: var(--text-primary);
        font-size: 0.78rem;
    }
    .status-bar-sep { width: 1px; height: 14px; background: var(--border-default); }

    .live-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--status-ok);
        flex-shrink: 0;
    }
    .live-dot.inactive { background: var(--text-muted); }
    .live-dot.running  { animation: live-pulse 1.8s ease-in-out infinite; }
    @keyframes live-pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(63,185,80,0.4); }
        50%       { opacity: 0.7; box-shadow: 0 0 0 4px rgba(63,185,80,0); }
    }
    @media (prefers-reduced-motion: reduce) { .live-dot.running { animation: none; } }

    /* Health panel */
    .health-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        padding: 1.25rem 1.5rem;
    }
    .health-panel-label {
        font-size: 0.63rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .health-readout {
        font-family: var(--font-data);
        font-weight: 700;
        font-size: 3.5rem;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .health-readout.ok   { color: var(--status-ok); }
    .health-readout.warn { color: var(--status-warn); }
    .health-readout.crit { color: var(--status-crit); }
    .health-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
        font-family: var(--font-data);
    }
    .health-bar-track {
        height: 4px;
        background: var(--bg-panel-alt);
        border-radius: 2px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .health-bar-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; }
    .health-bar-fill.ok   { background: var(--status-ok); }
    .health-bar-fill.warn { background: var(--status-warn); }
    .health-bar-fill.crit { background: var(--status-crit); }

    /* KPI tiles */
    .kpi-tile {
        background: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        padding: 0.85rem 1rem;
    }
    .kpi-tile-label {
        font-size: 0.61rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .kpi-tile-value {
        font-family: var(--font-data);
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.15;
        color: var(--text-primary);
    }
    .kpi-tile-value.ok    { color: var(--status-ok); }
    .kpi-tile-value.warn  { color: var(--status-warn); }
    .kpi-tile-value.crit  { color: var(--status-crit); }
    .kpi-tile-value.accent { color: var(--accent); }
    .kpi-tile-meta {
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
        font-family: var(--font-data);
    }
    .kpi-tile-value.fault-name {
        font-size: 1rem;
        text-transform: capitalize;
        letter-spacing: 0.01em;
    }

    /* Section dividers */
    .section-label {
        font-size: 0.63rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin: 1.4rem 0 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border-muted);
    }

    /* Sensor readout row */
    .sensor-row {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.55rem;
        margin-bottom: 0.75rem;
    }
    .sensor-cell {
        background: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--r-base);
        padding: 0.55rem 0.7rem;
    }
    .sensor-cell-label {
        font-size: 0.58rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.18rem;
    }
    .sensor-cell-value {
        font-family: var(--font-data);
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
    }
    .sensor-cell-unit {
        font-size: 0.62rem;
        color: var(--text-muted);
        margin-left: 2px;
    }

    /* Alert banner */
    .alert-banner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 1rem;
        border-radius: var(--r-base);
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1rem;
        border-left: 3px solid;
    }
    .alert-banner.warn { background: var(--status-warn-dim); color: var(--status-warn); border-color: var(--status-warn); }
    .alert-banner.crit { background: var(--status-crit-dim); color: var(--status-crit); border-color: var(--status-crit); }

    /* Event log */
    .event-log-entry {
        display: flex;
        gap: 0.75rem;
        padding: 0.38rem 0;
        border-bottom: 1px solid var(--border-muted);
        font-size: 0.72rem;
    }
    .event-log-entry:last-child { border-bottom: none; }
    .event-log-step {
        font-family: var(--font-data);
        font-size: 0.65rem;
        color: var(--text-muted);
        white-space: nowrap;
        min-width: 44px;
    }
    .event-log-msg { color: var(--text-secondary); }

    /* Idle state */
    .idle-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3.5rem 2rem;
        background: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        text-align: center;
    }
    .idle-state-icon { font-size: 1.75rem; margin-bottom: 0.75rem; opacity: 0.35; }
    .idle-state-title { font-weight: 600; font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.3rem; }
    .idle-state-sub { font-size: 0.78rem; color: var(--text-muted); max-width: 420px; }

    /* Streamlit widget overrides */
    [data-testid="stMetric"] {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        backdrop-filter: none !important;
    }
    .stButton > button {
        background: var(--bg-panel-alt) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--r-base) !important;
        font-family: var(--font-ui) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.85rem !important;
        transition: background 0.15s, border-color 0.15s !important;
    }
    .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: var(--border-strong) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #fff !important;
        border-color: var(--accent) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1f6feb !important;
        border-color: #1f6feb !important;
    }
    .stSelectbox label, .stSlider label, .stSelectSlider label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: var(--text-muted) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--bg-panel-alt) !important;
        border-color: var(--border-default) !important;
        border-radius: var(--r-base) !important;
    }
    .stPlotlyChart { border-radius: var(--r-lg); overflow: hidden; }
    [data-testid="stArrowVegaLiteChart"] { border-radius: var(--r-lg); overflow: hidden; }
    hr { border-color: var(--border-muted) !important; margin: 0.65rem 0 !important; }
    [data-testid="stToast"] {
        background: var(--bg-panel-alt) !important;
        border: 1px solid var(--border-default) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stAlert"] { border-radius: var(--r-base) !important; }
</style>
""", unsafe_allow_html=True)


# Load ML models (cached)
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
    st.error("Models not found. Run `python -m src.train` to train the ML models.")
    st.stop()

scaler, iso, clf = M["scaler"], M["iso"], M["clf"]
s_good, s_bad = M["s_good"], M["s_bad"]

# Session state initialisation
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

MAX_HISTORY = 500
twin = st.session_state.twin


# Helper functions
def alert_level(h: float) -> str:
    if h >= 70:
        return "OK"
    elif h >= 40:
        return "WARNING"
    else:
        return "CRITICAL"


def alert_css_class(level: str) -> str:
    return {"OK": "ok", "WARNING": "warn", "CRITICAL": "crit"}.get(level, "ok")


def health_css_class(h: float) -> str:
    if h >= 70:
        return "ok"
    elif h >= 40:
        return "warn"
    return "crit"


def estimate_rul(h_hist) -> float:
    if len(h_hist) < 45:
        return -1.0
    y = np.array(h_hist)
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    if slope > -0.05:
        return -1.0
    return float(max(0.0, (y[-10:].mean() - FAIL_THRESHOLD) / -slope))


def format_fault(fault: str) -> str:
    return fault.replace("_", " ").title() if fault != "healthy" else "Healthy"


def run_simulation_step():
    """Execute one step of the physics simulation + ML inference."""
    tel, truth = twin.step()
    st.session_state.step_count += 1

    buf = st.session_state.buf
    buf.append([tel[s] for s in SIGNALS])

    for s in SIGNALS:
        st.session_state.sensor_history[s].append(tel[s])
        if len(st.session_state.sensor_history[s]) > MAX_HISTORY:
            st.session_state.sensor_history[s] = st.session_state.sensor_history[s][-MAX_HISTORY:]

    st.session_state.ts_history.append(st.session_state.step_count)
    if len(st.session_state.ts_history) > MAX_HISTORY:
        st.session_state.ts_history = st.session_state.ts_history[-MAX_HISTORY:]

    st.session_state.health_true_history.append(truth["health_true"])
    if len(st.session_state.health_true_history) > MAX_HISTORY:
        st.session_state.health_true_history = st.session_state.health_true_history[-MAX_HISTORY:]

    if len(buf) >= WINDOW:
        x = window_features(buf).reshape(1, -1)

        score = float(iso.decision_function(scaler.transform(x))[0])
        raw = float(np.clip((score - s_bad) / (s_good - s_bad), 0, 1) * 100)
        st.session_state.ema = raw if st.session_state.ema is None else 0.8 * st.session_state.ema + 0.2 * raw
        health = st.session_state.ema

        proba = clf.predict_proba(x)[0]
        k = int(proba.argmax())
        fault_pred = str(clf.classes_[k])
        conf = float(proba[k])

        st.session_state.health_history.append(health)
        if len(st.session_state.health_history) > MAX_HISTORY:
            st.session_state.health_history = st.session_state.health_history[-MAX_HISTORY:]

        rul = estimate_rul(st.session_state.health_history)
        level = alert_level(health)

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


# Fault label map (used in sidebar and status bar)
FAULT_LABELS = {
    "healthy":         "No Fault",
    "bearing_wear":    "Bearing Wear",
    "imbalance":       "Rotor Imbalance",
    "cavitation":      "Cavitation",
    "cooling_failure": "Cooling Failure",
}

# Sidebar
with st.sidebar:
    active_fault = st.session_state.active_fault
    running = st.session_state.running

    if not running:
        pill_class, pill_label = "idle", "Idle"
    elif active_fault != "healthy":
        pill_class, pill_label = "crit", "Fault Active"
    else:
        pill_class, pill_label = "ok", "Running"

    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="asset-id">ASSET / PUMP01</div>
        <div class="asset-name">Motor-Pump Unit</div>
        <div class="status-pill {pill_class}">{pill_label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Simulation</div>', unsafe_allow_html=True)

    sim_speed = st.select_slider(
        "Speed",
        options=[0.5, 1.0, 2.0, 5.0, 10.0],
        value=st.session_state.get("speed", 1.0),
        format_func=lambda x: f"{x}x",
    )
    st.session_state.speed = sim_speed

    steps_per_refresh = st.slider(
        "Steps per Refresh", 1, 50, 5,
        help="Simulation steps computed between each UI update"
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        start = st.button("Start", use_container_width=True, type="primary")
    with col_stop:
        stop = st.button("Stop", use_container_width=True)

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Fault Injection</div>', unsafe_allow_html=True)

    fault_type = st.selectbox(
        "Fault Type",
        list(FAULT_LABELS.keys()),
        format_func=lambda x: FAULT_LABELS[x],
    )

    rate_mult = st.slider(
        "Progression Rate", 0.5, 3.0, 1.0, 0.1,
        help="Fault severity accumulation multiplier"
    )

    col_inject, col_repair = st.columns(2)
    with col_inject:
        inject = st.button("Inject", use_container_width=True,
                           disabled=(fault_type == "healthy"), type="primary")
    with col_repair:
        repair = st.button("Repair", use_container_width=True)

    if inject and fault_type != "healthy":
        twin.set_fault(fault_type, rate_mult)
        st.session_state.active_fault = fault_type
        st.session_state.event_log.append({
            "step": st.session_state.step_count,
            "msg": f"Injected {FAULT_LABELS[fault_type]} (rate {rate_mult:.1f}x)"
        })
        st.toast(f"Fault injected: {FAULT_LABELS[fault_type]}", icon="!")

    if repair:
        twin.repair()
        st.session_state.active_fault = "healthy"
        st.session_state.event_log.append({
            "step": st.session_state.step_count,
            "msg": "System repaired — fault cleared"
        })
        st.toast("Repair complete")

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Telemetry</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding: 0 1rem; font-family: var(--font-data); font-size: 0.72rem; color: var(--text-secondary); line-height: 2.1;">
        Steps run: <strong style="color: var(--text-primary);">{st.session_state.step_count}</strong><br>
        Buffer: <strong style="color: var(--text-primary);">{len(st.session_state.buf)}/{WINDOW}</strong><br>
        Speed: <strong style="color: var(--text-primary);">{st.session_state.speed}x</strong>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.event_log:
        st.markdown("---")
        st.markdown('<div class="sidebar-section-label">Event Log</div>', unsafe_allow_html=True)
        recent = list(reversed(st.session_state.event_log[-8:]))
        log_html = ""
        for ev in recent:
            log_html += f"""
            <div class="event-log-entry">
                <span class="event-log-step">#{ev['step']}</span>
                <span class="event-log-msg">{ev['msg']}</span>
            </div>"""
        st.markdown(f'<div style="padding: 0 0.35rem;">{log_html}</div>', unsafe_allow_html=True)


# Main dashboard

# Run simulation steps
latest = None
if st.session_state.running:
    for _ in range(steps_per_refresh):
        result = run_simulation_step()
        if result is not None:
            latest = result

# Recover latest from state if not running
if latest is None and st.session_state.health_history:
    latest = {
        "health":      st.session_state.health_history[-1],
        "health_true": st.session_state.health_true_history[-1] if st.session_state.health_true_history else 100.0,
        "fault_pred":  st.session_state.fault_history[-1] if st.session_state.fault_history else "healthy",
        "confidence":  0.0,
        "rul":         st.session_state.rul_history[-1] if st.session_state.rul_history else -1.0,
        "alert":       st.session_state.alert_history[-1] if st.session_state.alert_history else "OK",
    }

# Status bar
dot_class  = "live-dot running" if st.session_state.running else "live-dot inactive"
run_label  = "Live" if st.session_state.running else "Idle"
fault_label = FAULT_LABELS.get(st.session_state.active_fault, st.session_state.active_fault)

st.markdown(f"""
<div class="status-bar">
    <div class="status-bar-item"><div class="{dot_class}"></div><strong>{run_label}</strong></div>
    <div class="status-bar-sep"></div>
    <div class="status-bar-item">Asset <strong>Pump01</strong></div>
    <div class="status-bar-sep"></div>
    <div class="status-bar-item">Step <strong>{st.session_state.step_count}</strong></div>
    <div class="status-bar-sep"></div>
    <div class="status-bar-item">Buffer <strong>{len(st.session_state.buf)}/{WINDOW}</strong></div>
    <div class="status-bar-sep"></div>
    <div class="status-bar-item">Speed <strong>{st.session_state.speed}x</strong></div>
    <div class="status-bar-sep"></div>
    <div class="status-bar-item">Fault <strong>{fault_label}</strong></div>
</div>
""", unsafe_allow_html=True)

# Main content
if latest is None:
    st.markdown("""
    <div class="idle-state">
        <div class="idle-state-icon">&#9651;</div>
        <div class="idle-state-title">Simulation not started</div>
        <div class="idle-state-sub">Press Start in the sidebar to begin the physics simulation and ML inference pipeline.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    h = latest["health"]
    ht = latest["health_true"]
    alert = latest["alert"]
    h_class  = health_css_class(h)
    al_class = alert_css_class(alert)
    rul = latest["rul"]
    rul_display = f'{rul:.0f}s' if rul >= 0 else "Stable"
    rul_class = "ok" if rul < 0 else ("warn" if rul > 60 else "crit")
    conf = latest.get("confidence", 0.0)
    conf_class = "ok" if conf >= 0.8 else ("warn" if conf >= 0.5 else "crit")
    fault_display = format_fault(latest["fault_pred"])
    err = abs(h - ht)
    err_class = "ok" if err < 10 else ("warn" if err < 25 else "crit")
    ht_class = health_css_class(ht)

    # Alert banner (only when degraded)
    if alert != "OK":
        banner_msg = {
            "WARNING":  f"Health degrading — index at {h:.1f}. Monitor trend and prepare intervention.",
            "CRITICAL": f"Critical condition — health index {h:.1f}. Immediate action required.",
        }.get(alert, "")
        st.markdown(f'<div class="alert-banner {al_class}">{banner_msg}</div>', unsafe_allow_html=True)

    # Row 1: Health readout + KPI tiles
    col_health, col_kpis = st.columns([1, 2])

    with col_health:
        bar_width = max(0, min(100, h))
        st.markdown(f"""
        <div class="health-panel">
            <div class="health-panel-label">Health Index</div>
            <div class="health-readout {h_class}">{h:.1f}</div>
            <div class="health-sub">ML estimate &middot; EMA-smoothed &middot; out of 100</div>
            <div class="health-bar-track">
                <div class="health-bar-fill {h_class}" style="width:{bar_width}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpis:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-tile-label">Alert Status</div>
                <div class="kpi-tile-value {al_class}">{alert}</div>
                <div class="kpi-tile-meta">Threshold 70 / 40</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-tile-label">Detected Fault</div>
                <div class="kpi-tile-value fault-name">{fault_display}</div>
                <div class="kpi-tile-meta">Random Forest</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-tile-label">Confidence</div>
                <div class="kpi-tile-value {conf_class}">{conf:.0%}</div>
                <div class="kpi-tile-meta">Classifier output</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-tile-label">RUL Estimate</div>
                <div class="kpi-tile-value {rul_class}">{rul_display}</div>
                <div class="kpi-tile-meta">Linear trend</div>
            </div>""", unsafe_allow_html=True)

    # Ground truth row
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.markdown(f"""
        <div class="kpi-tile" style="margin-top:0.55rem;">
            <div class="kpi-tile-label">Ground Truth Health</div>
            <div class="kpi-tile-value {ht_class}">{ht:.1f}</div>
            <div class="kpi-tile-meta">Physics model</div>
        </div>""", unsafe_allow_html=True)
    with gc2:
        st.markdown(f"""
        <div class="kpi-tile" style="margin-top:0.55rem;">
            <div class="kpi-tile-label">Estimation Error</div>
            <div class="kpi-tile-value {err_class}">{err:.1f}</div>
            <div class="kpi-tile-meta">|ML &minus; truth|</div>
        </div>""", unsafe_allow_html=True)
    with gc3:
        st.markdown(f"""
        <div class="kpi-tile" style="margin-top:0.55rem;">
            <div class="kpi-tile-label">Simulation Step</div>
            <div class="kpi-tile-value accent">{st.session_state.step_count}</div>
            <div class="kpi-tile-meta">@ {st.session_state.speed}x speed</div>
        </div>""", unsafe_allow_html=True)

    # Live sensor readout
    sensors = latest.get("sensors", {})
    if sensors:
        st.markdown('<div class="section-label">Live Sensor Readout</div>', unsafe_allow_html=True)
        sensor_meta = [
            ("rpm",           "Speed",       "rpm"),
            ("current_a",     "Current",     "A"),
            ("flow_m3h",      "Flow",        "m3/h"),
            ("pressure_bar",  "Pressure",    "bar"),
            ("vibration_mms", "Vibration",   "mm/s"),
            ("temp_c",        "Temperature", "C"),
        ]
        cells_html = ""
        for key, label, unit in sensor_meta:
            val = sensors.get(key, 0.0)
            cells_html += f"""
            <div class="sensor-cell">
                <div class="sensor-cell-label">{label}</div>
                <div class="sensor-cell-value">{val:.1f}<span class="sensor-cell-unit">{unit}</span></div>
            </div>"""
        st.markdown(f'<div class="sensor-row">{cells_html}</div>', unsafe_allow_html=True)

    # Health trend charts
    if st.session_state.health_history:
        st.markdown('<div class="section-label">Health Trend</div>', unsafe_allow_html=True)
        col_trend, col_fault_chart = st.columns([3, 2])

        with col_trend:
            n_pts = len(st.session_state.health_history)
            ts_start = len(st.session_state.ts_history) - n_pts
            ts_for_health = (st.session_state.ts_history[ts_start:]
                             if ts_start >= 0 else list(range(n_pts)))
            health_df = pd.DataFrame({
                "Step":         ts_for_health[:n_pts],
                "ML Health":    st.session_state.health_history[:n_pts],
                "Ground Truth": (st.session_state.health_true_history[-n_pts:]
                                 if st.session_state.health_true_history else [100] * n_pts),
            })
            st.line_chart(
                health_df.set_index("Step"),
                color=["#388bfd", "#3fb950"],
                use_container_width=True,
                height=250,
            )

        with col_fault_chart:
            if st.session_state.fault_history:
                fault_counts = pd.Series(st.session_state.fault_history[-100:]).value_counts()
                st.markdown(
                    '<div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;'
                    'letter-spacing:0.08em;color:var(--text-muted);margin-bottom:0.4rem;">'
                    'Fault Distribution &mdash; last 100 steps</div>',
                    unsafe_allow_html=True
                )
                st.bar_chart(fault_counts, color="#388bfd", height=220)

    # Condition indicators
    if st.session_state.sensor_history["vibration_mms"]:
        st.markdown('<div class="section-label">Condition Indicators</div>', unsafe_allow_html=True)
        col_vib, col_temp = st.columns(2)

        with col_vib:
            n_vib = len(st.session_state.sensor_history["vibration_mms"])
            vib_df = pd.DataFrame({
                "Step": st.session_state.ts_history[-n_vib:],
                "Vibration (mm/s)": st.session_state.sensor_history["vibration_mms"][-n_vib:],
            })
            st.line_chart(vib_df.set_index("Step"), color=["#d29922"], height=195, use_container_width=True)

        with col_temp:
            n_temp = len(st.session_state.sensor_history["temp_c"])
            temp_df = pd.DataFrame({
                "Step": st.session_state.ts_history[-n_temp:],
                "Temperature (C)": st.session_state.sensor_history["temp_c"][-n_temp:],
            })
            st.line_chart(temp_df.set_index("Step"), color=["#f85149"], height=195, use_container_width=True)

    # Process variables
    if st.session_state.sensor_history["rpm"]:
        st.markdown('<div class="section-label">Process Variables</div>', unsafe_allow_html=True)
        col_rpm, col_cur, col_flow, col_press = st.columns(4)
        process_signals = [
            ("rpm",          "Speed (rpm)",   "#388bfd", col_rpm),
            ("current_a",    "Current (A)",   "#8b5cf6", col_cur),
            ("flow_m3h",     "Flow (m3/h)",   "#06b6d4", col_flow),
            ("pressure_bar", "Pressure (bar)","#3fb950", col_press),
        ]
        for sig, label, color, col in process_signals:
            with col:
                n_sig = len(st.session_state.sensor_history[sig])
                sig_df = pd.DataFrame({
                    "Step": st.session_state.ts_history[-n_sig:],
                    label: st.session_state.sensor_history[sig][-n_sig:],
                })
                st.line_chart(sig_df.set_index("Step"), color=[color], height=155, use_container_width=True)

    # All sensors normalised
    if st.session_state.sensor_history["rpm"] and len(st.session_state.sensor_history["rpm"]) > 10:
        st.markdown('<div class="section-label">All Sensors &mdash; Normalised</div>', unsafe_allow_html=True)
        combined = {}
        min_len = min(len(st.session_state.sensor_history[s]) for s in SIGNALS)
        for s in SIGNALS:
            vals = np.array(st.session_state.sensor_history[s][-min_len:])
            v_min, v_max = vals.min(), vals.max()
            span = v_max - v_min if v_max > v_min else 1.0
            combined[s] = ((vals - v_min) / span).tolist()
        combined["Step"] = st.session_state.ts_history[-min_len:]
        combined_df = pd.DataFrame(combined).set_index("Step")
        st.line_chart(combined_df, height=230, use_container_width=True)


# Auto-refresh for continuous simulation
if st.session_state.running:
    delay = max(0.05, 1.0 / st.session_state.speed / 10)
    time.sleep(delay)
    st.rerun()
