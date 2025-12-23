from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import streamlit as st


def _ensure_project_root_on_path() -> None:
    """
    Streamlit can execute pages with a sys.path that doesn't include the project root.
    This ensures imports like `from app.ui.state import ...` work reliably.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_ensure_project_root_on_path()

from app.ui.state import (
    init_session_state,
    get_site,
    get_outage_events,
    get_cdf_inputs,
    get_results,
)

# Backward/forward compatibility: some versions expose additional helpers.
try:
    from app.ui.state import get_presto_events  # type: ignore
except Exception:
    get_presto_events = None  # type: ignore

# -------------------------------------------------------------------
# Status checks
# -------------------------------------------------------------------

def _has_site() -> bool:
    try:
        site = get_site()
    except Exception:
        site = st.session_state.get("site")
    return bool(isinstance(site, dict) and site)


def _has_outage_events() -> bool:
    # Prefer a dedicated getter if present
    if callable(get_presto_events):
        try:
            ev = get_presto_events()
            if ev:
                return True
        except Exception:
            pass

    try:
        payload = get_outage_events()
    except Exception:
        payload = st.session_state.get("outage_events")

    if payload is None:
        return False

    # Legacy: list of events
    if isinstance(payload, list):
        return len(payload) > 0

    if isinstance(payload, dict):
        # Newer: events_tidy
        tidy = payload.get("events_tidy")
        if isinstance(tidy, list) and len(tidy) > 0:
            return True

        # Newer: nested results.events
        results = payload.get("results")
        if isinstance(results, dict):
            for k in ("events", "interruptions", "outages"):
                v = results.get(k)
                if isinstance(v, list) and len(v) > 0:
                    return True

        # Some older shapes stored events at top-level
        for k in ("events", "interruptions", "outages"):
            v = payload.get(k)
            if isinstance(v, list) and len(v) > 0:
                return True

    return False


def _has_cdf_inputs() -> bool:
    try:
        cdf = get_cdf_inputs()
    except Exception:
        cdf = st.session_state.get("cdf_inputs")

    # Accept either a dict payload or a truthy object
    if cdf is None:
        return False
    if isinstance(cdf, dict):
        return bool(cdf)
    return True


def _has_results() -> bool:
    # Prefer canonical getter
    try:
        res = get_results()
    except Exception:
        res = None

    if res is not None:
        # get_results may return dict/list; any non-empty is considered ready
        if isinstance(res, dict):
            return bool(res)
        if isinstance(res, list):
            return len(res) > 0
        return True

    # Fallback: check common session_state keys used across iterations
    for k in ("results", "cdf_results", "results_payload", "analysis_results"):
        v = st.session_state.get(k)
        if v is None:
            continue
        if isinstance(v, dict) and v:
            return True
        if isinstance(v, list) and len(v) > 0:
            return True
        if not isinstance(v, (dict, list)):
            return True

    return False

st.set_page_config(
    page_title="Outage Cost Impact Tool",
    page_icon="⚡",
    layout="wide"
)

init_session_state()

# -------------------------------------------------------------------
# App header
# -------------------------------------------------------------------
st.title("Grid Outage Financial Impact Tool")
st.subheader("Quantifying what power interruptions actually cost your business")

st.markdown(
    """
This tool helps commercial and industrial facilities understand the true financial
impact of grid outages using:

- Realistic outage simulations from LBNL’s PRESTO model
- Cost modeling aligned with NREL and FEMP Customer Damage Functions (CDF)
- Transparent math you can inspect and export

You will be guided step by step through:
1. Defining your facility and operations
2. Simulating realistic outage events for your county
3. Assigning outage-related costs
4. Reviewing expected costs, bad-year risk, and cost drivers

Use the navigation on the left to get started.
"""
)

# -------------------------------------------------------------------
# Status checks
# -------------------------------------------------------------------
st.divider()
st.subheader("Progress")

site_defined = _has_site()
outages_generated = _has_outage_events()
costs_defined = _has_cdf_inputs()
results_ready = _has_results()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Site defined", "Yes" if site_defined else "No")

with col2:
    st.metric("Outages loaded", "Yes" if outages_generated else "No")

with col3:
    st.metric("Costs defined", "Yes" if costs_defined else "No")

with col4:
    st.metric("Results ready", "Yes" if results_ready else "No")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.divider()
st.caption(
    "This tool is intended for planning and decision support. "
    "Results depend on assumptions and simulated outage behavior."
)