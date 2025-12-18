from dotenv import load_dotenv
load_dotenv()
from __future__ import annotations

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

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
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

site_defined = bool(get_site())
outages_generated = get_outage_events() is not None
costs_defined = bool(get_cdf_inputs())
results_ready = get_results() is not None

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Site defined", "Yes" if site_defined else "No")

with col2:
    st.metric("Outages simulated", "Yes" if outages_generated else "No")

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