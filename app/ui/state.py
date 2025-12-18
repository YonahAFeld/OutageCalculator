

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st


@dataclass(frozen=True)
class StateKeys:
    """
    Central registry for Streamlit session_state keys.

    Keeping keys in one place prevents subtle bugs caused by typos and
    makes it easier to change the app's state shape over time.
    """

    SITE: str = "site"
    OUTAGE_EVENTS: str = "outage_events"
    CDF_INPUTS: str = "cdf_inputs"
    RESULTS: str = "results"


DEFAULT_SESSION_STATE: Dict[str, Any] = {
    StateKeys.SITE: {},
    StateKeys.OUTAGE_EVENTS: None,
    StateKeys.CDF_INPUTS: {},
    StateKeys.RESULTS: None,
}


def init_session_state() -> None:
    """
    Initialize required session_state keys with default values.
    Safe to call multiple times.
    """
    for key, default in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_site() -> Dict[str, Any]:
    init_session_state()
    return st.session_state[StateKeys.SITE]


def set_site(site: Dict[str, Any]) -> None:
    init_session_state()
    st.session_state[StateKeys.SITE] = site


def get_outage_events() -> Optional[Any]:
    """
    Outage events payload (shape defined later).
    Often: list[dict] or a pandas DataFrame.
    """
    init_session_state()
    return st.session_state[StateKeys.OUTAGE_EVENTS]


def set_outage_events(events: Any) -> None:
    init_session_state()
    st.session_state[StateKeys.OUTAGE_EVENTS] = events


def get_cdf_inputs() -> Dict[str, Any]:
    init_session_state()
    return st.session_state[StateKeys.CDF_INPUTS]


def set_cdf_inputs(cdf_inputs: Dict[str, Any]) -> None:
    init_session_state()
    st.session_state[StateKeys.CDF_INPUTS] = cdf_inputs


def get_results() -> Optional[Dict[str, Any]]:
    init_session_state()
    return st.session_state[StateKeys.RESULTS]


def set_results(results: Dict[str, Any]) -> None:
    init_session_state()
    st.session_state[StateKeys.RESULTS] = results


def reset_all_state() -> None:
    """
    Reset the app to a blank slate.
    Useful for a "Start over" button.
    """
    for key, default in DEFAULT_SESSION_STATE.items():
        st.session_state[key] = default