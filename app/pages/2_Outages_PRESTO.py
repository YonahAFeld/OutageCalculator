from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from app.ui.state import get_outage_events, get_site, set_outage_events


def _ensure_project_root_on_path() -> None:
    """
    Streamlit executes pages from their own directory, so the project root may not be on sys.path.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_ensure_project_root_on_path()


# -------------------------------------------------------------------
# PRESTO API helpers
# -------------------------------------------------------------------
DEFAULT_PRESTO_BASE_URL = "https://presto.lbl.gov/api/v2"


@dataclass(frozen=True)
class PrestoConfig:
    base_url: str
    api_key: str


class PrestoApiError(RuntimeError):
    pass


def _headers(api_key: str) -> Dict[str, str]:
    # Docs specify x-api-key; tolerate X-API-Key too by sending only x-api-key.
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _raise_for_status(resp: requests.Response) -> None:
    if 200 <= resp.status_code < 300:
        return
    try:
        payload = resp.json()
    except Exception:
        payload = {"text": resp.text}
    msg = f"PRESTO API error ({resp.status_code}): {payload}"
    raise PrestoApiError(msg)


@st.cache_data(show_spinner=False)
def presto_get_reliability_metrics(
    base_url: str,
    api_key: str,
    fips: str,
    years: Optional[Tuple[int, ...]] = None,
    timeout_s: int = 30,
) -> Dict[str, Any]:
    """
    GET /reliability-metric/{fips}?years=YYYY&years=YYYY

    Returns raw JSON response.
    """
    fips = str(fips).zfill(5)
    url = f"{base_url}/reliability-metric/{fips}"

    params: List[Tuple[str, str]] = []
    if years:
        for y in years:
            params.append(("years", str(int(y))))

    resp = requests.get(url, headers=_headers(api_key), params=params, timeout=timeout_s)
    _raise_for_status(resp)
    return resp.json()


def presto_run_model(
    base_url: str,
    api_key: str,
    fips: str,
    simulations: int,
    reliability_metrics_override: Optional[List[List[float]]] = None,
    model_years: Optional[List[int]] = None,
    timeout_s: int = 90,
) -> Dict[str, Any]:
    """
    POST /model

    Returns raw JSON response.
    """
    fips = str(fips).zfill(5)
    url = f"{base_url}/model"

    body: Dict[str, Any] = {
        "fips": fips,
        "simulations": int(simulations),
    }
    if reliability_metrics_override is not None:
        body["reliabilityMetrics"] = reliability_metrics_override
    if model_years is not None and len(model_years) > 0:
        body["modelYears"] = model_years

    resp = requests.post(url, headers=_headers(api_key), data=json.dumps(body), timeout=timeout_s)
    _raise_for_status(resp)
    return resp.json()


def _normalize_events_from_model_response(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Best-effort normalization of PRESTO simulation output.

    Since the exact schema can evolve, this function tries to locate a list of
    events and convert them into a tidy dataframe:
      - simulation (int)
      - start (datetime, optional)
      - duration_hours (float, optional)
      - duration_minutes (float, optional)
      - raw (json string, optional)
    """
    results = payload.get("results", payload)

    # Attempt to locate event list(s)
    candidate = None
    for key in ["interruptions", "events", "outages", "simulations", "runs", "years"]:
        if isinstance(results.get(key), list):
            candidate = results[key]
            break

    rows: List[Dict[str, Any]] = []

    # Case A: already a flat list of events
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
        for i, ev in enumerate(candidate):
            rows.append(_event_row(simulation=None, event=ev))

    # Case B: list of simulations, each containing events
    elif isinstance(candidate, list) and candidate and not isinstance(candidate[0], dict):
        for sim_idx, sim_obj in enumerate(candidate):
            if isinstance(sim_obj, dict):
                ev_list = None
                for k in ["interruptions", "events", "outages"]:
                    if isinstance(sim_obj.get(k), list):
                        ev_list = sim_obj.get(k)
                        break
                if ev_list:
                    for ev in ev_list:
                        if isinstance(ev, dict):
                            rows.append(_event_row(simulation=sim_idx, event=ev))
                else:
                    rows.append(
                        {
                            "simulation": sim_idx,
                            "start": None,
                            "duration_hours": None,
                            "duration_minutes": None,
                            "raw": json.dumps(sim_obj),
                        }
                    )
            else:
                rows.append(
                    {
                        "simulation": sim_idx,
                        "start": None,
                        "duration_hours": None,
                        "duration_minutes": None,
                        "raw": json.dumps(sim_obj),
                    }
                )

    # Fallback: store raw payload only
    if not rows:
        rows = [
            {
                "simulation": None,
                "start": None,
                "duration_hours": None,
                "duration_minutes": None,
                "raw": json.dumps(payload),
            }
        ]

    df = pd.DataFrame(rows)

    # Attempt to parse start into datetime
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce")

    return df


def _event_row(simulation: Optional[int], event: Dict[str, Any]) -> Dict[str, Any]:
    start = (
        event.get("start")
        or event.get("startTime")
        or event.get("timestamp")
        or event.get("time")
        or None
    )

    duration_minutes = (
        event.get("durationMinutes")
        or event.get("duration_minutes")
        or event.get("duration_mins")
        or None
    )
    duration_hours = (
        event.get("durationHours")
        or event.get("duration_hours")
        or None
    )

    # Some APIs return duration in seconds
    duration_seconds = event.get("durationSeconds") or event.get("duration_seconds") or None

    dm = None
    dh = None

    try:
        if duration_minutes is not None:
            dm = float(duration_minutes)
            dh = dm / 60.0
        elif duration_hours is not None:
            dh = float(duration_hours)
            dm = dh * 60.0
        elif duration_seconds is not None:
            dm = float(duration_seconds) / 60.0
            dh = dm / 60.0
    except Exception:
        dm = None
        dh = None

    row: Dict[str, Any] = {
        "simulation": simulation,
        "start": start,
        "duration_hours": dh,
        "duration_minutes": dm,
        "raw": None,
    }

    # Keep a raw copy if we could not confidently parse duration
    if dm is None and dh is None:
        row["raw"] = json.dumps(event)

    return row


def _metrics_to_dataframe(metrics_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = metrics_payload.get("results", [])
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["month", "saifi", "saidi", "caidi"])
    df = pd.DataFrame(rows)
    for c in ["month", "saifi", "saidi", "caidi"]:
        if c not in df.columns:
            df[c] = None
    return df[["month", "saifi", "saidi", "caidi"]].sort_values("month")


def _metrics_to_override_array(metrics_df: pd.DataFrame) -> List[List[float]]:
    """
    Convert metrics dataframe to the override format required by POST /model:
      12 elements where each element is [SAIDI, SAIFI]
    """
    override: List[List[float]] = []
    for m in range(1, 13):
        row = metrics_df.loc[metrics_df["month"] == m]
        if row.empty:
            override.append([0.0, 0.0])
            continue
        saidi = float(row["saidi"].iloc[0]) if pd.notna(row["saidi"].iloc[0]) else 0.0
        saifi = float(row["saifi"].iloc[0]) if pd.notna(row["saifi"].iloc[0]) else 0.0
        override.append([saidi, saifi])
    return override


# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------
st.set_page_config(page_title="Outages (PRESTO)", page_icon="⚡", layout="wide")

st.title("Outage profile from PRESTO")
st.caption("Run county-level outage simulations via the PRESTO API and store the generated interruption events.")

site = get_site() or {}

if not site:
    st.warning("Please complete 'Site inputs' first.")
    st.stop()

fips = str(site.get("fips", "")).strip()
county_name = str(site.get("county_name", "")).strip()
state_abbrev = str(site.get("state_abbrev", "")).strip()

st.subheader("Selected location")
loc_cols = st.columns(3)
with loc_cols[0]:
    st.metric("County", county_name or "Not set")
with loc_cols[1]:
    st.metric("State", state_abbrev or "Not set")
with loc_cols[2]:
    st.metric("FIPS", fips.zfill(5) if fips else "Not set")

st.subheader("API configuration")

with st.expander("Settings", expanded=True):
    base_url = st.text_input("PRESTO base URL", value=os.getenv("PRESTO_BASE_URL", DEFAULT_PRESTO_BASE_URL))

    api_key_env = os.getenv("PRESTO_API_KEY", "").strip()
    api_key = st.text_input(
        "PRESTO API key",
        value=api_key_env,
        type="password",
        help="Set PRESTO_API_KEY in your environment to avoid typing this each run.",
    )

    if not api_key:
        st.warning("Enter your PRESTO API key to continue.")
        st.stop()

st.subheader("Simulation controls")

simulations = st.slider("Number of annual simulations", min_value=1000, max_value=20000, value=2000, step=1000)

years = st.multiselect(
    "Model years (optional, used to construct default reliability metrics)",
    options=list(range(2014, 2024)),
    default=[],
    help="If omitted, PRESTO returns all available years for reliability metrics.",
)

col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    load_metrics_clicked = st.button("Load default reliability metrics", use_container_width=True)
with col_b:
    run_model_clicked = st.button("Run simulations", use_container_width=True)
with col_c:
    st.caption("Tip: Load metrics first if you want to inspect or override SAIDI and SAIFI by month later.")

metrics_payload: Optional[Dict[str, Any]] = None
metrics_df: Optional[pd.DataFrame] = None

if load_metrics_clicked:
    if not fips:
        st.error("Missing FIPS. Please set a county FIPS on the Site inputs page.")
        st.stop()

    try:
        metrics_payload = presto_get_reliability_metrics(
            base_url=base_url,
            api_key=api_key,
            fips=fips,
            years=tuple(int(y) for y in years) if years else None,
        )
        metrics_df = _metrics_to_dataframe(metrics_payload)
    except PrestoApiError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading metrics: {e}")
        st.stop()

    st.success("Loaded reliability metrics.")
    st.dataframe(metrics_df, use_container_width=True)

    st.line_chart(metrics_df.set_index("month")[["saidi", "saifi", "caidi"]])

# Store metrics in session for downstream use if we loaded them this run
if metrics_df is not None:
    st.session_state["presto_metrics_df_json"] = metrics_df.to_json(orient="records")

override_metrics = None
if "presto_metrics_df_json" in st.session_state:
    try:
        df_tmp = pd.DataFrame(json.loads(st.session_state["presto_metrics_df_json"]))
        override_metrics = _metrics_to_override_array(df_tmp)
    except Exception:
        override_metrics = None

if run_model_clicked:
    if not fips:
        st.error("Missing FIPS. Please set a county FIPS on the Site inputs page.")
        st.stop()

    with st.spinner("Calling PRESTO and generating simulations..."):
        try:
            model_payload = presto_run_model(
                base_url=base_url,
                api_key=api_key,
                fips=fips,
                simulations=int(simulations),
                reliability_metrics_override=override_metrics,
                model_years=[int(y) for y in years] if years else None,
            )
        except PrestoApiError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error running model: {e}")
            st.stop()

    events_df = _normalize_events_from_model_response(model_payload)

    # Persist both the raw response and normalized events
    saved = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "location": {"fips": fips.zfill(5), "county_name": county_name, "state_abbrev": state_abbrev},
        "simulations": int(simulations),
        "model_years": [int(y) for y in years] if years else [],
        "raw_response": model_payload,
        "events": json.loads(events_df.to_json(orient="records", date_format="iso")),
    }

    set_outage_events(saved)
    st.success("Saved PRESTO outage events. Next: go to 'CDF Costs' in the left navigation.")

st.divider()
st.subheader("Saved outage events (current session)")

saved_events = get_outage_events()
if saved_events is None:
    st.info("No outage simulations saved yet.")
else:
    st.caption("Preview of the saved payload stored in session state.")
    st.json(
        {
            "generated_at": saved_events.get("generated_at"),
            "location": saved_events.get("location"),
            "simulations": saved_events.get("simulations"),
            "model_years": saved_events.get("model_years"),
            "events_count": len(saved_events.get("events", [])),
            "raw_response_keys": list((saved_events.get("raw_response") or {}).keys())[:25],
        }
    )

    try:
        preview_df = pd.DataFrame(saved_events.get("events", []))
        if not preview_df.empty:
            st.dataframe(preview_df.head(200), use_container_width=True)
    except Exception:
        pass
