from __future__ import annotations

import json
import os
import sys
import random
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

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

def _safe_page_link(path: str, label: str, icon: str = "") -> None:
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:
        st.write(f"{icon} {label}: {path}")

# -------------------------------------------------------------------
# PRESTO API helpers
# -------------------------------------------------------------------

DEFAULT_PRESTO_BASE_URL = "https://presto.lbl.gov/api/v2"
DEFAULT_PRESTO_MODEL_ENDPOINT = "/model"
DEFAULT_PRESTO_RELIABILITY_ENDPOINT = "/reliability-metric"


def _normalize_base_url(raw: str) -> str:
    """
    Ensure base_url includes /api/v2 exactly once and has no trailing slash.
    """
    s = (raw or "").strip()
    if not s:
        return DEFAULT_PRESTO_BASE_URL
    s = s.rstrip("/")
    if s.endswith("/api/v2"):
        return s
    if "/api/v2" not in s:
        s = f"{s}/api/v2"
    return s


@dataclass(frozen=True)
class PrestoConfig:
    base_url: str
    api_key: str


class PrestoApiError(RuntimeError):
    pass


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "cdf_presto_streamlit/0.1",
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


def _call_presto_model(
    *,
    base_url: str,
    api_key: str,
    fips: str,
    simulations: int,
    mean_events_per_year: Optional[float] = None,
) -> Dict[str, Any]:
    """Call PRESTO to generate interruption events.

    Notes:
      - Endpoint path is configurable via PRESTO_MODEL_ENDPOINT env var.
      - Payload keys are best-effort; if PRESTO changes, update here.
    """
    endpoint = (os.getenv("PRESTO_MODEL_ENDPOINT", DEFAULT_PRESTO_MODEL_ENDPOINT) or "").strip()
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    url = f"{base_url}{endpoint}"

    payload: Dict[str, Any] = {
        "fips": str(fips).zfill(5),
        "simulations": int(simulations),
    }
    if mean_events_per_year is not None:
        payload["mean_events_per_year"] = float(mean_events_per_year)

    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=60)
    _raise_for_status(resp)
    return resp.json()


def _call_presto_reliability_metric(
    *,
    base_url: str,
    api_key: str,
    fips: str,
    years: List[int],
) -> Dict[str, Any]:
    """Fetch historical reliability metrics (SAIFI/SAIDI/CAIDI) from PRESTO."""
    endpoint = (os.getenv("PRESTO_RELIABILITY_ENDPOINT", DEFAULT_PRESTO_RELIABILITY_ENDPOINT) or "").strip()
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    years = [int(y) for y in years if y is not None]
    # PRESTO supports repeated `years=` query params.
    qs = "&".join([f"years={y}" for y in years])
    url = f"{base_url}{endpoint}/{str(fips).zfill(5)}"
    if qs:
        url = f"{url}?{qs}"

    resp = requests.get(url, headers=_headers(api_key), timeout=60)
    _raise_for_status(resp)
    return resp.json()


# -------------------------------------------------------------------
# Normalization helpers
# -------------------------------------------------------------------
def _normalize_events_from_model_response(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Best-effort normalization of simulation output into a tidy dataframe:
      - simulation (int)
      - start (datetime, optional)
      - end (datetime, optional)
      - duration_hours (float, optional)
      - duration_minutes (float, optional)
      - raw (json string, optional)
    """
    # PRESTO sometimes returns a dict payload, and sometimes a bare list of events.
    # Normalize both shapes.
    if isinstance(payload, list):
        results: Any = payload
    else:
        results = payload.get("results", payload)

    candidate: Any = None

    # If results is already a list, treat it as the candidate list.
    if isinstance(results, list):
        candidate = results
    elif isinstance(results, dict):
        for key in ["interruptions", "events", "outages", "simulations", "runs", "years"]:
            if isinstance(results.get(key), list):
                candidate = results[key]
                break

    rows: List[Dict[str, Any]] = []

    # Case A: flat list of events
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
        for ev in candidate:
            rows.append(_event_row(simulation=ev.get("simulation"), event=ev))

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
                            "end": None,
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
                        "end": None,
                        "duration_hours": None,
                        "duration_minutes": None,
                        "raw": json.dumps(sim_obj),
                    }
                )

    if not rows:
        rows = [
            {
                "simulation": None,
                "start": None,
                "end": None,
                "duration_hours": None,
                "duration_minutes": None,
                "raw": json.dumps(payload),
            }
        ]

    df = pd.DataFrame(rows)

    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], errors="coerce")

    return df


def _event_row(simulation: Optional[int], event: Dict[str, Any]) -> Dict[str, Any]:
    start = (
        event.get("start")
        or event.get("startTime")
        or event.get("timestamp")
        or event.get("time")
        or None
    )
    end = event.get("end") or event.get("endTime") or None

    duration_minutes = (
        event.get("durationMinutes")
        or event.get("duration_minutes")
        or event.get("duration_mins")
        or None
    )
    duration_hours = event.get("durationHours") or event.get("duration_hours") or None
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
        "end": end,
        "duration_hours": dh,
        "duration_minutes": dm,
        "raw": None,
    }

    if dm is None and dh is None:
        row["raw"] = json.dumps(event)

    return row


def _normalize_reliability_metrics(payload: Dict[str, Any]) -> pd.DataFrame:
    """Normalize PRESTO reliability-metric payload to a tidy dataframe."""
    if not isinstance(payload, dict):
        return pd.DataFrame([])

    rows = payload.get("results")
    if not isinstance(rows, list):
        return pd.DataFrame([])

    df = pd.DataFrame(rows)
    # Expected columns: month, saifi, saidi, caidi
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df = df.sort_values("month")
    for col in ["saifi", "saidi", "caidi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _extract_saved_events_list(saved: Any) -> List[Any]:
    """Return the best-effort list of event dicts from a saved outage payload."""
    if saved is None:
        return []

    # Some legacy sessions stored a bare list
    if isinstance(saved, list):
        return saved

    if not isinstance(saved, dict):
        return []

    results = saved.get("results")

    # Some payloads store results as a bare list
    if isinstance(results, list):
        return results

    if isinstance(results, dict):
        return (
            results.get("events")
            or results.get("interruptions")
            or results.get("outages")
            or []
        )

    # As a last resort, check top-level keys
    return (saved.get("events") or saved.get("interruptions") or saved.get("outages") or [])

# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------
st.set_page_config(page_title="Outages", page_icon="⚡", layout="wide")

st.title("Outage profile")
st.caption("This step builds the outage risk profile for your facility location using county data and PRESTO simulations.")

# ---------------- Location ----------------
site = get_site() or {}
if not site:
    st.warning("Please complete 'Site inputs' first.")
    st.stop()

fips = str(site.get("fips", "")).strip()
county_name = str(site.get("county_name", "")).strip()
state_abbrev = str(site.get("state_abbrev", "")).strip()

base_url = _normalize_base_url(os.getenv("PRESTO_BASE_URL", DEFAULT_PRESTO_BASE_URL))
api_key = os.getenv("PRESTO_API_KEY", "").strip()
debug = os.getenv("PRESTO_DEBUG", "").strip().lower() in ("1", "true", "yes")

# Visible marker so you can confirm the app is running this exact file version
st.caption("Outages page version: step1 metrics + step2 scenarios")

a, b, c = st.columns(3)
with a:
    st.metric("County", county_name or "Not set")
with b:
    st.metric("State", state_abbrev or "Not set")
with c:
    st.metric("FIPS", fips.zfill(5) if fips else "Not set")

if not fips:
    st.error("Missing FIPS. Please set a county FIPS on the Site inputs page.")
    st.stop()

if not api_key:
    st.warning("PRESTO_API_KEY is not set. Add it to your environment (or Streamlit secrets) to run this step.")

st.divider()

# ---------------- Step 1: Historical reliability ----------------
st.subheader("Step 1: Load historical reliability")

st.markdown(
    """
PRESTO provides county-level historical reliability metrics:

- **SAIFI**: average number of outages per customer per year (frequency)
- **SAIDI**: average total outage **hours** per customer per year (duration)
- **CAIDI**: average outage **hours per outage** (SAIDI / SAIFI)

These metrics ground the simulation in observed reliability for your county.
    """
)

saved = get_outage_events()

hist_ready = bool(isinstance(saved, dict) and isinstance(saved.get("historical_by_year"), dict) and saved.get("historical_by_year"))

load_hist_clicked = st.button("Load historical reliability metrics", type="primary", use_container_width=True, disabled=(not bool(api_key)))

if load_hist_clicked:
    years = list(range(2014, 2024))

    with st.spinner("Fetching historical reliability metrics from PRESTO..."):
        probe_payload = _call_presto_reliability_metric(
            base_url=base_url,
            api_key=api_key,
            fips=fips,
            years=years,
        )

        avail_years = probe_payload.get("availableYears") if isinstance(probe_payload, dict) else None
        if isinstance(avail_years, list) and avail_years:
            years = sorted({int(y) for y in avail_years if y is not None})

        per_year: Dict[str, Dict[str, Any]] = {}
        progress = st.progress(0)
        total = max(1, len(years))

        for i, y in enumerate(years, start=1):
            try:
                py = _call_presto_reliability_metric(
                    base_url=base_url,
                    api_key=api_key,
                    fips=fips,
                    years=[int(y)],
                )
                if isinstance(py, dict):
                    per_year[str(int(y))] = py
            except Exception as e:
                if debug:
                    st.write(f"Failed year {y}: {e}")
            progress.progress(int((i / total) * 100))

        # Build tidy yearly table: year + month + metrics
        yearly_rows: List[Dict[str, Any]] = []
        for y_str, py in per_year.items():
            df_y = _normalize_reliability_metrics(py)
            if not df_y.empty:
                df_y = df_y.copy()
                df_y["year"] = int(y_str)
                yearly_rows.extend(json.loads(df_y.to_json(orient="records")))

        hist_yearly_tidy: List[Dict[str, Any]] = yearly_rows

        # Average across years by month for lightweight display/consistency
        hist_tidy_avg: List[Dict[str, Any]] = []
        hist_payload: Optional[Dict[str, Any]] = None

        hist_df = pd.DataFrame(hist_yearly_tidy)
        if not hist_df.empty and "month" in hist_df.columns:
            agg = (
                hist_df.groupby("month", as_index=False)[[c for c in ["saifi", "saidi", "caidi"] if c in hist_df.columns]]
                .mean(numeric_only=True)
                .sort_values("month")
            )
            hist_tidy_avg = json.loads(agg.to_json(orient="records"))
            hist_payload = {
                "input": {"fips": fips.zfill(5), "years": years},
                "availableYears": years,
                "results": json.loads(agg.to_json(orient="records")),
            }
        else:
            hist_payload = probe_payload

        # Persist to session
        merged = saved if isinstance(saved, dict) else {}
        merged = dict(merged)
        merged["historical"] = hist_payload
        merged["historical_tidy"] = hist_tidy_avg
        merged["historical_by_year"] = per_year
        merged["historical_yearly_tidy"] = hist_yearly_tidy

        merged.setdefault("meta", {})
        if isinstance(merged.get("meta"), dict):
            merged["meta"]["historical_years"] = years
            merged["meta"]["historical_loaded_at"] = datetime.utcnow().isoformat() + "Z"

        set_outage_events(merged)
        saved = merged

hist_ready = bool(isinstance(saved, dict) and isinstance(saved.get("historical_by_year"), dict) and saved.get("historical_by_year"))

if hist_ready:
    # Compute annual SAIFI by year (sum monthly SAIFI values per year)
    hist_yearly = saved.get("historical_yearly_tidy") if isinstance(saved, dict) else None
    hist_yearly_df = pd.DataFrame(hist_yearly) if isinstance(hist_yearly, list) else pd.DataFrame([])

    annual_saifi: Dict[int, float] = {}
    annual_saidi: Dict[int, float] = {}

    if not hist_yearly_df.empty and "year" in hist_yearly_df.columns:
        if "saifi" in hist_yearly_df.columns:
            saifi_by_year = (
                hist_yearly_df.dropna(subset=["year"]).groupby("year")["saifi"].sum(min_count=1)
            )
            for k, v in saifi_by_year.to_dict().items():
                try:
                    annual_saifi[int(k)] = float(v)
                except Exception:
                    pass

        if "saidi" in hist_yearly_df.columns:
            saidi_by_year = (
                hist_yearly_df.dropna(subset=["year"]).groupby("year")["saidi"].sum(min_count=1)
            )
            for k, v in saidi_by_year.to_dict().items():
                try:
                    annual_saidi[int(k)] = float(v)
                except Exception:
                    pass

    saifi_series = pd.Series(list(annual_saifi.values()), dtype="float64") if annual_saifi else pd.Series([], dtype="float64")
    saidi_series = pd.Series(list(annual_saidi.values()), dtype="float64") if annual_saidi else pd.Series([], dtype="float64")

    base_saifi = float(saifi_series.mean()) if not saifi_series.empty else None
    p75_saifi = float(saifi_series.quantile(0.75)) if not saifi_series.empty else None
    p90_saifi = float(saifi_series.quantile(0.90)) if not saifi_series.empty else None

    st.success("Historical reliability metrics loaded.")

    # Show a small summary (no charts)
    yrs = sorted(list(annual_saifi.keys()))
    if yrs:
        st.caption(f"Years loaded: {min(yrs)} to {max(yrs)}")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("SAIFI (avg annual, outages/customer)", f"{base_saifi:.2f}" if base_saifi is not None else "N/A")
    with s2:
        saidi_annual_hours = float(saidi_series.mean()) if not saidi_series.empty else None
        st.metric("SAIDI (avg annual, hours/customer)", f"{saidi_annual_hours:.2f}" if saidi_annual_hours is not None else "N/A")
    with s3:
        if (base_saifi is not None) and (saidi_annual_hours is not None) and base_saifi > 0:
            caidi_hours = saidi_annual_hours / base_saifi
            caidi_minutes = caidi_hours * 60.0
            st.metric("CAIDI (implied avg)", f"{caidi_hours:.2f} hours/outage", help=f"{caidi_minutes:.0f} minutes per outage")
        else:
            st.metric("CAIDI (implied avg)", "N/A")

    st.divider()

    # ---------------- Step 2: Future simulations ----------------
    st.subheader("Step 2: Run future outage simulations")

    st.markdown(
        """
Choose how conservative you want the simulations to be. We use historical **annual SAIFI** (outages per customer per year) to set the expected number of outages per simulated year.

- **Base case**: use the historical **average** SAIFI (typical year)
- **Planning case**: use **P75 SAIFI** (a bad-but-plausible year, roughly 1 in 4)
- **Stress case**: use **P90 SAIFI** (rare high-outage year, roughly 1 in 10)

This only sets outage **frequency**. PRESTO still determines outage **durations**.
        """
    )

    scenario = st.radio(
        "Simulation scenario",
        options=["Base case (mean)", "Planning case (P75)", "Stress case (P90)"],
        index=1,
        horizontal=True,
    )

    scenario_to_freq = {
        "Base case (mean)": base_saifi,
        "Planning case (P75)": p75_saifi,
        "Stress case (P90)": p90_saifi,
    }

    mean_events_per_year = scenario_to_freq.get(scenario)

    if mean_events_per_year is None:
        st.error("Could not compute annual SAIFI from historical data. Cannot run simulations.")
        mean_events_per_year = None

    quality = st.selectbox(
        "Simulation quality",
        options=[
            "Fast (3,000 simulated years)",
            "Balanced (5,000 simulated years)",
            "Deep risk (10,000 simulated years)",
            "Research-grade (20,000 simulated years)",
        ],
        index=1,
    )

    quality_to_years = {
        "Fast (3,000 simulated years)": 3000,
        "Balanced (5,000 simulated years)": 5000,
        "Deep risk (10,000 simulated years)": 10000,
        "Research-grade (20,000 simulated years)": 20000,
    }

    simulations = int(quality_to_years.get(quality, 5000))

    st.caption(
        f"This run will simulate {simulations:,} years using an expected outage frequency of {mean_events_per_year:.2f} events/year (from {scenario})."
        if mean_events_per_year is not None
        else ""
    )

    run_sims_clicked = st.button("Run PRESTO simulations", type="primary", use_container_width=True, disabled=(mean_events_per_year is None))

    if run_sims_clicked and mean_events_per_year is not None:
        with st.spinner("Calling PRESTO model endpoint..."):
            presto_payload = _call_presto_model(
                base_url=base_url,
                api_key=api_key,
                fips=fips,
                simulations=int(simulations),
                mean_events_per_year=float(mean_events_per_year),
            )

        if isinstance(presto_payload, list):
            presto_payload = {"results": {"events": presto_payload}}

        events_df = _normalize_events_from_model_response(presto_payload)

        payload = presto_payload if isinstance(presto_payload, dict) else {"results": {"events": presto_payload}}
        payload["meta"] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "location": {"fips": fips.zfill(5), "county_name": county_name, "state_abbrev": state_abbrev},
            "simulations": int(simulations),
            "model_years": [],
            "source": "presto",
            "base_url": base_url,
            "endpoint": (os.getenv("PRESTO_MODEL_ENDPOINT", DEFAULT_PRESTO_MODEL_ENDPOINT) or DEFAULT_PRESTO_MODEL_ENDPOINT),
            "scenario": scenario,
            "mean_events_per_year": float(mean_events_per_year),
        }
        payload["events_tidy"] = json.loads(events_df.to_json(orient="records", date_format="iso"))

        payload["historical"] = saved.get("historical") if isinstance(saved, dict) else None
        payload["historical_tidy"] = saved.get("historical_tidy") if isinstance(saved, dict) else []
        payload["historical_by_year"] = saved.get("historical_by_year") if isinstance(saved, dict) else {}
        payload["historical_yearly_tidy"] = saved.get("historical_yearly_tidy") if isinstance(saved, dict) else []

        if isinstance(payload.get("meta"), dict) and isinstance(saved, dict) and isinstance(saved.get("meta"), dict):
            payload["meta"]["historical_years"] = saved.get("meta", {}).get("historical_years")
            payload["meta"]["historical_loaded_at"] = saved.get("meta", {}).get("historical_loaded_at")

        existing_saved = get_outage_events()
        if isinstance(existing_saved, dict):
            merged = dict(existing_saved)
            merged.update(payload)
            payload_to_save = merged
        else:
            payload_to_save = payload

        set_outage_events(payload_to_save)

        st.success("Saved simulated outages.")
        _safe_page_link("app/pages/3_CDF_Costs.py", label="Go to CDF Costs", icon="🧾")
else:
    st.info("Load historical reliability metrics to enable simulations.")

st.divider()

# ---------------- Saved state (minimal) ----------------
st.subheader("Saved status")
saved = get_outage_events()

if saved is None:
    st.info("Nothing saved yet. First click 'Load historical reliability metrics', then choose a scenario and run simulations.")
else:
    if isinstance(saved, list):
        saved = {"results": {"events": saved}, "meta": {"source": "presto"}}

    try:
        ev_count = len(_extract_saved_events_list(saved))
    except Exception:
        ev_count = 0

    meta = saved.get("meta") if isinstance(saved, dict) else {}
    loc = (meta or {}).get("location") if isinstance(meta, dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Events saved", str(ev_count))
    with c2:
        st.metric("Simulated years", str((meta or {}).get("simulations") or "0"))
    with c3:
        st.metric("Scenario", str((meta or {}).get("scenario") or "—"))
    with c4:
        mepy = (meta or {}).get("mean_events_per_year")
    st.metric("Events/year used", f"{float(mepy):.2f}" if mepy is not None else "—")

    with st.expander("Debug payload", expanded=False):
        st.json(
            {
                "generated_at": (meta or {}).get("generated_at"),
                "location": loc,
                "simulations": (meta or {}).get("simulations"),
                "historical_years": (meta or {}).get("historical_years"),
                "payload_keys": (list(saved.keys())[:30] if isinstance(saved, dict) else []),
            }
        )