from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

import sys


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

from app.ui.state import get_site, set_site


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------
@dataclass
class SiteInputs:
    facility_name: str
    county_name: str
    state_abbrev: str
    fips: str
    facility_type: str
    peak_demand_kw: float
    annual_energy_kwh: float
    critical_load_kw: float
    critical_annual_energy_kwh: float
    analysis_period_years: int
    discount_rate_nominal: float


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _reference_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "reference"


@st.cache_data(show_spinner=False)
def load_counties() -> pd.DataFrame:
    """
    Load counties reference data used to select PRESTO simulation geography.

    Expected columns in counties.csv (either format is supported):

    Preferred:
      - county_name
      - state_abbrev
      - fips

    Also supported (common Census-style headers):
      - FIPS State and County Code
      - County Name
      - State Name
    """
    path = _reference_data_path() / "counties.csv"
    if not path.exists():
        return pd.DataFrame(columns=["county_name", "state_abbrev", "fips"])

    df = pd.read_csv(path, dtype={"fips": str})
    # Normalize column names for robustness
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle common alternate headers (e.g., Census-style exports)
    if "fips state and county code" in df.columns and "fips" not in df.columns:
        df = df.rename(columns={"fips state and county code": "fips"})
    if "county name" in df.columns and "county_name" not in df.columns:
        df = df.rename(columns={"county name": "county_name"})
    if "state name" in df.columns and "state_abbrev" not in df.columns:
        # We'll convert state name -> state abbreviation below.
        df = df.rename(columns={"state name": "state_name"})

    needed = {"county_name", "fips"}
    missing = needed - set(df.columns)
    if missing:
        # Return empty but well-formed frame if user's file doesn't match expectations yet.
        return pd.DataFrame(columns=["county_name", "state_abbrev", "fips"])

    if "state_abbrev" not in df.columns and "state_name" not in df.columns:
        return pd.DataFrame(columns=["county_name", "state_abbrev", "fips"])

    df["county_name"] = df["county_name"].astype(str).str.strip()
    df["fips"] = df["fips"].astype(str).str.zfill(5).str.strip()

    # Normalize state information (supports either abbreviations or full state names)
    state_name_to_abbrev = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "district of columbia": "DC",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
    }

    if "state_abbrev" in df.columns:
        # If it's already a 2-letter code, keep it. If it's a full state name, convert it.
        cleaned = df["state_abbrev"].astype(str).str.strip()
        df["state_abbrev"] = cleaned.str.lower().map(state_name_to_abbrev).fillna(cleaned.str.upper())
    else:
        df["state_name"] = df["state_name"].astype(str).str.strip()
        df["state_abbrev"] = df["state_name"].str.lower().map(state_name_to_abbrev).fillna("")

    df = df.dropna(subset=["county_name", "state_abbrev", "fips"]).drop_duplicates()
    df = df.sort_values(["state_abbrev", "county_name"]).reset_index(drop=True)
    return df


def _county_display(row: pd.Series) -> str:
    return f"{row['county_name']}, {row['state_abbrev']} ({row['fips']})"


def _find_default_county_index(
    counties_df: pd.DataFrame,
    county_name: Optional[str],
    state_abbrev: Optional[str],
    fips: Optional[str],
) -> int:
    if counties_df.empty:
        return 0

    # Prefer exact FIPS match when available
    if fips:
        matches = counties_df.index[counties_df["fips"] == str(fips).zfill(5)].tolist()
        if matches:
            return int(matches[0])

    # Fall back to county + state match
    if county_name and state_abbrev:
        mask = (
            (counties_df["county_name"].str.lower() == str(county_name).strip().lower())
            & (counties_df["state_abbrev"].str.lower() == str(state_abbrev).strip().lower())
        )
        matches = counties_df.index[mask].tolist()
        if matches:
            return int(matches[0])

    return 0


# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------
st.set_page_config(page_title="Site Inputs", page_icon="🏭", layout="wide")

st.title("Site inputs")
st.caption("Define your facility and choose the county used for PRESTO outage simulations.")

existing = get_site() or {}
counties = load_counties()

# -------------------------------------------------------------------
# Demo autofill (testing convenience)
# -------------------------------------------------------------------
DEMO_SITE = {
    "facility_name": "Riverside Plastics Plant",
    "facility_type": "Manufacturing",
    "county_name": "Riverside County",
    "state_abbrev": "CA",
    "fips": "06065",
    "peak_demand_kw": 2500.0,
    "annual_energy_kwh": 12_000_000.0,
    "critical_load_kw": 500.0,
    "critical_annual_energy_kwh": 2_000_000.0,
    "analysis_period_years": 20,
    "discount_rate_nominal": 7.0,
}

demo_clicked = st.button("Load demo site data", help="Autofill this page with realistic sample values for testing.")
if demo_clicked:
    # Common fields
    st.session_state["site_facility_name"] = DEMO_SITE["facility_name"]
    st.session_state["site_facility_type"] = DEMO_SITE["facility_type"]
    st.session_state["site_peak_demand_kw"] = float(DEMO_SITE["peak_demand_kw"])
    st.session_state["site_annual_energy_kwh"] = float(DEMO_SITE["annual_energy_kwh"])
    st.session_state["site_critical_load_kw"] = float(DEMO_SITE["critical_load_kw"])
    st.session_state["site_critical_annual_energy_kwh"] = float(DEMO_SITE["critical_annual_energy_kwh"])
    st.session_state["site_analysis_period_years"] = int(DEMO_SITE["analysis_period_years"])
    st.session_state["site_discount_rate_nominal"] = float(DEMO_SITE["discount_rate_nominal"])

    # Location fields depend on whether counties.csv is present
    if counties.empty:
        st.session_state["site_manual_county_name"] = DEMO_SITE["county_name"]
        st.session_state["site_manual_state_abbrev"] = DEMO_SITE["state_abbrev"]
        st.session_state["site_manual_fips"] = DEMO_SITE["fips"]
    else:
        # Set the selectbox by its display string if the row exists
        match = counties[counties["fips"] == str(DEMO_SITE["fips"]).zfill(5)]
        if not match.empty:
            row = match.iloc[0]
            st.session_state["site_county_select"] = _county_display(row)
    st.rerun()

if counties.empty:
    st.info(
        "Counties reference file not found yet or not in the expected format. "
        "No problem: you can enter your county manually. "
        "You can also add app/data/reference/counties.csv later to enable a searchable dropdown."
    )

facility_type_options: List[str] = [
    "Manufacturing",
    "Data center",
    "Cold storage",
    "Warehouse / logistics",
    "Food and beverage",
    "Healthcare",
    "Office / commercial",
    "Other",
]

with st.form("site_form", clear_on_submit=False):
    st.subheader("Facility")

    facility_name = st.text_input(
        "Facility name (optional)",
        value=str(existing.get("facility_name", "")),
        placeholder="e.g., Riverside Plastics Plant",
        key="site_facility_name",
    )

    facility_type = st.selectbox(
        "Facility type",
        options=facility_type_options,
        index=max(0, facility_type_options.index(existing.get("facility_type", "Other")))
        if existing.get("facility_type", "Other") in facility_type_options
        else len(facility_type_options) - 1,
        help="Used to apply reasonable defaults later. You can override everything.",
        key="site_facility_type",
    )

    st.subheader("Location")

    default_idx = _find_default_county_index(
        counties_df=counties,
        county_name=existing.get("county_name"),
        state_abbrev=existing.get("state_abbrev"),
        fips=existing.get("fips"),
    )

    if counties.empty:
        manual_county_name = st.text_input(
            "County name",
            value=str(existing.get("county_name", "")),
            placeholder="e.g., Los Angeles County",
            key="site_manual_county_name",
        )
        manual_state_abbrev = st.text_input(
            "State (2-letter abbreviation)",
            value=str(existing.get("state_abbrev", "")),
            placeholder="e.g., CA",
            max_chars=2,
            key="site_manual_state_abbrev",
        )
        manual_fips = st.text_input(
            "County FIPS (optional, 5 digits)",
            value=str(existing.get("fips", "")),
            placeholder="e.g., 06037",
            key="site_manual_fips",
        )
        selected_row = None
    else:
        display_options = counties.apply(_county_display, axis=1).tolist()
        selected_display = st.selectbox(
            "County (for PRESTO simulations)",
            options=display_options,
            index=default_idx,
            key="site_county_select",
        )
        selected_row = counties.iloc[display_options.index(selected_display)]

    st.subheader("Electrical profile")

    peak_demand_kw = st.number_input(
        "Peak demand (kW)",
        min_value=0.0,
        value=float(existing.get("peak_demand_kw", 0.0) or 0.0),
        step=10.0,
        help="Your highest observed or expected kW demand.",
        key="site_peak_demand_kw",
    )

    annual_energy_kwh = st.number_input(
        "Annual energy (kWh)",
        min_value=0.0,
        value=float(existing.get("annual_energy_kwh", 0.0) or 0.0),
        step=10000.0,
        help="Total annual electricity consumption.",
        key="site_annual_energy_kwh",
    )

    critical_load_kw = st.number_input(
        "Critical load (kW)",
        min_value=0.0,
        value=float(existing.get("critical_load_kw", 0.0) or 0.0),
        step=10.0,
        help="The portion of peak load you must sustain during an outage.",
        key="site_critical_load_kw",
    )

    critical_annual_energy_kwh = st.number_input(
        "Critical annual energy (kWh)",
        min_value=0.0,
        value=float(existing.get("critical_annual_energy_kwh", 0.0) or 0.0),
        step=10000.0,
        help="A critical load may be housed in a building where critical functions are carried out, or associated with a piece of infrastructure, such as an onsite well pump, that supplies water to a critical function.",
        key="site_critical_annual_energy_kwh",
    )

    st.subheader("Financial assumptions")

    analysis_period_years = st.number_input(
        "Analysis period (years)",
        min_value=1,
        max_value=100,
        value=int(existing.get("analysis_period_years", 20)),
        step=1,
        key="site_analysis_period_years",
    )

    discount_rate_nominal = st.number_input(
        "Discount rate (nominal, %)",
        min_value=0.0,
        max_value=30.0,
        value=float(existing.get("discount_rate_nominal", 7.0)),
        step=0.5,
        help="The discount rate is a financial number used to convert future costs and benefits into an equivalent present value. Often times the discount rate is just the cost of borrowing money, sometimes called the cost of capital. This is because, if the cost of borrowing is 10%, then you can make money by borrowing today on any project which offers at least a 10% return.",
        key="site_discount_rate_nominal",
    )

    submitted = st.form_submit_button("Save site inputs")

if submitted:
    errors: List[str] = []

    if critical_load_kw > peak_demand_kw and peak_demand_kw > 0:
        errors.append("Critical load cannot exceed peak demand.")

    if critical_annual_energy_kwh > annual_energy_kwh and annual_energy_kwh > 0:
        errors.append("Critical annual energy cannot exceed facility annual energy.")

    if selected_row is None and not counties.empty:
        errors.append("County selection is required to run PRESTO simulations.")

    if counties.empty:
        if not str(manual_county_name).strip():
            errors.append("County name is required.")
        if not str(manual_state_abbrev).strip():
            errors.append("State abbreviation is required.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    if counties.empty:
        county_name = str(manual_county_name).strip()
        state_abbrev = str(manual_state_abbrev).strip().upper()
        fips = str(manual_fips).strip()
        if fips:
            fips = fips.zfill(5)
    else:
        county_name = str(selected_row["county_name"])
        state_abbrev = str(selected_row["state_abbrev"])
        fips = str(selected_row["fips"])

    payload = SiteInputs(
        facility_name=facility_name.strip(),
        county_name=county_name,
        state_abbrev=state_abbrev,
        fips=fips,
        facility_type=facility_type,
        peak_demand_kw=float(peak_demand_kw),
        annual_energy_kwh=float(annual_energy_kwh),
        critical_load_kw=float(critical_load_kw),
        critical_annual_energy_kwh=float(critical_annual_energy_kwh),
        analysis_period_years=int(analysis_period_years),
        discount_rate_nominal=float(discount_rate_nominal),
    )

    set_site(asdict(payload))
    st.success("Saved. Next: go to 'Outages (PRESTO)' in the left navigation.")

# Show current saved state for transparency
st.divider()
st.subheader("Saved site data")
st.json(get_site() or {})
