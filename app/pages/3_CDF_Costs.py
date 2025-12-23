
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from app.core.cdf.model import (
    CdfParameters,
    DamagedEquipmentCost,
    DowntimeCost,
    LostDataCost,
    MissedDeadlinesSpoilageCost,
    OtherFixedCost,
    OtherSpoilageCost,
    PerishableFoodSpoilageCost,
    ProcessInterruptionCost,
    RestartCost,
    fixed_cost_breakdown,
    outage_cost_fixed_only,
    spoilage_cost_breakdown,
    total_spoilage_cost,
    BackupFuelIncrementalCost,
    CustomerSalesIncrementalCost,
    InterruptedProductionIncrementalCost,
    OtherIncrementalCost,
    RentedEquipmentIncrementalCost,
    SafetyIncrementalCost,
    StaffProductivityIncrementalCost,
    incremental_cost_breakdown,
    total_incremental_cost,
    presto_event_duration_hours,
)
from app.ui.state import get_cdf_inputs, get_outage_events, get_site, set_cdf_inputs


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


def _pct_to_prob(pct: float) -> float:
    try:
        v = float(pct)
    except Exception:
        return 0.0
    if v < 0:
        v = 0.0
    if v > 100:
        v = 100.0
    return v / 100.0




def _prob_to_pct(prob: float) -> float:
    try:
        v = float(prob)
    except Exception:
        return 0.0
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    return v * 100.0


# Helper: Coerce Streamlit data_editor return value to DataFrame with specified columns.

def _coerce_editor_df(value: Any, *, columns: List[str]) -> pd.DataFrame:
    """Coerce Streamlit data_editor return types into a DataFrame.

    Streamlit may return:
      - a pandas DataFrame
      - a list[dict]
      - a dict-like structure
    We normalize to a DataFrame with the requested columns.
    """
    if isinstance(value, pd.DataFrame):
        df = value.copy()
    elif isinstance(value, list):
        df = pd.DataFrame(value)
    elif isinstance(value, dict):
        # Some Streamlit versions can return a dict of column->values
        try:
            df = pd.DataFrame(value)
        except Exception:
            df = pd.DataFrame([])
    else:
        df = pd.DataFrame([])

    for c in columns:
        if c not in df.columns:
            df[c] = "" if c == "name" else 0.0

    return df[columns]


# Helper: Robustly read a st.data_editor value (DataFrame, list[dict], or dict-style session_state)
def _read_data_editor_state_as_df(
    *,
    key: str,
    base_df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Read a st.data_editor value reliably, including Streamlit's dict-style session state.

    Depending on Streamlit version/config, `st.session_state[key]` may be:
      - a DataFrame
      - a list[dict]
      - a dict with `edited_rows`, `added_rows`, `deleted_rows`

    We reconstruct the final table against `base_df` and return a DataFrame with `columns`.
    """
    v = st.session_state.get(key, None)

    # Direct representations
    if isinstance(v, (pd.DataFrame, list, dict)) and not (
        isinstance(v, dict) and ("edited_rows" in v or "added_rows" in v or "deleted_rows" in v)
    ):
        return _coerce_editor_df(v, columns=columns)

    # Dict-style editor state: apply edits/additions/deletions to base_df
    if isinstance(v, dict) and ("edited_rows" in v or "added_rows" in v or "deleted_rows" in v):
        df = base_df.copy()

        # Ensure all requested columns exist
        for c in columns:
            if c not in df.columns:
                df[c] = "" if c == "name" else 0.0

        # Deletions
        deleted = v.get("deleted_rows")
        if isinstance(deleted, list) and deleted:
            try:
                df = df.drop(index=[int(i) for i in deleted if str(i).strip().isdigit()], errors="ignore")
            except Exception:
                pass

        # Edits
        edited = v.get("edited_rows")
        if isinstance(edited, dict):
            for idx_str, changes in edited.items():
                try:
                    idx = int(idx_str)
                except Exception:
                    continue
                if not isinstance(changes, dict):
                    continue
                if idx in df.index:
                    for c, new_val in changes.items():
                        if c in df.columns:
                            df.at[idx, c] = new_val

        # Additions
        added = v.get("added_rows")
        if isinstance(added, list) and added:
            try:
                add_df = pd.DataFrame([r for r in added if isinstance(r, dict)])
                for c in columns:
                    if c not in add_df.columns:
                        add_df[c] = "" if c == "name" else 0.0
                df = pd.concat([df, add_df[columns]], ignore_index=True)
            except Exception:
                pass

        return _coerce_editor_df(df, columns=columns)

    # Fallback: the base table
    return _coerce_editor_df(base_df, columns=columns)


def _extract_outage_events_list(outages_payload: Any) -> List[Any]:
    """Return the best-effort list of outage event dicts from the PRESTO session payload."""
    if outages_payload is None:
        return []

    # Some legacy sessions stored a bare list
    if isinstance(outages_payload, list):
        return outages_payload

    if not isinstance(outages_payload, dict):
        return []

    # Prefer normalized events
    tidy = outages_payload.get("events_tidy")
    if isinstance(tidy, list) and tidy:
        return tidy

    results = outages_payload.get("results")
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        events = results.get("events") or results.get("interruptions") or results.get("outages")
        if isinstance(events, list):
            return events

    # Last resort: top-level
    for key in ["events", "interruptions", "outages"]:
        v = outages_payload.get(key)
        if isinstance(v, list):
            return v

    return []


def _preview_outage_duration_hours(outages_payload: Dict[str, Any] | None) -> float:
    """Representative outage duration for previews.

    Uses the median duration across simulated events when available; otherwise defaults to 8 hours.
    """
    events = _extract_outage_events_list(outages_payload)
    durations: List[float] = []

    for e in events:
        if isinstance(e, dict):
            d = float(presto_event_duration_hours(e) or 0.0)
            if d > 0:
                durations.append(d)

    if not durations:
        return 8.0

    durations.sort()
    mid = len(durations) // 2
    return float(durations[mid]) if len(durations) % 2 == 1 else float((durations[mid - 1] + durations[mid]) / 2.0)


# Helper to ensure CDF input saves propagate across pages
def _save_cdf_inputs_and_rerun(updated: Dict[str, Any]) -> None:
    """Persist CDF inputs into session state and force a rerun.

    This prevents a common Streamlit gotcha where another page appears to be using stale inputs
    because it reran before the updated payload was the one referenced.
    """
    set_cdf_inputs(updated)
    # Touch a monotonic timestamp so other pages can use it as a cache-buster if they cache.
    st.session_state["cdf_inputs_last_saved_at"] = time.time()
    st.rerun()

# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------
st.set_page_config(page_title="CDF Costs", page_icon="💵", layout="wide")

st.title("CDF costs")
st.caption("Enter outage-related costs. Use the tabs to move between cost categories.")

site = get_site() or {}
if not site:
    st.warning("Please complete 'Site inputs' first.")
    st.stop()

outages = get_outage_events()
if outages is None:
    st.info("Outage simulations have not been generated yet. You can still enter costs now, then run PRESTO later.")
else:
    # This app is PRESTO-only; just sanity-check that events exist.
    events_for_preview = _extract_outage_events_list(outages)
    if not events_for_preview:
        st.warning("Outage payload is present but contains no events. Re-run PRESTO simulations on the Outage profile page.")

existing = get_cdf_inputs() or {}
existing_fixed: Dict[str, Any] = existing.get("fixed_costs", {})
existing_spoilage: Dict[str, Any] = existing.get("spoilage_costs", {})
existing_incremental: Dict[str, Any] = existing.get("incremental_costs", {})

tab_fixed, tab_spoilage, tab_incremental = st.tabs(["Fixed costs", "Spoilage costs", "Incremental costs"])


# -------------------------------------------------------------------
# Fixed costs tab
# -------------------------------------------------------------------
with tab_fixed:
    st.subheader("Fixed costs")
    st.caption("Fixed costs occur immediately when power is lost and are independent of outage duration.")

    # Demo autofill (testing convenience)
    DEMO_FIXED_COSTS = {
        "enabled": {
            "eq": True,
            "dt": True,
            "ld": True,
            "pi": True,
            "rs": True,
        },
        "eq": {"c": 25000.0, "n": 1.0, "p_pct": 15.0},
        "dt": {"c": 8000.0, "h": 4.0},
        "ld": {"v": 50000.0, "p_pct": 5.0},
        "pi": {"c": 85.0, "h": 6.0, "io": 25000.0},
        "rs": {"c": 85.0, "h": 3.0, "m": 5000.0},
        "other": [
            {"name": "QA scrap and rework", "cost": 12000.0},
            {"name": "Expedited shipping to recover schedule", "cost": 7500.0},
        ],
        "show_method": {"eq": False, "dt": False, "ld": False, "pi": False, "rs": False},
    }

    if st.button("Load demo fixed costs", help="Autofill fixed cost inputs with realistic sample values for testing."):
        st.session_state["cdf_costs_fc_eq_enabled"] = DEMO_FIXED_COSTS["enabled"]["eq"]
        st.session_state["cdf_costs_fc_eq_c"] = DEMO_FIXED_COSTS["eq"]["c"]
        st.session_state["cdf_costs_fc_eq_n"] = DEMO_FIXED_COSTS["eq"]["n"]
        st.session_state["cdf_costs_fc_eq_p"] = DEMO_FIXED_COSTS["eq"]["p_pct"]
        st.session_state["cdf_costs_fc_eq_method"] = DEMO_FIXED_COSTS["show_method"]["eq"]

        st.session_state["cdf_costs_fc_dt_enabled"] = DEMO_FIXED_COSTS["enabled"]["dt"]
        st.session_state["cdf_costs_fc_dt_c"] = DEMO_FIXED_COSTS["dt"]["c"]
        st.session_state["cdf_costs_fc_dt_h"] = DEMO_FIXED_COSTS["dt"]["h"]
        st.session_state["cdf_costs_fc_dt_method"] = DEMO_FIXED_COSTS["show_method"]["dt"]

        st.session_state["cdf_costs_fc_ld_enabled"] = DEMO_FIXED_COSTS["enabled"]["ld"]
        st.session_state["cdf_costs_fc_ld_v"] = DEMO_FIXED_COSTS["ld"]["v"]
        st.session_state["cdf_costs_fc_ld_p"] = DEMO_FIXED_COSTS["ld"]["p_pct"]
        st.session_state["cdf_costs_fc_ld_method"] = DEMO_FIXED_COSTS["show_method"]["ld"]

        st.session_state["cdf_costs_fc_pi_enabled"] = DEMO_FIXED_COSTS["enabled"]["pi"]
        st.session_state["cdf_costs_fc_pi_c"] = DEMO_FIXED_COSTS["pi"]["c"]
        st.session_state["cdf_costs_fc_pi_h"] = DEMO_FIXED_COSTS["pi"]["h"]
        st.session_state["cdf_costs_fc_pi_io"] = DEMO_FIXED_COSTS["pi"]["io"]
        st.session_state["cdf_costs_fc_pi_method"] = DEMO_FIXED_COSTS["show_method"]["pi"]

        st.session_state["cdf_costs_fc_rs_enabled"] = DEMO_FIXED_COSTS["enabled"]["rs"]
        st.session_state["cdf_costs_fc_rs_c"] = DEMO_FIXED_COSTS["rs"]["c"]
        st.session_state["cdf_costs_fc_rs_h"] = DEMO_FIXED_COSTS["rs"]["h"]
        st.session_state["cdf_costs_fc_rs_m"] = DEMO_FIXED_COSTS["rs"]["m"]
        st.session_state["cdf_costs_fc_rs_method"] = DEMO_FIXED_COSTS["show_method"]["rs"]

        st.session_state["cdf_costs_fc_other_seed"] = DEMO_FIXED_COSTS["other"]
        st.session_state["cdf_costs_fc_other_seed_active"] = True
        # Reset the data_editor state so the seeded rows render (data_editor caches by key)
        st.session_state.pop("cdf_costs_fc_other_editor", None)
        st.rerun()

    with st.form("cdf_fixed_costs_form", clear_on_submit=False):
        # 1) Damaged equipment
        st.markdown("### 1) Damaged equipment costs")
        has_equipment_damage = st.toggle(
            "Do you have any damaged equipment costs?",
            value=bool(existing_fixed.get("damaged_equipment", {}).get("enabled", False)),
            help="Power outages can cause machinery to stop abruptly, electronics to shut down improperly, and production processes to halt mid-process",
            key="cdf_costs_fc_eq_enabled",
        )
        eq_cols = st.columns(4)
        with eq_cols[0]:
            eq_c = st.number_input(
                "Average cost of equipment repair or replacement ($)",
                min_value=0.0,
                value=float(existing_fixed.get("damaged_equipment", {}).get("avg_cost", 0.0) or 0.0),
                step=1000.0,
                key="cdf_costs_fc_eq_c",
            )
        with eq_cols[1]:
            eq_n = st.number_input(
                "Number of pieces of equipment damaged",
                min_value=0.0,
                value=float(existing_fixed.get("damaged_equipment", {}).get("count", 0.0) or 0.0),
                step=1.0,
                key="cdf_costs_fc_eq_n",
            )
        with eq_cols[2]:
            eq_p_pct = st.number_input(
                "Probability of damage from outage (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(_prob_to_pct(existing_fixed.get("damaged_equipment", {}).get("probability", 0.0) or 0.0)),
                step=1.0,
                key="cdf_costs_fc_eq_p",
            )
        with eq_cols[3]:
            eq_cost = (
                DamagedEquipmentCost(
                    avg_repair_replacement_cost=eq_c,
                    number_of_pieces_damaged=eq_n,
                    probability_of_damage=_pct_to_prob(eq_p_pct),
                ).cost()
                if has_equipment_damage
                else 0.0
            )
            st.metric("Cost ($)", f"{eq_cost:,.0f}")

        show_eq_method = st.toggle(
            "Show methodology (equipment damage)",
            value=bool(existing_fixed.get("damaged_equipment", {}).get("show_method", False)),
            key="cdf_costs_fc_eq_method",
        )
        if has_equipment_damage and show_eq_method:
            st.info(
                "Equipment Damage Costs ($) = C x N x P\n\n"
                "C = average cost of equipment repair or replacement\n"
                "N = the number of pieces of equipment damaged\n"
                "P = probability of damage from outage"
            )

        st.divider()

        # 2) Downtime
        st.markdown("### 2) Downtime costs")
        has_downtime = st.toggle(
            "Do you have downtime costs?",
            value=bool(existing_fixed.get("downtime", {}).get("enabled", False)),
            help="A facility may not be operational immediately after power is restored",
            key="cdf_costs_fc_dt_enabled",
        )
        dt_cols = st.columns(3)
        with dt_cols[0]:
            dt_c = st.number_input(
                "Cost per hour facility is idle ($/hour)",
                min_value=0.0,
                value=float(existing_fixed.get("downtime", {}).get("cost_per_hour", 0.0) or 0.0),
                step=100.0,
                key="cdf_costs_fc_dt_c",
            )
        with dt_cols[1]:
            dt_h = st.number_input(
                "Hours of downtime",
                min_value=0.0,
                value=float(existing_fixed.get("downtime", {}).get("hours", 0.0) or 0.0),
                step=0.5,
                key="cdf_costs_fc_dt_h",
            )
        with dt_cols[2]:
            dt_cost = DowntimeCost(cost_per_hour_idle=dt_c, hours_of_downtime=dt_h).cost() if has_downtime else 0.0
            st.metric("Cost ($)", f"{dt_cost:,.0f}")

        show_dt_method = st.toggle(
            "Show methodology (downtime)",
            value=bool(existing_fixed.get("downtime", {}).get("show_method", False)),
            key="cdf_costs_fc_dt_method",
        )
        if has_downtime and show_dt_method:
            st.info(
                "Downtime Costs ($) = C x H\n\n"
                "C = cost per hour facility is idle\n"
                "H = hours of downtime"
            )

        st.divider()

        # 3) Lost data
        st.markdown("### 3) Lost data costs")
        has_lost_data = st.toggle(
            "Do you have any lost data costs?",
            value=bool(existing_fixed.get("lost_data", {}).get("enabled", False)),
            help="Power outages can cause computer data to be either lost or corrupted",
            key="cdf_costs_fc_ld_enabled",
        )
        ld_cols = st.columns(3)
        with ld_cols[0]:
            ld_v = st.number_input(
                "Value of stored data ($)",
                min_value=0.0,
                value=float(existing_fixed.get("lost_data", {}).get("value", 0.0) or 0.0),
                step=1000.0,
                key="cdf_costs_fc_ld_v",
            )
        with ld_cols[1]:
            ld_p_pct = st.number_input(
                "Probability of loss (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(_prob_to_pct(existing_fixed.get("lost_data", {}).get("probability", 0.0) or 0.0)),
                step=1.0,
                key="cdf_costs_fc_ld_p",
            )
        with ld_cols[2]:
            ld_cost = (
                LostDataCost(value_of_stored_data=ld_v, probability_of_loss=_pct_to_prob(ld_p_pct)).cost()
                if has_lost_data
                else 0.0
            )
            st.metric("Cost ($)", f"{ld_cost:,.0f}")

        show_ld_method = st.toggle(
            "Show methodology (lost data)",
            value=bool(existing_fixed.get("lost_data", {}).get("show_method", False)),
            key="cdf_costs_fc_ld_method",
        )
        if has_lost_data and show_ld_method:
            st.info(
                "Lost Data or Experiments ($) = V x P\n\n"
                "V = value of stored data\n"
                "P = probability of loss"
            )

        st.divider()

        # 4) Process interruption
        st.markdown("### 4) Process interruption costs")
        has_interruption = st.toggle(
            "Do you have any process interruption costs?",
            value=bool(existing_fixed.get("process_interruption", {}).get("enabled", False)),
            help="For processes which require continuous power, such as manufacturing lines or in-progress experiments, even short duration outages can result in additional costs",
            key="cdf_costs_fc_pi_enabled",
        )
        pi_cols = st.columns(4)
        with pi_cols[0]:
            pi_c = st.number_input(
                "Average fully-burdened hourly employee costs ($/hour)",
                min_value=0.0,
                value=float(existing_fixed.get("process_interruption", {}).get("hourly_cost", 0.0) or 0.0),
                step=10.0,
                key="cdf_costs_fc_pi_c",
            )
        with pi_cols[1]:
            pi_h = st.number_input(
                "Hours of staff time to reset process",
                min_value=0.0,
                value=float(existing_fixed.get("process_interruption", {}).get("hours", 0.0) or 0.0),
                step=0.5,
                key="cdf_costs_fc_pi_h",
            )
        with pi_cols[2]:
            pi_io = st.number_input(
                "Lost inputs and outputs ($) (optional)",
                min_value=0.0,
                value=float(existing_fixed.get("process_interruption", {}).get("io", 0.0) or 0.0),
                step=100.0,
                key="cdf_costs_fc_pi_io",
            )
        with pi_cols[3]:
            pi_cost = (
                ProcessInterruptionCost(
                    fully_burdened_hourly_employee_cost=pi_c,
                    hours_staff_time_to_reset=pi_h,
                    lost_inputs_outputs=pi_io,
                ).cost()
                if has_interruption
                else 0.0
            )
            st.metric("Cost ($)", f"{pi_cost:,.0f}")

        show_pi_method = st.toggle(
            "Show methodology (process interruption)",
            value=bool(existing_fixed.get("process_interruption", {}).get("show_method", False)),
            key="cdf_costs_fc_pi_method",
        )
        if has_interruption and show_pi_method:
            st.info(
                "Interruption Costs ($) = C x H + IO\n\n"
                "C = average fully-burdened hourly employee costs (wage plus overhead)\n"
                "H = hours of staff time to reset process\n"
                "IO = lost inputs and outputs"
            )

        st.divider()

        # 5) Restart
        st.markdown("### 5) Restart costs")
        has_restart = st.toggle(
            "Do you have any restart costs?",
            value=bool(existing_fixed.get("restart", {}).get("enabled", False)),
            help="Additional labor-hours and expenses may be required to restart a facility after a power outage",
            key="cdf_costs_fc_rs_enabled",
        )
        rs_cols = st.columns(4)
        with rs_cols[0]:
            rs_c = st.number_input(
                "Average fully-burdened hourly employee costs ($/hour)",
                min_value=0.0,
                value=float(existing_fixed.get("restart", {}).get("hourly_cost", 0.0) or 0.0),
                step=10.0,
                key="cdf_costs_fc_rs_c",
            )
        with rs_cols[1]:
            rs_h = st.number_input(
                "Hours of staff time required to restart",
                min_value=0.0,
                value=float(existing_fixed.get("restart", {}).get("hours", 0.0) or 0.0),
                step=0.5,
                key="cdf_costs_fc_rs_h",
            )
        with rs_cols[2]:
            rs_m = st.number_input(
                "Additional restart costs ($)",
                min_value=0.0,
                value=float(existing_fixed.get("restart", {}).get("additional_costs", 0.0) or 0.0),
                step=100.0,
                key="cdf_costs_fc_rs_m",
            )
        with rs_cols[3]:
            rs_cost = (
                RestartCost(
                    fully_burdened_hourly_employee_cost=rs_c,
                    hours_staff_time_to_restart=rs_h,
                    additional_restart_costs=rs_m,
                ).cost()
                if has_restart
                else 0.0
            )
            st.metric("Cost ($)", f"{rs_cost:,.0f}")

        show_rs_method = st.toggle(
            "Show methodology (restart)",
            value=bool(existing_fixed.get("restart", {}).get("show_method", False)),
            key="cdf_costs_fc_rs_method",
        )
        if has_restart and show_rs_method:
            st.info(
                "Restart Costs ($) = C x H + M\n\n"
                "C = average fully-burdened hourly employee costs (wage plus overhead)\n"
                "H = hours of staff time required to restart\n"
                "M = additional restart costs"
            )

        st.divider()

        # 6) Other fixed costs
        st.markdown("### 6) Other fixed costs")
        st.caption("Add any other one-time costs that happen immediately when power is lost.")

        other_existing = existing_fixed.get("other_fixed_costs", [])
        # If demo autofill was used this run, seed the editor rows from session state (allowed).
        if st.session_state.get("cdf_costs_fc_other_seed_active", False):
            other_existing = st.session_state.get("cdf_costs_fc_other_seed", other_existing)
            # Reset editor state so the seeded defaults display
            st.session_state.pop("cdf_costs_fc_other_editor", None)
            # Clear the flag so normal saved inputs take over after a save.
            st.session_state["cdf_costs_fc_other_seed_active"] = False
        if not isinstance(other_existing, list):
            other_existing = []

        # Normalize legacy rows that used `amount` into `cost`
        normalized_other: List[Dict[str, Any]] = []
        for row in other_existing:
            if isinstance(row, dict):
                name = row.get("name", "")
                if "cost" in row and row.get("cost") is not None:
                    cost = row.get("cost")
                else:
                    cost = row.get("amount", 0.0)
                normalized_other.append({"name": name, "cost": float(cost or 0.0)})
        if not normalized_other:
            normalized_other = [{"name": "", "cost": 0.0}]

        other_df = pd.DataFrame(normalized_other)
        if "name" not in other_df.columns:
            other_df["name"] = ""
        if "cost" not in other_df.columns:
            other_df["cost"] = 0.0

        edited_other = st.data_editor(
            other_df[["name", "cost"]],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name of cost"),
                "cost": st.column_config.NumberColumn("Cost ($)", min_value=0.0, step=100.0),
            },
            key="cdf_costs_fc_other_editor",
        )

        submitted = st.form_submit_button("Save fixed costs", use_container_width=True)

    if submitted:
        fixed_cost_objects: List[Any] = []
        fixed_payload: Dict[str, Any] = {}

        fixed_payload["damaged_equipment"] = {
            "enabled": bool(has_equipment_damage),
            "avg_cost": float(eq_c),
            "count": float(eq_n),
            "probability": float(_pct_to_prob(eq_p_pct)),
            "probability_pct": float(eq_p_pct),
            "show_method": bool(show_eq_method),
            "calculated_cost": float(eq_cost),
        }
        if has_equipment_damage:
            fixed_cost_objects.append(
                DamagedEquipmentCost(
                    avg_repair_replacement_cost=float(eq_c),
                    number_of_pieces_damaged=float(eq_n),
                    probability_of_damage=float(_pct_to_prob(eq_p_pct)),
                )
            )

        fixed_payload["downtime"] = {
            "enabled": bool(has_downtime),
            "cost_per_hour": float(dt_c),
            "hours": float(dt_h),
            "show_method": bool(show_dt_method),
            "calculated_cost": float(dt_cost),
        }
        if has_downtime:
            fixed_cost_objects.append(DowntimeCost(cost_per_hour_idle=float(dt_c), hours_of_downtime=float(dt_h)))

        fixed_payload["lost_data"] = {
            "enabled": bool(has_lost_data),
            "value": float(ld_v),
            "probability": float(_pct_to_prob(ld_p_pct)),
            "probability_pct": float(ld_p_pct),
            "show_method": bool(show_ld_method),
            "calculated_cost": float(ld_cost),
        }
        if has_lost_data:
            fixed_cost_objects.append(
                LostDataCost(value_of_stored_data=float(ld_v), probability_of_loss=float(_pct_to_prob(ld_p_pct)))
            )

        fixed_payload["process_interruption"] = {
            "enabled": bool(has_interruption),
            "hourly_cost": float(pi_c),
            "hours": float(pi_h),
            "io": float(pi_io),
            "show_method": bool(show_pi_method),
            "calculated_cost": float(pi_cost),
        }
        if has_interruption:
            fixed_cost_objects.append(
                ProcessInterruptionCost(
                    fully_burdened_hourly_employee_cost=float(pi_c),
                    hours_staff_time_to_reset=float(pi_h),
                    lost_inputs_outputs=float(pi_io),
                )
            )

        fixed_payload["restart"] = {
            "enabled": bool(has_restart),
            "hourly_cost": float(rs_c),
            "hours": float(rs_h),
            "additional_costs": float(rs_m),
            "show_method": bool(show_rs_method),
            "calculated_cost": float(rs_cost),
        }
        if has_restart:
            fixed_cost_objects.append(
                RestartCost(
                    fully_burdened_hourly_employee_cost=float(rs_c),
                    hours_staff_time_to_restart=float(rs_h),
                    additional_restart_costs=float(rs_m),
                )
            )

        other_items: List[Dict[str, Any]] = []
        # Reconstruct the final editor table reliably (handles dict-style editor state)
        other_df_saved = _read_data_editor_state_as_df(
            key="cdf_costs_fc_other_editor",
            base_df=other_df[["name", "cost"]],
            columns=["name", "cost"],
        )

        for _, row in other_df_saved.iterrows():
            name = str(row.get("name", "") or "").strip()
            # Handle NaN safely
            try:
                cost_val = row.get("cost", 0.0)
                cost = float(0.0 if pd.isna(cost_val) else cost_val)
            except Exception:
                cost = 0.0

            if name or (cost and cost > 0):
                other_items.append({"name": name, "cost": float(cost)})
                fixed_cost_objects.append(OtherFixedCost(name=name or "Other fixed cost", amount=float(cost)))

        fixed_payload["other_fixed_costs"] = other_items

        params = CdfParameters(fixed_costs=fixed_cost_objects)
        fixed_total = outage_cost_fixed_only(params)
        breakdown = fixed_cost_breakdown(fixed_cost_objects)

        new_cdf_inputs = dict(existing)
        new_cdf_inputs["fixed_costs"] = fixed_payload
        new_cdf_inputs["fixed_costs_summary"] = {
            "total_fixed_cost_per_outage": float(fixed_total),
            "breakdown": breakdown,
        }

        st.success("Saved fixed costs.")
        _save_cdf_inputs_and_rerun(new_cdf_inputs)


# -------------------------------------------------------------------
# Spoilage costs tab
# -------------------------------------------------------------------
with tab_spoilage:
    st.subheader("Spoilage costs")
    st.caption(
        "Spoilage costs are one-time losses when items spoil due to an outage. Once an item spoils, it will not spoil again in later periods."
    )

    preview_h = _preview_outage_duration_hours(outages if isinstance(outages, dict) else None)
    st.caption(f"Previewing spoilage costs at an outage duration of {preview_h:.1f} hours.")

    # Demo autofill (testing convenience)
    DEMO_SPOILAGE = {
        "enabled": {"food": True, "deadlines": True},
        "food": {"value": 40000.0, "start": 4.0, "end": 12.0, "half": 8.0},
        "deadlines": {"value": 20000.0},
        "other": [
            {"name": "Work-in-process material spoilage", "value": 60000.0, "end": 10.0, "half": 6.0},
        ],
    }

    if st.button("Load demo spoilage costs", help="Autofill spoilage cost inputs with realistic sample values for testing."):
        st.session_state["cdf_costs_sc_food_enabled"] = DEMO_SPOILAGE["enabled"]["food"]
        st.session_state["cdf_costs_sc_food_value"] = DEMO_SPOILAGE["food"]["value"]
        st.session_state["cdf_costs_sc_food_start"] = DEMO_SPOILAGE["food"]["start"]
        st.session_state["cdf_costs_sc_food_end"] = DEMO_SPOILAGE["food"]["end"]
        st.session_state["cdf_costs_sc_food_half"] = DEMO_SPOILAGE["food"]["half"]

        st.session_state["cdf_costs_sc_deadlines_enabled"] = DEMO_SPOILAGE["enabled"]["deadlines"]
        st.session_state["cdf_costs_sc_deadlines_value"] = DEMO_SPOILAGE["deadlines"]["value"]

        st.session_state["cdf_costs_sc_other_seed"] = DEMO_SPOILAGE["other"]
        st.session_state["cdf_costs_sc_other_seed_active"] = True
        # Reset the data_editor state so the seeded rows render (data_editor caches by key)
        st.session_state.pop("cdf_costs_sc_other_editor", None)
        st.rerun()

    with st.form("cdf_spoilage_costs_form", clear_on_submit=False):
        # 1) Perishable food
        st.markdown("### 1) Perishable food costs")
        has_food = st.toggle(
            "Do you have any perishable food costs?",
            value=bool(existing_spoilage.get("perishable_food", {}).get("enabled", False)),
            help="Extended power outages can cause refigerated or frozen foods to spoil",
            key="cdf_costs_sc_food_enabled",
        )

        food_cols = st.columns(4)
        with food_cols[0]:
            food_value = st.number_input(
                "Total value of spoilable product ($)",
                min_value=0.0,
                value=float(existing_spoilage.get("perishable_food", {}).get("value", 0.0) or 0.0),
                step=1000.0,
                key="cdf_costs_sc_food_value",
            )
        with food_cols[1]:
            food_start = st.number_input(
                "Hour spoilage begins",
                min_value=0.0,
                value=float(existing_spoilage.get("perishable_food", {}).get("start", 0.0) or 0.0),
                step=0.5,
                help=(
                    "This is the time that has elapsed since the beginning of the outage that the item begins spoiling. "
                    "Entering 0 means that the item begins spoiling immediately as an outage begins.\n\n"
                    "Example: Inputting 5 indicates that this item begins to spoil 5 hours after the beginning of the outage."
                ),
                key="cdf_costs_sc_food_start",
            )
        with food_cols[2]:
            food_end = st.number_input(
                "Hour of 100% spoilage",
                min_value=0.0,
                value=float(existing_spoilage.get("perishable_food", {}).get("end", 0.0) or 0.0),
                step=0.5,
                help=(
                    "This is the time that has elapsed since the beginning of the outage that the item stops spoiling.\n\n"
                    "Example: If you put 5 for the beginning of the spoilage, and 10 here, then the item will completely spoil after a 10 hour outage. "
                    "This linearly increases.\n\n"
                    "Continuing the example: If the value of the item is $500, on a 6 hour outage, $100 will have spoiled, a 7 hour outage, $200 will have spoiled and so on."
                ),
                key="cdf_costs_sc_food_end",
            )

        show_half = (food_end > food_start)
        food_half = (
            st.number_input(
                "Hour of 50% spoilage (optional)",
                min_value=0.0,
                value=float(existing_spoilage.get("perishable_food", {}).get("half", 0.0) or 0.0),
                step=0.5,
                help="This option is available once you've entered the start and end times for spoilage",
                key="cdf_costs_sc_food_half",
            )
            if show_half
            else 0.0
        )

        food_cost = (
            PerishableFoodSpoilageCost(
                total_value_spoilable_product=float(food_value),
                hour_spoilage_begins=float(food_start),
                hour_of_100_percent_spoilage=float(food_end),
                hour_of_50_percent_spoilage=float(food_half) if (food_half and food_half > 0) else None,
            ).cost(preview_h)
            if has_food
            else 0.0
        )
        st.metric("Preview cost ($)", f"{food_cost:,.0f}")

        st.divider()

        # 2) Missed deadlines
        st.markdown("### 2) Missed deadline costs")
        has_deadlines = st.toggle(
            "Do you have any costs due to missed deadlines?",
            value=bool(existing_spoilage.get("missed_deadlines", {}).get("enabled", False)),
            help="extended outages can cause expenses due to missing deadlines or goals",
            key="cdf_costs_sc_deadlines_enabled",
        )
        dl_cols = st.columns(2)
        with dl_cols[0]:
            dl_value = st.number_input(
                "Total value of missed deadline ($)",
                min_value=0.0,
                value=float(existing_spoilage.get("missed_deadlines", {}).get("value", 0.0) or 0.0),
                step=1000.0,
                key="cdf_costs_sc_deadlines_value",
            )
        with dl_cols[1]:
            dl_cost = (
                MissedDeadlinesSpoilageCost(total_value_missed_deadline=float(dl_value), trigger_hour=0.0).cost(preview_h)
                if has_deadlines
                else 0.0
            )
            st.metric("Preview cost ($)", f"{dl_cost:,.0f}")

        st.divider()

        # 3) Other spoilage costs
        st.markdown("### 3) Other spoilage costs")
        st.caption(
            "Add any other spoilable products. For these entries, spoilage is assumed to begin immediately (hour 0) and increase linearly to 100% at the provided hour."
        )

        other_existing = existing_spoilage.get("other_spoilage_costs", [])
        if st.session_state.get("cdf_costs_sc_other_seed_active", False):
            other_existing = st.session_state.get("cdf_costs_sc_other_seed", other_existing)
            # Reset editor state so the seeded defaults display
            st.session_state.pop("cdf_costs_sc_other_editor", None)
            st.session_state["cdf_costs_sc_other_seed_active"] = False

        if not isinstance(other_existing, list):
            other_existing = []
        if not other_existing:
            other_existing = [{"name": "", "value": 0.0, "end": 0.0, "half": 0.0}]

        other_df = pd.DataFrame(other_existing)
        for col, default in [("name", ""), ("value", 0.0), ("end", 0.0), ("half", 0.0)]:
            if col not in other_df.columns:
                other_df[col] = default

        edited_other = st.data_editor(
            other_df[["name", "value", "end", "half"]],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name of cost"),
                "value": st.column_config.NumberColumn("Total value of spoilable product ($)", min_value=0.0, step=1000.0),
                "end": st.column_config.NumberColumn(
                    "Hour of 100% spoilage",
                    min_value=0.0,
                    step=0.5,
                    help=(
                        "This is the time that has elapsed since the beginning of the outage that the item stops spoiling.\n\n"
                        "Example: If you put 5 for the beginning of the spoilage, and 10 here, then the item will completely spoil after a 10 hour outage. "
                        "This linearly increases.\n\n"
                        "Continuing the example: If the value of the item is $500, on a 6 hour outage, $100 will have spoiled, a 7 hour outage, $200 will have spoiled and so on."
                    ),
                ),
                "half": st.column_config.NumberColumn(
                    "Hour of 50% spoilage (optional)",
                    min_value=0.0,
                    step=0.5,
                    help="This option is available once you've entered the start and end times for spoilage",
                ),
            },
            key="cdf_costs_sc_other_editor",
        )

        submitted = st.form_submit_button("Save spoilage costs", use_container_width=True)

    if submitted:
        spoilage_objects: List[Any] = []
        spoilage_payload: Dict[str, Any] = {}

        spoilage_payload["perishable_food"] = {
            "enabled": bool(has_food),
            "value": float(food_value),
            "start": float(food_start),
            "end": float(food_end),
            "half": float(food_half) if (food_half and food_half > 0) else 0.0,
            "preview_cost": float(food_cost),
        }
        if has_food:
            spoilage_objects.append(
                PerishableFoodSpoilageCost(
                    total_value_spoilable_product=float(food_value),
                    hour_spoilage_begins=float(food_start),
                    hour_of_100_percent_spoilage=float(food_end),
                    hour_of_50_percent_spoilage=float(food_half) if (food_half and food_half > 0) else None,
                )
            )

        spoilage_payload["missed_deadlines"] = {
            "enabled": bool(has_deadlines),
            "value": float(dl_value),
            "preview_cost": float(dl_cost),
        }
        if has_deadlines:
            spoilage_objects.append(MissedDeadlinesSpoilageCost(total_value_missed_deadline=float(dl_value), trigger_hour=0.0))

        other_items: List[Dict[str, Any]] = []
        other_df_saved = _read_data_editor_state_as_df(
            key="cdf_costs_sc_other_editor",
            base_df=other_df[["name", "value", "end", "half"]],
            columns=["name", "value", "end", "half"],
        )

        for _, row in other_df_saved.iterrows():
            name = str(row.get("name", "") or "").strip()
            def _f(col: str) -> float:
                try:
                    v = row.get(col, 0.0)
                    return float(0.0 if pd.isna(v) else v)
                except Exception:
                    return 0.0

            value = _f("value")
            end = _f("end")
            half = _f("half")

            if name or value or end or half:
                other_items.append({"name": name, "value": float(value), "end": float(end), "half": float(half)})
                spoilage_objects.append(
                    OtherSpoilageCost(
                        name=name or "Other spoilage cost",
                        total_value_spoilable_product=float(value),
                        hour_spoilage_begins=0.0,
                        hour_of_100_percent_spoilage=float(end),
                        hour_of_50_percent_spoilage=float(half) if half > 0 else None,
                    )
                )

        spoilage_payload["other_spoilage_costs"] = other_items

        spoilage_total_preview = total_spoilage_cost(spoilage_objects, preview_h)
        spoilage_breakdown_preview = spoilage_cost_breakdown(spoilage_objects, preview_h)

        new_cdf_inputs = dict(existing)
        new_cdf_inputs["spoilage_costs"] = spoilage_payload
        new_cdf_inputs["spoilage_costs_summary"] = {
            "preview_outage_duration_hours": float(preview_h),
            "total_spoilage_cost_preview": float(spoilage_total_preview),
            "breakdown_preview": spoilage_breakdown_preview,
        }

        st.success("Saved spoilage costs.")
        _save_cdf_inputs_and_rerun(new_cdf_inputs)

    st.divider()
    st.subheader("Saved spoilage-cost summary")
    saved = get_cdf_inputs() or {}
    sc_summary = saved.get("spoilage_costs_summary", {})
    if not sc_summary:
        st.info("No spoilage costs saved yet.")
    else:
        st.metric("Total spoilage cost (preview) ($)", f"{float(sc_summary.get('total_spoilage_cost_preview', 0.0)):,.0f}")
        st.caption(f"Preview outage duration: {float(sc_summary.get('preview_outage_duration_hours', 0.0)):.1f} hours")
        bd = sc_summary.get("breakdown_preview", [])
        if isinstance(bd, list) and bd:
            st.dataframe(pd.DataFrame(bd), use_container_width=True)
            

# -------------------------------------------------------------------
# Incremental costs tab
# -------------------------------------------------------------------
with tab_incremental:
    st.subheader("Incremental costs")
    st.caption(
        "Incremental costs are incurred due to lost opportunities and additional costs that accumulate for each hour the power is out "
        "(e.g., lost staff productivity, loss of manufacturing production, backup generator fuel costs, or lost communications)."
    )

    preview_h = _preview_outage_duration_hours(outages if isinstance(outages, dict) else None)
    st.caption(f"Previewing incremental costs at an outage duration of {preview_h:.1f} hours.")

    # Demo autofill (testing convenience)
    DEMO_INCREMENTAL = {
        "enabled": {
            "bf": True,
            "re": True,
            "cs": True,
            "ip": True,
            "sp": True,
            "sf": False,
        },
        "bf": {
            "gl": 350.0,
            "fe": 13.0,
            "fc": 5.25,
            "a": 50.0,
            "start": 0.0,
            "mit": 24.0,
            "wh": 120.0,
            "show": False,
        },
        "re": {"hc": 250.0, "start": 2.0, "mit": 24.0, "wh": 120.0},
        "cs": {"wr": 500000.0, "pct": 8.0, "start": 0.0, "mit": 24.0, "show": False},
        "ip": {"no": 40.0, "pct": 60.0, "ov": 350.0, "start": 0.0, "mit": 24.0, "wh": 120.0, "show": False},
        "sp": {"c": 85.0, "pct": 70.0, "n": 25.0, "start": 0.0, "mit": 12.0, "wh": 120.0, "show": False},
        "sf": {"n": 0.0, "p": 0.0, "start": 0.0, "mit": 0.0, "wh": 0.0, "show": False},
        "other": [{"name": "Temporary comms / IT support", "hourly_cost": 150.0, "start": 0.0, "mit": 24.0, "wh": 120.0}],
    }

    if st.button("Load demo incremental costs", help="Autofill incremental cost inputs with realistic sample values for testing."):
        st.session_state["cdf_costs_ic_bf_enabled"] = DEMO_INCREMENTAL["enabled"]["bf"]
        st.session_state["cdf_costs_ic_bf_gl"] = DEMO_INCREMENTAL["bf"]["gl"]
        st.session_state["cdf_costs_ic_bf_fe"] = DEMO_INCREMENTAL["bf"]["fe"]
        st.session_state["cdf_costs_ic_bf_fc"] = DEMO_INCREMENTAL["bf"]["fc"]
        st.session_state["cdf_costs_ic_bf_a"] = DEMO_INCREMENTAL["bf"]["a"]
        st.session_state["cdf_costs_ic_bf_start"] = DEMO_INCREMENTAL["bf"]["start"]
        st.session_state["cdf_costs_ic_bf_mit"] = DEMO_INCREMENTAL["bf"]["mit"]
        st.session_state["cdf_costs_ic_bf_wh"] = DEMO_INCREMENTAL["bf"]["wh"]
        st.session_state["cdf_costs_ic_bf_method"] = DEMO_INCREMENTAL["bf"]["show"]

        st.session_state["cdf_costs_ic_re_enabled"] = DEMO_INCREMENTAL["enabled"]["re"]
        st.session_state["cdf_costs_ic_re_hc"] = DEMO_INCREMENTAL["re"]["hc"]
        st.session_state["cdf_costs_ic_re_start"] = DEMO_INCREMENTAL["re"]["start"]
        st.session_state["cdf_costs_ic_re_mit"] = DEMO_INCREMENTAL["re"]["mit"]
        st.session_state["cdf_costs_ic_re_wh"] = DEMO_INCREMENTAL["re"]["wh"]

        st.session_state["cdf_costs_ic_cs_enabled"] = DEMO_INCREMENTAL["enabled"]["cs"]
        st.session_state["cdf_costs_ic_cs_wr"] = DEMO_INCREMENTAL["cs"]["wr"]
        st.session_state["cdf_costs_ic_cs_pct"] = DEMO_INCREMENTAL["cs"]["pct"]
        st.session_state["cdf_costs_ic_cs_start"] = DEMO_INCREMENTAL["cs"]["start"]
        st.session_state["cdf_costs_ic_cs_mit"] = DEMO_INCREMENTAL["cs"]["mit"]
        st.session_state["cdf_costs_ic_cs_method"] = DEMO_INCREMENTAL["cs"]["show"]

        st.session_state["cdf_costs_ic_ip_enabled"] = DEMO_INCREMENTAL["enabled"]["ip"]
        st.session_state["cdf_costs_ic_ip_no"] = DEMO_INCREMENTAL["ip"]["no"]
        st.session_state["cdf_costs_ic_ip_pct"] = DEMO_INCREMENTAL["ip"]["pct"]
        st.session_state["cdf_costs_ic_ip_ov"] = DEMO_INCREMENTAL["ip"]["ov"]
        st.session_state["cdf_costs_ic_ip_start"] = DEMO_INCREMENTAL["ip"]["start"]
        st.session_state["cdf_costs_ic_ip_mit"] = DEMO_INCREMENTAL["ip"]["mit"]
        st.session_state["cdf_costs_ic_ip_wh"] = DEMO_INCREMENTAL["ip"]["wh"]
        st.session_state["cdf_costs_ic_ip_method"] = DEMO_INCREMENTAL["ip"]["show"]

        st.session_state["cdf_costs_ic_sp_enabled"] = DEMO_INCREMENTAL["enabled"]["sp"]
        st.session_state["cdf_costs_ic_sp_c"] = DEMO_INCREMENTAL["sp"]["c"]
        st.session_state["cdf_costs_ic_sp_pct"] = DEMO_INCREMENTAL["sp"]["pct"]
        st.session_state["cdf_costs_ic_sp_n"] = DEMO_INCREMENTAL["sp"]["n"]
        st.session_state["cdf_costs_ic_sp_start"] = DEMO_INCREMENTAL["sp"]["start"]
        st.session_state["cdf_costs_ic_sp_mit"] = DEMO_INCREMENTAL["sp"]["mit"]
        st.session_state["cdf_costs_ic_sp_wh"] = DEMO_INCREMENTAL["sp"]["wh"]
        st.session_state["cdf_costs_ic_sp_method"] = DEMO_INCREMENTAL["sp"]["show"]

        st.session_state["cdf_costs_ic_sf_enabled"] = DEMO_INCREMENTAL["enabled"]["sf"]
        st.session_state["cdf_costs_ic_sf_n"] = DEMO_INCREMENTAL["sf"]["n"]
        st.session_state["cdf_costs_ic_sf_p"] = DEMO_INCREMENTAL["sf"]["p"]
        st.session_state["cdf_costs_ic_sf_start"] = DEMO_INCREMENTAL["sf"]["start"]
        st.session_state["cdf_costs_ic_sf_mit"] = DEMO_INCREMENTAL["sf"]["mit"]
        st.session_state["cdf_costs_ic_sf_wh"] = DEMO_INCREMENTAL["sf"]["wh"]
        st.session_state["cdf_costs_ic_sf_method"] = DEMO_INCREMENTAL["sf"]["show"]

        st.session_state["cdf_costs_ic_other_seed"] = DEMO_INCREMENTAL["other"]
        st.session_state["cdf_costs_ic_other_seed_active"] = True
        # Reset the data_editor state so the seeded rows render (data_editor caches by key)
        st.session_state.pop("cdf_costs_ic_other_editor", None)
        st.rerun()

    with st.form("cdf_incremental_costs_form", clear_on_submit=False):
        # 1) Backup fuel costs
        st.markdown("### 1) Backup fuel costs")
        has_bf = st.toggle(
            "Do you have any backup fuel costs?",
            value=bool(existing_incremental.get("backup_fuel", {}).get("enabled", False)),
            key="cdf_costs_ic_bf_enabled",
        )
        bf_cols = st.columns(5)
        with bf_cols[0]:
            bf_gl = st.number_input("Generator load (kW)", min_value=0.0, value=float(existing_incremental.get("backup_fuel", {}).get("gl", 0.0) or 0.0), step=10.0, key="cdf_costs_ic_bf_gl")
        with bf_cols[1]:
            bf_fe = st.number_input(
                "Fuel efficiency (kWh/gallon)",
                min_value=0.0,
                value=float(existing_incremental.get("backup_fuel", {}).get("fe", 13.0) or 13.0),
                step=0.5,
                help="The default of 13 kWh per gallon is 1 / burn rate (0.076)",
                key="cdf_costs_ic_bf_fe",
            )
        with bf_cols[2]:
            bf_fc = st.number_input("Cost of fuel ($/gallon)", min_value=0.0, value=float(existing_incremental.get("backup_fuel", {}).get("fc", 0.0) or 0.0), step=0.25, key="cdf_costs_ic_bf_fc")
        with bf_cols[3]:
            bf_a = st.number_input("Additional hourly backup system costs ($/hour) (optional)", min_value=0.0, value=float(existing_incremental.get("backup_fuel", {}).get("a", 0.0) or 0.0), step=10.0, key="cdf_costs_ic_bf_a")
        with bf_cols[4]:
            bf_hourly = BackupFuelIncrementalCost(
                generator_load_kw=float(bf_gl),
                fuel_efficiency_kwh_per_gallon=float(bf_fe),
                fuel_cost_per_gallon=float(bf_fc),
                additional_hourly_backup_costs=float(bf_a),
            ).hourly_cost()
            st.metric("Hourly cost ($/hour)", f"{(bf_hourly if has_bf else 0.0):,.0f}")

        bf_time_cols = st.columns(3)
        with bf_time_cols[0]:
            bf_start = st.number_input(
                "Hour cost starts (optional)",
                min_value=0.0,
                value=float(existing_incremental.get("backup_fuel", {}).get("start", 0.0) or 0.0),
                step=0.5,
                help="This is the time that has elapsed since the beginning of the outage that the item begins incurring it's cost.",
                key="cdf_costs_ic_bf_start",
            )
        with bf_time_cols[1]:
            bf_mit = st.number_input(
                "Hour cost is mitigated",
                min_value=0.0,
                value=float(existing_incremental.get("backup_fuel", {}).get("mit", 0.0) or 0.0),
                step=0.5,
                help=(
                    "This is the time that has elapsed since the beginning of the outage that the item STOPS incurring it's cost.\n\n"
                    "Example: If you provide 10 here and set the hour the cost starts as 5, the cost will continue to apply until an outage of 10 hours is reached, at which point this cost will flatten to its maximum value."
                ),
                key="cdf_costs_ic_bf_mit",
            )
        with bf_time_cols[2]:
            bf_wh = st.number_input(
                "Avg. weekly business hours (optional)",
                min_value=0.0,
                max_value=168.0,
                value=float(existing_incremental.get("backup_fuel", {}).get("wh", 0.0) or 0.0),
                step=1.0,
                help="The hours in a week that this cost would apply. This is applied as a fractional multiplier for your cost.",
                key="cdf_costs_ic_bf_wh",
            )

        show_bf_method = st.toggle(
            "Show methodology (backup fuel)",
            value=bool(existing_incremental.get("backup_fuel", {}).get("show_method", False)),
            key="cdf_costs_ic_bf_method",
        )
        if has_bf and show_bf_method:
            st.info(
                "Backup Fuel Cost ($/hour) = GL x 1/FE x FC + A\n\n"
                "Generator Load (GL) = average load on the generator [kW]\n"
                "Fuel Efficiency (FE) = average kWh of energy generated per gallon of fuel burned [kWh/gallon]\n"
                "Fuel Cost (FC) = cost of fuel [$/gallon]\n"
                "A = additional hourly backup system costs"
            )

        bf_preview = (
            BackupFuelIncrementalCost(
                generator_load_kw=float(bf_gl),
                fuel_efficiency_kwh_per_gallon=float(bf_fe),
                fuel_cost_per_gallon=float(bf_fc),
                additional_hourly_backup_costs=float(bf_a),
                hour_cost_starts=float(bf_start) if bf_start > 0 else None,
                hour_cost_mitigated=float(bf_mit) if bf_mit > 0 else None,
                avg_weekly_business_hours=float(bf_wh) if bf_wh > 0 else None,
            ).cost(preview_h)
            if has_bf
            else 0.0
        )
        st.metric("Preview cost ($)", f"{bf_preview:,.0f}")

        st.divider()

        # 2) Rental equipment costs
        st.markdown("### 2) Rental equipment costs")
        has_re = st.toggle(
            "Do you have any rental equipment costs related to a loss of power?",
            value=bool(existing_incremental.get("rented_equipment", {}).get("enabled", False)),
            key="cdf_costs_ic_re_enabled",
        )
        re_cols = st.columns(4)
        with re_cols[0]:
            re_hc = st.number_input("Hourly cost ($/hour)", min_value=0.0, value=float(existing_incremental.get("rented_equipment", {}).get("hc", 0.0) or 0.0), step=10.0, key="cdf_costs_ic_re_hc")
        with re_cols[1]:
            re_start = st.number_input("Hourly cost starts (optional)", min_value=0.0, value=float(existing_incremental.get("rented_equipment", {}).get("start", 0.0) or 0.0), step=0.5, help="This is the time that has elapsed since the beginning of the outage that the item begins incurring it's cost.", key="cdf_costs_ic_re_start")
        with re_cols[2]:
            re_mit = st.number_input(
                "Hour cost is mitigated (optional)",
                min_value=0.0,
                value=float(existing_incremental.get("rented_equipment", {}).get("mit", 0.0) or 0.0),
                step=0.5,
                help=(
                    "This is the time that has elapsed since the beginning of the outage that the item STOPS incurring it's cost.\n\n"
                    "Example: If you provide 10 here and set the hour the cost starts as 5, the cost will continue to apply until an outage of 10 hours is reached, at which point this cost will flatten to its maximum value."
                ),
                key="cdf_costs_ic_re_mit",
            )
        with re_cols[3]:
            re_wh = st.number_input(
                "Avg. weekly business hours (optional)",
                min_value=0.0,
                max_value=168.0,
                value=float(existing_incremental.get("rented_equipment", {}).get("wh", 0.0) or 0.0),
                step=1.0,
                help="The hours in a week that this cost would apply. This is applied as a fractional multiplier for your cost.",
                key="cdf_costs_ic_re_wh",
            )

        re_preview = (
            RentedEquipmentIncrementalCost(
                hourly_cost=float(re_hc),
                hour_cost_starts=float(re_start) if re_start > 0 else None,
                hour_cost_mitigated=float(re_mit) if re_mit > 0 else None,
                avg_weekly_business_hours=float(re_wh) if re_wh > 0 else None,
            ).cost(preview_h)
            if has_re
            else 0.0
        )
        st.metric("Preview cost ($)", f"{re_preview:,.0f}")

        st.divider()

        # 3) Lost customer sales
        st.markdown("### 3) Lost customer sales")
        has_cs = st.toggle(
            "Do you have any costs due to lost customer sales?",
            value=bool(existing_incremental.get("customer_sales", {}).get("enabled", False)),
            key="cdf_costs_ic_cs_enabled",
        )

        cs_cols = st.columns(4)
        with cs_cols[0]:
            cs_wr = st.number_input("Weekly revenues ($/week)", min_value=0.0, value=float(existing_incremental.get("customer_sales", {}).get("wr", 0.0) or 0.0), step=10000.0, key="cdf_costs_ic_cs_wr")
        with cs_cols[1]:
            cs_pct = st.number_input("Percentage decrease in sales (%)", min_value=0.0, max_value=100.0, value=float(existing_incremental.get("customer_sales", {}).get("pct", 0.0) or 0.0), step=1.0, key="cdf_costs_ic_cs_pct")
        with cs_cols[2]:
            cs_start = st.number_input("Hour cost starts (optional)", min_value=0.0, value=float(existing_incremental.get("customer_sales", {}).get("start", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_cs_start")
        with cs_cols[3]:
            cs_mit = st.number_input("Hour cost is mitigated (optional)", min_value=0.0, value=float(existing_incremental.get("customer_sales", {}).get("mit", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_cs_mit")

        show_cs_method = st.toggle(
            "Show methodology (customer sales)",
            value=bool(existing_incremental.get("customer_sales", {}).get("show_method", False)),
            key="cdf_costs_ic_cs_method",
        )
        if has_cs and show_cs_method:
            st.info(
                "Lost Customer Sales = WR x P\n\n"
                "WR = weekly revenues\n"
                "P = percent decrease in sales"
            )

        cs_obj = CustomerSalesIncrementalCost(
            weekly_revenues=float(cs_wr),
            percent_decrease_in_sales=_pct_to_prob(float(cs_pct)),
            hour_cost_starts=float(cs_start) if cs_start > 0 else None,
            hour_cost_mitigated=float(cs_mit) if cs_mit > 0 else None,
            avg_weekly_business_hours=None,
        )
        cs_hourly = cs_obj.hourly_cost()
        st.metric("Hourly cost ($/hour)", f"{(cs_hourly if has_cs else 0.0):,.0f}")

        cs_preview = cs_obj.cost(preview_h) if has_cs else 0.0
        st.metric("Preview cost ($)", f"{cs_preview:,.0f}")

        st.divider()

        # 4) Interrupted production
        st.markdown("### 4) Interrupted production")
        has_ip = st.toggle(
            "Do you have any costs caused by interrupted production?",
            value=bool(existing_incremental.get("interrupted_production", {}).get("enabled", False)),
            key="cdf_costs_ic_ip_enabled",
        )

        ip_cols = st.columns(6)
        with ip_cols[0]:
            ip_no = st.number_input("Units of output per hour", min_value=0.0, value=float(existing_incremental.get("interrupted_production", {}).get("no", 0.0) or 0.0), step=1.0, key="cdf_costs_ic_ip_no")
        with ip_cols[1]:
            ip_pct = st.number_input("Percent reduction in output due to outage (%)", min_value=0.0, max_value=100.0, value=float(existing_incremental.get("interrupted_production", {}).get("pct", 0.0) or 0.0), step=1.0, key="cdf_costs_ic_ip_pct")
        with ip_cols[2]:
            ip_ov = st.number_input("Dollar value per unit of output ($)", min_value=0.0, value=float(existing_incremental.get("interrupted_production", {}).get("ov", 0.0) or 0.0), step=10.0, key="cdf_costs_ic_ip_ov")
        with ip_cols[3]:
            ip_start = st.number_input("Hour cost starts (optional)", min_value=0.0, value=float(existing_incremental.get("interrupted_production", {}).get("start", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_ip_start")
        with ip_cols[4]:
            ip_mit = st.number_input("Hour cost is mitigated (optional)", min_value=0.0, value=float(existing_incremental.get("interrupted_production", {}).get("mit", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_ip_mit")
        with ip_cols[5]:
            ip_wh = st.number_input(
                "Avg. weekly business hours (optional)",
                min_value=0.0,
                max_value=168.0,
                value=float(existing_incremental.get("interrupted_production", {}).get("wh", 0.0) or 0.0),
                step=1.0,
                help="The hours in a week that this cost would apply. This is applied as a fractional multiplier for your cost.",
                key="cdf_costs_ic_ip_wh",
            )

        show_ip_method = st.toggle(
            "Show methodology (interrupted production)",
            value=bool(existing_incremental.get("interrupted_production", {}).get("show_method", False)),
            key="cdf_costs_ic_ip_method",
        )
        if has_ip and show_ip_method:
            st.info(
                "Lost Output ($/hour) = NO x P x OV\n\n"
                "Normal Output (NO) = units of output per hour during normal conditions\n"
                "P = percent reduction in output due to outage\n"
                "Outage Value (OV) = dollar value per unit of output"
            )

        ip_obj = InterruptedProductionIncrementalCost(
            normal_output_units_per_hour=float(ip_no),
            percent_reduction_output=_pct_to_prob(float(ip_pct)),
            outage_value_per_unit=float(ip_ov),
            hour_cost_starts=float(ip_start) if ip_start > 0 else None,
            hour_cost_mitigated=float(ip_mit) if ip_mit > 0 else None,
            avg_weekly_business_hours=float(ip_wh) if ip_wh > 0 else None,
        )
        ip_hourly = ip_obj.hourly_cost()
        st.metric("Hourly cost ($/hour)", f"{(ip_hourly if has_ip else 0.0):,.0f}")
        ip_preview = ip_obj.cost(preview_h) if has_ip else 0.0
        st.metric("Preview cost ($)", f"{ip_preview:,.0f}")

        st.divider()

        # 5) Staff productivity
        st.markdown("### 5) Staff productivity")
        has_sp = st.toggle(
            "Do you have any costs from loss of staff productivity?",
            value=bool(existing_incremental.get("staff_productivity", {}).get("enabled", False)),
            key="cdf_costs_ic_sp_enabled",
        )

        sp_cols = st.columns(6)
        with sp_cols[0]:
            sp_c = st.number_input("Average fully-burdened hourly employee costs ($/hour)", min_value=0.0, value=float(existing_incremental.get("staff_productivity", {}).get("c", 0.0) or 0.0), step=5.0, key="cdf_costs_ic_sp_c")
        with sp_cols[1]:
            sp_pct = st.number_input("Percent of work that cannot be completed (%)", min_value=0.0, max_value=100.0, value=float(existing_incremental.get("staff_productivity", {}).get("pct", 0.0) or 0.0), step=1.0, key="cdf_costs_ic_sp_pct")
        with sp_cols[2]:
            sp_n = st.number_input("Number of people affected", min_value=0.0, value=float(existing_incremental.get("staff_productivity", {}).get("n", 0.0) or 0.0), step=1.0, key="cdf_costs_ic_sp_n")
        with sp_cols[3]:
            sp_start = st.number_input("Hour cost starts (optional)", min_value=0.0, value=float(existing_incremental.get("staff_productivity", {}).get("start", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_sp_start")
        with sp_cols[4]:
            sp_mit = st.number_input("Hour cost is mitigated (optional)", min_value=0.0, value=float(existing_incremental.get("staff_productivity", {}).get("mit", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_sp_mit")
        with sp_cols[5]:
            sp_wh = st.number_input(
                "Avg. weekly business hours (optional)",
                min_value=0.0,
                max_value=168.0,
                value=float(existing_incremental.get("staff_productivity", {}).get("wh", 0.0) or 0.0),
                step=1.0,
                help="The hours in a week that this cost would apply. This is applied as a fractional multiplier for your cost.",
                key="cdf_costs_ic_sp_wh",
            )

        show_sp_method = st.toggle(
            "Show methodology (staff productivity)",
            value=bool(existing_incremental.get("staff_productivity", {}).get("show_method", False)),
            key="cdf_costs_ic_sp_method",
        )
        if has_sp and show_sp_method:
            st.info(
                "Outage Productivity Costs ($/hour) = C x P x N\n\n"
                "C = average fully-burdened hourly employee costs (wage plus overhead)\n"
                "P = percent of work that cannot be completed during outage\n"
                "N = the number of people affected"
            )

        sp_obj = StaffProductivityIncrementalCost(
            fully_burdened_hourly_employee_cost=float(sp_c),
            percent_work_not_completed=_pct_to_prob(float(sp_pct)),
            number_people_affected=float(sp_n),
            hour_cost_starts=float(sp_start) if sp_start > 0 else None,
            hour_cost_mitigated=float(sp_mit) if sp_mit > 0 else None,
            avg_weekly_business_hours=float(sp_wh) if sp_wh > 0 else None,
        )
        sp_hourly = sp_obj.hourly_cost()
        st.metric("Hourly cost ($/hour)", f"{(sp_hourly if has_sp else 0.0):,.0f}")
        sp_preview = sp_obj.cost(preview_h) if has_sp else 0.0
        st.metric("Preview cost ($)", f"{sp_preview:,.0f}")

        st.divider()

        # 6) Safety
        st.markdown("### 6) Safety (injuries or lives lost)")
        st.caption(
            "While it is difficult to place a value on impaired health and safety or loss of life, the following sources provide example values: "
            "EPA’s value of statistical life (2016): $10M; Median wrongful death jury award (2009-2013): $2.2M; Median 9-11 settlement (2003): $1.7M; "
            "Average life-insurance policy face value (2015): $160k; Average worth of U.S. household (2013): $80,039."
        )

        has_sf = st.toggle(
            "Are there any costs associated with injuries or lives lost?",
            value=bool(existing_incremental.get("safety", {}).get("enabled", False)),
            key="cdf_costs_ic_sf_enabled",
        )

        sf_cols = st.columns(5)
        with sf_cols[0]:
            sf_n = st.number_input("Number of people", min_value=0.0, value=float(existing_incremental.get("safety", {}).get("n", 0.0) or 0.0), step=1.0, key="cdf_costs_ic_sf_n")
        with sf_cols[1]:
            sf_p = st.number_input("Hourly cost of reduced safety per person ($/hour)", min_value=0.0, value=float(existing_incremental.get("safety", {}).get("p", 0.0) or 0.0), step=10.0, key="cdf_costs_ic_sf_p")
        with sf_cols[2]:
            sf_start = st.number_input("Hour cost starts (optional)", min_value=0.0, value=float(existing_incremental.get("safety", {}).get("start", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_sf_start")
        with sf_cols[3]:
            sf_mit = st.number_input("Hour cost is mitigated (optional)", min_value=0.0, value=float(existing_incremental.get("safety", {}).get("mit", 0.0) or 0.0), step=0.5, key="cdf_costs_ic_sf_mit")
        with sf_cols[4]:
            sf_wh = st.number_input(
                "Avg. weekly business hours (optional)",
                min_value=0.0,
                max_value=168.0,
                value=float(existing_incremental.get("safety", {}).get("wh", 0.0) or 0.0),
                step=1.0,
                help="The hours in a week that this cost would apply. This is applied as a fractional multiplier for your cost.",
                key="cdf_costs_ic_sf_wh",
            )

        show_sf_method = st.toggle(
            "Show methodology (safety)",
            value=bool(existing_incremental.get("safety", {}).get("show_method", False)),
            key="cdf_costs_ic_sf_method",
        )
        if has_sf and show_sf_method:
            st.info(
                "Cost of Safety ($/hour) = N x P\n\n"
                "N = the number of people\n"
                "P = hourly cost of reduced safety per person"
            )

        sf_obj = SafetyIncrementalCost(
            number_people=float(sf_n),
            hourly_cost_per_person=float(sf_p),
            hour_cost_starts=float(sf_start) if sf_start > 0 else None,
            hour_cost_mitigated=float(sf_mit) if sf_mit > 0 else None,
            avg_weekly_business_hours=float(sf_wh) if sf_wh > 0 else None,
        )
        sf_hourly = sf_obj.hourly_cost()
        st.metric("Hourly cost ($/hour)", f"{(sf_hourly if has_sf else 0.0):,.0f}")
        sf_preview = sf_obj.cost(preview_h) if has_sf else 0.0
        st.metric("Preview cost ($)", f"{sf_preview:,.0f}")

        st.divider()

        # 7) Other incremental costs
        st.markdown("### 7) Other incremental costs")
        st.caption("Add any other hourly costs associated with a loss of power.")

        other_existing = existing_incremental.get("other_incremental_costs", [])
        if st.session_state.get("cdf_costs_ic_other_seed_active", False):
            other_existing = st.session_state.get("cdf_costs_ic_other_seed", other_existing)
            # Reset editor state so the seeded defaults display
            st.session_state.pop("cdf_costs_ic_other_editor", None)
            st.session_state["cdf_costs_ic_other_seed_active"] = False
        if not isinstance(other_existing, list):
            other_existing = []
        if not other_existing:
            other_existing = [{"name": "", "hourly_cost": 0.0, "start": 0.0, "mit": 0.0, "wh": 0.0}]

        other_df = pd.DataFrame(other_existing)
        for col, default in [("name", ""), ("hourly_cost", 0.0), ("start", 0.0), ("mit", 0.0), ("wh", 0.0)]:
            if col not in other_df.columns:
                other_df[col] = default

        edited_other = st.data_editor(
            other_df[["name", "hourly_cost", "start", "mit", "wh"]],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name of cost"),
                "hourly_cost": st.column_config.NumberColumn("Hourly cost ($/hour)", min_value=0.0, step=10.0),
                "start": st.column_config.NumberColumn(
                    "Hour cost starts (optional)",
                    min_value=0.0,
                    step=0.5,
                    help="This is the time that has elapsed since the beginning of the outage that the item begins incurring it's cost.",
                ),
                "mit": st.column_config.NumberColumn(
                    "Hour cost is mitigated (optional)",
                    min_value=0.0,
                    step=0.5,
                    help=(
                        "This is the time that has elapsed since the beginning of the outage that the item STOPS incurring it's cost.\n\n"
                        "Example: If you provide 10 here and set the hour the cost starts as 5, the cost will continue to apply until an outage of 10 hours is reached, at which point this cost will flatten to its maximum value."
                    ),
                ),
                "wh": st.column_config.NumberColumn(
                    "Avg. weekly business hours (optional)",
                    min_value=0.0,
                    max_value=168.0,
                    step=1.0,
                    help="The hours in a week that this cost would apply. This is applied as a fractional multiplier for your cost.",
                ),
            },
            key="cdf_costs_ic_other_editor",
        )

        submitted = st.form_submit_button("Save incremental costs", use_container_width=True)

    if submitted:
        incremental_objects: List[Any] = []
        incremental_payload: Dict[str, Any] = {}

        # Backup fuel
        incremental_payload["backup_fuel"] = {
            "enabled": bool(has_bf),
            "gl": float(bf_gl),
            "fe": float(bf_fe),
            "fc": float(bf_fc),
            "a": float(bf_a),
            "start": float(bf_start),
            "mit": float(bf_mit),
            "wh": float(bf_wh),
            "show_method": bool(show_bf_method),
        }
        if has_bf:
            incremental_objects.append(
                BackupFuelIncrementalCost(
                    generator_load_kw=float(bf_gl),
                    fuel_efficiency_kwh_per_gallon=float(bf_fe),
                    fuel_cost_per_gallon=float(bf_fc),
                    additional_hourly_backup_costs=float(bf_a),
                    hour_cost_starts=float(bf_start) if bf_start > 0 else None,
                    hour_cost_mitigated=float(bf_mit) if bf_mit > 0 else None,
                    avg_weekly_business_hours=float(bf_wh) if bf_wh > 0 else None,
                )
            )

        # Rented equipment
        incremental_payload["rented_equipment"] = {
            "enabled": bool(has_re),
            "hc": float(re_hc),
            "start": float(re_start),
            "mit": float(re_mit),
            "wh": float(re_wh),
        }
        if has_re:
            incremental_objects.append(
                RentedEquipmentIncrementalCost(
                    hourly_cost=float(re_hc),
                    hour_cost_starts=float(re_start) if re_start > 0 else None,
                    hour_cost_mitigated=float(re_mit) if re_mit > 0 else None,
                    avg_weekly_business_hours=float(re_wh) if re_wh > 0 else None,
                )
            )

        # Customer sales
        incremental_payload["customer_sales"] = {
            "enabled": bool(has_cs),
            "wr": float(cs_wr),
            "pct": float(cs_pct),
            "start": float(cs_start),
            "mit": float(cs_mit),
            "show_method": bool(show_cs_method),
        }
        if has_cs:
            incremental_objects.append(
                CustomerSalesIncrementalCost(
                    weekly_revenues=float(cs_wr),
                    percent_decrease_in_sales=_pct_to_prob(float(cs_pct)),
                    hour_cost_starts=float(cs_start) if cs_start > 0 else None,
                    hour_cost_mitigated=float(cs_mit) if cs_mit > 0 else None,
                    avg_weekly_business_hours=None,
                )
            )

        # Interrupted production
        incremental_payload["interrupted_production"] = {
            "enabled": bool(has_ip),
            "no": float(ip_no),
            "pct": float(ip_pct),
            "ov": float(ip_ov),
            "start": float(ip_start),
            "mit": float(ip_mit),
            "wh": float(ip_wh),
            "show_method": bool(show_ip_method),
        }
        if has_ip:
            incremental_objects.append(
                InterruptedProductionIncrementalCost(
                    normal_output_units_per_hour=float(ip_no),
                    percent_reduction_output=_pct_to_prob(float(ip_pct)),
                    outage_value_per_unit=float(ip_ov),
                    hour_cost_starts=float(ip_start) if ip_start > 0 else None,
                    hour_cost_mitigated=float(ip_mit) if ip_mit > 0 else None,
                    avg_weekly_business_hours=float(ip_wh) if ip_wh > 0 else None,
                )
            )

        # Staff productivity
        incremental_payload["staff_productivity"] = {
            "enabled": bool(has_sp),
            "c": float(sp_c),
            "pct": float(sp_pct),
            "n": float(sp_n),
            "start": float(sp_start),
            "mit": float(sp_mit),
            "wh": float(sp_wh),
            "show_method": bool(show_sp_method),
        }
        if has_sp:
            incremental_objects.append(
                StaffProductivityIncrementalCost(
                    fully_burdened_hourly_employee_cost=float(sp_c),
                    percent_work_not_completed=_pct_to_prob(float(sp_pct)),
                    number_people_affected=float(sp_n),
                    hour_cost_starts=float(sp_start) if sp_start > 0 else None,
                    hour_cost_mitigated=float(sp_mit) if sp_mit > 0 else None,
                    avg_weekly_business_hours=float(sp_wh) if sp_wh > 0 else None,
                )
            )

        # Safety
        incremental_payload["safety"] = {
            "enabled": bool(has_sf),
            "n": float(sf_n),
            "p": float(sf_p),
            "start": float(sf_start),
            "mit": float(sf_mit),
            "wh": float(sf_wh),
            "show_method": bool(show_sf_method),
        }
        if has_sf:
            incremental_objects.append(
                SafetyIncrementalCost(
                    number_people=float(sf_n),
                    hourly_cost_per_person=float(sf_p),
                    hour_cost_starts=float(sf_start) if sf_start > 0 else None,
                    hour_cost_mitigated=float(sf_mit) if sf_mit > 0 else None,
                    avg_weekly_business_hours=float(sf_wh) if sf_wh > 0 else None,
                )
            )

        # Other incremental
        other_items: List[Dict[str, Any]] = []
        other_df_saved = _read_data_editor_state_as_df(
            key="cdf_costs_ic_other_editor",
            base_df=other_df[["name", "hourly_cost", "start", "mit", "wh"]],
            columns=["name", "hourly_cost", "start", "mit", "wh"],
        )

        for _, row in other_df_saved.iterrows():
            name = str(row.get("name", "") or "").strip()
            def _f(col: str) -> float:
                try:
                    v = row.get(col, 0.0)
                    return float(0.0 if pd.isna(v) else v)
                except Exception:
                    return 0.0

            hc = _f("hourly_cost")
            start = _f("start")
            mit = _f("mit")
            wh = _f("wh")

            if name or hc or start or mit or wh:
                other_items.append({"name": name, "hourly_cost": float(hc), "start": float(start), "mit": float(mit), "wh": float(wh)})
                incremental_objects.append(
                    OtherIncrementalCost(
                        name=name or "Other incremental cost",
                        hourly_cost=float(hc),
                        hour_cost_starts=float(start) if start > 0 else None,
                        hour_cost_mitigated=float(mit) if mit > 0 else None,
                        avg_weekly_business_hours=float(wh) if wh > 0 else None,
                    )
                )
        incremental_payload["other_incremental_costs"] = other_items

        inc_total_preview = total_incremental_cost(incremental_objects, preview_h)
        inc_breakdown_preview = incremental_cost_breakdown(incremental_objects, preview_h)

        new_cdf_inputs = dict(existing)
        new_cdf_inputs["incremental_costs"] = incremental_payload
        new_cdf_inputs["incremental_costs_summary"] = {
            "preview_outage_duration_hours": float(preview_h),
            "total_incremental_cost_preview": float(inc_total_preview),
            "breakdown_preview": inc_breakdown_preview,
        }

        st.success("Saved incremental costs.")
        _save_cdf_inputs_and_rerun(new_cdf_inputs)

    st.divider()
    st.subheader("Saved incremental-cost summary")
    saved = get_cdf_inputs() or {}
    ic_summary = saved.get("incremental_costs_summary", {})
    if not ic_summary:
        st.info("No incremental costs saved yet.")
    else:
        st.metric("Total incremental cost (preview) ($)", f"{float(ic_summary.get('total_incremental_cost_preview', 0.0)):,.0f}")
        st.caption(f"Preview outage duration: {float(ic_summary.get('preview_outage_duration_hours', 0.0)):.1f} hours")
        bd = ic_summary.get("breakdown_preview", [])
        if isinstance(bd, list) and bd:
            st.dataframe(pd.DataFrame(bd), use_container_width=True)


# -------------------------------------------------------------------
# Current saved summary (fixed costs)
# -------------------------------------------------------------------
st.divider()
st.subheader("Saved fixed-cost summary")

saved = get_cdf_inputs() or {}
summary = saved.get("fixed_costs_summary", {})

if not summary:
    st.info("No fixed costs saved yet.")
else:
    st.metric("Total fixed cost per outage ($)", f"{float(summary.get('total_fixed_cost_per_outage', 0.0)):,.0f}")
    bd = summary.get("breakdown", [])
    if isinstance(bd, list) and bd:
        st.dataframe(pd.DataFrame(bd), use_container_width=True)

st.divider()
st.subheader("Saved CDF inputs (current session)")
st.json(get_cdf_inputs() or {})