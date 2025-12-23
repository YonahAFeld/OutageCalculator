import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from app.core.cdf.results import (
    run_outage_cost_analysis,
    coerce_outages_payload,
    safe_results_keys,
)


def _safe_page_link(path: str, label: str, icon: str = "") -> None:
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:
        st.write(f"{icon} {label}: {path}")


def _get_state_value() -> tuple[dict, dict | None, dict]:
    site: dict = {}
    outages_payload = None
    cdf_inputs: dict = {}

    try:
        from app.ui.state import get_site, get_outage_events, get_cdf_inputs  # type: ignore
        site = get_site() or {}
        outages_payload = get_outage_events()
        cdf_inputs = get_cdf_inputs() or {}
    except Exception:
        site = st.session_state.get("site", {}) or {}
        outages_payload = (
            st.session_state.get("outage_events")
            or st.session_state.get("presto_outages")
            or st.session_state.get("outages")
        )
        cdf_inputs = st.session_state.get("cdf_inputs", {}) or {}

    return site, outages_payload, cdf_inputs


def _fmt_money(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"


def _cdf_cost_from_curve(duration_hours: float, cdf_cost_curve: list[dict]) -> float:
    if not cdf_cost_curve:
        return 0.0
    dh = max(float(duration_hours), 0.0)
    curve_sorted = sorted(cdf_cost_curve, key=lambda r: float(r.get("duration_hours", 0.0)))
    for row in curve_sorted:
        if float(row.get("duration_hours", 0.0)) >= dh:
            return float(row.get("total", 0.0))
    return float(curve_sorted[-1].get("total", 0.0))


def _plot_cdf_curve(cdf_cost_curve: list[dict], title: str, vlines_hours: dict[str, float] | None = None) -> None:
    if not cdf_cost_curve:
        st.info("No CDF cost curve available to plot.")
        return

    curve_sorted = sorted(cdf_cost_curve, key=lambda r: float(r.get("duration_hours", 0.0)))
    x = np.array([float(r.get("duration_hours", 0.0)) for r in curve_sorted], dtype=float)
    y = np.array([float(r.get("total", 0.0)) / 1000.0 for r in curve_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("Outage duration (hours)")
    ax.set_ylabel("Outage cost ($ thousands)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, axis="both", alpha=0.25)

    if vlines_hours:
        ymax = float(np.max(y)) if y.size else 0.0
        for lab, dh in vlines_hours.items():
            ax.axvline(float(dh), linestyle="--", linewidth=1)
            ax.text(float(dh), ymax * 0.98 if ymax > 0 else 0.0, lab, rotation=90, va="top", ha="right")

    st.pyplot(fig)


def _extract_presto_durations_hours(results: dict) -> np.ndarray:
    eby = results.get("event_costs_by_year", {}) or {}
    durations: list[float] = []
    for events in eby.values():
        for ev in events or []:
            if isinstance(ev, dict) and "duration_hours" in ev:
                try:
                    durations.append(float(ev["duration_hours"]))
                except Exception:
                    pass
    return np.array(durations, dtype=float)


st.set_page_config(page_title="Results", layout="wide")

st.title("Results")
st.caption("Displays outage-cost results using your CDF inputs and PRESTO outage simulations.")

site, outages_payload, cdf_inputs = _get_state_value()
outages_payload = coerce_outages_payload(outages_payload)

if not cdf_inputs:
    st.error("No CDF inputs found yet. Please complete the Costs page first.")
    _safe_page_link("app/pages/3_CDF_Costs.py", label="Go to CDF Costs", icon="🧾")
    st.stop()

if not outages_payload:
    st.error("No outage profile found yet. Please run or load outage events first.")
    _safe_page_link("app/pages/2_Outages_PRESTO.py", label="Go to Outages", icon="⚡")
    st.stop()

results = run_outage_cost_analysis(
    cdf_inputs=cdf_inputs,
    outages_payload=outages_payload,
)

meta = results.get("meta", {}) or {}
cdf_curve = results.get("cdf_cost_curve", []) or []
durations = _extract_presto_durations_hours(results)

# ------------------- UI -------------------

st.subheader("Single-outage cost")
st.caption("Change the outage duration to see the estimated cost from your CDF curve.")

max_h = max([float(r.get("duration_hours", 0.0)) for r in cdf_curve], default=24.0)
sel_hours = st.number_input("Outage duration (hours)", min_value=0.0, max_value=float(max_h), value=2.0, step=0.25)
sel_cost = _cdf_cost_from_curve(sel_hours, cdf_curve)

st.markdown(f"### If a single outage lasts **{sel_hours:.2f} hours**, it will cost my facility **{_fmt_money(sel_cost)}**.")

st.info("Mental model: if an outage lasts X hours, what does it cost this facility?")

st.subheader("CDF curve")
_plot_cdf_curve(cdf_curve, "CDF curve: outage duration vs cost")

st.subheader("CDF curve with PRESTO duration percentiles")
st.markdown(
    """
    **What these percentiles mean**

    PRESTO simulates thousands of possible outage events. Each percentile line shows how long outages last
    in that simulated universe:

    - **P25**: 25% of simulated outages end *before* this duration (shorter outages are common).
    - **P50 (median)**: Half of outages are shorter and half are longer.
    - **P75**: Only 25% of outages last longer than this duration.
    - **P95**: Severe but plausible events — only 5% of outages exceed this length.

    When these vertical lines intersect the CDF cost curve, they translate outage *duration risk* into
    **financial exposure**.
    """
)
if durations.size:
    p25, p50, p75, p95 = np.percentile(durations, [25, 50, 75, 95])
    cols = st.columns(4)
    cols[0].metric("P25", f"{p25:.2f} h")
    cols[1].metric("P50", f"{p50:.2f} h")
    cols[2].metric("P75", f"{p75:.2f} h")
    cols[3].metric("P95", f"{p95:.2f} h")
    _plot_cdf_curve(cdf_curve, "CDF + PRESTO percentiles", {"P25": p25, "P50": p50, "P75": p75, "P95": p95})
else:
    st.info("No PRESTO outage durations found.")

with st.expander("Debug", expanded=False):
    st.write("meta:", meta)
    for k in [
        "has_spoilage_payload",
        "has_incremental_payload",
        "n_fixed_objects",
        "n_spoilage_objects",
        "n_incremental_objects",
        "curve_point_2h",
    ]:
        if k in meta:
            st.write(f"{k}:", meta[k])

    st.write(
        {
            "cdf_inputs_has_spoilage_costs": bool(cdf_inputs.get("spoilage_costs")),
            "cdf_inputs_has_incremental_costs": bool(cdf_inputs.get("incremental_costs")),
            "cdf_inputs_keys": sorted(list(cdf_inputs.keys())),
        }
    )

    st.write("# of PRESTO events:", int(durations.size))