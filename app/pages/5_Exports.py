

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Prefer project state getters if available; fall back to st.session_state
try:
    from app.ui.state import get_site, get_outage_events  # type: ignore
except Exception:  # pragma: no cover
    get_site = None  # type: ignore
    get_outage_events = None  # type: ignore


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_json(obj: Any) -> str:
    """Serialize to JSON with best-effort handling."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return json.dumps({"unserializable": True, "repr": repr(obj)}, indent=2)


def _get_site_payload() -> Dict[str, Any]:
    if callable(get_site):
        site = get_site() or {}
    else:
        site = st.session_state.get("site") or {}
    if not isinstance(site, dict):
        site = {}
    return site


def _get_outage_payload() -> Any:
    if callable(get_outage_events):
        return get_outage_events()
    return st.session_state.get("outage_events")


def _get_first_existing_key(keys: List[str]) -> Tuple[Optional[str], Any]:
    for k in keys:
        if k in st.session_state:
            return k, st.session_state.get(k)
    return None, None


def _get_cdf_payloads() -> Dict[str, Any]:
    """Best-effort extraction of CDF inputs + curve from session state.

    The project has evolved; this function tries multiple known keys.
    """
    out: Dict[str, Any] = {}

    # Common keys used across iterations
    k_inputs, v_inputs = _get_first_existing_key(
        [
            "cdf_inputs",
            "cdf_cost_inputs",
            "saved_cdf_inputs",
            "cdf",
            "cdf_payload",
        ]
    )
    if v_inputs is not None:
        out["cdf_inputs_key"] = k_inputs
        out["cdf_inputs"] = v_inputs

    k_curve, v_curve = _get_first_existing_key(
        [
            "cdf_cost_curve",
            "cdf_curve",
            "cdf_costs_curve",
            "cdf_cost_curve",
        ]
    )
    if v_curve is not None:
        out["cdf_curve_key"] = k_curve
        out["cdf_curve"] = v_curve

    # Some versions store curve inside a results payload
    if "cdf_curve" not in out:
        k_results, v_results = _get_first_existing_key(["cdf_results", "results", "results_payload"])
        if isinstance(v_results, dict):
            curve = v_results.get("cdf_cost_curve") or v_results.get("cdf_curve")
            if curve is not None:
                out["cdf_curve_key"] = f"{k_results}.(cdf_cost_curve)"
                out["cdf_curve"] = curve

    return out


def _get_results_payloads() -> Dict[str, Any]:
    """Best-effort extraction of results outputs from session state."""
    out: Dict[str, Any] = {}

    k_res, v_res = _get_first_existing_key(
        [
            "cdf_results",
            "results",
            "results_payload",
            "analysis_results",
        ]
    )
    if v_res is not None:
        out["results_key"] = k_res
        out["results"] = v_res

    # Event-level rows
    k_ev, v_ev = _get_first_existing_key(
        [
            "results_by_event",
            "event_costs",
            "events_priced",
        ]
    )
    if v_ev is not None:
        out["results_by_event_key"] = k_ev
        out["results_by_event"] = v_ev

    # Year-level rows
    k_yr, v_yr = _get_first_existing_key(
        [
            "results_by_year",
            "results_by_sim_year",
            "annual_costs",
        ]
    )
    if v_yr is not None:
        out["results_by_sim_year_key"] = k_yr
        out["results_by_sim_year"] = v_yr

    # Summary row/table
    k_sum, v_sum = _get_first_existing_key(
        [
            "results_summary",
            "summary",
            "results_topline",
        ]
    )
    if v_sum is not None:
        out["results_summary_key"] = k_sum
        out["results_summary"] = v_sum

    return out


def _to_dataframe(obj: Any) -> pd.DataFrame:
    """Convert common payload shapes to a dataframe."""
    if obj is None:
        return pd.DataFrame([])

    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, list):
        # list of dicts
        if len(obj) == 0:
            return pd.DataFrame([])
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return pd.DataFrame({"value": obj})

    if isinstance(obj, dict):
        # common: {"results": [...]} or {"events_tidy": [...]}
        if isinstance(obj.get("results"), list):
            return pd.DataFrame(obj["results"])  # type: ignore[index]
        if isinstance(obj.get("events_tidy"), list):
            return pd.DataFrame(obj["events_tidy"])  # type: ignore[index]
        # single record
        return pd.DataFrame([obj])

    # scalar
    return pd.DataFrame([{"value": obj}])


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _readme_text(
    *,
    site: Dict[str, Any],
    outage_payload: Any,
    cdf_payloads: Dict[str, Any],
    results_payloads: Dict[str, Any],
) -> str:
    fips = str(site.get("fips", "")).zfill(5) if site.get("fips") is not None else ""
    county = site.get("county_name") or ""
    state = site.get("state_abbrev") or ""

    # Pull scenario metadata if present
    scenario = ""
    mepy = ""
    sims = ""
    if isinstance(outage_payload, dict):
        meta = outage_payload.get("meta") if isinstance(outage_payload.get("meta"), dict) else {}
        scenario = str(meta.get("scenario") or "")
        mepy = str(meta.get("mean_events_per_year") or "")
        sims = str(meta.get("simulations") or "")

    lines = []
    lines.append("OutageCost Analysis Bundle")
    lines.append("=")
    lines.append("")
    lines.append(f"Exported at (UTC): {_now_utc_iso()}")
    if fips or county or state:
        lines.append(f"Location: {county}, {state} (FIPS {fips})")
    if scenario:
        lines.append(f"Scenario: {scenario}")
    if mepy:
        lines.append(f"Events/year used: {mepy}")
    if sims:
        lines.append(f"Simulated years: {sims}")
    lines.append("")
    lines.append("What is included")
    lines.append("-")
    lines.append("- site.json / site.csv: facility location inputs")
    lines.append("- reliability_*.csv and reliability_raw.json: historical reliability metrics from PRESTO")
    lines.append("- presto_events.csv and presto_raw.json: simulated outage events from PRESTO")
    lines.append("- cdf_inputs.json and cdf_curve.csv: your outage cost model (duration -> $)")
    lines.append("- results_*.csv: financial outputs derived by pricing simulated outages through the CDF")
    lines.append("")
    lines.append("Notes")
    lines.append("-")
    lines.append("- Reliability units are shown as provided by PRESTO. In the PRESTO UI, SAIDI/CAIDI are typically displayed in hours.")
    lines.append("- This bundle is meant to be shareable with finance/ops as an audit trail of assumptions and outputs.")

    # Debug keys (helps you later)
    if cdf_payloads.get("cdf_inputs_key") or cdf_payloads.get("cdf_curve_key"):
        lines.append("")
        lines.append("Internal keys")
        lines.append("-")
        if cdf_payloads.get("cdf_inputs_key"):
            lines.append(f"- CDF inputs key: {cdf_payloads.get('cdf_inputs_key')}")
        if cdf_payloads.get("cdf_curve_key"):
            lines.append(f"- CDF curve key: {cdf_payloads.get('cdf_curve_key')}")
        if results_payloads.get("results_key"):
            lines.append(f"- Results key: {results_payloads.get('results_key')}")

    return "\n".join(lines) + "\n"


def _build_zip_bundle() -> Tuple[bytes, Dict[str, Any]]:
    site = _get_site_payload()
    outage_payload = _get_outage_payload()
    cdf_payloads = _get_cdf_payloads()
    results_payloads = _get_results_payloads()

    # Reliability pieces live inside the outage payload (new structure)
    reliability_monthly_by_year = None
    reliability_raw = None
    reliability_summary_df = pd.DataFrame([])

    if isinstance(outage_payload, dict):
        reliability_monthly_by_year = outage_payload.get("historical_yearly_tidy")
        reliability_raw = outage_payload.get("historical_by_year")

        # Annual summary from the tidy data if present
        try:
            df = _to_dataframe(reliability_monthly_by_year)
            if not df.empty and "year" in df.columns:
                # Annual SAIFI/SAIDI (sums across months)
                out_rows = []
                if "saifi" in df.columns:
                    saifi_annual = df.groupby("year")["saifi"].sum(min_count=1)
                else:
                    saifi_annual = pd.Series(dtype=float)
                if "saidi" in df.columns:
                    saidi_annual = df.groupby("year")["saidi"].sum(min_count=1)
                else:
                    saidi_annual = pd.Series(dtype=float)

                years = sorted(set(df["year"].dropna().astype(int).tolist()))
                for y in years:
                    out_rows.append(
                        {
                            "year": int(y),
                            "annual_saifi": float(saifi_annual.get(y)) if y in saifi_annual.index else None,
                            "annual_saidi": float(saidi_annual.get(y)) if y in saidi_annual.index else None,
                        }
                    )
                reliability_summary_df = pd.DataFrame(out_rows)

                # Add percentile/mean rows
                if not reliability_summary_df.empty and "annual_saifi" in reliability_summary_df.columns:
                    s = pd.to_numeric(reliability_summary_df["annual_saifi"], errors="coerce").dropna()
                    if not s.empty:
                        reliability_summary_df.attrs["saifi_mean"] = float(s.mean())
                        reliability_summary_df.attrs["saifi_p75"] = float(s.quantile(0.75))
                        reliability_summary_df.attrs["saifi_p90"] = float(s.quantile(0.90))
        except Exception:
            reliability_summary_df = pd.DataFrame([])

    # Build zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # README
        z.writestr(
            "README.txt",
            _readme_text(site=site, outage_payload=outage_payload, cdf_payloads=cdf_payloads, results_payloads=results_payloads),
        )

        # --- Site ---
        z.writestr("site.json", _safe_json(site))
        z.writestr("site.csv", _df_to_csv_bytes(_to_dataframe(site)))

        # --- Reliability ---
        if reliability_monthly_by_year is not None:
            z.writestr("reliability_monthly_by_year.csv", _df_to_csv_bytes(_to_dataframe(reliability_monthly_by_year)))
        if not reliability_summary_df.empty:
            # Put summary stats in header row as separate file for clarity
            summ = reliability_summary_df.copy()
            stats = {
                "saifi_mean": summ.attrs.get("saifi_mean"),
                "saifi_p75": summ.attrs.get("saifi_p75"),
                "saifi_p90": summ.attrs.get("saifi_p90"),
            }
            z.writestr("reliability_annual_summary.csv", _df_to_csv_bytes(summ))
            z.writestr("reliability_summary_stats.json", _safe_json(stats))
        if reliability_raw is not None:
            z.writestr("reliability_raw.json", _safe_json(reliability_raw))

        # --- PRESTO ---
        if isinstance(outage_payload, dict):
            # Prefer tidy events if present
            events = outage_payload.get("events_tidy")
            if events is None:
                # fall back to nested results
                results = outage_payload.get("results")
                if isinstance(results, dict):
                    events = results.get("events") or results.get("interruptions") or results.get("outages")
            if events is not None:
                z.writestr("presto_events.csv", _df_to_csv_bytes(_to_dataframe(events)))
            z.writestr("presto_raw.json", _safe_json(outage_payload))
        elif outage_payload is not None:
            # legacy
            z.writestr("presto_raw.json", _safe_json(outage_payload))

        # --- CDF ---
        if "cdf_inputs" in cdf_payloads:
            z.writestr("cdf_inputs.json", _safe_json(cdf_payloads.get("cdf_inputs")))
        if "cdf_curve" in cdf_payloads:
            z.writestr("cdf_curve.csv", _df_to_csv_bytes(_to_dataframe(cdf_payloads.get("cdf_curve"))))

        # --- Results ---
        if "results_summary" in results_payloads:
            z.writestr("results_summary.csv", _df_to_csv_bytes(_to_dataframe(results_payloads.get("results_summary"))))
        # Prefer explicit event/year tables, fall back to monolithic results payload
        if "results_by_event" in results_payloads:
            z.writestr("results_by_event.csv", _df_to_csv_bytes(_to_dataframe(results_payloads.get("results_by_event"))))
        if "results_by_sim_year" in results_payloads:
            z.writestr("results_by_sim_year.csv", _df_to_csv_bytes(_to_dataframe(results_payloads.get("results_by_sim_year"))))
        if "results" in results_payloads:
            z.writestr("results_raw.json", _safe_json(results_payloads.get("results")))

        # --- Metadata ---
        meta = {
            "exported_at_utc": _now_utc_iso(),
            "site_has_fips": bool(site.get("fips")),
            "has_historical": bool(isinstance(outage_payload, dict) and outage_payload.get("historical")),
            "has_presto_events": bool(isinstance(outage_payload, dict) and (outage_payload.get("events_tidy") or (outage_payload.get("results") or {}).get("events"))),
            "has_cdf_inputs": "cdf_inputs" in cdf_payloads,
            "has_cdf_curve": "cdf_curve" in cdf_payloads,
            "has_results": bool(results_payloads),
            "session_state_keys_sample": list(st.session_state.keys())[:50],
        }
        z.writestr("metadata.json", _safe_json(meta))

    diagnostics = {
        "site": site,
        "has_outage_payload": outage_payload is not None,
        "cdf_payload_keys": {k: v for k, v in cdf_payloads.items() if k.endswith("_key")},
        "results_payload_keys": {k: v for k, v in results_payloads.items() if k.endswith("_key")},
    }
    return buf.getvalue(), diagnostics


# ---------------- Page ----------------
st.set_page_config(page_title="Exports", page_icon="📦", layout="wide")

st.title("Exports")
st.caption("Download a complete bundle of inputs, assumptions, and outputs for sharing or recordkeeping.")

site = _get_site_payload()
if not site:
    st.warning("Please complete 'Site inputs' first.")

# Primary download
bundle_bytes, diag = _build_zip_bundle()

# Naming
fips = str(site.get("fips", "")).zfill(5) if site.get("fips") is not None else ""
county = str(site.get("county_name", "")).strip().replace(" ", "_")
state = str(site.get("state_abbrev", "")).strip().replace(" ", "_")
stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
filename = f"outagecost_bundle_{state}_{county}_{fips}_{stamp}.zip".replace("__", "_")

st.download_button(
    label="Download full analysis bundle (.zip)",
    data=bundle_bytes,
    file_name=filename,
    mime="application/zip",
    type="primary",
    use_container_width=True,
)

st.markdown(
    """
**Includes**: site info, historical reliability, PRESTO simulations, CDF inputs/curve, and results outputs (when available).

If a section hasn’t been run yet, the bundle will still download, but may omit those files.
    """
)

# Optional individual downloads
with st.expander("Download individual files", expanded=False):
    st.subheader("Site")
    st.download_button("site.json", data=_safe_json(site).encode("utf-8"), file_name="site.json", mime="application/json")
    st.download_button("site.csv", data=_df_to_csv_bytes(_to_dataframe(site)), file_name="site.csv", mime="text/csv")

    outage_payload = _get_outage_payload()
    if isinstance(outage_payload, dict):
        st.subheader("Reliability")
        if outage_payload.get("historical_yearly_tidy") is not None:
            st.download_button(
                "reliability_monthly_by_year.csv",
                data=_df_to_csv_bytes(_to_dataframe(outage_payload.get("historical_yearly_tidy"))),
                file_name="reliability_monthly_by_year.csv",
                mime="text/csv",
            )
        if outage_payload.get("historical_by_year") is not None:
            st.download_button(
                "reliability_raw.json",
                data=_safe_json(outage_payload.get("historical_by_year")).encode("utf-8"),
                file_name="reliability_raw.json",
                mime="application/json",
            )

        st.subheader("PRESTO simulations")
        events = outage_payload.get("events_tidy")
        if events is None:
            results = outage_payload.get("results")
            if isinstance(results, dict):
                events = results.get("events") or results.get("interruptions") or results.get("outages")
        if events is not None:
            st.download_button(
                "presto_events.csv",
                data=_df_to_csv_bytes(_to_dataframe(events)),
                file_name="presto_events.csv",
                mime="text/csv",
            )
        st.download_button(
            "presto_raw.json",
            data=_safe_json(outage_payload).encode("utf-8"),
            file_name="presto_raw.json",
            mime="application/json",
        )

    cdf_payloads = _get_cdf_payloads()
    if cdf_payloads:
        st.subheader("CDF")
        if "cdf_inputs" in cdf_payloads:
            st.download_button(
                "cdf_inputs.json",
                data=_safe_json(cdf_payloads.get("cdf_inputs")).encode("utf-8"),
                file_name="cdf_inputs.json",
                mime="application/json",
            )
        if "cdf_curve" in cdf_payloads:
            st.download_button(
                "cdf_curve.csv",
                data=_df_to_csv_bytes(_to_dataframe(cdf_payloads.get("cdf_curve"))),
                file_name="cdf_curve.csv",
                mime="text/csv",
            )

    results_payloads = _get_results_payloads()
    if results_payloads:
        st.subheader("Results")
        if "results_summary" in results_payloads:
            st.download_button(
                "results_summary.csv",
                data=_df_to_csv_bytes(_to_dataframe(results_payloads.get("results_summary"))),
                file_name="results_summary.csv",
                mime="text/csv",
            )
        if "results_by_event" in results_payloads:
            st.download_button(
                "results_by_event.csv",
                data=_df_to_csv_bytes(_to_dataframe(results_payloads.get("results_by_event"))),
                file_name="results_by_event.csv",
                mime="text/csv",
            )
        if "results_by_sim_year" in results_payloads:
            st.download_button(
                "results_by_sim_year.csv",
                data=_df_to_csv_bytes(_to_dataframe(results_payloads.get("results_by_sim_year"))),
                file_name="results_by_sim_year.csv",
                mime="text/csv",
            )
        if "results" in results_payloads:
            st.download_button(
                "results_raw.json",
                data=_safe_json(results_payloads.get("results")).encode("utf-8"),
                file_name="results_raw.json",
                mime="application/json",
            )

with st.expander("Debug", expanded=False):
    st.json(diag)
    st.caption("If a file is missing from the bundle, run the relevant step first (Outage profile, CDF Costs, Results).")