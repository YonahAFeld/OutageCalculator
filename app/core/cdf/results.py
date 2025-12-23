from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from app.core.cdf.model import (
    DamagedEquipmentCost,
    DowntimeCost,
    LostDataCost,
    ProcessInterruptionCost,
    RestartCost,
    OtherFixedCost,
    PerishableFoodSpoilageCost,
    MissedDeadlinesSpoilageCost,
    OtherSpoilageCost,
    BackupFuelIncrementalCost,
    RentedEquipmentIncrementalCost,
    CustomerSalesIncrementalCost,
    InterruptedProductionIncrementalCost,
    StaffProductivityIncrementalCost,
    SafetyIncrementalCost,
    OtherIncrementalCost,
    total_fixed_cost,
    total_spoilage_cost,
    total_incremental_cost,
)


def coerce_outages_payload(outages: Any) -> Dict[str, Any]:
    """Coerce various outage payload shapes into a dict.

    Handles legacy shapes and PRESTO variants where `results` may be a dict or a bare list.
    """
    if outages is None:
        return {}

    # Some legacy sessions stored a bare list of events
    if isinstance(outages, list):
        return {"results": {"events": outages}}

    if not isinstance(outages, dict):
        return {}

    results = outages.get("results")

    # If results is already a dict, keep as-is.
    if isinstance(results, dict):
        return outages

    # If PRESTO returns `results` as a bare list of events, wrap it.
    if isinstance(results, list):
        coerced = dict(outages)
        coerced["results"] = {"events": results}
        return coerced

    return outages


def safe_results_keys(outages: Any) -> List[str]:
    """Best-effort list of keys under outages['results'] for debugging/UI."""
    o = coerce_outages_payload(outages)
    r = o.get("results")
    if isinstance(r, dict):
        return list(r.keys())
    return []


# -----------------------------
# Core data structures
# -----------------------------
@dataclass
class OutageEvent:
    duration_hours: float


@dataclass
class SimulationYear:
    year_index: int
    events: List[OutageEvent]


@dataclass
class EventCost:
    duration_hours: float
    fixed: float
    spoilage: float
    incremental: float
    total: float


@dataclass
class YearCostSummary:
    year_index: int
    total_cost: float
    total_fixed: float
    total_spoilage: float
    total_incremental: float
    n_events: int


# -----------------------------
# Utilities
# -----------------------------
def _clamp(x: Any) -> float:
    try:
        return max(0.0, float(x))
    except Exception:
        return 0.0


def _pct_to_prob(pct: Any) -> float:
    return min(1.0, _clamp(pct) / 100.0)


def _pick(d: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _none_if_empty(x: Any) -> Optional[float]:
    """Return float(x) unless x is None or empty string; preserves 0."""
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round(q * (len(values) - 1)))
    return values[idx]


def _poisson(rng: random.Random, lam: float) -> int:
    """Poisson sampler without numpy. Good enough for small lambdas (SAIFI monthly is typically < 1)."""
    lam = max(0.0, float(lam or 0.0))
    if lam <= 0.0:
        return 0
    # Knuth
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


def _lognormal_with_mean(rng: random.Random, mean: float, sigma: float) -> float:
    """Sample lognormal with a specified mean."""
    mean = max(0.0, float(mean or 0.0))
    if mean <= 0.0:
        return 0.0
    sigma = max(0.05, float(sigma))
    mu = math.log(mean) - 0.5 * sigma * sigma
    return float(rng.lognormvariate(mu, sigma))
# -----------------------------
# Historical reliability metrics -> modeled outages
# -----------------------------

def _extract_reliability_rows(outages_payload: Any, *, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return reliability metric rows (month, saifi, saidi, caidi) for a specific year.

    Expected storage shapes in `outages_payload`:
      - outages_payload['historical_by_year'][str(year)]['results'] -> list of 12 rows
      - outages_payload['historical_by_year'][year]['results'] -> list
      - outages_payload['historical']['results'] -> list (aggregated across years)
      - outages_payload['historical_tidy'] -> list (already tidy, aggregated)

    If per-year data is missing, this returns the aggregated rows.
    """
    if outages_payload is None:
        return []

    if not isinstance(outages_payload, dict):
        return []

    # Aggregated tidy rows (preferred for typical pattern)
    tidy = outages_payload.get("historical_tidy")
    if isinstance(tidy, list) and tidy:
        return [r for r in tidy if isinstance(r, dict)]

    # Preferred: per-year payloads
    by_year = outages_payload.get("historical_by_year")
    if isinstance(by_year, dict) and year is not None:
        y_key_str = str(year)
        candidate = by_year.get(y_key_str) or by_year.get(year)
        if isinstance(candidate, dict):
            rows = candidate.get("results")
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]

    hist = outages_payload.get("historical")
    if isinstance(hist, dict):
        rows = hist.get("results")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]

    return []


def _reliability_month_map(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Map month -> {saifi, saidi, caidi} with numeric coercion."""
    out: Dict[int, Dict[str, float]] = {}
    for r in rows:
        try:
            m = int(r.get("month"))
        except Exception:
            continue
        if m < 1 or m > 12:
            continue

        def f(k: str) -> float:
            try:
                return float(r.get(k))
            except Exception:
                return 0.0

        saifi = max(0.0, f("saifi"))
        saidi = max(0.0, f("saidi"))
        caidi = max(0.0, f("caidi"))
        # If caidi missing, compute from saidi/saifi when possible
        if caidi <= 0.0 and saifi > 0.0:
            caidi = saidi / saifi

        out[m] = {"saifi": saifi, "saidi": saidi, "caidi": caidi}

    return out


def _modeled_events_from_reliability(
    *,
    rng: random.Random,
    month_map: Dict[int, Dict[str, float]],
    duration_sigma: float = 0.9,
    max_event_hours: float = 72.0,
) -> List[OutageEvent]:
    """Convert monthly SAIFI/CAIDI into a modeled list of outage events for one synthetic year."""
    events: List[OutageEvent] = []
    for m in range(1, 13):
        mm = month_map.get(m)
        if not mm:
            continue
        n = _poisson(rng, mm.get("saifi", 0.0))
        caidi_h = float(mm.get("caidi", 0.0) or 0.0)
        for _ in range(n):
            d = _lognormal_with_mean(rng, caidi_h, duration_sigma)
            d = max(0.0, min(float(d), float(max_event_hours)))
            events.append(OutageEvent(duration_hours=d))
    return events


def run_historical_cost_analysis(
    *,
    cdf_inputs: Dict[str, Any],
    outages_payload: Optional[Dict[str, Any]] = None,
    years: Optional[List[int]] = None,
    monte_carlo_runs: int = 300,
    duration_sigma: float = 0.9,
    range_q_low: float = 0.10,
    range_q_high: float = 0.90,
) -> Dict[str, Any]:
    """Estimate historical annual outage cost per year using reliability metrics.

    Returns, for each year, a "typical" annual cost (median) and a "likely range" (P10–P90 by default).

    Notes:
      - This is modeled: PRESTO metrics do not provide exact event lists.
      - We generate synthetic outage events per month consistent with SAIFI and CAIDI.
    """
    outages_payload = outages_payload or {}

    fixed = build_fixed_costs(cdf_inputs)
    spoilage = build_spoilage_costs(cdf_inputs)
    incremental = build_incremental_costs(cdf_inputs)

    fixed_per_event = total_fixed_cost(fixed)

    # Determine which historical years we can report.
    if years is None:
        # Prefer stored per-year keys
        by_year = outages_payload.get("historical_by_year") if isinstance(outages_payload, dict) else None
        if isinstance(by_year, dict) and by_year:
            parsed: List[int] = []
            for k in by_year.keys():
                try:
                    parsed.append(int(k))
                except Exception:
                    continue
            years = sorted(set(parsed))
        else:
            # Fallback: we only have an aggregated profile
            years = []

    results_by_year: List[Dict[str, Any]] = []

    for y in years:
        rows = _extract_reliability_rows(outages_payload, year=y)
        month_map = _reliability_month_map(rows)
        if not month_map:
            continue

        rng = random.Random(int(y))
        annual_costs: List[float] = []
        annual_event_counts: List[int] = []
        annual_outage_hours: List[float] = []

        for _ in range(int(max(1, monte_carlo_runs))):
            events = _modeled_events_from_reliability(
                rng=rng,
                month_map=month_map,
                duration_sigma=duration_sigma,
            )
            annual_event_counts.append(len(events))
            annual_outage_hours.append(sum(e.duration_hours for e in events))

            fixed_total = fixed_per_event * len(events)
            spoil_total = sum(total_spoilage_cost(spoilage, e.duration_hours) for e in events)
            inc_total = sum(total_incremental_cost(incremental, e.duration_hours) for e in events)
            annual_costs.append(float(fixed_total + spoil_total + inc_total))

        typical = _percentile(annual_costs, 0.5)
        low = _percentile(annual_costs, float(range_q_low))
        high = _percentile(annual_costs, float(range_q_high))

        results_by_year.append(
            {
                "year": int(y),
                "typical": float(typical),
                "likely_low": float(low),
                "likely_high": float(high),
                "mean": float(sum(annual_costs) / len(annual_costs)) if annual_costs else 0.0,
                "events_typical": float(_percentile([float(n) for n in annual_event_counts], 0.5)) if annual_event_counts else 0.0,
                "outage_hours_typical": float(_percentile(annual_outage_hours, 0.5)) if annual_outage_hours else 0.0,
                "monte_carlo_runs": int(monte_carlo_runs),
            }
        )

    return {
        "historical_years": results_by_year,
        "range_q_low": float(range_q_low),
        "range_q_high": float(range_q_high),
        "duration_sigma": float(duration_sigma),
        "monte_carlo_runs": int(monte_carlo_runs),
        "notes": "Historical costs are modeled from PRESTO monthly SAIFI/SAIDI/CAIDI (not reconstructed from exact events).",
    }


# -----------------------------
# Aggregated monthly historical cost analysis
# -----------------------------

def run_historical_monthly_cost_analysis(
    *,
    cdf_inputs: Dict[str, Any],
    outages_payload: Optional[Dict[str, Any]] = None,
    years_label: str = "2014–2023",
    duration_low_multiplier: float = 0.70,
    duration_high_multiplier: float = 1.30,
) -> Dict[str, Any]:
    """Compute a typical monthly outage cost pattern from aggregated reliability metrics.

    Uses the aggregated monthly SAIFI/SAIDI/CAIDI profile (no year dimension) and estimates:
      - typical monthly cost (expected value)
      - a small likely range by varying the typical duration (CAIDI) up/down

    This is intentionally simple and stable: it does not attempt to reconstruct exact historical outages.
    """
    outages_payload = outages_payload or {}

    fixed = build_fixed_costs(cdf_inputs)
    spoilage = build_spoilage_costs(cdf_inputs)
    incremental = build_incremental_costs(cdf_inputs)

    fixed_per_event = total_fixed_cost(fixed)

    # Pull aggregated monthly rows. If per-year data exists, prefer the precomputed average profile.
    rows = _extract_reliability_rows(outages_payload, year=None)
    month_map = _reliability_month_map(rows)

    monthly: List[Dict[str, Any]] = []
    total_typical = 0.0
    total_low = 0.0
    total_high = 0.0

    for m in range(1, 13):
        mm = month_map.get(m) or {"saifi": 0.0, "saidi": 0.0, "caidi": 0.0}
        saifi = float(mm.get("saifi", 0.0) or 0.0)
        saidi = float(mm.get("saidi", 0.0) or 0.0)
        caidi = float(mm.get("caidi", 0.0) or 0.0)

        # If CAIDI missing, back-compute from SAIDI/SAIFI when possible
        if caidi <= 0.0 and saifi > 0.0:
            caidi = saidi / saifi

        d_typ = max(0.0, caidi)
        d_low = max(0.0, d_typ * float(duration_low_multiplier))
        d_high = max(0.0, d_typ * float(duration_high_multiplier))

        # Cost per outage event at each representative duration
        per_event_typ = fixed_per_event + total_spoilage_cost(spoilage, d_typ) + total_incremental_cost(incremental, d_typ)
        per_event_low = fixed_per_event + total_spoilage_cost(spoilage, d_low) + total_incremental_cost(incremental, d_low)
        per_event_high = fixed_per_event + total_spoilage_cost(spoilage, d_high) + total_incremental_cost(incremental, d_high)

        # Expected monthly cost: SAIFI is expected outages per month
        cost_typ = float(saifi) * float(per_event_typ)
        cost_low = float(saifi) * float(per_event_low)
        cost_high = float(saifi) * float(per_event_high)

        total_typical += cost_typ
        total_low += cost_low
        total_high += cost_high

        monthly.append(
            {
                "month": int(m),
                "saifi": float(saifi),
                "saidi": float(saidi),
                "caidi": float(caidi),
                "duration_hours_typical": float(d_typ),
                "cost_typical": float(cost_typ),
                "cost_low": float(cost_low),
                "cost_high": float(cost_high),
            }
        )

    return {
        "years_label": years_label,
        "monthly": monthly,
        "annual_typical": float(total_typical),
        "annual_low": float(total_low),
        "annual_high": float(total_high),
        "duration_low_multiplier": float(duration_low_multiplier),
        "duration_high_multiplier": float(duration_high_multiplier),
        "notes": (
            "Historical monthly costs are estimated from aggregated PRESTO reliability metrics (monthly SAIFI/SAIDI/CAIDI). "
            "Costs are computed as expected outages per month times the cost of a representative outage duration (CAIDI)."
        ),
    }


# -----------------------------
# Parse outage payload
# -----------------------------
def parse_outages(outages: Dict[str, Any]) -> List[SimulationYear]:
    """Parse outage payloads from legacy or PRESTO-like shapes.

    Supported shapes:
    - {"events": [...]}  (flat)
    - {"simulations": [{"events": [...]}, ...]} (nested)
    - {"results": {"events": [...]}} (PRESTO-like)
    - {"events_tidy": [...]} (normalized helper list)

    For flat event lists that include a `simulation` index, this groups by simulation.
    """
    outages = coerce_outages_payload(outages)
    if not outages:
        return []

    # PRESTO-like saved payload: payload.results.events
    results = outages.get("results") if isinstance(outages.get("results"), dict) else None
    if results and (
        isinstance(results.get("events"), list)
        or isinstance(results.get("interruptions"), list)
        or isinstance(results.get("outages"), list)
    ):
        flat_events = (
            results.get("events")
            or results.get("interruptions")
            or results.get("outages")
            or []
        )
        # PRESTO v2 schema: results.events is a list of simulated years.
        # Each element contains a list of outage durations (observed to be in hours).
        if isinstance(flat_events, list) and flat_events and isinstance(flat_events[0], dict):
            first = flat_events[0]
            if isinstance(first.get("durations"), list) and ("numInterruptions" in first):
                sim_years: List[SimulationYear] = []
                for i, yobj in enumerate(flat_events, start=1):
                    if not isinstance(yobj, dict):
                        continue
                    durs = yobj.get("durations")
                    if not isinstance(durs, list):
                        continue

                    evs: List[OutageEvent] = []
                    for d in durs:
                        if d is None:
                            continue
                        try:
                            dh = float(d)
                        except Exception:
                            continue
                        if dh < 0:
                            continue
                        evs.append(OutageEvent(duration_hours=dh))

                    sim_years.append(SimulationYear(i, evs))

                if sim_years:
                    return sim_years
        by_sim: Dict[int, List[OutageEvent]] = {}
        for ev in flat_events:
            if not isinstance(ev, dict):
                continue
            sim_idx = ev.get("simulation")
            try:
                # If simulation is missing/null, treat each row as its own simulated year
                sim_i = int(sim_idx) if sim_idx is not None else (flat_events.index(ev) + 1)
            except Exception:
                sim_i = flat_events.index(ev) + 1

            dh = ev.get("duration_hours")
            if dh is None:
                dh = ev.get("durationHours")
            if dh is None:
                dm = ev.get("duration_minutes")
                if dm is None:
                    dm = ev.get("durationMinutes")
                if dm is not None:
                    try:
                        dh = float(dm) / 60.0
                    except Exception:
                        dh = None

            if dh is None:
                # Compute duration from start/end timestamps if present
                start = ev.get("start")
                end = ev.get("end")
                if start and end:
                    try:
                        from datetime import datetime

                        def _parse_dt(s: str) -> datetime:
                            s = str(s)
                            if s.endswith("Z"):
                                s = s[:-1] + "+00:00"
                            return datetime.fromisoformat(s)

                        ds = _parse_dt(start)
                        de = _parse_dt(end)
                        dh = (de - ds).total_seconds() / 3600.0
                    except Exception:
                        dh = None

            if dh is None:
                continue

            by_sim.setdefault(sim_i, []).append(OutageEvent(duration_hours=_clamp(dh)))

        years: List[SimulationYear] = []
        for sim_i in sorted(by_sim.keys()):
            years.append(SimulationYear(sim_i, by_sim[sim_i]))
        return years

    # Normalized helper list: payload.events_tidy
    if isinstance(outages.get("events_tidy"), list):
        flat_events = outages.get("events_tidy") or []
        by_sim: Dict[int, List[OutageEvent]] = {}
        for idx, ev in enumerate(flat_events):
            if not isinstance(ev, dict):
                continue
            sim_idx = ev.get("simulation")
            try:
                # If simulation is missing/null, treat each row as its own simulated year
                sim_i = int(sim_idx) if sim_idx is not None else (idx + 1)
            except Exception:
                sim_i = idx + 1

            dh = ev.get("duration_hours")
            if dh is None:
                dh = ev.get("durationHours")
            if dh is None:
                dm = ev.get("duration_minutes")
                if dm is None:
                    dm = ev.get("durationMinutes")
                if dm is not None:
                    try:
                        dh = float(dm) / 60.0
                    except Exception:
                        dh = None

            if dh is None:
                # Compute duration from start/end timestamps if present
                start = ev.get("start")
                end = ev.get("end")
                if start and end:
                    try:
                        from datetime import datetime

                        def _parse_dt(s: str) -> datetime:
                            s = str(s)
                            if s.endswith("Z"):
                                s = s[:-1] + "+00:00"
                            return datetime.fromisoformat(s)

                        ds = _parse_dt(start)
                        de = _parse_dt(end)
                        dh = (de - ds).total_seconds() / 3600.0
                    except Exception:
                        dh = None

            if dh is None:
                continue

            by_sim.setdefault(sim_i, []).append(OutageEvent(duration_hours=_clamp(dh)))

        years: List[SimulationYear] = []
        for sim_i in sorted(by_sim.keys()):
            years.append(SimulationYear(sim_i, by_sim[sim_i]))
        return years

    # Legacy nested simulations
    if isinstance(outages.get("simulations"), list):
        years = []
        for i, sim in enumerate(outages["simulations"]):
            if not isinstance(sim, dict):
                continue
            events = [
                OutageEvent(duration_hours=_clamp((e or {}).get("duration_hours", 0)))
                for e in (sim.get("events") or [])
                if isinstance(e, dict)
            ]
            years.append(SimulationYear(i, events))
        return years

    # Legacy flat events
    if isinstance(outages.get("events"), list):
        events = [
            OutageEvent(duration_hours=_clamp((e or {}).get("duration_hours", 0)))
            for e in outages.get("events")
            if isinstance(e, dict)
        ]
        return [SimulationYear(0, events)]

    return []


# -----------------------------
# Build cost objects from saved inputs
# -----------------------------
def build_fixed_costs(inputs: Dict[str, Any]) -> List[Any]:
    fixed = inputs.get("fixed_costs", {})
    objs = []

    de = fixed.get("damaged_equipment", {})
    if de.get("enabled"):
        objs.append(
            DamagedEquipmentCost(
                _clamp(_pick(de, "avg_cost")),
                _clamp(_pick(de, "count")),
                _clamp(_pick(de, "probability")),
            )
        )

    dt = fixed.get("downtime", {})
    if dt.get("enabled"):
        objs.append(
            DowntimeCost(
                _clamp(_pick(dt, "cost_per_hour")),
                _clamp(_pick(dt, "hours")),
            )
        )

    ld = fixed.get("lost_data", {})
    if ld.get("enabled"):
        objs.append(
            LostDataCost(
                _clamp(_pick(ld, "value")),
                _clamp(_pick(ld, "probability")),
            )
        )

    pi = fixed.get("process_interruption", {})
    if pi.get("enabled"):
        objs.append(
            ProcessInterruptionCost(
                _clamp(_pick(pi, "hourly_cost")),
                _clamp(_pick(pi, "hours")),
                _clamp(_pick(pi, "io")),
            )
        )

    rs = fixed.get("restart", {})
    if rs.get("enabled"):
        objs.append(
            RestartCost(
                _clamp(_pick(rs, "hourly_cost")),
                _clamp(_pick(rs, "hours")),
                _clamp(_pick(rs, "additional_costs")),
            )
        )

    for row in fixed.get("other_fixed_costs", []):
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("label") or "Other"
        # UI/editor schemas sometimes store this as `amount` instead of `cost`.
        val = row.get("cost")
        if val is None:
            val = row.get("amount")
        if val is None:
            val = row.get("value")
        objs.append(OtherFixedCost(str(name), _clamp(val)))

    return objs


def build_spoilage_costs(inputs: Dict[str, Any]) -> List[Any]:
    # Robustly accept alternate storage keys
    spoil = (
        inputs.get("spoilage_costs")
        or inputs.get("spoilage")
        or inputs.get("spoilage_inputs")
        or {}
    )
    objs: List[Any] = []

    pf = spoil.get("perishable_food", {})
    if isinstance(pf, dict) and pf.get("enabled"):
        val = _clamp(_pick(pf, "value", "amount", "cost"))
        start = _clamp(_pick(pf, "start", "begins", default=0.0))
        end = _clamp(_pick(pf, "end", "full", "hour_of_100_percent_spoilage", default=0.0))
        half_raw = pf.get("half")
        half = _clamp(half_raw) if half_raw is not None else 0.0
        objs.append(
            PerishableFoodSpoilageCost(
                total_value_spoilable_product=val,
                hour_spoilage_begins=start,
                hour_of_100_percent_spoilage=end,
                hour_of_50_percent_spoilage=(half or None),
            )
        )

    md = spoil.get("missed_deadlines", {})
    if isinstance(md, dict) and md.get("enabled"):
        val = _clamp(_pick(md, "value", "amount", "cost"))
        objs.append(
            MissedDeadlinesSpoilageCost(
                total_value_missed_deadline=val,
                trigger_hour=0.0,
            )
        )

    other_rows = (
        spoil.get("other_spoilage_costs")
        or spoil.get("other")
        or spoil.get("other_rows")
        or []
    )
    for row in other_rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("label") or "Other spoilage"
        val = row.get("value")
        if val is None:
            val = row.get("cost")
        if val is None:
            val = row.get("amount")
        start = row.get("start")
        if start is None:
            start = 0
        objs.append(
            OtherSpoilageCost(
                name=str(name),
                total_value_spoilable_product=_clamp(val),
                hour_spoilage_begins=_clamp(start),
                hour_of_100_percent_spoilage=_clamp(row.get("end")),
                hour_of_50_percent_spoilage=(_clamp(row.get("half")) or None),
            )
        )

    return objs


def build_incremental_costs(inputs: Dict[str, Any]) -> List[Any]:
    # Robustly accept alternate storage keys
    inc = (
        inputs.get("incremental_costs")
        or inputs.get("incremental")
        or inputs.get("incremental_inputs")
        or {}
    )
    objs: List[Any] = []

    bf = inc.get("backup_fuel", {})
    if isinstance(bf, dict) and bf.get("enabled"):
        objs.append(
            BackupFuelIncrementalCost(
                generator_load_kw=_clamp(bf.get("gl")),
                fuel_efficiency_kwh_per_gallon=_clamp((bf.get("fe") if bf.get("fe") is not None else 13)),
                fuel_cost_per_gallon=_clamp(bf.get("fc")),
                additional_hourly_backup_costs=_clamp(bf.get("a")),
                hour_cost_starts=_none_if_empty(bf.get("start")),
                hour_cost_mitigated=_none_if_empty(bf.get("mit")),
                avg_weekly_business_hours=_none_if_empty(bf.get("wh")),
            )
        )

    re = inc.get("rented_equipment", {})
    if isinstance(re, dict) and re.get("enabled"):
        objs.append(
            RentedEquipmentIncrementalCost(
                hourly_cost=_clamp(re.get("hc")),
                hour_cost_starts=_none_if_empty(re.get("start")),
                hour_cost_mitigated=_none_if_empty(re.get("mit")),
                avg_weekly_business_hours=_none_if_empty(re.get("wh")),
            )
        )

    cs = inc.get("customer_sales", {})
    if isinstance(cs, dict) and cs.get("enabled"):
        objs.append(
            CustomerSalesIncrementalCost(
                weekly_revenues=_clamp(cs.get("wr")),
                percent_decrease_in_sales=_pct_to_prob(cs.get("pct")),
                hour_cost_starts=_none_if_empty(cs.get("start")),
                hour_cost_mitigated=_none_if_empty(cs.get("mit")),
                avg_weekly_business_hours=_none_if_empty(cs.get("wh")),
            )
        )

    ip = inc.get("interrupted_production", {})
    if isinstance(ip, dict) and ip.get("enabled"):
        objs.append(
            InterruptedProductionIncrementalCost(
                normal_output_units_per_hour=_clamp(ip.get("no")),
                percent_reduction_output=_pct_to_prob(ip.get("pct")),
                outage_value_per_unit=_clamp(ip.get("ov")),
                hour_cost_starts=_none_if_empty(ip.get("start")),
                hour_cost_mitigated=_none_if_empty(ip.get("mit")),
                avg_weekly_business_hours=_none_if_empty(ip.get("wh")),
            )
        )

    sp = inc.get("staff_productivity", {})
    if isinstance(sp, dict) and sp.get("enabled"):
        objs.append(
            StaffProductivityIncrementalCost(
                fully_burdened_hourly_employee_cost=_clamp(sp.get("c")),
                percent_work_not_completed=_pct_to_prob(sp.get("pct")),
                number_people_affected=_clamp(sp.get("n")),
                hour_cost_starts=_none_if_empty(sp.get("start")),
                hour_cost_mitigated=_none_if_empty(sp.get("mit")),
                avg_weekly_business_hours=_none_if_empty(sp.get("wh")),
            )
        )

    sf = inc.get("safety", {})
    if isinstance(sf, dict) and sf.get("enabled"):
        objs.append(
            SafetyIncrementalCost(
                number_people=_clamp(sf.get("n")),
                hourly_cost_per_person=_clamp(sf.get("p")),
                hour_cost_starts=_none_if_empty(sf.get("start")),
                hour_cost_mitigated=_none_if_empty(sf.get("mit")),
                avg_weekly_business_hours=_none_if_empty(sf.get("wh")),
            )
        )

    other_rows = (
        inc.get("other_incremental_costs")
        or inc.get("other")
        or inc.get("other_rows")
        or []
    )
    for row in other_rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("label") or "Other incremental"
        hc = row.get("hourly_cost")
        if hc is None:
            hc = row.get("hc")
        if hc is None:
            hc = row.get("cost_per_hour")
        if hc is None:
            hc = row.get("amount_per_hour")
        objs.append(
            OtherIncrementalCost(
                name=str(name),
                hourly_cost=_clamp(hc),
                hour_cost_starts=_none_if_empty(row.get("start")),
                hour_cost_mitigated=_none_if_empty(row.get("mit")),
                avg_weekly_business_hours=_none_if_empty(row.get("wh")),
            )
        )

    return objs


# -----------------------------
# Main analysis function
# -----------------------------
def run_outage_cost_analysis(
    *,
    cdf_inputs: Dict[str, Any],
    outages_payload: Optional[Dict[str, Any]] = None,
    outages: Optional[Dict[str, Any]] = None,
    analysis_years: int = 20,
    discount_rate_nominal: float = 0.07,
    discount_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute annual outage costs across simulated years.

    Backward/forward compatible args:
    - accept `outages_payload` or `outages`
    - accept `discount_rate_nominal` or `discount_rate`

    Note: PV calculations are not yet returned; `analysis_years` and discount rate are kept for future expansion.
    """
    if outages_payload is None:
        outages_payload = outages or {}

    if discount_rate is None:
        discount_rate = float(discount_rate_nominal)

    # Allow 7 to mean 7%
    if discount_rate > 1.0:
        discount_rate = discount_rate / 100.0

    fixed = build_fixed_costs(cdf_inputs)
    spoilage = build_spoilage_costs(cdf_inputs)
    incremental = build_incremental_costs(cdf_inputs)

    spoilage_payload_present = bool(
        (cdf_inputs.get("spoilage_costs") or cdf_inputs.get("spoilage") or cdf_inputs.get("spoilage_inputs"))
    )
    incremental_payload_present = bool(
        (cdf_inputs.get("incremental_costs") or cdf_inputs.get("incremental") or cdf_inputs.get("incremental_inputs"))
    )

    years = parse_outages(outages_payload)
    summaries: List[YearCostSummary] = []

    # Fixed costs are per outage event.
    # Primary path: compute from model objects.
    fixed_per_event_model = float(total_fixed_cost(fixed))

    # Fallback/verification path: compute from saved input payload (what the UI shows the user).
    fixed_per_event_inputs = None
    try:
        fc_sum = (cdf_inputs.get("fixed_costs_summary") or {}).get("total_fixed_cost_per_outage")
        if fc_sum is not None:
            fixed_per_event_inputs = float(fc_sum)
    except Exception:
        fixed_per_event_inputs = None

    if fixed_per_event_inputs is None:
        try:
            fc = cdf_inputs.get("fixed_costs") or {}
            parts: List[float] = []
            for k in ["damaged_equipment", "downtime", "lost_data", "process_interruption", "restart"]:
                v = (fc.get(k) or {}).get("calculated_cost")
                if v is not None:
                    parts.append(float(v))
            for row in fc.get("other_fixed_costs", []) or []:
                if isinstance(row, dict):
                    v = row.get("cost")
                    if v is None:
                        v = row.get("amount")
                    if v is None:
                        v = row.get("value")
                    if v is not None:
                        parts.append(float(v))
            if parts:
                fixed_per_event_inputs = float(sum(parts))
        except Exception:
            fixed_per_event_inputs = None

    # Choose fixed_per_event.
    # If the UI-derived fixed cost exists and differs materially from the model-derived value,
    # prefer the UI-derived number (keeps results consistent with what the user entered).
    fixed_per_event = float(fixed_per_event_model)
    if fixed_per_event_inputs is not None:
        diff = abs(float(fixed_per_event_inputs) - float(fixed_per_event_model))
        if diff > max(1.0, 0.01 * max(1.0, float(fixed_per_event_model))):
            fixed_per_event = float(fixed_per_event_inputs)

    event_costs_by_year: Dict[int, List[Dict[str, Any]]] = {}

    def _event_cost_components(duration_hours: float) -> Tuple[float, float, float, float]:
        """Return (fixed, spoilage, incremental, total) for a single outage event."""
        f = float(fixed_per_event)
        s = float(total_spoilage_cost(spoilage, duration_hours))
        i = float(total_incremental_cost(incremental, duration_hours))
        return f, s, i, f + s + i

    def _event_cost_from_curve(duration_hours: float) -> Tuple[float, float, float, float]:
        """Return (fixed, spoilage, incremental, total) for a single outage event by looking up the sampled CDF curve.

        Uses a stepwise (ceil) lookup: picks the first curve point whose duration >= event duration.
        If the event duration exceeds the max curve duration, uses the max point.
        """
        dh = float(_clamp(duration_hours))
        if not cdf_cost_curve:
            # Fallback to analytic components if curve missing
            return _event_cost_components(dh)

        # Ensure sorted by duration
        curve_sorted = sorted(cdf_cost_curve, key=lambda r: float(r.get("duration_hours", 0.0)))
        for row in curve_sorted:
            if float(row.get("duration_hours", 0.0)) >= dh:
                return (
                    float(row.get("fixed", 0.0)),
                    float(row.get("spoilage", 0.0)),
                    float(row.get("incremental", 0.0)),
                    float(row.get("total", 0.0)),
                )

        last = curve_sorted[-1]
        return (
            float(last.get("fixed", 0.0)),
            float(last.get("spoilage", 0.0)),
            float(last.get("incremental", 0.0)),
            float(last.get("total", 0.0)),
        )

    # Sample the implied CDF cost function at standard durations (hours)
    curve_hours = [0.25, 0.5, 1, 2, 4, 8, 12, 24, 48, 72]
    cdf_cost_curve = []
    for h in curve_hours:
        f, s, i, t = _event_cost_components(float(h))
        cdf_cost_curve.append(
            {
                "duration_hours": float(h),
                "fixed": float(f),
                "spoilage": float(s),
                "incremental": float(i),
                "total": float(t),
            }
        )

    for y in years:
        # Fixed costs apply at outage start, once per event
        fixed_total = float(fixed_per_event) * float(len(y.events))

        spoil_total = 0.0
        inc_total = 0.0
        year_events: List[Dict[str, Any]] = []

        for ev in y.events:
            dh = float(ev.duration_hours)
            # Pass the outage duration through the duration-based cost function (CDF curve)
            f, s, i, t = _event_cost_from_curve(dh)
            spoil_total += s
            inc_total += i
            year_events.append(
                {
                    "duration_hours": dh,
                    "fixed": float(f),
                    "spoilage": float(s),
                    "incremental": float(i),
                    "total": float(t),
                    # Helpful for debugging: which curve duration was used
                    "curve_duration_hours": float(
                        next(
                            (
                                row.get("duration_hours")
                                for row in sorted(cdf_cost_curve, key=lambda r: float(r.get("duration_hours", 0.0)))
                                if float(row.get("duration_hours", 0.0)) >= float(_clamp(dh))
                            ),
                            (sorted(cdf_cost_curve, key=lambda r: float(r.get("duration_hours", 0.0)))[-1].get("duration_hours") if cdf_cost_curve else None),
                        )
                    )
                    if cdf_cost_curve
                    else None,
                }
            )

        event_costs_by_year[int(y.year_index)] = year_events

        summaries.append(
            YearCostSummary(
                int(y.year_index),
                float(fixed_total + spoil_total + inc_total),
                float(fixed_total),
                float(spoil_total),
                float(inc_total),
                int(len(y.events)),
            )
        )

    totals = [float(s.total_cost) for s in summaries]

    return {
        "annual_costs": totals,
        "mean": float(sum(totals) / len(totals)) if totals else 0.0,
        "p50": float(_percentile(totals, 0.5)),
        "p90": float(_percentile(totals, 0.9)),
        "summaries": [s.__dict__ for s in summaries],
        # Event-level costs for transparency and better charts
        "event_costs_by_year": event_costs_by_year,
        # Sampled implied cost function (fixed + spoilage(duration) + incremental(duration))
        "cdf_cost_curve": cdf_cost_curve,
        "meta": {
            "analysis_years": int(analysis_years),
            "discount_rate": float(discount_rate),
            "fixed_cost_per_event": float(fixed_per_event),
            "fixed_cost_per_event_model": float(fixed_per_event_model),
            "fixed_cost_per_event_inputs": float(fixed_per_event_inputs) if fixed_per_event_inputs is not None else None,
            "n_simulated_years": int(len(summaries)),
            "uses_duration_based_curve_lookup": True,
            "has_spoilage_payload": bool(spoilage_payload_present),
            "has_incremental_payload": bool(incremental_payload_present),
            "n_fixed_objects": int(len(fixed)),
            "n_spoilage_objects": int(len(spoilage)),
            "n_incremental_objects": int(len(incremental)),
            "curve_point_2h": next((r for r in cdf_cost_curve if float(r.get("duration_hours", 0.0)) == 2.0), None),
        },
    }