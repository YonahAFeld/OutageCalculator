from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union


# -------------------------------------------------------------------
# Fixed costs (occur immediately when power is lost; independent of outage duration)
# -------------------------------------------------------------------
class FixedCost(Protocol):
    name: str

    def cost(self) -> float: ...


class SpoilageCost(Protocol):
    name: str

    def cost(self, outage_duration_hours: float) -> float: ...


class IncrementalCost(Protocol):
    name: str

    def cost(self, outage_duration_hours: float) -> float: ...


def _weekly_hours_multiplier(avg_weekly_business_hours: Optional[float]) -> float:
    """Return a fractional multiplier in [0, 1] based on weekly business hours."""
    if avg_weekly_business_hours is None:
        return 1.0
    h = _clamp_nonnegative(avg_weekly_business_hours)
    if h <= 0:
        return 1.0
    if h >= 168:
        return 1.0
    return h / 168.0


def _active_hours(duration_h: float, start_h: Optional[float] = None, mitigated_h: Optional[float] = None) -> float:
    """Compute the number of outage hours a cost applies, respecting start and mitigation hours.

    Interpretation:
    - start_h: the hour within the outage when the cost begins accruing (0 means immediately).
    - mitigated_h: the hour within the outage when the cost stops accruing (None means it runs through the full outage).

    The cost accrues over the half-open interval [start_h, mitigated_h) clipped to [0, duration_h].
    """
    d = _clamp_hours(duration_h)

    s_raw = 0.0 if start_h is None else float(start_h)
    s = _clamp_hours(s_raw)

    if mitigated_h is None:
        m = d
    else:
        m = _clamp_hours(float(mitigated_h))

    if m > d:
        m = d
    if m <= s:
        return 0.0
    return m - s


def _clamp_nonnegative(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if v > 0 else 0.0


def _clamp_probability(p: float) -> float:
    """
    Clamp probability into [0, 1]. If parsing fails, return 0.
    """
    try:
        v = float(p)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _fraction_between(x: float, x0: float, x1: float) -> float:
    """
    Return fraction of x between [x0, x1] clamped to [0, 1].
    """
    if x1 <= x0:
        return 1.0 if x >= x0 else 0.0
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    return (x - x0) / (x1 - x0)


def _spoilage_fraction(duration_h: float, start_h: float, end_h: float, half_h: Optional[float] = None) -> float:
    """
    A one-time spoilage fraction that increases from 0 to 1 as outage duration increases.

    - If duration < start_h: 0
    - If duration >= end_h: 1
    - If half_h is provided and lies between start and end, then:
        * fraction is linear from 0 at start_h to 0.5 at half_h
        * and linear from 0.5 at half_h to 1 at end_h
      Otherwise: linear from 0 at start_h to 1 at end_h.
    """
    d = _clamp_hours(duration_h)
    s = _clamp_hours(start_h)
    e = _clamp_hours(end_h)

    if d <= s:
        return 0.0
    if e <= s:
        return 1.0

    if d >= e:
        return 1.0

    if half_h is None:
        return _fraction_between(d, s, e)

    h = _clamp_hours(half_h)
    if h <= s or h >= e:
        return _fraction_between(d, s, e)

    # Piecewise linear with a 50% point
    if d <= h:
        return 0.5 * _fraction_between(d, s, h)
    return 0.5 + 0.5 * _fraction_between(d, h, e)


def _clamp_hours(h: float) -> float:
    return _clamp_nonnegative(h)


@dataclass(frozen=True)
class DamagedEquipmentCost:
    """
    Equipment Damage Costs ($) = C x N x P

    C = average cost of equipment repair or replacement
    N = number of pieces of equipment damaged
    P = probability of damage from outage
    """

    name: str = "Damaged equipment cost"
    avg_repair_replacement_cost: float = 0.0  # C
    number_of_pieces_damaged: float = 0.0  # N (float to tolerate partial/estimated counts)
    probability_of_damage: float = 0.0  # P

    def cost(self) -> float:
        c = _clamp_nonnegative(self.avg_repair_replacement_cost)
        n = _clamp_nonnegative(self.number_of_pieces_damaged)
        p = _clamp_probability(self.probability_of_damage)
        return c * n * p


@dataclass(frozen=True)
class DowntimeCost:
    """
    Downtime Costs ($) = C x H

    C = cost per hour facility is idle
    H = hours of downtime
    """

    name: str = "Downtime cost"
    cost_per_hour_idle: float = 0.0  # C
    hours_of_downtime: float = 0.0  # H

    def cost(self) -> float:
        c = _clamp_nonnegative(self.cost_per_hour_idle)
        h = _clamp_nonnegative(self.hours_of_downtime)
        return c * h


@dataclass(frozen=True)
class LostDataCost:
    """
    Lost Data or Experiments ($) = V x P

    V = value of stored data
    P = probability of loss
    """

    name: str = "Lost data cost"
    value_of_stored_data: float = 0.0  # V
    probability_of_loss: float = 0.0  # P

    def cost(self) -> float:
        v = _clamp_nonnegative(self.value_of_stored_data)
        p = _clamp_probability(self.probability_of_loss)
        return v * p


@dataclass(frozen=True)
class ProcessInterruptionCost:
    """
    Interruption Costs ($) = C x H + IO

    C = average fully-burdened hourly employee costs (wage plus overhead)
    H = hours of staff time to reset process
    IO = lost inputs and outputs
    """

    name: str = "Process interruption cost"
    fully_burdened_hourly_employee_cost: float = 0.0  # C
    hours_staff_time_to_reset: float = 0.0  # H
    lost_inputs_outputs: float = 0.0  # IO (optional)

    def cost(self) -> float:
        c = _clamp_nonnegative(self.fully_burdened_hourly_employee_cost)
        h = _clamp_nonnegative(self.hours_staff_time_to_reset)
        io = _clamp_nonnegative(self.lost_inputs_outputs)
        return (c * h) + io


@dataclass(frozen=True)
class RestartCost:
    """
    Restart Costs ($) = C x H + M

    C = average fully-burdened hourly employee costs (wage plus overhead)
    H = hours of staff time required to restart
    M = additional restart costs
    """

    name: str = "Restart cost"
    fully_burdened_hourly_employee_cost: float = 0.0  # C
    hours_staff_time_to_restart: float = 0.0  # H
    additional_restart_costs: float = 0.0  # M

    def cost(self) -> float:
        c = _clamp_nonnegative(self.fully_burdened_hourly_employee_cost)
        h = _clamp_nonnegative(self.hours_staff_time_to_restart)
        m = _clamp_nonnegative(self.additional_restart_costs)
        return (c * h) + m


@dataclass(frozen=True)
class OtherFixedCost:
    """
    A catch-all fixed cost entered directly as a dollar amount.
    """

    name: str
    amount: float = 0.0

    def cost(self) -> float:
        return _clamp_nonnegative(self.amount)


# -------------------------------------------------------------------
# Spoilage costs (one-time losses that depend on outage duration)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class PerishableFoodSpoilageCost:
    """
    A linear (or piecewise-linear) spoilage curve for perishable products.

    Cost ($) = V x F(duration)

    V = total value of spoilable product
    F(duration) = spoilage fraction in [0, 1] based on outage duration
    """

    name: str = "Perishable food"
    total_value_spoilable_product: float = 0.0  # V
    hour_spoilage_begins: float = 0.0  # start
    hour_of_100_percent_spoilage: float = 0.0  # end
    hour_of_50_percent_spoilage: Optional[float] = None  # optional midpoint

    def cost(self, outage_duration_hours: float) -> float:
        v = _clamp_nonnegative(self.total_value_spoilable_product)
        frac = _spoilage_fraction(
            duration_h=outage_duration_hours,
            start_h=self.hour_spoilage_begins,
            end_h=self.hour_of_100_percent_spoilage,
            half_h=self.hour_of_50_percent_spoilage,
        )
        return v * frac


@dataclass(frozen=True)
class MissedDeadlinesSpoilageCost:
    """
    A one-time missed-deadline value triggered by an outage.

    By default, this cost is incurred if the outage lasts at least trigger_hour(s).
    """

    name: str = "Missed deadlines"
    total_value_missed_deadline: float = 0.0
    trigger_hour: float = 0.0  # 0 means any outage triggers

    def cost(self, outage_duration_hours: float) -> float:
        v = _clamp_nonnegative(self.total_value_missed_deadline)
        t = _clamp_hours(self.trigger_hour)
        d = _clamp_hours(outage_duration_hours)
        return v if d >= t else 0.0


@dataclass(frozen=True)
class OtherSpoilageCost:
    """
    A generic spoilage curve for any spoilable product.
    """

    name: str
    total_value_spoilable_product: float = 0.0
    hour_spoilage_begins: float = 0.0
    hour_of_100_percent_spoilage: float = 0.0
    hour_of_50_percent_spoilage: Optional[float] = None

    def cost(self, outage_duration_hours: float) -> float:
        v = _clamp_nonnegative(self.total_value_spoilable_product)
        frac = _spoilage_fraction(
            duration_h=outage_duration_hours,
            start_h=self.hour_spoilage_begins,
            end_h=self.hour_of_100_percent_spoilage,
            half_h=self.hour_of_50_percent_spoilage,
        )
        return v * frac


# -------------------------------------------------------------------
# Incremental costs (hourly costs that accumulate with outage duration)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class BackupFuelIncrementalCost:
    """
    Backup Fuel Cost ($/hour) = GL x 1/FE x FC + A

    GL = generator load [kW]
    FE = fuel efficiency [kWh/gallon]
    FC = fuel cost [$/gallon]
    A = additional hourly backup system costs
    """

    name: str = "Backup fuel cost"
    generator_load_kw: float = 0.0  # GL
    fuel_efficiency_kwh_per_gallon: float = 13.0  # FE
    fuel_cost_per_gallon: float = 0.0  # FC
    additional_hourly_backup_costs: float = 0.0  # A
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def hourly_cost(self) -> float:
        gl = _clamp_nonnegative(self.generator_load_kw)
        fe = _clamp_nonnegative(self.fuel_efficiency_kwh_per_gallon)
        fc = _clamp_nonnegative(self.fuel_cost_per_gallon)
        a = _clamp_nonnegative(self.additional_hourly_backup_costs)
        if fe <= 0:
            return a
        return (gl * (1.0 / fe) * fc) + a

    def cost(self, outage_duration_hours: float) -> float:
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        return self.hourly_cost() * hours * _weekly_hours_multiplier(self.avg_weekly_business_hours)


@dataclass(frozen=True)
class RentedEquipmentIncrementalCost:
    name: str = "Rented equipment costs"
    hourly_cost: float = 0.0
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def cost(self, outage_duration_hours: float) -> float:
        hc = _clamp_nonnegative(self.hourly_cost)
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        return hc * hours * _weekly_hours_multiplier(self.avg_weekly_business_hours)


@dataclass(frozen=True)
class CustomerSalesIncrementalCost:
    """
    Lost Customer Sales = WR x P

    WR = weekly revenues
    P = percent decrease in sales

    Hourly cost is computed as (WR x P) / H_week where H_week defaults to 168
    (or uses avg_weekly_business_hours if provided).
    """

    name: str = "Customer sales cost"
    weekly_revenues: float = 0.0  # WR
    percent_decrease_in_sales: float = 0.0  # P in [0,1]
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def hourly_cost(self) -> float:
        wr = _clamp_nonnegative(self.weekly_revenues)
        p = _clamp_probability(self.percent_decrease_in_sales)
        denom = _clamp_nonnegative(self.avg_weekly_business_hours) if self.avg_weekly_business_hours is not None else 168.0
        if denom <= 0:
            denom = 168.0
        return (wr * p) / denom

    def cost(self, outage_duration_hours: float) -> float:
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        # No extra weekly-hours multiplier here because hourly_cost already accounts for it.
        return self.hourly_cost() * hours


@dataclass(frozen=True)
class InterruptedProductionIncrementalCost:
    """
    Lost Output ($/hour) = NO x P x OV

    NO = units of output per hour during normal conditions
    P = percent reduction in output due to outage
    OV = dollar value per unit of output
    """

    name: str = "Interrupted production cost"
    normal_output_units_per_hour: float = 0.0  # NO
    percent_reduction_output: float = 0.0  # P in [0,1]
    outage_value_per_unit: float = 0.0  # OV
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def hourly_cost(self) -> float:
        no = _clamp_nonnegative(self.normal_output_units_per_hour)
        p = _clamp_probability(self.percent_reduction_output)
        ov = _clamp_nonnegative(self.outage_value_per_unit)
        return no * p * ov

    def cost(self, outage_duration_hours: float) -> float:
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        return self.hourly_cost() * hours * _weekly_hours_multiplier(self.avg_weekly_business_hours)


@dataclass(frozen=True)
class StaffProductivityIncrementalCost:
    """
    Outage Productivity Costs ($/hour) = C x P x N

    C = fully-burdened hourly employee costs
    P = percent of work that cannot be completed during outage
    N = number of people affected
    """

    name: str = "Staff productivity cost"
    fully_burdened_hourly_employee_cost: float = 0.0  # C
    percent_work_not_completed: float = 0.0  # P in [0,1]
    number_people_affected: float = 0.0  # N
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def hourly_cost(self) -> float:
        c = _clamp_nonnegative(self.fully_burdened_hourly_employee_cost)
        p = _clamp_probability(self.percent_work_not_completed)
        n = _clamp_nonnegative(self.number_people_affected)
        return c * p * n

    def cost(self, outage_duration_hours: float) -> float:
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        return self.hourly_cost() * hours * _weekly_hours_multiplier(self.avg_weekly_business_hours)


@dataclass(frozen=True)
class SafetyIncrementalCost:
    """
    Cost of Safety ($/hour) = N x P

    N = number of people
    P = hourly cost of reduced safety per person
    """

    name: str = "Safety cost"
    number_people: float = 0.0  # N
    hourly_cost_per_person: float = 0.0  # P
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def hourly_cost(self) -> float:
        n = _clamp_nonnegative(self.number_people)
        p = _clamp_nonnegative(self.hourly_cost_per_person)
        return n * p

    def cost(self, outage_duration_hours: float) -> float:
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        return self.hourly_cost() * hours * _weekly_hours_multiplier(self.avg_weekly_business_hours)


@dataclass(frozen=True)
class OtherIncrementalCost:
    name: str
    hourly_cost: float = 0.0
    hour_cost_starts: Optional[float] = None
    hour_cost_mitigated: Optional[float] = None
    avg_weekly_business_hours: Optional[float] = None

    def cost(self, outage_duration_hours: float) -> float:
        hc = _clamp_nonnegative(self.hourly_cost)
        hours = _active_hours(outage_duration_hours, self.hour_cost_starts, self.hour_cost_mitigated)
        return hc * hours * _weekly_hours_multiplier(self.avg_weekly_business_hours)


FixedCostItem = Union[
    DamagedEquipmentCost,
    DowntimeCost,
    LostDataCost,
    ProcessInterruptionCost,
    RestartCost,
    OtherFixedCost,
]

SpoilageCostItem = Union[
    PerishableFoodSpoilageCost,
    MissedDeadlinesSpoilageCost,
    OtherSpoilageCost,
]

IncrementalCostItem = Union[
    BackupFuelIncrementalCost,
    RentedEquipmentIncrementalCost,
    CustomerSalesIncrementalCost,
    InterruptedProductionIncrementalCost,
    StaffProductivityIncrementalCost,
    SafetyIncrementalCost,
    OtherIncrementalCost,
]


def total_fixed_cost(costs: List[FixedCostItem]) -> float:
    """
    Sum of all fixed costs for a single outage event.
    """
    return float(sum(_clamp_nonnegative(c.cost()) for c in (costs or [])))


def fixed_cost_breakdown(costs: List[FixedCostItem]) -> List[dict]:
    """
    Return a simple breakdown suitable for UI tables:
      [{"name": "...", "cost": 123.0}, ...]
    """
    breakdown: List[dict] = []
    for c in (costs or []):
        breakdown.append({"name": getattr(c, "name", "Fixed cost"), "cost": float(_clamp_nonnegative(c.cost()))})
    return breakdown


def total_spoilage_cost(costs: List[SpoilageCostItem], outage_duration_hours: float) -> float:
    """
    Sum of all spoilage costs for a single outage event, given the outage duration.
    """
    return float(sum(_clamp_nonnegative(c.cost(outage_duration_hours)) for c in (costs or [])))


def spoilage_cost_breakdown(costs: List[SpoilageCostItem], outage_duration_hours: float) -> List[dict]:
    """
    Return a simple breakdown suitable for UI tables:
      [{"name": "...", "cost": 123.0}, ...]
    """
    breakdown: List[dict] = []
    for c in (costs or []):
        breakdown.append(
            {"name": getattr(c, "name", "Spoilage cost"), "cost": float(_clamp_nonnegative(c.cost(outage_duration_hours)))}
        )
    return breakdown


def total_incremental_cost(costs: List[IncrementalCostItem], outage_duration_hours: float) -> float:
    """Sum of all incremental costs for a single outage event."""
    return float(sum(_clamp_nonnegative(c.cost(outage_duration_hours)) for c in (costs or [])))


def incremental_cost_breakdown(costs: List[IncrementalCostItem], outage_duration_hours: float) -> List[dict]:
    """Return a simple breakdown suitable for UI tables."""
    breakdown: List[dict] = []
    for c in (costs or []):
        breakdown.append(
            {"name": getattr(c, "name", "Incremental cost"), "cost": float(_clamp_nonnegative(c.cost(outage_duration_hours)))}
        )
    return breakdown


# -------------------------------------------------------------------
# PRESTO outage event helpers
# -------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Accept trailing Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def presto_event_duration_hours(event: Dict[str, Any]) -> float:
    """Best-effort duration in hours from a PRESTO event record.

    Supports both the app's normalized `events_tidy` shape and several raw PRESTO variants.
    Returns 0.0 if duration cannot be determined.
    """
    if not isinstance(event, dict):
        return 0.0

    # Preferred normalized keys
    for key in ["duration_hours", "durationHour", "durationHours", "outage_duration_hours"]:
        v = _to_float(event.get(key))
        if v is not None:
            return max(0.0, v)

    # Minutes variants
    for key in ["duration_minutes", "durationMinute", "durationMinutes", "duration_mins", "minutes"]:
        v = _to_float(event.get(key))
        if v is not None:
            return max(0.0, v / 60.0)

    # Derive from timestamps if present
    start = _parse_iso_datetime(event.get("start")) or _parse_iso_datetime(event.get("start_time"))
    end = _parse_iso_datetime(event.get("end")) or _parse_iso_datetime(event.get("end_time"))
    if start is not None and end is not None:
        try:
            delta_h = (end - start).total_seconds() / 3600.0
            return max(0.0, float(delta_h))
        except Exception:
            return 0.0

    return 0.0


def presto_event_start(event: Dict[str, Any]) -> Optional[datetime]:
    """Best-effort start datetime from a PRESTO event record."""
    if not isinstance(event, dict):
        return None
    return (
        _parse_iso_datetime(event.get("start"))
        or _parse_iso_datetime(event.get("start_time"))
        or _parse_iso_datetime(event.get("begin"))
        or _parse_iso_datetime(event.get("begin_time"))
    )


def presto_event_end(event: Dict[str, Any]) -> Optional[datetime]:
    """Best-effort end datetime from a PRESTO event record."""
    if not isinstance(event, dict):
        return None
    return (
        _parse_iso_datetime(event.get("end"))
        or _parse_iso_datetime(event.get("end_time"))
        or _parse_iso_datetime(event.get("finish"))
        or _parse_iso_datetime(event.get("finish_time"))
    )

# -------------------------------------------------------------------
# Placeholders for later layers (incremental costs, spoilage, aggregation)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class CdfParameters:
    """Container for the full CDF model inputs.

    - fixed_costs: one-time costs incurred at outage start (independent of duration)
    - spoilage_costs: one-time, duration-dependent costs (each item triggers once as duration crosses thresholds)
    - incremental_costs: ongoing costs that accrue over time during an outage, respecting start/mitigation hours
    """

    fixed_costs: List[FixedCostItem] = field(default_factory=list)
    spoilage_costs: List[SpoilageCostItem] = field(default_factory=list)
    incremental_costs: List[IncrementalCostItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Defensive: coerce None to empty list if any caller passes None.
        object.__setattr__(self, "fixed_costs", self.fixed_costs or [])
        object.__setattr__(self, "spoilage_costs", self.spoilage_costs or [])
        object.__setattr__(self, "incremental_costs", self.incremental_costs or [])


def outage_cost_fixed_only(params: CdfParameters) -> float:
    """
    Compute the outage cost using only fixed costs (one-time costs at outage start).
    """
    return total_fixed_cost(params.fixed_costs)


def outage_cost_fixed_and_spoilage(params: CdfParameters, outage_duration_hours: float) -> float:
    """
    Compute the outage cost using fixed costs plus spoilage costs (duration-dependent, one-time triggers).
    """
    return total_fixed_cost(params.fixed_costs) + total_spoilage_cost(params.spoilage_costs, outage_duration_hours)


def outage_cost_fixed_spoilage_and_incremental(
    params: CdfParameters, outage_duration_hours: float
) -> float:
    """Compute outage cost from fixed + spoilage + incremental components."""
    return (
        total_fixed_cost(params.fixed_costs)
        + total_spoilage_cost(params.spoilage_costs, outage_duration_hours)
        + total_incremental_cost(params.incremental_costs, outage_duration_hours)
    )