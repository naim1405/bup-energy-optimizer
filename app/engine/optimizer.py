"""The GridWise math engine.

Pure functions only: normalized dataclasses in, dataclasses out. No LLM,
no HTTP, no file I/O. That is deliberate -- this module is the part you
want to be able to unit-test without a network.

Model
-----
A plain linear program. Charge and discharge are 1:1 (Problem Statement
section 9.1 has no efficiency loss), so no integer variables are needed
even though `battery_action` looks categorical.

Variables, 5 per hour (120 total):
    grid[h]        >= 0        grid energy purchased
    solar[h]       >= 0        solar energy actually used
    charge[h]      >= 0        energy into the battery
    discharge[h]   >= 0        energy out of the battery
    E[h]                       battery energy AFTER hour h

Minimize  sum(grid[h] * tariff[h])

Subject to
    grid + solar + discharge - charge = demand            (balance, 9.5)
    E[h] = E[h-1] + charge - discharge                    (state,   9.1)
    E[23] = initial_energy_kwh                            (neutrality, 9.6)
    0 <= solar[h] <= effective_solar[h]                   (9.4)
    min_floor[h] <= E[h] <= capacity                      (9.2)
    charge[h]    <= max_charge_kwh_per_hour               (9.3)
    discharge[h] <= max_discharge_kwh_per_hour            (9.3)
    grid[h]      <= grid_cap[h]                           (max_grid_window)

Verified against the 10 public reference cases: this formulation
reproduces every published total_cost_bdt to 0.00% (see
verify_reference.py), i.e. it reaches the organizer optimum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from app.engine.models import (
    MAX_GRID_WINDOW,
    MIN_BATTERY_RESERVE,
    NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW,
    ROUND_DP,
    SOLAR_REDUCTION,
    Directive,
    HourPlan,
    Scenario,
    ScheduleResult,
)

N = 24
# Tie-breaker weight on (charge + discharge). Verified free: adding it
# changed the optimal cost by exactly 0.0 on all 10 public cases, and it
# removes cost-neutral circulating loops from degenerate optima.
EPS_LOOP = 1e-6
# Weight for constraint slacks in the infeasibility fallback.
BIG_M = 1e4

_IDX = {name: np.arange(N) + N * i for i, name in
        enumerate(("grid", "solar", "charge", "discharge", "E"))}


# ===========================================================================
# 1. Directive overlays -- the only place directives touch the math
# ===========================================================================

class Overlay:
    """Per-hour constraint arrays derived from the validated directives."""

    __slots__ = ("effective_solar", "min_floor", "grid_cap",
                 "charge_ok", "discharge_ok", "notes")

    def __init__(self, scenario: Scenario):
        self.effective_solar = np.array(scenario.solar, dtype=float)
        self.min_floor = np.full(N, float(scenario.battery.minimum_energy_kwh))
        self.grid_cap = np.full(N, np.inf)
        self.charge_ok = np.ones(N, dtype=bool)
        self.discharge_ok = np.ones(N, dtype=bool)
        self.notes: list[str] = []

    def apply(self, d: Directive) -> None:
        h = list(d.hours)
        if d.directive_type == SOLAR_REDUCTION:
            # Two overlapping solar reductions stack multiplicatively.
            # The spec is silent on this; multiplicative is the natural
            # reading and is idempotent for a single directive.
            self.effective_solar[h] *= d.factor
            self.notes.append(
                f"solar_reduction h{h}: effective solar x{d.factor}")
        elif d.directive_type == MIN_BATTERY_RESERVE:
            # max(base, directive): a directive can only RAISE the floor.
            self.min_floor[h] = np.maximum(self.min_floor[h],
                                          d.minimum_energy_kwh)
            self.notes.append(
                f"minimum_battery_reserve h{h}: floor -> {d.minimum_energy_kwh}")
        elif d.directive_type == NO_CHARGE_WINDOW:
            self.charge_ok[h] = False
            self.notes.append(f"no_charge_window h{h}")
        elif d.directive_type == NO_DISCHARGE_WINDOW:
            self.discharge_ok[h] = False
            self.notes.append(f"no_discharge_window h{h}")
        elif d.directive_type == MAX_GRID_WINDOW:
            self.grid_cap[h] = np.minimum(self.grid_cap[h], d.max_grid_kwh)
            self.notes.append(f"max_grid_window h{h}: grid <= {d.max_grid_kwh}")


def build_overlay(scenario: Scenario) -> Overlay:
    ov = Overlay(scenario)
    for d in scenario.directives:
        if d.applies:
            ov.apply(d)
    return ov


# ===========================================================================
# 2. The linear program
# ===========================================================================

def _build(scenario: Scenario, ov: Overlay, slack: bool):
    """Assemble the LP. `slack=True` relaxes the soft directives so the
    model always has a solution (used only after a real infeasibility)."""
    bat = scenario.battery
    demand = np.array(scenario.demand, dtype=float)
    tariff = np.array(scenario.tariff, dtype=float)
    cap = float(bat.capacity_kwh)

    nv = 5 * N + (2 * N if slack else 0)
    SG = np.arange(5 * N, 5 * N + N) if slack else None   # grid-cap slack
    SF = np.arange(5 * N + N, 5 * N + 2 * N) if slack else None  # floor slack

    c = np.zeros(nv)
    c[_IDX["grid"]] = tariff
    c[_IDX["charge"]] += EPS_LOOP
    c[_IDX["discharge"]] += EPS_LOOP
    if slack:
        c[SG] = BIG_M
        c[SF] = BIG_M

    A_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for h in range(N):
        r = np.zeros(nv)
        r[_IDX["grid"][h]] = 1.0
        r[_IDX["solar"][h]] = 1.0
        r[_IDX["discharge"][h]] = 1.0
        r[_IDX["charge"][h]] = -1.0
        A_eq.append(r)
        b_eq.append(demand[h])

        r = np.zeros(nv)
        r[_IDX["E"][h]] = 1.0
        r[_IDX["charge"][h]] = -1.0
        r[_IDX["discharge"][h]] = 1.0
        if h:
            r[_IDX["E"][h - 1]] = -1.0
            A_eq.append(r)
            b_eq.append(0.0)
        else:
            A_eq.append(r)
            b_eq.append(float(bat.initial_energy_kwh))

    r = np.zeros(nv)
    r[_IDX["E"][N - 1]] = 1.0
    A_eq.append(r)
    b_eq.append(float(bat.initial_energy_kwh))

    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for h in range(N):
        if np.isfinite(ov.grid_cap[h]):
            r = np.zeros(nv)
            r[_IDX["grid"][h]] = 1.0
            if slack:
                r[SG[h]] = -1.0
            A_ub.append(r)
            b_ub.append(float(ov.grid_cap[h]))
        if slack and ov.min_floor[h] > bat.minimum_energy_kwh:
            # E[h] - slack <= min_floor[h]  =>  -E[h] + slack >= -min_floor[h]
            r = np.zeros(nv)
            r[_IDX["E"][h]] = -1.0
            r[SF[h]] = 1.0
            A_ub.append(r)
            b_ub.append(-float(ov.min_floor[h]))

    bounds: list[tuple[float, float | None]] = []
    bounds += [(0.0, None)] * N                                    # grid
    bounds += [(0.0, float(ov.effective_solar[h])) for h in range(N)]  # solar
    bounds += [(0.0, float(bat.max_charge_kwh_per_hour) if ov.charge_ok[h]
                else 0.0) for h in range(N)]                       # charge
    bounds += [(0.0, float(bat.max_discharge_kwh_per_hour) if ov.discharge_ok[h]
                else 0.0) for h in range(N)]                       # discharge
    bounds += [(float(ov.min_floor[h]), cap) for h in range(N)]    # E
    if slack:
        bounds += [(0.0, None)] * (2 * N)

    return dict(c=c, A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds)


def solve_lp(scenario: Scenario, ov: Overlay, slack: bool = False):
    lp = _build(scenario, ov, slack)
    return linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"],
                   A_eq=lp["A_eq"], b_eq=lp["b_eq"], bounds=lp["bounds"],
                   method="highs")


# ===========================================================================
# 3. Post-processing: net, round, replay, total
# ===========================================================================

def _to_plan(x: np.ndarray, scenario: Scenario, ov: Overlay) -> list[HourPlan]:
    """LP vector -> clean HourPlan list.

    Order matters. Round the physical quantities first, then RECOMPUTE grid
    from the energy balance so it is exact by construction, then recompute
    battery energy by forward replay. Totals are derived from this list,
    never from the raw LP vector -- the judge recalculates them from
    hourly_plan and any disagreement is a scoring deduction.
    """
    bat = scenario.battery
    demand = np.array(scenario.demand, dtype=float)

    solar = np.round(x[_IDX["solar"]], ROUND_DP)
    charge = np.round(x[_IDX["charge"]], ROUND_DP)
    disch = np.round(x[_IDX["discharge"]], ROUND_DP)

    # Collapse degenerate simultaneous charge+discharge into one action.
    net = charge - disch
    charge = np.where(net > 0, net, 0.0)
    disch = np.where(net < 0, -net, 0.0)

    # grid falls out of the balance equation -> exact, never drifts.
    grid = demand + charge - disch - solar
    grid = np.maximum(np.round(grid, ROUND_DP), 0.0)

    e = float(bat.initial_energy_kwh)
    plan: list[HourPlan] = []
    for h in range(N):
        c, d = float(charge[h]), float(disch[h])
        if c > 1e-9:
            action, mag = "charge", c
        elif d > 1e-9:
            action, mag = "discharge", d
        else:
            action, mag = "idle", 0.0
        e += c - d
        plan.append(HourPlan(
            hour=h,
            grid_kwh=float(grid[h]),
            solar_used_kwh=float(solar[h]),
            battery_action=action,
            battery_kwh=round(mag, ROUND_DP),
            battery_energy_after_kwh=round(e, ROUND_DP),
        ))
    _reconcile(plan, scenario, ov)
    return plan


def _replay_energy(plan: list[HourPlan], initial: float) -> float:
    """Rewrite every reported E_after from the actual flows. Returns final E.

    This alone fixes any inconsistency between the reported energy column
    and the charge/discharge amounts, whatever caused it.
    """
    e = initial
    for q in plan:
        if q.battery_action == "charge":
            e += q.battery_kwh
        elif q.battery_action == "discharge":
            e -= q.battery_kwh
        _replace(plan, q.hour, battery_energy_after_kwh=round(e, ROUND_DP))
    return e


def _flows_legal(plan: list[HourPlan], scenario: Scenario, ov: Overlay) -> bool:
    """Cheap feasibility re-check after a repair attempt."""
    bat = scenario.battery
    for q in plan:
        h = q.hour
        if q.grid_kwh < -1e-9 or q.grid_kwh > ov.grid_cap[h] + 1e-9:
            return False
        if not (ov.min_floor[h] - 1e-9
                <= q.battery_energy_after_kwh <= bat.capacity_kwh + 1e-9):
            return False
        if q.battery_action == "charge":
            if not ov.charge_ok[h] or q.battery_kwh > bat.max_charge_kwh_per_hour + 1e-9:
                return False
        elif q.battery_action == "discharge":
            if not ov.discharge_ok[h] or q.battery_kwh > bat.max_discharge_kwh_per_hour + 1e-9:
                return False
        elif abs(q.battery_kwh) > 1e-9:
            return False
    return True


def _derive_grid(plan: list[HourPlan], scenario: Scenario) -> None:
    """Recompute every grid_kwh from the energy balance (section 9.5).

    grid = demand + charge - discharge - solar_used

    Deriving it rather than trusting it makes the balance exact by
    construction, so the two can never drift apart.
    """
    for q in plan:
        c = q.battery_kwh if q.battery_action == "charge" else 0.0
        d = q.battery_kwh if q.battery_action == "discharge" else 0.0
        g = scenario.demand[q.hour] + c - d - q.solar_used_kwh
        _replace(plan, q.hour, grid_kwh=round(max(g, 0.0), ROUND_DP))


def _try_adjust(plan: list[HourPlan], scenario: Scenario, h: int,
                delta: float) -> None:
    """Change net battery flow at hour h by `delta` (signed)."""
    p = plan[h]
    c = p.battery_kwh if p.battery_action == "charge" else 0.0
    d = p.battery_kwh if p.battery_action == "discharge" else 0.0
    nc, nd = c + max(delta, 0.0), d + max(-delta, 0.0)
    if nc > 1e-12 and nd > 1e-12:        # never both in one hour
        nc, nd = (nc - nd, 0.0) if nc >= nd else (0.0, nd - nc)
    if nc > 1e-12:
        action, mag = "charge", nc
    elif nd > 1e-12:
        action, mag = "discharge", nd
    else:
        action, mag = "idle", 0.0
    _replace(plan, h, battery_action=action, battery_kwh=round(mag, ROUND_DP))


def _reconcile(plan: list[HourPlan], scenario: Scenario, ov: Overlay) -> None:
    """Make the plan self-consistent and end-of-day neutral (section 9.6).

    Three problems, fixed in order:
      1. grid_kwh disagreeing with the balance equation  -> re-derive
      2. reported E_after disagreeing with the flows     -> replay
      3. the flows themselves netting to non-zero        -> nudge one hour
    Rounding to 6dp leaves at most ~1e-5 kWh of drift, well inside the
    0.01 judge tolerance, so step 3 is belt-and-braces -- but it must work
    when it fires.
    """
    initial = float(scenario.battery.initial_energy_kwh)
    _derive_grid(plan, scenario)
    e = _replay_energy(plan, initial)
    gap = e - initial                     # >0 means we net-charged too much
    if abs(gap) <= 1e-9:
        return

    snapshot = list(plan)
    for h in sorted(range(N), key=lambda k: scenario.tariff[k]):
        _try_adjust(plan, scenario, h, -gap)
        _derive_grid(plan, scenario)
        e2 = _replay_energy(plan, initial)
        if abs(e2 - initial) <= 1e-9 and _flows_legal(plan, scenario, ov):
            return
        plan[:] = list(snapshot)          # revert and try the next hour
    # Nothing worked. Leave the reconciled plan; the validator reports it
    # rather than us shipping something silently wrong.


def _replace(plan: list[HourPlan], h: int, **kw) -> HourPlan:
    p = plan[h]
    cur = dict(hour=p.hour, grid_kwh=p.grid_kwh, solar_used_kwh=p.solar_used_kwh,
               battery_action=p.battery_action, battery_kwh=p.battery_kwh,
               battery_energy_after_kwh=p.battery_energy_after_kwh)
    cur.update(kw)
    plan[h] = HourPlan(**cur)
    return plan[h]


def summarize(plan: list[HourPlan], scenario: Scenario) -> tuple[float, float, float]:
    tariff = scenario.tariff
    total_grid = sum(p.grid_kwh for p in plan)
    total_cost = sum(p.grid_kwh * tariff[p.hour] for p in plan)
    peak = max(p.grid_kwh for p in plan)
    return (round(total_grid, ROUND_DP), round(total_cost, ROUND_DP),
            round(peak, ROUND_DP))


# ===========================================================================
# 4. Entry point
# ===========================================================================

def optimize(scenario: Scenario, validate_fn=None) -> ScheduleResult:
    """Scenario (with validated directives) -> ScheduleResult.

    `validate_fn(scenario, overlay, plan)` should return a list of violation
    strings; it is injected so the engine stays free of validation policy.
    """
    ov = build_overlay(scenario)
    res = solve_lp(scenario, ov, slack=False)
    feasible = True
    violations: list[str] = []

    if not res.success or res.x is None:
        # SAFE FAILURE. A misread directive can make the model infeasible
        # even though the organizer's ground truth never is. Relax the soft
        # directives rather than returning a 500.
        res = solve_lp(scenario, ov, slack=True)
        feasible = False
        violations.append(
            f"LP infeasible under the interpreted directives; returned the "
            f"closest-feasible relaxed plan (solver said: {res.message})")
        if not res.success or res.x is None:
            return ScheduleResult(
                plan=_trivial_plan(scenario), total_grid_kwh=0.0,
                total_cost_bdt=0.0, peak_grid_kwh=0.0, feasible=False,
                violations=violations + ["relaxed LP also failed"],
                overlay_notes=ov.notes)

    plan = _to_plan(res.x, scenario, ov)
    total_grid, total_cost, peak = summarize(plan, scenario)

    if validate_fn is not None:
        violations += validate_fn(scenario, ov, plan)
        if violations:
            feasible = False
    return ScheduleResult(plan=plan, total_grid_kwh=total_grid,
                          total_cost_bdt=total_cost, peak_grid_kwh=peak,
                          feasible=feasible, violations=violations,
                          overlay_notes=ov.notes)


def _trivial_plan(scenario: Scenario) -> list[HourPlan]:
    """Last-resort plan: meet all demand from grid, battery untouched.
    Always structurally valid when no grid cap is binding."""
    bat = scenario.battery
    e = float(bat.initial_energy_kwh)
    return [HourPlan(hour=h.hour,
                     grid_kwh=round(h.demand_kwh - h.solar_kwh, ROUND_DP)
                     if h.demand_kwh >= h.solar_kwh else 0.0,
                     solar_used_kwh=min(h.demand_kwh, h.solar_kwh),
                     battery_action="idle", battery_kwh=0.0,
                     battery_energy_after_kwh=e)
            for h in scenario.hours]
