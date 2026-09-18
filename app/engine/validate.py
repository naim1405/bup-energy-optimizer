"""Independent hour-by-hour replay of every GridWise rule.

Deliberately written from the Problem Statement rather than from the
engine, so it can disagree with the engine. Run it on your own output
before responding -- the judge runs the same checks.
"""

from __future__ import annotations

from app.engine.optimizer import Overlay
from .models import HOURS_PER_DAY, HourPlan, Scenario

TOL = 0.01     # absolute kWh / BDT tolerance per section 11.5


def replay(scenario: Scenario, ov: Overlay,
           plan: list[HourPlan]) -> list[str]:
    """Return a list of violation strings. Empty list == valid."""
    errs: list[str] = []
    bat = scenario.battery
    cap = float(bat.capacity_kwh)

    if len(plan) != HOURS_PER_DAY:
        return [f"hourly_plan has {len(plan)} entries, expected 24"]
    if sorted(p.hour for p in plan) != list(range(HOURS_PER_DAY)):
        return ["hourly_plan hours are not exactly 0-23 once each"]

    e = float(bat.initial_energy_kwh)
    for p in sorted(plan, key=lambda z: z.hour):
        h = p.hour
        g, s, b = p.grid_kwh, p.solar_used_kwh, p.battery_kwh

        # 11.3 finite & non-negative
        for name, v in (("grid_kwh", g), ("solar_used_kwh", s),
                        ("battery_kwh", b),
                        ("battery_energy_after_kwh", p.battery_energy_after_kwh)):
            if v != v or abs(v) == float("inf"):
                errs.append(f"h{h}: {name} is not finite")
            elif v < -TOL:
                errs.append(f"h{h}: {name} is negative ({v})")

        # 9.4 solar usage
        if s > ov.effective_solar[h] + TOL:
            errs.append(f"h{h}: solar_used {s} exceeds effective solar "
                        f"{ov.effective_solar[h]:.4f}")

        # 10.3 action consistency
        if p.battery_action not in ("charge", "discharge", "idle"):
            errs.append(f"h{h}: invalid battery_action {p.battery_action!r}")
        if p.battery_action == "idle" and abs(b) > TOL:
            errs.append(f"h{h}: idle but battery_kwh={b}")

        # 9.1 state transition + 9.3 rate limits + directive windows
        if p.battery_action == "charge":
            if not ov.charge_ok[h]:
                errs.append(f"h{h}: charging inside a no_charge_window")
            if b > float(bat.max_charge_kwh_per_hour) + TOL:
                errs.append(f"h{h}: charge {b} exceeds max_charge_kwh_per_hour")
            e_new = e + b
        elif p.battery_action == "discharge":
            if not ov.discharge_ok[h]:
                errs.append(f"h{h}: discharging inside a no_discharge_window")
            if b > float(bat.max_discharge_kwh_per_hour) + TOL:
                errs.append(f"h{h}: discharge {b} exceeds max_discharge_kwh_per_hour")
            e_new = e - b
        else:
            e_new = e

        if abs(e_new - p.battery_energy_after_kwh) > TOL:
            errs.append(f"h{h}: reported E_after {p.battery_energy_after_kwh} "
                        f"!= replayed {e_new:.4f}")

        # 9.2 bounds (incl. any raised reserve floor)
        if e_new < ov.min_floor[h] - TOL:
            errs.append(f"h{h}: E_after {e_new:.4f} below floor "
                        f"{ov.min_floor[h]:.4f}")
        if e_new > cap + TOL:
            errs.append(f"h{h}: E_after {e_new:.4f} above capacity {cap}")

        # max_grid_window
        if g > ov.grid_cap[h] + TOL:
            errs.append(f"h{h}: grid {g} exceeds cap {ov.grid_cap[h]}")

        # 9.5 energy balance
        disch = b if p.battery_action == "discharge" else 0.0
        chg = b if p.battery_action == "charge" else 0.0
        lhs = g + s + disch
        rhs = scenario.hours[h].demand_kwh + chg
        if abs(lhs - rhs) > TOL:
            errs.append(f"h{h}: balance {lhs:.4f} != {rhs:.4f}")

        e = e_new

    # 9.6 end-of-day neutrality
    if abs(e - float(bat.initial_energy_kwh)) > TOL:
        errs.append(f"end-of-day: final E {e:.4f} != initial "
                    f"{bat.initial_energy_kwh}")
    return errs


def make_validator():
    """Adapter for engine.optimize(validate_fn=...)."""
    def _v(scenario: Scenario, ov: Overlay, plan: list[HourPlan]) -> list[str]:
        return replay(scenario, ov, plan)
    return _v
