"""Runs the full Module C study and returns every number the notebook reports."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import cohorts as co
from . import cost
from . import simulation as sim
from .config import StudyConfig
from .synthetic import Programme, simulate, without_exploiters


@dataclass
class StudyResult:
    cfg: StudyConfig
    prog: Programme
    summary: dict
    edge_curves: dict[str, pd.DataFrame]
    delayed: pd.DataFrame
    tradeoff: pd.DataFrame
    flags: pd.DataFrame


def run_study(cfg: StudyConfig | None = None) -> StudyResult:
    cfg = cfg or StudyConfig()
    det, prog_cfg, pop = cfg.detection, cfg.programme, cfg.population
    prog = simulate(cfg)
    accounts = prog.accounts
    truth = accounts.set_index("account_id")["segment"]
    seg_ids = {s: set(truth.index[truth == s]) for s in truth.unique()}
    complete_entries = prog.entries[prog.entries["cycle_complete"]]
    complete_steps = prog.steps[prog.steps["cycle_complete"]]
    n_cycles = int(complete_steps["cycle"].nunique())

    disc_e, disc_s = co.split_cycles(prog.entries, prog.steps, 0)
    hold_e, hold_s = co.split_cycles(prog.entries, prog.steps, 1)
    theo_all = co.theoretical_return(complete_steps, prog_cfg.base_return_ratio)
    theo_disc = co.theoretical_return(disc_s, prog_cfg.base_return_ratio)
    theo_hold = co.theoretical_return(hold_s, prog_cfg.base_return_ratio)

    def cohort(e, ids, theo):
        t = co.account_totals(e, ids)
        return {"accounts": len(t), "return": co.bca_return(t, cfg.bootstrap_resamples, cfg.seed).as_dict(),
                "benchmark": co.benchmark_z(t, theo)}

    # 1. Circularity: a list built partly from outcomes, validated on the data that built it.
    listed = co.outcome_list(disc_e, disc_s, det)
    # 2. Behaviour-only cohort, flagged on discovery cycles.
    flags = co.behaviour_cohort(disc_e, disc_s, det)
    behaviour = set(flags.loc[flags["flagged"], "account_id"])
    unknown_to_list = behaviour - listed

    summary = {
        "programme": {
            "entries": len(prog.entries),
            "complete_cycles": n_cycles,
            "mean_cycle_steps": float(prog.cycles["length_steps"].mean()),
            "late_time_share": co.late_baseline(disc_s, det.late_phase),
            "theoretical_return": {"all": theo_all, "discovery": theo_disc, "holdout": theo_hold},
        },
        "circularity": {
            "listed_accounts": len(listed),
            "listed_share_exploiting": float(np.mean([a in seg_ids["exploiting"] for a in listed])) if listed else None,
            "listed_in_sample": cohort(disc_e, listed, theo_disc),
            "listed_holdout": cohort(hold_e, listed, theo_hold),
        },
        "behaviour": {
            "assessed": len(flags),
            **co.detection_quality(behaviour, accounts),
            "in_sample": cohort(disc_e, behaviour, theo_disc),
            "holdout": cohort(hold_e, behaviour, theo_hold),
            "unknown_to_list_accounts": len(unknown_to_list),
            "unknown_to_list_holdout": cohort(hold_e, unknown_to_list, theo_hold),
        },
        "truth": {s: cohort(complete_entries, ids, theo_all) for s, ids in seg_ids.items()},
    }

    # 4. Monte Carlo time-to-edge for the planted strategy, against the timing-independent one.
    regular_spend = float(prog.steps.loc[prog.steps["phase"] < det.late_phase, "spend"].mean())
    late_from = (pop.exploit_phase_low + pop.exploit_phase_high) / 2
    n_exp = len(seg_ids["exploiting"])
    late_extra = n_exp * pop.exploit_active_rate * pop.mean_spend * pop.exploit_spend_multiplier
    pop_spend = sim.expected_population_spend(prog_cfg, regular_spend, late_extra, late_from)
    own = pop.exploit_active_rate * pop.mean_spend * pop.exploit_spend_multiplier
    strategies = {"late_strategy": sim.Strategy(late_from, own), "timing_independent": sim.Strategy(0.0, own)}
    curves, ratios, mc = {}, {}, {}
    for name, strat in strategies.items():
        ratio, spend, reward = sim.simulate_paths(prog_cfg, strat, pop_spend, cfg.mc_paths, cfg.mc_cycles,
                                                  np.random.default_rng(cfg.seed + 1))
        curve = sim.edge_curve(ratio)
        curves[name], ratios[name] = curve, ratio
        tte = {f"+{int(lv * 100)}%": {f"{int(c * 100)}%": sim.time_to_edge(curve, lv, c) for c in (0.5, 0.7, 0.9)}
               for lv in (0.01, 0.02, 0.03)}
        mc[name] = {
            "expected_return": sim.expected_return(prog_cfg, strat, pop_spend),
            "simulated_return": float(reward[:, -1].sum() / spend[:, -1].sum()),
            "time_to_edge_cycles": tte,
            "expected_spend_per_cycle": float(spend[:, -1].mean() / cfg.mc_cycles),
        }
    late = mc["late_strategy"]
    # Sustained-crossing check on individual paths: of the paths, how many stay above +1%
    # for every cycle after the curve-level 90% point.
    late["sustained_share_+1%_from_90%_point"] = sim.sustained_share(
        ratios["late_strategy"], 0.01, late["time_to_edge_cycles"]["+1%"]["90%"] or cfg.mc_cycles
    )
    late["predicted_extraction_per_cycle"] = (
        (late["expected_return"] - mc["timing_independent"]["expected_return"]) * late["expected_spend_per_cycle"]
    )
    summary["monte_carlo"] = mc

    # 5. Cost to the business, from the behaviour cohort measured on holdout cycles.
    n_hold_cycles = int(hold_s["cycle"].nunique())
    hold_totals = co.account_totals(hold_e, behaviour)
    per_acc = cost.extraction_per_account_cycle(hold_totals, theo_hold, n_hold_cycles,
                                                cfg.bootstrap_resamples, cfg.seed)
    delayed = cost.delayed_detection_curve(per_acc, len(behaviour), cfg.cost.max_detection_cycles)
    tradeoff = cost.threshold_tradeoff(prog, cfg, per_acc.value)
    best = tradeoff.loc[tradeoff["expected_cost"].idxmin()]
    never = per_acc.value * cfg.cost.horizon_cycles * n_exp

    counterfactual = simulate(without_exploiters(cfg))
    cf_complete = counterfactual.entries[counterfactual.entries["cycle_complete"]]
    cf_regular = set(counterfactual.accounts.loc[counterfactual.accounts["segment"] == "regular", "account_id"])
    regular_without = co.bca_return(co.account_totals(cf_complete, cf_regular), cfg.bootstrap_resamples, cfg.seed)

    summary["cost"] = {
        "extraction_per_account_cycle": per_acc.as_dict(),
        "extraction_cohort_total_per_cycle": cost.scale(per_acc, len(behaviour)).as_dict(),
        "extraction_cohort_total_period": cost.scale(per_acc, len(behaviour) * n_cycles).as_dict(),
        "mc_predicted_extraction_per_cycle": late["predicted_extraction_per_cycle"],
        "recommended_detection_cycles": int(best["detect_after_cycles"]),
        "expected_cost_at_recommended": float(best["expected_cost"]),
        "expected_cost_never_detect": float(never),
        "expected_cost_avoided": float(never - best["expected_cost"]),
        "pool_flows": cost.pool_flows(
            complete_entries, {"Flagged cohort": behaviour, "Everyone else": set(truth.index) - behaviour},
            prog_cfg.contribution_rate, n_cycles).to_dict("records"),
        "pool_flows_truth": cost.pool_flows(
            complete_entries, {"Planted exploiting": seg_ids["exploiting"], "Heavy legitimate": seg_ids["heavy"],
                               "Regular": seg_ids["regular"]}, prog_cfg.contribution_rate, n_cycles).to_dict("records"),
        "pool_health": cost.pool_health(complete_entries, behaviour, prog_cfg.contribution_rate),
        "pool_health_truth": cost.pool_health(complete_entries, seg_ids["exploiting"], prog_cfg.contribution_rate),
        "regular_return_with_exploiters": summary["truth"]["regular"]["return"],
        "regular_return_without_exploiters": regular_without.as_dict(),
    }
    return StudyResult(cfg, prog, summary, curves, delayed, tradeoff, flags)


def quick_config(seed: int = 7) -> StudyConfig:
    """Smaller study for tests and smoke runs."""
    cfg = StudyConfig(seed=seed)
    return replace(
        cfg,
        population=replace(cfg.population, n_accounts=2000, n_steps=6 * 200),
        cost=replace(cfg.cost, max_detection_cycles=6, window_stride=6),
        bootstrap_resamples=300,
        mc_paths=600,
        mc_cycles=120,
    )
