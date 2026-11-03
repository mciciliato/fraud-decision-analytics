"""Synthetic reward programme with planted ground truth.

Stand-in for the shared generator (Repo 0) until it exists. Fully vectorised
over accounts and steps; only the handful of payouts is processed in a loop.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import StudyConfig


@dataclass
class Programme:
    accounts: pd.DataFrame  # account_id, segment (ground truth), active_from_phase
    entries: pd.DataFrame  # one row per qualifying entry
    steps: pd.DataFrame  # step, cycle, cycle_step, phase, payout, spend, award
    cycles: pd.DataFrame  # one row per completed cycle


def _schedule(cfg: StudyConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Cycle step and payout flag for every step. Payouts do not depend on activity."""
    prog, n = cfg.programme, cfg.population.n_steps
    hazard = prog.hazard()
    draws = rng.random(n)
    cycle_step = np.empty(n, dtype=np.int64)
    payout = np.zeros(n, dtype=bool)
    a = 1
    for t in range(n):
        cycle_step[t] = a
        payout[t] = draws[t] < hazard[a - 1]
        a = 1 if payout[t] else a + 1
    cycle = np.concatenate(([0], np.cumsum(payout)[:-1]))
    return pd.DataFrame(
        {
            "step": np.arange(n),
            "cycle": cycle,
            "cycle_step": cycle_step,
            "phase": cycle_step / prog.max_cycle_steps,
            "payout": payout,
        }
    )


def _accounts(cfg: StudyConfig, rng: np.random.Generator) -> pd.DataFrame:
    pop = cfg.population
    n = pop.n_accounts
    n_exp, n_heavy = round(n * pop.share_exploiting), round(n * pop.share_heavy)
    segment = np.array(["regular"] * (n - n_exp - n_heavy) + ["exploiting"] * n_exp + ["heavy"] * n_heavy)
    rng.shuffle(segment)
    return pd.DataFrame(
        {
            "account_id": [f"acc_{i:06d}" for i in range(n)],
            "segment": segment,
            "base_rate": rng.gamma(pop.rate_dispersion, pop.mean_entry_rate / pop.rate_dispersion, size=n),
            "active_from_phase": np.where(
                segment == "exploiting", rng.uniform(pop.exploit_phase_low, pop.exploit_phase_high, size=n), np.nan
            ),
        }
    )


def _rates(accounts: pd.DataFrame, phase: np.ndarray, cfg: StudyConfig) -> np.ndarray:
    """Expected entries per step, accounts x steps."""
    pop = cfg.population
    seg = accounts["segment"].to_numpy()
    rate = np.repeat(accounts["base_rate"].to_numpy()[:, None], phase.size, axis=1)

    exp = seg == "exploiting"
    active = phase[None, :] >= accounts["active_from_phase"].to_numpy()[exp, None]
    rate[exp] = np.where(active, pop.exploit_active_rate, pop.exploit_background_rate)

    heavy = seg == "heavy"
    s = pop.heavy_phase_sensitivity
    rate[heavy] *= pop.heavy_rate_multiplier * (1 + s * phase[None, :]) / (1 + s * phase.mean())
    return rate


def simulate(cfg: StudyConfig | None = None) -> Programme:
    cfg = cfg or StudyConfig()
    prog, pop = cfg.programme, cfg.population
    rng = np.random.default_rng(cfg.seed)

    steps = _schedule(cfg, rng)
    accounts = _accounts(cfg, rng)
    counts = rng.poisson(_rates(accounts, steps["phase"].to_numpy(), cfg))

    acc_idx, step_idx = np.nonzero(counts)
    reps = counts[acc_idx, step_idx]
    acc_idx, step_idx = np.repeat(acc_idx, reps), np.repeat(step_idx, reps)
    mult = np.where(accounts["segment"].to_numpy()[acc_idx] == "exploiting", pop.exploit_spend_multiplier, 1.0)
    spend = rng.lognormal(pop.spend_log_mean, pop.spend_log_sd, size=acc_idx.size) * mult
    theta = prog.base_return_ratio / prog.base_return_shape
    base_reward = rng.gamma(prog.base_return_shape, theta, size=acc_idx.size) * spend

    step_spend = np.bincount(step_idx, weights=spend, minlength=len(steps))
    steps["spend"] = step_spend
    steps["award"] = 0.0

    # Pool per cycle: seed plus contributions up to and including the payout step, shared
    # pro rata among the spend of the payout step. An empty payout step rolls the pool over.
    cycle_of_step = steps["cycle"].to_numpy()
    cycle_spend = np.bincount(cycle_of_step, weights=step_spend)
    payout_steps = np.flatnonzero(steps["payout"].to_numpy())
    pools = np.zeros(len(steps))
    carry = 0.0
    for t in payout_steps:
        pool = carry + prog.pool_seed + prog.contribution_rate * cycle_spend[cycle_of_step[t]]
        carry = pool if step_spend[t] == 0 else 0.0
        pools[t] = 0.0 if step_spend[t] == 0 else pool
    steps["award"] = pools
    share = np.divide(spend, step_spend[step_idx], out=np.zeros_like(spend), where=step_spend[step_idx] > 0)
    award = pools[step_idx] * share
    entries = pd.DataFrame(
        {
            "entry_id": [f"ent_{i:08d}" for i in range(acc_idx.size)],
            "account_id": accounts["account_id"].to_numpy()[acc_idx],
            "step": step_idx,
            "cycle": cycle_of_step[step_idx],
            "phase": steps["phase"].to_numpy()[step_idx],
            "spend": spend,
            "base_reward": base_reward,
            "pool_award": award,
        }
    )
    n_complete = int(steps["payout"].sum())
    entries["cycle_complete"] = entries["cycle"] < n_complete
    steps["cycle_complete"] = steps["cycle"] < n_complete
    paid = steps.loc[payout_steps]
    cycles = pd.DataFrame({"cycle": paid["cycle"].to_numpy(), "length_steps": paid["cycle_step"].to_numpy(),
                           "award": paid["award"].to_numpy(), "payout_step_spend": paid["spend"].to_numpy()})
    return Programme(accounts=accounts.drop(columns="base_rate"), entries=entries, steps=steps, cycles=cycles)


def without_exploiters(cfg: StudyConfig) -> StudyConfig:
    """Same programme with the exploiting segment removed (pool-health counterfactual)."""
    return replace(cfg, population=replace(cfg.population, share_exploiting=0.0))
