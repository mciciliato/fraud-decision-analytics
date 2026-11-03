"""Vectorised Monte Carlo engine for a timing strategy in the reward programme.

The strategy spends a fixed amount in every step whose cycle phase is at or
above its threshold. The rest of the population is summarised by its expected
spend per cycle step. Each simulated cycle draws its payout step from the
programme's payout distribution; the strategy receives its pro rata share of
the pool if it is active in that step. Nothing is looped per entry.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProgrammeConfig


@dataclass(frozen=True)
class Strategy:
    threshold: float  # 0 = timing-independent (spends every step)
    spend_per_step: float


def expected_population_spend(prog: ProgrammeConfig, regular_spend_per_step: float,
                              late_spend: float = 0.0, late_from: float = 1.0) -> np.ndarray:
    """Expected population spend at cycle step a = 1 .. max: a constant level plus optional late spend."""
    phase = np.arange(1, prog.max_cycle_steps + 1) / prog.max_cycle_steps
    return regular_spend_per_step + late_spend * (phase >= late_from)


def _per_cycle_terms(prog: ProgrammeConfig, strategy: Strategy, pop_spend: np.ndarray):
    """For every possible payout step tau: strategy entries, pool share if active, pool size."""
    a = np.arange(1, prog.max_cycle_steps + 1)
    active = a / prog.max_cycle_steps >= strategy.threshold
    n_active = np.cumsum(active)  # active steps up to and including tau
    own = strategy.spend_per_step
    share = np.where(active, own / (pop_spend + own), 0.0)
    prize = prog.pool_seed + prog.contribution_rate * (np.cumsum(pop_spend) + own * n_active)
    return n_active, share, prize


def expected_return(prog: ProgrammeConfig, strategy: Strategy, pop_spend: np.ndarray) -> float:
    """Exact long-run return ratio of the strategy (ratio of expectations over cycles)."""
    pmf = prog.payout_pmf()
    n_active, share, prize = _per_cycle_terms(prog, strategy, pop_spend)
    spend = (pmf * n_active).sum() * strategy.spend_per_step
    return float(prog.base_return_ratio + (pmf * share * prize).sum() / spend)


def simulate_paths(prog: ProgrammeConfig, strategy: Strategy, pop_spend: np.ndarray,
                   n_paths: int, n_cycles: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative return ratio, spend and reward, each paths x cycles."""
    pmf = prog.payout_pmf()
    n_active, share, prize = _per_cycle_terms(prog, strategy, pop_spend)
    tau = rng.choice(pmf.size, size=(n_paths, n_cycles), p=pmf)
    n = n_active[tau]
    theta = prog.base_return_ratio * strategy.spend_per_step / prog.base_return_shape
    base = np.where(n > 0, rng.gamma(np.maximum(n * prog.base_return_shape, 1e-12), theta), 0.0)
    spend = np.cumsum(n * strategy.spend_per_step, axis=1)
    reward = np.cumsum(base + share[tau] * prize[tau], axis=1)
    ratio = np.divide(reward, spend, out=np.zeros_like(reward), where=spend > 0)
    return ratio, spend, reward


def edge_curve(ratio: np.ndarray, levels=(0.01, 0.02, 0.03)) -> pd.DataFrame:
    """P(cumulative return ratio >= 1 + level) by cycle."""
    out = {"cycle": np.arange(1, ratio.shape[1] + 1)}
    for lv in levels:
        out[f"p_above_{lv:.2f}"] = (ratio >= 1 + lv).mean(axis=0)
    return pd.DataFrame(out)


def time_to_edge(curve: pd.DataFrame, level: float, confidence: float) -> int | None:
    """First cycle from which P(return >= 1 + level) stays >= confidence for every later cycle.

    The sustained-crossing check rejects a curve that touches the confidence
    level and falls back: only the last crossing counts.
    """
    p = curve[f"p_above_{level:.2f}"].to_numpy()
    below = np.flatnonzero(p < confidence)
    if below.size == 0:
        return 1
    if below[-1] == p.size - 1:
        return None
    return int(below[-1] + 2)


def sustained_share(ratio: np.ndarray, level: float, from_cycle: int) -> float:
    """Share of paths that stay at or above 1 + level for every cycle from `from_cycle` on."""
    return float((ratio[:, from_cycle - 1 :] >= 1 + level).all(axis=1).mean())
