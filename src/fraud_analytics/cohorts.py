"""Cohort builders, the theoretical-return benchmark, z-test and BCa bootstrap.

Results are cohort-level only: per-account return estimates are too unstable
for low-volume accounts to support individual verdicts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import DetectionConfig


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def as_dict(self) -> dict:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high}


NAN = Estimate(float("nan"), float("nan"), float("nan"))


# --- Splits -------------------------------------------------------------------------------


def split_cycles(entries: pd.DataFrame, steps: pd.DataFrame, parity: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entries and steps of complete cycles with the given index parity (0 discovery, 1 holdout)."""
    e = entries[entries["cycle_complete"] & (entries["cycle"] % 2 == parity)]
    s = steps[steps["cycle_complete"] & (steps["cycle"] % 2 == parity)]
    return e, s


def first_cycles(entries: pd.DataFrame, steps: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entries and steps of the first n complete cycles."""
    return entries[entries["cycle"] < n], steps[steps["cycle"] < n]


# --- Behaviour features and cohorts -------------------------------------------------------------


def late_baseline(steps: pd.DataFrame, late_phase: float) -> float:
    """Share of time spent in the late part of the cycle: the label-free reference."""
    return float((steps["phase"] >= late_phase).mean())


def account_features(entries: pd.DataFrame, late_phase: float) -> pd.DataFrame:
    late = entries["phase"] >= late_phase
    e = entries.assign(
        late=late, late_cycle=entries["cycle"].where(late), reward=entries["base_reward"] + entries["pool_award"]
    )
    return (
        e.groupby("account_id")
        .agg(
            n_entries=("late", "size"),
            n_late=("late", "sum"),
            breadth=("late_cycle", "nunique"),
            spend=("spend", "sum"),
            reward=("reward", "sum"),
        )
        .reset_index()
    )


def benjamini_hochberg(p_values: np.ndarray, fdr: float) -> np.ndarray:
    m = p_values.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values)
    passed = p_values[order] <= fdr * np.arange(1, m + 1) / m
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


def timing_test(features: pd.DataFrame, p0: float) -> np.ndarray:
    """One-sided exact binomial p-value: more late entries than timing independence allows."""
    return stats.binom.sf(features["n_late"] - 1, features["n_entries"], p0)


def behaviour_cohort(entries: pd.DataFrame, steps: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    """Flag on measured behaviour only: timing concentration (FDR-controlled) and exposure breadth.

    Outcomes (wins, returns) are never used, so the cohort can be validated on them.
    """
    p0 = late_baseline(steps, cfg.late_phase)
    f = account_features(entries, cfg.late_phase)
    f = f[f["n_entries"] >= cfg.min_entries].reset_index(drop=True)
    f["late_share"] = f["n_late"] / f["n_entries"]
    f["p_value"] = timing_test(f, p0)
    f["flagged"] = benjamini_hochberg(f["p_value"].to_numpy(), cfg.fdr) & (f["breadth"] >= cfg.min_breadth)
    return f


def outcome_list(entries: pd.DataFrame, steps: pd.DataFrame, cfg: DetectionConfig) -> set[str]:
    """A "flagged list" built partly from outcomes: late-leaning accounts in the top return-ratio tail.

    This mimics a list assembled by reviewing whoever did well. Validating the
    effect against it is circular, because doing well was a condition for being on it.
    """
    p0 = late_baseline(steps, cfg.late_phase)
    f = account_features(entries, cfg.late_phase)
    f = f[f["n_entries"] >= cfg.min_entries]
    ratio = f["reward"] / f["spend"]
    keep = (ratio >= ratio.quantile(1 - cfg.outcome_top_share)) & (f["n_late"] / f["n_entries"] > p0)
    return set(f.loc[keep, "account_id"])


def operational_flags(entries: pd.DataFrame, cfg: DetectionConfig) -> set[str]:
    """Simple rule a team can apply after a few cycles: enough late entries, a late majority, breadth."""
    f = account_features(entries, cfg.late_phase)
    keep = (
        (f["n_late"] >= cfg.operational_min_late)
        & (f["n_late"] / f["n_entries"] >= cfg.operational_min_share)
        & (f["breadth"] >= cfg.operational_min_breadth)
    )
    return set(f.loc[keep, "account_id"])


# --- Returns and benchmark --------------------------------------------------------------------


def account_totals(entries: pd.DataFrame, ids: set[str] | None = None) -> pd.DataFrame:
    if ids is not None:
        entries = entries[entries["account_id"].isin(ids)]
    return (
        entries.assign(reward=entries["base_reward"] + entries["pool_award"])
        .groupby("account_id")
        .agg(spend=("spend", "sum"), reward=("reward", "sum"), n_entries=("spend", "size"))
        .reset_index()
    )


def theoretical_return(steps: pd.DataFrame, base_return_ratio: float) -> float:
    """Expected return per credit for participation spread evenly over time.

    An account spending the same small amount every step wins each payout with
    probability proportional to its share of that step's spend, so its pool
    return per credit is the time-average of award / step spend. Built from the
    programme trajectory only (no labels, no account outcomes).
    """
    s = steps[steps["spend"] > 0]
    return float(base_return_ratio + (s["award"] / s["spend"]).mean())


def _ratio(reward: np.ndarray, spend: np.ndarray, axis: int = -1) -> np.ndarray:
    return reward.sum(axis=axis) / spend.sum(axis=axis)


def bca_return(totals: pd.DataFrame, n_resamples: int, seed: int, confidence: float = 0.95) -> Estimate:
    """Cohort return ratio (sum reward / sum spend) with a BCa interval resampling accounts."""
    if totals.empty:
        return NAN
    r, s = totals["reward"].to_numpy(), totals["spend"].to_numpy()
    if len(totals) < 3:
        return Estimate(float(_ratio(r, s)), float("nan"), float("nan"))
    res = stats.bootstrap(
        (r, s), _ratio, paired=True, vectorized=True, n_resamples=n_resamples,
        confidence_level=confidence, method="BCa", rng=np.random.default_rng(seed),
    )
    return Estimate(float(_ratio(r, s)), float(res.confidence_interval.low), float(res.confidence_interval.high))


def benchmark_z(totals: pd.DataFrame, theoretical: float) -> dict:
    """Observed vs theoretical return, z-test on the pooled (delta-method) standard error.

    The standard error treats accounts as the independent units: entries of one
    account share a timing policy and stake level.
    """
    n = len(totals)
    if n < 2:
        return {"observed": float("nan"), "theoretical": theoretical, "z": float("nan"), "p_value": float("nan")}
    r, s = totals["reward"].to_numpy(), totals["spend"].to_numpy()
    ratio = r.sum() / s.sum()
    se = np.sqrt(((r - ratio * s) ** 2).sum() / (n * (n - 1))) / s.mean()
    z = (ratio - theoretical) / se
    return {"observed": float(ratio), "theoretical": theoretical, "se": float(se), "z": float(z),
            "p_value": float(stats.norm.sf(z))}


# --- Evaluation against planted truth ----------------------------------------------------------


def wilson(successes: int, trials: int, confidence: float = 0.95) -> Estimate:
    if trials == 0:
        return NAN
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return Estimate(p, float(centre - half), float(centre + half))


def detection_quality(flagged: set[str], accounts: pd.DataFrame, positive: str = "exploiting") -> dict:
    """Precision and recall against ground truth; recall counts unassessed positives as missed."""
    truth = accounts.set_index("account_id")["segment"]
    seg = truth.reindex(sorted(flagged))
    tp = int((seg == positive).sum())
    n_pos = int((truth == positive).sum())
    return {
        "flagged": len(flagged),
        "by_segment": {k: int(v) for k, v in seg.value_counts().items()},
        "precision": wilson(tp, len(flagged)).as_dict(),
        "recall": wilson(tp, n_pos).as_dict(),
    }
