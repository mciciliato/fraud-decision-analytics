"""Cost to the business: extraction, cost of delayed detection, threshold choice, pool health."""

import numpy as np
import pandas as pd
from scipy import stats

from .cohorts import Estimate, operational_flags
from .config import StudyConfig
from .synthetic import Programme


def extraction_per_account_cycle(totals: pd.DataFrame, theoretical: float, n_cycles: int,
                                 n_resamples: int, seed: int) -> Estimate:
    """Mean excess over the theoretical return, per account and cycle, with a BCa interval.

    Excess = reward - theoretical return x spend: what the account received above
    what timing-independent participation would have paid for the same spend.
    """
    excess = (totals["reward"] - theoretical * totals["spend"]).to_numpy() / n_cycles
    if excess.size < 3:
        return Estimate(float(excess.mean()) if excess.size else float("nan"), float("nan"), float("nan"))
    res = stats.bootstrap((excess,), np.mean, n_resamples=n_resamples, method="BCa",
                          rng=np.random.default_rng(seed))
    return Estimate(float(excess.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high))


def scale(est: Estimate, factor: float) -> Estimate:
    return Estimate(est.value * factor, est.low * factor, est.high * factor)


def delayed_detection_curve(per_account_cycle: Estimate, n_accounts: int, max_cycles: int) -> pd.DataFrame:
    """Loss accrued by a cohort before detection, if detection happens after N cycles."""
    n = np.arange(0, max_cycles + 1)
    return pd.DataFrame(
        {
            "detect_after_cycles": n,
            "loss": per_account_cycle.value * n_accounts * n,
            "loss_low": per_account_cycle.low * n_accounts * n,
            "loss_high": per_account_cycle.high * n_accounts * n,
        }
    )


def threshold_tradeoff(prog: Programme, cfg: StudyConfig, per_account_cycle: float) -> pd.DataFrame:
    """Expected cost of detecting after N cycles: loss before detection + missed accounts + false positives.

    The rule is applied to windows of N consecutive cycles starting at many points
    in the year and the counts are averaged, so the answer does not hinge on how
    long the first few cycles happened to be. True and false positives come from
    the planted ground truth of the synthetic programme; in production they would
    come from reviewed cases.
    """
    truth = prog.accounts.set_index("account_id")["segment"]
    n_pos = int((truth == "exploiting").sum())
    entries = prog.entries[prog.entries["cycle_complete"]]
    n_cycles = int(entries["cycle"].max()) + 1
    rows = []
    for n in range(1, cfg.cost.max_detection_cycles + 1):
        counts = []
        for start in range(0, n_cycles - n + 1, cfg.cost.window_stride):
            window = entries[entries["cycle"].between(start, start + n - 1)]
            seg = truth.reindex(sorted(operational_flags(window, cfg.detection)))
            counts.append(((seg == "exploiting").sum(), (seg != "exploiting").sum(), (seg == "heavy").sum()))
        tp, fp, fp_heavy = np.mean(counts, axis=0)
        loss_before = per_account_cycle * n * tp
        loss_missed = per_account_cycle * cfg.cost.horizon_cycles * (n_pos - tp)
        fp_cost = cfg.cost.false_positive_cost * fp
        rows.append((n, tp, fp, fp_heavy, loss_before, loss_missed, fp_cost, loss_before + loss_missed + fp_cost))
    return pd.DataFrame(
        rows,
        columns=["detect_after_cycles", "true_positives", "false_positives", "false_positives_heavy",
                 "loss_before_detection", "loss_never_detected", "false_positive_cost", "expected_cost"],
    )


def pool_flows(entries: pd.DataFrame, groups: dict[str, set[str]], contribution_rate: float,
               n_cycles: int) -> pd.DataFrame:
    """What each group puts into the pool and takes out of it, in total and per account per cycle."""
    contributions = contribution_rate * entries["spend"]
    total_in, total_out = contributions.sum(), entries["pool_award"].sum()
    rows = []
    for name, ids in groups.items():
        mask = entries["account_id"].isin(ids)
        n_acc = int(entries.loc[mask, "account_id"].nunique())
        put_in, took_out = float(contributions[mask].sum()), float(entries.loc[mask, "pool_award"].sum())
        rows.append({
            "group": name,
            "accounts": n_acc,
            "contributed_per_account_cycle": put_in / n_acc / n_cycles,
            "received_per_account_cycle": took_out / n_acc / n_cycles,
            "received_per_credit_contributed": took_out / put_in,
            "share_of_contributions": put_in / total_in,
            "share_of_awards": took_out / total_out,
        })
    return pd.DataFrame(rows)


def pool_health(entries: pd.DataFrame, cohort_ids: set[str], contribution_rate: float) -> dict:
    """Share of pool awards a cohort receives vs share of pool contributions it funds."""
    in_cohort = entries["account_id"].isin(cohort_ids)
    awards = entries["pool_award"]
    contributions = contribution_rate * entries["spend"]
    return {
        "share_of_awards": float(awards[in_cohort].sum() / awards.sum()),
        "share_of_contributions": float(contributions[in_cohort].sum() / contributions.sum()),
    }
