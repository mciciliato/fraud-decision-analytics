"""Parameters of the synthetic reward programme and of the analysis.

Every value is invented for this study and justified in DECISIONS.md. None is
calibrated to, or copied from, any real programme.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProgrammeConfig:
    """A recurring reward pool whose chance of paying out rises over each cycle.

    Time moves in steps. A cycle starts after every payout; at the end of the
    a-th step of a cycle the pool pays out with probability (a / max_cycle_steps)
    ** hazard_power, so it always pays by the last step. The pool is shared pro rata
    among the spend of the payout step (a cashback pool).
    """

    step_hours: int = 4
    max_cycle_steps: int = 42  # 7 days at 4-hour steps
    hazard_power: float = 4.0
    pool_seed: float = 900.0  # credits added by the programme at the start of every cycle
    contribution_rate: float = 0.02  # share of each qualifying spend added to the pool
    base_return_ratio: float = 0.85  # expected base reward (cashback) per credit spent
    base_return_shape: float = 2.0  # gamma shape of the per-entry base reward

    def hazard(self) -> np.ndarray:
        """Payout probability at the end of cycle step a = 1 .. max_cycle_steps."""
        a = np.arange(1, self.max_cycle_steps + 1)
        return (a / self.max_cycle_steps) ** self.hazard_power

    def payout_pmf(self) -> np.ndarray:
        """P(the cycle pays out at step a), a = 1 .. max_cycle_steps."""
        h = self.hazard()
        survive = np.concatenate(([1.0], np.cumprod(1 - h)[:-1]))
        return survive * h


@dataclass(frozen=True)
class PopulationConfig:
    """Synthetic accounts with planted ground truth."""

    n_accounts: int = 4000
    share_exploiting: float = 0.03  # concentrate activity late in the cycle (planted positive)
    share_heavy: float = 0.05  # legitimate heavy users, mildly more active late (false-positive risk)
    n_steps: int = 6 * 365  # one year
    mean_entry_rate: float = 0.06  # entries per account per step, regular accounts
    rate_dispersion: float = 1.5  # gamma shape of per-account activity
    spend_log_mean: float = 2.0  # log-normal spend per entry
    spend_log_sd: float = 0.6
    exploit_background_rate: float = 0.004
    exploit_active_rate: float = 0.35
    exploit_spend_multiplier: float = 2.5
    exploit_phase_low: float = 0.55  # personal start of the active window, as a share of the max cycle
    exploit_phase_high: float = 0.80
    heavy_rate_multiplier: float = 4.0
    heavy_phase_sensitivity: float = 1.0

    @property
    def mean_spend(self) -> float:
        return float(np.exp(self.spend_log_mean + self.spend_log_sd**2 / 2))


@dataclass(frozen=True)
class DetectionConfig:
    late_phase: float = 0.6  # an entry is "late" when its cycle step is at least this share of the max cycle
    min_entries: int = 8
    min_breadth: int = 3  # distinct cycles with late entries (exposure breadth)
    fdr: float = 0.01
    # Operational rule for the detection-threshold trade-off, evaluated after N cycles.
    operational_min_late: int = 3
    operational_min_share: float = 0.5
    operational_min_breadth: int = 2
    outcome_top_share: float = 0.05  # outcome-built list: top share of accounts by return ratio


@dataclass(frozen=True)
class CostConfig:
    false_positive_cost: float = 60.0  # review effort + friction for a legitimate account, credits
    horizon_cycles: int = 40  # cycles an undetected account keeps extracting
    max_detection_cycles: int = 30
    window_stride: int = 3  # cycles between the starts of the rolling evaluation windows


@dataclass(frozen=True)
class StudyConfig:
    seed: int = 7
    programme: ProgrammeConfig = field(default_factory=ProgrammeConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    bootstrap_resamples: int = 2000
    mc_paths: int = 4000
    mc_cycles: int = 800
