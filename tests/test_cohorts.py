import numpy as np
import pandas as pd
import pytest

from fraud_analytics import cohorts as co
from fraud_analytics.study import quick_config, run_study


@pytest.fixture(scope="module")
def study():
    return run_study(quick_config())


def test_ci_covers_planted_ratio():
    rng = np.random.default_rng(3)
    spend = rng.uniform(50, 150, size=400)
    totals = pd.DataFrame({"spend": spend, "reward": spend * rng.normal(1.1, 0.05, size=400)})
    est = co.bca_return(totals, n_resamples=500, seed=1)
    assert est.low < 1.1 < est.high


def test_ci_covers_theoretical_return_for_timing_independent_accounts(study):
    # Regular accounts ignore the cycle: their planted excess is zero, so their
    # interval must contain the label-free theoretical return.
    regular = study.summary["truth"]["regular"]
    theo = study.summary["programme"]["theoretical_return"]["all"]
    assert regular["return"]["ci_low"] < theo < regular["return"]["ci_high"]


def test_circularity_inflation_reproduced(study):
    circ = study.summary["circularity"]
    in_sample = circ["listed_in_sample"]["return"]["value"]
    holdout = circ["listed_holdout"]["return"]
    assert in_sample > holdout["value"] + 0.1
    assert holdout["ci_high"] < in_sample


def test_behaviour_cohort_recovers_planted_exploiters(study):
    beh = study.summary["behaviour"]
    assert beh["precision"]["value"] >= 0.8
    assert beh["recall"]["value"] >= 0.3


def test_benchmark_z_detects_planted_excess(study):
    assert study.summary["truth"]["exploiting"]["benchmark"]["z"] > 2
    assert abs(study.summary["truth"]["regular"]["benchmark"]["z"]) < 2


def test_benjamini_hochberg_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    assert co.benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False, False, False]


def test_wilson_interval_bounds():
    est = co.wilson(0, 10)
    assert est.low == pytest.approx(0, abs=1e-12) and 0 < est.high < 0.35


def test_flagging_ignores_outcomes(study):
    # Shuffling rewards across entries must not change the behaviour-only flags.
    prog, cfg = study.prog, study.cfg
    disc_e, disc_s = co.split_cycles(prog.entries, prog.steps, 0)
    shuffled = disc_e.assign(pool_award=np.random.default_rng(0).permutation(disc_e["pool_award"].to_numpy()))
    a = co.behaviour_cohort(disc_e, disc_s, cfg.detection)
    b = co.behaviour_cohort(shuffled, disc_s, cfg.detection)
    assert a["flagged"].tolist() == b["flagged"].tolist()
