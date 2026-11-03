import numpy as np
import pandas as pd
import pytest

from fraud_analytics import simulation as sim
from fraud_analytics.config import ProgrammeConfig
from fraud_analytics.study import quick_config
from fraud_analytics.synthetic import simulate

PROG = ProgrammeConfig()
POP_SPEND = sim.expected_population_spend(PROG, 1000.0, 500.0, 0.7)


def test_synthetic_programme_reproducible_by_seed():
    a, b = simulate(quick_config(seed=11)), simulate(quick_config(seed=11))
    pd.testing.assert_frame_equal(a.entries, b.entries)
    pd.testing.assert_frame_equal(a.steps, b.steps)


def test_monte_carlo_reproducible_by_seed():
    strat = sim.Strategy(0.7, 20.0)
    a = sim.simulate_paths(PROG, strat, POP_SPEND, 200, 50, np.random.default_rng(5))[0]
    b = sim.simulate_paths(PROG, strat, POP_SPEND, 200, 50, np.random.default_rng(5))[0]
    np.testing.assert_array_equal(a, b)


def test_zero_edge_strategy_converges_to_programme_margin():
    # A timing-independent strategy earns the programme's design return: 1 - margin, below 1.
    strat = sim.Strategy(0.0, 20.0)
    expected = sim.expected_return(PROG, strat, POP_SPEND)
    ratio, spend, reward = sim.simulate_paths(PROG, strat, POP_SPEND, 2000, 200, np.random.default_rng(1))
    assert expected < 1
    assert reward[:, -1].sum() / spend[:, -1].sum() == pytest.approx(expected, abs=0.005)


def test_late_strategy_has_positive_edge():
    late = sim.expected_return(PROG, sim.Strategy(0.7, 20.0), POP_SPEND)
    flat = sim.expected_return(PROG, sim.Strategy(0.0, 20.0), POP_SPEND)
    assert late > 1 > flat


def test_payout_pmf_sums_to_one_and_ends_by_max_cycle():
    pmf = PROG.payout_pmf()
    assert pmf.sum() == pytest.approx(1.0)
    assert PROG.hazard()[-1] == pytest.approx(1.0)


def test_pool_is_shared_in_full_and_only_at_payout_steps():
    prog = simulate(quick_config())
    paid = prog.entries.groupby("step")["pool_award"].sum()
    awards = prog.steps.set_index("step")["award"]
    np.testing.assert_allclose(paid.reindex(awards.index, fill_value=0).to_numpy(), awards.to_numpy(), rtol=1e-9)
    payout_steps = prog.steps.loc[prog.steps["payout"], "step"]
    assert prog.entries.loc[prog.entries["pool_award"] > 0, "step"].isin(payout_steps).all()


def test_time_to_edge_requires_sustained_crossing():
    curve = pd.DataFrame({"cycle": [1, 2, 3, 4, 5], "p_above_0.01": [0.2, 0.95, 0.6, 0.92, 0.97]})
    assert sim.time_to_edge(curve, 0.01, 0.9) == 4
    never = curve.assign(**{"p_above_0.01": [0.95, 0.95, 0.95, 0.95, 0.5]})
    assert sim.time_to_edge(never, 0.01, 0.9) is None
