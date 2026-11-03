import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("IPython")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from fraud_analytics import charts as ch  # noqa: E402


@pytest.fixture
def captured(monkeypatch):
    out = []
    monkeypatch.setattr(ch, "_bundle", out.append)
    return out


def test_rich_outputs_carry_readable_plain_text(captured):
    ch.note("Detect after **15 cycles**.")
    ch.table(pd.DataFrame({"Cohort": ["A"], "Return": ["1.00"]}), caption="Returns")
    ch.kpis([("Flagged", "71", "of 3,768")])
    for bundle in captured:
        assert "text/plain" in bundle and "object at" not in bundle["text/plain"]
        assert "<IPython" not in bundle["text/plain"]
    assert captured[0]["text/plain"] == "Detect after 15 cycles."


def test_rounded_bar_has_square_baseline_and_full_length():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    ch._hbar(ax, 0, 6.0, ch.BLUE)
    xs = ax.patches[0].get_path().vertices[:, 0]
    assert xs.min() == 0 and xs.max() == pytest.approx(6.0)
    plt.close(fig)
