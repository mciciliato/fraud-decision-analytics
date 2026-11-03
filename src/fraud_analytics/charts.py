"""Chart, table and text helpers for the Module C notebook.

Follows one set of specs throughout: validated palette slots in fixed order,
2px lines, 8px markers with a 2px surface ring, bars no thicker than 16px with
a 4px rounded data end and a square baseline, solid hairline grid, text in ink
tokens (never the series colour), a legend for two or more series plus
selective direct labels, and a table view next to every chart.

Every rich output also carries a plain-text fallback, so a viewer that cannot
render Markdown or HTML still shows readable text instead of an object repr.
"""

import html
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Validated categorical slots (fixed order) and ordinal blue ramp, light surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_RAMP = {"light": "#6da7ec", "mid": "#2a78d6", "dark": "#184f95"}
BLUE_PAIR = {"before": "#86b6ef", "after": "#1c5cab"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"
DPI = 100  # 1pt = 1.39px, so 1.5pt lines are ~2px


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9.5, "text.color": INK,
        "axes.edgecolor": GRID, "axes.linewidth": 0.75, "axes.labelcolor": INK_2, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.75, "grid.linestyle": "-",
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6, "legend.frameon": False, "legend.fontsize": 8.5,
    })


# --- Text and tables ------------------------------------------------------------------------


def _bundle(data: dict) -> None:
    from IPython.display import display

    display(data, raw=True)


def note(markdown: str) -> None:
    """A short Markdown paragraph with HTML and plain-text fallbacks."""
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", markdown)
    rich = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(markdown))
    _bundle({
        "text/markdown": markdown,
        "text/html": f'<p style="font-size:14px;line-height:1.55;max-width:46em;color:{INK}">{rich}</p>',
        "text/plain": plain,
    })


def table(df: pd.DataFrame, caption: str | None = None, index: bool = False) -> None:
    """A clean, right-aligned numeric table (values must already be formatted strings)."""
    styler = df.style if index else df.style.hide(axis="index")
    styler = styler.set_table_styles([
        {"selector": "", "props": f"border-collapse:collapse;font-size:13px;color:{INK};margin:4px 0 12px 0"},
        {"selector": "caption", "props": f"caption-side:top;text-align:left;color:{INK_2};font-size:12px;"
                                         "padding-bottom:6px"},
        {"selector": "th", "props": f"text-align:right;font-weight:600;color:{INK_2};padding:6px 14px;"
                                    f"border-bottom:1px solid {GRID}"},
        {"selector": "th.row_heading, th.blank", "props": "text-align:left"},
        {"selector": "td", "props": f"text-align:right;padding:6px 14px;border-bottom:1px solid {GRID};"
                                    "font-variant-numeric:tabular-nums"},
    ] + ([] if index else [{"selector": "td.col0", "props": "text-align:left"}]))
    if caption:
        styler = styler.set_caption(caption)
    plain = (caption + "\n" if caption else "") + df.to_string(index=index)
    _bundle({"text/html": styler.to_html(), "text/plain": plain})


def kpis(items: list[tuple[str, str, str]]) -> None:
    """A row of stat tiles: (label, value, detail)."""
    tiles = "".join(
        f'<div style="padding:10px 18px 10px 0;min-width:170px">'
        f'<div style="font-size:12px;color:{INK_2}">{html.escape(label)}</div>'
        f'<div style="font-size:26px;font-weight:600;color:{INK};line-height:1.25">{html.escape(value)}</div>'
        f'<div style="font-size:12px;color:{INK_2}">{html.escape(detail)}</div></div>'
        for label, value, detail in items
    )
    plain = "\n".join(f"{label}: {value} ({detail})" for label, value, detail in items)
    _bundle({"text/html": f'<div style="display:flex;flex-wrap:wrap;gap:8px">{tiles}</div>',
             "text/plain": plain})


def fmt_ci(est: dict, digits: int = 3) -> str:
    return f"{est['value']:.{digits}f}  ({est['ci_low']:.{digits}f} – {est['ci_high']:.{digits}f})"


def fmt_credits(x: float) -> str:
    return f"{x:,.0f}"


# --- Chart scaffolding ------------------------------------------------------------------------


def _figure(width: float, height: float):
    return plt.subplots(figsize=(width, height))


def _header(ax, title: str, subtitle: str, legend_rows: int = 0) -> None:
    """Title and subtitle stacked above the plot (and above the legend, when there is one)."""
    base = 6 + 15 * legend_rows
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, base), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, color=INK_2)
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, base + 15), textcoords="offset points",
                ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=INK)


def _legend(ax, handles, labels, ncol=None):
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol or len(labels),
              handlelength=1.6, columnspacing=1.8, borderaxespad=0.3, borderpad=0.1)


def _key(color, kind="line"):
    if kind == "dot":
        return plt.Line2D([], [], ls="", marker="o", ms=7, color=color, mec=SURFACE, mew=1.5)
    if kind == "bar":
        return plt.Rectangle((0, 0), 1, 1, color=color)
    return plt.Line2D([], [], color=color, lw=1.5)


def _hbar(ax, y: float, value: float, color: str, thickness_px: float = 14, radius_px: float = 4) -> None:
    """Horizontal bar from 0 with a rounded data end and a square baseline (call after limits are final)."""
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (1, 1)])
    dx, dy = abs(x1 - x0), abs(y1 - y0)  # data units per pixel
    h, rx, ry = thickness_px * dy, min(radius_px * dx, value / 2), radius_px * dy
    b, t, w = y - h / 2, y + h / 2, value
    verts = [(0, b), (w - rx, b), (w, b), (w, b + ry), (w, t - ry), (w, t), (w - rx, t), (0, t), (0, b)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def _end_label(ax, x, y, text, dx=6, dy=0, ha="left", color=INK, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=8.5, color=color, **kw)


def _finish(fig):
    fig.tight_layout()
    plt.show()


# --- Charts -------------------------------------------------------------------------------------


def circularity_dumbbell(rows: dict[str, tuple[dict, dict]], theoretical: float) -> None:
    """Discovery -> holdout return per cohort, with the theoretical return and break-even as references."""
    fig, ax = _figure(7.6, 3.1)
    _header(ax, "Selected on outcomes, the list's premium disappears on holdout",
            "Return ratio (rewards / spend), 95% BCa interval", legend_rows=1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    names = list(rows)
    for i, name in enumerate(names):
        disc, hold = rows[name]
        y = len(names) - 1 - i
        for est, col, dy in ((disc, BLUE_PAIR["before"], 0.17), (hold, BLUE_PAIR["after"], -0.17)):
            ax.plot([est["ci_low"], est["ci_high"]], [y + dy, y + dy], color=col, lw=1.5, zorder=2)
            ax.plot(est["value"], y + dy, "o", ms=7, color=col, mec=SURFACE, mew=1.5, zorder=3)
            _end_label(ax, est["ci_high"], y + dy, f"{est['value']:.2f}")
    for x, label in ((theoretical, f"theoretical {theoretical:.2f}"), (1.0, "break-even")):
        ax.axvline(x, color=INK_2, lw=0.8, zorder=1)
        ax.annotate(label, (x, len(names) - 0.45), xytext=(3, 0), textcoords="offset points", fontsize=8,
                    color=INK_2, va="bottom")
    ax.set_yticks(range(len(names)), names[::-1])
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
    lo = min(theoretical, *(e["ci_low"] for pair in rows.values() for e in pair))
    hi = max(e["ci_high"] for pair in rows.values() for e in pair)
    ax.set_xlim(lo - 0.02, hi + 0.05)
    _legend(ax, [_key(BLUE_PAIR["before"], "dot"), _key(BLUE_PAIR["after"], "dot")],
            ["measured on discovery (the cycles used to select)", "measured on holdout"])
    _finish(fig)


def edge_curves(curve: pd.DataFrame, tte: dict, max_cycles: int) -> None:
    """P(cumulative return >= 1 + level) over cycles, log scale, with the 90% crossing marked."""
    fig, ax = _figure(7.6, 3.8)
    marks = ", ".join(f"{tte[lab]['90%']}" for lab in ("+1%", "+2%", "+3%") if tte[lab]["90%"])
    _header(ax, "A clear edge, but it takes hundreds of cycles to be reliable",
            "Probability that the late strategy's cumulative return is at least the level; "
            f"dots mark 90% ({marks} cycles)", legend_rows=1)
    levels = [("0.01", "+1%", BLUE_RAMP["light"]), ("0.02", "+2%", BLUE_RAMP["mid"]),
              ("0.03", "+3%", BLUE_RAMP["dark"])]
    for conf in (0.5, 0.7, 0.9):
        ax.axhline(conf, color=INK_2, lw=0.8, zorder=1)
    for key, label, col in levels:
        ax.plot(curve["cycle"], curve[f"p_above_{key}"], color=col, zorder=2)
        cyc = tte[label]["90%"]
        if cyc:
            ax.plot(cyc, 0.9, "o", ms=7, color=col, mec=SURFACE, mew=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, max_cycles)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks([1, 3, 10, 30, 100, 300, max_cycles])
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks([0, 0.5, 0.7, 0.9])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cycles of the late strategy (log scale)")
    _legend(ax, [_key(c) for _, _, c in levels], [f"return at least {lab}" for _, lab, _ in levels])
    _finish(fig)


def delayed_loss(d: pd.DataFrame, recommended: int) -> None:
    fig, ax = _figure(7.6, 3.0)
    _header(ax, "Every cycle of delay adds to the loss",
            "Loss the flagged cohort accrues before detection, credits (95% band)")
    x = d["detect_after_cycles"]
    ax.fill_between(x, d["loss_low"], d["loss_high"], color=BLUE, alpha=0.10, lw=0)
    ax.plot(x, d["loss"], color=BLUE)
    for n in (recommended, int(x.iloc[-1])):
        row = d.loc[d["detect_after_cycles"] == n].iloc[0]
        ax.plot(n, row["loss"], "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
        _end_label(ax, n, row["loss"], f"{row['loss']:,.0f} after {n} cycles", dx=-8, dy=10, ha="right")
    ax.set_xlim(0, x.iloc[-1] + 0.5)
    ax.set_ylim(0, d["loss_high"].max() * 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    ax.set_xlabel("Detect after N cycles")
    _finish(fig)


def tradeoff(tr: pd.DataFrame) -> None:
    fig, ax = _figure(7.6, 4.0)
    _header(ax, "Expected cost is lowest around the recommended threshold",
            "Expected cost of detecting after N cycles and its three components, credits", legend_rows=2)
    series = [("expected_cost", "expected cost (total)", BLUE),
              ("loss_never_detected", "loss from exploiters missed", ORANGE),
              ("loss_before_detection", "loss before detection", AQUA),
              ("false_positive_cost", "false-positive cost", YELLOW)]
    x = tr["detect_after_cycles"]
    for key, label, col in series:
        ax.plot(x, tr[key], color=col, zorder=3 if key == "expected_cost" else 2)
        _end_label(ax, x.iloc[-1], tr[key].iloc[-1], label.split(" (")[0], color=INK_2)
    best = tr.loc[tr["expected_cost"].idxmin()]
    ax.plot(best["detect_after_cycles"], best["expected_cost"], "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5,
            zorder=4)
    ax.annotate(f"minimum at N = {int(best['detect_after_cycles'])}: {best['expected_cost']:,.0f}",
                (best["detect_after_cycles"], best["expected_cost"]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK)
    ax.set_xlim(0.5, x.iloc[-1])
    ax.set_ylim(0, tr[[k for k, _, _ in series]].to_numpy().max() * 1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    ax.set_xlabel("Detect after N cycles")
    _legend(ax, [_key(c) for _, _, c in series], [lab for _, lab, _ in series], ncol=2)
    fig.subplots_adjust(right=0.8)
    plt.show()


def pool_flows(flows: pd.DataFrame) -> None:
    """Credits each group puts into and takes out of the pool, per account per cycle."""
    fig, ax = _figure(7.6, 2.8)
    _header(ax, "The flagged cohort takes far more from the pool than it puts in",
            "Pool credits per account per cycle", legend_rows=1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    groups = list(flows["group"])
    measures = [("contributed_per_account_cycle", "contributed to the pool", BLUE),
                ("received_per_account_cycle", "received from the pool", ORANGE)]
    vmax = flows[[m for m, _, _ in measures]].to_numpy().max()
    ax.set_xlim(0, vmax * 1.15)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    labels = [f"{g}\nreceives {m:.1f}x what it puts in"
              for g, m in zip(groups, flows["received_per_credit_contributed"], strict=True)]
    ax.set_yticks(range(len(groups)), labels[::-1])
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    step = 0.2  # centre offset; 14px bars leave a surface gap between the pair
    for i, row in flows.iterrows():
        y = len(groups) - 1 - i
        for j, (key, _, col) in enumerate(measures):
            yy = y + (step if j == 0 else -step)
            _hbar(ax, yy, row[key], col)
            _end_label(ax, row[key], yy, f"{row[key]:.2f}")
    _legend(ax, [_key(c, "bar") for _, _, c in measures], [lab for _, lab, _ in measures])
    _finish(fig)
