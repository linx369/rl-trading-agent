"""Static PNG figures for a run (same six panels as the thesis)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regime_rl import ACTIONS  # noqa: E402
from regime_rl.evaluation.metrics import rolling_ir  # noqa: E402

# Validated categorical palette (fixed order, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCH = "#8a8986"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
REGIME_COLORS = {
    "p_bull": SERIES[0],
    "p_volatile": SERIES[1],
    "p_sideways": SERIES[2],
    "p_bear": SERIES[3],
}


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _fig(h: float = 4.0):
    fig, ax = plt.subplots(figsize=(11, h), facecolor=SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_all(
    record: pd.DataFrame, regimes: pd.DataFrame, mc: dict, out: Path, label: str
) -> list[Path]:
    """``record``: monthly test record with ret, bench_ret, action columns."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    wealth = (1 + record["ret"]).cumprod() - 1
    bwealth = (1 + record["bench_ret"]).cumprod() - 1

    fig, ax = _fig()
    ax.plot(wealth.index, 100 * wealth, color=SERIES[0], lw=2, label=label)
    ax.plot(bwealth.index, 100 * bwealth, color=BENCH, lw=2, label="Benchmark buy & hold")
    ax.fill_between(wealth.index, 100 * wealth, 100 * bwealth, color=SERIES[0], alpha=0.08)
    ax.set_ylabel("Cumulative return (%)", color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT)
    _style(ax, "Out-of-sample cumulative return")
    paths.append(_save(fig, out / "panel_A_cumulative_return.png"))

    fig, ax = _fig(3.2)
    for series, color, name in (
        (record["ret"], SERIES[0], label),
        (record["bench_ret"], BENCH, "Benchmark"),
    ):
        w = (1 + series).cumprod()
        ax.plot(w.index, 100 * (w / w.cummax() - 1), color=color, lw=2, label=name)
    ax.axhline(-8, color=MUTED, lw=1, ls="--", label="−8% reference")
    ax.set_ylabel("Drawdown (%)", color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT)
    _style(ax, "Drawdown from previous peak")
    paths.append(_save(fig, out / "panel_B_drawdown.png"))

    fig, ax = _fig(3.2)
    reg = regimes.loc[record.index.min() - pd.offsets.MonthEnd(1) : record.index.max()]
    cols = [c for c in REGIME_COLORS if c in reg]
    ax.stackplot(
        reg.index,
        *[100 * reg[c] for c in cols],
        colors=[REGIME_COLORS[c] for c in cols],
        labels=[c[2:].capitalize() for c in cols],
        alpha=0.85,
        edgecolor=SURFACE,
        lw=0.5,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)", color=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12), labelcolor=TEXT)
    _style(ax, "HMM regime probabilities (month end)")
    paths.append(_save(fig, out / "panel_C_regime_probs.png"))

    fig, ax = _fig(3.2)
    for a, name in ACTIONS.items():
        m = record["action"] == a
        if m.any():
            ax.bar(
                record.index[m],
                np.ones(m.sum()),
                width=24,
                color=SERIES[a],
                label=f"{a}: {name}",
                edgecolor=SURFACE,
                linewidth=1,
            )
    ax.set_yticks([])
    ax.legend(
        frameon=False,
        ncol=4,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        labelcolor=TEXT,
    )
    _style(ax, "Strategy chosen each month")
    paths.append(_save(fig, out / "panel_D_strategy_selection.png"))

    fig, ax = _fig(3.2)
    rir = rolling_ir(record["ret"] - record["bench_ret"]).dropna()
    ax.plot(rir.index, rir, color=SERIES[0], lw=2, label="Rolling 12-month IR")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":", label="IR = 0.5")
    ax.legend(frameon=False, labelcolor=TEXT)
    _style(ax, "Rolling 12-month information ratio vs benchmark")
    paths.append(_save(fig, out / "panel_E_rolling_ir.png"))

    fig, ax = _fig(3.6)
    final = 100 * (mc["paths"][:, -1] - 1)
    ax.hist(final, bins=50, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    for key, name in (("p5", "5th pct"), ("median", "Median"), ("p95", "95th pct")):
        ax.axvline(100 * mc[key], color=TEXT, lw=1, ls="--")
        ax.annotate(
            f"{name} {100 * mc[key]:+.1f}%",
            (100 * mc[key], ax.get_ylim()[1] * 0.97),
            fontsize=8,
            color=TEXT,
            rotation=90,
            ha="right",
            va="top",
            xytext=(-3, 0),
            textcoords="offset points",
        )
    ax.axvline(
        100 * mc["bench_total"],
        color=BENCH,
        lw=2,
        label=f"Benchmark {100 * mc['bench_total']:+.1f}%",
    )
    ax.set_xlabel("Two-year return (%)", color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT)
    _style(ax, f"Monte Carlo: {mc['paths'].shape[0]:,} bootstrapped two-year paths")
    paths.append(_save(fig, out / "panel_F_monte_carlo.png"))
    return paths
