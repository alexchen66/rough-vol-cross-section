"""
Plot full-period rough-vs-no-rough comparisons.

Each chart contains three curves:
  1. full model with rough-volatility features
  2. no-rough ablation model
  3. market proxy

By default this script plots long-short portfolios for 4 models x 4 cost levels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from visualize_daily_vs_benchmark import load_market_proxy


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
OUT_DIR = ROOT / "data" / "figures" / "rough_vs_no_rough_full_period"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["elastic_net", "lgbm", "ranker", "ridge"]
COST_BPS = [0, 5, 10, 20]
PORTFOLIO = "long_short"


def _load_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_main = pd.read_csv(
        REPORTS / "portfolio_daily_returns_all_windows.csv",
        parse_dates=["date", "hold_date"],
    )
    full_elastic = pd.read_csv(
        REPORTS / "portfolio_daily_returns_elastic_net_all_windows.csv",
        parse_dates=["date", "hold_date"],
    )
    full = pd.concat([full_main, full_elastic], ignore_index=True)

    no_rough = pd.read_csv(
        REPORTS / "portfolio_daily_returns_no_rough_all_windows.csv",
        parse_dates=["date", "hold_date"],
    )
    return full, no_rough


def _daily_curve(data: pd.DataFrame, model: str, cost_bps: int, portfolio: str) -> pd.Series:
    g = data[
        (data["model"] == model)
        & (data["cost_bps"] == cost_bps)
        & (data["portfolio"] == portfolio)
    ]
    if g.empty:
        raise ValueError(f"No returns for model={model}, cost_bps={cost_bps}, portfolio={portfolio}")
    daily = g.groupby("hold_date")["net_return"].sum().sort_index()
    return (1.0 + daily).cumprod()


def _market_curve(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    market = load_market_proxy(start, end)
    return (1.0 + market).cumprod()


def plot_one(full: pd.DataFrame, no_rough: pd.DataFrame, model: str, cost_bps: int) -> Path:
    full_curve = _daily_curve(full, model, cost_bps, PORTFOLIO)
    no_rough_curve = _daily_curve(no_rough, model, cost_bps, PORTFOLIO)

    start = min(full_curve.index.min(), no_rough_curve.index.min())
    end = max(full_curve.index.max(), no_rough_curve.index.max())
    market_curve = _market_curve(start, end)

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.plot(full_curve.index, full_curve.values, linewidth=1.35, color="#2563eb", label="Full with rough")
    ax.plot(no_rough_curve.index, no_rough_curve.values, linewidth=1.35, color="#dc2626", label="No rough")
    ax.plot(market_curve.index, market_curve.values, linewidth=1.55, color="#111827", linestyle="--", label="Market proxy")

    ax.axhline(1.0, color="#64748b", linewidth=0.9)
    ax.set_title(f"{model}: 2016-2023 {PORTFOLIO} rough vs no-rough, {cost_bps} bps")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = OUT_DIR / f"daily_cumulative_returns_2016_2023_{model}_{PORTFOLIO}_{cost_bps}bps_rough_vs_no_rough.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    full, no_rough = _load_returns()
    outputs = []
    for model in MODELS:
        for cost_bps in COST_BPS:
            outputs.append(plot_one(full, no_rough, model, cost_bps))

    print(f"Saved {len(outputs)} rough-vs-no-rough full-period charts under: {OUT_DIR}")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
