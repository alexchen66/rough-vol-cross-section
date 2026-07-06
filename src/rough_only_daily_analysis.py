"""
Run all-windows daily backtest and plots for the rough-only ablation experiment.

Charts are written to two new folders under data/figures:
  data/figures/rough_only_by_year
  data/figures/rough_only_full_period
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from all_windows_daily_analysis import _max_drawdown, _plot_curves, _sharpe, _wealth
from config import COST_BPS_GRID, DATA_PROCESSED
from daily_backtest import build_daily_return_panel
from visualize_daily_vs_benchmark import load_market_proxy


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
YEARLY_FIGURES = ROOT / "data" / "figures" / "rough_only_by_year"
FULL_FIGURES = ROOT / "data" / "figures" / "rough_only_full_period"

PREDICTIONS_PATH = DATA_PROCESSED / "predictions_rough_only.parquet"
RETURNS_PATH = REPORTS / "portfolio_daily_returns_rough_only_all_windows.csv"


def load_or_run_rough_only(force: bool = True) -> pd.DataFrame:
    if force or not RETURNS_PATH.exists():
        if not PREDICTIONS_PATH.exists():
            raise FileNotFoundError(
                f"Missing {PREDICTIONS_PATH}. Run python src/train_rough_only.py first."
            )
        predictions = pd.read_parquet(PREDICTIONS_PATH)
        crsp = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet", columns=["date", "permno", "ret"])
        panel = build_daily_return_panel(
            predictions,
            crsp,
            cost_bps_grid=COST_BPS_GRID,
            start_date="2016-01-01",
            end_date="2023-12-31",
            hold_to_next_rebalance=True,
            weighting="equal",
        )
        RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(RETURNS_PATH, index=False)
    else:
        panel = pd.read_csv(RETURNS_PATH, parse_dates=["date", "hold_date"])

    panel["date"] = pd.to_datetime(panel["date"])
    panel["hold_date"] = pd.to_datetime(panel["hold_date"])
    panel["year"] = panel["hold_date"].dt.year
    panel["cost_bps"] = panel["cost_bps"].astype(int)
    return panel.sort_values(["model", "portfolio", "cost_bps", "hold_date"]).reset_index(drop=True)


def plot_yearly_rough_only(data: pd.DataFrame, market_daily: pd.Series) -> list[Path]:
    outputs = []
    for year in sorted(data["year"].unique()):
        year_data = data[data["year"] == year].copy()
        for model in sorted(year_data["model"].unique()):
            for portfolio in ["long_only", "long_short"]:
                out = YEARLY_FIGURES / f"daily_cumulative_returns_{year}_{model}_{portfolio}_fees_vs_market.png"
                title = f"{model} rough-only: {year} daily cumulative {portfolio} returns by fee vs market"
                outputs.append(_plot_curves(year_data, market_daily, model, portfolio, out, title))
    return outputs


def plot_full_period_rough_only(data: pd.DataFrame, market_daily: pd.Series) -> list[Path]:
    outputs = []
    for model in sorted(data["model"].unique()):
        for portfolio in ["long_only", "long_short"]:
            out = FULL_FIGURES / f"daily_cumulative_returns_2016_2023_{model}_{portfolio}_fees_vs_market.png"
            title = f"{model} rough-only: 2016-2023 daily cumulative {portfolio} returns by fee vs market"
            outputs.append(_plot_curves(data, market_daily, model, portfolio, out, title))
    return outputs


def make_rough_only_backtest_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(["year", "model", "portfolio", "cost_bps"], sort=True)
    for (year, model, portfolio, cost_bps), g in grouped:
        daily = g.groupby("hold_date")["net_return"].sum().sort_index()
        entry_turnover = g.groupby("date")["turnover"].first()
        rows.append(
            {
                "year": year,
                "model": model,
                "portfolio": portfolio,
                "cost_bps": cost_bps,
                "annual_return": float(_wealth(daily).iloc[-1] - 1.0) if not daily.empty else np.nan,
                "annualized_vol": float(np.sqrt(252) * daily.std(ddof=1)) if len(daily) > 1 else np.nan,
                "sharpe": _sharpe(daily),
                "max_drawdown": _max_drawdown(daily),
                "mean_daily_return": float(daily.mean()) if not daily.empty else np.nan,
                "trading_days": int(daily.shape[0]),
                "avg_rebalance_turnover": float(entry_turnover.mean()) if not entry_turnover.empty else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS / "daily_backtest_summary_rough_only_all_windows_by_year.csv", index=False)
    return summary


def make_rough_only_ic_summary() -> pd.DataFrame:
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions[(predictions["date"] >= "2016-01-01") & (predictions["date"] <= "2023-12-31")]

    score_cols = [col for col in predictions.columns if col.startswith("score_")]
    rows = []
    for date, g in predictions.groupby("date"):
        for score_col in score_cols:
            model = score_col.replace("score_", "")
            rows.append(
                {
                    "date": date,
                    "year": date.year,
                    "model": model,
                    "pearson_ic": g[score_col].corr(g["y_xs"], method="pearson"),
                    "spearman_rank_ic": g[score_col].corr(g["y_xs"], method="spearman"),
                    "n": int(g[[score_col, "y_xs"]].dropna().shape[0]),
                }
            )

    daily_ic = pd.DataFrame(rows)
    daily_ic.to_csv(REPORTS / "ic_by_rebalance_date_rough_only_all_windows.csv", index=False)

    summary_rows = []
    for (year, model), g in daily_ic.groupby(["year", "model"], sort=True):
        for col in ["pearson_ic", "spearman_rank_ic"]:
            vals = g[col].dropna()
            std = vals.std(ddof=1)
            summary_rows.append(
                {
                    "year": year,
                    "model": model,
                    "ic_type": col,
                    "mean_ic": float(vals.mean()) if not vals.empty else np.nan,
                    "median_ic": float(vals.median()) if not vals.empty else np.nan,
                    "std_ic": float(std) if not vals.empty else np.nan,
                    "ic_ir": float(vals.mean() / std) if len(vals) > 1 and std != 0 else np.nan,
                    "positive_ic_rate": float((vals > 0).mean()) if not vals.empty else np.nan,
                    "n_rebalance_dates": int(vals.shape[0]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS / "ic_summary_rough_only_all_windows_by_year.csv", index=False)
    return summary


def main() -> None:
    data = load_or_run_rough_only(force=True)
    start, end = data["hold_date"].min(), data["hold_date"].max()
    market_daily = load_market_proxy(start, end)

    yearly_outputs = plot_yearly_rough_only(data, market_daily)
    full_outputs = plot_full_period_rough_only(data, market_daily)
    summary = make_rough_only_backtest_summary(data)
    ic_summary = make_rough_only_ic_summary()

    print(f"Saved rough-only daily returns: {RETURNS_PATH}")
    print(f"Saved {len(yearly_outputs)} yearly charts under: {YEARLY_FIGURES}")
    print(f"Saved {len(full_outputs)} full-period charts under: {FULL_FIGURES}")
    print(f"Saved rough-only backtest summary rows: {len(summary)}")
    print(f"Saved rough-only IC summary rows: {len(ic_summary)}")


if __name__ == "__main__":
    main()
