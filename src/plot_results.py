"""
Generate three result charts:
1. Cumulative L/S portfolio return (ensemble vs LGBM vs Ridge)
2. Annual IC bar chart
3. Decile monotonicity chart (average return by score decile)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import spearmanr
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "data" / "reports"
REPORTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_predictions():
    return pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")


def monthly_ls_spread(pred, score_col, pct=0.10):
    spreads = {}
    for date, g in pred.groupby("date"):
        top_n = max(1, int(len(g) * pct))
        g2 = g.sort_values(score_col, ascending=False)
        spreads[date] = g2.head(top_n)["y_xs"].mean() - g2.tail(top_n)["y_xs"].mean()
    return pd.Series(spreads).sort_index()


# ── Chart 1: Cumulative L/S Return ──────────────────────────────────────────

def plot_cumulative_returns(pred):
    fig, ax = plt.subplots(figsize=(11, 5))

    styles = {
        "score_ensemble": ("Ensemble (70% LGBM + 30% Ridge)", "#1f77b4", 2.5),
        "score_lgbm":     ("LGBM",                            "#ff7f0e", 1.5),
        "score_ridge":    ("Ridge",                           "#2ca02c", 1.5),
    }

    def to_dt(s):
        return [datetime.strptime(d[:10], "%Y-%m-%d") for d in s]

    for col, (label, color, lw) in styles.items():
        s = monthly_ls_spread(pred, col)
        cum = (1 + s).cumprod()
        ax.plot(to_dt(cum.index), cum.values,
                label=label, color=color, linewidth=lw)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (1 = start)")
    ax.set_title("Long-Short Portfolio Cumulative Return\n"
                 "(Top 10% Long, Bottom 10% Short, Equal Weight, Monthly Rebalance)",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Shade drawdown periods
    for start, end in [("2020-01-01", "2020-04-01"), ("2022-01-01", "2022-12-01")]:
        ax.axvspan(datetime.strptime(start, "%Y-%m-%d"),
                   datetime.strptime(end,   "%Y-%m-%d"),
                   alpha=0.08, color="red", label="_")

    fig.tight_layout()
    out = REPORTS / "chart1_cumulative_returns.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Chart 2: Annual IC Bar Chart ─────────────────────────────────────────────

def plot_annual_ic(pred):
    pred = pred.copy()
    pred["year"] = pred["date"].str[:4].astype(int)

    models = {
        "score_ensemble": "Ensemble",
        "score_lgbm":     "LGBM",
        "score_ridge":    "Ridge",
    }
    colors = {"Ensemble": "#1f77b4", "LGBM": "#ff7f0e", "Ridge": "#2ca02c"}

    years = sorted(pred["year"].unique())
    x = np.arange(len(years))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (col, name) in enumerate(models.items()):
        ics = []
        for yr in years:
            g = pred[pred["year"] == yr]
            ic_series = g.groupby("date").apply(
                lambda d: spearmanr(d[col], d["y_xs"])[0]
            )
            ics.append(ic_series.mean())
        offset = (i - 1) * width
        bars = ax.bar(x + offset, ics, width,
                      label=name, color=colors[name], alpha=0.85)
        for bar, v in zip(bars, ics):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if v >= 0 else -0.005),
                    f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color=colors[name])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.03, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
               label="IC = 0.03 (useful)")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Rank IC (mean over month)")
    ax.set_title("Annual Rank IC by Model\n"
                 "(Spearman correlation between predicted rank and realized excess return)",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    out = REPORTS / "chart2_annual_ic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Chart 3: Decile Monotonicity ─────────────────────────────────────────────

def plot_decile_monotonicity(pred):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    models = [
        ("score_ensemble", "Ensemble", "#1f77b4"),
        ("score_lgbm",     "LGBM",     "#ff7f0e"),
        ("score_ridge",    "Ridge",    "#2ca02c"),
    ]
    n_deciles = 10

    for ax, (col, name, color) in zip(axes, models):
        decile_rets = []
        for date, g in pred.groupby("date"):
            g2 = g.copy()
            g2["decile"] = pd.qcut(g2[col], q=n_deciles,
                                   labels=False, duplicates="drop")
            dr = g2.groupby("decile")["y_xs"].mean()
            decile_rets.append(dr)

        avg = pd.concat(decile_rets, axis=1).mean(axis=1)
        ann = avg * 12

        bars = ax.bar(range(1, len(ann) + 1), ann.values,
                      color=color, alpha=0.75, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Score Decile (1=Lowest, 10=Highest)")
        ax.set_title(f"{name}", fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # Highlight top and bottom
        bars[0].set_edgecolor("red");   bars[0].set_linewidth(1.5)
        bars[-1].set_edgecolor("green"); bars[-1].set_linewidth(1.5)

    axes[0].set_ylabel("Avg Annualized Excess Return")
    fig.suptitle("Return Monotonicity by Score Decile\n"
                 "(ideal: monotonically increasing left → right)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = REPORTS / "chart3_decile_monotonicity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Long / Short stock list ───────────────────────────────────────────────────

def print_long_short_list(pred):
    latest_date = pred["date"].max()
    latest = pred[pred["date"] == latest_date].copy()
    latest["ensemble_pct_rank"] = latest["score_ensemble"].rank(pct=True)

    n = len(latest)
    top_n = max(1, int(n * 0.10))

    long_list  = latest.nlargest(top_n,  "score_ensemble")[["permno", "score_ensemble", "ensemble_pct_rank", "y_xs"]]
    short_list = latest.nsmallest(top_n, "score_ensemble")[["permno", "score_ensemble", "ensemble_pct_rank", "y_xs"]]

    # Load ticker names
    t = pq.read_table(ROOT / "data" / "raw" / "crsp_names.parquet")
    for i, f in enumerate(t.schema):
        if pa.types.is_timestamp(f.type):
            t = t.set_column(i, f.name, t.column(f.name).cast(pa.string()))
    names = t.to_pandas()
    names["namedt"]   = names["namedt"].str[:10]
    names["nameendt"] = names["nameendt"].str[:10].fillna("2099-12-31")
    valid = names[(names["namedt"] <= latest_date) & (names["nameendt"] >= latest_date)]
    ticker_map = valid.set_index("permno")["ticker"].to_dict()

    for df in [long_list, short_list]:
        df["ticker"] = df["permno"].map(ticker_map).fillna("N/A")

    out_long  = REPORTS / "long_list_2023_10.csv"
    out_short = REPORTS / "short_list_2023_10.csv"
    long_list.to_csv(out_long,  index=False)
    short_list.to_csv(out_short, index=False)

    print(f"\nRebalance date: {latest_date}  |  Universe size: {n:,} stocks")
    print(f"Top 10% ({top_n} stocks) → LONG")
    print(long_list[["ticker", "permno", "ensemble_pct_rank"]].head(20).to_string(index=False))
    print(f"\nBottom 10% ({top_n} stocks) → SHORT")
    print(short_list[["ticker", "permno", "ensemble_pct_rank"]].head(20).to_string(index=False))
    print(f"\nFull lists saved: {out_long}, {out_short}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading predictions...")
    pred = load_predictions()

    print("Generating Chart 1: Cumulative Returns...")
    plot_cumulative_returns(pred)

    print("Generating Chart 2: Annual IC...")
    plot_annual_ic(pred)

    print("Generating Chart 3: Decile Monotonicity...")
    plot_decile_monotonicity(pred)

    print("Generating Long/Short Stock List...")
    print_long_short_list(pred)

    print("\nAll done. Charts saved to data/reports/")


if __name__ == "__main__":
    main()
