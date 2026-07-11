"""
Generate forward return labels for each stock on each rebalance date.
All returns are cross-sectional excess (demeaned within universe).
Point-in-time: signal generated at close of rebalance date t,
trade executes at open/close of t+1, hold for HOLDING_PERIOD days.
"""
import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED, HOLDING_PERIOD


def compute_forward_returns(
    crsp: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (permno, rebalance_date) in universe, compute:
      r_fwd   = P(t+1+h) / P(t+1) - 1     raw forward return
      y_xs    = r_fwd - median(r_fwd)       cross-sectional excess
      y_ind   = r_fwd - industry median      industry-neutral label
    """
    crsp = crsp[["date", "permno", "prc", "ret", "siccd"]].copy()
    crsp = crsp.sort_values(["permno", "date"])

    # Build a price series aligned by trading day index
    # Use cumulative returns to get forward prices without price gaps
    crsp["cum_ret"] = crsp.groupby("permno")["ret"].transform(
        lambda x: (1 + x.fillna(0)).cumprod()
    )

    rebal_dates = np.sort(universe["date"].unique())

    all_trading_days = np.sort(crsp["date"].unique())
    day_to_idx = {d: i for i, d in enumerate(all_trading_days)}

    records = []
    for rebal_date in rebal_dates:
        idx_t = day_to_idx.get(rebal_date)
        if idx_t is None:
            continue

        # t+1 (entry) and t+1+h (exit)
        idx_entry = idx_t + 1
        idx_exit  = idx_t + 1 + HOLDING_PERIOD

        if idx_exit >= len(all_trading_days):
            continue

        entry_date = all_trading_days[idx_entry]
        exit_date  = all_trading_days[idx_exit]

        permnos = universe[universe["date"] == rebal_date]["permno"].values

        # Use cumulative return (from ret, which includes dividends + DLRET)
        # r_fwd = cum_ret(exit) / cum_ret(entry) - 1
        # This correctly handles dividends and delisting returns.
        entry_data = (
            crsp[crsp["date"] == entry_date]
            .set_index("permno")[["cum_ret", "siccd"]]
        )
        exit_cum = (
            crsp[crsp["date"] == exit_date]
            .set_index("permno")["cum_ret"]
        )

        df = entry_data.loc[entry_data.index.isin(permnos)].copy()
        df["cum_ret_exit"] = exit_cum
        df = df.dropna(subset=["cum_ret", "cum_ret_exit"])
        df = df[df["cum_ret"] > 0]

        df["r_fwd"] = df["cum_ret_exit"] / df["cum_ret"] - 1
        df["date"]  = rebal_date
        df = df.reset_index()

        records.append(df[["date", "permno", "r_fwd", "siccd"]].copy())

    labels = pd.concat(records, ignore_index=True)

    # Cross-sectional excess return (subtract universe median)
    labels["y_xs"] = labels.groupby("date")["r_fwd"].transform(
        lambda x: x - x.median()
    )

    # Industry-neutral label (subtract 2-digit SIC median)
    labels["sic2"] = (labels["siccd"] // 100).astype("Int64")
    labels["y_ind"] = labels.groupby(["date", "sic2"])["r_fwd"].transform(
        lambda x: x - x.median()
    )

    # Rank label (0-9 deciles) for LGBMRanker
    labels["rank_label"] = labels.groupby("date")["r_fwd"].transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop")
    )

    return labels.drop(columns=["siccd"])


def main():
    print("Loading data...")
    crsp     = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet")
    universe = pd.read_parquet(DATA_PROCESSED / "universe.parquet")

    print("Computing forward return labels...")
    labels = compute_forward_returns(crsp, universe)

    out = DATA_PROCESSED / "labels.parquet"
    labels.to_parquet(out, index=False)
    print(f"  Saved {out} — {len(labels):,} rows")
    print(f"  Label stats:\n{labels[['r_fwd', 'y_xs', 'y_ind']].describe().round(4)}")


if __name__ == "__main__":
    main()
