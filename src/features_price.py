"""
Price and momentum features — computed from daily CRSP returns.
All features use only data available at or before the rebalance date.
"""
import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES

DATA_FEATURES.mkdir(parents=True, exist_ok=True)


def compute_price_features(
    crsp: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns one row per (rebalance_date, permno) with price/momentum features.
    """
    crsp = crsp[["date", "permno", "prc", "ret"]].copy()
    crsp = crsp.sort_values(["permno", "date"])

    # Rolling returns (log returns for multi-period)
    crsp["log_ret"] = np.log1p(crsp["ret"].fillna(0))

    def rolling_cum_ret(log_ret_series, window):
        return np.expm1(log_ret_series.rolling(window, min_periods=int(window * 0.8)).sum())

    grp = crsp.groupby("permno")["log_ret"]

    crsp["ret_1d"]   = crsp["ret"]
    crsp["ret_5d"]   = grp.transform(lambda x: rolling_cum_ret(x, 5))
    crsp["ret_20d"]  = grp.transform(lambda x: rolling_cum_ret(x, 20))
    crsp["ret_60d"]  = grp.transform(lambda x: rolling_cum_ret(x, 60))
    crsp["ret_120d"] = grp.transform(lambda x: rolling_cum_ret(x, 120))
    crsp["ret_252d"] = grp.transform(lambda x: rolling_cum_ret(x, 252))

    # 12-1 momentum: 252d return minus last 20d return
    crsp["mom_12_1"] = (
        crsp.groupby("permno")["log_ret"]
        .transform(lambda x: rolling_cum_ret(x.iloc[:-20] if len(x) > 20 else x, 232))
    )
    # Cleaner: ret_252d / (1+ret_20d) - 1
    crsp["mom_12_1"] = (1 + crsp["ret_252d"]) / (1 + crsp["ret_20d"]) - 1

    # Moving averages
    crsp["ma20"] = crsp.groupby("permno")["prc"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    crsp["ma60"] = crsp.groupby("permno")["prc"].transform(
        lambda x: x.rolling(60, min_periods=45).mean()
    )
    crsp["close_to_ma20"] = crsp["prc"] / crsp["ma20"] - 1
    crsp["close_to_ma60"] = crsp["prc"] / crsp["ma60"] - 1
    crsp["ma20_to_ma60"]  = crsp["ma20"] / crsp["ma60"] - 1

    # Price position within 252-day range
    crsp["high_252"] = crsp.groupby("permno")["prc"].transform(
        lambda x: x.rolling(252, min_periods=200).max()
    )
    crsp["low_252"] = crsp.groupby("permno")["prc"].transform(
        lambda x: x.rolling(252, min_periods=200).min()
    )
    rng = crsp["high_252"] - crsp["low_252"]
    crsp["price_position_252d"] = np.where(
        rng > 0, (crsp["prc"] - crsp["low_252"]) / rng, np.nan
    )

    # 52-week high ratio: price / 52w-high (George & Hwang 2004)
    # Near 52-week high → anchoring bias → market underreacts → positive predictor
    crsp["close_to_52w_high"] = crsp["prc"] / crsp["high_252"].replace(0, np.nan)

    # MAX: largest single-day return over past 20 days (lottery-stock effect)
    # High MAX → investors overpay for skewness → future underperformance (negative predictor)
    crsp["max_ret_20d"] = crsp.groupby("permno")["ret"].transform(
        lambda x: x.rolling(20, min_periods=15).max()
    )

    feature_cols = [
        "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
        "mom_12_1", "close_to_ma20", "close_to_ma60", "ma20_to_ma60",
        "price_position_252d", "close_to_52w_high", "max_ret_20d",
    ]

    # Extract only on rebalance dates in universe
    rebal_pairs = universe[["date", "permno"]].copy()
    result = rebal_pairs.merge(
        crsp[["date", "permno"] + feature_cols],
        on=["date", "permno"],
        how="left",
    )
    return result


def main():
    print("Loading data...")
    crsp     = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet")
    universe = pd.read_parquet(DATA_PROCESSED / "universe.parquet")

    print("Computing price features...")
    features = compute_price_features(crsp, universe)

    out = DATA_FEATURES / "features_price.parquet"
    features.to_parquet(out, index=False)
    print(f"  Saved {out} — {len(features):,} rows, {features.shape[1]} columns")


if __name__ == "__main__":
    main()
