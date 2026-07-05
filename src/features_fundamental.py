"""
Fundamental features from Compustat quarterly data.
Point-in-time alignment: use rdq (report/announcement date) not datadate.
"""
import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES


def build_point_in_time_fundamentals(
    fundq: pd.DataFrame,
    link: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Compustat → CRSP via link table.
    For each (permno, date) pair, find the most recently announced quarterly report.
    """
    # Valid links only
    link = link[link["linktype"].isin(["LU", "LC"])].copy()
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    # Merge gvkey → permno
    fundq = fundq.merge(link[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    fundq = fundq[
        (fundq["rdq"] >= fundq["linkdt"]) &
        (fundq["rdq"] <= fundq["linkenddt"])
    ]

    # Compute fundamental ratios at report time
    fundq = fundq.copy()

    # We'll join to price data later; for now compute ratio numerators
    # that don't require price
    eps = lambda col: fundq[col].replace(0, np.nan)

    # Profitability
    fundq["roe"]   = fundq["niq"] / eps("ceqq")
    fundq["roa"]   = fundq["niq"] / eps("atq")
    fundq["gross_profitability"] = (fundq["saleq"] - fundq["cogsq"]) / eps("atq")
    fundq["operating_margin"]    = fundq["oibdpq"] / eps("saleq")

    # Growth (vs prior year = 4 quarters ago)
    fundq = fundq.sort_values(["permno", "rdq"])
    fundq["saleq_lag4"]  = fundq.groupby("permno")["saleq"].shift(4)
    fundq["niq_lag4"]    = fundq.groupby("permno")["niq"].shift(4)
    fundq["atq_lag4"]    = fundq.groupby("permno")["atq"].shift(4)

    fundq["revenue_growth"] = (fundq["saleq"] - fundq["saleq_lag4"]) / eps("saleq_lag4")
    fundq["earnings_growth"] = (fundq["niq"] - fundq["niq_lag4"]) / fundq["niq_lag4"].abs().replace(0, np.nan)
    fundq["asset_growth"] = (fundq["atq"] - fundq["atq_lag4"]) / eps("atq_lag4")

    # Leverage
    fundq["debt_to_equity"] = fundq["ltq"] / eps("ceqq")
    fundq["debt_to_assets"] = fundq["ltq"] / eps("atq")

    # --- Earnings surprise (SUE) — PEAD signal ---
    # SUE = (EPS_t - EPS_{t-4}) / rolling_std(EPS changes, 8 quarters)
    # Positive SUE → market underreacts → stock drifts up (Post-Earnings Announcement Drift)
    fundq["eps_chg"] = fundq["epspxq"] - fundq.groupby("permno")["epspxq"].shift(4)
    eps_chg_std = fundq.groupby("permno")["eps_chg"].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    fundq["sue"] = fundq["eps_chg"] / eps_chg_std.replace(0, np.nan)
    # Winsorize SUE at ±5 to avoid distortion from penny-stock EPS
    fundq["sue"] = fundq["sue"].clip(-5, 5)

    # --- Accruals (earnings quality) ---
    # accruals = (net income - EBITDA-based cash earnings proxy) / total assets
    # EBITDA * (1 - tax_rate) ≈ operating cash flow before working capital
    # Negative accruals (cash earnings > accounting earnings) → higher future returns
    fundq["accruals"] = (
        (fundq["niq"] - fundq["oibdpq"] * 0.65)
        / eps("atq")
    )

    # Store per-share values for later price-based ratios
    fundq["book_value_q"]  = fundq["ceqq"]
    fundq["earnings_q"]    = fundq["niq"]
    fundq["sales_q"]       = fundq["saleq"]

    keep_cols = [
        "permno", "rdq",
        "roe", "roa", "gross_profitability", "operating_margin",
        "revenue_growth", "earnings_growth", "asset_growth",
        "debt_to_equity", "debt_to_assets",
        "sue", "accruals",
        "book_value_q", "earnings_q", "sales_q", "cshoq",
    ]
    return fundq[keep_cols].dropna(subset=["rdq"])


def merge_fundamentals_to_universe(
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame,
    crsp: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (permno, rebalance_date), find the latest announced quarter
    (rdq <= rebalance_date) and merge.
    Then compute price-based ratios using current market cap.
    """
    # merge_asof requires the on-key to be globally sorted (not just within group)
    left = (
        universe[["date", "permno", "mktcap"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    right = (
        fundamentals.rename(columns={"rdq": "date_rdq"})
        .sort_values("date_rdq")
        .reset_index(drop=True)
    )

    result = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="date_rdq",
        by="permno",
        direction="backward",
    )

    # Price-based ratios using current mktcap
    mktcap = result["mktcap"].replace(0, np.nan)
    shares = result["cshoq"] * 1_000  # cshoq in thousands

    result["book_to_market"]  = (result["book_value_q"] * 1e6) / mktcap
    result["earnings_yield"]  = (result["earnings_q"]   * 1e6) / mktcap
    result["sales_to_price"]  = (result["sales_q"]      * 1e6) / mktcap

    feature_cols = [
        "date", "permno",
        "book_to_market", "earnings_yield", "sales_to_price",
        "roe", "roa", "gross_profitability", "operating_margin",
        "revenue_growth", "earnings_growth", "asset_growth",
        "debt_to_equity", "debt_to_assets",
        "sue", "accruals",
    ]
    return result[feature_cols]


def main():
    print("Loading data...")
    fundq    = pd.read_parquet(DATA_RAW / "compustat_fundq.parquet")
    link     = pd.read_parquet(DATA_RAW / "crsp_compustat_link.parquet")
    universe = pd.read_parquet(DATA_PROCESSED / "universe.parquet")
    crsp     = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet")

    print("Building point-in-time fundamentals...")
    fundamentals = build_point_in_time_fundamentals(fundq, link)

    print("Merging to universe rebalance dates...")
    features = merge_fundamentals_to_universe(fundamentals, universe, crsp)

    out = DATA_FEATURES / "features_fundamental.parquet"
    features.to_parquet(out, index=False)
    print(f"  Saved {out} — {len(features):,} rows")
    coverage = features[["book_to_market"]].notna().mean()
    print(f"  Coverage (book_to_market): {coverage.values[0]:.1%}")


if __name__ == "__main__":
    main()
