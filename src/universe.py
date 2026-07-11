"""
Build the investable universe for each monthly rebalance date.
Survivorship-bias-free: includes delisted stocks up to their delisting date.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    DATA_RAW, DATA_PROCESSED,
    START_DATE, END_DATE,
    MIN_PRICE, MIN_DOLLAR_VOL_20D, MIN_LISTING_DAYS,
)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def get_rebalance_dates(crsp: pd.DataFrame) -> pd.DatetimeIndex:
    """Last trading day of each month."""
    return (
        crsp.groupby(crsp["date"].dt.to_period("M"))["date"]
        .max()
        .values
    )


def build_universe(
    crsp: pd.DataFrame,
    names: pd.DataFrame,
    delist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: [date, permno, prc, mktcap, exchcd]
    representing the investable universe on each rebalance date.
    """
    # --- listing dates from names (MSEHEAD: one row per permno) ---
    name_col  = "namedt"  if "namedt"  in names.columns else "begdat"
    nend_col  = "nameendt" if "nameendt" in names.columns else "enddat"
    listing = names[["permno"]].copy()
    listing["first_listed"] = names[name_col]
    listing["last_nameendt"] = names[nend_col].fillna(pd.Timestamp("2099-12-31"))

    # --- attach earliest delisting date per permno ---
    delist_date = delist.groupby("permno")["dlstdt"].min().reset_index()
    listing = listing.merge(delist_date, on="permno", how="left")
    listing["last_date"] = listing[["last_nameendt", "dlstdt"]].min(axis=1)

    # --- rolling 20-day avg dollar volume ---
    crsp = crsp.sort_values(["permno", "date"])
    crsp["avg_dollar_vol_20d"] = (
        crsp.groupby("permno")["dollar_vol"]
        .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )

    # --- rebalance dates ---
    rebal_dates = get_rebalance_dates(crsp)
    rebal_dates = pd.DatetimeIndex(rebal_dates)
    rebal_dates = rebal_dates[
        (rebal_dates >= pd.Timestamp(START_DATE))
        & (rebal_dates <= pd.Timestamp(END_DATE))
    ]

    records = []
    for date in rebal_dates:
        day_data = crsp[crsp["date"] == date].copy()
        day_data = day_data.merge(listing, on="permno", how="left")

        # Filter 1: price > MIN_PRICE
        day_data = day_data[day_data["prc"] >= MIN_PRICE]

        # Filter 2: liquidity
        day_data = day_data[day_data["avg_dollar_vol_20d"] >= MIN_DOLLAR_VOL_20D]

        # Filter 3: listed long enough
        day_data = day_data[
            day_data["first_listed"].notna() &
            ((date - day_data["first_listed"]).dt.days >= MIN_LISTING_DAYS)
        ]

        # Filter 4: not yet delisted
        day_data = day_data[
            day_data["last_date"].isna() | (day_data["last_date"] >= date)
        ]

        # Filter 5: NYSE/AMEX/NASDAQ only (exchcd 1/2/3) — already in CRSP DSF
        if "exchcd" in day_data.columns:
            day_data = day_data[day_data["exchcd"].isin([1, 2, 3])]

        records.append(day_data[["date", "permno", "prc", "mktcap",
                                  "exchcd"] if "exchcd" in day_data.columns
                                 else ["date", "permno", "prc", "mktcap"]].copy())

    return pd.concat(records, ignore_index=True)


def apply_delisting_returns(
    crsp: pd.DataFrame,
    delist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge delisting returns into daily returns.
    When a stock is delisted, replace the last NaN return with dlret.
    Stocks with missing dlret get -30% (Shumway 1997 convention for NYSE/AMEX).
    """
    delist = delist.copy()
    delist["dlret"] = delist["dlret"].fillna(-0.30)
    delist["dlret"] = delist["dlret"].clip(lower=-1.0)

    crsp = crsp.merge(
        delist[["permno", "dlstdt", "dlret"]],
        left_on=["permno", "date"],
        right_on=["permno", "dlstdt"],
        how="left",
    )
    # CRSP canonical DLRET treatment:
    #   both present  → (1+RET)*(1+DLRET)-1  (compound)
    #   only DLRET    → DLRET
    #   neither       → leave as-is (may be NaN)
    mask_both       = crsp["dlret"].notna() & crsp["ret"].notna()
    mask_only_dlret = crsp["dlret"].notna() & crsp["ret"].isna()
    crsp.loc[mask_both, "ret"] = (
        (1 + crsp.loc[mask_both, "ret"]) *
        (1 + crsp.loc[mask_both, "dlret"]) - 1
    )
    crsp.loc[mask_only_dlret, "ret"] = crsp.loc[mask_only_dlret, "dlret"]
    crsp = crsp.drop(columns=["dlstdt", "dlret"])
    return crsp


def main():
    print("Loading raw data...")
    crsp  = pd.read_parquet(DATA_RAW / "crsp_dsf.parquet")
    names = pd.read_parquet(DATA_RAW / "crsp_names.parquet")
    delist = pd.read_parquet(DATA_RAW / "crsp_delist.parquet")

    print("Merging names into CRSP...")
    # exchcd and siccd already in CRSP DSF — just merge listing dates from names
    listing_dates = names[["permno"]].copy()
    if "namedt" in names.columns:
        listing_dates["namedt"]   = names["namedt"]
        listing_dates["nameendt"] = names["nameendt"].fillna(pd.Timestamp("2099-12-31"))
    elif "begdat" in names.columns:
        listing_dates["namedt"]   = names["begdat"]
        listing_dates["nameendt"] = names["enddat"].fillna(pd.Timestamp("2099-12-31"))

    crsp = crsp.merge(listing_dates, on="permno", how="left")

    print("Applying delisting returns...")
    crsp = apply_delisting_returns(crsp, delist)

    print("Saving cleaned CRSP...")
    crsp.to_parquet(DATA_PROCESSED / "crsp_clean.parquet", index=False)

    print("Building universe...")
    universe = build_universe(crsp, names, delist)
    universe.to_parquet(DATA_PROCESSED / "universe.parquet", index=False)
    print(f"  Universe: {len(universe):,} stock-date pairs across {universe['date'].nunique()} rebalance dates")
    print(f"  Avg stocks per date: {len(universe) / universe['date'].nunique():.0f}")


if __name__ == "__main__":
    main()
