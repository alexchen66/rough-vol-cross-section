"""
Build the investable universe for each monthly rebalance date.
Survivorship-bias-free: includes delisted stocks up to their delisting date.

Date handling: all dates kept as YYYY-MM-DD strings throughout to avoid
pd.to_datetime() segfaults on Python 3.14/macOS. ISO-8601 strings sort
lexicographically, so string comparisons are correct for date filtering.
"""
import datetime as _dt
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    DATA_RAW, DATA_PROCESSED,
    START_DATE, END_DATE,
    MIN_PRICE, MIN_DOLLAR_VOL_20D, MIN_LISTING_DAYS,
)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _to_ordinal(s) -> float:
    """Convert YYYY-MM-DD string to integer day ordinal; NaN on failure."""
    if pd.isna(s):
        return np.nan
    d = str(s)[:10]
    try:
        return float(_dt.date(int(d[:4]), int(d[5:7]), int(d[8:10])).toordinal())
    except (ValueError, TypeError):
        return np.nan


def get_rebalance_dates(crsp: pd.DataFrame) -> list:
    """Last trading day of each month — returns YYYY-MM-DD strings."""
    return (
        crsp.groupby(crsp["date"].str[:7])["date"]
        .max()
        .tolist()
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
    name_col = "namedt"   if "namedt"   in names.columns else "begdat"
    nend_col = "nameendt" if "nameendt" in names.columns else "enddat"

    listing = names[["permno"]].copy()
    listing["first_listed"]  = names[name_col].astype(str).str[:10]
    listing["last_nameendt"] = names[nend_col].fillna("2099-12-31").astype(str).str[:10]

    # Earliest delisting date per permno (string min works for ISO dates)
    delist_date = (
        delist.groupby("permno")["dlstdt"]
        .min()
        .reset_index()
        .rename(columns={"dlstdt": "dlstdt_str"})
    )
    delist_date["dlstdt_str"] = delist_date["dlstdt_str"].astype(str).str[:10]

    listing = listing.merge(delist_date, on="permno", how="left")

    # last_date = min(last_nameendt, dlstdt_str).
    # Can't use DataFrame.min(axis=1) because NaN is float and can't compare with str.
    # Vectorized: start with last_nameendt, then overwrite where dlstdt is earlier.
    listing["last_date"] = listing["last_nameendt"]
    has_delist = listing["dlstdt_str"].notna() & (
        listing["dlstdt_str"].astype(str) != "nan"
    )
    earlier = has_delist & (listing["dlstdt_str"] < listing["last_nameendt"])
    listing.loc[earlier, "last_date"] = listing.loc[earlier, "dlstdt_str"]

    # rolling 20-day avg dollar volume
    crsp = crsp.sort_values(["permno", "date"])
    crsp["avg_dollar_vol_20d"] = (
        crsp.groupby("permno")["dollar_vol"]
        .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )

    rebal_dates = get_rebalance_dates(crsp)
    rebal_dates = [d for d in rebal_dates if START_DATE <= d <= END_DATE]

    records = []
    for date in rebal_dates:
        day_data = crsp[crsp["date"] == date].copy()
        day_data = day_data.merge(listing, on="permno", how="left")

        # Filter 1: price
        day_data = day_data[day_data["prc"] >= MIN_PRICE]

        # Filter 2: liquidity
        day_data = day_data[day_data["avg_dollar_vol_20d"] >= MIN_DOLLAR_VOL_20D]

        # Filter 3: listed long enough (pure-Python ordinal arithmetic, no pd.Timestamp)
        date_ord = _to_ordinal(date)
        fl_ord   = day_data["first_listed"].apply(_to_ordinal)
        day_data = day_data[fl_ord.notna() & ((date_ord - fl_ord) >= MIN_LISTING_DAYS)]

        # Filter 4: not yet delisted (string comparison works for ISO dates)
        day_data = day_data[
            day_data["last_date"].isna() | (day_data["last_date"] >= date)
        ]

        # Filter 5: NYSE/AMEX/NASDAQ only
        if "exchcd" in day_data.columns:
            day_data = day_data[day_data["exchcd"].isin([1, 2, 3])]

        cols = ["date", "permno", "prc", "mktcap"]
        if "exchcd" in day_data.columns:
            cols.append("exchcd")
        records.append(day_data[cols].copy())

    return pd.concat(records, ignore_index=True)


def apply_delisting_returns(
    crsp: pd.DataFrame,
    delist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge delisting returns into daily returns.
    Stocks with missing dlret get -30% (Shumway 1997 convention).
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
    #   both present  → (1+RET)*(1+DLRET)-1
    #   only DLRET    → DLRET
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
    crsp   = pd.read_parquet(DATA_RAW / "crsp_dsf.parquet")
    names  = pd.read_parquet(DATA_RAW / "crsp_names.parquet")
    delist = pd.read_parquet(DATA_RAW / "crsp_delist.parquet")

    # Normalize to YYYY-MM-DD strings — no pd.to_datetime() calls at all.
    crsp["date"]     = crsp["date"].astype(str).str[:10]
    delist["dlstdt"] = delist["dlstdt"].astype(str).str[:10]
    for col in ["namedt", "nameendt", "begdat", "enddat"]:
        if col in names.columns:
            names[col] = names[col].astype(str).str[:10]

    print("Merging names into CRSP...")
    name_col = "namedt"   if "namedt"   in names.columns else "begdat"
    nend_col = "nameendt" if "nameendt" in names.columns else "enddat"
    listing_dates = names[["permno"]].copy()
    listing_dates["namedt"]   = names[name_col]
    listing_dates["nameendt"] = names[nend_col].fillna("2099-12-31")
    crsp = crsp.merge(listing_dates, on="permno", how="left")

    print("Applying delisting returns...")
    crsp = apply_delisting_returns(crsp, delist)

    print("Saving cleaned CRSP...")
    # Dates stay as YYYY-MM-DD strings — all pd.to_datetime / np.astype("datetime64")
    # paths segfault on Python 3.14/macOS ARM. ISO-8601 string comparison is correct
    # for sorting and filtering; downstream scripts are written accordingly.
    crsp.to_parquet(DATA_PROCESSED / "crsp_clean.parquet", index=False)

    print("Building universe...")
    universe = build_universe(crsp, names, delist)
    universe.to_parquet(DATA_PROCESSED / "universe.parquet", index=False)
    n_dates = universe["date"].nunique()
    print(f"  Universe: {len(universe):,} stock-date pairs across {n_dates} rebalance dates")
    print(f"  Avg stocks per date: {len(universe) / max(n_dates, 1):.0f}")


if __name__ == "__main__":
    main()
