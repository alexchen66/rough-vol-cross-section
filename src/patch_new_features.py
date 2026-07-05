"""
Patch script: add SUE, accruals, close_to_52w_high, max_ret_20d
to existing feature parquets without re-running the full pipeline.

Uses pyarrow throughout to avoid pandas timestamp segfault.
Dates are kept as strings (ISO format sorts lexicographically = chronologically).
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES


# ── safe loader ─────────────────────────────────────────────────────────────

def load(path, columns=None):
    """Load parquet; cast all timestamps to string so pandas doesn't segfault."""
    t = pq.read_table(path, columns=columns)
    casts = []
    for i, field in enumerate(t.schema):
        if pa.types.is_timestamp(field.type):
            casts.append((i, field.name))
    for i, col in casts:
        t = t.set_column(i, col, t.column(col).cast(pa.string()))
    df = t.to_pandas()
    # trim string timestamps to date-only prefix "YYYY-MM-DD"
    for _, col in casts:
        df[col] = df[col].str[:10]
    return df


# ── 1. SUE + Accruals (from Compustat fundq) ────────────────────────────────

def compute_sue_accruals() -> pd.DataFrame:
    print("  Loading Compustat + link table...", flush=True)
    fundq = load(DATA_RAW / "compustat_fundq.parquet")
    link  = load(DATA_RAW / "crsp_compustat_link.parquet")

    # Valid links only
    link = link[link["linktype"].isin(["LU", "LC"])].copy()
    link["linkenddt"] = link["linkenddt"].fillna("2099-12-31").str[:10]
    link["linkdt"]    = link["linkdt"].str[:10]

    # Merge gvkey → permno
    fundq = fundq.merge(link[["gvkey", "permno", "linkdt", "linkenddt"]],
                        on="gvkey", how="inner")
    fundq["rdq"] = fundq["rdq"].str[:10].fillna("")
    fundq = fundq[
        (fundq["rdq"] != "") &
        (fundq["rdq"] >= fundq["linkdt"]) &
        (fundq["rdq"] <= fundq["linkenddt"])
    ].copy()

    fundq = fundq.sort_values(["permno", "rdq"]).reset_index(drop=True)

    # SUE: (EPS_t - EPS_{t-4}) / rolling std of EPS changes
    fundq["eps_lag4"] = fundq.groupby("permno")["epspxq"].shift(4)
    fundq["eps_chg"]  = fundq["epspxq"] - fundq["eps_lag4"]
    eps_std = fundq.groupby("permno")["eps_chg"].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    fundq["sue"] = (fundq["eps_chg"] / eps_std.replace(0, np.nan)).clip(-5, 5)

    # Accruals: (net income - after-tax EBITDA proxy) / total assets
    # negative accruals = cash earnings dominate = higher quality
    fundq["accruals"] = (
        (fundq["niq"] - fundq["oibdpq"] * 0.65)
        / fundq["atq"].abs().replace(0, np.nan)
    )

    keep = fundq[["permno", "rdq", "sue", "accruals"]].dropna(subset=["rdq"])
    keep = keep.sort_values(["permno", "rdq"]).reset_index(drop=True)
    return keep


def merge_sue_to_fundamental(sue_df: pd.DataFrame) -> None:
    """
    Merge SUE/accruals into existing features_fundamental.parquet
    using a manual merge_asof (backward, per permno) on string dates.
    """
    print("  Loading existing fundamental features...", flush=True)
    fund = load(DATA_FEATURES / "features_fundamental.parquet")

    # Remove old columns if they exist (re-running idempotently)
    for col in ["sue", "accruals"]:
        if col in fund.columns:
            fund = fund.drop(columns=[col])

    # Sort both for manual merge_asof
    fund = fund.sort_values(["date"]).reset_index(drop=True)
    sue_df = sue_df.sort_values(["rdq"]).reset_index(drop=True)

    # Per permno: for each universe date, find latest rdq <= date
    results = []
    for permno, g_fund in fund.groupby("permno"):
        g_sue = sue_df[sue_df["permno"] == permno].copy()
        if g_sue.empty:
            g_fund = g_fund.copy()
            g_fund["sue"] = np.nan
            g_fund["accruals"] = np.nan
            results.append(g_fund)
            continue

        sue_dates = g_sue["rdq"].values
        sue_vals  = g_sue["sue"].values
        acc_vals  = g_sue["accruals"].values

        sue_out = []
        acc_out = []
        for d in g_fund["date"].values:
            # Latest rdq <= d
            mask = sue_dates <= d
            if mask.any():
                idx = np.where(mask)[0][-1]
                sue_out.append(sue_vals[idx])
                acc_out.append(acc_vals[idx])
            else:
                sue_out.append(np.nan)
                acc_out.append(np.nan)

        g_fund = g_fund.copy()
        g_fund["sue"]      = sue_out
        g_fund["accruals"] = acc_out
        results.append(g_fund)

        if len(results) % 2000 == 0:
            print(f"    {len(results):,} stocks processed...", flush=True)

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["date", "permno"]).reset_index(drop=True)

    # Save back (keep dates as strings — consistent with how we loaded)
    pq.write_table(pa.Table.from_pandas(out),
                   DATA_FEATURES / "features_fundamental.parquet")
    print(f"  Saved features_fundamental.parquet — {len(out):,} rows, "
          f"sue coverage: {out['sue'].notna().mean():.1%}", flush=True)


# ── 2. close_to_52w_high + max_ret_20d (from CRSP) ─────────────────────────

def compute_price_extras() -> None:
    """Add close_to_52w_high and max_ret_20d to features_price.parquet."""
    print("  Loading CRSP clean...", flush=True)
    crsp = load(DATA_PROCESSED / "crsp_clean.parquet",
                ["date", "permno", "prc", "ret"])

    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    print("  Computing 52w high ratio and MAX...", flush=True)
    grp = crsp.groupby("permno")

    high_252 = grp["prc"].transform(lambda x: x.rolling(252, min_periods=200).max())
    crsp["close_to_52w_high"] = crsp["prc"] / high_252.replace(0, np.nan)
    crsp["max_ret_20d"] = grp["ret"].transform(
        lambda x: x.rolling(20, min_periods=15).max()
    )

    print("  Merging into features_price.parquet...", flush=True)
    price = load(DATA_FEATURES / "features_price.parquet")
    for col in ["close_to_52w_high", "max_ret_20d"]:
        if col in price.columns:
            price = price.drop(columns=[col])

    extras = crsp[["date", "permno", "close_to_52w_high", "max_ret_20d"]]
    price = price.merge(extras, on=["date", "permno"], how="left")

    pq.write_table(pa.Table.from_pandas(price),
                   DATA_FEATURES / "features_price.parquet")
    print(f"  Saved features_price.parquet — {len(price):,} rows", flush=True)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 1: SUE + Accruals ===", flush=True)
    sue_df = compute_sue_accruals()
    print(f"  SUE computed for {sue_df['permno'].nunique():,} stocks, "
          f"{len(sue_df):,} quarterly obs", flush=True)
    merge_sue_to_fundamental(sue_df)

    print("\n=== Step 2: 52w high + MAX ===", flush=True)
    compute_price_extras()

    print("\nDone. Now run: preprocess.py → train.py → evaluation.py", flush=True)


if __name__ == "__main__":
    main()
