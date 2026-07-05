"""
Analyst revision features from IBES Summary History and Surprise History.

Signals:
  sue_ibes      — Standardized Unexpected Earnings (WRDS pre-computed suescore)
                  actual EPS vs analyst consensus / dispersion
  revision_1m   — % change in consensus EPS estimate over past month
                  positive = analysts collectively raising forecasts
  revision_3m   — same over 3 months (slower, more persistent signal)
  est_disp      — estimate dispersion = STDEV / |MEANEST|
                  high dispersion = high uncertainty = typically negative predictor
  num_analysts  — analyst coverage count (proxy for information environment)

All signals are point-in-time: for each rebalance date, only use IBES data
published strictly before that date.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES

SUMMARY_CSV  = Path.home() / "Downloads" / "dpaecejwdwblpkaq.csv"
SURPRISE_CSV = Path.home() / "Downloads" / "dpaehhg7mzubplzl.csv"


# ── helpers ─────────────────────────────────────────────────────────────────

def load_parquet(path, columns=None):
    t = pq.read_table(path, columns=columns)
    casts = [(i, f.name) for i, f in enumerate(t.schema)
             if pa.types.is_timestamp(f.type)]
    for i, col in casts:
        t = t.set_column(i, col, t.column(col).cast(pa.string()))
    df = t.to_pandas()
    for _, col in casts:
        df[col] = df[col].str[:10]
    return df


# ── build ticker → permno link ───────────────────────────────────────────────

def build_ticker_permno_link() -> pd.DataFrame:
    """
    Map IBES OFTIC → permno via Compustat tic → gvkey → permno.
    Where one tic maps to multiple permno, keep all (will be resolved by date).
    """
    comp = load_parquet(DATA_RAW / "compustat_fundq.parquet",
                        columns=["gvkey", "tic"])
    comp = comp[["gvkey", "tic"]].drop_duplicates().dropna(subset=["tic"])

    link = load_parquet(DATA_RAW / "crsp_compustat_link.parquet")
    link = link[link["linktype"].isin(["LC", "LU"])].copy()
    link["linkenddt"] = link["linkenddt"].fillna("2099-12-31").str[:10]
    link["linkdt"]    = link["linkdt"].str[:10]

    merged = comp.merge(
        link[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner"
    )
    # Upper-case tic for matching
    merged["tic"] = merged["tic"].str.upper().str.strip()
    return merged[["tic", "permno", "linkdt", "linkenddt"]].drop_duplicates()


# ── process Summary Statistics → consensus revision signals ─────────────────

def compute_revision_signals(ticker_permno: pd.DataFrame) -> pd.DataFrame:
    """
    From Summary Statistics, compute per (permno, STATPERS):
      revision_1m, revision_3m, est_disp, num_analysts
    """
    print("  Loading Summary Statistics...", flush=True)
    summ = pd.read_csv(SUMMARY_CSV)
    summ = summ[summ["MEASURE"] == "EPS"].copy()
    summ["OFTIC"] = summ["OFTIC"].str.upper().str.strip()
    summ["STATPERS"] = summ["STATPERS"].str[:10]

    # Link to permno via OFTIC → tic
    summ = summ.merge(
        ticker_permno.rename(columns={"tic": "OFTIC"}),
        on="OFTIC", how="inner"
    )
    # Keep only rows where STATPERS is within the valid link window
    summ = summ[
        (summ["STATPERS"] >= summ["linkdt"]) &
        (summ["STATPERS"] <= summ["linkenddt"])
    ].copy()

    summ = summ.sort_values(["permno", "STATPERS"]).reset_index(drop=True)
    grp = summ.groupby("permno")

    # Estimate dispersion
    summ["est_disp"] = summ["STDEV"] / summ["MEANEST"].abs().replace(0, np.nan)
    summ["num_analysts"] = summ["NUMEST"]

    # Consensus revision: % change in MEANEST
    summ["mean_lag1"] = grp["MEANEST"].shift(1)
    summ["mean_lag3"] = grp["MEANEST"].shift(3)

    denom1 = summ["mean_lag1"].abs().replace(0, np.nan)
    denom3 = summ["mean_lag3"].abs().replace(0, np.nan)

    summ["revision_1m"] = (summ["MEANEST"] - summ["mean_lag1"]) / denom1
    summ["revision_3m"] = (summ["MEANEST"] - summ["mean_lag3"]) / denom3

    # Winsorize revisions at ±2 (outlier EPS movements are noise)
    for col in ["revision_1m", "revision_3m"]:
        summ[col] = summ[col].clip(-2, 2)

    keep = summ[["permno", "STATPERS",
                 "revision_1m", "revision_3m", "est_disp", "num_analysts"]]
    return keep.dropna(subset=["STATPERS"])


# ── process Surprise History → SUE signal ───────────────────────────────────

def compute_sue_signal(ticker_permno: pd.DataFrame) -> pd.DataFrame:
    """
    From Surprise History, extract the pre-computed suescore per (permno, anndats).
    """
    print("  Loading Surprise History...", flush=True)
    surp = pd.read_csv(SURPRISE_CSV)
    surp = surp[surp["MEASURE"] == "EPS"].copy()
    surp["OFTIC"] = surp["OFTIC"].str.upper().str.strip()
    surp["anndats"] = surp["anndats"].str[:10]

    surp = surp.merge(
        ticker_permno.rename(columns={"tic": "OFTIC"}),
        on="OFTIC", how="inner"
    )
    surp = surp[
        (surp["anndats"] >= surp["linkdt"]) &
        (surp["anndats"] <= surp["linkenddt"])
    ].copy()

    # Keep most recent announcement per permno per date
    surp = surp.sort_values(["permno", "anndats"])
    keep = surp[["permno", "anndats", "suescore"]].copy()
    keep["suescore"] = keep["suescore"].clip(-5, 5)
    return keep.dropna(subset=["anndats"])


# ── merge to universe rebalance dates ────────────────────────────────────────

def merge_to_universe(
    revision: pd.DataFrame,
    sue: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (permno, rebalance_date), find the most recent signal date
    strictly before the rebalance date (point-in-time safe).
    """
    print("  Merging to rebalance dates...", flush=True)

    # universe date format should already be YYYY-MM-DD
    uni = universe[["date", "permno"]].copy()
    uni = uni.sort_values(["date"]).reset_index(drop=True)

    results = []
    all_permnos = uni["permno"].unique()

    for i, permno in enumerate(all_permnos):
        g_uni = uni[uni["permno"] == permno].copy()
        dates = g_uni["date"].values  # sorted strings

        # --- Revision signals ---
        g_rev = revision[revision["permno"] == permno].sort_values("STATPERS")
        rev_dates = g_rev["STATPERS"].values

        # --- SUE ---
        g_sue = sue[sue["permno"] == permno].sort_values("anndats")
        sue_dates = g_sue["anndats"].values

        rows = []
        for d in dates:
            row = {"date": d, "permno": permno}

            # Latest revision stats with STATPERS < d (point-in-time)
            mask_rev = rev_dates < d
            if mask_rev.any():
                idx = np.where(mask_rev)[0][-1]
                rev_row = g_rev.iloc[idx]
                row["revision_1m"]   = rev_row["revision_1m"]
                row["revision_3m"]   = rev_row["revision_3m"]
                row["est_disp"]      = rev_row["est_disp"]
                row["num_analysts"]  = rev_row["num_analysts"]
            else:
                row["revision_1m"]   = np.nan
                row["revision_3m"]   = np.nan
                row["est_disp"]      = np.nan
                row["num_analysts"]  = np.nan

            # Latest SUE with anndats < d
            mask_sue = sue_dates < d
            if mask_sue.any():
                idx = np.where(mask_sue)[0][-1]
                row["sue_ibes"] = g_sue.iloc[idx]["suescore"]
            else:
                row["sue_ibes"] = np.nan

            rows.append(row)

        results.append(pd.DataFrame(rows))

        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,}/{len(all_permnos):,} stocks...", flush=True)

    out = pd.concat(results, ignore_index=True)
    return out.sort_values(["date", "permno"]).reset_index(drop=True)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("Building IBES ticker → permno link...", flush=True)
    ticker_permno = build_ticker_permno_link()
    print(f"  {len(ticker_permno):,} tic→permno pairs", flush=True)

    revision = compute_revision_signals(ticker_permno)
    print(f"  Revision signals: {revision['permno'].nunique():,} stocks, "
          f"{len(revision):,} obs", flush=True)

    sue = compute_sue_signal(ticker_permno)
    print(f"  SUE: {sue['permno'].nunique():,} stocks, {len(sue):,} obs",
          flush=True)

    print("Loading universe...", flush=True)
    universe = load_parquet(DATA_PROCESSED / "universe.parquet",
                            columns=["date", "permno"])
    universe["date"] = universe["date"].str[:10]
    print(f"  {universe['permno'].nunique():,} stocks, "
          f"{universe['date'].nunique()} rebalance dates", flush=True)

    features = merge_to_universe(revision, sue, universe)

    # Coverage report
    for col in ["sue_ibes", "revision_1m", "revision_3m", "est_disp", "num_analysts"]:
        cov = features[col].notna().mean()
        print(f"  Coverage {col}: {cov:.1%}", flush=True)

    out_path = DATA_FEATURES / "features_analyst.parquet"
    pq.write_table(pa.Table.from_pandas(features), out_path)
    print(f"\nSaved {out_path} — {len(features):,} rows, "
          f"{features.shape[1]} columns", flush=True)


if __name__ == "__main__":
    main()
