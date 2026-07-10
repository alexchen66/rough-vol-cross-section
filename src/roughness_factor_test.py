"""
Pure roughness single-factor test: "Buy Rough, Sell Smooth".

Long top-decile (roughest, highest roughness_126d) / Short bottom-decile (smoothest).
"""
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_FEATURES, DATA_PROCESSED


def _load(path, columns):
    """Load parquet with date as string to avoid pandas timestamp segfault."""
    t = pq.read_table(path, columns=columns)
    for i, col in enumerate(t.schema.names):
        if pa.types.is_timestamp(t.schema.field(col).type):
            t = t.set_column(i, col, t.column(col).cast(pa.string()))
    return t.to_pandas()


def rank_ic_per_date(df, signal_col, label_col="y_xs"):
    def _ic(g):
        valid = g[[signal_col, label_col]].dropna()
        if len(valid) < 10:
            return np.nan
        return spearmanr(valid[signal_col], valid[label_col])[0]
    return df.groupby("date").apply(_ic, include_groups=False).rename("ic")


def long_short_returns(df, signal_col, ret_col="r_fwd", decile=0.10):
    def _spread(g):
        valid = g[[signal_col, ret_col]].dropna()
        if len(valid) < 20:
            return np.nan
        n = max(1, int(len(valid) * decile))
        ranked = valid.sort_values(signal_col)
        return ranked.tail(n)[ret_col].mean() - ranked.head(n)[ret_col].mean()
    return df.groupby("date").apply(_spread, include_groups=False).rename("ls_ret")


def ic_summary(ic):
    ic = ic.dropna()
    return {
        "mean_ic":   round(ic.mean(), 4),
        "median_ic": round(ic.median(), 4),
        "std_ic":    round(ic.std(), 4),
        "ic_ir":     round(ic.mean() / ic.std(), 3) if ic.std() > 0 else np.nan,
        "hit_ratio": round((ic > 0).mean(), 3),
        "n_dates":   len(ic),
    }


def backtest_summary(ls):
    ls = ls.dropna()
    ann = ls.mean() * 12
    vol = ls.std() * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else np.nan
    cum = (1 + ls).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "ann_return": round(ann, 4),
        "ann_vol":    round(vol, 4),
        "sharpe":     round(sharpe, 3),
        "max_dd":     round(max_dd, 4),
        "hit_ratio":  round((ls > 0).mean(), 3),
        "n_months":   len(ls),
    }


def main():
    print("Loading data...", flush=True)

    rough_cols = [
        "date", "permno",
        "roughness_126d", "roughness_252d", "idio_roughness_126d",
        "volterra_rough_spread_H005_H050_126d",
        "idio_volterra_rough_spread_H005_H050_126d",
        "vol_of_vol_126d",
    ]
    feat   = _load(DATA_FEATURES / "features_rough_vol.parquet", rough_cols)
    labels = _load(DATA_PROCESSED / "labels.parquet",
                   ["date", "permno", "r_fwd", "y_xs"])

    df = feat.merge(labels, on=["date", "permno"])
    # year from string date "YYYY-MM-DD ..." → first 4 chars
    df["year"] = df["date"].str[:4].astype(int)

    print(f"  {len(df):,} rows | "
          f"{df['date'].nunique()} rebalance dates | "
          f"{df['permno'].nunique()} stocks", flush=True)

    signals = {
        "roughness_126d":                        "Whittle_126d",
        "roughness_252d":                        "Whittle_252d",
        "idio_roughness_126d":                   "Idio_Whittle_126d",
        "volterra_rough_spread_H005_H050_126d":  "Volterra_spread",
        "idio_volterra_rough_spread_H005_H050_126d": "Idio_Volterra_spread",
        "vol_of_vol_126d":                       "Vol_of_vol_126d",
    }

    rows_ic, rows_bt, ic_yy = [], [], {}

    for col, name in signals.items():
        if col not in df.columns:
            print(f"  skip {col}", flush=True)
            continue
        print(f"  Computing {name}...", flush=True)

        ic = rank_ic_per_date(df, col)
        ls = long_short_returns(df, col)

        s = ic_summary(ic);  s["signal"] = name;  rows_ic.append(s)
        s = backtest_summary(ls); s["signal"] = name; rows_bt.append(s)

        # year-by-year IC: join year from df
        date_year = df[["date","year"]].drop_duplicates().set_index("date")["year"]
        ic_yy[name] = ic.map(date_year).groupby(
            lambda d: date_year.get(d, np.nan)
        ).mean().round(4)
        # Simpler: reindex then group
        ic_df = ic.to_frame("ic")
        ic_df["year"] = ic_df.index.map(date_year)
        ic_yy[name] = ic_df.groupby("year")["ic"].mean().round(4)

    # ── print ────────────────────────────────────────────────────────────────
    print("\n" + "="*72, flush=True)
    print("BUY ROUGH / SELL SMOOTH  —  Single-Factor Test", flush=True)
    print("="*72, flush=True)

    ic_tbl = pd.DataFrame(rows_ic).set_index("signal")
    print("\n--- Rank IC (Spearman, signal vs y_xs) ---")
    print(ic_tbl[["mean_ic","median_ic","std_ic","ic_ir","hit_ratio","n_dates"]].to_string())

    bt_tbl = pd.DataFrame(rows_bt).set_index("signal")
    print("\n--- Long-Short Backtest (top 10% long, bottom 10% short, 0 bps) ---")
    print(bt_tbl[["ann_return","ann_vol","sharpe","max_dd","hit_ratio"]].to_string())

    yy = pd.DataFrame(ic_yy).round(4)
    print("\n--- Year-by-Year Rank IC ---")
    print(yy.to_string())

    # Annual long-short return for main signal
    ls_main = long_short_returns(df, "roughness_126d")
    ls_df = ls_main.dropna().to_frame("ret")
    ls_df["year"] = ls_df.index.map(
        df[["date","year"]].drop_duplicates().set_index("date")["year"]
    )
    ann_yr = ls_df.groupby("year")["ret"].apply(
        lambda x: round((1 + x).prod() - 1, 4)
    )
    print("\n--- Annual Long-Short Return  (roughness_126d) ---")
    print(ann_yr.to_string())

    # ── save ────────────────────────────────────────────────────────────────
    out = ROOT / "data" / "reports"
    out.mkdir(exist_ok=True)
    ic_tbl.to_csv(out / "roughness_factor_ic.csv")
    bt_tbl.to_csv(out / "roughness_factor_backtest.csv")
    yy.to_csv(out / "roughness_factor_ic_by_year.csv")
    print(f"\nSaved to data/reports/  (roughness_factor_*.csv)")


if __name__ == "__main__":
    main()
