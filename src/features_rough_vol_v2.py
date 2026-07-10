"""
BRSS-like realized Hurst exponent using Garman-Klass daily volatility.

改进点（相比 features_rough_vol.py）：
  - 用 Garman-Klass 公式估计每日独立方差，相邻两天零重叠
  - 按 BRSS 原文用 lag=1,...,10 的 log-log 回归估 H（每月一次）
  - 朋友指出 rv_20d 相邻窗口 95% 重叠是粗糙度信号弱的主因

输出特征（加入 features_rough_vol.parquet）：
  gk_vol          — Garman-Klass 日度波动率（替代 rv_20d 作为 vol proxy）
  hurst_gk_126d   — BRSS-like Hurst H（126 天月度滚动估计）
  hurst_gk_252d   — BRSS-like Hurst H（252 天估计）
  roughness_gk_126d = 0.5 - hurst_gk_126d
  roughness_gk_252d = 0.5 - hurst_gk_252d
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

LAG_GRID = np.arange(1, 11)   # ell = 1,...,10 per BRSS


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── Step 1: Garman-Klass daily volatility ────────────────────────────────────

def compute_gk_vol(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Garman-Klass estimator:
      sigma_GK^2 = 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2

    Returns daily DataFrame with columns: permno, date, gk_vol
    Each row is fully independent — no rolling window overlap.
    """
    df = ohlc.copy()

    # Need positive prices for log
    valid = (
        (df["bidlo"] > 0) & (df["askhi"] > 0) &
        (df["openprc"] > 0) & (df["prc"] > 0) &
        (df["askhi"] >= df["bidlo"])
    )
    df = df[valid].copy()

    log_hl = np.log(df["askhi"] / df["bidlo"])
    log_co = np.log(df["prc"]   / df["openprc"])

    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    # Clamp to non-negative (can be tiny-negative due to floating point)
    gk_var = gk_var.clip(lower=0)
    df["gk_vol"] = np.sqrt(gk_var)

    return df[["permno", "date", "gk_vol"]]


# ── Step 2: BRSS-like Hurst estimation ───────────────────────────────────────

def estimate_hurst_brss(log_vol: np.ndarray, lag_grid: np.ndarray = LAG_GRID) -> float:
    """
    BRSS eq: log z_2(ell) = beta1 + beta2 * log(ell)
    H = beta2 / 2

    log_vol: 1-D array of log(gk_vol), no NaN, sorted by date
    Returns H or NaN if insufficient data.
    """
    T = len(log_vol)
    if T < lag_grid.max() + 5:
        return np.nan

    z2 = np.array([
        np.mean((log_vol[ell:] - log_vol[:-ell])**2)
        for ell in lag_grid
    ])

    # log-log regression
    log_ell = np.log(lag_grid.astype(float))
    log_z2  = np.log(z2 + 1e-30)

    # Simple OLS slope
    x = log_ell - log_ell.mean()
    y = log_z2  - log_z2.mean()
    beta2 = (x * y).sum() / (x * x).sum()

    return beta2 / 2.0


def compute_hurst_rolling(gk: pd.DataFrame, window: int) -> pd.Series:
    """
    For each permno, compute monthly BRSS-like H using a rolling window
    of `window` trading days of log(gk_vol).
    """
    gk = gk.sort_values(["permno", "date"]).copy()
    gk["log_gk"] = np.log(gk["gk_vol"].replace(0, np.nan))

    results = []
    for permno, g in gk.groupby("permno"):
        g = g.dropna(subset=["log_gk"]).copy()
        log_v = g["log_gk"].values
        dates  = g["date"].values
        n = len(log_v)

        h_vals = np.full(n, np.nan)
        for i in range(window - 1, n):
            window_vals = log_v[max(0, i - window + 1): i + 1]
            h_vals[i] = estimate_hurst_brss(window_vals)

        g = g.copy()
        g[f"hurst_gk_{window}d"] = h_vals
        results.append(g[["permno", "date", f"hurst_gk_{window}d"]])

    return pd.concat(results, ignore_index=True)


# ── Step 3: merge to universe rebalance dates ────────────────────────────────

def merge_to_universe(features_daily: pd.DataFrame,
                      universe: pd.DataFrame,
                      feat_cols: list) -> pd.DataFrame:
    """
    Point-in-time merge: for each (permno, rebalance_date),
    take the most recent feature value with date < rebalance_date.
    """
    uni = universe[["date", "permno"]].copy().sort_values(["date"])
    all_permnos = uni["permno"].unique()
    results = []

    for i, permno in enumerate(all_permnos):
        g_uni  = uni[uni["permno"] == permno]
        g_feat = features_daily[features_daily["permno"] == permno].sort_values("date")

        if g_feat.empty:
            rows = g_uni.copy()
            for c in feat_cols:
                rows[c] = np.nan
            results.append(rows)
            continue

        feat_dates = g_feat["date"].values
        feat_vals  = {c: g_feat[c].values for c in feat_cols}

        rows = []
        for d in g_uni["date"].values:
            row = {"date": d, "permno": permno}
            mask = feat_dates < d
            if mask.any():
                idx = np.where(mask)[0][-1]
                for c in feat_cols:
                    row[c] = feat_vals[c][idx]
            else:
                for c in feat_cols:
                    row[c] = np.nan
            rows.append(row)

        results.append(pd.DataFrame(rows))

        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(all_permnos):,} stocks...", flush=True)

    return pd.concat(results, ignore_index=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading CRSP OHLC...", flush=True)
    ohlc = load_parquet(DATA_RAW / "crsp_ohlc.parquet")
    print(f"  {len(ohlc):,} rows, {ohlc['permno'].nunique():,} permnos", flush=True)

    print("Loading crsp_clean for close price...", flush=True)
    crsp = load_parquet(DATA_PROCESSED / "crsp_clean.parquet",
                        columns=["permno", "date", "prc"])

    print("Merging close price into OHLC...", flush=True)
    ohlc = ohlc.merge(crsp, on=["permno", "date"], how="inner")
    print(f"  After merge: {len(ohlc):,} rows", flush=True)

    print("Computing Garman-Klass daily volatility...", flush=True)
    gk = compute_gk_vol(ohlc)
    cov = (gk["gk_vol"] > 0).mean()
    print(f"  GK vol coverage: {cov:.1%}, "
          f"median: {gk['gk_vol'].median():.4f}", flush=True)

    print("Estimating BRSS-like Hurst H (126d window)...", flush=True)
    hurst_126 = compute_hurst_rolling(gk, window=126)
    print(f"  H_126d: mean={hurst_126['hurst_gk_126d'].mean():.3f}, "
          f"std={hurst_126['hurst_gk_126d'].std():.3f}", flush=True)

    print("Estimating BRSS-like Hurst H (252d window)...", flush=True)
    hurst_252 = compute_hurst_rolling(gk, window=252)

    print("Merging H estimates...", flush=True)
    hurst = hurst_126.merge(hurst_252, on=["permno", "date"], how="outer")
    hurst = hurst.merge(gk, on=["permno", "date"], how="outer")
    hurst["roughness_gk_126d"] = (0.5 - hurst["hurst_gk_126d"]).clip(-0.5, 0.5)
    hurst["roughness_gk_252d"] = (0.5 - hurst["hurst_gk_252d"]).clip(-0.5, 0.5)

    print("Loading universe...", flush=True)
    universe = load_parquet(DATA_PROCESSED / "universe.parquet",
                            columns=["date", "permno"])
    universe["date"] = universe["date"].str[:10]

    feat_cols = ["gk_vol", "hurst_gk_126d", "hurst_gk_252d",
                 "roughness_gk_126d", "roughness_gk_252d"]

    print("Merging to universe rebalance dates...", flush=True)
    out = merge_to_universe(hurst, universe, feat_cols)

    for c in feat_cols:
        cov = out[c].notna().mean()
        print(f"  Coverage {c}: {cov:.1%}", flush=True)

    out_path = DATA_FEATURES / "features_rough_vol_v2.parquet"
    pq.write_table(pa.Table.from_pandas(out), out_path)
    print(f"\nSaved {out_path} — {len(out):,} rows", flush=True)

    print("\nNext: add these features to config.py and re-run preprocess + train")


if __name__ == "__main__":
    main()
