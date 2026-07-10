"""
Per-rebalance-date cross-sectional preprocessing:
  1. Winsorize at 1%/99%
  2. Z-score (cross-sectional)
  3. Industry + log(mktcap) neutralization
  4. Missing value flags + median fill
"""
import pandas as pd
import numpy as np
from config import DATA_FEATURES, DATA_PROCESSED, ALL_FEATURES, ANALYST_FEATURES


def winsorize_cross_section(df: pd.DataFrame, features: list, lo=0.01, hi=0.99) -> pd.DataFrame:
    def _win(x):
        lb = x.quantile(lo)
        ub = x.quantile(hi)
        return x.clip(lb, ub)
    df[features] = df.groupby("date")[features].transform(_win)
    return df


def zscore_cross_section(df: pd.DataFrame, features: list) -> pd.DataFrame:
    def _z(x):
        m, s = x.mean(), x.std()
        return (x - m) / s if s > 1e-10 else x - m
    df[features] = df.groupby("date")[features].transform(_z)
    return df


def neutralize(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    For each feature and each date, regress out industry dummies + log(mktcap).
    Replace feature with residual.
    """
    df = df.copy()
    df["log_mktcap"] = np.log(df["mktcap"].replace(0, np.nan))

    for date, grp in df.groupby("date"):
        idx = grp.index
        X_base = pd.get_dummies(grp["sic2"].fillna(-1).astype(int), prefix="ind", drop_first=True)
        X_base["log_mktcap"] = grp["log_mktcap"].fillna(grp["log_mktcap"].median())
        X_arr = X_base.values.astype(float)

        if X_arr.shape[0] < X_arr.shape[1] + 5:
            continue

        XtX_inv = np.linalg.pinv(X_arr.T @ X_arr)

        for feat in features:
            y = grp[feat].values
            valid = np.isfinite(y)
            if valid.sum() < 20:
                continue
            coef = XtX_inv @ (X_arr[valid].T @ y[valid])
            resid = y.copy()
            resid[valid] = y[valid] - X_arr[valid] @ coef
            df.loc[idx, feat] = resid

    return df


def add_missing_flags(df: pd.DataFrame, features: list) -> pd.DataFrame:
    for feat in features:
        df[f"{feat}_missing"] = df[feat].isna().astype(np.int8)
    return df


def fill_missing_with_median(df: pd.DataFrame, features: list) -> pd.DataFrame:
    def _fill(x):
        return x.fillna(x.median())
    df[features] = df.groupby("date")[features].transform(_fill)
    return df


def build_feature_panel() -> pd.DataFrame:
    """Merge all feature files and apply preprocessing."""
    print("Loading feature files...")
    price    = pd.read_parquet(DATA_FEATURES / "features_price.parquet")
    risk     = pd.read_parquet(DATA_FEATURES / "features_risk.parquet")
    fund     = pd.read_parquet(DATA_FEATURES / "features_fundamental.parquet")
    rough    = pd.read_parquet(DATA_FEATURES / "features_rough_vol.parquet")
    analyst_path  = DATA_FEATURES / "features_analyst.parquet"
    rough_v2_path = DATA_FEATURES / "features_rough_vol_v2.parquet"
    analyst  = pd.read_parquet(analyst_path)  if analyst_path.exists()  else None
    rough_v2 = pd.read_parquet(rough_v2_path) if rough_v2_path.exists() else None

    universe = pd.read_parquet(DATA_PROCESSED / "universe.parquet")

    # Merge all
    df = universe[["date", "permno", "mktcap"]].copy()
    feat_dfs = [price, risk, fund, rough]
    if rough_v2 is not None:
        feat_dfs.append(rough_v2)
    if analyst is not None:
        feat_dfs.append(analyst)
    for feat_df in feat_dfs:
        cols = ["date", "permno"] + [c for c in feat_df.columns if c not in ("date", "permno")]
        df = df.merge(feat_df[cols], on=["date", "permno"], how="left")

    # SIC2 for neutralization
    crsp_names = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet")[
        ["date", "permno", "siccd"]
    ].drop_duplicates()
    df = df.merge(crsp_names, on=["date", "permno"], how="left")
    df["sic2"] = (df["siccd"] // 100).astype("Int64")

    # Only keep features that actually exist in columns
    available = [f for f in ALL_FEATURES if f in df.columns]

    print(f"  {len(available)} features available out of {len(ALL_FEATURES)} configured")
    print("  Adding missing flags...")
    df = add_missing_flags(df, available)

    print("  Filling missing values...")
    df = fill_missing_with_median(df, available)

    print("  Winsorizing...")
    df = winsorize_cross_section(df, available)

    print("  Z-scoring...")
    df = zscore_cross_section(df, available)

    print("  Neutralizing vs industry + log(mktcap)...")
    df = neutralize(df, available)

    return df


def main():
    df = build_feature_panel()
    out = DATA_FEATURES / "features_preprocessed.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved {out} — {df.shape}")


if __name__ == "__main__":
    main()
