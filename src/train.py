"""
Walk-forward training with embargo.
Trains Ridge, LGBMRegressor, and LGBMRanker on rolling windows.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    DATA_FEATURES, DATA_PROCESSED, ALL_FEATURES,
    WALK_FORWARD_WINDOWS, EMBARGO_DAYS,
)
from models import RidgeModel, LGBMRegressorModel, LGBMRankerModel


def get_feature_cols(df: pd.DataFrame) -> list:
    base = [f for f in ALL_FEATURES if f in df.columns]
    flag_cols = [f"{f}_missing" for f in ALL_FEATURES if f"{f}_missing" in df.columns]
    return base + flag_cols


def add_embargo(train_end: str, embargo_days: int, trading_dates) -> str:
    """Return the first date string to use for validation after embargo."""
    idx = np.searchsorted(trading_dates, train_end[:10], side="right")
    embargo_idx = min(idx + embargo_days, len(trading_dates) - 1)
    return trading_dates[embargo_idx]


def make_group_array(df: pd.DataFrame) -> np.ndarray:
    """Number of stocks per rebalance date, in sorted date order."""
    return df.groupby("date").size().sort_index().values


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_col: str = "y_xs",
    rank_label_col: str = "rank_label",
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      date, permno, score_ridge, score_lgbm, score_ranker, y_xs, rank_label
    concatenated across all test windows.
    """
    # Merge features + labels
    data = features.merge(
        labels[["date", "permno", label_col, rank_label_col]],
        on=["date", "permno"],
        how="inner",
    )
    data = data.sort_values(["date", "permno"]).reset_index(drop=True)

    feature_cols = get_feature_cols(data)
    trading_dates = np.array(sorted(data["date"].str[:10].unique()))

    all_predictions = []

    for i, (tr_s, tr_e, va_s, va_e, te_s, te_e) in enumerate(WALK_FORWARD_WINDOWS):
        print(f"\nWindow {i+1}/{len(WALK_FORWARD_WINDOWS)}: "
              f"train={tr_s}~{tr_e} | val={va_s}~{va_e} | test={te_s}~{te_e}")

        # Embargo: shift val start forward
        embargo_start = add_embargo(tr_e, EMBARGO_DAYS, trading_dates)
        actual_va_s = max(va_s[:10], embargo_start)

        train = data[(data["date"] >= tr_s) & (data["date"] <= tr_e)].copy()
        val   = data[(data["date"] >= actual_va_s) & (data["date"] <= va_e)].copy()
        test  = data[(data["date"] >= te_s) & (data["date"] <= te_e)].copy()

        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            print(f"  {split_name}: {split['date'].nunique()} dates, {len(split):,} rows")

        if len(train) < 1000 or len(val) < 100 or len(test) < 100:
            print("  Skipping window — insufficient data")
            continue

        X_train = train[feature_cols].values
        y_train = train[label_col].values
        r_train = train[rank_label_col].fillna(0).astype(int).values

        X_val   = val[feature_cols].values
        y_val   = val[label_col].values
        r_val   = val[rank_label_col].fillna(0).astype(int).values

        X_test  = test[feature_cols].values

        group_train = make_group_array(train.sort_values("date"))
        group_val   = make_group_array(val.sort_values("date"))

        # --- Ridge ---
        print("  Fitting Ridge...")
        ridge = RidgeModel(alpha=1.0)
        ridge.fit(X_train, y_train)
        test = test.copy()
        test["score_ridge"] = ridge.predict(X_test)

        # --- LGBMRegressor ---
        print("  Fitting LGBMRegressor...")
        lgbm_reg = LGBMRegressorModel()
        lgbm_reg.fit(X_train, y_train, X_val, y_val)
        test["score_lgbm"] = lgbm_reg.predict(X_test)

        # --- LGBMRanker ---
        print("  Fitting LGBMRanker...")
        lgbm_rank = LGBMRankerModel()
        lgbm_rank.fit(
            X_train, r_train, group_train,
            X_val,   r_val,   group_val,
        )
        test["score_ranker"] = lgbm_rank.predict(X_test)

        # --- Ensemble: average Ridge + LGBM (after rank-normalising per date) ---
        # Rank-normalise each score to [0,1] cross-sectionally before averaging
        # so that Ridge's linear scale and LGBM's tree scale are commensurate.
        def _rank_norm(s):
            return s.groupby(test["date"]).rank(pct=True)

        test["score_ensemble"] = (
            0.3 * _rank_norm(test["score_ridge"]) + 0.7 * _rank_norm(test["score_lgbm"])
        )

        all_predictions.append(
            test[["date", "permno",
                  "score_ridge", "score_lgbm", "score_ranker", "score_ensemble",
                  label_col, rank_label_col]].copy()
        )

        # Save feature importances for last lgbm model
        if i == len(WALK_FORWARD_WINDOWS) - 1:
            _save_feature_importance(lgbm_reg, lgbm_rank, feature_cols)

    predictions = pd.concat(all_predictions, ignore_index=True)
    return predictions


def _save_feature_importance(lgbm_reg, lgbm_rank, feature_cols):
    out = DATA_PROCESSED.parent / "reports"
    out.mkdir(exist_ok=True)

    fi_reg = pd.DataFrame({
        "feature": feature_cols,
        "importance_regressor": lgbm_reg.feature_importances_,
    }).sort_values("importance_regressor", ascending=False)
    fi_reg.to_csv(out / "feature_importance_regressor.csv", index=False)

    fi_rank = pd.DataFrame({
        "feature": feature_cols,
        "importance_ranker": lgbm_rank.feature_importances_,
    }).sort_values("importance_ranker", ascending=False)
    fi_rank.to_csv(out / "feature_importance_ranker.csv", index=False)

    print("  Feature importances saved to reports/")


def main():
    print("Loading preprocessed features...")
    features = pd.read_parquet(DATA_FEATURES / "features_preprocessed.parquet")

    print("Loading labels...")
    labels = pd.read_parquet(DATA_PROCESSED / "labels.parquet")

    print("Running walk-forward training...")
    predictions = run_walk_forward(features, labels)

    out = DATA_PROCESSED / "predictions.parquet"
    predictions.to_parquet(out, index=False)
    print(f"\nSaved {out} — {len(predictions):,} rows")


if __name__ == "__main__":
    main()
