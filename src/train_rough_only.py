"""
Train four models on the same walk-forward windows using only rough-volatility features.

This is an ablation experiment. It leaves the original prediction files intact and writes:
  data/processed/predictions_rough_only.parquet
"""
from __future__ import annotations

import pandas as pd

from config import (
    DATA_FEATURES,
    DATA_PROCESSED,
    EMBARGO_DAYS,
    ROUGH_VOL_FEATURES,
    WALK_FORWARD_WINDOWS,
)
from models import ElasticNetModel, LGBMRankerModel, LGBMRegressorModel, RidgeModel
from train import add_embargo, get_feature_cols, make_group_array


def get_rough_only_feature_cols(df: pd.DataFrame) -> list[str]:
    rough = set(ROUGH_VOL_FEATURES)
    rough_missing = {f"{name}_missing" for name in ROUGH_VOL_FEATURES}
    return [
        col for col in get_feature_cols(df)
        if col in rough or col in rough_missing
    ]


def run_rough_only_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_col: str = "y_xs",
    rank_label_col: str = "rank_label",
) -> pd.DataFrame:
    data = features.merge(
        labels[["date", "permno", label_col, rank_label_col]],
        on=["date", "permno"],
        how="inner",
    )
    data = data.sort_values(["date", "permno"]).reset_index(drop=True)

    feature_cols = get_rough_only_feature_cols(data)
    trading_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    all_predictions = []

    print(f"Using {len(feature_cols)} rough-only features.")
    print("Features:")
    for col in feature_cols:
        print(f"  {col}")

    for i, (tr_s, tr_e, va_s, va_e, te_s, te_e) in enumerate(WALK_FORWARD_WINDOWS):
        print(
            f"\nRough-only window {i + 1}/{len(WALK_FORWARD_WINDOWS)}: "
            f"train={tr_s}~{tr_e} | val={va_s}~{va_e} | test={te_s}~{te_e}"
        )

        embargo_start = add_embargo(tr_e, EMBARGO_DAYS, trading_dates)
        actual_va_s = max(pd.Timestamp(va_s), embargo_start)

        train = data[(data["date"] >= tr_s) & (data["date"] <= tr_e)].copy()
        val = data[(data["date"] >= actual_va_s) & (data["date"] <= va_e)].copy()
        test = data[(data["date"] >= te_s) & (data["date"] <= te_e)].copy()

        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            print(f"  {split_name}: {split['date'].nunique()} dates, {len(split):,} rows")

        if len(train) < 1000 or len(val) < 100 or len(test) < 100:
            print("  Skipping window - insufficient data")
            continue

        X_train = train[feature_cols].values
        y_train = train[label_col].values
        r_train = train[rank_label_col].fillna(0).astype(int).values

        X_val = val[feature_cols].values
        y_val = val[label_col].values
        r_val = val[rank_label_col].fillna(0).astype(int).values

        X_test = test[feature_cols].values

        group_train = make_group_array(train.sort_values("date"))
        group_val = make_group_array(val.sort_values("date"))

        test = test.copy()

        print("  Fitting Ridge...")
        ridge = RidgeModel(alpha=1.0)
        ridge.fit(X_train, y_train)
        test["score_ridge"] = ridge.predict(X_test)

        print("  Fitting ElasticNet...")
        elastic_net = ElasticNetModel(alpha=0.001, l1_ratio=0.2)
        elastic_net.fit(X_train, y_train)
        test["score_elastic_net"] = elastic_net.predict(X_test)

        print("  Fitting LGBMRegressor...")
        lgbm_reg = LGBMRegressorModel()
        lgbm_reg.fit(X_train, y_train, X_val, y_val)
        test["score_lgbm"] = lgbm_reg.predict(X_test)

        print("  Fitting LGBMRanker...")
        lgbm_rank = LGBMRankerModel()
        lgbm_rank.fit(
            X_train, r_train, group_train,
            X_val, r_val, group_val,
        )
        test["score_ranker"] = lgbm_rank.predict(X_test)

        all_predictions.append(
            test[
                [
                    "date",
                    "permno",
                    "score_ridge",
                    "score_elastic_net",
                    "score_lgbm",
                    "score_ranker",
                    label_col,
                    rank_label_col,
                ]
            ].copy()
        )

    return pd.concat(all_predictions, ignore_index=True)


def main() -> None:
    print("Loading preprocessed features...")
    features = pd.read_parquet(DATA_FEATURES / "features_preprocessed.parquet")

    print("Loading labels...")
    labels = pd.read_parquet(DATA_PROCESSED / "labels.parquet")

    print("Running rough-only walk-forward training...")
    predictions = run_rough_only_walk_forward(features, labels)

    out = DATA_PROCESSED / "predictions_rough_only.parquet"
    predictions.to_parquet(out, index=False)
    print(f"\nSaved {out} - {len(predictions):,} rows")


if __name__ == "__main__":
    main()
