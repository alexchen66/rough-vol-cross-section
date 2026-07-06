"""
Train Elastic Net on the same walk-forward windows as the main models.

This script does not retrain Ridge, LGBM, or Ranker. It writes a separate
prediction file:
  data/processed/predictions_elastic_net.parquet
"""
from __future__ import annotations

import pandas as pd

from config import DATA_FEATURES, DATA_PROCESSED, WALK_FORWARD_WINDOWS, EMBARGO_DAYS
from models import ElasticNetModel
from train import add_embargo, get_feature_cols


def run_elastic_net_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_col: str = "y_xs",
    rank_label_col: str = "rank_label",
    alpha: float = 0.001,
    l1_ratio: float = 0.2,
) -> pd.DataFrame:
    data = features.merge(
        labels[["date", "permno", label_col, rank_label_col]],
        on=["date", "permno"],
        how="inner",
    )
    data = data.sort_values(["date", "permno"]).reset_index(drop=True)

    feature_cols = get_feature_cols(data)
    trading_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    all_predictions = []

    for i, (tr_s, tr_e, va_s, va_e, te_s, te_e) in enumerate(WALK_FORWARD_WINDOWS):
        print(
            f"\nElastic Net window {i + 1}/{len(WALK_FORWARD_WINDOWS)}: "
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
        X_test = test[feature_cols].values

        print(f"  Fitting ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})...")
        model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)

        test = test.copy()
        test["score_elastic_net"] = model.predict(X_test)
        all_predictions.append(
            test[["date", "permno", "score_elastic_net", label_col, rank_label_col]].copy()
        )

    return pd.concat(all_predictions, ignore_index=True)


def main() -> None:
    print("Loading preprocessed features...")
    features = pd.read_parquet(DATA_FEATURES / "features_preprocessed.parquet")

    print("Loading labels...")
    labels = pd.read_parquet(DATA_PROCESSED / "labels.parquet")

    print("Running Elastic Net walk-forward training...")
    predictions = run_elastic_net_walk_forward(features, labels)

    out = DATA_PROCESSED / "predictions_elastic_net.parquet"
    predictions.to_parquet(out, index=False)
    print(f"\nSaved {out} - {len(predictions):,} rows")


if __name__ == "__main__":
    main()
