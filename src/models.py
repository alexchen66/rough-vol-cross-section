"""
Model wrappers: Ridge, ElasticNet, LGBMRegressor, LGBMRanker.
Each exposes a uniform fit / predict interface.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor, LGBMRanker


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model  = Ridge(alpha=alpha)

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class ElasticNetModel:
    def __init__(self, alpha=0.001, l1_ratio=0.2, max_iter=5000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=42,
            selection="cyclic",
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class LGBMRegressorModel:
    def __init__(self, **params):
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        defaults.update(params)
        self.model = LGBMRegressor(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = []
        if eval_set:
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(50, verbose=False), log_evaluation(period=-1)]
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def booster_(self):
        return self.model.booster_


class LGBMRankerModel:
    """
    Learning-to-rank model. Requires group array (stocks per rebalance date).
    Optimizes lambdarank / rank_xendcg objective.
    """
    def __init__(self, objective="lambdarank", **params):
        defaults = dict(
            objective=objective,
            metric="ndcg",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        defaults.update(params)
        self.model = LGBMRanker(**defaults)

    def fit(self, X_train, y_train, group_train,
            X_val=None, y_val=None, group_val=None, **kwargs):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        eval_group = [group_val] if group_val is not None else None
        callbacks = []
        if eval_set:
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(50, verbose=False), log_evaluation(period=-1)]
        self.model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
