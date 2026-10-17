"""교차검증, 예측, 보정(calibration) 점검."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .config import CV_FOLDS, CV_REPEATS, SEED
from .metrics import evaluate


def cross_validate(models: dict, X: pd.DataFrame, y: pd.Series,
                   n_splits: int = CV_FOLDS, n_repeats: int = CV_REPEATS) -> pd.DataFrame:
    """반복 층화 k-fold로 모델별·fold별 지표를 구한다. 모든 모델이 같은 fold를 쓴다."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        for name, model in models.items():
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            rows.append({"model": name, "fold": fold, **evaluate(y.iloc[va], p)})
    return pd.DataFrame(rows)


def summarize_cv(cv_scores: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in cv_scores.columns if c not in ("model", "fold")]
    agg = cv_scores.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    return agg


def out_of_fold(model, X: pd.DataFrame, y: pd.Series, n_splits: int = CV_FOLDS) -> np.ndarray:
    """학습 데이터의 각 기업에 대해, 그 기업을 보지 않은 모델의 예측 PD를 만든다."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def calibration_bins(y_true, pd_hat, n_bins: int = 10) -> pd.DataFrame:
    """예측 PD 기준 동일 인원 구간마다 평균 예측 PD와 실제 부도율을 비교한다."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    frame["bin"] = pd.qcut(frame["pd"].rank(method="first"), n_bins, labels=False)
    return frame.groupby("bin").agg(predicted=("pd", "mean"), observed=("y", "mean"),
                                    n=("y", "size"))


def calibration_summary(y_true, pd_hat) -> dict[str, float]:
    """평균 예측 PD가 실제 부도율의 몇 배인지 (1에 가까울수록 좋다)."""
    y_true = np.asarray(y_true)
    pd_hat = np.asarray(pd_hat)
    return {"mean_predicted_pd": float(pd_hat.mean()), "observed_default_rate": float(y_true.mean()),
            "ratio": float(pd_hat.mean() / y_true.mean())}
