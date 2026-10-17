"""비교할 세 가지 모델.

1. Logit-10 : 변수 10개만 쓰는 로지스틱 회귀. 신용평가 스코어카드처럼 사람이 읽고 설명할 수 있는 모델
2. Logit    : 변수 전체를 쓰는 L2 정규화 로지스틱 회귀. 비교 기준선
3. XGBoost  : 그래디언트 부스팅 트리. 비선형 관계와 변수 간 상호작용을 잡는다

모든 전처리(결측 대체, 변환, 변수 선택)는 파이프라인 안에 넣어서 교차검증의 각 fold에서
학습 데이터로만 학습되게 했다. 밖에서 미리 하면 검증 데이터 정보가 새어 성능이 부풀려진다.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

from .config import SEED


def _preprocess() -> list:
    # 재무비율은 한쪽으로 크게 치우친 분포가 많아서, 순위 기반으로 정규분포 모양으로 바꾼다.
    # 이렇게 하면 극단값 몇 개가 로지스틱 회귀 계수를 좌우하지 못한다.
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("quantile", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                         random_state=SEED)),
    ]


def make_logit(C: float = 0.1, balanced: bool = False) -> Pipeline:
    return Pipeline(_preprocess() + [
        ("model", LogisticRegression(C=C, max_iter=5000,
                                     class_weight="balanced" if balanced else None)),
    ])


def make_sparse_logit(n_features: int = 10, C: float = 0.1, select_C: float = 0.05) -> Pipeline:
    selector = SelectFromModel(
        LogisticRegression(penalty="l1", C=select_C, solver="liblinear"),
        max_features=n_features, threshold=-np.inf,
    )
    return Pipeline(_preprocess() + [
        ("select", selector),
        ("model", LogisticRegression(C=C, max_iter=5000)),
    ])


XGB_DEFAULTS = dict(
    n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
)


def make_xgb(scale_pos_weight: float = 1.0, **params) -> XGBClassifier:
    # 결측은 대체하지 않는다. XGBoost는 결측값을 어느 쪽 가지로 보낼지 학습한다.
    cfg = {**XGB_DEFAULTS, **params}
    return XGBClassifier(**cfg, scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                         tree_method="hist", n_jobs=4, random_state=SEED)


def build_models(params: dict | None = None) -> dict:
    """메인 비교에 쓰는 세 모델. params로 튜닝 결과를 덮어쓴다."""
    params = params or {}
    return {
        "Logit-10": make_sparse_logit(**params.get("Logit-10", {})),
        "Logit": make_logit(**params.get("Logit", {})),
        "XGBoost": make_xgb(**params.get("XGBoost", {})),
    }
