"""부도 예측 모델 평가 지표.

부도율이 3% 남짓이라 정확도(accuracy)는 쓰지 않는다. 모든 기업을 '정상'이라고만 해도
정확도가 97%가 나오기 때문이다. 대신 신용평가 실무에서 쓰는 순위 지표와 확률 지표를 함께 본다.

- AUROC : 부도 기업이 정상 기업보다 위험하게 평가될 확률
- AR    : Accuracy Ratio(= Gini = 2*AUROC - 1). 신용평가모형 검증에서 주로 쓰는 표기
- KS    : 부도/정상 기업의 누적 점수 분포가 가장 크게 벌어지는 지점의 차이
- PR-AUC: 부도 기업을 찾아내는 정밀도-재현율 곡선 아래 면적 (무작위 기준선 = 부도율)
- Brier : 예측 PD와 실제 결과의 평균 제곱오차. PD를 '확률'로 쓸 수 있는지 본다
- 상위 10% 재현율: 가장 위험하다고 본 10% 안에 실제 부도 기업이 몇 %나 들어오는가
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def recall_at_top(y_true, y_score, frac: float = 0.10) -> float:
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * frac)))
    top = np.argsort(-np.asarray(y_score), kind="stable")[:k]
    return float(y_true[top].sum() / y_true.sum())


def evaluate(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "AUROC": auc,
        "AR": 2 * auc - 1,
        "KS": ks_statistic(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
        "Brier": brier_score_loss(y_true, y_score),
        "Recall@10%": recall_at_top(y_true, y_score, 0.10),
    }


def bootstrap_ci(y_true, y_score, metric=roc_auc_score, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """평가 데이터를 복원추출해 지표의 신뢰구간을 구한다.

    평가 데이터의 부도 기업이 44개뿐이라 점추정 하나만 보면 우연한 차이를 과대해석하기 쉽다.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    stats = []
    n = len(y_true)
    while len(stats) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # 한 클래스만 뽑히면 AUROC를 계산할 수 없다
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, score_a, score_b, metric=roc_auc_score,
                          n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """같은 표본으로 두 모델 지표의 차이(a - b)와 95% 신뢰구간을 구한다."""
    rng = np.random.default_rng(seed)
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    diffs = []
    n = len(y_true)
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(metric(y_true[idx], score_a[idx]) - metric(y_true[idx], score_b[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(metric(y_true, score_a) - metric(y_true, score_b)),
            "ci_low": float(lo), "ci_high": float(hi)}


def grade_table(y_true, pd_hat, n_grades: int = 10) -> pd.DataFrame:
    """예측 PD로 기업을 n개 위험등급(1=가장 안전)으로 나누고 등급별 실제 부도율을 본다.

    신용등급 체계가 쓸모 있으려면 등급이 나빠질수록 실제 부도율이 꾸준히 올라가야 한다(단조성).
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    frame["grade"] = pd.qcut(frame["pd"].rank(method="first"), n_grades, labels=range(1, n_grades + 1))
    table = frame.groupby("grade", observed=True).agg(
        기업수=("y", "size"), 부도기업수=("y", "sum"), 평균예측PD=("pd", "mean"),
    )
    table["실제부도율"] = table["부도기업수"] / table["기업수"]
    return table[["기업수", "부도기업수", "실제부도율", "평균예측PD"]]


def is_monotonic(table: pd.DataFrame, col: str = "실제부도율") -> bool:
    return bool(table[col].is_monotonic_increasing)
