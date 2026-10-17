"""모형 검증: 등급 체계, PD 정확성 검정, 안정성, 재보정.

금융권 신용평가모형 검증은 보통 세 가지를 본다.
1. 변별력: 위험한 기업을 위에 세우는가 (AUROC, AR, KS → metrics.py)
2. 정확성(보정): 등급별 예측 PD가 실제 부도율과 맞는가 (이항검정, Jeffreys 검정, Hosmer-Lemeshow)
3. 안정성: 점수 분포가 개발 시점과 달라지지 않았는가 (PSI)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, chi2
from sklearn.linear_model import LogisticRegression

# 마스터 스케일: 예측 PD 구간으로 정한 고정 등급 체계 (1 = 가장 안전).
# 위험할수록 구간을 넓게 잡아(로그 간격) 저위험 기업이 여러 등급에 흩어지지 않게 했다.
MASTER_SCALE = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.15, 0.40, 1.0 + 1e-12]
GRADE_LABELS = list(range(1, len(MASTER_SCALE)))

GREEN, YELLOW, RED = "적합", "주의", "경고"


def assign_grade(pd_hat) -> np.ndarray:
    return pd.cut(np.asarray(pd_hat), MASTER_SCALE, right=False, labels=GRADE_LABELS).astype(int)


def grade_ranges() -> list[str]:
    def f(v):
        return f"{v * 100:g}%"
    return [f"{f(lo)} ~ {f(min(hi, 1.0))}" for lo, hi in zip(MASTER_SCALE[:-1], MASTER_SCALE[1:])]


def binomial_pvalue(defaults: int, n: int, pd_hat: float) -> float:
    """예측 PD가 맞다면 이만큼 이상 부도가 날 확률 (단측). 작을수록 PD 과소추정 의심."""
    return float(binom.sf(defaults - 1, n, pd_hat))


def jeffreys_pvalue(defaults: int, n: int, pd_hat: float) -> float:
    """Jeffreys 사전분포 Beta(0.5, 0.5)를 쓴 단측 검정. 표본이 작을 때 이항검정보다 덜 보수적이다.

    실제 부도율의 사후분포 Beta(d + 0.5, n - d + 0.5)에서 '실제 부도율이 예측 PD 이하'일 확률.
    """
    return float(beta.cdf(pd_hat, defaults + 0.5, n - defaults + 0.5))


def jeffreys_interval(defaults: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return (float(beta.ppf(a, defaults + 0.5, n - defaults + 0.5)),
            float(beta.ppf(1 - a, defaults + 0.5, n - defaults + 0.5)))


def traffic_light(p: float) -> str:
    return GREEN if p >= 0.05 else YELLOW if p >= 0.01 else RED


def grade_validation(y_true, pd_hat) -> pd.DataFrame:
    """마스터 스케일 등급별로 예측 PD와 실제 부도율을 비교하고 검정한다."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    frame["grade"] = assign_grade(frame["pd"])
    ranges = dict(zip(GRADE_LABELS, grade_ranges()))
    rows = []
    for g in GRADE_LABELS:
        sub = frame[frame["grade"] == g]
        n, d = len(sub), int(sub["y"].sum())
        if n == 0:
            continue
        avg_pd = float(sub["pd"].mean())
        p_jef = jeffreys_pvalue(d, n, avg_pd)
        lo, hi = jeffreys_interval(d, n)
        rows.append({"등급": g, "PD 구간": ranges[g], "기업수": n, "부도수": d,
                     "평균예측PD": avg_pd, "실제부도율": d / n, "부도율 95% 구간": (lo, hi),
                     "이항검정 p": binomial_pvalue(d, n, avg_pd), "Jeffreys p": p_jef,
                     "판정": traffic_light(p_jef)})
    return pd.DataFrame(rows).set_index("등급")


def hosmer_lemeshow(y_true, pd_hat, groups: int = 10) -> dict[str, float]:
    """예측 PD 10분위별 기대 부도수와 실제 부도수 차이로 전체 보정을 검정한다. p가 크면 보정 양호."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    frame["g"] = pd.qcut(frame["pd"].rank(method="first"), groups, labels=False)
    agg = frame.groupby("g").agg(obs=("y", "sum"), exp=("pd", "sum"), n=("y", "size"))
    stat = float((((agg["obs"] - agg["exp"]) ** 2) / (agg["exp"] * (1 - agg["exp"] / agg["n"]))).sum())
    return {"statistic": stat, "df": groups - 2, "p_value": float(chi2.sf(stat, groups - 2))}


def psi(expected_pd, actual_pd) -> float:
    """Population Stability Index. 마스터 스케일 등급 분포로 계산한다.

    실무 기준: 0.1 미만 안정, 0.1~0.25 관찰 필요, 0.25 이상 큰 변화.
    """
    e = pd.Series(assign_grade(expected_pd)).value_counts(normalize=True).reindex(GRADE_LABELS, fill_value=0)
    a = pd.Series(assign_grade(actual_pd)).value_counts(normalize=True).reindex(GRADE_LABELS, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


class PlattCalibrator:
    """예측 PD의 로짓에 로지스틱 회귀를 한 번 더 적합해 PD 수준을 맞춘다 (파라미터 2개).

    순위는 바꾸지 않고(단조 변환) 확률의 높낮이만 고친다. 학습 데이터의 out-of-fold 예측으로
    적합하므로 평가 데이터 정보는 쓰지 않는다.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, pd_hat, y_true) -> "PlattCalibrator":
        self.model.fit(self._logit(pd_hat), np.asarray(y_true))
        return self

    def transform(self, pd_hat) -> np.ndarray:
        return self.model.predict_proba(self._logit(pd_hat))[:, 1]

    @property
    def params(self) -> dict[str, float]:
        return {"slope": float(self.model.coef_[0, 0]), "intercept": float(self.model.intercept_[0])}
