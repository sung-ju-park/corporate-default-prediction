"""스트레스 테스트: 나쁜 상황을 가정하고 PD와 손실이 얼마나 커지는지 본다.

두 가지 방식으로 본다.
1. 경기 충격 시나리오: 1요인 모형의 경기 충격 Z를 'n년에 한 번' 수준으로 고정해 조건부 PD를 구한다.
   모든 기업에 같은 경기 충격이 오지만, 기업마다 PD와 상관계수가 달라 영향이 다르게 나타난다.
2. 재무비율 악화 시나리오: 부도 예측에 가장 큰 영향을 주는 재무비율을 위험한 방향으로 옮겨
   모델에 다시 넣는다. 모델이 '재무제표가 나빠지면' 어떻게 반응하는지 보는 민감도 분석이다.
   (옮기는 크기는 가정이며, 실제 경기침체 때 재무비율 변화를 추정한 값은 아니다.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .credit_var import PD_FLOOR, basel_correlation, conditional_pd


def systemic_scenarios(pd_hat, lgd: float, ead, years=(10, 25, 100, 1000)) -> pd.DataFrame:
    pd_hat = np.maximum(np.asarray(pd_hat, dtype=float), PD_FLOOR)
    rho = basel_correlation(pd_hat)
    exposure = lgd * np.broadcast_to(np.asarray(ead, dtype=float), pd_hat.shape)
    rows = [{"시나리오": "기준 (평상시)", "평균PD": float(pd_hat.mean()),
             "예상손실": float(pd_hat @ exposure)}]
    for n in years:
        z = norm.ppf(1 / n)
        cpd = conditional_pd(pd_hat, rho, z)
        rows.append({"시나리오": f"{n}년에 한 번 불황", "평균PD": float(cpd.mean()),
                     "예상손실": float(cpd @ exposure)})
    return pd.DataFrame(rows).set_index("시나리오")


def shift_quantiles(X: pd.DataFrame, reference: pd.DataFrame, directions: dict[str, int],
                    shift: float) -> pd.DataFrame:
    """각 변수의 값을 기준 분포(학습 데이터)에서 shift 만큼 위험한 쪽 분위로 옮긴다.

    directions: {변수: +1(높을수록 위험) 또는 -1(낮을수록 위험)}
    예) shift=0.1이면 당좌비율이 학습 데이터 기준 50번째 백분위인 기업은 40번째 백분위 값이 된다.
    """
    out = X.copy()
    for col, sign in directions.items():
        ref = np.sort(reference[col].dropna().to_numpy())
        vals = X[col].to_numpy(dtype=float)
        ok = ~np.isnan(vals)
        rank = np.searchsorted(ref, vals[ok], side="right") / len(ref)
        new_rank = np.clip(rank + sign * shift, 0, 1)
        new_vals = vals.copy()
        new_vals[ok] = np.quantile(ref, new_rank)
        out[col] = new_vals
    return out
