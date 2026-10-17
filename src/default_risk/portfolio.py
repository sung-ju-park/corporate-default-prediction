"""예측 PD로 예상손실(EL)을 계산한다.

EL = PD x LGD x EAD
 - PD : 부도확률 (모델 예측값)
 - LGD: 부도 시 손실률 (회수하지 못하는 비율)
 - EAD: 부도 시점의 노출 금액

여기서는 평가 데이터 기업마다 같은 금액을 빌려줬다고 가정한 가상 포트폴리오로,
모델이 예측한 EL과 실제로 부도가 난 기업에서 생긴 손실을 비교한다.
순위만 잘 맞히는 모델이 아니라 확률 자체가 맞아야 EL이 맞는다는 점을 보이기 위한 예시다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EAD_PER_FIRM, LGD


def expected_loss_by_grade(y_true, pd_hat, grades, lgd: float = LGD,
                           ead: float = EAD_PER_FIRM) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat),
                          "grade": np.asarray(grades)})
    frame["EL"] = frame["pd"] * lgd * ead
    frame["realized"] = frame["y"] * lgd * ead
    table = frame.groupby("grade", observed=True).agg(
        기업수=("y", "size"), 예상손실=("EL", "sum"), 실현손실=("realized", "sum"))
    table.loc["합계"] = table.sum()
    return table


def portfolio_totals(y_true, pd_hat, lgd: float = LGD, ead: float = EAD_PER_FIRM) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pd_hat = np.asarray(pd_hat)
    el = float((pd_hat * lgd * ead).sum())
    realized = float((y_true * lgd * ead).sum())
    return {"exposure": float(len(y_true) * ead), "expected_loss": el,
            "realized_loss": realized, "ratio": el / realized}
