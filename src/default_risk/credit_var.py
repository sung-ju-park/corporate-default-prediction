"""신용 VaR: 포트폴리오 손실 분포와 필요 자본.

예상손실(EL)은 '평균적으로' 얼마를 잃는지만 알려준다. 기업들의 부도는 경기라는 공통 요인 때문에
함께 몰려서 일어나므로, 나쁜 해에는 EL보다 훨씬 큰 손실이 날 수 있다.

바젤 내부등급법(IRB)의 바탕인 1요인 모형(Vasicek)으로 이를 계산한다.
  기업 i의 자산가치 = sqrt(rho) * Z + sqrt(1 - rho) * e_i
  Z  : 모든 기업이 함께 받는 경기 충격 (표준정규)
  e_i: 기업 고유 충격 (표준정규)
  자산가치가 Phi^-1(PD_i) 아래로 떨어지면 부도

- EL           : 손실 분포의 평균 → 충당금으로 대비
- VaR(99.9%)   : 1,000년에 한 번 수준의 나쁜 해의 손실
- UL = VaR - EL: 예상하지 못한 손실 → 자본으로 대비 (경제적 자본)
- ES(99.9%)    : VaR를 넘어서는 최악 0.1% 경우의 평균 손실
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

PD_FLOOR = 0.0003  # 바젤 II 기업 PD 하한 0.03%


def basel_correlation(pd_hat) -> np.ndarray:
    """바젤 기업 익스포저 자산상관계수. PD가 낮은(우량) 기업일수록 경기와 더 같이 움직인다고 본다(12%~24%)."""
    pd_hat = np.asarray(pd_hat, dtype=float)
    w = (1 - np.exp(-50 * pd_hat)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def conditional_pd(pd_hat, rho, z) -> np.ndarray:
    """경기 충격이 z일 때의 조건부 부도확률. z가 음수(불황)일수록 커진다."""
    return norm.cdf((norm.ppf(pd_hat) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def irb_capital(pd_hat, lgd: float, ead, q: float = 0.999) -> float:
    """바젤 IRB 공식으로 구한 필요 자본(UL)의 합. 만기 조정은 생략했다.

    K_i = LGD * [ Phi((Phi^-1(PD) + sqrt(rho) * Phi^-1(q)) / sqrt(1 - rho)) - PD ]
    """
    pd_hat = np.maximum(np.asarray(pd_hat, dtype=float), PD_FLOOR)
    rho = basel_correlation(pd_hat)
    stressed = conditional_pd(pd_hat, rho, norm.ppf(1 - q))
    k = lgd * (stressed - pd_hat)
    return float(np.sum(k * np.broadcast_to(ead, pd_hat.shape)))


def simulate_losses(pd_hat, lgd: float, ead, n_sims: int = 100_000, seed: int = 0,
                    chunk: int = 5_000) -> np.ndarray:
    """몬테카를로로 포트폴리오 손실 분포를 만든다. 반환값은 시뮬레이션별 총손실."""
    rng = np.random.default_rng(seed)
    pd_hat = np.maximum(np.asarray(pd_hat, dtype=float), PD_FLOOR)
    rho = basel_correlation(pd_hat)
    exposure = lgd * np.broadcast_to(np.asarray(ead, dtype=float), pd_hat.shape)
    threshold = norm.ppf(pd_hat)
    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    losses = np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        z = rng.standard_normal((m, 1))
        eps = rng.standard_normal((m, pd_hat.size))
        default = (a * z + b * eps) < threshold
        losses[start:start + m] = default @ exposure
    return losses


def loss_summary(losses: np.ndarray, levels=(0.99, 0.999)) -> dict[str, float]:
    out = {"EL": float(losses.mean())}
    for q in levels:
        var = float(np.quantile(losses, q))
        tag = f"{q * 100:g}"
        out[f"VaR {tag}%"] = var
        out[f"ES {tag}%"] = float(losses[losses >= var].mean())
    out["UL (VaR 99.9% - EL)"] = out["VaR 99.9%"] - out["EL"]
    return out
