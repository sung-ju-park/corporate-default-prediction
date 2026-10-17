"""신용 VaR, 스트레스, 검증 모듈 테스트: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from default_risk import credit_var, stress, validation  # noqa: E402


def test_basel_correlation_range():
    rho = credit_var.basel_correlation(np.array([1e-6, 0.01, 0.5]))
    assert rho[0] == pytest.approx(0.24, abs=1e-3)   # 우량 기업일수록 경기와 함께 움직임
    assert rho[2] == pytest.approx(0.12, abs=1e-3)
    assert rho[0] > rho[1] > rho[2]


def test_conditional_pd_averages_back_to_pd():
    # 경기 충격 Z를 적분하면 원래 PD가 나와야 한다 (전확률 법칙)
    pd_hat, rho = 0.02, 0.2
    z = np.linspace(-8, 8, 4001)
    w = norm.pdf(z) * (z[1] - z[0])
    assert np.sum(credit_var.conditional_pd(pd_hat, rho, z) * w) == pytest.approx(pd_hat, rel=1e-4)


def test_conditional_pd_rises_in_bad_times():
    assert credit_var.conditional_pd(0.02, 0.2, -2.0) > 0.02 > credit_var.conditional_pd(0.02, 0.2, 2.0)


def test_monte_carlo_matches_irb_formula_for_granular_portfolio():
    # 같은 PD의 기업이 많으면 몬테카를로 UL이 바젤 공식(무한히 잘게 나뉜 포트폴리오 가정)에 가까워야 한다
    pd_hat = np.full(2000, 0.02)
    losses = credit_var.simulate_losses(pd_hat, lgd=0.45, ead=1.0, n_sims=40_000, seed=1)
    summary = credit_var.loss_summary(losses)
    assert summary["EL"] == pytest.approx(2000 * 0.02 * 0.45, rel=0.02)
    assert summary["UL (VaR 99.9% - EL)"] == pytest.approx(
        credit_var.irb_capital(pd_hat, 0.45, 1.0), rel=0.08)


def test_systemic_scenarios_increase_with_severity():
    table = stress.systemic_scenarios(np.full(100, 0.01), lgd=0.45, ead=1.0)
    assert table["예상손실"].is_monotonic_increasing


def test_shift_quantiles_moves_in_risky_direction():
    ref = pd.DataFrame({"a": np.linspace(0, 1, 101), "b": np.linspace(0, 1, 101)})
    X = pd.DataFrame({"a": [0.5, np.nan], "b": [0.5, 0.2]})
    out = stress.shift_quantiles(X, ref, {"a": +1, "b": -1}, 0.1)
    assert out.loc[0, "a"] > 0.5 and out.loc[0, "b"] < 0.5
    assert np.isnan(out.loc[1, "a"])  # 결측은 그대로


def test_binomial_and_jeffreys_flag_underestimation():
    assert validation.binomial_pvalue(0, 100, 0.01) == pytest.approx(1.0)
    assert validation.binomial_pvalue(10, 100, 0.01) < 0.001
    assert validation.jeffreys_pvalue(10, 100, 0.01) < 0.001
    assert validation.traffic_light(0.2) == "적합"
    assert validation.traffic_light(0.03) == "주의"
    assert validation.traffic_light(0.001) == "경고"


def test_master_scale_grades():
    grades = validation.assign_grade([0.0, 0.0009, 0.001, 0.03, 0.99, 1.0])
    assert list(grades) == [1, 1, 2, 6, 9, 9]


def test_psi_zero_for_identical_distributions():
    p = np.linspace(0.0001, 0.9, 500)
    assert validation.psi(p, p) == pytest.approx(0.0, abs=1e-12)


def test_platt_calibration_keeps_order():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.5, 2000)
    y = rng.uniform(size=2000) < p * 1.5
    cal = validation.PlattCalibrator().fit(p, y)
    out = cal.transform(p)
    assert np.all(np.diff(out[np.argsort(p)]) >= -1e-12)
    assert out.mean() == pytest.approx(y.mean(), rel=0.05)
