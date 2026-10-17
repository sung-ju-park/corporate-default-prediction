"""핵심 함수 단위 테스트: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from default_risk import data, metrics, portfolio  # noqa: E402


def test_ks_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.ks_statistic(y, np.full(6, 0.5)) == pytest.approx(0.0)


def test_ar_is_twice_auc_minus_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = 1 / (1 + np.exp(-(y + rng.normal(0, 1, 500))))  # 0~1 확률로 변환
    m = metrics.evaluate(y, s)
    assert m["AR"] == pytest.approx(2 * m["AUROC"] - 1)


def test_recall_at_top():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.8, 0, 0, 0, 0, 0, 0, 0])
    assert metrics.recall_at_top(y, s, 0.1) == pytest.approx(0.5)  # 상위 1개 안에 부도 1/2


def test_grade_table_counts_and_monotonic():
    y = np.array([0] * 90 + [1] * 10)
    pd_hat = np.linspace(0, 1, 100)
    table = metrics.grade_table(y, pd_hat, n_grades=10)
    assert table["기업수"].sum() == 100
    assert table["부도기업수"].sum() == 10
    assert table.loc[10, "실제부도율"] == pytest.approx(1.0)
    assert metrics.is_monotonic(table)


def test_clean_masks_out_of_range_and_drops_redundant_columns():
    df = pd.DataFrame({
        "bankrupt": [0, 1, 0, 1],
        "ratio": [0.2, 0.5, 3.0e9, 0.1],      # 범위 밖 값 1개
        "debt": [0.3, 0.6, 0.2, 0.9],
        "equity": [0.7, 0.4, 0.8, 0.1],       # = 1 - debt -> 중복
        "flag": [0, 0, 0, 0],                 # 상수
    })
    X, y, report = data.clean(df)
    assert np.isnan(X.loc[2, "ratio"])
    assert "equity" not in X and "flag" not in X
    assert report.dropped_duplicate == {"equity": "debt"}
    assert report.out_of_range["ratio"] == 1
    assert list(y) == [0, 1, 0, 1]


def test_expected_loss_formula():
    totals = portfolio.portfolio_totals([0, 1], [0.1, 0.5], lgd=0.45, ead=10)
    assert totals["expected_loss"] == pytest.approx((0.1 + 0.5) * 0.45 * 10)
    assert totals["realized_loss"] == pytest.approx(1 * 0.45 * 10)
