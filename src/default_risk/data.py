"""데이터 불러오기와 정제.

원본 데이터(UCI Taiwanese Bankruptcy Prediction)는 95개 재무비율이 0~1로 정규화돼 있다고
설명되지만, 실제로는 24개 변수에 정규화되지 않은 값(약 60만~100억)이 섞여 있다.
이 값들은 0~1 구간과 사이가 완전히 비어 있어 같은 척도로 볼 수 없으므로 결측으로 처리한다.

정제 규칙은 모두 라벨(부도 여부)을 보지 않고 정해지므로, 전체 데이터에 적용해도
학습/평가 데이터 사이에 정보가 새지 않는다. 결측 대체값(중앙값)은 모델 파이프라인 안에서
학습 데이터로만 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_PATH, TARGET

RAW_TARGET = "Bankrupt?"
VALID_MAX = 1.0  # 정규화된 비율의 상한


@dataclass
class CleaningReport:
    n_rows: int
    n_raw_features: int
    out_of_range: pd.Series                      # 변수별 범위 밖 값의 개수
    dropped_constant: list[str] = field(default_factory=list)
    dropped_duplicate: dict[str, str] = field(default_factory=dict)  # 제거한 변수 -> 남긴 짝

    @property
    def n_features(self) -> int:
        return self.n_raw_features - len(self.dropped_constant) - len(self.dropped_duplicate)

    def summary(self) -> dict:
        oor = self.out_of_range[self.out_of_range > 0]
        return {
            "rows": self.n_rows,
            "raw_features": self.n_raw_features,
            "features_after_cleaning": self.n_features,
            "columns_with_out_of_range_values": int(len(oor)),
            "out_of_range_cells": int(oor.sum()),
            "dropped_constant": self.dropped_constant,
            "dropped_duplicate": self.dropped_duplicate,
        }


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """원본 CSV를 읽고 열 이름 앞뒤 공백을 지운다. 타깃 열 이름은 `bankrupt`로 바꾼다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. `python scripts/download_data.py` 로 데이터를 먼저 받아 주세요."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={RAW_TARGET: TARGET})


def find_out_of_range(X: pd.DataFrame, upper: float = VALID_MAX) -> pd.DataFrame:
    """정규화 범위(0~1)를 벗어난 셀을 True로 표시한 마스크."""
    return (X > upper) | (X < 0)


def find_exact_duplicates(X: pd.DataFrame, tol: float = 1e-9) -> dict[str, str]:
    """값이 완전히 같거나 1에서 뺀 값과 같은(예: 부채비율 vs 자기자본비율) 변수 쌍을 찾는다.

    반환값은 {제거할 변수: 남길 변수}. 앞쪽 열을 남긴다.
    """
    cols = list(X.columns)
    values = X.to_numpy(dtype=float)
    drop: dict[str, str] = {}
    for i, a in enumerate(cols):
        if a in drop:
            continue
        for j in range(i + 1, len(cols)):
            b = cols[j]
            if b in drop:
                continue
            same = np.nanmax(np.abs(values[:, i] - values[:, j])) < tol
            complement = np.nanmax(np.abs(values[:, i] + values[:, j] - 1)) < tol
            if same or complement:
                drop[b] = a
    return drop


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    """범위 밖 값 -> 결측, 상수 변수 제거, 중복 변수 제거."""
    y = df[TARGET].astype(int)
    X = df.drop(columns=TARGET).astype(float)

    mask = find_out_of_range(X)
    report = CleaningReport(n_rows=len(X), n_raw_features=X.shape[1], out_of_range=mask.sum())
    X = X.mask(mask)

    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant)
    report.dropped_constant = constant

    duplicates = find_exact_duplicates(X)
    X = X.drop(columns=list(duplicates))
    report.dropped_duplicate = duplicates
    return X, y, report


def range_flags(X_raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """민감도 실험용: 원래 값이 범위 밖이었는지를 0/1 변수로 만든다."""
    flags = find_out_of_range(X_raw[columns]).astype(int)
    flags.columns = [f"{c} (범위밖)" for c in columns]
    return flags
