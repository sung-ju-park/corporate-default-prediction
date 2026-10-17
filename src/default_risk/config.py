"""프로젝트 전역 설정: 경로, 난수 시드, 분할 비율, 손실 가정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "data.csv"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

TARGET = "bankrupt"
SEED = 42
TEST_SIZE = 0.2          # 학습 80% / 최종 평가 20% (층화 분할)
CV_FOLDS = 5
CV_REPEATS = 3           # 5-fold를 3번 반복해 평균과 편차를 본다

# 예상손실(EL = PD x LGD x EAD) 예시에 쓰는 가정
LGD = 0.45               # 바젤 II 기초 내부등급법의 무담보 선순위 채권 LGD
EAD_PER_FIRM = 1.0       # 기업당 대출 1억원으로 가정한 가상 포트폴리오 (단위: 억원)

N_GRADES = 10            # 위험등급 수 (예측 PD 기준 10분위)
