"""결과 그림. 모든 그림은 같은 색 체계와 글꼴을 쓴다."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# 4·7번 그림(가중치 실험)은 모델이 아니라 학습 방식을 구분하므로 모델 색과 겹치지 않게 따로 둔다
BASELINE_GRAY, WARN_RED = "#898781", "#e34948"
MODEL_COLORS = {"Logit-10": AQUA, "Logit": ORANGE, "XGBoost": BLUE}

KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]


def setup_style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in KOREAN_FONTS if f in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 24, "axes.axisbelow": True, "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _subtitle(ax, text: str) -> None:
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                fontsize=9.5, color=INK_2, va="bottom")


def short(name: str, n: int = 42) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def plot_out_of_range(out_of_range: pd.Series, n_rows: int, path: Path) -> Path:
    s = (out_of_range[out_of_range > 0] / n_rows * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 0.27 * len(s) + 1.4))
    ax.barh([short(c) for c in s.index], s.values, color=BLUE, height=0.62)
    for i, v in enumerate(s.values):
        ax.text(v + 0.8, i, f"{v:.1f}%" if v >= 0.1 else f"{v:.2f}%", va="center",
                fontsize=8, color=INK_2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("범위(0~1)를 벗어난 값의 비율 (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("24개 변수에 정규화되지 않은 값이 섞여 있다")
    _subtitle(ax, "0~1과 60만 이상 사이에 값이 하나도 없어 같은 척도로 볼 수 없다 → 결측으로 처리")
    return _save(fig, path)


def plot_cv_comparison(cv_scores: pd.DataFrame, path: Path, metrics=("AUROC", "PR-AUC")) -> Path:
    models = list(dict.fromkeys(cv_scores["model"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.2, 2.9), sharey=True)
    for ax, metric in zip(axes, metrics):
        for i, m in enumerate(models):
            v = cv_scores.loc[cv_scores["model"] == m, metric].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(v))
            ax.scatter(v, i + jitter, s=12, color=MODEL_COLORS[m], alpha=0.35, linewidths=0)
            ax.errorbar(v.mean(), i, xerr=v.std(), fmt="o", ms=8, color=MODEL_COLORS[m],
                        ecolor=MODEL_COLORS[m], elinewidth=2, capsize=0,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(v.mean(), i + 0.3, f"{v.mean():.3f}", ha="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_xlabel(metric)
        ax.grid(axis="y", visible=False)
    axes[0].set_title("세 모델의 차이는 fold 간 흔들림보다 작다")
    _subtitle(axes[0], "5-fold × 3회 교차검증 · 큰 점 = 평균, 막대 = ±1 표준편차, 작은 점 = 각 fold")
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_pr(y_true, preds: dict, path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    base = float(np.mean(y_true))
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    a2.axhline(base, color=AXIS, lw=1, ls="--")
    a2.text(0.01, base + 0.015, f"무작위로 골랐을 때 {base:.1%}", ha="left", fontsize=8, color=MUTED)
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        prec, rec, _ = precision_recall_curve(y_true, p)
        a1.plot(fpr, tpr, color=MODEL_COLORS[name], label=name)
        a2.plot(rec, prec, color=MODEL_COLORS[name], label=name)
    a1.set(xlabel="정상 기업을 위험으로 잘못 본 비율 (FPR)", ylabel="부도 기업을 찾아낸 비율 (TPR)",
           xlim=(0, 1), ylim=(0, 1.01))
    a2.set(xlabel="재현율 (찾아낸 부도 기업 비율)", ylabel="정밀도 (위험 판정 중 실제 부도 비율)",
           xlim=(0, 1), ylim=(0, 1.01))
    a1.set_title("ROC 곡선")
    a2.set_title("정밀도-재현율 곡선")
    a1.legend(loc="lower right")
    fig.suptitle(f"평가 데이터 (부도 {int(np.sum(y_true))}개 / {len(y_true):,}개 기업)", x=0.01, ha="left", fontsize=9.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(bins: dict, path: Path) -> Path:
    """bins: {라벨: calibration_bins 결과}. 첫 번째가 기본 모델, 두 번째가 가중치 모델."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    top = max(max(b["predicted"].max(), b["observed"].max()) for b in bins.values()) * 1.08
    ax.plot([0, top], [0, top], color=AXIS, lw=1, ls="--")
    ax.text(top * 0.96, top * 0.9, "완벽하게 맞는 선", ha="right", fontsize=8, color=MUTED, rotation=38)
    for (label, b), color in zip(bins.items(), (BASELINE_GRAY, WARN_RED)):
        ax.plot(b["predicted"], b["observed"], color=color, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="평균 예측 PD (구간별)", ylabel="실제 부도율 (구간별)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.legend(loc="upper left")
    ax.set_title("부도 가중치를 주면 PD가 크게 부풀려진다")
    _subtitle(ax, "XGBoost · 학습 데이터 out-of-fold 예측을 10개 구간으로 나눔")
    return _save(fig, path)


def plot_grades(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(1, len(table) + 1)
    ax.bar(x, table["실제부도율"], color=BLUE, width=0.62, label="실제 부도율")
    ax.plot(x, table["평균예측PD"], color=ORANGE, marker="o", ms=7, lw=0,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="평균 예측 PD")
    for xi, top, (n_def, n) in zip(x, np.maximum(table["실제부도율"], table["평균예측PD"]),
                                  table[["부도기업수", "기업수"]].to_numpy()):
        ax.text(xi, top + 0.01, f"{int(n_def)}/{int(n)}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [str(g) for g in x])
    ax.set_xlabel("위험등급 (1 = 가장 안전, 10 = 가장 위험)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("부도는 가장 위험한 등급에 몰려 있다")
    _subtitle(ax, "XGBoost · 평가 데이터를 예측 PD 10분위로 나눔 · 막대 위 숫자 = 부도 기업 수 / 기업 수")
    return _save(fig, path)


def plot_shap_importance(importance: pd.DataFrame, path: Path) -> Path:
    """importance: index=변수, columns=[mean_abs, direction]. direction은 '높을수록 위험' 등."""
    s = importance.sort_values("mean_abs")
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(s) + 1.4))
    ax.barh([short(c, 46) for c in s.index], s["mean_abs"], color=BLUE, height=0.6)
    xmax = s["mean_abs"].max()
    for i, (v, d) in enumerate(zip(s["mean_abs"], s["direction"])):
        ax.text(v + xmax * 0.015, i, d, va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, xmax * 1.32)
    ax.set_xlabel("평균 |SHAP| (로그오즈 기준 영향 크기)")
    ax.grid(axis="y", visible=False)
    ax.set_title("부도 예측을 좌우하는 재무비율")
    _subtitle(ax, "XGBoost · 평가 데이터 · 오른쪽 글은 값이 어느 쪽일 때 부도 위험이 커지는지")
    return _save(fig, path)


def plot_expected_loss(totals: dict, realized: float, n_firms: int, path: Path) -> Path:
    labels = list(totals)
    vals = [totals[k] for k in labels]
    colors = [BASELINE_GRAY if "가중치" not in k else WARN_RED for k in labels]
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(labels) + 1.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.55)
    ax.axvline(realized, color=INK, lw=1.2)
    ax.text(realized, -0.62, f" 실제로 난 손실 {realized:.1f}억", fontsize=8.5, color=INK, va="center")
    ax.set_ylim(-0.9, len(labels) - 0.5)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f"{v:.1f}억 ({v / realized:.2f}배)", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("포트폴리오 예상손실 (억원)")
    ax.grid(axis="y", visible=False)
    ax.set_title("순위가 비슷해도 PD가 틀리면 예상손실이 틀린다")
    _subtitle(ax, f"평가 데이터 {n_firms:,}개 기업에 1억원씩 대출, LGD 45% 가정 · EL = PD × LGD × EAD")
    return _save(fig, path)


STATUS = {"적합": "#0ca30c", "주의": "#c98500", "경고": "#d03b3b"}


def _validation_panel(ax, table: pd.DataFrame, title: str) -> None:
    x = table.index.to_numpy()
    lo = np.array([v[0] for v in table["부도율 95% 구간"]])
    hi = np.array([v[1] for v in table["부도율 95% 구간"]])
    obs = table["실제부도율"].to_numpy()
    floor = 2e-4
    ax.vlines(x, np.maximum(lo, floor), hi, color=AXIS, lw=2.5, zorder=1)
    ax.scatter(x, table["평균예측PD"], marker="_", s=420, linewidths=2.5, color=BLUE, zorder=3,
               label="평균 예측 PD")
    shown = np.maximum(obs, floor)
    ax.scatter(x, shown, s=46, color=INK, edgecolors=SURFACE, linewidths=1.2, zorder=4,
               label="실제 부도율 (막대 = 95% 구간)")
    for xi, o, h, verdict, d in zip(x, obs, hi, table["판정"], table["부도수"]):
        ax.text(xi, h * 1.35, verdict, ha="center", fontsize=8.5, color=STATUS[verdict], fontweight="bold")
        if o == 0:
            ax.text(xi + 0.18, floor, "0건", fontsize=7.5, color=MUTED, va="center")
    ax.set_yscale("log")
    ax.set_ylim(1.2e-4, 2.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.set_xticks(x, [str(g) for g in x])
    ax.set_xlabel("마스터 스케일 등급 (1 = 가장 안전)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=11)


def plot_validation(before: pd.DataFrame, after: pd.DataFrame, hl_before: float, hl_after: float,
                    path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    _validation_panel(a1, before, f"보정 전 · 개발 데이터 (HL p = {hl_before:.3f})")
    _validation_panel(a2, after, f"보정 후 · 평가 데이터 (HL p = {hl_after:.2f})")
    a1.set_ylabel("부도율 (로그 축)")
    a1.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("중간 등급의 PD 과소추정을 찾아 보정했다", x=0.01, ha="left", fontsize=12.5,
                 fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.975, "등급별 Jeffreys 단측검정 · 적합 p ≥ 0.05, 주의 0.01~0.05, 경고 < 0.01 · "
             "HL = Hosmer-Lemeshow 전체 적합도 검정", fontsize=9.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


def plot_loss_distribution(losses: np.ndarray, summary: dict, exposure: float, path: Path,
                           unit: float = 0.45) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    # 손실은 기업 한 곳의 손실(unit = LGD x EAD) 단위로만 생긴다. 막대 폭을 그 배수로 맞춰야
    # 막대마다 들어가는 경우의 수가 같아져 분포가 들쭉날쭉하게 보이지 않는다.
    width = unit * 2
    bins = np.arange(-unit / 2, np.quantile(losses, 0.9995) + width, width)
    ax.hist(losses, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6, density=True)
    top = ax.get_ylim()[1]
    marks = [("EL", summary["EL"], INK_2), ("VaR 99.9%", summary["VaR 99.9%"], INK)]
    for name, v, c in marks:
        ax.axvline(v, color=c, lw=1.4)
        ax.text(v, top * 0.97, f" {name} {v:.1f}억", color=c, fontsize=9, va="top")
    ax.annotate("", xy=(summary["VaR 99.9%"], top * 0.62), xytext=(summary["EL"], top * 0.62),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.2))
    ax.text((summary["EL"] + summary["VaR 99.9%"]) / 2, top * 0.66,
            f"예상하지 못한 손실 {summary['UL (VaR 99.9% - EL)']:.1f}억 → 자본으로 대비",
            ha="center", fontsize=9, color=INK_2)
    ax.set_xlabel("1년 포트폴리오 손실 (억원)")
    ax.set_yticks([])
    ax.grid(visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"평균 손실은 {summary['EL']:.0f}억이지만 1,000년에 한 번 나쁜 해에는 {summary['VaR 99.9%']:.0f}억")
    _subtitle(ax, f"보정된 PD · 노출 {exposure:,.0f}억, LGD 45% · 바젤 1요인 모형 몬테카를로 10만 회")
    return _save(fig, path)


def plot_stress(table: pd.DataFrame, base_label: str, path: Path) -> Path:
    labels = list(table.index)
    vals = table["예상손실"].to_numpy()
    base = float(table.loc[base_label, "예상손실"])
    colors = [BASELINE_GRAY if k == base_label else BLUE for k in labels]
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(labels) + 1.9))
    # 경기 충격 시나리오와 재무비율 악화 시나리오 사이를 한 칸 띄운다
    gap = np.array([0.0 if "불황" in k or k == base_label else 0.7 for k in labels])
    y = (np.arange(len(labels)) + gap)[::-1].max() - (np.arange(len(labels)) + gap)
    ax.barh(y, vals, color=colors, height=0.55)
    for yi, v, pdm in zip(y, vals, table["평균PD"]):
        ax.text(v + vals.max() * 0.01, yi, f"{v:.1f}억 (×{v / base:.1f}) · 평균 PD {pdm:.1%}",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, vals.max() * 1.45)
    ax.set_xlabel("예상손실 (억원)")
    ax.grid(axis="y", visible=False)
    ax.set_title("불황과 재무 악화에서 손실은 몇 배로 커진다")
    _subtitle(ax, "보정된 PD · 위: 경기 충격 시나리오, 아래: 영향 큰 재무비율 5개를 위험한 쪽으로 이동")
    return _save(fig, path)
