"""전체 분석을 처음부터 끝까지 실행한다.

    python run_pipeline.py

결과: reports/metrics.json, reports/results.md, reports/figures/*.png
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from default_risk import (config, credit_var, data, evaluation, metrics, models, plotting,  # noqa: E402
                          portfolio, stress, validation)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 교차검증으로 고른 하이퍼파라미터 (notebooks/analysis.ipynb의 튜닝 절 참고)
PARAMS = {
    "Logit-10": {"n_features": 10, "select_C": 0.05, "C": 0.1},
    "Logit": {"C": 0.1},
    "XGBoost": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def df_to_md(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    frame = frame.copy()
    for c in frame.columns:
        if pd.api.types.is_float_dtype(frame[c]):
            frame[c] = frame[c].map(lambda v: format(v, floatfmt))
    header = "| " + " | ".join([str(frame.index.name or "")] + [str(c) for c in frame.columns]) + " |"
    sep = "|" + "---|" * (len(frame.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(frame.index, frame.values)]
    return "\n".join([header, sep, *rows])


def shap_importance(model, X: pd.DataFrame, top: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:top]
    rows = []
    for j in order:
        col = X.columns[j]
        ok = X[col].notna().to_numpy()
        rho = spearmanr(X[col].to_numpy()[ok], sv[ok, j]).statistic
        direction = "값이 높을수록 위험" if rho > 0.3 else "값이 낮을수록 위험" if rho < -0.3 else "방향이 일정하지 않음"
        rows.append({"feature": col, "mean_abs": float(mean_abs[j]), "spearman": float(rho),
                     "direction": direction})
    return pd.DataFrame(rows).set_index("feature"), sv


def main() -> None:
    plotting.setup_style()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    md: list[str] = ["# 분석 결과 (run_pipeline.py가 자동 생성)\n"]

    # 1. 데이터 정제 --------------------------------------------------------------
    log("데이터 불러오기와 정제")
    raw = data.load_raw()
    X, y, report = data.clean(raw)
    out["data"] = {**report.summary(), "default_rate": float(y.mean()), "n_defaults": int(y.sum())}
    plotting.plot_out_of_range(report.out_of_range, report.n_rows, config.FIG_DIR / "01_out_of_range.png")
    oor = report.out_of_range[report.out_of_range > 0].sort_values(ascending=False)
    md += ["## 1. 데이터", f"- 기업 {len(y):,}개, 부도 {int(y.sum())}개 (부도율 {y.mean():.2%})",
           f"- 범위를 벗어난 값이 있는 변수 {len(oor)}개, 셀 {int(oor.sum()):,}개 → 결측 처리",
           f"- 제거: 상수 변수 {report.dropped_constant}, 중복 변수 {report.dropped_duplicate}",
           f"- 정제 후 변수 {X.shape[1]}개\n"]

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    out["split"] = {"train": len(tr), "train_defaults": int(ytr.sum()),
                    "test": len(te), "test_defaults": int(yte.sum())}

    # 2. 교차검증으로 세 모델 비교 ------------------------------------------------------
    log("교차검증 (5-fold x 3회)")
    model_set = models.build_models(PARAMS)
    cv_scores = evaluation.cross_validate(model_set, Xtr, ytr)
    cv_summary = evaluation.summarize_cv(cv_scores)
    out["cv"] = {m: {k: {"mean": float(cv_summary.loc[m, (k, "mean")]), "std": float(cv_summary.loc[m, (k, "std")])}
                     for k in ["AUROC", "AR", "KS", "PR-AUC", "Brier", "Recall@10%"]} for m in cv_summary.index}
    plotting.plot_cv_comparison(cv_scores, config.FIG_DIR / "02_cv_comparison.png")
    cv_table = pd.DataFrame({k: [f"{cv_summary.loc[m, (k, 'mean')]:.3f} ± {cv_summary.loc[m, (k, 'std')]:.3f}"
                                 for m in cv_summary.index] for k in ["AUROC", "AR", "KS", "PR-AUC", "Recall@10%"]},
                            index=pd.Index(cv_summary.index, name="모델"))
    md += ["## 2. 교차검증 (학습 데이터, 5-fold × 3회, 평균 ± 표준편차)", df_to_md(cv_table), ""]

    # 3. 민감도 실험: 범위 밖 값 처리 방식 ------------------------------------------------
    log("민감도 실험: 범위 밖 값 처리")
    raw_X = raw.drop(columns=config.TARGET)
    oor_cols = [c for c in oor.index if c in X.columns]
    variants = {
        "원래 값 그대로": raw_X[X.columns],
        "결측 처리 (채택)": X,
        "결측 처리 + 범위밖 표시 변수": pd.concat([X, data.range_flags(raw_X, oor_cols)], axis=1),
        "범위밖 표시 변수만": data.range_flags(raw_X, list(oor.index)),
    }
    sens = {}
    sens_models = {"Logit": models.make_logit(**PARAMS["Logit"]), "XGBoost": models.make_xgb(**PARAMS["XGBoost"])}
    for name, XX in variants.items():
        s = evaluation.cross_validate(sens_models, XX.iloc[tr], ytr)
        sens[name] = s.groupby("model")["AUROC"].mean().to_dict()
    out["sensitivity_out_of_range"] = sens
    sens_df = pd.DataFrame(sens).T[["Logit", "XGBoost"]]
    sens_df.index.name = "처리 방식"
    md += ["## 3. 범위 밖 값 처리 방식별 교차검증 AUROC", df_to_md(sens_df), ""]

    # 4. 보정 실험: 부도 가중치를 주면 PD가 어떻게 되나 -------------------------------------
    log("보정 실험: 클래스 가중치")
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    weighting = {
        "XGBoost": models.make_xgb(**PARAMS["XGBoost"]),
        "XGBoost (부도 가중치)": models.make_xgb(scale_pos_weight=spw, **PARAMS["XGBoost"]),
        "Logit": models.make_logit(**PARAMS["Logit"]),
        "Logit (부도 가중치)": models.make_logit(balanced=True, **PARAMS["Logit"]),
    }
    oof = {k: evaluation.out_of_fold(m, Xtr, ytr) for k, m in weighting.items()}
    calib = {}
    for k, p in oof.items():
        calib[k] = {**evaluation.calibration_summary(ytr, p), **metrics.evaluate(ytr, p)}
    out["calibration_oof"] = calib
    plotting.plot_calibration(
        {"가중치 없음": evaluation.calibration_bins(ytr, oof["XGBoost"]),
         "부도 가중치 (scale_pos_weight)": evaluation.calibration_bins(ytr, oof["XGBoost (부도 가중치)"])},
        config.FIG_DIR / "04_calibration.png")
    calib_df = pd.DataFrame(calib).T[["AUROC", "Brier", "mean_predicted_pd", "observed_default_rate", "ratio"]]
    calib_df.columns = ["AUROC", "Brier", "평균예측PD", "실제부도율", "배율"]
    calib_df.index.name = "모델"
    md += ["## 4. 부도 가중치와 PD 보정 (학습 데이터 out-of-fold)", df_to_md(calib_df, ".4f"), ""]

    # 5. 최종 학습과 평가 데이터 검증 ------------------------------------------------------
    log("최종 학습과 평가 데이터 검증")
    fitted = {k: clone(m).fit(Xtr, ytr) for k, m in model_set.items()}
    preds = {k: m.predict_proba(Xte)[:, 1] for k, m in fitted.items()}
    test = {}
    for k, p in preds.items():
        lo, hi = metrics.bootstrap_ci(yte, p)
        test[k] = {**metrics.evaluate(yte, p), "AUROC_CI": [lo, hi]}
    out["test"] = test
    out["test_diff"] = {
        "XGBoost - Logit-10": metrics.paired_bootstrap_diff(yte, preds["XGBoost"], preds["Logit-10"]),
        "XGBoost - Logit": metrics.paired_bootstrap_diff(yte, preds["XGBoost"], preds["Logit"]),
    }
    plotting.plot_roc_pr(yte, preds, config.FIG_DIR / "03_roc_pr.png")
    test_df = pd.DataFrame({k: {"AUROC": f"{v['AUROC']:.3f}",
                                "AUROC 95% CI": f"{v['AUROC_CI'][0]:.3f} ~ {v['AUROC_CI'][1]:.3f}",
                                "AR": f"{v['AR']:.3f}", "KS": f"{v['KS']:.3f}", "PR-AUC": f"{v['PR-AUC']:.3f}",
                                "Brier": f"{v['Brier']:.4f}", "Recall@10%": f"{v['Recall@10%']:.3f}"}
                            for k, v in test.items()}).T
    test_df.index.name = "모델"
    diff_lines = [f"- {k}: AUROC 차이 {v['diff']:+.3f} (95% CI {v['ci_low']:+.3f} ~ {v['ci_high']:+.3f})"
                  for k, v in out["test_diff"].items()]
    md += ["## 5. 평가 데이터 (학습에 쓰지 않은 20%)", df_to_md(test_df), "", *diff_lines, ""]

    # 6. 위험등급표 ------------------------------------------------------------------
    log("위험등급표")
    grades = metrics.grade_table(yte, preds["XGBoost"], config.N_GRADES)
    out["grades"] = {"table": grades.reset_index().to_dict(orient="records"),
                     "monotonic": metrics.is_monotonic(grades),
                     "share_of_defaults_in_top2": float(grades["부도기업수"].iloc[-2:].sum() / grades["부도기업수"].sum())}
    plotting.plot_grades(grades, config.FIG_DIR / "05_risk_grades.png")
    g = grades.copy()
    g.index.name = "등급"
    g["실제부도율"] = g["실제부도율"].map(lambda v: f"{v:.1%}")
    g["평균예측PD"] = g["평균예측PD"].map(lambda v: f"{v:.2%}")
    md += ["## 6. XGBoost 위험등급표 (평가 데이터, 예측 PD 10분위)", df_to_md(g), ""]

    # 7. 설명: SHAP과 Logit-10이 고른 변수 ------------------------------------------------
    log("SHAP")
    imp, sv = shap_importance(fitted["XGBoost"], Xte, top=12)
    out["shap_top"] = imp.reset_index().to_dict(orient="records")
    plotting.plot_shap_importance(imp, config.FIG_DIR / "06_shap_importance.png")
    lr10 = fitted["Logit-10"]
    selected = list(Xtr.columns[lr10.named_steps["select"].get_support()])
    coefs = pd.Series(lr10.named_steps["model"].coef_[0], index=selected).sort_values(key=np.abs, ascending=False)
    overlap = sorted(set(selected) & set(imp.index[:10]))
    out["logit10"] = {"features": coefs.to_dict(), "overlap_with_shap_top10": overlap}
    coef_df = pd.DataFrame({"계수": coefs.round(3), "방향": np.where(coefs.abs() < 0.1, "영향 미미",
                                                     np.where(coefs > 0, "높을수록 위험", "낮을수록 위험"))})
    coef_df.index.name = "Logit-10 변수"
    imp_md = imp[["mean_abs", "direction"]].rename(columns={"mean_abs": "평균 |SHAP|", "direction": "방향"})
    imp_md.index.name = "XGBoost 상위 변수"
    md += ["## 7. 설명", df_to_md(imp_md), "", df_to_md(coef_df),
           f"\nXGBoost SHAP 상위 10개와 Logit-10 변수의 겹침: {len(overlap)}개 — {overlap}\n"]

    # 8. 놓친 부도 기업 -----------------------------------------------------------------
    p = preds["XGBoost"]
    cutoff = np.quantile(p, 0.90)
    yv = yte.to_numpy()
    caught, missed = (yv == 1) & (p >= cutoff), (yv == 1) & (p < cutoff)
    top3 = list(imp.index[:3])
    miss_df = pd.DataFrame({
        "잡아낸 부도": Xte.loc[caught, top3].median(),
        "놓친 부도": Xte.loc[missed, top3].median(),
        "정상 기업": Xte.loc[yv == 0, top3].median(),
    })
    miss_df.index.name = "변수 (중앙값)"
    out["missed"] = {"cutoff_pd": float(cutoff), "caught": int(caught.sum()), "missed": int(missed.sum()),
                     "medians": miss_df.to_dict()}
    md += ["## 8. 상위 10% 기준에서 놓친 부도 기업",
           f"- 잡아냄 {int(caught.sum())}개, 놓침 {int(missed.sum())}개 (기준 PD {cutoff:.2%})",
           df_to_md(miss_df, ".4f"), ""]

    # 9. 예상손실 -------------------------------------------------------------------
    log("예상손실")
    fitted_w = {k: clone(m).fit(Xtr, ytr) for k, m in weighting.items() if k not in fitted}
    all_preds = {"XGBoost": preds["XGBoost"], "XGBoost (부도 가중치)": fitted_w["XGBoost (부도 가중치)"].predict_proba(Xte)[:, 1],
                 "Logit": preds["Logit"], "Logit (부도 가중치)": fitted_w["Logit (부도 가중치)"].predict_proba(Xte)[:, 1]}
    totals = {k: portfolio.portfolio_totals(yte, v) for k, v in all_preds.items()}
    out["expected_loss"] = totals
    realized = totals["XGBoost"]["realized_loss"]
    plotting.plot_expected_loss({k: v["expected_loss"] for k, v in totals.items()}, realized, len(yte),
                                config.FIG_DIR / "07_expected_loss.png")
    el_by_grade = portfolio.expected_loss_by_grade(yte, preds["XGBoost"],
                                                   pd.qcut(pd.Series(preds["XGBoost"]).rank(method="first"),
                                                           config.N_GRADES, labels=range(1, config.N_GRADES + 1)))
    el_df = pd.DataFrame({k: {"예상손실(억)": f"{v['expected_loss']:.1f}", "실현손실(억)": f"{v['realized_loss']:.1f}",
                              "배율": f"{v['ratio']:.2f}"} for k, v in totals.items()}).T
    el_df.index.name = "모델"
    el_by_grade.index.name = "등급"
    el_by_grade["기업수"] = el_by_grade["기업수"].astype(int)
    md += ["## 9. 가상 포트폴리오 예상손실 (기업당 1억원, LGD 45%)", df_to_md(el_df), "",
           "XGBoost 등급별:", df_to_md(el_by_grade, ".2f"), ""]

    # 10. 모형 검증과 재보정 ------------------------------------------------------------
    log("모형 검증과 재보정")
    xgb = model_set["XGBoost"]
    oof_xgb = oof["XGBoost"]  # 4절에서 같은 설정·같은 fold로 만든 out-of-fold 예측
    calibrator = validation.PlattCalibrator().fit(oof_xgb, ytr)
    pd_final = calibrator.transform(preds["XGBoost"])
    checks = {
        "개발 데이터 · 보정 전": (ytr, oof_xgb),
        "평가 데이터 · 보정 전": (yte, preds["XGBoost"]),
        "평가 데이터 · 보정 후": (yte, pd_final),
    }
    val_tables, val_out = {}, {}
    for k, (yy, pp) in checks.items():
        t = validation.grade_validation(yy, pp)
        hl = validation.hosmer_lemeshow(yy, pp)
        val_tables[k] = (t, hl)
        val_out[k] = {"hosmer_lemeshow": hl, "mean_pd": float(np.mean(pp)), "default_rate": float(np.mean(yy)),
                      "verdicts": t["판정"].value_counts().to_dict(),
                      "table": t.drop(columns=["부도율 95% 구간"]).reset_index().to_dict(orient="records")}
    val_out["calibrator"] = calibrator.params
    val_out["psi_dev_vs_test"] = validation.psi(calibrator.transform(oof_xgb), pd_final)
    val_out["auroc_after_calibration"] = float(metrics.evaluate(yte, pd_final)["AUROC"])
    out["validation"] = val_out
    plotting.plot_validation(val_tables["개발 데이터 · 보정 전"][0], val_tables["평가 데이터 · 보정 후"][0],
                             val_tables["개발 데이터 · 보정 전"][1]["p_value"],
                             val_tables["평가 데이터 · 보정 후"][1]["p_value"],
                             config.FIG_DIR / "08_validation.png")
    md += ["## 10. 모형 검증 (마스터 스케일 등급, Jeffreys 단측검정)",
           f"- 보정 함수 (Platt): 기울기 {calibrator.params['slope']:.3f}, 절편 {calibrator.params['intercept']:.3f}",
           f"- 보정 후 평가 데이터 AUROC: {val_out['auroc_after_calibration']:.3f} (보정은 순위를 바꾸지 않음)",
           f"- PSI (개발 out-of-fold vs 평가, 보정 후 등급 분포): {val_out['psi_dev_vs_test']:.4f}", ""]
    for k, (t, hl) in val_tables.items():
        show = t.drop(columns=["부도율 95% 구간"]).copy()
        for c in ["평균예측PD", "실제부도율"]:
            show[c] = show[c].map(lambda v: f"{v:.2%}")
        for c in ["이항검정 p", "Jeffreys p"]:
            show[c] = show[c].map(lambda v: f"{v:.3f}")
        md += [f"### {k} — Hosmer-Lemeshow 통계량 {hl['statistic']:.2f}, p = {hl['p_value']:.3f}", df_to_md(show), ""]

    # 11. 신용 VaR -----------------------------------------------------------------
    log("신용 VaR 몬테카를로")
    ead = config.EAD_PER_FIRM
    losses = credit_var.simulate_losses(pd_final, config.LGD, ead, n_sims=100_000, seed=config.SEED)
    var_sum = credit_var.loss_summary(losses)
    irb_ul = credit_var.irb_capital(pd_final, config.LGD, ead)
    # 시뮬레이션 오차: 99.9% 분위수는 10만 회 중 가장 나쁜 100회로 정해져 시드에 따라 흔들린다
    seed_ul = [credit_var.loss_summary(credit_var.simulate_losses(pd_final, config.LGD, ead, 100_000, seed=s))
               ["UL (VaR 99.9% - EL)"] for s in range(1, 6)]
    exposure = len(yte) * ead
    out["credit_var"] = {**var_sum, "IRB UL (analytic)": irb_ul, "exposure": exposure,
                         "realized_loss": float(yte.sum() * config.LGD * ead), "n_sims": 100_000,
                         "UL_other_seeds": seed_ul}
    plotting.plot_loss_distribution(losses, var_sum, exposure, config.FIG_DIR / "09_loss_distribution.png")
    var_df = pd.DataFrame({"억원": {k: f"{v:.1f}" for k, v in var_sum.items()} |
                           {"바젤 IRB 공식 UL (해석해)": f"{irb_ul:.1f}"}})
    var_df["노출 대비"] = [f"{float(v) / exposure:.1%}" for v in var_df["억원"]]
    var_df.index.name = "항목"
    md += ["## 11. 신용 VaR (보정된 PD, 1요인 모형 몬테카를로 10만 회)", df_to_md(var_df),
           "", f"몬테카를로 UL과 IRB 공식 UL의 차이: {(var_sum['UL (VaR 99.9% - EL)'] / irb_ul - 1):+.1%}",
           f"다른 시드 5개의 몬테카를로 UL: {min(seed_ul):.1f} ~ {max(seed_ul):.1f}억 "
           f"(IRB 대비 {min(seed_ul) / irb_ul - 1:+.1%} ~ {max(seed_ul) / irb_ul - 1:+.1%})", ""]

    # 12. 스트레스 테스트 ---------------------------------------------------------------
    log("스트레스 테스트")
    stress_df = stress.systemic_scenarios(pd_final, config.LGD, ead)
    drivers = {f: (1 if imp.loc[f, "spearman"] > 0 else -1) for f in imp.index[:5]}
    shocked_pd = {}
    for shift in (0.10, 0.20):
        Xs = stress.shift_quantiles(Xte, Xtr, drivers, shift)
        p_s = calibrator.transform(fitted["XGBoost"].predict_proba(Xs)[:, 1])
        p_s_floor = np.maximum(p_s, credit_var.PD_FLOOR)
        label = f"재무비율 {int(shift * 100)}%p 악화"
        stress_df.loc[label] = {"평균PD": float(p_s_floor.mean()),
                                "예상손실": float(p_s_floor.sum() * config.LGD * ead)}
        shocked_pd[label] = p_s
    base_label = "기준 (평상시)"
    plotting.plot_stress(stress_df, base_label, config.FIG_DIR / "10_stress.png")
    migration = {k: {"share_grade_8_9": float((validation.assign_grade(v) >= 8).mean()),
                     "psi_vs_base": validation.psi(pd_final, v)} for k, v in shocked_pd.items()}
    migration["기준 (평상시)"] = {"share_grade_8_9": float((validation.assign_grade(pd_final) >= 8).mean()),
                              "psi_vs_base": 0.0}
    out["stress"] = {"table": stress_df.reset_index().to_dict(orient="records"),
                     "drivers": drivers, "migration": migration}
    st = stress_df.copy()
    st["평균PD"] = st["평균PD"].map(lambda v: f"{v:.2%}")
    st["배율"] = (stress_df["예상손실"] / stress_df.loc[base_label, "예상손실"]).map(lambda v: f"{v:.2f}")
    st["예상손실"] = stress_df["예상손실"].map(lambda v: f"{v:.1f}")
    md += ["## 12. 스트레스 테스트 (보정된 PD, 예상손실 단위 억원)", df_to_md(st), "",
           f"재무비율 악화 시나리오에서 옮긴 변수(방향 +1 = 높을수록 위험): {drivers}", "",
           *[f"- {k}: 8·9등급 비중 {v['share_grade_8_9']:.1%}, 기준 대비 PSI {v['psi_vs_base']:.3f}"
             for k, v in migration.items()], ""]

    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                                     encoding="utf-8")
    (config.REPORT_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    log("완료: reports/metrics.json, reports/results.md, reports/figures/")


if __name__ == "__main__":
    main()
