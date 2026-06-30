from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
PRED_DIR = ARTIFACT_DIR / "predictions"
REPORT_DIR = ARTIFACT_DIR / "reports"
FIG_DIR = ARTIFACT_DIR / "report_figures"
SUPPLEMENT_DIR = ARTIFACT_DIR / "report_supplement"


def list_model_records() -> list[dict]:
    records: list[dict] = []
    for path in MODEL_DIR.glob("*.joblib"):
        match = re.match(
            r"(?P<target>.+?)_(?P<target_col>y_[^_]+)_(?P<feature_set>F\d+)_(?P<model_name>.+)\.joblib$",
            path.name,
        )
        if match:
            records.append({**match.groupdict(), "family": "ML", "path": path})

    for path in MODEL_DIR.glob("*.keras"):
        match = re.match(r"(?P<target>.+?)_(?P<target_col>y_[^_]+)_(?P<feature_set>F\d+)_lstm\.keras$", path.name)
        if match:
            records.append({**match.groupdict(), "family": "DL", "model_name": "lstm", "path": path})

    return sorted(records, key=lambda r: (r["target"], r["target_col"], r["family"], r["feature_set"], r["model_name"]))


def select_records(records: list[dict], target: str, target_col: str, family: str, feature_set: str, model_name: str) -> list[dict]:
    filtered = [r for r in records if r["target"] == target and r["target_col"] == target_col]
    if family != "All":
        filtered = [r for r in filtered if r["family"] == family]
    if feature_set != "All":
        filtered = [r for r in filtered if r["feature_set"] == feature_set]
    if model_name != "All":
        filtered = [r for r in filtered if r["model_name"] == model_name]
    return filtered


def comparison_path_for(target_col: str) -> Path | None:
    paths = {
        "y_gap": REPORT_DIR / "ml_baseline_results.csv",
        "y_daily": REPORT_DIR / "y_daily_ml_baseline_results.csv",
    }
    path = paths.get(target_col)
    return path if path and path.exists() else None


def load_csv_if_exists(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction_path(record: dict) -> Path:
    return PRED_DIR / f"{record['target']}_{record['target_col']}_{record['feature_set']}_{record['model_name']}_predictions.csv"


def backtest_path(record: dict) -> Path:
    return REPORT_DIR / f"{record['target']}_{record['target_col']}_{record['feature_set']}_{record['model_name']}_backtest.csv"


def backtest_summary_path(record: dict) -> Path:
    return REPORT_DIR / f"{record['target']}_{record['target_col']}_{record['feature_set']}_{record['model_name']}_backtest_summary.csv"


def metric_path(record: dict) -> Path:
    return REPORT_DIR / f"{record['target']}_{record['target_col']}_{record['feature_set']}_{record['model_name']}_metrics.json"


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"이미지 파일이 없습니다: {path.name}")


def show_dataframe(path: Path, title: str, parse_dates: list[str] | None = None, max_rows: int | None = None) -> None:
    df = load_csv_if_exists(path, parse_dates=parse_dates)
    if df is None:
        st.info(f"{title} 파일이 없습니다.")
        return
    st.subheader(title)
    st.dataframe(df.head(max_rows) if max_rows else df, width="stretch", hide_index=True)


def show_equity_curve(record: dict) -> None:
    path = backtest_path(record)
    bt = load_csv_if_exists(path, parse_dates=["Date"])
    if bt is None:
        st.info("선택한 산출물의 백테스트 CSV가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    if "strategy_return_gross" in bt.columns:
        ax.plot(bt["Date"], (1 + bt["strategy_return_gross"].fillna(0)).cumprod(), label="Strategy gross", linewidth=2)
    if "strategy_return_net" in bt.columns:
        ax.plot(bt["Date"], (1 + bt["strategy_return_net"].fillna(0)).cumprod(), label="Strategy net", linewidth=2)
    if "always_in_return" in bt.columns:
        ax.plot(bt["Date"], (1 + bt["always_in_return"].fillna(0)).cumprod(), label="Always-in", linestyle="--")
    if "target_Open" in bt.columns and "target_Close" in bt.columns:
        ax.plot(bt["Date"], bt["target_Close"] / bt["target_Open"].iloc[0], label="Buy & hold", linestyle=":")

    ax.set_title(f"{record['target']} / {record['target_col']} / {record['feature_set']} / {record['model_name']}")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def render_overview() -> None:
    st.subheader("프로젝트 개요")
    st.write(
        """
        미국 반도체 시장의 직전 마감 정보를 이용해 SK하이닉스의 다음 거래일 방향성을 예측한 딥러닝 수업 개인 프로젝트입니다.
        앱은 저장된 CSV, 모델, 이미지 산출물만 읽으며 화면에서 모델을 다시 학습하지 않습니다.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML targets", "y_gap / y_daily")
    c2.metric("Feature sets", "F1 ~ F5")
    c3.metric("ML models", "Logistic / RF")
    c4.metric("DL model", "LSTM")

    st.markdown(
        """
        **핵심 결론**
        - `y_gap`에서는 `F1 + Logistic Regression`이 가장 안정적이었습니다.
        - `y_daily`에서는 `F5 + Logistic Regression`의 ROC-AUC가 가장 높았습니다.
        - 기본 LSTM은 ML baseline보다 낮았지만, L2/BatchNorm/Dropout 강화 실험에서는 개선 가능성을 확인했습니다.
        """
    )


def render_model_results(records: list[dict]) -> None:
    st.subheader("F1~F5 ML Baseline 성능 비교")
    gap = load_csv_if_exists(REPORT_DIR / "ml_baseline_results.csv")
    daily = load_csv_if_exists(REPORT_DIR / "y_daily_ml_baseline_results.csv")
    frames = []
    if gap is not None:
        gap["target_col"] = "y_gap"
        frames.append(gap)
    if daily is not None:
        daily["target_col"] = "y_daily"
        frames.append(daily)

    if frames:
        all_results = pd.concat(frames, ignore_index=True)
        cols = ["target_col", "feature_set", "model_name", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
        st.dataframe(all_results[cols], width="stretch", hide_index=True)

        best = all_results.sort_values(["target_col", "roc_auc", "balanced_accuracy"], ascending=[True, False, False]).groupby("target_col").head(1)
        st.write("타깃별 최고 ROC-AUC 조합")
        st.dataframe(best[cols], width="stretch", hide_index=True)
    else:
        st.info("ML 비교 결과 CSV가 없습니다.")

    st.divider()
    st.subheader("저장된 모델 산출물")
    st.dataframe(pd.DataFrame(records).drop(columns=["path"]).assign(path=[str(r["path"]) for r in records]), width="stretch", hide_index=True)


def render_prediction_backtest(records: list[dict]) -> None:
    st.subheader("예측 결과 및 백테스트")

    targets = sorted({r["target"] for r in records})
    target = st.selectbox("Target stock", targets)
    target_cols = sorted({r["target_col"] for r in records if r["target"] == target})
    target_col = st.selectbox("Prediction target", target_cols)
    families = ["All"] + sorted({r["family"] for r in records if r["target"] == target and r["target_col"] == target_col})
    family = st.selectbox("Model family", families)

    subset = select_records(records, target, target_col, family, "All", "All")
    feature_sets = ["All"] + sorted({r["feature_set"] for r in subset})
    feature_set = st.selectbox("Feature set", feature_sets)
    subset = select_records(records, target, target_col, family, feature_set, "All")
    model_names = ["All"] + sorted({r["model_name"] for r in subset})
    model_name = st.selectbox("Model", model_names)
    subset = select_records(records, target, target_col, family, feature_set, model_name)

    if not subset:
        st.warning("현재 선택 조건과 일치하는 산출물이 없습니다.")
        return

    record = subset[0]
    st.write(
        {
            "target": record["target"],
            "target_col": record["target_col"],
            "family": record["family"],
            "feature_set": record["feature_set"],
            "model_name": record["model_name"],
            "model_path": str(record["path"]),
        }
    )

    metrics = load_json_if_exists(metric_path(record))
    if metrics:
        cols = st.columns(4)
        for col, key in zip(cols, ["accuracy", "balanced_accuracy", "f1", "roc_auc"]):
            col.metric(key, f"{metrics.get(key, 0):.3f}")

    col1, col2 = st.columns(2)
    with col1:
        pred = load_csv_if_exists(prediction_path(record), parse_dates=["Date"])
        st.subheader("예측 결과")
        if pred is not None:
            show_cols = [c for c in ["Date", "y_true", "y_prob", "y_pred"] if c in pred.columns]
            st.dataframe(pred[show_cols].head(80), width="stretch", hide_index=True)
            st.metric("Test rows", len(pred))
        else:
            st.info("선택한 산출물의 예측 파일이 없습니다.")

    with col2:
        summary = load_csv_if_exists(backtest_summary_path(record))
        st.subheader("백테스트 요약")
        if summary is not None:
            st.dataframe(summary.T.astype(str), width="stretch", hide_index=False)
        else:
            st.info("선택한 산출물의 백테스트 요약이 없습니다.")

    st.subheader("수익 곡선")
    show_equity_curve(record)


def render_report_figures() -> None:
    st.subheader("보고서용 핵심 그림")
    st.write("노트북 결과를 보고서에 옮기기 쉽도록 저장한 주요 차트입니다.")
    show_image(FIG_DIR / "01_model_comparison_auc.png", "ML 모델 ROC-AUC 비교")
    c1, c2 = st.columns(2)
    with c1:
        show_image(FIG_DIR / "02_gap_backtest_equity.png", "y_gap 백테스트")
    with c2:
        show_image(FIG_DIR / "03_daily_backtest_equity.png", "y_daily 백테스트")
    c3, c4 = st.columns(2)
    with c3:
        show_image(FIG_DIR / "04_lstm_training_history.png", "기본 LSTM 학습 곡선")
    with c4:
        show_image(FIG_DIR / "05_gap_prediction_probability.png", "y_gap 예측 확률")


def render_supplement() -> None:
    st.subheader("보고서 보완 분석")
    st.write("다른 샘플 보고서 대비 부족했던 EDA, 전처리, 변수 중요도, LSTM 개선 실험을 보완한 산출물입니다.")

    tab_eda, tab_pre, tab_imp, tab_lstm = st.tabs(["EDA", "전처리", "변수 중요도", "LSTM Ablation"])

    with tab_eda:
        show_dataframe(SUPPLEMENT_DIR / "data_source_profile.csv", "원천 데이터 프로파일", max_rows=30)
        show_dataframe(SUPPLEMENT_DIR / "feature_profile_f5.csv", "F5 Feature 프로파일", max_rows=30)
        show_image(SUPPLEMENT_DIR / "06_data_source_profile.png", "원천 데이터별 수집 행 수와 결측 비율")
        show_image(SUPPLEMENT_DIR / "07_feature_missingness_skewness.png", "Feature 결측 비율과 왜도")
        show_image(SUPPLEMENT_DIR / "08_feature_distribution_examples.png", "주요 독립변수 분포")

    with tab_pre:
        show_dataframe(SUPPLEMENT_DIR / "preprocessing_protocol.csv", "전처리 및 데이터 클렌징 프로토콜")
        show_image(SUPPLEMENT_DIR / "09_preprocessing_protocol.png", "전처리 프로토콜 요약")
        st.markdown(
            """
            - 미국/매크로 데이터는 자산별 휴장일 차이를 고려해 calendar union 후 forward fill 했습니다.
            - 한국 거래일에는 `merge_asof`로 직전 미국 마감 정보만 연결했습니다.
            - 모델 학습 단계의 잔여 결측은 train 기준 median imputer로 처리했습니다.
            """
        )

    with tab_imp:
        metrics = load_json_if_exists(SUPPLEMENT_DIR / "rf_feature_importance_metrics.json")
        if metrics:
            cols = st.columns(4)
            for col, key in zip(cols, ["accuracy", "balanced_accuracy", "f1", "roc_auc"]):
                col.metric(key, f"{metrics.get(key, 0):.3f}")
        show_dataframe(SUPPLEMENT_DIR / "rf_feature_importance_y_gap_f5.csv", "RandomForest 보조 모델 변수 중요도", max_rows=30)
        show_image(SUPPLEMENT_DIR / "10_rf_feature_importance.png", "RandomForest 보조 모델 기반 변수 중요도")
        st.info("이 그림은 최종 best 모델이 아니라, 변수 영향도 해석을 위한 RandomForest 보조 모델 결과입니다.")

    with tab_lstm:
        show_dataframe(SUPPLEMENT_DIR / "lstm_ablation_compare.csv", "기본 LSTM vs 규제형 LSTM 비교")
        metrics = load_json_if_exists(SUPPLEMENT_DIR / "lstm_regularized_metrics.json")
        if metrics:
            cols = st.columns(4)
            for col, key in zip(cols, ["accuracy", "balanced_accuracy", "f1", "roc_auc"]):
                col.metric(key, f"{metrics.get(key, 0):.3f}")
        show_image(SUPPLEMENT_DIR / "11_lstm_regularized_ablation.png", "L2 + BatchNorm + Dropout LSTM 개선 실험")
        st.markdown(
            """
            규제형 LSTM은 기본 LSTM 언더퍼폼을 진단하기 위한 ablation study입니다.
            `kernel_regularizer=l2(0.01)`, `BatchNormalization`, `Dropout(0.35)`를 적용했습니다.
            """
        )


st.set_page_config(page_title="반도체 방향성 예측 MVP", layout="wide")
st.title("반도체 주가 방향성 예측 MVP")
st.caption("미국 반도체 시장 마감 정보를 활용해 SK하이닉스 다음 거래일 방향성을 검증한 결과 대시보드")

records = list_model_records()
if not records:
    st.warning("저장된 모델 산출물이 없습니다. 노트북을 먼저 실행해 주세요.")
    st.stop()

tabs = st.tabs(["Overview", "Model Results", "Predictions & Backtest", "Report Figures", "Supplement Analysis"])

with tabs[0]:
    render_overview()
with tabs[1]:
    render_model_results(records)
with tabs[2]:
    render_prediction_backtest(records)
with tabs[3]:
    render_report_figures()
with tabs[4]:
    render_supplement()
