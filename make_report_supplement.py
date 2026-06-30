from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
FIG_DIR = ARTIFACT_DIR / "report_supplement"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAME = "SK_HYNIX"
RANDOM_STATE = 42
TEST_MONTHS = 3
LOOK_BACK = 20

sns.set_theme(style="whitegrid", font="Malgun Gothic")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_price_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])


def evaluate_binary(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_source_profile() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        if path.name in {"manifest.csv", "failed_downloads.csv"}:
            continue
        df = load_price_csv(path)
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        rows.append(
            {
                "symbol": path.stem,
                "rows": len(df),
                "start": df["Date"].min().date(),
                "end": df["Date"].max().date(),
                "missing_price_ratio": float(df[price_cols].isna().mean().mean()) if price_cols else np.nan,
            }
        )

    profile = pd.DataFrame(rows).sort_values("rows", ascending=False)
    profile.to_csv(FIG_DIR / "data_source_profile.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    show = profile.sort_values("rows")
    axes[0].barh(show["symbol"], show["rows"], color="#4C78A8")
    axes[0].set_title("원천 데이터별 수집 행 수")
    axes[0].set_xlabel("Rows")
    axes[0].grid(axis="x", alpha=0.25)

    miss = profile.sort_values("missing_price_ratio", ascending=False).head(15).sort_values("missing_price_ratio")
    axes[1].barh(miss["symbol"], miss["missing_price_ratio"] * 100, color="#F58518")
    axes[1].set_title("원천 가격 컬럼 결측 비율 Top 15")
    axes[1].set_xlabel("Missing ratio (%)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_data_source_profile.png", dpi=160)
    plt.close(fig)
    return profile


def save_feature_profile(feature_df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    df = feature_df[feature_df["target_name"] == TARGET_NAME].sort_values("Date").reset_index(drop=True)
    feature_cols = feature_sets["F5"]
    x = df[feature_cols].astype(float)

    stats = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_ratio": x.isna().mean().values,
            "mean": x.mean(numeric_only=True).values,
            "std": x.std(numeric_only=True).values,
            "min": x.min(numeric_only=True).values,
            "max": x.max(numeric_only=True).values,
            "skewness": x.skew(numeric_only=True).values,
        }
    )
    stats.to_csv(FIG_DIR / "feature_profile_f5.csv", index=False, encoding="utf-8-sig")

    missing = stats.sort_values("missing_ratio", ascending=False).head(20).sort_values("missing_ratio")
    skew = stats.assign(abs_skew=stats["skewness"].abs()).sort_values("abs_skew", ascending=False).head(20).sort_values("abs_skew")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].barh(missing["feature"], missing["missing_ratio"] * 100, color="#E45756")
    axes[0].set_title("F5 Feature 결측 비율 Top 20")
    axes[0].set_xlabel("Missing ratio (%)")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(skew["feature"], skew["skewness"], color="#54A24B")
    axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].set_title("F5 Feature 왜도 Top 20")
    axes[1].set_xlabel("Skewness")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_feature_missingness_skewness.png", dpi=160)
    plt.close(fig)
    return stats


def save_distribution_examples(feature_df: pd.DataFrame) -> None:
    df = feature_df[feature_df["target_name"] == TARGET_NAME].sort_values("Date").reset_index(drop=True)
    candidates = ["SOXX_ret_1d", "SMH_ret_1d", "NVDA_ret_1d", "VIX_ret_1d", "USD_KRW_ret_1d", "target_ret_1d"]
    cols = [c for c in candidates if c in df.columns][:6]
    if not cols:
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=50, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(col)
        ax.set_xlabel("Return / change")
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.suptitle("주요 독립변수 분포 예시", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_distribution_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_preprocessing_protocol(feature_df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    us_names = [
        "SOXX",
        "SMH",
        "NVDA",
        "AMD",
        "MU",
        "INTC",
        "AVGO",
        "QCOM",
        "AMAT",
        "LRCX",
        "KLAC",
        "QQQ",
        "SPY",
        "VIX",
        "USD_KRW",
        "USD_JPY",
        "DOLLAR_INDEX",
        "US_10Y",
    ]
    tables = []
    for name in us_names:
        df = load_price_csv(RAW_DIR / f"{name}.csv")
        keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].rename(columns={c: f"{name}_{c}" for c in keep if c != "Date"})
        tables.append(df)

    wide = tables[0]
    for table in tables[1:]:
        wide = wide.merge(table, on="Date", how="outer")
    feature_cols = [c for c in wide.columns if c != "Date"]
    raw_missing = float(wide[feature_cols].isna().mean().mean())
    ffilled = wide.sort_values("Date").copy()
    ffilled[feature_cols] = ffilled[feature_cols].ffill()
    ffill_missing = float(ffilled[feature_cols].isna().mean().mean())

    target_df = feature_df[feature_df["target_name"] == TARGET_NAME].sort_values("Date").reset_index(drop=True)
    f5_missing = float(target_df[feature_sets["F5"]].isna().mean().mean())

    rows = [
        {
            "step": "1. US/macro outer merge",
            "method": "Calendar union by Date",
            "missing_ratio": raw_missing,
            "purpose": "미국/매크로 자산별 휴장일 차이 확인",
        },
        {
            "step": "2. Holiday gap fill",
            "method": "Forward Fill",
            "missing_ratio": ffill_missing,
            "purpose": "최근 확인 가능한 직전 시장값 유지",
        },
        {
            "step": "3. KR-US time alignment",
            "method": "merge_asof backward, exact date excluded",
            "missing_ratio": float(target_df["us_date"].isna().mean()),
            "purpose": "한국 거래일에 직전 미국 마감만 연결",
        },
        {
            "step": "4. Rolling feature NaN",
            "method": "Train-only median imputer in Pipeline",
            "missing_ratio": f5_missing,
            "purpose": "초기 이동창 결측과 일부 잔여 결측 처리",
        },
    ]
    protocol = pd.DataFrame(rows)
    protocol.to_csv(FIG_DIR / "preprocessing_protocol.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    cell_text = protocol.copy()
    cell_text["missing_ratio"] = (cell_text["missing_ratio"] * 100).map("{:.2f}%".format)
    table = ax.table(
        cellText=cell_text.values,
        colLabels=["Step", "Method", "Missing", "Purpose"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2F4B7C")
        else:
            cell.set_facecolor("#F7F9FC" if row % 2 else "white")
    ax.set_title("전처리 및 데이터 클렌징 프로토콜", fontsize=14, pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_preprocessing_protocol.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return protocol


def train_rf_importance(feature_df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    df = feature_df[feature_df["target_name"] == TARGET_NAME].sort_values("Date").reset_index(drop=True)
    max_date = df["Date"].max()
    test_start = max_date - pd.DateOffset(months=TEST_MONTHS)
    train_df = df[df["Date"] < test_start].copy()
    test_df = df[df["Date"] >= test_start].copy()
    feature_cols = feature_sets["F5"]

    X_train = train_df[feature_cols]
    y_train = train_df["y_gap"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["y_gap"].astype(int)

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    max_depth=6,
                    min_samples_leaf=10,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_test, y_prob)

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(FIG_DIR / "rf_feature_importance_y_gap_f5.csv", index=False, encoding="utf-8-sig")
    joblib.dump(rf, ARTIFACT_DIR / "models" / "SK_HYNIX_y_gap_F5_random_forest_importance.joblib")
    (FIG_DIR / "rf_feature_importance_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    top = importances.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"], color="#4C78A8")
    ax.set_title(f"RandomForest 변수 중요도 Top 20 (y_gap, F5, AUC={metrics['roc_auc']:.3f})")
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_rf_feature_importance.png", dpi=160)
    plt.close(fig)
    return importances


def build_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str, scaler: StandardScaler | None = None, fit: bool = False):
    work = df.sort_values("Date").reset_index(drop=True)
    x_raw = work[feature_cols].astype(float)
    y_raw = work[target_col].astype(int)
    scaler = scaler or StandardScaler()
    x_scaled = scaler.fit_transform(x_raw) if fit else scaler.transform(x_raw)

    x_seq, y_seq = [], []
    for i in range(LOOK_BACK, len(work)):
        x_seq.append(x_scaled[i - LOOK_BACK : i])
        y_seq.append(y_raw.iloc[i])
    return np.asarray(x_seq), np.asarray(y_seq), scaler


def run_lstm_ablation(feature_df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict:
    metrics_path = FIG_DIR / "lstm_regularized_metrics.json"
    history_path = FIG_DIR / "lstm_regularized_history.csv"
    compare_path = FIG_DIR / "lstm_ablation_compare.csv"
    if metrics_path.exists() and history_path.exists() and compare_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        compare = pd.read_csv(compare_path)
        hist = pd.read_csv(history_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        compare.set_index("model")[["roc_auc", "balanced_accuracy", "f1"]].plot(kind="bar", ax=axes[0], color=["#4C78A8", "#F58518", "#54A24B"])
        axes[0].set_title("LSTM 규제 개선 실험 지표 비교")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].plot(hist["loss"], label="Train loss")
        axes[1].plot(hist["val_loss"], label="Validation loss")
        if "val_auc" in hist.columns:
            axes[1].plot(hist["val_auc"], label="Validation AUC")
        axes[1].set_title("Regularized LSTM 학습 곡선")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "11_lstm_regularized_ablation.png", dpi=160)
        plt.close(fig)
        return metrics

    df = feature_df[feature_df["target_name"] == TARGET_NAME].sort_values("Date").reset_index(drop=True)
    max_date = df["Date"].max()
    test_start = max_date - pd.DateOffset(months=TEST_MONTHS)
    train_df = df[df["Date"] < test_start].copy()
    test_df = df[df["Date"] >= test_start].copy()
    feature_cols = feature_sets["F5"]

    scaler = StandardScaler()
    x_full, y_full, scaler = build_sequences(train_df, feature_cols, "y_daily", scaler=scaler, fit=True)
    x_test, y_test, _ = build_sequences(test_df, feature_cols, "y_daily", scaler=scaler, fit=False)
    val_size = max(1, int(len(x_full) * 0.2))
    x_train, x_val = x_full[:-val_size], x_full[-val_size:]
    y_train, y_val = y_full[:-val_size], y_full[-val_size:]

    model = keras.Sequential(
        [
            layers.Input(shape=(LOOK_BACK, len(feature_cols))),
            layers.LSTM(
                32,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(0.01),
                recurrent_regularizer=regularizers.l2(0.01),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.LSTM(16, kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)],
        verbose=0,
        shuffle=False,
    )

    y_prob = model.predict(x_test, verbose=0).ravel()
    metrics = evaluate_binary(y_test, y_prob)
    metrics["epochs"] = len(history.history["loss"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(history_path, index=False, encoding="utf-8-sig")

    baseline_path = ARTIFACT_DIR / "reports" / "SK_HYNIX_y_daily_F5_lstm_metrics.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    compare = pd.DataFrame(
        [
            {"model": "Baseline LSTM", "roc_auc": baseline.get("roc_auc", np.nan), "balanced_accuracy": baseline.get("balanced_accuracy", np.nan), "f1": baseline.get("f1", np.nan)},
            {"model": "L2+BN+Dropout LSTM", "roc_auc": metrics["roc_auc"], "balanced_accuracy": metrics["balanced_accuracy"], "f1": metrics["f1"]},
        ]
    )
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    compare.set_index("model")[["roc_auc", "balanced_accuracy", "f1"]].plot(kind="bar", ax=axes[0], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("LSTM 규제 개선 실험 지표 비교")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.25)

    hist = pd.DataFrame(history.history)
    axes[1].plot(hist["loss"], label="Train loss")
    axes[1].plot(hist["val_loss"], label="Validation loss")
    if "val_auc" in hist.columns:
        axes[1].plot(hist["val_auc"], label="Validation AUC")
    axes[1].set_title("Regularized LSTM 학습 곡선")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_lstm_regularized_ablation.png", dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    feature_df = pd.read_csv(PROCESSED_DIR / "preopen_feature_table.csv", parse_dates=["Date", "us_date"])
    feature_sets = json.loads((PROCESSED_DIR / "feature_sets.json").read_text(encoding="utf-8"))

    save_source_profile()
    save_feature_profile(feature_df, feature_sets)
    save_distribution_examples(feature_df)
    save_preprocessing_protocol(feature_df, feature_sets)
    train_rf_importance(feature_df, feature_sets)
    run_lstm_ablation(feature_df, feature_sets)
    print(f"saved supplement outputs to {FIG_DIR}")


if __name__ == "__main__":
    main()
