from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "artifacts" / "reports"
PRED_DIR = PROJECT_DIR / "artifacts" / "predictions"
FIG_DIR = PROJECT_DIR / "artifacts" / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def save_model_comparison() -> None:
    frames = []
    for target_name, path in [
        ("y_gap", REPORT_DIR / "ml_baseline_results.csv"),
        ("y_daily", REPORT_DIR / "y_daily_ml_baseline_results.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            df["target"] = target_name
            df["label"] = df["target"] + " / " + df["feature_set"] + " / " + df["model_name"]
            frames.append(df)

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("roc_auc", ascending=False).head(10).sort_values("roc_auc")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["label"], df["roc_auc"], color="#4C78A8")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_title("ML 모델별 ROC-AUC 비교")
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.4, max(0.85, df["roc_auc"].max() + 0.03))
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_model_comparison_auc.png", dpi=160)
    plt.close(fig)


def save_equity_curve(path: Path, out_name: str, title: str) -> None:
    if not path.exists():
        return

    bt = pd.read_csv(path, parse_dates=["Date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    if "strategy_return_gross" in bt.columns:
        ax.plot(bt["Date"], (1 + bt["strategy_return_gross"].fillna(0)).cumprod(), label="Strategy gross", linewidth=2)
    if "strategy_return_net" in bt.columns:
        ax.plot(bt["Date"], (1 + bt["strategy_return_net"].fillna(0)).cumprod(), label="Strategy net", linewidth=2)
    if "always_in_return" in bt.columns:
        ax.plot(bt["Date"], (1 + bt["always_in_return"].fillna(0)).cumprod(), label="Always-in", linestyle="--")
    if "target_Open" in bt.columns and "target_Close" in bt.columns:
        ax.plot(bt["Date"], bt["target_Close"] / bt["target_Open"].iloc[0], label="Buy & hold", linestyle=":")

    ax.set_title(title)
    ax.set_ylabel("누적 수익 배율")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def save_lstm_history() -> None:
    path = REPORT_DIR / "SK_HYNIX_y_daily_F5_lstm_history.csv"
    if not path.exists():
        return

    hist = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["loss"], label="Train loss")
    axes[0].plot(hist["val_loss"], label="Validation loss")
    axes[0].set_title("LSTM Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if "auc" in hist.columns and "val_auc" in hist.columns:
        axes[1].plot(hist["auc"], label="Train AUC")
        axes[1].plot(hist["val_auc"], label="Validation AUC")
        axes[1].set_title("LSTM AUC Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_lstm_training_history.png", dpi=160)
    plt.close(fig)


def save_prediction_probability() -> None:
    path = PRED_DIR / "SK_HYNIX_y_gap_F1_logistic_regression_predictions.csv"
    if not path.exists():
        return

    pred = pd.read_csv(path, parse_dates=["Date"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pred["Date"], pred["y_prob"], label="Predicted probability", color="#F58518")
    ax.scatter(pred["Date"], pred["y_true"], label="Actual direction", s=16, alpha=0.7, color="#4C78A8")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("최근 3개월 y_gap 예측 확률과 실제 방향")
    ax.set_ylabel("Probability / Actual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_gap_prediction_probability.png", dpi=160)
    plt.close(fig)


def save_summary_text() -> None:
    gap = pd.read_csv(REPORT_DIR / "ml_baseline_results.csv").sort_values(["roc_auc", "balanced_accuracy"], ascending=False).iloc[0]
    daily = pd.read_csv(REPORT_DIR / "y_daily_ml_baseline_results.csv").sort_values(["roc_auc", "balanced_accuracy"], ascending=False).iloc[0]
    metrics_path = REPORT_DIR / "SK_HYNIX_y_daily_F5_lstm_metrics.json"
    lstm = json.load(open(metrics_path, encoding="utf-8")) if metrics_path.exists() else {}

    lines = [
        "# Report Figure Summary",
        "",
        f"- Best y_gap ML: {gap['feature_set']} / {gap['model_name']} / AUC={gap['roc_auc']:.3f}, balanced accuracy={gap['balanced_accuracy']:.3f}",
        f"- Best y_daily ML: {daily['feature_set']} / {daily['model_name']} / AUC={daily['roc_auc']:.3f}, balanced accuracy={daily['balanced_accuracy']:.3f}",
        f"- LSTM y_daily F5: AUC={lstm.get('roc_auc', 0):.3f}, balanced accuracy={lstm.get('balanced_accuracy', 0):.3f}",
    ]
    (FIG_DIR / "figure_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    save_model_comparison()
    save_equity_curve(
        REPORT_DIR / "SK_HYNIX_y_gap_F1_logistic_regression_backtest.csv",
        "02_gap_backtest_equity.png",
        "y_gap Logistic Regression 백테스트",
    )
    save_equity_curve(
        REPORT_DIR / "SK_HYNIX_y_daily_F5_logistic_regression_backtest.csv",
        "03_daily_backtest_equity.png",
        "y_daily Logistic Regression 백테스트",
    )
    save_lstm_history()
    save_prediction_probability()
    save_summary_text()
    print(f"saved figures to {FIG_DIR}")
