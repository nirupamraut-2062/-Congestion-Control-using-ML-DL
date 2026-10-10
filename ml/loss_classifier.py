"""Train and evaluate the loss classifier: was a packet lost to congestion or to a noisy link?

Compares a single-threshold rule, logistic regression, a random forest and XGBoost on
scenarios none of them saw during training, then saves the best model to models/.

Usage: python -m ml.loss_classifier
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from experiments.style import INK, INK2, SURFACE, apply_style
from sim import PKTS_PER_MBPS

from .features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
MODEL_PATH = MODELS / "loss_classifier.joblib"

BLUE = "#2a78d6"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
CLASSES = ["Random loss", "Congestion loss"]


def build_models(pos_weight: float) -> dict:
    return {
        "Single-threshold rule": DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=0),
    }


def score(y, pred, prob) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "recall_random": float(np.mean(pred[y == 0] == 0)),
        "recall_congestion": float(np.mean(pred[y == 1] == 1)),
        "roc_auc": roc_auc_score(y, prob),
    }


def plot_models(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    order = metrics.iloc[::-1]
    ax.barh(order["model"], order["macro_f1"], height=0.55, color=BLUE)
    for name, value in zip(order["model"], order["macro_f1"]):
        ax.annotate(f"{value:.3f}", (value, name), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro F1 on unseen scenarios (1.0 = perfect)")
    ax.set_title("Loss classifier: model comparison")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_classifier_models.png")
    plt.close(fig)


def plot_confusion(y, pred, name: str) -> None:
    counts = confusion_matrix(y, pred)
    share = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{100 * share[i, j]:.1f}%\n{counts[i, j]:,}", ha="center", va="center",
                    color=SURFACE if share[i, j] > 0.5 else INK, fontsize=11)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name}: confusion matrix (test set)")
    ax.grid(visible=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_classifier_confusion.png")
    plt.close(fig)


def plot_importance(model, X, y, name: str) -> None:
    result = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=5, random_state=0)
    order = np.argsort(result.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(FEATURES)[order], result.importances_mean[order], height=0.55, color=BLUE)
    ax.set_xlabel("Drop in macro F1 when the feature is shuffled")
    ax.set_title(f"{name}: which signals the classifier relies on")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_classifier_importance.png")
    plt.close(fig)


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    apply_style()

    data = pd.read_csv(DATA / "loss_events.csv")
    X, y, runs = data[FEATURES].to_numpy(), data["congestion"].to_numpy(), data["run_id"]
    # Whole scenarios are held out, so the test set measures transfer to unseen networks.
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, runs))
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    print(f"train: {len(train)} events from {runs.iloc[train].nunique()} scenarios, "
          f"test: {len(test)} events from {runs.iloc[test].nunique()} scenarios")

    models = build_models(pos_weight=float(np.sum(y_train == 0) / np.sum(y_train == 1)))
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        rows.append({"model": name, **score(y_test, predictions[name], model.predict_proba(X_test)[:, 1])})
    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(RESULTS / "loss_classifier_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    # A controller classifies one loss at a time, where XGBoost is far faster than a
    # 200-tree forest, so XGBoost is deployed unless another model is clearly more accurate.
    f1 = metrics.set_index("model")["macro_f1"]
    best = "XGBoost" if f1.max() - f1["XGBoost"] < 0.005 else f1.idxmax()
    model = models[best]

    # Shallow buffers leave little delay signal before a drop, so they are the hard case.
    held_out = data.iloc[test]
    pipe = held_out["bandwidth_mbps"] * PKTS_PER_MBPS * 2 * held_out["delay_ms"] / 1000
    depth = pd.cut(held_out["queue_pkts"] / pipe, [0, 0.25, 0.5, 1, np.inf],
                   labels=["< 0.25", "0.25-0.5", "0.5-1", "> 1"]).rename("queue_size_in_pipes")
    correct = pd.Series(predictions[best] == y_test, index=held_out.index)
    by_buffer = correct.groupby(depth, observed=True).agg(events="size", accuracy="mean").reset_index()
    by_buffer.round(4).to_csv(RESULTS / "loss_classifier_by_buffer.csv", index=False)
    print(f"\n{best} accuracy by buffer depth (queue size / pipe size):")
    print(by_buffer.round(3).to_string(index=False))

    plot_models(metrics)
    plot_confusion(y_test, predictions[best], best)
    plot_importance(model, X_test, y_test, best)

    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    joblib.dump({"name": best, "model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\nSaved {best} to {MODEL_PATH}")


if __name__ == "__main__":
    main()
