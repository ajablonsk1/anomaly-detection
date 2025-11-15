import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrices(results, save_path):
    n_models = len(results)
    _, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 12))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    colors = ["Blues", "Greens", "Oranges"]

    for idx, result in enumerate(results):
        cm_bin = confusion_matrix(result["y_test_bin"], result["y_pred_bin"])
        sns.heatmap(
            cm_bin,
            annot=True,
            fmt="d",
            cmap=colors[idx % len(colors)],
            ax=axes[0, idx],
            xticklabels=["BENIGN", "ATTACK"],
            yticklabels=["BENIGN", "ATTACK"],
        )
        axes[0, idx].set_title(
            f"{result['model_name']} - Binary Classification",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, idx].set_ylabel("True class")
        axes[0, idx].set_xlabel("Predicted class")

        cm_multi = confusion_matrix(result["y_test_multi"], result["y_pred_multi"])
        sns.heatmap(
            cm_multi,
            annot=True,
            fmt="d",
            cmap=colors[idx % len(colors)],
            ax=axes[1, idx],
            xticklabels=result["class_names"],
            yticklabels=result["class_names"],
        )
        axes[1, idx].set_title(
            f"{result['model_name']} - Multiclass Classification",
            fontsize=14,
            fontweight="bold",
        )
        axes[1, idx].set_ylabel("True class")
        axes[1, idx].set_xlabel("Predicted class")
        axes[1, idx].tick_params(axis="x", rotation=45)
        axes[1, idx].tick_params(axis="y", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_feature_importance(models_data, feature_names, save_path):
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, model_data in enumerate(models_data):
        if model_data["importances"] is None:
            axes[idx].text(
                0.5,
                0.5,
                "Feature importance\nnot available",
                ha="center",
                va="center",
                fontsize=14,
            )
            axes[idx].set_title(f"{model_data['name']}", fontsize=14, fontweight="bold")
            continue

        importance_df = (
            pd.DataFrame(
                {"feature": feature_names, "importance": model_data["importances"]}
            )
            .sort_values("importance", ascending=False)
            .head(15)
        )

        axes[idx].barh(range(len(importance_df)), importance_df["importance"])
        axes[idx].set_yticks(range(len(importance_df)))
        axes[idx].set_yticklabels(importance_df["feature"])
        axes[idx].set_xlabel("Feature Importance")
        axes[idx].set_title(
            f"Top 15 Features - {model_data['name']}", fontsize=14, fontweight="bold"
        )
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_class_distribution(y, save_path):
    plt.figure(figsize=(10, 6))

    class_counts = y.value_counts()
    colors = plt.cm.Set3(range(len(class_counts)))

    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors)
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Class Distribution", fontsize=14, fontweight="bold")

    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count / len(y) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")
