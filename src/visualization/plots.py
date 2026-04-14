import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.ndimage import rotate

from src.config import (
    FIGURES_DIR
)
from src.evaluation.roc_auc import compute_roc_points


def plot_target_distribution(df, target_col='Risk'):
    counts = df[target_col].value_counts()
    plt.bar(counts.index.astype(str), counts.values)

    plt.title("Target Distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")

    plt.savefig(FIGURES_DIR / "target_distribution.png")
    plt.close()

def plot_histogram(df, column):
    plt.hist(df[column].dropna().values, bins=30)

    plt.title(f'{column.capitalize()} Histogram')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    plt.savefig(FIGURES_DIR / f"{column.lower().replace(" ", "_")}_histogram.png")
    plt.close()

def plot_categorical_counts(df, column):
    counts = df[column].value_counts(dropna=False)
    plt.bar(counts.index.astype(str), counts.values)

    plt.title(f"{column.capitalize()} Distribution")
    plt.xlabel(f'{column.capitalize()}')

    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / f'{column.lower().replace(" ", "_")}_distribution.png')
    plt.close()

def plot_numeric_by_target(df, column, target_col):
    labels = sorted(df[target_col].dropna().unique())
    groups = [
        df[df[target_col] == label][column].dropna()
        for label in labels
    ]
    plt.figure()
    plt.boxplot(groups, labels=labels)
    plt.title(f'{column.capitalize()} by {target_col.capitalize()}')
    plt.xlabel(target_col.capitalize())
    plt.ylabel(column.capitalize())
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / f'{column.lower().replace(" ", "_")}_by_target.png')
    plt.close()


def plot_categorical_by_target(df, column, target_col):
    ct = pd.crosstab(df[column], df[target_col], normalize="index", dropna=False)

    ax = ct.plot(kind="bar", stacked=True)

    ax.set_xlabel(column)
    ax.set_ylabel("Proportion")
    ax.set_title(f"{column} vs {target_col} (Proportions)")

    ax.tick_params(axis='x', labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    ax.legend(title=target_col)

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / f'{column.lower().replace(" ", "_")}_by_target.png')
    plt.close()

def plot_roc_curve(fpr_list, tpr_list, auc=None, filename="roc_curve.png"):
    fig, ax = plt.subplots()

    # ROC curve
    ax.plot(fpr_list, tpr_list, label="ROC Curve")

    # Diagonal (random model)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Model")

    # Labels
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")

    # AUC in Legend
    if auc is not None:
        ax.legend(title=f"AUC = {auc:.4f}")
    else:
        ax.legend()

    # Grid
    ax.grid(True)

    # Save
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path)

    plt.close(fig)