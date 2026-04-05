import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import rotate

from src.config import (
    FIGURES_DIR
)

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