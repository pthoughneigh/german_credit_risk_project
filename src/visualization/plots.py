import matplotlib.pyplot as plt
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