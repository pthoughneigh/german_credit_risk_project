import pandas as pd
from src.data.loader import load_raw_data

def print_shape(df) -> None:
    print("\n=== SHAPE ===")
    rows, columns = df.shape
    print(f"rows: {rows}, columns: {columns}")

def print_missing_values(df) -> None:
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

def print_target_distribution(df, target_col="Risk") -> None:
    print(f"\n=== TARGET DISTRIBUTION: {target_col} ===")
    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print("\nCounts:")
    print(counts)
    print("\nProportions:")
    print(proportions)

def print_unique_values(df):
    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        num_unique = df[col].nunique()
        unique_values = df[col].dropna().unique()
        print(f"Column: {col}")
        print(f"Number of unique values: {num_unique}")
        print(f"Values: {'Too many unique values to display.' if num_unique > 10 
                else unique_values }\n")