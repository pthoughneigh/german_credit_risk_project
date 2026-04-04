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

def print_feature_groups(numeric_cols, categorical_cols, special_cols):
    print("\n=== NUMERIC FEATURES ===")
    print(numeric_cols)

    print("\n=== CATEGORICAL FEATURES ===")
    print(categorical_cols)

    print("\n=== SPECIAL / CODED FEATURES ===")
    print(special_cols)

def print_numeric_summary(df, numeric_cols, categorical_cols, special_cols):
    print("\n=== NUMERIC COLUMNS SUMMARY ===")
    print(df[numeric_cols].describe())
    print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
    print(df[categorical_cols].describe())
    print("\n=== SPECIAL COLUMNS SUMMARY ===")
    print(df[special_cols].describe())