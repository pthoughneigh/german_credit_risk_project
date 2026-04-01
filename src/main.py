from src.data.loader import load_raw_data
from src.data.cleaning import drop_useless_columns
from src.analysis.eda import (
    print_shape,
    print_missing_values,
    print_target_distribution,
    print_unique_values,
)
def main():
    df = load_raw_data()
    df = drop_useless_columns(df)

    print_shape(df)
    print_missing_values(df)
    print_target_distribution(df)
    print_unique_values(df)

if __name__ == "__main__":
    main()
