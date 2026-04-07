from pathlib import Path

def save_feature_importance_table(table, output_path):
    """
    Save feature importance table to CSV.

    Parameters:
        table: pandas DataFrame
        output_path: str or Path
    """
    output_path = Path(output_path)

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_path, index=False)

    print(f"Feature importance table saved to: {output_path}")