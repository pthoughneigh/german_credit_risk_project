from pathlib import Path
import pandas as pd
from typing import Union

def save_feature_importance_table(
    table: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Save a feature importance table to a CSV file.

    This function writes a pandas DataFrame containing feature importance
    information to a specified file path. If the target directory does not
    exist, it is created automatically.

    Parameters
    ----------
    table : pandas.DataFrame
        DataFrame containing feature importance values.

    output_path : str or pathlib.Path
        Destination file path where the CSV file will be saved.

    Returns
    -------
    None
        The function saves the file to disk and does not return a value.

    Notes
    -----
    - The output directory is created if it does not exist.
    - The CSV file is saved without the index.
    - A confirmation message is printed after saving.
    """
    output_path = Path(output_path)

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_path, index=False)

    print(f"Feature importance table saved to: {output_path}")