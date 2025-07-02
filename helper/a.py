import pandas as pd
import re
import os
import numpy as np

def extract_main_score(cell):
    """Extract float before '±', or return NaN if OOM, Too slow*, or empty."""
    if pd.isna(cell) or str(cell).strip() in ["OOM", "Too slow*", ""]:
        return np.nan
    match = re.match(r"([0-9.]+)", str(cell))
    return float(match.group(1)) if match else np.nan

def clean_btxt_with_exceptions(input_path: str, output_path: str):
    # Read TSV (tab-separated) input
    df = pd.read_csv(input_path, sep='\t')

    # Clean each cell (except model column)
    for col in df.columns[1:]:
        df[col] = df[col].map(extract_main_score)

    # Save as CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")

if __name__ == "__main__":
    input_path = "helper/b.txt"
    output_path = "helper/multi_large.csv"
    # clean_btxt_with_exceptions(input_path, output_path)
