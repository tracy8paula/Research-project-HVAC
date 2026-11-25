import pandas as pd
import numpy as np

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the ASHRAE Global Thermal Comfort Database II CSV file."""
    print(f"Loading dataset from: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset successfully loaded. Total records: {df.shape[0]}, Total features: {df.shape[1]}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print high-level summary and structure of the dataset."""
    print("\n--- Dataset Overview ---")
    print("Columns available in CSV:")
    print(df.columns.tolist())
    
    print("\nMissing values per column (Top 15):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(15))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter the dataset for indoor environmental quality and thermal comfort research.
    """
    print("\n--- Preprocessing & Filtering Dataset ---")
    
    # Core target columns
    target_columns = [
        'building_id', 'country', 'city', 'season', 'building_type', 'cooling_type',
        'gender', 'age', 'ht', 'wt', 
        'ta', 'tr', 'top', 'tg', 'rh', 'vel', 'met', 'clo',
        'thermal_sensation', 'thermal_preference', 'thermal_comfort', 'pmv', 'ppd'
    ]
    
    # Match existing columns (case-insensitive column matching to prevent dropped columns)
    col_mapping = {col.lower(): col for col in df.columns}
    existing_cols = [col_mapping[col.lower()] for col in target_columns if col.lower() in col_mapping]
    
    if not existing_cols:
        print("Warning: None of the target columns were found in the dataset! Returning full dataset.")
        df_filtered = df.copy()
    else:
        df_filtered = df[existing_cols].copy()

    # Numeric conversion for sensor and target variables
    numeric_cols = ['age', 'ht', 'wt', 'ta', 'tr', 'top', 'tg', 'rh', 'vel', 'met', 'clo', 'pmv', 'ppd', 'thermal_sensation']
    for col in numeric_cols:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # Drop rows missing essential environmental variables only if rows remain
    if 'thermal_sensation' in df_filtered.columns:
        valid_rows = df_filtered.dropna(subset=['thermal_sensation'])
        if len(valid_rows) > 0:
            df_filtered = valid_rows

    if 'ta' in df_filtered.columns:
        valid_rows = df_filtered.dropna(subset=['ta'])
        if len(valid_rows) > 0:
            df_filtered = valid_rows

    # Impute missing values for remaining numeric sensor columns
    for col in ['tr', 'top', 'tg', 'rh', 'vel']:
        if col in df_filtered.columns and df_filtered[col].notnull().any():
            median_val = df_filtered[col].median()
            df_filtered[col] = df_filtered[col].fillna(median_val)

    print(f"Processed dataset shape: {df_filtered.shape}")
    return df_filtered


def generate_summary_statistics(df: pd.DataFrame) -> None:
    """Print key descriptive statistics for environmental parameters safely."""
    print("\n--- Environmental & Thermal Index Summary Statistics ---")
    
    # Filter for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Error Prevention Check: Ensure DataFrame has numeric columns and rows
    if numeric_df.empty or numeric_df.shape[1] == 0:
        print("Warning: No numeric columns available to compute summary statistics.")
        return

    summary = numeric_df.describe().T
    
    # Safely select present summary metric columns
    stat_cols = [c for c in ['mean', 'std', 'min', '50%', 'max'] if c in summary.columns]
    print(summary[stat_cols])


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned dataset to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"\nCleaned and structured dataset saved to: {output_path}")


def main():
    input_csv = "ashrae_db2.01.csv"
    output_csv = "ashrae_db2.01_processed.csv"

    try:
        data = load_dataset(input_csv)
        inspect_data(data)
        processed_data = preprocess_data(data)
        generate_summary_statistics(processed_data)
        save_processed_data(processed_data, output_csv)
        print("\nWorkflow completed successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_csv}'. Ensure the CSV file is in the working directory.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()