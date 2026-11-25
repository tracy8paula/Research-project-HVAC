import zipfile
import os
import glob
import pandas as pd
import numpy as np

# 1. SET YOUR ZIP FILE NAME HERE
ZIP_FILE_PATH = "building59.zip"
EXTRACT_DIR = "./building59_extracted"
OUTPUT_CSV = "building59_processed_clean.csv"


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extracts the multi-file ZIP archive if not already extracted."""
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print("Extraction directory already exists. Skipping unzipping.")


def identify_and_merge_csvs(extract_to: str) -> pd.DataFrame:
    """
    Finds all CSV files inside extracted subfolders, identifies key HVAC/IAQ 
    sensor streams, and merges them into a single pandas DataFrame.
    """
    print("\nScanning extracted directory for sensor CSV files...")
    all_csv_files = glob.glob(os.path.join(extract_to, "**", "*.csv"), recursive=True)
    print(f"Found {len(all_csv_files)} individual CSV files.")

    # Target variable keywords inside Building 59 file names or headers
    target_keywords = ['temp', 'co2', 'rh', 'humidity', 'power', 'flow', 'occupancy', 'air']
    
    selected_dfs = []
    
    for file in all_csv_files:
        file_name_lower = os.path.basename(file).lower()
        
        # Filter files that contain relevant sensor data
        if any(keyword in file_name_lower for keyword in target_keywords):
            try:
                temp_df = pd.read_csv(file, low_memory=False)
                # Standardize column names to lowercase strings
                temp_df.columns = [str(col).strip().lower() for col in temp_df.columns]
                selected_dfs.append(temp_df)
            except Exception:
                continue

    if not selected_dfs:
        print("Warning: No keyword-matched files found. Attempting to load primary CSV files...")
        for file in all_csv_files[:10]:
            try:
                temp_df = pd.read_csv(file, low_memory=False)
                temp_df.columns = [str(col).strip().lower() for col in temp_df.columns]
                selected_dfs.append(temp_df)
            except Exception:
                continue

    if not selected_dfs:
        raise ValueError("Could not read any valid CSV files from the extracted archive.")

    print(f"Successfully loaded {len(selected_dfs)} relevant data tables. Merging...")
    
    # Concatenate datasets safely
    combined_df = pd.concat(selected_dfs, axis=0, ignore_index=True)
    return combined_df


def preprocess_building59(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, maps, and structures variables for LSTM + QPSO/PPO research.
    Applies strict 1D Series checks to fix 'arg must be a list, tuple, 1-d array, or Series'.
    """
    print("\n--- Preprocessing & Mapping Building 59 Features ---")
    print(f"Raw Combined Shape: {df.shape}")

    # Map variant Building 59 sensor naming conventions to standardized variables
    column_mapping = {}
    for col in df.columns:
        col_str = str(col).lower()
        if 'co2' in col_str:
            column_mapping[col] = 'co2'
        elif 'temp' in col_str or 'ta' in col_str:
            column_mapping[col] = 'temperature'
        elif 'rh' in col_str or 'hum' in col_str:
            column_mapping[col] = 'humidity'
        elif 'power' in col_str or 'kw' in col_str or 'watt' in col_str:
            column_mapping[col] = 'power_kw'
        elif 'occ' in col_str or 'count' in col_str:
            column_mapping[col] = 'occupancy'

    df_mapped = df.rename(columns=column_mapping)
    
    # Standard features needed for your research
    core_features = ['time', 'timestamp', 'co2', 'temperature', 'humidity', 'power_kw', 'occupancy']
    existing_features = [col for col in core_features if col in df_mapped.columns]
    
    if len(existing_features) > 0:
        df_clean = df_mapped[existing_features].copy()
    else:
        df_clean = df_mapped.copy()

    # Deduplicate duplicate column names to ensure every column selection returns a 1D Series
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()].copy()

    # Identify numeric columns
    numeric_cols = [c for c in df_clean.columns if c not in ['time', 'timestamp']]

    # Safely convert each column to numeric 1D Series
    for col in numeric_cols:
        series_data = df_clean[col]
        # Force single column extraction if duplicate exists
        if isinstance(series_data, pd.DataFrame):
            series_data = series_data.iloc[:, 0]
        df_clean[col] = pd.to_numeric(series_data, errors='coerce')

    # Drop rows where all numerical values are NaN
    if len(numeric_cols) > 0:
        df_clean = df_clean.dropna(how='all', subset=numeric_cols)

    # Clean numeric data using linear interpolation
    for col in numeric_cols:
        if df_clean[col].notnull().any():
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
            # Fill remaining boundary NaNs with median
            median_val = df_clean[col].median()
            if pd.notnull(median_val):
                df_clean[col] = df_clean[col].fillna(median_val)

    print(f"Cleaned Dataset Shape: {df_clean.shape}")
    return df_clean


def generate_summary_statistics(df: pd.DataFrame) -> None:
    """Safely prints descriptive statistics without crashing."""
    print("\n--- Processed Building 59 Summary Statistics ---")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or numeric_df.shape[1] == 0:
        print("Warning: No numeric sensor columns available for summary.")
        return

    summary = numeric_df.describe().T
    stat_cols = [c for c in ['mean', 'std', 'min', '50%', 'max'] if c in summary.columns]
    print(summary[stat_cols])


def main():
    try:
        extract_zip(ZIP_FILE_PATH, EXTRACT_DIR)
        combined_data = identify_and_merge_csvs(EXTRACT_DIR)
        clean_data = preprocess_building59(combined_data)
        generate_summary_statistics(clean_data)
        
        clean_data.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSUCCESS! Clean research-ready dataset saved to: '{OUTPUT_CSV}'")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{ZIP_FILE_PATH}'. Make sure the ZIP file is in this folder and the name matches ZIP_FILE_PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()