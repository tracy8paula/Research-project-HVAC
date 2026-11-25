
import os
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "building59_uganda_augmented.csv"
CHUNK_SIZE = 100_000


# --------------------------------------------------
# LOCATE DATASET
# --------------------------------------------------

def locate_file(filename: str) -> str:

    script_directory = Path(__file__).resolve().parent
    project_directory = script_directory.parent

    possible_paths = [
        Path(filename),
        project_directory / filename,
        script_directory / filename
    ]

    for path in possible_paths:

        if path.is_file():
            return str(path.resolve())

    raise FileNotFoundError(
        f"Could not locate {filename}. "
        "Check that the dataset exists in the project directory."
    )


# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

def main():

    path = locate_file(DATA_PATH)

    print("=" * 65)
    print("BUILDING 59 DATASET VALIDATION")
    print("=" * 65)

    print("File:", path)

    file_size = os.path.getsize(path) / (1024 * 1024)

    print("File size (MB):", round(file_size, 2))

    # Read a small sample to identify columns.
    sample = pd.read_csv(path, nrows=10)

    print("\nAVAILABLE COLUMNS:")

    for column in sample.columns:
        print("-", column)

    print("\nSAMPLE RECORDS:")
    print(sample.head())

    # --------------------------------------------------
    # INITIALISE AUDIT VARIABLES
    # --------------------------------------------------

    total_rows = 0

    missing_counts = {}

    numeric_min = {}
    numeric_max = {}
    numeric_sum = {}
    numeric_count = {}

    duplicate_count = 0

    time_candidates = {
        "timestamp",
        "time",
        "datetime",
        "date"
    }

    time_column = next(
        (
            column
            for column in sample.columns
            if str(column).lower() in time_candidates
        ),
        None
    )

    previous_timestamp = None

    time_differences = []

    invalid_timestamps = 0

    # CO2 comparison statistics
    co2_difference_sum = 0.0
    co2_difference_min = float("inf")
    co2_difference_max = float("-inf")
    co2_comparison_count = 0

    # --------------------------------------------------
    # PROCESS DATA IN CHUNKS
    # --------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        total_rows += len(chunk)

        # Missing values
        for column in chunk.columns:

            missing_counts[column] = (
                missing_counts.get(column, 0)
                + int(chunk[column].isna().sum())
            )

        # Duplicate rows within this chunk
        duplicate_count += int(
            chunk.duplicated().sum()
        )

        # Numeric summaries
        numeric_columns = chunk.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_columns:

            values = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if values.empty:
                continue

            numeric_min[column] = min(
                numeric_min.get(column, float("inf")),
                float(values.min())
            )

            numeric_max[column] = max(
                numeric_max.get(column, float("-inf")),
                float(values.max())
            )

            numeric_sum[column] = (
                numeric_sum.get(column, 0.0)
                + float(values.sum())
            )

            numeric_count[column] = (
                numeric_count.get(column, 0)
                + len(values)
            )

        # Timestamp validation
        if time_column is not None:

            parsed_dates = pd.to_datetime(
                chunk[time_column],
                errors="coerce"
            )

            # Convert to a Series so pandas indexing
            # and datetime accessors are consistently typed.
            date_series = pd.Series(
                parsed_dates,
                index=chunk.index
            )

            invalid_timestamps += int(
                date_series.isna().sum()
            )

            valid_dates = date_series.dropna()

            if not valid_dates.empty:

                if previous_timestamp is not None:

                    first_timestamp = valid_dates.iloc[0]

                    difference = (
                        first_timestamp - previous_timestamp
                    ).total_seconds()

                    time_differences.append(
                        float(difference)
                    )

                differences = (
                    valid_dates.diff()
                    .dt.total_seconds()
                    .dropna()
                )

                time_differences.extend(
                    differences.tolist()
                )

                previous_timestamp = valid_dates.iloc[-1]

        # CO2 augmentation comparison
        if {"co2", "co2_raw"}.issubset(chunk.columns):

            original = pd.to_numeric(
                chunk["co2_raw"],
                errors="coerce"
            )

            augmented = pd.to_numeric(
                chunk["co2"],
                errors="coerce"
            )

            valid_mask = (
                original.notna()
                & augmented.notna()
                & np.isfinite(original)
                & np.isfinite(augmented)
            )

            if valid_mask.any():

                differences = (
                    augmented.loc[valid_mask]
                    - original.loc[valid_mask]
                )

                co2_difference_sum += float(
                    differences.sum()
                )

                co2_difference_min = min(
                    co2_difference_min,
                    float(differences.min())
                )

                co2_difference_max = max(
                    co2_difference_max,
                    float(differences.max())
                )

                co2_comparison_count += len(differences)

        print(
            f"Processed chunk {chunk_number} | "
            f"Rows checked: {total_rows:,}",
            flush=True
        )

    # --------------------------------------------------
    # DISPLAY AUDIT RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("DATASET AUDIT RESULTS")
    print("=" * 65)

    print("Total rows:", f"{total_rows:,}")
    print("Total columns:", len(sample.columns))

    print(
        "Timestamp column:",
        time_column if time_column else "Not detected"
    )

    print(
        "Duplicate rows within individual chunks:",
        duplicate_count
    )

    # Missing values
    print("\nMISSING VALUES:")

    for column, count in missing_counts.items():

        percentage = (
            count / total_rows * 100
            if total_rows > 0 else 0
        )

        print(
            f"{column}: {count:,} "
            f"({percentage:.2f}%)"
        )

    # Numeric statistics
    print("\nNUMERIC FEATURE SUMMARY:")

    for column in numeric_min:

        mean = (
            numeric_sum[column]
            / numeric_count[column]
        )

        print(
            f"{column}: "
            f"min={numeric_min[column]:.3f}, "
            f"mean={mean:.3f}, "
            f"max={numeric_max[column]:.3f}"
        )

    # Timestamp results
    if time_column is not None:

        print("\nTIMESTAMP VALIDATION:")

        print(
            "Invalid timestamps:",
            invalid_timestamps
        )

        if time_differences:

            differences_array = np.asarray(
                time_differences,
                dtype=float
            )

            print(
                "Median interval (seconds):",
                np.median(differences_array)
            )

            print(
                "Minimum interval (seconds):",
                np.min(differences_array)
            )

            print(
                "Maximum interval (seconds):",
                np.max(differences_array)
            )

            print(
                "Intervals equal to 300 seconds:",
                int(np.sum(differences_array == 300))
            )

            print(
                "Non-positive intervals:",
                int(np.sum(differences_array <= 0))
            )

    # CO2 augmentation results
    print("\nCO2 AUGMENTATION VALIDATION:")

    if co2_comparison_count > 0:

        mean_difference = (
            co2_difference_sum
            / co2_comparison_count
        )

        print(
            "Valid CO2 comparisons:",
            f"{co2_comparison_count:,}"
        )

        print(
            "Mean added CO2:",
            round(mean_difference, 3)
        )

        print(
            "Minimum added CO2:",
            round(co2_difference_min, 3)
        )

        print(
            "Maximum added CO2:",
            round(co2_difference_max, 3)
        )

    else:

        print(
            "CO2 comparison unavailable. "
            "Both co2 and co2_raw columns were not found "
            "with valid numeric values."
        )

    print("\n" + "=" * 65)
    print("DATASET AUDIT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
