import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_CSV = "building59_processed_clean.csv"
OUTPUT_CSV = "building59_uganda_augmented.csv"


def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load the cleaned Building 59 dataset."""
    print(f"Loading cleaned dataset from: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def apply_tropical_climate_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments time-series sensor data with Ugandan tropical environmental characteristics:
    1. High Ambient Relative Humidity (Elevated to 70% - 85% range)
    2. Low-Cost Optical Sensor Drift & Dust Spikes (PM2.5 / Humidity sensor noise)
    3. Natural / Hybrid Ventilation Dynamics (Window opening air-exchange shocks)
    """
    print("\n--- Applying Tropical Climate & Infrastructure Noise Augmentation ---")
    df_aug = df.copy()
    np.random.seed(42)  # For reproducible research benchmarks
    n_rows = len(df_aug)

    # 1. Elevate Relative Humidity to Tropical Base (70% - 85% shift)
    if 'humidity' in df_aug.columns:
        print("-> Applying tropical relative humidity bias (70% - 85%)...")
        # Shift baseline humidity upwards with diurnal sinusoidal variations
        diurnal_shift = 5.0 * np.sin(np.linspace(0, 12 * np.pi, n_rows))
        df_aug['humidity'] = df_aug['humidity'] + 25.0 + diurnal_shift
        df_aug['humidity'] = df_aug['humidity'].clip(lower=45.0, upper=95.0)
    else:
        # Create synthetic tropical humidity if absent
        df_aug['humidity'] = 75.0 + 8.0 * np.sin(np.linspace(0, 10 * np.pi, n_rows)) + np.random.normal(0, 1.5, n_rows)

    # 2. Simulate Sensor Drift & Dust Spikes (e.g., Harmattan/Dust season degradation)
    print("-> Injecting low-cost optical sensor drift & dust noise...")
    drift_trend = np.linspace(0.0, 15.0, n_rows)  # Gradual calibration loss over time
    dust_spikes = np.random.choice([0, 1], size=n_rows, p=[0.98, 0.02]) * np.random.uniform(20.0, 50.0, n_rows)

    if 'co2' in df_aug.columns:
        df_aug['co2_raw'] = df_aug['co2']  # Keep ground truth for accuracy evaluation
        df_aug['co2'] = df_aug['co2'] + drift_trend + dust_spikes

    # 3. Hybrid / Natural Ventilation Infiltration Effects (Unplanned Air Exchanges)
    print("-> Simulating hybrid natural ventilation shocks (Window opening events)...")
    window_open_events = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    
    if 'temperature' in df_aug.columns:
        # Temperature rapidly adjusts toward ambient outside temp (26°C - 30°C) during open window events
        ambient_temp = 28.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, n_rows))
        df_aug['temperature'] = np.where(
            window_open_events == 1,
            0.6 * df_aug['temperature'] + 0.4 * ambient_temp,
            df_aug['temperature']
        )

    # Indicator feature for hybrid ventilation state
    df_aug['natural_vent_active'] = window_open_events

    print(f"Augmentation complete. Output DataFrame shape: {df_aug.shape}")
    return df_aug


def calculate_pmv_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a simplified Thermal Comfort Index (PMV Proxy) using temperature 
    and humidity features for downstream LSTM + QPSO reward formulation.
    """
    print("\n--- Calculating Thermal Comfort (PMV Proxy Index) ---")
    if 'temperature' in df.columns and 'humidity' in df.columns:
        # Heat Index / Thermal Sensation Proxy formula normalized around 0 (Optimal Comfort)
        temp_c = df['temperature']
        rh = df['humidity']
        
        # Simplified PMV approximation: 0 = Neutral, +3 = Too Hot, -3 = Too Cold
        pmv_proxy = 0.22 * (temp_c - 24.5) + 0.015 * (rh - 50.0)
        df['pmv_proxy'] = np.clip(pmv_proxy, -3.0, 3.0)
        print("PMV Proxy calculation successful.")
    return df


def plot_augmentation_diagnostics(df_orig: pd.DataFrame, df_aug: pd.DataFrame) -> None:
    """Generates visual diagnostic charts comparing raw vs. augmented features."""
    print("\nGenerating visual diagnostic comparison plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    sample_size = min(1000, len(df_aug))
    
    # 1. Temperature Comparison
    if 'temperature' in df_aug.columns:
        axes[0].plot(df_aug['temperature'].iloc[:sample_size], label='Augmented (Ugandan Tropical)', color='crimson')
        if 'temperature' in df_orig.columns:
            axes[0].plot(df_orig['temperature'].iloc[:sample_size], label='Original Baseline', color='gray', alpha=0.6, linestyle='--')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].set_title('Indoor Temperature Dynamics with Natural Ventilation Infiltration')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Relative Humidity Comparison
    if 'humidity' in df_aug.columns:
        axes[1].plot(df_aug['humidity'].iloc[:sample_size], label='Augmented Humidity (High Tropical Shift)', color='teal')
        axes[1].set_ylabel('Relative Humidity (%)')
        axes[1].set_title('Tropical Humidity Elevation (70% - 85% Target Range)')
        axes[1].legend()
        axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. CO2 Sensor Drift & Dust Noise
    if 'co2' in df_aug.columns:
        axes[2].plot(df_aug['co2'].iloc[:sample_size], label='Augmented CO2 (Drift + Dust Noise)', color='darkorange')
        if 'co2_raw' in df_aug.columns:
            axes[2].plot(df_aug['co2_raw'].iloc[:sample_size], label='Ground Truth CO2', color='navy', alpha=0.6, linestyle='--')
        axes[2].set_ylabel('CO2 (ppm)')
        axes[2].set_xlabel('Time Steps (5-min intervals)')
        axes[2].set_title('CO2 Sensor Stream with Drift and Dust Particle Spikes')
        axes[2].legend()
        axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("tropical_augmentation_diagnostics.png", dpi=300)
    print("Diagnostic plot saved as 'tropical_augmentation_diagnostics.png'.")


def main():
    try:
        df_raw = load_cleaned_data(INPUT_CSV)
        df_augmented = apply_tropical_climate_augmentation(df_raw)
        df_augmented = calculate_pmv_proxy(df_augmented)
        
        plot_augmentation_diagnostics(df_raw, df_augmented)
        
        df_augmented.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSUCCESS! Augmented research dataset saved to: '{OUTPUT_CSV}'")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_CSV}'. Ensure previous processing step completed successfully.")
    except Exception as e:
        print(f"An error occurred during data augmentation: {e}")


if __name__ == "__main__":
    main()