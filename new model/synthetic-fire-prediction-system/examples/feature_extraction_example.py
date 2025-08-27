"""
Example script demonstrating the use of feature extractors and the registry.

This script shows how to use the feature extractors and the registry to extract features
from synthetic fire prediction data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering.feature_extractor_registry import registry
from src.feature_engineering.extractors.thermal.hotspot_detector import HotspotDetector
from src.feature_engineering.extractors.gas.gas_concentration_extractor import GasConcentrationExtractor
from src.feature_engineering.extractors.environmental.temperature_pattern_extractor import TemperaturePatternExtractor
from src.feature_engineering.extractors.temporal.trend_analyzer import TrendAnalyzer
from src.feature_engineering.extractors.temporal.seasonality_detector import SeasonalityDetector
from src.feature_engineering.extractors.temporal.change_point_detector import ChangePointDetector
from src.feature_engineering.extractors.temporal.temporal_anomaly_detector import TemporalAnomalyDetector


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for feature extraction.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame containing synthetic data
    """
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=n_samples)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate time index
    t = np.linspace(0, 10, n_samples)
    
    # Generate temperature data with trend, seasonality, and noise
    trend = 0.01 * t**2
    seasonality = 5 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 1, n_samples)
    temperature = 25 + trend + seasonality + noise
    
    # Add some anomalies
    anomaly_indices = [100, 200, 300, 400, 500]
    for idx in anomaly_indices:
        temperature[idx] += 15
    
    # Add a change point
    change_point_idx = 600
    temperature[change_point_idx:] += 10
    
    # Generate gas concentration data
    gas_concentration = 50 + 30 * np.sin(2 * np.pi * t / 48) + 5 * np.random.normal(0, 1, n_samples)
    
    # Generate humidity data
    humidity = 60 + 20 * np.sin(2 * np.pi * t / 24 + np.pi/3) + 5 * np.random.normal(0, 1, n_samples)
    
    # Generate pressure data
    pressure = 1013 + 5 * np.sin(2 * np.pi * t / 48 + np.pi/6) + 2 * np.random.normal(0, 1, n_samples)
    
    # Generate thermal image data (simplified as 2D arrays)
    thermal_images = []
    for i in range(n_samples):
        # Create a base thermal image
        base_image = np.ones((10, 10)) * temperature[i]
        
        # Add some hotspots
        if i % 100 == 0:
            base_image[3:7, 3:7] += 20
        
        thermal_images.append(base_image)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'gas_concentration': gas_concentration,
        'humidity': humidity,
        'pressure': pressure
    })
    
    # Add thermal images as a separate dictionary
    thermal_data = {
        'timestamps': timestamps,
        'images': thermal_images
    }
    
    return df, thermal_data


def main():
    """
    Main function demonstrating feature extraction.
    """
    print("Generating synthetic data...")
    df, thermal_data = generate_synthetic_data()
    
    print("Registering feature extractors...")
    # Register feature extractors
    registry.register("HotspotDetector", HotspotDetector)
    registry.register("GasConcentrationExtractor", GasConcentrationExtractor)
    registry.register("TemperaturePatternExtractor", TemperaturePatternExtractor)
    registry.register("TrendAnalyzer", TrendAnalyzer)
    registry.register("SeasonalityDetector", SeasonalityDetector)
    registry.register("ChangePointDetector", ChangePointDetector)
    registry.register("TemporalAnomalyDetector", TemporalAnomalyDetector)
    
    print("Available feature extractors:")
    for name in registry.list_extractors():
        print(f"  - {name}")
    
    print("\nExtracting features...")
    
    # Extract thermal features
    hotspot_config = {
        'threshold': 40.0,
        'min_size': 2,
        'max_size': 20
    }
    hotspot_detector = registry.create("HotspotDetector", hotspot_config)
    hotspot_features = hotspot_detector.extract_features(thermal_data)
    print("\nHotspot features:")
    print(f"  - Detected {len(hotspot_features.get('hotspots', []))} hotspots")
    
    # Extract gas features
    gas_config = {
        'gas_column': 'gas_concentration',
        'threshold': 70.0
    }
    gas_extractor = registry.create("GasConcentrationExtractor", gas_config)
    gas_features = gas_extractor.extract_features(df)
    print("\nGas concentration features:")
    print(f"  - Average concentration: {gas_features.get('average_concentration', 0):.2f}")
    
    # Extract environmental features
    env_config = {
        'temperature_column': 'temperature',
        'window_size': 24
    }
    env_extractor = registry.create("TemperaturePatternExtractor", env_config)
    env_features = env_extractor.extract_features(df)
    print("\nTemperature pattern features:")
    print(f"  - Pattern count: {env_features.get('pattern_count', 0)}")
    
    # Extract temporal features
    # 1. Trend analysis
    trend_config = {
        'time_series_column': 'temperature',
        'timestamp_column': 'timestamp',
        'trend_detection_method': 'linear'
    }
    trend_analyzer = registry.create("TrendAnalyzer", trend_config)
    trend_features = trend_analyzer.extract_features(df)
    trend_info = trend_features.get('trend_features', {})
    print("\nTrend features:")
    print(f"  - Direction: {trend_info.get('trend_direction', 'unknown')}")
    print(f"  - Slope: {trend_info.get('trend_slope', 0):.4f}")
    print(f"  - Strength (R²): {trend_info.get('trend_r_squared', 0):.4f}")
    
    # 2. Seasonality detection
    seasonality_config = {
        'time_series_column': 'temperature',
        'timestamp_column': 'timestamp',
        'seasonality_detection_method': 'fft'
    }
    seasonality_detector = registry.create("SeasonalityDetector", seasonality_config)
    seasonality_features = seasonality_detector.extract_features(df)
    seasonality_info = seasonality_features.get('seasonality_features', {})
    print("\nSeasonality features:")
    print(f"  - Has seasonality: {seasonality_info.get('has_seasonality', False)}")
    print(f"  - Period: {seasonality_info.get('seasonality_period', 0)}")
    print(f"  - Strength: {seasonality_info.get('seasonality_strength', 0):.4f}")
    
    # 3. Change point detection
    change_point_config = {
        'time_series_column': 'temperature',
        'timestamp_column': 'timestamp',
        'change_point_detection_method': 'window_based',
        'window_size': 20,
        'threshold': 2.0
    }
    change_point_detector = registry.create("ChangePointDetector", change_point_config)
    change_point_features = change_point_detector.extract_features(df)
    change_points = change_point_features.get('change_point_features', {}).get('change_points', [])
    print("\nChange point features:")
    print(f"  - Detected {len(change_points)} change points")
    if change_points:
        print(f"  - First change point at index {change_points[0].get('index', 0)}")
    
    # 4. Anomaly detection
    anomaly_config = {
        'time_series_column': 'temperature',
        'timestamp_column': 'timestamp',
        'anomaly_detection_method': 'z_score',
        'threshold': 3.0
    }
    anomaly_detector = registry.create("TemporalAnomalyDetector", anomaly_config)
    anomaly_features = anomaly_detector.extract_features(df)
    anomalies = anomaly_features.get('anomaly_features', {}).get('anomalies', [])
    print("\nAnomaly features:")
    print(f"  - Detected {len(anomalies)} anomalies")
    if anomalies:
        print(f"  - First anomaly at index {anomalies[0].get('index', 0)}")
    
    # Create a feature extraction pipeline
    print("\nCreating feature extraction pipeline...")
    pipeline_config = {
        "extractors": [
            {"name": "TrendAnalyzer", "config": trend_config},
            {"name": "SeasonalityDetector", "config": seasonality_config},
            {"name": "ChangePointDetector", "config": change_point_config},
            {"name": "TemporalAnomalyDetector", "config": anomaly_config}
        ]
    }
    pipeline = registry.create_extractor_pipeline(pipeline_config)
    
    # Extract features using the pipeline
    pipeline_features = registry.extract_features(df, pipeline)
    print(f"  - Extracted features using {len(pipeline)} extractors")
    
    # Plot the data and detected features
    print("\nPlotting data and features...")
    plt.figure(figsize=(12, 8))
    
    # Plot temperature data
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['temperature'], label='Temperature')
    
    # Plot change points
    for cp in change_points:
        idx = cp.get('index', 0)
        if 0 <= idx < len(df):
            plt.axvline(x=df['timestamp'][idx], color='r', linestyle='--', alpha=0.7)
    
    # Plot anomalies
    for anomaly in anomalies:
        idx = anomaly.get('index', 0)
        if 0 <= idx < len(df):
            plt.plot(df['timestamp'][idx], df['temperature'][idx], 'ro')
    
    plt.title('Temperature Data with Change Points and Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    # Plot gas concentration data
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['gas_concentration'], label='Gas Concentration')
    plt.title('Gas Concentration Data')
    plt.xlabel('Time')
    plt.ylabel('Concentration (ppm)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_extraction_example.png')
    print("  - Plot saved as 'feature_extraction_example.png'")
    
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()