#!/usr/bin/env python3
"""
Fire Detection Pattern Analysis Script

This script analyzes data from Grove Multichannel Gas Sensor v2 and 
Grove Thermal Imaging Camera (MLX90640) for fire detection patterns.
"""

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# AWS S3 configuration
S3_BUCKET = 'data-collector-of-first-device'
S3_GAS_PREFIX = 'gas-data/'
S3_THERMAL_PREFIX = 'thermal-data/'

class FireDetectionAnalyzer:
    def __init__(self, bucket_name=S3_BUCKET):
        """Initialize the fire detection analyzer."""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.gas_data = None
        self.thermal_data = None
        
    def download_sample_data(self, num_files=5):
        """
        Download sample data files from S3 for analysis.
        
        Args:
            num_files (int): Number of files to download for each sensor type
        """
        print("🔍 Downloading sample gas data files...")
        gas_files = self._list_s3_files(S3_GAS_PREFIX)
        for file_key in gas_files[:num_files]:
            local_filename = f"gas_{file_key.split('/')[-1]}"
            self.s3_client.download_file(self.bucket_name, file_key, local_filename)
            print(f"   Downloaded {local_filename}")
        
        print("🔍 Downloading sample thermal data files...")
        thermal_files = self._list_s3_files(S3_THERMAL_PREFIX)
        for file_key in thermal_files[:num_files]:
            local_filename = f"thermal_{file_key.split('/')[-1]}"
            self.s3_client.download_file(self.bucket_name, file_key, local_filename)
            print(f"   Downloaded {local_filename}")
    
    def _list_s3_files(self, prefix):
        """
        List files in S3 bucket with given prefix.
        
        Args:
            prefix (str): S3 prefix to filter files
            
        Returns:
            list: List of file keys
        """
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
        
        return files
    
    def load_gas_data(self, num_files=10):
        """
        Load gas sensor data from downloaded CSV files.
        
        Args:
            num_files (int): Number of files to load
        """
        print("📊 Loading gas sensor data...")
        gas_files = [f for f in os.listdir('.') if f.startswith('gas_') and f.endswith('.csv')]
        
        dataframes = []
        for file in gas_files[:num_files]:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dataframes.append(df)
            except Exception as e:
                print(f"   ⚠️  Error loading {file}: {e}")
        
        if dataframes:
            self.gas_data = pd.concat(dataframes, ignore_index=True)
            self.gas_data = self.gas_data.sort_values('timestamp').reset_index(drop=True)
            print(f"   ✅ Loaded {len(self.gas_data)} gas sensor readings")
        else:
            print("   ❌ No gas data files found")
    
    def load_thermal_data(self, num_files=10):
        """
        Load thermal sensor data from downloaded CSV files.
        
        Args:
            num_files (int): Number of files to load
        """
        print("🌡️  Loading thermal sensor data...")
        thermal_files = [f for f in os.listdir('.') if f.startswith('thermal_') and f.endswith('.csv')]
        
        dataframes = []
        for file in thermal_files[:num_files]:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dataframes.append(df)
            except Exception as e:
                print(f"   ⚠️  Error loading {file}: {e}")
        
        if dataframes:
            self.thermal_data = pd.concat(dataframes, ignore_index=True)
            self.thermal_data = self.thermal_data.sort_values('timestamp').reset_index(drop=True)
            print(f"   ✅ Loaded {len(self.thermal_data)} thermal sensor readings")
        else:
            print("   ❌ No thermal data files found")
    
    def analyze_gas_patterns(self):
        """Analyze gas sensor data for fire detection patterns."""
        if self.gas_data is None or self.gas_data.empty:
            print("   ❌ No gas data available for analysis")
            return
        
        print("🔬 Analyzing gas sensor patterns...")
        
        # Basic statistics
        print("\n📈 Gas Sensor Data Summary:")
        print(self.gas_data.describe())
        
        # Check for elevated gas levels (potential fire indicators)
        # For CO, normal levels are typically <10 ppm, dangerous at >50 ppm
        co_threshold = 20  # ppm - elevated level indicator
        high_co_readings = self.gas_data[self.gas_data['CO'] > co_threshold]
        
        if not high_co_readings.empty:
            print(f"\n⚠️  Elevated CO levels detected ({len(high_co_readings)} readings above {co_threshold} ppm)")
            print("   Timestamps with elevated CO:")
            for ts in high_co_readings['timestamp'].head(5):
                print(f"     - {ts}")
        else:
            print(f"\n✅ CO levels normal (all readings below {co_threshold} ppm)")
        
        # For NO2, normal levels are typically <0.1 ppm, concerning at >1 ppm
        no2_threshold = 0.5  # ppm - elevated level indicator
        high_no2_readings = self.gas_data[self.gas_data['NO2'] > no2_threshold]
        
        if not high_no2_readings.empty:
            print(f"\n⚠️  Elevated NO2 levels detected ({len(high_no2_readings)} readings above {no2_threshold} ppm)")
            print("   Timestamps with elevated NO2:")
            for ts in high_no2_readings['timestamp'].head(5):
                print(f"     - {ts}")
        else:
            print(f"\n✅ NO2 levels normal (all readings below {no2_threshold} ppm)")
        
        # For VOC, normal levels vary but concerning levels are typically >500 ppb
        voc_threshold = 300  # ppb - elevated level indicator
        high_voc_readings = self.gas_data[self.gas_data['VOC'] > voc_threshold]
        
        if not high_voc_readings.empty:
            print(f"\n⚠️  Elevated VOC levels detected ({len(high_voc_readings)} readings above {voc_threshold} ppb)")
            print("   Timestamps with elevated VOC:")
            for ts in high_voc_readings['timestamp'].head(5):
                print(f"     - {ts}")
        else:
            print(f"\n✅ VOC levels normal (all readings below {voc_threshold} ppb)")
        
        # Plot gas levels over time
        self._plot_gas_data()
    
    def analyze_thermal_patterns(self):
        """Analyze thermal sensor data for fire detection patterns."""
        if self.thermal_data is None or self.thermal_data.empty:
            print("   ❌ No thermal data available for analysis")
            return
        
        print("🔥 Analyzing thermal sensor patterns...")
        
        # Calculate thermal statistics for each reading
        thermal_stats = self.thermal_data.copy()
        pixel_columns = [col for col in self.thermal_data.columns if col.startswith('pixel_')]
        
        if not pixel_columns:
            print("   ❌ No pixel data found in thermal readings")
            return
        
        # Calculate statistics for each thermal image
        thermal_stats['mean_temp'] = self.thermal_data[pixel_columns].mean(axis=1)
        thermal_stats['max_temp'] = self.thermal_data[pixel_columns].max(axis=1)
        thermal_stats['min_temp'] = self.thermal_data[pixel_columns].min(axis=1)
        thermal_stats['temp_std'] = self.thermal_data[pixel_columns].std(axis=1)
        
        # Identify hot spots (potential fire indicators)
        # Normal room temperature is around 20-25°C, concerning at >50°C
        hot_spot_threshold = 40  # °C - elevated temperature indicator
        hot_spots = thermal_stats[thermal_stats['max_temp'] > hot_spot_threshold]
        
        if not hot_spots.empty:
            print(f"\n⚠️  Hot spots detected ({len(hot_spots)} readings with max temp above {hot_spot_threshold}°C)")
            print("   Timestamps with hot spots:")
            for ts, temp in zip(hot_spots['timestamp'].head(5), hot_spots['max_temp'].head(5)):
                print(f"     - {ts}: {temp:.1f}°C")
        else:
            print(f"\n✅ No significant hot spots detected (all max temps below {hot_spot_threshold}°C)")
        
        # Identify temperature variations (potential fire indicators)
        # High standard deviation may indicate uneven heating
        temp_std_threshold = 5  # °C - elevated variation indicator
        high_variation = thermal_stats[thermal_stats['temp_std'] > temp_std_threshold]
        
        if not high_variation.empty:
            print(f"\n⚠️  High temperature variations detected ({len(high_variation)} readings with std dev above {temp_std_threshold}°C)")
            print("   Timestamps with high variation:")
            for ts, std in zip(high_variation['timestamp'].head(5), high_variation['temp_std'].head(5)):
                print(f"     - {ts}: {std:.1f}°C std dev")
        else:
            print(f"\n✅ Temperature variations normal (all std devs below {temp_std_threshold}°C)")
        
        # Plot thermal statistics over time
        self._plot_thermal_data(thermal_stats)
        
        # Analyze thermal image patterns
        self._analyze_thermal_images()
    
    def _plot_gas_data(self):
        """Plot gas sensor data over time."""
        if self.gas_data is None or self.gas_data.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot each gas component
        plt.subplot(3, 1, 1)
        plt.plot(self.gas_data['timestamp'], self.gas_data['CO'], 'b-', label='CO (ppm)')
        plt.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Elevated CO threshold')
        plt.ylabel('CO (ppm)')
        plt.title('Gas Sensor Data Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(self.gas_data['timestamp'], self.gas_data['NO2'], 'g-', label='NO2 (ppm)')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Elevated NO2 threshold')
        plt.ylabel('NO2 (ppm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.gas_data['timestamp'], self.gas_data['VOC'], 'm-', label='VOC (ppb)')
        plt.axhline(y=300, color='r', linestyle='--', alpha=0.7, label='Elevated VOC threshold')
        plt.ylabel('VOC (ppb)')
        plt.xlabel('Timestamp')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gas_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Gas analysis plot saved as 'gas_analysis.png'")
    
    def _plot_thermal_data(self, thermal_stats):
        """Plot thermal statistics over time."""
        plt.figure(figsize=(12, 10))
        
        # Plot mean, max, and min temperatures
        plt.subplot(4, 1, 1)
        plt.plot(thermal_stats['timestamp'], thermal_stats['mean_temp'], 'b-', label='Mean Temperature')
        plt.plot(thermal_stats['timestamp'], thermal_stats['max_temp'], 'r-', label='Max Temperature')
        plt.plot(thermal_stats['timestamp'], thermal_stats['min_temp'], 'g-', label='Min Temperature')
        plt.axhline(y=40, color='r', linestyle='--', alpha=0.7, label='Hot spot threshold')
        plt.ylabel('Temperature (°C)')
        plt.title('Thermal Sensor Statistics Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot temperature standard deviation
        plt.subplot(4, 1, 2)
        plt.plot(thermal_stats['timestamp'], thermal_stats['temp_std'], 'purple', label='Temperature Std Dev')
        plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='High variation threshold')
        plt.ylabel('Std Dev (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot temperature range
        plt.subplot(4, 1, 3)
        temp_range = thermal_stats['max_temp'] - thermal_stats['min_temp']
        plt.plot(thermal_stats['timestamp'], temp_range, 'orange', label='Temperature Range')
        plt.ylabel('Range (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot number of hot pixels (>40°C)
        plt.subplot(4, 1, 4)
        pixel_columns = [col for col in self.thermal_data.columns if col.startswith('pixel_')]
        hot_pixel_counts = (self.thermal_data[pixel_columns] > 40).sum(axis=1)
        plt.plot(thermal_stats['timestamp'], hot_pixel_counts, 'red', label='Hot Pixels (>40°C)')
        plt.ylabel('Count')
        plt.xlabel('Timestamp')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('thermal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Thermal analysis plot saved as 'thermal_analysis.png'")
    
    def _analyze_thermal_images(self):
        """Analyze thermal image patterns."""
        if self.thermal_data is None or self.thermal_data.empty:
            return
        
        # Select a few sample images for analysis
        sample_indices = [0, len(self.thermal_data)//2, -1]  # First, middle, last
        pixel_columns = [col for col in self.thermal_data.columns if col.startswith('pixel_')]
        
        if len(pixel_columns) != 768:  # MLX90640 has 32x24=768 pixels
            print(f"   ⚠️  Unexpected number of pixels: {len(pixel_columns)}")
            return
        
        # Reshape pixel data to 2D array (32x24 for MLX90640)
        height, width = 24, 32
        
        plt.figure(figsize=(15, 5))
        
        for i, idx in enumerate(sample_indices):
            if abs(idx) >= len(self.thermal_data):
                continue
                
            # Get pixel values for this timestamp
            pixel_values = self.thermal_data.iloc[idx][pixel_columns].values
            # Convert to float to ensure proper data type
            pixel_values = pd.to_numeric(pixel_values, errors='coerce').astype(float)
            # Reshape to 2D
            image_2d = pixel_values.reshape(height, width)
            
            # Plot thermal image
            plt.subplot(1, 3, i+1)
            plt.imshow(image_2d, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Temperature (°C)')
            plt.title(f'Thermal Image\n{self.thermal_data.iloc[idx]["timestamp"]}')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')
        
        plt.tight_layout()
        plt.savefig('thermal_images.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   🖼️  Thermal image samples saved as 'thermal_images.png'")
    
    def generate_fire_risk_report(self):
        """Generate a comprehensive fire risk report based on sensor data."""
        print("\n📋 Generating Fire Risk Assessment Report...")
        
        report = []
        report.append("🔥 SAAFE FIRE DETECTION SYSTEM - RISK ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Gas sensor analysis summary
        report.append("📊 GAS SENSOR ANALYSIS")
        report.append("-" * 30)
        
        if self.gas_data is not None and not self.gas_data.empty:
            max_co = self.gas_data['CO'].max()
            max_no2 = self.gas_data['NO2'].max()
            max_voc = self.gas_data['VOC'].max()
            
            report.append(f"Maximum CO Level: {max_co:.2f} ppm")
            report.append(f"Maximum NO2 Level: {max_no2:.2f} ppm")
            report.append(f"Maximum VOC Level: {max_voc:.2f} ppb")
            
            # Risk assessment based on gas levels
            gas_risk = "LOW"
            if max_co > 50 or max_no2 > 1 or max_voc > 500:
                gas_risk = "HIGH"
            elif max_co > 20 or max_no2 > 0.5 or max_voc > 300:
                gas_risk = "MEDIUM"
            
            report.append(f"Gas Risk Level: {gas_risk}")
        else:
            report.append("No gas data available")
        
        report.append("")
        
        # Thermal sensor analysis summary
        report.append("🌡️  THERMAL SENSOR ANALYSIS")
        report.append("-" * 30)
        
        if self.thermal_data is not None and not self.thermal_data.empty:
            pixel_columns = [col for col in self.thermal_data.columns if col.startswith('pixel_')]
            if pixel_columns:
                max_temp = self.thermal_data[pixel_columns].max().max()
                mean_temp = self.thermal_data[pixel_columns].mean().mean()
                
                report.append(f"Maximum Temperature: {max_temp:.2f} °C")
                report.append(f"Average Temperature: {mean_temp:.2f} °C")
                
                # Risk assessment based on temperature
                thermal_risk = "LOW"
                if max_temp > 70:
                    thermal_risk = "HIGH"
                elif max_temp > 50:
                    thermal_risk = "MEDIUM"
                
                report.append(f"Thermal Risk Level: {thermal_risk}")
            else:
                report.append("No thermal pixel data available")
        else:
            report.append("No thermal data available")
        
        report.append("")
        
        # Overall risk assessment
        report.append("🚨 OVERALL FIRE RISK ASSESSMENT")
        report.append("-" * 40)
        
        # Simple risk combination logic
        overall_risk = "LOW"
        gas_data_available = self.gas_data is not None and not self.gas_data.empty
        thermal_data_available = self.thermal_data is not None and not self.thermal_data.empty
        
        if gas_data_available and thermal_data_available:
            # Both sensors available
            gas_levels = [
                self.gas_data['CO'].max() > 50,
                self.gas_data['NO2'].max() > 1,
                self.gas_data['VOC'].max() > 500
            ]
            
            pixel_columns = [col for col in self.thermal_data.columns if col.startswith('pixel_')]
            if pixel_columns:
                max_temp = self.thermal_data[pixel_columns].max().max()
                thermal_risk_high = max_temp > 70
                
                if any(gas_levels) or thermal_risk_high:
                    overall_risk = "HIGH"
                elif any([
                    self.gas_data['CO'].max() > 20,
                    self.gas_data['NO2'].max() > 0.5,
                    self.gas_data['VOC'].max() > 300
                ]) or max_temp > 50:
                    overall_risk = "MEDIUM"
        elif gas_data_available:
            # Only gas data available
            gas_levels = [
                self.gas_data['CO'].max() > 50,
                self.gas_data['NO2'].max() > 1,
                self.gas_data['VOC'].max() > 500
            ]
            
            if any(gas_levels):
                overall_risk = "HIGH"
            elif any([
                self.gas_data['CO'].max() > 20,
                self.gas_data['NO2'].max() > 0.5,
                self.gas_data['VOC'].max() > 300
            ]):
                overall_risk = "MEDIUM"
        elif thermal_data_available:
            # Only thermal data available
            pixel_columns = [col for col in self.thermal_data.columns if col.startswith('pixel_')]
            if pixel_columns:
                max_temp = self.thermal_data[pixel_columns].max().max()
                
                if max_temp > 70:
                    overall_risk = "HIGH"
                elif max_temp > 50:
                    overall_risk = "MEDIUM"
        
        report.append(f"Overall Risk Level: {overall_risk}")
        
        if overall_risk == "HIGH":
            report.append("⚠️  IMMEDIATE ACTION RECOMMENDED")
            report.append("   - Check for potential fire sources")
            report.append("   - Ensure fire suppression systems are ready")
            report.append("   - Consider evacuating the area if necessary")
        elif overall_risk == "MEDIUM":
            report.append("⚠️  INCREASED VIGILANCE RECOMMENDED")
            report.append("   - Monitor sensors closely")
            report.append("   - Check for potential heat sources")
            report.append("   - Ensure fire safety equipment is accessible")
        else:
            report.append("✅ CURRENTLY NO DETECTED FIRE RISK")
            report.append("   - Continue routine monitoring")
        
        report.append("")
        report.append("📝 RECOMMENDATIONS")
        report.append("-" * 20)
        report.append("1. Continue regular monitoring of both gas and thermal sensors")
        report.append("2. Calibrate sensors regularly to ensure accuracy")
        report.append("3. Establish baseline readings for normal conditions")
        report.append("4. Set up automated alerts for threshold crossings")
        report.append("5. Integrate with fire suppression systems for automatic response")
        
        # Save report to file
        with open('fire_risk_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report to console
        print('\n'.join(report))
        print("\n📄 Report saved as 'fire_risk_report.txt'")

def main():
    """Main function to run the fire detection analysis."""
    print("🔥 SAAFE Fire Detection Pattern Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = FireDetectionAnalyzer()
    
    # Download sample data
    analyzer.download_sample_data(num_files=10)
    
    # Load data
    import os
    analyzer.load_gas_data(num_files=10)
    analyzer.load_thermal_data(num_files=10)
    
    # Analyze patterns
    analyzer.analyze_gas_patterns()
    analyzer.analyze_thermal_patterns()
    
    # Generate report
    analyzer.generate_fire_risk_report()
    
    print("\n✅ Analysis complete! Check the generated plots and report for detailed findings.")

if __name__ == "__main__":
    main()