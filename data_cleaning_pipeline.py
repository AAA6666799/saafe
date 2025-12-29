import pandas as pd
import numpy as np
from pathlib import Path
import boto3
import json
from datetime import datetime

class DataCleaner:
    def __init__(self, data_dir="synthetic datasets"):
        self.data_dir = Path(data_dir)
        self.cleaned_data = {}
        
    def clean_dataset(self, file_path):
        """Clean individual dataset"""
        df = pd.read_csv(file_path)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Convert boolean strings to actual booleans
        if 'is_anomaly' in df.columns:
            df['is_anomaly'] = df['is_anomaly'].map({'True': 1, 'False': 0, True: 1, False: 0})
        
        # Remove outliers (values beyond 3 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'is_anomaly':  # Don't remove anomaly labels
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 3 * std]
        
        return df
    
    def process_all_datasets(self):
        """Process all CSV files in the data directory"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        for file_path in csv_files:
            print(f"Cleaning {file_path.name}...")
            cleaned_df = self.clean_dataset(file_path)
            
            # Save cleaned data
            output_path = f"cleaned_{file_path.name}"
            cleaned_df.to_csv(output_path, index=False)
            
            self.cleaned_data[file_path.stem] = {
                'original_rows': sum(1 for _ in open(file_path)) - 1,
                'cleaned_rows': len(cleaned_df),
                'file_path': output_path
            }
            
            print(f"  Original: {self.cleaned_data[file_path.stem]['original_rows']:,} rows")
            print(f"  Cleaned: {self.cleaned_data[file_path.stem]['cleaned_rows']:,} rows")
    
    def upload_to_s3(self, bucket_name):
        """Upload cleaned data to S3 for SageMaker"""
        s3 = boto3.client('s3')
        
        for dataset_name, info in self.cleaned_data.items():
            key = f"training-data/{info['file_path']}"
            s3.upload_file(info['file_path'], bucket_name, key)
            print(f"Uploaded {info['file_path']} to s3://{bucket_name}/{key}")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process_all_datasets()
    
    # Print summary
    print("\n=== Data Cleaning Summary ===")
    for dataset, info in cleaner.cleaned_data.items():
        reduction = (info['original_rows'] - info['cleaned_rows']) / info['original_rows'] * 100
        print(f"{dataset}: {reduction:.1f}% reduction")