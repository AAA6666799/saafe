#!/usr/bin/env python3
"""
Debug script to check the actual structure of your datasets
"""

import pandas as pd
import boto3

# Configuration
INPUT_BUCKET = "synthetic-data-4"
REGION = "us-east-1"

def check_dataset_structure():
    """Check the actual column structure of all datasets"""
    
    area_datasets = {
        'kitchen': 'datasets/voc_data.csv',
        'electrical': 'datasets/arc_data.csv', 
        'laundry_hvac': 'datasets/laundry_data.csv',
        'living_bedroom': 'datasets/asd_data.csv',
        'basement_storage': 'datasets/basement_data.csv'
    }
    
    print("🔍 Checking dataset structures...")
    
    for area_name, dataset_file in area_datasets.items():
        print(f"\n📊 {area_name.upper()} ({dataset_file}):")
        
        try:
            # Load just the first few rows to check structure
            df = pd.read_csv(f"s3://{INPUT_BUCKET}/{dataset_file}", nrows=5)
            
            print(f"  Columns: {list(df.columns)}")
            print(f"  Shape: {df.shape}")
            print(f"  Sample data:")
            print(df.head(2).to_string(index=False))
            
        except Exception as e:
            print(f"  ❌ Error loading {dataset_file}: {e}")

if __name__ == "__main__":
    check_dataset_structure()