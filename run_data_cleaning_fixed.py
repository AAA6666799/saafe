#!/usr/bin/env python3
import boto3
import pandas as pd
import numpy as np
from io import StringIO
import json
from datetime import datetime

class S3DataCleaner:
    def __init__(self, region='us-east-1'):
        self.s3_client = boto3.client('s3', region_name=region)
        
    def clean_csv_data(self, csv_content):
        """Clean CSV data using pandas"""
        # Read CSV from string
        df = pd.read_csv(StringIO(csv_content))
        
        print(f"Original dataset shape: {df.shape}")
        
        # 1. Remove duplicates
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # 2. Handle missing values
        df = df.dropna()
        print(f"After removing null values: {df.shape}")
        
        # 3. Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Sort by timestamp
            df = df.sort_values('timestamp')
        
        # 4. Convert boolean strings to integers
        if 'is_anomaly' in df.columns:
            df['is_anomaly'] = df['is_anomaly'].map({
                'True': 1, 'False': 0, True: 1, False: 0
            })
        
        # 5. Remove outliers (values beyond 3 standard deviations)
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            mean_val = df['value'].mean()
            std_val = df['value'].std()
            
            if std_val > 0:
                df = df[abs(df['value'] - mean_val) <= 3 * std_val]
                print(f"After removing outliers: {df.shape}")
        
        return df
    
    def process_s3_dataset(self, input_bucket, input_key, output_bucket, output_key):
        """Process a single dataset from S3"""
        try:
            # Download data from S3
            print(f"Processing s3://{input_bucket}/{input_key}")
            response = self.s3_client.get_object(Bucket=input_bucket, Key=input_key)
            csv_content = response['Body'].read().decode('utf-8')
            
            # Clean the data
            cleaned_df = self.clean_csv_data(csv_content)
            
            # Convert back to CSV
            cleaned_csv = cleaned_df.to_csv(index=False)
            
            # Upload cleaned data to S3
            self.s3_client.put_object(
                Bucket=output_bucket,
                Key=output_key,
                Body=cleaned_csv,
                ContentType='text/csv'
            )
            
            print(f"Cleaned data saved to s3://{output_bucket}/{output_key}")
            
            return {
                'status': 'success',
                'original_rows': len(csv_content.split('\n')) - 1,
                'cleaned_rows': len(cleaned_df),
                'reduction_percent': round((1 - len(cleaned_df) / (len(csv_content.split('\n')) - 1)) * 100, 2)
            }
            
        except Exception as e:
            print(f"Error processing {input_key}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

def main():
    # Your actual bucket configuration
    INPUT_BUCKET = "synthetic-data-4"
    OUTPUT_BUCKET = "processedd-synthetic-data"
    
    # Initialize cleaner
    cleaner = S3DataCleaner()
    
    # Define your datasets (based on what we found in your bucket)
    datasets = [
        "datasets/arc_data.csv",
        "datasets/asd_data.csv", 
        "datasets/basement_data.csv",
        "datasets/laundry_data.csv",
        "datasets/voc_data.csv"
    ]
    
    results = {}
    
    print("Starting data cleaning pipeline...")
    print(f"Input bucket: s3://{INPUT_BUCKET}")
    print(f"Output bucket: s3://{OUTPUT_BUCKET}")
    print("-" * 60)
    
    # Process each dataset
    for dataset_path in datasets:
        dataset_name = dataset_path.split('/')[-1].replace('.csv', '')
        output_key = f"cleaned-data/{dataset_name}_cleaned.csv"
        
        print(f"\nProcessing {dataset_name}...")
        result = cleaner.process_s3_dataset(
            INPUT_BUCKET, dataset_path,
            OUTPUT_BUCKET, output_key
        )
        
        results[dataset_name] = result
        print(f"Result: {result}")
        print("-" * 60)
    
    # Create summary report
    summary = {
        'cleaning_timestamp': datetime.now().isoformat(),
        'input_bucket': INPUT_BUCKET,
        'output_bucket': OUTPUT_BUCKET,
        'total_datasets': len(results),
        'successful_cleanings': sum(1 for r in results.values() if r['status'] == 'success'),
        'failed_cleanings': sum(1 for r in results.values() if r['status'] == 'error'),
        'details': results
    }
    
    # Upload summary to S3
    summary_json = json.dumps(summary, indent=2)
    cleaner.s3_client.put_object(
        Bucket=OUTPUT_BUCKET,
        Key='cleaning-reports/summary.json',
        Body=summary_json,
        ContentType='application/json'
    )
    
    print(f"\n=== CLEANING SUMMARY ===")
    print(f"Total datasets processed: {summary['total_datasets']}")
    print(f"Successful: {summary['successful_cleanings']}")
    print(f"Failed: {summary['failed_cleanings']}")
    
    for dataset, result in results.items():
        if result['status'] == 'success':
            print(f"{dataset}: {result['reduction_percent']}% data reduction ({result['original_rows']:,} → {result['cleaned_rows']:,} rows)")
        else:
            print(f"{dataset}: FAILED - {result.get('message', 'Unknown error')}")
    
    print(f"\nCleaning summary saved to s3://{OUTPUT_BUCKET}/cleaning-reports/summary.json")
    print("Your cleaned datasets are ready for training!")

if __name__ == "__main__":
    main()