import boto3
import pandas as pd
import numpy as np
from io import StringIO
import json

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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 4. Sort by timestamp
        df = df.sort_values('timestamp')
        
        # 5. Convert boolean strings to integers
        if 'is_anomaly' in df.columns:
            df['is_anomaly'] = df['is_anomaly'].map({
                'True': 1, 'False': 0, True: 1, False: 0
            })
        
        # 6. Remove outliers (values beyond 3 standard deviations)
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
    
    def clean_all_datasets(self, input_bucket, output_bucket, dataset_prefix="synthetic-datasets/"):
        """Clean all datasets in S3 bucket"""
        
        # List all CSV files in the input bucket
        response = self.s3_client.list_objects_v2(
            Bucket=input_bucket,
            Prefix=dataset_prefix
        )
        
        results = {}
        
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.csv'):
                    # Extract dataset name
                    dataset_name = obj['Key'].split('/')[-1].replace('.csv', '')
                    
                    # Define output path
                    output_key = f"cleaned-data/{dataset_name}_cleaned.csv"
                    
                    # Process the dataset
                    result = self.process_s3_dataset(
                        input_bucket, obj['Key'], 
                        output_bucket, output_key
                    )
                    
                    results[dataset_name] = result
                    print(f"Completed {dataset_name}: {result}")
                    print("-" * 50)
        
        return results
    
    def create_cleaning_summary(self, results, output_bucket):
        """Create and upload cleaning summary"""
        summary = {
            'cleaning_timestamp': pd.Timestamp.now().isoformat(),
            'total_datasets': len(results),
            'successful_cleanings': sum(1 for r in results.values() if r['status'] == 'success'),
            'failed_cleanings': sum(1 for r in results.values() if r['status'] == 'error'),
            'details': results
        }
        
        # Upload summary to S3
        summary_json = json.dumps(summary, indent=2)
        self.s3_client.put_object(
            Bucket=output_bucket,
            Key='cleaning-reports/summary.json',
            Body=summary_json,
            ContentType='application/json'
        )
        
        print(f"Cleaning summary saved to s3://{output_bucket}/cleaning-reports/summary.json")
        return summary

def main():
    # Configuration - Update these with your actual bucket names
    INPUT_BUCKET = "your-input-bucket"
    OUTPUT_BUCKET = "your-output-bucket"
    
    # Initialize cleaner
    cleaner = S3DataCleaner()
    
    # Clean all datasets
    print("Starting data cleaning pipeline...")
    results = cleaner.clean_all_datasets(INPUT_BUCKET, OUTPUT_BUCKET)
    
    # Create summary report
    summary = cleaner.create_cleaning_summary(results, OUTPUT_BUCKET)
    
    print("\n=== CLEANING SUMMARY ===")
    print(f"Total datasets processed: {summary['total_datasets']}")
    print(f"Successful: {summary['successful_cleanings']}")
    print(f"Failed: {summary['failed_cleanings']}")
    
    for dataset, result in results.items():
        if result['status'] == 'success':
            print(f"{dataset}: {result['reduction_percent']}% data reduction")

if __name__ == "__main__":
    main()