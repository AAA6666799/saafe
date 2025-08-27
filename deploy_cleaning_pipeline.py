#!/usr/bin/env python3
import boto3
import json
import time
from pathlib import Path

class DataCleaningDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.cloudformation = boto3.client('cloudformation', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        
    def create_s3_buckets(self, input_bucket, output_bucket):
        """Create S3 buckets if they don't exist"""
        buckets_to_create = [input_bucket, output_bucket]
        
        for bucket in buckets_to_create:
            try:
                if self.region == 'us-east-1':
                    self.s3.create_bucket(Bucket=bucket)
                else:
                    self.s3.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                print(f"Created S3 bucket: {bucket}")
            except self.s3.exceptions.BucketAlreadyOwnedByYou:
                print(f"Bucket {bucket} already exists")
            except Exception as e:
                print(f"Error creating bucket {bucket}: {e}")
    
    def deploy_infrastructure(self, stack_name, input_bucket, output_bucket):
        """Deploy CloudFormation stack"""
        template_path = Path("data_cleaning_infrastructure.yaml")
        
        if not template_path.exists():
            print("CloudFormation template not found!")
            return False
        
        with open(template_path, 'r') as f:
            template_body = f.read()
        
        parameters = [
            {'ParameterKey': 'InputBucketName', 'ParameterValue': input_bucket},
            {'ParameterKey': 'OutputBucketName', 'ParameterValue': output_bucket}
        ]
        
        try:
            response = self.cloudformation.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            print(f"Creating CloudFormation stack: {stack_name}")
            print(f"Stack ID: {response['StackId']}")
            
            # Wait for stack creation
            waiter = self.cloudformation.get_waiter('stack_create_complete')
            waiter.wait(StackName=stack_name)
            
            print("Stack created successfully!")
            return True
            
        except self.cloudformation.exceptions.AlreadyExistsException:
            print(f"Stack {stack_name} already exists")
            return True
        except Exception as e:
            print(f"Error creating stack: {e}")
            return False
    
    def upload_cleaning_script(self, output_bucket):
        """Upload the data cleaning script to S3"""
        script_path = Path("batch_data_cleaning.py")
        
        if script_path.exists():
            self.s3.upload_file(
                str(script_path),
                output_bucket,
                'scripts/batch_data_cleaning.py'
            )
            print(f"Uploaded cleaning script to s3://{output_bucket}/scripts/")
    
    def run_cleaning_job(self, input_bucket, output_bucket):
        """Execute the data cleaning process"""
        print("Starting data cleaning process...")
        
        # Import and run the cleaning script
        from batch_data_cleaning import S3DataCleaner
        
        cleaner = S3DataCleaner(region=self.region)
        results = cleaner.clean_all_datasets(input_bucket, output_bucket)
        summary = cleaner.create_cleaning_summary(results, output_bucket)
        
        return summary

def main():
    print("=== AWS Data Cleaning Pipeline Deployment ===")
    
    # Configuration
    REGION = input("Enter AWS region (default: us-east-1): ").strip() or 'us-east-1'
    INPUT_BUCKET = input("Enter input S3 bucket name: ").strip()
    OUTPUT_BUCKET = input("Enter output S3 bucket name: ").strip()
    STACK_NAME = input("Enter CloudFormation stack name (default: data-cleaning-stack): ").strip() or 'data-cleaning-stack'
    
    if not INPUT_BUCKET or not OUTPUT_BUCKET:
        print("Error: Bucket names are required!")
        return
    
    deployer = DataCleaningDeployer(region=REGION)
    
    # Step 1: Create S3 buckets
    print("\n1. Creating S3 buckets...")
    deployer.create_s3_buckets(INPUT_BUCKET, OUTPUT_BUCKET)
    
    # Step 2: Deploy infrastructure (optional)
    deploy_infra = input("\nDeploy CloudFormation infrastructure? (y/n): ").strip().lower()
    if deploy_infra == 'y':
        print("\n2. Deploying infrastructure...")
        deployer.deploy_infrastructure(STACK_NAME, INPUT_BUCKET, OUTPUT_BUCKET)
    
    # Step 3: Upload scripts
    print("\n3. Uploading cleaning scripts...")
    deployer.upload_cleaning_script(OUTPUT_BUCKET)
    
    # Step 4: Run cleaning job
    run_cleaning = input("\nRun data cleaning now? (y/n): ").strip().lower()
    if run_cleaning == 'y':
        print("\n4. Running data cleaning...")
        summary = deployer.run_cleaning_job(INPUT_BUCKET, OUTPUT_BUCKET)
        
        print("\n=== CLEANING COMPLETED ===")
        print(f"Processed {summary['total_datasets']} datasets")
        print(f"Success: {summary['successful_cleanings']}")
        print(f"Failed: {summary['failed_cleanings']}")
        print(f"Results saved to s3://{OUTPUT_BUCKET}/cleaning-reports/")
    
    print("\nDeployment complete! Your cleaned datasets are ready for training.")

if __name__ == "__main__":
    main()