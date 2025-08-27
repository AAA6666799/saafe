import boto3
import json
from datetime import datetime

class AWSDataCleaner:
    def __init__(self, region='us-east-1'):
        self.glue_client = boto3.client('glue', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.region = region
        
    def create_glue_database(self, database_name='sensor_data_db'):
        """Create Glue database for our datasets"""
        try:
            self.glue_client.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': 'Database for sensor anomaly detection datasets'
                }
            )
            print(f"Created Glue database: {database_name}")
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Database {database_name} already exists")
    
    def create_glue_crawler(self, crawler_name, s3_path, database_name='sensor_data_db'):
        """Create Glue crawler to catalog S3 data"""
        crawler_config = {
            'Name': crawler_name,
            'Role': 'arn:aws:iam::YOUR_ACCOUNT:role/GlueServiceRole',  # Replace with your role
            'DatabaseName': database_name,
            'Targets': {
                'S3Targets': [
                    {
                        'Path': s3_path
                    }
                ]
            },
            'SchemaChangePolicy': {
                'UpdateBehavior': 'UPDATE_IN_DATABASE',
                'DeleteBehavior': 'LOG'
            }
        }
        
        try:
            self.glue_client.create_crawler(**crawler_config)
            print(f"Created Glue crawler: {crawler_name}")
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Crawler {crawler_name} already exists")
    
    def create_data_cleaning_job(self, job_name, input_s3_path, output_s3_path):
        """Create Glue ETL job for data cleaning"""
        
        # Glue job script for data cleaning
        cleaning_script = f'''
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read data from S3
df = spark.read.option("header", "true").csv("{input_s3_path}")

# Data cleaning operations
# 1. Remove duplicates
df_clean = df.dropDuplicates()

# 2. Convert timestamp to proper format
df_clean = df_clean.withColumn("timestamp", F.to_timestamp("timestamp"))

# 3. Convert is_anomaly to integer
df_clean = df_clean.withColumn("is_anomaly", 
    F.when(F.col("is_anomaly") == "True", 1)
    .when(F.col("is_anomaly") == "False", 0)
    .otherwise(F.col("is_anomaly").cast("int")))

# 4. Convert value to double and handle nulls
df_clean = df_clean.withColumn("value", F.col("value").cast("double"))
df_clean = df_clean.filter(F.col("value").isNotNull())

# 5. Remove outliers (values beyond 3 standard deviations)
stats = df_clean.select(
    F.mean("value").alias("mean"),
    F.stddev("value").alias("std")
).collect()[0]

mean_val = stats["mean"]
std_val = stats["std"]

if std_val is not None and std_val > 0:
    df_clean = df_clean.filter(
        F.abs(F.col("value") - mean_val) <= 3 * std_val
    )

# 6. Sort by timestamp
df_clean = df_clean.orderBy("timestamp")

# Write cleaned data back to S3
df_clean.write.mode("overwrite").option("header", "true").csv("{output_s3_path}")

job.commit()
'''
        
        # Upload script to S3
        script_key = f"glue-scripts/{job_name}.py"
        script_bucket = output_s3_path.split('/')[2]  # Extract bucket from s3://bucket/path
        
        self.s3_client.put_object(
            Bucket=script_bucket,
            Key=script_key,
            Body=cleaning_script
        )
        
        # Create Glue job
        job_config = {
            'Name': job_name,
            'Role': 'arn:aws:iam::YOUR_ACCOUNT:role/GlueServiceRole',  # Replace with your role
            'Command': {
                'Name': 'glueetl',
                'ScriptLocation': f's3://{script_bucket}/{script_key}',
                'PythonVersion': '3'
            },
            'DefaultArguments': {
                '--job-language': 'python',
                '--job-bookmark-option': 'job-bookmark-disable'
            },
            'MaxRetries': 0,
            'Timeout': 60,
            'GlueVersion': '3.0'
        }
        
        try:
            self.glue_client.create_job(**job_config)
            print(f"Created Glue ETL job: {job_name}")
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Job {job_name} already exists")
    
    def run_cleaning_pipeline(self, input_bucket, output_bucket, datasets):
        """Run complete data cleaning pipeline"""
        
        # Create database
        self.create_glue_database()
        
        # Process each dataset
        for dataset in datasets:
            print(f"\\nProcessing {dataset}...")
            
            # Create crawler for input data
            crawler_name = f"{dataset}_crawler"
            input_s3_path = f"s3://{input_bucket}/synthetic-datasets/{dataset}.csv"
            self.create_glue_crawler(crawler_name, input_s3_path)
            
            # Create cleaning job
            job_name = f"clean_{dataset}_job"
            output_s3_path = f"s3://{output_bucket}/cleaned-data/{dataset}/"
            self.create_data_cleaning_job(job_name, input_s3_path, output_s3_path)
            
            # Run the job
            print(f"Starting cleaning job for {dataset}...")
            self.glue_client.start_job_run(JobName=job_name)
    
    def monitor_jobs(self, job_names):
        """Monitor Glue job status"""
        for job_name in job_names:
            response = self.glue_client.get_job_runs(JobName=job_name, MaxResults=1)
            if response['JobRuns']:
                status = response['JobRuns'][0]['JobRunState']
                print(f"{job_name}: {status}")

if __name__ == "__main__":
    # Configuration
    INPUT_BUCKET = "your-input-bucket"  # Replace with your S3 bucket
    OUTPUT_BUCKET = "your-output-bucket"  # Replace with your output bucket
    DATASETS = ["arc_data", "asd_data", "basement_data", "laundry_data", "voc_data"]
    
    # Initialize cleaner
    cleaner = AWSDataCleaner()
    
    # Run cleaning pipeline
    cleaner.run_cleaning_pipeline(INPUT_BUCKET, OUTPUT_BUCKET, DATASETS)