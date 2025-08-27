#!/usr/bin/env python3
"""
🔍 Enhanced S3 Dataset Inspector for 50M Fire Detection Training
Analyzes s3://processedd-synthetic-data/cleaned-data/ structure and format
"""

import boto3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError
import warnings
warnings.filterwarnings('ignore')

class DatasetInspector:
    def __init__(self, bucket_name="processedd-synthetic-data", prefix="cleaned-data/"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = None
        self.dataset_info = {}
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3')
            # Test credentials
            self.s3_client.head_bucket(Bucket=bucket_name)
            print("✅ AWS S3 connection established")
        except NoCredentialsError:
            print("❌ AWS credentials not found. Please run: aws configure")
            return
        except ClientError as e:
            print(f"❌ AWS S3 error: {e}")
            return
    
    def analyze_bucket_structure(self):
        """Analyze the complete bucket structure"""
        print(f"\n🔍 ANALYZING S3 BUCKET: s3://{self.bucket_name}/{self.prefix}")
        print("=" * 70)
        
        try:
            # List all objects with pagination
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            files = []
            total_size = 0
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_info = {
                            'key': obj['Key'],
                            'size_bytes': obj['Size'],
                            'size_mb': round(obj['Size'] / (1024*1024), 2),
                            'size_gb': round(obj['Size'] / (1024*1024*1024), 3),
                            'last_modified': obj['LastModified'],
                            'extension': obj['Key'].split('.')[-1].lower() if '.' in obj['Key'] else 'no_ext'
                        }
                        files.append(file_info)
                        total_size += obj['Size']
            
            # Analyze file types and sizes
            self.dataset_info = {
                'total_files': len(files),
                'total_size_gb': round(total_size / (1024*1024*1024), 3),
                'total_size_mb': round(total_size / (1024*1024), 1),
                'files': files
            }
            
            print(f"📊 DATASET OVERVIEW:")
            print(f"   Total files: {len(files):,}")
            print(f"   Total size: {self.dataset_info['total_size_gb']:.2f} GB ({self.dataset_info['total_size_mb']:,.1f} MB)")
            
            # Group by file type
            file_types = {}
            for file in files:
                ext = file['extension']
                if ext not in file_types:
                    file_types[ext] = {'count': 0, 'total_size_mb': 0}
                file_types[ext]['count'] += 1
                file_types[ext]['total_size_mb'] += file['size_mb']
            
            print(f"\n📁 FILE TYPES:")
            for ext, info in sorted(file_types.items(), key=lambda x: x[1]['total_size_mb'], reverse=True):
                print(f"   .{ext}: {info['count']:,} files, {info['total_size_mb']:,.1f} MB")
            
            # Show largest files
            largest_files = sorted(files, key=lambda x: x['size_bytes'], reverse=True)[:10]
            print(f"\n📈 LARGEST FILES:")
            for i, file in enumerate(largest_files, 1):
                print(f"   {i:2d}. {file['key']} ({file['size_mb']:.1f} MB)")
            
            return True
            
        except ClientError as e:
            print(f"❌ Error analyzing bucket: {e}")
            return False
    
    def sample_data_files(self, max_samples=5):
        """Sample and analyze data file contents"""
        print(f"\n🧪 SAMPLING DATA FILES (max {max_samples} files)")
        print("-" * 50)
        
        # Find data files (CSV, Parquet, JSON)
        data_files = [
            f for f in self.dataset_info['files'] 
            if f['extension'] in ['csv', 'parquet', 'json', 'npy', 'npz'] 
            and f['size_mb'] > 0.1  # Skip tiny files
        ]
        
        if not data_files:
            print("❌ No recognizable data files found")
            return {}
        
        # Sort by size and sample the largest ones
        data_files.sort(key=lambda x: x['size_bytes'], reverse=True)
        sample_files = data_files[:max_samples]
        
        sample_results = {}
        
        for i, file_info in enumerate(sample_files, 1):
            key = file_info['key']
            ext = file_info['extension']
            size_mb = file_info['size_mb']
            
            print(f"\n📄 SAMPLING FILE {i}/{len(sample_files)}: {key}")
            print(f"   Size: {size_mb:.1f} MB, Type: .{ext}")
            
            try:
                if ext == 'csv':
                    result = self._sample_csv(key)
                elif ext == 'parquet':
                    result = self._sample_parquet(key)
                elif ext == 'json':
                    result = self._sample_json(key)
                elif ext in ['npy', 'npz']:
                    result = self._sample_numpy(key)
                else:
                    result = {'error': 'Unsupported file type'}
                
                sample_results[key] = result
                
            except Exception as e:
                print(f"   ❌ Error sampling {key}: {e}")
                sample_results[key] = {'error': str(e)}
        
        return sample_results
    
    def _sample_csv(self, key):
        """Sample CSV file"""
        try:
            # Download small portion or read directly
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key,
                Range='bytes=0-1048576'  # First 1MB
            )
            
            # Read as pandas DataFrame
            from io import StringIO
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), nrows=1000)  # Max 1000 rows for sample
            
            result = {
                'format': 'csv',
                'columns': list(df.columns),
                'shape': df.shape,
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records'),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
                'null_counts': df.isnull().sum().to_dict(),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns)
            }
            
            print(f"   ✅ CSV Analysis:")
            print(f"      Columns: {len(df.columns)}")
            print(f"      Sample shape: {df.shape}")
            print(f"      Numeric cols: {len(result['numeric_columns'])}")
            print(f"      Memory usage: {result['memory_usage_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            return {'error': f'CSV sampling failed: {e}'}
    
    def _sample_parquet(self, key):
        """Sample Parquet file"""
        try:
            # Download file to temp location
            temp_file = f"/tmp/{os.path.basename(key)}"
            self.s3_client.download_file(self.bucket_name, key, temp_file)
            
            # Read with pandas
            df = pd.read_parquet(temp_file, nrows=1000)
            
            result = {
                'format': 'parquet',
                'columns': list(df.columns),
                'shape': df.shape,
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records'),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
                'null_counts': df.isnull().sum().to_dict(),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns)
            }
            
            print(f"   ✅ Parquet Analysis:")
            print(f"      Columns: {len(df.columns)}")
            print(f"      Sample shape: {df.shape}")
            print(f"      Numeric cols: {len(result['numeric_columns'])}")
            print(f"      Memory usage: {result['memory_usage_mb']:.2f} MB")
            
            # Cleanup
            os.remove(temp_file)
            return result
            
        except Exception as e:
            return {'error': f'Parquet sampling failed: {e}'}
    
    def _sample_json(self, key):
        """Sample JSON file"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key,
                Range='bytes=0-1048576'  # First 1MB
            )
            
            json_content = response['Body'].read().decode('utf-8')
            
            # Try to parse as JSON Lines or regular JSON
            try:
                # Try JSON Lines format first
                lines = json_content.strip().split('\n')[:1000]
                data = [json.loads(line) for line in lines if line.strip()]
                df = pd.DataFrame(data)
            except:
                # Try regular JSON
                data = json.loads(json_content)
                if isinstance(data, list):
                    df = pd.DataFrame(data[:1000])
                else:
                    df = pd.DataFrame([data])
            
            result = {
                'format': 'json',
                'columns': list(df.columns),
                'shape': df.shape,
                'sample_data': df.head(3).to_dict('records'),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024)
            }
            
            print(f"   ✅ JSON Analysis:")
            print(f"      Columns: {len(df.columns)}")
            print(f"      Sample shape: {df.shape}")
            
            return result
            
        except Exception as e:
            return {'error': f'JSON sampling failed: {e}'}
    
    def _sample_numpy(self, key):
        """Sample NumPy file"""
        try:
            temp_file = f"/tmp/{os.path.basename(key)}"
            self.s3_client.download_file(self.bucket_name, key, temp_file)
            
            if key.endswith('.npz'):
                data = np.load(temp_file)
                arrays = {k: v.shape for k, v in data.items()}
                result = {
                    'format': 'npz',
                    'arrays': arrays,
                    'total_elements': sum(np.prod(shape) for shape in arrays.values())
                }
            else:
                data = np.load(temp_file)
                result = {
                    'format': 'npy',
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'total_elements': data.size,
                    'memory_mb': data.nbytes / (1024*1024)
                }
            
            print(f"   ✅ NumPy Analysis:")
            print(f"      Format: {result['format']}")
            if 'shape' in result:
                print(f"      Shape: {result['shape']}")
            
            os.remove(temp_file)
            return result
            
        except Exception as e:
            return {'error': f'NumPy sampling failed: {e}'}
    
    def estimate_training_requirements(self, sample_results):
        """Estimate training requirements based on data analysis"""
        print(f"\n🎯 TRAINING REQUIREMENTS ESTIMATION")
        print("=" * 50)
        
        total_size_gb = self.dataset_info['total_size_gb']
        total_files = self.dataset_info['total_files']
        
        # Estimate number of samples
        estimated_samples = 0
        sample_shapes = []
        
        for key, result in sample_results.items():
            if 'error' not in result and 'shape' in result:
                file_size_mb = next(f['size_mb'] for f in self.dataset_info['files'] if f['key'] == key)
                sample_size_mb = result.get('memory_usage_mb', file_size_mb * 0.1)
                
                if sample_size_mb > 0:
                    scaling_factor = file_size_mb / sample_size_mb
                    file_samples = int(result['shape'][0] * scaling_factor)
                    estimated_samples += file_samples
                    sample_shapes.append(result['shape'])
        
        # Fallback estimation if no samples worked
        if estimated_samples == 0:
            # Rough estimate: assume 200 bytes per sample average
            estimated_samples = int((total_size_gb * 1024 * 1024 * 1024) / 200)
        
        print(f"📊 DATASET ESTIMATES:")
        print(f"   Total size: {total_size_gb:.2f} GB")
        print(f"   Estimated samples: {estimated_samples:,}")
        print(f"   Files to process: {total_files:,}")
        
        # Training approach recommendations
        print(f"\n🚀 RECOMMENDED TRAINING APPROACH:")
        
        if total_size_gb < 5:
            approach = "Production Ensemble (Local)"
            cost_range = "$100-300"
            time_range = "4-8 hours"
            script = "production_fire_ai_complete.py"
        elif total_size_gb < 20:
            approach = "AWS SageMaker Training"  
            cost_range = "$200-500"
            time_range = "2-6 hours"
            script = "aws_training_pipeline.py"
        else:
            approach = "S3 Streaming Training"
            cost_range = "$300-800"
            time_range = "8-24 hours" 
            script = "production_fire_ensemble_s3.py"
        
        print(f"   🎯 Best approach: {approach}")
        print(f"   💰 Estimated cost: {cost_range}")
        print(f"   ⏱️ Training time: {time_range}")
        print(f"   📄 Use script: {script}")
        
        # Memory requirements
        if sample_shapes:
            avg_features = np.mean([shape[1] if len(shape) > 1 else 1 for shape in sample_shapes])
            memory_per_batch = (10000 * avg_features * 8) / (1024*1024)  # 8 bytes per float64
            
            print(f"\n💾 MEMORY REQUIREMENTS:")
            print(f"   Estimated features per sample: {avg_features:.0f}")
            print(f"   Memory per 10K batch: {memory_per_batch:.1f} MB")
            
            if memory_per_batch > 1000:
                print(f"   ⚠️ Consider reducing batch size or using streaming")
            else:
                print(f"   ✅ Memory requirements look manageable")
        
        return {
            'total_size_gb': total_size_gb,
            'estimated_samples': estimated_samples,
            'recommended_approach': approach,
            'recommended_script': script,
            'cost_range': cost_range,
            'time_range': time_range
        }
    
    def generate_training_config(self, recommendations):
        """Generate training configuration based on analysis"""
        print(f"\n⚙️ GENERATING TRAINING CONFIGURATION")
        print("-" * 40)
        
        config = {
            'dataset': {
                's3_bucket': self.bucket_name,
                's3_prefix': self.prefix,
                'total_size_gb': self.dataset_info['total_size_gb'],
                'total_files': self.dataset_info['total_files'],
                'estimated_samples': recommendations['estimated_samples']
            },
            'training': {
                'recommended_approach': recommendations['recommended_approach'],
                'script_to_use': recommendations['recommended_script'],
                'batch_size': 10000 if self.dataset_info['total_size_gb'] < 10 else 50000,
                'epochs': 100 if self.dataset_info['total_size_gb'] < 5 else 50,
                'validation_split': 0.1,
                'save_checkpoints': True
            },
            'aws': {
                'region': 'us-east-1',
                'instance_type': 'ml.p3.2xlarge' if self.dataset_info['total_size_gb'] < 10 else 'ml.p3.8xlarge',
                'max_runtime_hours': 8 if self.dataset_info['total_size_gb'] < 10 else 24
            },
            'estimated_cost': recommendations['cost_range'],
            'estimated_time': recommendations['time_range'],
            'generated_at': datetime.now().isoformat()
        }
        
        # Save configuration
        config_file = 'fire_detection_50m_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_file}")
        
        return config

def main():
    print("🔥" * 80)
    print("FIRE DETECTION 50M DATASET INSPECTOR")
    print("🔥" * 80)
    
    # Initialize inspector
    inspector = DatasetInspector()
    
    if inspector.s3_client is None:
        return
    
    # Step 1: Analyze bucket structure
    if not inspector.analyze_bucket_structure():
        return
    
    # Step 2: Sample data files
    sample_results = inspector.sample_data_files(max_samples=5)
    
    # Step 3: Estimate training requirements
    recommendations = inspector.estimate_training_requirements(sample_results)
    
    # Step 4: Generate training configuration
    config = inspector.generate_training_config(recommendations)
    
    # Step 5: Next steps
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Review the generated config: fire_detection_50m_config.json")
    print(f"2. Run the recommended script: {recommendations['recommended_script']}")
    print(f"3. Monitor training progress and costs")
    print(f"4. Deploy trained models to production")
    
    print(f"\n✅ Dataset inspection complete!")
    print(f"📊 Ready to train on {recommendations['estimated_samples']:,} samples")

if __name__ == "__main__":
    main()