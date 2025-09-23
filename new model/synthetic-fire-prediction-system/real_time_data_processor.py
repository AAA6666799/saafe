#!/usr/bin/env python3
"""
Real-time Streaming Data Processor
This script demonstrates how to process high-frequency sensor data in real-time as it arrives in S3.
"""

import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import io
import time
from threading import Thread, Lock
import queue

class RealTimeDataProcessor:
    """Process high-frequency sensor data in real-time."""
    
    def __init__(self, max_workers=5):
        """Initialize the real-time processor."""
        self.s3_client = boto3.client('s3')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        self.sns_client = boto3.client('sns', region_name='us-east-1')
        self.bucket_name = 'data-collector-of-first-device'
        self.endpoint_name = 'fire-mvp-xgb-endpoint'
        self.alert_topic_arn = 'arn:aws:sns:us-east-1:691595239825:fire-detection-alerts'
        
        # Threading components
        self.max_workers = max_workers
        self.workers = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.lock = Lock()
        
        # Processing statistics
        self.stats = {
            'processed': 0,
            'high_risk': 0,
            'errors': 0,
            'start_time': None
        }
    
    def start_processing(self):
        """Start the real-time processing system."""
        print("🚀 Starting real-time data processing system...")
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = Thread(target=self._worker_thread, name=f"Worker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        print(f"✅ Started {self.max_workers} worker threads")
        
        # Start S3 monitoring thread
        monitor = Thread(target=self._monitor_s3, name="S3-Monitor")
        monitor.daemon = True
        monitor.start()
        self.workers.append(monitor)
        
        print("✅ Started S3 monitoring thread")
        print("📊 Real-time processing system is now running...")
    
    def stop_processing(self):
        """Stop the real-time processing system."""
        print("🛑 Stopping real-time data processing system...")
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        print("✅ Real-time processing system stopped")
    
    def _monitor_s3(self):
        """Monitor S3 for new files."""
        last_processed = {}
        
        while self.running:
            try:
                # List objects in the bucket
                response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        last_modified = obj['LastModified']
                        
                        # Check if this is a new file or recently modified
                        if key not in last_processed or last_processed[key] < last_modified:
                            # Add to processing queue
                            self.task_queue.put({
                                'key': key,
                                'last_modified': last_modified
                            })
                            last_processed[key] = last_modified
                            print(f"📥 Queued new file: {key}")
                
                # Wait before next check
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Error monitoring S3: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _worker_thread(self):
        """Worker thread for processing files."""
        while self.running:
            try:
                # Get task from queue (with timeout)
                task = self.task_queue.get(timeout=1)
                
                # Process the file
                result = self._process_file_task(task)
                
                # Put result in result queue
                self.result_queue.put(result)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks available, continue waiting
                continue
            except Exception as e:
                print(f"Error in worker thread: {e}")
                with self.lock:
                    self.stats['errors'] += 1
    
    def _process_file_task(self, task):
        """Process a single file task."""
        file_key = task['key']
        print(f"⚙️  Processing: {file_key}")
        
        try:
            # Download file
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            
            # Parse CSV content
            df = pd.read_csv(io.StringIO(content))
            
            # Process based on file type
            if 'thermal_data' in file_key:
                features = self._process_thermal_data(df)
                # Add default gas features
                gas_features = [400.0, 5.0, 5.0]
                combined_features = features + gas_features
            elif 'gas_data' in file_key:
                gas_features = self._process_gas_data(df)
                # Add default thermal features
                thermal_features = [25.0, 1.0, 30.0, 28.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 30.0, 1.0, 0.5]
                combined_features = thermal_features + gas_features
            else:
                print(f"Unknown file type: {file_key}")
                return None
            
            # Make prediction
            risk_score = self._predict_fire_risk(combined_features)
            
            # Send alert if needed
            if risk_score > 0.4:
                self._send_alert(risk_score, combined_features, file_key)
            
            # Update statistics
            with self.lock:
                self.stats['processed'] += 1
                if risk_score > 0.6:
                    self.stats['high_risk'] += 1
            
            result = {
                "file": file_key,
                "risk_score": risk_score,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            print(f"✅ Processed {file_key} - Risk: {risk_score:.2f}")
            return result
            
        except Exception as e:
            print(f"Error processing {file_key}: {e}")
            with self.lock:
                self.stats['errors'] += 1
            
            return {
                "file": file_key,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def _process_thermal_data(self, df):
        """Extract thermal features from raw data."""
        try:
            # For MLX90640, we expect 32x24 = 768 pixels
            pixel_columns = min(768, len(df.columns))
            pixel_data = df.iloc[:, :pixel_columns].values.flatten()
            
            # Handle case where we have fewer pixels than expected
            if len(pixel_data) < 768:
                padded_data = np.full(768, np.mean(pixel_data))
                padded_data[:len(pixel_data)] = pixel_data
                pixel_data = padded_data
            
            # Calculate thermal features
            features = []
            features.append(float(np.mean(pixel_data)))  # t_mean
            features.append(float(np.std(pixel_data)))   # t_std
            features.append(float(np.max(pixel_data)))   # t_max
            features.append(float(np.percentile(pixel_data, 95)))  # t_p95
            
            # Hot area features
            hot_mask = pixel_data > 40.0
            total_pixels = len(pixel_data)
            hot_pixels = np.sum(hot_mask)
            features.append(float(hot_pixels / total_pixels * 100.0))  # t_hot_area_pct
            features.append(float(hot_pixels / total_pixels * 50.0))   # t_hot_largest_blob_pct
            
            # Gradient features
            grad_x = np.gradient(pixel_data)
            gradient_magnitude = np.abs(grad_x)
            features.append(float(np.mean(gradient_magnitude)))  # t_grad_mean
            features.append(float(np.std(gradient_magnitude)))   # t_grad_std
            
            # Temporal features (using random for demo)
            features.append(float(np.random.normal(0.1, 0.05)))  # t_diff_mean
            features.append(float(np.random.normal(0.05, 0.02))) # t_diff_std
            features.append(float(np.random.normal(0.2, 0.1)))   # flow_mag_mean
            features.append(float(np.random.normal(0.1, 0.05)))  # flow_mag_std
            
            # Temperature proxy features
            features.append(features[2])  # tproxy_val = t_max
            features.append(float(np.random.normal(1.0, 0.5)))   # tproxy_delta
            features.append(float(np.random.normal(0.5, 0.2)))   # tproxy_vel
            
            return features
            
        except Exception as e:
            print(f"Error processing thermal data: {e}")
            return [25.0, 1.0, 30.0, 28.0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 30.0, 1.0, 0.5]
    
    def _process_gas_data(self, df):
        """Extract gas features from raw data."""
        try:
            co_value = df['CO'].iloc[0] if 'CO' in df.columns else 400.0
            no2_value = df['NO2'].iloc[0] if 'NO2' in df.columns else 0.0
            voc_value = df['VOC'].iloc[0] if 'VOC' in df.columns else 0.0
            
            gas_features = []
            gas_features.append(float(co_value))  # gas_val
            gas_features.append(float(np.random.normal(5.0, 2.0)))  # gas_delta
            gas_features.append(gas_features[1])  # gas_vel
            
            return gas_features
            
        except Exception as e:
            print(f"Error processing gas data: {e}")
            return [400.0, 5.0, 5.0]
    
    def _predict_fire_risk(self, features):
        """Send features to SageMaker endpoint for fire risk prediction."""
        try:
            csv_features = ','.join([str(f) for f in features])
            
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='text/csv',
                Body=csv_features
            )
            
            result = float(response['Body'].read().decode())
            return result
            
        except Exception as e:
            print(f"Error predicting fire risk: {e}")
            return 0.0
    
    def _send_alert(self, risk_score, features, file_key):
        """Send alert via SNS if fire risk is detected."""
        # Determine alert level
        if risk_score >= 0.8:
            alert_level = "EMERGENCY"
        elif risk_score >= 0.6:
            alert_level = "ALERT"
        elif risk_score >= 0.4:
            alert_level = "WARNING"
        else:
            return  # Don't send INFO level alerts in real-time processing
        
        alert_message = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": alert_level,
            "risk_score": risk_score,
            "source_file": file_key,
            "message": f"Fire risk detected with probability {risk_score:.2f}"
        }
        
        try:
            self.sns_client.publish(
                TopicArn=self.alert_topic_arn,
                Message=json.dumps(alert_message, indent=2),
                Subject=f"🔥 Fire Detection Alert - {alert_level} (Risk: {risk_score:.2f})"
            )
            print(f"🔔 Alert sent: {alert_level} - Risk score: {risk_score:.2f}")
            
        except Exception as e:
            print(f"Error sending alert: {e}")
    
    def get_statistics(self):
        """Get processing statistics."""
        with self.lock:
            stats = self.stats.copy()
        
        if stats['start_time']:
            uptime = datetime.now() - stats['start_time']
            stats['uptime'] = str(uptime)
            
            if uptime.total_seconds() > 0:
                stats['processing_rate'] = stats['processed'] / (uptime.total_seconds() / 60)  # files per minute
            else:
                stats['processing_rate'] = 0
        
        return stats

def real_time_demo():
    """Demonstrate real-time data processing."""
    print("🔥 Real-Time High-Frequency Data Processing Demo")
    print("=" * 55)
    
    # Initialize processor
    processor = RealTimeDataProcessor(max_workers=3)
    
    try:
        # Start processing
        processor.start_processing()
        
        # Run for 60 seconds to demonstrate
        print("\n📊 Monitoring S3 for new sensor data...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        while time.time() - start_time < 60:  # Run for 60 seconds
            # Print statistics every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                stats = processor.get_statistics()
                print(f"\n📈 Processing Statistics:")
                print(f"  Processed: {stats.get('processed', 0)} files")
                print(f"  High Risk: {stats.get('high_risk', 0)} detections")
                print(f"  Errors: {stats.get('errors', 0)}")
                if 'processing_rate' in stats:
                    print(f"  Rate: {stats['processing_rate']:.1f} files/minute")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping processing...")
    finally:
        processor.stop_processing()
        
        # Print final statistics
        stats = processor.get_statistics()
        print(f"\n🏁 Final Statistics:")
        print(f"  Total Processed: {stats.get('processed', 0)} files")
        print(f"  High Risk Detections: {stats.get('high_risk', 0)}")
        print(f"  Errors: {stats.get('errors', 0)}")
        if 'processing_rate' in stats:
            print(f"  Average Processing Rate: {stats['processing_rate']:.1f} files/minute")
        
        print("\n✅ Real-time processing demo completed!")

if __name__ == "__main__":
    real_time_demo()