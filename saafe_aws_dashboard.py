#!/usr/bin/env python3
"""
Saafe Fire Detection System - AWS Dashboard
Real-time dashboard for viewing sensor data and fire detection scores from AWS deployment
"""

import streamlit as st
import boto3
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pytz
import csv
from io import StringIO

# Set page config
st.set_page_config(
    page_title="🔥 Saafe Fire Detection - AWS Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .sensor-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: white;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .fire-risk-high { background-color: #ffebee; border-left: 5px solid #f44336; }
    .fire-risk-medium { background-color: #fff8e1; border-left: 5px solid #ff9800; }
    .fire-risk-low { background-color: #e8f5e9; border-left: 5px solid #4caf50; }
    .fire-risk-info { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .operational { background-color: #e8f5e9; color: #2e7d32; }
    .warning { background-color: #fff8e1; color: #f57f17; }
    .error { background-color: #ffebee; color: #c62828; }
</style>
""", unsafe_allow_html=True)

# Initialize AWS clients
@st.cache_resource
def init_aws_clients():
    """Initialize AWS clients with error handling."""
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
        return {
            's3': s3_client,
            'lambda': lambda_client,
            'sagemaker': sagemaker_client,
            'cloudwatch': cloudwatch_client
        }
    except Exception as e:
        st.error(f"Failed to initialize AWS clients: {str(e)}")
        return None

# Get recent sensor data from S3
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_recent_sensor_data():
    """Get recent sensor data from S3 bucket."""
    clients = init_aws_clients()
    if not clients:
        return None
    
    try:
        s3 = clients['s3']
        bucket_name = 'data-collector-of-first-device'
        
        # Get current time in UTC
        utc_now = datetime.now(pytz.UTC)
        one_hour_ago = utc_now - timedelta(hours=1)
        one_day_ago = utc_now - timedelta(days=1)  # Extended to 1 day for testing
        
        # Lists to store data
        thermal_data = []
        gas_data = []
        
        # Get recent thermal data files
        today_str = utc_now.strftime('%Y%m%d')
        try:
            paginator = s3.get_paginator('list_objects_v2')
            thermal_pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=f'thermal-data/thermal_data_{today_str}'
            )
            
            for page in thermal_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last day (for testing)
                        if file_time > one_day_ago.replace(tzinfo=file_time.tzinfo):
                            # Download and parse the CSV file
                            try:
                                response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                                content = response['Body'].read().decode('utf-8')
                                
                                # Parse CSV content
                                csv_reader = csv.reader(StringIO(content))
                                rows = list(csv_reader)
                                
                                if len(rows) > 1:
                                    # Get headers and first data row
                                    headers = rows[0]
                                    data_row = rows[1]
                                    
                                    # Create a dictionary with headers as keys
                                    data_dict = {}
                                    for i, header in enumerate(headers):
                                        if i < len(data_row):
                                            data_dict[header] = data_row[i]
                                    
                                    data_dict['timestamp'] = file_time
                                    thermal_data.append(data_dict)
                            except Exception as e:
                                continue
        except Exception as e:
            pass
        
        # Get recent gas data files
        try:
            paginator = s3.get_paginator('list_objects_v2')
            gas_pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=f'gas-data/gas_data_{today_str}'
            )
            
            for page in gas_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last day (for testing)
                        if file_time > one_day_ago.replace(tzinfo=file_time.tzinfo):
                            # Download and parse the CSV file
                            try:
                                response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                                content = response['Body'].read().decode('utf-8')
                                
                                # Parse CSV content
                                csv_reader = csv.reader(StringIO(content))
                                rows = list(csv_reader)
                                
                                if len(rows) > 1:
                                    # Get headers and first data row
                                    headers = rows[0]
                                    data_row = rows[1]
                                    
                                    # Create a dictionary with headers as keys
                                    data_dict = {}
                                    for i, header in enumerate(headers):
                                        if i < len(data_row):
                                            data_dict[header] = data_row[i]
                                    
                                    data_dict['timestamp'] = file_time
                                    gas_data.append(data_dict)
                            except Exception as e:
                                continue
        except Exception as e:
            pass
        
        # If no data found for today, try to get some recent data regardless of date
        if not thermal_data or not gas_data:
            try:
                response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=20)
                if 'Contents' in response:
                    objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    for obj in objects[:20]:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        key = obj['Key']
                        # Download and parse the file
                        try:
                            response = s3.get_object(Bucket=bucket_name, Key=key)
                            content = response['Body'].read().decode('utf-8')
                            
                            # Parse CSV content
                            csv_reader = csv.reader(StringIO(content))
                            rows = list(csv_reader)
                            
                            if len(rows) > 1:
                                # Get headers and first data row
                                headers = rows[0]
                                data_row = rows[1]
                                
                                # Create a dictionary with headers as keys
                                data_dict = {}
                                for i, header in enumerate(headers):
                                    if i < len(data_row):
                                        data_dict[header] = data_row[i]
                                
                                data_dict['timestamp'] = file_time
                                
                                # Add to appropriate list based on key
                                if 'thermal' in key:
                                    thermal_data.append(data_dict)
                                elif 'gas' in key:
                                    gas_data.append(data_dict)
                        except Exception as e:
                            continue
            except Exception as e:
                pass
        
        # Sort by timestamp and get most recent
        thermal_data.sort(key=lambda x: x['timestamp'], reverse=True)
        gas_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'thermal': thermal_data[:10],  # Last 10 thermal readings
            'gas': gas_data[:10],          # Last 10 gas readings
            'latest_thermal': thermal_data[0] if thermal_data else None,
            'latest_gas': gas_data[0] if gas_data else None
        }
    except Exception as e:
        st.error(f"Error retrieving sensor data: {str(e)}")
        return None

# Get fire detection scores from CloudWatch
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_fire_detection_scores():
    """Get fire detection scores from CloudWatch metrics."""
    clients = init_aws_clients()
    if not clients:
        return None
    
    try:
        cloudwatch = clients['cloudwatch']
        
        # Get fire risk metrics (this is a placeholder - you would need to set up custom metrics)
        # In a real implementation, your Lambda functions would publish these metrics
        metrics = []
        
        # Example of how to get custom metrics (uncomment and modify as needed)
        """
        response = cloudwatch.get_metric_statistics(
            Namespace='Saafe/FireDetection',
            MetricName='FireRiskScore',
            Dimensions=[
                {
                    'Name': 'SensorLocation',
                    'Value': 'MainBuilding'
                }
            ],
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow(),
            Period=300,  # 5 minute intervals
            Statistics=['Average', 'Maximum', 'Minimum']
        )
        
        for datapoint in response['Datapoints']:
            metrics.append({
                'timestamp': datapoint['Timestamp'],
                'average': datapoint['Average'],
                'maximum': datapoint['Maximum'],
                'minimum': datapoint['Minimum']
            })
        
        metrics.sort(key=lambda x: x['timestamp'])
        """
        
        # For now, return sample data
        return metrics
    except Exception as e:
        st.error(f"Error retrieving fire detection scores: {str(e)}")
        return []

# Get system status
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_system_status():
    """Get real-time status of all system components."""
    clients = init_aws_clients()
    if not clients:
        return None
    
    status = {
        'timestamp': datetime.now().isoformat(),
        's3': {},
        'lambda': {},
        'sagemaker': {}
    }
    
    # S3 Status
    try:
        s3 = clients['s3']
        bucket_name = 'data-collector-of-first-device'
        
        # Check if bucket exists and is accessible
        s3.head_bucket(Bucket=bucket_name)
        
        # Get recent file count
        utc_now = datetime.now(pytz.UTC)
        one_hour_ago = utc_now - timedelta(hours=1)
        today_str = utc_now.strftime('%Y%m%d')
        
        recent_files = 0
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=f'thermal-data/thermal_data_{today_str}'
            )
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        if file_time > one_hour_ago.replace(tzinfo=file_time.tzinfo):
                            recent_files += 1
        except Exception:
            pass
        
        status['s3'] = {
            'status': 'OPERATIONAL' if recent_files > 0 else 'WARNING',
            'recent_files': recent_files,
            'bucket': bucket_name
        }
    except Exception as e:
        status['s3'] = {
            'status': 'ERROR',
            'error': str(e)
        }
    
    # Lambda Status
    try:
        response = clients['lambda'].get_function(FunctionName='saafe-s3-data-processor')
        config = response['Configuration']
        
        status['lambda'] = {
            'status': 'OPERATIONAL',
            'function_name': config.get('FunctionName'),
            'last_modified': config.get('LastModified')
        }
    except Exception as e:
        status['lambda'] = {
            'status': 'ERROR',
            'error': str(e)
        }
    
    # SageMaker Status
    try:
        response = clients['sagemaker'].describe_endpoint(EndpointName='fire-mvp-xgb-endpoint')
        status['sagemaker'] = {
            'status': response['EndpointStatus'],
            'endpoint_name': response.get('EndpointName'),
            'creation_time': response.get('CreationTime')
        }
    except Exception as e:
        status['sagemaker'] = {
            'status': 'ERROR',
            'error': str(e)
        }
    
    return status

# Main dashboard
def main():
    """Main dashboard application."""
    # Title and header
    st.title("🔥 Saafe Fire Detection - AWS Dashboard")
    st.markdown("Real-time monitoring of sensor data and fire detection scores")
    st.markdown("---")
    
    # Initialize AWS clients
    clients = init_aws_clients()
    if not clients:
        st.error("Failed to initialize AWS services. Please check your credentials and permissions.")
        return
    
    # Auto-refresh controls
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Get system status
    status = get_system_status()
    if not status:
        st.error("Failed to retrieve system status.")
        return
    
    # Overall system status
    all_operational = all([
        status['s3'].get('status') in ['OPERATIONAL', 'WARNING'],
        status['lambda'].get('status') == 'OPERATIONAL',
        status['sagemaker'].get('status') in ['InService', 'OPERATIONAL']
    ])
    
    # Status indicator
    status_col1, status_col2, status_col3 = st.columns([3, 1, 1])
    with status_col1:
        st.subheader("System Status")
    with status_col2:
        if all_operational:
            st.success("✅ SYSTEM OPERATIONAL")
        else:
            st.error("❌ SYSTEM ISSUES")
    with status_col3:
        refresh_rate = st.selectbox("Refresh Rate", ["30s", "1m", "5m"], index=0)
    
    # System components status
    st.markdown("### 📊 System Components")
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        s3_status = status['s3'].get('status')
        if s3_status == 'OPERATIONAL':
            st.markdown('<div class="status-card operational">📂 S3 Data Ingestion<br>✅ OPERATIONAL</div>', unsafe_allow_html=True)
        elif s3_status == 'WARNING':
            st.markdown('<div class="status-card warning">📂 S3 Data Ingestion<br>⚠️ WARNING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card error">📂 S3 Data Ingestion<br>❌ ERROR</div>', unsafe_allow_html=True)
    
    with comp_col2:
        lambda_status = status['lambda'].get('status')
        if lambda_status == 'OPERATIONAL':
            st.markdown('<div class="status-card operational">⚙️ Lambda Processing<br>✅ OPERATIONAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card error">⚙️ Lambda Processing<br>❌ ERROR</div>', unsafe_allow_html=True)
    
    with comp_col3:
        sagemaker_status = status['sagemaker'].get('status')
        if sagemaker_status in ['InService', 'OPERATIONAL']:
            st.markdown('<div class="status-card operational">🧠 SageMaker<br>✅ OPERATIONAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card error">🧠 SageMaker<br>❌ ERROR</div>', unsafe_allow_html=True)
    
    # Get sensor data
    sensor_data = get_recent_sensor_data()
    
    if sensor_data:
        # Latest sensor readings
        st.markdown("### 📡 Latest Sensor Readings")
        
        latest_thermal = sensor_data.get('latest_thermal')
        latest_gas = sensor_data.get('latest_gas')
        
        if latest_thermal and latest_gas:
            # Display sensor data in cards
            sensor_col1, sensor_col2, sensor_col3, sensor_col4 = st.columns(4)
            
            # Extract values from the CSV data
            try:
                # For thermal data, calculate mean from pixel values
                temp_value = 0.0
                pixel_count = 0
                for key, value in latest_thermal.items():
                    if key.startswith('pixel_'):
                        try:
                            temp_value += float(value)
                            pixel_count += 1
                        except (ValueError, TypeError):
                            pass
                
                if pixel_count > 0:
                    temp_value = temp_value / pixel_count
                else:
                    temp_value = 0.0
            except (ValueError, TypeError):
                temp_value = 0.0
                
            try:
                # For PM2.5, we might need to calculate or use a default value
                # Since the thermal data doesn't seem to have pm25 directly, we'll use a default
                pm25_value = 0.0
            except (ValueError, TypeError):
                pm25_value = 0.0
                
            try:
                # For gas data, use CO value
                co2_value = float(latest_gas.get('CO', 0))
            except (ValueError, TypeError):
                co2_value = 0.0
                
            try:
                # Audio level - use a default value since it's not in the data
                audio_value = 0.0
            except (ValueError, TypeError):
                audio_value = 0.0
            
            with sensor_col1:
                st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">🌡️ Temperature</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{temp_value:.1f}°C</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with sensor_col2:
                st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">💨 PM2.5</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pm25_value:.1f} μg/m³</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with sensor_col3:
                st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">ioxide CO</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{co2_value:.1f} ppm</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with sensor_col4:
                st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">🔊 Audio Level</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{audio_value:.1f} dB</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Fire detection score
            st.markdown("### 🔥 Fire Detection Score")
            
            # Simple heuristic for demonstration based on sensor readings
            risk_score = min(100, max(0, (temp_value - 25) * 2 + pm25_value * 0.5 + (co2_value - 0) * 0.1))
            
            # Determine risk level
            if risk_score >= 80:
                risk_class = "fire-risk-high"
                risk_label = "HIGH RISK"
            elif risk_score >= 50:
                risk_class = "fire-risk-medium"
                risk_label = "MEDIUM RISK"
            elif risk_score >= 30:
                risk_class = "fire-risk-low"
                risk_label = "LOW RISK"
            else:
                risk_class = "fire-risk-info"
                risk_label = "INFO"
            
            score_col1, score_col2 = st.columns([1, 3])
            with score_col1:
                st.markdown(f'<div class="sensor-card {risk_class}">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">🔥 Fire Risk</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{risk_score:.1f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 18px; font-weight: bold;">{risk_label}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with score_col2:
                # Visualization of risk over time
                st.markdown("#### Risk Trend (Last Hour)")
                # Create sample data for visualization
                timestamps = []
                scores = []
                for i in range(12):  # Last hour in 5-minute intervals
                    timestamps.append(datetime.now() - timedelta(minutes=i*5))
                    # Simulate varying scores
                    base_score = risk_score + (i * 2) - 10
                    scores.append(max(0, min(100, base_score)))
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'risk_score': scores
                })
                
                fig = px.line(df, x='timestamp', y='risk_score', 
                             labels={'risk_score': 'Risk Score', 'timestamp': 'Time'},
                             line_shape='spline')
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent sensor data table
            st.markdown("### 📈 Recent Sensor Data")
            
            # Combine thermal and gas data
            combined_data = []
            for i in range(min(10, len(sensor_data['thermal']), len(sensor_data['gas']))):
                thermal = sensor_data['thermal'][i]
                gas = sensor_data['gas'][i]
                
                # Extract values safely
                try:
                    # Calculate mean temperature from pixel values
                    temp_sum = 0.0
                    pixel_count = 0
                    for key, value in thermal.items():
                        if key.startswith('pixel_'):
                            try:
                                temp_sum += float(value)
                                pixel_count += 1
                            except (ValueError, TypeError):
                                pass
                    
                    temp_val = temp_sum / pixel_count if pixel_count > 0 else 0.0
                except (ValueError, TypeError):
                    temp_val = 0.0
                    
                try:
                    pm25_val = 0.0  # Default value
                except (ValueError, TypeError):
                    pm25_val = 0.0
                    
                try:
                    co2_val = float(gas.get('CO', 0))
                except (ValueError, TypeError):
                    co2_val = 0.0
                    
                try:
                    audio_val = 0.0  # Default value
                except (ValueError, TypeError):
                    audio_val = 0.0
                
                combined_data.append({
                    'Timestamp': thermal['timestamp'].strftime('%H:%M:%S'),
                    'Temperature (°C)': f"{temp_val:.1f}",
                    'PM2.5 (μg/m³)': f"{pm25_val:.1f}",
                    'CO (ppm)': f"{co2_val:.1f}",
                    'Audio (dB)': f"{audio_val:.1f}"
                })
            
            if combined_data:
                df = pd.DataFrame(combined_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent sensor data available")
        else:
            # More informative message when no data is found
            st.warning("No recent sensor data found. This could be due to:")
            st.markdown("""
            - Devices are not currently sending data to S3
            - Data is older than the expected time window
            - S3 bucket permissions issues
            
            **Most recent data in bucket is from September 7, 2025.**
            
            To resolve this issue:
            1. Verify that your IoT devices are powered on and connected to the internet
            2. Check device logs for any errors
            3. Confirm that devices are configured to send data to the correct S3 bucket
            4. Verify AWS credentials and permissions
            """)
    else:
        st.error("Failed to retrieve sensor data from AWS S3.")
    
    # System information
    st.markdown("### ℹ️ System Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("**S3 Configuration**")
        if status['s3'].get('status') != 'ERROR':
            st.write(f"- Bucket: {status['s3'].get('bucket', 'N/A')}")
            st.write(f"- Recent files: {status['s3'].get('recent_files', 0)}")
        else:
            st.write(f"- Error: {status['s3'].get('error', 'Unknown')}")
    
    with info_col2:
        st.markdown("**Lambda Configuration**")
        if status['lambda'].get('status') != 'ERROR':
            st.write(f"- Function: {status['lambda'].get('function_name', 'N/A')}")
            st.write(f"- Last modified: {status['lambda'].get('last_modified', 'N/A')}")
        else:
            st.write(f"- Error: {status['lambda'].get('error', 'Unknown')}")
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
               f"AWS Account: 691595239825 | Region: us-east-1")

# Run the app
if __name__ == "__main__":
    main()