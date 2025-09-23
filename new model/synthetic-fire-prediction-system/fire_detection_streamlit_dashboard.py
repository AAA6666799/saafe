#!/usr/bin/env python3
"""
Fire Detection System - Real-Time Dashboard
This Streamlit dashboard shows real-time status of the fire detection system.
"""

import streamlit as st
import boto3
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pytz  # Add this import

# Set page config
st.set_page_config(
    page_title="🔥 Fire Detection System Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Initialize AWS clients
@st.cache_resource
def init_aws_clients():
    """Initialize AWS clients with error handling."""
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
        sns_client = boto3.client('sns', region_name='us-east-1')
        logs_client = boto3.client('logs', region_name='us-east-1')
        cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
        return {
            's3': s3_client,
            'lambda': lambda_client,
            'sagemaker': sagemaker_client,
            'sns': sns_client,
            'logs': logs_client,
            'cloudwatch': cloudwatch_client
        }
    except Exception as e:
        st.error(f"Failed to initialize AWS clients: {str(e)}")
        return None

# Helper function to calculate time since last file upload
def get_time_since_last_file(status_data):
    """Calculate human-readable time since last file upload."""
    if not status_data or not status_data.get('s3', {}).get('recent_files'):
        # Try to get the most recent file from all files if no recent files
        s3_data = status_data.get('s3', {})
        if s3_data.get('all_files'):
            # Get the most recent file from all files
            all_files = s3_data['all_files']
            if all_files:
                # Sort by last modified
                sorted_files = sorted(all_files, key=lambda x: x['modified'], reverse=True)
                if sorted_files:
                    # Parse the timestamp
                    try:
                        from datetime import datetime
                        last_modified_str = sorted_files[0]['modified']
                        # Handle different timestamp formats
                        if 'UTC' in last_modified_str:
                            last_modified = datetime.strptime(last_modified_str.replace(' UTC', ''), '%Y-%m-%d %H:%M:%S')
                        else:
                            last_modified = datetime.strptime(last_modified_str, '%Y-%m-%d %H:%M:%S')
                        
                        # Calculate time difference
                        now = datetime.utcnow()  # Use UTC for comparison
                        time_diff = now - last_modified
                        
                        # Convert to human-readable format
                        total_seconds = int(time_diff.total_seconds())
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        
                        if hours > 0:
                            return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
                        elif minutes > 0:
                            return f"{minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}"
                        else:
                            return f"{seconds} second{'s' if seconds != 1 else ''}"
                    except Exception as e:
                        return "unknown time"
        return "unknown time"
    
    # If we have recent files, get the most recent one
    recent_files = status_data['s3']['recent_files']
    if recent_files:
        try:
            # Get the most recent file
            from datetime import datetime
            most_recent = recent_files[0]
            last_modified_str = most_recent['modified']
            
            # Parse the timestamp
            if 'UTC' in last_modified_str:
                last_modified = datetime.strptime(last_modified_str.replace(' UTC', ''), '%Y-%m-%d %H:%M:%S')
            else:
                last_modified = datetime.strptime(last_modified_str, '%Y-%m-%d %H:%M:%S')
            
            # Calculate time difference
            now = datetime.utcnow()  # Use UTC for comparison
            time_diff = now - last_modified
            
            # Convert to human-readable format
            total_seconds = int(time_diff.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if hours > 0:
                return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
            elif minutes > 0:
                return f"{minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}"
            else:
                return f"{seconds} second{'s' if seconds != 1 else ''}"
        except Exception as e:
            return "unknown time"
    
    return "unknown time"

# Get system status
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_system_status():
    """Get real-time status of all system components."""
    # Import datetime and timedelta locally for Streamlit context
    from datetime import datetime, timedelta
    
    clients = init_aws_clients()
    if not clients:
        return None
    
    status = {
        'timestamp': datetime.now().isoformat(),
        's3': {},
        'lambda': {},
        'sagemaker': {},
        'sns': {},
        'performance': {}
    }
    
    # S3 Status - Filter for only recent files (last hour)
    try:
        # Initialize S3 client
        s3 = clients['s3']
        import pytz
        
        utc_now = datetime.now(pytz.UTC)
        one_hour_ago = utc_now - timedelta(hours=1)
        
        # Use a more efficient approach to find recent files
        # Instead of listing all files, we'll search for today's files by prefix
        today_str = utc_now.strftime('%Y%m%d')
        recent_files = []
        thermal_files = 0
        gas_files = 0
        
        # Search for gas data files from today
        try:
            paginator = s3.get_paginator('list_objects_v2')
            gas_pages = paginator.paginate(
                Bucket='data-collector-of-first-device',
                Prefix=f'gas-data/gas_data_{today_str}'
            )
            
            for page in gas_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last hour
                        if file_time > one_hour_ago.replace(tzinfo=file_time.tzinfo):
                            recent_files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
                            })
                            gas_files += 1
        except Exception as e:
            pass  # Continue even if gas data search fails
        
        # Search for thermal data files from today
        try:
            paginator = s3.get_paginator('list_objects_v2')
            thermal_pages = paginator.paginate(
                Bucket='data-collector-of-first-device',
                Prefix=f'thermal-data/thermal_data_{today_str}'
            )
            
            for page in thermal_pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_time = obj['LastModified']
                        if file_time.tzinfo is None:
                            file_time = file_time.replace(tzinfo=pytz.UTC)
                        
                        # Check if file was modified in the last hour
                        if file_time > one_hour_ago.replace(tzinfo=file_time.tzinfo):
                            recent_files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
                            })
                            thermal_files += 1
        except Exception as e:
            pass  # Continue even if thermal data search fails
        
        # Sort recent files by last modified (most recent first)
        recent_files.sort(key=lambda x: x['modified'], reverse=True)
        
        # Also get total file count for display
        try:
            response = s3.list_objects_v2(
                Bucket='data-collector-of-first-device',
                MaxKeys=1
            )
            total_files = response.get('KeyCount', 0)
            if response.get('IsTruncated', False):
                # If truncated, there are more files
                total_files = '1000+'  # Approximate
        except:
            total_files = 'Unknown'
        
        status['s3'] = {
            'status': 'OPERATIONAL',
            'total_files': total_files,
            'recent_thermal_files': thermal_files,
            'recent_gas_files': gas_files,
            'recent_files': recent_files[:10],  # Show only top 10 recent files
            'has_recent_data': len(recent_files) > 0
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
            'runtime': config.get('Runtime'),
            'timeout': config.get('Timeout'),
            'memory': config.get('MemorySize'),
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
            'creation_time': response.get('CreationTime').strftime('%Y-%m-%d %H:%M:%S UTC') if response.get('CreationTime') else 'N/A'
        }
    except Exception as e:
        status['sagemaker'] = {
            'status': 'ERROR',
            'error': str(e)
        }
    
    # SNS Status
    try:
        response = clients['sns'].get_topic_attributes(TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts')
        subs_response = clients['sns'].list_subscriptions_by_topic(TopicArn='arn:aws:sns:us-east-1:691595239825:fire-detection-alerts')
        subscription_count = len(subs_response.get('Subscriptions', []))
        
        status['sns'] = {
            'status': 'OPERATIONAL',
            'topic_arn': response['Attributes'].get('TopicArn'),
            'subscriptions': subscription_count
        }
    except Exception as e:
        status['sns'] = {
            'status': 'ERROR',
            'error': str(e)
        }
    
    # Performance Metrics (if available)
    try:
        # Import datetime and timedelta locally for Streamlit context
        from datetime import datetime, timedelta
        
        # Get Lambda metrics
        lambda_metrics = clients['cloudwatch'].get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Invocations',
            Dimensions=[{'Name': 'FunctionName', 'Value': 'saafe-s3-data-processor'}],
            StartTime=datetime.now().timestamp() - 3600,  # Last hour
            EndTime=datetime.now().timestamp(),
            Period=300,  # 5 minute periods
            Statistics=['Sum']
        )
        
        invocations = sum([datapoint['Sum'] for datapoint in lambda_metrics.get('Datapoints', [])])
        
        status['performance'] = {
            'lambda_invocations_last_hour': invocations
        }
    except Exception as e:
        status['performance'] = {
            'error': str(e)
        }
    
    return status

# Main dashboard
def main():
    """Main dashboard application."""
    # Title and header
    st.title("🔥 Fire Detection System Dashboard")
    st.markdown("---")
    
    # Initialize AWS clients
    clients = init_aws_clients()
    if not clients:
        st.error("Failed to initialize AWS services. Please check your credentials and permissions.")
        return
    
    # Auto-refresh
    refresh_placeholder = st.empty()
    status = get_system_status()
    
    if not status:
        st.error("Failed to retrieve system status. Please check your AWS configuration.")
        return
    
    # Overall system status
    all_operational = all([
        status['s3'].get('status') == 'OPERATIONAL',
        status['lambda'].get('status') == 'OPERATIONAL',
        status['sagemaker'].get('status') in ['InService', 'OPERATIONAL'],
        status['sns'].get('status') == 'OPERATIONAL'
    ])
    
    # Status indicator
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("System Overview")
    with col2:
        if all_operational:
            st.success("✅ SYSTEM OPERATIONAL")
        else:
            st.error("❌ SYSTEM ISSUES")
    with col3:
        if st.button("🔄 Manual Refresh", key="manual_refresh"):
            # Clear any cached data and force refresh
            if 'last_sensor_timestamp' in st.session_state:
                del st.session_state.last_sensor_timestamp
            st.rerun()
    
    # Last updated
    from datetime import datetime
    st.caption(f"Last updated: {datetime.fromisoformat(status['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # System architecture diagram
    st.markdown("### 🏗️ System Architecture")
    st.image("https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-end-to-end-mlops/main/images/mlops-architecture.png", 
             caption="Fire Detection System Architecture", use_column_width=True)
    
    # Data flow visualization
    st.markdown("### 🔄 Data Flow")
    flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
    
    with flow_col1:
        if status['s3'].get('status') == 'OPERATIONAL':
            st.success("📡\nDevices\n✅")
        else:
            st.error("📡\nDevices\n❌")
    
    with flow_col2:
        if status['s3'].get('status') == 'OPERATIONAL':
            st.success("📂\nS3\n✅")
        else:
            st.error("📂\nS3\n❌")
    
    with flow_col3:
        if status['lambda'].get('status') == 'OPERATIONAL':
            st.success("⚙️\nLambda\n✅")
        else:
            st.error("⚙️\nLambda\n❌")
    
    with flow_col4:
        if status['sagemaker'].get('status') in ['InService', 'OPERATIONAL']:
            st.success("🧠\nSageMaker\n✅")
        else:
            st.error("🧠\nSageMaker\n❌")
    
    with flow_col5:
        if status['sns'].get('status') == 'OPERATIONAL':
            st.success("🚨\nSNS\n✅")
        else:
            st.error("🚨\nSNS\n❌")
    
    # Detailed component status
    st.markdown("### 📊 Component Status")
    
    # S3 Status
    with st.expander("📂 S3 Data Ingestion - LIVE DATA FOCUS", expanded=True):
        if status['s3'].get('status') == 'OPERATIONAL':
            st.success("✅ OPERATIONAL")
            st.write(f"**Bucket:** data-collector-of-first-device")
            st.write(f"**Total Files in Bucket:** {status['s3'].get('total_files', 0)}")
            
            # Show live data status
            if status['s3'].get('has_recent_data', False):
                st.success(f"✅ LIVE DATA DETECTED")
                st.write(f"**Recent Thermal Files (Last Hour):** {status['s3'].get('recent_thermal_files', 0)}")
                st.write(f"**Recent Gas Files (Last Hour):** {status['s3'].get('recent_gas_files', 0)}")
                
                # Recent files
                if status['s3'].get('recent_files'):
                    st.write("**Recent Files (Last Hour):**")
                    df = pd.DataFrame(status['s3']['recent_files'])
                    st.dataframe(df, use_container_width=True)
            else:
                time_since_last = get_time_since_last_file(status)
                st.warning("⚠️ NO RECENT LIVE DATA")
                
                # More detailed explanation
                st.info(f"""
                **No files have been uploaded to the bucket in the last hour.** 
                The most recent file was uploaded {time_since_last} ago. 
                
                **Expected behavior for live system:**
                - Devices should send data every second/minute
                - New files should appear in S3 bucket frequently
                - Dashboard should show "LIVE DATA DETECTED"
                
                **Current status indicates:**
                - Devices are not currently sending data to S3
                - This may be due to connectivity, power, or configuration issues
                """)
                
                # Show the most recent file even if it's old
                if status['s3'].get('all_files'):
                    st.write("**Most Recent File (May be old):**")
                    df = pd.DataFrame(status['s3']['all_files'][:1])
                    st.dataframe(df, use_container_width=True)
                
                # Show additional debugging information
                st.markdown("#### Debugging Information:")
                from datetime import datetime, timedelta
                st.write(f"**Current UTC Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                st.write(f"**One Hour Ago:** {(datetime.utcnow() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
                # Add expected behavior section
                st.markdown("#### Expected vs. Current Behavior:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Expected (Live System):**")
                    st.markdown("- New files every 1-60 seconds")
                    st.markdown("- Dashboard shows green status")
                    st.markdown("- Real-time processing active")
                    st.markdown("- Alerts generated as needed")
                
                with col2:
                    st.markdown("**⚠️ Current (No Live Data):**")
                    st.markdown("- No new files in last hour")
                    st.markdown("- Dashboard shows warning")
                    st.markdown("- Processing inactive")
                    st.markdown("- No alerts generated")
                
                st.markdown("#### Troubleshooting Steps:")
                st.markdown("1. **Verify Device Status**")
                st.markdown("   - Check if devices are powered on")
                st.markdown("   - Confirm internet connectivity")
                st.markdown("   - Review device logs for errors")
                
                st.markdown("2. **Check Device Configuration**")
                st.markdown("   - Verify S3 bucket name: `data-collector-of-first-device`")
                st.markdown("   - Confirm AWS credentials are valid")
                st.markdown("   - Ensure correct file naming format")
                
                st.markdown("3. **Monitor System**")
                st.markdown("   - Refresh this dashboard regularly")
                st.markdown("   - Watch for status change to green")
                st.markdown("   - Check CloudWatch logs for processing activity")
        else:
            st.error(f"❌ ERROR - {status['s3'].get('error', 'Unknown error')}")
    
    # Lambda Status
    with st.expander("⚙️ Lambda Processing", expanded=True):
        if status['lambda'].get('status') == 'OPERATIONAL':
            st.success("✅ OPERATIONAL")
            st.write(f"**Function Name:** {status['lambda'].get('function_name')}")
            st.write(f"**Runtime:** {status['lambda'].get('runtime')}")
            st.write(f"**Memory:** {status['lambda'].get('memory')} MB")
            st.write(f"**Timeout:** {status['lambda'].get('timeout')} seconds")
            st.write(f"**Last Modified:** {status['lambda'].get('last_modified')}")
        else:
            st.error(f"❌ ERROR - {status['lambda'].get('error', 'Unknown error')}")
    
    # SageMaker Status
    with st.expander("🧠 SageMaker Inference", expanded=True):
        if status['sagemaker'].get('status') in ['InService', 'OPERATIONAL']:
            st.success(f"✅ {status['sagemaker'].get('status')}")
            st.write(f"**Endpoint Name:** {status['sagemaker'].get('endpoint_name')}")
            st.write(f"**Creation Time:** {status['sagemaker'].get('creation_time')}")
        else:
            st.error(f"❌ ERROR - {status['sagemaker'].get('error', 'Unknown error')}")
    
    # SNS Status
    with st.expander("🚨 SNS Alerting", expanded=True):
        if status['sns'].get('status') == 'OPERATIONAL':
            st.success("✅ OPERATIONAL")
            st.write(f"**Topic ARN:** {status['sns'].get('topic_arn')}")
            st.write(f"**Subscriptions:** {status['sns'].get('subscriptions', 0)}")
            if status['sns'].get('subscriptions', 0) == 0:
                st.warning("⚠️ No subscriptions configured - alerts will not be sent!")
                st.info("To receive alerts, configure email or SMS subscriptions in the SNS topic.")
        else:
            st.error(f"❌ ERROR - {status['sns'].get('error', 'Unknown error')}")
    
    # Performance Metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Lambda Invocations (Last Hour)", 
                  status['performance'].get('lambda_invocations_last_hour', 'N/A'))
    
    with col2:
        # Calculate data ingestion rate
        if status['s3'].get('has_recent_data', False):
            total_recent = status['s3'].get('recent_thermal_files', 0) + status['s3'].get('recent_gas_files', 0)
            st.metric("Recent Files (Last Hour)", total_recent)
        else:
            st.metric("Recent Files (Last Hour)", 0)
    
    # Alert Levels
    st.markdown("### 🚨 Alert Levels")
    alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
    
    with alert_col1:
        st.markdown("**EMERGENCY**\n\nRisk ≥ 0.8")
        st.error(" ")
    
    with alert_col2:
        st.markdown("**ALERT**\n\nRisk 0.6-0.79")
        st.warning(" ")
    
    with alert_col3:
        st.markdown("**WARNING**\n\nRisk 0.4-0.59")
        st.info(" ")
    
    with alert_col4:
        st.markdown("**INFO**\n\nRisk < 0.4")
        st.success(" ")
    
    # Next Steps
    st.markdown("### 📋 Next Steps")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.subheader("Immediate Actions")
        if not status['s3'].get('has_recent_data', False):
            st.error("1. ⚠️ VERIFY DEVICE CONNECTIVITY - No live data detected")
        elif status['sns'].get('subscriptions', 0) == 0:
            st.warning("1. ⚠️ Configure SNS subscriptions to receive alerts")
        else:
            st.success("1. ✅ System monitoring operational")
        
        st.info("2. Monitor CloudWatch logs for processing activity")
        st.info("3. Test end-to-end data flow with sample data")
    
    with steps_col2:
        st.subheader("Long-term Actions")
        st.info("1. Set up automated monitoring and alerting")
        st.info("2. Document system operations procedures")
        st.info("3. Schedule periodic model retraining")
    
    # Footer
    st.markdown("---")
    from datetime import datetime
    st.caption(f"Dashboard refreshed at: {datetime.fromisoformat(status['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
               f"AWS Account: 691595239825 | Region: us-east-1")

# Run the app
if __name__ == "__main__":
    main()