#!/bin/bash

# Stop Training Utility Script
# Stops all running SageMaker training jobs and cleans up background processes

echo "🛑 Stopping AWS Fire Detection Training Jobs and Background Processes"
echo "====================================================================="

# Function to check if AWS CLI is available
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Cannot stop SageMaker training jobs."
        return 1
    fi
    return 0
}

# Function to stop SageMaker training jobs
stop_sagemaker_jobs() {
    echo "🔍 Checking for running SageMaker training jobs..."
    
    # Get all running training jobs that start with 'fire-' or 'pytorch-training'
    RUNNING_JOBS=$(aws sagemaker list-training-jobs \
        --status-equals InProgress \
        --query 'TrainingJobSummaries[?starts_with(TrainingJobName, `fire-`) || starts_with(TrainingJobName, `pytorch-training`)].TrainingJobName' \
        --output text 2>/dev/null)
    
    if [ -z "$RUNNING_JOBS" ] || [ "$RUNNING_JOBS" = "None" ]; then
        echo "✅ No running fire detection training jobs found"
    else
        echo "📋 Found running training jobs:"
        echo "$RUNNING_JOBS"
        echo ""
        
        # Stop each training job
        for job in $RUNNING_JOBS; do
            echo "🛑 Stopping training job: $job"
            aws sagemaker stop-training-job --training-job-name "$job" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ Stopped: $job"
            else
                echo "❌ Failed to stop: $job"
            fi
        done
    fi
}

# Function to kill background processes
kill_background_processes() {
    echo ""
    echo "🔍 Checking for background training processes..."
    
    # Kill AWS ensemble trainer processes
    TRAINER_PIDS=$(pgrep -f "aws_ensemble_trainer" 2>/dev/null)
    if [ -n "$TRAINER_PIDS" ]; then
        echo "🛑 Stopping AWS ensemble trainer processes: $TRAINER_PIDS"
        pkill -f "aws_ensemble_trainer"
        echo "✅ Stopped ensemble trainer processes"
    else
        echo "✅ No ensemble trainer processes found"
    fi
    
    # Kill monitoring scripts
    MONITOR_PIDS=$(pgrep -f "monitor.*training" 2>/dev/null)
    if [ -n "$MONITOR_PIDS" ]; then
        echo "🛑 Stopping monitoring processes: $MONITOR_PIDS"
        pkill -f "monitor.*training"
        echo "✅ Stopped monitoring processes"
    else
        echo "✅ No monitoring processes found"
    fi
    
    # Kill auto-download scripts
    DOWNLOAD_PIDS=$(pgrep -f "auto_download" 2>/dev/null)
    if [ -n "$DOWNLOAD_PIDS" ]; then
        echo "🛑 Stopping auto-download processes: $DOWNLOAD_PIDS"
        pkill -f "auto_download"
        echo "✅ Stopped auto-download processes"
    else
        echo "✅ No auto-download processes found"
    fi
    
    # Kill any fast training processes
    FAST_TRAINING_PIDS=$(pgrep -f "aws_fast_training" 2>/dev/null)
    if [ -n "$FAST_TRAINING_PIDS" ]; then
        echo "🛑 Stopping fast training processes: $FAST_TRAINING_PIDS"
        pkill -f "aws_fast_training"
        echo "✅ Stopped fast training processes"
    else
        echo "✅ No fast training processes found"
    fi
}

# Function to clean up generated scripts
cleanup_scripts() {
    echo ""
    echo "🧹 Cleaning up auto-generated monitoring scripts..."
    
    # Remove monitoring scripts
    if [ -f "monitor_training.sh" ]; then
        rm -f "monitor_training.sh"
        echo "✅ Removed monitor_training.sh"
    fi
    
    if [ -f "auto_download.sh" ]; then
        rm -f "auto_download.sh"
        echo "✅ Removed auto_download.sh"
    fi
    
    if [ -f "launch_training.py" ]; then
        rm -f "launch_training.py"
        echo "✅ Removed launch_training.py"
    fi
    
    # Remove any temp directories created by training scripts
    if [ -d "temp_training_env" ]; then
        rm -rf "temp_training_env"
        echo "✅ Removed temp_training_env"
    fi
    
    echo "✅ Cleanup completed"
}

# Main execution
main() {
    # Stop SageMaker jobs if AWS CLI is available
    if check_aws_cli; then
        stop_sagemaker_jobs
    fi
    
    # Kill background processes
    kill_background_processes
    
    # Clean up scripts
    cleanup_scripts
    
    echo ""
    echo "🎯 CLEANUP SUMMARY"
    echo "=================="
    echo "✅ All fire detection training processes have been stopped"
    echo "✅ Background monitoring scripts have been terminated"
    echo "✅ Auto-generated scripts have been cleaned up"
    echo ""
    echo "💡 Training should no longer be running repeatedly"
    echo "   You can now start fresh training if needed"
}

# Run the main function
main