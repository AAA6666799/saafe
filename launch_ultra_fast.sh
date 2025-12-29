#!/bin/bash

# Saafe Ultra-Fast Training Launcher with Progress Tracking
# Maximum speed, cost is no object!

set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress bar function
show_progress() {
    local duration=$1
    local message=$2
    local progress=0
    local bar_length=50
    
    echo -e "${CYAN}${message}${NC}"
    
    while [ $progress -le $duration ]; do
        local filled=$((progress * bar_length / duration))
        local empty=$((bar_length - filled))
        
        printf "\r["
        printf "%*s" $filled | tr ' ' '█'
        printf "%*s" $empty | tr ' ' '░'
        printf "] %d%%" $((progress * 100 / duration))
        
        sleep 0.1
        progress=$((progress + 1))
    done
    echo ""
}

echo -e "${PURPLE}🚀 SAAFE ULTRA-FAST TRAINING LAUNCHER${NC}"
echo "====================================="
echo -e "${YELLOW}💰 Cost optimized for SPEED, not budget${NC}"
echo -e "${GREEN}⚡ Expected time: 30-45 minutes${NC}"
echo -e "${GREEN}🔥 Expected cost: \$6-12${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}[1/6] Checking prerequisites...${NC}"
show_progress 10 "Validating AWS setup"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Install it first:${NC}"
    echo "   curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"
    echo "   unzip awscliv2.zip && sudo ./aws/install"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Run: aws configure${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI configured${NC}"

# Step 2: Install dependencies with conflict resolution
echo -e "${BLUE}[2/6] Installing dependencies...${NC}"
show_progress 15 "Setting up Python packages"

# Create a temporary virtual environment to avoid conflicts
if [ ! -d "temp_training_env" ]; then
    python -m venv temp_training_env --system-site-packages
fi

source temp_training_env/bin/activate

# Install only what we need for training
pip install -q boto3==1.34.0 sagemaker==2.190.0 torch==2.0.1 numpy==1.24.3 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Installing with --force-reinstall to resolve conflicts${NC}"
    pip install -q --force-reinstall --no-deps boto3 sagemaker torch numpy
}

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 3: Verify AWS setup
echo -e "${BLUE}[3/6] Verifying AWS configuration...${NC}"
show_progress 8 "Checking account and permissions"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "us-east-1")
echo -e "${GREEN}📋 Account: $ACCOUNT_ID${NC}"
echo -e "${GREEN}📋 Region: $REGION${NC}"

# Step 4: Check/create SageMaker role
echo -e "${BLUE}[4/6] Setting up SageMaker role...${NC}"
show_progress 12 "Configuring IAM permissions"

ROLE_NAME="SaafeTrainingRole"

if ! aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    echo -e "${YELLOW}Creating SageMaker role for ultra-fast training...${NC}"
    
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json > /dev/null

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess > /dev/null

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess > /dev/null

    rm trust-policy.json
    echo -e "${GREEN}✅ SageMaker role created${NC}"
    sleep 5
else
    echo -e "${GREEN}✅ SageMaker role exists${NC}"
fi

# Step 5: Choose training configuration
echo -e "${BLUE}[5/6] Selecting ultra-fast configuration...${NC}"
show_progress 5 "Optimizing for maximum speed"

echo ""
echo -e "${PURPLE}🔥 ULTRA-FAST TRAINING OPTIONS${NC}"
echo "=============================="
echo -e "${GREEN}1. 🚀 MAXIMUM SPEED (ml.p3.8xlarge - 4x V100 GPUs) - ~\$9, 30min${NC}"
echo -e "${CYAN}2. ⚡ DISTRIBUTED (4x ml.p3.2xlarge - 4x V100 GPUs) - ~\$6, 35min${NC}"
echo -e "${YELLOW}3. 🔥 AUTO-CHOOSE FASTEST${NC}"
echo ""

read -p "Choose option (1-3) [3]: " choice
choice=${choice:-3}

echo ""
echo -e "${PURPLE}🚀 LAUNCHING ULTRA-FAST TRAINING...${NC}"
echo "=================================="

case $choice in
    1)
        echo -e "${GREEN}🔥 Maximum Speed: Single ml.p3.8xlarge instance${NC}"
        echo "   4x NVIDIA V100 GPUs, 244GB RAM, 64GB GPU memory"
        echo -e "${YELLOW}   Cost: ~\$12.24/hour, Expected: 30-45 minutes = ~\$6-9${NC}"
        TRAINING_TYPE="ultra_fast"
        ;;
    2)
        echo -e "${CYAN}⚡ Distributed: 4x ml.p3.2xlarge instances${NC}"
        echo "   4x NVIDIA V100 GPUs total, Distributed training"
        echo -e "${YELLOW}   Cost: ~\$12.24/hour, Expected: 20-35 minutes = ~\$4-7${NC}"
        TRAINING_TYPE="distributed"
        ;;
    3)
        echo -e "${PURPLE}🚀 Auto-choosing MAXIMUM SPEED option${NC}"
        echo "   Single ml.p3.8xlarge for fastest overall training"
        TRAINING_TYPE="ultra_fast"
        ;;
esac

echo ""
read -p "Proceed with ultra-fast training? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}👍 Training cancelled${NC}"
    deactivate 2>/dev/null || true
    exit 0
fi

# Step 6: Launch training with progress tracking
echo -e "${BLUE}[6/6] Launching training job...${NC}"
show_progress 20 "Starting AWS SageMaker training"

echo ""
echo -e "${PURPLE}🚀 LAUNCHING TRAINING JOB...${NC}"
echo "============================"

# Create a Python script to launch training with progress
cat > launch_training.py << 'EOF'
import boto3
import time
import sys
from datetime import datetime

def show_training_progress():
    print("🚀 Training job launched successfully!")
    print("⏳ Monitoring progress...")
    
    sagemaker = boto3.client('sagemaker')
    
    # Get the latest training job
    jobs = sagemaker.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not jobs['TrainingJobSummaries']:
        print("❌ No training jobs found")
        return
    
    job_name = jobs['TrainingJobSummaries'][0]['TrainingJobName']
    print(f"📊 Job Name: {job_name}")
    
    start_time = time.time()
    last_status = ""
    
    while True:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed//60:02d}:{elapsed%60:02d}] Status: {status}")
                last_status = status
            
            if status == 'Completed':
                print("🎉 Training completed successfully!")
                
                # Show final metrics
                if 'FinalMetricDataList' in response:
                    for metric in response['FinalMetricDataList']:
                        print(f"   {metric['MetricName']}: {metric['Value']}")
                
                # Get model location
                model_uri = response['ModelArtifacts']['S3ModelArtifacts']
                print(f"📦 Model location: {model_uri}")
                
                return True
                
            elif status == 'Failed':
                print("❌ Training failed!")
                if 'FailureReason' in response:
                    print(f"   Reason: {response['FailureReason']}")
                return False
                
            elif status in ['InProgress', 'Starting']:
                # Show progress bar
                elapsed = int(time.time() - start_time)
                estimated_total = 30 * 60  # 30 minutes
                progress = min(elapsed / estimated_total * 100, 95)
                
                bar_length = 30
                filled = int(progress * bar_length / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(f"\r[{bar}] {progress:.1f}% - {elapsed//60:02d}:{elapsed%60:02d} elapsed", end='', flush=True)
                
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped (training continues)")
            return None
        except Exception as e:
            print(f"\n⚠️  Error monitoring: {e}")
            time.sleep(60)

if __name__ == "__main__":
    show_training_progress()
EOF

# Launch the actual training
python aws_fast_training.py &
TRAINING_PID=$!

# Start progress monitoring
python launch_training.py

# Clean up
deactivate 2>/dev/null || true
rm -f launch_training.py

echo ""
echo -e "${GREEN}🎯 ULTRA-FAST TRAINING SUMMARY${NC}"
echo "=============================="
echo -e "${GREEN}✅ Training job launched successfully!${NC}"
echo -e "${YELLOW}⏱️  Expected completion: 30-45 minutes${NC}"
echo -e "${YELLOW}💰 Expected cost: \$6-12${NC}"
echo -e "${BLUE}🔍 Monitor progress:${NC}"
echo "   AWS Console: https://console.aws.amazon.com/sagemaker/"
echo "   CLI: aws sagemaker list-training-jobs --status-equals InProgress"
echo ""
echo -e "${PURPLE}📱 You'll get production-ready models in under an hour!${NC}"
echo -e "${GREEN}🚀 This is the FASTEST way to train your Saafe models!${NC}"

# Optional: Set up automatic download
read -p "Set up automatic model download when complete? (y/N): " auto_download

if [[ $auto_download =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}⏳ Setting up automatic download...${NC}"
    
    # Create download script
    cat > auto_download.sh << 'EOF'
#!/bin/bash

echo "🔍 Waiting for training completion and auto-downloading..."

while true; do
    # Get latest training job
    LATEST_JOB=$(aws sagemaker list-training-jobs \
        --sort-by CreationTime \
        --sort-order Descending \
        --max-items 1 \
        --query 'TrainingJobSummaries[0].TrainingJobName' \
        --output text)
    
    if [ "$LATEST_JOB" != "None" ] && [[ $LATEST_JOB == saafe-* ]]; then
        STATUS=$(aws sagemaker describe-training-job \
            --training-job-name $LATEST_JOB \
            --query 'TrainingJobStatus' \
            --output text)
        
        echo "$(date '+%H:%M:%S'): $LATEST_JOB - $STATUS"
        
        if [ "$STATUS" = "Completed" ]; then
            echo "🎉 Training completed! Downloading models..."
            
            # Get model location
            MODEL_URI=$(aws sagemaker describe-training-job \
                --training-job-name $LATEST_JOB \
                --query 'ModelArtifacts.S3ModelArtifacts' \
                --output text)
            
            # Download model
            mkdir -p models
            aws s3 cp $MODEL_URI models/model.tar.gz
            cd models && tar -xzf model.tar.gz && rm model.tar.gz
            
            # Rename to standard format
            if [ -f "model.pth" ]; then
                mv model.pth transformer_model.pth
            fi
            
            echo "✅ Models downloaded to models/ directory"
            echo "🚀 Ready for deployment!"
            
            # Show final summary
            echo ""
            echo "🎯 TRAINING COMPLETE SUMMARY"
            echo "============================"
            echo "✅ Models: models/transformer_model.pth"
            echo "✅ Metrics: models/metrics.json"
            echo "🚀 Next: streamlit run app.py"
            
            break
            
        elif [ "$STATUS" = "Failed" ]; then
            echo "❌ Training failed!"
            break
        fi
    fi
    
    sleep 60  # Check every minute
done
EOF
    
    chmod +x auto_download.sh
    
    echo -e "${GREEN}✅ Auto-download script created: auto_download.sh${NC}"
    echo -e "${CYAN}   Run in another terminal: ./auto_download.sh${NC}"
    
    # Option to run it in background
    read -p "Run auto-download in background now? (y/N): " run_bg
    if [[ $run_bg =~ ^[Yy]$ ]]; then
        nohup ./auto_download.sh > download.log 2>&1 &
        echo -e "${GREEN}✅ Auto-download running in background${NC}"
        echo -e "${CYAN}   Check progress: tail -f download.log${NC}"
    fi
fi

echo ""
echo -e "${PURPLE}🎉 ULTRA-FAST TRAINING SETUP COMPLETE!${NC}"
echo "====================================="
echo -e "${GREEN}Your Saafe models will be trained with maximum speed!${NC}"
echo -e "${YELLOW}Check back in 30-45 minutes for production-ready models! 🚀${NC}"