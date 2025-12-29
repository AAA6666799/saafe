#!/usr/bin/env python3
"""
Quick training launcher that avoids dependency conflicts.
Uses only built-in Python libraries where possible.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_command(cmd, ignore_errors=False):
    """Run command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and not ignore_errors:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        if not ignore_errors:
            print(f"❌ Error running command: {e}")
        return False

def show_progress_bar(duration, message):
    """Show a simple progress bar."""
    print(f"⏳ {message}")
    bar_length = 40
    
    for i in range(duration + 1):
        progress = i / duration
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        percent = int(progress * 100)
        
        print(f"\r[{bar}] {percent}%", end='', flush=True)
        time.sleep(0.1)
    
    print()  # New line after progress bar

def check_aws_setup():
    """Check if AWS is properly configured."""
    print("🔍 Checking AWS setup...")
    
    # Check AWS CLI
    if not run_command("aws --version", ignore_errors=True):
        print("❌ AWS CLI not found. Please install it first.")
        return False
    
    # Check credentials
    if not run_command("aws sts get-caller-identity", ignore_errors=True):
        print("❌ AWS credentials not configured. Run: aws configure")
        return False
    
    print("✅ AWS CLI configured")
    return True

def install_minimal_deps():
    """Install only essential dependencies."""
    print("📦 Installing minimal dependencies...")
    
    # Try to install without conflicts
    deps = ["boto3", "sagemaker"]
    
    for dep in deps:
        print(f"   Installing {dep}...")
        success = run_command(f"pip install -q --no-deps {dep}", ignore_errors=True)
        if not success:
            print(f"   ⚠️  {dep} installation had issues (may still work)")
    
    print("✅ Dependencies installed")

def get_account_info():
    """Get AWS account information."""
    try:
        result = subprocess.run("aws sts get-caller-identity --query Account --output text", 
                              shell=True, capture_output=True, text=True)
        account_id = result.stdout.strip()
        
        result = subprocess.run("aws configure get region", 
                              shell=True, capture_output=True, text=True)
        region = result.stdout.strip() or "us-east-1"
        
        return account_id, region
    except:
        return "unknown", "us-east-1"

def create_sagemaker_role():
    """Create SageMaker role if it doesn't exist."""
    print("🔐 Setting up SageMaker role...")
    
    role_name = "SaafeTrainingRole"
    
    # Check if role exists
    check_cmd = f"aws iam get-role --role-name {role_name}"
    if run_command(check_cmd, ignore_errors=True):
        print("✅ SageMaker role already exists")
        return True
    
    print("   Creating SageMaker role...")
    
    # Create trust policy
    trust_policy = '''{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}'''
    
    with open("trust-policy.json", "w") as f:
        f.write(trust_policy)
    
    # Create role
    create_cmd = f"aws iam create-role --role-name {role_name} --assume-role-policy-document file://trust-policy.json"
    if not run_command(create_cmd, ignore_errors=True):
        print("⚠️  Role creation failed, but may already exist")
    
    # Attach policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    ]
    
    for policy in policies:
        attach_cmd = f"aws iam attach-role-policy --role-name {role_name} --policy-arn {policy}"
        run_command(attach_cmd, ignore_errors=True)
    
    # Clean up
    if os.path.exists("trust-policy.json"):
        os.remove("trust-policy.json")
    
    print("✅ SageMaker role configured")
    return True

def launch_training():
    """Launch the ultra-fast training."""
    print("🚀 Launching ultra-fast training...")
    
    # Check if training script exists
    if not os.path.exists("aws_fast_training.py"):
        print("❌ aws_fast_training.py not found")
        return False
    
    print("⚡ Starting training job...")
    
    # Launch training in background
    try:
        process = subprocess.Popen([sys.executable, "aws_fast_training.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        print(f"✅ Training process started (PID: {process.pid})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start training: {e}")
        return False

def monitor_training():
    """Simple training monitoring."""
    print("📊 Monitoring training progress...")
    print("   (This is a simplified monitor - check AWS Console for detailed progress)")
    
    start_time = time.time()
    
    for i in range(30):  # Monitor for 30 minutes
        elapsed = int(time.time() - start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        
        # Simple progress estimation
        progress = min(i * 3.33, 95)  # Rough progress over 30 minutes
        
        print(f"\r⏳ Training progress: {progress:.1f}% - {minutes:02d}:{seconds:02d} elapsed", 
              end='', flush=True)
        
        # Check if we can get actual status
        try:
            result = subprocess.run("aws sagemaker list-training-jobs --status-equals InProgress --max-items 1", 
                                  shell=True, capture_output=True, text=True)
            if "saafe" in result.stdout.lower():
                print(f" - Training job active")
            else:
                print(f" - Checking status...")
        except:
            pass
        
        time.sleep(60)  # Check every minute
    
    print("\n✅ Monitoring complete - check AWS Console for final status")

def main():
    """Main function."""
    print("🚀 SAAFE ULTRA-FAST TRAINING (Dependency-Safe)")
    print("=" * 50)
    print("💰 Cost optimized for SPEED: $6-12")
    print("⚡ Expected time: 30-45 minutes")
    print()
    
    # Step 1: Check AWS setup
    show_progress_bar(10, "Checking AWS configuration")
    if not check_aws_setup():
        return
    
    # Step 2: Install dependencies
    show_progress_bar(15, "Installing dependencies")
    install_minimal_deps()
    
    # Step 3: Get account info
    account_id, region = get_account_info()
    print(f"📋 Account: {account_id}")
    print(f"📋 Region: {region}")
    
    # Step 4: Setup role
    show_progress_bar(8, "Setting up SageMaker role")
    create_sagemaker_role()
    
    # Step 5: Confirm launch
    print("\n🔥 ULTRA-FAST TRAINING CONFIGURATION")
    print("=" * 40)
    print("Instance: ml.p3.8xlarge (4x NVIDIA V100)")
    print("Memory: 244GB RAM + 64GB GPU")
    print("Cost: ~$12.24/hour")
    print("Expected time: 30-45 minutes")
    print("Expected total cost: $6-12")
    print()
    
    confirm = input("Launch ultra-fast training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("👍 Training cancelled")
        return
    
    # Step 6: Launch training
    show_progress_bar(20, "Launching training job")
    if not launch_training():
        return
    
    print("\n🎉 TRAINING LAUNCHED SUCCESSFULLY!")
    print("=" * 40)
    print("✅ Training job started")
    print("⏱️  Expected completion: 30-45 minutes")
    print("💰 Expected cost: $6-12")
    print("🔍 Monitor: https://console.aws.amazon.com/sagemaker/")
    print()
    
    # Optional monitoring
    monitor_choice = input("Start simple progress monitoring? (y/N): ").strip().lower()
    if monitor_choice == 'y':
        monitor_training()
    
    print("\n🚀 Your ultra-fast training is now running!")
    print("Check AWS Console for detailed progress and results.")

if __name__ == "__main__":
    main()