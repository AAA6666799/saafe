#!/usr/bin/env python3
"""
TOP-NOTCH Saafe Training Launcher
Maximum performance with 4x V100 GPUs - Cost no object!
"""

import subprocess
import sys
import json
import os
import time

def print_banner():
    """Print the top-notch banner."""
    print("🔥" * 50)
    print("🚀 SAAFE TOP-NOTCH TRAINING LAUNCHER 🚀")
    print("🔥" * 50)
    print("💰 COST: $6-12 (MAXIMUM PERFORMANCE)")
    print("⚡ TIME: 20-30 minutes")
    print("🔥 4x NVIDIA V100 GPUs (64GB GPU RAM)")
    print("🧠 ADVANCED MODEL: 500K+ parameters")
    print("📊 TRAINING DATA: 20,000 samples")
    print("🎯 PRODUCTION-GRADE QUALITY")
    print("🔥" * 50)
    print()

def quick_check():
    """Quick AWS check."""
    print("🔍 Quick AWS check...")
    
    try:
        result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            account_data = json.loads(result.stdout)
            print(f"✅ AWS Account: {account_data['Account']}")
            return True
        else:
            print("❌ AWS not configured. Run: aws configure")
            return False
    except:
        print("❌ AWS CLI issue. Please check your setup.")
        return False

def launch_topnotch():
    """Launch the top-notch training."""
    print("🚀 LAUNCHING TOP-NOTCH TRAINING...")
    print("=" * 40)
    
    # Confirm the beast mode
    print("🔥 TOP-NOTCH CONFIGURATION:")
    print("   Instance: ml.p3.8xlarge")
    print("   GPUs: 4x NVIDIA V100 (16GB each)")
    print("   Total GPU Memory: 64GB")
    print("   CPU: 32 vCPUs, 244GB RAM")
    print("   Model: Advanced TopNotchFireModel")
    print("   Training Samples: 20,000")
    print("   Epochs: 50 (high quality)")
    print("   Mixed Precision: Enabled")
    print("   Multi-GPU: DataParallel")
    print()
    print("💰 COST BREAKDOWN:")
    print("   Rate: $12.24/hour")
    print("   Expected Time: 25 minutes")
    print("   Total Cost: ~$5.10")
    print()
    
    confirm = input("🔥 LAUNCH TOP-NOTCH TRAINING? (y/N): ").strip().upper()
    
    if confirm != 'Y':
        print("👍 Cancelled - No problem!")
        return
    
    print("\n🚀 LAUNCHING THE BEAST...")
    print("=" * 30)
    
    try:
        # Launch the simple_train.py which now has top-notch config
        result = subprocess.run([sys.executable, 'simple_train.py'], 
                              input='y\ny\n',  # Auto-confirm the prompts
                              text=True,
                              timeout=600)  # 10 minute timeout for launch
        
        if result.returncode == 0:
            print("\n🎉 TOP-NOTCH TRAINING LAUNCHED!")
            print("=" * 35)
            print("✅ 4x V100 GPUs are now training your model")
            print("⚡ Expected completion: 20-30 minutes")
            print("💰 Expected cost: $4-6")
            print("🔍 Monitor: https://console.aws.amazon.com/sagemaker/")
            print("\n🔥 YOU'RE GETTING THE BEST FIRE DETECTION MODEL POSSIBLE!")
            
        else:
            print("❌ Launch failed. Try running: python simple_train.py")
            
    except subprocess.TimeoutExpired:
        print("⏰ Launch taking longer than expected...")
        print("🔍 Check AWS Console: https://console.aws.amazon.com/sagemaker/")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        print("🔧 Try running: python simple_train.py")

def main():
    """Main function."""
    print_banner()
    
    if not quick_check():
        return
    
    print("🎯 READY FOR TOP-NOTCH TRAINING!")
    print("This will create the BEST fire detection model possible.")
    print("Using maximum AWS compute power for production-grade results.")
    print()
    
    launch_topnotch()

if __name__ == "__main__":
    main()