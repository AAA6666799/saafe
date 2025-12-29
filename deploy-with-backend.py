#!/usr/bin/env python3
"""
SAAFE Dashboard Full-Stack AWS Deployment
Deploys both frontend and backend for complete functionality
"""

import json
import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, cwd=None):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error running: {cmd}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Exception running {cmd}: {e}")
        return None

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check AWS CLI
    if not run_command("aws --version"):
        print("❌ AWS CLI not found. Please install it first.")
        return False
    
    # Check AWS credentials
    if not run_command("aws sts get-caller-identity"):
        print("❌ AWS CLI not configured. Please run: aws configure")
        return False
    
    # Check Docker
    if not run_command("docker --version"):
        print("❌ Docker not found. Please install Docker Desktop.")
        return False
    
    print("✅ All prerequisites met")
    return True

def build_frontend():
    """Build the React frontend"""
    print("📦 Building frontend...")
    
    if not os.path.exists("saafe-lovable"):
        print("❌ saafe-lovable directory not found")
        return False
    
    # Install dependencies and build
    if not run_command("npm install", cwd="saafe-lovable"):
        return False
    
    if not run_command("npm run build", cwd="saafe-lovable"):
        return False
    
    print("✅ Frontend built successfully")
    return True

def create_dockerfile():
    """Create Dockerfile for full-stack deployment"""
    print("🐳 Creating Dockerfile...")
    
    dockerfile_content = """# Multi-stage build for SAAFE Dashboard
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY saafe-lovable/package*.json ./
RUN npm ci --only=production
COPY saafe-lovable/ ./
RUN npm run build

# Python backend stage
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY *.py ./
COPY config/ ./config/
COPY models/ ./models/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./static

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "app.py"]
"""
    
    with open("Dockerfile.fullstack", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created")
    return True

def create_eb_config():
    """Create Elastic Beanstalk configuration"""
    print("⚙️ Creating Elastic Beanstalk configuration...")
    
    # Create .ebextensions directory
    os.makedirs(".ebextensions", exist_ok=True)
    
    # Create environment configuration
    eb_config = {
        "option_settings": {
            "aws:elasticbeanstalk:container:python": {
                "WSGIPath": "app.py"
            },
            "aws:elasticbeanstalk:environment:proxy:staticfiles": {
                "/static": "static"
            },
            "aws:autoscaling:launchconfiguration": {
                "InstanceType": "t3.small"
            },
            "aws:elasticbeanstalk:healthreporting:system": {
                "SystemType": "enhanced"
            }
        }
    }
    
    with open(".ebextensions/01_python.config", "w") as f:
        f.write("""packages:
  yum:
    git: []
    
commands:
  01_install_dependencies:
    command: "pip install -r requirements.txt"
""")
    
    print("✅ Elastic Beanstalk configuration created")
    return True

def deploy_to_eb():
    """Deploy to Elastic Beanstalk"""
    print("🚀 Deploying to Elastic Beanstalk...")
    
    app_name = f"saafe-dashboard-{int(datetime.now().timestamp())}"
    env_name = f"{app_name}-env"
    
    # Initialize EB application
    if not run_command(f"eb init {app_name} --region us-east-1 --platform python-3.9"):
        return False, None, None
    
    # Create environment
    if not run_command(f"eb create {env_name} --instance-type t3.small"):
        return False, None, None
    
    # Get the URL
    url = run_command("eb status | grep CNAME")
    if url:
        url = url.split(":")[1].strip()
    
    print(f"✅ Deployed to Elastic Beanstalk: {url}")
    return True, app_name, url

def main():
    print("🚀 SAAFE Dashboard Full-Stack AWS Deployment")
    print("=" * 50)
    
    if not check_prerequisites():
        sys.exit(1)
    
    if not build_frontend():
        sys.exit(1)
    
    if not create_dockerfile():
        sys.exit(1)
    
    if not create_eb_config():
        sys.exit(1)
    
    success, app_name, url = deploy_to_eb()
    
    if success:
        print("\n🎉 Deployment Complete!")
        print("=" * 30)
        print(f"✅ Application: {app_name}")
        print(f"✅ URL: https://{url}")
        print("\n🌍 Your SAAFE dashboard is now live and accessible worldwide!")
        print("\n💡 Management commands:")
        print(f"   - View logs: eb logs")
        print(f"   - Update app: eb deploy")
        print(f"   - Terminate: eb terminate {app_name}-env")
    else:
        print("❌ Deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()