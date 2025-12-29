#!/usr/bin/env python3
"""
AWS CodeArtifact setup script for Saafe project
Configures pip to use CodeArtifact repository for package management
"""

import subprocess
import sys
import os
import json
from pathlib import Path

class CodeArtifactSetup:
    def __init__(self):
        self.domain = "saafeai"
        self.domain_owner = "691595239825"
        self.repository = "saafe"
        self.region = "eu-west-1"
        
    def login_to_codeartifact(self):
        """Login to AWS CodeArtifact"""
        print("🔐 Logging into AWS CodeArtifact...")
        
        cmd = [
            "aws", "codeartifact", "login",
            "--tool", "pip",
            "--repository", self.repository,
            "--domain", self.domain,
            "--domain-owner", self.domain_owner,
            "--region", self.region
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Successfully logged into CodeArtifact")
            print(f"   Domain: {self.domain}")
            print(f"   Repository: {self.repository}")
            print(f"   Region: {self.region}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to login to CodeArtifact: {e}")
            print(f"   Error output: {e.stderr}")
            return False
    
    def get_repository_endpoint(self):
        """Get the CodeArtifact repository endpoint"""
        print("🔗 Getting repository endpoint...")
        
        cmd = [
            "aws", "codeartifact", "get-repository-endpoint",
            "--domain", self.domain,
            "--domain-owner", self.domain_owner,
            "--repository", self.repository,
            "--format", "pypi",
            "--region", self.region
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            endpoint_data = json.loads(result.stdout)
            endpoint = endpoint_data['repositoryEndpoint']
            print(f"✅ Repository endpoint: {endpoint}")
            return endpoint
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get repository endpoint: {e}")
            return None
    
    def get_authorization_token(self):
        """Get authorization token for CodeArtifact"""
        print("🎫 Getting authorization token...")
        
        cmd = [
            "aws", "codeartifact", "get-authorization-token",
            "--domain", self.domain,
            "--domain-owner", self.domain_owner,
            "--region", self.region,
            "--query", "authorizationToken",
            "--output", "text"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            token = result.stdout.strip()
            print("✅ Authorization token obtained")
            return token
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get authorization token: {e}")
            return None
    
    def create_pip_conf(self):
        """Create pip.conf file for CodeArtifact"""
        print("📝 Creating pip configuration...")
        
        endpoint = self.get_repository_endpoint()
        token = self.get_authorization_token()
        
        if not endpoint or not token:
            print("❌ Cannot create pip.conf without endpoint and token")
            return False
        
        # Create pip.conf content
        pip_conf_content = f"""[global]
index-url = https://aws:{token}@{endpoint.replace('https://', '')}/simple/
trusted-host = {endpoint.replace('https://', '').split('/')[0]}

[install]
trusted-host = {endpoint.replace('https://', '').split('/')[0]}
"""
        
        # Write to pip.conf
        pip_conf_path = Path.home() / ".pip" / "pip.conf"
        pip_conf_path.parent.mkdir(exist_ok=True)
        
        with open(pip_conf_path, 'w') as f:
            f.write(pip_conf_content)
        
        print(f"✅ Created pip.conf at {pip_conf_path}")
        return True
    
    def create_requirements_codeartifact(self):
        """Create requirements file optimized for CodeArtifact"""
        print("📦 Creating CodeArtifact-optimized requirements...")
        
        # Read existing requirements
        requirements_path = Path("requirements.txt")
        if not requirements_path.exists():
            print("❌ requirements.txt not found")
            return False
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        # Create CodeArtifact version
        codeartifact_requirements = f"""# Saafe MVP - CodeArtifact Requirements
# Generated for AWS CodeArtifact repository: {self.repository}

{requirements}

# AWS-specific packages
boto3>=1.26.0
botocore>=1.29.0
awscli>=2.0.0
sagemaker>=2.0.0

# Additional packages for AWS deployment
gunicorn>=20.1.0
uvicorn>=0.18.0
fastapi>=0.95.0
"""
        
        codeartifact_path = Path("requirements-codeartifact.txt")
        with open(codeartifact_path, 'w') as f:
            f.write(codeartifact_requirements)
        
        print(f"✅ Created {codeartifact_path}")
        return True
    
    def test_installation(self):
        """Test package installation from CodeArtifact"""
        print("🧪 Testing package installation...")
        
        test_packages = ["boto3", "numpy", "torch"]
        
        for package in test_packages:
            print(f"   Testing {package}...")
            cmd = ["pip", "install", "--dry-run", package]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"   ✅ {package} available")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {package} failed: {e}")
                return False
        
        print("✅ All test packages available from CodeArtifact")
        return True
    
    def create_docker_setup(self):
        """Create Docker setup for CodeArtifact"""
        print("🐳 Creating Docker setup for CodeArtifact...")
        
        dockerfile_content = f"""# Dockerfile for Saafe MVP with CodeArtifact
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \\
    && unzip awscliv2.zip \\
    && ./aws/install \\
    && rm -rf aws awscliv2.zip

# Copy requirements first for better caching
COPY requirements-codeartifact.txt .

# Setup CodeArtifact authentication
ARG AWS_ACCOUNT_ID={self.domain_owner}
ARG AWS_REGION={self.region}
ARG CODEARTIFACT_DOMAIN={self.domain}
ARG CODEARTIFACT_REPO={self.repository}

# Login to CodeArtifact and install packages
RUN aws codeartifact login --tool pip \\
    --repository $CODEARTIFACT_REPO \\
    --domain $CODEARTIFACT_DOMAIN \\
    --domain-owner $AWS_ACCOUNT_ID \\
    --region $AWS_REGION \\
    && pip install --no-cache-dir -r requirements-codeartifact.txt

# Copy application code
COPY saafe_mvp/ ./saafe_mvp/
COPY models/ ./models/
COPY config/ ./config/
COPY app.py .
COPY main.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        with open("Dockerfile-codeartifact", 'w') as f:
            f.write(dockerfile_content)
        
        print("✅ Created Dockerfile-codeartifact")
        
        # Create docker-compose for local testing
        compose_content = f"""version: '3.8'

services:
  saafe-mvp:
    build:
      context: .
      dockerfile: Dockerfile-codeartifact
      args:
        AWS_ACCOUNT_ID: {self.domain_owner}
        AWS_REGION: {self.region}
        CODEARTIFACT_DOMAIN: {self.domain}
        CODEARTIFACT_REPO: {self.repository}
    ports:
      - "8501:8501"
    environment:
      - AWS_DEFAULT_REGION={self.region}
    volumes:
      - ~/.aws:/root/.aws:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
        
        with open("docker-compose-codeartifact.yml", 'w') as f:
            f.write(compose_content)
        
        print("✅ Created docker-compose-codeartifact.yml")
        return True
    
    def setup_complete(self):
        """Complete setup process"""
        print("\n" + "=" * 60)
        print("🚀 AWS CodeArtifact Setup Complete!")
        print("=" * 60)
        
        print(f"\n📋 Configuration Summary:")
        print(f"   Domain: {self.domain}")
        print(f"   Repository: {self.repository}")
        print(f"   Region: {self.region}")
        print(f"   Domain Owner: {self.domain_owner}")
        
        print(f"\n🔧 Files Created:")
        print(f"   • requirements-codeartifact.txt")
        print(f"   • Dockerfile-codeartifact")
        print(f"   • docker-compose-codeartifact.yml")
        print(f"   • ~/.pip/pip.conf")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Test local installation:")
        print(f"      pip install -r requirements-codeartifact.txt")
        print(f"   2. Build Docker image:")
        print(f"      docker-compose -f docker-compose-codeartifact.yml build")
        print(f"   3. Run locally:")
        print(f"      docker-compose -f docker-compose-codeartifact.yml up")
        print(f"   4. Deploy to AWS with CodeArtifact integration")
        
        print(f"\n💡 Tips:")
        print(f"   • CodeArtifact tokens expire after 12 hours")
        print(f"   • Re-run login command when token expires")
        print(f"   • Use requirements-codeartifact.txt for deployments")

def main():
    """Main setup function"""
    print("🔧 AWS CodeArtifact Setup for Saafe MVP")
    print("=" * 60)
    
    setup = CodeArtifactSetup()
    
    # Step 1: Login to CodeArtifact
    if not setup.login_to_codeartifact():
        print("❌ Setup failed at login step")
        sys.exit(1)
    
    # Step 2: Create pip configuration
    if not setup.create_pip_conf():
        print("❌ Setup failed at pip configuration step")
        sys.exit(1)
    
    # Step 3: Create optimized requirements
    if not setup.create_requirements_codeartifact():
        print("❌ Setup failed at requirements creation step")
        sys.exit(1)
    
    # Step 4: Test installation
    if not setup.test_installation():
        print("⚠️  Package testing failed, but continuing...")
    
    # Step 5: Create Docker setup
    if not setup.create_docker_setup():
        print("❌ Setup failed at Docker setup step")
        sys.exit(1)
    
    # Step 6: Complete setup
    setup.setup_complete()

if __name__ == "__main__":
    main()