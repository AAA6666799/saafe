#!/usr/bin/env python3
"""
Saafe Fire Detection System - Cloud Deployment Manager
This script provides a simple interface for deploying the system to various cloud platforms.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class SaafeDeploymentManager:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.deployment_dir = self.project_root / "deployment"
        
    def deploy_docker(self):
        """Deploy using Docker Compose"""
        print("🐳 Deploying with Docker Compose...")
        
        # Check if docker-compose.yml exists
        compose_file = self.deployment_dir / "docker-compose.yml"
        if not compose_file.exists():
            print("❌ docker-compose.yml not found")
            return False
            
        try:
            # Run docker-compose up
            result = subprocess.run([
                "docker-compose", "-f", str(compose_file), "up", "-d"
            ], cwd=self.deployment_dir, check=True, capture_output=True, text=True)
            
            print("✅ Docker deployment started successfully")
            print("🌐 Access the dashboard at: http://localhost:8501")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker deployment failed: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ Docker Compose not found. Please install Docker Desktop or Docker Compose.")
            return False
    
    def deploy_kubernetes(self):
        """Deploy to Kubernetes"""
        print("☸️  Deploying to Kubernetes...")
        
        # Check if kubectl is available
        try:
            subprocess.run(["kubectl", "version", "--client"], 
                         check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ kubectl not found. Please install kubectl.")
            return False
            
        kube_config = self.deployment_dir / "kubernetes-deployment.yaml"
        if not kube_config.exists():
            print("❌ kubernetes-deployment.yaml not found")
            return False
            
        try:
            # Apply the Kubernetes configuration
            result = subprocess.run([
                "kubectl", "apply", "-f", str(kube_config)
            ], cwd=self.deployment_dir, check=True, capture_output=True, text=True)
            
            print("✅ Kubernetes deployment started")
            print("📋 Check status with: kubectl get pods -n saafe-fire-detection")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Kubernetes deployment failed: {e.stderr}")
            return False
    
    def deploy_aws(self):
        """Deploy to AWS"""
        print("☁️  Deploying to AWS...")
        
        # Check if AWS CLI is available
        try:
            subprocess.run(["aws", "--version"], 
                         check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ AWS CLI not found. Please install AWS CLI.")
            return False
            
        deploy_script = self.deployment_dir / "deploy-aws.sh"
        if not deploy_script.exists():
            print("❌ deploy-aws.sh not found")
            return False
            
        try:
            # Make script executable and run it
            deploy_script.chmod(0o755)
            result = subprocess.run([str(deploy_script)], 
                                  cwd=self.deployment_dir, check=True)
            
            print("✅ AWS deployment initiated")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ AWS deployment failed")
            return False
    
    def deploy_systemd(self):
        """Deploy as systemd service"""
        print("🐧 Deploying as systemd service...")
        
        service_file = self.deployment_dir / "saafe-fire-detection.service"
        if not service_file.exists():
            print("❌ saafe-fire-detection.service not found")
            return False
            
        try:
            # Copy service file and enable it
            subprocess.run([
                "sudo", "cp", str(service_file), 
                "/etc/systemd/system/saafe-fire-detection.service"
            ], check=True)
            
            subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
            subprocess.run(["sudo", "systemctl", "enable", "saafe-fire-detection.service"], check=True)
            subprocess.run(["sudo", "systemctl", "start", "saafe-fire-detection.service"], check=True)
            
            print("✅ Systemd service deployed and started")
            print("📋 Check status with: sudo systemctl status saafe-fire-detection.service")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Systemd deployment failed: {e}")
            return False
        except PermissionError:
            print("❌ Permission denied. Run with sudo or as root user.")
            return False

def main():
    parser = argparse.ArgumentParser(description="Saafe Fire Detection System Deployment Manager")
    parser.add_argument("platform", choices=["docker", "kubernetes", "aws", "systemd"],
                       help="Deployment platform")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    print("🔥 Saafe Fire Detection System Deployment Manager")
    print("=" * 50)
    
    deployer = SaafeDeploymentManager(args.project_root)
    
    if args.platform == "docker":
        success = deployer.deploy_docker()
    elif args.platform == "kubernetes":
        success = deployer.deploy_kubernetes()
    elif args.platform == "aws":
        success = deployer.deploy_aws()
    elif args.platform == "systemd":
        success = deployer.deploy_systemd()
    else:
        print("❌ Unknown platform")
        return 1
    
    if success:
        print("\n✅ Deployment completed successfully!")
        return 0
    else:
        print("\n❌ Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())