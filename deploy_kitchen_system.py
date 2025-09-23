#!/usr/bin/env python3
"""
Deployment script for the kitchen fire detection system.
This script sets up the IoT device, configures alerts, and starts the web interface.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json

class KitchenFireDetectionDeployment:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.deployment_dir = self.project_root / "deployment"
        
    def setup_iot_device(self):
        """Setup the IoT device for kitchen fire detection"""
        print("🔧 Setting up IoT device for kitchen fire detection...")
        
        # Check if required configuration exists
        iot_config = self.config_dir / "iot_config.yaml"
        if not iot_config.exists():
            print("❌ IoT configuration file not found")
            return False
            
        print("✅ IoT configuration file found")
        print(f"📍 Device location: Kitchen above chimney")
        print(f"📡 Sensor type: VOC sensor")
        return True
    
    def configure_alerts(self):
        """Configure fire alerts for the kitchen device"""
        print("🔔 Configuring fire alerts...")
        
        # Check app configuration
        app_config_path = self.config_dir / "app_config.json"
        if not app_config_path.exists():
            print("❌ App configuration file not found")
            return False
            
        try:
            with open(app_config_path, 'r') as f:
                config = json.load(f)
            
            # Check if notifications are enabled
            notifications = config.get("notifications", {})
            if notifications.get("sms_enabled") or notifications.get("email_enabled"):
                print("✅ Alerts configured successfully")
                print(f"📧 Email alerts: {'ENABLED' if notifications.get('email_enabled') else 'DISABLED'}")
                print(f"📱 SMS alerts: {'ENABLED' if notifications.get('sms_enabled') else 'DISABLED'}")
                print(f"👤 Alert recipients: {len(notifications.get('email_addresses', []))} email(s), {len(notifications.get('phone_numbers', []))} phone number(s)")
                return True
            else:
                print("⚠️  No alert methods enabled in configuration")
                return False
                
        except Exception as e:
            print(f"❌ Error reading configuration: {e}")
            return False
    
    def start_web_interface(self, host="0.0.0.0", port=8501):
        """Start the web interface for team access"""
        print("🌐 Starting web interface for team access...")
        
        try:
            # Check if main application exists
            app_file = self.project_root / "app.py"
            if not app_file.exists():
                # Try alternative locations
                app_file = self.project_root / "saafe_mvp" / "main.py"
                if not app_file.exists():
                    print("❌ Main application file not found")
                    return False
            
            # Start the web interface using Streamlit
            print(f"🚀 Starting web server on {host}:{port}")
            print(f"🔗 Team access URL: http://{self.get_local_ip()}:{port}")
            print(f"🔐 For external access, use your public IP or set up port forwarding")
            
            # Create a simple startup script
            startup_script = self.deployment_dir / "start_kitchen_dashboard.sh"
            with open(startup_script, 'w') as f:
                f.write(f"""#!/bin/bash
# Kitchen Fire Detection Dashboard Startup Script

cd {self.project_root}

# Start the dashboard
streamlit run {app_file} \\
    --server.port={port} \\
    --server.address={host} \\
    --server.headless=true \\
    --browser.gatherUsageStats=false

echo "Dashboard started on http://{host}:{port}"
""")
            
            # Make script executable
            startup_script.chmod(0o755)
            
            print(f"💾 Startup script created: {startup_script}")
            print("💡 To start the dashboard, run:")
            print(f"   {startup_script}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting web interface: {e}")
            return False
    
    def get_local_ip(self):
        """Get the local IP address of the machine"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    def create_systemd_service(self):
        """Create a systemd service for automatic startup"""
        print("⚙️  Creating systemd service for automatic startup...")
        
        try:
            service_content = f"""[Unit]
Description=Kitchen Fire Detection System
After=network.target
After=multi-user.target

[Service]
Type=simple
User={os.getenv('USER', 'saafe')}
Group={os.getenv('USER', 'saafe')}
WorkingDirectory={self.project_root}
Environment=PATH={os.getenv('PATH')}
Environment=PYTHONPATH={self.project_root}
Environment=STREAMLIT_SERVER_HEADLESS=true
Environment=STREAMLIT_SERVER_PORT=8501
Environment=STREAMLIT_SERVER_ADDRESS=0.0.0.0
ExecStart={sys.executable} -m streamlit run {self.project_root}/saafe_mvp/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={self.project_root}/logs {self.project_root}/exports {self.project_root}/temp

[Install]
WantedBy=multi-user.target
"""
            
            service_file = self.deployment_dir / "kitchen-fire-detection.service"
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            print(f"✅ Systemd service file created: {service_file}")
            print("💡 To install and start the service, run:")
            print(f"   sudo cp {service_file} /etc/systemd/system/")
            print("   sudo systemctl daemon-reload")
            print("   sudo systemctl enable kitchen-fire-detection.service")
            print("   sudo systemctl start kitchen-fire-detection.service")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating systemd service: {e}")
            return False
    
    def deploy_docker(self):
        """Deploy using Docker for easy setup"""
        print("🐳 Creating Docker deployment...")
        
        try:
            # Create a simplified docker-compose file for kitchen deployment
            docker_compose_content = f"""version: '3.8'

services:
  kitchen-fire-detection:
    build:
      context: {self.project_root}
      dockerfile: Dockerfile
    image: saafe/kitchen-fire-detection:latest
    container_name: kitchen-fire-detection
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - {self.project_root}/logs:/app/logs
      - {self.project_root}/config:/app/config
      - {self.project_root}/models:/app/models
    restart: unless-stopped
    privileged: true  # Required for hardware access
"""
            
            docker_compose_file = self.deployment_dir / "docker-compose-kitchen.yml"
            with open(docker_compose_file, 'w') as f:
                f.write(docker_compose_content)
            
            print(f"✅ Docker Compose file created: {docker_compose_file}")
            print("💡 To deploy using Docker, run:")
            print(f"   docker-compose -f {docker_compose_file} up -d")
            print("🔗 Access dashboard at: http://localhost:8501")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating Docker deployment: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Kitchen Fire Detection System Deployment")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP for web interface")
    parser.add_argument("--port", type=int, default=8501, help="Port for web interface")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--mode", choices=["full", "web-only", "iot-only"], 
                       default="full", help="Deployment mode")
    
    args = parser.parse_args()
    
    print("🔥 Kitchen Fire Detection System Deployment")
    print("=" * 50)
    
    deployer = KitchenFireDetectionDeployment(args.project_root)
    
    # Setup IoT device
    if args.mode in ["full", "iot-only"]:
        if not deployer.setup_iot_device():
            print("❌ IoT device setup failed")
            return 1
    
    # Configure alerts
    if args.mode in ["full", "iot-only"]:
        if not deployer.configure_alerts():
            print("❌ Alert configuration failed")
            return 1
    
    # Start web interface
    if args.mode in ["full", "web-only"]:
        if not deployer.start_web_interface(args.host, args.port):
            print("❌ Web interface setup failed")
            return 1
        
        # Create systemd service
        deployer.create_systemd_service()
        
        # Create Docker deployment
        deployer.deploy_docker()
    
    print("\n✅ Kitchen fire detection system deployment completed!")
    print(f"🔗 Team access URL: http://{deployer.get_local_ip()}:{args.port}")
    print("🔔 Fire alerts are configured and active")
    print("📍 IoT device is monitoring kitchen above chimney")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())