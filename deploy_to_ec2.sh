#!/bin/bash

# Deployment script for Saafe dashboard on EC2

# Update system
sudo yum update -y

# Install Python 3 and pip
sudo yum install -y python3 python3-pip git

# Install required packages
pip3 install streamlit boto3 pytz plotly pandas

# Create app directory
mkdir -p /home/ec2-user/saafe-dashboard
cd /home/ec2-user/saafe-dashboard

# Download the dashboard files
wget https://raw.githubusercontent.com/AAA6666799/saafe/main/saafe_aws_dashboard.py
wget https://raw.githubusercontent.com/AAA6666799/saafe/main/run_aws_dashboard.py
wget https://raw.githubusercontent.com/AAA6666799/saafe/main/start_aws_dashboard.sh

# Make the start script executable
chmod +x start_aws_dashboard.sh

# Create systemd service file
sudo tee /etc/systemd/system/saafe-dashboard.service > /dev/null <<EOF
[Unit]
Description=Saafe Fire Detection Dashboard
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/saafe-dashboard
ExecStart=/home/ec2-user/saafe-dashboard/start_aws_dashboard.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable saafe-dashboard
sudo systemctl start saafe-dashboard

echo "Saafe dashboard deployment completed!"
echo "Access the dashboard at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8502"