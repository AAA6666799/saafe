# Saafe Fire Detection System - Cloud Deployment

This directory contains deployment configurations for running the Saafe Fire Detection System continuously in the cloud.

## Deployment Options

### 1. Docker Compose (Local/Single Server)
For local testing or single-server deployments:

```bash
# From the deployment directory
docker-compose up -d
```

This will start all system components:
- Main fire detection system (Streamlit UI on port 8501)
- IoT agent for sensor data collection
- Alert agent for notifications
- Monitoring agent for system health

### 2. Kubernetes (Cloud/Production)
For scalable cloud deployments:

```bash
# Apply the Kubernetes configuration
kubectl apply -f kubernetes-deployment.yaml

# Check the status
kubectl get pods -n saafe-fire-detection
```

### 3. Systemd Service (Linux Server)
For traditional Linux server deployments:

```bash
# Copy the service file to systemd directory
sudo cp saafe-fire-detection.service /etc/systemd/system/

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable saafe-fire-detection.service

# Start the service
sudo systemctl start saafe-fire-detection.service
```

### 4. AWS Deployment
For AWS deployments using ECS:

```bash
# Run the deployment script
./deploy-aws.sh
```

Follow the prompts to configure your AWS credentials and update the network settings with your VPC and subnet IDs.

## Configuration

All deployment methods use configuration files from the `config` directory. Make sure to update these files with your specific settings before deployment:

- `base_config.yaml` - Base system configuration
- `iot_config.yaml` - IoT sensor configuration
- `prod_config.yaml` - Production environment configuration

## Monitoring and Maintenance

The system includes built-in health checks and logging. Monitor the following:

- Container logs: `docker logs saafe-fire-detection`
- Kubernetes logs: `kubectl logs -n saafe-fire-detection <pod-name>`
- Systemd logs: `journalctl -u saafe-fire-detection.service`

## Scaling

For high-availability deployments, you can scale the Kubernetes deployment:

```bash
kubectl scale deployment saafe-fire-detection --replicas=3 -n saafe-fire-detection
```

Note that the IoT agent should typically run with a single replica to avoid conflicts with physical sensors.