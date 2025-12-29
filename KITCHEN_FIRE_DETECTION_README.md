# Kitchen Fire Detection System

This system monitors your kitchen area (specifically above the chimney) for fire risks using IoT sensors and provides real-time alerts.

## System Overview

- **Location**: Kitchen above chimney
- **Sensor Type**: VOC (Volatile Organic Compounds) sensor
- **Monitoring**: Continuous 24/7 fire risk detection
- **Alerts**: Email and SMS notifications when fire risk is detected
- **Web Interface**: Dashboard for team monitoring

## Team Access

Your team can access the fire detection dashboard at:
**http://[YOUR_SERVER_IP]:8501**

Replace `[YOUR_SERVER_IP]` with the actual IP address of the server where the system is running.

### Default Access
- **URL**: http://localhost:8501 (if accessing from the same machine)
- **Email Notifications**: Sent to ch.ajay1707@gmail.com
- **SMS Alerts**: Sent to configured phone numbers

## System Features

### Real-time Monitoring
- Continuous monitoring of VOC levels in the kitchen
- AI-powered fire risk prediction
- Early warning system with lead time prediction

### Alert System
- **Critical Alerts**: Immediate notifications for high fire risk
- **Elevated Alerts**: Warnings for moderate fire risk
- **Email Notifications**: Detailed alerts sent to team members
- **SMS Alerts**: Instant text messages for critical situations

### Dashboard Features
- Real-time sensor readings
- Fire risk probability visualization
- Historical data analysis
- System status monitoring

## Deployment Options

### Quick Start
1. Run the deployment script:
   ```bash
   python3 deploy_kitchen_system.py
   ```

2. Start the web interface:
   ```bash
   ./deployment/start_kitchen_dashboard.sh
   ```

### Docker Deployment
```bash
docker-compose -f deployment/docker-compose-kitchen.yml up -d
```

### Systemd Service (for automatic startup)
```bash
sudo cp deployment/kitchen-fire-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable kitchen-fire-detection.service
sudo systemctl start kitchen-fire-detection.service
```

## Configuration

### Alert Settings
Alert thresholds can be configured in `config/app_config.json`:
- **Normal**: 0-30% risk
- **Mild**: 31-50% risk
- **Elevated**: 51-85% risk
- **Critical**: 86-100% risk

### Team Members
Add additional team members to receive alerts by updating `config/app_config.json`:
```json
{
  "notifications": {
    "email_addresses": [
      "ch.ajay1707@gmail.com",
      "team-member1@company.com",
      "team-member2@company.com"
    ],
    "phone_numbers": [
      "+1234567890"
    ]
  }
}
```

## Troubleshooting

### No Alerts Received
1. Check network connectivity
2. Verify email/SMS configuration in `config/app_config.json`
3. Check system logs in the `logs/` directory

### Dashboard Not Accessible
1. Ensure the service is running:
   ```bash
   ps aux | grep streamlit
   ```
2. Check firewall settings
3. Verify port 8501 is not blocked

### Sensor Issues
1. Check physical sensor connection
2. Verify sensor drivers are installed
3. Check hardware logs in `logs/hardware.log`

## Support

For technical support, contact:
- **Email**: ch.ajay1707@gmail.com
- **Phone**: +1234567890

## System Maintenance

### Regular Tasks
- Check sensor calibration monthly
- Review alert logs weekly
- Update system software quarterly

### Data Backup
- System logs are stored in `logs/`
- Configuration files in `config/`
- Models in `models/`

Regular backups of these directories are recommended.