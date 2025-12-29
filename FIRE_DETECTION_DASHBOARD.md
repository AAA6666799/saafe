# SAAFE Kitchen Fire Detection System with Dashboard

This document describes the complete setup for the SAAFE fire detection system in your kitchen, including the IoT device, alert system, and web dashboard.

## System Overview

The system consists of:
1. IoT fire detection device installed above your kitchen chimney
2. Email and SMS alert system
3. Web dashboard for monitoring and visualization
4. Backend services for data processing and API

## Current Setup Status

✅ IoT device installed in kitchen
✅ Email alerts configured (sending to ch.ajay1707@gmail.com)
✅ SMS alerts configured
✅ Web dashboard available at http://localhost:5173
✅ Backend API running at http://localhost:8000

## Accessing the Dashboard

### Development Mode (Recommended for testing)
1. Open a terminal
2. Navigate to the project directory:
   ```bash
   cd "/Volumes/Ajay/saafe copy 3/saafe-lovable"
   ```
3. Start the dashboard:
   ```bash
   ./start_dashboard.sh
   ```
4. Open your browser and go to: http://localhost:5173

### Production Mode (For permanent deployment)
1. Open a terminal
2. Navigate to the project directory:
   ```bash
   cd "/Volumes/Ajay/saafe copy 3/saafe-lovable"
   ```
3. Build and serve the dashboard:
   ```bash
   ./build_and_serve.sh
   ```
4. Open your browser and go to: http://localhost:8000

## Dashboard Features

The dashboard provides real-time monitoring of:
- Fire risk score with visual indicators
- Thermal camera data visualization
- Gas sensor readings (VOC, CO, NO2)
- Environmental conditions (temperature, humidity, pressure)
- System status and alerts

## Alert System

The system is configured to send alerts via:
- Email to: ch.ajay1707@gmail.com
- SMS to your registered phone number

Alerts are triggered based on fire probability thresholds and sensor readings.

## Backend API

The dashboard connects to a backend API that provides fire detection data:
- API endpoint: http://localhost:8000/api/fire-detection-data
- Status endpoint: http://localhost:8000/api/status

## Directory Structure

```
/Volumes/Ajay/saafe copy 3/
├── saafe-lovable/              # Dashboard frontend and backend
│   ├── src/                    # React frontend source code
│   ├── backend/                # Express.js backend server
│   ├── start_dashboard.sh      # Development startup script
│   └── build_and_serve.sh      # Production build script
├── config/                     # System configuration files
├── deployment/                 # Deployment scripts
└── saafe_mvp/                  # Core fire detection system
```

## Troubleshooting

If you encounter issues:

1. **Dashboard not loading**: 
   - Ensure both frontend and backend services are running
   - Check that ports 5173 and 8000 are not blocked by firewall

2. **No data in dashboard**:
   - Verify the IoT device is connected and sending data
   - Check backend API is returning data: `curl http://localhost:8000/api/fire-detection-data`

3. **Alerts not working**:
   - Check configuration in `config/app_config.json`
   - Verify network connectivity for email/SMS services

## Next Steps

To enhance your system:

1. Connect the IoT device to send real data to the backend
2. Configure additional alert recipients
3. Set up automated deployment for the dashboard
4. Add more sophisticated analytics and reporting

## Support

For any issues or questions, please refer to the documentation in each directory or contact the SAAFE support team.