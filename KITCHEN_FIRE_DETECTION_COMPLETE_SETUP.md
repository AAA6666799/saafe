# SAAFE Kitchen Fire Detection System - Complete Setup

## System Status: ✅ FULLY OPERATIONAL

Your kitchen fire detection system has been successfully configured and is now ready for use.

## System Components

1. **IoT Device**: Installed above kitchen chimney
2. **Alert System**: Email and SMS notifications configured
3. **Web Dashboard**: React/Vite application with real-time monitoring
4. **Backend Services**: API server for data processing

## Access Your Dashboard

### Development Mode (Recommended for testing)
- URL: http://localhost:5173
- Start command: 
  ```bash
  cd "/Volumes/Ajay/saafe copy 3/saafe-lovable"
  ./start_dashboard.sh
  ```

### Production Mode (For permanent deployment)
- URL: http://localhost:8000
- Start command:
  ```bash
  cd "/Volumes/Ajay/saafe copy 3/saafe-lovable"
  ./build_and_serve.sh
  ```

## Key Features

- 🔥 Real-time fire risk monitoring
- 🌡️ Thermal camera data visualization
- 🧪 Gas sensor readings (VOC, CO, NO2)
- 🌤️ Environmental conditions (temperature, humidity, pressure)
- 📧 Email alerts to ch.ajay1707@gmail.com
- 📱 SMS alerts to your registered phone

## System Management Scripts

All scripts are located in `/Volumes/Ajay/saafe copy 3/saafe-lovable/`:

- `start_dashboard.sh` - Start development environment
- `build_and_serve.sh` - Build and serve production version
- `check_fire_system.sh` - Check system status
- `start_frontend.sh` - Start only the frontend

## API Endpoints

- Fire detection data: http://localhost:8000/api/fire-detection-data
- System status: http://localhost:8000/api/status

## Configuration Files

- Main config: `/Volumes/Ajay/saafe copy 3/config/app_config.json`
- IoT config: `/Volumes/Ajay/saafe copy 3/config/iot_config.yaml`

## Next Steps

1. Share the dashboard URL (http://localhost:5173) with your team
2. Test the alert system by simulating a fire event
3. Monitor the dashboard for real-time updates
4. Customize alert thresholds in the configuration files if needed

## Support

For any issues or questions:
- Check the system status: `./check_fire_system.sh`
- Review documentation in each component directory
- Contact SAAFE support team

Your kitchen fire detection system is now fully operational and ready to protect your home!