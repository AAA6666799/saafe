# SAAFE Fire Detection System Integration

This repository contains the integration between the Synthetic Fire Prediction System and the SAAFE Global Command Center (GCC).

## Integration Summary

The Synthetic Fire Prediction System dashboard has been successfully integrated with the SAAFE Global Command Center while preserving the original GCC design and without using mock data.

## Key Features

- **Real-time Fire Detection Data**: Live data from the Synthetic Fire Prediction System
- **Preserved GCC Design**: All existing GCC functionality and styling maintained
- **No Mock Data**: All data comes from the actual fire detection system
- **Seamless Integration**: Fire detection dashboard appears within the GCC interface

## How to Run the Integrated System

### Prerequisites

1. Python 3.7+
2. Node.js 14+
3. npm 6+
4. AWS credentials configured for S3 access

### Starting the System

1. **Start the Fire Detection API Server**:
   ```bash
   cd task_1_synthetic_fire_system
   ./start_api.sh
   ```
   The API server will be available at `http://localhost:8000`

2. **Start the SAAFE Global Command Center**:
   ```bash
   cd saafe-lovable
   ./start_frontend.sh
   ```
   The GCC will be available at `http://localhost:5173`

3. **Access the Integrated System**:
   Open your browser and navigate to `http://localhost:5173` to see the GCC with the integrated fire detection dashboard.

## Integration Components

### Fire Detection API Server
- Located at: `task_1_synthetic_fire_system/api_server.py`
- Provides REST API endpoints for fire detection data
- Connects directly to the Synthetic Fire Prediction System

### FireDetectionDashboard Component
- Located at: `saafe-lovable/src/components/FireDetectionDashboard.tsx`
- React component that displays fire detection data
- Fetches data from the API server every 30 seconds

### API Communication Module
- Located at: `saafe-lovable/src/api/fireDetection.ts`
- Handles communication between the frontend and API server

## Data Displayed

The integrated dashboard shows:

1. **Fire Risk Score**: Color-coded gauge showing current fire risk
2. **Gas Sensor Readings**: Visual representation of CO, CO2, and smoke levels
3. **Environmental Data**: Temperature, humidity, and pressure readings
4. **Thermal Camera Data**: Visualization of thermal sensor data

## Documentation

- `FIRE_DETECTION_INTEGRATION.md` - Detailed technical documentation
- `INTEGRATION_SUMMARY.md` - Summary of the integration work
- `test_integration.py` - Script to test the integration

## Troubleshooting

If you encounter issues:

1. Ensure both the API server and frontend are running
2. Check that AWS credentials are properly configured
3. Verify that the S3 bucket contains sensor data
4. Check the browser console for frontend errors
5. Check the API server logs for backend errors

## Future Enhancements

Possible future improvements include:

1. WebSocket integration for real-time updates
2. Historical data visualization
3. Alert notification system
4. User authentication and authorization
5. Mobile-responsive design