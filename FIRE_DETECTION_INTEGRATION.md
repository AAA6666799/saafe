# Fire Detection System Integration with SAAFE Global Command Center

This document explains how to integrate the Synthetic Fire Prediction System with the SAAFE Global Command Center (GCC).

## System Architecture

The integration consists of two main components:

1. **Synthetic Fire Prediction System API Server** - A Python FastAPI server that exposes fire detection data
2. **SAAFE Global Command Center** - A React frontend that displays the fire detection dashboard

## Prerequisites

- Python 3.7+
- Node.js 14+
- npm 6+
- AWS credentials configured (for S3 access)

## Setup Instructions

### 1. Start the Fire Detection API Server

Navigate to the fire detection system directory:
```bash
cd task_1_synthetic_fire_system
```

Start the API server:
```bash
./start_api.sh
```

The API server will be available at `http://localhost:8000`

### 2. Start the SAAFE Global Command Center

In a new terminal, navigate to the GCC directory:
```bash
cd saafe-lovable
```

Start the frontend:
```bash
./start_frontend.sh
```

The GCC will be available at `http://localhost:5173`

## API Endpoints

The Fire Detection API provides the following endpoints:

- `GET /api/status` - System status
- `GET /api/sensor-data` - Latest sensor data
- `GET /api/prediction` - Latest prediction
- `GET /api/risk-assessment` - Current risk assessment
- `GET /api/alert` - Latest alert
- `GET /api/fire-detection-data` - Comprehensive fire detection data (used by GCC)

## Integration Details

The integration is implemented through:

1. **FireDetectionDashboard Component** - A React component that fetches data from the Fire Detection API
2. **Fire Detection API Module** - TypeScript functions that communicate with the Python API
3. **API Server** - Python FastAPI server that exposes fire detection data

## Data Flow

1. The Synthetic Fire Prediction System continuously monitors sensors and makes predictions
2. The API Server exposes this data through REST endpoints
3. The GCC frontend periodically fetches data from the API
4. The FireDetectionDashboard component displays the data in a user-friendly format

## Customization

To customize the integration:

1. Modify the `FireDetectionDashboard.tsx` component to change the UI
2. Update the `fireDetection.ts` API module to modify data fetching
3. Extend the Python API server to expose additional endpoints
4. Adjust the polling interval in the FireDetectionDashboard component

## Troubleshooting

### API Server Issues

- Ensure AWS credentials are properly configured
- Check that the S3 bucket exists and contains data
- Verify that required Python packages are installed

### Frontend Issues

- Ensure Node.js and npm are properly installed
- Check that all dependencies are installed (`npm install`)
- Verify that the API server is running and accessible

### Common Errors

- "Failed to fetch fire detection data" - Check that the API server is running
- "No sensor data available" - Verify that the S3 bucket contains data
- CORS errors - Ensure CORS is properly configured in the API server

## Future Enhancements

1. Add real-time WebSocket support for live updates
2. Implement authentication and authorization
3. Add historical data visualization
4. Include more detailed sensor information
5. Add alert notification system