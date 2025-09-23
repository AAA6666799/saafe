# SAAFE Fire Detection Dashboard

This is the React/Vite dashboard for the SAAFE fire detection system, specifically configured for the kitchen fire detection device.

## Features

- Real-time fire risk monitoring
- Thermal camera data visualization
- Gas sensor readings (VOC, CO, NO2)
- Environmental data (temperature, humidity, pressure)
- Risk assessment and alert system
- Historical data trends

## Prerequisites

- Node.js (version 16 or higher)
- npm (comes with Node.js)

## Setup

1. Install dependencies:
   ```bash
   npm install
   cd backend && npm install && cd ..
   ```

## Development Mode

To run the dashboard in development mode with hot reloading:

```bash
./start_dashboard.sh
```

This will start:
- Frontend development server on http://localhost:5173
- Backend API server on http://localhost:8000

## Production Mode

To build and serve the dashboard in production mode:

```bash
./build_and_serve.sh
```

This will:
1. Build the React frontend for production
2. Copy the built files to the backend
3. Start the backend server which serves both the API and the frontend

The production dashboard will be available at http://localhost:8000

## API Endpoints

The backend provides the following API endpoints:

- `GET /api/fire-detection-data` - Returns current fire detection data
- `GET /api/status` - Returns system status information

## Live Data Integration

The dashboard connects to live data from AWS S3:

- **S3 Bucket**: `data-collector-of-first-device`
- **Data Sources**: Thermal camera and gas sensor readings
- **Update Frequency**: Real-time (as data arrives in S3)

## Configuration

The dashboard is configured to:
- Connect to backend API at http://localhost:8000
- Display data from the kitchen fire detection device
- Send alerts to ch.ajay1707@gmail.com (configured in the main system)

## Directory Structure

```
saafe-lovable/
├── src/
│   ├── components/          # React components
│   ├── api/                 # API integration functions
│   ├── data/                # Sample data files
│   └── styles/              # CSS styles
├── backend/                 # Express.js backend server
│   └── server.js            # Main backend server
├── start_dashboard.sh       # Development startup script
├── build_and_serve.sh       # Production build and serve script
└── vite.config.ts           # Vite configuration
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed: `npm install`
2. Check that ports 5173 (frontend) and 8000 (backend) are available
3. Verify that Node.js is properly installed
4. Check the browser console for any JavaScript errors