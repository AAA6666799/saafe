"""
API Server for Synthetic Fire Prediction System

This module provides a REST API to access fire detection data
for integration with the SAAFE Global Command Center.
"""

import sys
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

# Add the synthetic fire system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from ai_fire_prediction_platform.system.manager import SystemManager
from ai_fire_prediction_platform.core.config import ConfigurationManager
from ai_fire_prediction_platform.hardware.abstraction import S3HardwareInterface
from ai_fire_prediction_platform.core.interfaces import SensorData

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Synthetic Fire Prediction System API",
              description="API for accessing fire detection data",
              version="1.0.0")

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system manager instance
system_manager: Optional[SystemManager] = None

def initialize_system():
    """Initialize the fire detection system components"""
    global system_manager
    
    try:
        config_manager = ConfigurationManager()
        system_manager = SystemManager(config_manager)
        
        # Start the system
        system_manager.start()
        
        print("System initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    success = initialize_system()
    if not success:
        print("Failed to initialize fire detection system")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown system on exit"""
    global system_manager
    if system_manager and system_manager.is_running:
        system_manager.stop()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Synthetic Fire Prediction System API", "status": "running"}

@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    global system_manager
    if not system_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = system_manager.get_status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.get("/api/sensor-data")
async def get_sensor_data():
    """Get latest sensor data"""
    global system_manager
    if not system_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get the hardware interface to fetch sensor data directly
        if hasattr(system_manager, 'hardware_interface'):
            sensor_data = system_manager.hardware_interface.get_sensor_data()
            if sensor_data:
                # Convert sensor data to JSON-serializable format
                data = {
                    "timestamp": sensor_data.timestamp,
                    "gas_readings": sensor_data.gas_readings,
                    "environmental_data": sensor_data.environmental_data,
                    "sensor_health": sensor_data.sensor_health
                }
                
                # Process thermal data if available
                if sensor_data.thermal_frame is not None:
                    # Convert numpy array to list for JSON serialization
                    thermal_data = sensor_data.thermal_frame.tolist()
                    data["thermal_frame"] = thermal_data
                    data["thermal_stats"] = {
                        "max": float(np.max(sensor_data.thermal_frame)),
                        "min": float(np.min(sensor_data.thermal_frame)),
                        "mean": float(np.mean(sensor_data.thermal_frame))
                    }
                else:
                    data["thermal_frame"] = None
                    data["thermal_stats"] = None
                
                return {
                    "status": "success",
                    "data": data
                }
            else:
                raise HTTPException(status_code=404, detail="No sensor data available")
        else:
            raise HTTPException(status_code=500, detail="Hardware interface not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sensor data: {str(e)}")

@app.get("/api/prediction")
async def get_prediction():
    """Get latest prediction"""
    global system_manager
    if not system_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        prediction = system_manager.get_latest_prediction()
        if prediction:
            return {
                "status": "success",
                "data": {
                    "timestamp": prediction.timestamp,
                    "fire_probability": prediction.fire_probability,
                    "confidence_score": prediction.confidence_score,
                    "lead_time_estimate": prediction.lead_time_estimate,
                    "contributing_factors": prediction.contributing_factors,
                    "model_ensemble_votes": prediction.model_ensemble_votes
                }
            }
        else:
            raise HTTPException(status_code=404, detail="No prediction available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting prediction: {str(e)}")

@app.get("/api/risk-assessment")
async def get_risk_assessment():
    """Get current risk assessment"""
    global system_manager
    if not system_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        risk_assessment = system_manager.get_current_risk_assessment()
        if risk_assessment:
            return {
                "status": "success",
                "data": {
                    "timestamp": risk_assessment.timestamp,
                    "risk_level": risk_assessment.risk_level,
                    "fire_probability": risk_assessment.fire_probability,
                    "confidence_level": risk_assessment.confidence_level,
                    "contributing_sensors": risk_assessment.contributing_sensors,
                    "recommended_actions": risk_assessment.recommended_actions,
                    "escalation_required": risk_assessment.escalation_required
                }
            }
        else:
            raise HTTPException(status_code=404, detail="No risk assessment available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting risk assessment: {str(e)}")

@app.get("/api/alert")
async def get_latest_alert():
    """Get latest alert"""
    global system_manager
    if not system_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        alert = system_manager.get_latest_alert()
        if alert:
            return {
                "status": "success",
                "data": {
                    "alert_level": {
                        "level": alert.alert_level.level,
                        "description": alert.alert_level.description,
                        "icon": alert.alert_level.icon
                    },
                    "risk_score": alert.risk_score,
                    "confidence": alert.confidence,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "context_info": alert.context_info
                }
            }
        else:
            raise HTTPException(status_code=404, detail="No alert available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alert: {str(e)}")

@app.get("/api/fire-detection-data")
async def get_fire_detection_data():
    """Get comprehensive fire detection data for dashboard"""
    global system_manager
    if not system_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get sensor data
        sensor_data_response = await get_sensor_data()
        sensor_data = sensor_data_response.get("data", {}) if sensor_data_response.get("status") == "success" else {}
        
        # Get prediction
        prediction_response = await get_prediction()
        prediction_data = prediction_response.get("data", {}) if prediction_response.get("status") == "success" else {}
        
        # Get risk assessment
        risk_response = await get_risk_assessment()
        risk_data = risk_response.get("data", {}) if risk_response.get("status") == "success" else {}
        
        # Get alert
        alert_response = await get_latest_alert()
        alert_data = alert_response.get("data", {}) if alert_response.get("status") == "success" else {}
        
        # Combine all data
        combined_data = {
            "sensor_data": sensor_data,
            "prediction": prediction_data,
            "risk_assessment": risk_data,
            "alert": alert_data,
            "last_updated": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "data": combined_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting fire detection data: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)