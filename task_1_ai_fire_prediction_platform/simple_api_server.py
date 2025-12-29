"""
Simple API Server for Synthetic Fire Prediction System

This module provides a REST API with mock data to test the frontend integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import random
import json

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Synthetic Fire Prediction System API", "status": "running"}

@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    return {
        "status": "success",
        "data": {
            "system_status": "running",
            "uptime": "2 hours",
            "last_update": datetime.now().isoformat()
        }
    }

@app.get("/api/sensor-data")
async def get_sensor_data():
    """Get mock sensor data"""
    # Generate mock thermal data (20x20 grid with temperature values)
    thermal_frame = [[random.uniform(20, 40) for _ in range(20)] for _ in range(20)]
    
    # Flatten the 2D array to calculate stats
    flat_temps = [temp for row in thermal_frame for temp in row]
    
    return {
        "status": "success",
        "data": {
            "timestamp": datetime.now().timestamp(),
            "gas_readings": {
                "CO": random.uniform(0, 50),
                "NO2": random.uniform(0, 30),
                "VOC": random.uniform(0, 100)
            },
            "environmental_data": {
                "temperature": random.uniform(20, 30),
                "humidity": random.uniform(40, 60),
                "pressure": random.uniform(1000, 1020)
            },
            "sensor_health": {
                "thermal_camera": 98,
                "gas_sensors": 95,
                "environmental_sensors": 97
            },
            "thermal_frame": thermal_frame,
            "thermal_stats": {
                "max": max(flat_temps),
                "min": min(flat_temps),
                "mean": sum(flat_temps) / len(flat_temps)
            }
        }
    }

@app.get("/api/prediction")
async def get_prediction():
    """Get mock prediction"""
    return {
        "status": "success",
        "data": {
            "timestamp": datetime.now().timestamp(),
            "fire_probability": random.uniform(0, 100),
            "confidence_score": random.uniform(80, 95),
            "lead_time_estimate": random.uniform(5, 30),
            "contributing_factors": {
                "temperature_anomaly": random.uniform(0, 1),
                "gas_concentration": random.uniform(0, 1),
                "historical_patterns": random.uniform(0, 1)
            },
            "model_ensemble_votes": {
                "temporal_model": random.choice([0, 1]),
                "baseline_model": random.choice([0, 1]),
                "ensemble_model": random.choice([0, 1])
            }
        }
    }

@app.get("/api/risk-assessment")
async def get_risk_assessment():
    """Get mock risk assessment"""
    risk_levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    
    return {
        "status": "success",
        "data": {
            "timestamp": datetime.now().timestamp(),
            "risk_level": random.choice(risk_levels),
            "fire_probability": random.uniform(0, 100),
            "confidence_level": random.uniform(80, 95),
            "contributing_sensors": ["thermal_camera", "gas_sensors"],
            "recommended_actions": ["Increase monitoring frequency", "Verify sensor readings"],
            "escalation_required": random.choice([True, False])
        }
    }

@app.get("/api/alert")
async def get_latest_alert():
    """Get mock alert"""
    alert_levels = [
        {"level": 1, "description": "Normal", "icon": "✅"},
        {"level": 2, "description": "Mild Anomaly", "icon": "ℹ️"},
        {"level": 3, "description": "Elevated Risk", "icon": "⚠️"},
        {"level": 4, "description": "Critical Fire Alert", "icon": "🚨"}
    ]
    
    return {
        "status": "success",
        "data": {
            "alert_level": random.choice(alert_levels),
            "risk_score": random.uniform(0, 100),
            "confidence": random.uniform(80, 95),
            "message": "System operating normally",
            "timestamp": datetime.now().isoformat(),
            "context_info": {}
        }
    }

@app.get("/api/fire-detection-data")
async def get_fire_detection_data():
    """Get comprehensive mock fire detection data for dashboard"""
    # Get all data
    sensor_data_response = await get_sensor_data()
    prediction_response = await get_prediction()
    risk_response = await get_risk_assessment()
    alert_response = await get_latest_alert()
    
    # Combine all data
    combined_data = {
        "sensor_data": sensor_data_response.get("data", {}),
        "prediction": prediction_response.get("data", {}),
        "risk_assessment": risk_response.get("data", {}),
        "alert": alert_response.get("data", {}),
        "last_updated": datetime.now().isoformat()
    }
    
    return {
        "status": "success",
        "data": combined_data
    }

if __name__ == "__main__":
    import uvicorn
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)