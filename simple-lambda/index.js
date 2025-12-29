exports.handler = async (event) => {
    // CORS headers
    const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
    };
    
    // Handle preflight requests
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: headers,
            body: ''
        };
    }
    
    // Mock fire detection data
    const mockData = {
        status: "success",
        data: {
            sensor_data: {
                timestamp: Math.floor(Date.now() / 1000),
                thermal_frame: Array(20).fill().map(() => Array(20).fill(25 + Math.random() * 10)),
                thermal_stats: {
                    max: 35 + Math.random() * 5,
                    min: 20 + Math.random() * 5,
                    mean: 25 + Math.random() * 5
                },
                gas_readings: {
                    voc: 50 + Math.random() * 20,
                    co: 0.5 + Math.random() * 0.3,
                    no2: 0.1 + Math.random() * 0.05
                },
                environmental_data: {
                    temperature: 25 + Math.random() * 5,
                    humidity: 40 + Math.random() * 20,
                    pressure: 1013 + Math.random() * 10
                },
                sensor_health: {
                    thermal_camera: 0.95 + Math.random() * 0.05,
                    gas_sensor: 0.98 + Math.random() * 0.02,
                    environmental: 0.97 + Math.random() * 0.03
                }
            },
            prediction: {
                timestamp: Math.floor(Date.now() / 1000),
                fire_probability: Math.random() * 0.2,
                confidence_score: 0.8 + Math.random() * 0.2,
                lead_time_estimate: 30 + Math.random() * 60,
                contributing_factors: {
                    "voc_level": Math.random(),
                    "temperature_spike": Math.random(),
                    "smoke_detected": Math.random()
                }
            },
            risk_assessment: {
                timestamp: Math.floor(Date.now() / 1000),
                risk_level: "low",
                fire_probability: Math.random() * 0.2,
                confidence_level: 0.85 + Math.random() * 0.15,
                contributing_sensors: ["thermal_camera", "gas_sensor"],
                recommended_actions: ["increase monitoring frequency", "verify ventilation"],
                escalation_required: false
            },
            alert: {
                alert_level: {
                    level: 1,
                    description: "Normal",
                    icon: "✅"
                },
                risk_score: Math.floor(Math.random() * 20),
                confidence: 0.9 + Math.random() * 0.1,
                message: "System operating normally",
                timestamp: new Date().toISOString()
            },
            last_updated: new Date().toISOString()
        }
    };
    
    return {
        statusCode: 200,
        headers: headers,
        body: JSON.stringify(mockData)
    };
};
