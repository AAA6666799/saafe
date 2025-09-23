import { useState, useEffect } from "react";
import { fetchFireDetectionData } from "../api/fireDetection";

// Define types for our data
interface GasReadings {
  co?: number;
  no2?: number;
  voc?: number;
  CO?: number;
  NO2?: number;
  VOC?: number;
}

interface EnvironmentalData {
  temperature: number;
  humidity: number;
  pressure: number;
  voc?: number;
}

interface ThermalStats {
  max: number;
  min: number;
  mean: number;
}

interface SensorData {
  timestamp: number;
  thermal_frame: number[][];
  thermal_stats: ThermalStats;
  gas_readings: GasReadings;
  environmental_data: EnvironmentalData;
  sensor_health: Record<string, number>;
}

interface PredictionData {
  timestamp: number;
  fire_probability: number;
  confidence_score: number;
  lead_time_estimate: number;
  contributing_factors: Record<string, number>;
  model_ensemble_votes: Record<string, number>;
}

interface RiskAssessment {
  timestamp: number;
  risk_level: string;
  fire_probability: number;
  confidence_level: number;
  contributing_sensors: string[];
  recommended_actions: string[];
  escalation_required: boolean;
}

interface AlertLevel {
  level: number;
  description: string;
  icon: string;
}

interface AlertData {
  alert_level: AlertLevel;
  risk_score: number;
  confidence: number;
  message: string;
  timestamp: string;
  context_info: Record<string, any>;
}

interface FireDetectionData {
  sensor_data: SensorData;
  prediction: PredictionData;
  risk_assessment: RiskAssessment;
  alert: AlertData;
  last_updated: string;
}

// Fire Detection Dashboard Component
export default function FireDetectionDashboard() {
  const [fireData, setFireData] = useState<FireDetectionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch real data from the synthetic fire system
  const fetchFireData = async () => {
    try {
      setLoading(true);
      const data = await fetchFireDetectionData();
      setFireData(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch fire detection data: " + (err as Error).message);
      console.error("Error fetching fire data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFireData();
    // Set up polling for real-time updates
    const interval = setInterval(fetchFireData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Determine alert color based on level
  const getAlertColor = (level: number) => {
    switch(level) {
      case 4: return "#ef4444"; // Critical
      case 3: return "#f97316"; // Elevated
      case 2: return "#f59e0b"; // Mild
      default: return "#34d399"; // Normal
    }
  };

  // Get alert level description
  const getAlertDescription = (level: number) => {
    switch(level) {
      case 4: return "Immediate action required";
      case 3: return "Increased monitoring required";
      case 2: return "Monitor for changes";
      default: return "Normal conditions";
    }
  };

  if (loading) {
    return (
      <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white", marginTop: 12 }}>
        <div style={{ padding:12 }}>
          <strong style={{ color:"#0f172a" }}>🔥 Fire Detection Dashboard</strong>
        </div>
        <div style={{ padding: "20px", textAlign: "center" }}>
          <div>Loading fire detection data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white", marginTop: 12 }}>
        <div style={{ padding:12 }}>
          <strong style={{ color:"#0f172a" }}>🔥 Fire Detection Dashboard</strong>
        </div>
        <div style={{ padding: "20px", textAlign: "center", color: "#ef4444" }}>
          <div>{error}</div>
          <button onClick={fetchFireData} style={{ marginTop: 10, padding: "8px 12px", background: "#059669", color: "white", border: "none", borderRadius: 8, cursor: "pointer" }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!fireData) {
    return (
      <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white", marginTop: 12 }}>
        <div style={{ padding:12 }}>
          <strong style={{ color:"#0f172a" }}>🔥 Fire Detection Dashboard</strong>
        </div>
        <div style={{ padding: "20px", textAlign: "center" }}>
          <div>No fire detection data available</div>
          <button onClick={fetchFireData} style={{ marginTop: 10, padding: "8px 12px", background: "#059669", color: "white", border: "none", borderRadius: 8, cursor: "pointer" }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Extract data for display
  const { sensor_data, prediction, alert } = fireData;
  const riskScore = (prediction?.fire_probability || 0) * 100;
  const alertLevel = alert?.alert_level?.level || 1;
  const alertDescription = alert?.alert_level?.description || "Normal";
  
  // Process gas readings
  const gasReadings = sensor_data?.gas_readings || {};
  const gasEntries = Object.entries(gasReadings).filter(([key]) => 
    key !== 'timestamp' && key !== 'sensor_health'
  );

  // Process environmental data
  const environmentalData = sensor_data?.environmental_data || {
    temperature: 0,
    humidity: 0,
    pressure: 0
  };

  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white", marginTop: 12 }}>
      <div style={{ padding:12, display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <strong style={{ color:"#0f172a" }}>🔥 Fire Detection Dashboard</strong>
        <button 
          onClick={fetchFireData} 
          style={{ border:"1px solid #e5e7eb", background:"white", color:"#0f172a", padding:"6px 10px", borderRadius:8, cursor:"pointer", fontSize: "14px" }}
        >
          🔄 Refresh
        </button>
      </div>
      
      <div style={{ padding: "0 12px 12px" }}>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12, marginBottom: 12 }}>
          {/* Risk Gauge */}
          <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
            <div style={{ fontSize:12, color:"#64748b", marginBottom: 8 }}>Fire Risk Score</div>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height: 100 }}>
              <div style={{ 
                width: 80, 
                height: 80, 
                borderRadius: "50%", 
                background: `conic-gradient(${getAlertColor(alertLevel)} ${riskScore * 3.6}deg, #e2e8f0 0deg)`,
                position: "relative"
              }}>
                <div style={{ 
                  position: "absolute", 
                  top: 8, 
                  left: 8, 
                  right: 8, 
                  bottom: 8, 
                  background: "white", 
                  borderRadius: "50%", 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "center",
                  fontWeight: "bold",
                  fontSize: 20
                }}>
                  {Math.round(riskScore)}
                </div>
              </div>
            </div>
            <div style={{ textAlign: "center", marginTop: 8 }}>
              <div style={{ fontWeight: "bold", color: getAlertColor(alertLevel) }}>
                {alertDescription}
              </div>
              <div style={{ fontSize: 12, color: "#64748b" }}>
                {getAlertDescription(alertLevel)}
              </div>
            </div>
          </div>
          
          {/* Gas Readings */}
          <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
            <div style={{ fontSize:12, color:"#64748b", marginBottom: 8 }}>Gas Sensors</div>
            {gasEntries.length > 0 ? (
              gasEntries.map(([gas, value]) => (
                <div key={gas} style={{ marginBottom: 8 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
                    <span>{gas.toUpperCase()}</span>
                    <span>{typeof value === 'number' ? value.toFixed(1) : 'N/A'}</span>
                  </div>
                  <div style={{ height:6, background:"#f1f5f9", borderRadius:9999, marginTop:4 }}>
                    <div 
                      style={{ 
                        width:`${Math.min(100, typeof value === 'number' ? value : 0)}%`, 
                        height:"100%", 
                        background: (typeof value === 'number' && value > 50) ? "#ef4444" : 
                                   (typeof value === 'number' && value > 30) ? "#f97316" : "#34d399", 
                        borderRadius:9999 
                      }} 
                    />
                  </div>
                </div>
              ))
            ) : (
              <div style={{ fontSize: 12, color: "#64748b" }}>No gas data available</div>
            )}
          </div>
          
          {/* Environmental Data */}
          <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
            <div style={{ fontSize:12, color:"#64748b", marginBottom: 8 }}>Environmental</div>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom: 6 }}>
              <span>Temperature</span>
              <span style={{ fontWeight: "bold" }}>
                {environmentalData.temperature.toFixed(1)}°C
              </span>
            </div>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom: 6 }}>
              <span>Humidity</span>
              <span style={{ fontWeight: "bold" }}>
                {environmentalData.humidity.toFixed(1)}%
              </span>
            </div>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom: 6 }}>
              <span>Pressure</span>
              <span style={{ fontWeight: "bold" }}>
                {environmentalData.pressure.toFixed(1)} hPa
              </span>
            </div>
            <div style={{ marginTop: 8, fontSize: 12, color: "#64748b" }}>
              Updated: {new Date(sensor_data?.timestamp * 1000 || Date.now()).toLocaleTimeString()}
            </div>
          </div>
        </div>
        
        {/* Thermal Image Representation */}
        <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom: 8 }}>
            <div style={{ fontSize:12, color:"#64748b" }}>Thermal Camera Data</div>
            <div style={{ fontSize:12, color:"#64748b" }}>
              Max: {sensor_data?.thermal_stats?.max?.toFixed(1) || 'N/A'}°C
            </div>
          </div>
          <div style={{ 
            height: 150, 
            background: "linear-gradient(to right, #0000ff, #00ff00, #ffff00, #ff0000)",
            borderRadius: 8,
            position: "relative",
            overflow: "hidden"
          }}>
            {/* Simplified thermal visualization */}
            {sensor_data?.thermal_frame ? (
              sensor_data.thermal_frame.map((row, i) => (
                <div key={i} style={{ display: "flex", height: "5%" }}>
                  {row.map((temp, j) => (
                    <div 
                      key={`${i}-${j}`} 
                      style={{ 
                        flex: 1, 
                        background: `hsl(${(100 - (temp || 0)) * 2.4}, 100%, 50%)`,
                        opacity: 0.8
                      }} 
                    />
                  ))}
                </div>
              ))
            ) : (
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center", 
                height: "100%", 
                color: "#64748b" 
              }}>
                No thermal data available
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}