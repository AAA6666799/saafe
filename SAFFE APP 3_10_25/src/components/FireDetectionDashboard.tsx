import { useState, useEffect } from "react";
import { fetchFireDetectionData, predictSaafeModel, predictFeatures18Model, predictKaggleModel, predictTensorFlowModel } from "../api/fireDetection";

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
  std?: number;
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

interface DataProvenance {
  source: string;
  bucket: string;
  timestamp: string;
  sensor_timestamp?: string;
  data_age_seconds?: number;
  thermal_file?: {
    key: string;
    last_modified: string;
  };
  gas_file?: {
    key: string;
    last_modified: string;
  };
}

interface FireDetectionData {
  sensor_data: SensorData;
  prediction: PredictionData;
  risk_assessment: RiskAssessment;
  alert: AlertData;
  data_provenance?: DataProvenance;
  last_updated: string;
}

// Model prediction interface
interface ModelPrediction {
  status: string;
  model: string;
  prediction: any;
  fire_detected: boolean;
  timestamp: string;
  error?: string;
}

// Fire Detection Dashboard Component
export default function FireDetectionDashboard() {
  const [fireData, setFireData] = useState<FireDetectionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Model predictions state
  const [modelPredictions, setModelPredictions] = useState<Record<string, ModelPrediction>>({});
  const [predictingModels, setPredictingModels] = useState<Set<string>>(new Set());

  // Email settings state
  const [userEmail, setUserEmail] = useState(() => {
    return localStorage.getItem('saafe-user-email') || '';
  });
  const [emailSaving, setEmailSaving] = useState(false);
  const [emailStatus, setEmailStatus] = useState<string | null>(null);
  const [testEmailSending, setTestEmailSending] = useState(false);

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

  // Function to run model prediction
  const runModelPrediction = async (modelName: string, payload: any) => {
    setPredictingModels(prev => new Set(prev).add(modelName));

    try {
      let result: ModelPrediction;

      switch (modelName) {
        case 'saafe':
          result = await predictSaafeModel(payload);
          break;
        case 'features18':
          result = await predictFeatures18Model(payload);
          break;
        case 'kaggle':
          result = await predictKaggleModel(payload);
          break;
        case 'tensorflow':
          result = await predictTensorFlowModel(payload);
          break;
        default:
          throw new Error(`Unknown model: ${modelName}`);
      }

      setModelPredictions(prev => ({
        ...prev,
        [modelName]: result
      }));

    } catch (error) {
      console.error(`Error predicting with ${modelName}:`, error);
      setModelPredictions(prev => ({
        ...prev,
        [modelName]: {
          status: "error",
          model: modelName,
          prediction: null,
          fire_detected: false,
          timestamp: new Date().toISOString(),
          error: (error as Error).message
        }
      }));
    } finally {
      setPredictingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(modelName);
        return newSet;
      });
    }
  };

  // Function to save user email
  const saveUserEmail = async () => {
    if (!userEmail.trim()) {
      setEmailStatus('Please enter a valid email address');
      return;
    }

    console.log('Saving email:', userEmail);
    setEmailSaving(true);
    setEmailStatus('Saving...');

    try {
      // Save to localStorage first
      localStorage.setItem('saafe-user-email', userEmail);
      console.log('Email saved to localStorage');

      // Send to backend to update configuration
      const response = await fetch('/api/update-email-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ recipient_email: userEmail })
      });

      console.log('Backend response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Backend error response:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('Backend response:', result);

      setEmailStatus('✅ Email configuration updated successfully!');
      setTimeout(() => setEmailStatus(null), 3000);
    } catch (error) {
      console.error('Error saving email:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setEmailStatus(`❌ Failed to save email: ${errorMessage}`);
      setTimeout(() => setEmailStatus(null), 5000);
    } finally {
      setEmailSaving(false);
    }
  };

  // Function to send test email
  const sendTestEmail = async () => {
    if (!userEmail.trim()) {
      setEmailStatus('Please configure your email first');
      return;
    }

    console.log('Sending test email to:', userEmail);
    setTestEmailSending(true);
    setEmailStatus('Sending test email...');

    try {
      const response = await fetch('/api/send-test-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ test_email: userEmail })
      });

      console.log('Test email response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Test email error response:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('Test email response:', result);

      setEmailStatus('✅ Test email sent successfully! Check your inbox.');
      setTimeout(() => setEmailStatus(null), 5000);
    } catch (error) {
      console.error('Error sending test email:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setEmailStatus(`❌ Failed to send test email: ${errorMessage}`);
      setTimeout(() => setEmailStatus(null), 5000);
    } finally {
      setTestEmailSending(false);
    }
  };

  // Function to run all models with current sensor data
  const runAllPredictions = async () => {
    if (!fireData?.sensor_data) return;

    // Prepare payload for Saafe and TensorFlow models
    const saafePayload = {
      frame: Math.floor(Date.now() / 1000),
      timestamp: new Date().toISOString(),
      features: {
        t_mean: fireData.sensor_data.thermal_stats.mean,
        t_std: fireData.sensor_data.thermal_stats.std || 2.0,
        t_max: fireData.sensor_data.thermal_stats.max,
        t_p95: fireData.sensor_data.thermal_stats.max * 0.95,
        t_hot_area_pct: 5.0,
        t_hot_largest_blob_pct: 3.0,
        t_grad_mean: 0.5,
        t_grad_std: 0.2,
        t_diff_mean: 0.1,
        t_diff_std: 0.05,
        flow_mag_mean: 0.1,
        flow_mag_std: 0.05,
        tproxy_val: fireData.sensor_data.thermal_stats.mean,
        tproxy_delta: 1.0,
        tproxy_vel: 0.5,
        CO: fireData.sensor_data.gas_readings.co || 0.5,
        VOC: fireData.sensor_data.gas_readings.voc || 50,
        NO2: fireData.sensor_data.gas_readings.no2 || 0.1,
        CO_diff: 0.02,
        VOC_diff: 0.03,
        NO2_diff: 0.0,
        VOC_ma5: fireData.sensor_data.gas_readings.voc || 50,
        CO_ma5: fireData.sensor_data.gas_readings.co || 0.5,
        NO2_ma5: fireData.sensor_data.gas_readings.no2 || 0.1,
        VOC_z: 0.1,
        CO_z: 0.1,
        NO2_z: 0.0,
        temp_rise_c_per_min: 0.2,
        temp_slope_30s: 0.1,
        gas_var_30s: 0.05,
        delta_temp_30s: 0.2,
        delta_gas_10s: 0.01,
        spike_count_voc_2m: 0,
        temp_co_corr_lag_0s: 0.1,
        temp_co_corr_lag_15s: 0.08,
        temp_co_corr_lag_60s: 0.05,
        temp_voc_corr_lag_0s: 0.12,
        temp_voc_corr_lag_15s: 0.1,
        temp_voc_corr_lag_60s: 0.08,
        temp_co_xcorr_max_abs: 0.15,
        temp_voc_xcorr_max_abs: 0.18,
        is_weekend: 0,
        asleep_window: 0,
        hrblk_0: 0,
        hrblk_1: 0,
        hrblk_2: 0,
        hrblk_3: 0,
        hrblk_4: 0,
        hrblk_5: 0
      },
      decision_threshold: 0.4
    };

    // Prepare payload for 18 Features model
    const features18Payload = {
      data: {
        features_dict: {
          t_mean: fireData.sensor_data.thermal_stats.mean,
          t_std: 2.0,
          t_max: fireData.sensor_data.thermal_stats.max,
          t_p95: fireData.sensor_data.thermal_stats.max * 0.95,
          t_hot_area_pct: 5.0,
          t_hot_largest_blob_pct: 3.0,
          t_grad_mean: 0.5,
          t_grad_std: 0.2,
          t_diff_mean: 0.1,
          t_diff_std: 0.05,
          flow_mag_mean: 0.1,
          flow_mag_std: 0.05,
          gas_val: fireData.sensor_data.gas_readings.voc || 400,
          gas_delta: 5.0,
          gas_vel: 0.5,
          tproxy_val: fireData.sensor_data.thermal_stats.mean,
          tproxy_delta: 1.0,
          tproxy_vel: 0.2
        }
      },
      threshold: 0.5
    };

    // Prepare payload for Kaggle model
    const kagglePayload = {
      data: {
        Temperature: fireData.sensor_data.environmental_data.temperature,
        Humidity: fireData.sensor_data.environmental_data.humidity,
        TVOC: fireData.sensor_data.gas_readings.voc || 50,
        eCO2: 420,
        PM1_0: 1.2,
        PM2_5: 2.3,
        PM10: 3.4,
        Pressure: fireData.sensor_data.environmental_data.pressure,
        Raw_H2: 14500,
        Raw_Ethanol: 21000,
        CNT: 0,
        UTC: Math.floor(Date.now() / 1000),
        NC0_5: 0,
        NC1_0: 0,
        NC2_5: 0
      }
    };

    // Run all predictions
    await Promise.all([
      runModelPrediction('saafe', saafePayload),
      runModelPrediction('features18', features18Payload),
      runModelPrediction('kaggle', kagglePayload),
      runModelPrediction('tensorflow', saafePayload) // TensorFlow uses same payload as Saafe
    ]);
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

      {/* Email Configuration Section */}
      <div style={{ borderTop:"1px solid #e5e7eb", padding:12, background:"#f8fafc" }}>
        <div style={{ fontSize:14, fontWeight:"bold", marginBottom: 8, color:"#0f172a" }}>📧 Email Alert Configuration</div>
        <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom: 8 }}>
          <input
            type="email"
            placeholder="Enter your email for fire alerts"
            value={userEmail}
            onChange={(e) => setUserEmail(e.target.value)}
            style={{ flex:1, border:"1px solid #e5e7eb", borderRadius:8, padding:"8px 12px", outline:"none", fontSize:14 }}
          />
          <button
            onClick={saveUserEmail}
            disabled={emailSaving || !userEmail.trim()}
            style={{
              border:"1px solid #059669",
              background: emailSaving ? "#f3f4f6" : "#059669",
              color: emailSaving ? "#64748b" : "white",
              padding:"8px 16px",
              borderRadius:8,
              cursor: emailSaving || !userEmail.trim() ? "not-allowed" : "pointer",
              fontSize:14
            }}
          >
            {emailSaving ? "Saving..." : "Save Email"}
          </button>
          <button
            onClick={sendTestEmail}
            disabled={testEmailSending || !userEmail.trim()}
            style={{
              border:"1px solid #3b82f6",
              background: testEmailSending ? "#f3f4f6" : "#3b82f6",
              color: testEmailSending ? "#64748b" : "white",
              padding:"8px 16px",
              borderRadius:8,
              cursor: testEmailSending || !userEmail.trim() ? "not-allowed" : "pointer",
              fontSize:14
            }}
          >
            {testEmailSending ? "Sending..." : "Test Email"}
          </button>
        </div>
        {emailStatus && (
          <div style={{
            fontSize:12,
            color: emailStatus.startsWith('✅') ? "#059669" : emailStatus.startsWith('❌') ? "#ef4444" : "#64748b",
            marginBottom: 4,
            padding: "4px 8px",
            background: "white",
            borderRadius: 4,
            border: "1px solid #e5e7eb"
          }}>
            {emailStatus}
          </div>
        )}
        <div style={{ fontSize:12, color:"#64748b" }}>
          You will receive email alerts when fire is detected by any AI model. Use "Test Email" to verify your configuration.
        </div>
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
              Sensor Reading: {fireData.data_provenance?.sensor_timestamp ?
                new Date(fireData.data_provenance.sensor_timestamp).toLocaleString() :
                new Date(sensor_data?.timestamp * 1000 || Date.now()).toLocaleString()}
              {fireData.data_provenance?.data_age_seconds && (
                <span style={{ marginLeft: 8, color: fireData.data_provenance.data_age_seconds > 300 ? "#ef4444" : "#64748b" }}>
                  ({Math.floor(fireData.data_provenance.data_age_seconds / 60)}m ago)
                </span>
              )}
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

        {/* Model Predictions Section */}
        <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12, marginTop: 12 }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom: 12 }}>
            <div style={{ fontSize:12, color:"#64748b" }}>AI Model Predictions</div>
            <button
              onClick={runAllPredictions}
              disabled={predictingModels.size > 0}
              style={{
                border:"1px solid #e5e7eb",
                background: predictingModels.size > 0 ? "#f3f4f6" : "#059669",
                color: predictingModels.size > 0 ? "#64748b" : "white",
                padding:"6px 12px",
                borderRadius:8,
                cursor: predictingModels.size > 0 ? "not-allowed" : "pointer",
                fontSize: "14px"
              }}
            >
              {predictingModels.size > 0 ? "Running..." : "Run All Predictions"}
            </button>
          </div>

          <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(250px, 1fr))", gap:12 }}>
            {/* Saafe Model */}
            <div style={{ border:"1px solid #e5e7eb", borderRadius:8, padding:12 }}>
              <div style={{ fontSize:14, fontWeight:"bold", marginBottom: 8 }}>Saafe Model</div>
              {modelPredictions.saafe ? (
                <div>
                  <div style={{ display:"flex", alignItems:"center", marginBottom: 4 }}>
                    <span style={{ fontSize:12, color: modelPredictions.saafe.fire_detected ? "#ef4444" : "#34d399" }}>
                      {modelPredictions.saafe.fire_detected ? "🔥 Fire Detected" : "✅ No Fire"}
                    </span>
                  </div>
                  {modelPredictions.saafe.prediction && (
                    <div style={{ fontSize:12, color:"#64748b" }}>
                      Label: {modelPredictions.saafe.prediction.label || 'N/A'}
                      <br />
                      Probability: {modelPredictions.saafe.prediction.fire_probability ?
                        (modelPredictions.saafe.prediction.fire_probability * 100).toFixed(2) + '%' : 'N/A'}
                    </div>
                  )}
                  {modelPredictions.saafe.error && (
                    <div style={{ fontSize:12, color:"#ef4444" }}>
                      Error: {modelPredictions.saafe.error}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize:12, color:"#64748b" }}>No prediction yet</div>
              )}
            </div>

            {/* 18 Features Model */}
            <div style={{ border:"1px solid #e5e7eb", borderRadius:8, padding:12 }}>
              <div style={{ fontSize:14, fontWeight:"bold", marginBottom: 8 }}>18 Features Model</div>
              {modelPredictions.features18 ? (
                <div>
                  <div style={{ display:"flex", alignItems:"center", marginBottom: 4 }}>
                    <span style={{ fontSize:12, color: modelPredictions.features18.fire_detected ? "#ef4444" : "#34d399" }}>
                      {modelPredictions.features18.fire_detected ? "🔥 Fire Detected" : "✅ No Fire"}
                    </span>
                  </div>
                  {modelPredictions.features18.prediction && (
                    <div style={{ fontSize:12, color:"#64748b" }}>
                      Score: {modelPredictions.features18.prediction.score ?
                        (modelPredictions.features18.prediction.score * 100).toFixed(2) + '%' : 'N/A'}
                    </div>
                  )}
                  {modelPredictions.features18.error && (
                    <div style={{ fontSize:12, color:"#ef4444" }}>
                      Error: {modelPredictions.features18.error}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize:12, color:"#64748b" }}>No prediction yet</div>
              )}
            </div>

            {/* Kaggle Base Model */}
            <div style={{ border:"1px solid #e5e7eb", borderRadius:8, padding:12 }}>
              <div style={{ fontSize:14, fontWeight:"bold", marginBottom: 8 }}>Kaggle Base Model</div>
              {modelPredictions.kaggle ? (
                <div>
                  <div style={{ display:"flex", alignItems:"center", marginBottom: 4 }}>
                    <span style={{ fontSize:12, color: modelPredictions.kaggle.fire_detected ? "#ef4444" : "#34d399" }}>
                      {modelPredictions.kaggle.fire_detected ? "🔥 Fire Detected" : "✅ No Fire"}
                    </span>
                  </div>
                  {modelPredictions.kaggle.prediction && (
                    <div style={{ fontSize:12, color:"#64748b" }}>
                      Score: {modelPredictions.kaggle.prediction.score ?
                        (modelPredictions.kaggle.prediction.score * 100).toFixed(2) + '%' : 'N/A'}
                    </div>
                  )}
                  {modelPredictions.kaggle.error && (
                    <div style={{ fontSize:12, color:"#ef4444" }}>
                      Error: {modelPredictions.kaggle.error}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize:12, color:"#64748b" }}>No prediction yet</div>
              )}
            </div>

            {/* TensorFlow Enhanced Model */}
            <div style={{ border:"1px solid #e5e7eb", borderRadius:8, padding:12 }}>
              <div style={{ fontSize:14, fontWeight:"bold", marginBottom: 8 }}>TensorFlow Enhanced</div>
              {modelPredictions.tensorflow ? (
                <div>
                  <div style={{ display:"flex", alignItems:"center", marginBottom: 4 }}>
                    <span style={{ fontSize:12, color: modelPredictions.tensorflow.fire_detected ? "#ef4444" : "#34d399" }}>
                      {modelPredictions.tensorflow.fire_detected ? "🔥 Fire Detected" : "✅ No Fire"}
                    </span>
                  </div>
                  {modelPredictions.tensorflow.prediction && (
                    <div style={{ fontSize:12, color:"#64748b" }}>
                      Label: {modelPredictions.tensorflow.prediction.label || 'N/A'}
                      <br />
                      Probability: {modelPredictions.tensorflow.prediction.fire_probability ?
                        (modelPredictions.tensorflow.prediction.fire_probability * 100).toFixed(2) + '%' : 'N/A'}
                    </div>
                  )}
                  {modelPredictions.tensorflow.error && (
                    <div style={{ fontSize:12, color:"#ef4444" }}>
                      Error: {modelPredictions.tensorflow.error}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize:12, color:"#64748b" }}>No prediction yet</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}