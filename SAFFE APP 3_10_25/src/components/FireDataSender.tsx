import { useState } from 'react';
import axios from 'axios';

// Alert state types
type AlertState = 'non-fire' | 'fire-predicted' | 'fire';

interface AlertResponse {
  status: string;
  message: string;
  alert_state: AlertState;
  timestamp: string;
  data?: any;
}

export default function FireDataSender() {
  const [loading, setLoading] = useState(false);
  const [lastResponse, setLastResponse] = useState<AlertResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Function to send alert to backend
  const sendAlert = async (alertState: AlertState) => {
    setLoading(true);
    setError(null);

    // Prepare payload based on alert state
    let payload: any;

    switch (alertState) {
      case 'non-fire':
        // Non-fire scenario - normal conditions
        payload = {
          frame: 9012,
          timestamp: new Date().toISOString(),
          features: {
            t_mean: 22.0, t_std: 0.1, t_max: 22.3, t_p95: 22.2,
            t_hot_area_pct: 0.0, t_hot_largest_blob_pct: 0.0,
            t_grad_mean: 0.0, t_grad_std: 0.0,
            t_diff_mean: 0.0, t_diff_std: 0.0,
            flow_mag_mean: 0.02, flow_mag_std: 0.002,
            tproxy_val: 22.1, tproxy_delta: 0.0, tproxy_vel: 0.0,
            CO: 0.03, VOC: 0.06, NO2: 0.004,
            CO_diff: -0.001, VOC_diff: -0.001, NO2_diff: 0.0,
            VOC_ma5: 0.06, CO_ma5: 0.03, NO2_ma5: 0.004,
            VOC_z: -0.15, CO_z: -0.15, NO2_z: -0.05,
            temp_rise_c_per_min: 0.0, temp_slope_30s: 0.0,
            gas_var_30s: 0.0, delta_temp_30s: 0.0, delta_gas_10s: 0.0,
            spike_count_voc_2m: 0,
            temp_co_corr_lag_0s: 0.0, temp_co_corr_lag_15s: 0.0, temp_co_corr_lag_60s: 0.0,
            temp_voc_corr_lag_0s: 0.0, temp_voc_corr_lag_15s: 0.0, temp_voc_corr_lag_60s: 0.0,
            temp_co_xcorr_max_abs: 0.005, temp_voc_xcorr_max_abs: 0.005,
            is_weekend: 0, asleep_window: 0,
            hrblk_0: 0, hrblk_1: 1, hrblk_2: 0, hrblk_3: 0, hrblk_4: 0, hrblk_5: 0
          },
          decision_threshold: 0.40
        };
        break;

      case 'fire-predicted':
        // Fire predicted scenario - early warning signs
        payload = {
          frame: 5678,
          timestamp: new Date().toISOString(),
          features: {
            t_mean: 44.0, t_std: 0.5, t_max: 28.0, t_p95: 27.5,
            t_hot_area_pct: 0.4, t_hot_largest_blob_pct: 0.1,
            t_grad_mean: 0.05, t_grad_std: 0.02, t_diff_mean: 0.03, t_diff_std: 0.01,
            flow_mag_mean: 0.1, flow_mag_std: 0.01,
            tproxy_val: 28.0, tproxy_delta: 0.2, tproxy_vel: 0.05,
            CO: 0.2, VOC: 0.5, NO2: 0.01,
            CO_diff: 0.02, VOC_diff: 0.03, NO2_diff: 0.0,
            VOC_ma5: 0.4, CO_ma5: 0.15, NO2_ma5: 0.01,
            VOC_z: 0.1, CO_z: 0.1, NO2_z: 0.0,
            temp_rise_c_per_min: 0.2, temp_slope_30s: 0.1,
            gas_var_30s: 0.05, delta_temp_30s: 0.2, delta_gas_10s: 0.01,
            spike_count_voc_2m: 0,
            temp_co_corr_lag_0s: 0.20, temp_co_corr_lag_15s: 0.08, temp_co_corr_lag_60s: 0.05,
            temp_voc_corr_lag_0s: 0.12, temp_voc_corr_lag_15s: 0.10, temp_voc_corr_lag_60s: 0.08,
            temp_co_xcorr_max_abs: 0.15, temp_voc_xcorr_max_abs: 0.18,
            is_weekend: 0, asleep_window: 4,
            hrblk_0: 0, hrblk_1: 0, hrblk_2: 2, hrblk_3: 0, hrblk_4: 5, hrblk_5: 0
          },
          decision_threshold: 0.4
        };
        break;

      case 'fire':
        // Fire detected scenario - critical conditions
        payload = {
          frame: 1234,
          timestamp: new Date().toISOString(),
          features: {
            t_mean: 28.12, t_std: 0.83, t_max: 74.56, t_p95: 71.92,
            t_hot_area_pct: 8.20, t_hot_largest_blob_pct: 5.47,
            t_grad_mean: 0.42, t_grad_std: 0.25, t_diff_mean: 0.18, t_diff_std: 0.09,
            flow_mag_mean: 0.50, flow_mag_std: 0.05,
            tproxy_val: 74.56, tproxy_delta: 1.32, tproxy_vel: 0.87,
            CO: 0.9, VOC: 2.5, NO2: 0.03,
            CO_diff: 0.30, VOC_diff: 0.40, NO2_diff: -0.01,
            VOC_ma5: 2.10, CO_ma5: 0.75, NO2_ma5: 0.02,
            VOC_z: 2.2, CO_z: 1.1, NO2_z: -0.2,
            temp_rise_c_per_min: 12.5, temp_slope_30s: 3.2,
            gas_var_30s: 0.45, delta_temp_30s: 8.7, delta_gas_10s: 0.6,
            spike_count_voc_2m: 4,
            temp_co_corr_lag_0s: 0.72, temp_co_corr_lag_15s: 0.68, temp_co_corr_lag_60s: 0.55,
            temp_voc_corr_lag_0s: 0.81, temp_voc_corr_lag_15s: 0.77, temp_voc_corr_lag_60s: 0.60,
            temp_co_xcorr_max_abs: 0.74, temp_voc_xcorr_max_abs: 0.83,
            is_weekend: 0, asleep_window: 1,
            hrblk_0: 0, hrblk_1: 0, hrblk_2: 0, hrblk_3: 0, hrblk_4: 1, hrblk_5: 0
          },
          decision_threshold: 0.4
        };
        break;
    }

    try {
      // Send to the Saafe model endpoint
      const response = await axios.post(
        '/api/predict/saafe',
        payload,
        { headers: { 'Content-Type': 'application/json' } }
      );

      console.log('Alert sent successfully:', response.data);

      // Update the backend alert state based on the scenario
      let alertStatePayload: any = {
        isActive: alertState !== 'non-fire',
        timestamp: new Date().toISOString()
      };

      switch (alertState) {
        case 'non-fire':
          alertStatePayload = {
            ...alertStatePayload,
            level: 1,
            message: "System operating normally",
            riskScore: 10,
            confidence: 0.9
          };
          break;
        case 'fire-predicted':
          alertStatePayload = {
            ...alertStatePayload,
            level: 5,
            message: "Fire predicted - elevated risk detected",
            riskScore: 55,
            confidence: 0.75
          };
          break;
        case 'fire':
          alertStatePayload = {
            ...alertStatePayload,
            level: 9,
            message: "FIRE DETECTED - immediate action required",
            riskScore: 95,
            confidence: 0.95
          };
          break;
      }

      // Update alert state on backend
      await axios.post(
        '/api/alert-state',
        alertStatePayload,
        { headers: { 'Content-Type': 'application/json' } }
      );

      console.log('Alert state updated on backend:', alertStatePayload);
      
      setLastResponse({
        status: 'success',
        message: `${alertState.toUpperCase()} alert sent successfully`,
        alert_state: alertState,
        timestamp: new Date().toISOString(),
        data: response.data
      });
    } catch (err: any) {
      console.error('Error sending alert:', err);
      setError(err.response?.data?.message || err.message || 'Failed to send alert');
      setLastResponse(null);
    } finally {
      setLoading(false);
    }
  };

  // Get button style based on alert state
  const getButtonStyle = (alertState: AlertState) => {
    const baseStyle = {
      padding: '16px 32px',
      fontSize: '18px',
      fontWeight: 'bold',
      border: 'none',
      borderRadius: '12px',
      cursor: loading ? 'not-allowed' : 'pointer',
      transition: 'all 0.3s ease',
      opacity: loading ? 0.6 : 1,
      minWidth: '200px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
    };

    switch (alertState) {
      case 'non-fire':
        return {
          ...baseStyle,
          background: '#22c55e',
          color: 'white'
        };
      case 'fire-predicted':
        return {
          ...baseStyle,
          background: '#eab308',
          color: 'black'
        };
      case 'fire':
        return {
          ...baseStyle,
          background: '#ef4444',
          color: 'white'
        };
    }
  };

  // Get status badge style
  const getStatusBadgeStyle = (alertState: AlertState) => {
    const baseStyle = {
      padding: '8px 16px',
      borderRadius: '8px',
      fontWeight: 'bold',
      display: 'inline-block',
      marginTop: '12px'
    };

    switch (alertState) {
      case 'non-fire':
        return {
          ...baseStyle,
          background: '#dcfce7',
          color: '#166534'
        };
      case 'fire-predicted':
        return {
          ...baseStyle,
          background: '#fef3c7',
          color: '#854d0e'
        };
      case 'fire':
        return {
          ...baseStyle,
          background: '#fee2e2',
          color: '#991b1b'
        };
    }
  };

  return (
    <div style={{ 
      border: '1px solid #e5e7eb', 
      borderRadius: 16, 
      overflow: 'hidden', 
      background: 'white', 
      marginTop: 12 
    }}>
      {/* Header */}
      <div style={{ 
        padding: 16, 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white'
      }}>
        <h2 style={{ margin: 0, fontSize: 24, fontWeight: 'bold' }}>
          📡 Fire Data Sender
        </h2>
        <p style={{ margin: '8px 0 0 0', fontSize: 14, opacity: 0.9 }}>
          Send fire detection sensor data to the backend API for testing and monitoring
        </p>
      </div>

      {/* Content */}
      <div style={{ padding: 24 }}>
        {/* Instructions */}
        <div style={{ 
          background: '#f8fafc', 
          border: '1px solid #e2e8f0', 
          borderRadius: 12, 
          padding: 16, 
          marginBottom: 24 
        }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: 16, color: '#0f172a' }}>
            📋 Instructions
          </h3>
          <ul style={{ margin: 0, paddingLeft: 20, color: '#475569', fontSize: 14 }}>
            <li>Click any button below to simulate different fire alert scenarios</li>
            <li>Each button sends specific sensor data to the backend API</li>
            <li>The system will process the data and trigger appropriate alerts</li>
            <li>If email alerts are configured, you'll receive notifications for fire events</li>
          </ul>
        </div>

        {/* Alert Buttons */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: 16, 
          marginBottom: 24 
        }}>
          {/* Non-Fire Button */}
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={() => sendAlert('non-fire')}
              disabled={loading}
              style={getButtonStyle('non-fire')}
            >
              ✅ Non-Fire
            </button>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>
              Normal conditions<br />No fire detected
            </div>
          </div>

          {/* Fire Predicted Button */}
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={() => sendAlert('fire-predicted')}
              disabled={loading}
              style={getButtonStyle('fire-predicted')}
            >
              ⚠️ Fire Predicted
            </button>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>
              Early warning signs<br />Elevated risk
            </div>
          </div>

          {/* Fire Button */}
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={() => sendAlert('fire')}
              disabled={loading}
              style={getButtonStyle('fire')}
            >
              🔥 Fire
            </button>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>
              Critical conditions<br />Fire detected
            </div>
          </div>
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div style={{ 
            textAlign: 'center', 
            padding: 16, 
            background: '#f1f5f9', 
            borderRadius: 8,
            marginBottom: 16
          }}>
            <div style={{ fontSize: 16, color: '#475569' }}>
              🔄 Sending alert to backend...
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div style={{ 
            background: '#fee2e2', 
            border: '1px solid #fecaca', 
            borderRadius: 12, 
            padding: 16, 
            marginBottom: 16 
          }}>
            <div style={{ fontSize: 16, fontWeight: 'bold', color: '#991b1b', marginBottom: 8 }}>
              ❌ Error
            </div>
            <div style={{ fontSize: 14, color: '#7f1d1d' }}>
              {error}
            </div>
          </div>
        )}

        {/* Response Display */}
        {lastResponse && (
          <div style={{ 
            background: '#f8fafc', 
            border: '1px solid #e2e8f0', 
            borderRadius: 12, 
            padding: 16 
          }}>
            <div style={{ fontSize: 16, fontWeight: 'bold', color: '#0f172a', marginBottom: 12 }}>
              📊 Last Response
            </div>
            
            <div style={{ marginBottom: 12 }}>
              <span style={getStatusBadgeStyle(lastResponse.alert_state)}>
                {lastResponse.alert_state.toUpperCase().replace('-', ' ')}
              </span>
            </div>

            <div style={{ fontSize: 14, color: '#475569', marginBottom: 8 }}>
              <strong>Status:</strong> {lastResponse.status}
            </div>
            <div style={{ fontSize: 14, color: '#475569', marginBottom: 8 }}>
              <strong>Message:</strong> {lastResponse.message}
            </div>
            <div style={{ fontSize: 14, color: '#475569', marginBottom: 8 }}>
              <strong>Timestamp:</strong> {new Date(lastResponse.timestamp).toLocaleString()}
            </div>

            {lastResponse.data && (
              <details style={{ marginTop: 12 }}>
                <summary style={{ 
                  cursor: 'pointer', 
                  fontSize: 14, 
                  fontWeight: 'bold', 
                  color: '#475569',
                  padding: '8px 0'
                }}>
                  View Full Response Data
                </summary>
                <pre style={{ 
                  background: '#1e293b', 
                  color: '#e2e8f0', 
                  padding: 12, 
                  borderRadius: 8, 
                  overflow: 'auto',
                  fontSize: 12,
                  marginTop: 8
                }}>
                  {JSON.stringify(lastResponse.data, null, 2)}
                </pre>
              </details>
            )}
          </div>
        )}

        {/* API Information */}
        <div style={{ 
          marginTop: 24, 
          padding: 16, 
          background: '#eff6ff', 
          border: '1px solid #bfdbfe', 
          borderRadius: 12 
        }}>
          <div style={{ fontSize: 14, fontWeight: 'bold', color: '#1e40af', marginBottom: 8 }}>
            ℹ️ API Information
          </div>
          <div style={{ fontSize: 13, color: '#1e3a8a' }}>
            <strong>Endpoint:</strong> POST /api/predict/saafe<br />
            <strong>Backend:</strong> Node.js Express Server<br />
            <strong>Model:</strong> Saafe Fire Detection AI
          </div>
        </div>
      </div>
    </div>
  );
}