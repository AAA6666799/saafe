// API functions for Fire Detection System integration

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

// Updated response structure to match the new backend
interface FireDetectionResponse {
  status: string;
  data: FireDetectionData;
  message?: string;
}

// Function to fetch fire detection data from the backend API
export async function fetchFireDetectionData(): Promise<FireDetectionData> {
  try {
    // Connect to the backend API
    const response = await fetch('/api/fire-detection-data');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result: FireDetectionResponse = await response.json();
    
    if (result.status !== "success") {
      throw new Error(result.message || "Failed to fetch fire detection data");
    }
    
    return result.data;
  } catch (error) {
    console.error("Error fetching fire detection data:", error);
    throw new Error("Failed to fetch fire detection data: " + (error as Error).message);
  }
}

// Function to fetch system status
export async function fetchSystemStatus() {
  try {
    const response = await fetch('/api/status');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (result.status !== "success") {
      throw new Error(result.message || "Failed to fetch system status");
    }
    
    return result.data;
  } catch (error) {
    console.error("Error fetching system status:", error);
    throw new Error("Failed to fetch system status: " + (error as Error).message);
  }
}