import { useState, useEffect } from "react";
import { Camera } from "@/pages/Index";

interface ApiResponse {
  frame: number;
  device_name: string;
  device_id: string;
  device_location: [number, number];
  timestamp: string;
  prediction: {
    fire_probability: number;
    label: string;
  };
  explanation: {
    global_top_features: Array<{
      feature: string;
      importance: number;
    }>;
    local_contributions: Array<{
      feature: string;
      value: number;
      contribution: number;
    }>;
    notes: string;
  };
}

const API_URL = "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/";
const POLL_INTERVAL = 5000; // Poll every 5 seconds

const mapApiResponseToCamera = (data: ApiResponse): Camera => {
  // Determine status based on prediction label
  let status: "fire" | "no-fire" | "predicted" = "no-fire";
  if (data.prediction.label === "Fire") {
    status = "fire";
  } else if (data.prediction.fire_probability > 0.0001) {
    status = "predicted";
  }

  // Calculate temperature estimate based on fire probability (mock calculation)
  const temperature = Math.round(70 + (data.prediction.fire_probability * 10000));

  return {
    id: `CAM-${data.device_id}`,
    name: data.device_name,
    location: `Device ${data.device_id}`,
    coordinates: [data.device_location[1], data.device_location[0]], // [lon, lat] for mapbox
    status,
    temperature,
    lastUpdated: new Date(data.timestamp).toLocaleTimeString(),
  };
};

export const useRealtimeFireData = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(API_URL);
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        const data: ApiResponse = await response.json();
        const camera = mapApiResponseToCamera(data);
        
        // Update or add camera to list
        setCameras(prev => {
          const existing = prev.findIndex(c => c.id === camera.id);
          if (existing >= 0) {
            const updated = [...prev];
            updated[existing] = camera;
            return updated;
          }
          return [...prev, camera];
        });
        
        setError(null);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching fire data:", err);
        setError(err instanceof Error ? err.message : "Failed to fetch data");
        setLoading(false);
      }
    };

    // Initial fetch
    fetchData();

    // Set up polling
    const interval = setInterval(fetchData, POLL_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  return { cameras, loading, error };
};
