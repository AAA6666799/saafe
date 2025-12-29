import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Flame } from "lucide-react";
import axios from "axios";

import { API_BASE_URL } from "../config/api";

export interface Camera {
  id: string;
  name: string;
  location: string;
  coordinates: [number, number];
  status: "fire" | "predicted" | "no-fire" | "black";
  temperature: number;
  lastUpdated: string;
}

interface FireDetectionProps {
  onCameraSelect: (camera: Camera) => void;
}

const FireDetection = ({ onCameraSelect }: FireDetectionProps) => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchBackendData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/api/alert-state`);
      const alertData = (response.data as any).data;

      // Determine status based on risk score
      let finalStatus: Camera["status"] = "no-fire";
      if (alertData.riskScore >= 80) {
        finalStatus = "fire";
      } else if (alertData.riskScore >= 40) {
        finalStatus = "predicted";
      }

      // Build camera data
      const mappedCamera: Camera = {
        id: "CAM-001",
        name: "Kitchen Camera",
        location: "Kyle Park",
        coordinates: [51.476782, -0.373907],
        status: alertData.isActive ? finalStatus : "no-fire",
        temperature: Math.round(alertData.riskScore / 3),
        lastUpdated: new Date(alertData.timestamp).toLocaleTimeString(),
      };

      // Only show cameras if fire or predicted
      setCameras(
        finalStatus === "fire" || finalStatus === "predicted" ? [mappedCamera] : []
      );
    } catch (err) {
      console.error("❌ Error fetching backend data:", err);
      setError("Failed to connect to backend service.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBackendData(); // Initial call
    const interval = setInterval(fetchBackendData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  // Status badge styling
  const getStatusBadge = (status: Camera["status"]) => {
    const styles = {
      fire: "bg-primary text-primary-foreground shadow-glow-fire",
      predicted: "bg-warning text-warning-foreground shadow-glow-warning",
      "no-fire": "bg-safe text-safe-foreground shadow-glow-safe",
      black: "bg-black text-white shadow-glow-black",
    };

    return (
      <Badge className={styles[status]}>
        {status === "no-fire"
          ? "Safe"
          : status === "black"
          ? "Offline"
          : status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  return (
    <div className="h-full p-6">
      <div className="mb-6">
        <h2 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <Flame className="h-8 w-8 text-primary" />
          Fire Prediction System
        </h2>
        <p className="text-muted-foreground">
          Real-time fire prediction based on backend analysis
        </p>
      </div>

      {loading && <p>Loading camera data…</p>}
      {error && <p className="text-red-500">{error}</p>}

      {cameras.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {cameras.map((camera) => (
            <Card
              key={camera.id}
              className={`border-2 cursor-pointer transition-all hover:scale-[1.02] ${
                camera.status === "fire"
                  ? "border-primary shadow-glow-fire"
                  : camera.status === "predicted"
                  ? "border-warning shadow-glow-warning"
                  : camera.status === "black"
                  ? "border-black bg-gray-900 text-white"
                  : "border-safe shadow-glow-safe"
              }`}
              onClick={() => onCameraSelect(camera)}
            >
              <div className="p-6 space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-xl font-bold mb-1">{camera.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {camera.location}
                    </p>
                  </div>
                  {getStatusBadge(camera.status)}
                </div>

                <div className="aspect-video bg-muted rounded-lg relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-destructive/20" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center space-y-2">
                      <Flame className="h-12 w-12 mx-auto text-primary animate-pulse" />
                      <p className="text-sm font-medium">Thermal Camera Feed</p>
                      <p className="text-xs text-muted-foreground">
                        Connect to AWS for live thermal imagery
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Temperature</p>
                    <p className="text-2xl font-bold text-primary">
                      {camera.temperature}°C
                    </p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Status</p>
                    <p className="text-2xl font-bold text-secondary">
                      {camera.status === "fire"
                        ? "🔥 94%"
                        : camera.status === "predicted"
                        ? "⚠️ 67%"
                        : camera.status === "black"
                        ? "Offline"
                        : "—"}
                    </p>
                  </div>
                </div>

                <div className="text-xs text-muted-foreground">
                  Last updated: {camera.lastUpdated}
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        !loading && (
          <Card className="border-border">
            <div className="p-12 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-safe/10 mb-4">
                <Flame className="h-8 w-8 text-safe" />
              </div>
              <h3 className="text-xl font-semibold mb-2">All Clear</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                No active fires or predicted threats detected. System is
                monitoring all cameras.
              </p>
            </div>
          </Card>
        )
      )}
    </div>
  );
};

export default FireDetection;
