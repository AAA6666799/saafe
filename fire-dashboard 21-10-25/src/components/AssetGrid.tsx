import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Camera as CameraIcon } from "lucide-react";
import { Camera } from "@/pages/Index";
import axios from "axios";

import { API_BASE_URL } from "../config/api";

interface AssetGridProps {
  onCameraSelect: (camera: Camera) => void;
}

const AssetGrid = ({ onCameraSelect }: AssetGridProps) => {
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
      let status: Camera["status"] = "no-fire";
      if (alertData.riskScore >= 80) {
        status = "fire";
      } else if (alertData.riskScore >= 40) {
        status = "predicted";
      }

      // Map alert data to camera format
      const mapped: Camera[] = [
        {
          id: "CAM-001",
          name: "Kitchen Camera",
          location: "Kitchen - Main Area",
          coordinates: [-0.373907, 51.476782],
          status: alertData.isActive ? status : "no-fire",
          temperature: Math.round(alertData.riskScore / 3),
          lastUpdated: new Date(alertData.timestamp).toLocaleTimeString(),
        },
      ];

      setCameras(mapped);
    } catch (err) {
      console.error("Backend Fetch Error:", err);
      setError("Failed to fetch data from backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBackendData(); // Initial call
    const interval = setInterval(fetchBackendData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status: Camera["status"]) => {
    const styles = {
      fire: "bg-primary text-primary-foreground shadow-glow-fire",
      predicted: "bg-warning text-warning-foreground shadow-glow-warning",
      "no-fire": "bg-safe text-safe-foreground shadow-glow-safe",
      black: "bg-black text-white",
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
        <h2 className="text-3xl font-bold mb-2">Grid — Asset Manager</h2>
        <p className="text-muted-foreground">
          Comprehensive view of all monitored assets
        </p>
      </div>

      {loading && <p className="text-sm text-muted-foreground">Loading assets...</p>}
      {error && <p className="text-sm text-red-500">{error}</p>}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {cameras.map((camera) => (
          <Card
            key={camera.id}
            className={`cursor-pointer transition-all hover:scale-[1.02] border-2 ${
              camera.status === "fire"
                ? "border-primary shadow-glow-fire"
                : camera.status === "predicted"
                ? "border-warning shadow-glow-warning"
                : camera.status === "black"
                ? "border-black bg-gray-900"
                : "border-safe shadow-glow-safe"
            }`}
            onClick={() => onCameraSelect(camera)}
          >
            <div className="p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <CameraIcon className="h-5 w-5 text-muted-foreground" />
                  <span className="font-semibold text-sm">{camera.id}</span>
                </div>
                {getStatusBadge(camera.status)}
              </div>

              <div>
                <h3 className="font-bold text-lg mb-1">{camera.name}</h3>
                <p className="text-xs text-muted-foreground">{camera.location}</p>
              </div>

              <div className="aspect-video bg-muted rounded-lg relative overflow-hidden">
                <div
                  className={`absolute inset-0 ${
                    camera.status === "fire"
                      ? "bg-gradient-to-br from-primary/30 to-destructive/30"
                      : camera.status === "predicted"
                      ? "bg-gradient-to-br from-warning/30 to-warning/10"
                      : "bg-gradient-to-br from-safe/20 to-safe/5"
                  }`}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <CameraIcon className="h-8 w-8 text-muted-foreground/50" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-muted-foreground">Temperature</p>
                  <p className="font-bold text-lg">{camera.temperature}°C</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Last Update</p>
                  <p className="font-bold text-sm">{camera.lastUpdated}</p>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {!loading && cameras.length === 0 && (
        <div className="text-center py-12">
          <CameraIcon className="h-12 w-12 mx-auto text-muted-foreground/50 mb-4" />
          <p className="text-muted-foreground">No assets found</p>
        </div>
      )}
    </div>
  );
};

export default AssetGrid;
