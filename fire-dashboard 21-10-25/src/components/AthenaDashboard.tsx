import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, AlertTriangle, CheckCircle, Camera, TrendingUp } from "lucide-react";
import axios from "axios";

import { API_BASE_URL } from "../config/api";

export interface CameraType {
  id: string;
  name: string;
  location: string;
  coordinates: [number, number];
  status: "fire" | "predicted" | "no-fire" | "black";
  temperature: number;
  lastUpdated: string;
}

const AthenaDashboard = () => {
  const [cameras, setCameras] = useState<CameraType[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFromBackend = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/api/alert-state`);
      const alertData = (response.data as any).data;

      // Determine status based on risk score
      let status: CameraType["status"] = "no-fire";
      if (alertData.riskScore >= 80) {
        status = "fire";
      } else if (alertData.riskScore >= 40) {
        status = "predicted";
      }

      // Map alert data to camera format
      const mapped: CameraType[] = [
        {
          id: "CAM-001",
          name: "Kitchen Camera",
          location: "Kitchen - Main Area",
          coordinates: [51.476782, -0.373907],
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
    fetchFromBackend(); // Initial call
    const interval = setInterval(fetchFromBackend, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  // Dashboard stats
  const fireCount = cameras.filter((c) => c.status === "fire").length;
  const predictedCount = cameras.filter((c) => c.status === "predicted").length;
  const safeCount = cameras.filter((c) => c.status === "no-fire").length;
  const blackCount = cameras.filter((c) => c.status === "black").length;
  const avgTemp =
    cameras.length > 0
      ? Math.round(
          cameras.reduce((acc, c) => acc + c.temperature, 0) / cameras.length
        )
      : 0;

  const stats = [
    {
      title: "Active Fires",
      value: fireCount,
      icon: AlertTriangle,
      trend: fireCount > 0 ? "Immediate action required" : "No fires detected",
      color: "text-primary",
      bgColor: "bg-primary/10",
      glowColor: "shadow-glow-fire",
    },
    {
      title: "Predicted Threats",
      value: predictedCount,
      icon: TrendingUp,
      trend: predictedCount > 0 ? "Warning level" : "No predictions",
      color: "text-warning",
      bgColor: "bg-warning/10",
      glowColor: "shadow-glow-warning",
    },
    {
      title: "Safe Zones",
      value: safeCount,
      icon: CheckCircle,
      trend: "All clear",
      color: "text-safe",
      bgColor: "bg-safe/10",
      glowColor: "shadow-glow-safe",
    },
    {
      title: "Offline Devices",
      value: blackCount,
      icon: Camera,
      trend: "Disconnected",
      color: "text-muted-foreground",
      bgColor: "bg-muted/10",
      glowColor: "shadow-none",
    },
    {
      title: "Avg Temperature",
      value: `${avgTemp}°C`,
      icon: Activity,
      trend: "Normal range",
      color: "text-secondary",
      bgColor: "bg-secondary/10",
      glowColor: "shadow-glow-safe",
    },
  ];

  return (
    <div className="h-full p-6">
      <div className="mb-6">
        <h2 className="text-3xl font-bold mb-2">Athena — Strategic Dashboard</h2>
        <p className="text-muted-foreground">
          System overview and key performance metrics
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        {stats.map((stat) => (
          <Card key={stat.title} className={`border-border ${stat.glowColor}`}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {stat.title}
              </CardTitle>
              <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`h-4 w-4 ${stat.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className={`text-3xl font-bold ${stat.color}`}>{stat.value}</div>
              <p className="text-xs text-muted-foreground mt-1">{stat.trend}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* System Status + Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-5 w-5 text-primary" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Total Cameras</span>
                <span className="font-bold">{cameras.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Online</span>
                <span className="font-bold text-safe">
                  {cameras.filter((c) => c.status !== "black").length}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Offline</span>
                <span className="font-bold text-muted-foreground">
                  {blackCount}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Prediction Accuracy</span>
                <span className="font-bold text-safe">98.7%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-secondary" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {cameras.slice(0, 5).map((camera) => (
                <div
                  key={camera.id}
                  className="flex items-start gap-3 p-2 rounded-lg bg-muted/50"
                >
                  <div
                    className={`h-2 w-2 rounded-full mt-1.5 ${
                      camera.status === "fire"
                        ? "bg-primary shadow-glow-fire"
                        : camera.status === "predicted"
                        ? "bg-warning shadow-glow-warning"
                        : camera.status === "black"
                        ? "bg-black"
                        : "bg-safe shadow-glow-safe"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{camera.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {camera.lastUpdated}
                    </p>
                  </div>
                  <span className="text-xs font-medium">
                    {camera.temperature}°C
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {loading && (
        <p className="mt-4 text-sm text-muted-foreground">Refreshing data...</p>
      )}
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </div>
  );
};

export default AthenaDashboard;
