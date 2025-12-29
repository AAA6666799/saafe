import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { Button } from "@/components/ui/button";
import { Play, Pause } from "lucide-react";
import { Camera } from "@/pages/Index";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import axios from "axios";

mapboxgl.accessToken =
  "pk.eyJ1IjoiaGFybGVxdWluaXJlIiwiYSI6ImNtZGdpMWRrejBsYzcybHB6eXp0Z25pYzcifQ.Mwu0v3MGK-eo6mEsbYjVng";

interface HeliosMapProps {
  onCameraSelect: (camera: Camera) => void;
}

import { API_BASE_URL } from "../config/api";

export default function HeliosMap({ onCameraSelect }: HeliosMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [isSpinning, setIsSpinning] = useState(false);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const animationRef = useRef<number | null>(null);
  const [activeAlert, setActiveAlert] = useState<{
    deviceId: string;
    status: string;
    riskScore: number;
  } | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>("—");

  // Fetch alert state from backend
  const fetchAlertState = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/alert-state`);
      if (response.data.status === 'success') {
        const alertData = response.data.data;
        
        // Determine status based on risk score
        let status: Camera["status"] = "no-fire";
        if (alertData.riskScore >= 80) {
          status = "fire";
        } else if (alertData.riskScore >= 40) {
          status = "predicted";
        }

        // Create camera data from alert state
        const camera: Camera = {
          id: "CAM-001",
          name: "Kitchen Camera",
          location: "Kyle Park Estate",
          coordinates: [-6.8649841, 52.7808874], // [lng, lat] for Mapbox - Kyle Park Estate, Ireland
          status: alertData.isActive ? status : "no-fire",
          temperature: Math.round(alertData.riskScore / 3), // Approximate temp from risk
          lastUpdated: new Date(alertData.timestamp).toLocaleTimeString()
        };

        setCameras([camera]);
        setLastUpdate(new Date(alertData.timestamp).toLocaleTimeString());

        // Show alert if fire or predicted
        if (alertData.isActive && (status === "fire" || status === "predicted")) {
          setActiveAlert({
            deviceId: "SAAFE-KITCHEN-001",
            status: status,
            riskScore: alertData.riskScore
          });
          
          // Auto hide after 10 seconds
          setTimeout(() => setActiveAlert(null), 10000);
        } else {
          setActiveAlert(null);
        }
      }
    } catch (err) {
      console.error("❌ Error fetching alert state:", err);
    }
  };

  useEffect(() => {
    fetchAlertState(); // Initial fetch
    const interval = setInterval(fetchAlertState, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!map.current && mapContainer.current) {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: "mapbox://styles/mapbox/satellite-v9",
        center: [0, 20],
        zoom: 1.5,
        projection: "globe",
      });
      map.current.addControl(new mapboxgl.NavigationControl(), "top-right");
      
      // Wait for style to load before setting up clustering
      map.current.on('load', () => {
        if (cameras.length) setupClustering();
      });
    } else if (map.current && cameras.length && map.current.isStyleLoaded()) {
      setupClustering();
    }
  }, [cameras]);

  const statusToHsl = (status: string) => {
    switch (status) {
      case "fire":
        return "hsl(0, 80%, 50%)";
      case "predicted":
        return "hsl(45, 95%, 55%)";
      case "no-fire":
        return "hsl(120, 80%, 45%)";
      case "black":
        return "hsl(0, 0%, 20%)";
      default:
        return "hsl(120, 80%, 45%)";
    }
  };

  const setupClustering = () => {
    if (!map.current) return;
    const geojson: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: cameras.map((c) => ({
        type: "Feature",
        properties: {
          id: c.id,
          name: c.name,
          location: c.location,
          status: c.status,
          temperature: c.temperature,
          lastUpdated: c.lastUpdated,
        },
        geometry: { type: "Point", coordinates: [c.coordinates[0], c.coordinates[1]] },
      })),
    };

    const src = map.current.getSource("cameras") as mapboxgl.GeoJSONSource;
    if (src) {
      src.setData(geojson);
    } else {
      map.current.addSource("cameras", { 
        type: "geojson", 
        data: geojson, 
        cluster: true, 
        clusterMaxZoom: 14, 
        clusterRadius: 50 
      });
      
      map.current.addLayer({
        id: "clusters",
        type: "circle",
        source: "cameras",
        filter: ["has", "point_count"],
        paint: {
          "circle-color": "#11b4da",
          "circle-radius": 20,
        },
      });

      map.current.addLayer({
        id: "cluster-count",
        type: "symbol",
        source: "cameras",
        filter: ["has", "point_count"],
        layout: { 
          "text-field": "{point_count_abbreviated}", 
          "text-font": ["DIN Offc Pro Medium", "Arial Unicode MS Bold"], 
          "text-size": 14 
        },
        paint: { "text-color": "#fff" },
      });

      map.current.addLayer({
        id: "unclustered-point",
        type: "circle",
        source: "cameras",
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-color": [
            "match",
            ["get", "status"],
            "fire", statusToHsl("fire"),
            "predicted", statusToHsl("predicted"),
            "no-fire", statusToHsl("no-fire"),
            "black", statusToHsl("black"),
            statusToHsl("no-fire")
          ],
          "circle-radius": 10,
          "circle-stroke-width": 2,
          "circle-stroke-color": "#fff",
        },
      });
    }

    const clickPin = (e: mapboxgl.MapLayerMouseEvent) => {
      const feat = e.features![0];
      const cam = cameras.find((c) => c.id === feat.properties!.id);
      if (!cam) return;
      onCameraSelect(cam);
      map.current!.flyTo({ center: cam.coordinates as [number, number], zoom: 12 });
    };
    map.current.off("click", "unclustered-point", clickPin);
    map.current.on("click", "unclustered-point", clickPin);

    const handleClusterClick = (e: mapboxgl.MapLayerMouseEvent) => {
      const feats = map.current!.queryRenderedFeatures(e.point, { layers: ["clusters"] });
      const clustId = feats[0].properties!.cluster_id;
      (map.current!.getSource("cameras") as mapboxgl.GeoJSONSource).getClusterExpansionZoom(clustId, (err, zoom) => {
        if (err || zoom == null) return;
        map.current!.easeTo({ center: (feats[0].geometry as GeoJSON.Point).coordinates as [number, number], zoom });
      });
    };
    map.current.off("click", "clusters", handleClusterClick);
    map.current.on("click", "clusters", handleClusterClick);
  };

  const startSpinning = () => {
    if (!map.current || !isSpinning) return;
    const spin = () => {
      if (!map.current || !isSpinning) return;
      const center = map.current.getCenter();
      center.lng += 0.1;
      map.current.setCenter(center);
      animationRef.current = requestAnimationFrame(spin);
    };
    animationRef.current = requestAnimationFrame(spin);
  };

  const stopSpinning = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
  };

  useEffect(() => {
    isSpinning ? startSpinning() : stopSpinning();
  }, [isSpinning]);

  return (
    <div className="h-full relative flex">
      <div ref={mapContainer} className="flex-1" />
      <div className="absolute left-4 top-4 z-20">
        <Button onClick={() => setIsSpinning((s) => !s)} className="gap-2">
          {isSpinning ? (
            <>
              <Pause className="h-4 w-4" /> Stop Spin
            </>
          ) : (
            <>
              <Play className="h-4 w-4" /> Start Spin
            </>
          )}
        </Button>
        <div className="mt-2 text-xs text-muted-foreground bg-background/80 p-2 rounded">
          Last update: {lastUpdate}
        </div>
      </div>

      {activeAlert && (
        <div className="absolute left-4 bottom-10 z-20 w-80">
          <Alert variant={activeAlert.status === "fire" ? "destructive" : "default"}>
            <AlertTitle>
              {activeAlert.status === "fire" ? "🔥 FIRE DETECTED" : "⚠️ WARNING"}
            </AlertTitle>
            <AlertDescription>
              Device <b>{activeAlert.deviceId}</b> reported status:{" "}
              <b>{activeAlert.status.toUpperCase()}</b>
              <br />
              Risk Score: <b>{activeAlert.riskScore}/100</b>
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
}