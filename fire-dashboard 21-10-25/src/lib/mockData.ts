import { Camera } from "@/pages/Index";

export const mockCameras: Camera[] = [
  {
    id: "CAM-001",
    name: "Tokyo Central Alpha",
    location: "Shibuya District, Tokyo, Japan",
    coordinates: [139.6917, 35.6895], // Tokyo
    status: "fire",
    temperature: 187,
    lastUpdated: "2 mins ago"
  },
  {
    id: "CAM-002",
    name: "Berlin East Beta",
    location: "Kreuzberg, Berlin, Germany",
    coordinates: [13.4050, 52.5200], // Berlin
    status: "predicted",
    temperature: 145,
    lastUpdated: "5 mins ago"
  },
  {
    id: "CAM-003",
    name: "Sydney Gateway Gamma",
    location: "Port Botany, Sydney, Australia",
    coordinates: [151.2093, -33.8688], // Sydney
    status: "no-fire",
    temperature: 78,
    lastUpdated: "1 min ago"
  },
  {
    id: "CAM-004",
    name: "Toronto West Delta",
    location: "Industrial Park, Toronto, Canada",
    coordinates: [-79.3832, 43.6532], // Toronto
    status: "no-fire",
    temperature: 82,
    lastUpdated: "3 mins ago"
  },
  {
    id: "CAM-005",
    name: "Dubai Central Epsilon",
    location: "Logistics Hub, Dubai, UAE",
    coordinates: [55.2708, 25.2048], // Dubai
    status: "no-fire",
    temperature: 75,
    lastUpdated: "4 mins ago"
  },
  {
    id: "CAM-006",
    name: "São Paulo Northeast Zeta",
    location: "Research Park, São Paulo, Brazil",
    coordinates: [-46.6333, -23.5505], // São Paulo
    status: "no-fire",
    temperature: 71,
    lastUpdated: "2 mins ago"
  },
  {
    id: "CAM-007",
    name: "Cape Town Southwest Eta",
    location: "Harbor Area, Cape Town, South Africa",
    coordinates: [18.4241, -33.9249], // Cape Town
    status: "predicted",
    temperature: 132,
    lastUpdated: "6 mins ago"
  },
  {
    id: "CAM-008",
    name: "Mumbai Northwest Theta",
    location: "Chemical Zone, Mumbai, India",
    coordinates: [72.8777, 19.0760], // Mumbai
    status: "no-fire",
    temperature: 85,
    lastUpdated: "1 min ago"
  }
];
