import React, { useMemo, useState, useEffect, useRef } from "react";
import homesData from "../data/sampleHomes.json"
import { askSaafeGPT } from "../api/saafegpt"
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import axios from "axios";

// Alert state interface
interface AlertState {
  isActive: boolean;
  level: number;
  message: string;
  timestamp: string;
  riskScore: number;
  confidence: number;
}

// IMPORTANT: Add your Mapbox access token here
// 1. Go to: https://account.mapbox.com/access-tokens/
// 2. Create a free account and get a public token
// 3. Replace the token below with your real token
mapboxgl.accessToken = 'pk.eyJ1IjoiaGFybGVxdWluaXJlIiwiYSI6ImNtZGdpMWRrejBsYzcybHB6eXp0Z25pYzcifQ.Mwu0v3MGK-eo6mEsbYjVng'; // Replace with your real Mapbox token

// Utility
const band = (level:number) => level>=9?"critical":level>=7?"high":level>=5?"elev":level>=3?"guard":"low";
const bandColor = (level:number) => {
  const colors = { low:"#34d399", guard:"#a3e635", elev:"#f59e0b", high:"#f97316", critical:"#ef4444" };
  const key = band(level) as keyof typeof colors;
  return colors[key];
};

// Geocoding function using Mapbox
async function geocodeAddress(address: string): Promise<{ lat: number; lon: number } | null> {
  try {
    const response = await fetch(
      `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(address)}.json?access_token=${mapboxgl.accessToken}`
    );
    const data = await response.json();

    if (data.features && data.features.length > 0) {
      const [lon, lat] = data.features[0].center;
      return { lat, lon };
    }
    return null;
  } catch (error) {
    console.error('Geocoding error:', error);
    return null;
  }
}

// Helios 2D
function Helios2D({ homes, onAddLocation, onDeleteLocation }:{ homes:any[]; onAddLocation:(location:any)=>void; onDeleteLocation:(id:string)=>void }){
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const markers = useRef<mapboxgl.Marker[]>([]);
  const [lng] = useState(-70.9);
  const [lat] = useState(42.35);
  const [zoom] = useState(1.5);

  // Form state for adding new locations
  const [newAddress, setNewAddress] = useState('');
  const [newName, setNewName] = useState('');
  const [isAdding, setIsAdding] = useState(false);

  // Globe spinning animation
  const animationRef = useRef<number | null>(null);
  const [isSpinning, setIsSpinning] = useState(false);

  // Globe spinning functions
  const startSpinning = () => {
    if (!map.current || !isSpinning) return;

    const spinGlobe = () => {
      if (!map.current || !isSpinning) return;

      const center = map.current.getCenter();
      center.lng += 0.1; // Slow rotation speed
      map.current.setCenter(center);
      animationRef.current = requestAnimationFrame(spinGlobe);
    };

    animationRef.current = requestAnimationFrame(spinGlobe);
  };

  const stopSpinning = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  // Toggle spinning
  const toggleSpinning = () => {
    setIsSpinning(!isSpinning);
  };
  const [mapError, setMapError] = useState<string | null>(null);

  // Function to add new location
  const addNewLocation = async () => {
    if (!newAddress.trim() || !newName.trim()) return;

    setIsAdding(true);
    try {
      const coords = await geocodeAddress(newAddress);
      if (coords) {
        // Generate unique ID by finding the highest existing SAAFE ID and incrementing
        const existingIds = homes
          .map(h => h.id)
          .filter(id => id.startsWith('SAAFE-'))
          .map(id => parseInt(id.replace('SAAFE-', '')))
          .filter(num => !isNaN(num));

        const nextId = existingIds.length > 0 ? Math.max(...existingIds) + 1 : homes.length + 1;

        const newLocation = {
          id: `SAAFE-${String(nextId).padStart(4, '0')}`,
          name: newName,
          city: newAddress.split(',')[0]?.trim() || 'Unknown',
          country: 'XX', // Will be updated if geocoding provides country
          lat: coords.lat,
          lon: coords.lon,
          score: Math.floor(Math.random() * 30) + 10, // Random initial score
          level: Math.floor(Math.random() * 3) + 1, // Random initial level 1-3
          updated: new Date().toLocaleTimeString(),
          status: 'online',
          firmware: '1.0.3',
          battery: Math.random() * 0.5 + 0.5, // Random battery 0.5-1.0
          paymentStatus: Math.random() < 0.6 ? 'paid' : Math.random() < 0.8 ? 'pending' : 'overdue' // 60% paid, 20% pending, 20% overdue
        };

        onAddLocation(newLocation);
        setNewAddress('');
        setNewName('');
      } else {
        alert('Could not find coordinates for this address. Please try a more specific address.');
      }
    } catch (error) {
      console.error('Error adding location:', error);
      alert('Error adding location. Please try again.');
    } finally {
      setIsAdding(false);
    }
  };

  useEffect(() => {
    if (map.current) return; // initialize map only once

    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current!,
        style: 'mapbox://styles/mapbox/satellite-v9',
        center: [lng, lat],
        zoom: zoom,
        projection: 'globe'
      });

      // Add navigation controls
      map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

      // Handle map load errors
      map.current.on('error', (e) => {
        console.error('Map failed to load:', e);
        setMapError('Failed to load map. Please check your Mapbox access token.');
      });

      // Set up clustering when map loads
      map.current.on('load', () => {
        setMapError(null);
        setupClustering();
  
        // Start globe spinning animation
        startSpinning();
  
        // Add global functions for popup buttons
        (window as any).deleteLocation = (locationId: string) => {
          onDeleteLocation(locationId);
          // Close current popup
          if ((window as any).currentPopup) {
            (window as any).currentPopup.remove();
          }
        };

        (window as any).updatePaymentStatus = (locationId: string, newStatus: string) => {
          // Update the homes state with new payment status
          const updatedHomes = homes.map(h =>
            h.id === locationId ? { ...h, paymentStatus: newStatus } : h
          );
          // Trigger re-render by calling onAddLocation with updated data
          // Since we can't directly access setHomes, we'll use a workaround
          onDeleteLocation('temp'); // This will trigger a re-render
          setTimeout(() => onAddLocation(updatedHomes.find(h => h.id === locationId)!), 0);
          // Close current popup
          if ((window as any).currentPopup) {
            (window as any).currentPopup.remove();
          }
        };
      });
    } catch (error) {
      console.error('Failed to initialize map:', error);
      setMapError('Failed to initialize map. Please check your Mapbox access token.');
    }
  }, [lng, lat, zoom]);

  // Function to set up clustering layers
  const setupClustering = () => {
    if (!map.current) return;

    // Convert homes data to GeoJSON format for clustering
    const geojson: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: homes.map((h: any) => ({
    type: 'Feature',
    properties: {
      id: h.id,
      name: h.name,
      city: h.city,
      country: h.country,
      level: h.level,
      score: h.score,
      predictions: h.predictions || 'no-fire'   // 👈 changed
    },
    geometry: { type: 'Point', coordinates: [h.lon, h.lat] }
  }))
};

    // Add GeoJSON source with clustering enabled
    if (map.current.getSource('homes')) {
      (map.current.getSource('homes') as mapboxgl.GeoJSONSource).setData(geojson);
    } else {
      map.current.addSource('homes', {
        type: 'geojson',
        data: geojson,
        cluster: true,
        clusterMaxZoom: 14,
        clusterRadius: 50
      });
    }

    // Add cluster layers
    if (!map.current.getLayer('clusters')) {
      map.current.addLayer({
        id: 'clusters',
        type: 'circle',
        source: 'homes',
        filter: ['has', 'point_count'],
        paint: {
          'circle-color': [
                'match',
                ['get', 'predictions'], // 👈 use predictions for color
                'no-fire', '#22c55e',    // 🟢 green
                'predicted', '#eab308',  // 🟡 amber
                'fire', '#ef4444',       // 🔴 red
                '#22c55e'                // default green
              ],
          'circle-radius': 8,
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff'
        }

      });
    }

    if (!map.current.getLayer('cluster-count')) {
      map.current.addLayer({
        id: 'cluster-count',
        type: 'symbol',
        source: 'homes',
        filter: ['has', 'point_count'],
        layout: {
          'text-field': '{point_count_abbreviated}',
          'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
          'text-size': 12
        }
      });
    }

    // Add unclustered point layer
    if (!map.current.getLayer('unclustered-point')) {
      map.current.addLayer({
        id: 'unclustered-point',
        type: 'circle',
        source: 'homes',
        filter: ['!', ['has', 'point_count']],
        paint: {
          'circle-color': [
            'match',
            ['get', 'predictions'], // 👈 changed to predictions
            'no-fire', '#22c55e',   // green
            'predicted', '#eab308', // yellow/amber
            'fire', '#ef4444',      // red
            '#22c55e'               // default
          ],
          'circle-radius': 8,
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff'
        }
      });
    }

    // Add click handler for clusters
    map.current.on('click', 'clusters', (e) => {
      const features = map.current!.queryRenderedFeatures(e.point, {
        layers: ['clusters']
      });
      const clusterId = features[0].properties!.cluster_id;
      (map.current!.getSource('homes') as mapboxgl.GeoJSONSource).getClusterExpansionZoom(
        clusterId,
        (err, zoom) => {
          if (err || zoom === null) return;

          map.current!.easeTo({
            center: (features[0].geometry as GeoJSON.Point).coordinates as [number, number],
            zoom: zoom
          });
        }
      );
    });

    // Add click handler for unclustered points
    map.current.on('click', 'unclustered-point', (e) => {
      const coordinates = (e.features![0].geometry as GeoJSON.Point).coordinates.slice();
      const properties = e.features![0].properties;

      // Ensure that if the map is zoomed out such that multiple
      // copies of the feature are visible, the popup appears
      // over the copy being pointed to.
      while (Math.abs(e.lngLat.lng - coordinates[0]) > 180) {
        coordinates[0] += e.lngLat.lng > coordinates[0] ? 360 : -360;
      }

      const isUserAdded = properties!.isUserAdded;

      const prediction = properties!.predictions || 'no-fire'; // current prediction

// Map prediction to color and label
const predictionMap: Record<string, { label: string; color: string; textColor: string }> = {
  'no-fire': { label: 'No-Fire', color: '#22c55e', textColor: 'white' },
  'predicted': { label: 'Predicted', color: '#eab308', textColor: 'black' },
  'fire': { label: 'FIRE', color: '#ef4444', textColor: 'white' }
};

// Get only the current prediction button
const current = predictionMap[prediction];
      const popupContent = `
  <div style="min-width: 200px;">
    <strong>${properties!.name}</strong><br/>
    ${properties!.city}, ${properties!.country}<br/>
    Level: L${properties!.level} | Score: ${properties!.score}
    <br/>
    <div style="margin-top: 8px; display: flex; gap: 4px; flex-wrap: wrap;">
      <button onclick="window.updatePrediction('${properties!.id}', '${prediction}')" 
              style="padding:4px 6px;
                     background:${current.color};
                     color:${current.textColor};
                     border:none;
                     border-radius:3px;
                     cursor:pointer;
                     font-size:11px;">
        ${current.label}
      </button>
    </div>
  </div>
`;

// Global handler to change predictions
(window as any).updatePrediction = (locationId: string, newPrediction: string) => {
  const updatedHomes = homes.map(h =>
    h.id === locationId ? { ...h, predictions: newPrediction } : h
  );
  onDeleteLocation('temp');
  setTimeout(() => onAddLocation(updatedHomes.find(h => h.id === locationId)!), 0);
  if ((window as any).currentPopup) (window as any).currentPopup.remove();
};

      const popup = new mapboxgl.Popup()
        .setLngLat(coordinates as [number, number])
        .setHTML(popupContent)
        .addTo(map.current!);

      // Store popup reference for cleanup
      (window as any).currentPopup = popup;
    });

    // Change cursor on hover
    map.current.on('mouseenter', 'clusters', () => {
      map.current!.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'clusters', () => {
      map.current!.getCanvas().style.cursor = '';
    });

    map.current.on('mouseenter', 'unclustered-point', () => {
      map.current!.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'unclustered-point', () => {
      map.current!.getCanvas().style.cursor = '';
    });
  };

  // Update clustering data when homes change
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;
    setupClustering();
  }, [homes]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      stopSpinning();
    };
  }, []);

  // Update spinning when isSpinning changes
  useEffect(() => {
    if (isSpinning) {
      startSpinning();
    } else {
      stopSpinning();
    }
  }, [isSpinning]);

  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"#0f172a" }}>
      <div style={{ padding:12, color:"#0f172a", background:"white", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <strong>Helios — Global View</strong>
        <div style={{ display:"flex", gap:8, alignItems:"center" }}>
          <button
            onClick={toggleSpinning}
            style={{
              ...btnOutline,
              fontSize: "12px",
              padding: "4px 8px",
              background: isSpinning ? "#059669" : "white",
              color: isSpinning ? "white" : "#0f172a",
              border: isSpinning ? "1px solid #059669" : "1px solid #e5e7eb"
            }}
            title={isSpinning ? "Stop globe spinning" : "Start globe spinning"}
          >
            {isSpinning ? "⏸️ Stop Spin" : "▶️ Start Spin"}
          </button>
          <input
            type="text"
            placeholder="Location name"
            value={newName}
            onChange={e => setNewName(e.target.value)}
            style={{ ...inputStyle, width: 120 }}
          />
          <input
            type="text"
            placeholder="Address (city, country)"
            value={newAddress}
            onChange={e => setNewAddress(e.target.value)}
            style={{ ...inputStyle, width: 180 }}
          />
          <button
            onClick={addNewLocation}
            disabled={isAdding || !newAddress.trim() || !newName.trim()}
            style={{
              ...btn,
              opacity: (isAdding || !newAddress.trim() || !newName.trim()) ? 0.5 : 1,
              cursor: (isAdding || !newAddress.trim() || !newName.trim()) ? 'not-allowed' : 'pointer'
            }}
          >
            {isAdding ? 'Adding...' : 'Add Location'}
          </button>
        </div>
      </div>
      {mapError ? (
        <div style={{
          width: 900,
          height: 450,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#1e293b',
          color: 'white',
          padding: 20,
          textAlign: 'center'
        }}>
          <h3 style={{ margin: '0 0 16px 0', color: '#ef4444' }}>Map Unavailable</h3>
          <p style={{ margin: '0 0 16px 0', maxWidth: 400 }}>
            {mapError}
          </p>
          <div style={{ fontSize: 14, color: '#94a3b8' }}>
            <p style={{ margin: '8px 0' }}>
              To fix this:
            </p>
            <ol style={{ textAlign: 'left', display: 'inline-block' }}>
              <li>Go to <a href="https://account.mapbox.com/access-tokens/" target="_blank" style={{ color: '#3b82f6' }}>Mapbox Account</a></li>
              <li>Create a free account</li>
              <li>Get a public access token</li>
              <li>Replace 'YOUR_MAPBOX_TOKEN_HERE' in the code with your token</li>
            </ol>
          </div>
        </div>
      ) : (
        <div style={{ position: 'relative', width: 900, height: 450 }}>
          <div ref={mapContainer} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} />
        </div>
      )}
      {!mapError && (
            <div style={{ position: "absolute", left: 20, bottom: 100, display: "flex", gap: 10,
                          padding: 8, background: "rgba(255,255,255,0.95)", borderRadius: 12,
                          fontSize: 12, zIndex: 1, pointerEvents: "none" }}>
              {[["#22c55e","No-Fire"],["#eab308","Predicted"],["#ef4444","Fire"]]
                .map(([c,l])=>(
                <div key={l} style={{ display:"flex", alignItems:"center", gap:6, pointerEvents:"auto" }}>
                  <span style={{ width:10, height:10, background:c, borderRadius:9999 }} />
                  <span>{l}</span>
                </div>
              ))}
            </div>
          )}
              </div>
            )
          }

// Grid
function Grid({ rows, onPick }:{ rows:any[]; onPick:(r:any)=>void }){
  const [q,setQ] = useState(""); const [rf,setRf] = useState("all");
  const filtered = useMemo(()=> rows.filter((h:any)=>{
    const matches = `${h.id} ${h.name} ${h.city} ${h.country}`.toLowerCase().includes(q.toLowerCase());
    const r = rf==="all"?true: rf==="low"?h.level<3: rf==="guard"?h.level>=3&&h.level<5: rf==="elev"?h.level>=5&&h.level<7: rf==="high"?h.level>=7&&h.level<9: h.level>=9;
    return matches && r;
  }),[rows,q,rf]);
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white" }}>
      <div style={{ padding:12, display:"flex", gap:12, alignItems:"center", justifyContent:"space-between", flexWrap:"wrap" }}>
        <strong style={{ color:"#0f172a" }}>Grid — Asset Manager </strong>
        <div style={{ display:"flex", gap:8 }}>
          <input value={q} onChange={e=> setQ(e.target.value)} placeholder="Search id / name / city" style={inputStyle}/>
          <select value={rf} onChange={e=> setRf(e.target.value)} style={inputStyle}>
            <option value="all">All</option><option value="low">Low (L1–2)</option><option value="guard">Guarded (L3–4)</option><option value="elev">Elevated (L5–6)</option><option value="high">High (L7–8)</option><option value="crit">Critical (L9–10)</option>
          </select>
        </div>
      </div>
      <div style={{ overflow:"auto", maxHeight:420 }}>

        {/* table starts here */}
        <table style={{ width:"100%", borderCollapse:"collapse", fontSize:14 }}>
         <thead>
  <tr>
    {['ID','Name','Location','Score','Level','Prediction','Status','Battery','Firmware','Updated']
      .map(h=>(<th key={h} style={th}>{h}</th>))}
  </tr>
</thead>

<tbody>
{filtered.map((h:any)=>(
  <tr key={h.id} onClick={()=> onPick(h)} style={{ cursor:"pointer" }}>
    <td style={td}>{h.id}</td>
    <td style={td}>{h.name}</td>
    <td style={tdMuted}>{h.city}, {h.country}</td>
    <td style={td}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        {/* <div style={{ width:64, height:8, borderRadius:9999, background:bandColor(h.level) }}/> */}
        <span style={{ fontVariantNumeric:"tabular-nums" }}>{h.score}</span>
      </div>
    </td>
    <td style={td}>
      <span style={{ padding:"2px 8px", borderRadius:9999,
                     background:h.level>=9?"#fee2e2":"#e2e8f0",
                     color:h.level>=9?"#991b1b":"#0f172a" }}>L{h.level}</span>
            </td>

            {/* 🔥 Prediction column */}
            <td style={td}>
              <span style={{
                padding:"2px 8px",
                borderRadius:9999,
                background: h.predictions==='fire' ? '#ee4445'
                          : h.predictions==='predicted' ? '#eab308'
                          : '#049668',
                color: h.predictions==='fire' ? '#ffffffff'
                    : h.predictions==='predicted' ? '#ffffffff'
                    : '#ffffffff'
              }}>
                {h.predictions}
              </span>
            </td>

            <td style={td}>
              {h.status==='online'
                ? <span style={{ color:"#059669" }}>Online</span>
                : <span style={{ color:"#64748b" }}>Offline</span>}
            </td>
            <td style={td}>{Math.round(h.battery*100)}%</td>
            <td style={td}>{h.firmware}</td>
            <td style={tdMuted}>{h.updated}</td>
          </tr>
        ))}
        </tbody>
        </table>
      </div>
    </div>
  )
}



// Chronos with real API call to fetch sensor data

// Alert history interface
interface AlertHistoryEvent {
  id: number;
  timestamp: string;
  eventType: string;
  riskScore: number;
  level: number;
  message: string;
  confidence: number;
}

function Chronos({ sel, onClose }: { sel: any; onClose: () => void }) {
  const [sensorData, setSensorData] = useState<any | null>(null);
  const [alertState, setAlertState] = useState<AlertState | null>(null);
  const [alertHistory, setAlertHistory] = useState<AlertHistoryEvent[]>([]);
  console.log(sensorData)
  // const [safeData, setSafeData] = useState(null);
  // const [safeError, setSafeError] = useState("");

  console.log("Fetched sensor data:", sensorData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch alert state and history from backend
  useEffect(() => {
    if (!sel) return;

    const fetchAlertState = async () => {
      try {
        const response = await axios.get('/api/alert-state');
        if (response.data.status === 'success') {
          setAlertState(response.data.data);
        }
      } catch (err: any) {
        console.error('Error fetching alert state:', err);
      }
    };

    const fetchAlertHistory = async () => {
      try {
        const response = await axios.get('/api/alert-history?limit=10');
        if (response.data.status === 'success') {
          setAlertHistory(response.data.data);
        }
      } catch (err: any) {
        console.error('Error fetching alert history:', err);
      }
    };

    // Fetch immediately
    fetchAlertState();
    fetchAlertHistory();

    // Poll every 5 seconds for real-time updates
    const interval = setInterval(() => {
      fetchAlertState();
      fetchAlertHistory();
    }, 5000);

    return () => clearInterval(interval);
  }, [sel]);

  // Fetch sensor data when component mounts or sel changes
  useEffect(() => {
  if (!sel) return;

  const fetchSensors = async () => {
    setLoading(true);
    setError(null);


    // API FOR NON_FIRE ENDPOINT
    const payload = {
  "frame": 9012,
  "timestamp": "2025-09-08T13:10:00Z",
  "features": {
    "t_mean": 22.0, "t_std": 0.1, "t_max": 22.3, "t_p95": 22.2,
    "t_hot_area_pct": 0.0, "t_hot_largest_blob_pct": 0.0,

    "t_grad_mean": 0.0, "t_grad_std": 0.0,
    "t_diff_mean": 0.0, "t_diff_std": 0.0,

    "flow_mag_mean": 0.02, "flow_mag_std": 0.002,

    "tproxy_val": 22.1, "tproxy_delta": 0.0, "tproxy_vel": 0.0,

    "CO": 0.03, "VOC": 0.06, "NO2": 0.004,
    "CO_diff": -0.001, "VOC_diff": -0.001, "NO2_diff": 0.0,
    "VOC_ma5": 0.06, "CO_ma5": 0.03, "NO2_ma5": 0.004,

    "VOC_z": -0.15, "CO_z": -0.15, "NO2_z": -0.05,

    "temp_rise_c_per_min": 0.0, "temp_slope_30s": 0.0,
    "gas_var_30s": 0.0, "delta_temp_30s": 0.0, "delta_gas_10s": 0.0,
    "spike_count_voc_2m": 0,

    "temp_co_corr_lag_0s": 0.0, "temp_co_corr_lag_15s": 0.0, "temp_co_corr_lag_60s": 0.0,
    "temp_voc_corr_lag_0s": 0.0, "temp_voc_corr_lag_15s": 0.0, "temp_voc_corr_lag_60s": 0.0,
    "temp_co_xcorr_max_abs": 0.005, "temp_voc_xcorr_max_abs": 0.005,

    "is_weekend": 0, "asleep_window": 0,

    "hrblk_0": 0, "hrblk_1": 1, "hrblk_2": 0, "hrblk_3": 0, "hrblk_4": 0, "hrblk_5": 0
  },
  "decision_threshold": 0.40
}


// // API FOR fire started ENDPOINT
// const payload = {
//   "frame": 1234,
//   "timestamp": "2025-09-08T12:34:56Z",
//   "features": {
//     "t_mean": 28.12, "t_std": 0.83, "t_max": 74.56, "t_p95": 71.92,
//     "t_hot_area_pct": 8.20, "t_hot_largest_blob_pct": 5.47,
//     "t_grad_mean": 0.42, "t_grad_std": 0.25, "t_diff_mean": 0.18, "t_diff_std": 0.09,
//     "flow_mag_mean": 0.50, "flow_mag_std": 0.05,
//     "tproxy_val": 74.56, "tproxy_delta": 1.32, "tproxy_vel": 0.87,
//     "CO": 0.9, "VOC": 2.5, "NO2": 0.03,
//     "CO_diff": 0.30, "VOC_diff": 0.40, "NO2_diff": -0.01,
//     "VOC_ma5": 2.10, "CO_ma5": 0.75, "NO2_ma5": 0.02,
//     "VOC_z": 2.2, "CO_z": 1.1, "NO2_z": -0.2,
//     "temp_rise_c_per_min": 12.5, "temp_slope_30s": 3.2,
//     "gas_var_30s": 0.45, "delta_temp_30s": 8.7, "delta_gas_10s": 0.6,
//     "spike_count_voc_2m": 4,
//     "temp_co_corr_lag_0s": 0.72, "temp_co_corr_lag_15s": 0.68, "temp_co_corr_lag_60s": 0.55,
//     "temp_voc_corr_lag_0s": 0.81, "temp_voc_corr_lag_15s": 0.77, "temp_voc_corr_lag_60s": 0.60,
//     "temp_co_xcorr_max_abs": 0.74, "temp_voc_xcorr_max_abs": 0.83,
//     "is_weekend": 0, "asleep_window": 1,
//     "hrblk_0": 0, "hrblk_1": 0, "hrblk_2": 0, "hrblk_3": 0, "hrblk_4": 1, "hrblk_5": 0
//   },
//   "decision_threshold": 0.4
// }


// // API FOR fire PREDICTED ENDPOINT
// const payload = {
//   "frame": 5678,
//   "timestamp": "2025-09-08T12:45:00Z",
//   "features": {
//     "t_mean": 44.0, "t_std": 0.5, "t_max": 28.0, "t_p95": 27.5,
//     "t_hot_area_pct": 0.4, "t_hot_largest_blob_pct": 0.1,
//     "t_grad_mean": 0.05, "t_grad_std": 0.02, "t_diff_mean": 0.03, "t_diff_std": 0.01,
//     "flow_mag_mean": 0.1, "flow_mag_std": 0.01,
//     "tproxy_val": 28.0, "tproxy_delta": 0.2, "tproxy_vel": 0.05,

//     "CO": 0.2, "VOC": 0.5, "NO2": 0.01,
//     "CO_diff": 0.02, "VOC_diff": 0.03, "NO2_diff": 0.0,
//     "VOC_ma5": 0.4, "CO_ma5": 0.15, "NO2_ma5": 0.01,
//     "VOC_z": 0.1, "CO_z": 0.1, "NO2_z": 0.0,

//     "temp_rise_c_per_min": 0.2, "temp_slope_30s": 0.1,
//     "gas_var_30s": 0.05, "delta_temp_30s": 0.2, "delta_gas_10s": 0.01,
//     "spike_count_voc_2m": 0,

//     "temp_co_corr_lag_0s": 0.20, "temp_co_corr_lag_15s": 0.08, "temp_co_corr_lag_60s": 0.05,
//     "temp_voc_corr_lag_0s": 0.12, "temp_voc_corr_lag_15s": 0.10, "temp_voc_corr_lag_60s": 0.08,
//     "temp_co_xcorr_max_abs": 0.15, "temp_voc_xcorr_max_abs": 0.18,

//     "is_weekend": 0, "asleep_window": 4,
//     "hrblk_0": 0, "hrblk_1": 0, "hrblk_2": 2, "hrblk_3": 0, "hrblk_4": 5, "hrblk_5": 0
//   },
//   "decision_threshold": 0.4
// }





   try {
  const res = await axios.post(
    // "https://b6vmdcuw7b.execute-api.us-east-1.amazonaws.com/predict",
    "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws",
    payload,
    { headers: { "Content-Type": "application/json" } }
  );
  console.log("Response", res);
  setSensorData(res.data);
} catch (err: any) {
  console.log("Full error object", err);
  console.log("Error message", err.message);
  console.log("Response data", err.response?.data);
  console.log("Response status", err.response?.status);
  setError(err.message || "Error fetching sensor data");
}
  };

  fetchSensors();
}, [sel]);

  if (!sel) return null;




  return (
    <div
      style={{
        position: "fixed",
        inset: "0 0 0 auto",
        width: 520,
        background: "white",
        boxShadow: "-16px 0 40px rgba(0,0,0,.1)",
        borderLeft: "1px solid #e5e7eb",
        padding: 16,
        overflowY: "auto",
        zIndex: 99
        
      }}
    >
      {/* ---------- Header ---------- */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ margin: 0, color: "#0f172a" }}>{sel.name}</h3>
        <button onClick={onClose} style={btn}>Close</button>
      </div>

      {/* ---------- Stats (Updated from Alert State) ---------- */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12 }}>
        <MiniStat
          label="Risk Score"
          value={alertState ? String(alertState.riskScore) : String(sel.score)}
          color={alertState ? bandColor(alertState.level) : bandColor(sel.level)}
        />
        <MiniStat
          label="Alert Level"
          value={alertState ? `L${alertState.level}` : `L${sel.level}`}
          color={alertState ? bandColor(alertState.level) : bandColor(sel.level)}
        />
        <MiniStat label="Battery" value={`${Math.round(sel.battery * 100)}%`} color="#34d399" />
        <MiniStat
          label="Status"
          value={alertState?.isActive ? "ACTIVE ALERT" : sel.status}
          color={alertState?.isActive ? "#ef4444" : (sel.status === "online" ? "#34d399" : "#94a3b8")}
        />
      </div>

      {/* ---------- Current Alert Status ---------- */}
      {alertState && (
        <section style={{ marginTop: 16 }}>
          <h4 style={h4}>Current Alert Status</h4>
          <div style={{
            ...box,
            background: alertState.isActive ? "#fee2e2" : "#dcfce7",
            borderColor: alertState.isActive ? "#ef4444" : "#22c55e"
          }}>
            <p style={{ margin: 0, color: "#334155", fontWeight: 600 }}>
              {alertState.message}
            </p>
            <p style={{ margin: "8px 0 0 0", fontSize: 12, color: "#64748b" }}>
              Confidence: {(alertState.confidence * 100).toFixed(0)}% |
              Last Updated: {new Date(alertState.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </section>
      )}

      {/* ---------- 24 AI Agents Consensus ---------- */}
      {alertState && (
        <section style={{ marginTop: 16 }}>
          <h4 style={h4}>24 AI Agents Consensus</h4>
          <div style={box}>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(6, 1fr)",
              gap: 8,
              marginBottom: 12
            }}>
              {Array.from({ length: 24 }, (_, i) => {
                // Determine agent status based on alert state
                let agentStatus: 'fire' | 'predicted' | 'no-fire';
                let agentColor: string;
                let agentIcon: string;
                
                if (alertState.riskScore >= 80) {
                  // Fire detected - most agents agree
                  agentStatus = Math.random() < 0.85 ? 'fire' : 'predicted';
                } else if (alertState.riskScore >= 40) {
                  // Fire predicted - mixed consensus
                  const rand = Math.random();
                  agentStatus = rand < 0.6 ? 'predicted' : rand < 0.8 ? 'fire' : 'no-fire';
                } else {
                  // No fire - most agents agree
                  agentStatus = Math.random() < 0.9 ? 'no-fire' : 'predicted';
                }

                if (agentStatus === 'fire') {
                  agentColor = '#ef4444';
                  agentIcon = '🔥';
                } else if (agentStatus === 'predicted') {
                  agentColor = '#eab308';
                  agentIcon = '⚠️';
                } else {
                  agentColor = '#22c55e';
                  agentIcon = '✅';
                }

                return (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: 8,
                      borderRadius: 8,
                      background: 'white',
                      border: `2px solid ${agentColor}`,
                      fontSize: 10,
                      fontWeight: 600
                    }}
                    title={`Agent ${i + 1}: ${agentStatus}`}
                  >
                    <div style={{ fontSize: 20, marginBottom: 4 }}>{agentIcon}</div>
                    <div style={{ color: agentColor }}>A{i + 1}</div>
                  </div>
                );
              })}
            </div>
            
            {/* Consensus Summary */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-around',
              padding: '12px 0',
              borderTop: '1px solid #e5e7eb'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ef4444' }}>
                  {alertState.riskScore >= 80 ? Math.floor(24 * 0.85) :
                   alertState.riskScore >= 40 ? Math.floor(24 * 0.2) :
                   Math.floor(24 * 0.05)}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>🔥 Fire</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#eab308' }}>
                  {alertState.riskScore >= 80 ? Math.floor(24 * 0.15) :
                   alertState.riskScore >= 40 ? Math.floor(24 * 0.6) :
                   Math.floor(24 * 0.1)}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>⚠️ Predicted</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#22c55e' }}>
                  {alertState.riskScore >= 80 ? Math.floor(24 * 0.0) :
                   alertState.riskScore >= 40 ? Math.floor(24 * 0.2) :
                   Math.floor(24 * 0.85)}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>✅ No-Fire</div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ---------- LLM Incident Summary ---------- */}
      <section style={{ marginTop: 16 }}>
        <h4 style={h4}>LLM Incident Summary</h4>
        <div style={box}>
          <p style={{ margin: 0, color: "#334155" }}>
            {alertState ? alertState.message :
             "At 11:32, temperature rose to 65°C with PM2.5 spikes and crackling audio near Kitchen. System escalated to L9, user notified; awaiting confirmation."}
          </p>
        </div>
      </section>

      {/* ---------- Live History ---------- */}
      <section style={{ marginTop: 16 }}>
        <h4 style={h4}>Live History</h4>
        <div style={box}>
          {alertHistory.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {alertHistory.map((event) => {
                // Determine color based on event type
                let badgeColor = '#22c55e'; // Green for Normal
                let badgeBg = '#dcfce7';
                let badgeIcon = '✅';
                
                if (event.eventType === 'Fire Detected') {
                  badgeColor = '#ef4444'; // Red
                  badgeBg = '#fee2e2';
                  badgeIcon = '🔥';
                } else if (event.eventType === 'Fire Predicted') {
                  badgeColor = '#eab308'; // Yellow
                  badgeBg = '#fef3c7';
                  badgeIcon = '⚠️';
                }
                
                return (
                  <div
                    key={event.id}
                    style={{
                      padding: 12,
                      borderRadius: 8,
                      border: `1px solid ${badgeColor}`,
                      background: badgeBg,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 6
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span
                        style={{
                          padding: '4px 12px',
                          borderRadius: 9999,
                          background: badgeColor,
                          color: 'white',
                          fontSize: 12,
                          fontWeight: 600,
                          display: 'flex',
                          alignItems: 'center',
                          gap: 4
                        }}
                      >
                        {badgeIcon} {event.eventType}
                      </span>
                      <span style={{ fontSize: 11, color: '#64748b' }}>
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div style={{ fontSize: 13, color: '#334155' }}>
                      <div><strong>Risk Score:</strong> {event.riskScore}</div>
                      <div><strong>Level:</strong> L{event.level}</div>
                      <div><strong>Confidence:</strong> {(event.confidence * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p style={{ margin: 0, color: '#64748b', textAlign: 'center' }}>
              No alert history available
            </p>
          )}
        </div>
      </section>







      {/* ---------- History ---------- */}
      <section style={{ marginTop: 16 }}>
        <h4 style={h4}>History</h4>
        <ul style={{ margin: 0, paddingLeft: 18, color: "#475569" }}>
          <li>10:54 — L7 escalated → user app push</li>
          <li>10:58 — L9 critical → SMS + auto-call</li>
          <li>11:03 — Resolved; ventilation restored</li>
        </ul>
      </section>

      {/* ---------- Actions ---------- */}
      <section style={{ marginTop: 16 }}>
        <h4 style={h4}>Actions</h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <button style={btnDestructive}>Force Escalation</button>
          <button style={btnOutline}>Mark False Positive</button>
          <button style={btnOutline}>Trigger Test</button>
          <button style={btnOutline}>Remote Mute</button>
        </div>
        <p style={{ fontSize: 12, color: "#64748b", marginTop: 8 }}>
          <strong>Policy:</strong> L6–7 notify & require user confirm; L8–10 escalate via redundant channels (push/SMS/auto-call).
        </p>
      </section>
    </div>
  );
}










// function Chronos({ sel, onClose }: { sel: any; onClose: () => void }) {
//   if (!sel) return null;

//   return (
//     <div
//       style={{
//         position: "fixed",
//         inset: "0 0 0 auto",
//         width: 520,
//         background: "white",
//         boxShadow: "-16px 0 40px rgba(0,0,0,.1)",
//         borderLeft: "1px solid #e5e7eb",
//         padding: 16,
//         overflowY: "auto",
//       }}
//     >
//       {/* ---------- Header ---------- */}
//       <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
//         <h3 style={{ margin: 0, color: "#0f172a" }}>{sel.name}</h3>
//         <button onClick={onClose} style={btn}>
//           Close
//         </button>
//       </div>

//       {/* ---------- Stats ---------- */}
//       <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12 }}>
//         <MiniStat label="Risk Score" value={String(sel.score)} color={bandColor(sel.level)} />
//         <MiniStat label="Alert Level" value={`L${sel.level}`} color={bandColor(sel.level)} />
//         <MiniStat label="Battery" value={`${Math.round(sel.battery * 100)}%`} color="#34d399" />
//         <MiniStat label="Status" value={sel.status} color={sel.status === "online" ? "#34d399" : "#94a3b8"} />
//       </div>

//       {/* ---------- LLM Incident Summary ---------- */}
//       <section style={{ marginTop: 16 }}>
//         <h4 style={h4}>LLM Incident Summary</h4>
//         <div style={box}>
//           <p style={{ margin: 0, color: "#334155" }}>
//             At 11:32, temperature rose to 65°C with PM2.5 spikes and crackling audio near <strong>Kitchen</strong>.
//             System escalated to <strong>L9</strong>, user notified; awaiting confirmation.
//           </p>
//         </div>
//       </section>

//       {/* ---------- Sensors (static payload) ---------- */}
//       <section style={{ marginTop: 16 }}>
//         <h4 style={h4}>Sensors</h4>

//         {/* Fire-insight summary */}
//         <div style={box}>
//           <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
//             <span style={{ fontWeight: 600 }}>Fire probability</span>
//             <span style={{ fontWeight: 700, color: "#ef4444" }}>62.4%</span>
//             <span style={{ fontSize: 12, padding: "2px 6px", borderRadius: 4, background: "#fee2e2", color: "#991b1b" }}>Fire</span>
//           </div>

//           {/* Global drivers */}
//           <div style={{ fontSize: 13, color: "#475569", margin: "12px 0 4px" }}>Top global drivers</div>
//           <table style={{ width: "100%", fontSize: 13 }}>
//             <thead>
//               <tr style={{ color: "#64748b" }}>
//                 <th style={{ textAlign: "left" }}>Feature</th>
//                 <th style={{ textAlign: "right" }}>Importance</th>
//               </tr>
//             </thead>
//             <tbody>
//               <tr><td>hrblk_3</td><td style={{ textAlign: "right" }}>1602.56</td></tr>
//               <tr><td>spike_count_voc_2m</td><td style={{ textAlign: "right" }}>733.78</td></tr>
//               <tr><td>temp_slope_30s</td><td style={{ textAlign: "right" }}>58.65</td></tr>
//               <tr><td>hrblk_4</td><td style={{ textAlign: "right" }}>52.05</td></tr>
//               <tr><td>asleep_window</td><td style={{ textAlign: "right" }}>35.74</td></tr>
//               <tr><td>delta_temp_30s</td><td style={{ textAlign: "right" }}>23.75</td></tr>
//             </tbody>
//           </table>

//           {/* Local contributions */}
//           <div style={{ fontSize: 13, color: "#475569", margin: "12px 0 4px" }}>Local contributions</div>
//           <table style={{ width: "100%", fontSize: 13 }}>
//             <thead>
//               <tr style={{ color: "#64748b" }}>
//                 <th style={{ textAlign: "left" }}>Feature</th>
//                 <th style={{ textAlign: "right" }}>Value</th>
//                 <th style={{ textAlign: "right" }}>Score</th>
//               </tr>
//             </thead>
//             <tbody>
//               <tr><td>hrblk_3</td><td style={{ textAlign: "right" }}>0.0</td><td style={{ textAlign: "right" }}>-4.07</td></tr>
//               <tr><td>spike_count_voc_2m</td><td style={{ textAlign: "right" }}>4.0</td><td style={{ textAlign: "right" }}>2.02</td></tr>
//               <tr><td>temp_slope_30s</td><td style={{ textAlign: "right" }}>3.2</td><td style={{ textAlign: "right" }}>1.41</td></tr>
//               <tr><td>is_weekend</td><td style={{ textAlign: "right" }}>0.0</td><td style={{ textAlign: "right" }}>0.60</td></tr>
//               <tr><td>hrblk_4</td><td style={{ textAlign: "right" }}>1.0</td><td style={{ textAlign: "right" }}>-0.34</td></tr>
//               <tr><td>delta_temp_30s</td><td style={{ textAlign: "right" }}>8.7</td><td style={{ textAlign: "right" }}>0.32</td></tr>
//               <tr><td>delta_gas_10s</td><td style={{ textAlign: "right" }}>0.6</td><td style={{ textAlign: "right" }}>0.25</td></tr>
//             </tbody>
//           </table>
//         </div>
//       </section>

//       {/* ---------- History ---------- */}
//       <section style={{ marginTop: 16 }}>
//         <h4 style={h4}>History</h4>
//         <ul style={{ margin: 0, paddingLeft: 18, color: "#475569" }}>
//           <li>10:54 — L7 escalated → user app push</li>
//           <li>10:58 — L9 critical → SMS + auto-call</li>
//           <li>11:03 — Resolved; ventilation restored</li>
//         </ul>
//       </section>

//       {/* ---------- Actions ---------- */}
//       <section style={{ marginTop: 16 }}>
//         <h4 style={h4}>Actions</h4>
//         <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
//           <button style={btnDestructive}>Force Escalation</button>
//           <button style={btnOutline}>Mark False Positive</button>
//           <button style={btnOutline}>Trigger Test</button>
//           <button style={btnOutline}>Remote Mute</button>
//         </div>
//         <p style={{ fontSize: 12, color: "#64748b", marginTop: 8 }}>
//           <strong>Policy:</strong> L6–7 notify & require user confirm; L8–10 escalate via redundant channels (push/SMS/auto-call).
//         </p>
//       </section>
//     </div>
//   );
// }

// Athena
function Athena({ rows }:{ rows:any[] }){
  const total = rows.length, online = rows.filter(r=> r.status==='online').length;
  const critical = rows.filter(r=> r.level>=9).length, high = rows.filter(r=> r.level>=7 && r.level<9).length;
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white" }}>
      <div style={{ padding:12 }}><strong style={{ color:"#0f172a" }}>Athena — Strategic Dashboard</strong></div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap:8, padding:12 }}>
        <KPI label="Devices Online" value={`${online}/${total}`} />
        <KPI label="Critical (L9–10)" value={String(critical)} accent="#ef4444" />
        <KPI label="High (L7–8)" value={String(high)} accent="#f97316" />
        <KPI label="Avg Score" value={String(Math.round(rows.reduce((a,b)=>a+b.score,0)/rows.length))} />
      </div>
      <div style={{ padding:12 }}>
        <TinyTrend title="Incidents (last 24h)" data={[2,3,5,4,6,7,5,4,3,6,9,8]} />
      </div>
    </div>
  )
}

// SAAFEGPT
function SAAFEGPT(){
  const [q,setQ] = useState(""); const [log,setLog] = useState<any[]>([]);
  async function ask(){
    const userMsg = { role:'user', content:q }; setLog(L=> [...L, userMsg]); setQ("");
    const res = await askSaafeGPT([...log, userMsg]); setLog(L=> [...L, res.message]);
  }
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"white" }}>
      <div style={{ padding:12 }}><strong style={{ color:"#0f172a" }}>SAAFEGPT</strong></div>
      <div style={{ height:180, overflow:"auto", padding:"0 12px" }}>
        {log.map((m,i)=>(<div key={i} style={{ margin:"8px 0", color: m.role==='user'?"#0f172a":"#334155" }}><b>{m.role==='user'?'You':'SAAFEGPT'}:</b> {m.content}</div>))}
      </div>
      <div style={{ padding:12, display:"flex", gap:8 }}>
        <input placeholder="Ask about an incident or KPI…" value={q} onChange={e=> setQ(e.target.value)} style={{ ...inputStyle, flex:1 }} />
        <button onClick={ask} style={btn}>Ask</button>
      </div>
      <div style={{ padding:"0 12px 12px", fontSize:12, color:"#64748b" }}>Backend: POST /api/saafegpt → {"{ message: { role, content } }"}</div>
    </div>
  )
}

// Alert Status Display Component
function AlertStatusDisplay({ alertState }: { alertState: AlertState | null }) {
  if (!alertState) {
    return (
      <div style={{
        border: "1px solid #e5e7eb",
        borderRadius: 16,
        padding: 20,
        background: "white",
        marginBottom: 12,
        textAlign: "center"
      }}>
        <div style={{ fontSize: 14, color: "#64748b" }}>Loading alert status...</div>
      </div>
    );
  }

  // Determine alert display based on level and message
  let alertColor = "#22c55e"; // Green for normal
  let alertIcon = "✅";
  let alertLabel = "Non-Fire";
  let alertBg = "#dcfce7";
  let alertTextColor = "#166534";

  if (alertState.message.toLowerCase().includes("fire") && !alertState.message.toLowerCase().includes("non")) {
    if (alertState.message.toLowerCase().includes("predicted")) {
      alertColor = "#eab308"; // Yellow for predicted
      alertIcon = "⚠️";
      alertLabel = "Fire Predicted";
      alertBg = "#fef3c7";
      alertTextColor = "#854d0e";
    } else {
      alertColor = "#ef4444"; // Red for fire
      alertIcon = "🔥";
      alertLabel = "FIRE DETECTED";
      alertBg = "#fee2e2";
      alertTextColor = "#991b1b";
    }
  }

  return (
    <div style={{
      border: `3px solid ${alertColor}`,
      borderRadius: 16,
      padding: 24,
      background: alertBg,
      marginBottom: 12,
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)"
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
        {/* Alert Status */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            fontSize: 48,
            lineHeight: 1
          }}>
            {alertIcon}
          </div>
          <div>
            <div style={{
              fontSize: 28,
              fontWeight: "bold",
              color: alertTextColor,
              marginBottom: 4
            }}>
              {alertLabel}
            </div>
            <div style={{
              fontSize: 16,
              color: alertTextColor,
              opacity: 0.8
            }}>
              {alertState.message}
            </div>
          </div>
        </div>

        {/* Alert Metrics */}
        <div style={{ display: "flex", gap: 16 }}>
          <div style={{
            background: "white",
            padding: 12,
            borderRadius: 12,
            minWidth: 120,
            textAlign: "center"
          }}>
            <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>Risk Score</div>
            <div style={{ fontSize: 24, fontWeight: "bold", color: alertTextColor }}>
              {alertState.riskScore}
            </div>
          </div>
          <div style={{
            background: "white",
            padding: 12,
            borderRadius: 12,
            minWidth: 120,
            textAlign: "center"
          }}>
            <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>Confidence</div>
            <div style={{ fontSize: 24, fontWeight: "bold", color: alertTextColor }}>
              {(alertState.confidence * 100).toFixed(0)}%
            </div>
          </div>
          <div style={{
            background: "white",
            padding: 12,
            borderRadius: 12,
            minWidth: 120,
            textAlign: "center"
          }}>
            <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>Alert Level</div>
            <div style={{ fontSize: 24, fontWeight: "bold", color: alertTextColor }}>
              L{alertState.level}
            </div>
          </div>
        </div>
      </div>

      {/* Timestamp */}
      <div style={{
        marginTop: 12,
        fontSize: 12,
        color: alertTextColor,
        opacity: 0.7,
        textAlign: "right"
      }}>
        Last updated: {new Date(alertState.timestamp).toLocaleString()}
      </div>
    </div>
  );
}

// Main
export default function SaafeLovable(){
  const [selected, setSelected] = useState<any|null>(null);
  const [alertState, setAlertState] = useState<AlertState | null>(null);
  const [userLocations, setUserLocations] = useState<any[]>(() => {
    // Load user locations from localStorage and deduplicate
    const saved = localStorage.getItem('saafe-user-locations');
    if (saved) {
      const parsed = JSON.parse(saved);
      // Remove duplicates based on ID
      const uniqueLocations = parsed.filter((location: any, index: number, self: any[]) =>
        index === self.findIndex((l: any) => l.id === location.id)
      );
      // If duplicates were found, update localStorage
      if (uniqueLocations.length !== parsed.length) {
        localStorage.setItem('saafe-user-locations', JSON.stringify(uniqueLocations));
      }
      return uniqueLocations;
    }
    return [];
  });

  // State to hold updated homes data with alert-based modifications
  const [homesData_updated, setHomesData_updated] = useState<any[]>([...(homesData as any)]);

  // Combine updated homes data with user locations
  const homes = [...homesData_updated, ...userLocations];

  // Debug: Log homes data
  console.log('Homes data:', homes);
  console.log('HomesData length:', homesData?.length);
  console.log('User locations length:', userLocations?.length);

  const handleAddLocation = (newLocation: any) => {
    const locationWithId = { ...newLocation, isUserAdded: true };
    const updatedLocations = [...userLocations, locationWithId];
    setUserLocations(updatedLocations);
    localStorage.setItem('saafe-user-locations', JSON.stringify(updatedLocations));
  };

  const handleDeleteLocation = (locationId: string) => {
    const updatedLocations = userLocations.filter(loc => loc.id !== locationId);
    setUserLocations(updatedLocations);
    localStorage.setItem('saafe-user-locations', JSON.stringify(updatedLocations));
  };

  // Fetch alert state from backend and update homes data
  useEffect(() => {
    const fetchAlertState = async () => {
      try {
        const response = await axios.get('/api/alert-state');
        if (response.data.status === 'success') {
          const newAlertState = response.data.data;
          setAlertState(newAlertState);
          
          // Update homes data based on alert state
          const updatedHomes = (homesData as any[]).map((home: any) => {
            // Determine prediction status based on risk score
            let predictions = 'no-fire';
            let level = home.level;
            let score = home.score;
            
            if (newAlertState.riskScore >= 80) {
              predictions = 'fire';
              level = Math.max(level, 9); // Ensure level is at least 9 for fire
              score = Math.max(score, newAlertState.riskScore);
            } else if (newAlertState.riskScore >= 40) {
              predictions = 'predicted';
              level = Math.max(level, 5); // Ensure level is at least 5 for predicted
              score = Math.max(score, newAlertState.riskScore);
            } else {
              predictions = 'no-fire';
              level = Math.min(level, 3); // Keep level low for no-fire
              score = Math.min(score, 30);
            }
            
            return {
              ...home,
              predictions,
              level,
              score,
              updated: new Date().toLocaleTimeString()
            };
          });
          
          setHomesData_updated(updatedHomes);
        }
      } catch (error) {
        console.error('Error fetching alert state:', error);
      }
    };

    // Fetch immediately
    fetchAlertState();

    // Poll every 5 seconds for updates
    const interval = setInterval(fetchAlertState, 5000);

    return () => clearInterval(interval);
  }, []);

  // Optional mock stream: uncomment to simulate updates
  // useEffect(()=>{
  //   const id = setInterval(()=>{
  //     const idx = Math.floor(Math.random()*homes.length);
  //     homes[idx].level = Math.max(1, Math.min(10, homes[idx].level + (Math.random()<0.5?-1:1)));
  //     homes[idx].score = Math.max(0, Math.min(100, homes[idx].score + (Math.random()<0.5?-5:5)));
  //   }, 1500);
  //   return ()=> clearInterval(id);
  // },[]);
  return (
    <div style={{ minHeight:"100vh", padding:16 }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:12 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ padding:8, borderRadius:12, background:"#059669", color:"white", fontWeight:600 }}>S</div>
          <h2 style={{ margin:0, color:"#0f172a" }}>SAAFE Global Command Center 1</h2>
          <span style={{ marginLeft:8, fontSize:12, padding:"2px 8px", background:"#e2e8f0", borderRadius:9999 }}>MVP</span>
        </div>
        <div style={{ display:"flex", gap:8 }}>
          <button style={btnOutline}>Sync</button>
          <button style={btn}>Safe Mode</button>
        </div>
      </div>

      {/* Alert Status Display */}
      <AlertStatusDisplay alertState={alertState} />

      <div style={{ display:"grid", gridTemplateColumns:"2fr 1fr", gap:12, marginBottom:12 }}>
        <Helios2D homes={homes} onAddLocation={handleAddLocation} onDeleteLocation={handleDeleteLocation} />
        <Athena rows={homes} />
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"2fr 1fr", gap:12 }}>
        <Grid rows={homes} onPick={setSelected} />
        <SAAFEGPT />
      </div>

      <Chronos sel={selected} onClose={()=> setSelected(null)} />
    </div>
  )
}

// UI atoms
function KPI({ label, value, accent="#0ea5e9" }:{ label:string; value:string; accent?:string }){
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
      <div style={{ fontSize:12, color:"#64748b" }}>{label}</div>
      <div style={{ fontSize:22, fontWeight:700, color:"#0f172a" }}>{value}</div>
      <div style={{ height:6, background:"#f1f5f9", borderRadius:9999, marginTop:8 }}>
        <div style={{ width:"60%", height:"100%", background:accent, borderRadius:9999 }} />
      </div>
    </div>
  )
}

function TinyTrend({ title, data }:{ title:string; data:number[] }){
  const max = Math.max(...data);
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
      <div style={{ fontSize:12, color:"#64748b", marginBottom:8 }}>{title}</div>
      <svg width="100%" height="60" viewBox={`0 0 ${data.length*12} 60`}>
        {data.map((v,i)=>{ const h = Math.max(2,(v/max)*54); return <rect key={i} x={i*12} y={60-h} width={8} height={h} fill="#0ea5e9" rx={3} /> })}
      </svg>
    </div>
  )
}

function MiniStat({ label, value, color }:{ label:string; value:string; color:string }){
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:12 }}>
      <div style={{ fontSize:12, color:"#64748b" }}>{label}</div>
      <div style={{ display:"flex", alignItems:"center", gap:8, marginTop:6 }}>
        <div style={{ width:6, height:24, borderRadius:6, background:color }} />
        <div style={{ fontWeight:700, color:"#0f172a" }}>{value}</div>
      </div>
    </div>
  )
}

function Bar({ label, value, unit, warn }:{ label:string; value:number; unit:string; warn?:boolean }){
  const pct = Math.min(100, Math.round(value));
  const fill = warn?"#f97316":"#10b981";
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:12, padding:10 }}>
      <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, color:"#64748b" }}>
        <span>{label}</span><span style={{ color: warn?"#b45309":"#0f172a" }}>{value} {unit}</span>
      </div>
      <div style={{ height:8, background:"#f1f5f9", borderRadius:9999, marginTop:8 }}>
        <div style={{ width:`${pct}%`, height:"100%", background:fill, borderRadius:9999 }} />
      </div>
    </div>
  )
}

const inputStyle:any = { border:"1px solid #e5e7eb", borderRadius:8, padding:"8px 10px", outline:"none" };
const th:any = { textAlign:"left", padding:"10px 12px", borderBottom:"1px solid #e5e7eb" };
const td:any = { padding:"10px 12px", borderBottom:"1px solid #e5e7eb", color:"#0f172a" };
const tdMuted:any = { padding:"10px 12px", borderBottom:"1px solid #e5e7eb", color:"#64748b" };
const btn:any = { border:"1px solid #059669", background:"#059669", color:"white", padding:"8px 12px", borderRadius:10, cursor:"pointer" };
const btnOutline:any = { border:"1px solid #e5e7eb", background:"white", color:"#0f172a", padding:"8px 12px", borderRadius:10, cursor:"pointer" };
const btnDestructive:any = { border:"1px solid #ef4444", background:"#ef4444", color:"white", padding:"8px 12px", borderRadius:10, cursor:"pointer" };
const box:any = { border:"1px solid #e5e7eb", borderRadius:12, padding:12, background:"#f8fafc" };
const h4:any = { margin:"6px 0", color:"#0f172a" } ;
