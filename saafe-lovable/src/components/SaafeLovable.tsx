import React, { useMemo, useState, useEffect, useRef } from "react";
import homesData from "../data/sampleHomes.json"
import { askSaafeGPT } from "../api/saafegpt"
import FireDetectionDashboard from "./FireDetectionDashboard"
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

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
  const [isSpinning, setIsSpinning] = useState(true);

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
        const newLocation = {
          id: `SAAFE-${String(homes.length + 1).padStart(4, '0')}`,
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
          battery: Math.random() * 0.5 + 0.5 // Random battery 0.5-1.0
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
  
        // Add global delete function for popup buttons
        (window as any).deleteLocation = (locationId: string) => {
          onDeleteLocation(locationId);
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
          score: h.score
        },
        geometry: {
          type: 'Point',
          coordinates: [h.lon, h.lat]
        }
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
            'step',
            ['get', 'point_count'],
            '#51bbd6',
            100,
            '#f1f075',
            750,
            '#f28cb1'
          ],
          'circle-radius': [
            'step',
            ['get', 'point_count'],
            20,
            100,
            30,
            750,
            40
          ]
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
            ['get', 'level'],
            1, '#34d399',
            2, '#34d399',
            3, '#a3e635',
            4, '#a3e635',
            5, '#f59e0b',
            6, '#f59e0b',
            7, '#f97316',
            8, '#f97316',
            9, '#ef4444',
            10, '#ef4444',
            '#34d399'
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
      const popupContent = `
        <div style="min-width: 200px;">
          <strong>${properties!.name}</strong><br/>
          ${properties!.city}, ${properties!.country}<br/>
          Level: L${properties!.level} | Score: ${properties!.score}
          ${isUserAdded ? `<br/><button onclick="window.deleteLocation('${properties!.id}')" style="margin-top: 8px; padding: 4px 8px; background: #ef4444; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Delete Location</button>` : ''}
        </div>
      `;

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
        <div style={{
          position: "absolute",
          left: 12,
          bottom: 12,
          display: "flex",
          gap: 10,
          padding: 8,
          background: "rgba(255,255,255,0.95)",
          borderRadius: 12,
          fontSize: 12,
          zIndex: 1,
          pointerEvents: "none"
        }}>
          {[["#34d399","L1–2 Low"],["#a3e635","L3–4 Guarded"],["#f59e0b","L5–6 Elevated"],["#f97316","L7–8 High"],["#ef4444","L9–10 Critical"]].map(([c,l])=>(
            <div key={String(l)} style={{ display:"flex", alignItems:"center", gap:6, pointerEvents: "auto" }}>
              <span style={{ width:10, height:10, background:String(c), borderRadius:9999 }} /><span>{l}</span>
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
        <strong style={{ color:"#0f172a" }}>Grid — Asset Manager</strong>
        <div style={{ display:"flex", gap:8 }}>
          <input value={q} onChange={e=> setQ(e.target.value)} placeholder="Search id / name / city" style={inputStyle}/>
          <select value={rf} onChange={e=> setRf(e.target.value)} style={inputStyle}>
            <option value="all">All</option><option value="low">Low (L1–2)</option><option value="guard">Guarded (L3–4)</option><option value="elev">Elevated (L5–6)</option><option value="high">High (L7–8)</option><option value="crit">Critical (L9–10)</option>
          </select>
        </div>
      </div>
      <div style={{ overflow:"auto", maxHeight:420 }}>
        <table style={{ width:"100%", borderCollapse:"collapse", fontSize:14 }}>
          <thead style={{ position:"sticky", top:0, background:"#f8fafc", color:"#475569" }}>
            <tr>{['ID','Name','Location','Score','Level','Status','Battery','Firmware','Updated'].map(h=>(<th key={h} style={th}>{h}</th>))}</tr>
          </thead>
          <tbody>
            {filtered.map((h:any)=>(
              <tr key={h.id} onClick={()=> onPick(h)} style={{ cursor:"pointer" }}>
                <td style={td}>{h.id}</td><td style={td}>{h.name}</td><td style={tdMuted}>{h.city}, {h.country}</td>
                <td style={td}><div style={{ display:"flex", alignItems:"center", gap:8 }}><div style={{ width:64, height:8, borderRadius:9999, background:bandColor(h.level) }}/><span style={{ fontVariantNumeric:"tabular-nums" }}>{h.score}</span></div></td>
                <td style={td}><span style={{ padding:"2px 8px", borderRadius:9999, background:h.level>=9?"#fee2e2":"#e2e8f0", color:h.level>=9?"#991b1b":"#0f172a" }}>L{h.level}</span></td>
                <td style={td}>{h.status==='online'?<span style={{ color:"#059669" }}>Online</span>:<span style={{ color:"#64748b" }}>Offline</span>}</td>
                <td style={td}><span style={{ fontVariantNumeric:"tabular-nums" }}>{Math.round(h.battery*100)}%</span></td>
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

// Chronos
function Chronos({ sel, onClose }:{ sel:any; onClose:()=>void }){
  if (!sel) return null;
  return (
    <div style={{ position:"fixed", inset:"0 0 0 auto", width:520, background:"white", boxShadow:"-16px 0 40px rgba(0,0,0,.1)", borderLeft:"1px solid #e5e7eb", padding:16, overflowY:"auto" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <h3 style={{ margin:0, color:"#0f172a" }}>{sel.name}</h3>
        <button onClick={onClose} style={btn}>Close</button>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginTop:12 }}>
        <MiniStat label="Risk Score" value={String(sel.score)} color={bandColor(sel.level)} />
        <MiniStat label="Alert Level" value={`L${sel.level}`} color={bandColor(sel.level)} />
        <MiniStat label="Battery" value={`${Math.round(sel.battery*100)}%`} color="#34d399" />
        <MiniStat label="Status" value={sel.status} color={sel.status==='online'?"#34d399":"#94a3b8"} />
      </div>
      <section style={{ marginTop:16 }}>
        <h4 style={h4}>LLM Incident Summary</h4>
        <div style={box}><p style={{ margin:0, color:"#334155" }}>At 11:32, temperature rose to 65°C with PM2.5 spikes and crackling audio near <strong>Kitchen</strong>. System escalated to <strong>L9</strong>, user notified; awaiting confirmation.</p></div>
      </section>
      <section style={{ marginTop:16 }}>
        <h4 style={h4}>Sensors</h4>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
          <Bar label="Temperature" value={65} unit="°C" warn />
          <Bar label="PM2.5" value={180} unit="µg/m³" warn />
          <Bar label="CO₂" value={780} unit="ppm" warn />
          <Bar label="Humidity" value={35} unit="%" />
          <Bar label="Light" value={310} unit="lux" />
          <Bar label="Audio" value={0.6} unit="Δ dB" />
        </div>
      </section>
      <section style={{ marginTop:16 }}>
        <h4 style={h4}>History</h4>
        <ul style={{ margin:0, paddingLeft:18, color:"#475569" }}>
          <li>10:54 — L7 escalated → user app push</li>
          <li>10:58 — L9 critical → SMS + auto‑call</li>
          <li>11:03 — Resolved; ventilation restored</li>
        </ul>
      </section>
      <section style={{ marginTop:16 }}>
        <h4 style={h4}>Actions</h4>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
          <button style={btnDestructive}>Force Escalation</button>
          <button style={btnOutline}>Mark False Positive</button>
          <button style={btnOutline}>Trigger Test</button>
          <button style={btnOutline}>Remote Mute</button>
        </div>
        <p style={{ fontSize:12, color:"#64748b", marginTop:8 }}><strong>Policy:</strong> L6–7 notify & require user confirm; L8–10 escalate via redundant channels (push/SMS/auto‑call).</p>
      </section>
    </div>
  )
}

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

// Main
export default function SaafeLovable(){
  const [selected, setSelected] = useState<any|null>(null);
  const [userLocations, setUserLocations] = useState<any[]>(() => {
    // Load user locations from localStorage
    const saved = localStorage.getItem('saafe-user-locations');
    return saved ? JSON.parse(saved) : [];
  });

  // Combine original homes data with user locations
  const homes = [...(homesData as any), ...userLocations];

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
          <h2 style={{ margin:0, color:"#0f172a" }}>SAAFE Global Command Center</h2>
          <span style={{ marginLeft:8, fontSize:12, padding:"2px 8px", background:"#e2e8f0", borderRadius:9999 }}>MVP</span>
        </div>
        <div style={{ display:"flex", gap:8 }}>
          <button style={btnOutline}>Sync</button>
          <button style={btn}>Safe Mode</button>
        </div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"2fr 1fr", gap:12, marginBottom:12 }}>
        <Helios2D homes={homes} onAddLocation={handleAddLocation} onDeleteLocation={handleDeleteLocation} />
        <Athena rows={homes} />
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"2fr 1fr", gap:12 }}>
        <Grid rows={homes} onPick={setSelected} />
        <SAAFEGPT />
      </div>

      {/* Add the Fire Detection Dashboard */}
      <FireDetectionDashboard />

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
