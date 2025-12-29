
// // HeliosMap.tsx PROPER WORKING LIVE VERSION
// import { useEffect, useRef, useState } from "react";
// import mapboxgl from "mapbox-gl";
// import "mapbox-gl/dist/mapbox-gl.css";
// import { Button } from "@/components/ui/button";
// import { Play, Pause } from "lucide-react";
// import { Client } from "@gradio/client";
// import { Camera } from "@/pages/Index";

// mapboxgl.accessToken =
//   "pk.eyJ1IjoiaGFybGVxdWluaXJlIiwiYSI6ImNtZGdpMWRrejBsYzcybHB6eXp0Z25pYzcifQ.Mwu0v3MGK-eo6mEsbYjVng";

// const API_URL  = "https://08d6b4685d6057126e.gradio.live/";
// const AWS_URL  = "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws";

// interface HeliosMapProps {
//   onCameraSelect: (camera: Camera) => void;
// }


// export default function HeliosMap({ onCameraSelect }: HeliosMapProps) {
//   const mapContainer = useRef<HTMLDivElement>(null);
//   const map = useRef<mapboxgl.Map | null>(null);
//   const [isSpinning, setIsSpinning] = useState(false);
//   const [cameras, setCameras] = useState<Camera[]>([]);
//   const animationRef = useRef<number | null>(null);

//   const STORAGE_KEY = "helios-gradio-payload";
//   const savePayload = (obj: any) => localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));

//   /* ----------  NEW: probability → colour only  ---------- */
//   const probToColour = (p: number) => {
//     if (p >= 0.623939) return "fire";      // red
//     if (p <= 0.000028) return "no-fire";   // green
//     return "predicted";                    // yellow (in-between)
//   };


//   const statusToHsl = (status: string) => {
//   switch (status) {
//     case "fire":
//       return "hsl(15,90%,55%)";
//     case "predicted":
//       return "hsl(45,95%,55%)";
//     case "no-fire":
//       return "hsl(120,80%,45%)";   // green
//     case "black":
//       return "hsl(0,0%,0%)";       //  ➜  BLACK
  
//   }
// };

//   /* ----------  data fetching  ---------- */
//   const fetchCameras = async () => {
//     try {
//       const client = await Client.connect(API_URL);
//       // const now = new Date();
//       // const year = now.getUTCFullYear();
//       // const month = String(now.getUTCMonth() + 1).padStart(2, "0");
//       // const day = String(now.getUTCDate()).padStart(2, "0");
//       // const hours = String(now.getUTCHours()).padStart(2, "0");
//       // const minutes = String(now.getUTCMinutes()).padStart(2, "0");
//       // const timestamp = `${year}-${month}-${day} ${hours}:${minutes}`;

//       // const res = await client.predict("/predict", { timestamp_str: timestamp });

// /* ----------  UK (Europe/London) time  ---------- */
// const now = new Date();
// const ukTime = new Intl.DateTimeFormat("en-GB", {
//   timeZone: "Europe/London",
//   year: "numeric",
//   month: "2-digit",
//   day: "2-digit",
//   hour: "2-digit",
//   minute: "2-digit",
//   hour12: false,
// }).format(now);

// /*  DD/MM/YYYY HH:mm  →  split & re-format to YYYY-MM-DD HH:mm  */
// const [datePart, timePart] = ukTime.split(", ");
// const [day, month, year] = datePart.split("/");
// const timestamp = `${year}-${month}-${day} ${timePart}`;
// const [hours, minutes] = timePart.split(":");

// /*  use it  */
// const res = await client.predict("/predict", { timestamp_str: timestamp });

//       const transformedPayload = Array.isArray(res.data)
//         ? res.data.map((item: any) => ({
//             frame: 0,
//             target_minute: `dt=${year}-${month}-${day}/hour=${hours}/shane_${timestamp.replace(/:/g, "")}.json`,
//             timestamp: `${year}-${month}-${day} ${hours}:${minutes}:00+00:00`,
//             device_name: item.device_name || "shane",
//             device_id: item.device_id || "02",
//             device_location: item.device_location || [51.476782, -0.373907],
//             device_status: item.device_status ?? true,
//             features: {
//               t_mean: item.features?.t_mean ?? 26.82,
//               t_std: item.features?.t_std ?? 0.32,
//               t_max: item.features?.t_max ?? 27.88,
//               t_p95: item.features?.t_p95 ?? 27.42,
//               t_hot_area_pct: item.features?.t_hot_area_pct ?? 4.04,
//               t_hot_largest_blob_pct: item.features?.t_hot_largest_blob_pct ?? 2.21,
//               t_grad_mean: item.features?.t_grad_mean ?? 0.31,
//               t_grad_std: item.features?.t_grad_std ?? 0.19,
//               t_diff_mean: item.features?.t_diff_mean ?? 0,
//               t_diff_std: item.features?.t_diff_std ?? 0,
//               flow_mag_mean: item.features?.flow_mag_mean ?? 0,
//               flow_mag_std: item.features?.flow_mag_std ?? 0,
//               tproxy_val: item.features?.tproxy_val ?? 27.88,
//               tproxy_delta: item.features?.tproxy_delta ?? 0,
//               tproxy_vel: item.features?.tproxy_vel ?? 0,
//               CO: item.features?.CO ?? 0,
//               VOC: item.features?.VOC ?? 0,
//               NO2: item.features?.NO2 ?? 0,
//               CO_diff: item.features?.CO_diff ?? 0,
//               VOC_diff: item.features?.VOC_diff ?? 0,
//               NO2_diff: item.features?.NO2_diff ?? 0,
//               VOC_ma5: item.features?.VOC_ma5 ?? 0,
//               CO_ma5: item.features?.CO_ma5 ?? 0,
//               NO2_ma5: item.features?.NO2_ma5 ?? 0,
//               VOC_z: item.features?.VOC_z ?? 0,
//               CO_z: item.features?.CO_z ?? 0,
//               NO2_z: item.features?.NO2_z ?? 0,
//               temp_rise_c_per_min: item.features?.temp_rise_c_per_min ?? 0,
//               temp_slope_30s: item.features?.temp_slope_30s ?? 0,
//               gas_var_30s: item.features?.gas_var_30s ?? 0,
//               delta_temp_30s: item.features?.delta_temp_30s ?? 0,
//               delta_gas_10s: item.features?.delta_gas_10s ?? 0,
//               spike_count_voc_2m: item.features?.spike_count_voc_2m ?? 0,
//               is_weekend: item.features?.is_weekend ?? 0,
//               asleep_window: item.features?.asleep_window ?? 0,
//               hrblk_0: item.features?.hrblk_0 ?? 0,
//               hrblk_1: item.features?.hrblk_1 ?? 0,
//               hrblk_2: item.features?.hrblk_2 ?? 0,
//               hrblk_3: item.features?.hrblk_3 ?? 1,
//               hrblk_4: item.features?.hrblk_4 ?? 0,
//               hrblk_5: item.features?.hrblk_5 ?? 0,
//             },
//             decision_threshold: item.decision_threshold ?? 0.4,
//           }))
//         : [];

//       const mapped: Camera[] = Array.isArray(res.data)
//         ? res.data.map((item: any) => ({
//             id: `CAM-${item.device_id}`,
//             name: item.device_name || "Unknown Device",
//             location: item.device_name || "Unknown Location",
//             coordinates: [item.device_location[1], item.device_location[0]],
//             status: item.device_status
//               ? "no-fire"
//               : item.features?.t_max > item.decision_threshold
//               ? "predicted"
//               : "fire",
//             temperature: Math.round(item.features?.t_max || 25),
//             lastUpdated: new Date(item.timestamp).toLocaleTimeString(),
//           }))
//         : [];

//       setCameras(mapped);
//       const singleObj = transformedPayload[0] ?? null;
//       savePayload(singleObj);
//       transformedPayload.forEach((obj) => sendToAws(obj));
//     } catch (err) {
//       console.error("❌ Error fetching cameras:", err);
//     }
//   };

//   /* ----------  AWS call + NEW colour rule  ---------- */
//   const sendToAws = async (singleObj: any) => {
//   try {
//     const body = JSON.stringify(singleObj);
//     const res = await fetch(AWS_URL, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body,
//     });
//     const aws: any = await res.json();

//     /*  ➜  1.  hardware flag = OFF  →  BLACK  */
//     if (singleObj.device_status === false) {
//       setCameras((prev) =>
//         prev.map((cam) =>
//           cam.id === `CAM-${singleObj.device_id}`
//             ? { ...cam, status: "black", temperature: Math.round(singleObj.features.t_max) }
//             : cam
//         )
//       );
//       return; //  skip probability logic
//     }

//     /*  ➜  2.  normal probability rule  */
//     const prob = Number(aws?.fire_probability ?? -1);
//     const finalStatus = probToColour(prob);
//     setCameras((prev) =>
//       prev.map((cam) =>
//         cam.id === `CAM-${singleObj.device_id}`
//           ? { ...cam, status: finalStatus, temperature: Math.round(singleObj.features.t_max) }
//           : cam
//       )
//     );
//   } catch (e) {
//     console.error("AWS error for device", singleObj.device_id, e);
//   }
// };


//   /* ----------  map lifecycle  ---------- */
//   useEffect(() => {
//     fetchCameras();
//     const intv = setInterval(fetchCameras, 60000);
//     return () => clearInterval(intv);
//   }, []);

//   useEffect(() => {
//     if (!map.current) {
//       map.current = new mapboxgl.Map({
//         container: mapContainer.current!,
//         style: "mapbox://styles/mapbox/satellite-v9",
//         center: [0, 20],
//         zoom: 1.5,
//         projection: "globe",
//       });
//       map.current.addControl(new mapboxgl.NavigationControl(), "top-right");
//     }
//     if (map.current && cameras.length) setupClustering();
//   }, [cameras]);

//   const setupClustering = () => {
//     if (!map.current) return;
//     const geojson: GeoJSON.FeatureCollection = {
//       type: "FeatureCollection",
//       features: cameras.map((c) => ({
//         type: "Feature",
//         properties: {
//           id: c.id,
//           name: c.name,
//           location: c.location,
//           status: c.status,
//           temperature: c.temperature,
//           lastUpdated: c.lastUpdated,
//         },
//         geometry: { type: "Point", coordinates: [c.coordinates[0], c.coordinates[1]] },
//       })),
//     };

//     const src = map.current.getSource("cameras") as mapboxgl.GeoJSONSource;
//     if (src) {
//       src.setData(geojson);
//     } else {
//       map.current.addSource("cameras", { type: "geojson", data: geojson, cluster: true, clusterMaxZoom: 14, clusterRadius: 50 });
//       map.current.addLayer({
//         id: "clusters",
//         type: "circle",
//         source: "cameras",
//         filter: ["has", "point_count"],
//         paint: { "circle-color": "#3b82f6", "circle-radius": 20, "circle-stroke-width": 2, "circle-stroke-color": "#fff" },
//       });
//       map.current.addLayer({
//         id: "cluster-count",
//         type: "symbol",
//         source: "cameras",
//         filter: ["has", "point_count"],
//         layout: { "text-field": "{point_count_abbreviated}", "text-font": ["DIN Offc Pro Medium", "Arial Unicode MS Bold"], "text-size": 14 },
//         paint: { "text-color": "#fff" },
//       });
//       map.current.addLayer({
//         id: "unclustered-point",
//         type: "circle",
//         source: "cameras",
//         filter: ["!", ["has", "point_count"]],
//         paint: {
//           "circle-color": ["match", ["get", "status"], "fire", statusToHsl("fire"), "predicted", statusToHsl("predicted"), statusToHsl("no-fire")],
//           "circle-radius": 10,
//           "circle-stroke-width": 2,
//           "circle-stroke-color": "#fff",
//         },
//       });
//     }

//     const clickPin = (e: mapboxgl.MapLayerMouseEvent) => {
//       const feat = e.features![0];
//       const cam = cameras.find((c) => c.id === feat.properties!.id);
//       if (!cam) return;
//       onCameraSelect(cam);
//       map.current!.flyTo({ center: cam.coordinates as [number, number], zoom: 12 });
//     };
//     map.current.off("click", "unclustered-point", clickPin);
//     map.current.on("click", "unclustered-point", clickPin);

//     const handleClusterClick = (e: mapboxgl.MapLayerMouseEvent) => {
//       const feats = map.current!.queryRenderedFeatures(e.point, { layers: ["clusters"] });
//       const clustId = feats[0].properties!.cluster_id;
//       (map.current!.getSource("cameras") as mapboxgl.GeoJSONSource).getClusterExpansionZoom(clustId, (err, zoom) => {
//         if (err || zoom == null) return;
//         map.current!.easeTo({ center: (feats[0].geometry as GeoJSON.Point).coordinates as [number, number], zoom });
//       });
//     };
//     map.current.off("click", "clusters", handleClusterClick);
//     map.current.on("click", "clusters", handleClusterClick);
//   };

//   /* ----------  spin  ---------- */
//   const startSpinning = () => {
//     if (!map.current || !isSpinning) return;
//     const spin = () => {
//       if (!map.current || !isSpinning) return;
//       const center = map.current.getCenter();
//       center.lng += 0.1;
//       map.current.setCenter(center);
//       animationRef.current = requestAnimationFrame(spin);
//     };
//     animationRef.current = requestAnimationFrame(spin);
//   };
//   const stopSpinning = () => {
//     if (animationRef.current) cancelAnimationFrame(animationRef.current);
//   };
//   useEffect(() => {
//     isSpinning ? startSpinning() : stopSpinning();
//   }, [isSpinning]);

//   return (
//     <div className="h-full relative flex">
//       <div ref={mapContainer} className="flex-1" />
//       <div className="absolute left-4 top-4 z-20">
//         <Button onClick={() => setIsSpinning((s) => !s)} className="gap-2">
//           {isSpinning ? (
//             <>
//               <Pause className="h-4 w-4" /> Stop Spin
//             </>
//           ) : (
//             <>
//               <Play className="h-4 w-4" /> Start Spin
//             </>
//           )}
//         </Button>
//       </div>
//     </div>
//   );
// }


// // HeliosMap.tsx  (+  AWS last-call timer) FINAL WORKING VERSION
// import { useEffect, useRef, useState } from "react";
// import mapboxgl from "mapbox-gl";
// import "mapbox-gl/dist/mapbox-gl.css";
// import { Button } from "@/components/ui/button";
// import { Play, Pause } from "lucide-react";
// import { Client } from "@gradio/client";
// import { Camera } from "@/pages/Index";

// mapboxgl.accessToken =
//   "pk.eyJ1IjoiaGFybGVxdWluaXJlIiwiYSI6ImNtZGdpMWRrejBsYzcybHB6eXp0Z25pYzcifQ.Mwu0v3MGK-eo6mEsbYjVng";

// // const API_URL  = "https://08d6b4685d6057126e.gradio.live/";
// const API_URL  = "https://42afcf4e97e90fbef2.gradio.live";
// const AWS_URL  = "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws";

// interface HeliosMapProps {
//   onCameraSelect: (camera: Camera) => void;
// }

// export default function HeliosMap({ onCameraSelect }: HeliosMapProps) {
//   const mapContainer = useRef<HTMLDivElement>(null);
//   const map = useRef<mapboxgl.Map | null>(null);
//   const [isSpinning, setIsSpinning] = useState(false);
//   const [cameras, setCameras] = useState<Camera[]>([]);
//   const animationRef = useRef<number | null>(null);

//   const STORAGE_KEY = "helios-gradio-payload";
//   const savePayload = (obj: any) => localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));

//   /* ➜  AWS last-call timer  */
//   const [lastAwsCall, setLastAwsCall] = useState<string>("—");

//   /* ----------  NEW: probability → colour only  ---------- */
//   const probToColour = (p: number) => {
//     if (p >= 0.623939) return "fire";      // red
//     if (p <= 0.000028) return "no-fire";   // green
//     return "predicted";                    // yellow (in-between)
//   };
//   const statusToHsl = (status: string) => {
//     switch (status) {
//       case "fire":
//         return "hsl(15,90%,55%)";
//       case "predicted":
//         return "hsl(45,95%,55%)";
//     case "no-fire":
//       return "hsl(120,80%,45%)";
//       case "black":
//         return "hsl(0,0%,0%)";
//       default:
//         return "hsl(120,80%,45%)";
//     }
//   };

//   /* ----------  data fetching  ---------- */
//   const fetchCameras = async () => {
//     try {
//       const client = await Client.connect(API_URL);
//       const now = new Date();
//       const year = now.getUTCFullYear();
//       const month = String(now.getUTCMonth() + 1).padStart(2, "0");
//       const day = String(now.getUTCDate()).padStart(2, "0");
//       const hours = String(now.getUTCHours()).padStart(2, "0");
//       const minutes = String(now.getUTCMinutes()).padStart(2, "0");
//       const timestamp = `${year}-${month}-${day} ${hours}:${minutes}`;

//       const res = await client.predict("/predict", { timestamp_str: timestamp });

//       const transformedPayload = Array.isArray(res.data)
//         ? res.data.map((item: any) => ({
//             frame: 0,
//             target_minute: `dt=${year}-${month}-${day}/hour=${hours}/shane_${timestamp.replace(/:/g, "")}.json`,
//             timestamp: `${year}-${month}-${day} ${hours}:${minutes}:00+00:00`,
//             device_name: item.device_name || "shane",
//             device_id: item.device_id || "02",
//             device_location: item.device_location || [51.476782, -0.373907],
//             device_status: item.device_status ?? true,
//             features: {
//               t_mean: item.features?.t_mean ?? 26.82,
//               t_std: item.features?.t_std ?? 0.32,
//               t_max: item.features?.t_max ?? 27.88,
//               t_p95: item.features?.t_p95 ?? 27.42,
//               t_hot_area_pct: item.features?.t_hot_area_pct ?? 4.04,
//               t_hot_largest_blob_pct: item.features?.t_hot_largest_blob_pct ?? 2.21,
//               t_grad_mean: item.features?.t_grad_mean ?? 0.31,
//               t_grad_std: item.features?.t_grad_std ?? 0.19,
//               t_diff_mean: item.features?.t_diff_mean ?? 0,
//               t_diff_std: item.features?.t_diff_std ?? 0,
//               flow_mag_mean: item.features?.flow_mag_mean ?? 0,
//               flow_mag_std: item.features?.flow_mag_std ?? 0,
//               tproxy_val: item.features?.tproxy_val ?? 27.88,
//               tproxy_delta: item.features?.tproxy_delta ?? 0,
//               tproxy_vel: item.features?.tproxy_vel ?? 0,
//               CO: item.features?.CO ?? 0,
//               VOC: item.features?.VOC ?? 0,
//               NO2: item.features?.NO2 ?? 0,
//               CO_diff: item.features?.CO_diff ?? 0,
//               VOC_diff: item.features?.VOC_diff ?? 0,
//               NO2_diff: item.features?.NO2_diff ?? 0,
//               VOC_ma5: item.features?.VOC_ma5 ?? 0,
//               CO_ma5: item.features?.CO_ma5 ?? 0,
//               NO2_ma5: item.features?.NO2_ma5 ?? 0,
//               VOC_z: item.features?.VOC_z ?? 0,
//               CO_z: item.features?.CO_z ?? 0,
//               NO2_z: item.features?.NO2_z ?? 0,
//               temp_rise_c_per_min: item.features?.temp_rise_c_per_min ?? 0,
//               temp_slope_30s: item.features?.temp_slope_30s ?? 0,
//               gas_var_30s: item.features?.gas_var_30s ?? 0,
//               delta_temp_30s: item.features?.delta_temp_30s ?? 0,
//               delta_gas_10s: item.features?.delta_gas_10s ?? 0,
//               spike_count_voc_2m: item.features?.spike_count_voc_2m ?? 0,
//               is_weekend: item.features?.is_weekend ?? 0,
//               asleep_window: item.features?.asleep_window ?? 0,
//               hrblk_0: item.features?.hrblk_0 ?? 0,
//               hrblk_1: item.features?.hrblk_1 ?? 0,
//               hrblk_2: item.features?.hrblk_2 ?? 0,
//               hrblk_3: item.features?.hrblk_3 ?? 1,
//               hrblk_4: item.features?.hrblk_4 ?? 0,
//               hrblk_5: item.features?.hrblk_5 ?? 0,
//             },
//             decision_threshold: item.decision_threshold ?? 0.4,
//           }))
//         : [];

//       const mapped: Camera[] = Array.isArray(res.data)
//         ? res.data.map((item: any) => ({
//             id: `CAM-${item.device_id}`,
//             name: item.device_name || "Unknown Device",
//             location: item.device_name || "Unknown Location",
//             coordinates: [item.device_location[1], item.device_location[0]],
//             status: item.device_status
//               ? "no-fire"
//               : item.features?.t_max > item.decision_threshold
//               ? "predicted"
//               : "fire",
//             temperature: Math.round(item.features?.t_max || 25),
//             lastUpdated: new Date(item.timestamp).toLocaleTimeString(),
//           }))
//         : [];

//       setCameras(mapped);
//       const singleObj = transformedPayload[0] ?? null;
//       savePayload(singleObj);
//       transformedPayload.forEach((obj) => sendToAws(obj));
//     } catch (err) {
//       console.error("❌ Error fetching cameras:", err);
//     }
//   };

//   /* ----------  AWS call + NEW colour rule  ---------- */
//   const sendToAws = async (singleObj: any) => {
//     try {
//       const body = JSON.stringify(singleObj);
//       const res = await fetch(AWS_URL, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body,
//       });
//       const aws: any = await res.json();

//       /*  ➜  1.  hardware flag = OFF  →  BLACK  */
//       if (singleObj.device_status === false) {
//         setCameras((prev) =>
//           prev.map((cam) =>
//             cam.id === `CAM-${singleObj.device_id}`
//               ? { ...cam, status: "black", temperature: Math.round(singleObj.features.t_max) }
//               : cam
//           )
//         );
//         /* ➜  reset timer  */
//         setLastAwsCall(new Date().toLocaleTimeString());
//         return; //  skip probability logic
//       }

//       /*  ➜  2.  normal probability rule  */
//       const prob = Number(aws?.fire_probability ?? -1);
//       const finalStatus = probToColour(prob);
//       setCameras((prev) =>
//         prev.map((cam) =>
//           cam.id === `CAM-${singleObj.device_id}`
//             ? { ...cam, status: finalStatus, temperature: Math.round(singleObj.features.t_max) }
//             : cam
//         )
//       );
//       /* ➜  reset timer  */
//       setLastAwsCall(new Date().toLocaleTimeString());
//     } catch (e) {
//       console.error("AWS error for device", singleObj.device_id, e);
//     }
//   };

//   /* ----------  map lifecycle  ---------- */
//   useEffect(() => {
//     fetchCameras();
//     const intv = setInterval(fetchCameras, 60000);
//     return () => clearInterval(intv);
//   }, []);

//   useEffect(() => {
//     if (!map.current) {
//       map.current = new mapboxgl.Map({
//         container: mapContainer.current!,
//         style: "mapbox://styles/mapbox/satellite-v9",
//         center: [0, 20],
//         zoom: 1.5,
//         projection: "globe",
//       });
//       map.current.addControl(new mapboxgl.NavigationControl(), "top-right");
//     }
//     if (map.current && cameras.length) setupClustering();
//   }, [cameras]);

//   const setupClustering = () => {
//     if (!map.current) return;
//     const geojson: GeoJSON.FeatureCollection = {
//       type: "FeatureCollection",
//       features: cameras.map((c) => ({
//         type: "Feature",
//         properties: {
//           id: c.id,
//           name: c.name,
//           location: c.location,
//           status: c.status,
//           temperature: c.temperature,
//           lastUpdated: c.lastUpdated,
//         },
//         geometry: { type: "Point", coordinates: [c.coordinates[0], c.coordinates[1]] },
//       })),
//     };

//     const src = map.current.getSource("cameras") as mapboxgl.GeoJSONSource;
//     if (src) {
//       src.setData(geojson);
//     } else {
//       map.current.addSource("cameras", { type: "geojson", data: geojson, cluster: true, clusterMaxZoom: 14, clusterRadius: 50 });
//       map.current.addLayer({
//         id: "clusters",
//         type: "circle",
//         source: "cameras",
//         filter: ["has", "point_count"],
//         paint: { "circle-color": "#3b82f6", "circle-radius": 20, "circle-stroke-width": 2, "circle-stroke-color": "#fff" },
//       });
//       map.current.addLayer({
//         id: "cluster-count",
//         type: "symbol",
//         source: "cameras",
//         filter: ["has", "point_count"],
//         layout: { "text-field": "{point_count_abbreviated}", "text-font": ["DIN Offc Pro Medium", "Arial Unicode MS Bold"], "text-size": 14 },
//         paint: { "text-color": "#fff" },
//       });
//       map.current.addLayer({
//         id: "unclustered-point",
//         type: "circle",
//         source: "cameras",
//         filter: ["!", ["has", "point_count"]],
//         paint: {
//           "circle-color": ["match", ["get", "status"], "fire", statusToHsl("fire"), "predicted", statusToHsl("predicted"), statusToHsl("no-fire")],
//           "circle-radius": 10,
//           "circle-stroke-width": 2,
//           "circle-stroke-color": "#fff",
//         },
//       });
//     }

//     const clickPin = (e: mapboxgl.MapLayerMouseEvent) => {
//       const feat = e.features![0];
//       const cam = cameras.find((c) => c.id === feat.properties!.id);
//       if (!cam) return;
//       onCameraSelect(cam);
//       map.current!.flyTo({ center: cam.coordinates as [number, number], zoom: 12 });
//     };
//     map.current.off("click", "unclustered-point", clickPin);
//     map.current.on("click", "unclustered-point", clickPin);

//     const handleClusterClick = (e: mapboxgl.MapLayerMouseEvent) => {
//       const feats = map.current!.queryRenderedFeatures(e.point, { layers: ["clusters"] });
//       const clustId = feats[0].properties!.cluster_id;
//       (map.current!.getSource("cameras") as mapboxgl.GeoJSONSource).getClusterExpansionZoom(clustId, (err, zoom) => {
//         if (err || zoom == null) return;
//         map.current!.easeTo({ center: (feats[0].geometry as GeoJSON.Point).coordinates as [number, number], zoom });
//       });
//     };
//     map.current.off("click", "clusters", handleClusterClick);
//     map.current.on("click", "clusters", handleClusterClick);
//   };

//   /* ----------  spin  ---------- */
//   const startSpinning = () => {
//     if (!map.current || !isSpinning) return;
//     const spin = () => {
//       if (!map.current || !isSpinning) return;
//       const center = map.current.getCenter();
//       center.lng += 0.1;
//       map.current.setCenter(center);
//       animationRef.current = requestAnimationFrame(spin);
//     };
//     animationRef.current = requestAnimationFrame(spin);
//   };
//   const stopSpinning = () => {
//     if (animationRef.current) cancelAnimationFrame(animationRef.current);
//   };
//   useEffect(() => {
//     isSpinning ? startSpinning() : stopSpinning();
//   }, [isSpinning]);

//   return (
//     <div className="h-full relative flex">
//       <div ref={mapContainer} className="flex-1" />
//       <div className="absolute left-4 top-4 z-20">
//         <Button onClick={() => setIsSpinning((s) => !s)} className="gap-2">
//           {isSpinning ? (
//             <>
//               <Pause className="h-4 w-4" /> Stop Spin
//             </>
//           ) : (
//             <>
//               <Play className="h-4 w-4" /> Start Spin
//             </>
//           )}
//         </Button>
//         {/* ➜  AWS last-call timer  */}
//         <div className="mt-2 text-xs text-muted-foreground">
//           AWS last call: {lastAwsCall}
//         </div>
//       </div>
//     </div>
//   );
// }