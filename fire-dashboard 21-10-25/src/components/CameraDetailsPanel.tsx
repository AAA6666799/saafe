// // CameraDetailsPanel.tsx
// import { useEffect, useState } from "react";
// import { X, MapPin, Thermometer, Clock, Activity, ChevronDown, ChevronUp } from "lucide-react";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { Button } from "@/components/ui/button";
// import { Badge } from "@/components/ui/badge";
// import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
// import { Camera } from "@/pages/Index";

// const AWS_SAAFE  = "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws";
// const AWS_TENSOR = "https://b6vmdcuw7b.execute-api.us-east-1.amazonaws.com/predict";
// const STORAGE_KEY = "helios-gradio-payload";

// interface CameraDetailsPanelProps {
//   camera: Camera | null;
//   onClose: () => void;
// }

// export default function CameraDetailsPanel({ camera, onClose }: CameraDetailsPanelProps) {
//   const [saafeData, setSaafeData] = useState<any>(null);
//   const [tensorData, setTensorData] = useState<any>(null);
//   const [lastUpdate, setLastUpdate] = useState<string>("");
//   const [expandSaafe, setExpandSaafe] = useState(false);
//   const [expandTensor, setExpandTensor] = useState(false);
//   // 🆕 Timer state
//   const [timer, setTimer] = useState<number>(60);
  
//   /* shared function to call both AWS endpoints */
//   const callBoth = async (payloadArg?: any) => {
//     const payload = payloadArg ?? (() => {
//       try {
//         return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
//       } catch {
//         return null;
//       }
//     })();
//     if (!payload) return;

//     try {
//       const body = JSON.stringify(payload);
//       const [resS, resT] = await Promise.all([
//         fetch(AWS_SAAFE,  { method: "POST", headers: { "Content-Type": "application/json" }, body }),
//         fetch(AWS_TENSOR, { method: "POST", headers: { "Content-Type": "application/json" }, body }),
//       ]);
//       const [dataS, dataT] = await Promise.all([resS.json(), resT.json()]);

//       setSaafeData(dataS);
//       setTensorData(dataT);
//       setLastUpdate(new Date().toLocaleTimeString());

//       // 🆕 Reset timer whenever API hits
//       setTimer(60);
//     } catch (err) {
//       console.error("callBoth error:", err);
//     }
//   };

  
//   /*  ➜  auto-refresh every 60 s  */
// useEffect(() => {
//   if (!camera) return;
//   const id = setInterval(() => {
//     const payload = (() => {
//       try {
//         return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
//       } catch {
//         return null;
//       }
//     })();
//     if (payload) callBoth(payload); // re-use existing function
//   }, 60000);
//   return () => clearInterval(id); // cleanup on unmount
// }, [camera]);
  

//   /* ----------  read payload from localStorage & call both AWS  ---------- */
//   useEffect(() => {
//     if (!camera) return;
//     const payload = (() => {
//       try {
//         return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
//       } catch {
//         return null;
//       }
//     })();
//     if (!payload) return;

//     callBoth(payload);
//   }, [camera]);

//   // 🆕 Countdown effect
//   useEffect(() => {
//     const interval = setInterval(() => {
//       setTimer(prev => (prev > 0 ? prev - 1 : 0));
//     }, 1000);
//     return () => clearInterval(interval);
//   }, []);


//   if (!camera) return null;

//   const tempHistory = [
//     { time: "14:55", temp: camera.temperature },
//     { time: "14:50", temp: camera.temperature - 2 },
//     { time: "14:45", temp: camera.temperature - 5 },
//     { time: "14:40", temp: camera.temperature - 3 },
//     { time: "14:35", temp: camera.temperature - 7 },
//   ];

//   const getStatusColor = (status: Camera["status"]) => {
//     return status === "fire" ? "text-primary" : status === "predicted" ? "text-warning" : "text-safe";
//   };

  
//   /* ----------  reusable render for one AWS card  ---------- */
//   const AwsCard = ({
//     title,
//     data,
//     expand,
//     setExpand,
//   }: {
//     title: string;
//     data: any;
//     expand: boolean;
//     setExpand: (v: boolean) => void;
//   }) => (
//     <Card>
//       <CardHeader>
//         <CardTitle className="text-sm">{title}</CardTitle>
//         <p className="text-xs text-muted-foreground">Last AWS call: {lastUpdate || "—"}</p>
//       </CardHeader>
//       <CardContent className="space-y-2">
//         {!data ? (
//           <p className="text-sm text-muted-foreground">Calling AWS …</p>
//         ) : (
//           <>
//             <Table>
//               <TableBody>
//                 <TableRow>
//                   <TableCell className="font-medium">fire_probability</TableCell>
//                   <TableCell className="text-right">
//   {data?.prediction?.fire_probability != null
//     ? data.prediction.fire_probability.toFixed(6)
//     : "—"}
// </TableCell>
//                 </TableRow>
//                 <TableRow>
//                   <TableCell className="font-medium">label</TableCell>
//                   <TableCell className="text-right">{data.prediction?.label ?? "—"}</TableCell>
//                 </TableRow>
//               </TableBody>
//             </Table>

//             {/* Explanation accordion */}
//             <Button
//               variant="ghost"
//               size="sm"
//               className="mt-2"
//               onClick={() => setExpand(!expand)}
//             >
//               {expand ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
//               <span className="ml-1 text-xs">Explanation</span>
//             </Button>
//             {expand && (
//               <div className="text-xs text-muted-foreground max-h-40 overflow-y-auto border rounded p-2">
//                 <p>{data.explanation?.notes ?? "No explanation available."}</p>
//                 <p className="mt-2 font-semibold">Top global features:</p>
//                 <ul className="list-disc ml-4">
//                   {data.explanation?.global_top_features?.map((f: any, i: number) => (
//                     <li key={i}>{f.feature} (imp. {f.importance.toFixed(2)})</li>
//                   ))}
//                 </ul>
//               </div>
//             )}
//           </>
//         )}
//       </CardContent>
//     </Card>
//   );

//   return (
//     <div className="fixed inset-y-0 right-0 w-96 bg-card border-l border-border shadow-2xl z-50 overflow-auto animate-in slide-in-from-right">
//       {/* header */}
//       <div className="sticky top-0 bg-card border-b border-border p-4 flex items-center justify-between">
//         <h2 className="font-bold text-lg">Camera Details</h2>
//         <Button variant="ghost" size="icon" onClick={onClose}>
//           <X className="h-4 w-4" />
//         </Button>
//       </div>

//       {/* 🆕 Timer Display */}
//       <div className="bg-muted text-center py-2 text-sm font-medium">
//         Refreshing in: {timer}s
//       </div>

//       <div className="p-4 space-y-4">
//         {/* camera info */}
//         <div>
//           <h3 className="text-xl font-bold mb-1">{camera.name}</h3>
//           <p className="text-sm text-muted-foreground">{camera.id}</p>
//         </div>

//         {/* status badge */}
//         <Badge
//           className={
//             camera.status === "fire"
//               ? "bg-primary text-primary-foreground shadow-glow-fire"
//               : camera.status === "predicted"
//               ? "bg-warning text-warning-foreground shadow-glow-warning"
//               : "bg-safe text-safe-foreground shadow-glow-safe"
//           }
//         >
//           {camera.status === "no-fire" ? "Safe" : camera.status.charAt(0).toUpperCase() + camera.status.slice(1)}
//         </Badge>

//         {/* location */}
//         <Card>
//           <CardHeader><CardTitle className="text-sm">Location Information</CardTitle></CardHeader>
//           <CardContent className="space-y-3">
//             <div className="flex items-start gap-2">
//               <MapPin className="h-4 w-4 mt-0.5 text-muted-foreground" />
//               <div>
//                 <p className="text-sm font-medium">{camera.location}</p>
//                 <p className="text-xs text-muted-foreground font-mono">
//                   {camera.coordinates[0].toFixed(6)}, {camera.coordinates[1].toFixed(6)}
//                 </p>
//               </div>
//             </div>
//           </CardContent>
//         </Card>

//         {/* current status */}
//         <Card>
//           <CardHeader><CardTitle className="text-sm">Current Status</CardTitle></CardHeader>
//           <CardContent className="space-y-3">
//             <div className="flex items-center justify-between">
//               <div className="flex items-center gap-2"><Thermometer className="h-4 w-4 text-muted-foreground" /><span className="text-sm">Temperature</span></div>
//               <span className={`text-lg font-bold ${getStatusColor(camera.status)}`}>{camera.temperature}°C</span>
//             </div>
//             <div className="flex items-center justify-between">
//               <div className="flex items-center gap-2"><Clock className="h-4 w-4 text-muted-foreground" /><span className="text-sm">Last Updated</span></div>
//               <span className="text-sm font-medium">{lastUpdate || "—"}</span>
//             </div>
//             <div className="flex items-center justify-between">
//               <div className="flex items-center gap-2"><Activity className="h-4 w-4 text-muted-foreground" /><span className="text-sm">Detection Rate</span></div>
//               <span className="text-sm font-medium">2.5 Hz</span>
//             </div>
//           </CardContent>
//         </Card>

//         {/* temperature history */}
//         <Card>
//           <CardHeader><CardTitle className="text-sm">Temperature History</CardTitle></CardHeader>
//           <CardContent>
//             <Table>
//               <TableHeader>
//                 <TableRow>
//                   <TableHead>Time</TableHead>
//                   <TableHead className="text-right">Temp (°C)</TableHead>
//                 </TableRow>
//               </TableHeader>
//               <TableBody>
//                 {tempHistory.map((entry, index) => (
//                   <TableRow key={index}>
//                     <TableCell className="font-medium">{entry.time}</TableCell>
//                     <TableCell className="text-right">{entry.temp}°C</TableCell>
//                   </TableRow>
//                 ))}
//               </TableBody>
//             </Table>
//           </CardContent>
//         </Card>

//         {/* AWS PREDICTIONS – two separate cards */}
//         <AwsCard title="SAAFE Prediction" data={saafeData} expand={expandSaafe} setExpand={setExpandSaafe} />
//         <AwsCard title="Tensor Prediction" data={tensorData} expand={expandTensor} setExpand={setExpandTensor} />

//         {/* actions */}
//         <Card>
//           <CardHeader><CardTitle className="text-sm">Actions</CardTitle></CardHeader>
//           <CardContent className="space-y-2">
//             <Button className="w-full" variant="outline">View Full History</Button>
//             <Button className="w-full" variant="outline">Download Report</Button>
//             <Button className="w-full" variant="outline">Configure Alerts</Button>
//           </CardContent>
//         </Card>
//       </div>
//     </div>
//   );
// }


// CameraDetailsPanel.tsx
import { useEffect, useState } from "react";
import {
  X,
  MapPin,
  Thermometer,
  Clock,
  Activity,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Camera } from "@/pages/Index";

// const AWS_SAAFE = "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws";
const AWS_SAAFE = "https://b6vmdcuw7b.execute-api.us-east-1.amazonaws.com/predict";
const STORAGE_KEY = "helios-gradio-payload";

interface CameraDetailsPanelProps {
  camera: Camera | null;
  onClose: () => void;
}

export default function CameraDetailsPanel({ camera, onClose }: CameraDetailsPanelProps) {
  const [saafeData, setSaafeData] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<string>("");
  const [expandDetails, setExpandDetails] = useState(false);
  const [timer, setTimer] = useState<number>(60);

  // ✅ Color logic (includes black for device_status = false)
  const getStatusColor = (status: Camera["status"]) => {
    if (status === "fire") return "text-primary";
    if (status === "predicted") return "text-warning";
    if (status === "black") return "text-gray-700";
    return "text-safe";
  };

  // ✅ Fetch only SAAFE API using stored payload
  const callSaafe = async (payloadArg?: any) => {
    const payload =
      payloadArg ??
      (() => {
        try {
          return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        } catch {
          return null;
        }
      })();
    if (!payload) return;

    try {
      const res = await fetch(AWS_SAAFE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setSaafeData(data);
      setLastUpdate(payload.timestamp || "—");
      setTimer(30); // reset refresh timer
    } catch (err) {
      console.error("SAAFE fetch error:", err);
    }
  };

  // 🔁 Auto-refresh every 60 seconds
  useEffect(() => {
    if (!camera) return;
    const interval = setInterval(() => {
      const payload = (() => {
        try {
          return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        } catch {
          return null;
        }
      })();
      if (payload) callSaafe(payload);
    }, 30000);
    return () => clearInterval(interval);
  }, [camera]);

  // ▶️ Initial fetch when camera opens
  useEffect(() => {
    if (!camera) return;
    const payload = (() => {
      try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      } catch {
        return null;
      }
    })();
    if (payload) callSaafe(payload);
  }, [camera]);

  // ⏱ Countdown
  useEffect(() => {
    const interval = setInterval(() => {
      setTimer((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  if (!camera) return null;

  const tempHistory = [
    { time: "14:55", temp: camera.temperature },
    { time: "14:50", temp: camera.temperature - 2 },
    { time: "14:45", temp: camera.temperature - 5 },
    { time: "14:40", temp: camera.temperature - 3 },
    { time: "14:35", temp: camera.temperature - 7 },
  ];

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-card border-l border-border shadow-2xl z-50 overflow-auto animate-in slide-in-from-right">
      {/* Header */}
      <div className="sticky top-0 bg-card border-b border-border p-4 flex items-center justify-between">
        <h2 className="font-bold text-lg">Camera Details</h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Timer */}
      <div className="bg-muted text-center py-2 text-sm font-medium">
        Refreshing in: {timer}s
      </div>

      <div className="p-4 space-y-4">
        {/* Basic Info */}
        <div>
          <h3 className="text-xl font-bold mb-1">{camera.name}</h3>
          <p className="text-sm text-muted-foreground">{camera.id}</p>
        </div>

        {/* Status Badge */}
        <Badge
          className={
            camera.status === "fire"
              ? "bg-primary text-primary-foreground shadow-glow-fire"
              : camera.status === "predicted"
              ? "bg-warning text-warning-foreground shadow-glow-warning"
              : camera.status === "black"
              ? "bg-gray-700 text-white"
              : "bg-safe text-safe-foreground shadow-glow-safe"
          }
        >
          {camera.status === "no-fire"
            ? "Safe"
            : camera.status === "black"
            ? "Offline"
            : camera.status.charAt(0).toUpperCase() + camera.status.slice(1)}
        </Badge>

        {/* Location */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Location Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-start gap-2">
              <MapPin className="h-4 w-4 mt-0.5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">{camera.location}</p>
                <p className="text-xs text-muted-foreground font-mono">
                  {camera.coordinates[0].toFixed(6)}, {camera.coordinates[1].toFixed(6)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Status Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Current Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Thermometer className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">Temperature</span>
              </div>
              <span className={`text-lg font-bold ${getStatusColor(camera.status)}`}>
                {camera.temperature}°C
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">Last Updated</span>
              </div>
              <span className="text-sm font-medium">{lastUpdate || "—"}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">Detection Rate</span>
              </div>
              <span className="text-sm font-medium">2.5 Hz</span>
            </div>
          </CardContent>
        </Card>

        {/* AWS SAAFE Prediction */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">TENSOR Prediction</CardTitle>
            <p className="text-xs text-muted-foreground">
              Last AWS call: {lastUpdate || "—"}
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {!saafeData ? (
              <p className="text-sm text-muted-foreground">Calling AWS …</p>
            ) : (
              <>
                <Table>
                  <TableBody>
                    <TableRow>
                      <TableCell className="font-medium">Fire Probability</TableCell>
                      <TableCell className="text-right">
                        {saafeData?.prediction?.fire_probability?.toFixed(6) ?? "—"}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Label</TableCell>
                      <TableCell className="text-right">
                        {saafeData?.prediction?.label ?? "—"}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>

                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-2"
                  onClick={() => setExpandDetails(!expandDetails)}
                >
                  {expandDetails ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                  <span className="ml-1 text-xs">Full Explanation</span>
                </Button>

                {expandDetails && (
                  <div className="text-xs text-muted-foreground max-h-60 overflow-y-auto border rounded p-2 space-y-2">
                    <p className="font-semibold">Notes:</p>
                    <p>{saafeData.explanation?.notes ?? "No notes available."}</p>

                    <p className="mt-2 font-semibold">Top Global Features:</p>
                    <ul className="list-disc ml-4">
                      {saafeData.explanation?.global_top_features?.map((f: any, i: number) => (
                        <li key={i}>
                          {f.feature} (importance: {f.importance.toFixed(2)})
                        </li>
                      ))}
                    </ul>

                    <p className="mt-2 font-semibold">Local Contributions:</p>
                    <ul className="list-disc ml-4">
                      {saafeData.explanation?.local_contributions?.map((f: any, i: number) => (
                        <li key={i}>
                          {f.feature}: value {f.value}, contrib {f.contribution.toFixed(3)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
