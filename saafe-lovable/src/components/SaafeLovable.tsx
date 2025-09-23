import React, { useMemo, useState, useEffect, useRef } from "react";
import homesData from "../data/sampleHomes.json"
import { askSaafeGPT } from "../api/saafegpt"
import FireDetectionDashboard from "./FireDetectionDashboard"

// Utility
const band = (level:number) => level>=9?"critical":level>=7?"high":level>=5?"elev":level>=3?"guard":"low";
const bandColor = (level:number) => ({ low:"#34d399", guard:"#a3e635", elev:"#f59e0b", high:"#f97316", critical:"#ef4444" }[band(level) as keyof any]);

function ll2xy(lat:number, lon:number, w=900, h=450){ const x=(lon+180)*(w/360); const y=(90-lat)*(h/180); return [x,y]; }

// Helios 2D
function Helios2D({ homes }:{ homes:any[] }){
  const width = 900, height = 450;
  const MAP_URL = "https://upload.wikimedia.org/wikipedia/commons/8/80/Blue_Marble_2002.png";
  return (
    <div style={{ border:"1px solid #e5e7eb", borderRadius:16, overflow:"hidden", background:"#0f172a" }}>
      <div style={{ padding:12, color:"#0f172a", background:"white" }}><strong>Helios — Global View</strong></div>
      <div style={{ position:"relative", width, height, background:"#0b1020" }}>
        <img src={MAP_URL} alt="world" width={width} height={height} style={{ display:"block", filter:"saturate(0.9) brightness(0.95)" }}/>
        {homes.map((h:any)=>(
          <div key={h.id} title={`${h.city}, ${h.country} — L${h.level}`}
               style={{ position:"absolute", left:ll2xy(h.lat,h.lon,width,height)[0]-4, top:ll2xy(h.lat,h.lon,width,height)[1]-4, width:8,height:8, background:bandColor(h.level), borderRadius:9999, boxShadow:`0 0 10px ${bandColor(h.level)}`}}/>
        ))}
        <div style={{ position:"absolute", left:12, bottom:12, display:"flex", gap:10, padding:8, background:"rgba(255,255,255,0.9)", borderRadius:12, fontSize:12 }}>
          {[["#34d399","L1–2 Low"],["#a3e635","L3–4 Guarded"],["#f59e0b","L5–6 Elevated"],["#f97316","L7–8 High"],["#ef4444","L9–10 Critical"]].map(([c,l])=>(
            <div key={String(l)} style={{ display:"flex", alignItems:"center", gap:6 }}>
              <span style={{ width:10, height:10, background:String(c), borderRadius:9999 }} /><span>{l}</span>
            </div>
          ))}
        </div>
      </div>
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
  const homes:any[] = (homesData as any);
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
        <Helios2D homes={homes} />
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
