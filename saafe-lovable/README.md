# SAAFE.ai — Lovable-Ready Global Command Center

This repository contains a **drop-in React codebase** designed for **Lovable** with zero external UI dependencies, plus **Grafana dashboard provisioning**.

## Features
- **Helios** (Global View 2D) — world map with risk markers
- **Grid** (Asset Manager) — search/filter, click to drill
- **Chronos** (Incident Panel) — mini-stats, sensors, history, actions
- **Athena** (Strategic Dashboard) — KPIs + tiny trend chart
- **SAAFEGPT** (LLM stub) — chat UI with simple API contract
- **Mock stream** — optional random updates for demo

## Run in Lovable
1. Create/import a React project in Lovable.
2. Copy `src/`, `public/` into your project (or drop this repo).
3. Render the app by importing the default export:
   ```tsx
   import SaafeLovable from "./src/components/SaafeLovable";
   export default function App(){ return <SaafeLovable/> }
   ```
4. Replace `src/data/sampleHomes.json` with your live feed or wire the WebSocket in `SaafeLovable.tsx`.

## API Contracts
### Realtime PATCH
```json
{ "type":"PATCH", "id":"SAAFE-0003", "changes": { "level": 9, "score": 92, "updated": "2025-09-10T18:05:14Z" } }
```
Apply per-device patches; batch per animation frame.

### SAAFEGPT
- **POST** `/api/saafegpt`
- Request:
```json
{ "messages": [ {"role":"system","content":"You are SAAFEGPT..."}, {"role":"user","content":"Summarize incident SAAFE-0003"} ] }
```
- Response:
```json
{ "message": { "role":"assistant", "content":"At 11:32..." } }
```

## Grafana
Provisioned dashboard JSON at `monitoring/grafana/provisioning/dashboards/saafe_overview.json`.
Import or auto-provision via Grafana provisioning. Assumes a datasource named `saafe_ts` (configure per your stack).

## License
Copyright © SAAFE.ai
