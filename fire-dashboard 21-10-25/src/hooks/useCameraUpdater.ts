// src/hooks/useCameraUpdater.ts
import { useEffect } from "react";
import { Client } from "@gradio/client";

const API_URL = "https://42afcf4e97e90fbef2.gradio.live";
const AWS_URL = "https://b6vmdcuw7b.execute-api.us-east-1.amazonaws.com/predict";

export const useCameraUpdater = () => {
  const sendToAws = async (singleObj: any) => {
    try {
      const res = await fetch(AWS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(singleObj),
      });
      const aws: any = await res.json();
      console.log("AWS updated for device:", singleObj.device_id, aws);
    } catch (err) {
      console.error("AWS error for device", singleObj.device_id, err);
    }
  };

  const fetchCameras = async () => {
    try {
      const client = await Client.connect(API_URL);
      const now = new Date();
      const timestamp = now.toISOString();

      const res = await client.predict("/predict", { timestamp_str: timestamp });
      const transformed = Array.isArray(res.data)
        ? res.data.map((item: any) => ({
            ...item,
            timestamp,
          }))
        : [];

      transformed.forEach((obj) => sendToAws(obj));
    } catch (err) {
      console.error("Error fetching cameras:", err);
    }
  };

  useEffect(() => {
    fetchCameras(); // immediate call
    const interval = setInterval(fetchCameras, 30000); // repeat every 30s
    return () => clearInterval(interval);
  }, []);
};
