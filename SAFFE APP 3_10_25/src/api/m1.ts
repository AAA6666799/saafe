/// <reference types="vite/client" />

export async function callM1(payload: any) {
  const res = await fetch(import.meta.env.VITE_M1_URL!, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
}
