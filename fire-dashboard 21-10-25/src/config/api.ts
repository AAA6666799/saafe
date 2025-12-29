// API Configuration
// This will use the deployed API URL in production and localhost in development

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
  (import.meta.env.PROD 
    ? 'https://fire-dashboard-xi.vercel.app' 
    : 'http://localhost:8080');

console.log('API_BASE_URL:', API_BASE_URL);