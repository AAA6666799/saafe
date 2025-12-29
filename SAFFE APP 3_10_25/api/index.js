const AWS = require('aws-sdk');
const { kv } = require('@vercel/kv');

let nodemailer;
try {
  nodemailer = require('nodemailer');
} catch (error) {
  console.warn('Nodemailer not installed. Email functionality will be disabled.');
  nodemailer = null;
}

// Configure AWS SDK
AWS.config.update({ region: process.env.AWS_REGION || 'us-east-1' });
const s3 = new AWS.S3();

// Model endpoints
const MODEL_URLS = {
  saafe: "https://bjggbotpq6aglni3wd3qe5luf40wszod.lambda-url.us-east-1.on.aws/",
  features18: "https://rnmsxp5s53.us-east-1.awsapprunner.com/predict_features",
  kaggle: "https://mfyemzf28h.us-east-1.awsapprunner.com/predict",
  tensorflow: "https://b6vmdcuw7b.execute-api.us-east-1.amazonaws.com/predict"
};

// Email configuration
let EMAIL_CONFIG = {
  sender_email: process.env.SENDER_EMAIL || "ch.ajay1707@gmail.com",
  sender_password: process.env.SENDER_PASSWORD || "oznfunikrcfutxxn",
  recipients: [
    {
      email: process.env.RECIPIENT_EMAIL || "ch.ajay1707@gmail.com",
      name: "Primary Admin",
      alertLevels: ["all"],
      enabled: true
    }
  ]
};

// Track last email sent
let lastEmailSent = {
  timestamp: null,
  riskScore: 0,
  level: 0
};

const EMAIL_COOLDOWN = 5 * 60 * 1000;

// Vercel KV keys
const ALERT_STATE_KEY = 'saafe:alert-state';
const ALERT_HISTORY_KEY = 'saafe:alert-history';
const MAX_HISTORY_SIZE = 50;

// Default alert state
const DEFAULT_ALERT_STATE = {
  isActive: false,
  level: 1,
  message: "System operating normally",
  timestamp: new Date().toISOString(),
  riskScore: 0,
  confidence: 0.9
};

// Helper functions for KV storage
async function getAlertState() {
  try {
    const state = await kv.get(ALERT_STATE_KEY);
    return state || DEFAULT_ALERT_STATE;
  } catch (error) {
    console.error('Error getting alert state from KV:', error);
    return DEFAULT_ALERT_STATE;
  }
}

async function setAlertState(state) {
  try {
    await kv.set(ALERT_STATE_KEY, state);
    return true;
  } catch (error) {
    console.error('Error setting alert state in KV:', error);
    return false;
  }
}

async function getAlertHistory() {
  try {
    const history = await kv.get(ALERT_HISTORY_KEY);
    return history || [];
  } catch (error) {
    console.error('Error getting alert history from KV:', error);
    return [];
  }
}

async function setAlertHistory(history) {
  try {
    await kv.set(ALERT_HISTORY_KEY, history);
    return true;
  } catch (error) {
    console.error('Error setting alert history in KV:', error);
    return false;
  }
}

// Email transporter
const emailTransporter = nodemailer ? nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: EMAIL_CONFIG.sender_email,
    pass: EMAIL_CONFIG.sender_password
  }
}) : null;

// Helper functions (imported from backend/server.js)
async function addToAlertHistory(eventType, riskScore, level, message, confidence) {
  const event = {
    id: Date.now() + Math.random(),
    timestamp: new Date().toISOString(),
    eventType,
    riskScore,
    level,
    message,
    confidence
  };
  
  const history = await getAlertHistory();
  history.unshift(event);
  
  if (history.length > MAX_HISTORY_SIZE) {
    history.splice(MAX_HISTORY_SIZE);
  }
  
  await setAlertHistory(history);
}

async function fetchDeviceDataFromS3() {
  try {
    const bucketName = 'data-collector-of-first-device';
    const deviceFileKey = 'devices.json';

    const deviceObject = await s3.getObject({
      Bucket: bucketName,
      Key: deviceFileKey
    }).promise();

    const deviceContent = deviceObject.Body.toString('utf-8');
    const devices = JSON.parse(deviceContent);

    return {
      status: "success",
      data: devices,
      count: devices.length,
      source: "AWS S3",
      bucket: bucketName,
      key: deviceFileKey,
      lastModified: deviceObject.LastModified,
      fetchedAt: new Date().toISOString()
    };
  } catch (error) {
    return {
      status: "error",
      message: "Failed to fetch device data from AWS S3",
      error: error.message,
      bucket: 'data-collector-of-first-device',
      key: 'devices.json',
      attemptedAt: new Date().toISOString()
    };
  }
}

// Main serverless handler
module.exports = async (req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { url, method } = req;
  const path = url.replace('/api', '');

  try {
    // Health check
    if (path === '/health' && method === 'GET') {
      return res.status(200).send('ok');
    }

    // Fire detection data
    if (path === '/fire-detection-data' && method === 'GET') {
      // This would need the full implementation from backend/server.js
      return res.status(200).json({
        status: "success",
        message: "Fire detection data endpoint - implementation needed"
      });
    }

    // Devices endpoints
    if (path === '/devices' && method === 'GET') {
      const deviceResult = await fetchDeviceDataFromS3();
      if (deviceResult.status === "error") {
        return res.status(503).json(deviceResult);
      }
      return res.status(200).json(deviceResult);
    }

    // Alert state endpoints
    if (path === '/alert-state' && method === 'GET') {
      const alertState = await getAlertState();
      return res.status(200).json({
        status: "success",
        data: alertState
      });
    }

    if (path === '/alert-state' && method === 'POST') {
      const { isActive, level, message, riskScore, confidence } = req.body;
      
      if (typeof isActive !== 'boolean') {
        return res.status(400).json({
          status: "error",
          message: "isActive field is required and must be a boolean"
        });
      }

      const currentAlertState = await getAlertState();
      const previousRiskScore = currentAlertState.riskScore;
      const previousIsActive = currentAlertState.isActive;

      const newAlertState = {
        isActive,
        level: level !== undefined ? level : currentAlertState.level,
        message: message || currentAlertState.message,
        riskScore: riskScore !== undefined ? riskScore : currentAlertState.riskScore,
        confidence: confidence !== undefined ? confidence : currentAlertState.confidence,
        timestamp: new Date().toISOString()
      };

      await setAlertState(newAlertState);

      let eventType = 'Normal';
      if (riskScore >= 80) {
        eventType = 'Fire Detected';
      } else if (riskScore >= 40) {
        eventType = 'Fire Predicted';
      }
      
      if (previousRiskScore !== riskScore || previousIsActive !== isActive) {
        await addToAlertHistory(eventType, riskScore, level, message, confidence);
      }

      return res.status(200).json({
        status: "success",
        message: "Alert state updated successfully",
        data: newAlertState
      });
    }

    // Alert history
    if (path === '/alert-history' && method === 'GET') {
      const limit = parseInt(req.query.limit) || 10;
      const alertHistory = await getAlertHistory();
      const limitedHistory = alertHistory.slice(0, Math.min(limit, alertHistory.length));
      
      return res.status(200).json({
        status: "success",
        data: limitedHistory,
        total: alertHistory.length
      });
    }

    // Email recipients endpoints
    if (path === '/email-recipients' && method === 'GET') {
      return res.status(200).json({
        status: "success",
        data: EMAIL_CONFIG.recipients,
        count: EMAIL_CONFIG.recipients.length
      });
    }

    // Predict endpoint - forward to Saafe model
    if (path === '/predict/saafe' && method === 'POST') {
      try {
        const response = await fetch(MODEL_URLS.saafe, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(req.body)
        });

        const data = await response.json();
        
        return res.status(response.status).json({
          status: "success",
          message: "Prediction completed",
          data: data,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Prediction error:', error);
        return res.status(500).json({
          status: "error",
          message: "Failed to get prediction from model",
          error: error.message
        });
      }
    }

    // Default 404
    return res.status(404).json({
      status: "error",
      message: "Endpoint not found",
      path: path,
      method: method
    });

  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({
      status: "error",
      message: "Internal server error",
      error: error.message
    });
  }
};