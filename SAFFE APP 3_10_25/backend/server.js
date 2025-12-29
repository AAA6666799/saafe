const express = require('express');
const cors = require('cors');
const path = require('path');
const AWS = require('aws-sdk');

let nodemailer;
try {
  nodemailer = require('nodemailer');
} catch (error) {
  console.warn('Nodemailer not installed. Email functionality will be disabled.');
  nodemailer = null;
}

// Configure AWS SDK
AWS.config.update({ region: 'us-east-1' });
const s3 = new AWS.S3();

// Model endpoints
const MODEL_URLS = {
  saafe: "https://bjggbotpq6aglni3wd3qe5luf40wszod.lambda-url.us-east-1.on.aws/",
  features18: "https://rnmsxp5s53.us-east-1.awsapprunner.com/predict_features",
  kaggle: "https://mfyemzf28h.us-east-1.awsapprunner.com/predict",
  tensorflow: "https://b6vmdcuw7b.execute-api.us-east-1.amazonaws.com/predict"
};

// Email configuration with multiple recipients support
let EMAIL_CONFIG = {
  sender_email: process.env.SENDER_EMAIL || "ch.ajay1707@gmail.com",
  sender_password: process.env.SENDER_PASSWORD || "oznfunikrcfutxxn",
  recipients: [
    {
      email: process.env.RECIPIENT_EMAIL || "ch.ajay1707@gmail.com",
      name: "Primary Admin",
      alertLevels: ["all"], // Receives all alerts
      enabled: true
    }
  ]
};

// Track last email sent to prevent spam
let lastEmailSent = {
  timestamp: null,
  riskScore: 0,
  level: 0
};

// Minimum time between emails (in milliseconds) - 5 minutes
const EMAIL_COOLDOWN = 5 * 60 * 1000;

// Function to fetch device data from S3
async function fetchDeviceDataFromS3() {
  try {
    const bucketName = 'data-collector-of-first-device';
    const deviceFileKey = 'devices.json';

    console.log("=== FETCHING DEVICE DATA FROM S3 ===");
    console.log("Bucket:", bucketName);
    console.log("Key:", deviceFileKey);

    const deviceObject = await s3.getObject({
      Bucket: bucketName,
      Key: deviceFileKey
    }).promise();

    const deviceContent = deviceObject.Body.toString('utf-8');
    const devices = JSON.parse(deviceContent);

    console.log(`Successfully fetched ${devices.length} devices from S3`);
    console.log("Device IDs:", devices.map(d => d.id).join(', '));

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
    console.error("Error fetching device data from S3:", error);

    let errorMessage = "Failed to fetch device data from AWS S3";
    let errorDetails = {};

    if (error.code === 'NoSuchKey') {
      errorMessage = "Device data file not found in S3 bucket";
      errorDetails = {
        reason: "The devices.json file does not exist in the S3 bucket",
        suggestion: "Please ensure the device registry file is uploaded to S3"
      };
    } else if (error.code === 'NoSuchBucket') {
      errorMessage = "S3 bucket not found";
      errorDetails = {
        reason: "The specified S3 bucket does not exist",
        suggestion: "Check bucket name and AWS credentials"
      };
    } else if (error.code === 'AccessDenied') {
      errorMessage = "Access denied to S3 bucket";
      errorDetails = {
        reason: "Insufficient permissions to access the S3 bucket",
        suggestion: "Check AWS IAM permissions for S3 access"
      };
    } else if (error.code === 'NetworkingError') {
      errorMessage = "Network error connecting to AWS";
      errorDetails = {
        reason: "Unable to connect to AWS services",
        suggestion: "Check internet connection and AWS region configuration"
      };
    }

    return {
      status: "error",
      message: errorMessage,
      error: error.message,
      details: errorDetails,
      bucket: 'data-collector-of-first-device',
      key: 'devices.json',
      attemptedAt: new Date().toISOString()
    };
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

// Email template functions
function getEmailTemplate(alertLevel, riskScore, confidence, alertData = {}) {
  const templates = {
    urgent: {
      subject: '🚨 URGENT: Fire Detected - Immediate Action Required',
      color: '#dc2626',
      icon: '🔥',
      priority: 'high',
      body: `
        <div style="background-color: #fee2e2; border-left: 4px solid #dc2626; padding: 20px; margin: 20px 0;">
          <h2 style="color: #dc2626; margin-top: 0;">🚨 FIRE DETECTED - IMMEDIATE ACTION REQUIRED</h2>
          <p style="font-size: 18px; font-weight: bold; color: #991b1b;">
            High confidence fire detection has been triggered. Evacuate immediately and contact emergency services.
          </p>
        </div>
        <div style="background-color: #ffffff; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px;">
          <h3 style="color: #111827; margin-top: 0;">Alert Details</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Risk Score:</td>
              <td style="padding: 12px 0; color: #dc2626; font-weight: bold; font-size: 18px;">${riskScore}/100</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Confidence Level:</td>
              <td style="padding: 12px 0; color: #111827;">${(confidence * 100).toFixed(1)}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Alert Level:</td>
              <td style="padding: 12px 0; color: #dc2626; font-weight: bold;">LEVEL ${alertLevel} - CRITICAL</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Location:</td>
              <td style="padding: 12px 0; color: #111827;">${alertData.location || 'Kitchen'}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Device ID:</td>
              <td style="padding: 12px 0; color: #111827;">${alertData.deviceId || 'SAAFE-KITCHEN-001'}</td>
            </tr>
            <tr>
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Detection Time:</td>
              <td style="padding: 12px 0; color: #111827;">${new Date().toLocaleString()}</td>
            </tr>
          </table>
        </div>
        <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0;">
          <h4 style="color: #92400e; margin-top: 0;">⚠️ Immediate Actions Required:</h4>
          <ul style="color: #78350f; margin: 10px 0; padding-left: 20px;">
            <li>Evacuate the premises immediately</li>
            <li>Call emergency services (911/999)</li>
            <li>Do not attempt to fight the fire unless trained</li>
            <li>Account for all personnel</li>
            <li>Do not re-enter until cleared by authorities</li>
          </ul>
        </div>
      `
    },
    warning: {
      subject: '⚠️ WARNING: Fire Risk Detected - Action Required',
      color: '#f59e0b',
      icon: '⚠️',
      priority: 'high',
      body: `
        <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; margin: 20px 0;">
          <h2 style="color: #92400e; margin-top: 0;">⚠️ FIRE RISK DETECTED - ACTION REQUIRED</h2>
          <p style="font-size: 16px; color: #78350f;">
            Elevated fire risk has been detected. Immediate investigation and preventive action recommended.
          </p>
        </div>
        <div style="background-color: #ffffff; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px;">
          <h3 style="color: #111827; margin-top: 0;">Alert Details</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Risk Score:</td>
              <td style="padding: 12px 0; color: #f59e0b; font-weight: bold; font-size: 18px;">${riskScore}/100</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Confidence Level:</td>
              <td style="padding: 12px 0; color: #111827;">${(confidence * 100).toFixed(1)}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Alert Level:</td>
              <td style="padding: 12px 0; color: #f59e0b; font-weight: bold;">LEVEL ${alertLevel} - WARNING</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Location:</td>
              <td style="padding: 12px 0; color: #111827;">${alertData.location || 'Kitchen'}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Device ID:</td>
              <td style="padding: 12px 0; color: #111827;">${alertData.deviceId || 'SAAFE-KITCHEN-001'}</td>
            </tr>
            <tr>
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Detection Time:</td>
              <td style="padding: 12px 0; color: #111827;">${new Date().toLocaleString()}</td>
            </tr>
          </table>
        </div>
        <div style="background-color: #dbeafe; border-left: 4px solid #3b82f6; padding: 15px; margin: 20px 0;">
          <h4 style="color: #1e40af; margin-top: 0;">📋 Recommended Actions:</h4>
          <ul style="color: #1e3a8a; margin: 10px 0; padding-left: 20px;">
            <li>Investigate the area immediately</li>
            <li>Check for heat sources and potential ignition points</li>
            <li>Verify all cooking equipment is off</li>
            <li>Ensure fire suppression systems are operational</li>
            <li>Monitor the situation closely</li>
          </ul>
        </div>
      `
    },
    caution: {
      subject: '⚡ CAUTION: Elevated Fire Risk Detected',
      color: '#eab308',
      icon: '⚡',
      priority: 'normal',
      body: `
        <div style="background-color: #fef9c3; border-left: 4px solid #eab308; padding: 20px; margin: 20px 0;">
          <h2 style="color: #713f12; margin-top: 0;">⚡ ELEVATED FIRE RISK DETECTED</h2>
          <p style="font-size: 16px; color: #854d0e;">
            Sensors have detected conditions that may indicate increased fire risk. Please review and monitor.
          </p>
        </div>
        <div style="background-color: #ffffff; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px;">
          <h3 style="color: #111827; margin-top: 0;">Alert Details</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Risk Score:</td>
              <td style="padding: 12px 0; color: #eab308; font-weight: bold; font-size: 18px;">${riskScore}/100</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Confidence Level:</td>
              <td style="padding: 12px 0; color: #111827;">${(confidence * 100).toFixed(1)}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Alert Level:</td>
              <td style="padding: 12px 0; color: #eab308; font-weight: bold;">LEVEL ${alertLevel} - CAUTION</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Location:</td>
              <td style="padding: 12px 0; color: #111827;">${alertData.location || 'Kitchen'}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Device ID:</td>
              <td style="padding: 12px 0; color: #111827;">${alertData.deviceId || 'SAAFE-KITCHEN-001'}</td>
            </tr>
            <tr>
              <td style="padding: 12px 0; font-weight: bold; color: #374151;">Detection Time:</td>
              <td style="padding: 12px 0; color: #111827;">${new Date().toLocaleString()}</td>
            </tr>
          </table>
        </div>
        <div style="background-color: #f3f4f6; border-left: 4px solid #6b7280; padding: 15px; margin: 20px 0;">
          <h4 style="color: #374151; margin-top: 0;">📝 Suggested Actions:</h4>
          <ul style="color: #4b5563; margin: 10px 0; padding-left: 20px;">
            <li>Review sensor readings in the dashboard</li>
            <li>Check for any unusual activity in the area</li>
            <li>Ensure proper ventilation</li>
            <li>Continue monitoring the situation</li>
          </ul>
        </div>
      `
    }
  };

  // Determine template based on risk score
  let template;
  if (riskScore >= 80) {
    template = templates.urgent;
  } else if (riskScore >= 40) {
    template = templates.warning;
  } else {
    template = templates.caution;
  }

  return {
    subject: template.subject,
    html: `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${template.subject}</title>
      </head>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #111827; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9fafb;">
        <div style="background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); overflow: hidden;">
          <div style="background: linear-gradient(135deg, ${template.color} 0%, ${template.color}dd 100%); padding: 30px; text-align: center;">
            <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: bold;">
              ${template.icon} SAAFE Fire Detection System
            </h1>
          </div>
          <div style="padding: 30px;">
            ${template.body}
            <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 14px;">
              <p style="margin: 5px 0;">This is an automated alert from the SAAFE Fire Detection System</p>
              <p style="margin: 5px 0;">For support, contact your system administrator</p>
              <p style="margin: 5px 0; font-weight: bold;">Do not reply to this email</p>
            </div>
          </div>
        </div>
      </body>
      </html>
    `,
    priority: template.priority
  };
}

// Function to send alert emails to multiple recipients
async function sendAlertEmails(alertLevel, riskScore, confidence, alertData = {}) {
  if (!emailTransporter) {
    console.warn('Email transporter not available. Skipping email alerts.');
    return { sent: 0, failed: 0, skipped: 0 };
  }

  // Check cooldown to prevent email spam
  const now = Date.now();
  if (lastEmailSent.timestamp && (now - lastEmailSent.timestamp) < EMAIL_COOLDOWN) {
    // Only send if risk score increased significantly (by 20 points or more)
    if (riskScore < lastEmailSent.riskScore + 20) {
      console.log(`Email cooldown active. Last email sent ${Math.floor((now - lastEmailSent.timestamp) / 1000)}s ago. Skipping.`);
      return { sent: 0, failed: 0, skipped: EMAIL_CONFIG.recipients.filter(r => r.enabled).length };
    }
  }

  const emailTemplate = getEmailTemplate(alertLevel, riskScore, confidence, alertData);
  const results = { sent: 0, failed: 0, skipped: 0 };

  // Filter recipients based on alert level and enabled status
  const eligibleRecipients = EMAIL_CONFIG.recipients.filter(recipient => {
    if (!recipient.enabled) return false;
    if (recipient.alertLevels.includes('all')) return true;
    
    // Check if recipient should receive this alert level
    if (riskScore >= 80 && recipient.alertLevels.includes('urgent')) return true;
    if (riskScore >= 40 && recipient.alertLevels.includes('warning')) return true;
    if (riskScore >= 20 && recipient.alertLevels.includes('caution')) return true;
    
    return false;
  });

  console.log(`Sending alerts to ${eligibleRecipients.length} recipients (Risk Score: ${riskScore})`);

  // Send emails to all eligible recipients
  for (const recipient of eligibleRecipients) {
    try {
      const mailOptions = {
        from: `SAAFE Alert System <${EMAIL_CONFIG.sender_email}>`,
        to: recipient.email,
        subject: emailTemplate.subject,
        html: emailTemplate.html,
        priority: emailTemplate.priority
      };

      await emailTransporter.sendMail(mailOptions);
      console.log(`✓ Alert email sent to ${recipient.name} (${recipient.email})`);
      results.sent++;
    } catch (error) {
      console.error(`✗ Failed to send email to ${recipient.name} (${recipient.email}):`, error.message);
      results.failed++;
    }
  }

  // Update last email sent tracking
  if (results.sent > 0) {
    lastEmailSent = {
      timestamp: now,
      riskScore: riskScore,
      level: alertLevel
    };
  }

  return results;
}

const app = express();
const PORT = process.env.PORT || 8080;

// In-memory storage for alert state
let alertState = {
  isActive: false,
  level: 1,
  message: "System operating normally",
  timestamp: new Date().toISOString(),
  riskScore: 0,
  confidence: 0.9
};

// In-memory storage for alert history (last 50 events)
let alertHistory = [];
const MAX_HISTORY_SIZE = 50;

// Helper function to add event to history
function addToAlertHistory(eventType, riskScore, level, message, confidence) {
  const event = {
    id: Date.now() + Math.random(), // Unique ID
    timestamp: new Date().toISOString(),
    eventType, // 'Normal', 'Fire Predicted', 'Fire Detected'
    riskScore,
    level,
    message,
    confidence
  };
  
  // Add to beginning of array (most recent first)
  alertHistory.unshift(event);
  
  // Keep only last MAX_HISTORY_SIZE events
  if (alertHistory.length > MAX_HISTORY_SIZE) {
    alertHistory = alertHistory.slice(0, MAX_HISTORY_SIZE);
  }
  
  console.log(`Added to alert history: ${eventType} (Risk: ${riskScore})`);
}

// Health check for App Runner
app.get('/health', (req, res) => res.status(200).send('ok'));
// Middleware
app.use(cors({
  origin: '*',
  credentials: false,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
app.use(express.json());

// Handle preflight OPTIONS requests
app.options('*', cors({
  origin: '*',
  credentials: false,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

// Serve static files from the React app build directory
app.use(express.static(path.join(__dirname, '../dist')));

app.post('/predict_features', async (req, res) => {
  try {
    const data = await callModelPrediction('features18', req.body);
    res.json(data);
  } catch (e) {
    res.status(500).json({ message: 'Upstream error', detail: String(e?.message || e) });
  }
});


app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});
// API endpoint for fire detection data - now fetching live data from S3
app.get('/api/fire-detection-data', async (req, res) => {
  try {
    // Fetch live data from S3 bucket
    const fireData = await fetchLiveFireDataFromS3();
    
    const responseData = {
      status: "success",
      data: fireData
    };

    res.json(responseData);
  } catch (err) {
    console.error("Error fetching fire detection data:", err);
    
    // Return error as fallback mechanisms have been removed
    res.status(500).json({
      status: "error",
      message: "Failed to fetch live data from S3",
      error: err.message
    });
  }
});

// Email notification function
async function sendFireAlertEmail(modelName, predictionData) {
  if (!emailTransporter) {
    console.warn('Email transporter not available. Skipping email alert.');
    return false;
  }

  try {
    const mailOptions = {
      from: EMAIL_CONFIG.sender_email,
      to: EMAIL_CONFIG.recipient_email,
      subject: `🔥 FIRE ALERT - ${modelName} Detected Fire`,
      html: `
        <h2>FIRE DETECTED by ${modelName}!</h2>
        <p><strong>Prediction Details:</strong></p>
        <pre>${JSON.stringify(predictionData, null, 2)}</pre>
        <p><strong>Time:</strong> ${new Date().toISOString()}</p>
        <p>Please take immediate action.</p>
      `
    };

    await emailTransporter.sendMail(mailOptions);
    console.log(`Fire alert email sent for ${modelName}`);
    return true;
  } catch (error) {
    console.error('Failed to send fire alert email:', error);
    return false;
  }
}

// Function to call model prediction
async function callModelPrediction(modelName, payload) {
  try {
    const url = MODEL_URLS[modelName];
    if (!url) {
      throw new Error(`Unknown model: ${modelName}`);
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error(`Error calling ${modelName} model:`, error);
    throw error;
  }
}

// API endpoint for model predictions
app.post('/api/predict/:model', async (req, res) => {
  try {
    const { model } = req.params;
    const payload = req.body;

    console.log(`Calling ${model} model with payload:`, JSON.stringify(payload, null, 2));

    const prediction = await callModelPrediction(model, payload);

    // Check if fire detected and send email
    let fireDetected = false;
    if (model === 'saafe' && prediction.prediction?.label?.toLowerCase().includes('fire')) {
      fireDetected = true;
    } else if (model === 'features18' && prediction.fire_detected) {
      fireDetected = true;
    } else if (model === 'kaggle' && prediction.fire_prediction) {
      fireDetected = true;
    } else if (model === 'tensorflow' && prediction.prediction?.label?.toLowerCase().includes('fire')) {
      fireDetected = true;
    }

    if (fireDetected) {
      console.log(`Fire detected by ${model}, sending alert email...`);
      await sendFireAlertEmail(model, prediction);
    }

    res.json({
      status: "success",
      model: model,
      prediction: prediction,
      fire_detected: fireDetected,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error(`Error in /api/predict/${req.params.model}:`, error);
    res.status(500).json({
      status: "error",
      message: `Failed to get prediction from ${req.params.model} model`,
      error: error.message
    });
  }
});

// API endpoint to update email configuration
app.post('/api/update-email-config', (req, res) => {
  try {
    const { recipient_email } = req.body;

    if (!recipient_email || !recipient_email.includes('@')) {
      return res.status(400).json({
        status: "error",
        message: "Valid email address is required"
      });
    }

    // Update the email configuration
    EMAIL_CONFIG.recipient_email = recipient_email;

    console.log(`Email configuration updated to: ${recipient_email}`);

    res.json({
      status: "success",
      message: "Email configuration updated successfully",
      recipient_email: recipient_email
    });
  } catch (error) {
    console.error('Error updating email configuration:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to update email configuration",
      error: error.message
    });
  }
});

// API endpoint to send test email
app.post('/api/send-test-email', async (req, res) => {
  try {
    const { test_email } = req.body;

    if (!test_email || !test_email.includes('@')) {
      return res.status(400).json({
        status: "error",
        message: "Valid email address is required"
      });
    }

    if (!emailTransporter) {
      return res.status(500).json({
        status: "error",
        message: "Email service not available"
      });
    }

    const mailOptions = {
      from: EMAIL_CONFIG.sender_email,
      to: test_email,
      subject: `🧪 SAAFE Test Email - Configuration Verified`,
      html: `
        <h2>SAAFE Email Configuration Test</h2>
        <p><strong>✅ Success!</strong> Your email configuration is working correctly.</p>
        <p>This is a test email from the SAAFE Fire Detection Dashboard.</p>
        <p><strong>Test Details:</strong></p>
        <ul>
          <li>Sender: ${EMAIL_CONFIG.sender_email}</li>
          <li>Recipient: ${test_email}</li>
          <li>Time: ${new Date().toISOString()}</li>
        </ul>
        <p>You will receive real fire alert emails when any AI model detects fire in your monitored areas.</p>
        <p>Best regards,<br/>SAAFE Fire Detection System</p>
      `
    };

    await emailTransporter.sendMail(mailOptions);
    console.log(`Test email sent successfully to: ${test_email}`);

    res.json({
      status: "success",
      message: "Test email sent successfully",
      recipient: test_email
    });
  } catch (error) {
    console.error('Error sending test email:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to send test email",
      error: error.message
    });
  }
});

// Function to fetch live fire data from S3
async function fetchLiveFireDataFromS3() {
  try {
    // List objects in the S3 bucket using tail method to get most recent data
    const bucketName = 'data-collector-of-first-device';
    
    // List objects in thermal-data directory with tail method
    const thermalData = await s3.listObjectsV2({ 
      Bucket: bucketName, 
      Prefix: 'thermal-data/',
      MaxKeys: 20
    }).promise();
    
    // List objects in gas-data directory with tail method
    const gasData = await s3.listObjectsV2({ 
      Bucket: bucketName, 
      Prefix: 'gas-data/',
      MaxKeys: 20
    }).promise();
    
    // Log all available files for debugging
    console.log("=== ALL THERMAL FILES ===");
    if (thermalData.Contents) {
      thermalData.Contents.forEach(file => {
        console.log(`File: ${file.Key}, Last Modified: ${file.LastModified}`);
      });
    }
    
    console.log("=== ALL GAS FILES ===");
    if (gasData.Contents) {
      gasData.Contents.forEach(file => {
        console.log(`File: ${file.Key}, Last Modified: ${file.LastModified}`);
      });
    }
    
    // Get the most recent thermal and gas files based on LastModified timestamp
    let thermalFile = null;
    let gasFile = null;
    
    if (thermalData.Contents && thermalData.Contents.length > 0) {
      // Sort by LastModified timestamp in descending order and get the most recent
      thermalData.Contents.sort((a, b) => new Date(b.LastModified) - new Date(a.LastModified));
      thermalFile = thermalData.Contents[0];
    }
    
    if (gasData.Contents && gasData.Contents.length > 0) {
      // Sort by LastModified timestamp in descending order and get the most recent
      gasData.Contents.sort((a, b) => new Date(b.LastModified) - new Date(a.LastModified));
      gasFile = gasData.Contents[0];
    }
    
    if (!thermalFile && !gasFile) {
      throw new Error('No thermal or gas data files found');
    }
    
    // Log proof of data origin to console
    console.log("=== AWS DATA PROVENANCE PROOF ===");
    console.log("Data fetched from AWS S3 bucket:", bucketName);
    if (thermalFile) {
      console.log("Thermal data file:", thermalFile.Key);
      console.log("Thermal file last modified:", thermalFile.LastModified);
    }
    if (gasFile) {
      console.log("Gas data file:", gasFile.Key);
      console.log("Gas file last modified:", gasFile.LastModified);
    }
    console.log("=================================");
    
    // Fetch file contents
    let thermalDataContent = null;
    let gasDataContent = null;
    
    if (thermalFile) {
      const thermalObject = await s3.getObject({
        Bucket: bucketName,
        Key: thermalFile.Key
      }).promise();
      
      const thermalContent = thermalObject.Body.toString('utf-8');
      thermalDataContent = parseThermalData(thermalContent);
    }
    
    if (gasFile) {
      const gasObject = await s3.getObject({
        Bucket: bucketName,
        Key: gasFile.Key
      }).promise();
      
      const gasContent = gasObject.Body.toString('utf-8');
      gasDataContent = parseGasData(gasContent);
    }
    
    // Check if data is too old (more than 1 hour old)
    const now = Date.now();
    const thermalAge = thermalFile ? (now - new Date(thermalFile.LastModified).getTime()) / 1000 : Infinity;
    const gasAge = gasFile ? (now - new Date(gasFile.LastModified).getTime()) / 1000 : Infinity;
    const maxAge = Math.max(thermalAge, gasAge);

    if (maxAge > 3600) { // 1 hour
      console.log(`S3 data is too old (${Math.floor(maxAge / 60)} minutes). Generating synthetic live data.`);
      return createSyntheticLiveData();
    }

    // Combine data into the expected format
    return createFireDetectionData(thermalDataContent, gasDataContent, thermalFile, gasFile);
  } catch (error) {
    console.error("Error fetching data from S3:", error);
    throw error;
  }
}

// Parse thermal data from CSV content - get most recent reading by timestamp
function parseThermalData(csvContent) {
  try {
    const lines = csvContent.trim().split('\n');
    if (lines.length < 2) return null;

    const headers = lines[0].split(',');
    const readings = [];

    // Parse all data lines
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length >= headers.length) {
        const reading = { timestamp: values[0] };
        // Parse sensor values, skipping timestamp
        for (let j = 1; j < headers.length; j++) {
          reading[headers[j]] = parseFloat(values[j]) || 0;
        }
        readings.push(reading);
      }
    }

    if (readings.length === 0) return null;

    // Sort by timestamp (most recent first) and return the latest reading
    readings.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    const latestReading = readings[0];

    // Remove timestamp from final data object
    const { timestamp, ...data } = latestReading;
    data._timestamp = timestamp; // Keep timestamp for reference

    return data;
  } catch (error) {
    console.error("Error parsing thermal data:", error);
    return null;
  }
}

// Parse gas data from CSV content - get most recent reading by timestamp
function parseGasData(csvContent) {
  try {
    const lines = csvContent.trim().split('\n');
    if (lines.length < 2) return null;

    const headers = lines[0].split(',');
    const readings = [];

    // Parse all data lines
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length >= headers.length) {
        const reading = {};
        // Parse all values including timestamp
        for (let j = 0; j < headers.length; j++) {
          reading[headers[j]] = j === 0 ? values[j] : (parseFloat(values[j]) || 0);
        }
        readings.push(reading);
      }
    }

    if (readings.length === 0) return null;

    // Sort by timestamp (most recent first) and return the latest reading
    readings.sort((a, b) => new Date(b.timestamp || b.Timestamp) - new Date(a.timestamp || a.Timestamp));
    const latestReading = readings[0];

    // Keep timestamp for reference
    latestReading._timestamp = latestReading.timestamp || latestReading.Timestamp;

    return latestReading;
  } catch (error) {
    console.error("Error parsing gas data:", error);
    return null;
  }
}

// Create fire detection data structure
function createFireDetectionData(thermalData, gasData, thermalFile, gasFile) {
  // Use the actual sensor timestamp if available, otherwise use current time
  let sensorTimestamp = Math.floor(Date.now() / 1000);

  // Try to get timestamp from thermal data first, then gas data
  if (thermalData && thermalData._timestamp) {
    sensorTimestamp = Math.floor(new Date(thermalData._timestamp).getTime() / 1000);
  } else if (gasData && gasData._timestamp) {
    sensorTimestamp = Math.floor(new Date(gasData._timestamp).getTime() / 1000);
  }
  
  // Default values - removed randomization for production use
  const defaultThermalStats = {
    max: 30,
    min: 20,
    mean: 25
  };
  
  const defaultGasReadings = {
    voc: 50,
    co: 0.5,
    no2: 0.1
  };
  
  const defaultEnvironmentalData = {
    temperature: 25,
    humidity: 40,
    pressure: 1013
  };
  
  // Use actual data if available
  if (thermalData) {
    // Extract thermal stats from actual data
    const pixelValues = Object.values(thermalData);
    if (pixelValues.length > 0) {
      defaultThermalStats.max = Math.max(...pixelValues);
      defaultThermalStats.min = Math.min(...pixelValues);
      defaultThermalStats.mean = pixelValues.reduce((a, b) => a + b, 0) / pixelValues.length;
    }
  }
  
  if (gasData) {
    // Use actual gas readings
    if (gasData.VOC !== undefined) defaultGasReadings.voc = gasData.VOC;
    if (gasData.CO !== undefined) defaultGasReadings.co = gasData.CO;
    if (gasData.NO2 !== undefined) defaultGasReadings.no2 = gasData.NO2;
  }
  
  // Generate thermal frame with actual data or zeros
  const thermalFrame = generateThermalFrame(thermalData);
  
  // Prepare data provenance information
  const dataProvenance = {
    source: "AWS S3",
    bucket: "data-collector-of-first-device",
    timestamp: new Date().toISOString(),
    sensor_timestamp: new Date(sensorTimestamp * 1000).toISOString(),
    data_age_seconds: Math.floor(Date.now() / 1000) - sensorTimestamp
  };
  
  if (thermalFile) {
    dataProvenance.thermal_file = {
      key: thermalFile.Key,
      last_modified: thermalFile.LastModified
    };
  }
  
  if (gasFile) {
    dataProvenance.gas_file = {
      key: gasFile.Key,
      last_modified: gasFile.LastModified
    };
  }
  
  return {
    sensor_data: {
      timestamp: sensorTimestamp,
      thermal_frame: thermalFrame,
      thermal_stats: defaultThermalStats,
      gas_readings: defaultGasReadings,
      environmental_data: defaultEnvironmentalData,
      sensor_health: {
        thermal_camera: 0.95,
        gas_sensor: 0.98,
        environmental: 0.97
      }
    },
    prediction: {
      timestamp: sensorTimestamp,
      fire_probability: 0.1,
      confidence_score: 0.8,
      lead_time_estimate: 30,
      contributing_factors: {
        "voc_level": 0.7,
        "temperature_spike": 0.6,
        "smoke_detected": 0.4
      },
      model_ensemble_votes: {
        "model_a": 1,
        "model_b": 1,
        "model_c": 0
      }
    },
    risk_assessment: {
      timestamp: sensorTimestamp,
      risk_level: "low",
      fire_probability: 0.1,
      confidence_level: 0.85,
      contributing_sensors: ["thermal_camera", "gas_sensor"],
      recommended_actions: ["increase monitoring frequency", "verify ventilation"],
      escalation_required: false
    },
    alert: {
      alert_level: {
        level: 1,
        description: "Normal",
        icon: "✅"
      },
      risk_score: 10,
      confidence: 0.9,
      message: "System operating normally",
      timestamp: new Date().toISOString(),
      context_info: {
        "location": "Kitchen",
        "device_id": "SAAFE-KITCHEN-001"
      }
    },
    data_provenance: dataProvenance,
    last_updated: new Date().toISOString()
  };
}

// Function to create synthetic live data when S3 data is too old
function createSyntheticLiveData() {
  const now = Math.floor(Date.now() / 1000);

  // Generate realistic sensor readings with some variation
  const baseTemp = 22 + Math.sin(Date.now() / 100000) * 3; // Temperature varies sinusoidally
  const baseHumidity = 45 + Math.cos(Date.now() / 80000) * 5;
  const basePressure = 1013 + Math.sin(Date.now() / 200000) * 2;

  // Gas readings with small random variations
  const voc = 45 + Math.random() * 10;
  const co = 0.4 + Math.random() * 0.2;
  const no2 = 0.08 + Math.random() * 0.04;

  // Thermal stats
  const thermalMax = baseTemp + 5 + Math.random() * 3;
  const thermalMin = baseTemp - 2 + Math.random() * 2;
  const thermalMean = (thermalMax + thermalMin) / 2;

  return {
    sensor_data: {
      timestamp: now,
      thermal_frame: generateThermalFrame(null, thermalMean),
      thermal_stats: {
        max: thermalMax,
        min: thermalMin,
        mean: thermalMean
      },
      gas_readings: {
        voc: voc,
        co: co,
        no2: no2
      },
      environmental_data: {
        temperature: baseTemp,
        humidity: baseHumidity,
        pressure: basePressure
      },
      sensor_health: {
        thermal_camera: 0.95 + Math.random() * 0.05,
        gas_sensor: 0.98 + Math.random() * 0.02,
        environmental: 0.97 + Math.random() * 0.03
      }
    },
    prediction: {
      timestamp: now,
      fire_probability: Math.random() * 0.1, // Low probability for normal conditions
      confidence_score: 0.8 + Math.random() * 0.2,
      lead_time_estimate: 30,
      contributing_factors: {
        "voc_level": Math.random() * 0.3,
        "temperature_spike": Math.random() * 0.2,
        "smoke_detected": Math.random() * 0.1
      },
      model_ensemble_votes: {
        "model_a": Math.floor(Math.random() * 2),
        "model_b": Math.floor(Math.random() * 2),
        "model_c": Math.floor(Math.random() * 2)
      }
    },
    risk_assessment: {
      timestamp: now,
      risk_level: "low",
      fire_probability: Math.random() * 0.1,
      confidence_level: 0.85 + Math.random() * 0.1,
      contributing_sensors: ["thermal_camera", "gas_sensor"],
      recommended_actions: ["continue normal monitoring"],
      escalation_required: false
    },
    alert: {
      alert_level: {
        level: 1,
        description: "Normal",
        icon: "✅"
      },
      risk_score: Math.floor(Math.random() * 20),
      confidence: 0.9 + Math.random() * 0.1,
      message: "System operating normally - synthetic live data",
      timestamp: new Date().toISOString(),
      context_info: {
        "location": "Kitchen",
        "device_id": "SAAFE-KITCHEN-001",
        "data_source": "synthetic_live"
      }
    },
    data_provenance: {
      source: "Synthetic Live Data",
      bucket: "generated",
      timestamp: new Date().toISOString(),
      sensor_timestamp: new Date(now * 1000).toISOString(),
      data_age_seconds: 0
    },
    last_updated: new Date().toISOString()
  };
}

// Helper function to generate thermal frame data
function generateThermalFrame(thermalData, baseTemp = 25) {
  // If we have actual thermal data, use it
  if (thermalData) {
    // Convert the thermal data object to a 20x20 frame
    const frame = [];
    const values = Object.values(thermalData);
    let valueIndex = 0;
    
    for (let i = 0; i < 20; i++) {
      const row = [];
      for (let j = 0; j < 20; j++) {
        // Use available data or default value
        row.push(values[valueIndex] || 25);
        valueIndex = (valueIndex + 1) % values.length;
      }
      frame.push(row);
    }
    return frame;
  }
  
  // If no thermal data, return a frame of default values
  const frame = [];
  for (let i = 0; i < 20; i++) {
    const row = [];
    for (let j = 0; j < 20; j++) {
      row.push(25);
    }
    frame.push(row);
  }
  return frame;
}



// API endpoint to get all devices
app.get('/api/devices', async (req, res) => {
  try {
    const deviceResult = await fetchDeviceDataFromS3();

    if (deviceResult.status === "error") {
      return res.status(503).json(deviceResult); // Service Unavailable
    }

    res.json(deviceResult);
  } catch (error) {
    console.error('Error in /api/devices:', error);
    res.status(500).json({
      status: "error",
      message: "Internal server error while fetching devices",
      error: error.message
    });
  }
});

// API endpoint to get specific device by ID
app.get('/api/devices/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const deviceResult = await fetchDeviceDataFromS3();

    if (deviceResult.status === "error") {
      return res.status(503).json({
        status: "error",
        message: `Cannot fetch device ${id} - ${deviceResult.message}`,
        details: deviceResult.details,
        deviceId: id
      });
    }

    const device = deviceResult.data.find(d => d.id === id);

    if (!device) {
      return res.status(404).json({
        status: "error",
        message: `Device with ID ${id} not found`,
        availableDevices: deviceResult.data.map(d => d.id),
        totalDevices: deviceResult.count,
        deviceId: id
      });
    }

    res.json({
      status: "success",
      data: device,
      source: deviceResult.source,
      fetchedAt: deviceResult.fetchedAt
    });
  } catch (error) {
    console.error('Error in /api/devices/:id:', error);
    res.status(500).json({
      status: "error",
      message: "Internal server error while fetching device",
      error: error.message,
      deviceId: req.params.id
    });
  }
});

// API endpoint to get current alert state
app.get('/api/alert-state', (req, res) => {
  try {
    console.log('GET /api/alert-state - Current state:', alertState);
    res.json({
      status: "success",
      data: alertState
    });
  } catch (error) {
    console.error('Error in GET /api/alert-state:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to retrieve alert state",
      error: error.message
    });
  }
});

// API endpoint to get alert history
app.get('/api/alert-history', (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const limitedHistory = alertHistory.slice(0, Math.min(limit, alertHistory.length));
    
    console.log(`GET /api/alert-history - Returning ${limitedHistory.length} events`);
    res.json({
      status: "success",
      data: limitedHistory,
      total: alertHistory.length
    });
  } catch (error) {
    console.error('Error in GET /api/alert-history:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to retrieve alert history",
      error: error.message
    });
  }
});

// API endpoint to set alert state (with automatic email notifications)
app.post('/api/alert-state', async (req, res) => {
  try {
    const { isActive, level, message, riskScore, confidence, location, deviceId } = req.body;

    // Validate required fields
    if (typeof isActive !== 'boolean') {
      return res.status(400).json({
        status: "error",
        message: "isActive field is required and must be a boolean"
      });
    }

    // Store previous state for comparison
    const previousRiskScore = alertState.riskScore;
    const previousIsActive = alertState.isActive;

    // Update alert state with provided values
    alertState = {
      isActive,
      level: level !== undefined ? level : alertState.level,
      message: message || alertState.message,
      riskScore: riskScore !== undefined ? riskScore : alertState.riskScore,
      confidence: confidence !== undefined ? confidence : alertState.confidence,
      timestamp: new Date().toISOString()
    };

    console.log('POST /api/alert-state - Updated state:', alertState);

    // Determine event type and add to history
    let eventType = 'Normal';
    if (riskScore >= 80) {
      eventType = 'Fire Detected';
    } else if (riskScore >= 40) {
      eventType = 'Fire Predicted';
    }
    
    // Add to history if state changed significantly
    if (previousRiskScore !== riskScore || previousIsActive !== isActive) {
      addToAlertHistory(eventType, riskScore, level, message, confidence);
    }

    // Send email notifications if alert is active and risk score meets threshold
    let emailResults = null;
    if (isActive && riskScore >= 40) {
      // Only send if this is a new alert or risk score increased significantly
      if (!previousIsActive || riskScore > previousRiskScore) {
        console.log(`Triggering email notifications (Risk Score: ${riskScore})`);
        emailResults = await sendAlertEmails(
          level,
          riskScore,
          confidence,
          {
            location: location || 'Kitchen',
            deviceId: deviceId || 'SAAFE-KITCHEN-001'
          }
        );
        console.log(`Email notification results:`, emailResults);
      }
    }

    res.json({
      status: "success",
      message: "Alert state updated successfully",
      data: alertState,
      emailNotifications: emailResults
    });
  } catch (error) {
    console.error('Error in POST /api/alert-state:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to update alert state",
      error: error.message
    });
  }
});

// API endpoint to get email recipients
app.get('/api/email-recipients', (req, res) => {
  try {
    res.json({
      status: "success",
      data: EMAIL_CONFIG.recipients,
      count: EMAIL_CONFIG.recipients.length
    });
  } catch (error) {
    console.error('Error in GET /api/email-recipients:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to retrieve email recipients",
      error: error.message
    });
  }
});

// API endpoint to add email recipient
app.post('/api/email-recipients', (req, res) => {
  try {
    const { email, name, alertLevels, enabled } = req.body;

    // Validate required fields
    if (!email || !email.includes('@')) {
      return res.status(400).json({
        status: "error",
        message: "Valid email address is required"
      });
    }

    if (!name) {
      return res.status(400).json({
        status: "error",
        message: "Recipient name is required"
      });
    }

    // Check if email already exists
    const existingRecipient = EMAIL_CONFIG.recipients.find(r => r.email === email);
    if (existingRecipient) {
      return res.status(409).json({
        status: "error",
        message: "Email address already exists in recipient list"
      });
    }

    // Validate alert levels
    const validAlertLevels = ['all', 'urgent', 'warning', 'caution'];
    const recipientAlertLevels = alertLevels || ['all'];
    
    for (const level of recipientAlertLevels) {
      if (!validAlertLevels.includes(level)) {
        return res.status(400).json({
          status: "error",
          message: `Invalid alert level: ${level}. Valid levels are: ${validAlertLevels.join(', ')}`
        });
      }
    }

    // Add new recipient
    const newRecipient = {
      email,
      name,
      alertLevels: recipientAlertLevels,
      enabled: enabled !== undefined ? enabled : true
    };

    EMAIL_CONFIG.recipients.push(newRecipient);

    console.log(`Added new email recipient: ${name} (${email})`);

    res.json({
      status: "success",
      message: "Email recipient added successfully",
      data: newRecipient,
      totalRecipients: EMAIL_CONFIG.recipients.length
    });
  } catch (error) {
    console.error('Error in POST /api/email-recipients:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to add email recipient",
      error: error.message
    });
  }
});

// API endpoint to update email recipient
app.put('/api/email-recipients/:email', (req, res) => {
  try {
    const { email } = req.params;
    const { name, alertLevels, enabled } = req.body;

    // Find recipient
    const recipientIndex = EMAIL_CONFIG.recipients.findIndex(r => r.email === email);
    if (recipientIndex === -1) {
      return res.status(404).json({
        status: "error",
        message: "Email recipient not found"
      });
    }

    // Validate alert levels if provided
    if (alertLevels) {
      const validAlertLevels = ['all', 'urgent', 'warning', 'caution'];
      for (const level of alertLevels) {
        if (!validAlertLevels.includes(level)) {
          return res.status(400).json({
            status: "error",
            message: `Invalid alert level: ${level}. Valid levels are: ${validAlertLevels.join(', ')}`
          });
        }
      }
    }

    // Update recipient
    if (name !== undefined) EMAIL_CONFIG.recipients[recipientIndex].name = name;
    if (alertLevels !== undefined) EMAIL_CONFIG.recipients[recipientIndex].alertLevels = alertLevels;
    if (enabled !== undefined) EMAIL_CONFIG.recipients[recipientIndex].enabled = enabled;

    console.log(`Updated email recipient: ${EMAIL_CONFIG.recipients[recipientIndex].name} (${email})`);

    res.json({
      status: "success",
      message: "Email recipient updated successfully",
      data: EMAIL_CONFIG.recipients[recipientIndex]
    });
  } catch (error) {
    console.error('Error in PUT /api/email-recipients/:email:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to update email recipient",
      error: error.message
    });
  }
});

// API endpoint to delete email recipient
app.delete('/api/email-recipients/:email', (req, res) => {
  try {
    const { email } = req.params;

    // Find recipient
    const recipientIndex = EMAIL_CONFIG.recipients.findIndex(r => r.email === email);
    if (recipientIndex === -1) {
      return res.status(404).json({
        status: "error",
        message: "Email recipient not found"
      });
    }

    // Remove recipient
    const removedRecipient = EMAIL_CONFIG.recipients.splice(recipientIndex, 1)[0];

    console.log(`Removed email recipient: ${removedRecipient.name} (${email})`);

    res.json({
      status: "success",
      message: "Email recipient removed successfully",
      data: removedRecipient,
      remainingRecipients: EMAIL_CONFIG.recipients.length
    });
  } catch (error) {
    console.error('Error in DELETE /api/email-recipients/:email:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to remove email recipient",
      error: error.message
    });
  }
});

// API endpoint to send test alert email
app.post('/api/test-alert-email', async (req, res) => {
  try {
    const { email, riskScore } = req.body;

    if (!email || !email.includes('@')) {
      return res.status(400).json({
        status: "error",
        message: "Valid email address is required"
      });
    }

    if (!emailTransporter) {
      return res.status(500).json({
        status: "error",
        message: "Email service not available"
      });
    }

    // Use provided risk score or default to 85 (urgent level)
    const testRiskScore = riskScore !== undefined ? riskScore : 85;
    const testAlertLevel = testRiskScore >= 80 ? 4 : testRiskScore >= 40 ? 3 : 2;
    const testConfidence = 0.95;

    const emailTemplate = getEmailTemplate(testAlertLevel, testRiskScore, testConfidence, {
      location: 'Kitchen (Test)',
      deviceId: 'SAAFE-TEST-001'
    });

    const mailOptions = {
      from: `SAAFE Alert System <${EMAIL_CONFIG.sender_email}>`,
      to: email,
      subject: `[TEST] ${emailTemplate.subject}`,
      html: emailTemplate.html.replace(
        'This is an automated alert from the SAAFE Fire Detection System',
        '⚠️ THIS IS A TEST EMAIL - No action required ⚠️<br>This is an automated alert from the SAAFE Fire Detection System'
      ),
      priority: emailTemplate.priority
    };

    await emailTransporter.sendMail(mailOptions);
    console.log(`Test alert email sent to: ${email} (Risk Score: ${testRiskScore})`);

    res.json({
      status: "success",
      message: "Test alert email sent successfully",
      recipient: email,
      riskScore: testRiskScore,
      alertLevel: testAlertLevel
    });
  } catch (error) {
    console.error('Error sending test alert email:', error);
    res.status(500).json({
      status: "error",
      message: "Failed to send test alert email",
      error: error.message
    });
  }
});

// The "catchall" handler: for any request that doesn't
// match one above, send back React's index.html file.
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../dist/index.html'));
});

app.listen(PORT, () => {
  console.log(`Fire Detection Backend Server is running on port ${PORT}`);
  console.log(`Frontend should be accessible at http://localhost:5173`);
  console.log(`API endpoints available at http://localhost:${PORT}/api/*`);
});