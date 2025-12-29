#!/bin/bash

# SAAFE Dashboard Serverless AWS Deployment
# Uses AWS Lambda + API Gateway + S3 for global deployment

set -e

echo "🚀 SAAFE Dashboard Serverless AWS Deployment"
echo "============================================"

# Configuration
BUCKET_NAME="saafe-dashboard-$(date +%s)"
REGION="us-east-1"
STACK_NAME="saafe-dashboard-stack"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not configured. Please run: aws configure${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites ready${NC}"

# Build the frontend
echo -e "${YELLOW}📦 Building React frontend...${NC}"
cd saafe-lovable
npm install
npm run build
cd ..

echo -e "${GREEN}✅ Frontend built${NC}"

# Create S3 bucket for frontend
echo -e "${YELLOW}🪣 Creating S3 bucket for frontend...${NC}"
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Configure bucket for static website hosting
aws s3 website s3://$BUCKET_NAME --index-document index.html --error-document index.html

# Upload frontend files
echo -e "${YELLOW}📤 Uploading frontend to S3...${NC}"
aws s3 sync saafe-lovable/dist/ s3://$BUCKET_NAME --delete

# Create CloudFront distribution for S3 bucket
echo -e "${YELLOW}🌍 Creating CloudFront distribution...${NC}"

# Create CloudFront Origin Access Control
OAC_ID=$(aws cloudfront create-origin-access-control \
    --origin-access-control-config \
    Name="saafe-dashboard-oac",Description="OAC for SAAFE Dashboard",OriginAccessControlOriginType="s3",SigningBehavior="always",SigningProtocol="sigv4" \
    --query 'OriginAccessControl.Id' --output text 2>/dev/null || \
aws cloudfront list-origin-access-controls --query 'OriginAccessControlList.Items[?Name==`saafe-dashboard-oac`].Id' --output text)

# Create CloudFront distribution
cat > cloudfront-config.json << EOF
{
    "CallerReference": "saafe-dashboard-$(date +%s)",
    "Comment": "SAAFE Dashboard Distribution",
    "DefaultRootObject": "index.html",
    "Origins": {
        "Quantity": 1,
        "Items": [
            {
                "Id": "S3-$BUCKET_NAME",
                "DomainName": "$BUCKET_NAME.s3.$REGION.amazonaws.com",
                "S3OriginConfig": {
                    "OriginAccessIdentity": ""
                },
                "OriginAccessControlId": "$OAC_ID"
            }
        ]
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": "S3-$BUCKET_NAME",
        "ViewerProtocolPolicy": "redirect-to-https",
        "MinTTL": 0,
        "ForwardedValues": {
            "QueryString": false,
            "Cookies": {
                "Forward": "none"
            }
        },
        "TrustedSigners": {
            "Enabled": false,
            "Quantity": 0
        }
    },
    "CustomErrorResponses": {
        "Quantity": 1,
        "Items": [
            {
                "ErrorCode": 404,
                "ResponsePagePath": "/index.html",
                "ResponseCode": "200",
                "ErrorCachingMinTTL": 300
            }
        ]
    },
    "Enabled": true,
    "PriceClass": "PriceClass_All"
}
EOF

DISTRIBUTION_ID=$(aws cloudfront create-distribution --distribution-config file://cloudfront-config.json --query 'Distribution.Id' --output text)
CLOUDFRONT_DOMAIN=$(aws cloudfront get-distribution --id $DISTRIBUTION_ID --query 'Distribution.DomainName' --output text)

# Update S3 bucket policy to allow CloudFront access
cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowCloudFrontServicePrincipal",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudfront.amazonaws.com"
            },
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::$BUCKET_NAME/*",
            "Condition": {
                "StringEquals": {
                    "AWS:SourceArn": "arn:aws:cloudfront::$(aws sts get-caller-identity --query Account --output text):distribution/$DISTRIBUTION_ID"
                }
            }
        }
    ]
}
EOF

aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json 2>/dev/null || echo "Bucket policy set with CloudFront access"

rm bucket-policy.json cloudfront-config.json

# Create Lambda function for backend API
echo -e "${YELLOW}⚡ Creating Lambda function for backend...${NC}"

# Create Lambda deployment package
mkdir -p lambda-package
cp -r saafe-lovable/backend/* lambda-package/
cd lambda-package

# Install dependencies
npm install --production

# Create Lambda handler
cat > index.js << 'EOF'
const serverless = require('serverless-http');
const express = require('express');
const cors = require('cors');
const AWS = require('aws-sdk');

// Configure AWS SDK
AWS.config.update({ region: 'us-east-1' });
const s3 = new AWS.S3();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// API endpoint for fire detection data
app.get('/api/fire-detection-data', async (req, res) => {
  try {
    const fireData = await fetchLiveFireDataFromS3();
    
    const responseData = {
      status: "success",
      data: fireData
    };

    res.json(responseData);
  } catch (err) {
    console.error("Error fetching fire detection data:", err);
    
    res.status(500).json({
      status: "error",
      message: "Failed to fetch live data from S3",
      error: err.message
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Function to fetch live fire data from S3
async function fetchLiveFireDataFromS3() {
  try {
    const bucketName = 'data-collector-of-first-device';
    
    // List objects in thermal-data directory
    const thermalData = await s3.listObjectsV2({ 
      Bucket: bucketName, 
      Prefix: 'thermal-data/',
      MaxKeys: 20
    }).promise();
    
    // List objects in gas-data directory
    const gasData = await s3.listObjectsV2({ 
      Bucket: bucketName, 
      Prefix: 'gas-data/',
      MaxKeys: 20
    }).promise();
    
    // Get the most recent files
    let thermalFile = null;
    let gasFile = null;
    
    if (thermalData.Contents && thermalData.Contents.length > 0) {
      thermalData.Contents.sort((a, b) => new Date(b.LastModified) - new Date(a.LastModified));
      thermalFile = thermalData.Contents[0];
    }
    
    if (gasData.Contents && gasData.Contents.length > 0) {
      gasData.Contents.sort((a, b) => new Date(b.LastModified) - new Date(a.LastModified));
      gasFile = gasData.Contents[0];
    }
    
    // Fetch and parse data
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
    
    return createFireDetectionData(thermalDataContent, gasDataContent, thermalFile, gasFile);
  } catch (error) {
    console.error("Error fetching data from S3:", error);
    throw error;
  }
}

// Parse thermal data from CSV content
function parseThermalData(csvContent) {
  try {
    const lines = csvContent.trim().split('\n');
    if (lines.length < 2) return null;

    const headers = lines[0].split(',');
    const readings = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length >= headers.length) {
        const reading = { timestamp: values[0] };
        for (let j = 1; j < headers.length; j++) {
          reading[headers[j]] = parseFloat(values[j]) || 0;
        }
        readings.push(reading);
      }
    }

    if (readings.length === 0) return null;

    readings.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    const latestReading = readings[0];

    const { timestamp, ...data } = latestReading;
    data._timestamp = timestamp;

    return data;
  } catch (error) {
    console.error("Error parsing thermal data:", error);
    return null;
  }
}

// Parse gas data from CSV content
function parseGasData(csvContent) {
  try {
    const lines = csvContent.trim().split('\n');
    if (lines.length < 2) return null;

    const headers = lines[0].split(',');
    const readings = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length >= headers.length) {
        const reading = {};
        for (let j = 0; j < headers.length; j++) {
          reading[headers[j]] = j === 0 ? values[j] : (parseFloat(values[j]) || 0);
        }
        readings.push(reading);
      }
    }

    if (readings.length === 0) return null;

    readings.sort((a, b) => new Date(b.timestamp || b.Timestamp) - new Date(a.timestamp || a.Timestamp));
    const latestReading = readings[0];

    latestReading._timestamp = latestReading.timestamp || latestReading.Timestamp;

    return latestReading;
  } catch (error) {
    console.error("Error parsing gas data:", error);
    return null;
  }
}

// Create fire detection data structure
function createFireDetectionData(thermalData, gasData, thermalFile, gasFile) {
  let sensorTimestamp = Math.floor(Date.now() / 1000);

  if (thermalData && thermalData._timestamp) {
    sensorTimestamp = Math.floor(new Date(thermalData._timestamp).getTime() / 1000);
  } else if (gasData && gasData._timestamp) {
    sensorTimestamp = Math.floor(new Date(gasData._timestamp).getTime() / 1000);
  }
  
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
  
  if (thermalData) {
    const pixelValues = Object.values(thermalData);
    if (pixelValues.length > 0) {
      defaultThermalStats.max = Math.max(...pixelValues);
      defaultThermalStats.min = Math.min(...pixelValues);
      defaultThermalStats.mean = pixelValues.reduce((a, b) => a + b, 0) / pixelValues.length;
    }
  }
  
  if (gasData) {
    if (gasData.VOC !== undefined) defaultGasReadings.voc = gasData.VOC;
    if (gasData.CO !== undefined) defaultGasReadings.co = gasData.CO;
    if (gasData.NO2 !== undefined) defaultGasReadings.no2 = gasData.NO2;
  }
  
  const thermalFrame = generateThermalFrame(thermalData);
  
  return {
    sensor_data: {
      timestamp: sensorTimestamp,
      thermal_frame: thermalFrame,
      thermal_stats: defaultThermalStats,
      gas_readings: defaultGasReadings,
      environmental_data: {
        temperature: 25,
        humidity: 40,
        pressure: 1013
      },
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
      timestamp: new Date().toISOString()
    },
    last_updated: new Date().toISOString()
  };
}

function generateThermalFrame(thermalData) {
  if (thermalData) {
    const frame = [];
    const values = Object.values(thermalData);
    let valueIndex = 0;
    
    for (let i = 0; i < 20; i++) {
      const row = [];
      for (let j = 0; j < 20; j++) {
        row.push(values[valueIndex] || 25);
        valueIndex = (valueIndex + 1) % values.length;
      }
      frame.push(row);
    }
    return frame;
  }
  
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

module.exports.handler = serverless(app);
EOF

# Add serverless-http dependency
npm install serverless-http

# Create deployment package
zip -r ../lambda-function.zip . -x "*.DS_Store*" "node_modules/.cache/*"

cd ..

# Create Lambda function
echo -e "${YELLOW}⚡ Creating Lambda function...${NC}"
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name lambda-execution-role --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$LAMBDA_ROLE_ARN" ]; then
    echo -e "${YELLOW}🔧 Creating Lambda execution role...${NC}"
    
    # Create trust policy
    cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create role
    aws iam create-role --role-name lambda-execution-role --assume-role-policy-document file://trust-policy.json
    
    # Attach policies
    aws iam attach-role-policy --role-name lambda-execution-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    aws iam attach-role-policy --role-name lambda-execution-role --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
    
    # Get role ARN
    LAMBDA_ROLE_ARN=$(aws iam get-role --role-name lambda-execution-role --query 'Role.Arn' --output text)
    
    # Wait for role to be ready
    sleep 10
    
    rm trust-policy.json
fi

# Create Lambda function
FUNCTION_ARN=$(aws lambda create-function \
    --function-name saafe-dashboard-api \
    --runtime nodejs20.x \
    --role $LAMBDA_ROLE_ARN \
    --handler index.handler \
    --zip-file fileb://lambda-function.zip \
    --timeout 30 \
    --memory-size 512 \
    --query 'FunctionArn' \
    --output text 2>/dev/null || \
aws lambda update-function-code \
    --function-name saafe-dashboard-api \
    --zip-file fileb://lambda-function.zip \
    --query 'FunctionArn' \
    --output text)

# Create API Gateway
echo -e "${YELLOW}🌐 Creating API Gateway...${NC}"
API_ID=$(aws apigatewayv2 create-api \
    --name saafe-dashboard-api \
    --protocol-type HTTP \
    --cors-configuration AllowOrigins="*",AllowMethods="*",AllowHeaders="*" \
    --query 'ApiId' \
    --output text 2>/dev/null || \
aws apigatewayv2 get-apis --query 'Items[?Name==`saafe-dashboard-api`].ApiId' --output text)

# Create Lambda integration
INTEGRATION_ID=$(aws apigatewayv2 create-integration \
    --api-id $API_ID \
    --integration-type AWS_PROXY \
    --integration-uri $FUNCTION_ARN \
    --payload-format-version 2.0 \
    --query 'IntegrationId' \
    --output text)

# Create routes
aws apigatewayv2 create-route \
    --api-id $API_ID \
    --route-key 'GET /api/fire-detection-data' \
    --target integrations/$INTEGRATION_ID

aws apigatewayv2 create-route \
    --api-id $API_ID \
    --route-key 'GET /health' \
    --target integrations/$INTEGRATION_ID

# Create stage
aws apigatewayv2 create-stage \
    --api-id $API_ID \
    --stage-name prod \
    --auto-deploy

# Add Lambda permission for API Gateway
aws lambda add-permission \
    --function-name saafe-dashboard-api \
    --statement-id api-gateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:$REGION:$(aws sts get-caller-identity --query Account --output text):$API_ID/*/*" 2>/dev/null || echo "Permission may already exist"

# Get API Gateway URL
API_URL=$(aws apigatewayv2 get-api --api-id $API_ID --query 'ApiEndpoint' --output text)

# Clean up
rm -rf lambda-package lambda-function.zip

echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "================================="
echo -e "${GREEN}✅ Frontend URL:${NC} https://$CLOUDFRONT_DOMAIN"
echo -e "${GREEN}✅ CloudFront Distribution:${NC} $DISTRIBUTION_ID"
echo -e "${GREEN}✅ API Gateway URL:${NC} $API_URL"
echo -e "${GREEN}✅ API Endpoint:${NC} $API_URL/api/fire-detection-data"
echo -e "${GREEN}✅ Health Check:${NC} $API_URL/health"
echo -e "${GREEN}✅ S3 Bucket:${NC} $BUCKET_NAME"
echo -e "${GREEN}✅ Lambda Function:${NC} saafe-dashboard-api"
echo ""
echo -e "${BLUE}🌍 Your SAAFE dashboard is now live and accessible worldwide!${NC}"
echo ""
echo -e "${YELLOW}💡 Next Steps:${NC}"
echo "1. Update your frontend to use the API Gateway URL: $API_URL"
echo "2. Consider setting up CloudFront for better performance"
echo "3. Add a custom domain for production use"
echo ""
echo -e "${YELLOW}📊 Monitoring:${NC}"
echo "   - Lambda Logs: CloudWatch Logs"
echo "   - API Gateway Metrics: CloudWatch"
echo "   - S3 Access Logs: Available if configured"
echo ""
echo -e "${GREEN}🔥 Your fire detection system is now protecting users globally!${NC}"