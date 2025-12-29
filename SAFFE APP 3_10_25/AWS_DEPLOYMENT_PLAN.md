# AWS Deployment Plan for SAAFE Fire Detection Dashboard

## Executive Summary

This document provides a comprehensive deployment strategy for the SAAFE Fire Detection Dashboard application on AWS. The application is a full-stack Node.js/React application that integrates with AWS services (S3, Lambda) and requires secure environment variable management for email notifications and AWS credentials.

**Recommended Deployment Service:** AWS Elastic Beanstalk (Node.js Platform)

---

## Table of Contents

1. [Application Architecture Overview](#1-application-architecture-overview)
2. [AWS Service Comparison & Recommendation](#2-aws-service-comparison--recommendation)
3. [Deployment Architecture Design](#3-deployment-architecture-design)
4. [Environment Variables Management](#4-environment-variables-management)
5. [AWS Credentials & IAM Configuration](#5-aws-credentials--iam-configuration)
6. [SSL/HTTPS Configuration](#6-sslhttps-configuration)
7. [Required Configuration Files](#7-required-configuration-files)
8. [Step-by-Step Deployment Instructions](#8-step-by-step-deployment-instructions)
9. [Pre-Deployment Checklist](#9-pre-deployment-checklist)
10. [Post-Deployment Verification](#10-post-deployment-verification)
11. [Monitoring & Maintenance](#11-monitoring--maintenance)
12. [Troubleshooting Guide](#12-troubleshooting-guide)

---

## 1. Application Architecture Overview

### Current Stack
- **Frontend:** React 19.1.1 + TypeScript + Vite
- **Backend:** Node.js + Express (Port 8080)
- **Build Output:** Frontend builds to `dist/`, copied to `backend/dist/`
- **Deployment Model:** Backend serves frontend static files

### AWS Services Integration
- **S3 Bucket:** `data-collector-of-first-device` (sensor data storage)
- **Lambda Functions:** ML model endpoints for fire detection
- **AWS SDK:** Direct S3 access from backend
- **Region:** us-east-1

### External Dependencies
- **Gmail SMTP:** Email alerts for fire detection
- **Model Endpoints:** 4 different ML model APIs

### Key Application Features
1. Real-time fire detection data from S3
2. Multi-model AI predictions
3. Email alerting system
4. Device management dashboard
5. Live sensor data visualization

---

## 2. AWS Service Comparison & Recommendation

### Service Evaluation Matrix

| Criteria | Elastic Beanstalk | App Runner | Amplify Hosting | ECS Fargate |
|----------|------------------|------------|-----------------|-------------|
| **Ease of Deployment** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Good | ⭐⭐ Moderate |
| **Node.js Support** | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐⭐ Container | ⭐⭐⭐⭐ Native | ⭐⭐⭐⭐ Container |
| **Environment Variables** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Good |
| **IAM Integration** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Auto-scaling** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Limited | ⭐⭐⭐⭐⭐ Excellent |
| **SSL/HTTPS** | ⭐⭐⭐⭐⭐ Free ACM | ⭐⭐⭐⭐ Automatic | ⭐⭐⭐⭐⭐ Free ACM | ⭐⭐⭐⭐ Manual Setup |
| **Cost (Low Traffic)** | ⭐⭐⭐⭐ $20-40/mo | ⭐⭐⭐⭐ $25-50/mo | ⭐⭐⭐⭐⭐ $15-30/mo | ⭐⭐ $50-100/mo |
| **Monitoring** | ⭐⭐⭐⭐⭐ CloudWatch | ⭐⭐⭐⭐ CloudWatch | ⭐⭐⭐ Basic | ⭐⭐⭐⭐⭐ CloudWatch |
| **Static File Serving** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good |
| **Learning Curve** | ⭐⭐⭐⭐ Low | ⭐⭐⭐⭐ Low | ⭐⭐⭐⭐ Low | ⭐⭐ High |

### Recommendation: AWS Elastic Beanstalk

**Why Elastic Beanstalk is the Best Choice:**

1. **Perfect Fit for Architecture**
   - Native Node.js platform support (no containerization required)
   - Excellent for applications where backend serves frontend
   - Built-in load balancing and auto-scaling
   - Seamless integration with other AWS services

2. **Environment Management**
   - Secure environment variable storage
   - Easy secrets management
   - Multiple environment support (dev, staging, prod)
   - Configuration versioning

3. **IAM Integration**
   - Instance profiles for S3 access (no hardcoded credentials)
   - Fine-grained permission control
   - Automatic credential rotation

4. **Developer Experience**
   - Simple deployment via EB CLI or console
   - Rolling updates with zero downtime
   - Easy rollback capabilities
   - Comprehensive logging and monitoring

5. **Cost-Effective**
   - Pay only for underlying EC2 instances
   - Free tier eligible for small instances
   - Predictable pricing model

6. **SSL/HTTPS**
   - Free SSL certificates via AWS Certificate Manager
   - Automatic certificate renewal
   - Easy custom domain configuration

**Why Not Other Services:**

- **App Runner:** Requires containerization (unnecessary complexity), higher cost
- **Amplify Hosting:** Better for frontend-only or serverless apps, limited backend control
- **ECS Fargate:** Overkill for this application, higher complexity and cost

---

## 3. Deployment Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Internet                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Route 53 (Optional - Custom Domain)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Application Load Balancer (with SSL/TLS)             │
│                    (Port 443 HTTPS)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Elastic Beanstalk Environment                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         EC2 Instance(s) - Auto Scaling Group          │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │   Node.js Application (Port 8080)               │ │  │
│  │  │   ├── Express Backend                           │ │  │
│  │  │   └── React Frontend (Static Files)             │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                         │  │
│  │  Environment Variables (Encrypted):                   │  │
│  │  - SENDER_EMAIL, SENDER_PASSWORD                      │  │
│  │  - RECIPIENT_EMAIL, PORT                              │  │
│  │  - VITE_M1_URL                                        │  │
│  │                                                         │  │
│  │  IAM Instance Profile:                                │  │
│  │  - S3 Read Access (data-collector-of-first-device)   │  │
│  │  - Lambda Invoke (if needed)                          │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AWS Services Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   S3 Bucket  │  │    Lambda    │  │  CloudWatch  │      │
│  │  (Sensor     │  │  (ML Models) │  │  (Logs &     │      │
│  │   Data)      │  │              │  │  Metrics)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  External Services                           │
│              Gmail SMTP (Email Alerts)                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Application Load Balancer (ALB)
- **Purpose:** SSL termination, traffic distribution
- **Configuration:**
  - Listener: Port 443 (HTTPS)
  - Target: EC2 instances on port 8080
  - Health check: `/health` endpoint
  - SSL Certificate: AWS Certificate Manager (ACM)

#### 2. EC2 Instances (Managed by Elastic Beanstalk)
- **Instance Type:** t3.small or t3.medium (recommended)
- **Operating System:** Amazon Linux 2
- **Node.js Version:** 18.x (LTS)
- **Auto Scaling:**
  - Min: 1 instance
  - Max: 4 instances
  - Scaling trigger: CPU > 70% or Request count

#### 3. Security Groups
- **ALB Security Group:**
  - Inbound: Port 443 (HTTPS) from 0.0.0.0/0
  - Outbound: Port 8080 to EC2 instances
  
- **EC2 Security Group:**
  - Inbound: Port 8080 from ALB only
  - Outbound: All traffic (for S3, Lambda, Gmail SMTP)

#### 4. IAM Roles
- **Instance Profile Role:** Grants EC2 instances permissions to:
  - Read from S3 bucket `data-collector-of-first-device`
  - Write CloudWatch logs
  - (Optional) Invoke Lambda functions

---

## 4. Environment Variables Management

### Environment Variables Required

#### Backend Environment Variables
```bash
# Email Configuration (Required)
SENDER_EMAIL=ch.ajay1707@gmail.com
SENDER_PASSWORD=<gmail-app-password>
RECIPIENT_EMAIL=<alert-recipient-email>

# Server Configuration
PORT=8080
NODE_ENV=production

# AWS Configuration (Optional - use IAM roles instead)
# AWS_ACCESS_KEY_ID=<not-recommended>
# AWS_ACCESS_KEY_ID=<not-recommended>
AWS_REGION=us-east-1
```

#### Frontend Environment Variables
```bash
# Lambda Model Endpoint
VITE_M1_URL=https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/
```

### Elastic Beanstalk Environment Variable Configuration

#### Method 1: Via EB Console (Recommended for Secrets)
1. Navigate to Elastic Beanstalk Console
2. Select your environment
3. Go to Configuration → Software
4. Add environment properties:
   - `SENDER_EMAIL`: `ch.ajay1707@gmail.com`
   - `SENDER_PASSWORD`: `<your-gmail-app-password>`
   - `RECIPIENT_EMAIL`: `<recipient-email>`
   - `PORT`: `8080`
   - `NODE_ENV`: `production`
   - `AWS_REGION`: `us-east-1`
   - `VITE_M1_URL`: `https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/`

#### Method 2: Via `.ebextensions` Configuration File
Create `.ebextensions/environment.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:application:environment:
    NODE_ENV: production
    PORT: 8080
    AWS_REGION: us-east-1
    VITE_M1_URL: https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/
    # DO NOT put sensitive values here - use console or AWS Secrets Manager
```

#### Method 3: AWS Secrets Manager (Best Practice for Production)
1. Store sensitive values in AWS Secrets Manager
2. Grant EC2 instance profile permission to read secrets
3. Modify application to fetch secrets on startup

**Example Secrets Manager Setup:**
```bash
# Create secret
aws secretsmanager create-secret \
  --name saafe-app-secrets \
  --secret-string '{
    "SENDER_EMAIL":"ch.ajay1707@gmail.com",
    "SENDER_PASSWORD":"<gmail-app-password>",
    "RECIPIENT_EMAIL":"<recipient-email>"
  }' \
  --region us-east-1
```

**Modify server.js to fetch secrets:**
```javascript
// Add at the top of server.js
const AWS = require('aws-sdk');
const secretsManager = new AWS.SecretsManager({ region: 'us-east-1' });

async function loadSecrets() {
  try {
    const data = await secretsManager.getSecretValue({ 
      SecretId: 'saafe-app-secrets' 
    }).promise();
    const secrets = JSON.parse(data.SecretString);
    
    EMAIL_CONFIG.sender_email = secrets.SENDER_EMAIL;
    EMAIL_CONFIG.sender_password = secrets.SENDER_PASSWORD;
    EMAIL_CONFIG.recipient_email = secrets.RECIPIENT_EMAIL;
    
    console.log('Secrets loaded successfully from AWS Secrets Manager');
  } catch (error) {
    console.error('Error loading secrets:', error);
    // Fall back to environment variables
  }
}

// Call before starting server
loadSecrets().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
```

### Security Best Practices
1. **Never commit `.env` files** to version control
2. **Use IAM roles** instead of AWS access keys when possible
3. **Rotate Gmail app passwords** regularly
4. **Use AWS Secrets Manager** for production deployments
5. **Enable encryption** for environment variables in EB

---

## 5. AWS Credentials & IAM Configuration

### IAM Role Strategy (Recommended Approach)

**DO NOT use AWS access keys in environment variables.** Instead, use IAM instance profiles.

#### Step 1: Create IAM Policy for S3 Access

Create a policy named `SAAFE-S3-Access-Policy`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadSensorData",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::data-collector-of-first-device",
        "arn:aws:s3:::data-collector-of-first-device/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/aws/elasticbeanstalk/*"
    }
  ]
}
```

#### Step 2: Create IAM Role for EC2 Instances

Create a role named `SAAFE-EB-EC2-Role`:

1. **Trust Relationship:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

2. **Attach Policies:**
   - `SAAFE-S3-Access-Policy` (custom policy created above)
   - `AWSElasticBeanstalkWebTier` (AWS managed policy)
   - `AWSElasticBeanstalkWorkerTier` (AWS managed policy)
   - `AWSElasticBeanstalkMulticontainerDocker` (AWS managed policy)

#### Step 3: Create Instance Profile

```bash
# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name SAAFE-EB-Instance-Profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name SAAFE-EB-Instance-Profile \
  --role-name SAAFE-EB-EC2-Role
```

#### Step 4: Configure Elastic Beanstalk to Use Instance Profile

In `.ebextensions/iam.config`:
```yaml
option_settings:
  aws:autoscaling:launchconfiguration:
    IamInstanceProfile: SAAFE-EB-Instance-Profile
```

Or via EB Console:
1. Configuration → Security
2. Select `SAAFE-EB-Instance-Profile` as IAM instance profile

### AWS SDK Configuration in Application

The AWS SDK will automatically use the instance profile credentials. No code changes needed!

```javascript
// This automatically uses instance profile credentials
const AWS = require('aws-sdk');
AWS.config.update({ region: 'us-east-1' });
const s3 = new AWS.S3();
```

### Optional: Secrets Manager Access

If using AWS Secrets Manager, add this to the IAM policy:

```json
{
  "Sid": "ReadSecrets",
  "Effect": "Allow",
  "Action": [
    "secretsmanager:GetSecretValue"
  ],
  "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:saafe-app-secrets-*"
}
```

---

## 6. SSL/HTTPS Configuration

### Option 1: AWS Certificate Manager (Recommended - Free)

#### Step 1: Request SSL Certificate

1. **Via AWS Console:**
   - Navigate to AWS Certificate Manager (ACM)
   - Click "Request a certificate"
   - Choose "Request a public certificate"
   - Enter domain name (e.g., `saafe.yourdomain.com`)
   - Choose DNS validation (recommended) or Email validation
   - Complete validation process

2. **Via AWS CLI:**
```bash
aws acm request-certificate \
  --domain-name saafe.yourdomain.com \
  --validation-method DNS \
  --region us-east-1
```

#### Step 2: Validate Domain Ownership

For DNS validation:
1. ACM provides CNAME records
2. Add these records to your DNS provider
3. Wait for validation (usually 5-30 minutes)

#### Step 3: Configure Elastic Beanstalk Load Balancer

In `.ebextensions/https.config`:
```yaml
option_settings:
  aws:elbv2:listener:443:
    Protocol: HTTPS
    SSLCertificateArns: arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERTIFICATE_ID
    DefaultProcess: default
  aws:elbv2:listener:default:
    ListenerEnabled: false
```

Or via EB Console:
1. Configuration → Load Balancer
2. Add listener on port 443
3. Select HTTPS protocol
4. Choose your ACM certificate
5. Save configuration

#### Step 4: Redirect HTTP to HTTPS

Create `.ebextensions/http-redirect.config`:
```yaml
files:
  "/etc/nginx/conf.d/https_redirect.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      server {
        listen 8080;
        
        if ($http_x_forwarded_proto != 'https') {
          return 301 https://$host$request_uri;
        }
      }
```

### Option 2: Using Elastic Beanstalk Default Domain

If you don't have a custom domain, Elastic Beanstalk provides a default HTTPS endpoint:

- Format: `https://<environment-name>.<region>.elasticbeanstalk.com`
- SSL certificate automatically provided by AWS
- No additional configuration needed

### Option 3: Let's Encrypt (Alternative)

For custom domains without using ACM:

Create `.ebextensions/certbot.config`:
```yaml
packages:
  yum:
    certbot: []

container_commands:
  10_install_certbot:
    command: "sudo certbot certonly --standalone --non-interactive --agree-tos --email admin@yourdomain.com -d saafe.yourdomain.com"
    ignoreErrors: true
```

**Note:** ACM is strongly recommended over Let's Encrypt for AWS deployments.

---

## 7. Required Configuration Files

### File Structure

```
SAFFE APP 3_10_25/
├── .ebextensions/
│   ├── 01_environment.config
│   ├── 02_iam.config
│   ├── 03_https.config
│   ├── 04_http-redirect.config
│   └── 05_nodejs.config
├── .elasticbeanstalk/
│   └── config.yml
├── .ebignore
├── backend/
│   ├── server.js
│   ├── package.json
│   └── dist/ (created during build)
├── src/
├── package.json
└── buildspec.yml (optional - for CodePipeline)
```

### 1. `.ebextensions/01_environment.config`

```yaml
option_settings:
  aws:elasticbeanstalk:application:environment:
    NODE_ENV: production
    PORT: 8080
    AWS_REGION: us-east-1
    VITE_M1_URL: https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/
  
  aws:elasticbeanstalk:container:nodejs:
    NodeCommand: "npm start"
    NodeVersion: 18.19.0
  
  aws:elasticbeanstalk:environment:proxy:
    ProxyServer: nginx
```

### 2. `.ebextensions/02_iam.config`

```yaml
option_settings:
  aws:autoscaling:launchconfiguration:
    IamInstanceProfile: SAAFE-EB-Instance-Profile
    EC2KeyName: your-key-pair-name  # Optional - for SSH access
```

### 3. `.ebextensions/03_https.config`

```yaml
option_settings:
  # Load Balancer Configuration
  aws:elasticbeanstalk:environment:
    LoadBalancerType: application
  
  # HTTPS Listener (Port 443)
  aws:elbv2:listener:443:
    Protocol: HTTPS
    SSLCertificateArns: arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERT_ID
    DefaultProcess: default
  
  # HTTP Listener (Port 80) - Optional, for redirect
  aws:elbv2:listener:80:
    Protocol: HTTP
    DefaultProcess: default
```

### 4. `.ebextensions/04_http-redirect.config`

```yaml
files:
  "/etc/nginx/conf.d/https_redirect.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      server {
        listen 8080;
        
        location / {
          if ($http_x_forwarded_proto != 'https') {
            return 301 https://$host$request_uri;
          }
          proxy_pass http://127.0.0.1:8080;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
        }
      }

container_commands:
  01_reload_nginx:
    command: "sudo service nginx reload"
```

### 5. `.ebextensions/05_nodejs.config`

```yaml
option_settings:
  aws:elasticbeanstalk:container:nodejs:
    NodeCommand: "npm start"
    NodeVersion: 18.19.0
  
  aws:elasticbeanstalk:application:environment:
    NPM_USE_PRODUCTION: true

commands:
  01_node_install:
    command: "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    ignoreErrors: true
```

### 6. `.elasticbeanstalk/config.yml`

```yaml
branch-defaults:
  main:
    environment: saafe-production
    group_suffix: null

global:
  application_name: saafe-fire-detection
  branch: null
  default_ec2_keyname: your-key-pair
  default_platform: Node.js 18 running on 64bit Amazon Linux 2
  default_region: us-east-1
  include_git_submodules: true
  instance_profile: SAAFE-EB-Instance-Profile
  platform_name: null
  platform_version: null
  profile: null
  repository: null
  sc: git
  workspace_type: Application
```

### 7. `.ebignore`

```
# Development files
node_modules/
npm-debug.log
.env
.env.local
.env.*.local

# Build artifacts
dist/
build/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Testing
coverage/
.nyc_output/

# Documentation
*.md
docs/

# Backup files
*.backup
__MACOSX/
```

### 8. `Procfile` (Optional - for custom start command)

```
web: cd backend && npm start
```

### 9. `.npmrc` (Optional - for build optimization)

```
production=true
optional=false
```

### 10. `buildspec.yml` (Optional - for AWS CodePipeline CI/CD)

```yaml
version: 0.2

phases:
  pre_build:
    commands:
      - echo Installing dependencies...
      - npm install
      - cd backend && npm install && cd ..
  
  build:
    commands:
      - echo Building frontend...
      - npm run build:frontend
      - echo Copying build to backend...
      - npm run copy:dist
  
  post_build:
    commands:
      - echo Build completed successfully

artifacts:
  files:
    - '**/*'
  base-directory: '.'
```

---

## 8. Step-by-Step Deployment Instructions

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **EB CLI** installed (`pip install awsebcli`)
4. **Node.js 18.x** installed locally
5. **Git** installed

### Phase 1: Local Preparation

#### Step 1: Verify Application Builds Locally

```bash
# Navigate to project directory
cd "SAFFE APP 3_10_25"

# Install dependencies
npm install
cd backend && npm install && cd ..

# Build frontend
npm run build:frontend

# Copy to backend
npm run copy:dist

# Test backend locally
cd backend
PORT=8080 node server.js
```

Verify at `http://localhost:8080`

#### Step 2: Create Required Configuration Files

Create all files from Section 7 above:
- `.ebextensions/` directory with all config files
- `.elasticbeanstalk/config.yml`
- `.ebignore`

#### Step 3: Update package.json Scripts

Ensure `package.json` has:
```json
{
  "scripts": {
    "start": "cd backend && node server.js",
    "build": "npm run build:frontend && npm run copy:dist",
    "build:frontend": "vite build",
    "copy:dist": "mkdir -p backend/dist && cp -r dist/* backend/dist/",
    "postinstall": "npm run build"
  }
}
```

### Phase 2: AWS IAM Setup

#### Step 4: Create IAM Policy

```bash
# Create policy file
cat > saafe-s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::data-collector-of-first-device",
        "arn:aws:s3:::data-collector-of-first-device/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:*"
    }
  ]
}
EOF

# Create policy
aws iam create-policy \
  --policy-name SAAFE-S3-Access-Policy \
  --policy-document file://saafe-s3-policy.json \
  --region us-east-1
```

#### Step 5: Create IAM Role

```bash
# Create trust policy
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name SAAFE-EB-EC2-Role \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name SAAFE-EB-EC2-Role \
  --policy-arn arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier

aws iam attach-role-policy \
  --role-name SAAFE-EB-EC2-Role \
  --policy-arn arn:aws:iam::ACCOUNT_ID:policy/SAAFE-S3-Access-Policy

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name SAAFE-EB-Instance-Profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name SAAFE-EB-Instance-Profile \
  --role-name SAAFE-EB-EC2-Role
```

### Phase 3: Elastic Beanstalk Setup

#### Step 6: Initialize Elastic Beanstalk Application

```bash
# Initialize EB in project directory
eb init

# Follow prompts:
# - Select region: us-east-1
# - Application name: saafe-fire-detection
# - Platform: Node.js
# - Platform version: Node.js 18 running on 64bit Amazon Linux 2
# - SSH: Yes (optional, for debugging)
# - Key pair: Select or create new
```

#### Step 7: Create Elastic Beanstalk Environment

```bash
# Create environment
eb create saafe-production \
  --instance-type t3.small \
  --instance-profile SAAFE-EB-Instance-Profile \
  --envvars NODE_ENV=production,PORT=8080,AWS_REGION=us-east-1 \
  --elb-type application \
  --enable-spot

# This will:
# - Create EC2 instances
# - Set up load balancer
# - Configure auto-scaling
# - Deploy application
```

#### Step 8: Configure Environment Variables (Sensitive Data)

```bash
# Set environment variables via EB CLI
eb setenv \
  SENDER_EMAIL=ch.ajay1707@gmail.com \
  SENDER_PASSWORD=your-gmail-app-password \
  RECIPIENT_EMAIL=recipient@example.com \
  VITE_M1_URL=https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/
```

Or via AWS Console:
1. Go to Elastic Beanstalk Console
2. Select `saafe-production` environment
3. Configuration → Software → Environment properties
4. Add all environment variables
5. Apply changes

### Phase 4: SSL/HTTPS Configuration

#### Step 9: Request SSL Certificate (if using custom domain)

```bash
# Request certificate
aws acm request-certificate \
  --domain-name saafe.yourdomain.com \
  --validation-method DNS \
  --region us-east-1

# Note the CertificateArn from output
```

#### Step 10: Configure Load Balancer for HTTPS

```bash
# Update .ebextensions/03_https.config with your certificate ARN
# Then deploy:
eb deploy
```

Or via Console:
1. Configuration → Load Balancer
2. Add listener: Port 443, Protocol HTTPS
3. Select ACM certificate
4. Save

### Phase 5: Deployment

#### Step 11: Deploy Application

```bash
# Deploy to Elastic Beanstalk
eb deploy

# Monitor deployment
eb status
eb health
eb logs
```

#### Step 12: Verify Deployment

```bash
# Get environment URL
eb status

# Test endpoints
curl https://your-environment.us-east-1.elasticbeanstalk.com/health
curl https://your-environment.us-east-1.elasticbeanstalk.com/api/fire-detection-data
```

### Phase 6: Post-Deployment Configuration

#### Step 13: Configure Auto-Scaling

Via Console:
1. Configuration → Capacity
2. Set:
   - Min instances: 1
   - Max instances: 4
   - Scaling trigger: CPU > 70%

#### Step 14: Set Up CloudWatch Alarms

```bash
# Create alarm for high CPU
aws cloudwatch put-metric-alarm \
  --alarm-name saafe-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

#### Step 15: Configure Custom Domain (Optional)

1. In Route 53, create A record:
   - Name: `saafe.yourdomain.com`
   - Type: A - Alias
   - Target: Your EB environment load balancer

2. Update EB environment:
```bash
eb setenv DOMAIN=saafe.yourdomain.com
```

---

## 9. Pre-Deployment Checklist

### Code Preparation
- [ ] All dependencies listed in `package.json` and `backend/package.json`
- [ ] Application builds successfully locally (`npm run build`)
- [ ] Backend serves frontend correctly (`npm start`)
- [ ] All API endpoints tested locally
- [ ] No hardcoded credentials in code
- [ ] `.env` files added to `.gitignore`
- [ ] Error handling implemented for all AWS service calls

### AWS Account Setup
- [ ] AWS account created and verified
- [ ] AWS CLI installed and configured
- [ ] EB CLI installed (`pip install awsebcli`)
- [ ] IAM user has necessary permissions
- [ ] S3 bucket `data-collector-of-first-device` exists and has data
- [ ] Lambda functions are deployed and accessible

### IAM Configuration
- [ ] IAM policy `SAAFE-S3-Access-Policy` created
- [ ] IAM role `SAAFE-EB-EC2-Role` created
- [ ] Instance profile `SAAFE-EB-Instance-Profile` created
- [ ] Policies attached to role
- [ ] Role added to instance profile

### Configuration Files
- [ ] `.ebextensions/` directory created with all config files
- [ ] `.elasticbeanstalk/config.yml` created
- [ ] `.ebignore` file created
- [ ] `Procfile` created (if needed)
- [ ] All placeholder values replaced (ACCOUNT_ID, CERT_ID, etc.)

### Environment Variables
- [ ] Gmail app password generated (not regular password)
- [ ] All required environment variables documented
- [ ] Sensitive values NOT committed to Git
- [ ] Environment variables ready to set in EB

### SSL/HTTPS (if using custom domain)
- [ ] Domain name registered
- [ ] DNS access available
- [ ] ACM certificate requested
- [ ] Certificate validated
- [ ] Certificate ARN noted

### Testing
- [ ] Application tested locally on port 8080
- [ ] S3 access tested with IAM credentials
- [ ] Email sending tested
- [ ] All API endpoints return expected responses
- [ ] Frontend loads and displays data correctly

### Documentation
- [ ] Deployment plan reviewed
- [ ] Team members briefed on deployment process
- [ ] Rollback plan documented
- [ ] Monitoring strategy defined

---

## 10. Post-Deployment Verification

### Immediate Checks (Within 5 minutes)

#### 1. Environment Health
```bash
# Check environment status
eb status

# Expected output:
# Environment details for: saafe-production
# Status: Ready
# Health: Green
```

#### 2. Application Accessibility
```bash
# Get environment URL
EB_URL=$(eb status | grep "CNAME" | awk '{print $2}')

# Test health endpoint
curl https://$EB_URL/health
# Expected: "ok"

# Test API endpoint
curl https://$EB_URL/api/fire-detection-data
# Expected: JSON response with sensor data
```

#### 3. SSL Certificate
```bash
# Verify SSL
curl -I https://$EB_URL
# Check for: HTTP/2 200 or HTTP/1.1 200

# Detailed SSL check
openssl s_client -connect $EB_URL:443 -servername $EB_URL
```

### Functional Tests (Within 15 minutes)

#### 4. Frontend Loading
- Open browser to `https://your-environment-url`
- Verify dashboard loads
- Check browser console for errors
- Verify map displays correctly

#### 5. API Endpoints
```bash
# Test device list
curl https://$EB_URL/api/devices

# Test specific device
curl https://$EB_URL/api/devices/SAAFE-KITCHEN-001

# Test fire detection data
curl https://$EB_URL/api/fire-detection-data
```

#### 6. S3 Integration
- Check CloudWatch logs for S3 access
- Verify sensor data is being fetched
- Confirm no access denied errors

#### 7. Email Functionality
```bash
# Send test email via API
curl -X POST https://$EB_URL/api/send-test-email \
  -H "Content-Type: application/json" \
  -d '{"test_email":"your-email@example.com"}'

# Check email inbox for test message
```

### Performance Tests (Within 30 minutes)

#### 8. Load Testing
```bash
# Simple load test with Apache Bench
ab -n 1000 -c 10 https://$EB_URL/api/fire-detection-data

# Check response times and success rate
```

#### 9. Auto-Scaling Verification
- Monitor CloudWatch metrics
- Verify CPU, memory, network metrics are being collected
- Check auto-scaling group configuration

### Monitoring Setup (Within 1 hour)

#### 10. CloudWatch Logs
```bash
# View logs
eb logs

# Stream logs in real-time
eb logs --stream
```

#### 11. CloudWatch Metrics
- Navigate to CloudWatch Console
- Verify metrics for:
  - Application requests
  - Response times
  - Error rates
  - CPU utilization
  - Network traffic

#### 12. Alarms Configuration
- Verify CloudWatch alarms are active
- Test alarm notifications (if configured)

### Security Verification

#### 13. Security Group Rules
```bash
# List security groups
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=*saafe*" \
  --region us-east-1

# Verify:
# - ALB allows 443 from 0.0.0.0/0
# - EC2 allows 8080 from ALB only
```

#### 14. IAM Permissions
```bash
# Verify instance profile
aws iam get-instance-profile \
  --instance-profile-name SAAFE-EB-Instance-Profile

# Test S3 access from EC2 (via SSH)
aws s3 ls s3://data-collector-of-first-device/
```

### Checklist Summary

- [ ] Environment status is "Ready" and health is "Green"
- [ ] Application accessible via HTTPS
- [ ] SSL certificate valid and trusted
- [ ] Frontend loads without errors
- [ ] All API endpoints respond correctly
- [ ] S3 data fetching works
- [ ] Email sending works
- [ ] Load balancer distributes traffic
- [ ] Auto-scaling configured correctly
- [ ] CloudWatch logs are being collected
- [ ] CloudWatch metrics are being reported
- [ ] Security groups configured correctly
- [ ] IAM permissions working as expected
- [ ] No errors in application logs
- [ ] Performance meets expectations

---

## 11. Monitoring & Maintenance

### CloudWatch Monitoring

#### Key Metrics to Monitor

1. **Application Health**
   - Request count
   - Response time (latency)
   - HTTP 4xx errors
   - HTTP 5xx errors
   - Healthy host count

2. **Infrastructure Health**
   - CPU utilization
   - Memory utilization
   - Network in/out
   - Disk I/O
   - Instance health

3. **Custom Application Metrics**
   - Fire detection events
   - S3 fetch success/failure rate
   - Email send success/failure rate
   - Model prediction latency

#### Setting Up CloudWatch Dashboard

```bash
# Create custom dashboard
aws cloudwatch put-dashboard \
  --dashboard-name SAAFE-Dashboard \
  --dashboard-body file://dashboard-config.json
```

**dashboard-config.json:**
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ElasticBeanstalk", "EnvironmentHealth", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Environment Health"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ApplicationELB", "TargetResponseTime", {"stat": "Average"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Response Time"
      }
    }
  ]
}
```

### Log Management

#### Accessing Logs

```bash
# View recent logs
eb logs

# Download logs
eb logs --all

# Stream logs in real-time
eb logs --stream

# View specific log file
eb logs --log-group /aws/elasticbeanstalk/saafe-production/var/log/nodejs/nodejs.log
```

#### Log Retention

Configure in `.ebextensions/logging.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:cloudwatch:logs:
    StreamLogs: true
    DeleteOnTerminate: false
    RetentionInDays: 30
  
  aws:elasticbeanstalk:cloudwatch:logs:health:
    HealthStreamingEnabled: true
    DeleteOnTerminate: false
    RetentionInDays: 7
```

### Alerting

#### Critical Alerts

1. **High Error Rate**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name saafe-high-error-rate \
  --alarm-description "Alert when 5xx errors exceed threshold" \
  --metric-name HTTPCode_Target_5XX_Count \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:saafe-alerts
```

2. **Environment Health Degraded**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name saafe-health-degraded \
  --alarm-description "Alert when environment health is not OK" \
  --metric-name EnvironmentHealth \
  --namespace AWS/ElasticBeanstalk \
  --statistic Average \
  --period 300 \
  --threshold 15 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

3. **High CPU Usage**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name saafe-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

### Maintenance Tasks

#### Weekly Tasks
- [ ] Review CloudWatch logs for errors
- [ ] Check application performance metrics
- [ ] Verify auto-scaling is working correctly
- [ ] Review security group rules
- [ ] Check SSL certificate expiration (ACM auto-renews)

#### Monthly Tasks
- [ ] Review and optimize instance types
- [ ] Analyze cost and usage reports
- [ ] Update dependencies (`npm audit fix`)
- [ ] Review and update IAM policies
- [ ] Test disaster recovery procedures

#### Quarterly Tasks
- [ ] Perform load testing
- [ ] Review and update security policies
- [ ] Audit access logs
- [ ] Update documentation
- [ ] Review and optimize costs

### Backup Strategy

#### Application Code
- Store in Git repository (GitHub, CodeCommit)
- Tag releases: `git tag -a v1.0.0 -m "Production release"`

#### Configuration
- Export EB configuration: `eb config save saafe-production`
- Store `.ebextensions` in version control

#### Data
- S3 bucket versioning enabled
- Regular S3 bucket backups to another region

### Scaling Strategy

#### Vertical Scaling (Instance Size)
```bash
# Update instance type
eb scale 1 --instance-type t3.medium
```

#### Horizontal Scaling (Instance Count)
```bash
# Manual scaling
eb scale 3

# Auto-scaling configuration
# Edit .ebextensions/autoscaling.config
```

**autoscaling.config:**
```yaml
option_settings:
  aws:autoscaling:asg:
    MinSize: 1
    MaxSize: 4
  
  aws:autoscaling:trigger:
    MeasureName: CPUUtilization
    Statistic: Average
    Unit: Percent
    UpperThreshold: 70
    UpperBreachScaleIncrement: 1
    LowerThreshold: 30
    LowerBreachScaleIncrement: -1
```

---

## 12. Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Environment Creation Fails

**Symptoms:**
- `eb create` command fails
- Environment stuck in "Launching" state

**Possible Causes:**
1. Insufficient IAM permissions
2. Invalid instance profile
3. Service limits exceeded
4. Invalid configuration

**Solutions:**
```bash
# Check IAM permissions
aws iam get-user

# Verify instance profile exists
aws iam get-instance-profile --instance-profile-name SAAFE-EB-Instance-Profile

# Check service limits
aws service-quotas list-service-quotas --service-code elasticbeanstalk

# Review EB events
eb events --follow
```

#### Issue 2: Application Not Accessible

**Symptoms:**
- 502 Bad Gateway error
- Connection timeout
- Application URL not responding

**Possible Causes:**
1. Application not starting on correct port
2. Security group misconfiguration
3. Health check failing

**Solutions:**
```bash
# Check application logs
eb logs

# Verify port configuration
eb printenv | grep PORT

# Check security groups
aws ec2 describe-security-groups --filters "Name=group-name,Values=*saafe*"

# Test health endpoint
eb ssh
curl localhost:8080/health
```

#### Issue 3: S3 Access Denied

**Symptoms:**
- "Access Denied" errors in logs
- Unable to fetch sensor data
- 403 errors when accessing S3

**Possible Causes:**
1. IAM role missing S3 permissions
2. Bucket policy blocking access
3. Instance profile not attached

**Solutions:**
```bash
# Verify IAM role permissions
aws iam get-role-policy --role-name SAAFE-EB-EC2-Role --policy-name S3Access

# Check instance profile
eb ssh
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://data-collector-of-first-device/

# Update IAM policy if needed
aws iam put-role-policy --role-name SAAFE-EB-EC2-Role --policy-name S3Access --policy-document file://policy.json
```

#### Issue 4: Email Sending Fails

**Symptoms:**
- Email alerts not received
- "Authentication failed" errors
- SMTP connection errors

**Possible Causes:**
1. Incorrect Gmail app password
2. Gmail security settings blocking access
3. Environment variables not set correctly

**Solutions:**
```bash
# Verify environment variables
eb printenv | grep EMAIL

# Test email configuration
curl -X POST https://your-env.elasticbeanstalk.com/api/send-test-email \
  -H "Content-Type: application/json" \
  -d '{"test_email":"test@example.com"}'

# Check application logs for SMTP errors
eb logs | grep -i "email\|smtp"

# Generate new Gmail app password:
# 1. Go to Google Account settings
# 2. Security → 2-Step Verification → App passwords
# 3. Generate new password
# 4. Update environment variable
eb setenv SENDER_PASSWORD=new-app-password
```

#### Issue 5: High Memory Usage / Application Crashes

**Symptoms:**
- Application restarts frequently
- Out of memory errors
- Slow response times

**Possible Causes:**
1. Memory leaks in application
2. Insufficient instance size
3. Too many concurrent requests

**Solutions:**
```bash
# Check memory usage
eb ssh
top
free -m

# Increase instance size
eb scale 1 --instance-type t3.medium

# Add memory monitoring
# Create .ebextensions/monitoring.config
```

**monitoring.config:**
```yaml
option_settings:
  aws:elasticbeanstalk:healthreporting:system:
    SystemType: enhanced
  
  aws:elasticbeanstalk:cloudwatch:logs:
    StreamLogs: true
```

#### Issue 6: SSL Certificate Issues

**Symptoms:**
- "Certificate not trusted" warnings
- HTTPS not working
- Certificate validation errors

**Possible Causes:**
1. Certificate not validated
2. Wrong certificate ARN in configuration
3. Certificate in wrong region

**Solutions:**
```bash
# List certificates
aws acm list-certificates --region us-east-1

# Check certificate status
aws acm describe-certificate --certificate-arn YOUR_CERT_ARN --region us-east-1

# Verify load balancer configuration
aws elbv2 describe-listeners --load-balancer-arn YOUR_LB_ARN

# Update certificate in EB
# Edit .ebextensions/03_https.config with correct ARN
eb deploy
```

#### Issue 7: Build Failures

**Symptoms:**
- `npm install` fails during deployment
- Build process times out
- Missing dependencies

**Possible Causes:**
1. Network issues during npm install
2. Incompatible Node.js version
3. Missing build dependencies

**Solutions:**
```bash
# Specify Node.js version in .ebextensions
# Create .ebextensions/nodejs.config

# Use npm ci instead of npm install
# Update package.json scripts

# Increase deployment timeout
# Edit .ebextensions/timeout.config
```

**timeout.config:**
```yaml
option_settings:
  aws:elasticbeanstalk:command:
    DeploymentPolicy: Rolling
    Timeout: 600
```

#### Issue 8: Environment Variables Not Loading

**Symptoms:**
- Application uses default values
- "undefined" errors for environment variables
- Configuration not applied

**Possible Causes:**
1. Variables not set in EB environment
2. Typo in variable names
3. Application not reading variables correctly

**Solutions:**
```bash
# List all environment variables
eb printenv

# Set variables
eb setenv VAR_NAME=value

# Verify in application
eb ssh
echo $VAR_NAME

# Check application code reads process.env correctly
```

### Emergency Procedures

#### Rollback to Previous Version

```bash
# List application versions
aws elasticbeanstalk describe-application-versions \
  --application-name saafe-fire-detection

# Deploy previous version
aws elasticbeanstalk update-environment \
  --environment-name saafe-production \
  --version-label previous-version-label
```

#### Emergency Shutdown

```bash
# Terminate environment (preserves configuration)
eb terminate saafe-production --force

# Or scale down to 0 instances
eb scale 0
```

#### Quick Recovery

```bash
# Rebuild environment from saved configuration
eb create saafe-production-recovery --cfg saved-config-name

# Or restore from backup
eb restore saafe-production
```

### Getting Help

#### AWS Support Resources
- AWS Support Center: https://console.aws.amazon.com/support/
- Elastic Beanstalk Documentation: https://docs.aws.amazon.com/elasticbeanstalk/
- AWS Forums: https://forums.aws.amazon.com/

#### Useful Commands for Debugging

```bash
# Get detailed environment information
eb status --verbose

# View all events
eb events --follow

# SSH into instance
eb ssh

# Download all logs
eb logs --all --zip

# Check environment health
eb health --refresh

# View configuration
eb config
```

---

## Appendix A: Cost Estimation

### Monthly Cost Breakdown (Estimated)

#### Elastic Beanstalk (Free - only pay for resources)
- Service: $0

#### EC2 Instances (t3.small)
- 1 instance × 730 hours × $0.0208/hour = $15.18
- 2 instances (with auto-scaling) = $30.36

#### Application Load Balancer
- ALB: $16.20/month (fixed)
- LCU hours: ~$5-10/month (variable)
- Total: ~$21-26/month

#### Data Transfer
- First 1 GB: Free
- Next 9.999 TB: $0.09/GB
- Estimated: $5-10/month

#### CloudWatch
- Logs: $0.50/GB ingested
- Metrics: First 10 custom metrics free
- Estimated: $2-5/month

#### S3 (Existing bucket)
- Already in use, no additional cost

#### ACM SSL Certificate
- Free

### Total Estimated Monthly Cost
- **Minimum (1 instance):** $43-56/month
- **Average (2 instances):** $58-71/month
- **Maximum (4 instances):** $88-116/month

### Cost Optimization Tips
1. Use Reserved Instances for predictable workloads (up to 72% savings)
2. Enable auto-scaling to scale down during low traffic
3. Use Spot Instances for non-critical environments (up to 90% savings)
4. Set up billing alerts
5. Review and delete unused resources regularly

---

## Appendix B: Security Best Practices

### Application Security
1. **Never commit secrets to Git**
2. **Use environment variables for all configuration**
3. **Implement rate limiting on API endpoints**
4. **Validate all user inputs**
5. **Use HTTPS only (redirect HTTP to HTTPS)**
6. **Keep dependencies updated** (`npm audit fix`)
7. **Implement proper error handling** (don't expose stack traces)

### AWS Security
1. **Use IAM roles instead of access keys**
2. **Follow principle of least privilege**
3. **Enable MFA on AWS account**
4. **Use AWS Secrets Manager for sensitive data**
5. **Enable CloudTrail for audit logging**
6. **Regularly review security groups**
7. **Enable VPC Flow Logs**
8. **Use AWS WAF for additional protection**

### Network Security
1. **Restrict security group rules to minimum required**
2. **Use private subnets for EC2 instances**
3. **Enable encryption in transit (HTTPS)**
4. **Enable encryption at rest (S3, EBS)**
5. **Use VPC endpoints for AWS services**

---

## Appendix C: Useful Commands Reference

### EB CLI Commands

```bash
# Initialize EB
eb init

# Create environment
eb create environment-name

# Deploy application
eb deploy

# View status
eb status

# View logs
eb logs
eb logs --stream

# SSH into instance
eb ssh

# Scale environment
eb scale 2

# Set environment variables
eb setenv VAR=value

# View environment variables
eb printenv

# Terminate environment
eb terminate

# List environments
eb list

# Open in browser
eb open

# View events
eb events

# Health check
eb health
```

### AWS CLI Commands

```bash
# List EB applications
aws elasticbeanstalk describe-applications

# List environments
aws elasticbeanstalk describe-environments

# Update environment
aws elasticbeanstalk update-environment --environment-name name --option-settings file://options.json

# Create application version
aws elasticbeanstalk create-application-version --application-name app --version-label v1

# List S3 buckets
aws s3 ls

# List IAM roles
aws iam list-roles

# Describe security groups
aws ec2 describe-security-groups

# View CloudWatch logs
aws logs tail /aws/elasticbeanstalk/environment-name/var/log/nodejs/nodejs.log --follow
```

---

## Conclusion

This deployment plan provides a comprehensive guide for deploying the SAAFE Fire Detection Dashboard to AWS using Elastic Beanstalk. The recommended architecture leverages AWS best practices for security, scalability, and cost-effectiveness.

### Key Takeaways

1. **Elastic Beanstalk** is the optimal choice for this application
2. **IAM roles** should be used instead of hardcoded credentials
3. **Environment variables** must be managed securely
4. **SSL/HTTPS** is essential and easily configured with ACM
5. **Monitoring and logging** are critical for production operations

### Next Steps

1. Review this plan with your team
2. Complete the pre-deployment checklist
3. Set up AWS account and IAM roles
4. Follow the step-by-step deployment instructions
5. Verify deployment using the post-deployment checklist
6. Set up monitoring and alerting
7. Document any customizations or deviations from this plan

### Support

For questions or issues during deployment:
- Review the Troubleshooting Guide (Section 12)
- Check AWS Elastic Beanstalk documentation
- Contact AWS Support if needed

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-04  
**Author:** SAAFE Deployment Team  
**Status:** Ready for Implementation