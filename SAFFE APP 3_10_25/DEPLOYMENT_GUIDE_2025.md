# SAAFE Fire Dashboard - Production Deployment Guide 2025

## 🚀 Deployment Status

**Environment:** saafe-fire-dashboard-prod  
**Platform:** Node.js 20 on Amazon Linux 2023  
**Region:** us-east-1 (US East - N. Virginia)  
**Instance Type:** t3.small  
**Load Balancer:** Application Load Balancer (ALB)  
**Deployment Date:** October 6, 2025

---

## 📋 Deployment Summary

This deployment replaces the previous environment (saafe-fire-dashboard-env-5) which is no longer accessible. The new environment has been configured with:

### ✅ Pre-configured Environment Variables
- `NODE_ENV=production`
- `PORT=8080`
- `AWS_REGION=us-east-1`
- `VITE_M1_URL=https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/`

### 🔐 Required Email Configuration (Post-Deployment)
The following environment variables need to be configured for email alerts:
- `SENDER_EMAIL` - Gmail address for sending alerts
- `SENDER_PASSWORD` - Gmail app password (16-character)
- `RECIPIENT_EMAIL` - Primary recipient for fire alerts

---

## 🔧 Post-Deployment Configuration

### Step 1: Configure Email Notifications

1. **Generate Gmail App Password:**
   - Go to [Google Account Security](https://myaccount.google.com/security)
   - Enable 2-Step Verification if not already enabled
   - Navigate to: Security → 2-Step Verification → App passwords
   - Select "Mail" and generate a new app password
   - Copy the 16-character password

2. **Set Environment Variables:**
   ```bash
   cd "SAFFE APP 3_10_25"
   eb setenv SENDER_EMAIL=your-email@gmail.com \
             SENDER_PASSWORD=your-16-char-app-password \
             RECIPIENT_EMAIL=alert-recipient@example.com
   ```

3. **Verify Configuration:**
   ```bash
   eb printenv
   ```

### Step 2: Verify Deployment

1. **Check Environment Status:**
   ```bash
   eb status
   ```

2. **Get Application URL:**
   ```bash
   eb status | grep CNAME
   ```

3. **Test Application:**
   - Open the URL in your browser
   - Verify all three components load:
     - Dashboard (SaafeLovable)
     - Fire Data Sender
     - Email Recipient Manager

4. **Test Email Functionality:**
   - Use the Email Recipient Manager to send a test email
   - Verify email delivery

---

## 📊 Application Architecture

### Frontend Components
- **React 19.1.1** with TypeScript
- **Vite** build system
- **Three Main Components:**
  1. SaafeLovable Dashboard - Real-time fire detection monitoring
  2. Fire Data Sender - Manual fire alert testing
  3. Email Recipient Manager - Configure alert recipients

### Backend Services
- **Node.js + Express** on port 8080
- **AWS S3 Integration** - Reads from `data-collector-of-first-device` bucket
- **Lambda Integration** - Calls ML models for fire prediction
- **Email Service** - Gmail SMTP for alert notifications

### AWS Services Used
- **Elastic Beanstalk** - Application hosting
- **EC2** - Compute instances (t3.small)
- **Application Load Balancer** - Traffic distribution
- **S3** - Sensor data storage
- **Lambda** - ML model endpoints
- **IAM** - Access management
- **CloudWatch** - Logging and monitoring

---

## 🛠️ Management Commands

### Deployment & Updates
```bash
# Deploy code changes
cd "SAFFE APP 3_10_25"
eb deploy

# View deployment status
eb status

# Open application in browser
eb open
```

### Monitoring & Logs
```bash
# View recent logs
eb logs

# Stream logs in real-time
eb logs --stream

# Check environment health
eb health

# View detailed health information
eb health --refresh
```

### Environment Management
```bash
# List all environments
eb list

# Switch between environments
eb use environment-name

# Restart application
eb restart

# Scale instances
eb scale 2  # Scale to 2 instances

# Update environment variables
eb setenv KEY=value

# View all environment variables
eb printenv
```

### SSH Access
```bash
# SSH into EC2 instance
eb ssh

# Once connected, useful commands:
# - View application logs: sudo tail -f /var/log/nodejs/nodejs.log
# - Check running processes: ps aux | grep node
# - View environment variables: env | grep NODE
```

---

## 🔍 Troubleshooting

### Application Not Loading
1. Check environment status: `eb status`
2. View logs: `eb logs`
3. Verify health: `eb health --refresh`
4. Check if build completed: Look for `backend/dist/` directory

### Email Alerts Not Working
1. Verify environment variables: `eb printenv | grep EMAIL`
2. Check Gmail app password is correct (16 characters, no spaces)
3. Test email from dashboard Email Recipient Manager
4. Check application logs: `eb logs | grep -i email`

### S3 Access Issues
1. Verify IAM role permissions in `.ebextensions/02_instance_profile.config`
2. Check bucket name: `data-collector-of-first-device`
3. Test S3 access from EC2: `eb ssh` then `aws s3 ls s3://data-collector-of-first-device/`

### Build Failures
1. Test build locally: `npm run build`
2. Check Node.js version compatibility
3. Verify all dependencies in `package.json`
4. Review build logs: `eb logs`

---

## 📈 Monitoring & Alerts

### CloudWatch Metrics
Monitor these key metrics in AWS CloudWatch:
- **Application Health** - Environment health status
- **Request Count** - Number of requests per minute
- **Response Time** - Average latency
- **Error Rate** - 4xx and 5xx errors
- **CPU Utilization** - Instance CPU usage
- **Network Traffic** - Inbound/outbound data

### Setting Up Alerts
```bash
# Create CloudWatch alarm for high CPU
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

---

## 🔒 Security Best Practices

### Current Security Measures
✅ IAM roles instead of hardcoded AWS credentials  
✅ Environment variables for sensitive data  
✅ S3 bucket access restricted via IAM policies  
✅ Application Load Balancer for traffic distribution  
✅ Security groups configured for minimal access  

### Recommended Enhancements
- [ ] Enable HTTPS with AWS Certificate Manager (ACM)
- [ ] Set up AWS WAF for additional protection
- [ ] Enable VPC Flow Logs
- [ ] Implement rate limiting on API endpoints
- [ ] Regular security audits with AWS Inspector
- [ ] Enable CloudTrail for audit logging

---

## 💰 Cost Optimization

### Current Monthly Costs (Estimated)
- **EC2 (t3.small):** ~$15-30/month
- **Application Load Balancer:** ~$20-25/month
- **Data Transfer:** ~$5-10/month
- **CloudWatch Logs:** ~$2-5/month
- **Total:** ~$42-70/month

### Cost Reduction Tips
1. Use Reserved Instances for 1-year commitment (save up to 40%)
2. Enable auto-scaling to scale down during low traffic
3. Set up billing alerts in AWS Console
4. Review and delete unused resources regularly
5. Use AWS Cost Explorer to identify optimization opportunities

---

## 🔄 Backup & Recovery

### Application Code
- Stored in Git repository
- Tagged releases for version control
- `.ebextensions` configuration backed up

### Configuration Backup
```bash
# Save current configuration
eb config save saafe-fire-dashboard-prod --cfg production-config

# Restore from saved configuration
eb create new-environment --cfg production-config
```

### Disaster Recovery
1. Keep `.ebextensions` and `.elasticbeanstalk` in version control
2. Document all environment variables
3. Maintain list of AWS resources (S3 buckets, Lambda functions)
4. Test recovery process quarterly

---

## 📞 Support & Resources

### AWS Documentation
- [Elastic Beanstalk Node.js](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create_deploy_nodejs.html)
- [EB CLI Reference](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3.html)
- [Environment Configuration](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/customize-containers.html)

### Project Documentation
- `AWS_DEPLOYMENT_PLAN.md` - Comprehensive deployment architecture
- `EMAIL_NOTIFICATION_GUIDE.md` - Email configuration details
- `ALERT_FLOW_TEST_REPORT.md` - Alert system testing

### Getting Help
1. Check application logs: `eb logs`
2. Review AWS Elastic Beanstalk console
3. Check CloudWatch metrics and logs
4. Contact AWS Support if needed

---

## ✅ Deployment Checklist

### Pre-Deployment
- [x] Application builds successfully (`npm run build`)
- [x] Backend serves frontend correctly
- [x] AWS credentials configured
- [x] EB CLI installed and configured
- [x] `.ebextensions` configuration files created
- [x] Environment variables documented

### Post-Deployment
- [ ] Environment status is "Ready" and health is "Green"
- [ ] Application accessible via provided URL
- [ ] All three components load correctly
- [ ] Email configuration completed
- [ ] Test email sent successfully
- [ ] S3 data fetching works
- [ ] Lambda model endpoints responding
- [ ] CloudWatch logs being collected
- [ ] Monitoring alerts configured

---

## 📝 Version History

### v2.0 - October 6, 2025
- New deployment to replace terminated environment
- Updated to Node.js 20 on Amazon Linux 2023
- Improved IAM role configuration
- Enhanced monitoring and logging
- Updated documentation

### v1.0 - September 16, 2025
- Initial production deployment
- Python platform (deprecated)
- Basic monitoring setup

---

**Last Updated:** October 6, 2025  
**Maintained By:** SAAFE Development Team  
**Status:** Active Deployment