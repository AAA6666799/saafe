# SAAFE Dashboard - AWS Global Deployment Guide

## Overview
Deploy the SAAFE fire detection dashboard as a globally accessible website using AWS services.

## Deployment Options

### Option 1: AWS S3 + CloudFront (Recommended for Static Sites)
- **Cost**: ~$1-5/month for small traffic
- **Global**: CloudFront CDN for worldwide access
- **SSL**: Free SSL certificate via AWS Certificate Manager
- **Custom Domain**: Support for your own domain

### Option 2: AWS Amplify (Easiest)
- **Cost**: ~$1-15/month
- **CI/CD**: Automatic deployments from Git
- **Global**: Built-in CDN
- **SSL**: Automatic HTTPS

### Option 3: AWS Elastic Beanstalk (Full Stack)
- **Cost**: ~$10-50/month
- **Backend**: Can host both frontend and backend APIs
- **Scaling**: Auto-scaling capabilities

## Quick Start - S3 + CloudFront Deployment

This is the most cost-effective option for a React dashboard.