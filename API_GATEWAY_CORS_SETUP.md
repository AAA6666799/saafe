# API Gateway CORS Setup for Lambda Function

## Overview
This document provides instructions to set up API Gateway in front of your Lambda function to properly handle CORS requests.

## Current Issue
Your Lambda function URL `https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/` is returning CORS errors when accessed from web browsers.

## Solution: API Gateway with CORS

### Step 1: Create API Gateway

1. Go to AWS Console → API Gateway
2. Click "Create API" → "REST API" → "Build"
3. Choose "REST" protocol
4. Create API:
   - API name: `saafe-fire-detection-api`
   - Description: `API Gateway for SAAFE Fire Detection with CORS`
   - Endpoint Type: `Regional`

### Step 2: Create Resources and Methods

1. **Create Resource:**
   - Resource Name: `predict`
   - Resource Path: `predict`

2. **Create Method:**
   - Method Type: `POST`
   - Integration Type: `Lambda Function`
   - Lambda Function: Select your existing Lambda function
   - Use Lambda Proxy integration: `Yes`

3. **Create OPTIONS Method for CORS:**
   - Method Type: `OPTIONS`
   - Integration Type: `Mock`
   - Add these headers in Method Response:
     ```
     Access-Control-Allow-Origin: '*'
     Access-Control-Allow-Methods: 'POST,OPTIONS'
     Access-Control-Allow-Headers: 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'
     ```

### Step 3: Configure CORS

1. Go to your API → Resources → predict
2. Click "Enable CORS"
3. Configure:
   - Access-Control-Allow-Origin: `*`
   - Access-Control-Allow-Methods: `POST, OPTIONS`
   - Access-Control-Allow-Headers: `Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token`

### Step 4: Deploy API

1. Go to API Gateway → Resources
2. Click "Actions" → "Deploy API"
3. Create new stage: `prod`
4. Deploy

### Step 5: Get API Gateway URL

After deployment, you'll get a URL like:
`https://your-api-id.execute-api.us-east-1.amazonaws.com/prod`

The full endpoint will be:
`https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict`

## Update Server.js

Replace the Lambda URL in your `MODEL_URLS`:

```javascript
const MODEL_URLS = {
  saafe: "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict",
  // ... other URLs
};
```

## CloudFormation Template (Alternative)

If you prefer infrastructure as code:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'API Gateway with CORS for SAAFE Lambda'

Resources:
  ApiGatewayRestApi:
    Type: 'AWS::ApiGateway::RestApi'
    Properties:
      Name: saafe-fire-detection-api
      Description: API Gateway for SAAFE Fire Detection with CORS

  PredictResource:
    Type: 'AWS::ApiGateway::Resource'
    Properties:
      RestApiId: !Ref ApiGatewayRestApi
      ParentId: !GetAtt ApiGatewayRestApi.RootResourceId
      PathPart: 'predict'

  PredictMethod:
    Type: 'AWS::ApiGateway::Method'
    Properties:
      RestApiId: !Ref ApiGatewayRestApi
      ResourceId: !Ref PredictResource
      HttpMethod: 'POST'
      AuthorizationType: 'NONE'
      Integration:
        Type: 'AWS_PROXY'
        IntegrationHttpMethod: 'POST'
        Uri: !Sub 'arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${LambdaFunctionArn}/invocations'

  OptionsMethod:
    Type: 'AWS::ApiGateway::Method'
    Properties:
      RestApiId: !Ref ApiGatewayRestApi
      ResourceId: !Ref PredictResource
      HttpMethod: 'OPTIONS'
      AuthorizationType: 'NONE'
      Integration:
        Type: 'MOCK'
        IntegrationResponses:
          - StatusCode: '200'
            ResponseParameters:
              'method.response.header.Access-Control-Allow-Origin': "'*'"
              'method.response.header.Access-Control-Allow-Methods': "'POST,OPTIONS'"
              'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
        RequestTemplates:
          'application/json': '{"statusCode": 200}'

  PredictMethodResponse:
    Type: 'AWS::ApiGateway::MethodResponse'
    Properties:
      RestApiId: !Ref ApiGatewayRestApi
      ResourceId: !Ref PredictResource
      HttpMethod: 'POST'
      StatusCode: '200'
      ResponseParameters:
        'method.response.header.Access-Control-Allow-Origin': "'*'"

  OptionsMethodResponse:
    Type: 'AWS::ApiGateway::MethodResponse'
    Properties:
      RestApiId: !Ref ApiGatewayRestApi
      ResourceId: !Ref PredictResource
      HttpMethod: 'OPTIONS'
      StatusCode: '200'
      ResponseParameters:
        'method.response.header.Access-Control-Allow-Origin': "'*'"
        'method.response.header.Access-Control-Allow-Methods': "'POST,OPTIONS'"
        'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"

  ApiGatewayDeployment:
    Type: 'AWS::ApiGateway::Deployment'
    DependsOn:
      - PredictMethod
      - OptionsMethod
    Properties:
      RestApiId: !Ref ApiGatewayRestApi
      StageName: 'prod'

Parameters:
  LambdaFunctionArn:
    Type: String
    Description: ARN of the Lambda function

Outputs:
  ApiGatewayUrl:
    Description: API Gateway URL
    Value: !Sub 'https://${ApiGatewayRestApi}.execute-api.${AWS::Region}.amazonaws.com/prod'
    Export:
      Name: ApiGatewayUrl
```

## Testing

After setup, test your new API Gateway endpoint:
```bash
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

The endpoint should now work from web browsers without CORS errors.