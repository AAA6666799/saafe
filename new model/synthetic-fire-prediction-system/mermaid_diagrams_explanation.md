# Detailed Explanation of Mermaid Diagrams

This document provides detailed explanations of the three Mermaid diagrams that illustrate the architecture and flow of the Synthetic Fire Prediction System.

**Important Note**: As you correctly pointed out, not all components shown in these diagrams are actually deployed to AWS. Only the machine learning model is deployed to an AWS SageMaker endpoint. The multi-agent system runs locally or on IoT devices, not in the cloud.

## 1. System Architecture Diagram

### Overview
This diagram shows the complete system architecture, illustrating how different components interact with each other in layers. It demonstrates the flow of data from hardware sensors through feature engineering, machine learning models, agent framework, and finally to output systems.

**Implementation Status**: Only the machine learning components are deployed to AWS SageMaker. The agent framework runs locally or on IoT devices.

### Components Explained

#### Hardware Layer
- **FLIR Lepton 3.5 Thermal Camera**: Captures thermal imaging data to detect heat signatures that may indicate fire
- **SCD41 CO₂ Sensor**: Measures carbon dioxide concentration, which increases during combustion
- **Environmental Sensors**: Provide additional context such as ambient temperature and humidity
- **Data Processing**: Initial processing of raw sensor data before feature extraction

#### Feature Engineering Layer
- **Feature Extraction**: Transforms raw sensor data into standardized features
- **18 Features**: The system uses 18 standardized features (15 thermal + 3 gas) that match the input format of our trained models

#### Machine Learning Layer
- **Model Ensemble Manager**: Central component that orchestrates all machine learning models
- **Random Forest Model**: Baseline classifier with 200 estimators for robust fire detection
- **XGBoost Model**: Gradient boosting classifier that improves accuracy over the baseline
- **LSTM Model**: Temporal pattern recognition model for sequence data analysis
- **Ensemble Classifier**: Combines predictions from individual models using weighted voting

#### Agent Framework Layer
- **Agent Coordinator**: Manages communication and coordination between all agents
- **Monitoring Agent**: Continuously monitors sensor health and data quality
- **Analysis Agent**: Primary consumer of ML models that processes sensor data
- **Response Agent**: Determines appropriate responses and generates alerts
- **Learning Agent**: Tracks performance and manages model retraining

#### Output Layer
- **Alerts & Notifications**: Generates real-time alerts when fire is detected
- **System Dashboard**: Provides visualization of system status and metrics
- **IoT Communication**: Handles communication with IoT devices and cloud services

### Data Flow
1. Raw data is collected from sensors
2. Data is processed and features are extracted
3. Features are fed to the model ensemble manager
4. Individual models make predictions
5. Ensemble classifier combines predictions
6. Analysis agent processes results
7. Agent coordinator routes information to appropriate agents
8. Output systems generate alerts and visualizations

## 2. ML Integration Flow Diagram

### Overview
This diagram focuses specifically on the machine learning integration flow, showing how data moves through the ML pipeline from collection to model deployment. It highlights the ensemble approach and the feedback loop for continuous learning.

**Implementation Status**: The ML training pipeline runs locally or on AWS SageMaker for training, but only the final model is deployed to AWS SageMaker endpoints. The feedback loop for continuous learning is conceptual and not fully automated in the current implementation.

### Components Explained

#### Data Collection and Preprocessing
- **Sensor Data Collection**: Continuous collection of thermal and gas sensor data
- **Feature Engineering**: Transformation of raw data into standardized features
- **Feature Validation**: Ensures all required features are present and valid

#### Model Ensemble Processing
- **Model Ensemble Manager**: Central orchestrator for all ML models
- **Individual Model Predictions**: Each model makes independent predictions
- **Random Forest**: Decision tree ensemble for baseline classification
- **XGBoost**: Gradient boosting machine for improved accuracy
- **LSTM**: Recurrent neural network for temporal pattern recognition
- **Model Confidence Scoring**: Calculates confidence for each model's predictions
- **Ensemble Decision**: Combines individual predictions using weighted voting
- **Confidence Assessment**: Evaluates overall confidence of ensemble prediction
- **Fire Detection Result**: Final fire detection decision with confidence score

#### Agent Integration
- **Analysis Agent Processing**: Primary consumer of ML predictions
- **Agent Coordinator**: Routes analysis results to appropriate agents
- **Response Agent**: Generates alerts based on detection results
- **Monitoring Agent**: Monitors system health and data quality
- **Learning Agent**: Tracks performance metrics for continuous improvement

#### Feedback Loop
- **Performance Analysis**: Learning agent analyzes model performance
- **Model Retraining Trigger**: Initiates retraining when performance degrades
- **New Training Job**: Creates new training job using AWS SageMaker
- **Model Deployment**: Deploys newly trained models to production

### Key Integration Points
1. The Analysis Agent is the primary consumer of ML predictions
2. All agents are connected to and use the AI models we created
3. The system implements continuous learning through performance monitoring
4. Model retraining is automated based on performance degradation

## 3. Agent-Model Integration Diagram

### Overview
This diagram specifically illustrates how the agent framework integrates with the machine learning models, showing the detailed flow from sensors through feature engineering to agent processing and the feedback loop for model improvement.

**Implementation Status**: The agent framework runs locally or on IoT devices and connects to the AWS SageMaker endpoint for ML predictions. The continuous learning feedback loop is not fully implemented in AWS.

### Components Explained

#### Sensor Layer
- **FLIR Lepton 3.5 Thermal Camera**: Provides thermal imaging data
- **SCD41 CO₂ Sensor**: Measures carbon dioxide concentration
- **Environmental Sensors**: Additional environmental context data
- **Data Preprocessing**: Initial processing of raw sensor data

#### Feature Engineering Layer
- **Feature Extraction**: Converts raw sensor data into standardized features
- **18-Feature Vector**: Standardized input format (15 thermal + 3 gas features) that matches our trained models

#### ML Model Layer
- **Model Ensemble Manager**: Central component managing all ML models
- **Random Forest Model**: Baseline classifier with 200 estimators
- **XGBoost Model**: Gradient boosting classifier for improved accuracy
- **LSTM Model**: Temporal pattern recognition for sequence data
- **Confidence Scoring**: Calculates confidence for each model's predictions
- **Weighted Voting**: Combines predictions using performance-based weights
- **Final Prediction & Confidence**: Ensemble result with overall confidence score

#### Agent Framework Processing
- **Agent Coordinator**: Central coordination point for all agents
- **Monitoring Agent**: Monitors sensor health and data quality
- **Analysis Agent**: Primary consumer of ML predictions
- **Response Agent**: Generates alerts and responses
- **Learning Agent**: Tracks performance and manages improvement
- **Pattern Analysis**: Analysis agent performs detailed pattern recognition
- **Rule-Based Detection**: Traditional rule-based fire detection methods
- **ML-Based Detection**: Machine learning-based fire detection using our models
- **Confidence Fusion**: Combines rule-based and ML-based confidence scores
- **Final Analysis Result**: Comprehensive analysis result for decision making

#### System Output
- **Alert Generation**: Creates alerts based on analysis results
- **Dashboard Updates**: Updates system dashboard with latest information
- **IoT Communication**: Communicates with IoT devices and cloud services

#### Feedback Loop
- **Performance Monitoring**: Learning agent continuously monitors model performance
- **Model Retraining Trigger**: Automatically triggers retraining when needed
- **New Training Job - SageMaker**: Uses AWS SageMaker for model retraining
- **Model Deployment**: Deploys improved models back to the ensemble manager

### Integration Details

#### Analysis Agent - ML Connection
The Analysis Agent is the primary integration point between the agent framework and ML models:
- Receives sensor data and extracts the same 18 features used in training
- Uses the Model Ensemble Manager to make predictions
- Combines ML predictions with rule-based analysis
- Calculates confidence scores that inform response actions

#### Agent Coordinator Role
The Agent Coordinator manages communication between agents and ensures ML predictions flow through the system:
- Routes analysis results from the Analysis Agent to the Response Agent
- Passes performance data from the Response Agent to the Learning Agent
- Maintains system state and metrics

#### Continuous Learning Loop
The Learning Agent connects model performance to retraining:
- Tracks accuracy, precision, and recall of ML predictions
- Identifies performance degradation that triggers retraining
- Uses AWS SageMaker to train new models with the same pipeline

### Configuration Integration
The system configuration explicitly connects agents and models:
- Defines the 18-feature input format that matches trained models
- Configures analysis thresholds based on model performance
- Sets up ensemble weights for combining model predictions
- Links agent behaviors to model confidence thresholds

## Key Integration Points Summary

### Direct ML Model Consumption
1. **Analysis Agent**: Directly consumes predictions from all trained models
2. **Feature Compatibility**: All agents use the same 18-feature input format used during training
3. **Confidence Integration**: Agents utilize ML confidence scores for decision making

### Agent Framework Integration
1. **Monitoring Agent**: Uses ML models to detect anomalies in sensor data
2. **Response Agent**: Bases alert generation on ML confidence scores
3. **Learning Agent**: Tracks ML performance and triggers retraining

### Continuous Improvement
1. **Performance Monitoring**: Real-time tracking of ML model accuracy
2. **Automated Retraining**: Performance degradation triggers new training jobs
3. **Seamless Deployment**: New models are automatically deployed to production

### AWS Integration
1. **SageMaker Training**: All model training uses AWS SageMaker
2. **S3 Storage**: Models and data are stored in S3 buckets
3. **IoT Core**: Device communication managed through AWS IoT Core

In the actual implementation, all agents in the system are fully connected to and utilize the AI models we created, with the Analysis Agent serving as the primary consumer of ML predictions. However, the agents run locally or on IoT devices, while only the ML models are deployed to AWS SageMaker endpoints. The communication between agents and the cloud-based ML model happens through API calls to the SageMaker endpoint.

## Actual AWS Implementation Status

### What's Deployed to AWS
1. **ML Model**: The trained ensemble model is deployed to AWS SageMaker endpoints
2. **S3 Storage**: Training data and model artifacts are stored in S3 buckets
3. **SageMaker Training Jobs**: Used for large-scale model training (100K+ samples)

### What's NOT Deployed to AWS
1. **Agent Framework**: Runs locally or on IoT devices
2. **Data Collection**: Happens at the edge/IoT device level
3. **Real-time Coordination**: Agent communication occurs locally
4. **Continuous Learning**: Retraining happens through manual pipeline execution

### Communication Flow
1. IoT devices collect sensor data locally
2. Agents process data and extract features locally
3. Agents make API calls to SageMaker endpoint for predictions
4. Results are processed by agents locally for decision making
5. Training data is periodically uploaded to S3 for model retraining
6. New models are deployed manually to SageMaker endpoints

This hybrid approach keeps real-time processing latency low while leveraging AWS for model training and hosting.