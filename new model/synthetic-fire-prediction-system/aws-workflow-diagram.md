# AWS Workflow Diagrams for Synthetic Fire Prediction System

## Data Flow Diagram

```mermaid
flowchart TD
    %% Data Generation Flow
    Start([Start]) --> SynConfig[Configure Synthetic Data Parameters]
    SynConfig --> BatchJob[AWS Batch Job]
    BatchJob --> |Thermal Data| S3Thermal[S3: Thermal Images]
    BatchJob --> |Gas Data| S3Gas[S3: Gas Sensor Data]
    BatchJob --> |Environmental Data| S3Env[S3: Environmental Data]
    
    %% Feature Engineering Flow
    S3Thermal --> GlueETL[AWS Glue ETL Jobs]
    S3Gas --> GlueETL
    S3Env --> GlueETL
    GlueETL --> EMRProcess[Amazon EMR Processing]
    EMRProcess --> |Extracted Features| S3Features[S3: Feature Storage]
    S3Features --> Athena[Amazon Athena]
    Athena --> |Feature Analysis| DataScientist([Data Scientist])
    
    %% Model Training Flow
    S3Features --> SMTraining[SageMaker Training Job]
    SMTraining --> |Model Artifacts| S3Models[S3: Model Artifacts]
    S3Models --> SMRegistry[SageMaker Model Registry]
    SMRegistry --> |Approved Models| SMEndpoint[SageMaker Endpoint]
    
    %% Agent System Flow
    S3Features --> |Real-time Features| MonitorAgent[Monitoring Agent - Lambda]
    MonitorAgent --> |Anomalies| SQSQueue[SQS Queue]
    SQSQueue --> AnalysisAgent[Analysis Agent - ECS]
    AnalysisAgent --> |Risk Assessment| SNSTopic[SNS Topic]
    SNSTopic --> ResponseAgent[Response Agent - Lambda]
    SNSTopic --> LearningAgent[Learning Agent - ECS]
    ResponseAgent --> |Alerts| Notification([Notification System])
    LearningAgent --> |Model Updates| SMTraining
    
    %% Inference Flow
    DataSource([Data Source]) --> |Raw Data| APIGateway[API Gateway]
    APIGateway --> |Process Request| StepFunction[Step Functions]
    StepFunction --> |Extract Features| LambdaProcess[Lambda Processor]
    LambdaProcess --> |Features| SMEndpoint
    SMEndpoint --> |Predictions| StepFunction
    StepFunction --> |Response| APIGateway
    APIGateway --> |Results| User([End User])
    
    %% Monitoring Flow
    BatchJob --> |Logs| CloudWatchLogs[CloudWatch Logs]
    GlueETL --> |Logs| CloudWatchLogs
    EMRProcess --> |Logs| CloudWatchLogs
    SMTraining --> |Logs| CloudWatchLogs
    MonitorAgent --> |Logs| CloudWatchLogs
    AnalysisAgent --> |Logs| CloudWatchLogs
    ResponseAgent --> |Logs| CloudWatchLogs
    LearningAgent --> |Logs| CloudWatchLogs
    SMEndpoint --> |Logs| CloudWatchLogs
    
    CloudWatchLogs --> |Metrics| CloudWatchDash[CloudWatch Dashboards]
    CloudWatchDash --> |Alerts| SNSAlerts[SNS Alerts]
    SNSAlerts --> Admin([Administrator])
```

## Training Workflow

```mermaid
flowchart TD
    Start([Start]) --> S3Data[S3: Feature Data]
    S3Data --> SMNotebook[SageMaker Notebook]
    SMNotebook --> |Experiment| SMExperiment[SageMaker Experiment]
    SMExperiment --> |Hyperparameters| SMTuning[SageMaker Hyperparameter Tuning]
    SMTuning --> |Best Parameters| SMTraining[SageMaker Training Job]
    SMTraining --> |Model Artifacts| S3Model[S3: Model Artifacts]
    S3Model --> SMModel[SageMaker Model]
    SMModel --> |Evaluation| SMBatch[SageMaker Batch Transform]
    SMBatch --> |Metrics| CloudWatch[CloudWatch Metrics]
    CloudWatch --> |Evaluation| SMRegistry[SageMaker Model Registry]
    SMRegistry --> Decision{Meets Criteria?}
    Decision --> |Yes| SMEndpoint[SageMaker Endpoint]
    Decision --> |No| SMNotebook
    SMEndpoint --> End([End])
```

## Synthetic Data Generation Workflow

```mermaid
flowchart TD
    Start([Start]) --> StepFunction[Step Functions]
    StepFunction --> ScenarioConfig[Configure Scenario]
    ScenarioConfig --> ParallelGen{Parallel Generation}
    
    ParallelGen --> ThermalBatch[AWS Batch: Thermal Generation]
    ParallelGen --> GasBatch[AWS Batch: Gas Generation]
    ParallelGen --> EnvBatch[AWS Batch: Environmental Generation]
    
    ThermalBatch --> |Images| S3Thermal[S3: Thermal Data]
    GasBatch --> |Time Series| S3Gas[S3: Gas Data]
    EnvBatch --> |Time Series| S3Env[S3: Environmental Data]
    
    S3Thermal --> LambdaValidate[Lambda: Data Validation]
    S3Gas --> LambdaValidate
    S3Env --> LambdaValidate
    
    LambdaValidate --> ValidationCheck{Valid Data?}
    ValidationCheck --> |Yes| S3Valid[S3: Validated Data]
    ValidationCheck --> |No| SNSNotify[SNS: Validation Failure]
    SNSNotify --> StepFunction
    
    S3Valid --> MetadataGlue[Glue: Metadata Extraction]
    MetadataGlue --> DynamoDB[DynamoDB: Metadata Catalog]
    DynamoDB --> End([End])
```

## Agent System Workflow

```mermaid
flowchart TD
    Start([Start]) --> DataStreams[Data Streams]
    DataStreams --> MonitorAgent[Monitoring Agent - Lambda]
    MonitorAgent --> AnomalyDetected{Anomaly Detected?}
    
    AnomalyDetected --> |No| ContinueMonitoring[Continue Monitoring]
    ContinueMonitoring --> MonitorAgent
    
    AnomalyDetected --> |Yes| SQSQueue[SQS Queue]
    SQSQueue --> AnalysisAgent[Analysis Agent - ECS]
    AnalysisAgent --> RiskAssessment[Risk Assessment]
    RiskAssessment --> RiskLevel{Risk Level}
    
    RiskLevel --> |Low| SNSLow[SNS: Low Risk]
    RiskLevel --> |Medium| SNSMedium[SNS: Medium Risk]
    RiskLevel --> |High| SNSHigh[SNS: High Risk]
    
    SNSLow --> ResponseAgentLow[Response Agent - Low Priority]
    SNSMedium --> ResponseAgentMed[Response Agent - Medium Priority]
    SNSHigh --> ResponseAgentHigh[Response Agent - High Priority]
    
    ResponseAgentLow --> |Log| CloudWatch[CloudWatch]
    ResponseAgentMed --> |Alert| Notification[Notification System]
    ResponseAgentHigh --> |Emergency| AlertSystem[Emergency Alert System]
    
    CloudWatch --> LearningAgent[Learning Agent - ECS]
    Notification --> LearningAgent
    AlertSystem --> LearningAgent
    
    LearningAgent --> ModelUpdate{Update Model?}
    ModelUpdate --> |Yes| TriggerTraining[Trigger Model Retraining]
    ModelUpdate --> |No| UpdateRules[Update Detection Rules]
    
    TriggerTraining --> End([End])
    UpdateRules --> End
```

## Deployment Pipeline

```mermaid
flowchart TD
    Start([Start]) --> CodeCommit[AWS CodeCommit/GitHub]
    CodeCommit --> |Code Changes| CodeBuild[AWS CodeBuild]
    CodeBuild --> |Build Artifacts| CodePipeline[AWS CodePipeline]
    
    CodePipeline --> DeployType{Deployment Type}
    
    DeployType --> |Infrastructure| CFNValidate[CloudFormation Validate]
    CFNValidate --> CFNDeploy[CloudFormation Deploy]
    
    DeployType --> |Application| ECRBuild[ECR Image Build]
    ECRBuild --> ECRPush[ECR Image Push]
    ECRPush --> ECSUpdate[ECS Service Update]
    
    DeployType --> |Model| SMBuild[SageMaker Model Build]
    SMBuild --> SMTest[SageMaker Model Test]
    SMTest --> SMDeploy[SageMaker Model Deploy]
    
    CFNDeploy --> |Success| NotifySuccess[SNS: Success Notification]
    ECSUpdate --> |Success| NotifySuccess
    SMDeploy --> |Success| NotifySuccess
    
    CFNDeploy --> |Failure| NotifyFailure[SNS: Failure Notification]
    ECSUpdate --> |Failure| NotifyFailure
    SMTest --> |Failure| NotifyFailure
    
    NotifySuccess --> End([End])
    NotifyFailure --> Rollback[Automatic Rollback]
    Rollback --> End