# Saafe Fire Detection System - Architecture Diagram

```mermaid
graph TB
    subgraph "Saafe Fire Detection System"
        direction TB
        
        subgraph "Data Layer"
            direction LR
            S1[Thermal Sensors<br/>FLIR Lepton 3.5] --> H1[Hardware Abstraction Layer]
            S2[Gas Sensors<br/>SCD41 CO₂] --> H1
            S3[Environmental Sensors<br/>Temp, Humidity, Pressure] --> H1
            H1 --> D1[Synthetic Data<br/>Generator]
        end
        
        subgraph "Processing Layer"
            direction TB
            D1 --> F1[Feature<br/>Engineering]
            H1 --> F1
            F1 --> M1[Spatio-Temporal<br/>Transformer<br/>Model]
            F1 --> M2[Anti-Hallucination<br/>Engine]
            M1 --> E1[Ensemble<br/>System]
            M2 --> E1
        end
        
        subgraph "Intelligence Layer"
            direction TB
            E1 --> A1[Analysis Agent]
            A1 --> C1[Agent<br/>Coordinator]
            C1 --> MO1[Monitoring Agent]
            C1 --> R1[Response Agent]
            C1 --> L1[Learning Agent]
        end
        
        subgraph "Interface Layer"
            direction TB
            A1 --> U1[Dashboard<br/>UI]
            R1 --> N1[Notification<br/>System]
            L1 --> P1[Performance<br/>Monitor]
        end
        
        subgraph "External Services"
            direction TB
            N1 --> E2[SMS Gateway]
            N1 --> E3[Email Service]
            N1 --> E4[Push Notification<br/>Service]
            P1 --> E5[CloudWatch<br/>Monitoring]
            U1 --> E6[AWS S3<br/>Storage]
        end
    end
    
    style S1 fill:#4CAF50,stroke:#388E3C
    style S2 fill:#4CAF50,stroke:#388E3C
    style S3 fill:#4CAF50,stroke:#388E3C
    style H1 fill:#2196F3,stroke:#0D47A1
    style D1 fill:#2196F3,stroke:#0D47A1
    style F1 fill:#FF9800,stroke:#E65100
    style M1 fill:#9C27B0,stroke:#4A148C
    style M2 fill:#9C27B0,stroke:#4A148C
    style E1 fill:#9C27B0,stroke:#4A148C
    style A1 fill:#FF5722,stroke:#BF360C
    style MO1 fill:#FF5722,stroke:#BF360C
    style R1 fill:#FF5722,stroke:#BF360C
    style L1 fill:#FF5722,stroke:#BF360C
    style C1 fill:#FF5722,stroke:#BF360C
    style U1 fill:#607D8B,stroke:#263238
    style N1 fill:#607D8B,stroke:#263238
    style P1 fill:#607D8B,stroke:#263238
    style E2 fill:#795548,stroke:#3E2723
    style E3 fill:#795548,stroke:#3E2723
    style E4 fill:#795548,stroke:#3E2723
    style E5 fill:#795548,stroke:#3E2723
    style E6 fill:#795548,stroke:#3E2723
```

## Component Descriptions

### Data Layer
- **Thermal Sensors**: FLIR Lepton 3.5 thermal imaging cameras providing 384x288 pixel thermal data
- **Gas Sensors**: SCD41 CO₂ sensors with VOC detection capabilities
- **Environmental Sensors**: Temperature, humidity, and pressure sensors
- **Hardware Abstraction Layer**: Unified interface enabling seamless transition from synthetic to real hardware
- **Synthetic Data Generator**: Creates realistic sensor data for training and testing

### Processing Layer
- **Feature Engineering**: Extracts 18 meaningful features from raw sensor data (15 thermal + 3 gas)
- **Spatio-Temporal Transformer Model**: Deep learning model with 6 transformer layers for pattern recognition
- **Anti-Hallucination Engine**: Prevents false positives during cooking scenarios through ensemble validation
- **Ensemble System**: Combines multiple models for improved accuracy and robustness

### Intelligence Layer
- **Analysis Agent**: Primary consumer of ML models that processes sensor data and generates predictions
- **Agent Coordinator**: Manages communication and coordination between all agents
- **Monitoring Agent**: Continuously monitors sensor health and data quality
- **Response Agent**: Determines appropriate responses and generates alerts
- **Learning Agent**: Tracks performance and manages model retraining

### Interface Layer
- **Dashboard UI**: Real-time monitoring interface with sensor data visualization and alert management
- **Notification System**: Multi-channel alerting system (SMS, Email, Push notifications)
- **Performance Monitor**: System health and performance metrics tracking

### External Services
- **SMS Gateway**: SMS notification delivery service
- **Email Service**: Email notification delivery service
- **Push Notification Service**: Mobile push notification delivery
- **CloudWatch Monitoring**: AWS CloudWatch for system monitoring and alerting
- **AWS S3 Storage**: Amazon S3 for data storage and model artifacts

## Data Flow

1. **Sensor Data Collection**: Real or synthetic sensor data is collected through the Hardware Abstraction Layer
2. **Feature Engineering**: Raw sensor data is processed to extract meaningful features
3. **Model Inference**: Features are fed to the Spatio-Temporal Transformer and Anti-Hallucination models
4. **Ensemble Decision**: Model predictions are combined and validated through the ensemble system
5. **Agent Processing**: Analysis Agent processes results and coordinates with other agents
6. **Response Generation**: Response Agent determines appropriate alerts and notifications
7. **User Interface**: Dashboard displays real-time data and alerts to users
8. **Continuous Learning**: Learning Agent tracks performance and recommends improvements