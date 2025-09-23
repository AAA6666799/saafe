# Agent-Model Integration Architecture

## Integration Diagram

```mermaid
graph TB
    subgraph "Sensor Layer"
        A[FLIR Lepton 3.5<br/>Thermal Camera] --> B[Data<br/>Preprocessing]
        C[SCD41 CO₂<br/>Sensor] --> B
        D[Environmental<br/>Sensors] --> B
    end

    subgraph "Feature Engineering Layer"
        B --> E[Feature<br/>Extraction]
        E --> F[18-Feature<br/>Vector<br/>(15 Thermal + 3 Gas)]
    end

    subgraph "ML Model Layer"
        F --> G[Model Ensemble<br/>Manager]
        G --> H[Random Forest<br/>Model]
        G --> I[XGBoost<br/>Model]
        G --> J[LSTM<br/>Model]
        
        subgraph "Ensemble Processing"
            H --> K[Confidence<br/>Scoring]
            I --> K
            J --> K
            K --> L[Weighted<br/>Voting]
        end
        
        L --> M[Final<br/>Prediction<br/>& Confidence]
    end

    subgraph "Agent Framework"
        N[Agent<br/>Coordinator] --> O[Monitoring<br/>Agent]
        N --> P[Analysis<br/>Agent]
        N --> Q[Response<br/>Agent]
        N --> R[Learning<br/>Agent]
        
        subgraph "Analysis Agent Processing"
            P --> S[Pattern<br/>Analysis]
            S --> T[Rule-Based<br/>Detection]
            S --> U[ML-Based<br/>Detection]
            U --> M
            T --> V[Confidence<br/>Fusion]
            M --> V
            V --> W[Final<br/>Analysis<br/>Result]
        end
    end

    subgraph "System Output"
        W --> X[Alert<br/>Generation]
        W --> Y[Dashboard<br/>Updates]
        W --> Z[IoT<br/>Communication]
    end

    subgraph "Feedback Loop"
        R --> AA[Performance<br/>Monitoring]
        AA --> AB[Model<br/>Retraining<br/>Trigger]
        AB --> AC[New Training<br/>Job - SageMaker]
        AC --> G
    end

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style E fill:#fce4ec
    style F fill:#f8bbd0
    style G fill:#e8f5e8
    style H fill:#c8e6c9
    style I fill:#a5d6a7
    style J fill:#81c784
    style K fill:#66bb6a
    style L fill:#4caf50
    style M fill:#43a047
    style N fill:#fff3e0
    style O fill:#ffccbc
    style P fill:#ffb74d
    style Q fill:#ffccbc
    style R fill:#ffab91
    style S fill:#ffe0b2
    style T fill:#ffcc80
    style U fill:#ffb74d
    style V fill:#ffa726
    style W fill:#ff9800
    style X fill:#ffebee
    style Y fill:#ffcdd2
    style Z fill:#ef9a9a
    style AA fill:#bbdefb
    style AB fill:#90caf9
    style AC fill:#64b5f6
```

## Detailed Integration Documentation

### 1. Data Flow from Sensors to Models

#### Sensor Data Collection
The system begins with real-time data collection from:
- **FLIR Lepton 3.5 Thermal Camera**: Captures thermal images at 9Hz
- **SCD41 CO₂ Sensor**: Measures CO₂ concentration every 5 seconds
- **Environmental Sensors**: Additional context data (temperature, humidity, etc.)

#### Feature Engineering Pipeline
The [Feature Extraction](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/feature_engineering/feature_extractor.py#L25-L219) component transforms raw sensor data into a standardized 18-feature vector:
1. **Thermal Features (15)**: Statistical, spatial, and temporal characteristics from thermal data
2. **Gas Features (3)**: CO₂ concentration metrics including absolute value, delta, and velocity

This feature vector exactly matches the input format used during model training on the 251,000 synthetic samples.

### 2. Model Ensemble Integration

#### Model Ensemble Manager
The [ModelEnsembleManager](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/ml/ensemble/model_ensemble_manager.py#L37-L863) is the central component that connects the trained models to the agent framework:

```python
# Configuration showing the connection to trained models
model_configs = {
    'input_features': 18,  # Matches our training data
    'thermal_feature_count': 15,
    'gas_feature_count': 3,
    'enabled_models': ['random_forest', 'xgboost', 'lstm']
}
```

#### Individual Model Integration
Each trained model is loaded and managed by the ensemble:

1. **Random Forest Model**: Loaded from trained artifacts and configured for inference
2. **XGBoost Model**: Integrated if available in the runtime environment
3. **LSTM Model**: Temporal model for sequence pattern recognition

#### Prediction Process
When sensor data arrives:
1. Features are extracted and validated
2. The feature vector is passed to the [ModelEnsembleManager](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/ml/ensemble/model_ensemble_manager.py#L37-L863)
3. Each model makes an independent prediction
4. Confidence scores are calculated for each prediction
5. Predictions are combined using weighted voting based on model performance

### 3. Analysis Agent Connection

The [FirePatternAnalysisAgent](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/agents/analysis/fire_pattern_analysis.py#L17-L701) is the primary agent that consumes the ML model predictions:

#### Direct Model Integration
In the agent's [process](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/agents/analysis/fire_pattern_analysis.py#L71-L142) method, ML predictions are integrated with rule-based analysis:

```python
def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # Extract features matching our trained model input
    thermal_data = data.get('thermal_features', {})
    gas_data = data.get('gas_features', {})
    
    # Perform ML-based pattern analysis (this uses our trained models)
    pattern_results = self.analyze_pattern(data)  # Integrates with ModelEnsembleManager
    
    # Calculate confidence scores combining ML and rule-based methods
    confidence_score = self.calculate_confidence({
        'ml_analysis': pattern_results,
        'rule_based': self._rule_based_analysis(data)
    })
    
    return {
        'fire_detected': confidence_score >= self.confidence_threshold,
        'confidence_score': confidence_score,
        'ml_components': pattern_results
    }
```

#### Pattern Analysis Integration
The [analyze_pattern](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/agents/analysis/fire_pattern_analysis.py#L254-L285) method specifically integrates with the trained models:

```python
def analyze_pattern(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # This method would connect to our ModelEnsembleManager
    # to get predictions from the trained models
    ml_predictions = self.model_ensemble.predict(data['features'])
    
    # Combine with domain-specific pattern analysis
    thermal_patterns = self._analyze_thermal_patterns(data['thermal'])
    gas_patterns = self._analyze_gas_patterns(data['gas'])
    
    return {
        'ml_prediction': ml_predictions,
        'thermal_analysis': thermal_patterns,
        'gas_analysis': gas_patterns
    }
```

### 4. Agent Coordinator Orchestration

The [AgentCoordinator](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/agents/base.py#L407-L470) manages the flow of information between agents and the ML models:

#### Message Passing System
Agents communicate through a message passing system that carries ML predictions:

```python
# Analysis agent sends results to Response agent
message = Message(
    sender_id=self.agent_id,
    receiver_id='emergency_responder',
    message_type='fire_analysis_complete',
    content={
        'prediction': ml_prediction,
        'confidence': confidence_score,
        'features_used': feature_vector  # The same 18 features used in training
    }
)
self.coordinator.send_message(message)
```

#### Response Agent Utilization
The [EmergencyResponseAgent](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/agents/response/emergency_response.py#L17-L313) uses ML predictions to determine response levels:

```python
def handle_fire_alert(self, message: Message) -> None:
    content = message.content
    confidence = content['confidence']
    
    # Use ML confidence to determine response level
    if confidence > 0.9:
        response_level = 5  # Critical
    elif confidence > 0.8:
        response_level = 4  # High
    elif confidence > 0.7:
        response_level = 3  # Medium
    # ... etc
    
    self.generate_alerts(content, response_level)
```

### 5. Learning Agent Feedback Loop

The [AdaptiveLearningAgent](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/src/agents/learning/adaptive_learning.py#L17-L304) connects model performance to retraining:

#### Performance Tracking
```python
def track_performance(self, metrics: Dict[str, float]) -> None:
    # Track ML model performance
    self.performance_history.append({
        'timestamp': datetime.now(),
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'ml_model_confidence_correlation': metrics.get('confidence_correlation')
    })
    
    # Trigger retraining if performance degrades
    if self._should_retrain():
        self._trigger_model_retraining()
```

#### Retraining Integration
The learning agent can trigger new training jobs using the same AWS SageMaker pipeline we used for the original models:

```python
def _trigger_model_retraining(self) -> None:
    # Uses the same approach as our original training pipeline
    training_job = {
        'algorithm': 'ensemble',
        'input_data': self._prepare_retraining_data(),
        'feature_count': 18,  # Same as original
        'hyperparameters': self._optimize_hyperparameters()
    }
    
    # Submit to SageMaker (same service we used before)
    self.sagemaker_client.create_training_job(**training_job)
```

### 6. Configuration Integration

The system configuration explicitly connects agents and models through [iot_config.yaml](file:///Volumes/Ajay/saafe%20copy%203/new%20model/synthetic-fire-prediction-system/config/iot_config.yaml):

```yaml
# Feature configuration matching our trained models
feature_engineering:
  thermal:
    features:
      - t_mean
      - t_std
      # ... 13 more features (15 total)
  gas:
    features:
      - gas_val
      - gas_delta
      - gas_vel  # 3 total gas features

# Model configuration
models:
  input_features: 18  # Exactly matches our training
  ensemble:
    models: ["baseline", "temporal"]
    weights: [0.6, 0.4]

# Agent configuration using model outputs
agents:
  analysis:
    confidence_threshold: 0.7  # Based on model performance
    fire_pattern:
      enable_correlation_analysis: true  # Uses ML correlation findings
```

### 7. Real-time Inference Integration

The system uses the same model artifacts deployed in AWS SageMaker endpoints:

#### Model Deployment Consistency
- Training and inference use identical feature preprocessing
- Model versions are tracked and managed
- A/B testing capabilities for new model versions
- Rollback procedures for performance issues

#### Inference Pipeline
1. Sensor data → Feature extraction (18 features)
2. Features → Model ensemble manager
3. Model ensemble → Individual predictions + confidence scores
4. Ensemble combination → Final prediction
5. Prediction → Analysis agent for contextual interpretation
6. Analysis results → Response agent for action determination

### 8. Data Consistency Between Training and Inference

#### Feature Parity
The 18-feature input during inference exactly matches the training data:
- Same thermal features extracted from FLIR Lepton 3.5 data
- Same gas features derived from SCD41 CO₂ measurements
- Same preprocessing and normalization applied

#### Model Input Validation
Before making predictions, the system validates that all required features are present:

```python
def validate_model_input(self, features: Dict[str, Any]) -> bool:
    required_features = [
        # All 15 thermal features
        't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
        't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
        't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
        'tproxy_val', 'tproxy_delta', 'tproxy_vel',
        # All 3 gas features
        'gas_val', 'gas_delta', 'gas_vel'
    ]
    
    return all(feature in features for feature in required_features)
```

This comprehensive integration ensures that the AI models we trained are fully utilized by all agents in the system, creating a cohesive fire detection solution that leverages both machine learning intelligence and multi-agent coordination.