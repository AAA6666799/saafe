# Fire Detection Multi-Agent System

This folder contains all 24 agents for the comprehensive fire detection and response system.

## 📁 Folder Structure

```
agents/
├── base.py                          # Base agent classes (6 core agents)
├── __init__.py
│
├── architecture/                    # Agent Architecture & Management (5 agents)
│   ├── __init__.py
│   └── architecture.py              # AgentState, AgentConfiguration, AgentLifecycleManager, 
│                                    # AgentRegistry, AgentArchitecture
│
├── monitoring/                      # Monitoring Agents (1 primary + 3 additional)
│   ├── __init__.py
│   ├── system_health.py            # SystemHealthMonitor (772 lines)
│   ├── alert_monitor.py            # Alert monitoring
│   ├── data_quality.py             # Data quality monitoring
│   └── model_performance.py        # Model performance monitoring
│
├── analysis/                        # Analysis Agents (1 agent)
│   ├── __init__.py
│   └── fire_pattern_analysis.py    # FirePatternAnalysisAgent (702 lines)
│                                    # - FLIR Lepton 3.5 thermal analysis
│                                    # - SCD41 CO₂ sensor analysis
│                                    # - Multi-modal pattern fusion
│
├── decision/                        # Decision Agents (5 agents)
│   ├── __init__.py
│   ├── fire_detection.py           # FireDetectionAgent (339 lines)
│   ├── fire_classification.py      # FireClassificationAgent (471 lines)
│   ├── alert_generation.py         # AlertGenerationAgent (595 lines)
│   ├── flir_scd41_alert_generation.py  # FLIRSCD41AlertGenerationAgent (756 lines)
│   └── response_recommendation.py  # ResponseRecommendationAgent (820 lines)
│
├── response/                        # Response Agents (1 agent)
│   ├── __init__.py
│   └── emergency_response.py       # EmergencyResponseAgent (705 lines)
│                                    # - 5 response levels
│                                    # - Automated emergency coordination
│
├── learning/                        # Learning Agents (1 agent)
│   ├── __init__.py
│   └── adaptive_learning.py        # AdaptiveLearningAgent (734 lines)
│                                    # - Performance tracking
│                                    # - Error analysis
│                                    # - Improvement recommendations
│
├── coordination/                    # Coordination System (1 agent)
│   ├── __init__.py
│   └── multi_agent_coordinator.py  # MultiAgentFireDetectionSystem (764 lines)
│                                    # - 4-stage pipeline orchestration
│
└── communication/                   # Communication utilities
    └── __init__.py
```

## 🎯 Agent Categories

### **Base Architecture (6 agents)**
1. `Agent` - Abstract base class
2. `MonitoringAgent` - Base for monitoring agents
3. `AnalysisAgent` - Base for analysis agents
4. `ResponseAgent` - Base for response agents
5. `LearningAgent` - Base for learning agents
6. `AgentCoordinator` - Message routing and coordination

### **Architecture & Management (5 agents)**
7. `AgentState` - State management
8. `AgentConfiguration` - Configuration handling
9. `AgentLifecycleManager` - Lifecycle operations
10. `AgentRegistry` - Agent discovery and registration
11. `AgentArchitecture` - Overall system architecture

### **Specialized Agents (13 agents)**

#### Monitoring (4 agents)
12. `SystemHealthMonitor` - System health monitoring
13. `AlertMonitor` - Alert monitoring
14. `DataQualityMonitor` - Data quality checks
15. `ModelPerformanceMonitor` - Model performance tracking

#### Analysis (1 agent)
16. `FirePatternAnalysisAgent` - Multi-sensor fire pattern analysis

#### Decision (5 agents)
17. `FireDetectionAgent` - Fire detection decisions
18. `FireClassificationAgent` - Fire type classification
19. `AlertGenerationAgent` - Alert generation
20. `FLIRSCD41AlertGenerationAgent` - FLIR+SCD41 specific alerts
21. `ResponseRecommendationAgent` - Response planning

#### Response (1 agent)
22. `EmergencyResponseAgent` - Emergency coordination

#### Learning (1 agent)
23. `AdaptiveLearningAgent` - Continuous learning and improvement

#### Coordination (1 agent)
24. `MultiAgentFireDetectionSystem` - System orchestration

## 🔥 Key Features

### Hardware Integration
- **FLIR Lepton 3.5** thermal camera (160×120 resolution)
- **SCD41 CO₂ sensor** (400-40,000 ppm range)
- Real-time system health monitoring

### Sophisticated Algorithms
- Multi-modal sensor fusion
- Weighted confidence scoring
- Temporal pattern analysis
- Fire signature matching
- Adaptive learning with trend prediction

### Production-Ready
- Message-based inter-agent communication
- State persistence (save/load)
- Configuration validation
- Comprehensive error handling
- Threading support for real-time operation
- Performance metrics tracking

## 📊 Code Statistics

- **Total Lines**: ~6,000+ lines of functional code
- **Largest Agent**: ResponseRecommendationAgent (820 lines)
- **Most Complex**: MultiAgentFireDetectionSystem (764 lines)
- **Documentation**: Comprehensive docstrings throughout
- **Type Hints**: Full type annotation coverage

## 🚀 Usage

```python
from agents.coordination.multi_agent_coordinator import create_multi_agent_fire_system

# Create the multi-agent system
system = create_multi_agent_fire_system()

# Initialize all agents
system.initialize()

# Start the system
system.start()

# Process sensor data
sensor_data = {
    'flir': {'flir_lepton35': {...}},
    'scd41': {'scd41_co2': {...}}
}
results = system.process_sensor_data(sensor_data)

# Get system status
status = system.get_system_status()
```

## 📝 License

Part of the SAAFE Fire Detection System