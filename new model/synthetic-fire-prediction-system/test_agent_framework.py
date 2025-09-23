#!/usr/bin/env python3
"""
Test script to verify that the agent framework is working correctly.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_imports():
    """Test that all agent classes can be imported successfully."""
    try:
        # Test base agent imports
        from src.agents.base import Agent, AnalysisAgent, ResponseAgent, LearningAgent, MonitoringAgent
        print("✅ Base agent classes imported successfully")
        
        # Test concrete agent imports
        from src.agents.analysis.fire_pattern_analysis import FirePatternAnalysisAgent
        print("✅ FirePatternAnalysisAgent imported successfully")
        
        from src.agents.response.emergency_response import EmergencyResponseAgent
        print("✅ EmergencyResponseAgent imported successfully")
        
        from src.agents.learning.adaptive_learning import AdaptiveLearningAgent
        print("✅ AdaptiveLearningAgent imported successfully")
        
        # Test monitoring agents
        from src.agents.monitoring.alert_monitor import AlertMonitor
        from src.agents.monitoring.data_quality import DataQualityMonitor
        from src.agents.monitoring.model_performance import ModelPerformanceMonitor
        from src.agents.monitoring.system_health import SystemHealthMonitor
        print("✅ Monitoring agents imported successfully")
        
        # Test coordination
        from src.agents.coordination.multi_agent_coordinator import MultiAgentFireDetectionSystem
        print("✅ MultiAgentFireDetectionSystem imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_agent_instantiation():
    """Test that agent classes can be instantiated."""
    try:
        # Test base agent instantiation (should fail as it's abstract)
        from src.agents.base import AnalysisAgent
        try:
            agent = AnalysisAgent("test_agent", {})
            print("❌ AnalysisAgent instantiated (should be abstract)")
            return False
        except TypeError:
            print("✅ AnalysisAgent is properly abstract")
        
        # Test concrete agent instantiation
        from src.agents.analysis.fire_pattern_analysis import FirePatternAnalysisAgent
        agent = FirePatternAnalysisAgent("test_fire_pattern_agent", {
            "confidence_threshold": 0.7,
            "pattern_window_size": 50
        })
        print("✅ FirePatternAnalysisAgent instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Agent Framework")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    import_success = test_agent_imports()
    instantiation_success = test_agent_instantiation()
    
    print("\n" + "=" * 50)
    if import_success and instantiation_success:
        print("🎉 All tests passed! Agent framework is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())