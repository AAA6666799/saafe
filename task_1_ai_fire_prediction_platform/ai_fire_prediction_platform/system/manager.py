"""
System manager for synthetic fire prediction system
"""

import time
import threading
import numpy as np
from typing import Dict, Any, Optional
from ai_fire_prediction_platform.core.config import config_manager
from ai_fire_prediction_platform.core.interfaces import SensorData, FeatureVector, PredictionResult, RiskAssessment
from ai_fire_prediction_platform.feature_engineering.fusion import FeatureFusionEngine
from ai_fire_prediction_platform.models.ensemble import EnsembleModel
from ai_fire_prediction_platform.hardware.abstraction import S3HardwareInterface
from ai_fire_prediction_platform.alerting.engine import AlertEngine, create_alert_engine
from ai_fire_prediction_platform.alerting.notifications import NotificationManager, create_notification_manager


class SystemManager:
    """Main system manager that coordinates all components"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.is_running = False
        self.system_components = {}
        
        # Initialize core components
        self._initialize_components()
        
        # System state
        self.current_risk_assessment: Optional[RiskAssessment] = None
        self.last_prediction: Optional[PredictionResult] = None
        self.system_health = "UNKNOWN"
        
        # Alerting system
        self.alert_engine: Optional[AlertEngine] = None
        self.notification_manager: Optional[NotificationManager] = None
        self._initialize_alerting_system()
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize feature fusion engine
            self.feature_fusion = FeatureFusionEngine(
                self.config_manager.synthetic_data_config.__dict__
            )
            
            # Initialize ensemble model
            self.model = EnsembleModel(
                self.config_manager.model_config.__dict__
            )
            
            # Initialize hardware interface (S3 interface for real data)
            self.hardware_interface = S3HardwareInterface({
                's3_bucket': 'data-collector-of-first-device',
                'thermal_prefix': 'thermal-data/',
                'gas_prefix': 'gas-data/'
            })
            
            # Mark components as initialized
            self.system_components = {
                'feature_fusion': True,
                'model': True,
                'hardware_interface': self.hardware_interface.is_connected()
            }
            
            self.system_health = "INITIALIZED"
            
        except Exception as e:
            print(f"Error initializing system components: {e}")
            self.system_health = "ERROR"
    
    def _initialize_alerting_system(self):
        """Initialize alerting system components"""
        try:
            # Initialize alert engine
            self.alert_engine = create_alert_engine()
            
            # Initialize notification manager
            self.notification_manager = create_notification_manager()
            
            print("Alerting system initialized successfully")
        except Exception as e:
            print(f"Error initializing alerting system: {e}")
    
    def start(self):
        """Start the system"""
        if self.is_running:
            print("System is already running")
            return
        
        # Verify S3 connection
        if not self.hardware_interface.is_connected():
            print("Error: Cannot connect to S3 bucket")
            return False
        
        print("Starting Synthetic Fire Prediction System...")
        
        # Verify all components are initialized
        if not all(self.system_components.values()):
            print("Error: Not all system components are properly initialized")
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start main processing thread
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        print("System started successfully")
        return True
    
    def stop(self):
        """Stop the system"""
        if not self.is_running:
            print("System is not running")
            return
        
        print("Stopping Synthetic Fire Prediction System...")
        
        # Signal stop and wait for thread to finish
        self.stop_event.set()
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
        
        self.is_running = False
        self.system_health = "STOPPED"
        
        print("System stopped successfully")
    
    def _main_loop(self):
        """Main processing loop"""
        print("Entering main processing loop...")
        
        processing_interval = 1.0  # Process data every second
        
        while not self.stop_event.is_set():
            try:
                # Get sensor data from S3 hardware interface
                sensor_data = self.hardware_interface.get_sensor_data()
                
                if sensor_data:
                    # Extract and fuse features
                    feature_vector = self.feature_fusion.extract_features(sensor_data)
                    
                    # Make prediction
                    if feature_vector:
                        # Flatten all features for model input
                        all_features = []
                        if feature_vector.thermal_features is not None:
                            all_features.append(feature_vector.thermal_features)
                        if feature_vector.gas_features is not None:
                            all_features.append(feature_vector.gas_features)
                        if feature_vector.environmental_features is not None:
                            all_features.append(feature_vector.environmental_features)
                        if feature_vector.fusion_features is not None:
                            all_features.append(feature_vector.fusion_features)
                        
                        if all_features:
                            combined_features = np.concatenate(all_features)
                            
                            # Make prediction
                            fire_probability, confidence = self.model.predict(combined_features)
                            
                            # Create prediction result
                            self.last_prediction = PredictionResult(
                                timestamp=sensor_data.timestamp,
                                fire_probability=fire_probability,
                                confidence_score=confidence,
                                lead_time_estimate=0.0,  # Not implemented in this version
                                contributing_factors={},  # Not implemented in this version
                                model_ensemble_votes={}  # Not implemented in this version
                            )
                            
                            # Perform risk assessment (simplified)
                            self._perform_risk_assessment(fire_probability, confidence, sensor_data)
                            
                            # Generate alert using alert engine
                            if self.alert_engine:
                                alert = self.alert_engine.process_prediction(
                                    self.last_prediction, sensor_data, self.current_risk_assessment
                                )
                                
                                # Send notifications for critical alerts
                                if self.notification_manager and alert.alert_level.level >= 3:  # Elevated or Critical
                                    self.notification_manager.send_alert_notifications(alert)
                
                # Wait for next processing cycle
                if self.stop_event.wait(processing_interval):
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                if self.stop_event.wait(1.0):  # Wait a bit before retrying
                    break
        
        print("Exiting main processing loop")
    
    def _perform_risk_assessment(self, fire_probability: float, confidence: float, sensor_data: SensorData):
        """Perform risk assessment based on prediction"""
        # Determine risk level based on fire probability
        if fire_probability < 0.3:
            risk_level = "LOW"
        elif fire_probability < 0.6:
            risk_level = "MEDIUM"
        elif fire_probability < 0.85:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Determine contributing sensors (simplified)
        contributing_sensors = []
        if sensor_data.thermal_frame is not None:
            contributing_sensors.append("thermal")
        if sensor_data.gas_readings:
            contributing_sensors.extend(list(sensor_data.gas_readings.keys()))
        if sensor_data.environmental_data:
            contributing_sensors.extend(list(sensor_data.environmental_data.keys()))
        
        # Determine recommended actions based on risk level
        recommended_actions = []
        if risk_level == "LOW":
            recommended_actions = ["Continue monitoring"]
        elif risk_level == "MEDIUM":
            recommended_actions = ["Increase monitoring frequency", "Verify sensor readings"]
        elif risk_level == "HIGH":
            recommended_actions = ["Alert personnel", "Prepare response team", "Verify immediately"]
        elif risk_level == "CRITICAL":
            recommended_actions = ["EMERGENCY ALERT", "Evacuate area", "Contact fire department"]
        
        # Create risk assessment
        self.current_risk_assessment = RiskAssessment(
            timestamp=sensor_data.timestamp,
            risk_level=risk_level,
            fire_probability=fire_probability,
            confidence_level=confidence,
            contributing_sensors=contributing_sensors,
            recommended_actions=recommended_actions,
            escalation_required=(risk_level in ["HIGH", "CRITICAL"])
        )
        
        # Log the assessment if risk is elevated
        if risk_level in ["HIGH", "CRITICAL"]:
            self._log_risk_assessment()
    
    def _log_risk_assessment(self):
        """Log risk assessment to console"""
        if self.current_risk_assessment:
            print(f"\n🚨 RISK ASSESSMENT ALERT 🚨")
            print(f"Time: {self.current_risk_assessment.timestamp}")
            print(f"Risk Level: {self.current_risk_assessment.risk_level}")
            print(f"Fire Probability: {self.current_risk_assessment.fire_probability:.2f}")
            print(f"Confidence: {self.current_risk_assessment.confidence_level:.2f}")
            print(f"Recommended Actions: {', '.join(self.current_risk_assessment.recommended_actions)}")
            if self.current_risk_assessment.escalation_required:
                print("⚠️  ESCALATION REQUIRED ⚠️")
            print("-" * 40)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = {
            "is_running": self.is_running,
            "system_health": self.system_health,
            "components_status": self.system_components
        }
        
        # Add risk assessment information if available
        if self.current_risk_assessment:
            status["current_risk_level"] = self.current_risk_assessment.risk_level
            status["last_prediction_time"] = self.current_risk_assessment.timestamp
        
        # Add alert engine information if available
        if self.alert_engine:
            status["alert_engine_status"] = "active"
            stats = self.alert_engine.get_alert_statistics()
            status["current_alert_level"] = stats.get("current_level", "UNKNOWN")
        
        return status
    
    def get_latest_prediction(self) -> Optional[PredictionResult]:
        """Get the latest prediction result"""
        return self.last_prediction
    
    def get_current_risk_assessment(self) -> Optional[RiskAssessment]:
        """Get the current risk assessment"""
        return self.current_risk_assessment
    
    def get_latest_alert(self):
        """Get the latest alert"""
        if self.alert_engine and self.alert_engine.last_alert:
            return self.alert_engine.last_alert
        return None
    
    def get_recent_alerts(self, hours: int = 24):
        """Get recent alerts"""
        if self.alert_engine:
            return self.alert_engine.get_recent_alerts(hours)
        return []
    
    def get_alert_statistics(self, hours: int = 24):
        """Get alert statistics"""
        if self.alert_engine:
            return self.alert_engine.get_alert_statistics(hours)
        return {}
    
    def train_model(self, training_data, labels):
        """Train the ensemble model with provided data"""
        try:
            self.model.train(training_data, labels)
            print("Model trained successfully")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def save_model(self, path: str):
        """Save the trained model to disk"""
        try:
            self.model.save(path)
            print(f"Model saved to {path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str):
        """Load a trained model from disk"""
        try:
            self.model.load(path)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False