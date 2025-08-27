"""
Main application entry point for Saafe IoT-based Predictive Fire Detection System.
"""

import logging
import sys
import asyncio
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from saafe_mvp.models.transformer import SpatioTemporalTransformer, ModelConfig
from saafe_mvp.models.model_manager import ModelManager
from saafe_mvp.data.iot_data_loader import IoTFireDataset
from saafe_mvp.services.notification_manager import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class IoTFireDetectionSystem:
    """Main IoT-based fire detection system."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        """
        Initialize the IoT fire detection system.
        
        Args:
            model_path (str): Path to trained model
            device (torch.device): Device for inference
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model manager
        self.model_manager = ModelManager(device=self.device)
        
        # Load or create model
        if model_path and Path(model_path).exists():
            success, message = self.model_manager.load_model(model_path)
            if not success:
                logger.warning(f"Failed to load model: {message}")
                self._create_fallback_model()
        else:
            self._create_fallback_model()
        
        # Get the model
        self.model = self.model_manager.get_model()
        if self.model is None:
            raise RuntimeError("No model available for inference")
        
        # Initialize notification system
        self.notification_manager = NotificationManager()
        
        # Area configuration
        self.area_config = {
            'kitchen': {
                'name': 'Kitchen',
                'sensor_type': 'VOC + ML',
                'vendor': 'Honeywell MiCS',
                'lead_time_range': 'Minutes to Hours',
                'critical_threshold': 250.0
            },
            'electrical': {
                'name': 'Electrical Panel',
                'sensor_type': 'Arc Detection',
                'vendor': 'Ting/Eaton AFDD',
                'lead_time_range': 'Days to Weeks',
                'critical_threshold': 8
            },
            'laundry_hvac': {
                'name': 'Laundry/HVAC',
                'sensor_type': 'Thermal + Current',
                'vendor': 'Honeywell Thermal',
                'lead_time_range': 'Hours to Days',
                'critical_threshold': {'temp': 60.0, 'current': 2.5}
            },
            'living_bedroom': {
                'name': 'Living/Bedroom',
                'sensor_type': 'Aspirating Smoke',
                'vendor': 'Xtralis VESDA-E',
                'lead_time_range': 'Minutes to Hours',
                'critical_threshold': 15.0
            },
            'basement_storage': {
                'name': 'Basement/Storage',
                'sensor_type': 'Environmental IoT',
                'vendor': 'Bosch/Airthings',
                'lead_time_range': 'Hours to Days',
                'critical_threshold': {'gas': 30.0, 'temp_trend': 5.0}
            }
        }
        
        logger.info("IoT Fire Detection System initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Areas monitored: {list(self.area_config.keys())}")
    
    def _create_fallback_model(self):
        """Create a fallback model if no trained model is available."""
        logger.info("Creating fallback IoT model...")
        config = ModelConfig()
        success, message = self.model_manager.create_fallback_model()
        if not success:
            raise RuntimeError(f"Failed to create fallback model: {message}")
    
    def predict_fire_risk(self, area_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Predict fire risk from area sensor data.
        
        Args:
            area_data (Dict[str, torch.Tensor]): Area-specific sensor data
            
        Returns:
            Dict containing predictions and risk assessment
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            device_data = {}
            for area_name, data in area_data.items():
                device_data[area_name] = data.to(self.device)
            
            # Get model predictions
            outputs = self.model(device_data)
            
            # Process predictions
            lead_time_probs = torch.softmax(outputs['lead_time_logits'], dim=1)
            lead_time_pred = torch.argmax(lead_time_probs, dim=1)
            
            area_risks = outputs['area_risks'].cpu().numpy()
            time_to_ignition = outputs['time_to_ignition'].cpu().numpy()
            
            # Convert to interpretable results
            lead_time_categories = ['Immediate', 'Hours', 'Days', 'Weeks']
            predicted_category = lead_time_categories[lead_time_pred.item()]
            
            # Area-specific risk assessment
            area_risk_dict = {}
            for i, area_name in enumerate(self.area_config.keys()):
                area_risk_dict[area_name] = {
                    'risk_probability': float(area_risks[0, i]),
                    'risk_level': self._categorize_risk(area_risks[0, i]),
                    'sensor_info': self.area_config[area_name]
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_lead_time': predicted_category,
                'lead_time_confidence': float(lead_time_probs.max()),
                'time_to_ignition_hours': float(time_to_ignition[0, 0]),
                'area_risks': area_risk_dict,
                'requires_action': self._requires_immediate_action(area_risks[0], predicted_category),
                'recommended_actions': self._get_recommended_actions(area_risks[0], predicted_category)
            }
    
    def _categorize_risk(self, risk_prob: float) -> str:
        """Categorize risk probability into levels."""
        if risk_prob < 0.3:
            return 'Low'
        elif risk_prob < 0.6:
            return 'Medium'
        elif risk_prob < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def _requires_immediate_action(self, area_risks: list, lead_time_category: str) -> bool:
        """Determine if immediate action is required."""
        # Immediate action if any area has critical risk or lead time is immediate
        critical_risk = any(risk > 0.8 for risk in area_risks)
        immediate_timeline = lead_time_category == 'Immediate'
        
        return critical_risk or immediate_timeline
    
    def _get_recommended_actions(self, area_risks: list, lead_time_category: str) -> list:
        """Get recommended actions based on risk assessment."""
        actions = []
        
        area_names = list(self.area_config.keys())
        
        for i, risk in enumerate(area_risks):
            area_name = area_names[i]
            area_info = self.area_config[area_name]
            
            if risk > 0.8:  # Critical risk
                if area_name == 'kitchen':
                    actions.append(f"🔥 CRITICAL: Check kitchen appliances immediately - possible overheating detected")
                elif area_name == 'electrical':
                    actions.append(f"⚡ CRITICAL: Electrical panel inspection required - arc faults detected")
                elif area_name == 'laundry_hvac':
                    actions.append(f"🌡️ CRITICAL: Check HVAC/dryer systems - overheating detected")
                elif area_name == 'living_bedroom':
                    actions.append(f"💨 CRITICAL: Smoke detected in living areas - investigate immediately")
                elif area_name == 'basement_storage':
                    actions.append(f"🏠 CRITICAL: Basement environmental hazard - check for chemical issues")
            
            elif risk > 0.6:  # High risk
                actions.append(f"⚠️ HIGH: Monitor {area_info['name']} closely - elevated risk detected")
            
            elif risk > 0.3:  # Medium risk
                actions.append(f"📊 MEDIUM: {area_info['name']} showing elevated readings - routine check recommended")
        
        # Add timeline-specific actions
        if lead_time_category == 'Immediate':
            actions.insert(0, "🚨 IMMEDIATE ACTION REQUIRED - Fire risk detected")
        elif lead_time_category == 'Hours':
            actions.append("⏰ Schedule inspection within next few hours")
        elif lead_time_category == 'Days':
            actions.append("📅 Schedule maintenance within next few days")
        elif lead_time_category == 'Weeks':
            actions.append("📋 Add to routine maintenance schedule")
        
        return actions if actions else ["✅ All systems normal - continue monitoring"]
    
    async def monitor_continuous(self, data_source: str = "synthetic datasets", 
                                interval_seconds: int = 60):
        """
        Continuous monitoring mode.
        
        Args:
            data_source (str): Data source for monitoring
            interval_seconds (int): Monitoring interval
        """
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        # Load dataset for simulation
        try:
            dataset = IoTFireDataset(data_source, sequence_length=60)
            logger.info(f"Loaded dataset with {len(dataset)} sequences")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
        
        sequence_idx = 0
        
        while True:
            try:
                # Get next sequence (simulate real-time data)
                if sequence_idx >= len(dataset):
                    sequence_idx = 0  # Loop back to start
                
                area_data, labels = dataset[sequence_idx]
                sequence_idx += 1
                
                # Add batch dimension
                batched_data = {}
                for area_name, data in area_data.items():
                    batched_data[area_name] = data.unsqueeze(0)
                
                # Get predictions
                predictions = self.predict_fire_risk(batched_data)
                
                # Log results
                logger.info(f"Monitoring Update:")
                logger.info(f"  Lead Time: {predictions['overall_lead_time']}")
                logger.info(f"  Time to Ignition: {predictions['time_to_ignition_hours']:.1f} hours")
                
                # Check for alerts
                if predictions['requires_action']:
                    logger.warning("🚨 ALERT TRIGGERED")
                    for action in predictions['recommended_actions']:
                        logger.warning(f"  {action}")
                    
                    # Send notifications (if configured)
                    try:
                        await self.notification_manager.send_alert(
                            title="Fire Risk Alert",
                            message=f"Lead time: {predictions['overall_lead_time']}",
                            priority="high" if predictions['overall_lead_time'] == 'Immediate' else "medium"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        model_status = self.model_manager.get_system_status()
        
        return {
            'system_name': 'Saafe IoT Predictive Fire Detection',
            'version': '3.0.0',
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'model_info': model_status,
            'areas_monitored': self.area_config,
            'capabilities': {
                'predictive_detection': True,
                'area_specific_analysis': True,
                'lead_time_prediction': True,
                'vendor_integration': True,
                'false_alarm_prevention': True
            }
        }


async def main():
    """Main application entry point."""
    logger.info("🚀 Starting Saafe IoT Predictive Fire Detection System")
    logger.info("=" * 60)
    
    try:
        # Initialize system
        model_path = "models/iot_transformer_model.pth"
        system = IoTFireDetectionSystem(model_path)
        
        # Display system status
        status = system.get_system_status()
        logger.info("System Status:")
        logger.info(f"  Version: {status['version']}")
        logger.info(f"  Model Status: {status['model_info']['device']}")
        logger.info(f"  Areas Monitored: {len(status['areas_monitored'])}")
        
        # Start monitoring
        logger.info("\n🔍 Starting continuous monitoring...")
        logger.info("Press Ctrl+C to stop")
        
        await system.monitor_continuous(interval_seconds=30)
        
    except KeyboardInterrupt:
        logger.info("\n👋 System shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    
    logger.info("Saafe IoT system stopped")


if __name__ == "__main__":
    asyncio.run(main())