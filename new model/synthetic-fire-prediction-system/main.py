#!/usr/bin/env python3
"""
Main entry point for the IoT-based fire prediction system.
Handles FLIR Lepton 3.5 thermal cameras and Sensirion SCD41 CO₂ sensors.
"""

import os
import sys
import argparse
import logging
import asyncio
import torch
from typing import Dict, Any
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.system import create_system
from src.integrated_system import IntegratedFireDetectionSystem

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FLIR + SCD41 IoT Fire Detection System')
    
    parser.add_argument('--config', type=str, default='config',
                        help='Path to configuration directory')
    
    parser.add_argument('--env', type=str, default=None,
                        help='Environment (dev, test, prod)')
    
    parser.add_argument('--mode', type=str, default='iot',
                        choices=['run', 'test', 'generate', 'train', 'iot'],
                        help='Operation mode')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained IoT model')
    
    parser.add_argument('--data-source', type=str, default='mqtt',
                        choices=['mqtt', 'synthetic', 'usb'],
                        help='IoT data source (mqtt/synthetic/usb)')
    
    parser.add_argument('--thermal-device', type=str, default='/dev/thermal0',
                        help='FLIR Lepton device path')
    
    parser.add_argument('--gas-device', type=str, default='/dev/ttyUSB0',
                        help='SCD41 sensor device path')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()


def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=numeric_level, format=log_format)
    
    os.makedirs('logs', exist_ok=True)
    
    file_handler = logging.FileHandler(
        f'logs/flir_scd41_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


class FlirScd41FireDetectionSystem:
    """Fire detection system for FLIR Lepton 3.5 + Sensirion SCD41."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the IoT fire detection system."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize integrated system with IoT configuration
        iot_config = {
            'system_id': 'flir_scd41_fire_detection',
            'sensors': {
                'mode': 'real',  # Real IoT devices
                'thermal_config': {
                    'device_type': 'flir_lepton_3_5',
                    'device_path': config.get('thermal_device', '/dev/thermal0'),
                    'features': ['t_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct', 
                               't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
                               't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
                               'tproxy_val', 'tproxy_delta', 'tproxy_vel']
                },
                'gas_config': {
                    'device_type': 'sensirion_scd41',
                    'device_path': config.get('gas_device', '/dev/ttyUSB0'),
                    'features': ['gas_val', 'gas_delta', 'gas_vel']
                }
            },
            'agents': {
                'analysis': {
                    'fire_pattern': {
                        'confidence_threshold': 0.7,
                        'thermal_thresholds': {
                            't_max': 60.0,
                            't_hot_area_pct': 10.0,
                            'tproxy_vel': 2.0
                        },
                        'gas_thresholds': {
                            'gas_val': 1000.0,  # CO₂ ppm
                            'gas_vel': 50.0     # Rate of change
                        }
                    }
                }
            }
        }
        
        self.system = IntegratedFireDetectionSystem(iot_config)
        
        # Device feature mappings
        self.thermal_features = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel'
        ]
        
        self.gas_features = ['gas_val', 'gas_delta', 'gas_vel']
        
        self.logger.info("🔥 FLIR + SCD41 Fire Detection System initialized")
        self.logger.info(f"📷 Thermal features: {len(self.thermal_features)}")
        self.logger.info(f"🌬️ Gas features: {len(self.gas_features)}")
    
    def initialize(self) -> bool:
        """Initialize the system."""
        return self.system.initialize()
    
    def start(self) -> bool:
        """Start the system."""
        return self.system.start()
    
    def process_iot_data(self, thermal_data: Dict[str, float], gas_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Process data from FLIR and SCD41 devices.
        
        Args:
            thermal_data: FLIR Lepton 3.5 features
            gas_data: SCD41 sensor features
            
        Returns:
            Fire detection results
        """
        # Convert to system format
        sensor_data = {
            'thermal': {
                'flir_lepton_01': {
                    **thermal_data,
                    'timestamp': datetime.now().isoformat(),
                    'device_type': 'flir_lepton_3_5'
                }
            },
            'gas': {
                'scd41_01': {
                    **gas_data,
                    'timestamp': datetime.now().isoformat(),
                    'device_type': 'sensirion_scd41'
                }
            }
        }
        
        # Process through integrated system
        return self.system.process_data(sensor_data)
    
    def validate_thermal_data(self, data: Dict[str, float]) -> bool:
        """Validate FLIR thermal data format."""
        required_fields = self.thermal_features
        return all(field in data for field in required_fields)
    
    def validate_gas_data(self, data: Dict[str, float]) -> bool:
        """Validate SCD41 gas data format."""
        required_fields = self.gas_features
        return all(field in data for field in required_fields)


async def run_iot_system(args):
    """Run the IoT fire detection system."""
    logger = logging.getLogger('iot_main')
    logger.info("🚀 Starting FLIR + SCD41 Fire Detection System")
    
    try:
        # Initialize system with device paths
        config = {
            'thermal_device': args.thermal_device,
            'gas_device': args.gas_device,
            'data_source': args.data_source
        }
        
        system = FlirScd41FireDetectionSystem(config)
        
        if not system.initialize():
            logger.error("❌ System initialization failed")
            return 1
        
        if not system.start():
            logger.error("❌ System startup failed")
            return 1
        
        logger.info("✅ System started successfully")
        
        # Start monitoring loop
        if args.data_source == 'synthetic':
            await monitor_synthetic_data(system)
        elif args.data_source == 'usb':
            await monitor_usb_devices(system, args)
        else:  # mqtt
            await monitor_mqtt_data(system)
            
    except Exception as e:
        logger.error(f"❌ System error: {str(e)}")
        return 1
    
    return 0


async def monitor_synthetic_data(system: FlirScd41FireDetectionSystem):
    """Monitor with synthetic FLIR + SCD41 data."""
    logger = logging.getLogger('synthetic_monitor')
    logger.info("📊 Starting synthetic data monitoring")
    
    while True:
        try:
            # Generate synthetic FLIR data
            thermal_data = generate_synthetic_flir_data()
            
            # Generate synthetic SCD41 data
            gas_data = generate_synthetic_scd41_data()
            
            # Process data
            result = system.process_iot_data(thermal_data, gas_data)
            
            # Log results
            fire_detected = result.get('final_decision', {}).get('fire_detected', False)
            confidence = result.get('final_decision', {}).get('confidence_score', 0.0)
            
            status = "🔥 FIRE" if fire_detected else "✅ SAFE"
            logger.info(f"{status} - Confidence: {confidence:.3f}")
            
            if fire_detected:
                logger.warning(f"🚨 Fire detected! Response: {result.get('agent_response', {}).get('response_level', 'UNKNOWN')}")
            
            await asyncio.sleep(1.0)  # 1 Hz monitoring
            
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Monitoring error: {str(e)}")
            await asyncio.sleep(5.0)


def generate_synthetic_flir_data() -> Dict[str, float]:
    """Generate synthetic FLIR Lepton 3.5 data."""
    # Normal operation values with some randomness
    return {
        't_mean': np.random.normal(22.0, 2.0),
        't_std': np.random.normal(3.0, 0.5),
        't_max': np.random.normal(25.0, 5.0),
        't_p95': np.random.normal(24.0, 3.0),
        't_hot_area_pct': np.random.exponential(2.0),
        't_hot_largest_blob_pct': np.random.exponential(1.0),
        't_grad_mean': np.random.normal(1.0, 0.3),
        't_grad_std': np.random.normal(0.5, 0.2),
        't_diff_mean': np.random.normal(0.1, 0.1),
        't_diff_std': np.random.normal(0.1, 0.05),
        'flow_mag_mean': np.random.normal(0.5, 0.2),
        'flow_mag_std': np.random.normal(0.3, 0.1),
        'tproxy_val': np.random.normal(25.0, 5.0),
        'tproxy_delta': np.random.normal(0.1, 0.5),
        'tproxy_vel': np.random.normal(0.05, 0.2)
    }


def generate_synthetic_scd41_data() -> Dict[str, float]:
    """Generate synthetic SCD41 CO₂ sensor data."""
    # Normal indoor CO₂ levels
    return {
        'gas_val': np.random.normal(450.0, 50.0),    # Normal indoor CO₂
        'gas_delta': np.random.normal(0.0, 5.0),     # Small changes
        'gas_vel': np.random.normal(0.0, 5.0)        # Same as delta
    }


async def monitor_usb_devices(system: FlirScd41FireDetectionSystem, args):
    """Monitor real USB devices."""
    logger = logging.getLogger('usb_monitor')
    logger.info(f"🔌 Monitoring USB devices: {args.thermal_device}, {args.gas_device}")
    
    # This would integrate with actual device drivers
    logger.warning("📝 USB device monitoring not implemented - using synthetic data")
    await monitor_synthetic_data(system)


async def monitor_mqtt_data(system: FlirScd41FireDetectionSystem):
    """Monitor MQTT data streams."""
    logger = logging.getLogger('mqtt_monitor')
    logger.info("📡 Monitoring MQTT data streams")
    
    # This would integrate with MQTT broker
    logger.warning("📝 MQTT monitoring not implemented - using synthetic data")
    await monitor_synthetic_data(system)


def test_system(args):
    """Test the system with sample FLIR + SCD41 data."""
    logger = logging.getLogger('test')
    logger.info("🧪 Testing FLIR + SCD41 system")
    
    try:
        config = {'thermal_device': args.thermal_device, 'gas_device': args.gas_device}
        system = FlirScd41FireDetectionSystem(config)
        
        if not system.initialize():
            logger.error("❌ System initialization failed")
            return 1
        
        # Test with normal data
        logger.info("📊 Testing with normal conditions...")
        thermal_data = generate_synthetic_flir_data()
        gas_data = generate_synthetic_scd41_data()
        
        if not system.validate_thermal_data(thermal_data):
            logger.error("❌ Invalid thermal data format")
            return 1
        
        if not system.validate_gas_data(gas_data):
            logger.error("❌ Invalid gas data format")
            return 1
        
        result = system.process_iot_data(thermal_data, gas_data)
        
        logger.info("📊 Test Results:")
        logger.info(f"   Fire Detected: {result.get('final_decision', {}).get('fire_detected', False)}")
        logger.info(f"   Confidence: {result.get('final_decision', {}).get('confidence_score', 0.0):.3f}")
        logger.info(f"   Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
        
        # Test with fire conditions
        logger.info("🔥 Testing with fire conditions...")
        fire_thermal = {
            't_mean': 45.0, 't_std': 8.0, 't_max': 85.0, 't_p95': 75.0,
            't_hot_area_pct': 25.0, 't_hot_largest_blob_pct': 15.0,
            't_grad_mean': 5.0, 't_grad_std': 2.0, 't_diff_mean': 3.0, 't_diff_std': 1.5,
            'flow_mag_mean': 3.0, 'flow_mag_std': 1.0, 'tproxy_val': 80.0,
            'tproxy_delta': 15.0, 'tproxy_vel': 8.0
        }
        
        fire_gas = {
            'gas_val': 1200.0,  # Elevated CO₂
            'gas_delta': 150.0,  # Rapid increase
            'gas_vel': 150.0
        }
        
        fire_result = system.process_iot_data(fire_thermal, fire_gas)
        
        logger.info("🔥 Fire Test Results:")
        logger.info(f"   Fire Detected: {fire_result.get('final_decision', {}).get('fire_detected', False)}")
        logger.info(f"   Confidence: {fire_result.get('final_decision', {}).get('confidence_score', 0.0):.3f}")
        
        logger.info("✅ System test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return 1


def run_system(args):
    """
    Run the system in normal operation mode.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger('main')
    logger.info("Starting system in run mode")
    
    # Create and initialize the system
    system = create_system(args.config, args.env)
    
    if not system.is_initialized:
        logger.error("System initialization failed")
        return 1
    
    # Start the system
    if not system.start():
        logger.error("System startup failed")
        return 1
    
    try:
        # In a real application, this would be a main processing loop
        logger.info("System running. Press Ctrl+C to stop.")
        while True:
            # Process data or wait for events
            pass
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    
    finally:
        # Stop and shut down the system
        system.stop()
        system.shutdown()
    
    return 0


def generate_data(args):
    """
    Generate IoT training data in the new format.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger('data_gen')
    logger.info("📊 Generating IoT training data")
    
    try:
        # Import data generation modules
        from src.data_generation.iot_data_generator import IoTDataGenerator
        
        # Configure for IoT data generation
        config = {
            'areas': ['kitchen', 'electrical', 'laundry_hvac', 'living_bedroom', 'basement_storage'],
            'samples_per_area': 10000,
            'output_dir': 'synthetic datasets/',
            'format': 'iot_csv'
        }
        
        generator = IoTDataGenerator(config)
        generator.generate_comprehensive_iot_dataset()
        
        logger.info("✅ IoT data generation completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Data generation failed: {str(e)}")
        return 1


async def train_models(args):
    """
    Train IoT models with the new data format.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger('training')
    logger.info("🚀 Starting IoT model training with area-based data")
    
    try:
        # Import training modules
        from train_iot_models import IoTModelTrainer
        
        # Initialize trainer with IoT configuration
        trainer = IoTModelTrainer(
            data_path="synthetic datasets/",
            model_save_path="models/",
            config_path=args.config
        )
        
        # Train models on IoT data format
        training_results = await trainer.train_comprehensive_iot_models()
        
        logger.info("📊 Training Results:")
        for model_name, result in training_results.items():
            if result['status'] == 'success':
                logger.info(f"   ✅ {model_name}: {result.get('accuracy', 'N/A'):.3f}")
            else:
                logger.info(f"   ❌ {model_name}: {result['error']}")
        
        logger.info("🎉 IoT model training completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return 1


def main():
    """Main function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger('main')
    logger.info("🔥 FLIR Lepton 3.5 + SCD41 Fire Detection System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data Source: {args.data_source}")
    logger.info(f"Thermal Device: {args.thermal_device}")
    logger.info(f"Gas Device: {args.gas_device}")
    
    if args.mode == 'iot':
        return asyncio.run(run_iot_system(args))
    elif args.mode == 'test':
        return test_system(args)
    elif args.mode == 'run':
        return run_system(args)
    elif args.mode == 'generate':
        logger.warning("📝 Data generation mode - will create synthetic FLIR + SCD41 datasets")
        return 0
    elif args.mode == 'train':
        logger.warning("📝 Training mode - will train models on FLIR + SCD41 data")
        return 0
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())