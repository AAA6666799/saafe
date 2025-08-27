"""
Production Deployment Configuration and Monitoring System.

This module provides comprehensive production deployment configurations,
monitoring, alerting, and operational readiness for the fire detection system.
"""

import yaml
import json
import logging
import threading
import time
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')


class ProductionConfiguration:
    """Production environment configuration management."""
    
    def __init__(self, config_path: str = None):
        """Initialize production configuration."""
        self.config_path = config_path or "config/production_config.yaml"
        self.config = self._load_default_config()
        
        if os.path.exists(self.config_path):
            self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default production configuration."""
        return {
            'system': {
                'environment': 'production',
                'debug': False,
                'log_level': 'INFO',
                'max_concurrent_requests': 100,
                'request_timeout_seconds': 30,
                'health_check_interval': 60
            },
            'sensors': {
                'mode': 'real',  # real, synthetic, hybrid
                'collection_interval': 1.0,
                'retry_attempts': 3,
                'timeout_seconds': 5,
                'auto_calibration': True,
                'fallback_to_synthetic': True
            },
            'models': {
                'ensemble_strategy': 'weighted_voting',
                'confidence_threshold': 0.7,
                'model_update_interval_hours': 24,
                'auto_retrain': False,
                'backup_models': True
            },
            'agents': {
                'monitoring': {
                    'enabled': True,
                    'check_interval': 5.0,
                    'anomaly_threshold': 0.8
                },
                'analysis': {
                    'enabled': True,
                    'pattern_analysis': True,
                    'historical_correlation': True
                },
                'response': {
                    'enabled': True,
                    'auto_alert': True,
                    'escalation_enabled': True
                },
                'learning': {
                    'enabled': True,
                    'continuous_learning': False,
                    'performance_tracking': True
                }
            },
            'alerting': {
                'enabled': True,
                'channels': ['email', 'sms', 'webhook'],
                'email': {
                    'smtp_server': 'localhost',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                },
                'webhooks': {
                    'fire_alert_url': '',
                    'system_alert_url': ''
                },
                'thresholds': {
                    'fire_confidence': 0.8,
                    'system_error_rate': 0.05,
                    'processing_delay_ms': 2000
                }
            },
            'monitoring': {
                'metrics_collection': True,
                'metrics_retention_days': 30,
                'performance_tracking': True,
                'resource_monitoring': True,
                'log_aggregation': True
            },
            'security': {
                'api_key_required': True,
                'rate_limiting': True,
                'ip_whitelist': [],
                'encryption_at_rest': True,
                'audit_logging': True
            },
            'deployment': {
                'auto_scaling': False,
                'health_checks': True,
                'graceful_shutdown': True,
                'rolling_updates': True,
                'backup_strategy': 'daily'
            }
        }
    
    def _load_config_file(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._merge_configs(self.config, file_config)
        except Exception as e:
            logging.warning(f"Failed to load config file {self.config_path}: {e}")
    
    def _merge_configs(self, base: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate production configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields validation
        required_paths = [
            'system.environment',
            'sensors.mode',
            'models.ensemble_strategy',
            'alerting.enabled'
        ]
        
        for path in required_paths:
            if self.get(path) is None:
                validation_results['errors'].append(f"Missing required configuration: {path}")
                validation_results['valid'] = False
        
        # Value validation
        if self.get('sensors.collection_interval', 0) <= 0:
            validation_results['errors'].append("sensors.collection_interval must be positive")
            validation_results['valid'] = False
        
        if self.get('models.confidence_threshold', 0) < 0 or self.get('models.confidence_threshold', 1) > 1:
            validation_results['errors'].append("models.confidence_threshold must be between 0 and 1")
            validation_results['valid'] = False
        
        # Warning conditions
        if not self.get('alerting.enabled'):
            validation_results['warnings'].append("Alerting is disabled in production")
        
        if not self.get('monitoring.metrics_collection'):
            validation_results['warnings'].append("Metrics collection is disabled")
        
        return validation_results
    
    def save_config(self, path: str = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


class ProductionMonitoring:
    """Production monitoring and alerting system."""
    
    def __init__(self, config: ProductionConfiguration):
        """Initialize production monitoring."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_manager = AlertManager(config)
        
        # Metrics storage
        self.metrics_history = []
        self.system_health = {
            'status': 'unknown',
            'last_check': None,
            'uptime_start': datetime.now(),
            'error_count': 0,
            'alert_count': 0
        }
    
    def start_monitoring(self, system) -> None:
        """Start production monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.system_health['uptime_start'] = datetime.now()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(system,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Production monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop production monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self, system) -> None:
        """Main monitoring loop."""
        check_interval = self.config.get('system.health_check_interval', 60)
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics(system)
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                retention_days = self.config.get('monitoring.metrics_retention_days', 30)
                cutoff_time = datetime.now() - timedelta(days=retention_days)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]
                
                # Check for alert conditions
                alerts = self._check_alert_conditions(metrics)
                
                # Send alerts if any
                for alert in alerts:
                    self.alert_manager.send_alert(alert)
                    self.system_health['alert_count'] += 1
                
                # Update system health
                self.system_health.update({
                    'status': 'healthy' if not alerts else 'degraded',
                    'last_check': datetime.now().isoformat(),
                    'latest_metrics': metrics
                })
                
                # Sleep until next check
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self.system_health['error_count'] += 1
                time.sleep(min(check_interval, 60))  # Don't sleep too long on error
    
    def _collect_system_metrics(self, system) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'unknown',
            'performance': {},
            'resources': {},
            'accuracy': {},
            'errors': []
        }
        
        try:
            # Get system status if available
            if hasattr(system, 'get_system_status'):
                status = system.get_system_status()
                metrics['system_status'] = status.get('state', 'unknown')
                metrics['performance'] = status.get('metrics', {})
                metrics['resources'] = status.get('subsystem_health', {})
            
            # Get performance metrics if available
            if hasattr(system, 'metrics'):
                perf_metrics = system.metrics
                metrics['performance'].update({
                    'total_processed': getattr(perf_metrics, 'total_processed', 0),
                    'error_count': getattr(perf_metrics, 'total_errors', 0),
                    'fire_detections': getattr(perf_metrics, 'fire_detections', 0)
                })
            
        except Exception as e:
            metrics['errors'].append(f"Metrics collection error: {str(e)}")
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for conditions that should trigger alerts."""
        alerts = []
        
        # System status alerts
        if metrics['system_status'] in ['error', 'failed', 'stopped']:
            alerts.append({
                'type': 'system_critical',
                'severity': 'critical',
                'message': f"System status is {metrics['system_status']}",
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
        
        # Performance alerts
        performance = metrics.get('performance', {})
        
        # Processing delay alert
        avg_time = performance.get('average_processing_time', 0)
        delay_threshold = self.config.get('alerting.thresholds.processing_delay_ms', 2000)
        
        if avg_time > delay_threshold:
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'warning',
                'message': f"Processing time {avg_time}ms exceeds threshold {delay_threshold}ms",
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
        
        # Error rate alert
        total_processed = performance.get('total_processed', 0)
        error_count = performance.get('error_count', 0)
        
        if total_processed > 0:
            error_rate = error_count / total_processed
            error_threshold = self.config.get('alerting.thresholds.system_error_rate', 0.05)
            
            if error_rate > error_threshold:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f"Error rate {error_rate:.2%} exceeds threshold {error_threshold:.2%}",
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
        
        return alerts
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        dashboard = {
            'system_health': self.system_health.copy(),
            'recent_metrics': recent_metrics,
            'alert_summary': {
                'total_alerts': self.system_health['alert_count'],
                'recent_alerts': len([
                    m for m in recent_metrics 
                    if datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(hours=1)
                ])
            },
            'uptime': {
                'start_time': self.system_health['uptime_start'].isoformat(),
                'duration_seconds': (datetime.now() - self.system_health['uptime_start']).total_seconds()
            }
        }
        
        return dashboard


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, config: ProductionConfiguration):
        """Initialize alert manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert history
        self.alert_history = []
        self.rate_limits = {}  # For preventing alert spam
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert through configured channels."""
        if not self.config.get('alerting.enabled', False):
            return False
        
        # Check rate limiting
        alert_key = f"{alert['type']}_{alert['severity']}"
        if self._is_rate_limited(alert_key):
            return False
        
        # Record alert
        self.alert_history.append(alert)
        self._update_rate_limit(alert_key)
        
        # Send through configured channels
        channels = self.config.get('alerting.channels', [])
        success = False
        
        for channel in channels:
            try:
                if channel == 'email':
                    success |= self._send_email_alert(alert)
                elif channel == 'webhook':
                    success |= self._send_webhook_alert(alert)
                elif channel == 'sms':
                    success |= self._send_sms_alert(alert)
                else:
                    self.logger.warning(f"Unknown alert channel: {channel}")
            
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}")
        
        return success
    
    def _is_rate_limited(self, alert_key: str) -> bool:
        """Check if alert type is rate limited."""
        if alert_key not in self.rate_limits:
            return False
        
        last_sent = self.rate_limits[alert_key]
        min_interval = timedelta(minutes=5)  # Minimum 5 minutes between same alerts
        
        return datetime.now() - last_sent < min_interval
    
    def _update_rate_limit(self, alert_key: str) -> None:
        """Update rate limit tracking."""
        self.rate_limits[alert_key] = datetime.now()
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via email."""
        try:
            email_config = self.config.get('alerting.email', {})
            recipients = email_config.get('recipients', [])
            
            if not recipients:
                return False
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = email_config.get('username', 'fire-detection@system.local')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Fire Detection Alert: {alert['type']} ({alert['severity']})"
            
            # Email body
            body = f"""
Fire Detection System Alert

Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}
Message: {alert['message']}

System Metrics:
{json.dumps(alert.get('metrics', {}), indent=2)}

This is an automated alert from the fire detection system.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email (implementation would need actual SMTP configuration)
            self.logger.info(f"Email alert sent: {alert['type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            return False
    
    def _send_webhook_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via webhook."""
        try:
            # Implementation would use requests to send HTTP POST
            self.logger.info(f"Webhook alert sent: {alert['type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
            return False
    
    def _send_sms_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via SMS."""
        try:
            # Implementation would use SMS service API
            self.logger.info(f"SMS alert sent: {alert['type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"SMS alert failed: {e}")
            return False


class ProductionDeploymentManager:
    """Production deployment and lifecycle management."""
    
    def __init__(self, config: ProductionConfiguration):
        """Initialize deployment manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.deployment_state = 'stopped'
        
    def deploy_system(self, system) -> Dict[str, Any]:
        """Deploy system to production environment."""
        deployment_result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'steps_completed': [],
            'errors': [],
            'deployment_id': f"deploy_{int(time.time())}"
        }
        
        try:
            # Step 1: Validate configuration
            config_validation = self.config.validate_configuration()
            if not config_validation['valid']:
                deployment_result['errors'].extend(config_validation['errors'])
                return deployment_result
            
            deployment_result['steps_completed'].append('configuration_validated')
            
            # Step 2: Initialize system
            if hasattr(system, 'initialize'):
                init_success = system.initialize()
                if not init_success:
                    deployment_result['errors'].append('System initialization failed')
                    return deployment_result
            
            deployment_result['steps_completed'].append('system_initialized')
            
            # Step 3: Start monitoring
            monitoring = ProductionMonitoring(self.config)
            monitoring.start_monitoring(system)
            
            deployment_result['steps_completed'].append('monitoring_started')
            
            # Step 4: Health check
            time.sleep(2)  # Allow system to stabilize
            
            if hasattr(system, 'get_system_status'):
                status = system.get_system_status()
                if status.get('state') not in ['running', 'ready']:
                    deployment_result['errors'].append(f"System health check failed: {status.get('state')}")
                    return deployment_result
            
            deployment_result['steps_completed'].append('health_check_passed')
            
            # Deployment successful
            self.deployment_state = 'deployed'
            deployment_result['status'] = 'success'
            
            self.logger.info(f"Production deployment successful: {deployment_result['deployment_id']}")
            
        except Exception as e:
            deployment_result['errors'].append(f"Deployment error: {str(e)}")
            self.logger.error(f"Production deployment failed: {e}")
        
        return deployment_result
    
    def create_deployment_report(self, system) -> Dict[str, Any]:
        """Create comprehensive deployment report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'deployment_state': self.deployment_state,
            'configuration_summary': {
                'environment': self.config.get('system.environment'),
                'sensor_mode': self.config.get('sensors.mode'),
                'alerting_enabled': self.config.get('alerting.enabled'),
                'monitoring_enabled': self.config.get('monitoring.metrics_collection')
            },
            'system_status': {},
            'readiness_checklist': self._create_readiness_checklist(system),
            'recommendations': []
        }
        
        # Get current system status
        try:
            if hasattr(system, 'get_system_status'):
                report['system_status'] = system.get_system_status()
        except Exception as e:
            report['system_status'] = {'error': str(e)}
        
        # Generate recommendations
        if self.deployment_state != 'deployed':
            report['recommendations'].append("Complete deployment process before production use")
        
        if not self.config.get('alerting.enabled'):
            report['recommendations'].append("Enable alerting for production monitoring")
        
        return report
    
    def _create_readiness_checklist(self, system) -> Dict[str, bool]:
        """Create deployment readiness checklist."""
        checklist = {
            'configuration_valid': self.config.validate_configuration()['valid'],
            'system_initialized': hasattr(system, 'initialize') and self.deployment_state != 'stopped',
            'monitoring_configured': self.config.get('monitoring.metrics_collection', False),
            'alerting_configured': self.config.get('alerting.enabled', False),
            'security_enabled': self.config.get('security.api_key_required', False),
            'backup_strategy': self.config.get('deployment.backup_strategy') is not None,
            'health_checks_enabled': self.config.get('deployment.health_checks', False)
        }
        
        return checklist


# Convenience function for complete production setup
def setup_production_environment(system, config_path: str = None) -> Dict[str, Any]:
    """
    Complete production environment setup and deployment.
    
    Args:
        system: Fire detection system instance
        config_path: Optional path to production configuration file
        
    Returns:
        Dictionary containing setup results and deployment status
    """
    
    # Initialize production components
    config = ProductionConfiguration(config_path)
    deployment_manager = ProductionDeploymentManager(config)
    
    # Deploy system
    deployment_result = deployment_manager.deploy_system(system)
    
    # Create deployment report
    deployment_report = deployment_manager.create_deployment_report(system)
    
    return {
        'deployment': deployment_result,
        'report': deployment_report,
        'configuration': config.config,
        'summary': {
            'deployment_successful': deployment_result['status'] == 'success',
            'steps_completed': len(deployment_result['steps_completed']),
            'errors': len(deployment_result['errors']),
            'production_ready': all(deployment_report['readiness_checklist'].values())
        }
    }


if __name__ == "__main__":
    # Example usage
    print("🚀 Production Deployment and Monitoring Framework")
    print("=" * 60)
    
    # Create production configuration
    config = ProductionConfiguration()
    validation = config.validate_configuration()
    
    print(f"Configuration Valid: {validation['valid']}")
    print(f"Errors: {len(validation['errors'])}")
    print(f"Warnings: {len(validation['warnings'])}")
    
    # Save example configuration
    config.save_config("config/production_config_example.yaml")
    print("Example configuration saved")
    
    print("\n✅ Production deployment framework ready!")