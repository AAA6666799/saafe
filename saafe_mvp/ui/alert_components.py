"""
Saafe MVP - Alert Status and Notification Components
Implements system status display, notification configuration, and alert history
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from ..core.data_models import AlertData


class AlertStatusPanel:
    """Alert status and notification management panel"""
    
    def __init__(self):
        self.alert_history = []
        self.max_history = 100
        self.notification_status = {
            'sms': {'enabled': False, 'configured': False, 'last_test': None},
            'email': {'enabled': False, 'configured': False, 'last_test': None},
            'push': {'enabled': True, 'configured': True, 'last_test': None}
        }
        self.system_status = {
            'ai_model': 'operational',
            'data_stream': 'operational',
            'notifications': 'operational',
            'last_update': datetime.now()
        }
    
    def add_alert(self, alert: AlertData):
        """Add new alert to history"""
        self.alert_history.insert(0, {
            'timestamp': alert.timestamp,
            'alert_level': alert.alert_level,
            'alert_type': alert.alert_level.description,  # Use alert_level.description instead
            'message': alert.message,
            'risk_score': alert.risk_score
        })
        
        # Maintain history size
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop()
    
    def get_system_status_color(self, status: str) -> str:
        """Get color for system status"""
        status_colors = {
            'operational': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'offline': '#6c757d'
        }
        return status_colors.get(status, '#6c757d')
    
    def get_system_status_icon(self, status: str) -> str:
        """Get icon for system status"""
        status_icons = {
            'operational': '🟢',
            'warning': '🟡',
            'error': '🔴',
            'offline': '⚫'
        }
        return status_icons.get(status, '⚫')
    
    def render_system_status(self):
        """Render current system status display"""
        st.markdown("### System Status")
        
        # Overall system status
        overall_status = 'operational'
        if any(status != 'operational' for status in self.system_status.values() if isinstance(status, str)):
            overall_status = 'warning'
        
        overall_color = self.get_system_status_color(overall_status)
        overall_icon = self.get_system_status_icon(overall_status)
        
        st.markdown(f"""
        <div style="background: white; border-radius: 1rem; padding: 1.5rem; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 2px solid {overall_color}; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-right: 1rem;">{overall_icon}</div>
                <div>
                    <h4 style="margin: 0; color: {overall_color};">
                        {'All Systems Operational' if overall_status == 'operational' else 'System Issues Detected'}
                    </h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">
                        Last updated: {self.system_status['last_update'].strftime('%H:%M:%S')}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual component status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_component_status('AI Model', self.system_status['ai_model'], 
                                        'Fire detection models loaded and running')
        
        with col2:
            self._render_component_status('Data Stream', self.system_status['data_stream'], 
                                        'Sensor data generation active')
        
        with col3:
            self._render_component_status('Notifications', self.system_status['notifications'], 
                                        'Alert delivery services ready')
    
    def _render_component_status(self, component: str, status: str, description: str):
        """Render individual component status card"""
        color = self.get_system_status_color(status)
        icon = self.get_system_status_icon(status)
        
        st.markdown(f"""
        <div style="background: white; border-radius: 0.75rem; padding: 1rem; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-weight: 600; color: {color}; margin-bottom: 0.5rem;">{component}</div>
            <div style="font-size: 0.8rem; color: #666; text-transform: capitalize;">{status}</div>
            <div style="font-size: 0.7rem; color: #999; margin-top: 0.5rem;">{description}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_notification_status(self):
        """Render mobile notification configuration status"""
        st.markdown("### Notification Services")
        
        # Notification service cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_notification_card('SMS', 'sms', '📱', 
                                         'Text message alerts to mobile devices')
        
        with col2:
            self._render_notification_card('Email', 'email', '📧', 
                                         'Email alerts with detailed information')
        
        with col3:
            self._render_notification_card('Push', 'push', '🔔', 
                                         'Browser push notifications')
        
        # Notification configuration summary
        enabled_count = sum(1 for service in self.notification_status.values() if service['enabled'])
        configured_count = sum(1 for service in self.notification_status.values() if service['configured'])
        
        if enabled_count > 0:
            st.success(f"✅ {enabled_count} notification service(s) enabled and ready")
        else:
            st.warning("⚠️ No notification services enabled - configure alerts in settings")
    
    def _render_notification_card(self, name: str, key: str, icon: str, description: str):
        """Render individual notification service card"""
        service = self.notification_status[key]
        
        # Determine status and color
        if service['enabled'] and service['configured']:
            status_text = "Active"
            status_color = "#28a745"
            border_color = "#28a745"
        elif service['configured']:
            status_text = "Configured"
            status_color = "#ffc107"
            border_color = "#ffc107"
        else:
            status_text = "Not Configured"
            status_color = "#6c757d"
            border_color = "#dee2e6"
        
        st.markdown(f"""
        <div style="background: white; border-radius: 0.75rem; padding: 1rem; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 2px solid {border_color}; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">{name}</div>
            <div style="font-size: 0.9rem; color: {status_color}; font-weight: 500; margin-bottom: 0.5rem;">
                {status_text}
            </div>
            <div style="font-size: 0.8rem; color: #666;">{description}</div>
            {f'<div style="font-size: 0.7rem; color: #999; margin-top: 0.5rem;">Last test: {service["last_test"].strftime("%H:%M")}</div>' if service["last_test"] else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def render_alert_history(self):
        """Render alert history with timestamps and details"""
        st.markdown("### Recent Alerts")
        
        if not self.alert_history:
            st.markdown("""
            <div style="background: #f8f9fa; border-radius: 0.5rem; padding: 2rem; text-align: center; color: #666;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📋</div>
                <div>No alerts in history</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">Alert events will appear here</div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Display recent alerts
        for i, alert in enumerate(self.alert_history[:10]):  # Show last 10 alerts
            self._render_alert_item(alert, i)
        
        # Show more button if there are more alerts
        if len(self.alert_history) > 10:
            if st.button(f"Show {len(self.alert_history) - 10} more alerts"):
                for i, alert in enumerate(self.alert_history[10:], 10):
                    self._render_alert_item(alert, i)
    
    def _render_alert_item(self, alert: dict, index: int):
        """Render individual alert history item"""
        # Determine alert styling
        alert_colors = {
            'normal': '#28a745',
            'mild': '#ffc107',
            'elevated': '#fd7e14',
            'critical': '#dc3545'
        }
        
        alert_icons = {
            'normal': '🟢',
            'mild': '🟡',
            'elevated': '🟠',
            'critical': '🔴'
        }
        
        color = alert_colors.get(alert['alert_type'], '#6c757d')
        icon = alert_icons.get(alert['alert_type'], '⚫')
        
        # Time formatting
        time_str = alert['timestamp'].strftime('%H:%M:%S')
        date_str = alert['timestamp'].strftime('%Y-%m-%d')
        
        st.markdown(f"""
        <div style="background: white; border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.5rem;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.05); border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center;">
                    <div style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</div>
                    <div>
                        <div style="font-weight: 600; color: #333;">{alert['message']}</div>
                        <div style="font-size: 0.8rem; color: #666;">
                            Risk Score: {alert['risk_score']:.0f} | Level: {alert['alert_level']}
                        </div>
                    </div>
                </div>
                <div style="text-align: right; font-size: 0.8rem; color: #666;">
                    <div>{time_str}</div>
                    <div>{date_str}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_anti_hallucination_display(self, is_cooking_detected: bool = False, 
                                        explanation: str = ""):
        """Render anti-hallucination explanation for cooking scenarios"""
        if not is_cooking_detected:
            return
        
        st.markdown("### Anti-Hallucination System")
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                    border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem;
                    border: 2px solid #ffc107;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-right: 1rem;">🛡️</div>
                <div>
                    <h4 style="margin: 0; color: #856404;">Cooking Activity Detected</h4>
                    <p style="margin: 0; color: #856404; font-size: 0.9rem;">
                        Anti-hallucination system active - Fire alert suppressed
                    </p>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.7); border-radius: 0.5rem; padding: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #856404;">Why no fire alert?</h5>
                <p style="margin: 0; color: #856404; font-size: 0.9rem;">
                    {explanation if explanation else 
                     "The AI detected elevated PM2.5 and CO₂ levels consistent with normal cooking activity. "
                     "Temperature patterns and other indicators do not suggest fire risk. The anti-hallucination "
                     "system prevents false alarms during routine cooking activities."}
                </p>
            </div>
            
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #856404;">
                <strong>System Status:</strong> Monitoring continues - will alert if fire indicators detected
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_complete_alert_panel(self, current_alert: Optional[AlertData] = None,
                                  is_cooking_detected: bool = False):
        """Render the complete alert status and notification panel"""
        
        # Add current alert to history if provided
        if current_alert:
            self.add_alert(current_alert)
        
        # System status
        self.render_system_status()
        
        # Notification services status
        self.render_notification_status()
        
        # Anti-hallucination display (if cooking detected)
        if is_cooking_detected:
            self.render_anti_hallucination_display(
                is_cooking_detected=True,
                explanation="Elevated PM2.5 and CO₂ detected, but temperature patterns and temporal analysis indicate normal cooking activity rather than fire emergency."
            )
        
        # Alert history
        self.render_alert_history()
    
    def update_notification_config(self, service: str, enabled: bool, configured: bool):
        """Update notification service configuration"""
        if service in self.notification_status:
            self.notification_status[service]['enabled'] = enabled
            self.notification_status[service]['configured'] = configured
            if enabled and configured:
                self.notification_status[service]['last_test'] = datetime.now()
    
    def update_system_status(self, component: str, status: str):
        """Update system component status"""
        if component in self.system_status:
            self.system_status[component] = status
            self.system_status['last_update'] = datetime.now()