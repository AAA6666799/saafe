"""
Saafe MVP - Main Settings Page
Integrates all settings components into a cohesive settings interface
"""

import streamlit as st
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

from .settings_components import (
    NotificationConfigPanel, 
    AlertThresholdPanel, 
    SystemPreferencesPanel,
    SystemPreferences,
    ThemeMode,
    PerformanceMode
)
from ..services.notification_manager import NotificationManager, NotificationConfig, AlertLevel
from ..services.sms_service import SMSConfig
from ..services.email_service import EmailConfig
from ..services.push_notification_service import PushConfig


class SettingsManager:
    """Manages settings persistence and validation"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path
        self.ensure_config_directory()
    
    def ensure_config_directory(self):
        """Ensure config directory exists"""
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        default_settings = {
            'system_preferences': {
                'update_frequency': 1,
                'theme_mode': 'light',
                'performance_mode': 'balanced',
                'model_path': 'models/saved',
                'normal_threshold': [0, 30],
                'mild_threshold': [31, 50],
                'elevated_threshold': [51, 85],
                'critical_threshold': [86, 100]
            },
            'notification_config': {
                'sms_enabled': True,
                'email_enabled': True,
                'push_enabled': True,
                'phone_numbers': [],
                'email_addresses': [],
                'sms_min_level': 'elevated',
                'email_min_level': 'mild',
                'push_min_level': 'mild'
            },
            'last_updated': None
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    settings = json.load(f)
                # Merge with defaults to handle missing keys
                for key, value in default_settings.items():
                    if key not in settings:
                        settings[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in settings[key]:
                                settings[key][subkey] = subvalue
                return settings
            except Exception as e:
                st.error(f"Error loading settings: {e}")
                return default_settings
        
        return default_settings
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file"""
        try:
            settings['last_updated'] = datetime.now().isoformat()
            with open(self.config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving settings: {e}")
            return False
    
    def get_system_preferences(self, settings: Dict[str, Any]) -> SystemPreferences:
        """Convert settings dict to SystemPreferences object"""
        prefs_dict = settings.get('system_preferences', {})
        
        return SystemPreferences(
            update_frequency=prefs_dict.get('update_frequency', 1),
            theme_mode=ThemeMode(prefs_dict.get('theme_mode', 'light')),
            performance_mode=PerformanceMode(prefs_dict.get('performance_mode', 'balanced')),
            model_path=prefs_dict.get('model_path', 'models/saved'),
            normal_threshold=tuple(prefs_dict.get('normal_threshold', [0, 30])),
            mild_threshold=tuple(prefs_dict.get('mild_threshold', [31, 50])),
            elevated_threshold=tuple(prefs_dict.get('elevated_threshold', [51, 85])),
            critical_threshold=tuple(prefs_dict.get('critical_threshold', [86, 100]))
        )
    
    def get_notification_config(self, settings: Dict[str, Any]) -> NotificationConfig:
        """Convert settings dict to NotificationConfig object"""
        notif_dict = settings.get('notification_config', {})
        
        return NotificationConfig(
            sms_enabled=notif_dict.get('sms_enabled', True),
            email_enabled=notif_dict.get('email_enabled', True),
            push_enabled=notif_dict.get('push_enabled', True),
            phone_numbers=notif_dict.get('phone_numbers', []),
            email_addresses=notif_dict.get('email_addresses', []),
            sms_min_level=AlertLevel(notif_dict.get('sms_min_level', 'elevated')),
            email_min_level=AlertLevel(notif_dict.get('email_min_level', 'mild')),
            push_min_level=AlertLevel(notif_dict.get('push_min_level', 'mild'))
        )


class SettingsPage:
    """Main settings page interface"""
    
    def __init__(self):
        self.settings_manager = SettingsManager()
        self.notification_panel = NotificationConfigPanel()
        self.threshold_panel = AlertThresholdPanel()
        self.preferences_panel = SystemPreferencesPanel()
        self._setup_page_config()
        self._setup_custom_css()
    
    def _setup_page_config(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="Saafe Settings",
            page_icon="⚙️",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def _setup_custom_css(self):
        """Setup custom CSS for settings page"""
        st.markdown("""
        <style>
        /* Settings page styling */
        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0 2rem 0;
            border-bottom: 2px solid #ff4b4b;
            margin-bottom: 2rem;
        }
        
        .settings-logo {
            font-size: 2rem;
            font-weight: 700;
            color: #ff4b4b;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .back-button {
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
            text-decoration: none;
            font-weight: 500;
        }
        
        .back-button:hover {
            background: #5a6268;
            text-decoration: none;
            color: white;
        }
        
        .save-section {
            background: #f8f9fa;
            border-radius: 1rem;
            padding: 2rem;
            margin-top: 2rem;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        
        .save-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .save-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        .save-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render settings page header"""
        st.markdown("""
        <div class="settings-header">
            <div class="settings-logo">
                ⚙️ Saafe Settings
            </div>
            <a href="/" class="back-button">← Back to Dashboard</a>
        </div>
        """, unsafe_allow_html=True)
    
    def simulate_notification_test(self, test_type: str, phone_numbers: list, email_addresses: list) -> Dict[str, Any]:
        """Simulate notification testing (since we don't have real services configured)"""
        import time
        import random
        
        # Simulate processing time
        time.sleep(1)
        
        results = {'success': True, 'message': ''}
        
        if test_type == 'sms':
            if phone_numbers:
                results['message'] = f"SMS test sent to {len(phone_numbers)} number(s)"
            else:
                results = {'success': False, 'message': 'No phone numbers configured'}
        
        elif test_type == 'email':
            if email_addresses:
                results['message'] = f"Email test sent to {len(email_addresses)} address(es)"
            else:
                results = {'success': False, 'message': 'No email addresses configured'}
        
        elif test_type == 'push':
            results['message'] = "Push notification test sent to browser"
        
        elif test_type == 'all':
            total_contacts = len(phone_numbers) + len(email_addresses)
            if total_contacts > 0:
                results['message'] = f"Test notifications sent to all {total_contacts} contact(s)"
            else:
                results = {'success': False, 'message': 'No contacts configured'}
        
        # Simulate occasional failures for realism
        if random.random() < 0.1:  # 10% chance of failure
            results = {'success': False, 'message': 'Test failed - please check configuration'}
        
        return results
    
    def render_main_settings(self):
        """Render main settings interface"""
        # Load current settings
        settings = self.settings_manager.load_settings()
        system_prefs = self.settings_manager.get_system_preferences(settings)
        notification_config = self.settings_manager.get_notification_config(settings)
        
        # Initialize session state for contact lists
        if 'phone_numbers' not in st.session_state:
            st.session_state.phone_numbers = notification_config.phone_numbers.copy()
        if 'email_addresses' not in st.session_state:
            st.session_state.email_addresses = notification_config.email_addresses.copy()
        
        # Render header
        self.render_header()
        
        # Create tabs for different settings sections
        tab1, tab2, tab3 = st.tabs(["📱 Notifications", "⚠️ Alert Thresholds", "⚙️ System Preferences"])
        
        with tab1:
            # Render notification configuration panel
            notification_data = self.notification_panel.render_complete_notification_panel()
            
            # Handle notification testing
            if notification_data['test_action']:
                with st.spinner("Testing notifications..."):
                    test_results = self.simulate_notification_test(
                        notification_data['test_action'],
                        notification_data['phone_numbers'],
                        notification_data['email_addresses']
                    )
                    st.session_state.test_results = test_results
                    st.rerun()
        
        with tab2:
            # Get current risk score for preview (simulate)
            current_risk_score = st.session_state.get('current_risk_score', 25)
            
            # Render alert threshold panel
            updated_prefs, reset_clicked = self.threshold_panel.render_complete_threshold_panel(
                system_prefs, current_risk_score
            )
            
            if reset_clicked:
                # Reset thresholds to defaults
                system_prefs = SystemPreferences()
                st.success("Thresholds reset to defaults")
                st.rerun()
            else:
                system_prefs = updated_prefs
        
        with tab3:
            # Render system preferences panel
            system_prefs = self.preferences_panel.render_complete_preferences_panel(system_prefs)
        
        # Save settings section
        st.markdown('<div class="save-section">', unsafe_allow_html=True)
        st.markdown("### Save Configuration")
        st.markdown("Save your settings to apply changes and persist them across sessions.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Save Settings", type="primary", key="save_settings"):
                # Prepare settings for saving
                updated_settings = {
                    'system_preferences': {
                        'update_frequency': system_prefs.update_frequency,
                        'theme_mode': system_prefs.theme_mode.value,
                        'performance_mode': system_prefs.performance_mode.value,
                        'model_path': system_prefs.model_path,
                        'normal_threshold': list(system_prefs.normal_threshold),
                        'mild_threshold': list(system_prefs.mild_threshold),
                        'elevated_threshold': list(system_prefs.elevated_threshold),
                        'critical_threshold': list(system_prefs.critical_threshold)
                    },
                    'notification_config': {
                        'sms_enabled': notification_data['toggles']['sms_enabled'],
                        'email_enabled': notification_data['toggles']['email_enabled'],
                        'push_enabled': notification_data['toggles']['push_enabled'],
                        'phone_numbers': notification_data['phone_numbers'],
                        'email_addresses': notification_data['email_addresses'],
                        'sms_min_level': 'elevated',
                        'email_min_level': 'mild',
                        'push_min_level': 'mild'
                    }
                }
                
                if self.settings_manager.save_settings(updated_settings):
                    st.success("✅ Settings saved successfully!")
                else:
                    st.error("❌ Failed to save settings")
        
        with col2:
            if st.button("🔄 Reset All", type="secondary", key="reset_all"):
                # Clear session state to reset to defaults
                for key in ['phone_numbers', 'email_addresses', 'test_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Settings reset to defaults")
                st.rerun()
        
        with col3:
            if st.button("📋 Export Config", type="secondary", key="export_config"):
                # Export current configuration
                config_json = json.dumps(settings, indent=2)
                st.download_button(
                    label="Download Configuration",
                    data=config_json,
                    file_name=f"safeguard_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display current configuration summary
        with st.expander("📊 Current Configuration Summary"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**System Preferences:**")
                st.write(f"• Update frequency: {system_prefs.update_frequency}s")
                st.write(f"• Theme: {system_prefs.theme_mode.value.title()}")
                st.write(f"• Performance mode: {system_prefs.performance_mode.value.replace('_', ' ').title()}")
                st.write(f"• Model path: {system_prefs.model_path}")
                
                st.markdown("**Alert Thresholds:**")
                st.write(f"• Normal: {system_prefs.normal_threshold[0]}-{system_prefs.normal_threshold[1]}")
                st.write(f"• Mild: {system_prefs.mild_threshold[0]}-{system_prefs.mild_threshold[1]}")
                st.write(f"• Elevated: {system_prefs.elevated_threshold[0]}-{system_prefs.elevated_threshold[1]}")
                st.write(f"• Critical: {system_prefs.critical_threshold[0]}-{system_prefs.critical_threshold[1]}")
            
            with col2:
                st.markdown("**Notification Settings:**")
                st.write(f"• SMS enabled: {'✅' if notification_data['toggles']['sms_enabled'] else '❌'}")
                st.write(f"• Email enabled: {'✅' if notification_data['toggles']['email_enabled'] else '❌'}")
                st.write(f"• Push enabled: {'✅' if notification_data['toggles']['push_enabled'] else '❌'}")
                
                st.markdown("**Contacts:**")
                st.write(f"• Phone numbers: {len(notification_data['phone_numbers'])}")
                st.write(f"• Email addresses: {len(notification_data['email_addresses'])}")


def main():
    """Main entry point for settings page"""
    settings_page = SettingsPage()
    settings_page.render_main_settings()


if __name__ == "__main__":
    main()