"""
Saafe MVP - Settings and Configuration Interface Components
Implements clean, professional settings panels for notification configuration,
alert thresholds, and system preferences
"""

import streamlit as st
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..services.notification_manager import NotificationManager, NotificationConfig, AlertLevel
from ..services.sms_service import SMSConfig
from ..services.email_service import EmailConfig
from ..services.push_notification_service import PushConfig


class ThemeMode(Enum):
    """UI theme modes"""
    LIGHT = "light"
    DARK = "dark"


class PerformanceMode(Enum):
    """System performance modes"""
    BALANCED = "balanced"
    HIGH_ACCURACY = "high_accuracy"
    FAST_RESPONSE = "fast_response"


@dataclass
class SystemPreferences:
    """System configuration preferences"""
    update_frequency: int = 1  # seconds
    theme_mode: ThemeMode = ThemeMode.LIGHT
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    model_path: str = "models/saved"
    
    # Alert thresholds (risk score ranges)
    normal_threshold: Tuple[int, int] = (0, 30)
    mild_threshold: Tuple[int, int] = (31, 50)
    elevated_threshold: Tuple[int, int] = (51, 85)
    critical_threshold: Tuple[int, int] = (86, 100)


class NotificationConfigPanel:
    """Notification configuration panel with clean interface"""
    
    def __init__(self):
        self.notification_manager = None
        self._setup_custom_css()
    
    def _setup_custom_css(self):
        """Setup custom CSS for settings interface"""
        st.markdown("""
        <style>
        /* Settings panel styling */
        .settings-panel {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid #f0f0f0;
        }
        
        .settings-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #333;
            border-bottom: 2px solid #ff4b4b;
            padding-bottom: 0.5rem;
        }
        
        .settings-section {
            margin-bottom: 2rem;
        }
        
        .settings-section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #555;
        }
        
        .contact-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .contact-item .contact-text {
            font-family: monospace;
            color: #333;
        }
        
        .contact-item .remove-btn {
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 0.25rem;
            padding: 0.25rem 0.5rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        
        .test-result {
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            font-weight: 500;
        }
        
        .test-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .test-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .test-warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        /* Toggle switch styling */
        .toggle-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .toggle-label {
            font-weight: 500;
            color: #333;
            min-width: 100px;
        }
        
        /* Threshold slider styling */
        .threshold-container {
            margin-bottom: 1.5rem;
        }
        
        .threshold-label {
            font-weight: 500;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .threshold-range {
            font-size: 0.9rem;
            color: #666;
            margin-left: 1rem;
        }
        
        /* Performance warning */
        .performance-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-top: 0.5rem;
            color: #856404;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _validate_phone_number(self, phone: str) -> Tuple[bool, str]:
        """Validate phone number format"""
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        
        # Check if it's a valid US phone number (10 or 11 digits)
        if len(digits_only) == 10:
            formatted = f"+1-{digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"
            return True, formatted
        elif len(digits_only) == 11 and digits_only.startswith('1'):
            formatted = f"+{digits_only[0]}-{digits_only[1:4]}-{digits_only[4:7]}-{digits_only[7:]}"
            return True, formatted
        else:
            return False, "Invalid phone number format"
    
    def _validate_email(self, email: str) -> Tuple[bool, str]:
        """Validate email address format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email.strip()):
            return True, email.strip().lower()
        return False, "Invalid email address format"
    
    def render_notification_toggles(self) -> Dict[str, bool]:
        """Render notification service toggle switches"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Notification Services</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sms_enabled = st.checkbox("📱 SMS Alerts", value=True, key="sms_enabled")
        
        with col2:
            email_enabled = st.checkbox("📧 Email Alerts", value=True, key="email_enabled")
        
        with col3:
            push_enabled = st.checkbox("🔔 Push Notifications", value=True, key="push_enabled")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'sms_enabled': sms_enabled,
            'email_enabled': email_enabled,
            'push_enabled': push_enabled
        }
    
    def render_phone_number_management(self) -> List[str]:
        """Render phone number management interface"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">SMS Phone Numbers</div>', unsafe_allow_html=True)
        
        # Initialize phone numbers in session state
        if 'phone_numbers' not in st.session_state:
            st.session_state.phone_numbers = []
        
        # Add new phone number
        col1, col2 = st.columns([3, 1])
        with col1:
            new_phone = st.text_input("Add phone number", placeholder="+1-555-123-4567", key="new_phone")
        with col2:
            if st.button("Add", key="add_phone"):
                if new_phone:
                    is_valid, formatted = self._validate_phone_number(new_phone)
                    if is_valid:
                        if formatted not in st.session_state.phone_numbers:
                            st.session_state.phone_numbers.append(formatted)
                            st.success(f"Added: {formatted}")
                            st.rerun()
                        else:
                            st.warning("Phone number already exists")
                    else:
                        st.error(formatted)  # Error message
        
        # Display existing phone numbers
        if st.session_state.phone_numbers:
            st.markdown("**Current phone numbers:**")
            for i, phone in enumerate(st.session_state.phone_numbers):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(phone)
                with col2:
                    if st.button("Remove", key=f"remove_phone_{i}"):
                        st.session_state.phone_numbers.remove(phone)
                        st.rerun()
        else:
            st.info("No phone numbers configured")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return st.session_state.phone_numbers.copy()
    
    def render_email_management(self) -> List[str]:
        """Render email address management interface"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Email Addresses</div>', unsafe_allow_html=True)
        
        # Initialize email addresses in session state
        if 'email_addresses' not in st.session_state:
            st.session_state.email_addresses = []
        
        # Add new email address
        col1, col2 = st.columns([3, 1])
        with col1:
            new_email = st.text_input("Add email address", placeholder="user@example.com", key="new_email")
        with col2:
            if st.button("Add", key="add_email"):
                if new_email:
                    is_valid, formatted = self._validate_email(new_email)
                    if is_valid:
                        if formatted not in st.session_state.email_addresses:
                            st.session_state.email_addresses.append(formatted)
                            st.success(f"Added: {formatted}")
                            st.rerun()
                        else:
                            st.warning("Email address already exists")
                    else:
                        st.error(formatted)  # Error message
        
        # Display existing email addresses
        if st.session_state.email_addresses:
            st.markdown("**Current email addresses:**")
            for i, email in enumerate(st.session_state.email_addresses):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(email)
                with col2:
                    if st.button("Remove", key=f"remove_email_{i}"):
                        st.session_state.email_addresses.remove(email)
                        st.rerun()
        else:
            st.info("No email addresses configured")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return st.session_state.email_addresses.copy()
    
    def render_notification_testing(self, phone_numbers: List[str], email_addresses: List[str]) -> Optional[str]:
        """Render notification testing interface"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Test Notifications</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        test_result = None
        
        with col1:
            if st.button("Test SMS", key="test_sms", disabled=not phone_numbers):
                test_result = "sms"
        
        with col2:
            if st.button("Test Email", key="test_email", disabled=not email_addresses):
                test_result = "email"
        
        with col3:
            if st.button("Test Push", key="test_push"):
                test_result = "push"
        
        with col4:
            if st.button("Test All", key="test_all", disabled=not (phone_numbers or email_addresses)):
                test_result = "all"
        
        # Display test results
        if 'test_results' in st.session_state and st.session_state.test_results:
            results = st.session_state.test_results
            if results['success']:
                st.markdown(f'<div class="test-result test-success">✅ {results["message"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="test-result test-error">❌ {results["message"]}</div>', 
                           unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return test_result
    
    def render_complete_notification_panel(self) -> Dict[str, Any]:
        """Render complete notification configuration panel"""
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        st.markdown('<div class="settings-title">📱 Notification Configuration</div>', unsafe_allow_html=True)
        
        # Render notification toggles
        toggles = self.render_notification_toggles()
        
        # Render contact management
        phone_numbers = self.render_phone_number_management()
        email_addresses = self.render_email_management()
        
        # Render testing interface
        test_action = self.render_notification_testing(phone_numbers, email_addresses)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'toggles': toggles,
            'phone_numbers': phone_numbers,
            'email_addresses': email_addresses,
            'test_action': test_action
        }


class AlertThresholdPanel:
    """Alert threshold configuration panel"""
    
    def __init__(self):
        self._setup_custom_css()
    
    def _setup_custom_css(self):
        """Setup custom CSS for threshold panel"""
        st.markdown("""
        <style>
        .threshold-panel {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid #f0f0f0;
        }
        
        .threshold-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        
        .threshold-normal { border-left-color: #28a745; }
        .threshold-mild { border-left-color: #ffc107; }
        .threshold-elevated { border-left-color: #fd7e14; }
        .threshold-critical { border-left-color: #dc3545; }
        
        .threshold-name {
            font-weight: 600;
            color: #333;
        }
        
        .threshold-range {
            font-family: monospace;
            color: #666;
            background: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
        }
        
        .impact-preview {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        .impact-title {
            font-weight: 600;
            color: #1976d2;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_threshold_sliders(self, current_prefs: SystemPreferences) -> SystemPreferences:
        """Render threshold adjustment sliders"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Risk Score Thresholds</div>', unsafe_allow_html=True)
        
        # Normal threshold (0-30 default)
        normal_max = st.slider(
            "Normal → Mild Anomaly threshold",
            min_value=10,
            max_value=50,
            value=current_prefs.normal_threshold[1],
            step=1,
            key="normal_threshold",
            help="Risk scores below this value are considered normal"
        )
        
        # Mild threshold (31-50 default)  
        mild_max = st.slider(
            "Mild Anomaly → Elevated threshold",
            min_value=normal_max + 1,
            max_value=80,
            value=max(current_prefs.mild_threshold[1], normal_max + 1),
            step=1,
            key="mild_threshold",
            help="Risk scores in this range indicate mild anomalies"
        )
        
        # Elevated threshold (51-85 default)
        elevated_max = st.slider(
            "Elevated → Critical threshold", 
            min_value=mild_max + 1,
            max_value=95,
            value=max(current_prefs.elevated_threshold[1], mild_max + 1),
            step=1,
            key="elevated_threshold",
            help="Risk scores in this range indicate elevated risk"
        )
        
        # Update preferences
        updated_prefs = SystemPreferences(
            update_frequency=current_prefs.update_frequency,
            theme_mode=current_prefs.theme_mode,
            performance_mode=current_prefs.performance_mode,
            model_path=current_prefs.model_path,
            normal_threshold=(0, normal_max),
            mild_threshold=(normal_max + 1, mild_max),
            elevated_threshold=(mild_max + 1, elevated_max),
            critical_threshold=(elevated_max + 1, 100)
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return updated_prefs
    
    def render_threshold_preview(self, prefs: SystemPreferences, current_risk_score: float = 25):
        """Render threshold impact preview"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Current Thresholds</div>', unsafe_allow_html=True)
        
        # Display threshold ranges
        thresholds = [
            ("Normal", prefs.normal_threshold, "threshold-normal"),
            ("Mild Anomaly", prefs.mild_threshold, "threshold-mild"),
            ("Elevated", prefs.elevated_threshold, "threshold-elevated"),
            ("Critical", prefs.critical_threshold, "threshold-critical")
        ]
        
        for name, (min_val, max_val), css_class in thresholds:
            st.markdown(f"""
            <div class="threshold-item {css_class}">
                <div class="threshold-name">{name}</div>
                <div class="threshold-range">{min_val} - {max_val}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show impact on current risk score
        def get_alert_level(score: float) -> str:
            if score <= prefs.normal_threshold[1]:
                return "Normal"
            elif score <= prefs.mild_threshold[1]:
                return "Mild Anomaly"
            elif score <= prefs.elevated_threshold[1]:
                return "Elevated"
            else:
                return "Critical"
        
        current_level = get_alert_level(current_risk_score)
        
        st.markdown(f"""
        <div class="impact-preview">
            <div class="impact-title">Impact Preview</div>
            <div>Current risk score: <strong>{current_risk_score}</strong></div>
            <div>Alert level: <strong>{current_level}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_reset_button(self) -> bool:
        """Render reset to defaults button"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            reset_clicked = st.button("Reset to Defaults", key="reset_thresholds", type="secondary")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return reset_clicked
    
    def render_complete_threshold_panel(self, current_prefs: SystemPreferences, current_risk_score: float = 25) -> Tuple[SystemPreferences, bool]:
        """Render complete alert threshold configuration panel"""
        st.markdown('<div class="threshold-panel">', unsafe_allow_html=True)
        st.markdown('<div class="settings-title">⚠️ Alert Thresholds</div>', unsafe_allow_html=True)
        
        # Render threshold sliders
        updated_prefs = self.render_threshold_sliders(current_prefs)
        
        # Render threshold preview
        self.render_threshold_preview(updated_prefs, current_risk_score)
        
        # Render reset button
        reset_clicked = self.render_reset_button()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return updated_prefs, reset_clicked


class SystemPreferencesPanel:
    """System preferences configuration panel"""
    
    def __init__(self):
        self._setup_custom_css()
    
    def _setup_custom_css(self):
        """Setup custom CSS for system preferences panel"""
        st.markdown("""
        <style>
        .preferences-panel {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid #f0f0f0;
        }
        
        .preference-item {
            margin-bottom: 1.5rem;
        }
        
        .preference-label {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .preference-description {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_update_frequency(self, current_frequency: int) -> int:
        """Render update frequency configuration"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Update Frequency</div>', unsafe_allow_html=True)
        
        frequency = st.selectbox(
            "Data update interval",
            options=[0.5, 1, 2, 5],
            index=[0.5, 1, 2, 5].index(current_frequency) if current_frequency in [0.5, 1, 2, 5] else 1,
            format_func=lambda x: f"{x} second{'s' if x != 1 else ''}",
            key="update_frequency",
            help="How often sensor data and AI analysis updates"
        )
        
        # Performance impact warning
        if frequency < 1:
            st.markdown("""
            <div class="performance-warning">
                ⚠️ High update frequencies may impact system performance on slower computers
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return frequency
    
    def render_theme_selection(self, current_theme: ThemeMode) -> ThemeMode:
        """Render theme selection"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Theme</div>', unsafe_allow_html=True)
        
        theme_options = {
            "Light": ThemeMode.LIGHT,
            "Dark": ThemeMode.DARK
        }
        
        selected_theme = st.selectbox(
            "Interface theme",
            options=list(theme_options.keys()),
            index=0 if current_theme == ThemeMode.LIGHT else 1,
            key="theme_mode",
            help="Choose between light and dark interface themes"
        )
        
        st.info("Theme changes will take effect on next restart")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return theme_options[selected_theme]
    
    def render_performance_mode(self, current_mode: PerformanceMode) -> PerformanceMode:
        """Render performance mode selection"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Performance Mode</div>', unsafe_allow_html=True)
        
        mode_options = {
            "Balanced": PerformanceMode.BALANCED,
            "High Accuracy": PerformanceMode.HIGH_ACCURACY,
            "Fast Response": PerformanceMode.FAST_RESPONSE
        }
        
        mode_descriptions = {
            "Balanced": "Optimal balance of speed and accuracy",
            "High Accuracy": "Maximum accuracy, slower processing",
            "Fast Response": "Fastest processing, reduced accuracy"
        }
        
        current_index = list(mode_options.values()).index(current_mode)
        
        selected_mode = st.selectbox(
            "AI processing mode",
            options=list(mode_options.keys()),
            index=current_index,
            key="performance_mode",
            help="Choose processing mode based on your priorities"
        )
        
        st.info(mode_descriptions[selected_mode])
        
        st.markdown('</div>', unsafe_allow_html=True)
        return mode_options[selected_mode]
    
    def render_model_path_config(self, current_path: str) -> str:
        """Render model path configuration"""
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-section-title">Model Configuration</div>', unsafe_allow_html=True)
        
        model_path = st.text_input(
            "Model directory path",
            value=current_path,
            key="model_path",
            help="Path to directory containing AI model files"
        )
        
        # Model path validation
        import os
        if model_path and not os.path.exists(model_path):
            st.warning(f"⚠️ Path does not exist: {model_path}")
        elif model_path and os.path.exists(model_path):
            st.success(f"✅ Valid path: {model_path}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model_path
    
    def render_complete_preferences_panel(self, current_prefs: SystemPreferences) -> SystemPreferences:
        """Render complete system preferences panel"""
        st.markdown('<div class="preferences-panel">', unsafe_allow_html=True)
        st.markdown('<div class="settings-title">⚙️ System Preferences</div>', unsafe_allow_html=True)
        
        # Render all preference sections
        frequency = self.render_update_frequency(current_prefs.update_frequency)
        theme = self.render_theme_selection(current_prefs.theme_mode)
        performance = self.render_performance_mode(current_prefs.performance_mode)
        model_path = self.render_model_path_config(current_prefs.model_path)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Return updated preferences
        return SystemPreferences(
            update_frequency=frequency,
            theme_mode=theme,
            performance_mode=performance,
            model_path=model_path,
            normal_threshold=current_prefs.normal_threshold,
            mild_threshold=current_prefs.mild_threshold,
            elevated_threshold=current_prefs.elevated_threshold,
            critical_threshold=current_prefs.critical_threshold
        )