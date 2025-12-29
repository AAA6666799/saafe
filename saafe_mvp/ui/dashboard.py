"""
Saafe MVP - Main Dashboard Interface
Implements a sleek, minimalist Streamlit dashboard for fire detection demonstration
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
from datetime import datetime
from typing import Dict, List, Optional
import time
import traceback

from ..core.data_models import SensorReading, PredictionResult, AlertData
from ..core.scenario_manager import ScenarioManager
from ..core.fire_detection_pipeline import FireDetectionPipeline
from ..core.alert_engine import AlertEngine
from .sensor_components import SensorGrid
from .ai_analysis_components import AIAnalysisPanel
from .alert_components import AlertStatusPanel


class SafeguardDashboard:
    """Main dashboard interface with real-time updates and professional styling"""
    
    def __init__(self):
        self.scenario_manager = None
        self.fire_pipeline = None
        self.alert_engine = None
        self.sensor_grid = SensorGrid()
        self.ai_analysis_panel = AIAnalysisPanel()
        self.alert_status_panel = AlertStatusPanel()
        self._setup_page_config()
        self._setup_custom_css()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings for professional appearance"""
        st.set_page_config(
            page_title="Saafe Fire Detection MVP",
            page_icon="🔥",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "Saafe Fire Detection MVP - Intelligent fire detection system"
            }
        )
    
    def _initialize_scenario_manager(self, scenario_type: str):
        """Initialize data streaming for the selected scenario"""
        try:
            from ..core.data_stream import get_data_stream_manager
            from ..core.scenario_manager import ScenarioType
            from ..core.data_models import ScenarioConfig
            
            # Create a demo config for faster updates
            demo_config = ScenarioConfig(
                duration_seconds=60,  # 1 minute total duration
                update_frequency=2.0,  # 2 updates per second
                noise_level=0.1
            )
            
            # Get or create data stream manager (ensure it persists across reruns)
            if not hasattr(st.session_state, 'data_stream_manager') or st.session_state.data_stream_manager is None:
                from ..core.data_stream import DataStreamManager  # Import directly to avoid global instance issues
                st.session_state.data_stream_manager = DataStreamManager(demo_config)
                print(f"Created new DataStreamManager for session")
            
            # Only reset if switching to a different scenario
            current_scenario_key = f'current_scenario_{scenario_type}'
            if st.session_state.get('active_scenario_key') != current_scenario_key:
                # Stop any existing streaming
                st.session_state.data_stream_manager.stop_streaming()
                
                # Set new scenario tracking
                st.session_state.active_scenario_key = current_scenario_key
                
                # Start streaming for the new scenario
                scenario_enum = ScenarioType(scenario_type)
                success = st.session_state.data_stream_manager.start_streaming(scenario_enum)
                
                if success:
                    print(f"Started data streaming for {scenario_type} scenario")
                else:
                    st.error(f"Failed to start streaming for {scenario_type} scenario")
            
            # Initialize AI components if needed
            if not hasattr(st.session_state, 'fire_pipeline') or st.session_state.fire_pipeline is None:
                from ..core.fire_detection_pipeline import FireDetectionPipeline
                from ..models.model_manager import ModelManager
                
                # Initialize model manager first
                model_manager = ModelManager()
                model_manager.create_fallback_model()  # Ensure we have a model
                st.session_state.fire_pipeline = FireDetectionPipeline(model_manager, enable_anti_hallucination=False)
            
            if not hasattr(st.session_state, 'alert_engine') or st.session_state.alert_engine is None:
                from ..core.alert_engine import AlertEngine
                st.session_state.alert_engine = AlertEngine()
                
        except Exception as e:
            st.error(f"Error initializing data streaming: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    def _setup_custom_css(self):
        """Apply custom CSS for sleek, minimalist design"""
        st.markdown("""
        <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Header styling */
        .saafe-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0 2rem 0;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        
        .saafe-logo {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ff4b4b;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .saafe-nav {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .nav-icon {
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 0.5rem;
            transition: background-color 0.2s;
        }
        
        .nav-icon:hover {
            background-color: #f0f0f0;
        }
        
        /* Scenario button styling */
        .scenario-buttons {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
            padding: 2rem;
            background: #f8f9fa;
            border-radius: 1rem;
        }
        
        .scenario-button {
            padding: 1rem 2rem;
            border: 2px solid #e0e0e0;
            border-radius: 0.75rem;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            min-width: 180px;
        }
        
        .scenario-button.active {
            border-color: #ff4b4b;
            background: #fff5f5;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.15);
        }
        
        .scenario-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        }
        
        .scenario-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        .scenario-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin: 0 auto;
            background: #e0e0e0;
        }
        
        .scenario-indicator.active {
            background: #ff4b4b;
        }
        
        /* Card styling */
        .dashboard-card {
            background: white;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid #f0f0f0;
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #333;
        }
        
        /* Sensor display styling */
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .sensor-card {
            background: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            border: 1px solid #f0f0f0;
        }
        
        .sensor-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        .sensor-label {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }
        
        .sensor-unit {
            font-size: 0.8rem;
            color: #999;
        }
        
        /* Status indicators */
        .status-normal { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        
        /* Alert panel styling */
        .alert-panel {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .alert-normal {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        
        .alert-warning {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        
        .alert-danger {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .scenario-buttons {
                flex-direction: column;
                gap: 1rem;
            }
            
            .sensor-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
            }
            
            .saafe-header {
                flex-direction: column;
                gap: 1rem;
                text-align: center;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render professional header with logo and navigation"""
        st.markdown("""
        <div class="saafe-header">
            <div class="saafe-logo">
                🔥 Saafe
            </div>
            <div class="saafe-nav">
                <div class="nav-icon" title="Settings">⚙️</div>
                <div class="nav-icon" title="Analytics">📊</div>
                <div class="nav-icon" title="Help">❓</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add settings button functionality
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col4:
            if st.button("⚙️ Settings", key="settings_btn", help="Open settings panel"):
                st.session_state.show_settings = True
                st.rerun()
    
    def render_scenario_controls(self, current_scenario: Optional[str] = None):
        """Render scenario selection buttons with active state indicators"""
        st.markdown("### 🎯 Select Scenario")
        
        # Create three columns for the scenario buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Normal Environment", 
                        key="normal_btn",
                        type="primary" if current_scenario == 'normal' else "secondary",
                        use_container_width=True):
                st.session_state.current_scenario = 'normal'
                self._initialize_scenario_manager('normal')
                st.rerun()
        
        with col2:
            if st.button("🍳 Cooking Activity", 
                        key="cooking_btn",
                        type="primary" if current_scenario == 'cooking' else "secondary",
                        use_container_width=True):
                st.session_state.current_scenario = 'cooking'
                self._initialize_scenario_manager('cooking')
                st.rerun()
        
        with col3:
            if st.button("🔥 Fire Emergency", 
                        key="fire_btn",
                        type="primary" if current_scenario == 'fire' else "secondary",
                        use_container_width=True):
                st.session_state.current_scenario = 'fire'
                self._initialize_scenario_manager('fire')
                st.rerun()
    
    def render_main_dashboard(self):
        """Render the complete main dashboard layout"""
        # Check if settings should be shown
        if st.session_state.get('show_settings', False):
            self.render_settings_page()
            return
        
        # Initialize session state for current scenario
        if 'current_scenario' not in st.session_state:
            st.session_state.current_scenario = None
        
        # Add auto-refresh controls when scenario is running
        if st.session_state.current_scenario:
            # Create controls for auto-refresh
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info("📡 **Live Data Streaming** - Sensor readings update automatically")
            with col2:
                auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_refresh_enabled")
            with col3:
                if st.button("🔄 Manual Refresh", key="manual_refresh"):
                    # Clear any cached data and force refresh
                    if 'last_sensor_timestamp' in st.session_state:
                        del st.session_state.last_sensor_timestamp
                    st.rerun()
            
            # Auto-refresh mechanism - stable approach
            if auto_refresh:
                st.caption("🔄 Auto-refresh enabled - data updates automatically")
                
                # Use a much more conservative refresh approach
                if 'page_load_time' not in st.session_state:
                    st.session_state.page_load_time = time.time()
                
                # Only refresh every 3 seconds to prevent crashes
                current_time = time.time()
                time_since_load = current_time - st.session_state.page_load_time
                
                # Refresh every 3 seconds (much more stable)
                if time_since_load >= 3.0:
                    st.session_state.page_load_time = current_time
                    st.rerun()
        
        # Render header
        self.render_header()
        
        # Render scenario controls
        self.render_scenario_controls(st.session_state.current_scenario)
        
# Initialize scenario start time only if not set
        if 'scenario_start_time' not in st.session_state:
            st.session_state.scenario_start_time = time.time()

        # Get current sensor data from data stream manager (moved up to define variables)
        current_sensor_data = None
        current_prediction = None
        stream_status = None
        scenario_progress_info = None
        
        # Debug: Show session state info
        with st.expander("🔧 Session State Debug", expanded=False):
            st.write(f"Current scenario: {st.session_state.current_scenario}")
            st.write(f"Has data_stream_manager: {hasattr(st.session_state, 'data_stream_manager')}")
            st.write(f"Auto-refresh enabled: {st.session_state.get('auto_refresh_enabled', False)}")
            if hasattr(st.session_state, 'data_stream_manager'):
                st.write(f"Stream manager type: {type(st.session_state.data_stream_manager)}")
                # Test data retrieval
                test_data = st.session_state.data_stream_manager.get_current_reading()
                if test_data:
                    st.write(f"✅ Data available: T={test_data.temperature:.1f}°C at {test_data.timestamp}")
                else:
                    st.write("❌ No data from stream manager")
        
        if st.session_state.current_scenario and hasattr(st.session_state, 'data_stream_manager'):
            try:
                # Get latest sensor reading from data stream manager
                current_sensor_data = st.session_state.data_stream_manager.get_current_reading()
                
                # Get stream status and progress
                stream_status = st.session_state.data_stream_manager.get_stream_status()
                scenario_progress_info = st.session_state.data_stream_manager.get_scenario_progress()
                
                # Debug: Show raw data
                if current_sensor_data:
                    st.info(f"🔍 **DEBUG**: Got live data at {current_sensor_data.timestamp.strftime('%H:%M:%S')} - T={current_sensor_data.temperature:.1f}°C, PM2.5={current_sensor_data.pm25:.1f}")
                else:
                    st.warning("🔍 **DEBUG**: No sensor data from stream manager")
                
                # Get AI prediction if pipeline is available and we have data
                if hasattr(st.session_state, 'fire_pipeline') and current_sensor_data:
                    # Pipeline expects a list of sensor readings
                    sensor_readings_list = [current_sensor_data]
                    current_prediction = st.session_state.fire_pipeline.predict(sensor_readings_list)
            except Exception as e:
                st.error(f"Error getting streaming data: {e}")
                import traceback
                st.error(traceback.format_exc())

        # Main content area
        if st.session_state.current_scenario:
            # Get progress information from data stream manager
            if scenario_progress_info:
                scenario_progress = scenario_progress_info['progress']
                current_phase = scenario_progress_info['phase']
                remaining_time = scenario_progress_info['remaining_time']
                elapsed_time = scenario_progress_info['elapsed_time']
                
                # Get phase description based on scenario and progress
                if st.session_state.current_scenario == 'fire':
                    if scenario_progress < 0.20:
                        phase_desc = "Initial heating, minimal visible signs"
                    elif scenario_progress < 0.35:
                        phase_desc = "Visible flames, rapid development"
                    elif scenario_progress < 0.40:
                        phase_desc = "Sudden intense burning"
                    elif scenario_progress < 0.85:
                        phase_desc = "Peak fire conditions"
                    else:
                        phase_desc = "Fire consuming available fuel"
                elif st.session_state.current_scenario == 'cooking':
                    if scenario_progress < 0.3:
                        phase_desc = "Stove/oven warming up"
                    elif scenario_progress < 0.7:
                        phase_desc = "Food preparation in progress"
                    else:
                        phase_desc = "Finishing cooking process"
                else:  # normal
                    phase_desc = "Normal room conditions"
                
                # Auto-restart scenario when completed
                if scenario_progress >= 1.0:
                    # Restart the scenario automatically
                    try:
                        from ..core.scenario_manager import ScenarioType
                        scenario_enum = ScenarioType(st.session_state.current_scenario)
                        st.session_state.data_stream_manager.start_streaming(scenario_enum)
                        scenario_progress = 0.0
                        current_phase = "🔄 Restarting..."
                        phase_desc = "Scenario restarting automatically"
                    except Exception as e:
                        st.error(f"Error restarting scenario: {e}")
            else:
                # Fallback values if no progress info available
                scenario_progress = 0.0
                current_phase = "Initializing..."
                phase_desc = "Starting scenario..."
                remaining_time = 60
            
            # Display the scenario information
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 🔥 {st.session_state.current_scenario.title()} Scenario")
                st.markdown(f"**Status:** {current_phase}")
                st.caption(phase_desc)
            
            with col2:
                st.markdown("**Progress**")
                st.progress(scenario_progress)
                st.caption(f"{scenario_progress*100:.0f}% complete")
                
                # Show time remaining
                if remaining_time > 0:
                    st.caption(f"⏱️ {remaining_time:.0f}s remaining")
        else:
            st.markdown("""
            <div class="dashboard-card">
                <div class="card-title">Welcome to Saafe Fire Detection MVP</div>
                <p>Select a scenario above to begin the demonstration. The system will simulate realistic sensor environments and show how our AI models detect and classify different situations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sensor readings section
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Real-time Sensor Readings</div>
        </div>
        """, unsafe_allow_html=True)
        

        
        # Display sensor readings - simple and stable approach
        if current_sensor_data:
            # Show live data status
            st.success(
                f"📡 **LIVE DATA** {current_sensor_data.timestamp.strftime('%H:%M:%S')} - "
                f"T={current_sensor_data.temperature:.1f}°C, PM2.5={current_sensor_data.pm25:.1f}, "
                f"CO₂={current_sensor_data.co2:.1f}, Audio={current_sensor_data.audio_level:.1f}dB"
            )
            
            # Render sensor cards directly (no complex containers)
            self.sensor_grid.render_sensor_cards(current_sensor_data)
        else:
            # Show no data message
            st.warning("⚠️ **NO LIVE DATA** - Select a scenario and enable auto-refresh")
            self.sensor_grid.render_sensor_cards(None)
        
        # AI Analysis section
        if st.session_state.current_scenario:
            st.markdown("### 🤖 AI Analysis")
            self.ai_analysis_panel.render_complete_analysis_panel(current_prediction)
        
        # Alert Status and Notifications section
        if st.session_state.current_scenario:
            is_cooking = st.session_state.current_scenario == 'cooking'
            # Convert prediction to alert if available
            current_alert = None
            if current_prediction and hasattr(st.session_state, 'alert_engine'):
                try:
                    current_alert = st.session_state.alert_engine.process_prediction(
                        current_prediction, 
                        current_sensor_data
                    )
                except Exception as e:
                    st.error(f"Error generating alert: {e}")
                    # Create a fallback alert
                    from ..core.alert_engine import AlertLevel, AlertData
                    from datetime import datetime
                    current_alert = AlertData(
                        alert_level=AlertLevel.NORMAL,
                        risk_score=current_prediction.risk_score if current_prediction else 0,
                        confidence=current_prediction.confidence if current_prediction else 0,
                        message="System processing...",
                        timestamp=datetime.now()
                    )
            
            self.alert_status_panel.render_complete_alert_panel(
                current_alert, 
                is_cooking_detected=is_cooking
            )
        
        # Refresh controls and status
        if st.session_state.current_scenario:
            # Initialize refresh counter
            if 'refresh_counter' not in st.session_state:
                st.session_state.refresh_counter = 0
            
            # Status and controls
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                # Show streaming status from data stream manager
                if stream_status and stream_status['streaming']:
                    readings_per_sec = stream_status.get('readings_per_second', 0)
                    st.success(f"🟢 **Streaming Active** - {readings_per_sec:.1f} readings/sec")
                else:
                    st.warning("🟡 **Streaming Inactive** - Click refresh to restart")
            
            with col2:
                if st.button("🔄 Refresh", key="refresh_btn", use_container_width=True):
                    st.session_state.refresh_counter += 1
                    st.rerun()
            
            with col3:
                # Auto-refresh toggle (enabled by default)
                auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_refresh", 
                                         help="Automatically refresh every 2 seconds to show live data")
            
            with col4:
                st.metric("Updates", st.session_state.refresh_counter)
            
            # Auto-refresh mechanism using Streamlit's built-in rerun
            if auto_refresh:
                # Use a more reliable auto-refresh mechanism
                if 'last_refresh_time' not in st.session_state:
                    st.session_state.last_refresh_time = time.time()
                
                current_time = time.time()
                if current_time - st.session_state.last_refresh_time > 2.0:  # 2 second intervals for faster updates
                    st.session_state.last_refresh_time = current_time
                    st.session_state.refresh_counter += 1
                    # Force rerun to get fresh data
                    st.rerun()
                
                # Show countdown
                time_since_refresh = current_time - st.session_state.last_refresh_time
                next_refresh_in = max(0, 2.0 - time_since_refresh)
                st.caption(f"⏱️ Next refresh in {next_refresh_in:.1f}s")
                
                # Also add a JavaScript-based refresh as backup
                st.markdown("""
                <script>
                setTimeout(function(){
                    window.location.reload();
                }, 2000);
                </script>
                """, unsafe_allow_html=True)
        
        # Debug information (collapsed by default)
        if st.session_state.current_scenario:
            with st.expander("🔧 Debug Information", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Streaming Status:**")
                    st.write(f"• Current Scenario: {st.session_state.current_scenario}")
                    st.write(f"• Data Stream Manager: {'✓' if hasattr(st.session_state, 'data_stream_manager') else '✗'}")
                    
                    if stream_status:
                        st.write(f"• Streaming Active: {'✓' if stream_status['streaming'] else '✗'}")
                        st.write(f"• Total Readings: {stream_status['total_readings']}")
                        st.write(f"• Uptime: {stream_status['uptime']:.1f}s")
                        st.write(f"• Buffer Size: {stream_status['buffer_size']}")
                        st.write(f"• Update Frequency: {stream_status['update_frequency']} Hz")
                
                with col2:
                    st.write("**Data Status:**")
                    st.write(f"• Current Data: {'✓' if current_sensor_data else '✗'}")
                    st.write(f"• AI Pipeline: {'✓' if hasattr(st.session_state, 'fire_pipeline') else '✗'}")
                    st.write(f"• Alert Engine: {'✓' if hasattr(st.session_state, 'alert_engine') else '✗'}")
                    
                    if current_sensor_data:
                        st.write("**Latest Reading:**")
                        st.write(f"• Temperature: {current_sensor_data.temperature:.1f}°C")
                        st.write(f"• PM2.5: {current_sensor_data.pm25:.1f} μg/m³")
                        st.write(f"• CO₂: {current_sensor_data.co2:.1f} ppm")
                        st.write(f"• Audio: {current_sensor_data.audio_level:.1f} dB")
                        st.write(f"• Timestamp: {current_sensor_data.timestamp.strftime('%H:%M:%S')}")
                    
                    if scenario_progress_info:
                        st.write("**Progress Info:**")
                        st.write(f"• Progress: {scenario_progress_info['progress']*100:.1f}%")
                        st.write(f"• Elapsed: {scenario_progress_info['elapsed_time']:.1f}s")
                        st.write(f"• Remaining: {scenario_progress_info['remaining_time']:.1f}s")
            
            # Instructions for users
            st.info("💡 **Tip**: The system streams data in real-time. Click the '🔄 Refresh Now' button above to see the latest sensor readings, or enable auto-refresh for continuous updates.")
    
    def render_settings_page(self):
        """Render the settings page"""
        from .settings_page import SettingsPage
        
        # Back button
        if st.button("← Back to Dashboard", key="back_to_dashboard"):
            st.session_state.show_settings = False
            st.rerun()
        
        # Render settings page
        settings_page = SettingsPage()
        settings_page.render_main_settings()


def main():
    """Main entry point for the Streamlit dashboard"""
    dashboard = SafeguardDashboard()
    dashboard.render_main_dashboard()


if __name__ == "__main__":
    main()