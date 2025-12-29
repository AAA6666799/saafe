"""
UI components for export and performance monitoring functionality.

This module provides Streamlit components for:
- Session data export controls
- Performance metrics display
- Export status and history
- System diagnostics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
import json

from ..services.session_manager import SessionManager
from ..services.export_service import ExportConfig
from ..services.performance_monitor import PerformanceMonitor


class ExportControlPanel:
    """Export control panel for the dashboard."""
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize export control panel.
        
        Args:
            session_manager (SessionManager): Session manager instance
        """
        self.session_manager = session_manager
    
    def render(self):
        """Render the export control panel."""
        st.subheader("📊 Export & Reports")
        
        # Current session info
        if self.session_manager.current_session_id:
            stats = self.session_manager.get_session_statistics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Session Duration",
                    f"{stats.get('duration_minutes', 0):.1f} min"
                )
            
            with col2:
                st.metric(
                    "Data Points",
                    f"{stats.get('total_readings', 0):,}"
                )
            
            with col3:
                st.metric(
                    "Predictions",
                    f"{stats.get('total_predictions', 0):,}"
                )
            
            # Export controls
            st.write("**Export Current Session**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                export_formats = st.multiselect(
                    "Select Export Formats",
                    options=['JSON', 'CSV', 'PDF'],
                    default=['JSON', 'CSV'],
                    help="Choose which formats to export"
                )
            
            with col2:
                if st.button("🚀 Export Now", type="primary"):
                    if export_formats:
                        self._perform_export(export_formats)
                    else:
                        st.warning("Please select at least one export format")
            
            # Auto-export settings
            with st.expander("⚙️ Auto-Export Settings"):
                auto_export = st.checkbox(
                    "Enable Auto-Export",
                    value=False,
                    help="Automatically export session data at regular intervals"
                )
                
                if auto_export:
                    interval = st.slider(
                        "Export Interval (minutes)",
                        min_value=5,
                        max_value=60,
                        value=15,
                        help="How often to automatically export data"
                    )
                    
                    auto_formats = st.multiselect(
                        "Auto-Export Formats",
                        options=['JSON', 'CSV'],
                        default=['JSON'],
                        help="Formats for automatic exports"
                    )
        else:
            st.info("No active session. Start a scenario to begin data collection.")
    
    def _perform_export(self, formats: List[str]):
        """Perform export operation."""
        try:
            # Convert format names to lowercase
            format_list = [fmt.lower() for fmt in formats]
            
            with st.spinner("Exporting session data..."):
                results = self.session_manager.export_current_session(
                    formats=format_list,
                    immediate=True
                )
            
            st.success("✅ Export completed successfully!")
            
            # Display export results
            for format_name, file_path in results.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    st.write(f"📄 **{format_name.upper()}**: `{os.path.basename(file_path)}` ({file_size:.1f} KB)")
                    
                    # Provide download link for smaller files
                    if file_size < 1024:  # Less than 1MB
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f"Download {format_name.upper()}",
                                data=f.read(),
                                file_name=os.path.basename(file_path),
                                mime=self._get_mime_type(format_name)
                            )
        
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
    
    def _get_mime_type(self, format_name: str) -> str:
        """Get MIME type for file format."""
        mime_types = {
            'json': 'application/json',
            'csv': 'text/csv',
            'pdf': 'application/pdf'
        }
        return mime_types.get(format_name.lower(), 'application/octet-stream')


class PerformanceMetricsPanel:
    """Performance metrics display panel."""
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize performance metrics panel.
        
        Args:
            session_manager (SessionManager): Session manager instance
        """
        self.session_manager = session_manager
    
    def render(self):
        """Render the performance metrics panel."""
        st.subheader("⚡ Performance Metrics")
        
        if not self.session_manager.performance_monitor:
            st.warning("Performance monitoring is disabled")
            return
        
        # Get performance summary
        perf_summary = self.session_manager.get_performance_summary()
        
        if 'reliability' in perf_summary:
            reliability = perf_summary['reliability']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Success Rate",
                    f"{reliability.get('success_rate', 0):.1%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Avg Processing Time",
                    f"{reliability.get('average_processing_time', 0):.1f}ms",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Total Predictions",
                    f"{reliability.get('total_predictions', 0):,}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "System Uptime",
                    f"{reliability.get('uptime_seconds', 0) / 3600:.1f}h",
                    delta=None
                )
            
            # Processing time chart
            if 'processing_times' in perf_summary:
                processing_stats = perf_summary['processing_times']
                
                if 'prediction' in processing_stats:
                    pred_stats = processing_stats['prediction']
                    
                    st.write("**Processing Time Distribution**")
                    
                    # Create processing time metrics chart
                    metrics_data = {
                        'Metric': ['Mean', 'Median', 'P95', 'P99', 'Max'],
                        'Time (ms)': [
                            pred_stats.get('mean', 0),
                            pred_stats.get('median', 0),
                            pred_stats.get('p95', 0),
                            pred_stats.get('p99', 0),
                            pred_stats.get('max', 0)
                        ]
                    }
                    
                    fig = px.bar(
                        x=metrics_data['Metric'],
                        y=metrics_data['Time (ms)'],
                        title="Processing Time Statistics",
                        labels={'x': 'Metric', 'y': 'Time (ms)'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # System resource usage
            if 'resource_usage' in perf_summary:
                resource_stats = perf_summary['resource_usage']
                
                st.write("**System Resource Usage**")
                
                resource_cols = st.columns(2)
                
                with resource_cols[0]:
                    if 'cpu' in resource_stats:
                        cpu_stats = resource_stats['cpu']
                        st.metric(
                            "CPU Usage",
                            f"{cpu_stats.get('current', 0):.1f}%",
                            delta=f"Avg: {cpu_stats.get('mean', 0):.1f}%"
                        )
                
                with resource_cols[1]:
                    if 'memory' in resource_stats:
                        memory_stats = resource_stats['memory']
                        st.metric(
                            "Memory Usage",
                            f"{memory_stats.get('current', 0):.1f}%",
                            delta=f"Avg: {memory_stats.get('mean', 0):.1f}%"
                        )
            
            # Performance alerts
            if 'alerts' in perf_summary:
                alerts = perf_summary['alerts']
                
                if alerts:
                    st.write("**Performance Alerts**")
                    for alert in alerts:
                        severity_icon = "🔴" if alert['severity'] == 'critical' else "⚠️"
                        st.warning(f"{severity_icon} {alert['message']}")
        
        # Diagnostic information
        with st.expander("🔧 System Diagnostics"):
            self._render_diagnostics()
    
    def _render_diagnostics(self):
        """Render system diagnostic information."""
        if not self.session_manager.performance_monitor:
            st.write("Performance monitoring not available")
            return
        
        diagnostic_info = self.session_manager.performance_monitor.get_diagnostic_info()
        
        # System information
        if 'system_info' in diagnostic_info:
            st.write("**System Information**")
            sys_info = diagnostic_info['system_info']
            
            info_data = {
                'Property': ['Python Version', 'Platform', 'CPU Cores', 'Total Memory', 'GPU Available'],
                'Value': [
                    sys_info.get('python_version', 'Unknown'),
                    sys_info.get('platform', 'Unknown'),
                    f"{sys_info.get('cpu_count', 0)} cores",
                    f"{sys_info.get('total_memory_gb', 0):.1f} GB",
                    "Yes" if sys_info.get('gpu_available', False) else "No"
                ]
            }
            
            df = pd.DataFrame(info_data)
            st.dataframe(df, hide_index=True)
        
        # Recent errors
        if 'recent_errors' in diagnostic_info:
            recent_errors = diagnostic_info['recent_errors']
            
            if recent_errors:
                st.write("**Recent Errors**")
                for error in recent_errors[-5:]:  # Show last 5 errors
                    st.error(f"🕐 {error['timestamp']}: {error['error']}")
            else:
                st.success("✅ No recent errors")
        
        # Performance recommendations
        if 'recommendations' in diagnostic_info:
            recommendations = diagnostic_info['recommendations']
            
            if recommendations:
                st.write("**Performance Recommendations**")
                for rec in recommendations:
                    st.info(f"💡 {rec}")


class ExportHistoryPanel:
    """Panel for viewing export history and status."""
    
    def __init__(self, export_directory: str = "exports"):
        """
        Initialize export history panel.
        
        Args:
            export_directory (str): Directory where exports are stored
        """
        self.export_directory = export_directory
    
    def render(self):
        """Render the export history panel."""
        st.subheader("📁 Export History")
        
        if not os.path.exists(self.export_directory):
            st.info("No exports found. Export some session data to see history here.")
            return
        
        # Get list of export files
        export_files = []
        for filename in os.listdir(self.export_directory):
            if filename.startswith('safeguard_session_'):
                file_path = os.path.join(self.export_directory, filename)
                file_stat = os.stat(file_path)
                
                export_files.append({
                    'filename': filename,
                    'size_kb': file_stat.st_size / 1024,
                    'created': datetime.fromtimestamp(file_stat.st_ctime),
                    'format': filename.split('.')[-1].upper(),
                    'path': file_path
                })
        
        if not export_files:
            st.info("No export files found.")
            return
        
        # Sort by creation time (newest first)
        export_files.sort(key=lambda x: x['created'], reverse=True)
        
        # Display export files
        for export_file in export_files[:10]:  # Show last 10 exports
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"📄 **{export_file['filename']}**")
                    st.caption(f"Created: {export_file['created'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    st.write(f"**{export_file['format']}**")
                
                with col3:
                    st.write(f"**{export_file['size_kb']:.1f} KB**")
                
                with col4:
                    # Download button
                    try:
                        with open(export_file['path'], 'rb') as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f.read(),
                                file_name=export_file['filename'],
                                key=f"download_{export_file['filename']}"
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                st.divider()


def render_export_and_performance_tab(session_manager: SessionManager):
    """
    Render the complete export and performance monitoring tab.
    
    Args:
        session_manager (SessionManager): Session manager instance
    """
    st.header("📊 Export & Performance")
    
    # Create tabs for different sections
    export_tab, performance_tab, history_tab = st.tabs([
        "📤 Export Controls",
        "⚡ Performance Metrics", 
        "📁 Export History"
    ])
    
    with export_tab:
        export_panel = ExportControlPanel(session_manager)
        export_panel.render()
    
    with performance_tab:
        performance_panel = PerformanceMetricsPanel(session_manager)
        performance_panel.render()
    
    with history_tab:
        history_panel = ExportHistoryPanel()
        history_panel.render()


def render_performance_sidebar(session_manager: SessionManager):
    """
    Render performance metrics in the sidebar.
    
    Args:
        session_manager (SessionManager): Session manager instance
    """
    st.sidebar.subheader("⚡ Performance")
    
    if session_manager.performance_monitor:
        perf_summary = session_manager.get_performance_summary()
        
        if 'reliability' in perf_summary:
            reliability = perf_summary['reliability']
            
            # Key metrics in sidebar
            st.sidebar.metric(
                "Success Rate",
                f"{reliability.get('success_rate', 0):.1%}"
            )
            
            st.sidebar.metric(
                "Avg Processing",
                f"{reliability.get('average_processing_time', 0):.1f}ms"
            )
            
            # System status indicator
            if 'alerts' in perf_summary and perf_summary['alerts']:
                st.sidebar.error("⚠️ Performance Issues")
            else:
                st.sidebar.success("✅ System Healthy")
    else:
        st.sidebar.info("Performance monitoring disabled")


def render_export_quick_actions(session_manager: SessionManager):
    """
    Render quick export actions in the sidebar.
    
    Args:
        session_manager (SessionManager): Session manager instance
    """
    st.sidebar.subheader("📤 Quick Export")
    
    if session_manager.current_session_id:
        if st.sidebar.button("📊 Export JSON", key="quick_json"):
            try:
                results = session_manager.export_current_session(formats=['json'])
                st.sidebar.success("✅ JSON exported!")
            except Exception as e:
                st.sidebar.error(f"❌ Export failed: {e}")
        
        if st.sidebar.button("📈 Export CSV", key="quick_csv"):
            try:
                results = session_manager.export_current_session(formats=['csv'])
                st.sidebar.success("✅ CSV exported!")
            except Exception as e:
                st.sidebar.error(f"❌ Export failed: {e}")
    else:
        st.sidebar.info("No active session")