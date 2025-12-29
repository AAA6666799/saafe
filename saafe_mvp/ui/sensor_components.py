"""
Saafe MVP - Sensor Display Components
Implements elegant sensor gauges and real-time visualization components
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import time

from ..core.data_models import SensorReading


class SensorDisplay:
    """Individual sensor display with real-time updates and elegant styling"""
    
    def __init__(self, sensor_type: str, unit: str, normal_range: Tuple[float, float], 
                 warning_range: Tuple[float, float], danger_range: Tuple[float, float]):
        self.sensor_type = sensor_type
        self.unit = unit
        self.normal_range = normal_range
        self.warning_range = warning_range
        self.danger_range = danger_range
        self.history = []
        self.max_history = 300  # 5 minutes at 1-second intervals
    
    def get_status_color(self, value: float) -> str:
        """Get color based on sensor value ranges"""
        if self.danger_range[0] <= value <= self.danger_range[1]:
            return "#dc3545"  # Red
        elif self.warning_range[0] <= value <= self.warning_range[1]:
            return "#ffc107"  # Yellow
        else:
            return "#28a745"  # Green
    
    def get_status_text(self, value: float) -> str:
        """Get status text based on sensor value"""
        if self.danger_range[0] <= value <= self.danger_range[1]:
            return "DANGER"
        elif self.warning_range[0] <= value <= self.warning_range[1]:
            return "WARNING"
        else:
            return "NORMAL"
    
    def add_reading(self, value: float, timestamp: datetime):
        """Add new sensor reading to history"""
        self.history.append({'value': value, 'timestamp': timestamp})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def render_gauge(self, current_value: float) -> go.Figure:
        """Render elegant circular gauge for current reading"""
        color = self.get_status_color(current_value)
        
        # Determine gauge range based on sensor type
        if self.sensor_type.lower() == 'temperature':
            gauge_max = 100
        elif self.sensor_type.lower() == 'pm2.5':
            gauge_max = 200
        elif self.sensor_type.lower() == 'co2':
            gauge_max = 2000
        elif self.sensor_type.lower() == 'audio':
            gauge_max = 100
        else:
            gauge_max = max(self.danger_range[1] * 1.2, current_value * 1.2)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{self.sensor_type}<br><span style='font-size:0.8em;color:gray'>{self.unit}</span>"},
            delta = {'reference': (self.normal_range[0] + self.normal_range[1]) / 2},
            gauge = {
                'axis': {'range': [None, gauge_max]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, self.normal_range[1]], 'color': "lightgray"},
                    {'range': [self.warning_range[0], self.warning_range[1]], 'color': "yellow"},
                    {'range': [self.danger_range[0], gauge_max], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': self.danger_range[0]
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333", size=12)
        )
        
        return fig
    
    def render_bar_chart(self, current_value: float) -> go.Figure:
        """Render elegant bar chart visualization"""
        color = self.get_status_color(current_value)
        
        # Determine max value for bar chart
        if self.sensor_type.lower() == 'temperature':
            max_val = 100
        elif self.sensor_type.lower() == 'pm2.5':
            max_val = 200
        elif self.sensor_type.lower() == 'co2':
            max_val = 2000
        elif self.sensor_type.lower() == 'audio':
            max_val = 100
        else:
            max_val = max(self.danger_range[1] * 1.2, current_value * 1.2)
        
        fig = go.Figure()
        
        # Background bar
        fig.add_trace(go.Bar(
            x=[max_val],
            y=[self.sensor_type],
            orientation='h',
            marker_color='#f0f0f0',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Value bar
        fig.add_trace(go.Bar(
            x=[current_value],
            y=[self.sensor_type],
            orientation='h',
            marker_color=color,
            showlegend=False,
            text=f"{current_value:.1f} {self.unit}",
            textposition='inside',
            textfont=dict(color='white', size=14, family='Arial Black')
        ))
        
        fig.update_layout(
            height=80,
            margin=dict(l=100, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=True, zeroline=False),
            barmode='overlay',
            font=dict(color="#333", size=12)
        )
        
        return fig
    
    def render_trend_chart(self) -> go.Figure:
        """Render time-series trend chart with historical data"""
        if not self.history:
            # Return empty chart if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                height=200,
                margin=dict(l=40, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig
        
        # Prepare data
        df = pd.DataFrame(self.history)
        
        # Create trend line
        fig = go.Figure()
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name=self.sensor_type,
            line=dict(color='#007bff', width=2),
            fill='tonexty' if len(df) > 1 else None,
            fillcolor='rgba(0, 123, 255, 0.1)'
        ))
        
        # Add range indicators
        if len(df) > 0:
            x_range = [df['timestamp'].min(), df['timestamp'].max()]
            
            # Normal range
            fig.add_hrect(
                y0=self.normal_range[0], y1=self.normal_range[1],
                fillcolor="rgba(40, 167, 69, 0.1)",
                layer="below", line_width=0,
            )
            
            # Warning range
            fig.add_hrect(
                y0=self.warning_range[0], y1=self.warning_range[1],
                fillcolor="rgba(255, 193, 7, 0.1)",
                layer="below", line_width=0,
            )
            
            # Danger range
            fig.add_hrect(
                y0=self.danger_range[0], y1=self.danger_range[1],
                fillcolor="rgba(220, 53, 69, 0.1)",
                layer="below", line_width=0,
            )
        
        fig.update_layout(
            height=200,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)',
                title="Time"
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)',
                title=f"{self.sensor_type} ({self.unit})"
            ),
            showlegend=False,
            font=dict(color="#333", size=10)
        )
        
        return fig


class SensorGrid:
    """Grid layout for multiple sensor displays"""
    
    def __init__(self):
        self.sensors = {
            'temperature': SensorDisplay(
                'Temperature', '°C', 
                normal_range=(18, 25), 
                warning_range=(25, 35), 
                danger_range=(35, 100)
            ),
            'pm25': SensorDisplay(
                'PM2.5', 'μg/m³', 
                normal_range=(0, 25), 
                warning_range=(25, 75), 
                danger_range=(75, 200)
            ),
            'co2': SensorDisplay(
                'CO₂', 'ppm', 
                normal_range=(300, 500), 
                warning_range=(500, 800), 
                danger_range=(800, 2000)
            ),
            'audio': SensorDisplay(
                'Audio Level', 'dB', 
                normal_range=(30, 50), 
                warning_range=(50, 70), 
                danger_range=(70, 100)
            )
        }
    
    def update_readings(self, sensor_reading: SensorReading):
        """Update all sensor displays with new readings"""
        timestamp = sensor_reading.timestamp
        
        self.sensors['temperature'].add_reading(sensor_reading.temperature, timestamp)
        self.sensors['pm25'].add_reading(sensor_reading.pm25, timestamp)
        self.sensors['co2'].add_reading(sensor_reading.co2, timestamp)
        self.sensors['audio'].add_reading(sensor_reading.audio_level, timestamp)
    
    def render_sensor_cards(self, sensor_reading: Optional[SensorReading] = None):
        """Render sensor cards in a responsive grid layout"""
        if sensor_reading is None:
            # Show "No Data" instead of fake demo data
            st.warning("⚠️ No sensor data available. Please select a scenario and ensure streaming is active.")
            
            # Create 4-column layout for sensors with "No Data" placeholders
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self._render_no_data_card('Temperature', '°C')
            with col2:
                self._render_no_data_card('PM2.5', 'μg/m³')
            with col3:
                self._render_no_data_card('CO₂', 'ppm')
            with col4:
                self._render_no_data_card('Audio Level', 'dB')
            return
        
        # Update readings
        self.update_readings(sensor_reading)
        
        # Create 4-column layout for sensors
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_sensor_card('temperature', sensor_reading.temperature)
        
        with col2:
            self._render_sensor_card('pm25', sensor_reading.pm25)
        
        with col3:
            self._render_sensor_card('co2', sensor_reading.co2)
        
        with col4:
            self._render_sensor_card('audio', sensor_reading.audio_level)
    
    def _render_no_data_card(self, sensor_name: str, unit: str):
        """Render a card showing no data available"""
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 0.75rem;
            padding: 1.5rem;
            text-align: center;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 2rem; color: #6c757d; margin-bottom: 0.5rem;">--</div>
            <div style="font-weight: 600; color: #495057; margin-bottom: 0.25rem;">{sensor_name}</div>
            <div style="font-size: 0.8rem; color: #6c757d;">{unit}</div>
            <div style="font-size: 0.7rem; color: #6c757d; margin-top: 0.5rem;">NO DATA</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sensor_card(self, sensor_key: str, current_value: float):
        """Render individual sensor card"""
        sensor = self.sensors[sensor_key]
        
        # Card container - simplified to prevent conflicts
        with st.container():
            # Sensor value and status
            status_color = sensor.get_status_color(current_value)
            status_text = sensor.get_status_text(current_value)
            
            # Display current value with large text (add timestamp to force refresh)
            timestamp_ms = int(time.time() * 1000)
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 0.75rem; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; margin-bottom: 1rem;"
                        data-timestamp="{timestamp_ms}">
                <div style="font-size: 2rem; font-weight: 700; color: {status_color}; margin-bottom: 0.5rem;">
                    {current_value:.1f}
                </div>
                <div style="font-size: 0.9rem; color: #666; font-weight: 500; margin-bottom: 0.5rem;">
                    {sensor.sensor_type}
                </div>
                <div style="font-size: 0.8rem; color: #999;">
                    {sensor.unit}
                </div>
                <div style="font-size: 0.8rem; color: {status_color}; font-weight: 600; margin-top: 0.5rem;">
                    {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bar chart visualization - simplified
            bar_fig = sensor.render_bar_chart(current_value)
            st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_trend_charts(self):
        """Render trend charts for all sensors"""
        st.markdown("### Sensor Trends")
        
        # Create 2x2 grid for trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Temperature & PM2.5")
            temp_fig = self.sensors['temperature'].render_trend_chart()
            st.plotly_chart(temp_fig, use_container_width=True, config={'displayModeBar': False})
            
            pm25_fig = self.sensors['pm25'].render_trend_chart()
            st.plotly_chart(pm25_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("#### CO₂ & Audio Level")
            co2_fig = self.sensors['co2'].render_trend_chart()
            st.plotly_chart(co2_fig, use_container_width=True, config={'displayModeBar': False})
            
            audio_fig = self.sensors['audio'].render_trend_chart()
            st.plotly_chart(audio_fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_combined_overview(self, sensor_reading: Optional[SensorReading] = None):
        """Render combined sensor overview with gauges"""
        if sensor_reading is None:
            # Generate sample data
            sensor_reading = SensorReading(
                timestamp=datetime.now(),
                temperature=22.5,
                pm25=12.3,
                co2=420.0,
                audio_level=35.2,
                location="Demo"
            )
        
        st.markdown("### Sensor Overview")
        
        # Create 2x2 grid for gauges
        col1, col2 = st.columns(2)
        
        with col1:
            temp_gauge = self.sensors['temperature'].render_gauge(sensor_reading.temperature)
            st.plotly_chart(temp_gauge, use_container_width=True, config={'displayModeBar': False})
            
            pm25_gauge = self.sensors['pm25'].render_gauge(sensor_reading.pm25)
            st.plotly_chart(pm25_gauge, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            co2_gauge = self.sensors['co2'].render_gauge(sensor_reading.co2)
            st.plotly_chart(co2_gauge, use_container_width=True, config={'displayModeBar': False})
            
            audio_gauge = self.sensors['audio'].render_gauge(sensor_reading.audio_level)
            st.plotly_chart(audio_gauge, use_container_width=True, config={'displayModeBar': False})