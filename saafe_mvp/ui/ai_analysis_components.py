"""
Saafe MVP - AI Analysis Display Components
Implements risk score visualization, confidence displays, and model transparency features
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.data_models import PredictionResult, AlertData


class AIAnalysisPanel:
    """AI analysis display panel with risk scores, confidence, and transparency"""
    
    def __init__(self):
        self.risk_history = []
        self.confidence_history = []
        self.max_history = 300  # 5 minutes at 1-second intervals
        
        # Alert level thresholds
        self.alert_thresholds = {
            'normal': (0, 30),
            'mild': (31, 50),
            'elevated': (51, 85),
            'critical': (86, 100)
        }
        
        # Alert level colors
        self.alert_colors = {
            'normal': '#28a745',
            'mild': '#ffc107',
            'elevated': '#fd7e14',
            'critical': '#dc3545'
        }
    
    def get_alert_level(self, risk_score: float) -> str:
        """Determine alert level based on risk score"""
        if risk_score <= self.alert_thresholds['normal'][1]:
            return 'normal'
        elif risk_score <= self.alert_thresholds['mild'][1]:
            return 'mild'
        elif risk_score <= self.alert_thresholds['elevated'][1]:
            return 'elevated'
        else:
            return 'critical'
    
    def get_alert_text(self, alert_level: str) -> str:
        """Get human-readable alert text"""
        alert_texts = {
            'normal': 'Normal',
            'mild': 'Mild Anomaly',
            'elevated': 'Elevated Risk',
            'critical': 'CRITICAL ALERT'
        }
        return alert_texts.get(alert_level, 'Unknown')
    
    def add_prediction(self, prediction: PredictionResult):
        """Add new prediction to history"""
        self.risk_history.append({
            'timestamp': datetime.now(),
            'risk_score': prediction.risk_score,
            'predicted_class': prediction.predicted_class
        })
        
        self.confidence_history.append({
            'timestamp': datetime.now(),
            'confidence': prediction.confidence,
            'processing_time': prediction.processing_time
        })
        
        # Maintain history size
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)
    
    def render_risk_score_display(self, risk_score: float, predicted_class: str) -> None:
        """Render main risk score display with progress bar"""
        alert_level = self.get_alert_level(risk_score)
        alert_text = self.get_alert_text(alert_level)
        color = self.alert_colors[alert_level]
        
        # Create a container with custom styling
        with st.container():
            # Header row
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 🔥 Risk Assessment")
                st.caption("AI Model Prediction")
            with col2:
                st.markdown(f"""
                <div style="text-align: right;">
                    <div style="font-size: 3rem; font-weight: 700; color: {color}; line-height: 1;">
                        {risk_score:.0f}
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">Risk Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Alert status row
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{alert_text}**")
            with col2:
                st.markdown(f"*Predicted: {predicted_class.title()}*")
            
            # Progress bar using Streamlit's native progress bar
            progress_value = min(risk_score / 100.0, 1.0)
            st.progress(progress_value)
            
            # Scale labels
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("0 - Safe")
            with col2:
                st.caption("50 - Moderate")
            with col3:
                st.caption("100 - Critical")
    
    def render_confidence_metrics(self, confidence: float, processing_time: float) -> None:
        """Render model confidence and performance metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence display using native Streamlit components
            st.markdown("#### 🎯 Model Confidence")
            confidence_percent = confidence * 100
            
            # Use metric for clean display
            if confidence > 0.8:
                st.success(f"**{confidence_percent:.0f}%** - High Confidence")
            elif confidence > 0.6:
                st.warning(f"**{confidence_percent:.0f}%** - Moderate Confidence")
            else:
                st.error(f"**{confidence_percent:.0f}%** - Low Confidence")
            
            # Progress bar for confidence
            st.progress(confidence)
        
        with col2:
            # Processing time display
            st.markdown("#### ⚡ Processing Time")
            
            # Use metric for clean display
            if processing_time < 50:
                st.success(f"**{processing_time:.0f}ms** - Fast")
            elif processing_time < 100:
                st.warning(f"**{processing_time:.0f}ms** - Moderate")
            else:
                st.error(f"**{processing_time:.0f}ms** - Slow")
    
    def render_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """Render feature importance for model transparency"""
        if not feature_importance:
            return
        
        st.markdown("#### Model Decision Factors")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Create horizontal bar chart
        features = [f[0].replace('_', ' ').title() for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        colors = ['#dc3545' if imp > 0 else '#28a745' for imp in importances]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color=colors,
            text=[f"{imp:+.2f}" for imp in importances],
            textposition='outside',
            showlegend=False
        ))
        
        fig.update_layout(
            height=max(200, len(features) * 40),
            margin=dict(l=120, r=60, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Feature Influence",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinecolor='rgba(128,128,128,0.5)'
            ),
            yaxis=dict(showgrid=False),
            font=dict(color="#333", size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Add explanation
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 0.5rem; padding: 1rem; margin-top: 1rem;">
            <small style="color: #666;">
                <strong>Feature Influence:</strong> Red bars indicate factors increasing fire risk, 
                green bars indicate factors decreasing risk. Longer bars have more influence on the prediction.
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    def render_risk_trend_chart(self) -> None:
        """Render risk score trend over time"""
        if not self.risk_history:
            st.markdown("*No risk score history available*")
            return
        
        # Prepare data
        timestamps = [entry['timestamp'] for entry in self.risk_history]
        risk_scores = [entry['risk_score'] for entry in self.risk_history]
        
        fig = go.Figure()
        
        # Add risk score line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=risk_scores,
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='#007bff', width=3),
            marker=dict(size=4),
            fill='tonexty',
            fillcolor='rgba(0, 123, 255, 0.1)'
        ))
        
        # Add threshold lines
        fig.add_hline(y=30, line_dash="dash", line_color="#28a745", 
                     annotation_text="Normal Threshold", annotation_position="right")
        fig.add_hline(y=50, line_dash="dash", line_color="#ffc107", 
                     annotation_text="Mild Threshold", annotation_position="right")
        fig.add_hline(y=85, line_dash="dash", line_color="#fd7e14", 
                     annotation_text="Elevated Threshold", annotation_position="right")
        
        # Add colored background regions
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(40, 167, 69, 0.1)", layer="below", line_width=0)
        fig.add_hrect(y0=30, y1=50, fillcolor="rgba(255, 193, 7, 0.1)", layer="below", line_width=0)
        fig.add_hrect(y0=50, y1=85, fillcolor="rgba(253, 126, 20, 0.1)", layer="below", line_width=0)
        fig.add_hrect(y0=85, y1=100, fillcolor="rgba(220, 53, 69, 0.1)", layer="below", line_width=0)
        
        fig.update_layout(
            title="Risk Score Trend",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Time",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="Risk Score",
                range=[0, 100],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            showlegend=False,
            font=dict(color="#333", size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_ensemble_voting(self, ensemble_votes: Dict[str, float]) -> None:
        """Render ensemble voting results for transparency"""
        if not ensemble_votes:
            return
        
        st.markdown("#### Ensemble Model Voting")
        
        # Create voting visualization
        models = list(ensemble_votes.keys())
        votes = list(ensemble_votes.values())
        
        # Create pie chart for voting
        fig = go.Figure(data=[go.Pie(
            labels=models,
            values=votes,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=['#007bff', '#28a745', '#ffc107', '#dc3545'][:len(models)])
        )])
        
        fig.update_layout(
            title="Model Agreement",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333", size=11),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_complete_analysis_panel(self, prediction: Optional[PredictionResult] = None) -> None:
        """Render the complete AI analysis panel"""
        if prediction is None:
            # Generate sample prediction for demonstration
            prediction = PredictionResult(
                risk_score=25.0,
                confidence=0.92,
                predicted_class='normal',
                feature_importance={
                    'temperature': -0.15,
                    'pm25': 0.08,
                    'co2': 0.05,
                    'audio_level': -0.12,
                    'temporal_pattern': -0.20
                },
                processing_time=23.5
            )
        
        # Add to history
        self.add_prediction(prediction)
        
        st.markdown("### AI Analysis & Risk Assessment")
        
        # Main risk score display
        self.render_risk_score_display(prediction.risk_score, prediction.predicted_class)
        
        # Confidence and performance metrics
        self.render_confidence_metrics(prediction.confidence, prediction.processing_time)
        
        # Feature importance for transparency
        self.render_feature_importance(prediction.feature_importance)
        
        # Risk trend chart
        st.markdown("#### Risk Score History")
        self.render_risk_trend_chart()