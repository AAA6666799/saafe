"""
Export and reporting services for session data.

This module provides functionality to export sensor data, predictions, and alerts
in various formats including PDF reports, CSV data exports, and dashboard screenshots.
"""

import os
import csv
import json
import base64
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.lineplots import LinePlot
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Screenshot functionality
try:
    import selenium
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Plotting for charts
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..core.data_models import SensorReading
from ..core.alert_engine import AlertData

# Handle optional torch dependency
try:
    from ..core.fire_detection_pipeline import PredictionResult
except ImportError:
    # Create mock PredictionResult for testing without torch
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    
    @dataclass
    class PredictionResult:
        risk_score: float
        confidence: float
        predicted_class: str
        feature_importance: Dict[str, float]
        processing_time: float
        ensemble_votes: Dict[str, float]
        anti_hallucination: Any
        timestamp: datetime
        model_metadata: Dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Container for session data to be exported."""
    session_id: str
    start_time: datetime
    end_time: datetime
    scenario_type: str
    sensor_readings: List[SensorReading]
    predictions: List[PredictionResult]
    alerts: List[AlertData]
    performance_metrics: Dict[str, Any]
    configuration: Dict[str, Any]


@dataclass
class ExportConfig:
    """Configuration for export operations."""
    output_directory: str = "exports"
    include_charts: bool = True
    include_raw_data: bool = True
    include_summary: bool = True
    chart_resolution: int = 300  # DPI
    max_data_points: int = 10000  # Limit for large datasets


class ExportService:
    """Main export service for session data and reports."""
    
    def __init__(self, config: ExportConfig = None):
        """
        Initialize the export service.
        
        Args:
            config (ExportConfig): Export configuration
        """
        self.config = config or ExportConfig()
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        # Check available dependencies
        self.pdf_available = REPORTLAB_AVAILABLE
        self.screenshot_available = SELENIUM_AVAILABLE
        self.charts_available = MATPLOTLIB_AVAILABLE
        
        if not self.pdf_available:
            self.logger.warning("ReportLab not available - PDF export disabled")
        if not self.screenshot_available:
            self.logger.warning("Selenium not available - screenshot capture disabled")
        if not self.charts_available:
            self.logger.warning("Matplotlib not available - chart generation disabled")
    
    def export_session_data(self, 
                          session_data: SessionData,
                          formats: List[str] = None) -> Dict[str, str]:
        """
        Export session data in specified formats.
        
        Args:
            session_data (SessionData): Session data to export
            formats (List[str]): Export formats ('pdf', 'csv', 'json', 'screenshot')
            
        Returns:
            Dict[str, str]: Mapping of format to output file path
        """
        if formats is None:
            formats = ['pdf', 'csv', 'json']
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"safeguard_session_{session_data.session_id}_{timestamp}"
        
        try:
            # Export CSV data
            if 'csv' in formats:
                csv_path = self._export_csv(session_data, base_filename)
                results['csv'] = csv_path
                self.logger.info(f"CSV export completed: {csv_path}")
            
            # Export JSON data
            if 'json' in formats:
                json_path = self._export_json(session_data, base_filename)
                results['json'] = json_path
                self.logger.info(f"JSON export completed: {json_path}")
            
            # Export PDF report
            if 'pdf' in formats and self.pdf_available:
                pdf_path = self._export_pdf_report(session_data, base_filename)
                results['pdf'] = pdf_path
                self.logger.info(f"PDF export completed: {pdf_path}")
            elif 'pdf' in formats:
                self.logger.warning("PDF export requested but ReportLab not available")
            
            # Capture screenshot
            if 'screenshot' in formats and self.screenshot_available:
                screenshot_path = self._capture_screenshot(base_filename)
                results['screenshot'] = screenshot_path
                self.logger.info(f"Screenshot captured: {screenshot_path}")
            elif 'screenshot' in formats:
                self.logger.warning("Screenshot requested but Selenium not available")
                
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise
        
        return results
    
    def _export_csv(self, session_data: SessionData, base_filename: str) -> str:
        """Export sensor data and predictions to CSV format."""
        csv_path = os.path.join(self.config.output_directory, f"{base_filename}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'timestamp', 'temperature', 'pm25', 'co2', 'audio_level',
                'risk_score', 'confidence', 'predicted_class', 'alert_level',
                'processing_time'
            ])
            
            # Combine sensor readings with predictions
            for i, reading in enumerate(session_data.sensor_readings):
                prediction = session_data.predictions[i] if i < len(session_data.predictions) else None
                alert = session_data.alerts[i] if i < len(session_data.alerts) else None
                
                row = [
                    reading.timestamp.isoformat(),
                    reading.temperature,
                    reading.pm25,
                    reading.co2,
                    reading.audio_level,
                    prediction.risk_score if prediction else '',
                    prediction.confidence if prediction else '',
                    prediction.predicted_class if prediction else '',
                    alert.alert_level.description if alert else '',
                    prediction.processing_time if prediction else ''
                ]
                writer.writerow(row)
        
        return csv_path
    
    def _export_json(self, session_data: SessionData, base_filename: str) -> str:
        """Export complete session data to JSON format."""
        json_path = os.path.join(self.config.output_directory, f"{base_filename}.json")
        
        # Convert session data to serializable format
        export_data = {
            'session_info': {
                'session_id': session_data.session_id,
                'start_time': session_data.start_time.isoformat(),
                'end_time': session_data.end_time.isoformat(),
                'scenario_type': session_data.scenario_type,
                'duration_minutes': (session_data.end_time - session_data.start_time).total_seconds() / 60
            },
            'sensor_readings': [reading.to_dict() for reading in session_data.sensor_readings],
            'predictions': [self._prediction_to_dict(pred) for pred in session_data.predictions],
            'alerts': [alert.to_dict() for alert in session_data.alerts],
            'performance_metrics': session_data.performance_metrics,
            'configuration': session_data.configuration,
            'export_metadata': {
                'export_time': datetime.now().isoformat(),
                'total_readings': len(session_data.sensor_readings),
                'total_predictions': len(session_data.predictions),
                'total_alerts': len(session_data.alerts)
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
        
        return json_path
    
    def _export_pdf_report(self, session_data: SessionData, base_filename: str) -> str:
        """Generate comprehensive PDF report with charts and statistics."""
        if not self.pdf_available:
            raise RuntimeError("ReportLab not available for PDF generation")
        
        pdf_path = os.path.join(self.config.output_directory, f"{base_filename}.pdf")
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Saafe Fire Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        # Session Information
        story.append(Paragraph("Session Information", styles['Heading2']))
        session_info = [
            ['Session ID:', session_data.session_id],
            ['Scenario Type:', session_data.scenario_type],
            ['Start Time:', session_data.start_time.strftime("%Y-%m-%d %H:%M:%S")],
            ['End Time:', session_data.end_time.strftime("%Y-%m-%d %H:%M:%S")],
            ['Duration:', f"{(session_data.end_time - session_data.start_time).total_seconds() / 60:.1f} minutes"],
            ['Total Readings:', str(len(session_data.sensor_readings))],
            ['Total Alerts:', str(len(session_data.alerts))]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 3*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(session_table)
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        stats = self._calculate_summary_stats(session_data)
        
        stats_data = [
            ['Metric', 'Value'],
            ['Average Risk Score', f"{stats['avg_risk_score']:.1f}"],
            ['Maximum Risk Score', f"{stats['max_risk_score']:.1f}"],
            ['Average Temperature', f"{stats['avg_temperature']:.1f}°C"],
            ['Average PM2.5', f"{stats['avg_pm25']:.1f} μg/m³"],
            ['Average CO₂', f"{stats['avg_co2']:.1f} ppm"],
            ['Critical Alerts', str(stats['critical_alerts'])],
            ['Average Processing Time', f"{stats['avg_processing_time']:.1f}ms"]
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Generate and include charts if available
        if self.charts_available and self.config.include_charts:
            chart_paths = self._generate_charts(session_data, base_filename)
            for chart_title, chart_path in chart_paths.items():
                story.append(Paragraph(chart_title, styles['Heading3']))
                try:
                    img = Image(chart_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                except Exception as e:
                    self.logger.warning(f"Failed to include chart {chart_title}: {e}")
        
        # Build PDF
        doc.build(story)
        
        return pdf_path
    
    def _generate_charts(self, session_data: SessionData, base_filename: str) -> Dict[str, str]:
        """Generate charts for the PDF report."""
        if not self.charts_available:
            return {}
        
        charts = {}
        
        # Extract data for plotting
        timestamps = [reading.timestamp for reading in session_data.sensor_readings]
        temperatures = [reading.temperature for reading in session_data.sensor_readings]
        pm25_values = [reading.pm25 for reading in session_data.sensor_readings]
        co2_values = [reading.co2 for reading in session_data.sensor_readings]
        risk_scores = [pred.risk_score for pred in session_data.predictions]
        
        # Temperature chart
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, temperatures, 'r-', linewidth=2, label='Temperature')
        plt.title('Temperature Over Time')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        temp_chart_path = os.path.join(self.config.output_directory, f"{base_filename}_temperature.png")
        plt.savefig(temp_chart_path, dpi=self.config.chart_resolution, bbox_inches='tight')
        plt.close()
        charts['Temperature Trend'] = temp_chart_path
        
        # Risk score chart
        if risk_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps[:len(risk_scores)], risk_scores, 'b-', linewidth=2, label='Risk Score')
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Normal Threshold')
            plt.axhline(y=85, color='r', linestyle='--', alpha=0.7, label='Critical Threshold')
            plt.title('Risk Score Over Time')
            plt.xlabel('Time')
            plt.ylabel('Risk Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            risk_chart_path = os.path.join(self.config.output_directory, f"{base_filename}_risk_score.png")
            plt.savefig(risk_chart_path, dpi=self.config.chart_resolution, bbox_inches='tight')
            plt.close()
            charts['Risk Score Trend'] = risk_chart_path
        
        return charts
    
    def _capture_screenshot(self, base_filename: str) -> str:
        """Capture screenshot of the dashboard interface."""
        if not self.screenshot_available:
            raise RuntimeError("Selenium not available for screenshot capture")
        
        screenshot_path = os.path.join(self.config.output_directory, f"{base_filename}_dashboard.png")
        
        # Configure Chrome options for headless operation
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            # Assume Streamlit is running on localhost:8501
            driver = webdriver.Chrome(options=chrome_options)
            driver.get("http://localhost:8501")
            
            # Wait for page to load
            time.sleep(3)
            
            # Take screenshot
            driver.save_screenshot(screenshot_path)
            driver.quit()
            
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            # Create a placeholder image instead
            if self.charts_available:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, 'Dashboard Screenshot\nNot Available', 
                        ha='center', va='center', fontsize=20)
                plt.axis('off')
                plt.savefig(screenshot_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        return screenshot_path
    
    def _calculate_summary_stats(self, session_data: SessionData) -> Dict[str, Any]:
        """Calculate summary statistics for the session."""
        stats = {}
        
        # Sensor statistics
        if session_data.sensor_readings:
            temperatures = [r.temperature for r in session_data.sensor_readings]
            pm25_values = [r.pm25 for r in session_data.sensor_readings]
            co2_values = [r.co2 for r in session_data.sensor_readings]
            
            stats['avg_temperature'] = sum(temperatures) / len(temperatures)
            stats['avg_pm25'] = sum(pm25_values) / len(pm25_values)
            stats['avg_co2'] = sum(co2_values) / len(co2_values)
        else:
            stats['avg_temperature'] = 0
            stats['avg_pm25'] = 0
            stats['avg_co2'] = 0
        
        # Prediction statistics
        if session_data.predictions:
            risk_scores = [p.risk_score for p in session_data.predictions]
            processing_times = [p.processing_time for p in session_data.predictions]
            
            stats['avg_risk_score'] = sum(risk_scores) / len(risk_scores)
            stats['max_risk_score'] = max(risk_scores)
            stats['avg_processing_time'] = sum(processing_times) / len(processing_times)
        else:
            stats['avg_risk_score'] = 0
            stats['max_risk_score'] = 0
            stats['avg_processing_time'] = 0
        
        # Alert statistics
        critical_alerts = sum(1 for alert in session_data.alerts 
                            if alert.alert_level.level >= 4)
        stats['critical_alerts'] = critical_alerts
        
        return stats
    
    def _prediction_to_dict(self, prediction: PredictionResult) -> Dict[str, Any]:
        """Convert PredictionResult to dictionary for JSON serialization."""
        return {
            'risk_score': prediction.risk_score,
            'confidence': prediction.confidence,
            'predicted_class': prediction.predicted_class,
            'feature_importance': prediction.feature_importance,
            'processing_time': prediction.processing_time,
            'ensemble_votes': prediction.ensemble_votes,
            'timestamp': prediction.timestamp.isoformat(),
            'anti_hallucination': {
                'is_valid': prediction.anti_hallucination.is_valid,
                'confidence_adjustment': prediction.anti_hallucination.confidence_adjustment,
                'reasoning': prediction.anti_hallucination.reasoning
            } if prediction.anti_hallucination else None
        }
    
    def generate_pdf_report(self, session_data: Dict[str, Any], output_path: str) -> tuple[bool, str]:
        """
        Generate PDF report from session data.
        
        Args:
            session_data: Dictionary containing session information
            output_path: Path where PDF should be saved
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.pdf_available:
                return False, "PDF generation not available - ReportLab not installed"
            
            # Convert dict to SessionData if needed
            if isinstance(session_data, dict):
                # Create a minimal SessionData-like object
                from types import SimpleNamespace
                session_obj = SimpleNamespace()
                session_obj.session_id = session_data.get('session_id', 'unknown')
                session_obj.start_time = session_data.get('start_time')
                session_obj.end_time = session_data.get('end_time')
                session_obj.scenario_type = session_data.get('scenario_type', 'unknown')
                session_obj.sensor_readings = session_data.get('sensor_readings', [])
                session_obj.predictions = session_data.get('predictions', [])
                session_obj.alerts = session_data.get('alerts', [])
                session_obj.performance_metrics = session_data.get('performance_metrics', {})
            else:
                session_obj = session_data
            
            # Use the existing private method
            base_filename = os.path.splitext(os.path.basename(output_path))[0]
            temp_output_dir = self.config.output_directory
            self.config.output_directory = os.path.dirname(output_path)
            
            try:
                result_path = self._export_pdf_report(session_obj, base_filename)
                # Move file to desired location if different
                if result_path != output_path:
                    import shutil
                    shutil.move(result_path, output_path)
                return True, f"PDF report generated successfully at {output_path}"
            finally:
                self.config.output_directory = temp_output_dir
                
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            return False, f"PDF generation failed: {str(e)}"
    
    def generate_session_summary(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate session summary statistics.
        
        Args:
            session_data: Dictionary containing session information
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Convert dict to SessionData if needed
            if isinstance(session_data, dict):
                from types import SimpleNamespace
                session_obj = SimpleNamespace()
                session_obj.session_id = session_data.get('session_id', 'unknown')
                session_obj.start_time = session_data.get('start_time')
                session_obj.end_time = session_data.get('end_time')
                session_obj.scenario_type = session_data.get('scenario_type', 'unknown')
                session_obj.sensor_readings = session_data.get('sensor_readings', [])
                session_obj.predictions = session_data.get('predictions', [])
                session_obj.alerts = session_data.get('alerts', [])
                session_obj.performance_metrics = session_data.get('performance_metrics', {})
            else:
                session_obj = session_data
            
            # Use existing private method
            summary = self._calculate_summary_stats(session_obj)
            
            # Add additional summary information
            summary.update({
                'session_id': session_obj.session_id,
                'scenario_type': session_obj.scenario_type,
                'duration_minutes': (session_obj.end_time - session_obj.start_time).total_seconds() / 60 if session_obj.start_time and session_obj.end_time else 0,
                'total_sensor_readings': len(session_obj.sensor_readings),
                'total_predictions': len(session_obj.predictions),
                'total_alerts': len(session_obj.alerts),
                'performance_metrics': session_obj.performance_metrics
            })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate session summary: {e}")
            return {
                'error': f"Summary generation failed: {str(e)}",
                'session_id': session_data.get('session_id', 'unknown') if isinstance(session_data, dict) else 'unknown'
            }
    
    def export_to_csv(self, session_data: Dict[str, Any], output_path: str) -> tuple[bool, str]:
        """
        Export session data to CSV format.
        
        Args:
            session_data: Dictionary containing session information
            output_path: Path where CSV should be saved
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Convert dict to SessionData if needed
            if isinstance(session_data, dict):
                from types import SimpleNamespace
                session_obj = SimpleNamespace()
                session_obj.session_id = session_data.get('session_id', 'unknown')
                session_obj.start_time = session_data.get('start_time')
                session_obj.end_time = session_data.get('end_time')
                session_obj.scenario_type = session_data.get('scenario_type', 'unknown')
                session_obj.sensor_readings = session_data.get('sensor_readings', [])
                session_obj.predictions = session_data.get('predictions', [])
                session_obj.alerts = session_data.get('alerts', [])
                session_obj.performance_metrics = session_data.get('performance_metrics', {})
            else:
                session_obj = session_data
            
            # Use the existing private method
            base_filename = os.path.splitext(os.path.basename(output_path))[0]
            temp_output_dir = self.config.output_directory
            self.config.output_directory = os.path.dirname(output_path)
            
            try:
                result_path = self._export_csv(session_obj, base_filename)
                # Move file to desired location if different
                if result_path != output_path:
                    import shutil
                    shutil.move(result_path, output_path)
                return True, f"CSV export successful: {output_path}"
            finally:
                self.config.output_directory = temp_output_dir
                
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return False, f"CSV export failed: {str(e)}"


class BatchExportManager:
    """Manages batch export operations and scheduling."""
    
    def __init__(self, export_service: ExportService):
        """
        Initialize batch export manager.
        
        Args:
            export_service (ExportService): Export service instance
        """
        self.export_service = export_service
        self.scheduled_exports = []
        self.logger = logging.getLogger(__name__)
    
    def schedule_export(self, 
                       session_data: SessionData,
                       export_time: datetime,
                       formats: List[str] = None) -> str:
        """
        Schedule an export operation.
        
        Args:
            session_data (SessionData): Data to export
            export_time (datetime): When to perform the export
            formats (List[str]): Export formats
            
        Returns:
            str: Export job ID
        """
        job_id = f"export_{int(datetime.now().timestamp())}"
        
        export_job = {
            'job_id': job_id,
            'session_data': session_data,
            'export_time': export_time,
            'formats': formats or ['pdf', 'csv'],
            'status': 'scheduled'
        }
        
        self.scheduled_exports.append(export_job)
        self.logger.info(f"Export scheduled: {job_id} for {export_time}")
        
        return job_id
    
    def process_scheduled_exports(self) -> List[Dict[str, Any]]:
        """
        Process any scheduled exports that are due.
        
        Returns:
            List[Dict[str, Any]]: Results of processed exports
        """
        current_time = datetime.now()
        processed = []
        
        for export_job in self.scheduled_exports[:]:
            if (export_job['export_time'] <= current_time and 
                export_job['status'] == 'scheduled'):
                
                try:
                    export_job['status'] = 'processing'
                    results = self.export_service.export_session_data(
                        export_job['session_data'],
                        export_job['formats']
                    )
                    
                    export_job['status'] = 'completed'
                    export_job['results'] = results
                    export_job['completed_time'] = current_time
                    
                    processed.append(export_job)
                    self.logger.info(f"Export completed: {export_job['job_id']}")
                    
                except Exception as e:
                    export_job['status'] = 'failed'
                    export_job['error'] = str(e)
                    self.logger.error(f"Export failed: {export_job['job_id']} - {e}")
        
        # Remove completed/failed jobs older than 24 hours
        cutoff_time = current_time - timedelta(hours=24)
        self.scheduled_exports = [
            job for job in self.scheduled_exports
            if (job['status'] == 'scheduled' or 
                job.get('completed_time', current_time) > cutoff_time)
        ]
        
        return processed
    
    def get_export_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an export job.
        
        Args:
            job_id (str): Export job ID
            
        Returns:
            Optional[Dict[str, Any]]: Job status information
        """
        for job in self.scheduled_exports:
            if job['job_id'] == job_id:
                return {
                    'job_id': job['job_id'],
                    'status': job['status'],
                    'export_time': job['export_time'],
                    'formats': job['formats'],
                    'results': job.get('results'),
                    'error': job.get('error'),
                    'completed_time': job.get('completed_time')
                }
        return None