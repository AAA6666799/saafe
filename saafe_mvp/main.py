"""
Main application entry point for Saafe MVP.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from saafe_mvp.core.data_stream import get_data_stream_manager
from saafe_mvp.models.model_manager import ModelManager
from saafe_mvp.core.fire_detection_pipeline import FireDetectionPipeline
from saafe_mvp.core.alert_engine import AlertEngine
from saafe_mvp.services.notification_manager import NotificationManager
from saafe_mvp.ui.dashboard import SafeguardDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    logger.info("Starting Saafe MVP...")
    
    try:
        # Initialize core components
        model_manager = ModelManager()
        pipeline = FireDetectionPipeline(model_manager)
        alert_engine = AlertEngine()
        
        # Start dashboard
        dashboard = SafeguardDashboard()
        dashboard.render_main_dashboard()
        
        logger.info("Saafe MVP started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Saafe MVP: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()