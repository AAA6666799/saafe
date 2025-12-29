"""
Main entry point for the Synthetic Fire Prediction System
"""

import argparse
import sys
import time
from pathlib import Path

from ai_fire_prediction_platform.core.config import config_manager
from ai_fire_prediction_platform.core.utils import setup_logging
from ai_fire_prediction_platform.system.manager import SystemManager


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Synthetic Fire Prediction System"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parser.add_argument(
        "--run-time",
        type=int,
        default=60,
        help="Run time in seconds (0 for infinite)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting Synthetic Fire Prediction System")
    
    try:
        # Initialize configuration manager
        global_config_manager = config_manager
        if args.config_dir != "config":
            from ai_fire_prediction_platform.core.config import ConfigurationManager
            global_config_manager = ConfigurationManager(args.config_dir)
        
        # Validate configuration
        config_errors = global_config_manager.validate_configuration()
        if config_errors:
            logger.error("Configuration validation failed:")
            for section, errors in config_errors.items():
                for error in errors:
                    logger.error(f"  {section}: {error}")
            sys.exit(1)
        
        logger.info("Configuration validated successfully")
        logger.info(f"System environment: {global_config_manager.system_config.environment}")
        
        # Initialize and start system manager
        system_manager = SystemManager(global_config_manager)
        
        if system_manager.start():
            logger.info("System manager started successfully")
            
            # Run for specified time
            if args.run_time > 0:
                logger.info(f"Running system for {args.run_time} seconds...")
                time.sleep(args.run_time)
                system_manager.stop()
            else:
                logger.info("Running system indefinitely. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal. Stopping system...")
                    system_manager.stop()
            
            logger.info("System operation completed")
        else:
            logger.error("Failed to start system manager")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()