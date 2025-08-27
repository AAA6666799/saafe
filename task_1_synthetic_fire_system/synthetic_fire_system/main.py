"""
Main entry point for the Synthetic Fire Prediction System
"""

import argparse
import sys
from pathlib import Path

from synthetic_fire_system.core.config import config_manager
from synthetic_fire_system.core.utils import setup_logging


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
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting Synthetic Fire Prediction System")
    
    try:
        # Initialize configuration manager
        global_config_manager = config_manager
        if args.config_dir != "config":
            from synthetic_fire_system.core.config import ConfigurationManager
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
        
        # TODO: Initialize and start system manager
        # This will be implemented in future tasks
        logger.info("System initialization complete")
        logger.info("System is ready for operation")
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()