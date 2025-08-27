#!/usr/bin/env python3
"""
Script to set up the necessary directories for the synthetic fire prediction system.

This script creates the required directory structure for data, models, and logs.
"""

import os
import argparse
import logging


def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('setup_directories')


def create_directory(path, logger):
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        logger: Logger instance
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {str(e)}")
    else:
        logger.info(f"Directory already exists: {path}")


def setup_directories(base_dir, env, logger):
    """
    Set up the directory structure.
    
    Args:
        base_dir: Base directory for the project
        env: Environment (dev, test, prod)
        logger: Logger instance
    """
    # Create data directories
    data_dir = os.path.join(base_dir, 'data', env)
    create_directory(data_dir, logger)
    
    # Create subdirectories for different data types
    create_directory(os.path.join(data_dir, 'thermal'), logger)
    create_directory(os.path.join(data_dir, 'gas'), logger)
    create_directory(os.path.join(data_dir, 'environmental'), logger)
    create_directory(os.path.join(data_dir, 'scenarios'), logger)
    
    # Create model directories
    models_dir = os.path.join(base_dir, 'models', env)
    create_directory(models_dir, logger)
    
    # Create subdirectories for different model types
    create_directory(os.path.join(models_dir, 'baseline'), logger)
    create_directory(os.path.join(models_dir, 'temporal'), logger)
    create_directory(os.path.join(models_dir, 'ensemble'), logger)
    
    # Create logs directory
    logs_dir = os.path.join(base_dir, 'logs', env)
    create_directory(logs_dir, logger)
    
    # Create config directories
    config_dir = os.path.join(base_dir, 'config')
    create_directory(config_dir, logger)
    create_directory(os.path.join(config_dir, 'environments'), logger)
    create_directory(os.path.join(config_dir, 'secrets'), logger)
    
    logger.info(f"Directory setup complete for environment: {env}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Set up directories for the synthetic fire prediction system')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory for the project')
    parser.add_argument('--env', type=str, default='dev', choices=['dev', 'test', 'prod'], help='Environment')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Setting up directories for environment: {args.env}")
    setup_directories(args.base_dir, args.env, logger)


if __name__ == '__main__':
    main()