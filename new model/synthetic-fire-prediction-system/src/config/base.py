"""
Configuration management system.

This module provides a unified interface for managing configuration settings,
including AWS resources, environment-specific settings, and secrets.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class SecretManager:
    """
    Class for managing secrets.
    
    This class provides methods for securely accessing and storing secrets,
    with support for AWS Secrets Manager and local secrets files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the secret manager.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.secrets_cache = {}
        
        # Initialize AWS Secrets Manager client if configured
        self.secrets_manager_client = None
        if self.config.get('use_aws_secrets_manager', False):
            session_kwargs = {}
            if 'aws_region' in self.config:
                session_kwargs['region_name'] = self.config['aws_region']
            
            if 'aws_profile' in self.config:
                session_kwargs['profile_name'] = self.config['aws_profile']
                
            session = boto3.Session(**session_kwargs)
            self.secrets_manager_client = session.client('secretsmanager')
    
    def get_secret(self, secret_name: str, use_cache: bool = True) -> str:
        """
        Get a secret value.
        
        Args:
            secret_name: Name of the secret
            use_cache: Whether to use cached values
            
        Returns:
            Secret value
            
        Raises:
            ConfigurationError: If the secret cannot be retrieved
        """
        # Check cache first if enabled
        if use_cache and secret_name in self.secrets_cache:
            return self.secrets_cache[secret_name]
        
        # Try environment variables first
        env_var_name = secret_name.upper().replace('-', '_')
        if env_var_name in os.environ:
            secret_value = os.environ[env_var_name]
            if use_cache:
                self.secrets_cache[secret_name] = secret_value
            return secret_value
        
        # Try AWS Secrets Manager if configured
        if self.secrets_manager_client:
            try:
                response = self.secrets_manager_client.get_secret_value(SecretId=secret_name)
                if 'SecretString' in response:
                    secret_value = response['SecretString']
                    if use_cache:
                        self.secrets_cache[secret_name] = secret_value
                    return secret_value
            except ClientError as e:
                logging.warning(f"Failed to retrieve secret {secret_name} from AWS Secrets Manager: {str(e)}")
        
        # Try local secrets file
        secrets_file = self.config.get('secrets_file', 'secrets.json')
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
                    if secret_name in secrets:
                        secret_value = secrets[secret_name]
                        if use_cache:
                            self.secrets_cache[secret_name] = secret_value
                        return secret_value
            except Exception as e:
                logging.warning(f"Failed to read secret {secret_name} from local file: {str(e)}")
        
        raise ConfigurationError(f"Secret {secret_name} not found")
    
    def set_secret(self, secret_name: str, secret_value: str, update_aws: bool = False) -> None:
        """
        Set a secret value.
        
        Args:
            secret_name: Name of the secret
            secret_value: Value of the secret
            update_aws: Whether to update the secret in AWS Secrets Manager
            
        Raises:
            ConfigurationError: If the secret cannot be set
        """
        # Update cache
        self.secrets_cache[secret_name] = secret_value
        
        # Update AWS Secrets Manager if configured and requested
        if update_aws and self.secrets_manager_client:
            try:
                self.secrets_manager_client.put_secret_value(
                    SecretId=secret_name,
                    SecretString=secret_value
                )
            except ClientError as e:
                logging.error(f"Failed to update secret {secret_name} in AWS Secrets Manager: {str(e)}")
                raise ConfigurationError(f"Failed to update secret {secret_name} in AWS Secrets Manager")
        
        # Update local secrets file
        secrets_file = self.config.get('secrets_file', 'secrets.json')
        secrets = {}
        
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read secrets file: {str(e)}")
        
        secrets[secret_name] = secret_value
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(secrets_file)), exist_ok=True)
            with open(secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to write secret {secret_name} to local file: {str(e)}")
            raise ConfigurationError(f"Failed to write secret {secret_name} to local file")


class ConfigurationManager:
    """
    Class for managing configuration settings.
    
    This class provides a unified interface for accessing configuration settings
    from various sources, including files, environment variables, and AWS services.
    """
    
    def __init__(self, 
                base_config_path: str = 'config',
                environment: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            base_config_path: Base path for configuration files
            environment: Optional environment name (dev, test, prod)
        """
        self.base_config_path = base_config_path
        self.environment = environment or os.environ.get('APP_ENV', 'dev')
        self.config_cache = {}
        
        # Load base configuration
        self.base_config = self._load_config_file('base_config.yaml')
        
        # Load environment-specific configuration
        env_config = self._load_config_file(f'{self.environment}_config.yaml')
        
        # Merge configurations
        self.config = self._merge_configs(self.base_config, env_config)
        
        # Initialize secret manager
        self.secret_manager = SecretManager(self.config.get('secrets', {}))
    
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            filename: Name of the configuration file
            
        Returns:
            Dictionary containing configuration settings
        """
        filepath = os.path.join(self.base_config_path, filename)
        if not os.path.exists(filepath):
            logging.warning(f"Configuration file {filepath} not found")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                elif filename.endswith('.json'):
                    return json.load(f)
                else:
                    logging.warning(f"Unsupported configuration file format: {filename}")
                    return {}
        except Exception as e:
            logging.error(f"Failed to load configuration file {filepath}: {str(e)}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        if key in self.config_cache:
            return self.config_cache[key]
        
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            
            self.config_cache[key] = value
            return value
        except (KeyError, TypeError):
            return default
    
    def get_secret(self, secret_name: str, use_cache: bool = True) -> str:
        """
        Get a secret value.
        
        Args:
            secret_name: Name of the secret
            use_cache: Whether to use cached values
            
        Returns:
            Secret value
            
        Raises:
            ConfigurationError: If the secret cannot be retrieved
        """
        return self.secret_manager.get_secret(secret_name, use_cache)
    
    def set_secret(self, secret_name: str, secret_value: str, update_aws: bool = False) -> None:
        """
        Set a secret value.
        
        Args:
            secret_name: Name of the secret
            secret_value: Value of the secret
            update_aws: Whether to update the secret in AWS Secrets Manager
            
        Raises:
            ConfigurationError: If the secret cannot be set
        """
        self.secret_manager.set_secret(secret_name, secret_value, update_aws)
    
    def get_aws_config(self, service_name: str) -> Dict[str, Any]:
        """
        Get AWS service configuration.
        
        Args:
            service_name: Name of the AWS service
            
        Returns:
            Dictionary containing AWS service configuration
        """
        aws_config = self.get('aws', {})
        service_config = aws_config.get(service_name, {})
        
        # Add common AWS configuration
        if 'region' not in service_config and 'region' in aws_config:
            service_config['region_name'] = aws_config['region']
        
        if 'profile' not in service_config and 'profile' in aws_config:
            service_config['profile_name'] = aws_config['profile']
        
        return service_config
    
    def reload(self) -> None:
        """
        Reload configuration from files.
        """
        # Clear caches
        self.config_cache = {}
        
        # Reload base configuration
        self.base_config = self._load_config_file('base_config.yaml')
        
        # Reload environment-specific configuration
        env_config = self._load_config_file(f'{self.environment}_config.yaml')
        
        # Merge configurations
        self.config = self._merge_configs(self.base_config, env_config)
    
    def save(self, filename: str = None) -> None:
        """
        Save current configuration to a file.
        
        Args:
            filename: Optional filename to save to
        """
        if filename is None:
            filename = f'{self.environment}_config.yaml'
        
        filepath = os.path.join(self.base_config_path, filename)
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif filepath.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {filename}")
        except Exception as e:
            logging.error(f"Failed to save configuration to {filepath}: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration to {filepath}")


# Create a singleton instance
config_manager = None


def initialize_config(base_config_path: str = 'config', environment: Optional[str] = None) -> ConfigurationManager:
    """
    Initialize the configuration manager.
    
    Args:
        base_config_path: Base path for configuration files
        environment: Optional environment name (dev, test, prod)
        
    Returns:
        ConfigurationManager instance
    """
    global config_manager
    config_manager = ConfigurationManager(base_config_path, environment)
    return config_manager


def get_config() -> ConfigurationManager:
    """
    Get the configuration manager instance.
    
    Returns:
        ConfigurationManager instance
        
    Raises:
        ConfigurationError: If the configuration manager has not been initialized
    """
    global config_manager
    if config_manager is None:
        raise ConfigurationError("Configuration manager has not been initialized")
    return config_manager