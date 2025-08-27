# Configuration Management

This directory contains configuration files for the Synthetic Fire Prediction System.

## Directory Structure

- `environments/`: Environment-specific configuration files
  - `base_config.yaml`: Base configuration with default values
  - `dev_config.yaml`: Development environment configuration
  - `test_config.yaml`: Testing environment configuration
  - `prod_config.yaml`: Production environment configuration

- `secrets/`: Secret configuration files (not committed to version control)
  - `secrets_template.json`: Template for secrets configuration
  - `dev_secrets.json`: Development environment secrets (create this file locally)
  - `test_secrets.json`: Testing environment secrets (create this file locally)
  - `prod_secrets.json`: Production environment secrets (create this file locally)

## Usage

The configuration system loads the base configuration first, then overlays the environment-specific configuration on top. This allows for environment-specific overrides while maintaining a common base configuration.

### Environment Selection

The environment is selected using the following methods (in order of precedence):

1. Command-line argument: `--env`
2. Environment variable: `APP_ENV`
3. Default: `dev`

### Secrets Management

Secrets are managed separately from regular configuration to ensure they are not accidentally committed to version control. The system looks for secrets in the following locations (in order of precedence):

1. Environment variables
2. AWS Secrets Manager (if configured)
3. Local secrets file

To set up your local secrets:

1. Copy `secrets_template.json` to `dev_secrets.json` (or the appropriate environment)
2. Fill in your secret values
3. Ensure the file is not committed to version control (it's included in `.gitignore`)

## Configuration Schema

See `base_config.yaml` for the complete configuration schema with comments explaining each option.