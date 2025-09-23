#!/bin/bash

# Fixed Fire Detection Dashboard Deployment Script for Elastic Beanstalk
# This script deploys the Streamlit dashboard to AWS Elastic Beanstalk with proper virtual environment handling

set -e  # Exit on any error

echo "🚀 Starting Fire Detection Dashboard Deployment to Elastic Beanstalk (Fixed Version)..."

# Check if AWS CLI and EB CLI are installed
if ! command -v aws &> /dev/null
then
    echo "❌ AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

if ! command -v eb &> /dev/null
then
    echo "❌ EB CLI is not installed. Please install EB CLI."
    echo "Installation instructions: https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html"
    exit 1
fi

# Variables
APP_NAME="fire-detection-dashboard"
ENV_NAME="fire-detection-dashboard-env"
REGION="us-east-1"

echo "📋 Deployment Configuration:"
echo "   Application: $APP_NAME"
echo "   Environment: $ENV_NAME"
echo "   Region: $REGION"

# Clean hidden files
echo "🧹 Cleaning hidden files..."
find . -name "._*" -type f -delete

# Create a clean deployment package
echo "📦 Creating clean deployment package..."
rm -f deploy-package.zip
zip -r deploy-package.zip . -x ".*" "__MACOSX*" "*.git*" "*.DS_Store" "deploy-package.zip" "venv*" "*.pyc" "*.pyo" "*__pycache__*" "temp*" "test*" "notebooks*" "docs*" "examples*" "scripts*" "src*" "tests*" "data*" "*.zip" "deploy_eb*.sh" "deploy_dashboard.sh" "deploy_lambda_agents.sh" "deploy_models*.sh" "deploy_s3_processor.sh" "run_*.sh" "setup_*.sh" "start_training.sh" "stop_training.sh" "validate_setup.sh" "check_*.py" "test_*.py" "demo_*.py" "verify_*.py" "monitor_*.py" "create_*.py" "debug_*.py" "simple_*.py" "advanced_*.py" "medium_*.py" "very_simple_*.py" "local_train_demo.py" "train_*.py" "sagemaker_*.py" "flir_scd41_*.py" "execute_*.py" "package_*.py" "evaluate_*.py" "inference_*.py" "integration_test.py" "main.py" "setup.py" "pytest.ini" "requirements-aws.txt" "requirements_unified_notebook.txt" "response.json" "response_v2.json" "test_output.txt" "phase1_benefit_analysis_results.json" "feature_info.json" "cloudwatch_dashboard.json" "tasks.md" "design.md" "requirements.md" "system_architecture.md" "system_overview.md" "aws-*.md" "implementation_plan.md" "implementation_summary.md" "ml_integration_flow.md" "mermaid_diagrams_explanation.md" "system_architecture_diagram.mmd" "ml_integration_flow_diagram.mmd" "cloud_deployment_diagram.mmd" "agent_model_integration_diagram.mmd" "model_performance_comparison.ipynb" "model_performance_comparison.png" "performance_improvements.png" "domain_specialization_comparison.png" "sourcedir.tar.gz" "corrected_code*.tar.gz" "ensemble_debug_code.tar.gz" "flir_scd41_*.tar.gz" "*.zip" "*.json" "*.png" "*.mmd" "*.ipynb" "temp_fix/*" "updated_packages/*" "extracted_source/*" "test_sensor_data/*" "test_sensor_data_csv/*" "test_training/*" "venv_clean/*" "config/*" "notebooks/*" "examples/*" "scripts/*" "src/*" "tests/*" "data/*" "docs/*" "temp/*" ".ebextensions/._*" ".elasticbeanstalk/*" ".streamlit/*" ".vscode/*" ".idea/*"

# Initialize Elastic Beanstalk application if it doesn't exist
if ! eb list | grep -q $APP_NAME; then
    echo "🏗️ Creating new Elastic Beanstalk application..."
    eb init -p "Python 3.9" $APP_NAME --region $REGION
else
    echo "✅ Application already exists"
fi

# Deploy using the clean package
echo "📤 Deploying application with clean package..."
eb deploy $ENV_NAME --region $REGION

echo "✅ Deployment completed successfully!"

echo "🌐 To access your dashboard:"
echo "   1. Wait 5-10 minutes for the environment to be created"
echo "   2. Get the URL:"
echo "      eb status"
echo "   3. Open the URL in your browser"

echo "🔄 To update the application after changes:"
echo "   eb deploy"

echo "🧹 To terminate the environment when no longer needed:"
echo "   eb terminate $ENV_NAME"

echo "🎉 Deployment script finished!"