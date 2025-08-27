#!/bin/bash
# AWS Setup Validation Script

echo "🔍 Validating AWS Setup"
echo "======================"

python3 aws_ensemble_trainer.py \
    --config config/base_config.yaml \
    --data-bucket fire-detection-training-691595239825 \
    --dry-run

echo "Validation completed!"