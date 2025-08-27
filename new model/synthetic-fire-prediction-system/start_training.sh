#!/bin/bash
# AWS Fire Detection Training Start Script

echo "🔥 Starting AWS Fire Detection Training"
echo "======================================"

python3 aws_ensemble_trainer.py \
    --config config/base_config.yaml \
    --data-bucket fire-detection-training-691595239825

echo "Training completed!"