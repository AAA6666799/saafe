#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - Simple Model Evaluation
This script displays the final metrics from the training job.
"""

import boto3
import json
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'

def get_training_metrics():
    """Get the final metrics from the training job."""
    print("Getting training metrics...")
    
    try:
        # Initialize SageMaker client
        sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
        
        # Get training job details
        job_name = 'flir-scd41-xgboost-simple-20250828-154649'
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        
        # Extract final metrics
        final_metrics = response.get('FinalMetricDataList', [])
        
        # Print results
        print("\n" + "=" * 50)
        print("MODEL TRAINING METRICS")
        print("=" * 50)
        
        # Display key metrics
        auc = None
        accuracy = None
        f1 = None
        
        for metric in final_metrics:
            metric_name = metric['MetricName']
            metric_value = metric['Value']
            
            if metric_name == 'train:auc':
                auc = metric_value
                print(f"AUC (Area Under Curve):     {metric_value:.4f}")
            elif metric_name == 'train:accuracy':
                accuracy = metric_value
                print(f"Accuracy:                   {metric_value:.4f}")
            elif metric_name == 'train:f1':
                f1 = metric_value
                print(f"F1-Score:                   {metric_value:.4f}")
        
        # Display additional information
        print("-" * 50)
        print(f"Training time:              {response.get('TrainingTimeInSeconds', 'N/A')} seconds")
        print(f"Billable time:              {response.get('BillableTimeInSeconds', 'N/A')} seconds")
        print(f"Instance type:              {response.get('ResourceConfig', {}).get('InstanceType', 'N/A')}")
        print("=" * 50)
        
        # Save metrics to file
        metrics_data = {
            "job_name": job_name,
            "auc": auc,
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time_seconds": response.get('TrainingTimeInSeconds'),
            "billable_time_seconds": response.get('BillableTimeInSeconds'),
            "instance_type": response.get('ResourceConfig', {}).get('InstanceType'),
            "all_metrics": []  # Simplified to avoid datetime serialization issues
        }
        
        # Add simplified metrics without datetime objects
        for metric in final_metrics:
            metrics_data["all_metrics"].append({
                "MetricName": metric['MetricName'],
                "Value": metric['Value']
            })
        
        with open('/tmp/model_training_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print("Metrics saved to /tmp/model_training_metrics.json")
        
        return metrics_data
        
    except Exception as e:
        print(f"Error getting training metrics: {e}")
        return None

def interpret_metrics(metrics):
    """Interpret the model metrics."""
    if not metrics:
        return
    
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE INTERPRETATION")
    print("=" * 50)
    
    auc = metrics.get('auc')
    accuracy = metrics.get('accuracy')
    f1 = metrics.get('f1_score')
    
    # Interpret AUC if available
    if auc is not None:
        if auc >= 0.9:
            auc_interpretation = "Excellent"
        elif auc >= 0.8:
            auc_interpretation = "Good"
        elif auc >= 0.7:
            auc_interpretation = "Fair"
        elif auc >= 0.6:
            auc_interpretation = "Poor"
        else:
            auc_interpretation = "Fail"
        print(f"AUC ({auc:.4f}):           {auc_interpretation}")
    
    # Interpret Accuracy if available
    if accuracy is not None:
        if accuracy >= 0.9:
            accuracy_interpretation = "Excellent"
        elif accuracy >= 0.8:
            accuracy_interpretation = "Good"
        elif accuracy >= 0.7:
            accuracy_interpretation = "Fair"
        elif accuracy >= 0.6:
            accuracy_interpretation = "Poor"
        else:
            accuracy_interpretation = "Fail"
        print(f"Accuracy ({accuracy:.4f}):     {accuracy_interpretation}")
    
    # Interpret F1-Score if available
    if f1 is not None:
        if f1 >= 0.9:
            f1_interpretation = "Excellent"
        elif f1 >= 0.8:
            f1_interpretation = "Good"
        elif f1 >= 0.7:
            f1_interpretation = "Fair"
        elif f1 >= 0.6:
            f1_interpretation = "Poor"
        else:
            f1_interpretation = "Fail"
        print(f"F1-Score ({f1:.4f}):       {f1_interpretation}")
    
    print("\nOverall Assessment:")
    
    # If we only have AUC, base our assessment on that
    if auc is not None and accuracy is None and f1 is None:
        if auc >= 0.8:
            print("✅ Model performance is GOOD based on AUC metric - suitable for deployment")
        elif auc >= 0.7:
            print("⚠️ Model performance is FAIR based on AUC metric - may be suitable for deployment with monitoring")
        else:
            print("❌ Model performance is POOR based on AUC metric - not recommended for deployment")
    else:
        # Calculate average of available metrics
        available_metrics = [m for m in [auc, accuracy, f1] if m is not None]
        if available_metrics:
            avg_score = sum(available_metrics) / len(available_metrics)
            if avg_score >= 0.8:
                print("✅ Model performance is GOOD - suitable for deployment")
            elif avg_score >= 0.7:
                print("⚠️ Model performance is FAIR - may be suitable for deployment with monitoring")
            else:
                print("❌ Model performance is POOR - not recommended for deployment")
        else:
            print("⚠️ No metrics available for assessment")
    
    print("=" * 50)

def main():
    """Main function to evaluate the trained model."""
    print("FLIR+SCD41 Fire Detection - Simple Model Evaluation")
    print("=" * 50)
    
    # Get training metrics
    metrics = get_training_metrics()
    
    if metrics:
        # Interpret metrics
        interpret_metrics(metrics)
        
        print("\n✅ Model evaluation completed!")
        print("\nBased on these metrics, you can decide whether to proceed with deployment.")
    else:
        print("\n❌ Model evaluation failed.")

if __name__ == "__main__":
    main()