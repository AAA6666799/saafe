#!/usr/bin/env python3
"""
FLIR+SCD41 Fire Detection System - SageMaker Training Script for 100K Samples Ensemble
This script is compatible with SageMaker's training environment and trains an ensemble of models.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import sys
import traceback

def load_json_data(data_dir):
    """Load JSON data from directory, handling multiple files."""
    print(f"🔍 Looking for JSON files in {data_dir}")
    try:
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        data_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        print(f"📁 Found {len(data_files)} JSON files: {data_files}")
        
        if not data_files:
            raise ValueError(f"No JSON files found in {data_dir}")
        
        # Load data from all files
        all_samples = []
        for data_file in data_files:
            full_path = os.path.join(data_dir, data_file)
            print(f"📂 Loading data from {full_path}")
            
            try:
                with open(full_path, "r") as f:
                    data = json.load(f)
                
                samples = data.get('samples', [])
                all_samples.extend(samples)
                print(f"  Loaded {len(samples)} samples from {data_file}")
            except Exception as e:
                print(f"  Warning: Error loading {data_file}: {e}, skipping")
                continue
        
        print(f"✅ Successfully loaded total of {len(all_samples)} samples from {len(data_files)} files")
        
        # Return data in the expected format
        return {'samples': all_samples}
    except Exception as e:
        print(f"❌ Error in load_json_data: {e}")
        raise

def convert_to_csv(data, output_path):
    """Convert JSON data to CSV format."""
    print("🔄 Converting JSON data to CSV")
    try:
        # Extract samples
        samples = data["samples"]
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            **sample["features"],
            "label": sample["label"]
        } for sample in samples])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as CSV
        df.to_csv(output_path, index=False)
        print(f"💾 Data converted to CSV and saved to {output_path}")
        print(f"📊 DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error in convert_to_csv: {e}")
        raise

def calculate_ensemble_weights(model_scores):
    """Calculate ensemble weights based on validation performance."""
    print("⚖️  Calculating ensemble weights")
    try:
        # Normalize scores to [0, 1] range
        min_score = min(model_scores)
        max_score = max(model_scores)
        
        if max_score == min_score:
            # All models have same performance, equal weights
            return [1.0/len(model_scores)] * len(model_scores)
        
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in model_scores]
        
        # Apply exponential scaling
        scaling_factor = 3.0
        weighted_scores = [np.exp(scaling_factor * score) for score in normalized_scores]
        
        # Normalize to sum to 1
        total_weight = sum(weighted_scores)
        weights = [w / total_weight for w in weighted_scores]
        
        return weights
    except Exception as e:
        print(f"❌ Error in calculate_ensemble_weights: {e}")
        raise

def evaluate_ensemble(X_test, y_test, models, weights):
    """Evaluate ensemble model on test set."""
    print("🧪 Evaluating ensemble model")
    try:
        # Get predictions from each model
        probabilities = {}
        
        for i, model in enumerate(models):
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.decision_function(X_test)
                # Normalize to [0, 1] range
                probs = (probs - probs.min()) / (probs.max() - probs.min())
            
            probabilities[i] = probs
        
        # Calculate weighted ensemble predictions
        ensemble_probs = np.zeros(len(X_test))
        for i, weight in enumerate(weights):
            if i in probabilities:
                ensemble_probs += weight * probabilities[i]
        
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        # Calculate ensemble metrics
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, ensemble_preds),
            'f1_score': f1_score(y_test, ensemble_preds),
            'precision': precision_score(y_test, ensemble_preds),
            'recall': recall_score(y_test, ensemble_preds),
            'auc': roc_auc_score(y_test, ensemble_probs) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        return ensemble_preds, ensemble_probs, ensemble_metrics
    except Exception as e:
        print(f"❌ Error in evaluate_ensemble: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-data-dir", type=str, default="/opt/ml/output/data")
    args = parser.parse_args()
    
    print("🚀 Starting FLIR+SCD41 100K ensemble training...")
    print(f"📂 Data path: {args.data_path}")
    print(f"📂 Model dir: {args.model_dir}")
    print(f"📂 Output data dir: {args.output_data_dir}")
    print("=" * 60)
    
    try:
        # Check if data directory exists
        if not os.path.exists(args.data_path):
            print(f"❌ ERROR: Data directory {args.data_path} does not exist")
            sys.exit(1)
        
        # List contents of data directory
        print(f"📁 Contents of data directory: {os.listdir(args.data_path)}")
        
        # Load data
        print(f"📥 Loading data from {args.data_path}")
        data = load_json_data(args.data_path)
        print(f"📊 Loaded {len(data['samples'])} samples")
        
        # Convert to CSV for algorithm mode compatibility
        csv_path = os.path.join(args.output_data_dir, "train_data.csv")
        print(f"🔄 Converting data to CSV format")
        df = convert_to_csv(data, csv_path)
        
        # Prepare features and target
        feature_names = [
            't_mean', 't_std', 't_max', 't_p95', 't_hot_area_pct',
            't_hot_largest_blob_pct', 't_grad_mean', 't_grad_std',
            't_diff_mean', 't_diff_std', 'flow_mag_mean', 'flow_mag_std',
            'tproxy_val', 'tproxy_delta', 'tproxy_vel',
            'gas_val', 'gas_delta', 'gas_vel'
        ]
        
        print(f"⚙️  Preparing features and target")
        X = df[feature_names]
        y = df["label"]
        
        print(f"📈 Feature matrix shape: {X.shape}")
        print(f"🎯 Target vector shape: {y.shape}")
        print(f"🔥 Fire samples: {sum(y):,} ({sum(y)/len(y)*100:.2f}%)")
        
        # Split data
        print("✂️  Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15/(1-0.15), random_state=42, stratify=y_train)
        
        print(f"  Train set: {X_train.shape[0]:,} samples")
        print(f"  Validation set: {X_val.shape[0]:,} samples")
        print(f"  Test set: {X_test.shape[0]:,} samples")
        
        # Train multiple models
        models = []
        model_names = []
        val_scores = []
        
        # Train Random Forest
        print("🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_val_pred = rf_model.predict(X_val)
        rf_val_score = accuracy_score(y_val, rf_val_pred)
        models.append(rf_model)
        model_names.append("Random Forest")
        val_scores.append(rf_val_score)
        print(f"  Random Forest validation accuracy: {rf_val_score:.4f}")
        
        # Train Gradient Boosting
        print("📊 Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_val_pred = gb_model.predict(X_val)
        gb_val_score = accuracy_score(y_val, gb_val_pred)
        models.append(gb_model)
        model_names.append("Gradient Boosting")
        val_scores.append(gb_val_score)
        print(f"  Gradient Boosting validation accuracy: {gb_val_score:.4f}")
        
        # Train Logistic Regression
        print("📊 Training Logistic Regression...")
        lr_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_val_pred = lr_model.predict(X_val)
        lr_val_score = accuracy_score(y_val, lr_val_pred)
        models.append(lr_model)
        model_names.append("Logistic Regression")
        val_scores.append(lr_val_score)
        print(f"  Logistic Regression validation accuracy: {lr_val_score:.4f}")
        
        # Calculate ensemble weights
        print("⚖️  Calculating ensemble weights...")
        ensemble_weights = calculate_ensemble_weights(val_scores)
        for name, weight in zip(model_names, ensemble_weights):
            print(f"  {name} weight: {weight:.4f}")
        
        # Evaluate ensemble on test set
        print("🧪 Evaluating ensemble on test set...")
        ensemble_preds, ensemble_probs, ensemble_metrics = evaluate_ensemble(X_test, y_test, models, ensemble_weights)
        
        # Evaluate individual models
        print("📋 Evaluating individual models...")
        model_metrics = {}
        for i, (model, name) in enumerate(zip(models, model_names)):
            pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_test)[:, 1]
            else:
                prob = model.decision_function(X_test)
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            
            model_metrics[name] = {
                'accuracy': accuracy_score(y_test, pred),
                'f1_score': f1_score(y_test, pred),
                'precision': precision_score(y_test, pred),
                'recall': recall_score(y_test, pred),
                'auc': roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.0
            }
        
        # Add ensemble metrics
        model_metrics["Ensemble"] = ensemble_metrics
        
        # Save models
        print("💾 Saving models...")
        os.makedirs(args.model_dir, exist_ok=True)
        for i, (model, name) in enumerate(zip(models, model_names)):
            model_path = os.path.join(args.model_dir, f"{name.lower().replace(' ', '_')}_model.joblib")
            joblib.dump(model, model_path)
            print(f"  {name} model saved to {model_path}")
        
        # Save ensemble weights
        print("💾 Saving ensemble weights...")
        weights_data = {
            'models': model_names,
            'weights': ensemble_weights,
            'validation_scores': val_scores
        }
        with open(os.path.join(args.model_dir, "ensemble_weights.json"), "w") as f:
            json.dump(weights_data, f, indent=2)
        print("  Ensemble weights saved")
        
        # Save evaluation results
        print("💾 Saving evaluation results...")
        with open(os.path.join(args.model_dir, "evaluation.json"), "w") as f:
            json.dump(model_metrics, f, indent=2)
        print("  Evaluation results saved")
        
        # Print summary
        print("\n" + "="*60)
        print("🏆 MODEL PERFORMANCE SUMMARY")
        print("="*60)
        for model_name, metrics in model_metrics.items():
            if model_name == "Ensemble":
                print(f"\n🏆 {model_name}:")
            else:
                print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
        
        print("\n🎉 Training completed successfully!")
        print("✅ All models trained and saved successfully!")
        
    except Exception as e:
        print(f"💥 FATAL ERROR during training: {e}")
        print("🔍 Detailed traceback:")
        traceback.print_exc()
        sys.exit(1)