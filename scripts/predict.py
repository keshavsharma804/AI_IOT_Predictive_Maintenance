"""
Prediction Script
Use trained ensemble model for real-time predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.config import get_config
from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble_model import EnsembleModel


def main():
    """Main prediction script"""
    parser = argparse.ArgumentParser(description='Make predictions with ensemble model')
    parser.add_argument('--features', type=str, required=True,
                       help='Feature file for prediction')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/ensemble_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file for predictions')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANOMALY DETECTION - PREDICTION MODE".center(80))
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    ensemble = EnsembleModel(config)
    ensemble.load(args.model)
    print("✓ Model loaded")
    
    # Load features
    print(f"\nLoading features from {args.features}...")
    engineer = FeatureEngineer(config)
    features_df = engineer.load_features(args.features)
    print(f"✓ Loaded {len(features_df)} samples")

    # Prepare features
    feature_cols = [col for col in features_df.columns 
                if col not in ['window_id', 'window_start', 'window_end',
                                'machine_id', 'is_anomaly', 'failure_type', 'severity']]
    X = features_df[feature_cols]

    # Make predictions
    print("\nMaking predictions...")
    predictions, confidence, individual_preds = ensemble.predict_with_confidence(X)
    ensemble_scores, individual_scores = ensemble.predict_scores(X)

    # Create results DataFrame
    results_df = features_df[['window_id', 'window_start', 'window_end', 'machine_id']].copy()
    results_df['prediction'] = predictions
    results_df['prediction_label'] = results_df['prediction'].map({0: 'Normal', 1: 'Anomaly'})
    results_df['confidence'] = confidence
    results_df['anomaly_score'] = ensemble_scores

    # Add individual model predictions
    for model_name, preds in individual_preds.items():
        results_df[f'{model_name}_prediction'] = preds

    # Add true labels if available
    if 'is_anomaly' in features_df.columns:
        results_df['actual'] = features_df['is_anomaly']
        results_df['correct'] = (results_df['prediction'] == results_df['actual'])

    # Save results
    output_path = Path(config.get('paths.results', 'results')) / args.output
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Predictions saved to {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)

    anomaly_count = predictions.sum()
    anomaly_pct = (anomaly_count / len(predictions)) * 100

    print(f"\nTotal Samples:    {len(predictions):,}")
    print(f"Predicted Normal: {len(predictions) - anomaly_count:,} ({100-anomaly_pct:.2f}%)")
    print(f"Predicted Anomaly: {anomaly_count:,} ({anomaly_pct:.2f}%)")
    print(f"Average Confidence: {confidence.mean():.2%}")

    # High-confidence anomalies
    high_conf_anomalies = ((predictions == 1) & (confidence > 0.8)).sum()
    print(f"\nHigh-Confidence Anomalies: {high_conf_anomalies} (confidence > 80%)")

    # If true labels available, show accuracy
    if 'is_anomaly' in features_df.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = features_df['is_anomaly']
        
        print("\n" + "-"*80)
        print("ACCURACY METRICS")
        print("-"*80)
        print(f"Accuracy:  {accuracy_score(y_true, predictions):.2%}")
        print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.2%}")
        print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.2%}")
        print(f"F1-Score:  {f1_score(y_true, predictions, zero_division=0):.2%}")

    # Show some high-priority alerts
    print("\n" + "-"*80)
    print("HIGH-PRIORITY ALERTS (Top 5)")
    print("-"*80)

    alert_df = results_df[results_df['prediction'] == 1].nlargest(5, 'confidence')

    for idx, row in alert_df.iterrows():
        print(f"\nAlert #{idx+1}:")
        print(f"  Time: {row['window_start']} to {row['window_end']}")
        print(f"  Machine: {row['machine_id']}")
        print(f"  Confidence: {row['confidence']:.1%}")
        print(f"  Anomaly Score: {row['anomaly_score']:.4f}")

    print("\n" + "="*80)
    print("✓ Prediction complete!")
    print("="*80)