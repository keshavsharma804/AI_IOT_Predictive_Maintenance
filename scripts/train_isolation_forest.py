"""
Train Isolation Forest Model
Complete training pipeline for anomaly detection
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.config import get_config
from src.features.feature_engineering import FeatureEngineer
from src.models.isolation_forest_model import IsolationForestModel
from src.models.model_trainer import ModelTrainer


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train Isolation Forest model')
    parser.add_argument('--features', type=str, default='machine_001_features.csv',
                       help='Feature file name')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ISOLATION FOREST TRAINING PIPELINE".center(80))
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Initialize components
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    
    # Load features
    print(f"\nLoading features from {args.features}...")
    features_df = engineer.load_features(args.features)
    
    print(f"  Loaded: {features_df.shape}")
    print(f"  Anomalies: {features_df['is_anomaly'].sum()} ({features_df['is_anomaly'].mean()*100:.2f}%)")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        features_df,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Initialize and train model
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    model = IsolationForestModel(config)
    model.train(X_train)
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    
    val_metrics = model.evaluate(X_val, y_val)
    model.print_metrics()
    
    # Optimize threshold
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    
    optimal_threshold = model.optimize_threshold(X_val, y_val, target_metric='f1')
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    test_metrics = model.evaluate(X_test, y_test)
    model.print_metrics()
    
    # Get predictions for visualization
    y_pred = model.predict(X_test)
    y_scores = model.predict_scores(X_test)
    
    # Plot confusion matrix
    print("\nGenerating visualizations...")
    figures_path = Path(config.get('paths.figures', 'results/figures'))
    
    trainer.plot_confusion_matrix(
        y_test, y_pred,
        save_path=figures_path / 'isolation_forest_confusion_matrix.png'
    )
    
        # Plot ROC curve
    trainer.plot_roc_curve(
        y_test, y_scores,
        save_path=figures_path / 'isolation_forest_roc_curve.png'
    )

    # Plot Precision-Recall curve
    trainer.plot_precision_recall_curve(
        y_test, y_scores,
        save_path=figures_path / 'isolation_forest_pr_curve.png'
    )

    # Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    print("\nCalculating feature importance (this may take a moment)...")
    importance = model.get_feature_importance(X_test.iloc[:200], n_top=20)

    print("\nTop 20 Most Important Features:")
    print(importance.to_string(index=False))

    # Save feature importance
    importance_path = Path(config.get('paths.results', 'results')) / 'feature_importance.csv'
    importance.to_csv(importance_path, index=False)
    print(f"\n✓ Feature importance saved to: {importance_path}")

    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    model_path = Path(config.get('paths.models', 'models/saved_models')) / 'isolation_forest.pkl'
    model.save(str(model_path))
    print(f"✓ Model saved to: {model_path}")

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    print(f"\n✓ Model: Isolation Forest")
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Test samples: {len(X_test):,}")
    print(f"✓ Features: {X_train.shape[1]}")

    print(f"\n✓ Performance Metrics:")
    print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Precision:    {test_metrics['precision']:.4f}")
    print(f"  Recall:       {test_metrics['recall']:.4f}")
    print(f"  F1-Score:     {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:      {test_metrics['roc_auc']:.4f}")

    print(f"\n✓ Business Metrics:")
    print(f"  True Positives:  {test_metrics['true_positive']}")
    print(f"  False Positives: {test_metrics['false_positive']}")
    print(f"  True Negatives:  {test_metrics['true_negative']}")
    print(f"  False Negatives: {test_metrics['false_negative']}")
    print(f"  False Alarm Rate: {test_metrics['false_positive_rate']:.4f}")

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)

    print("\nNext steps:")
    print("  1. Review visualizations in results/figures/")
    print("  2. Analyze feature importance")
    print("  3. Deploy model for predictions")
    print("  4. Compare with other models (LSTM, Random Forest)")