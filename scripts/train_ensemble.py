"""
Train Complete Ensemble Model
Final production-ready model combining all approaches
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import get_config
from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble_model import EnsembleModel
from src.models.model_trainer import ModelTrainer


def main():
    """Main ensemble training script"""
    parser = argparse.ArgumentParser(description='Train Ensemble Model')
    parser.add_argument('--features', type=str, default='machine_001_features.csv',
                       help='Feature file name')
    parser.add_argument('--optimize-weights', action='store_true',
                       help='Optimize ensemble weights')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE MODEL TRAINING - FINAL PRODUCTION MODEL".center(80))
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Initialize
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    
    # Load features
    print(f"\nLoading features...")
    features_df = engineer.load_features(args.features)
    print(f"  Shape: {features_df.shape}")
    print(f"  Anomalies: {features_df['is_anomaly'].sum()} "
          f"({features_df['is_anomaly'].mean()*100:.2f}%)")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        features_df, test_size=0.2, val_size=0.1
    )
    
    # Train ensemble
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE")
    print("="*80)
    
    ensemble = EnsembleModel(config)
    ensemble.train(X_train, y_train)
    
    # Optimize weights if requested
    if args.optimize_weights:
        print("\n" + "="*80)
        print("OPTIMIZING WEIGHTS")
        print("="*80)
        
        optimal_weights = ensemble.optimize_weights(X_val, y_val, metric='f1')
    
    # Evaluate
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    metrics = ensemble.evaluate(X_test, y_test)
    ensemble.print_metrics()
    
    # Get predictions with confidence
    predictions, confidence, individual_preds = ensemble.predict_with_confidence(X_test)
    ensemble_scores, individual_scores = ensemble.predict_scores(X_test)
    
    # Visualizations
    print("\nGenerating visualizations...")
    figures_path = Path(config.get('paths.figures', 'results/figures'))
    
    # 1. Confusion Matrix
    trainer.plot_confusion_matrix(
        y_test, predictions,
        save_path=figures_path / 'ensemble_confusion_matrix.png'
    )
    
    # 2. ROC Curve
    trainer.plot_roc_curve(
        y_test, ensemble_scores,
        save_path=figures_path / 'ensemble_roc_curve.png'
    )
    
    # 3. Confidence Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidence[y_test == 0], bins=30, alpha=0.6, label='Normal', 
           color='green', density=True, edgecolor='black')
    ax.hist(confidence[y_test == 1], bins=30, alpha=0.6, label='Anomaly',
           color='red', density=True, edgecolor='black')
    
    ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Prediction Confidence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_path / 'ensemble_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model Agreement
    pred_array = np.array(list(individual_preds.values()))
    agreement = np.mean(pred_array, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if y == 0 else 'red' for y in y_test]
    ax.scatter(range(len(agreement)), agreement, c=colors, alpha=0.5, s=10)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    
    ax.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Agreement', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Model Agreement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_path / 'ensemble_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualizations to {figures_path}")
    
    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    model_path = Path(config.get('paths.models', 'models/saved_models')) / 'ensemble_model.pkl'
    ensemble.save(str(model_path))
    
    # Final Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    
    print(f"\n✓ Ensemble Model Performance:")
    print(f"  Accuracy:        {metrics['accuracy']:.2%}")
    print(f"  Precision:       {metrics['precision']:.2%}")
    print(f"  Recall:          {metrics['recall']:.2%}")
    print(f"  F1-Score:        {metrics['f1_score']:.2%}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  Avg Confidence:  {metrics['avg_confidence']:.2%}")
    
    print(f"\n✓ Business Impact:")
    cost_downtime = config.get('business.cost_unplanned_downtime', 5000)
    tp = metrics['true_positive']
    fp = metrics['false_positive']
    
    downtime_prevented = tp * 4 * cost_downtime
    false_alarm_cost = fp * 100
    net_benefit = downtime_prevented - false_alarm_cost
    
    print(f"  Downtime Prevented:  ${downtime_prevented:,.2f}")
    print(f"  False Alarm Costs:   ${false_alarm_cost:,.2f}")
    print(f"  Net Benefit:         ${net_benefit:,.2f}")
    
    print("\n✓ Model saved and ready for deployment!")
    
    print("\n" + "="*80)
    print("🎉 COMPLETE ML PIPELINE FINISHED! 🎉".center(80))
    print("="*80)


if __name__ == "__main__":
    main()