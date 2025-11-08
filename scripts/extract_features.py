import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from tqdm import tqdm

from src.config import get_config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer


def main():
    """Main feature extraction script - COMPLETE PIPELINE"""
    parser = argparse.ArgumentParser(
        description='Extract ALL features (time + frequency + temporal)'
    )
    parser.add_argument('--window-size', type=int, default=1000,
                       help='Window size for feature extraction')
    parser.add_argument('--stride', type=int, default=500,
                       help='Stride for sliding window')
    parser.add_argument('--machine', type=str, default='machine_001',
                       help='Machine ID or "all" for all machines')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPLETE FEATURE EXTRACTION - ALL DOMAINS".center(80))
    print("="*80)
    
    # Load configuration
    config = get_config()
    
    # Bearing parameters
    bearing_params = {
        'shaft_speed': 1800,
        'num_balls': 9,
        'ball_diameter': 10,
        'pitch_diameter': 50,
        'contact_angle': 0
    }
    
    # Initialize
    loader = DataLoader(config)
    engineer = FeatureEngineer(config)
    
    print(f"\nConfiguration:")
    print(f"  Window Size: {args.window_size}")
    print(f"  Stride: {args.stride}")
    print(f"  Target: {args.machine}")
    print(f"  Feature Domains:")
    print(f"    ✓ Time-domain")
    print(f"    ✓ Frequency-domain")
    print(f"    ✓ Temporal")
    print(f"    ✓ Rolling windows")
    
    # Determine machines to process
    if args.machine == 'all':
        data_path = Path(config.get('paths.data_synthetic', 'data/synthetic'))
        machine_files = list(data_path.glob('machine_*_data.csv'))
        machine_ids = [f.stem.replace('_data', '') for f in machine_files]
        print(f"\nFound {len(machine_ids)} machines")
    else:
        machine_ids = [args.machine]
    
    # Process each machine
    all_features = []
    
    for machine_id in tqdm(machine_ids, desc="Processing"):
        try:
            print(f"\n{'='*80}")
            print(f"Processing {machine_id}")
            print('='*80)
            
            # Load data
            print(f"Loading data...")
            df = loader.load_machine_data(machine_id)
            print(f"  Loaded {len(df):,} samples")
            
            # Extract ALL features
            print(f"\nExtracting complete feature set...")
            features_df = engineer.engineer_complete_features(
                df,
                window_size=args.window_size,
                stride=args.stride,
                include_rolling=True,
                include_frequency=True,
                include_temporal=True,
                bearing_params=bearing_params
            )
            
            print(f"\n✓ Extraction complete!")
            print(f"  Windows: {len(features_df):,}")
            print(f"  Features per window: {len(features_df.columns)}")
            print(f"  Dataset size: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Save
            output_file = f'{machine_id}_features_all.csv'
            engineer.save_features(features_df, output_file)
            
            all_features.append(features_df)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all
    if len(all_features) > 1:
        print(f"\n{'='*80}")
        print("Combining all machines...")
        print('='*80)
        
        combined = pd.concat(all_features, ignore_index=True)
        engineer.save_features(combined, 'all_machines_features_all.csv')
        
        print(f"✓ Combined dataset:")
        print(f"  Total windows: {len(combined):,}")
        print(f"  Total features: {len(combined.columns)}")
        print(f"  Size: {combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Analysis
    print(f"\n{'='*80}")
    print("FEATURE ANALYSIS")
    print('='*80)
    
    if all_features:
        sample = all_features[0]
        
        # Categorize features
        all_cols = [col for col in sample.columns 
                   if col not in ['window_id', 'window_start', 'window_end',
                                 'machine_id', 'is_anomaly', 'failure_type', 'severity']]
        
        time_domain = [c for c in all_cols if any(x in c for x in 
                      ['mean', 'std', 'rms', 'kurtosis', 'skewness', 'peak', 
                       'crest', 'shape', 'impulse', 'clearance', 'energy'])]
        
        frequency = [c for c in all_cols if any(x in c for x in 
                    ['fft', 'spectral', 'power', 'band', 'harmonic', 
                     'bpfo', 'bpfi', 'bsf', 'ftf'])]
        
        temporal = [c for c in all_cols if any(x in c for x in 
                   ['lag', 'velocity', 'acceleration', 'trend', 'autocorr', 
                    'xcorr', 'cumsum', 'degradation'])]
        
        rolling = [c for c in all_cols if 'rolling' in c or 'ewma' in c]
        
        print(f"\n✓ Feature Breakdown:")
        print(f"  Time-domain features:      {len(time_domain):>4}")
        print(f"  Frequency-domain features: {len(frequency):>4}")
        print(f"  Temporal features:         {len(temporal):>4}")
        print(f"  Rolling window features:   {len(rolling):>4}")
        print(f"  " + "-"*40)
        print(f"  TOTAL:                     {len(all_cols):>4}")
        
        # Top features
        stats = engineer.get_feature_importance_analysis(sample)
        
        print(f"\nTop 20 Features by Variance:")
        print(stats.head(20)[['feature', 'variance', 'cv']].to_string(index=False))
        
        # Save
        stats_path = Path(config.get('paths.data_features')) / 'feature_statistics_all.csv'
        stats.to_csv(stats_path, index=False)
        print(f"\n✓ Statistics saved: {stats_path}")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE FEATURE EXTRACTION FINISHED!")
    print('='*80)
    
    print(f"\nYou now have a COMPLETE feature set ready for ML models!")
    print(f"\nNext steps:")
    print(f"  1. Feature selection (remove low-variance features)")
    print(f"  2. Feature scaling/normalization")
    print(f"  3. Train ML models (Isolation Forest, LSTM, etc.)")
    print(f"  4. Evaluate and optimize")


if __name__ == "__main__":
    main()