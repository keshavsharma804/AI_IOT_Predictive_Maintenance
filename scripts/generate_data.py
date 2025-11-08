"""
Script to generate synthetic IoT sensor data
Run this to create training data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.sensor_simulator import SensorSimulator
from src.data.data_validator import DataValidator
from src.data.data_loader import DataLoader
import argparse


def main():
    """Main data generation script"""
    parser = argparse.ArgumentParser(description='Generate synthetic IoT sensor data')
    parser.add_argument('--machines', type=int, default=5, help='Number of machines')
    parser.add_argument('--days', type=int, default=30, help='Days of data to generate')
    parser.add_argument('--validate', action='store_true', help='Run validation after generation')
    
    args = parser.parse_args()
    
    print("="*80)
    print("IoT SENSOR DATA GENERATION".center(80))
    print("="*80)
    
    # Load configuration
    config = get_config()
    
    # Update config with command line arguments
    config.config['data_generation']['num_machines'] = args.machines
    config.config['data_generation']['duration_days'] = args.days
    
    # Initialize simulator
    print(f"\nInitializing simulator...")
    print(f"  Machines: {args.machines}")
    print(f"  Duration: {args.days} days")
    print(f"  Sampling Rate: {config.get('data_generation.sampling_rate')} Hz")
    
    simulator = SensorSimulator(config)
    
    # Generate data for all machines
    print(f"\nGenerating data...")
    df = simulator.generate_multiple_machines()
    
    # Save combined data
    output_path = Path(config.get('paths.data_synthetic')) / 'all_machines_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved combined data to: {output_path}")
    
    # Save individual machine data
    for machine_id in df['machine_id'].unique():
        machine_df = df[df['machine_id'] == machine_id]
        machine_path = Path(config.get('paths.data_synthetic')) / f'{machine_id}_data.csv'
        machine_df.to_csv(machine_path, index=False)
        print(f"Saved {machine_id} data to: {machine_path}")
    
    # Save failure summary
    failure_summary = simulator.get_failure_summary()
    summary_path = Path(config.get('paths.data_synthetic')) / 'failure_summary.csv'
    failure_summary.to_csv(summary_path, index=False)
    print(f"\nSaved failure summary to: {summary_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("GENERATION SUMMARY".center(80))
    print("="*80)
    
    loader = DataLoader(config)
    loader.print_summary(df)
    
    print("\nFailure Events Summary:")
    print(failure_summary)
    
    # Validate if requested
    if args.validate:
        print("\n" + "="*80)
        print("RUNNING VALIDATION".center(80))
        print("="*80)
        
        validator = DataValidator()
        results = validator.validate(df)
        validator.print_report()
    
    print("\n✓ Data generation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the generated data in: {config.get('paths.data_synthetic')}")
    print(f"  2. Run exploratory data analysis (EDA)")
    print(f"  3. Proceed to feature engineering")


if __name__ == "__main__":
    main()