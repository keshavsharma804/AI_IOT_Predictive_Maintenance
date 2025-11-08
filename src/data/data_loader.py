"""
Data Loading Module
Handles loading and preprocessing of sensor data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from ..utils.logger import LoggerMixin
from ..config import get_config


class DataLoader(LoggerMixin):
    """Loads and preprocesses sensor data"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.data_path = self.config.get('paths.data_synthetic', 'data/synthetic')
    
    def load_machine_data(
        self, 
        machine_id: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific machine
        
        Args:
            machine_id: Machine identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with sensor data
        """
        filepath = Path(self.data_path) / f"{machine_id}_data.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range if specified
        if start_date is not None:
            df = df[df['timestamp'] >= start_date]
        if end_date is not None:
            df = df[df['timestamp'] <= end_date]
        
        self.logger.info(f"Loaded {len(df)} samples for {machine_id}")
        
        return df
    
    def load_all_machines(self) -> pd.DataFrame:
        """
        Load data for all machines
        
        Returns:
            Combined DataFrame with all machines
        """
        data_path = Path(self.data_path)
        csv_files = list(data_path.glob("machine_*_data.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No machine data files found in {data_path}")
        
        self.logger.info(f"Found {len(csv_files)} machine data files")
        
        all_data = []
        for filepath in csv_files:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        self.logger.info(f"Loaded total {len(combined_df)} samples from {len(csv_files)} machines")
        
        return combined_df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess sensor data
        
        - Handle missing values
        - Remove outliers (optional)
        - Sort by timestamp
        
        Args:
            df: Raw sensor data
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing data...")
        
        # Sort by timestamp
        df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
        
        # Handle missing values (forward fill then backward fill)
        sensor_cols = ['vibration_x', 'vibration_y', 'vibration_z', 
                      'temperature', 'pressure', 'current']
        
        for col in sensor_cols:
            if col in df.columns:
                # Forward fill
                df[col] = df.groupby('machine_id')[col].fillna(method='ffill')
                # Backward fill for remaining NaNs
                df[col] = df.groupby('machine_id')[col].fillna(method='bfill')
        
        # Remove any remaining rows with NaN
        initial_len = len(df)
        df = df.dropna(subset=sensor_cols)
        removed = initial_len - len(df)
        
        if removed > 0:
            self.logger.warning(f"Removed {removed} rows with missing values")
        
        self.logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        
        return df
    
    def get_normal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only normal (non-anomalous) data
        
        Args:
            df: DataFrame with labels
            
        Returns:
            DataFrame with only normal data
        """
        if 'is_anomaly' not in df.columns:
            self.logger.warning("No 'is_anomaly' column found, returning all data")
            return df
        
        normal_df = df[df['is_anomaly'] == 0].copy()
        self.logger.info(f"Extracted {len(normal_df)} normal samples ({len(normal_df)/len(df)*100:.1f}%)")
        
        return normal_df
    
    def get_anomaly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only anomalous data
        
        Args:
            df: DataFrame with labels
            
        Returns:
            DataFrame with only anomalies
        """
        if 'is_anomaly' not in df.columns:
            self.logger.warning("No 'is_anomaly' column found, returning empty DataFrame")
            return pd.DataFrame()
        
        anomaly_df = df[df['is_anomaly'] == 1].copy()
        self.logger.info(f"Extracted {len(anomaly_df)} anomalous samples ({len(anomaly_df)/len(df)*100:.1f}%)")
        
        return anomaly_df
    
    def split_by_machine(self, df: pd.DataFrame) -> dict:
        """
        Split data by machine ID
        
        Args:
            df: Combined DataFrame
            
        Returns:
            Dictionary with machine_id as keys and DataFrames as values
        """
        machine_data = {}
        
        for machine_id in df['machine_id'].unique():
            machine_data[machine_id] = df[df['machine_id'] == machine_id].copy()
        
        self.logger.info(f"Split data into {len(machine_data)} machines")
        
        return machine_data
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the data
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            'total_samples': len(df),
            'num_machines': df['machine_id'].nunique() if 'machine_id' in df.columns else 1,
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'anomalies': {
                'count': df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0,
                'percentage': df['is_anomaly'].mean() * 100 if 'is_anomaly' in df.columns else 0
            },
            'failure_types': df['failure_type'].value_counts().to_dict() if 'failure_type' in df.columns else {},
            'sensor_stats': {}
        }
        
        # Sensor statistics
        sensor_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
        for col in sensor_cols:
            if col in df.columns:
                summary['sensor_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
        
        return summary
    
    def print_summary(self, df: pd.DataFrame):
        """Print data summary"""
        summary = self.get_data_summary(df)
        
        print("\n" + "="*80)
        print("DATA SUMMARY".center(80))
        print("="*80)
        
        print(f"\nTotal Samples: {summary['total_samples']:,}")
        print(f"Number of Machines: {summary['num_machines']}")
        
        print(f"\nDate Range:")
        print(f"  Start: {summary['date_range']['start']}")
        print(f"  End: {summary['date_range']['end']}")
        print(f"  Duration: {summary['date_range']['duration_days']} days")
        
        print(f"\nAnomalies:")
        print(f"  Count: {summary['anomalies']['count']:,}")
        print(f"  Percentage: {summary['anomalies']['percentage']:.2f}%")
        
        if summary['failure_types']:
            print(f"\nFailure Types:")
            for failure_type, count in summary['failure_types'].items():
                print(f"  {failure_type}: {count:,}")
        
        print(f"\nSensor Statistics:")
        for sensor, stats in summary['sensor_stats'].items():
            print(f"\n  {sensor}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std: {stats['std']:.2f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"    Median: {stats['median']:.2f}")
        
        print("\n" + "="*80 + "\n")


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Load single machine
    df = loader.load_machine_data("machine_001")
    loader.print_summary(df)
    
    # Preprocess
    df_clean = loader.preprocess(df)
    
    # Get normal vs anomaly data
    normal_data = loader.get_normal_data(df_clean)
    anomaly_data = loader.get_anomaly_data(df_clean)
    
    print(f"Normal samples: {len(normal_data)}")
    print(f"Anomaly samples: {len(anomaly_data)}")