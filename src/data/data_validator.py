"""
Data Validation Module
Validates sensor data quality and detects issues
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from ..utils.logger import LoggerMixin


class DataValidator(LoggerMixin):
    """Validates IoT sensor data quality"""
    
    def __init__(self):
        """Initialize data validator"""
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Run all validation checks
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Running data validation...")
        
        results = {
            'missing_values': self._check_missing_values(df),
            'outliers': self._check_outliers(df),
            'stuck_sensors': self._check_stuck_values(df),
            'range_violations': self._check_range_violations(df),
            'temporal_gaps': self._check_temporal_gaps(df),
            'summary': {}
        }
        
        # Summary
        total_issues = sum([
            results['missing_values']['total'],
            results['outliers']['total'],
            results['stuck_sensors']['total'],
            results['range_violations']['total']
        ])
        
        results['summary'] = {
            'total_samples': len(df),
            'total_issues': total_issues,
            'data_quality_score': 1 - (total_issues / (len(df) * 6)),  # 6 sensor columns
            'pass': total_issues < len(df) * 0.05  # Pass if <5% issues
        }
        
        self.validation_results = results
        self.logger.info(f"Validation complete. Quality score: {results['summary']['data_quality_score']:.2%}")
        
        return results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        sensor_cols = ['vibration_x', 'vibration_y', 'vibration_z', 
                      'temperature', 'pressure', 'current']
        
        missing = df[sensor_cols].isnull().sum()
        
        return {
            'total': missing.sum(),
            'by_column': missing.to_dict(),
            'percentage': (missing.sum() / (len(df) * len(sensor_cols))) * 100
        }
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Check for statistical outliers (IQR method)"""
        sensor_cols = ['vibration_x', 'vibration_y', 'vibration_z', 
                      'temperature', 'pressure', 'current']
        
        outlier_counts = {}
        
        for col in sensor_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = int(outliers)
        
        return {
            'total': sum(outlier_counts.values()),
            'by_column': outlier_counts
        }
    
    def _check_stuck_values(self, df: pd.DataFrame, threshold: int = 10) -> Dict:
        """Check for stuck sensor values (same value repeated)"""
        sensor_cols = ['vibration_x', 'vibration_y', 'vibration_z', 
                      'temperature', 'pressure', 'current']
        
        stuck_counts = {}
        
        for col in sensor_cols:
            # Find sequences of identical values
            diff = df[col].diff().fillna(1)
            stuck = (diff == 0)
            
            # Count runs longer than threshold
            runs = stuck.astype(int).groupby((stuck != stuck.shift()).cumsum()).sum()
            long_runs = (runs >= threshold).sum()
            
            stuck_counts[col] = int(long_runs)
        
        return {
            'total': sum(stuck_counts.values()),
            'by_column': stuck_counts,
            'threshold': threshold
        }
    
    def _check_range_violations(self, df: pd.DataFrame) -> Dict:
        """Check if values are within expected ranges"""
        ranges = {
            'vibration_x': (0, 20),
            'vibration_y': (0, 20),
            'vibration_z': (0, 20),
            'temperature': (0, 150),
            'pressure': (0, 200),
            'current': (0, 30)
        }
        
        violations = {}
        
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                violations[col] = int(((df[col] < min_val) | (df[col] > max_val)).sum())
        
        return {
            'total': sum(violations.values()),
            'by_column': violations
        }
    
    def _check_temporal_gaps(self, df: pd.DataFrame) -> Dict:
        """Check for gaps in timestamp sequence"""
        if 'timestamp' not in df.columns:
            return {'gaps': 0, 'max_gap_seconds': 0}
        
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
        
        # Expected time between samples
        expected_diff = 1.0 / 100  # 100Hz = 0.01 seconds
        
        # Find gaps larger than 2x expected
        gaps = time_diffs[time_diffs > expected_diff * 2]
        
        return {
            'gaps': len(gaps),
            'max_gap_seconds': float(time_diffs.max()) if len(time_diffs) > 0 else 0
        }
    
    def print_report(self):
        """Print validation report"""
        if not self.validation_results:
            self.logger.warning("No validation results available. Run validate() first.")
            return
        
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT".center(80))
        print("="*80)
        
        summary = self.validation_results['summary']
        print(f"\nTotal Samples: {summary['total_samples']:,}")
        print(f"Total Issues: {summary['total_issues']:,}")
        print(f"Data Quality Score: {summary['data_quality_score']:.2%}")
        print(f"Status: {'✓ PASS' if summary['pass'] else '✗ FAIL'}")
        
        print("\n" + "-"*80)
        print("MISSING VALUES")
        print("-"*80)
        mv = self.validation_results['missing_values']
        print(f"Total: {mv['total']} ({mv['percentage']:.3f}%)")
        for col, count in mv['by_column'].items():
            if count > 0:
                print(f"  {col}: {count}")
        
        print("\n" + "-"*80)
        print("OUTLIERS (IQR Method)")
        print("-"*80)
        outliers = self.validation_results['outliers']
        print(f"Total: {outliers['total']}")
        for col, count in outliers['by_column'].items():
            if count > 0:
                print(f"  {col}: {count}")
        
        print("\n" + "-"*80)
        print("STUCK SENSORS")
        print("-"*80)
        stuck = self.validation_results['stuck_sensors']
        print(f"Total Events: {stuck['total']} (threshold: {stuck['threshold']} samples)")
        for col, count in stuck['by_column'].items():
            if count > 0:
                print(f"  {col}: {count} events")
        
        print("\n" + "-"*80)
        print("RANGE VIOLATIONS")
        print("-"*80)
        ranges = self.validation_results['range_violations']
        print(f"Total: {ranges['total']}")
        for col, count in ranges['by_column'].items():
            if count > 0:
                print(f"  {col}: {count}")
        
        print("\n" + "-"*80)
        print("TEMPORAL GAPS")
        print("-"*80)
        gaps = self.validation_results['temporal_gaps']
        print(f"Gaps Found: {gaps['gaps']}")
        print(f"Max Gap: {gaps['max_gap_seconds']:.4f} seconds")
        
        print("\n" + "="*80 + "\n")


# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('data/synthetic/machine_001_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Validate
    validator = DataValidator()
    results = validator.validate(df)
    validator.print_report()