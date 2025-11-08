"""
Report Generation Module
Create professional PDF/HTML reports with visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin
from ..config import get_config
from .plots import SensorPlotter


class ReportGenerator(LoggerMixin):
    """
    Generate professional analysis reports
    
    Features:
    - Executive summary
    - Detailed analysis with visualizations
    - Performance metrics
    - Recommendations
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize report generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.report_path = Path(self.config.get('paths.reports', 'results/reports'))
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        self.plotter = SensorPlotter(config)
        
        self.logger.info("ReportGenerator initialized")
    
    def generate_data_quality_report(
        self,
        df: pd.DataFrame,
        validation_results: Dict,
        filename: str = 'data_quality_report.txt'
    ) -> str:
        """
        Generate data quality assessment report
        
        Args:
            df: DataFrame with sensor data
            validation_results: Results from DataValidator
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        self.logger.info("Generating data quality report")
        
        report_path = self.report_path / filename
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA QUALITY ASSESSMENT REPORT\n".center(80))
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            summary = validation_results.get('summary', {})
            f.write(f"Total Samples: {summary.get('total_samples', 0):,}\n")
            f.write(f"Data Quality Score: {summary.get('data_quality_score', 0):.2%}\n")
            f.write(f"Status: {'PASS' if summary.get('pass', False) else 'FAIL'}\n\n")
            
            # Missing Values
            f.write("MISSING VALUES\n")
            f.write("-"*80 + "\n")
            mv = validation_results.get('missing_values', {})
            f.write(f"Total: {mv.get('total', 0)} ({mv.get('percentage', 0):.3f}%)\n")
            f.write("\nBreakdown by column:\n")
            for col, count in mv.get('by_column', {}).items():
                if count > 0:
                    f.write(f"  {col}: {count}\n")
            f.write("\n")
            
            # Outliers
            f.write("OUTLIERS (3x IQR Method)\n")
            f.write("-"*80 + "\n")
            outliers = validation_results.get('outliers', {})
            f.write(f"Total: {outliers.get('total', 0)}\n")
            f.write("\nBreakdown by column:\n")
            for col, count in outliers.get('by_column', {}).items():
                if count > 0:
                    f.write(f"  {col}: {count}\n")
            f.write("\n")
            
            # Stuck Sensors
            f.write("STUCK SENSOR EVENTS\n")
            f.write("-"*80 + "\n")
            stuck = validation_results.get('stuck_sensors', {})
            f.write(f"Total Events: {stuck.get('total', 0)}\n")
            f.write(f"Detection Threshold: {stuck.get('threshold', 10)} consecutive samples\n")
            f.write("\nBreakdown by sensor:\n")
            for col, count in stuck.get('by_column', {}).items():
                if count > 0:
                    f.write(f"  {col}: {count} events\n")
            f.write("\n")
            
            # Range Violations
            f.write("RANGE VIOLATIONS\n")
            f.write("-"*80 + "\n")
            ranges = validation_results.get('range_violations', {})
            f.write(f"Total: {ranges.get('total', 0)}\n")
            f.write("\nBreakdown by sensor:\n")
            for col, count in ranges.get('by_column', {}).items():
                if count > 0:
                    f.write(f"  {col}: {count}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            
            if summary.get('data_quality_score', 0) < 0.95:
                f.write("⚠ Data quality is below acceptable threshold (95%)\n\n")
                f.write("Recommended actions:\n")
                f.write("1. Investigate and repair sensors with high outlier rates\n")
                f.write("2. Review data collection pipeline for missing values\n")
                f.write("3. Implement sensor health monitoring\n")
                f.write("4. Consider recalibration of affected sensors\n")
            else:
                f.write("✓ Data quality is acceptable\n\n")
                f.write("Continue with:\n")
                f.write("1. Regular quality monitoring\n")
                f.write("2. Preventive maintenance schedules\n")
                f.write("3. Feature engineering and model development\n")
            
            f.write("\n" + "="*80 + "\n")
        
        self.logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def generate_anomaly_report(
        self,
        df: pd.DataFrame,
        failure_summary: pd.DataFrame,
        filename: str = 'anomaly_analysis_report.txt'
    ) -> str:
        """
        Generate anomaly analysis report
        
        Args:
            df: DataFrame with sensor data and labels
            failure_summary: DataFrame with failure events
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        self.logger.info("Generating anomaly analysis report")
        
        report_path = self.report_path / filename
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ANOMALY ANALYSIS REPORT\n".center(80))
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")
            
            # Overview
            f.write("OVERVIEW\n")
            f.write("-"*80 + "\n")
            total_samples = len(df)
            anomaly_samples = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            anomaly_rate = (anomaly_samples / total_samples * 100) if total_samples > 0 else 0
            
            f.write(f"Total Samples: {total_samples:,}\n")
            f.write(f"Anomalous Samples: {anomaly_samples:,}\n")
            f.write(f"Anomaly Rate: {anomaly_rate:.2f}%\n\n")
            
            # Failure Events
            f.write("FAILURE EVENTS\n")
            f.write("-"*80 + "\n")
            if not failure_summary.empty:
                f.write(f"Total Failures: {len(failure_summary)}\n\n")
                
                failure_counts = failure_summary['failure_type'].value_counts()
                f.write("Failure Type Distribution:\n")
                for failure_type, count in failure_counts.items():
                    f.write(f"  {failure_type}: {count}\n")
                
                f.write("\nDetailed Failure Events:\n")
                for idx, row in failure_summary.iterrows():
                    f.write(f"\n  Event {idx + 1}:\n")
                    f.write(f"    Machine: {row['machine_id']}\n")
                    f.write(f"    Type: {row['failure_type']}\n")
                    f.write(f"    Start: {row['start_time']}\n")
                    f.write(f"    End: {row['end_time']}\n")
                    f.write(f"    Duration: {row['duration_days']} days\n")
                    f.write(f"    Severity: {row['severity']:.2f}\n")
                    f.write(f"    Affected Sensors: {row['affected_sensors']}\n")
            else:
                f.write("No failure events recorded.\n")
            
            f.write("\n")
            
            # Sensor Statistics During Anomalies
            f.write("SENSOR BEHAVIOR DURING ANOMALIES\n")
            f.write("-"*80 + "\n")
            
            if 'is_anomaly' in df.columns and df['is_anomaly'].sum() > 0:
                sensor_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
                
                f.write("Comparison: Normal vs Anomalous Conditions\n\n")
                
                for col in sensor_cols:
                    if col in df.columns:
                        normal_stats = df.loc[df['is_anomaly'] == 0, col].describe()
                        anomaly_stats = df.loc[df['is_anomaly'] == 1, col].describe()
                        
                        f.write(f"{col.replace('_', ' ').title()}:\n")
                        f.write(f"  Normal  - Mean: {normal_stats['mean']:.2f}, "
                               f"Std: {normal_stats['std']:.2f}, "
                               f"Max: {normal_stats['max']:.2f}\n")
                        f.write(f"  Anomaly - Mean: {anomaly_stats['mean']:.2f}, "
                               f"Std: {anomaly_stats['std']:.2f}, "
                               f"Max: {anomaly_stats['max']:.2f}\n")
                        
                        # Calculate change percentage
                        change = ((anomaly_stats['mean'] - normal_stats['mean']) / 
                                 normal_stats['mean'] * 100)
                        f.write(f"  Change: {change:+.1f}%\n\n")
            
            # Business Impact
            f.write("ESTIMATED BUSINESS IMPACT\n")
            f.write("-"*80 + "\n")
            
            # Get costs from config
            cost_downtime = self.config.get('business.cost_unplanned_downtime', 5000)
            
            if not failure_summary.empty:
                total_failure_hours = failure_summary['duration_days'].sum() * 24
                potential_downtime_cost = total_failure_hours * cost_downtime
                
                f.write(f"Total Failure Duration: {total_failure_hours:.0f} hours\n")
                f.write(f"Potential Downtime Cost: ${potential_downtime_cost:,.2f}\n")
                f.write(f"\nNote: Early detection could reduce downtime by 40-60%\n")
                f.write(f"Potential Savings: ${potential_downtime_cost * 0.5:,.2f}\n")
            else:
                f.write("No failures detected - system operating nominally\n")
            
            f.write("\n" + "="*80 + "\n")
        
        self.logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def generate_full_report(
        self,
        df: pd.DataFrame,
        validation_results: Dict,
        failure_summary: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Generate complete analysis package
        
        Args:
            df: DataFrame with sensor data
            validation_results: Validation results
            failure_summary: Failure events summary
            
        Returns:
            Dictionary with paths to generated files
        """
        self.logger.info("Generating full analysis report package")
        
        # Generate text reports
        quality_report = self.generate_data_quality_report(df, validation_results)
        anomaly_report = self.generate_anomaly_report(df, failure_summary)
        
        # Generate visualizations
        self.plotter.plot_sensor_timeseries(df, save=True)
        self.plotter.plot_correlation_matrix(df, save=True)
        self.plotter.plot_distributions(df, save=True)
        self.plotter.plot_boxplots(df, save=True)
        
        if not failure_summary.empty:
            self.plotter.plot_failure_timeline(failure_summary, save=True)
        
        report_files = {
            'data_quality_report': quality_report,
            'anomaly_report': anomaly_report,
            'figures': str(self.plotter.figure_path)
        }
        
        self.logger.info("Full report package generated successfully")
        
        return report_files


# Example usage
if __name__ == "__main__":
    from ..config import get_config
    from ..data.data_loader import DataLoader
    from ..data.data_validator import DataValidator
    from ..data.sensor_simulator import SensorSimulator
    
    # Load configuration
    config = get_config()
    
    # Load data
    loader = DataLoader(config)
    df = loader.load_machine_data("machine_001")
    
    # Validate data
    validator = DataValidator()
    validation_results = validator.validate(df)
    
    # Get failure summary
    simulator = SensorSimulator(config)
    # Assuming we have failure_summary from generation
    failure_summary = pd.DataFrame()  # Replace with actual data
    
    # Generate reports
    report_gen = ReportGenerator(config)
    reports = report_gen.generate_full_report(df, validation_results, failure_summary)
    
    print("\n✓ Reports generated successfully:")
    for report_type, path in reports.items():
        print(f"  {report_type}: {path}")