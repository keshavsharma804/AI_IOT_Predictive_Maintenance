"""
Feature Engineering Pipeline
Orchestrates extraction of all features from raw sensor data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin
from ..config import get_config
from .time_domain import TimeDomainFeatures, RollingWindowFeatures


class FeatureEngineer(LoggerMixin):
    """
    Complete feature engineering pipeline
    
    Orchestrates extraction of:
    - Time-domain features
    - Rolling window features
    - Aggregated features
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        
        # Initialize feature extractors
        self.time_domain = TimeDomainFeatures()
        self.rolling_window = RollingWindowFeatures()
        
        self.logger.info("FeatureEngineer initialized")
    
    def engineer_complete_features(
        self,
        df: pd.DataFrame,
        window_size: int = 1000,
        stride: int = 500,
        sensor_cols: List[str] = None,
        include_rolling: bool = false,
        include_frequency: bool = false,
        include_temporal: bool = false,
        bearing_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Extract ALL features: time-domain, frequency-domain, and temporal
        
        Args:
            df: DataFrame with raw sensor data
            window_size: Window size for feature extraction
            stride: Stride for sliding window
            sensor_cols: List of sensor columns
            include_rolling: Include rolling window features
            include_frequency: Include frequency-domain features
            include_temporal: Include temporal features
            bearing_params: Bearing parameters for vibration analysis
            
        Returns:
            DataFrame with complete feature set
        """
        # Import extractors
        from .frequency_domain import FrequencyDomainFeatures
        from .temporal_features import TemporalFeatures
        
        if sensor_cols is None:
            sensor_cols = ['vibration_x', 'vibration_y', 'vibration_z',
                          'vibration_rms', 'temperature', 'pressure', 'current']
        
        available_cols = [col for col in sensor_cols if col in df.columns]
        
        self.logger.info("="*80)
        self.logger.info("COMPLETE FEATURE ENGINEERING PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Sensors: {len(available_cols)}")
        self.logger.info(f"Window: {window_size}, Stride: {stride}")
        self.logger.info(f"Time-domain: ✓")
        self.logger.info(f"Frequency-domain: {'✓' if include_frequency else '✗'}")
        self.logger.info(f"Temporal: {'✓' if include_temporal else '✗'}")
        self.logger.info(f"Rolling: {'✓' if include_rolling else '✗'}")
        
        # Initialize extractors
        sampling_rate = self.config.get('data_generation.sampling_rate', 100)
        
        if include_frequency:
            freq_extractor = FrequencyDomainFeatures(sampling_rate=sampling_rate)
        
        if include_temporal:
            temporal_extractor = TemporalFeatures(sampling_rate=sampling_rate)
        
        feature_list = []
        num_windows = (len(df) - window_size) // stride + 1
        
        self.logger.info(f"Processing {num_windows} windows...")
        
        for i in range(0, len(df) - window_size + 1, stride):
            window_data = df.iloc[i:i+window_size]
            window_features = {}
            
            # Metadata
            window_features['window_id'] = i // stride
            window_features['window_start'] = window_data['timestamp'].iloc[0]
            window_features['window_end'] = window_data['timestamp'].iloc[-1]
            
            if 'machine_id' in df.columns:
                window_features['machine_id'] = window_data['machine_id'].iloc[0]
            
            # Extract features for each sensor
            for col in available_cols:
                signal = window_data[col].values
                
                # 1. Time-domain features
                time_features = self.time_domain.extract_all_features(
                    signal, prefix=f'{col}_'
                )
                window_features.update(time_features)
                
                # 2. Frequency-domain features (vibration only)
                if include_frequency and 'vibration' in col.lower():
                    freq_features = freq_extractor.extract_all_features(
                        signal,
                        prefix=f'{col}_',
                        bearing_params=bearing_params
                    )
                    window_features.update(freq_features)
                
                # 3. Temporal features
                if include_temporal:
                    # Lag features
                    lags = [10, 50, 100]
                    lag_features = temporal_extractor.extract_lag_features(
                        signal, lags=lags, prefix=f'{col}_'
                    )
                    window_features.update(lag_features)
                    
                    # Rate of change
                    roc_features = temporal_extractor.extract_rate_of_change_features(
                        signal, prefix=f'{col}_'
                    )
                    window_features.update(roc_features)
                    
                    # Trend
                    trend_features = temporal_extractor.extract_trend_features(
                        signal, prefix=f'{col}_'
                    )
                    window_features.update(trend_features)
                    
                    # Autocorrelation
                    acf_features = temporal_extractor.extract_autocorrelation_features(
                        signal, max_lag=100, prefix=f'{col}_'
                    )
                    window_features.update(acf_features)
                    
                    # Cumulative
                    cum_features = temporal_extractor.extract_cumulative_features(
                        signal, prefix=f'{col}_'
                    )
                    window_features.update(cum_features)
                    
                    # Degradation
                    deg_features = temporal_extractor.extract_degradation_features(
                        signal, prefix=f'{col}_'
                    )
                    window_features.update(deg_features)
            
            # 4. Cross-correlations between sensors (temporal)
            if include_temporal and len(available_cols) > 1:
                # Vibration vs Temperature
                if 'vibration_rms' in available_cols and 'temperature' in available_cols:
                    xcorr = temporal_extractor.extract_cross_correlation_features(
                        window_data['vibration_rms'].values,
                        window_data['temperature'].values,
                        prefix='vib_temp_'
                    )
                    window_features.update(xcorr)
                
                # Vibration vs Temperature
            if 'vibration_rms' in available_cols and 'temperature' in available_cols:
                xcorr = temporal_extractor.extract_cross_correlation_features(
                    window_data['vibration_rms'].values,
                    window_data['temperature'].values,
                    prefix='vib_temp_'
                )
                window_features.update(xcorr)
            
            # Vibration vs Current
            if 'vibration_rms' in available_cols and 'current' in available_cols:
                xcorr = temporal_extractor.extract_cross_correlation_features(
                    window_data['vibration_rms'].values,
                    window_data['current'].values,
                    prefix='vib_current_'
                )
                window_features.update(xcorr)
            
            # Temperature vs Current
            if 'temperature' in available_cols and 'current' in available_cols:
                xcorr = temporal_extractor.extract_cross_correlation_features(
                    window_data['temperature'].values,
                    window_data['current'].values,
                    prefix='temp_current_'
                )
                window_features.update(xcorr)
        
            # Add labels
            if 'is_anomaly' in df.columns:
                window_features['is_anomaly'] = int(
                    window_data['is_anomaly'].mean() > 0.5
                )
            
            if 'failure_type' in df.columns:
                window_features['failure_type'] = window_data['failure_type'].mode()[0]
            
            if 'severity' in df.columns:
                window_features['severity'] = window_data['severity'].mean()
            
            feature_list.append(window_features)
            
            # Progress
            if (i // stride) % 100 == 0:
                progress = (i // stride) / num_windows * 100
                self.logger.info(f"Progress: {progress:.1f}%")
        
        # Create DataFrame
        features_df = pd.DataFrame(feature_list)
        
        self.logger.info(f"Extracted {len(features_df)} feature windows")
        self.logger.info(f"Features per window: {len(features_df.columns)}")
        
        # Add rolling features
        if include_rolling:
            self.logger.info("Adding rolling window features...")
            
            rolling_cols = [col for col in features_df.columns 
                        if any(sensor in col for sensor in available_cols)]
            
            features_df = self.rolling_window.extract_rolling_features(
                features_df,
                sensor_cols=rolling_cols[:10],
                windows=[3, 5, 10]
            )
        
        self.logger.info("="*80)
        self.logger.info(f"✓ COMPLETE! Total features: {len(features_df.columns)}")
        self.logger.info("="*80)
        
        return features_df
