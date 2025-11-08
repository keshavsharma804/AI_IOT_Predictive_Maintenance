"""
Time-Domain Feature Extraction
Extract statistical and signal characteristics from time-series sensor data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin


class TimeDomainFeatures(LoggerMixin):
    """
    Extract time-domain features from sensor signals
    
    Features include:
    - Basic statistics (mean, std, min, max, median)
    - Statistical moments (skewness, kurtosis)
    - Signal characteristics (RMS, peak-to-peak, crest factor)
    - Shape factors (impulse, clearance, shape)
    - Energy metrics
    - Zero-crossing rate
    - Peak analysis
    """
    
    def __init__(self):
        """Initialize time-domain feature extractor"""
        self.logger.info("TimeDomainFeatures initialized")
    
    def extract_all_features(
        self, 
        signal: Union[np.ndarray, pd.Series],
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract all time-domain features from a signal
        
        Args:
            signal: Input signal (1D array)
            prefix: Prefix for feature names (e.g., 'vibration_')
            
        Returns:
            Dictionary of feature name: value pairs
        """
        # Convert to numpy array if pandas Series
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        # Remove NaN values
        signal = signal[~np.isnan(signal)]
        
        if len(signal) == 0:
            self.logger.warning("Empty signal after removing NaN")
            return {}
        
        features = {}
        
        # Basic statistics
        features.update(self._basic_statistics(signal, prefix))
        
        # Statistical moments
        features.update(self._statistical_moments(signal, prefix))
        
        # Signal characteristics
        features.update(self._signal_characteristics(signal, prefix))
        
        # Shape factors
        features.update(self._shape_factors(signal, prefix))
        
        # Energy and power
        features.update(self._energy_features(signal, prefix))
        
        # Zero-crossing
        features.update(self._zero_crossing_features(signal, prefix))
        
        # Peak analysis
        features.update(self._peak_features(signal, prefix))
        
        # Percentiles
        features.update(self._percentile_features(signal, prefix))
        
        return features
    
    def _basic_statistics(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract basic statistical features
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {
            f'{prefix}mean': float(np.mean(signal)),
            f'{prefix}std': float(np.std(signal)),
            f'{prefix}var': float(np.var(signal)),
            f'{prefix}min': float(np.min(signal)),
            f'{prefix}max': float(np.max(signal)),
            f'{prefix}median': float(np.median(signal)),
            f'{prefix}range': float(np.ptp(signal)),  # peak-to-peak
            f'{prefix}iqr': float(np.percentile(signal, 75) - np.percentile(signal, 25))
        }
        
        return features
    
    def _statistical_moments(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract higher-order statistical moments
        
        Skewness: Measure of asymmetry (0 = symmetric)
        Kurtosis: Measure of tailedness (3 = normal distribution)
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {
            f'{prefix}skewness': float(stats.skew(signal)),
            f'{prefix}kurtosis': float(stats.kurtosis(signal)),
            f'{prefix}moment_3': float(stats.moment(signal, moment=3)),
            f'{prefix}moment_4': float(stats.moment(signal, moment=4)),
            f'{prefix}moment_5': float(stats.moment(signal, moment=5))
        }
        
        return features
    
    def _signal_characteristics(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract signal characteristic features
        
        RMS: Root Mean Square - overall energy level
        Peak-to-Peak: Maximum amplitude variation
        Crest Factor: Peak / RMS - indicates impulsiveness
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(signal**2))
        
        # Peak value (absolute)
        peak = np.max(np.abs(signal))
        
        # Peak-to-peak
        peak_to_peak = np.ptp(signal)
        
        # Crest factor (Peak / RMS)
        crest_factor = peak / (rms + 1e-10)
        
        # Mean absolute value
        mean_abs = np.mean(np.abs(signal))
        
        # Mean absolute deviation
        mad = np.mean(np.abs(signal - np.mean(signal)))
        
        features = {
            f'{prefix}rms': float(rms),
            f'{prefix}peak': float(peak),
            f'{prefix}peak_to_peak': float(peak_to_peak),
            f'{prefix}crest_factor': float(crest_factor),
            f'{prefix}mean_absolute': float(mean_abs),
            f'{prefix}mean_abs_deviation': float(mad)
        }
        
        return features
    
    def _shape_factors(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract shape factor features
        
        These are commonly used in vibration analysis for bearing diagnostics
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        # RMS
        rms = np.sqrt(np.mean(signal**2))
        
        # Mean absolute value
        mean_abs = np.mean(np.abs(signal))
        
        # Peak value
        peak = np.max(np.abs(signal))
        
        # Shape factor (RMS / Mean Absolute)
        shape_factor = rms / (mean_abs + 1e-10)
        
        # Impulse factor (Peak / Mean Absolute)
        impulse_factor = peak / (mean_abs + 1e-10)
        
        # Clearance factor (Peak / RMS of sqrt of absolute)
        sqrt_mean_abs = np.mean(np.sqrt(np.abs(signal)))
        clearance_factor = peak / ((sqrt_mean_abs**2) + 1e-10)
        
        features = {
            f'{prefix}shape_factor': float(shape_factor),
            f'{prefix}impulse_factor': float(impulse_factor),
            f'{prefix}clearance_factor': float(clearance_factor)
        }
        
        return features
    
    def _energy_features(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract energy-based features
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        # Total energy
        energy = np.sum(signal**2)
        
        # Average power
        power = np.mean(signal**2)
        
        # Log energy
        log_energy = np.log(energy + 1e-10)
        
        # Teager-Kaiser energy operator (approximation)
        if len(signal) > 2:
            teager = signal[1:-1]**2 - signal[:-2] * signal[2:]
            teager_energy = np.sum(teager)
        else:
            teager_energy = 0
        
        features = {
            f'{prefix}energy': float(energy),
            f'{prefix}power': float(power),
            f'{prefix}log_energy': float(log_energy),
            f'{prefix}teager_energy': float(teager_energy)
        }
        
        return features
    
    def _zero_crossing_features(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract zero-crossing features
        
        Zero-crossing rate indicates frequency content
        High rate = high frequency, Low rate = low frequency
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        zcr = len(zero_crossings) / len(signal)
        
        # Mean-crossing rate (crossing the mean)
        mean_signal = np.mean(signal)
        mean_crossings = np.where(np.diff(np.sign(signal - mean_signal)))[0]
        mcr = len(mean_crossings) / len(signal)
        
        features = {
            f'{prefix}zero_crossing_rate': float(zcr),
            f'{prefix}mean_crossing_rate': float(mcr)
        }
        
        return features
    
    def _peak_features(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract peak-related features
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        # Find peaks
        peaks, properties = find_peaks(signal, height=np.mean(signal))
        
        # Number of peaks
        num_peaks = len(peaks)
        
        # Peak density
        peak_density = num_peaks / len(signal)
        
        # Average peak height
        if num_peaks > 0:
            avg_peak_height = np.mean(properties['peak_heights'])
            max_peak_height = np.max(properties['peak_heights'])
            std_peak_height = np.std(properties['peak_heights'])
        else:
            avg_peak_height = 0
            max_peak_height = 0
            std_peak_height = 0
        
        features = {
            f'{prefix}num_peaks': float(num_peaks),
            f'{prefix}peak_density': float(peak_density),
            f'{prefix}avg_peak_height': float(avg_peak_height),
            f'{prefix}max_peak_height': float(max_peak_height),
            f'{prefix}std_peak_height': float(std_peak_height)
        }
        
        return features
    
    def _percentile_features(
        self, 
        signal: np.ndarray, 
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract percentile features
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        features = {}
        
        for p in percentiles:
            features[f'{prefix}percentile_{p}'] = float(np.percentile(signal, p))
        
        return features


class RollingWindowFeatures(LoggerMixin):
    """
    Extract rolling window features for time-series data
    
    Features:
    - Rolling mean, std, min, max
    - Exponentially weighted moving average (EWMA)
    - Rate of change
    - Momentum
    """
    
    def __init__(self):
        """Initialize rolling window feature extractor"""
        self.logger.info("RollingWindowFeatures initialized")
    
    def extract_rolling_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        windows: List[int] = [5, 30, 300]
    ) -> pd.DataFrame:
        """
        Extract rolling window features for multiple sensors
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            windows: List of window sizes (in samples)
            
        Returns:
            DataFrame with rolling features added
        """
        self.logger.info(f"Extracting rolling features for {len(sensor_cols)} sensors, "
                        f"windows: {windows}")
        
        df_features = df.copy()
        
        for col in sensor_cols:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found, skipping")
                continue
            
            for window in windows:
                # Rolling mean
                df_features[f'{col}_rolling_mean_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                df_features[f'{col}_rolling_std_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min
                df_features[f'{col}_rolling_min_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                
                # Rolling max
                df_features[f'{col}_rolling_max_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )
                
                # Rolling range
                df_features[f'{col}_rolling_range_{window}'] = (
                    df_features[f'{col}_rolling_max_{window}'] - 
                    df_features[f'{col}_rolling_min_{window}']
                )
            
            # Exponentially weighted moving average (EWMA)
            for span in [10, 50, 100]:
                df_features[f'{col}_ewma_{span}'] = (
                    df[col].ewm(span=span).mean()
                )
            
            # Rate of change
            for lag in [1, 5, 10]:
                df_features[f'{col}_roc_{lag}'] = df[col].diff(lag)
                df_features[f'{col}_pct_change_{lag}'] = df[col].pct_change(lag)
            
            # Momentum (difference from moving average)
            for window in [10, 30]:
                ma = df[col].rolling(window=window, min_periods=1).mean()
                df_features[f'{col}_momentum_{window}'] = df[col] - ma
        
        # Fill NaN values from rolling operations
        df_features = df_features.fillna(method='bfill').fillna(method='ffill')
        
        self.logger.info(f"Added {len(df_features.columns) - len(df.columns)} rolling features")
        
        return df_features


class AggregatedFeatures(LoggerMixin):
    """
    Extract aggregated features over time windows
    
    Features aggregated over fixed time periods (e.g., 1 min, 5 min, 1 hour)
    """
    
    def __init__(self):
        """Initialize aggregated feature extractor"""
        self.logger.info("AggregatedFeatures initialized")
    
    def extract_aggregated_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        time_windows: List[str] = ['1min', '5min', '1H'],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract aggregated features over time windows
        
        Args:
            df: DataFrame with timestamp and sensor data
            sensor_cols: List of sensor columns
            time_windows: List of time window strings (pandas resample format)
            group_by: Optional column to group by (e.g., 'machine_id')
            
        Returns:
            DataFrame with aggregated features
        """
        if 'timestamp' not in df.columns:
            self.logger.error("DataFrame must have 'timestamp' column")
            return df
        
        self.logger.info(f"Extracting aggregated features for windows: {time_windows}")
        
        # Set timestamp as index
        df_temp = df.set_index('timestamp')
        
        all_features = []
        
        for window in time_windows:
            self.logger.info(f"Processing window: {window}")
            
            if group_by and group_by in df.columns:
                # Group by machine/device first, then resample
                grouped = df_temp.groupby(group_by)
                
                for name, group in grouped:
                    resampled = group[sensor_cols].resample(window).agg([
                        'mean', 'std', 'min', 'max', 'median'
                    ])
                    
                    # Flatten column names
                    resampled.columns = [
                        f'{col}_{agg}_{window}' 
                        for col, agg in resampled.columns
                    ]
                    
                    resampled[group_by] = name
                    all_features.append(resampled)
            else:
                # Simple resample
                resampled = df_temp[sensor_cols].resample(window).agg([
                    'mean', 'std', 'min', 'max', 'median'
                ])
                
                # Flatten column names
                resampled.columns = [
                    f'{col}_{agg}_{window}' 
                    for col, agg in resampled.columns
                ]
                
                all_features.append(resampled)
        
        # Combine all features
        if all_features:
            df_aggregated = pd.concat(all_features, axis=1)
            df_aggregated = df_aggregated.reset_index()
            
            self.logger.info(f"Created {len(df_aggregated.columns)} aggregated features")
            
            return df_aggregated
        else:
            return df


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.config import get_config
    from src.data.data_loader import DataLoader
    
    # Load configuration
    config = get_config()
    
    # Load sample data
    loader = DataLoader(config)
    df = loader.load_machine_data("machine_001")
    
    print("\n" + "="*80)
    print("TIME-DOMAIN FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Test 1: Extract features from single signal
    print("\nTest 1: Single Signal Features")
    print("-"*80)
    
    extractor = TimeDomainFeatures()
    
    # Extract features from vibration signal (first 1000 samples)
    signal = df['vibration_rms'].iloc[:1000].values
    features = extractor.extract_all_features(signal, prefix='vib_')
    
    print(f"Extracted {len(features)} features:")
    for name, value in list(features.items())[:10]:  # Show first 10
        print(f"  {name}: {value:.4f}")
    print(f"  ... and {len(features) - 10} more features")
    
    # Test 2: Rolling window features
    print("\n\nTest 2: Rolling Window Features")
    print("-"*80)
    
    rolling_extractor = RollingWindowFeatures()
    
    # Extract rolling features (use small subset for speed)
    df_sample = df.iloc[:10000].copy()
    df_rolling = rolling_extractor.extract_rolling_features(
        df_sample,
        sensor_cols=['vibration_rms', 'temperature'],
        windows=[10, 50, 100]
    )
    
    new_cols = [col for col in df_rolling.columns if col not in df_sample.columns]
    print(f"Added {len(new_cols)} rolling features")
    print(f"Examples: {new_cols[:5]}")
    
    # Test 3: Feature extraction for multiple signals
    print("\n\nTest 3: Multi-Sensor Feature Extraction")
    print("-"*80)
    
    sensor_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
    all_features_dict = {}
    
    for col in sensor_cols:
        signal = df[col].iloc[:1000].values
        features = extractor.extract_all_features(signal, prefix=f'{col}_')
        all_features_dict.update(features)
    
    print(f"Total features extracted: {len(all_features_dict)}")
    print(f"Features per sensor: {len(all_features_dict) // len(sensor_cols)}")
    
    # Test 4: Create feature DataFrame
    print("\n\nTest 4: Creating Feature DataFrame")
    print("-"*80)
    
    features_df = pd.DataFrame([all_features_dict])
    print(f"Feature DataFrame shape: {features_df.shape}")
    print(f"\nSample features:")
    print(features_df.iloc[0, :5])
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)