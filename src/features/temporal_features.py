"""
Temporal Feature Extraction
Extract time-dependent features including lags, trends, seasonality, and correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats, signal
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin


class TemporalFeatures(LoggerMixin):
    """
    Extract temporal features from time-series sensor data
    
    Features include:
    - Lag features (past values)
    - Rate of change
    - Trend analysis (linear, polynomial)
    - Acceleration (second derivative)
    - Autocorrelation
    - Cross-correlation between sensors
    - Time since last anomaly
    - Cumulative statistics
    - Seasonality indicators
    """
    
    def __init__(self, sampling_rate: float = 100):
        """
        Initialize temporal feature extractor
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.logger.info(f"TemporalFeatures initialized (fs={sampling_rate}Hz)")
    
    def extract_lag_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        lags: List[int],
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract lag features (past values)
        
        Args:
            signal: Input signal
            lags: List of lag periods (in samples)
            prefix: Feature name prefix
            
        Returns:
            Dictionary of lag features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        features = {}
        
        for lag in lags:
            if lag < len(signal):
                # Lag value
                features[f'{prefix}lag_{lag}'] = float(signal[-lag])
                
                # Change from lag
                current_value = signal[-1]
                lag_value = signal[-lag]
                features[f'{prefix}change_from_lag_{lag}'] = float(current_value - lag_value)
                
                # Percent change from lag
                if lag_value != 0:
                    features[f'{prefix}pct_change_from_lag_{lag}'] = float(
                        (current_value - lag_value) / lag_value * 100
                    )
                else:
                    features[f'{prefix}pct_change_from_lag_{lag}'] = 0.0
        
        return features
    
    def extract_rate_of_change_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract rate of change features
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of rate features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        features = {}
        
        # First derivative (velocity)
        first_deriv = np.diff(signal)
        features[f'{prefix}velocity_mean'] = float(np.mean(first_deriv))
        features[f'{prefix}velocity_std'] = float(np.std(first_deriv))
        features[f'{prefix}velocity_max'] = float(np.max(np.abs(first_deriv)))
        
        # Second derivative (acceleration)
        if len(first_deriv) > 1:
            second_deriv = np.diff(first_deriv)
            features[f'{prefix}acceleration_mean'] = float(np.mean(second_deriv))
            features[f'{prefix}acceleration_std'] = float(np.std(second_deriv))
            features[f'{prefix}acceleration_max'] = float(np.max(np.abs(second_deriv)))
        
        # Rate of change in different windows
        for window in [10, 50, 100]:
            if len(signal) > window:
                recent = signal[-window:]
                older = signal[-2*window:-window] if len(signal) > 2*window else signal[:window]
                
                change = np.mean(recent) - np.mean(older)
                features[f'{prefix}roc_window_{window}'] = float(change)
        
        return features
    
    def extract_trend_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract trend analysis features
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of trend features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        features = {}
        
        # Remove NaN values
        signal = signal[~np.isnan(signal)]
        
        if len(signal) < 2:
            return features
        
        # Time axis
        x = np.arange(len(signal))
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = linregress(x, signal)
        
        features[f'{prefix}trend_slope'] = float(slope)
        features[f'{prefix}trend_intercept'] = float(intercept)
        features[f'{prefix}trend_r_squared'] = float(r_value ** 2)
        features[f'{prefix}trend_p_value'] = float(p_value)
        features[f'{prefix}trend_std_error'] = float(std_err)
        
        # Trend strength (normalized slope)
        signal_range = np.ptp(signal)
        if signal_range > 0:
            features[f'{prefix}trend_strength'] = float(slope * len(signal) / signal_range)
        else:
            features[f'{prefix}trend_strength'] = 0.0
        
        # Polynomial trend (degree 2)
        if len(signal) > 3:
            poly_coef = np.polyfit(x, signal, 2)
            features[f'{prefix}poly_coef_0'] = float(poly_coef[0])  # Quadratic term
            features[f'{prefix}poly_coef_1'] = float(poly_coef[1])  # Linear term
            features[f'{prefix}poly_coef_2'] = float(poly_coef[2])  # Constant term
            
            # Curvature (second derivative of polynomial)
            features[f'{prefix}curvature'] = float(2 * poly_coef[0])
        
        # Trend direction
        features[f'{prefix}trend_direction'] = float(np.sign(slope))
        
        # Monotonicity (percentage of time series that is monotonic)
        diffs = np.diff(signal)
        increasing = np.sum(diffs > 0) / len(diffs)
        decreasing = np.sum(diffs < 0) / len(diffs)
        features[f'{prefix}monotonicity_increasing'] = float(increasing)
        features[f'{prefix}monotonicity_decreasing'] = float(decreasing)
        
        return features
    
    def extract_autocorrelation_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        max_lag: int = 100,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract autocorrelation features
        
        Args:
            signal: Input signal
            max_lag: Maximum lag for autocorrelation
            prefix: Feature name prefix
            
        Returns:
            Dictionary of autocorrelation features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        # Remove NaN
        signal = signal[~np.isnan(signal)]
        
        features = {}
        
        if len(signal) < max_lag:
            return features
        
        # Normalize signal
        signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Compute autocorrelation for specific lags
        lags = [1, 5, 10, 20, 50, 100]
        
        for lag in lags:
            if lag < len(signal_normalized):
                # Autocorrelation at this lag
                acf = np.corrcoef(
                    signal_normalized[:-lag],
                    signal_normalized[lag:]
                )[0, 1]
                
                features[f'{prefix}autocorr_lag_{lag}'] = float(acf)
        
        # Find first zero crossing of autocorrelation
        acf_values = []
        for lag in range(1, min(max_lag, len(signal_normalized))):
            acf = np.corrcoef(
                signal_normalized[:-lag],
                signal_normalized[lag:]
            )[0, 1]
            acf_values.append(acf)
        
        acf_values = np.array(acf_values)
        
        # First lag where ACF crosses zero
        zero_crossings = np.where(np.diff(np.sign(acf_values)))[0]
        if len(zero_crossings) > 0:
            features[f'{prefix}acf_first_zero_crossing'] = float(zero_crossings[0] + 1)
        else:
            features[f'{prefix}acf_first_zero_crossing'] = float(max_lag)
        
        # Maximum autocorrelation value (excluding lag 0)
        if len(acf_values) > 0:
            features[f'{prefix}acf_max'] = float(np.max(np.abs(acf_values)))
            features[f'{prefix}acf_mean'] = float(np.mean(acf_values))
        
        return features
    
    def extract_cumulative_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract cumulative statistics
        
        Args:
            signal: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of cumulative features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        features = {}
        
        # Cumulative sum
        cumsum = np.cumsum(signal)
        features[f'{prefix}cumsum'] = float(cumsum[-1])
        features[f'{prefix}cumsum_mean'] = float(np.mean(cumsum))
        features[f'{prefix}cumsum_std'] = float(np.std(cumsum))
        
        # Cumulative product (for small signals)
        # Use log to prevent overflow
        log_signal = np.log(np.abs(signal) + 1e-10)
        log_cumprod = np.cumsum(log_signal)
        features[f'{prefix}log_cumprod'] = float(log_cumprod[-1])
        
        # Cumulative min/max
        cummin = np.minimum.accumulate(signal)
        cummax = np.maximum.accumulate(signal)
        features[f'{prefix}cummin'] = float(cummin[-1])
        features[f'{prefix}cummax'] = float(cummax[-1])
        features[f'{prefix}cumrange'] = float(cummax[-1] - cummin[-1])
        
        # Range expansion (how much range has grown)
        initial_range = np.ptp(signal[:len(signal)//2])
        final_range = np.ptp(signal[len(signal)//2:])
        if initial_range > 0:
            features[f'{prefix}range_expansion'] = float(final_range / initial_range)
        else:
            features[f'{prefix}range_expansion'] = 1.0
        
        return features
    
    def extract_seasonality_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        timestamps: Optional[pd.DatetimeIndex] = None,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract seasonality indicators
        
        Args:
            signal: Input signal
            timestamps: Optional timestamps for time-based seasonality
            prefix: Feature name prefix
            
        Returns:
            Dictionary of seasonality features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        features = {}
        
        # If timestamps provided, extract time-based features
        if timestamps is not None:
            # Hour of day effect
            if len(timestamps) > 0:
                hours = timestamps.hour
                unique_hours = np.unique(hours)
                
                if len(unique_hours) > 1:
                    # Group by hour and calculate variance
                    hour_means = []
                    for hour in unique_hours:
                        hour_mask = hours == hour
                        hour_means.append(np.mean(signal[hour_mask]))
                    
                    features[f'{prefix}hourly_variance'] = float(np.var(hour_means))
                
                # Day of week effect
                days = timestamps.dayofweek
                unique_days = np.unique(days)
                
                if len(unique_days) > 1:
                    day_means = []
                    for day in unique_days:
                        day_mask = days == day
                        day_means.append(np.mean(signal[day_mask]))
                    
                    features[f'{prefix}daily_variance'] = float(np.var(day_means))
        
        # Periodic component detection using FFT
        n = len(signal)
        if n > 10:
            # Remove trend
            detrended = signal - np.linspace(signal[0], signal[-1], n)
            
            # FFT
            fft_vals = np.fft.fft(detrended)
            power = np.abs(fft_vals[:n//2]) ** 2
            freqs = np.fft.fftfreq(n, 1/self.sampling_rate)[:n//2]
            
            # Find dominant periodic component
            if len(power) > 0:
                dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
                dominant_period = 1 / (freqs[dominant_idx] + 1e-10)
                dominant_power = power[dominant_idx]
                
                features[f'{prefix}dominant_period'] = float(dominant_period)
                features[f'{prefix}periodic_strength'] = float(
                    dominant_power / (np.sum(power) + 1e-10)
                )
        
        return features
    
    def extract_cross_correlation_features(
        self,
        signal1: Union[np.ndarray, pd.Series],
        signal2: Union[np.ndarray, pd.Series],
        max_lag: int = 50,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract cross-correlation between two signals
        
        Args:
            signal1: First signal
            signal2: Second signal
            max_lag: Maximum lag for cross-correlation
            prefix: Feature name prefix
            
        Returns:
            Dictionary of cross-correlation features
        """
        if isinstance(signal1, pd.Series):
            signal1 = signal1.values
        if isinstance(signal2, pd.Series):
            signal2 = signal2.values
        
        features = {}
        
        # Ensure same length
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
        
        if min_len < 2:
            return features
        
        # Normalize signals
        signal1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        signal2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)
        
        # Cross-correlation at zero lag
        features[f'{prefix}xcorr_zero_lag'] = float(np.corrcoef(signal1, signal2)[0, 1])
        
        # Cross-correlation at specific lags
        lags = [1, 5, 10, 20]
        for lag in lags:
            if lag < len(signal1_norm):
                # Positive lag (signal2 leads signal1)
                xcorr_pos = np.corrcoef(
                    signal1_norm[lag:],
                    signal2_norm[:-lag]
                )[0, 1]
                features[f'{prefix}xcorr_lag_pos_{lag}'] = float(xcorr_pos)
                
                # Negative lag (signal1 leads signal2)
                xcorr_neg = np.corrcoef(
                    signal1_norm[:-lag],
                    signal2_norm[lag:]
                )[0, 1]
                features[f'{prefix}xcorr_lag_neg_{lag}'] = float(xcorr_neg)
        
        # Maximum cross-correlation and lag at which it occurs
        max_lag = min(max_lag, len(signal1_norm) - 1)
        xcorr_values = []
        lags_tested = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                xcorr = np.corrcoef(signal1_norm, signal2_norm)[0, 1]
            elif lag > 0:
                xcorr = np.corrcoef(signal1_norm[lag:], signal2_norm[:-lag])[0, 1]
            else:
                xcorr = np.corrcoef(signal1_norm[:lag], signal2_norm[-lag:])[0, 1]
            
            xcorr_values.append(xcorr)
            lags_tested.append(lag)
        
        xcorr_values = np.array(xcorr_values)
        max_xcorr_idx = np.argmax(np.abs(xcorr_values))
        
        features[f'{prefix}xcorr_max'] = float(xcorr_values[max_xcorr_idx])
        features[f'{prefix}xcorr_max_lag'] = float(lags_tested[max_xcorr_idx])
        
        return features
    
    def extract_degradation_features(
        self,
        signal: Union[np.ndarray, pd.Series],
        baseline_signal: Optional[np.ndarray] = None,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Extract degradation indicators
        
        Args:
            signal: Current signal
            baseline_signal: Baseline (healthy) signal for comparison
            prefix: Feature name prefix
            
        Returns:
            Dictionary of degradation features
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        features = {}
        
        # Trend-based degradation
        x = np.arange(len(signal))
        slope, _, _, _, _ = linregress(x, signal)
        
        # Positive slope might indicate degradation (e.g., increasing vibration)
        features[f'{prefix}degradation_trend'] = float(slope)
        
        # Variance growth (increasing noise/instability)
        first_half_var = np.var(signal[:len(signal)//2])
        second_half_var = np.var(signal[len(signal)//2:])
        
        if first_half_var > 0:
            features[f'{prefix}variance_growth'] = float(
                (second_half_var - first_half_var) / first_half_var
            )
        else:
            features[f'{prefix}variance_growth'] = 0.0
        
        # Mean shift
        first_half_mean = np.mean(signal[:len(signal)//2])
        second_half_mean = np.mean(signal[len(signal)//2:])
        features[f'{prefix}mean_shift'] = float(second_half_mean - first_half_mean)
        
        # If baseline provided, compare
        if baseline_signal is not None:
            # Distance from baseline
            features[f'{prefix}baseline_deviation'] = float(
                np.mean(np.abs(signal - baseline_signal[:len(signal)]))
            )
            
            # Correlation with baseline
            if len(baseline_signal) >= len(signal):
                features[f'{prefix}baseline_correlation'] = float(
                    np.corrcoef(signal, baseline_signal[:len(signal)])[0, 1]
                )
        
        # Number of threshold exceedances (anomaly count proxy)
        threshold = np.mean(signal) + 2 * np.std(signal)
        exceedances = np.sum(signal > threshold)
        features[f'{prefix}threshold_exceedances'] = float(exceedances)
        features[f'{prefix}exceedance_rate'] = float(exceedances / len(signal))
        
        return features


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.config import get_config
    from src.data.data_loader import DataLoader
    
    # Load configuration
    config = get_config()
    sampling_rate = config.get('data_generation.sampling_rate', 100)
    
    # Load sample data
    loader = DataLoader(config)
    df = loader.load_machine_data("machine_001")
    
    print("\n" + "="*80)
    print("TEMPORAL FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Test 1: Lag features
    print("\nTest 1: Lag Features")
    print("-"*80)
    
    extractor = TemporalFeatures(sampling_rate=sampling_rate)
    
    signal = df['vibration_rms'].iloc[:1000].values
    
    lag_features = extractor.extract_lag_features(
        signal,
        lags=[1, 10, 50, 100],
        prefix='vib_'
    )
    
    print(f"Extracted {len(lag_features)} lag features")
    for key, value in list(lag_features.items())[:5]:
        print(f"  {key}: {value:.4f}")
    
    # Test 2: Rate of change
    print("\n\nTest 2: Rate of Change Features")
    print("-"*80)
    
    roc_features = extractor.extract_rate_of_change_features(signal, prefix='vib_')
    print(f"Extracted {len(roc_features)} rate of change features")
    for key, value in roc_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 3: Trend analysis
    print("\n\nTest 3: Trend Features")
    print("-"*80)
    
    trend_features = extractor.extract_trend_features(signal, prefix='vib_')
    print(f"Extracted {len(trend_features)} trend features")
    for key, value in list(trend_features.items())[:8]:
        print(f"  {key}: {value:.4f}")
    
    # Test 4: Autocorrelation
    print("\n\nTest 4: Autocorrelation Features")
    print("-"*80)
    
    acf_features = extractor.extract_autocorrelation_features(signal, prefix='vib_')
    print(f"Extracted {len(acf_features)} autocorrelation features")
    for key, value in acf_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 5: Cross-correlation
    print("\n\nTest 5: Cross-Correlation Features")
    print("-"*80)
    
    signal1 = df['vibration_rms'].iloc[:1000].values
    signal2 = df['temperature'].iloc[:1000].values
    
    xcorr_features = extractor.extract_cross_correlation_features(
        signal1, signal2, prefix='vib_temp_'
    )
    print(f"Extracted {len(xcorr_features)} cross-correlation features")
    for key, value in xcorr_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 6: All temporal features
    print("\n\nTest 6: Complete Temporal Feature Set")
    print("-"*80)
    
    all_features = {}
    all_features.update(lag_features)
    all_features.update(roc_features)
    all_features.update(trend_features)
    all_features.update(acf_features)
    all_features.update(extractor.extract_cumulative_features(signal, prefix='vib_'))
    all_features.update(extractor.extract_degradation_features(signal, prefix='vib_'))
    
    print(f"\nTotal temporal features: {len(all_features)}")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)