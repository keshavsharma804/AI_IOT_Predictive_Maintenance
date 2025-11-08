"""
Frequency-Domain Feature Extraction
Extract spectral features using FFT and signal processing techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin


class FrequencyDomainFeatures(LoggerMixin):
    """
    Extract frequency-domain features from sensor signals
    
    Features include:
    - FFT coefficients and statistics
    - Power Spectral Density (PSD)
    - Dominant frequencies
    - Spectral features (centroid, rolloff, flatness, entropy)
    - Band power analysis
    - Bearing-specific frequencies (BPFI, BPFO, BSF, FTF)
    - Harmonic analysis
    """
    
    def __init__(self, sampling_rate: float = 100):
        """
        Initialize frequency-domain feature extractor
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.logger.info(f"FrequencyDomainFeatures initialized (fs={sampling_rate}Hz)")
    
    def extract_all_features(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        prefix: str = '',
        bearing_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Extract all frequency-domain features from a signal
        
        Args:
            signal_data: Input signal (1D array)
            prefix: Prefix for feature names
            bearing_params: Bearing geometry parameters (optional)
                          {'shaft_speed': rpm, 'num_balls': int, etc.}
            
        Returns:
            Dictionary of feature name: value pairs
        """
        # Convert to numpy array
        if isinstance(signal_data, pd.Series):
            signal_data = signal_data.values
        
        # Remove NaN values
        signal_data = signal_data[~np.isnan(signal_data)]
        
        if len(signal_data) == 0:
            self.logger.warning("Empty signal after removing NaN")
            return {}
        
        features = {}
        
        # Compute FFT
        fft_values, frequencies = self._compute_fft(signal_data)
        
        # FFT-based features
        features.update(self._fft_features(fft_values, frequencies, prefix))
        
        # Power Spectral Density
        features.update(self._psd_features(signal_data, prefix))
        
        # Spectral characteristics
        features.update(self._spectral_features(fft_values, frequencies, prefix))
        
        # Band power analysis
        features.update(self._band_power_features(fft_values, frequencies, prefix))
        
        # Harmonic analysis
        features.update(self._harmonic_features(fft_values, frequencies, prefix))
        
        # Bearing-specific features (if parameters provided)
        if bearing_params:
            features.update(self._bearing_features(
                fft_values, frequencies, bearing_params, prefix
            ))
        
        return features
    
    def _compute_fft(
        self,
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of signal
        
        Args:
            signal_data: Input signal
            
        Returns:
            Tuple of (fft_magnitude, frequencies)
        """
        n = len(signal_data)
        
        # Compute FFT
        fft_values = fft(signal_data)
        frequencies = fftfreq(n, 1/self.sampling_rate)
        
        # Get positive frequencies only
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        fft_magnitude = np.abs(fft_values[positive_freq_idx])
        
        return fft_magnitude, frequencies
    
    def _fft_features(
        self,
        fft_magnitude: np.ndarray,
        frequencies: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract basic FFT features
        
        Args:
            fft_magnitude: FFT magnitude spectrum
            frequencies: Frequency array
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic statistics of FFT magnitude
        features[f'{prefix}fft_mean'] = float(np.mean(fft_magnitude))
        features[f'{prefix}fft_std'] = float(np.std(fft_magnitude))
        features[f'{prefix}fft_max'] = float(np.max(fft_magnitude))
        features[f'{prefix}fft_min'] = float(np.min(fft_magnitude))
        features[f'{prefix}fft_range'] = float(np.ptp(fft_magnitude))
        
        # Dominant frequency
        dominant_idx = np.argmax(fft_magnitude)
        features[f'{prefix}dominant_frequency'] = float(frequencies[dominant_idx])
        features[f'{prefix}dominant_magnitude'] = float(fft_magnitude[dominant_idx])
        
        # Frequency of max power
        power = fft_magnitude ** 2
        max_power_idx = np.argmax(power)
        features[f'{prefix}max_power_frequency'] = float(frequencies[max_power_idx])
        
        # Mean frequency (weighted by magnitude)
        features[f'{prefix}mean_frequency'] = float(
            np.sum(frequencies * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10)
        )
        
        # Median frequency
        cumsum_magnitude = np.cumsum(fft_magnitude)
        median_idx = np.searchsorted(cumsum_magnitude, cumsum_magnitude[-1] / 2)
        features[f'{prefix}median_frequency'] = float(frequencies[median_idx])
        
        # FFT coefficient bins (first 10 bins)
        num_bins = min(10, len(fft_magnitude))
        for i in range(num_bins):
            features[f'{prefix}fft_coef_{i}'] = float(fft_magnitude[i])
        
        return features
    
    def _psd_features(
        self,
        signal_data: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract Power Spectral Density features
        
        Args:
            signal_data: Input signal
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        # Compute PSD using Welch's method
        frequencies, psd = welch(
            signal_data,
            fs=self.sampling_rate,
            nperseg=min(256, len(signal_data))
        )
        
        features = {}
        
        # Total power
        features[f'{prefix}total_power'] = float(np.sum(psd))
        
        # Peak power
        features[f'{prefix}peak_power'] = float(np.max(psd))
        
        # Power at dominant frequency
        peak_idx = np.argmax(psd)
        features[f'{prefix}peak_power_frequency'] = float(frequencies[peak_idx])
        
        # Power in specific frequency bands
        # Low: 0-10 Hz, Mid: 10-50 Hz, High: 50+ Hz
        low_band_mask = frequencies < 10
        mid_band_mask = (frequencies >= 10) & (frequencies < 50)
        high_band_mask = frequencies >= 50
        
        features[f'{prefix}power_low_band'] = float(np.sum(psd[low_band_mask]))
        features[f'{prefix}power_mid_band'] = float(np.sum(psd[mid_band_mask]))
        features[f'{prefix}power_high_band'] = float(np.sum(psd[high_band_mask]))
        
        # Power ratios
        total_power = np.sum(psd) + 1e-10
        features[f'{prefix}power_ratio_low'] = float(
            np.sum(psd[low_band_mask]) / total_power
        )
        features[f'{prefix}power_ratio_mid'] = float(
            np.sum(psd[mid_band_mask]) / total_power
        )
        features[f'{prefix}power_ratio_high'] = float(
            np.sum(psd[high_band_mask]) / total_power
        )
        
        return features
    
    def _spectral_features(
        self,
        fft_magnitude: np.ndarray,
        frequencies: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract spectral shape features
        
        Args:
            fft_magnitude: FFT magnitude spectrum
            frequencies: Frequency array
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Normalize magnitude
        magnitude_sum = np.sum(fft_magnitude) + 1e-10
        normalized_magnitude = fft_magnitude / magnitude_sum
        
        # Spectral centroid (center of mass of spectrum)
        features[f'{prefix}spectral_centroid'] = float(
            np.sum(frequencies * fft_magnitude) / magnitude_sum
        )
        
        # Spectral spread (standard deviation around centroid)
        centroid = features[f'{prefix}spectral_centroid']
        features[f'{prefix}spectral_spread'] = float(
            np.sqrt(np.sum(((frequencies - centroid) ** 2) * normalized_magnitude))
        )
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum_magnitude = np.cumsum(fft_magnitude)
        rolloff_threshold = 0.85 * cumsum_magnitude[-1]
        rolloff_idx = np.searchsorted(cumsum_magnitude, rolloff_threshold)
        features[f'{prefix}spectral_rolloff'] = float(frequencies[rolloff_idx])
        
        # Spectral flatness (measure of noisiness vs tonality)
        # Geometric mean / Arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
        arithmetic_mean = np.mean(fft_magnitude)
        features[f'{prefix}spectral_flatness'] = float(
            geometric_mean / (arithmetic_mean + 1e-10)
        )
        
        # Spectral entropy
        # Measure of spectral complexity
        psd_normalized = fft_magnitude / (np.sum(fft_magnitude) + 1e-10)
        features[f'{prefix}spectral_entropy'] = float(
            -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
        )
        
        # Spectral slope (linear regression of log-magnitude spectrum)
        log_magnitude = np.log(fft_magnitude + 1e-10)
        slope, _ = np.polyfit(frequencies, log_magnitude, 1)
        features[f'{prefix}spectral_slope'] = float(slope)
        
        # Spectral kurtosis
        features[f'{prefix}spectral_kurtosis'] = float(
            np.sum(((frequencies - centroid) ** 4) * normalized_magnitude) /
            (features[f'{prefix}spectral_spread'] ** 4 + 1e-10)
        )
        
        # Spectral skewness
        features[f'{prefix}spectral_skewness'] = float(
            np.sum(((frequencies - centroid) ** 3) * normalized_magnitude) /
            (features[f'{prefix}spectral_spread'] ** 3 + 1e-10)
        )
        
        return features
    
    def _band_power_features(
        self,
        fft_magnitude: np.ndarray,
        frequencies: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract power in specific frequency bands
        
        Args:
            fft_magnitude: FFT magnitude spectrum
            frequencies: Frequency array
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Define frequency bands (common for vibration analysis)
        bands = {
            'very_low': (0, 2),      # 0-2 Hz (slow variations)
            'low': (2, 10),          # 2-10 Hz (low frequency)
            'medium': (10, 100),     # 10-100 Hz (medium frequency)
            'high': (100, 1000),     # 100-1000 Hz (high frequency)
            'very_high': (1000, self.sampling_rate/2)  # > 1000 Hz
        }
        
        total_power = np.sum(fft_magnitude ** 2)
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequencies in this band
            band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
            
            if np.any(band_mask):
                # Power in band
                band_power = np.sum(fft_magnitude[band_mask] ** 2)
                features[f'{prefix}band_power_{band_name}'] = float(band_power)
                
                # Relative power
                features[f'{prefix}band_power_ratio_{band_name}'] = float(
                    band_power / (total_power + 1e-10)
                )
                
                # Peak frequency in band
                band_fft = fft_magnitude[band_mask]
                band_freqs = frequencies[band_mask]
                if len(band_fft) > 0:
                    peak_idx = np.argmax(band_fft)
                    features[f'{prefix}band_peak_freq_{band_name}'] = float(
                        band_freqs[peak_idx]
                    )
        
        return features
    
    def _harmonic_features(
        self,
        fft_magnitude: np.ndarray,
        frequencies: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract harmonic features
        
        Args:
            fft_magnitude: FFT magnitude spectrum
            frequencies: Frequency array
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Find peaks in spectrum
        peaks, properties = find_peaks(
            fft_magnitude,
            height=np.mean(fft_magnitude),
            distance=5
        )
        
        if len(peaks) > 0:
            # Number of peaks
            features[f'{prefix}num_peaks'] = float(len(peaks))
            
            # Peak frequencies
            peak_freqs = frequencies[peaks]
            peak_mags = fft_magnitude[peaks]
            
            # Sort by magnitude
            sorted_indices = np.argsort(peak_mags)[::-1]
            top_peaks = sorted_indices[:min(5, len(sorted_indices))]
            
            # Top 5 peak frequencies and magnitudes
            for i, idx in enumerate(top_peaks):
                features[f'{prefix}peak_freq_{i+1}'] = float(peak_freqs[idx])
                features[f'{prefix}peak_mag_{i+1}'] = float(peak_mags[idx])
            
            # Fundamental frequency (strongest peak)
            fundamental_idx = sorted_indices[0]
            fundamental_freq = peak_freqs[fundamental_idx]
            features[f'{prefix}fundamental_frequency'] = float(fundamental_freq)
            
            # Harmonic ratio (power in harmonics / total power)
            if fundamental_freq > 0:
                harmonic_power = 0
                for harmonic in [2, 3, 4, 5]:
                    harmonic_freq = fundamental_freq * harmonic
                    # Find closest frequency bin
                    freq_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                    if freq_idx < len(fft_magnitude):
                        harmonic_power += fft_magnitude[freq_idx] ** 2
                
                total_power = np.sum(fft_magnitude ** 2)
                features[f'{prefix}harmonic_ratio'] = float(
                    harmonic_power / (total_power + 1e-10)
                )
        else:
            features[f'{prefix}num_peaks'] = 0.0
            features[f'{prefix}fundamental_frequency'] = 0.0
            features[f'{prefix}harmonic_ratio'] = 0.0
        
        return features
    
    def _bearing_features(
        self,
        fft_magnitude: np.ndarray,
        frequencies: np.ndarray,
        bearing_params: Dict,
        prefix: str
    ) -> Dict[str, float]:
        """
        Extract bearing-specific frequency features
        
        Bearing defect frequencies:
        - BPFI: Ball Pass Frequency Inner race
        - BPFO: Ball Pass Frequency Outer race
        - BSF: Ball Spin Frequency
        - FTF: Fundamental Train Frequency (cage)
        
        Args:
            fft_magnitude: FFT magnitude spectrum
            frequencies: Frequency array
            bearing_params: Bearing geometry parameters
            prefix: Feature name prefix
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Extract parameters
        shaft_speed_rpm = bearing_params.get('shaft_speed', 1800)  # RPM
        num_balls = bearing_params.get('num_balls', 9)
        ball_diameter = bearing_params.get('ball_diameter', 10)  # mm
        pitch_diameter = bearing_params.get('pitch_diameter', 50)  # mm
        contact_angle = bearing_params.get('contact_angle', 0)  # degrees
        
        # Convert to Hz
        shaft_speed_hz = shaft_speed_rpm / 60
        
        # Calculate bearing frequencies
        contact_angle_rad = np.radians(contact_angle)
        diameter_ratio = ball_diameter / pitch_diameter
        
        # BPFO (Ball Pass Frequency Outer race)
        bpfo = (shaft_speed_hz * num_balls / 2) * (
            1 - diameter_ratio * np.cos(contact_angle_rad)
        )
        
        # BPFI (Ball Pass Frequency Inner race)
        bpfi = (shaft_speed_hz * num_balls / 2) * (
            1 + diameter_ratio * np.cos(contact_angle_rad)
        )
        
        # BSF (Ball Spin Frequency)
        bsf = (shaft_speed_hz * pitch_diameter / (2 * ball_diameter)) * (
            1 - (diameter_ratio * np.cos(contact_angle_rad)) ** 2
        )
        
        # FTF (Fundamental Train Frequency - cage)
        ftf = (shaft_speed_hz / 2) * (
            1 - diameter_ratio * np.cos(contact_angle_rad)
        )
        
        # Store calculated frequencies
        features[f'{prefix}bpfo_frequency'] = float(bpfo)
        features[f'{prefix}bpfi_frequency'] = float(bpfi)
        features[f'{prefix}bsf_frequency'] = float(bsf)
        features[f'{prefix}ftf_frequency'] = float(ftf)
        
        # Look for energy at these frequencies (±2% tolerance)
        tolerance = 0.02
        
        for freq_name, freq_value in [
            ('bpfo', bpfo), ('bpfi', bpfi), ('bsf', bsf), ('ftf', ftf)
        ]:
            # Find indices within tolerance
            lower = freq_value * (1 - tolerance)
            upper = freq_value * (1 + tolerance)
            mask = (frequencies >= lower) & (frequencies <= upper)
            
            if np.any(mask):
                # Peak magnitude in this range
                peak_mag = np.max(fft_magnitude[mask])
                features[f'{prefix}{freq_name}_magnitude'] = float(peak_mag)
                
                # Average magnitude in this range
                avg_mag = np.mean(fft_magnitude[mask])
                features[f'{prefix}{freq_name}_avg_magnitude'] = float(avg_mag)
            else:
                features[f'{prefix}{freq_name}_magnitude'] = 0.0
                features[f'{prefix}{freq_name}_avg_magnitude'] = 0.0
        
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
    print("FREQUENCY-DOMAIN FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Test 1: Extract features from single signal
    print("\nTest 1: Single Signal Frequency Features")
    print("-"*80)
    
    extractor = FrequencyDomainFeatures(sampling_rate=sampling_rate)
    
    # Extract features from vibration signal (first 1000 samples)
    signal_data = df['vibration_rms'].iloc[:1000].values
    
    # Bearing parameters (example)
    bearing_params = {
        'shaft_speed': 1800,  # RPM
        'num_balls': 9,
        'ball_diameter': 10,  # mm
        'pitch_diameter': 50,  # mm
        'contact_angle': 0    # degrees
    }
    
    features = extractor.extract_all_features(
        signal_data,
        prefix='vib_',
        bearing_params=bearing_params
    )
    
    print(f"Extracted {len(features)} frequency-domain features")
    
    # Display sample features
    print("\nSample FFT Features:")
    for key in ['vib_dominant_frequency', 'vib_fft_mean', 'vib_fft_max']:
        if key in features:
            print(f"  {key}: {features[key]:.4f}")
    
    print("\nPower Spectral Density Features:")
    for key in ['vib_total_power', 'vib_peak_power', 'vib_power_low_band']:
        if key in features:
            print(f"  {key}: {features[key]:.4f}")
    
    print("\nSpectral Features:")
    for key in ['vib_spectral_centroid', 'vib_spectral_rolloff', 'vib_spectral_entropy']:
        if key in features:
            print(f"  {key}: {features[key]:.4f}")
    
    print("\nBearing Frequencies:")
    for key in ['vib_bpfo_frequency', 'vib_bpfi_frequency', 'vib_bsf_frequency']:
        if key in features:
            print(f"  {key}: {features[key]:.4f}")
    
    # Test 2: Visualize FFT
    print("\n\nTest 2: FFT Visualization")
    print("-"*80)
    
    import matplotlib.pyplot as plt
    
    fft_magnitude, frequencies = extractor._compute_fft(signal_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time domain
    axes[0].plot(signal_data, linewidth=0.5)
    axes[0].set_title('Time Domain Signal', fontweight='bold')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Frequency domain
    axes[1].plot(frequencies, fft_magnitude, linewidth=1)
    axes[1].set_title('Frequency Domain (FFT)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlim(0, sampling_rate/2)
    axes[1].grid(True, alpha=0.3)
    
    # Mark dominant frequency
    dominant_idx = np.argmax(fft_magnitude)
    dominant_freq = frequencies[dominant_idx]
    axes[1].axvline(dominant_freq, color='red', linestyle='--',
                   label=f'Dominant: {dominant_freq:.2f} Hz')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('frequency_analysis_test.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to frequency_analysis_test.png")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)