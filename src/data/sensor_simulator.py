"""
IoT Sensor Data Generator
Physics-based simulation of industrial equipment sensors with realistic failure patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin
from ..config import get_config


@dataclass
class FailureEvent:
    """Represents a failure event in the system"""
    machine_id: str
    failure_type: str
    start_time: datetime
    end_time: datetime
    severity: float  # 0-1 scale
    affected_sensors: List[str]


class SensorSimulator(LoggerMixin):
    """
    Simulates realistic IoT sensor data for industrial equipment
    
    Features:
    - Physics-based sensor modeling
    - Realistic failure scenarios
    - Multi-sensor correlation
    - Temporal patterns (daily, seasonal)
    - Data quality issues (noise, drift, missing)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sensor simulator
        
        Args:
            config: Configuration dictionary (uses default if None)
        """
        self.config = config or get_config()
        self.sampling_rate = self.config.get('data_generation.sampling_rate', 100)
        self.duration_days = self.config.get('data_generation.duration_days', 30)
        self.num_machines = self.config.get('data_generation.num_machines', 5)
        
        # Sensor configurations
        self.sensor_config = self.config.get('data_generation.sensors', {})
        self.failure_config = self.config.get('data_generation.failures', {})
        
        # Calculate total samples
        self.total_samples = int(self.sampling_rate * 60 * 60 * 24 * self.duration_days)
        
        # Storage for failure events
        self.failure_events: List[FailureEvent] = []
        
        self.logger.info(f"Initialized SensorSimulator: {self.num_machines} machines, "
                        f"{self.duration_days} days, {self.sampling_rate}Hz")
    
    def generate_data(self, machine_id: str = "machine_001") -> pd.DataFrame:
        """
        Generate complete sensor data for a machine
        
        Args:
            machine_id: Unique machine identifier
            
        Returns:
            DataFrame with all sensor readings and labels
        """
        self.logger.info(f"Generating data for {machine_id}...")
        
        # Create time index
        start_time = datetime.now() - timedelta(days=self.duration_days)
        time_index = pd.date_range(
            start=start_time,
            periods=self.total_samples,
            freq=f'{1000/self.sampling_rate}ms'
        )
        
        # Initialize DataFrame
        df = pd.DataFrame({'timestamp': time_index})
        df['machine_id'] = machine_id
        
        # Generate failure events for this machine
        failure_events = self._generate_failure_events(machine_id, start_time)
        self.failure_events.extend(failure_events)
        
        # Generate sensor data
        df = self._generate_vibration_data(df, failure_events)
        df = self._generate_temperature_data(df, failure_events)
        df = self._generate_pressure_data(df, failure_events)
        df = self._generate_current_data(df, failure_events)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add labels
        df = self._add_labels(df, failure_events)
        
        # Add data quality issues
        df = self._add_data_quality_issues(df)
        
        self.logger.info(f"Generated {len(df)} samples with {len(failure_events)} failures")
        
        return df
    
    def _generate_failure_events(
        self, 
        machine_id: str, 
        start_time: datetime
    ) -> List[FailureEvent]:
        """
        Generate random failure events based on configuration
        
        Args:
            machine_id: Machine identifier
            start_time: Simulation start time
            
        Returns:
            List of FailureEvent objects
        """
        events = []
        
        for failure_type, config in self.failure_config.items():
            probability = config.get('probability', 0.05)
            degradation_days = config.get('degradation_days', 7)
            affected_sensors = config.get('affected_sensors', [])
            
            # Randomly decide if this failure occurs
            if np.random.random() < probability:
                # Random start time (not too early, not too late)
                days_offset = np.random.randint(5, self.duration_days - degradation_days - 2)
                failure_start = start_time + timedelta(days=days_offset)
                failure_end = failure_start + timedelta(days=degradation_days)
                
                severity = np.random.uniform(0.6, 1.0)
                
                event = FailureEvent(
                    machine_id=machine_id,
                    failure_type=failure_type,
                    start_time=failure_start,
                    end_time=failure_end,
                    severity=severity,
                    affected_sensors=affected_sensors
                )
                events.append(event)
                
                self.logger.info(f"Created {failure_type} event: "
                               f"{failure_start.date()} to {failure_end.date()}")
        
        return events
    
    def _generate_vibration_data(
        self, 
        df: pd.DataFrame, 
        failure_events: List[FailureEvent]
    ) -> pd.DataFrame:
        """
        Generate realistic vibration sensor data (3-axis accelerometer)
        
        Physics:
        - Normal operation: Low frequency vibration (bearing rotation)
        - Imbalance: Increases at 1x rotation frequency
        - Bearing wear: High frequency components increase
        - Looseness: Sub-harmonic frequencies appear
        
        Args:
            df: DataFrame with timestamp
            failure_events: List of failure events
            
        Returns:
            DataFrame with vibration columns added
        """
        config = self.sensor_config.get('vibration', {})
        normal_range = config.get('normal_range', [0.5, 2.0])
        noise_level = config.get('noise_level', 0.1)
        
        n = len(df)
        t = np.arange(n) / self.sampling_rate  # Time in seconds
        
        # Base frequencies (typical motor speeds)
        rotation_freq = 30  # Hz (1800 RPM)
        bearing_freq = rotation_freq * 5.2  # BPFI frequency
        
        # Initialize vibration signals
        vib_x = np.zeros(n)
        vib_y = np.zeros(n)
        vib_z = np.zeros(n)
        
        # Normal operation - low amplitude sinusoids
        base_amplitude = np.random.uniform(normal_range[0], normal_range[1])
        vib_x = base_amplitude * np.sin(2 * np.pi * rotation_freq * t)
        vib_y = base_amplitude * 0.8 * np.sin(2 * np.pi * rotation_freq * t + np.pi/4)
        vib_z = base_amplitude * 0.6 * np.sin(2 * np.pi * rotation_freq * t + np.pi/2)
        
        # Add harmonics (normal operation)
        for harmonic in [2, 3]:
            amplitude = base_amplitude / (harmonic * 2)
            vib_x += amplitude * np.sin(2 * np.pi * rotation_freq * harmonic * t)
            vib_y += amplitude * np.sin(2 * np.pi * rotation_freq * harmonic * t)
        
        # Add daily variation (temperature affects vibration)
        hours = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        daily_variation = 0.2 * np.sin(2 * np.pi * hours / 24)
        vib_x *= (1 + daily_variation)
        vib_y *= (1 + daily_variation)
        vib_z *= (1 + daily_variation)
        
        # Apply failure effects
        for event in failure_events:
            if 'vibration' not in event.affected_sensors:
                continue
            
            # Get indices affected by this failure
            mask = (df['timestamp'] >= event.start_time) & (df['timestamp'] <= event.end_time)
            affected_indices = np.where(mask)[0]
            
            if len(affected_indices) == 0:
                continue
            
            # Calculate degradation progression (0 to 1)
            progression = np.linspace(0, event.severity, len(affected_indices))
            
            if event.failure_type == 'bearing_wear':
                # Increase high-frequency components
                high_freq_component = progression[:, np.newaxis] * 3 * np.sin(
                    2 * np.pi * bearing_freq * t[affected_indices, np.newaxis]
                )
                vib_x[affected_indices] += high_freq_component.flatten()
                vib_y[affected_indices] += high_freq_component.flatten() * 0.8
                vib_z[affected_indices] += high_freq_component.flatten() * 0.6
                
            elif event.failure_type == 'imbalance':
                # Increase at 1x rotation frequency
                imbalance_component = progression * 4 * np.sin(2 * np.pi * rotation_freq * t[affected_indices])
                vib_x[affected_indices] += imbalance_component
                vib_y[affected_indices] += imbalance_component * 0.9
                
            elif event.failure_type == 'cavitation':
                # Add random impulses
                num_impulses = int(len(affected_indices) * 0.01)
                impulse_indices = np.random.choice(affected_indices, num_impulses, replace=False)
                impulse_magnitude = progression[impulse_indices - affected_indices[0]] * 5
                vib_x[impulse_indices] += impulse_magnitude * np.random.randn(num_impulses)
                vib_y[impulse_indices] += impulse_magnitude * np.random.randn(num_impulses)
        
        # Add noise
        vib_x += noise_level * np.random.randn(n)
        vib_y += noise_level * np.random.randn(n)
        vib_z += noise_level * np.random.randn(n)
        
        # Ensure non-negative (absolute values for vibration)
        df['vibration_x'] = np.abs(vib_x)
        df['vibration_y'] = np.abs(vib_y)
        df['vibration_z'] = np.abs(vib_z)
        
        # Calculate RMS (commonly used metric)
        df['vibration_rms'] = np.sqrt((df['vibration_x']**2 + df['vibration_y']**2 + df['vibration_z']**2) / 3)
        
        return df
    
    def _generate_temperature_data(
        self, 
        df: pd.DataFrame, 
        failure_events: List[FailureEvent]
    ) -> pd.DataFrame:
        """
        Generate realistic temperature sensor data
        
        Physics:
        - Normal: Stable with daily variation
        - Overheating: Gradual temperature rise
        - Bearing wear: Increased friction -> higher temperature
        
        Args:
            df: DataFrame with timestamp and vibration data
            failure_events: List of failure events
            
        Returns:
            DataFrame with temperature column added
        """
        config = self.sensor_config.get('temperature', {})
        normal_range = config.get('normal_range', [40, 60])
        noise_level = config.get('noise_level', 0.5)
        
        n = len(df)
        
        # Base temperature (with daily variation)
        base_temp = np.random.uniform(normal_range[0], normal_range[1])
        hours = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        
        # Daily cycle (cooler at night, warmer during day)
        daily_cycle = 5 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at noon
        
        # Weekly cycle (weekend vs weekday operation)
        weekly_cycle = 2 * np.sin(2 * np.pi * hours / (24 * 7))
        
        temperature = base_temp + daily_cycle + weekly_cycle
        
        # Temperature correlates with vibration (friction generates heat)
        vibration_effect = (df['vibration_rms'] - df['vibration_rms'].mean()) * 2
        temperature += vibration_effect
        
        # Apply failure effects
        for event in failure_events:
            if 'temperature' not in event.affected_sensors:
                continue
            
            mask = (df['timestamp'] >= event.start_time) & (df['timestamp'] <= event.end_time)
            affected_indices = np.where(mask)[0]
            
            if len(affected_indices) == 0:
                continue
            
            progression = np.linspace(0, event.severity, len(affected_indices))
            
            if event.failure_type in ['bearing_wear', 'overheating']:
                # Gradual temperature increase
                temp_increase = progression * 25  # Up to 25°C increase
                temperature[affected_indices] += temp_increase
        
        # Add noise
        temperature += noise_level * np.random.randn(n)
        
        # Add sensor drift (realistic issue)
        drift = np.linspace(0, 0.5, n)  # 0.5°C drift over time
        temperature += drift
        
        df['temperature'] = temperature
        
        return df
    
    def _generate_pressure_data(
        self, 
        df: pd.DataFrame, 
        failure_events: List[FailureEvent]
    ) -> pd.DataFrame:
        """
        Generate realistic pressure sensor data
        
        Physics:
        - Normal: Stable pressure with small fluctuations
        - Cavitation: Pressure drops and fluctuates
        - Blockage: Pressure increases
        
        Args:
            df: DataFrame with timestamp
            failure_events: List of failure events
            
        Returns:
            DataFrame with pressure column added
        """
        config = self.sensor_config.get('pressure', {})
        normal_range = config.get('normal_range', [80, 120])
        noise_level = config.get('noise_level', 1.0)
        
        n = len(df)
        
        # Base pressure
        base_pressure = np.random.uniform(normal_range[0], normal_range[1])
        pressure = np.ones(n) * base_pressure
        
        # Small fluctuations (pump cycles)
        t = np.arange(n) / self.sampling_rate
        pump_freq = 0.5  # Hz (slow pump cycle)
        pressure += 5 * np.sin(2 * np.pi * pump_freq * t)
        
        # Apply failure effects
        for event in failure_events:
            if 'pressure' not in event.affected_sensors:
                continue
            
            mask = (df['timestamp'] >= event.start_time) & (df['timestamp'] <= event.end_time)
            affected_indices = np.where(mask)[0]
            
            if len(affected_indices) == 0:
                continue
            
            progression = np.linspace(0, event.severity, len(affected_indices))
            
            if event.failure_type == 'cavitation':
                # Pressure drops and becomes erratic
                pressure_drop = progression * 30
                pressure[affected_indices] -= pressure_drop
                # Add large fluctuations
                erratic_noise = progression[:, np.newaxis] * 10 * np.random.randn(len(affected_indices), 1).flatten()
                pressure[affected_indices] += erratic_noise
        
        # Add noise
        pressure += noise_level * np.random.randn(n)
        
        # Ensure positive pressure
        pressure = np.maximum(pressure, 10)
        
        df['pressure'] = pressure
        
        return df
    
    def _generate_current_data(
        self, 
        df: pd.DataFrame, 
        failure_events: List[FailureEvent]
    ) -> pd.DataFrame:
        """
        Generate realistic current sensor data
        
        Physics:
        - Normal: Stable current draw
        - Overheating: Current increases
        - Mechanical issues: Current fluctuates
        
        Args:
            df: DataFrame with timestamp
            failure_events: List of failure events
            
        Returns:
            DataFrame with current column added
        """
        config = self.sensor_config.get('current', {})
        normal_range = config.get('normal_range', [8, 12])
        noise_level = config.get('noise_level', 0.2)
        
        n = len(df)
        
        # Base current
        base_current = np.random.uniform(normal_range[0], normal_range[1])
        current = np.ones(n) * base_current
        
        # Current correlates with load (simulated by vibration)
        load_effect = (df['vibration_rms'] - df['vibration_rms'].mean()) * 0.5
        current += load_effect
        
        # Apply failure effects
        for event in failure_events:
            if 'current' not in event.affected_sensors:
                continue
            
            mask = (df['timestamp'] >= event.start_time) & (df['timestamp'] <= event.end_time)
            affected_indices = np.where(mask)[0]
            
            if len(affected_indices) == 0:
                continue
            
            progression = np.linspace(0, event.severity, len(affected_indices))
            
            if event.failure_type == 'overheating':
                # Current increases with overheating
                current_increase = progression * 4
                current[affected_indices] += current_increase
        
        # Add noise
        current += noise_level * np.random.randn(n)
        
        # Ensure positive current
        current = np.maximum(current, 0)
        
        df['current'] = current
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (hour, day of week, etc.)
        
        Args:
            df: DataFrame with timestamp
            
        Returns:
            DataFrame with temporal features added
        """
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_month'] = df['timestamp'].dt.day
        
        return df
    
    def _add_labels(
        self, 
        df: pd.DataFrame, 
        failure_events: List[FailureEvent]
    ) -> pd.DataFrame:
        """
        Add anomaly labels based on failure events
        
        Args:
            df: DataFrame with sensor data
            failure_events: List of failure events
            
        Returns:
            DataFrame with labels added
        """
        # Initialize labels
        df['is_anomaly'] = 0
        df['failure_type'] = 'normal'
        df['severity'] = 0.0
        df['time_to_failure'] = -1  # -1 means no failure
        
        for event in failure_events:
            mask = (df['timestamp'] >= event.start_time) & (df['timestamp'] <= event.end_time)
            
            df.loc[mask, 'is_anomaly'] = 1
            df.loc[mask, 'failure_type'] = event.failure_type
            
            # Severity increases over time
            affected_timestamps = df.loc[mask, 'timestamp']
            if len(affected_timestamps) > 0:
                time_progression = (affected_timestamps - event.start_time).dt.total_seconds()
                total_duration = (event.end_time - event.start_time).total_seconds()
                severity_progression = (time_progression / total_duration) * event.severity
                df.loc[mask, 'severity'] = severity_progression.values
            
            # Time to failure (in hours)
            time_to_failure = (event.end_time - df.loc[mask, 'timestamp']).dt.total_seconds() / 3600
            df.loc[mask, 'time_to_failure'] = time_to_failure.values
        
        return df
    
    def _add_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add realistic data quality issues
        
        - Missing values (sensor dropout)
        - Outliers (sensor spikes)
        - Stuck values (sensor freeze)
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with quality issues added
        """
        sensor_columns = ['vibration_x', 'vibration_y', 'vibration_z', 
                         'temperature', 'pressure', 'current']
        
        n = len(df)
        
        # 1. Random missing values (0.1% of data)
        missing_rate = 0.001
        for col in sensor_columns:
            missing_indices = np.random.choice(n, int(n * missing_rate), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        # 2. Random outliers (0.05% of data)
        outlier_rate = 0.0005
        for col in sensor_columns:
            outlier_indices = np.random.choice(n, int(n * outlier_rate), replace=False)
            # Spike to 5x normal value
            df.loc[outlier_indices, col] *= np.random.uniform(3, 7, len(outlier_indices))
        
        # 3. Stuck sensor values (sensor freeze for 10-50 samples)
        num_stuck_events = 5
        for _ in range(num_stuck_events):
            col = np.random.choice(sensor_columns)
            start_idx = np.random.randint(0, n - 100)
            duration = np.random.randint(10, 50)
            stuck_value = df.loc[start_idx, col]
            df.loc[start_idx:start_idx+duration, col] = stuck_value
        
        return df
    
    def generate_multiple_machines(self) -> pd.DataFrame:
        """
        Generate data for all machines
        
        Returns:
            Combined DataFrame with all machines
        """
        self.logger.info(f"Generating data for {self.num_machines} machines...")
        
        all_data = []
        
        for i in range(self.num_machines):
            machine_id = f"machine_{i+1:03d}"
            df = self.generate_data(machine_id)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        self.logger.info(f"Generated total {len(combined_df)} samples")
        self.logger.info(f"Total failures: {len(self.failure_events)}")
        
        return combined_df
    
    def get_failure_summary(self) -> pd.DataFrame:
        """
        Get summary of all failure events
        
        Returns:
            DataFrame with failure event details
        """
        if not self.failure_events:
            return pd.DataFrame()
        
        summary_data = []
        for event in self.failure_events:
            summary_data.append({
                'machine_id': event.machine_id,
                'failure_type': event.failure_type,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'duration_days': (event.end_time - event.start_time).days,
                'severity': event.severity,
                'affected_sensors': ', '.join(event.affected_sensors)
            })
        
        return pd.DataFrame(summary_data)


# Example usage and testing
if __name__ == "__main__":
    from ..config import get_config
    
    # Initialize
    config = get_config()
    simulator = SensorSimulator(config)
    
    # Generate data for one machine
    print("Generating data for single machine...")
    df = simulator.generate_data("machine_001")
    
    print(f"\nGenerated {len(df)} samples")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nAnomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    print(f"\nFailure types:\n{df['failure_type'].value_counts()}")
    
    # Get failure summary
    failure_summary = simulator.get_failure_summary()
    print(f"\nFailure Events:\n{failure_summary}")
    
    # Save to file
    output_path = config.get('paths.data_synthetic') + '/machine_001_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved data to: {output_path}")