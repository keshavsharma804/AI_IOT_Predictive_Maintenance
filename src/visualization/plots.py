"""
Advanced Plotting Module
Professional visualization utilities for IoT sensor data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin
from ..config import get_config


class SensorPlotter(LoggerMixin):
    """
    Advanced plotting utilities for sensor data visualization
    
    Features:
    - Time series plots with anomaly highlighting
    - Multi-sensor comparison plots
    - Statistical distribution analysis
    - Correlation heatmaps
    - Failure event timelines
    - Frequency domain analysis
    """
    
    def __init__(self, config: Optional[dict] = None, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize plotter
        
        Args:
            config: Configuration dictionary
            style: Matplotlib style
        """
        self.config = config or get_config()
        self.style = style
        self.figure_path = Path(self.config.get('paths.figures', 'results/figures'))
        self.figure_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.style)
        
        # Color scheme from config
        self.colors = {
            'normal': self.config.get('visualization.colors.normal', '#2ECC71'),
            'warning': self.config.get('visualization.colors.warning', '#F39C12'),
            'critical': self.config.get('visualization.colors.critical', '#E74C3C'),
            'anomaly': self.config.get('visualization.colors.anomaly', '#C0392B')
        }
        
        self.logger.info("SensorPlotter initialized")
    
    def plot_sensor_timeseries(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str] = None,
        highlight_anomalies: bool = True,
        figsize: Tuple[int, int] = (16, 12),
        save: bool = True,
        filename: str = 'sensor_timeseries.png'
    ) -> plt.Figure:
        """
        Plot time series for multiple sensors with anomaly highlighting
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns to plot
            highlight_anomalies: Whether to highlight anomalies
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        if sensor_cols is None:
            sensor_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
        
        self.logger.info(f"Plotting time series for {len(sensor_cols)} sensors")
        
        fig, axes = plt.subplots(len(sensor_cols), 1, figsize=figsize, sharex=True)
        
        if len(sensor_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(sensor_cols):
            ax = axes[i]
            
            if highlight_anomalies and 'is_anomaly' in df.columns:
                # Plot normal data
                normal_mask = df['is_anomaly'] == 0
                ax.plot(df.loc[normal_mask, 'timestamp'],
                       df.loc[normal_mask, col],
                       color=self.colors['normal'],
                       alpha=0.7,
                       linewidth=0.5,
                       label='Normal')
                
                # Highlight anomalies
                anomaly_mask = df['is_anomaly'] == 1
                if anomaly_mask.sum() > 0:
                    ax.scatter(df.loc[anomaly_mask, 'timestamp'],
                             df.loc[anomaly_mask, col],
                             color=self.colors['anomaly'],
                             s=5,
                             alpha=0.8,
                             label='Anomaly',
                             zorder=5)
            else:
                # Plot all data
                ax.plot(df['timestamp'], df[col],
                       color='blue',
                       alpha=0.7,
                       linewidth=0.8)
            
            ax.set_ylabel(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        plt.suptitle('Sensor Time Series Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_failure_timeline(
        self,
        failure_summary: pd.DataFrame,
        figsize: Tuple[int, int] = (16, 8),
        save: bool = True,
        filename: str = 'failure_timeline.png'
    ) -> plt.Figure:
        """
        Plot timeline of failure events
        
        Args:
            failure_summary: DataFrame with failure event details
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        if failure_summary.empty:
            self.logger.warning("No failure events to plot")
            return None
        
        self.logger.info(f"Plotting timeline for {len(failure_summary)} failure events")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color map for failure types
        failure_types = failure_summary['failure_type'].unique()
        colors_map = plt.cm.Set3(np.linspace(0, 1, len(failure_types)))
        failure_colors = dict(zip(failure_types, colors_map))
        
        for idx, row in failure_summary.iterrows():
            start = pd.to_datetime(row['start_time'])
            end = pd.to_datetime(row['end_time'])
            duration = (end - start).days
            
            color = failure_colors[row['failure_type']]
            
            # Plot bar
            ax.barh(idx, duration, left=start, height=0.8,
                   color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add text label
            mid_point = start + (end - start) / 2
            ax.text(mid_point, idx, f"{row['failure_type']}\n({row['machine_id']})",
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Failure Event #', fontsize=12, fontweight='bold')
        ax.set_title('Failure Event Timeline', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=failure_colors[ft], alpha=0.7) 
                  for ft in failure_types]
        ax.legend(handles, failure_types, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True,
        filename: str = 'correlation_matrix.png'
    ) -> plt.Figure:
        """
        Plot correlation heatmap for sensors
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        if sensor_cols is None:
            sensor_cols = ['vibration_x', 'vibration_y', 'vibration_z',
                          'vibration_rms', 'temperature', 'pressure', 'current']
        
        # Filter available columns
        available_cols = [col for col in sensor_cols if col in df.columns]
        
        self.logger.info(f"Plotting correlation matrix for {len(available_cols)} sensors")
        
        correlation = df[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(correlation,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"},
                   ax=ax)
        
        ax.set_title('Sensor Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_distributions(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str] = None,
        compare_normal_anomaly: bool = True,
        figsize: Tuple[int, int] = (14, 10),
        save: bool = True,
        filename: str = 'sensor_distributions.png'
    ) -> plt.Figure:
        """
        Plot distribution histograms for sensors
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            compare_normal_anomaly: Whether to compare normal vs anomaly
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        if sensor_cols is None:
            sensor_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
        
        self.logger.info(f"Plotting distributions for {len(sensor_cols)} sensors")
        
        n_cols = 2
        n_rows = (len(sensor_cols) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(sensor_cols):
            ax = axes[i]
            
            if compare_normal_anomaly and 'is_anomaly' in df.columns:
                # Normal distribution
                normal_data = df.loc[df['is_anomaly'] == 0, col].dropna()
                ax.hist(normal_data, bins=50, alpha=0.6,
                       label='Normal', color=self.colors['normal'],
                       density=True, edgecolor='black', linewidth=0.5)
                
                # Anomaly distribution
                anomaly_data = df.loc[df['is_anomaly'] == 1, col].dropna()
                if len(anomaly_data) > 0:
                    ax.hist(anomaly_data, bins=50, alpha=0.6,
                           label='Anomaly', color=self.colors['anomaly'],
                           density=True, edgecolor='black', linewidth=0.5)
                
                # Add statistics
                ax.axvline(normal_data.mean(), color=self.colors['normal'],
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Normal Mean: {normal_data.mean():.2f}')
                if len(anomaly_data) > 0:
                    ax.axvline(anomaly_data.mean(), color=self.colors['anomaly'],
                              linestyle='--', linewidth=2, alpha=0.8,
                              label=f'Anomaly Mean: {anomaly_data.mean():.2f}')
            else:
                # Plot all data
                data = df[col].dropna()
                ax.hist(data, bins=50, alpha=0.7, color='blue',
                       density=True, edgecolor='black', linewidth=0.5)
                ax.axvline(data.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {data.mean():.2f}')
            
            ax.set_xlabel(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(sensor_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Sensor Data Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_frequency_spectrum(
        self,
        signal: np.ndarray,
        sampling_rate: float = 100,
        sensor_name: str = 'Vibration',
        figsize: Tuple[int, int] = (14, 6),
        save: bool = True,
        filename: str = 'frequency_spectrum.png'
    ) -> plt.Figure:
        """
        Plot frequency spectrum using FFT
        
        Args:
            signal: Time-domain signal
            sampling_rate: Sampling rate in Hz
            sensor_name: Name of the sensor
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        self.logger.info(f"Plotting frequency spectrum for {sensor_name}")
        
        # Perform FFT
        n = len(signal)
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(n, 1/sampling_rate)
        
        # Get positive frequencies only
        positive_freq_idx = fft_freq > 0
        fft_freq_positive = fft_freq[positive_freq_idx]
        fft_magnitude = np.abs(fft_values[positive_freq_idx])
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Time domain
        ax1 = axes[0]
        time = np.arange(len(signal)) / sampling_rate
        ax1.plot(time, signal, linewidth=0.5, color='blue', alpha=0.7)
        ax1.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        ax1.set_title(f'{sensor_name} - Time Domain', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain
        ax2 = axes[1]
        ax2.plot(fft_freq_positive, fft_magnitude, linewidth=1, color='red', alpha=0.7)
        ax2.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Magnitude', fontsize=10, fontweight='bold')
        ax2.set_title(f'{sensor_name} - Frequency Domain', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, sampling_rate/2)  # Nyquist frequency
        
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_anomaly_detection_results(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        scores: np.ndarray,
        sensor_col: str = 'vibration_rms',
        figsize: Tuple[int, int] = (16, 10),
        save: bool = True,
        filename: str = 'anomaly_detection_results.png'
    ) -> plt.Figure:
        """
        Plot anomaly detection results with scores
        
        Args:
            df: DataFrame with sensor data
            predictions: Anomaly predictions (0=normal, 1=anomaly)
            scores: Anomaly scores
            sensor_col: Sensor column to plot
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        self.logger.info("Plotting anomaly detection results")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Sensor data with predictions
        ax1 = axes[0]
        
        # Normal points
        normal_mask = predictions == 0
        ax1.scatter(df.loc[normal_mask, 'timestamp'],
                   df.loc[normal_mask, sensor_col],
                   c=self.colors['normal'],
                   s=1,
                   alpha=0.5,
                   label='Normal')
        
        # Anomaly points
        anomaly_mask = predictions == 1
        ax1.scatter(df.loc[anomaly_mask, 'timestamp'],
                   df.loc[anomaly_mask, sensor_col],
                   c=self.colors['anomaly'],
                   s=10,
                   alpha=0.8,
                   label='Detected Anomaly',
                   zorder=5)
        
        # True anomalies if available
        if 'is_anomaly' in df.columns:
            true_anomaly_mask = df['is_anomaly'] == 1
            ax1.scatter(df.loc[true_anomaly_mask, 'timestamp'],
                       df.loc[true_anomaly_mask, sensor_col],
                       c='none',
                       s=20,
                       edgecolors='black',
                       linewidths=1.5,
                       alpha=0.6,
                       label='True Anomaly',
                       zorder=4)
        
        ax1.set_ylabel(sensor_col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Sensor Data with Anomaly Detection', fontsize=12, fontweight='bold')
        
        # Plot 2: Anomaly scores
        ax2 = axes[1]
        ax2.plot(df['timestamp'], scores, linewidth=0.8, color='purple', alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(df['timestamp'], scores, 0,
                        where=(predictions == 1),
                        color=self.colors['anomaly'],
                        alpha=0.3,
                        label='Anomaly Region')
        
        ax2.set_xlabel('Timestamp', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Anomaly Score', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Anomaly Scores Over Time', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_boxplots(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str] = None,
        group_by: str = 'is_anomaly',
        figsize: Tuple[int, int] = (14, 8),
        save: bool = True,
        filename: str = 'sensor_boxplots.png'
    ) -> plt.Figure:
        """
        Plot boxplots for sensors grouped by category
        
        Args:
            df: DataFrame with sensor data
            sensor_cols: List of sensor columns
            group_by: Column to group by
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        if sensor_cols is None:
            sensor_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
        
        if group_by not in df.columns:
            self.logger.warning(f"Column '{group_by}' not found, plotting without grouping")
            group_by = None
        
        self.logger.info(f"Plotting boxplots for {len(sensor_cols)} sensors")
        
        n_cols = 2
        n_rows = (len(sensor_cols) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(sensor_cols):
            ax = axes[i]
            
            if group_by:
                # Create boxplot by group
                data_to_plot = [df.loc[df[group_by] == val, col].dropna() 
                               for val in df[group_by].unique()]
                labels = [str(val) for val in df[group_by].unique()]
                
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                               showmeans=True, meanline=True)
                
                # Color boxes
                for patch, label in zip(bp['boxes'], labels):
                    if label == '0':
                        patch.set_facecolor(self.colors['normal'])
                    else:
                        patch.set_facecolor(self.colors['anomaly'])
                    patch.set_alpha(0.6)
            else:
                # Single boxplot
                ax.boxplot(df[col].dropna(), patch_artist=True,
                          showmeans=True, meanline=True)
            
            ax.set_ylabel(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            if group_by:
                ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=10)
        
        # Hide extra subplots
        for i in range(len(sensor_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Sensor Boxplot Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_machine_comparison(
        self,
        df: pd.DataFrame,
        sensor_col: str = 'vibration_rms',
        figsize: Tuple[int, int] = (16, 10),
        save: bool = True,
        filename: str = 'machine_comparison.png'
    ) -> plt.Figure:
        """
        Compare sensor readings across different machines
        
        Args:
            df: DataFrame with sensor data (multiple machines)
            sensor_col: Sensor column to compare
            figsize: Figure size
            save: Whether to save the figure
            filename: Output filename
            
        Returns:
            Figure object
        """
        if 'machine_id' not in df.columns:
            self.logger.warning("No machine_id column found")
            return None
        
        machines = df['machine_id'].unique()
        self.logger.info(f"Comparing {sensor_col} across {len(machines)} machines")
        
        fig, axes = plt.subplots(len(machines), 1, figsize=figsize, sharex=True)
        
        if len(machines) == 1:
            axes = [axes]
        
        for i, machine in enumerate(machines):
            ax = axes[i]
            machine_data = df[df['machine_id'] == machine]
            
            # Plot data
            if 'is_anomaly' in machine_data.columns:
                normal_mask = machine_data['is_anomaly'] == 0
                ax.plot(machine_data.loc[normal_mask, 'timestamp'],
                       machine_data.loc[normal_mask, sensor_col],
                       color=self.colors['normal'],
                       alpha=0.7,
                       linewidth=0.5)
                
                anomaly_mask = machine_data['is_anomaly'] == 1
                if anomaly_mask.sum() > 0:
                    ax.scatter(machine_data.loc[anomaly_mask, 'timestamp'],
                             machine_data.loc[anomaly_mask, sensor_col],
                             color=self.colors['anomaly'],
                             s=5,
                             alpha=0.8,
                             zorder=5)
            else:
                ax.plot(machine_data['timestamp'],
                       machine_data[sensor_col],
                       color='blue',
                       alpha=0.7,
                       linewidth=0.5)
            
            ax.set_ylabel(machine, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = machine_data[sensor_col].mean()
            ax.axhline(mean_val, color='red', linestyle='--',
                      linewidth=1, alpha=0.5, label=f'Mean: {mean_val:.2f}')
            ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        plt.suptitle(f'{sensor_col.replace("_", " ").title()} - Machine Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            save_path = self.figure_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig


# Example usage
if __name__ == "__main__":
    from ..config import get_config
    from ..data.data_loader import DataLoader
    
    # Load data
    config = get_config()
    loader = DataLoader(config)
    df = loader.load_machine_data("machine_001")
    
    # Initialize plotter
    plotter = SensorPlotter(config)
    
    # Create various plots
    plotter.plot_sensor_timeseries(df)
    plotter.plot_correlation_matrix(df)
    plotter.plot_distributions(df)
    plotter.plot_boxplots(df)
    
    print("✓ All plots generated successfully")