"""
Real-Time Dashboard Module
Interactive dashboard for monitoring sensor data and anomalies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Callable
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import LoggerMixin
from ..config import get_config


class RealtimeDashboard(LoggerMixin):
    """
    Real-time monitoring dashboard for IoT sensors
    
    Features:
    - Live sensor readings
    - Anomaly alerts
    - Historical trends
    - Performance metrics
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        window_size: int = 1000,
        update_interval: int = 100
    ):
        """
        Initialize dashboard
        
        Args:
            config: Configuration dictionary
            window_size: Number of samples to display
            update_interval: Update interval in milliseconds
        """
        self.config = config or get_config()
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Data buffers (deque for efficient FIFO)
        self.time_buffer = deque(maxlen=window_size)
        self.vibration_buffer = deque(maxlen=window_size)
        self.temperature_buffer = deque(maxlen=window_size)
        self.pressure_buffer = deque(maxlen=window_size)
        self.current_buffer = deque(maxlen=window_size)
        self.anomaly_buffer = deque(maxlen=window_size)
        
        # Anomaly counters
        self.total_samples = 0
        self.total_anomalies = 0
        
        # Color scheme
        self.colors = {
            'normal': '#2ECC71',
            'anomaly': '#E74C3C',
            'warning': '#F39C12'
        }
        
        self.logger.info("RealtimeDashboard initialized")
    
    def update_data(
        self,
        timestamp: float,
        vibration: float,
        temperature: float,
        pressure: float,
        current: float,
        is_anomaly: int = 0
    ):
        """
        Update dashboard with new data point
        
        Args:
            timestamp: Unix timestamp
            vibration: Vibration RMS value
            temperature: Temperature value
            pressure: Pressure value
            current: Current value
            is_anomaly: Anomaly flag (0 or 1)
        """
        self.time_buffer.append(timestamp)
        self.vibration_buffer.append(vibration)
        self.temperature_buffer.append(temperature)
        self.pressure_buffer.append(pressure)
        self.current_buffer.append(current)
        self.anomaly_buffer.append(is_anomaly)
        
        self.total_samples += 1
        if is_anomaly:
            self.total_anomalies += 1
    
    def create_dashboard(self) -> plt.Figure:
        """
        Create dashboard layout
        
        Returns:
            Figure object
        """
        self.logger.info("Creating dashboard layout")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax_vibration = fig.add_subplot(gs[0, :2])
        self.ax_temperature = fig.add_subplot(gs[1, :2])
        self.ax_pressure = fig.add_subplot(gs[2, :2])
        self.ax_current = fig.add_subplot(gs[3, :2])
        
        # Statistics panel
        self.ax_stats = fig.add_subplot(gs[0:2, 2])
        self.ax_stats.axis('off')
        
        # Anomaly indicator
        self.ax_anomaly = fig.add_subplot(gs[2:4, 2])
        self.ax_anomaly.axis('off')
        
        # Configure sensor plots
        for ax, label in zip(
            [self.ax_vibration, self.ax_temperature, self.ax_pressure, self.ax_current],
            ['Vibration RMS', 'Temperature (°C)', 'Pressure (PSI)', 'Current (A)']
        ):
            ax.set_ylabel(label, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.window_size)
        
        self.ax_current.set_xlabel('Sample', fontsize=10, fontweight='bold')
        
        fig.suptitle('Real-Time IoT Sensor Monitoring Dashboard', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def update_plots(self, frame):
        """
        Update callback for animation
        
        Args:
            frame: Frame number (unused, required by FuncAnimation)
        """
        # Clear plots
        self.ax_vibration.clear()
        self.ax_temperature.clear()
        self.ax_pressure.clear()
        self.ax_current.clear()
        
        if len(self.time_buffer) == 0:
            return
        
        # Prepare data
        x = list(range(len(self.time_buffer)))
        
        # Plot sensor data with anomaly highlighting
        for ax, buffer, label, threshold in zip(
            [self.ax_vibration, self.ax_temperature, self.ax_pressure, self.ax_current],
            [self.vibration_buffer, self.temperature_buffer, self.pressure_buffer, self.current_buffer],
            ['Vibration RMS', 'Temperature (°C)', 'Pressure (PSI)', 'Current (A)'],
            [10.0, 95.0, 150.0, 18.0]  # Thresholds
        ):
            data = list(buffer)
            anomalies = list(self.anomaly_buffer)
            
            # Plot normal data
            normal_x = [x[i] for i in range(len(x)) if anomalies[i] == 0]
            normal_y = [data[i] for i in range(len(data)) if anomalies[i] == 0]
            ax.plot(normal_x, normal_y, color=self.colors['normal'], 
                   linewidth=1.5, alpha=0.8)
            
            # Plot anomalies
            anomaly_x = [x[i] for i in range(len(x)) if anomalies[i] == 1]
            anomaly_y = [data[i] for i in range(len(data)) if anomalies[i] == 1]
            if anomaly_x:
                ax.scatter(anomaly_x, anomaly_y, color=self.colors['anomaly'],
                          s=30, zorder=5, alpha=0.9, marker='x', linewidths=2)
            
            # Add threshold line
            ax.axhline(threshold, color=self.colors['warning'], 
                      linestyle='--', linewidth=2, alpha=0.6,
                      label=f'Threshold: {threshold}')
            
            ax.set_ylabel(label, fontsize=10, fontweight='bold')
            ax.set_xlim(0, self.window_size)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        self.ax_current.set_xlabel('Sample', fontsize=10, fontweight='bold')
        
        # Update statistics panel
        self._update_stats_panel()
        
        # Update anomaly indicator
        self._update_anomaly_indicator()
    
    def _update_stats_panel(self):
        """Update statistics text panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if len(self.vibration_buffer) == 0:
            return
        
        # Calculate statistics
        stats_text = "STATISTICS\n" + "="*30 + "\n\n"
        stats_text += f"Total Samples: {self.total_samples:,}\n"
        stats_text += f"Window Size: {len(self.time_buffer)}\n\n"
        
        # Current values
        stats_text += "CURRENT VALUES\n" + "-"*30 + "\n"
        stats_text += f"Vibration: {self.vibration_buffer[-1]:.2f}\n"
        stats_text += f"Temperature: {self.temperature_buffer[-1]:.2f}°C\n"
        stats_text += f"Pressure: {self.pressure_buffer[-1]:.2f} PSI\n"
        stats_text += f"Current: {self.current_buffer[-1]:.2f} A\n\n"
        
        # Statistics
        stats_text += "WINDOW STATISTICS\n" + "-"*30 + "\n"
        stats_text += f"Vibration Mean: {np.mean(self.vibration_buffer):.2f}\n"
        stats_text += f"Vibration Std: {np.std(self.vibration_buffer):.2f}\n\n"
        stats_text += f"Temp Mean: {np.mean(self.temperature_buffer):.2f}°C\n"
        stats_text += f"Temp Std: {np.std(self.temperature_buffer):.2f}°C\n\n"
        
        # Anomaly rate
        anomaly_rate = (self.total_anomalies / self.total_samples * 100) if self.total_samples > 0 else 0
        stats_text += f"Anomaly Rate: {anomaly_rate:.2f}%\n"
        stats_text += f"Total Anomalies: {self.total_anomalies}"
        
        self.ax_stats.text(0.05, 0.95, stats_text,
                          transform=self.ax_stats.transAxes,
                          fontsize=9,
                          verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def _update_anomaly_indicator(self):
        """Update anomaly status indicator"""
        self.ax_anomaly.clear()
        self.ax_anomaly.axis('off')
        
        if len(self.anomaly_buffer) == 0:
            return
        
        # Check if latest reading is anomaly
        is_current_anomaly = self.anomaly_buffer[-1] == 1
        
        # Create status indicator
        if is_current_anomaly:
            status_text = "⚠ ANOMALY DETECTED"
            status_color = self.colors['anomaly']
            detail_text = "ALERT: Abnormal sensor reading detected!\n\n"
            detail_text += "Recommended Actions:\n"
            detail_text += "• Inspect equipment immediately\n"
            detail_text += "• Check sensor calibration\n"
            detail_text += "• Review maintenance logs\n"
            detail_text += "• Notify maintenance team"
        else:
            status_text = "✓ NORMAL OPERATION"
            status_color = self.colors['normal']
            detail_text = "All systems operating within\nnormal parameters.\n\n"
            detail_text += "Status: HEALTHY\n"
            detail_text += "No action required."
        
        # Status box
        self.ax_anomaly.text(0.5, 0.85, status_text,
                            transform=self.ax_anomaly.transAxes,
                            fontsize=14,
                            fontweight='bold',
                            ha='center',
                            color='white',
                            bbox=dict(boxstyle='round,pad=0.8',
                                    facecolor=status_color,
                                    edgecolor='black',
                                    linewidth=2))
        
        # Details
        self.ax_anomaly.text(0.5, 0.45, detail_text,
                            transform=self.ax_anomaly.transAxes,
                            fontsize=9,
                            ha='center',
                            va='center',
                            bbox=dict(boxstyle='round',
                                    facecolor='lightyellow',
                                    alpha=0.5))
    
    def run_simulation(self, data_source: Callable):
        """
        Run dashboard with simulated data stream
        
        Args:
            data_source: Function that yields data tuples
                        (timestamp, vibration, temperature, pressure, current, is_anomaly)
        """
        fig = self.create_dashboard()
        
        def update_wrapper(frame):
            try:
                data = next(data_source)
                self.update_data(*data)
                self.update_plots(frame)
            except StopIteration:
                self.logger.info("Data stream ended")
        
        anim = FuncAnimation(fig, update_wrapper, interval=self.update_interval,
                           cache_frame_data=False)
        plt.show()
        
        return anim


# Example usage
if __name__ == "__main__":
    from ..config import get_config
    import time
    
    config = get_config()
    dashboard = RealtimeDashboard(config, window_size=500)
    
    # Simulate data stream
    def simulate_data():
        """Generate simulated sensor data"""
        t = 0
        while True:
            # Normal operation with some noise
            vibration = 2.0 + 0.5 * np.sin(t * 0.1) + np.random.randn() * 0.2
            temperature = 50 + 5 * np.sin(t * 0.05) + np.random.randn() * 1
            pressure = 100 + 10 * np.sin(t * 0.08) + np.random.randn() * 2
            current = 10 + 1 * np.sin(t * 0.06) + np.random.randn() * 0.3
            
            # Randomly inject anomalies
            is_anomaly = 1 if np.random.random() < 0.02 else 0
            if is_anomaly:
                vibration *= 3
                temperature += 20
            
            yield (time.time(), vibration, temperature, pressure, current, is_anomaly)
            t += 1
    
    print("Starting real-time dashboard simulation...")
    print("Close the window to stop.")
    
    dashboard.run_simulation(simulate_data())