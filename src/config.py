"""
Configuration Management Module
Loads and manages project configuration from YAML file
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        paths = self.config.get('paths', {})
        
        for path_name, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}')"


# Global config instance
_config = None


def get_config(config_path: str = "config/config.yaml") -> Config:
    """
    Get global configuration instance (Singleton pattern)
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = "config/config.yaml"):
    """
    Reload configuration (useful for testing)
    
    Args:
        config_path: Path to configuration file
    """
    global _config
    _config = Config(config_path)
    return _config


# Example usage
if __name__ == "__main__":
    # Load config
    config = get_config()
    
    # Access configuration
    print("Project Name:", config.get('project.name'))
    print("Sampling Rate:", config.get('data_generation.sampling_rate'))
    print("Models:", config.get('models'))
    
    # Access with default value
    print("Unknown Key:", config.get('unknown.key', 'default_value'))