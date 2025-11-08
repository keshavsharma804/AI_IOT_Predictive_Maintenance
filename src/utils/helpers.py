"""
Helper Utilities
Common utility functions used across the project
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
from pathlib import Path
import json
import pickle
import joblib
from datetime import datetime, timedelta


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Load JSON file
    
    Args:
        filepath: JSON file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj, filepath: Union[str, Path]) -> None:
    """
    Save object using pickle
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]):
    """
    Load pickled object
    
    Args:
        filepath: Pickle file path
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_model(model, filepath: Union[str, Path]) -> None:
    """
    Save model using joblib (efficient for sklearn models)
    
    Args:
        model: Model to save
        filepath: Output file path
    """
    ensure_dir(Path(filepath).parent)
    joblib.dump(model, filepath)


def load_model(filepath: Union[str, Path]):
    """
    Load model from joblib file
    
    Args:
        filepath: Model file path
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def timestamp_to_string(timestamp: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Convert timestamp to string
    
    Args:
        timestamp: Datetime object
        format: String format
        
    Returns:
        Formatted string
    """
    return timestamp.strftime(format)


def string_to_timestamp(date_string: str, format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Convert string to timestamp
    
    Args:
        date_string: Date string
        format: String format
        
    Returns:
        Datetime object
    """
    return datetime.strptime(date_string, format)


def calculate_memory_usage(df: pd.DataFrame) -> Tuple[float, dict]:
    """
    Calculate memory usage of DataFrame
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Total memory in MB and per-column breakdown
    """
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / 1024**2
    
    breakdown = {
        col: f"{mem / 1024**2:.2f} MB"
        for col, mem in memory_usage.items()
    }
    
    return total_mb, breakdown


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize array
    
    Args:
        arr: Input array
        method: 'minmax' or 'zscore'
        
    Returns:
        Normalized array
    """
    if method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val + 1e-8)
    elif method == 'zscore':
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def split_time_series(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train/val/test
    
    Args:
        df: DataFrame with time series
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Splits must sum to 1.0"
    
    n = len(df)
    train_idx = int(n * train_size)
    val_idx = int(n * (train_size + val_size))
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    return train_df, val_df, test_df


def print_section_header(title: str, char: str = '=', width: int = 80) -> None:
    """
    Print formatted section header
    
    Args:
        title: Section title
        char: Character for border
        width: Total width
    """
    print('\n' + char * width)
    print(f"{title:^{width}}")
    print(char * width + '\n')


# Example usage
if __name__ == "__main__":
    print_section_header("Helper Utilities Test")
    
    # Test directory creation
    test_dir = ensure_dir("test_output/subdir")
    print(f"Created directory: {test_dir}")
    
    # Test JSON save/load
    test_data = {"key": "value", "number": 42}
    save_json(test_data, "test_output/test.json")
    loaded = load_json("test_output/test.json")
    print(f"JSON test: {loaded}")
    
    # Test memory calculation
    df = pd.DataFrame(np.random.rand(1000, 10))
    total_mb, breakdown = calculate_memory_usage(df)
    print(f"DataFrame memory: {total_mb:.2f} MB")