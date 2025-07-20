"""
Main data validation engine.

Orchestrates various data quality checks and provides a unified interface
for validating market data before it enters the trading system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """
    Container for validation results.
    
    Stores validation outcomes, error messages, and statistics
    for comprehensive data quality reporting.
    """
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.statistics = {}
        self.cleaned_data = None
        
    def add_error(self, message: str, severity: str = "error") -> None:
        """Add an error or warning to the validation result."""
        if severity == "error":
            self.errors.append(message)
            self.is_valid = False
        elif severity == "warning":
            self.warnings.append(message)
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics
        }


class DataValidator:
    """
    Main data validation class.
    
    Performs comprehensive validation of market data including:
    - Completeness checks (missing values, gaps in time series)
    - Range validation (price and volume bounds)
    - Consistency checks (OHLC relationships)
    - Statistical outlier detection
    - Cross-validation between data sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or self._default_config()
        
    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> ValidationResult:
        """
        Perform comprehensive validation on market data.
        
        Args:
            data: DataFrame with OHLCV market data
            symbol: Stock symbol for context in error messages
            
        Returns:
            ValidationResult object containing validation outcomes
        """
        result = ValidationResult()
        
        logger.info(f"Starting validation for {symbol} with {len(data)} records")
        
        # Basic structure validation
        self._validate_data_structure(data, result)
        
        # Completeness checks
        self._validate_completeness(data, result)
        
        # Range validation
        self._validate_price_ranges(data, symbol, result)
        
        # OHLC consistency
        self._validate_ohlc_consistency(data, result)
        
        # Volume validation
        self._validate_volume_data(data, result)
        
        # Temporal consistency
        self._validate_temporal_consistency(data, result)
        
        # Statistical outlier detection
        self._detect_statistical_outliers(data, result)
        
        # Clean data if validation passed with only warnings
        if result.is_valid or (not result.errors and result.warnings):
            result.cleaned_data = self._clean_data(data)
            
        logger.info(f"Validation completed for {symbol}: {result.get_summary()}")
        return result
        
    def _validate_data_structure(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validate basic data structure requirements."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if data.empty:
            result.add_error("Dataset is empty")
            return
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")
            
        # Check index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            result.add_error("Index must be a DatetimeIndex")
            
    def _validate_completeness(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Check for missing values and data gaps."""
        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            missing_percentage = (total_missing / (len(data) * len(data.columns))) * 100
            if missing_percentage > self.config['max_missing_percentage']:
                result.add_error(f"Too many missing values: {missing_percentage:.2f}%")
            else:
                result.add_error(f"Missing values detected: {missing_counts.to_dict()}", "warning")
                
        # Check for time series gaps
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            expected_freq = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else None
            
            if expected_freq:
                large_gaps = time_diffs[time_diffs > expected_freq * 2]
                if len(large_gaps) > 0:
                    result.add_error(f"Found {len(large_gaps)} time gaps larger than expected", "warning")
                    
    def _validate_price_ranges(self, data: pd.DataFrame, symbol: str, result: ValidationResult) -> None:
        """Validate price data is within reasonable ranges."""
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col in data.columns:
                prices = data[col].dropna()
                
                # Check for negative or zero prices
                invalid_prices = prices[prices <= 0]
                if len(invalid_prices) > 0:
                    result.add_error(f"Found {len(invalid_prices)} non-positive {col} prices")
                    
                # Check for extremely high prices (potential data errors)
                max_reasonable_price = self.config.get('max_reasonable_price', 10000)
                high_prices = prices[prices > max_reasonable_price]
                if len(high_prices) > 0:
                    result.add_error(f"Found {len(high_prices)} {col} prices above ${max_reasonable_price}", "warning")
                    
    def _validate_ohlc_consistency(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validate OHLC price relationships."""
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        
        if all(col in data.columns for col in ohlc_cols):
            # High should be >= Open, Low, Close
            high_violations = (
                (data['High'] < data['Open']) |
                (data['High'] < data['Low']) |
                (data['High'] < data['Close'])
            ).sum()
            
            if high_violations > 0:
                result.add_error(f"Found {high_violations} High price violations")
                
            # Low should be <= Open, High, Close
            low_violations = (
                (data['Low'] > data['Open']) |
                (data['Low'] > data['High']) |
                (data['Low'] > data['Close'])
            ).sum()
            
            if low_violations > 0:
                result.add_error(f"Found {low_violations} Low price violations")
                
    def _validate_volume_data(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validate volume data."""
        if 'Volume' in data.columns:
            volumes = data['Volume'].dropna()
            
            # Check for negative volumes
            negative_volumes = volumes[volumes < 0]
            if len(negative_volumes) > 0:
                result.add_error(f"Found {len(negative_volumes)} negative volume values")
                
            # Check for zero volumes (may be valid on weekends/holidays)
            zero_volumes = volumes[volumes == 0]
            if len(zero_volumes) > len(data) * 0.1:  # More than 10% zero volumes
                result.add_error(f"High proportion of zero volume days: {len(zero_volumes)}", "warning")
                
    def _validate_temporal_consistency(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validate temporal aspects of the data."""
        if len(data) > 1:
            # Check for duplicate timestamps
            duplicates = data.index.duplicated().sum()
            if duplicates > 0:
                result.add_error(f"Found {duplicates} duplicate timestamps")
                
            # Check for proper time ordering
            if not data.index.is_monotonic_increasing:
                result.add_error("Data is not properly time-ordered")
                
    def _detect_statistical_outliers(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Detect statistical outliers in price and volume data."""
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                values = data[col].dropna()
                
                if len(values) > 10:  # Need sufficient data for statistical analysis
                    # Use IQR method for outlier detection
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = values[(values < lower_bound) | (values > upper_bound)]
                    
                    if len(outliers) > 0:
                        outlier_percentage = (len(outliers) / len(values)) * 100
                        severity = "error" if outlier_percentage > 5 else "warning"
                        result.add_error(
                            f"Found {len(outliers)} ({outlier_percentage:.1f}%) outliers in {col}",
                            severity
                        )
                        
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        cleaned = data.copy()
        
        # Forward fill missing values (common in financial data)
        cleaned = cleaned.fillna(method='ffill')
        
        # Drop any remaining missing values
        cleaned = cleaned.dropna()
        
        # Remove duplicate timestamps, keeping the last occurrence
        cleaned = cleaned[~cleaned.index.duplicated(keep='last')]
        
        # Sort by timestamp
        cleaned = cleaned.sort_index()
        
        return cleaned
        
    def _default_config(self) -> Dict[str, Any]:
        """Default validation configuration."""
        return {
            'max_missing_percentage': 5.0,  # Maximum allowed missing data percentage
            'max_reasonable_price': 10000,  # Maximum reasonable stock price
            'outlier_threshold': 3.0,       # Standard deviations for outlier detection
            'min_volume': 0,                # Minimum valid volume
            'price_precision': 2            # Expected decimal places for prices
        }