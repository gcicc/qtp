"""
Pytest configuration and shared fixtures for QTP testing.

Provides common test fixtures, configuration, and utilities used across
all test modules. Ensures consistent test environment setup.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import tempfile
import os

# Test data generation utilities
def generate_sample_market_data(
    symbols: list = None,
    start_date: datetime = None,
    end_date: datetime = None,
    frequency: str = "1D"
) -> Dict[str, pd.DataFrame]:
    """
    Generate sample market data for testing.
    
    Args:
        symbols: List of symbol names (default: ['AAPL', 'MSFT', 'GOOGL'])
        start_date: Start date for data (default: 1 year ago)
        end_date: End date for data (default: today)
        frequency: Data frequency (default: daily)
        
    Returns:
        Dictionary mapping symbols to OHLCV DataFrames
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    if end_date is None:
        end_date = datetime.now()
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    market_data = {}
    
    for symbol in symbols:
        # Generate realistic price data with random walk
        num_periods = len(date_range)
        initial_price = np.random.uniform(50, 200)  # Starting price
        
        # Random walk for price movement
        returns = np.random.normal(0.001, 0.02, num_periods)  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(0.01, new_price))  # Ensure positive prices
        
        prices = np.array(prices)
        
        # Generate OHLC from closing prices
        # Open: previous close with small gap
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        opens = opens * np.random.normal(1.0, 0.005, num_periods)
        
        # High: max of open/close plus some upward movement
        highs = np.maximum(opens, prices) * np.random.uniform(1.0, 1.03, num_periods)
        
        # Low: min of open/close minus some downward movement  
        lows = np.minimum(opens, prices) * np.random.uniform(0.97, 1.0, num_periods)
        
        # Volume: random but realistic
        volumes = np.random.lognormal(15, 1, num_periods).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=date_range)
        
        market_data[symbol] = df
    
    return market_data


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for testing."""
    return generate_sample_market_data()


@pytest.fixture 
def single_symbol_data():
    """Fixture providing data for a single symbol."""
    data = generate_sample_market_data(symbols=['TEST'])
    return data['TEST']


@pytest.fixture
def strategy_config():
    """Fixture providing a basic strategy configuration."""
    from src.strategies.base import StrategyConfig
    
    return StrategyConfig(
        name="TestStrategy",
        description="Strategy for testing purposes",
        max_position_size=0.1,
        lookback_window=20,
        parameters={
            'test_param': 42,
            'another_param': 'test_value'
        }
    )


@pytest.fixture
def agent_config():
    """Fixture providing a basic agent configuration."""
    from src.agents.base import AgentConfig
    
    return AgentConfig(
        name="TestAgent",
        agent_type="test",
        description="Agent for testing purposes",
        max_concurrent_tasks=3,
        parameters={
            'test_setting': True,
            'threshold': 0.5
        }
    )


@pytest.fixture
def temp_data_dir():
    """Fixture providing a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_data_validator():
    """Fixture providing a mock data validator."""
    from src.data.validation import ValidationResult
    
    class MockValidator:
        def validate_market_data(self, data, symbol):
            result = ValidationResult()
            result.is_valid = True
            result.cleaned_data = data
            return result
    
    return MockValidator()


@pytest.fixture
def sample_signals():
    """Fixture providing sample trading signals."""
    from src.strategies.base import Signal, SignalType, SignalStrength
    
    signals = []
    
    # Buy signal
    signals.append(Signal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        timestamp=datetime.now(),
        price=150.0,
        position_size=0.05,
        confidence=0.8,
        strength=SignalStrength.STRONG,
        strategy_name="TestStrategy",
        reason="Test buy signal"
    ))
    
    # Sell signal
    signals.append(Signal(
        symbol="MSFT",
        signal_type=SignalType.SELL,
        timestamp=datetime.now(),
        price=300.0,
        position_size=-0.03,
        confidence=0.6,
        strength=SignalStrength.MODERATE,
        strategy_name="TestStrategy", 
        reason="Test sell signal"
    ))
    
    return signals


@pytest.fixture(scope="session")
def test_database():
    """Session-scoped fixture for test database setup."""
    # This would set up a test database if needed
    # For now, just return a placeholder
    return {"status": "mock_db_ready"}


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require external data"
    )


# Custom test utilities
class TestUtils:
    """Utility class for common testing operations."""
    
    @staticmethod
    def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype=True):
        """Assert that two DataFrames are equal with better error messages."""
        try:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
        except AssertionError as e:
            print(f"DataFrame comparison failed:\n{e}")
            print(f"DF1 shape: {df1.shape}, DF2 shape: {df2.shape}")
            print(f"DF1 columns: {list(df1.columns)}")
            print(f"DF2 columns: {list(df2.columns)}")
            raise
    
    @staticmethod
    def assert_signal_valid(signal):
        """Assert that a signal object is valid."""
        from src.strategies.base import Signal, SignalType
        
        assert isinstance(signal, Signal)
        assert isinstance(signal.signal_type, SignalType)
        assert signal.price > 0
        assert 0 <= signal.confidence <= 1
        assert signal.symbol is not None
        assert signal.timestamp is not None


@pytest.fixture
def test_utils():
    """Fixture providing test utility functions."""
    return TestUtils()