"""
Comprehensive market data pipeline orchestration.

This module provides a unified data pipeline that orchestrates data flow from
ingestion through validation, feature engineering, and storage. It handles
real-time and batch processing with error handling, retry logic, and monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import json
import os

from .sources.yahoo_finance import YahooFinanceConnector
from .sources.alpha_vantage import AlphaVantageConnector
from .sources.websocket_feeds import WebSocketDataFeed, WebSocketConfig
from .sources.mock_data import MockDataGenerator
from .validation import EnhancedDataValidator
from .features import FeatureEngineering
from .ingestion.base import DataSourceConfig
from .structures import OHLCV, Trade, Quote, MarketEvent

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution statuses."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class DataSourceType(Enum):
    """Supported data source types."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    WEBSOCKET = "websocket"
    MOCK = "mock"


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    # Data sources
    enabled_sources: List[DataSourceType] = field(default_factory=lambda: [DataSourceType.YAHOO_FINANCE])
    source_configs: Dict[DataSourceType, DataSourceConfig] = field(default_factory=dict)
    
    # Processing settings
    batch_size: int = 100
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay: int = 5
    
    # Validation settings
    enable_validation: bool = True
    enable_cross_validation: bool = True
    validation_config: Optional[Dict[str, Any]] = None
    
    # Feature engineering settings
    enable_feature_engineering: bool = True
    feature_config: Optional[Dict[str, Any]] = None
    
    # Storage settings
    cache_directory: str = "data/cache"
    processed_directory: str = "data/processed"
    enable_caching: bool = True
    cache_expiry_hours: int = 24
    
    # Real-time settings
    enable_real_time: bool = False
    real_time_symbols: List[str] = field(default_factory=list)
    real_time_interval: int = 1  # seconds
    
    # Monitoring settings
    enable_monitoring: bool = True
    log_level: str = "INFO"
    max_log_size_mb: int = 100


class DataPipelineManager:
    """
    Comprehensive data pipeline manager.
    
    Orchestrates the entire data pipeline from ingestion to storage:
    - Multi-source data ingestion with failover
    - Data validation and quality checks
    - Feature engineering and transformation
    - Caching and storage management
    - Real-time data processing
    - Error handling and monitoring
    - Performance optimization
    
    Example:
        >>> config = PipelineConfig(
        ...     enabled_sources=[DataSourceType.YAHOO_FINANCE, DataSourceType.MOCK],
        ...     enable_validation=True,
        ...     enable_feature_engineering=True
        ... )
        >>> pipeline = DataPipelineManager(config)
        >>> result = await pipeline.process_symbols(["AAPL", "MSFT"], "2023-01-01", "2023-12-31")
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize data pipeline manager.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.status = PipelineStatus.IDLE
        
        # Initialize components
        self.data_sources = self._initialize_data_sources()
        self.validator = EnhancedDataValidator(config.validation_config) if config.enable_validation else None
        self.feature_engineer = FeatureEngineering(config.feature_config) if config.enable_feature_engineering else None
        
        # Storage and caching
        self._ensure_directories()
        self.cache = {}
        
        # Real-time processing
        self.real_time_queue = Queue()
        self.real_time_handlers = []
        self.websocket_manager = None
        
        # Monitoring and statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'validation_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'last_update': None,
            'errors': []
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.lock = threading.Lock()
        
        logger.info("Data pipeline manager initialized")
    
    async def process_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple symbols through the complete pipeline.
        
        Orchestrates data ingestion, validation, feature engineering,
        and storage for multiple symbols with parallel processing.
        
        Args:
            symbols: List of symbols to process
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data frequency
            force_refresh: Force refresh of cached data
            
        Returns:
            Dictionary mapping symbols to processed DataFrames
            
        Example:
            >>> pipeline = DataPipelineManager(config)
            >>> results = await pipeline.process_symbols(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     "2023-01-01",
            ...     "2023-12-31"
            ... )
            >>> print(f"Processed {len(results)} symbols")
        """
        start_time = time.time()
        self.status = PipelineStatus.RUNNING
        results = {}
        
        try:
            logger.info(f"Starting pipeline processing for {len(symbols)} symbols")
            
            # Process symbols in batches
            batches = [symbols[i:i + self.config.batch_size] 
                      for i in range(0, len(symbols), self.config.batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} symbols")
                
                # Submit batch for parallel processing
                futures = []
                for symbol in batch:
                    future = self.executor.submit(
                        self._process_single_symbol,
                        symbol, start_date, end_date, interval, force_refresh
                    )
                    futures.append((symbol, future))
                
                # Collect results
                for symbol, future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        if result is not None:
                            results[symbol] = result
                            logger.info(f"Successfully processed {symbol}")
                        else:
                            logger.warning(f"No data returned for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process {symbol}: {e}")
                        self._record_error(symbol, e)
                
                # Small delay between batches to avoid overwhelming APIs
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(1)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(symbols), len(results), processing_time)
            
            logger.info(f"Pipeline processing completed: {len(results)}/{len(symbols)} symbols successful")
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            self.status = PipelineStatus.ERROR
            raise
        finally:
            self.status = PipelineStatus.IDLE
        
        return results
    
    def _process_single_symbol(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """Process a single symbol through the complete pipeline."""
        try:
            with self.lock:
                self.stats['total_requests'] += 1
            
            # Check cache first
            if not force_refresh:
                cached_data = self._get_cached_data(symbol, start_date, end_date, interval)
                if cached_data is not None:
                    with self.lock:
                        self.stats['cache_hits'] += 1
                    logger.debug(f"Using cached data for {symbol}")
                    return cached_data
            
            with self.lock:
                self.stats['cache_misses'] += 1
            
            # Ingest raw data
            raw_data = self._ingest_data(symbol, start_date, end_date, interval)
            if raw_data is None or raw_data.empty:
                logger.warning(f"No raw data available for {symbol}")
                return None
            
            # Validate data
            validated_data = self._validate_data(raw_data, symbol)
            if validated_data is None or validated_data.empty:
                logger.warning(f"Data validation failed for {symbol}")
                with self.lock:
                    self.stats['validation_failures'] += 1
                return None
            
            # Feature engineering
            processed_data = self._engineer_features(validated_data, symbol)
            
            # Cache results
            if self.config.enable_caching:
                self._cache_data(symbol, start_date, end_date, interval, processed_data)
            
            # Store processed data
            self._store_data(symbol, processed_data, interval)
            
            with self.lock:
                self.stats['successful_requests'] += 1
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Processing failed for {symbol}: {e}")
            with self.lock:
                self.stats['failed_requests'] += 1
            self._record_error(symbol, e)
            return None
    
    def _ingest_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Ingest data from configured sources with failover."""
        last_error = None
        
        for source_type in self.config.enabled_sources:
            source = self.data_sources.get(source_type)
            if source is None:
                continue
            
            for attempt in range(self.config.retry_attempts):
                try:
                    logger.debug(f"Attempting data ingestion for {symbol} from {source_type.value} (attempt {attempt + 1})")
                    
                    data = source.get_historical_data(symbol, start_date, end_date, interval)
                    
                    if data is not None and not data.empty:
                        logger.info(f"Successfully ingested {len(data)} records for {symbol} from {source_type.value}")
                        return data
                    else:
                        logger.warning(f"Empty data returned for {symbol} from {source_type.value}")
                        
                except Exception as e:
                    last_error = e
                    logger.warning(f"Data ingestion failed for {symbol} from {source_type.value}: {e}")
                    
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
        
        logger.error(f"All data sources failed for {symbol}. Last error: {last_error}")
        return None
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Validate data quality and integrity."""
        if not self.config.enable_validation or self.validator is None:
            return data
        
        try:
            # Perform basic validation
            validation_result = self.validator.validate_market_data_enhanced(data, symbol)
            
            if not validation_result.is_valid:
                logger.error(f"Data validation failed for {symbol}: {validation_result.errors}")
                return None
            
            if validation_result.warnings:
                logger.warning(f"Data validation warnings for {symbol}: {validation_result.warnings}")
            
            # Use cleaned data if available
            if validation_result.cleaned_data is not None:
                return validation_result.cleaned_data
            
            return data
            
        except Exception as e:
            logger.error(f"Data validation error for {symbol}: {e}")
            return None
    
    def _engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply feature engineering to the data."""
        if not self.config.enable_feature_engineering or self.feature_engineer is None:
            return data
        
        try:
            features = self.feature_engineer.create_comprehensive_features(data, symbol)
            
            # Combine original data with features
            combined_data = pd.concat([data, features], axis=1)
            
            logger.debug(f"Created {features.shape[1]} features for {symbol}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {symbol}: {e}")
            # Return original data if feature engineering fails
            return data
    
    def _get_cached_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if available and fresh."""
        if not self.config.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval)
        cache_file = os.path.join(self.config.cache_directory, f"{cache_key}.parquet")
        
        try:
            if os.path.exists(cache_file):
                # Check cache expiry
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                expiry_time = datetime.now() - timedelta(hours=self.config.cache_expiry_hours)
                
                if file_time > expiry_time:
                    data = pd.read_parquet(cache_file)
                    logger.debug(f"Cache hit for {symbol}")
                    return data
                else:
                    logger.debug(f"Cache expired for {symbol}")
                    os.remove(cache_file)
            
        except Exception as e:
            logger.warning(f"Cache read failed for {symbol}: {e}")
        
        return None
    
    def _cache_data(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str,
        data: pd.DataFrame
    ) -> None:
        """Cache processed data for future use."""
        if not self.config.enable_caching:
            return
        
        try:
            cache_key = self._generate_cache_key(symbol, start_date, end_date, interval)
            cache_file = os.path.join(self.config.cache_directory, f"{cache_key}.parquet")
            
            data.to_parquet(cache_file, compression='snappy')
            logger.debug(f"Cached data for {symbol}")
            
        except Exception as e:
            logger.warning(f"Cache write failed for {symbol}: {e}")
    
    def _store_data(self, symbol: str, data: pd.DataFrame, interval: str) -> None:
        """Store processed data to persistent storage."""
        try:
            # Create subdirectories
            interval_dir = os.path.join(self.config.processed_directory, interval)
            os.makedirs(interval_dir, exist_ok=True)
            
            # Save as parquet
            file_path = os.path.join(interval_dir, f"{symbol}.parquet")
            data.to_parquet(file_path, compression='snappy')
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'interval': interval,
                'records': len(data),
                'columns': list(data.columns),
                'start_date': data.index.min().isoformat(),
                'end_date': data.index.max().isoformat(),
                'processed_at': datetime.now().isoformat()
            }
            
            metadata_file = os.path.join(interval_dir, f"{symbol}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Stored processed data for {symbol}")
            
        except Exception as e:
            logger.warning(f"Data storage failed for {symbol}: {e}")
    
    async def start_real_time_processing(self) -> None:
        """Start real-time data processing."""
        if not self.config.enable_real_time or not self.config.real_time_symbols:
            logger.info("Real-time processing not enabled")
            return
        
        try:
            logger.info("Starting real-time data processing")
            
            # Setup WebSocket connections for real-time data
            if DataSourceType.WEBSOCKET in self.config.enabled_sources:
                await self._setup_websocket_feeds()
            
            # Start real-time processing loop
            asyncio.create_task(self._real_time_processing_loop())
            
            logger.info("Real-time processing started")
            
        except Exception as e:
            logger.error(f"Failed to start real-time processing: {e}")
            raise
    
    async def stop_real_time_processing(self) -> None:
        """Stop real-time data processing."""
        try:
            if self.websocket_manager:
                await self.websocket_manager.stop_all()
            
            logger.info("Real-time processing stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop real-time processing: {e}")
    
    def add_real_time_handler(self, handler: Callable[[str, pd.DataFrame], None]) -> None:
        """Add handler for real-time data updates."""
        self.real_time_handlers.append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics and performance metrics."""
        with self.lock:
            stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['failure_rate'] = stats['failed_requests'] / stats['total_requests']
            stats['validation_failure_rate'] = stats['validation_failures'] / stats['total_requests']
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['median_processing_time'] = np.median(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
        
        stats['cache_hit_rate'] = (stats['cache_hits'] / 
                                 (stats['cache_hits'] + stats['cache_misses'])) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        with self.lock:
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'validation_failures': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'processing_times': [],
                'last_update': None,
                'errors': []
            }
        
        logger.info("Pipeline statistics reset")
    
    def _initialize_data_sources(self) -> Dict[DataSourceType, Any]:
        """Initialize configured data sources."""
        sources = {}
        
        for source_type in self.config.enabled_sources:
            try:
                if source_type == DataSourceType.YAHOO_FINANCE:
                    config = self.config.source_configs.get(source_type, DataSourceConfig())
                    sources[source_type] = YahooFinanceConnector(config)
                    
                elif source_type == DataSourceType.ALPHA_VANTAGE:
                    config = self.config.source_configs.get(source_type)
                    if config and config.api_key:
                        sources[source_type] = AlphaVantageConnector(config)
                    else:
                        logger.warning("Alpha Vantage requires API key")
                        
                elif source_type == DataSourceType.MOCK:
                    sources[source_type] = MockDataGenerator()
                    
                logger.info(f"Initialized {source_type.value} data source")
                
            except Exception as e:
                logger.error(f"Failed to initialize {source_type.value}: {e}")
        
        return sources
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.config.cache_directory,
            self.config.processed_directory
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _generate_cache_key(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str
    ) -> str:
        """Generate cache key for data."""
        start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
        return f"{symbol}_{start_str}_{end_str}_{interval}"
    
    def _update_stats(self, total_symbols: int, successful_symbols: int, processing_time: float) -> None:
        """Update pipeline statistics."""
        with self.lock:
            self.stats['processing_times'].append(processing_time)
            self.stats['last_update'] = datetime.now()
            
            # Keep only recent processing times
            if len(self.stats['processing_times']) > 100:
                self.stats['processing_times'] = self.stats['processing_times'][-100:]
    
    def _record_error(self, symbol: str, error: Exception) -> None:
        """Record error for monitoring."""
        with self.lock:
            error_record = {
                'symbol': symbol,
                'error': str(error),
                'timestamp': datetime.now().isoformat()
            }
            self.stats['errors'].append(error_record)
            
            # Keep only recent errors
            if len(self.stats['errors']) > 100:
                self.stats['errors'] = self.stats['errors'][-100:]
    
    async def _setup_websocket_feeds(self) -> None:
        """Setup WebSocket feeds for real-time data."""
        # This would be implemented based on specific WebSocket providers
        # For now, this is a placeholder
        pass
    
    async def _real_time_processing_loop(self) -> None:
        """Main loop for real-time data processing."""
        while self.config.enable_real_time:
            try:
                # Process queued real-time data
                while not self.real_time_queue.empty():
                    try:
                        data_item = self.real_time_queue.get_nowait()
                        await self._process_real_time_data(data_item)
                    except Empty:
                        break
                
                await asyncio.sleep(self.config.real_time_interval)
                
            except Exception as e:
                logger.error(f"Real-time processing error: {e}")
                await asyncio.sleep(5)  # Error backoff
    
    async def _process_real_time_data(self, data_item: Any) -> None:
        """Process individual real-time data item."""
        try:
            # Extract symbol and data from the item
            symbol = data_item.get('symbol')
            
            if symbol:
                # Create DataFrame from real-time data
                df = self._convert_real_time_to_dataframe(data_item)
                
                # Apply real-time handlers
                for handler in self.real_time_handlers:
                    try:
                        handler(symbol, df)
                    except Exception as e:
                        logger.error(f"Real-time handler error for {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"Real-time data processing error: {e}")
    
    def _convert_real_time_to_dataframe(self, data_item: Any) -> pd.DataFrame:
        """Convert real-time data item to DataFrame."""
        # This would depend on the specific real-time data format
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass