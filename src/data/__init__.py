"""
Data Pipeline Module

Handles market data ingestion, validation, and feature engineering for the QTP platform.

This module provides comprehensive data management capabilities including:
- Real-time and historical market data ingestion from multiple sources
- Statistical data quality validation and completeness checks
- Technical indicators, fundamental ratios, and custom feature engineering
- Data preprocessing and normalization for machine learning models

Submodules:
- ingestion: Market data collection from various data providers
- validation: Data quality checks and statistical validation
- features: Technical indicators and feature engineering pipelines
"""

from .ingestion import *
from .validation import *
from .features import *

__all__ = [
    # Re-export key classes and functions from submodules
    "MarketDataIngester",
    "DataValidator", 
    "FeatureEngineer",
    "TechnicalIndicators"
]