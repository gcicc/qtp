"""
Feature Engineering Module

Provides technical indicators, fundamental ratios, and custom feature engineering
for quantitative trading strategies.

Key Components:
- TechnicalIndicators: Common technical analysis indicators (RSI, MACD, Bollinger Bands)
- FundamentalFeatures: Financial ratio calculations and fundamental analysis
- CustomFeatures: Advanced feature engineering and factor construction
- FeaturePipeline: Automated feature generation and preprocessing
"""

from .technical import TechnicalIndicators
from .fundamental import FundamentalFeatures  
from .custom import CustomFeatures
from .pipeline import FeaturePipeline

__all__ = [
    "TechnicalIndicators",
    "FundamentalFeatures",
    "CustomFeatures", 
    "FeaturePipeline"
]