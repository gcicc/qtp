"""
Explanation Engine

Natural language explanation system that provides detailed rationale for trading
decisions, market context, and educational content. Core to QTP's transparency mission.

Key Features:
- Multi-level explanations (novice to expert)
- Causal reasoning for market movements
- Statistical significance communication
- Risk-return trade-off articulation
- Historical precedent identification

Components:
- ExplanationEngine: Main explanation generation system
- SignalExplainer: Detailed signal rationale generation
- MarketContextualizer: Market regime and condition explanation
- EducationalContentGenerator: Learning-focused explanations
- TemplateManager: Explanation template management
"""

from .engine import ExplanationEngine
from .signal_explainer import SignalExplainer
from .market_context import MarketContextualizer
from .educational import EducationalContentGenerator
from .templates import ExplanationTemplate, TemplateManager

__all__ = [
    "ExplanationEngine",
    "SignalExplainer",
    "MarketContextualizer", 
    "EducationalContentGenerator",
    "ExplanationTemplate",
    "TemplateManager"
]