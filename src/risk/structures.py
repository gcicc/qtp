"""
Risk management structures for the QTP platform.

This module defines data structures for portfolio management, risk metrics,
and risk monitoring alerts. All structures include comprehensive validation
and methods for risk calculation and assessment.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

from ..strategies.signals import Position


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of risk alerts."""
    POSITION_SIZE = "position_size"
    PORTFOLIO_VAR = "portfolio_var"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MARGIN = "margin"
    LIQUIDITY = "liquidity"
    STRESS_TEST = "stress_test"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class Portfolio(BaseModel):
    """
    Portfolio data structure with comprehensive risk metrics.
    
    Represents a trading portfolio with positions, cash, and calculated
    risk metrics for monitoring and management.
    
    Example:
        >>> portfolio = Portfolio(
        ...     name="Main Trading Portfolio",
        ...     cash_balance=Decimal("100000.00"),
        ...     positions=[position1, position2],
        ...     base_currency="USD"
        ... )
        >>> print(f"Total value: ${portfolio.total_value()}")
    """
    # Portfolio identification
    portfolio_id: Optional[str] = Field(None, description="Unique portfolio identifier")
    name: str = Field(..., description="Portfolio name")
    account_id: Optional[str] = Field(None, description="Associated account ID")
    
    # Holdings
    cash_balance: Decimal = Field(..., ge=0, description="Available cash balance")
    positions: List[Position] = Field(default_factory=list, description="Current positions")
    base_currency: str = Field(default="USD", description="Base currency")
    
    # Portfolio settings
    initial_capital: Decimal = Field(..., gt=0, description="Initial portfolio capital")
    max_position_size: Optional[float] = Field(None, gt=0, le=1, description="Max position as % of portfolio")
    max_sector_allocation: Optional[float] = Field(None, gt=0, le=1, description="Max sector allocation")
    target_leverage: Optional[float] = Field(default=1.0, gt=0, description="Target leverage ratio")
    
    # Risk parameters
    var_confidence: float = Field(default=0.95, gt=0, lt=1, description="VaR confidence level")
    max_drawdown_threshold: float = Field(default=0.20, gt=0, lt=1, description="Max drawdown threshold")
    correlation_threshold: float = Field(default=0.70, gt=0, lt=1, description="Correlation alert threshold")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Portfolio creation time")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional portfolio data")
    
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        total = Decimal('0')
        for position in self.positions:
            market_value = position.market_value()
            if market_value is not None:
                total += market_value
        return total
    
    def total_value(self) -> Decimal:
        """Calculate total portfolio value (cash + positions)."""
        return self.cash_balance + self.total_market_value()
    
    def net_liquidation_value(self) -> Decimal:
        """Calculate net liquidation value."""
        # For now, same as total value. Could include margin requirements in future
        return self.total_value()
    
    def total_pnl(self) -> Decimal:
        """Calculate total portfolio P&L."""
        total = Decimal('0')
        for position in self.positions:
            pnl = position.total_pnl()
            if pnl is not None:
                total += pnl
        return total
    
    def return_pct(self) -> float:
        """Calculate portfolio return percentage."""
        current_value = self.total_value()
        return float((current_value - self.initial_capital) / self.initial_capital)
    
    def leverage_ratio(self) -> float:
        """Calculate current leverage ratio."""
        net_value = self.net_liquidation_value()
        if net_value <= 0:
            return 0.0
        
        gross_exposure = sum(
            abs(pos.market_value() or Decimal('0'))
            for pos in self.positions
        )
        return float(gross_exposure / net_value)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def position_count(self) -> int:
        """Get number of positions."""
        return len([pos for pos in self.positions if pos.quantity != 0])
    
    def largest_position_pct(self) -> float:
        """Get largest position as percentage of portfolio."""
        if not self.positions:
            return 0.0
        
        total_value = self.total_value()
        if total_value <= 0:
            return 0.0
        
        max_position_value = max(
            abs(pos.market_value() or Decimal('0'))
            for pos in self.positions
        )
        return float(max_position_value / total_value)


class RiskMetrics(BaseModel):
    """
    Comprehensive risk metrics for a portfolio or position.
    
    Contains calculated risk measures including VaR, volatility,
    correlation metrics, and other risk indicators.
    
    Example:
        >>> metrics = RiskMetrics(
        ...     symbol="PORTFOLIO",
        ...     timestamp=datetime.now(),
        ...     var_1d=Decimal("5000.00"),
        ...     volatility_annualized=0.18,
        ...     sharpe_ratio=1.25,
        ...     max_drawdown=0.08
        ... )
    """
    # Identification
    symbol: str = Field(..., description="Symbol or portfolio identifier")
    timestamp: datetime = Field(..., description="Metrics calculation timestamp")
    lookback_period: int = Field(default=252, gt=0, description="Lookback period in days")
    
    # Value at Risk
    var_1d: Optional[Decimal] = Field(None, description="1-day Value at Risk")
    var_5d: Optional[Decimal] = Field(None, description="5-day Value at Risk")
    var_10d: Optional[Decimal] = Field(None, description="10-day Value at Risk")
    cvar_1d: Optional[Decimal] = Field(None, description="1-day Conditional VaR")
    confidence_level: float = Field(default=0.95, gt=0, lt=1, description="VaR confidence level")
    
    # Volatility metrics
    volatility_daily: Optional[float] = Field(None, ge=0, description="Daily volatility")
    volatility_annualized: Optional[float] = Field(None, ge=0, description="Annualized volatility")
    volatility_regime: Optional[str] = Field(None, description="Volatility regime (low/normal/high)")
    
    # Return metrics
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    information_ratio: Optional[float] = Field(None, description="Information ratio")
    
    # Drawdown metrics
    current_drawdown: Optional[float] = Field(None, ge=0, le=1, description="Current drawdown")
    max_drawdown: Optional[float] = Field(None, ge=0, le=1, description="Maximum drawdown")
    drawdown_duration: Optional[int] = Field(None, ge=0, description="Drawdown duration in days")
    recovery_time: Optional[int] = Field(None, ge=0, description="Time to recover from max drawdown")
    
    # Correlation and beta
    market_beta: Optional[float] = Field(None, description="Beta relative to market")
    correlation_to_market: Optional[float] = Field(None, ge=-1, le=1, description="Correlation to market")
    tracking_error: Optional[float] = Field(None, ge=0, description="Tracking error")
    
    # Concentration metrics
    concentration_herfindahl: Optional[float] = Field(None, ge=0, le=1, description="Herfindahl concentration index")
    top_positions_weight: Optional[float] = Field(None, ge=0, le=1, description="Weight of top 10 positions")
    sector_concentration: Optional[Dict[str, float]] = Field(None, description="Sector concentration")
    
    # Liquidity metrics
    avg_daily_volume: Optional[int] = Field(None, ge=0, description="Average daily volume")
    liquidity_score: Optional[float] = Field(None, ge=0, le=1, description="Liquidity score (0-1)")
    days_to_liquidate: Optional[float] = Field(None, ge=0, description="Estimated days to liquidate")
    
    # Stress test results
    stress_test_scenarios: Optional[Dict[str, Decimal]] = Field(None, description="Stress test P&L by scenario")
    tail_risk: Optional[Decimal] = Field(None, description="Tail risk estimate")
    
    # Additional metrics
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional risk metrics")
    
    def risk_level(self) -> RiskLevel:
        """Determine overall risk level based on metrics."""
        risk_indicators = []
        
        # Check drawdown
        if self.current_drawdown is not None:
            if self.current_drawdown > 0.15:
                risk_indicators.append("high_drawdown")
            elif self.current_drawdown > 0.10:
                risk_indicators.append("moderate_drawdown")
        
        # Check volatility
        if self.volatility_annualized is not None:
            if self.volatility_annualized > 0.30:
                risk_indicators.append("high_volatility")
            elif self.volatility_annualized > 0.20:
                risk_indicators.append("moderate_volatility")
        
        # Check concentration
        if self.concentration_herfindahl is not None:
            if self.concentration_herfindahl > 0.20:
                risk_indicators.append("high_concentration")
        
        # Determine overall risk level
        if len(risk_indicators) >= 3 or "high_drawdown" in risk_indicators:
            return RiskLevel.CRITICAL
        elif len(risk_indicators) >= 2:
            return RiskLevel.HIGH
        elif len(risk_indicators) >= 1:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def is_healthy(self) -> bool:
        """Check if risk metrics indicate a healthy portfolio."""
        return self.risk_level() in [RiskLevel.LOW, RiskLevel.MODERATE]


class Alert(BaseModel):
    """
    Risk management alert structure.
    
    Represents a risk alert with severity, trigger conditions,
    and recommended actions.
    
    Example:
        >>> alert = Alert(
        ...     alert_type=AlertType.DRAWDOWN,
        ...     severity=AlertSeverity.WARNING,
        ...     symbol="PORTFOLIO",
        ...     message="Current drawdown (12%) exceeds warning threshold (10%)",
        ...     triggered_value=0.12,
        ...     threshold=0.10,
        ...     recommended_actions=["Reduce position sizes", "Review strategy performance"]
        ... )
    """
    # Alert identification
    alert_id: Optional[str] = Field(None, description="Unique alert identifier")
    alert_type: AlertType = Field(..., description="Type of risk alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    
    # Alert context
    symbol: Optional[str] = Field(None, description="Affected symbol or portfolio")
    strategy_name: Optional[str] = Field(None, description="Related strategy")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")
    
    # Alert details
    message: str = Field(..., description="Human-readable alert message")
    triggered_value: Optional[Union[float, Decimal]] = Field(None, description="Value that triggered alert")
    threshold: Optional[Union[float, Decimal]] = Field(None, description="Alert threshold")
    current_risk_level: Optional[RiskLevel] = Field(None, description="Current risk level")
    
    # Actions and resolution
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    auto_actions_taken: List[str] = Field(default_factory=list, description="Automatic actions taken")
    is_acknowledged: bool = Field(default=False, description="Whether alert has been acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged alert")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    
    # Alert lifecycle
    is_active: bool = Field(default=True, description="Whether alert is still active")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolution_notes: str = Field(default="", description="Resolution notes")
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional alert data")
    
    def acknowledge(self, user: str, notes: str = "") -> None:
        """Acknowledge the alert."""
        self.is_acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
        if notes:
            self.resolution_notes = notes
    
    def resolve(self, notes: str = "") -> None:
        """Resolve the alert."""
        self.is_active = False
        self.resolved_at = datetime.now()
        if notes:
            self.resolution_notes = notes
    
    def age_minutes(self) -> float:
        """Calculate alert age in minutes."""
        return (datetime.now() - self.timestamp).total_seconds() / 60.0
    
    def requires_immediate_action(self) -> bool:
        """Check if alert requires immediate action."""
        immediate_severities = {AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY}
        immediate_types = {AlertType.MARGIN, AlertType.LIQUIDITY}
        
        return (
            self.severity in immediate_severities or
            self.alert_type in immediate_types or
            (self.severity == AlertSeverity.WARNING and self.age_minutes() > 30)
        )


class RiskLimits(BaseModel):
    """
    Risk limits configuration for portfolio management.
    
    Defines various risk thresholds and limits for automated
    risk management and alerting.
    
    Example:
        >>> limits = RiskLimits(
        ...     max_portfolio_var=Decimal("10000.00"),
        ...     max_position_size=0.10,
        ...     max_drawdown=0.15,
        ...     max_leverage=2.0
        ... )
    """
    # Portfolio-level limits
    max_portfolio_var: Optional[Decimal] = Field(None, gt=0, description="Maximum portfolio VaR")
    max_drawdown: Optional[float] = Field(None, gt=0, lt=1, description="Maximum drawdown threshold")
    max_leverage: Optional[float] = Field(default=1.0, gt=0, description="Maximum leverage ratio")
    max_correlation: Optional[float] = Field(default=0.70, gt=0, lt=1, description="Maximum position correlation")
    
    # Position-level limits
    max_position_size: Optional[float] = Field(None, gt=0, le=1, description="Max position as % of portfolio")
    max_sector_allocation: Optional[float] = Field(None, gt=0, le=1, description="Max sector allocation")
    max_single_stock_weight: Optional[float] = Field(None, gt=0, le=1, description="Max single stock weight")
    
    # Concentration limits
    max_top5_concentration: Optional[float] = Field(default=0.50, gt=0, le=1, description="Max top 5 positions weight")
    max_top10_concentration: Optional[float] = Field(default=0.70, gt=0, le=1, description="Max top 10 positions weight")
    
    # Volatility limits
    max_portfolio_volatility: Optional[float] = Field(None, gt=0, description="Maximum portfolio volatility")
    min_sharpe_ratio: Optional[float] = Field(None, description="Minimum Sharpe ratio")
    
    # Liquidity limits
    min_liquidity_score: Optional[float] = Field(default=0.30, gt=0, le=1, description="Minimum liquidity score")
    max_days_to_liquidate: Optional[float] = Field(default=5.0, gt=0, description="Max days to liquidate")
    
    # Dynamic limits
    stress_test_threshold: Optional[Decimal] = Field(None, description="Stress test loss threshold")
    margin_requirement_buffer: Optional[float] = Field(default=0.20, ge=0, description="Margin requirement buffer")
    
    # Metadata
    effective_date: datetime = Field(default_factory=datetime.now, description="When limits become effective")
    review_date: Optional[datetime] = Field(None, description="Next review date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional limit data")
    
    def check_position_limit(self, position_weight: float) -> bool:
        """Check if position weight exceeds limits."""
        if self.max_position_size is None:
            return True
        return position_weight <= self.max_position_size
    
    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """Check if drawdown exceeds limits."""
        if self.max_drawdown is None:
            return True
        return current_drawdown <= self.max_drawdown
    
    def check_leverage_limit(self, current_leverage: float) -> bool:
        """Check if leverage exceeds limits."""
        if self.max_leverage is None:
            return True
        return current_leverage <= self.max_leverage
    
    def violations(self, portfolio: Portfolio, metrics: RiskMetrics) -> List[str]:
        """Check for any limit violations."""
        violations = []
        
        # Check drawdown
        if metrics.current_drawdown and not self.check_drawdown_limit(metrics.current_drawdown):
            violations.append(f"Drawdown ({metrics.current_drawdown:.2%}) exceeds limit ({self.max_drawdown:.2%})")
        
        # Check leverage
        leverage = portfolio.leverage_ratio()
        if not self.check_leverage_limit(leverage):
            violations.append(f"Leverage ({leverage:.2f}) exceeds limit ({self.max_leverage:.2f})")
        
        # Check position concentration
        largest_position = portfolio.largest_position_pct()
        if not self.check_position_limit(largest_position):
            violations.append(f"Largest position ({largest_position:.2%}) exceeds limit ({self.max_position_size:.2%})")
        
        return violations