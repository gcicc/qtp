"""
Unit tests for risk management structures.

Tests all risk-related structures including Portfolio, RiskMetrics, Alert,
and RiskLimits with comprehensive validation and functionality verification.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.risk.structures import (
    Portfolio, RiskMetrics, Alert, RiskLimits,
    RiskLevel, AlertType, AlertSeverity
)
from src.strategies.signals import Position, PositionSide


class TestPortfolio:
    """Test cases for Portfolio data structure."""
    
    def test_valid_portfolio_creation(self):
        """Test creation of valid portfolio."""
        portfolio = Portfolio(
            name="Main Trading Portfolio",
            cash_balance=Decimal("100000.00"),
            initial_capital=Decimal("100000.00"),
            base_currency="USD"
        )
        
        assert portfolio.name == "Main Trading Portfolio"
        assert portfolio.cash_balance == Decimal("100000.00")
        assert portfolio.initial_capital == Decimal("100000.00")
        assert portfolio.base_currency == "USD"
        assert len(portfolio.positions) == 0
    
    def test_portfolio_with_positions(self):
        """Test portfolio with positions."""
        # Create test positions
        position1 = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            entry_timestamp=datetime.now()
        )
        
        position2 = Position(
            symbol="GOOGL",
            side=PositionSide.LONG,
            quantity=Decimal("50"),
            avg_entry_price=Decimal("2800.00"),
            current_price=Decimal("2850.00"),
            entry_timestamp=datetime.now()
        )
        
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Decimal("50000.00"),
            initial_capital=Decimal("200000.00"),
            positions=[position1, position2]
        )
        
        # Test portfolio calculations
        # Position 1 market value: 100 * 155 = 15,500
        # Position 2 market value: 50 * 2850 = 142,500
        # Total market value: 15,500 + 142,500 = 158,000
        expected_market_value = Decimal("158000.00")
        assert portfolio.total_market_value() == expected_market_value
        
        # Total portfolio value: cash + positions = 50,000 + 158,000 = 208,000
        expected_total_value = Decimal("208000.00")
        assert portfolio.total_value() == expected_total_value
        
        # Return percentage: (208,000 - 200,000) / 200,000 = 4%
        expected_return = 0.04
        assert abs(portfolio.return_pct() - expected_return) < 1e-10
    
    def test_portfolio_position_management(self):
        """Test portfolio position management methods."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            entry_timestamp=datetime.now()
        )
        
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Decimal("50000.00"),
            initial_capital=Decimal("100000.00"),
            positions=[position]
        )
        
        # Get position by symbol
        found_position = portfolio.get_position("AAPL")
        assert found_position is not None
        assert found_position.symbol == "AAPL"
        
        # Position not found
        not_found = portfolio.get_position("MSFT")
        assert not_found is None
        
        # Position count
        assert portfolio.position_count() == 1
    
    def test_portfolio_leverage_calculation(self):
        """Test portfolio leverage calculation."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("1000"),  # Large position
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("150.00"),
            entry_timestamp=datetime.now()
        )
        
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Decimal("50000.00"),  # Less cash than position value
            initial_capital=Decimal("200000.00"),
            positions=[position]
        )
        
        # Position value: 1000 * 150 = 150,000
        # Net liquidation value: 50,000 + 150,000 = 200,000
        # Gross exposure: 150,000
        # Leverage: 150,000 / 200,000 = 0.75
        expected_leverage = 0.75
        actual_leverage = portfolio.leverage_ratio()
        assert abs(actual_leverage - expected_leverage) < 1e-10


class TestRiskMetrics:
    """Test cases for RiskMetrics data structure."""
    
    def test_valid_risk_metrics_creation(self):
        """Test creation of valid risk metrics."""
        metrics = RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            var_1d=Decimal("5000.00"),
            volatility_annualized=0.18,
            sharpe_ratio=1.25,
            max_drawdown=0.08
        )
        
        assert metrics.symbol == "PORTFOLIO"
        assert metrics.var_1d == Decimal("5000.00")
        assert metrics.volatility_annualized == 0.18
        assert metrics.sharpe_ratio == 1.25
        assert metrics.max_drawdown == 0.08
    
    def test_risk_level_determination(self):
        """Test risk level determination based on metrics."""
        # Low risk metrics
        low_risk_metrics = RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            current_drawdown=0.05,  # Low drawdown
            volatility_annualized=0.15,  # Moderate volatility
            concentration_herfindahl=0.10  # Low concentration
        )
        assert low_risk_metrics.risk_level() == RiskLevel.MODERATE
        
        # High risk metrics
        high_risk_metrics = RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            current_drawdown=0.20,  # High drawdown (>15%)
            volatility_annualized=0.35,  # High volatility (>30%)
            concentration_herfindahl=0.25  # High concentration (>20%)
        )
        assert high_risk_metrics.risk_level() == RiskLevel.CRITICAL
    
    def test_risk_metrics_health_check(self):
        """Test risk metrics health assessment."""
        # Healthy portfolio
        healthy_metrics = RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            current_drawdown=0.05,
            volatility_annualized=0.15
        )
        assert healthy_metrics.is_healthy() is True
        
        # Unhealthy portfolio
        unhealthy_metrics = RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            current_drawdown=0.18,  # High drawdown
            volatility_annualized=0.35  # High volatility
        )
        assert unhealthy_metrics.is_healthy() is False


class TestAlert:
    """Test cases for Alert data structure."""
    
    def test_valid_alert_creation(self):
        """Test creation of valid risk alert."""
        alert = Alert(
            alert_type=AlertType.DRAWDOWN,
            severity=AlertSeverity.WARNING,
            symbol="PORTFOLIO",
            message="Current drawdown (12%) exceeds warning threshold (10%)",
            triggered_value=0.12,
            threshold=0.10
        )
        
        assert alert.alert_type == AlertType.DRAWDOWN
        assert alert.severity == AlertSeverity.WARNING
        assert alert.symbol == "PORTFOLIO"
        assert alert.triggered_value == 0.12
        assert alert.threshold == 0.10
        assert alert.is_active is True
        assert alert.is_acknowledged is False
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment functionality."""
        alert = Alert(
            alert_type=AlertType.CONCENTRATION,
            severity=AlertSeverity.INFO,
            message="Portfolio concentration alert"
        )
        
        # Initially not acknowledged
        assert alert.is_acknowledged is False
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None
        
        # Acknowledge alert
        alert.acknowledge("trader1", "Reviewed and acceptable")
        
        assert alert.is_acknowledged is True
        assert alert.acknowledged_by == "trader1"
        assert alert.acknowledged_at is not None
        assert "Reviewed and acceptable" in alert.resolution_notes
    
    def test_alert_resolution(self):
        """Test alert resolution functionality."""
        alert = Alert(
            alert_type=AlertType.VOLATILITY,
            severity=AlertSeverity.WARNING,
            message="High volatility detected"
        )
        
        # Initially active
        assert alert.is_active is True
        assert alert.resolved_at is None
        
        # Resolve alert
        alert.resolve("Volatility returned to normal levels")
        
        assert alert.is_active is False
        assert alert.resolved_at is not None
        assert "Volatility returned to normal levels" in alert.resolution_notes
    
    def test_alert_immediate_action_requirement(self):
        """Test immediate action requirement detection."""
        # Critical severity - requires immediate action
        critical_alert = Alert(
            alert_type=AlertType.MARGIN,
            severity=AlertSeverity.CRITICAL,
            message="Margin requirement exceeded"
        )
        assert critical_alert.requires_immediate_action() is True
        
        # Emergency severity - requires immediate action
        emergency_alert = Alert(
            alert_type=AlertType.LIQUIDITY,
            severity=AlertSeverity.EMERGENCY,
            message="Liquidity crisis detected"
        )
        assert emergency_alert.requires_immediate_action() is True
        
        # Margin alert type - always requires immediate action
        margin_alert = Alert(
            alert_type=AlertType.MARGIN,
            severity=AlertSeverity.INFO,
            message="Margin usage high"
        )
        assert margin_alert.requires_immediate_action() is True
        
        # Regular warning - doesn't require immediate action (unless old)
        regular_alert = Alert(
            alert_type=AlertType.CONCENTRATION,
            severity=AlertSeverity.WARNING,
            message="Portfolio concentration warning"
        )
        assert regular_alert.requires_immediate_action() is False
    
    def test_alert_age_calculation(self):
        """Test alert age calculation."""
        # Create alert with specific timestamp
        past_time = datetime.now() - timedelta(minutes=45)
        alert = Alert(
            alert_type=AlertType.DRAWDOWN,
            severity=AlertSeverity.WARNING,
            message="Drawdown alert",
            timestamp=past_time
        )
        
        age_minutes = alert.age_minutes()
        assert 44 <= age_minutes <= 46  # Allow for small timing differences


class TestRiskLimits:
    """Test cases for RiskLimits data structure."""
    
    def test_valid_risk_limits_creation(self):
        """Test creation of valid risk limits."""
        limits = RiskLimits(
            max_portfolio_var=Decimal("10000.00"),
            max_position_size=0.10,
            max_drawdown=0.15,
            max_leverage=2.0
        )
        
        assert limits.max_portfolio_var == Decimal("10000.00")
        assert limits.max_position_size == 0.10
        assert limits.max_drawdown == 0.15
        assert limits.max_leverage == 2.0
    
    def test_risk_limits_validation(self):
        """Test risk limits validation methods."""
        limits = RiskLimits(
            max_position_size=0.10,
            max_drawdown=0.15,
            max_leverage=2.0
        )
        
        # Position limit validation
        assert limits.check_position_limit(0.08) is True  # Within limit
        assert limits.check_position_limit(0.12) is False  # Exceeds limit
        
        # Drawdown limit validation
        assert limits.check_drawdown_limit(0.10) is True  # Within limit
        assert limits.check_drawdown_limit(0.18) is False  # Exceeds limit
        
        # Leverage limit validation
        assert limits.check_leverage_limit(1.5) is True  # Within limit
        assert limits.check_leverage_limit(2.5) is False  # Exceeds limit
    
    def test_risk_limits_violations_detection(self):
        """Test risk limits violations detection."""
        limits = RiskLimits(
            max_position_size=0.10,
            max_drawdown=0.15,
            max_leverage=2.0
        )
        
        # Create portfolio with violations
        portfolio = Portfolio(
            name="Test Portfolio",
            cash_balance=Decimal("10000.00"),
            initial_capital=Decimal("100000.00")
        )
        
        # Create risk metrics with violations
        metrics = RiskMetrics(
            symbol="PORTFOLIO",
            timestamp=datetime.now(),
            current_drawdown=0.18  # Exceeds 15% limit
        )
        
        # Mock portfolio methods for testing
        def mock_leverage_ratio():
            return 2.5  # Exceeds 2.0 limit
        
        def mock_largest_position_pct():
            return 0.12  # Exceeds 10% limit
        
        portfolio.leverage_ratio = mock_leverage_ratio
        portfolio.largest_position_pct = mock_largest_position_pct
        
        violations = limits.violations(portfolio, metrics)
        
        # Should detect all three violations
        assert len(violations) == 3
        assert any("Drawdown" in violation for violation in violations)
        assert any("Leverage" in violation for violation in violations)
        assert any("position" in violation for violation in violations)