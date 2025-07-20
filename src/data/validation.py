"""
Enhanced data validation engine with statistical checks and cross-validation.

This module provides comprehensive data quality validation including statistical analysis,
outlier detection, cross-validation between data sources, and financial data specific checks.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings

from .validation.validator import ValidationResult, DataValidator

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Advanced statistical validation for market data.
    
    Performs sophisticated statistical analysis including:
    - Distribution analysis and normality tests
    - Outlier detection using multiple algorithms
    - Time series stationarity tests
    - Autocorrelation analysis
    - Volatility clustering detection
    - Cross-correlation analysis between assets
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical validator.
        
        Args:
            config: Configuration dictionary with statistical parameters
        """
        self.config = config or self._default_config()
        
    def validate_distribution(self, data: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Analyze the statistical distribution of data.
        
        Performs comprehensive distribution analysis including:
        - Normality tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera)
        - Distribution parameter estimation
        - Skewness and kurtosis analysis
        - Goodness-of-fit tests for common distributions
        
        Args:
            data: Time series data
            column_name: Name of the data column
            
        Returns:
            Dictionary containing distribution analysis results
        """
        results = {
            'column': column_name,
            'n_observations': len(data),
            'missing_values': data.isnull().sum(),
            'descriptive_stats': {},
            'normality_tests': {},
            'distribution_fit': {},
            'outliers': {}
        }
        
        # Remove missing values
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            logger.warning(f"Insufficient data for statistical analysis: {len(clean_data)} observations")
            return results
        
        # Descriptive statistics
        results['descriptive_stats'] = {
            'mean': float(clean_data.mean()),
            'median': float(clean_data.median()),
            'std': float(clean_data.std()),
            'variance': float(clean_data.var()),
            'skewness': float(stats.skew(clean_data)),
            'kurtosis': float(stats.kurtosis(clean_data)),
            'min': float(clean_data.min()),
            'max': float(clean_data.max()),
            'range': float(clean_data.max() - clean_data.min()),
            'iqr': float(clean_data.quantile(0.75) - clean_data.quantile(0.25))
        }
        
        # Normality tests
        if len(clean_data) >= 3:
            try:
                # Shapiro-Wilk test (sensitive to small deviations)
                shapiro_stat, shapiro_p = stats.shapiro(clean_data[:5000])  # Limited to 5000 observations
                results['normality_tests']['shapiro'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
                
                # Jarque-Bera test (tests skewness and kurtosis)
                jb_stat, jb_p = stats.jarque_bera(clean_data)
                results['normality_tests']['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > 0.05
                }
                
                # Anderson-Darling test
                ad_stat, ad_critical, ad_significance = stats.anderson(clean_data, dist='norm')
                results['normality_tests']['anderson_darling'] = {
                    'statistic': float(ad_stat),
                    'critical_values': ad_critical.tolist(),
                    'significance_levels': ad_significance.tolist(),
                    'is_normal': ad_stat < ad_critical[2]  # 5% significance level
                }
                
            except Exception as e:
                logger.warning(f"Normality tests failed for {column_name}: {e}")
        
        # Distribution fitting
        try:
            # Fit common distributions
            distributions = ['norm', 'lognorm', 't', 'gamma', 'beta']
            best_fit = None
            best_aic = float('inf')
            
            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(clean_data)
                    
                    # Calculate AIC (Akaike Information Criterion)
                    log_likelihood = np.sum(dist.logpdf(clean_data, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_fit = {
                            'distribution': dist_name,
                            'parameters': params,
                            'aic': aic,
                            'log_likelihood': log_likelihood
                        }
                        
                except Exception:
                    continue
            
            if best_fit:
                results['distribution_fit'] = best_fit
                
        except Exception as e:
            logger.warning(f"Distribution fitting failed for {column_name}: {e}")
        
        # Outlier detection
        results['outliers'] = self._detect_outliers_multiple_methods(clean_data)
        
        return results
    
    def validate_time_series_properties(self, data: pd.Series, timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Validate time series specific properties.
        
        Analyzes temporal characteristics including:
        - Stationarity tests (ADF, KPSS)
        - Autocorrelation and partial autocorrelation
        - Volatility clustering (ARCH effects)
        - Trend analysis
        - Seasonality detection
        
        Args:
            data: Time series data
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary containing time series analysis results
        """
        results = {
            'stationarity': {},
            'autocorrelation': {},
            'volatility_clustering': {},
            'trend_analysis': {},
            'seasonality': {}
        }
        
        clean_data = data.dropna()
        
        if len(clean_data) < 50:
            logger.warning("Insufficient data for time series analysis")
            return results
        
        try:
            # Stationarity tests
            from statsmodels.tsa.stattools import adfuller, kpss
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(clean_data, autolag='AIC')
            results['stationarity']['adf'] = {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'used_lag': int(adf_result[2]),
                'n_observations': int(adf_result[3]),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS test
            try:
                kpss_result = kpss(clean_data, regression='c')
                results['stationarity']['kpss'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'used_lag': int(kpss_result[2]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception as e:
                logger.warning(f"KPSS test failed: {e}")
            
        except ImportError:
            logger.warning("Statsmodels not available for advanced time series tests")
        except Exception as e:
            logger.warning(f"Stationarity tests failed: {e}")
        
        # Autocorrelation analysis
        try:
            # Calculate autocorrelation function
            max_lags = min(40, len(clean_data) // 4)
            autocorr = [clean_data.autocorr(lag=i) for i in range(1, max_lags + 1)]
            
            results['autocorrelation']['autocorrelation_function'] = autocorr
            results['autocorrelation']['significant_lags'] = [
                i for i, corr in enumerate(autocorr, 1) 
                if abs(corr) > 1.96 / np.sqrt(len(clean_data))
            ]
            
            # Ljung-Box test for autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(clean_data, lags=min(10, len(clean_data)//5), return_df=True)
            results['autocorrelation']['ljung_box'] = {
                'statistics': lb_result['lb_stat'].tolist(),
                'p_values': lb_result['lb_pvalue'].tolist(),
                'has_autocorrelation': any(lb_result['lb_pvalue'] < 0.05)
            }
            
        except ImportError:
            logger.warning("Statsmodels not available for autocorrelation tests")
        except Exception as e:
            logger.warning(f"Autocorrelation analysis failed: {e}")
        
        # Volatility clustering (ARCH effects)
        try:
            # Calculate squared returns for ARCH test
            returns = clean_data.pct_change().dropna()
            squared_returns = returns ** 2
            
            if len(squared_returns) > 10:
                from statsmodels.stats.diagnostic import het_arch
                arch_result = het_arch(squared_returns, maxlag=5)
                results['volatility_clustering']['arch_test'] = {
                    'statistic': float(arch_result[0]),
                    'p_value': float(arch_result[1]),
                    'f_statistic': float(arch_result[2]),
                    'f_p_value': float(arch_result[3]),
                    'has_arch_effects': arch_result[1] < 0.05
                }
                
        except ImportError:
            logger.warning("Statsmodels not available for ARCH test")
        except Exception as e:
            logger.warning(f"ARCH test failed: {e}")
        
        # Trend analysis
        try:
            # Linear trend
            x = np.arange(len(clean_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_data)
            
            results['trend_analysis']['linear_trend'] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'std_error': float(std_err),
                'is_significant': p_value < 0.05
            }
            
            # Mann-Kendall trend test
            try:
                from scipy.stats import kendalltau
                tau, p_val = kendalltau(x, clean_data)
                results['trend_analysis']['mann_kendall'] = {
                    'tau': float(tau),
                    'p_value': float(p_val),
                    'has_trend': p_val < 0.05,
                    'trend_direction': 'increasing' if tau > 0 else 'decreasing'
                }
            except Exception as e:
                logger.warning(f"Mann-Kendall test failed: {e}")
                
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
        
        return results
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies using multiple machine learning algorithms.
        
        Uses ensemble methods including:
        - Isolation Forest
        - Local Outlier Factor
        - DBSCAN clustering
        - Statistical outliers (Z-score, IQR)
        
        Args:
            data: DataFrame with numerical data
            
        Returns:
            Dictionary containing anomaly detection results
        """
        results = {
            'isolation_forest': {},
            'local_outlier_factor': {},
            'dbscan': {},
            'statistical_outliers': {},
            'ensemble_anomalies': []
        }
        
        # Prepare data
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data) < 10:
            logger.warning("Insufficient data for anomaly detection")
            return results
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        try:
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.config.get('anomaly_contamination', 0.1),
                random_state=42
            )
            iso_anomalies = iso_forest.fit_predict(scaled_data)
            iso_scores = iso_forest.decision_function(scaled_data)
            
            results['isolation_forest'] = {
                'anomalies': (iso_anomalies == -1).tolist(),
                'scores': iso_scores.tolist(),
                'n_anomalies': sum(iso_anomalies == -1),
                'contamination_rate': sum(iso_anomalies == -1) / len(iso_anomalies)
            }
            
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
        
        try:
            # Local Outlier Factor
            from sklearn.neighbors import LocalOutlierFactor
            
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(scaled_data) // 2),
                contamination=self.config.get('anomaly_contamination', 0.1)
            )
            lof_anomalies = lof.fit_predict(scaled_data)
            lof_scores = lof.negative_outlier_factor_
            
            results['local_outlier_factor'] = {
                'anomalies': (lof_anomalies == -1).tolist(),
                'scores': lof_scores.tolist(),
                'n_anomalies': sum(lof_anomalies == -1),
                'contamination_rate': sum(lof_anomalies == -1) / len(lof_anomalies)
            }
            
        except Exception as e:
            logger.warning(f"Local Outlier Factor failed: {e}")
        
        try:
            # DBSCAN clustering
            eps = self.config.get('dbscan_eps', 0.5)
            min_samples = max(2, len(scaled_data) // 50)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(scaled_data)
            
            # Points with label -1 are considered outliers
            dbscan_anomalies = cluster_labels == -1
            
            results['dbscan'] = {
                'anomalies': dbscan_anomalies.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_anomalies': sum(dbscan_anomalies),
                'contamination_rate': sum(dbscan_anomalies) / len(dbscan_anomalies)
            }
            
        except Exception as e:
            logger.warning(f"DBSCAN failed: {e}")
        
        # Statistical outliers
        statistical_anomalies = []
        for i, column in enumerate(numeric_data.columns):
            col_data = numeric_data[column]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = z_scores > self.config.get('z_score_threshold', 3.0)
            
            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            iqr_outliers = (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
            
            # Modified Z-score (using median)
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad
            modified_z_outliers = np.abs(modified_z_scores) > self.config.get('modified_z_threshold', 3.5)
            
            statistical_anomalies.append({
                'column': column,
                'z_score_outliers': z_outliers.tolist(),
                'iqr_outliers': iqr_outliers.tolist(),
                'modified_z_outliers': modified_z_outliers.tolist(),
                'n_z_outliers': sum(z_outliers),
                'n_iqr_outliers': sum(iqr_outliers),
                'n_modified_z_outliers': sum(modified_z_outliers)
            })
        
        results['statistical_outliers'] = statistical_anomalies
        
        # Ensemble anomalies (consensus across methods)
        if 'isolation_forest' in results and 'local_outlier_factor' in results:
            iso_anomalies = np.array(results['isolation_forest'].get('anomalies', []))
            lof_anomalies = np.array(results['local_outlier_factor'].get('anomalies', []))
            
            if len(iso_anomalies) == len(lof_anomalies):
                # Points identified as anomalies by multiple methods
                ensemble_anomalies = iso_anomalies & lof_anomalies
                results['ensemble_anomalies'] = ensemble_anomalies.tolist()
        
        return results
    
    def cross_validate_sources(
        self, 
        source1_data: pd.DataFrame, 
        source2_data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Cross-validate data between two sources.
        
        Compares data from different sources to identify discrepancies:
        - Price correlation analysis
        - Volume correlation analysis
        - Statistical difference tests
        - Systematic bias detection
        - Data completeness comparison
        
        Args:
            source1_data: Data from first source
            source2_data: Data from second source
            symbol: Symbol being compared
            
        Returns:
            Dictionary containing cross-validation results
        """
        results = {
            'symbol': symbol,
            'data_alignment': {},
            'price_comparison': {},
            'volume_comparison': {},
            'statistical_tests': {},
            'bias_analysis': {},
            'quality_score': 0.0
        }
        
        try:
            # Align data by timestamp
            aligned_data = pd.merge(
                source1_data, source2_data, 
                left_index=True, right_index=True, 
                suffixes=('_src1', '_src2'), 
                how='inner'
            )
            
            if len(aligned_data) == 0:
                logger.warning(f"No overlapping data for cross-validation of {symbol}")
                return results
            
            results['data_alignment'] = {
                'source1_records': len(source1_data),
                'source2_records': len(source2_data),
                'aligned_records': len(aligned_data),
                'alignment_ratio': len(aligned_data) / max(len(source1_data), len(source2_data))
            }
            
            # Price comparison
            price_columns = ['Open', 'High', 'Low', 'Close']
            price_correlations = {}
            price_differences = {}
            
            for col in price_columns:
                col1 = f"{col}_src1"
                col2 = f"{col}_src2"
                
                if col1 in aligned_data.columns and col2 in aligned_data.columns:
                    # Correlation
                    correlation = aligned_data[col1].corr(aligned_data[col2])
                    price_correlations[col] = float(correlation)
                    
                    # Percentage differences
                    pct_diff = ((aligned_data[col2] - aligned_data[col1]) / aligned_data[col1] * 100)
                    price_differences[col] = {
                        'mean_pct_diff': float(pct_diff.mean()),
                        'std_pct_diff': float(pct_diff.std()),
                        'max_pct_diff': float(pct_diff.abs().max()),
                        'median_pct_diff': float(pct_diff.median())
                    }
            
            results['price_comparison'] = {
                'correlations': price_correlations,
                'differences': price_differences,
                'avg_correlation': np.mean(list(price_correlations.values())) if price_correlations else 0
            }
            
            # Volume comparison
            if 'Volume_src1' in aligned_data.columns and 'Volume_src2' in aligned_data.columns:
                vol_corr = aligned_data['Volume_src1'].corr(aligned_data['Volume_src2'])
                vol_pct_diff = ((aligned_data['Volume_src2'] - aligned_data['Volume_src1']) / 
                               aligned_data['Volume_src1'] * 100)
                
                results['volume_comparison'] = {
                    'correlation': float(vol_corr),
                    'mean_pct_diff': float(vol_pct_diff.mean()),
                    'std_pct_diff': float(vol_pct_diff.std()),
                    'max_pct_diff': float(vol_pct_diff.abs().max())
                }
            
            # Statistical tests
            for col in price_columns:
                col1 = f"{col}_src1"
                col2 = f"{col}_src2"
                
                if col1 in aligned_data.columns and col2 in aligned_data.columns:
                    # Paired t-test
                    t_stat, t_p = stats.ttest_rel(aligned_data[col1], aligned_data[col2])
                    
                    # Wilcoxon signed-rank test (non-parametric)
                    w_stat, w_p = stats.wilcoxon(aligned_data[col1], aligned_data[col2])
                    
                    results['statistical_tests'][col] = {
                        't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
                        'wilcoxon': {'statistic': float(w_stat), 'p_value': float(w_p)},
                        'significant_difference': min(t_p, w_p) < 0.05
                    }
            
            # Bias analysis
            close1 = aligned_data.get('Close_src1')
            close2 = aligned_data.get('Close_src2')
            
            if close1 is not None and close2 is not None:
                bias = ((close2 - close1) / close1 * 100).mean()
                results['bias_analysis'] = {
                    'systematic_bias_pct': float(bias),
                    'has_systematic_bias': abs(bias) > self.config.get('bias_threshold', 0.1)
                }
            
            # Quality score (0-1, higher is better)
            quality_factors = []
            
            # Data alignment factor
            quality_factors.append(results['data_alignment']['alignment_ratio'])
            
            # Price correlation factor
            avg_corr = results['price_comparison']['avg_correlation']
            if not np.isnan(avg_corr):
                quality_factors.append(max(0, avg_corr))  # Ensure non-negative
            
            # Low difference factor
            if price_differences:
                avg_diff = np.mean([abs(diff['mean_pct_diff']) for diff in price_differences.values()])
                quality_factors.append(max(0, 1 - avg_diff / 100))  # Convert to 0-1 scale
            
            results['quality_score'] = float(np.mean(quality_factors)) if quality_factors else 0.0
            
        except Exception as e:
            logger.error(f"Cross-validation failed for {symbol}: {e}")
            
        return results
    
    def _detect_outliers_multiple_methods(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple statistical methods."""
        outlier_results = {}
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = z_scores > self.config.get('z_score_threshold', 3.0)
        outlier_results['z_score'] = {
            'outliers': z_outliers.tolist(),
            'n_outliers': sum(z_outliers),
            'percentage': (sum(z_outliers) / len(data)) * 100
        }
        
        # IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))
        outlier_results['iqr'] = {
            'outliers': iqr_outliers.tolist(),
            'n_outliers': sum(iqr_outliers),
            'percentage': (sum(iqr_outliers) / len(data)) * 100,
            'lower_bound': q1 - 1.5 * iqr,
            'upper_bound': q3 + 1.5 * iqr
        }
        
        # Modified Z-score (using median absolute deviation)
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad != 0:
            modified_z_scores = 0.6745 * (data - median) / mad
            modified_z_outliers = np.abs(modified_z_scores) > self.config.get('modified_z_threshold', 3.5)
            outlier_results['modified_z_score'] = {
                'outliers': modified_z_outliers.tolist(),
                'n_outliers': sum(modified_z_outliers),
                'percentage': (sum(modified_z_outliers) / len(data)) * 100
            }
        
        return outlier_results
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for statistical validation."""
        return {
            'z_score_threshold': 3.0,
            'modified_z_threshold': 3.5,
            'anomaly_contamination': 0.1,
            'dbscan_eps': 0.5,
            'bias_threshold': 0.1,  # 0.1% systematic bias threshold
            'correlation_threshold': 0.95,  # High correlation expected between sources
            'max_missing_percentage': 5.0,
            'outlier_percentage_threshold': 5.0
        }


class EnhancedDataValidator(DataValidator):
    """
    Enhanced data validator combining basic and statistical validation.
    
    Extends the base DataValidator with advanced statistical analysis,
    cross-validation capabilities, and financial data specific checks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced validator."""
        super().__init__(config)
        self.statistical_validator = StatisticalValidator(config)
        
    def validate_market_data_enhanced(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        reference_data: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """
        Perform enhanced validation including statistical analysis.
        
        Args:
            data: DataFrame with OHLCV market data
            symbol: Stock symbol for context
            reference_data: Optional reference data for cross-validation
            
        Returns:
            Enhanced ValidationResult with statistical analysis
        """
        # Start with basic validation
        result = super().validate_market_data(data, symbol)
        
        if not data.empty:
            try:
                # Add statistical analysis
                price_columns = ['Open', 'High', 'Low', 'Close']
                result.statistics['distribution_analysis'] = {}
                result.statistics['time_series_analysis'] = {}
                
                for col in price_columns:
                    if col in data.columns:
                        # Distribution analysis
                        dist_analysis = self.statistical_validator.validate_distribution(
                            data[col], col
                        )
                        result.statistics['distribution_analysis'][col] = dist_analysis
                        
                        # Time series analysis
                        ts_analysis = self.statistical_validator.validate_time_series_properties(
                            data[col], data.index
                        )
                        result.statistics['time_series_analysis'][col] = ts_analysis
                
                # Anomaly detection
                anomaly_results = self.statistical_validator.detect_anomalies(data)
                result.statistics['anomaly_detection'] = anomaly_results
                
                # Cross-validation if reference data provided
                if reference_data is not None:
                    cross_val_results = self.statistical_validator.cross_validate_sources(
                        data, reference_data, symbol
                    )
                    result.statistics['cross_validation'] = cross_val_results
                    
                    # Add warnings based on cross-validation
                    if cross_val_results.get('quality_score', 1.0) < 0.8:
                        result.add_error(
                            f"Low cross-validation quality score: {cross_val_results['quality_score']:.2f}",
                            "warning"
                        )
                
                # Financial data specific checks
                self._validate_financial_relationships(data, result)
                
            except Exception as e:
                logger.error(f"Enhanced validation failed for {symbol}: {e}")
                result.add_error(f"Statistical analysis failed: {e}", "warning")
        
        return result
    
    def _validate_financial_relationships(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validate financial relationships specific to market data."""
        try:
            # Price momentum analysis
            if 'Close' in data.columns and len(data) > 20:
                close_prices = data['Close']
                returns = close_prices.pct_change().dropna()
                
                # Check for excessive volatility
                daily_volatility = returns.std()
                annualized_vol = daily_volatility * np.sqrt(252)
                
                if annualized_vol > 1.0:  # 100% annualized volatility
                    result.add_error(
                        f"Extremely high volatility detected: {annualized_vol:.1%} annualized",
                        "warning"
                    )
                
                # Check for return autocorrelation (should be minimal for efficient markets)
                if len(returns) > 10:
                    autocorr_1 = returns.autocorr(lag=1)
                    if abs(autocorr_1) > 0.1:
                        result.add_error(
                            f"High return autocorrelation detected: {autocorr_1:.3f}",
                            "warning"
                        )
            
            # Volume-price relationship
            if all(col in data.columns for col in ['Close', 'Volume']):
                price_changes = data['Close'].pct_change().abs()
                volume_changes = data['Volume'].pct_change().abs()
                
                # Volume should generally increase with large price moves
                vol_price_corr = price_changes.corr(volume_changes)
                if vol_price_corr < 0.1:
                    result.add_error(
                        f"Low volume-price correlation: {vol_price_corr:.3f}",
                        "warning"
                    )
            
            # Bid-ask spread analysis (if available)
            if all(col in data.columns for col in ['Bid', 'Ask']):
                spreads = (data['Ask'] - data['Bid']) / data['Bid']
                avg_spread = spreads.mean()
                
                if avg_spread > 0.01:  # 1% average spread is quite high
                    result.add_error(
                        f"High average bid-ask spread: {avg_spread:.1%}",
                        "warning"
                    )
                
        except Exception as e:
            logger.warning(f"Financial relationship validation failed: {e}")