"""
Temporal Analysis & Early Warning System
========================================

Time-series analysis of vegetation indices for crop stress detection,
trend identification, and proactive alert generation.

Dependencies:
    - numpy
    - pandas
    - scipy
    - matplotlib

Author: Agricultural AI Research Project
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """
    Analyzes temporal patterns in vegetation indices for early warning.
    """
    
    def __init__(self):
        self.alert_thresholds = {
            'rapid_decline': 0.15,  # NDVI drop over 7 days
            'sustained_decline_days': 14,
            'critical_ndvi': 0.4,
            'water_stress_ndwi': 0.3
        }
        
    def detect_trend(self, timeseries, window_days=30):
        """
        Detect overall trend in NDVI time series using linear regression.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI' columns
            window_days: Days to consider for trend
            
        Returns:
            dict: Trend statistics
        """
        df = timeseries.copy().sort_values('date')
        
        # Use recent window
        if len(df) > window_days // 5:  # Assuming ~5 day revisit
            df = df.tail(window_days // 5)
        
        # Remove NaN
        df = df.dropna(subset=['NDVI'])
        
        if len(df) < 3:
            return {'trend': 'insufficient_data', 'slope': None, 'p_value': None}
        
        # Linear regression
        x = np.arange(len(df))
        y = df['NDVI'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Classify trend
        if p_value > 0.05:
            trend_type = 'stable'
        elif slope > 0.01:
            trend_type = 'increasing'
        elif slope < -0.01:
            trend_type = 'decreasing'
        else:
            trend_type = 'stable'
        
        return {
            'trend': trend_type,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'prediction_7d': intercept + slope * (len(df) + 1.4),  # ~7 days ahead
            'confidence': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
        }
    
    def detect_anomalies(self, timeseries, std_threshold=2.0):
        """
        Detect anomalous NDVI values using statistical outlier detection.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI' columns
            std_threshold: Number of standard deviations for outlier
            
        Returns:
            DataFrame: Original data with 'is_anomaly' column
        """
        df = timeseries.copy()
        
        # Rolling statistics
        window = min(6, len(df))  # ~30 days or less
        df['rolling_mean'] = df['NDVI'].rolling(window=window, center=True, min_periods=2).mean()
        df['rolling_std'] = df['NDVI'].rolling(window=window, center=True, min_periods=2).std()
        
        # Identify anomalies
        df['deviation'] = np.abs(df['NDVI'] - df['rolling_mean'])
        df['is_anomaly'] = df['deviation'] > (std_threshold * df['rolling_std'])
        
        # Sudden drops are more concerning
        df['is_sudden_drop'] = (df['NDVI'] - df['rolling_mean']) < -(std_threshold * df['rolling_std'])
        
        return df
    
    def detect_rapid_decline(self, timeseries, days=7):
        """
        Detect rapid NDVI decline over short period.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI' columns
            days: Period to check for decline
            
        Returns:
            dict: Decline detection results
        """
        df = timeseries.copy().sort_values('date')
        
        if len(df) < 2:
            return {'detected': False, 'message': 'Insufficient data'}
        
        # Compare recent observations
        recent = df.tail(2)
        
        if len(recent) == 2:
            ndvi_change = recent.iloc[-1]['NDVI'] - recent.iloc[0]['NDVI']
            days_elapsed = (recent.iloc[-1]['date'] - recent.iloc[0]['date']).days
            
            # Normalize to 7-day equivalent
            if days_elapsed > 0:
                decline_rate = ndvi_change / days_elapsed * 7
            else:
                decline_rate = 0
            
            if decline_rate < -self.alert_thresholds['rapid_decline']:
                return {
                    'detected': True,
                    'decline_amount': abs(ndvi_change),
                    'decline_rate': decline_rate,
                    'severity': 'high' if decline_rate < -0.2 else 'medium',
                    'message': f'Rapid decline detected: NDVI dropped {abs(ndvi_change):.3f} in {days_elapsed} days'
                }
        
        return {'detected': False, 'message': 'No rapid decline'}
    
    def analyze_sustained_decline(self, timeseries, min_observations=3):
        """
        Check for sustained negative trend over multiple observations.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI' columns
            min_observations: Minimum consecutive declining observations
            
        Returns:
            dict: Sustained decline analysis
        """
        df = timeseries.copy().sort_values('date')
        
        # Calculate differences
        df['ndvi_diff'] = df['NDVI'].diff()
        
        # Count consecutive declines
        declining = df['ndvi_diff'] < 0
        consecutive_declines = 0
        max_consecutive = 0
        
        for is_declining in declining:
            if is_declining:
                consecutive_declines += 1
                max_consecutive = max(max_consecutive, consecutive_declines)
            else:
                consecutive_declines = 0
        
        is_sustained = max_consecutive >= min_observations
        
        if is_sustained:
            # Calculate total decline
            recent_decline_period = df.tail(max_consecutive + 1)
            total_decline = recent_decline_period.iloc[-1]['NDVI'] - recent_decline_period.iloc[0]['NDVI']
            
            return {
                'detected': True,
                'consecutive_declines': max_consecutive,
                'total_decline': total_decline,
                'severity': 'high' if total_decline < -0.2 else 'medium',
                'message': f'Sustained decline: {max_consecutive} consecutive drops, total change: {total_decline:.3f}'
            }
        
        return {'detected': False, 'message': 'No sustained decline'}
    
    def check_critical_threshold(self, timeseries):
        """
        Check if NDVI has fallen below critical threshold.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI' columns
            
        Returns:
            dict: Threshold check results
        """
        df = timeseries.copy().sort_values('date')
        
        # Recent observations below threshold
        recent = df.tail(3)
        below_threshold = recent['NDVI'] < self.alert_thresholds['critical_ndvi']
        
        consecutive_below = 0
        for is_below in below_threshold:
            if is_below:
                consecutive_below += 1
            else:
                break
        
        if consecutive_below >= 2:
            current_ndvi = recent.iloc[-1]['NDVI']
            return {
                'detected': True,
                'current_ndvi': current_ndvi,
                'threshold': self.alert_thresholds['critical_ndvi'],
                'consecutive_days': consecutive_below * 5,  # Approximate
                'severity': 'critical' if current_ndvi < 0.3 else 'high',
                'message': f'Critical threshold breached: NDVI = {current_ndvi:.3f}'
            }
        
        return {'detected': False, 'message': 'Above critical threshold'}
    
    def detect_water_stress(self, timeseries):
        """
        Detect water stress pattern (high NDWI + declining NDVI).
        
        Args:
            timeseries: DataFrame with 'date', 'NDVI', 'NDWI' columns
            
        Returns:
            dict: Water stress detection results
        """
        df = timeseries.copy().sort_values('date')
        
        if 'NDWI' not in df.columns:
            return {'detected': False, 'message': 'NDWI data not available'}
        
        recent = df.tail(3)
        
        # Check for pattern
        high_ndwi = recent['NDWI'].mean() > self.alert_thresholds['water_stress_ndwi']
        declining_ndvi = (recent.iloc[-1]['NDVI'] - recent.iloc[0]['NDVI']) < -0.05
        
        if high_ndwi and declining_ndvi:
            return {
                'detected': True,
                'avg_ndwi': recent['NDWI'].mean(),
                'ndvi_decline': recent.iloc[-1]['NDVI'] - recent.iloc[0]['NDVI'],
                'severity': 'high',
                'message': 'Water stress pattern detected: High NDWI with declining NDVI'
            }
        
        return {'detected': False, 'message': 'No water stress pattern'}
    
    def generate_alert(self, timeseries):
        """
        Comprehensive alert generation based on all detection methods.
        
        Args:
            timeseries: DataFrame with temporal vegetation data
            
        Returns:
            dict: Alert information with level and recommendations
        """
        alerts = []
        max_severity = 'normal'
        
        # Run all detection methods
        trend_result = self.detect_trend(timeseries)
        rapid_decline = self.detect_rapid_decline(timeseries)
        sustained_decline = self.analyze_sustained_decline(timeseries)
        threshold_check = self.check_critical_threshold(timeseries)
        water_stress = self.detect_water_stress(timeseries)
        
        # Collect alerts
        if rapid_decline['detected']:
            alerts.append({
                'type': 'rapid_decline',
                'severity': rapid_decline['severity'],
                'message': rapid_decline['message']
            })
            if rapid_decline['severity'] == 'high':
                max_severity = 'critical'
            elif max_severity == 'normal':
                max_severity = 'warning'
        
        if sustained_decline['detected']:
            alerts.append({
                'type': 'sustained_decline',
                'severity': sustained_decline['severity'],
                'message': sustained_decline['message']
            })
            if sustained_decline['severity'] == 'high' and max_severity != 'critical':
                max_severity = 'warning'
            elif max_severity == 'normal':
                max_severity = 'watch'
        
        if threshold_check['detected']:
            alerts.append({
                'type': 'critical_threshold',
                'severity': threshold_check['severity'],
                'message': threshold_check['message']
            })
            max_severity = 'critical'
        
        if water_stress['detected']:
            alerts.append({
                'type': 'water_stress',
                'severity': water_stress['severity'],
                'message': water_stress['message']
            })
            if max_severity not in ['critical', 'warning']:
                max_severity = 'warning'
        
        # Trend-based alerts
        if trend_result['trend'] == 'decreasing' and trend_result['p_value'] < 0.05:
            alerts.append({
                'type': 'negative_trend',
                'severity': 'medium',
                'message': f'Negative trend detected (slope: {trend_result["slope"]:.4f})'
            })
            if max_severity == 'normal':
                max_severity = 'watch'
        
        # Determine overall alert level
        alert_levels = {
            'normal': {'icon': '🟢', 'label': 'Normal', 'action': 'Continue monitoring'},
            'watch': {'icon': '🟡', 'label': 'Watch', 'action': 'Increased monitoring recommended'},
            'warning': {'icon': '🟠', 'label': 'Warning', 'action': 'Investigation recommended'},
            'critical': {'icon': '🔴', 'label': 'Critical', 'action': 'Immediate action required'}
        }
        
        return {
            'alert_level': max_severity,
            'icon': alert_levels[max_severity]['icon'],
            'label': alert_levels[max_severity]['label'],
            'action': alert_levels[max_severity]['action'],
            'alerts': alerts,
            'alert_count': len(alerts),
            'timestamp': datetime.now().isoformat(),
            'trend_analysis': trend_result
        }
    
    def visualize_timeseries(self, timeseries, save_path='timeseries_analysis.png'):
        """
        Create comprehensive time series visualization with anomalies and trends.
        
        Args:
            timeseries: DataFrame with 'date', 'NDVI', optional 'NDWI'
            save_path: Path to save figure
        """
        df = timeseries.copy().sort_values('date')
        
        # Detect anomalies
        df_anomalies = self.detect_anomalies(df)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # NDVI plot
        axes[0].plot(df['date'], df['NDVI'], 'o-', label='NDVI', linewidth=2, markersize=6)
        
        # Highlight anomalies
        anomalies = df_anomalies[df_anomalies['is_anomaly']]
        if len(anomalies) > 0:
            axes[0].scatter(anomalies['date'], anomalies['NDVI'], 
                          color='red', s=100, marker='x', label='Anomaly', zorder=5)
        
        # Trend line
        trend = self.detect_trend(df)
        if trend['slope'] is not None:
            x_trend = np.arange(len(df))
            y_trend = trend['slope'] * x_trend + (df['NDVI'].mean() - trend['slope'] * x_trend.mean())
            axes[0].plot(df['date'], y_trend, '--', color='gray', 
                        label=f'Trend (slope={trend["slope"]:.4f})', linewidth=2)
        
        # Critical threshold
        axes[0].axhline(y=self.alert_thresholds['critical_ndvi'], 
                       color='red', linestyle=':', linewidth=2, label='Critical Threshold')
        
        axes[0].set_ylabel('NDVI', fontsize=12, fontweight='bold')
        axes[0].set_title('NDVI Time Series Analysis', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # NDWI plot (if available)
        if 'NDWI' in df.columns:
            axes[1].plot(df['date'], df['NDWI'], 'o-', color='blue', 
                        label='NDWI', linewidth=2, markersize=6)
            axes[1].axhline(y=self.alert_thresholds['water_stress_ndwi'], 
                          color='orange', linestyle=':', linewidth=2, 
                          label='Water Stress Threshold')
            axes[1].set_ylabel('NDWI', fontsize=12, fontweight='bold')
            axes[1].legend(loc='best')
        else:
            # Show NDVI derivative
            axes[1].plot(df['date'].iloc[1:], df['NDVI'].diff().iloc[1:], 
                        'o-', color='green', label='ΔNDVI', linewidth=2, markersize=6)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[1].set_ylabel('ΔNDVI (Change)', fontsize=12, fontweight='bold')
            axes[1].legend(loc='best')
        
        axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Time series visualization saved to '{save_path}'")
    
    def seasonal_decomposition(self, timeseries, period=12):
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI'
            period: Seasonal period (number of observations)
            
        Returns:
            dict: Decomposition components
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df = timeseries.copy().sort_values('date').set_index('date')
        
        if len(df) < 2 * period:
            return {'error': 'Insufficient data for seasonal decomposition'}
        
        try:
            decomposition = seasonal_decompose(df['NDVI'], model='additive', period=period, extrapolate_trend='freq')
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
        except Exception as e:
            return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('processed_sentinel2_timeseries.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer()
    
    # Generate alert
    alert_result = analyzer.generate_alert(df)
    
    print("\n" + "="*60)
    print("EARLY WARNING SYSTEM - ALERT REPORT")
    print("="*60)
    print(f"\nAlert Level: {alert_result['icon']} {alert_result['label']}")
    print(f"Action: {alert_result['action']}")
    print(f"\nActive Alerts: {alert_result['alert_count']}")
    
    for alert in alert_result['alerts']:
        print(f"\n{alert['type'].upper()} ({alert['severity']})")
        print(f"  → {alert['message']}")
    
    print("\nTrend Analysis:")
    trend = alert_result['trend_analysis']
    print(f"  Trend: {trend['trend']}")
    print(f"  Confidence: {trend['confidence']}")
    if trend['slope'] is not None:
        print(f"  Slope: {trend['slope']:.6f}")
        print(f"  R²: {trend['r_squared']:.4f}")
    
    # Visualize
    analyzer.visualize_timeseries(df, 'ndvi_timeseries_analysis.png')
    
    print("\n" + "="*60)
