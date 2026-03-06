"""
Relative Yield Potential Estimation
===================================

IMPORTANT DISCLAIMER:
This module provides RELATIVE yield potential indices based on integrated
NDVI values. It does NOT predict absolute crop yields and should NOT be
used for commercial yield forecasting without proper calibration and
ground-truth validation.

Limitations:
- No crop-specific calibration
- No soil/weather/management factor integration
- Proxy-based approach without ground truth
- For research and educational purposes only

Dependencies:
    - numpy
    - pandas
    - matplotlib

Author: Agricultural AI Research Project
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings


class YieldEstimator:
    """
    Estimates relative yield potential from NDVI time series.
    
    CRITICAL: This is a PROXY-based approach for demonstration purposes.
    """
    
    def __init__(self):
        self.assumptions = [
            'Linear relationship between integrated NDVI and biomass',
            'No crop-specific parameters',
            'No soil quality adjustment',
            'No weather extreme considerations',
            'No pest/disease impact modeling',
            'No irrigation/fertilization history',
            'For comparative analysis only'
        ]
        
        self.confidence_level = 'LOW'
        
    def integrate_ndvi(self, timeseries, start_date=None, end_date=None):
        """
        Calculate integrated NDVI over growing season.
        
        Concept: Cumulative vegetation vigor approximates biomass accumulation.
        
        Args:
            timeseries: DataFrame with 'date' and 'NDVI' columns
            start_date: Season start (optional, uses first date if None)
            end_date: Season end (optional, uses last date if None)
            
        Returns:
            float: Integrated NDVI value
        """
        df = timeseries.copy().sort_values('date')
        
        # Filter by date range if provided
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        if len(df) < 2:
            warnings.warn("Insufficient data for integration")
            return np.nan
        
        # Numerical integration using trapezoidal rule
        # Assuming approximately 5-day intervals between observations
        days = (df['date'] - df['date'].iloc[0]).dt.days.values
        ndvi_values = df['NDVI'].values
        
        integrated_ndvi = np.trapz(ndvi_values, days)
        
        return integrated_ndvi
    
    def estimate_relative_yield(self, current_timeseries, baseline_timeseries=None, 
                                baseline_value=None):
        """
        Estimate relative yield potential compared to baseline.
        
        Args:
            current_timeseries: DataFrame for current season
            baseline_timeseries: DataFrame for baseline season (optional)
            baseline_value: Pre-computed baseline integrated NDVI (optional)
            
        Returns:
            dict: Relative yield index and metadata
        """
        # Integrate current season
        current_integrated = self.integrate_ndvi(current_timeseries)
        
        # Determine baseline
        if baseline_value is not None:
            baseline = baseline_value
        elif baseline_timeseries is not None:
            baseline = self.integrate_ndvi(baseline_timeseries)
        else:
            # Use current season max as reference (least reliable)
            baseline = current_timeseries['NDVI'].max() * 100  # Approximate
            warnings.warn("No baseline provided, using current season max as reference")
        
        # Calculate relative index
        if baseline > 0:
            relative_yield_index = (current_integrated / baseline) * 100
        else:
            relative_yield_index = np.nan
        
        # Qualitative assessment
        if relative_yield_index > 95:
            performance = 'Excellent'
        elif relative_yield_index > 85:
            performance = 'Good'
        elif relative_yield_index > 70:
            performance = 'Fair'
        elif relative_yield_index > 50:
            performance = 'Poor'
        else:
            performance = 'Very Poor'
        
        return {
            'relative_yield_index': relative_yield_index,
            'performance_category': performance,
            'current_integrated_ndvi': current_integrated,
            'baseline_integrated_ndvi': baseline,
            'confidence': self.confidence_level,
            'assumptions': self.assumptions,
            'timestamp': datetime.now().isoformat(),
            'interpretation': self._generate_interpretation(relative_yield_index)
        }
    
    def _generate_interpretation(self, index):
        """
        Generate plain-language interpretation with caveats.
        
        Args:
            index: Relative yield index (percentage)
            
        Returns:
            str: Interpretation text
        """
        if np.isnan(index):
            return "Unable to compute yield estimate due to insufficient data."
        
        interpretation = f"""
RELATIVE YIELD POTENTIAL INTERPRETATION:

Index Value: {index:.1f}% of baseline

This value represents the current season's vegetation vigor relative to a 
baseline period. It is NOT a prediction of absolute crop yield.

What this means:
- {index:.1f}% suggests vegetation performance is {"above" if index > 100 else "at" if index >= 95 else "below"} baseline levels
- Higher values generally correlate with better biomass accumulation
- This metric does NOT account for:
  * Crop type and variety
  * Soil characteristics
  * Weather events (drought, flooding, frost)
  * Pest and disease pressure
  * Management practices (irrigation, fertilization)
  * Market factors

CRITICAL CAVEAT:
This is a COMPARATIVE tool, not a yield forecasting system. Use alongside
agronomic expertise and field observations. Do not make management decisions
based solely on this metric.
        """
        
        return interpretation.strip()
    
    def analyze_yield_variability(self, timeseries_list, labels=None):
        """
        Compare yield potential across multiple fields or seasons.
        
        Args:
            timeseries_list: List of DataFrames
            labels: List of labels for each timeseries
            
        Returns:
            DataFrame: Comparative analysis
        """
        if labels is None:
            labels = [f"Field {i+1}" for i in range(len(timeseries_list))]
        
        results = []
        
        for ts, label in zip(timeseries_list, labels):
            integrated = self.integrate_ndvi(ts)
            max_ndvi = ts['NDVI'].max()
            mean_ndvi = ts['NDVI'].mean()
            
            results.append({
                'field': label,
                'integrated_ndvi': integrated,
                'max_ndvi': max_ndvi,
                'mean_ndvi': mean_ndvi
            })
        
        df = pd.DataFrame(results)
        
        # Calculate relative performance
        baseline = df['integrated_ndvi'].max()
        df['relative_index'] = (df['integrated_ndvi'] / baseline) * 100
        df['rank'] = df['relative_index'].rank(ascending=False)
        
        return df
    
    def visualize_yield_comparison(self, comparison_df, save_path='yield_comparison.png'):
        """
        Create visualization of yield potential comparison.
        
        Args:
            comparison_df: DataFrame from analyze_yield_variability
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of relative yield index
        colors = ['green' if x > 90 else 'yellow' if x > 75 else 'orange' 
                 for x in comparison_df['relative_index']]
        
        axes[0].bar(comparison_df['field'], comparison_df['relative_index'], color=colors)
        axes[0].set_ylabel('Relative Yield Index (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Relative Yield Potential by Field', fontsize=14, fontweight='bold')
        axes[0].axhline(y=100, color='black', linestyle='--', label='Baseline')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Scatter: Mean NDVI vs Integrated NDVI
        axes[1].scatter(comparison_df['mean_ndvi'], comparison_df['integrated_ndvi'], 
                       s=100, c=comparison_df['relative_index'], cmap='RdYlGn', 
                       vmin=50, vmax=110)
        
        for idx, row in comparison_df.iterrows():
            axes[1].annotate(row['field'], (row['mean_ndvi'], row['integrated_ndvi']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1].set_xlabel('Mean NDVI', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Integrated NDVI', fontsize=12, fontweight='bold')
        axes[1].set_title('NDVI Metrics Correlation', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
        cbar.set_label('Relative Index (%)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Yield comparison visualization saved to '{save_path}'")
    
    def generate_report(self, yield_estimate, timeseries, save_path='yield_report.txt'):
        """
        Generate detailed text report with all caveats.
        
        Args:
            yield_estimate: Dict from estimate_relative_yield
            timeseries: Original timeseries data
            save_path: Path to save report
        """
        report = f"""
{'='*70}
RELATIVE YIELD POTENTIAL REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER:
This report provides a RELATIVE yield index for comparative purposes only.
It is NOT an absolute yield prediction and should NOT replace agronomic
expertise or field-based assessments.

{'='*70}
RESULTS SUMMARY
{'='*70}

Relative Yield Index: {yield_estimate['relative_yield_index']:.1f}%
Performance Category: {yield_estimate['performance_category']}
Confidence Level: {yield_estimate['confidence']}

Integrated NDVI (Current): {yield_estimate['current_integrated_ndvi']:.2f}
Integrated NDVI (Baseline): {yield_estimate['baseline_integrated_ndvi']:.2f}

{'='*70}
VEGETATION STATISTICS
{'='*70}

Observations: {len(timeseries)}
Date Range: {timeseries['date'].min().strftime('%Y-%m-%d')} to {timeseries['date'].max().strftime('%Y-%m-%d')}

NDVI Statistics:
  Mean: {timeseries['NDVI'].mean():.3f}
  Max: {timeseries['NDVI'].max():.3f}
  Min: {timeseries['NDVI'].min():.3f}
  Std Dev: {timeseries['NDVI'].std():.3f}

{'='*70}
INTERPRETATION
{'='*70}

{yield_estimate['interpretation']}

{'='*70}
ASSUMPTIONS & LIMITATIONS
{'='*70}

This analysis is based on the following assumptions:

"""
        for i, assumption in enumerate(self.assumptions, 1):
            report += f"{i}. {assumption}\n"
        
        report += f"""
{'='*70}
RECOMMENDED ACTIONS
{'='*70}

1. Use this index as ONE input among many for decision-making
2. Validate with field observations and scouting
3. Consider local weather, pest, and disease reports
4. Consult with agronomists for crop-specific interpretation
5. Do NOT use for financial or insurance purposes without proper validation

{'='*70}
TECHNICAL NOTES
{'='*70}

Method: Trapezoidal integration of NDVI time series
Data Source: Sentinel-2 Level-2A satellite imagery
Spatial Resolution: 10 meters
Temporal Resolution: ~5 days (weather permitting)

Contact: Agricultural AI Research Project
For more information: [Your contact/documentation link]

{'='*70}
END OF REPORT
{'='*70}
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"\nYield report saved to '{save_path}'")
        
        return report


# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('processed_sentinel2_timeseries.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize estimator
    estimator = YieldEstimator()
    
    print("\n" + "="*70)
    print("YIELD POTENTIAL ESTIMATION MODULE - DEMONSTRATION")
    print("="*70)
    
    # Show assumptions
    print("\nIMPORTANT ASSUMPTIONS:")
    for i, assumption in enumerate(estimator.assumptions, 1):
        print(f"  {i}. {assumption}")
    
    # Estimate yield (using historical max as baseline for demo)
    baseline_value = df['NDVI'].max() * 150  # Simulated historical baseline
    
    yield_estimate = estimator.estimate_relative_yield(
        current_timeseries=df,
        baseline_value=baseline_value
    )
    
    print("\n" + "="*70)
    print("ESTIMATION RESULTS")
    print("="*70)
    print(f"Relative Yield Index: {yield_estimate['relative_yield_index']:.1f}%")
    print(f"Performance: {yield_estimate['performance_category']}")
    print(f"Confidence: {yield_estimate['confidence']}")
    
    # Generate full report
    report = estimator.generate_report(yield_estimate, df, 'yield_estimate_report.txt')
    
    print("\n⚠️  REMEMBER: This is a RELATIVE metric for COMPARISON, not absolute yield prediction!")
    print("="*70)
