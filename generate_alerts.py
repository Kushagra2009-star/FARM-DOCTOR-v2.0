"""
Generate crop health alerts from time series data
"""
from temporal_analysis import TemporalAnalyzer
import pandas as pd

print("="*60)
print("CROP HEALTH ALERT SYSTEM")
print("="*60)

print("\nLoading data...")
df = pd.read_csv('processed_data.csv')

# Check what columns we actually have
print(f"Columns in CSV: {df.columns.tolist()}")

# Convert date to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
else:
    print("❌ Error: 'date' column not found!")
    exit(1)

print(f"Analyzing {len(df)} observations from {df['date'].min()} to {df['date'].max()}")

# IMPORTANT: TemporalAnalyzer expects lowercase column names
# So we need to rename columns to lowercase for the analyzer
df_analyzer = df.copy()

# Create a mapping of current names to what analyzer expects
column_mapping = {
    'NDVI': 'NDVI',  # Keep as is (analyzer uses NDVI uppercase)
    'NDWI': 'NDWI',  # Keep as is
    'EVI': 'EVI',    # Keep as is
    'SAVI': 'SAVI',  # Keep as is
}

# The analyzer actually uses UPPERCASE for indices
# Check the temporal_analysis.py to see what it expects

# Initialize analyzer
print("\nInitializing temporal analyzer...")
analyzer = TemporalAnalyzer()

# Generate alert
print("Running temporal analysis...")
try:
    alert = analyzer.generate_alert(df)
    
    # Display results
    print("\n" + "="*60)
    print("CROP HEALTH ALERT REPORT")
    print("="*60)
    print(f"\nAlert Level: {alert['icon']} {alert['label']}")
    print(f"Action Required: {alert['action']}")
    print(f"\nActive Alerts: {alert['alert_count']}")
    
    if alert['alerts']:
        print("\nDetailed Alerts:")
        for i, a in enumerate(alert['alerts'], 1):
            print(f"\n{i}. {a['type'].upper()} ({a['severity']})")
            print(f"   → {a['message']}")
    else:
        print("\n✓ No alerts - vegetation health is normal")
    
    print("\nTrend Analysis:")
    trend = alert['trend_analysis']
    print(f"  Trend: {trend['trend']}")
    print(f"  Confidence: {trend.get('confidence', 'N/A')}")
    if trend.get('slope') is not None:
        print(f"  Slope: {trend['slope']:.6f}")
        print(f"  R²: {trend.get('r_squared', 'N/A')}")
    
    # Create visualization
    print("\nGenerating time series visualization...")
    analyzer.visualize_timeseries(df, 'ndvi_timeseries.png')
    
    print("\n" + "="*60)
    print("✅ Alert report complete!")
    print("Chart saved to: ndvi_timeseries.png")
    print("="*60)

except KeyError as e:
    print(f"\n❌ Column Error: {e}")
    print("\nThe temporal analyzer is looking for a column that doesn't exist.")
    print("Available columns:", df.columns.tolist())
    print("\nPlease check temporal_analysis.py to see what column names it expects.")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nTroubleshooting:")
    print("1. Check that processed_data.csv has NDVI, NDWI columns")
    print("2. Ensure date column exists and is valid")
    print("3. Verify you have enough data (>3 observations)")