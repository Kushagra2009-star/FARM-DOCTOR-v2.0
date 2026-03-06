"""
Quick start script - Download satellite data for a location
"""

from sentinel2_processor import Sentinel2Processor, FeatureExtractor
import ee

# Initialize Earth Engine
try:
    ee.Initialize(project='farm-doctor-489111')
    print("Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error: {e}") 
    print("Authrnticationwith Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project='farm-doctor-489111')
    print("Authentication complete!")

# Create processor
processor = Sentinel2Processor()


latitude = 27.505328 
longitude = 79.401622

print(f"\nFetching data for: ({latitude}, {longitude})")

# Create point geometry
point = ee.Geometry.Point([longitude, latitude])

# Get time series data
print("Downloading Sentinel-2 data (this may take 1-2 minutes)...")
timeseries = processor.extract_point_timeseries(
    geometry=point,
    start_date='2023-01-01',
    end_date='2025-06-30',  # 12 months of data
    scale=10
)

print(f"\n✓ Downloaded {len(timeseries)} observations!")

# Show sample data
print("\nFirst 5 observations:")
print(timeseries[['date', 'NDVI', 'NDWI', 'EVI']].head())

# Save to CSV
output_file = 'sample_data.csv'
timeseries.to_csv(output_file, index=False)
print(f"\n✓ Data saved to: {output_file}")

# Add temporal features
print("\nComputing temporal features...")
extractor = FeatureExtractor()
features = extractor.compute_temporal_features(timeseries)

# Generate health labels
features['health_label'] = extractor.generate_proxy_labels(features)

print("\nHealth class distribution:")
print(features['health_label'].value_counts())

# Save enriched data
features.to_csv('processed_data.csv', index=False)
print("\n✓ Processed data saved to: processed_data.csv")

print("\n" + "="*60)
print("SUCCESS! Your system is working!")
print("="*60)