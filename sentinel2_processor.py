"""
Sentinel-2 Data Acquisition and Preprocessing Pipeline
======================================================

This module handles downloading Sentinel-2 imagery, cloud masking,
and computation of vegetation indices (NDVI, NDWI, EVI, SAVI).

Dependencies:
    - earthengine-api (Google Earth Engine)
    - pandas
    - numpy

Author: Agricultural AI Research Project
License: MIT
"""

import ee
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class Sentinel2Processor:
    """
    Processes Sentinel-2 Level-2A imagery for crop health monitoring.
    """
    
    def __init__(self):
        """
        Initialize processor with default parameters.
        """
        self.cloud_threshold = 20  # Maximum cloud cover percentage
        
    def get_sentinel2_collection(self, geometry, start_date, end_date):
        """
        Retrieve Sentinel-2 Level-2A image collection for specified area and time.
        
        Args:
            geometry: ee.Geometry object (field boundary)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            ee.ImageCollection: Filtered Sentinel-2 collection
        """
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_threshold))
                     .sort('system:time_start'))
        
        return collection
    
    def mask_clouds(self, image):
        """
        Apply cloud and shadow masking using Scene Classification Layer (SCL).
        
        Args:
            image: ee.Image - Sentinel-2 image
            
        Returns:
            ee.Image: Cloud-masked image
        """
        # SCL band classification:
        # 0: No data, 1: Saturated, 2: Dark areas, 3: Cloud shadows
        # 4: Vegetation, 5: Not vegetated, 6: Water
        # 7: Unclassified, 8: Cloud medium prob, 9: Cloud high prob
        # 10: Thin cirrus, 11: Snow/ice
        
        scl = image.select('SCL')
        
        # Mask clouds, shadows, and other problematic pixels
        mask = (scl.eq(4)  # Vegetation
               .Or(scl.eq(5))  # Not vegetated
               .Or(scl.eq(6))  # Water
               .Or(scl.eq(11)))  # Snow (keep for winter monitoring)
        
        return image.updateMask(mask)
    
    def compute_ndvi(self, image):
        """
        Compute Normalized Difference Vegetation Index.
        
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1 (vegetation typically 0.2-0.9)
        
        Args:
            image: ee.Image with B4 (Red) and B8 (NIR) bands
            
        Returns:
            ee.Image: Original image with NDVI band added
        """
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def compute_ndwi(self, image):
        """
        Compute Normalized Difference Water Index.
        
        NDWI = (Green - NIR) / (Green + NIR)
        Indicates water content in vegetation
        
        Args:
            image: ee.Image with B3 (Green) and B8 (NIR) bands
            
        Returns:
            ee.Image: Original image with NDWI band added
        """
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands(ndwi)
    
    def compute_evi(self, image):
        """
        Compute Enhanced Vegetation Index.
        
        EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        More sensitive than NDVI in high biomass regions
        
        Args:
            image: ee.Image with B2, B4, B8 bands
            
        Returns:
            ee.Image: Original image with EVI band added
        """
        nir = image.select('B8')
        red = image.select('B4')
        blue = image.select('B2')
        
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': nir,
                'RED': red,
                'BLUE': blue
            }
        ).rename('EVI')
        
        return image.addBands(evi)
    
    def compute_savi(self, image, L=0.5):
        """
        Compute Soil-Adjusted Vegetation Index.
        
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        L = 0.5 is typical for moderate vegetation cover
        
        Args:
            image: ee.Image with B4, B8 bands
            L: Soil brightness correction factor (0-1)
            
        Returns:
            ee.Image: Original image with SAVI band added
        """
        nir = image.select('B8')
        red = image.select('B4')
        
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
            {
                'NIR': nir,
                'RED': red,
                'L': L
            }
        ).rename('SAVI')
        
        return image.addBands(savi)
    
    def process_image(self, image):
        """
        Complete preprocessing pipeline for single image.
        
        Pipeline:
        1. Cloud masking
        2. Scale to reflectance (0-1)
        3. Compute all vegetation indices
        
        Args:
            image: ee.Image - Raw Sentinel-2 image
            
        Returns:
            ee.Image: Processed image with all indices
        """
        # Cloud masking
        masked = self.mask_clouds(image)
        
        # Scale reflectance values
        scaled = masked.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
                       .multiply(0.0001)
        
        # Compute all indices step by step (NO .pipe() - that's Pandas, not Earth Engine!)
        with_ndvi = self.compute_ndvi(scaled)
        with_ndwi = self.compute_ndwi(with_ndvi)
        with_evi = self.compute_evi(with_ndwi)
        with_indices = self.compute_savi(with_evi)
        
        # Copy metadata
        return with_indices.copyProperties(image, ['system:time_start'])
    
    def extract_point_timeseries(self, geometry, start_date, end_date, scale=10):
        """
        Extract time series of vegetation indices for a point or small region.
        
        Args:
            geometry: ee.Geometry - Point or polygon
            start_date: str - Start date (YYYY-MM-DD)
            end_date: str - End date (YYYY-MM-DD)
            scale: int - Spatial resolution in meters
            
        Returns:
            pandas.DataFrame: Time series of indices
        """
        collection = self.get_sentinel2_collection(geometry, start_date, end_date)
        processed = collection.map(self.process_image)
        
        def extract_values(image):
            """Extract mean values for each band"""
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9
            )
            
            return ee.Feature(None, stats).set('date', image.date().format('YYYY-MM-dd'))
        
        features = processed.map(extract_values)
        
        # Convert to pandas DataFrame
        info = features.getInfo()
        data = [feat['properties'] for feat in info['features']]
        df = pd.DataFrame(data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    def export_image(self, image, geometry, filename, scale=10):
        """
        Export processed image as GeoTIFF.
        
        Args:
            image: ee.Image to export
            geometry: ee.Geometry - Export region
            filename: str - Output filename
            scale: int - Resolution in meters
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            scale=scale,
            region=geometry,
            fileFormat='GeoTIFF',
            maxPixels=1e9
        )
        
        task.start()
        print(f"Export task started: {filename}")
        return task


class FeatureExtractor:
    """
    Extracts machine learning features from processed imagery.
    """
    
    def __init__(self):
        self.feature_columns = []
        
    def compute_temporal_features(self, timeseries_df):
        """
        Compute temporal features from NDVI time series.
        
        Features:
        - Current value
        - Change from previous observation (ΔNDVI)
        - 7-day, 14-day, 30-day trends
        - Standard deviation over 30 days
        - Days since peak NDVI
        - Rate of change
        
        Args:
            timeseries_df: DataFrame with 'date' and 'NDVI' columns
            
        Returns:
            DataFrame: Original data with temporal features added
        """
        df = timeseries_df.copy()
        df = df.sort_values('date')
        
        # Compute differences
        df['NDVI_delta'] = df['NDVI'].diff()
        
        # Rolling statistics
        df['NDVI_7d_mean'] = df['NDVI'].rolling(window=2, min_periods=1).mean()
        df['NDVI_14d_mean'] = df['NDVI'].rolling(window=3, min_periods=1).mean()
        df['NDVI_30d_mean'] = df['NDVI'].rolling(window=6, min_periods=1).mean()
        df['NDVI_30d_std'] = df['NDVI'].rolling(window=6, min_periods=1).std()
        
        # Trend calculation (linear regression slope)
        df['NDVI_trend_7d'] = self._calculate_trend(df['NDVI'], window=2)
        df['NDVI_trend_30d'] = self._calculate_trend(df['NDVI'], window=6)
        
        # Peak detection
        df['NDVI_season_max'] = df['NDVI'].expanding().max()
        
        # Days since peak (handle case where peak might not exist yet)
        if len(df) > 0 and df['NDVI'].max() > 0:
            peak_idx = df['NDVI'].idxmax()
            df['days_since_peak'] = (df['date'] - df.loc[peak_idx, 'date']).dt.days
        else:
            df['days_since_peak'] = 0
        
        # Rate of change (derivative approximation)
        df['NDVI_rate'] = df['NDVI_delta'] / 5  # Assuming 5-day revisit
        
        return df
    
    def _calculate_trend(self, series, window):
        """
        Calculate linear trend (slope) over rolling window.
        
        Args:
            series: pandas.Series
            window: int - Window size
            
        Returns:
            pandas.Series: Trend values
        """
        trends = []
        for i in range(len(series)):
            if i < window - 1:
                trends.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    trends.append(slope)
                else:
                    trends.append(np.nan)
        return pd.Series(trends, index=series.index)
    
    def create_feature_vector(self, row):
        """
        Create feature vector for ML model from single observation.
        
        Args:
            row: DataFrame row with all computed features
            
        Returns:
            dict: Feature vector
        """
        features = {
            # Spectral features
            'ndvi': row.get('NDVI', np.nan),
            'ndwi': row.get('NDWI', np.nan),
            'evi': row.get('EVI', np.nan),
            'savi': row.get('SAVI', np.nan),
            'red': row.get('B4', np.nan),
            'nir': row.get('B8', np.nan),
            
            # Temporal features
            'ndvi_delta': row.get('NDVI_delta', np.nan),
            'ndvi_7d_mean': row.get('NDVI_7d_mean', np.nan),
            'ndvi_30d_mean': row.get('NDVI_30d_mean', np.nan),
            'ndvi_30d_std': row.get('NDVI_30d_std', np.nan),
            'ndvi_trend_7d': row.get('NDVI_trend_7d', np.nan),
            'ndvi_trend_30d': row.get('NDVI_trend_30d', np.nan),
            'ndvi_season_max': row.get('NDVI_season_max', np.nan),
            'days_since_peak': row.get('days_since_peak', np.nan),
            'ndvi_rate': row.get('NDVI_rate', np.nan),
        }
        
        return features
    
    def generate_proxy_labels(self, df):
        """
        Generate health class labels using heuristic rules.
        
        IMPORTANT: These are PROXY labels, not ground truth!
        
        Classes:
        - Healthy: NDVI > 0.65, stable or increasing
        - Mild Stress: NDVI 0.45-0.65, or moderate decline
        - Severe Stress: NDVI < 0.45, or rapid decline
        - Water Stressed: High NDWI + declining NDVI
        
        Args:
            df: DataFrame with NDVI, NDWI, and temporal features
            
        Returns:
            pandas.Series: Health class labels
        """
        labels = []
        
        for idx, row in df.iterrows():
            ndvi = row.get('NDVI', 0)
            ndwi = row.get('NDWI', 0)
            trend_30d = row.get('NDVI_trend_30d', 0)
            trend_7d = row.get('NDVI_trend_7d', 0)
            
            # Handle NaN values
            if pd.isna(ndvi):
                labels.append('Unknown')
                continue
            
            if pd.isna(trend_30d):
                trend_30d = 0
            if pd.isna(trend_7d):
                trend_7d = 0
            if pd.isna(ndwi):
                ndwi = 0
            
            # Water stress pattern
            if ndwi > 0.3 and trend_7d < -0.1:
                label = 'Water Stressed'
            # Severe stress
            elif ndvi < 0.45 or trend_30d < -0.15:
                label = 'Severe Stress'
            # Healthy vegetation
            elif ndvi > 0.65 and trend_30d >= -0.05:
                label = 'Healthy'
            # Mild stress (default)
            else:
                label = 'Mild Stress'
            
            labels.append(label)
        
        return pd.Series(labels, index=df.index, name='health_label')


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("SENTINEL-2 PROCESSOR - EXAMPLE USAGE")
    print("="*60)
    
    # This will only run if you execute this file directly
    print("\nThis is a module - import it in your scripts!")
    print("\nExample:")
    print("  from sentinel2_processor import Sentinel2Processor, FeatureExtractor")
    print("  processor = Sentinel2Processor()")
    print("  point = ee.Geometry.Point([lon, lat])")
    print("  data = processor.extract_point_timeseries(point, '2024-01-01', '2024-06-30')")
    print("\nSee quick_start.py for complete example!")
    print("="*60)