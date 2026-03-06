# AI-Assisted Satellite-Driven Crop Health Monitoring System
## Project Design Document

### Executive Summary
This document outlines the design and implementation of a satellite-based crop health monitoring system integrating Sentinel-2 imagery, spectral analysis, and machine learning for automated crop health classification, stress detection, and yield potential estimation.

---

## 1. System Architecture

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION                         │
│  Sentinel-2 Level-2A API → Cloud Filtering → Data Storage   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                     │
│  Cloud Masking → Spectral Band Extraction → Index Calc      │
│  (NDVI, NDWI, EVI, SAVI) → Normalization                   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                          │
│  Current Indices → Temporal Trends (ΔNDVI) →                │
│  Statistical Features → Spatial Context                      │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                  ML CLASSIFICATION                           │
│  Random Forest Classifier → Health Categories →             │
│  Confidence Scores → Uncertainty Quantification              │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              TEMPORAL ANALYSIS & ALERTS                      │
│  Time-Series Decomposition → Trend Detection →              │
│  Early Warning System → Anomaly Detection                    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│            YIELD POTENTIAL ESTIMATION                        │
│  Integrated NDVI → Baseline Comparison →                    │
│  Relative Yield Index (with clear limitations)              │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              VISUALIZATION & INTERFACE                       │
│  Leaflet.js Web GIS → Interactive Maps →                    │
│  Temporal Charts → Alert Dashboard                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Data Acquisition & Preprocessing

### 2.1 Sentinel-2 Specifications
- **Source**: ESA Copernicus / Google Earth Engine / Sentinel Hub
- **Product**: Level-2A (atmospherically corrected)
- **Resolution**: 10m (B2,B3,B4,B8), 20m (B5,B6,B7,B8A,B11,B12)
- **Temporal**: 5-day revisit (with both satellites)
- **Bands Used**:
  - B2 (Blue, 490nm)
  - B3 (Green, 560nm)
  - B4 (Red, 665nm)
  - B8 (NIR, 842nm)
  - B11 (SWIR1, 1610nm)
  - B12 (SWIR2, 2190nm)

### 2.2 Preprocessing Pipeline

```python
# Pseudocode for preprocessing
def preprocess_sentinel2(image, cloud_threshold=20):
    """
    1. Cloud masking using SCL band (Scene Classification)
    2. Remove pixels with cloud probability > threshold
    3. Scale reflectance values (0-10000 → 0-1)
    4. Interpolate missing values (spatial/temporal)
    """
    cloud_mask = image.select('SCL').lt(7)  # Remove clouds, shadows
    masked = image.updateMask(cloud_mask)
    scaled = masked.multiply(0.0001)
    return scaled
```

### 2.3 Vegetation Indices

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (NIR - Red) / (NIR + Red) | General vegetation health |
| NDWI | (Green - NIR) / (Green + NIR) | Water stress detection |
| EVI | 2.5 × ((NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)) | Enhanced sensitivity |
| SAVI | ((NIR - Red) / (NIR + Red + L)) × (1 + L), L=0.5 | Soil-adjusted |

---

## 3. Feature Engineering

### 3.1 Feature Vector Composition

For each field/patch at time t:

**Spectral Features (6)**:
- Current NDVI, NDWI, EVI, SAVI
- Red, NIR reflectance values

**Temporal Features (8)**:
- ΔNDVI (change from previous observation)
- 7-day, 14-day, 30-day NDVI trends
- Standard deviation over 30 days
- Maximum NDVI in growing season
- Days since peak NDVI
- Rate of change (derivative)

**Spatial Features (4)**:
- Mean NDVI of 3×3 neighborhood
- Spatial variance (texture)
- Distance to field boundary
- Field elevation (if DEM available)

**Meteorological Context (optional, 3)**:
- Cumulative rainfall (30-day)
- Growing degree days
- Days since last rain

**Total**: ~21 features per observation

### 3.2 Data Structure

```python
# Feature matrix format
features = {
    'field_id': str,
    'date': datetime,
    'lat': float,
    'lon': float,
    'ndvi_current': float,
    'ndvi_7d_trend': float,
    'ndvi_30d_std': float,
    # ... 18 more features
    'health_label': categorical  # Target variable
}
```

---

## 4. Machine Learning Model

### 4.1 Health Classification Categories

| Class | NDVI Range | Description | Proxy Label Logic |
|-------|------------|-------------|-------------------|
| **Healthy** | > 0.65 | Dense, vigorous vegetation | NDVI > 0.65 AND stable/increasing |
| **Mild Stress** | 0.45 - 0.65 | Moderate vegetation, early stress | NDVI declining OR 0.45-0.65 range |
| **Severe Stress** | 0.25 - 0.45 | Significant degradation | NDVI < 0.45 OR rapid decline |
| **Water Stressed** | Variable | High NDWI, specific pattern | NDWI > threshold AND NDVI declining |

### 4.2 Labeling Strategy (Proxy-Based)

Since ground truth is unavailable:

```python
def generate_proxy_labels(features):
    """
    Heuristic-based labeling using multiple indicators
    """
    if features['ndvi'] > 0.65 and features['ndvi_trend_30d'] >= -0.05:
        return 'Healthy'
    elif features['ndwi'] > 0.3 and features['ndvi_trend_7d'] < -0.1:
        return 'Water Stressed'
    elif features['ndvi'] < 0.45 or features['ndvi_trend_30d'] < -0.15:
        return 'Severe Stress'
    else:
        return 'Mild Stress'
```

**Limitations Acknowledged**:
- No agronomic ground truth
- Thresholds vary by crop type, region, season
- Indirect indicators only
- Subject to validation with domain experts

### 4.3 Random Forest Model

**Choice Rationale**:
- Handles non-linear relationships in spectral data
- Feature importance for interpretability
- Robust to outliers
- Probability outputs for uncertainty

**Architecture**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
```

**Training Strategy**:
- Train/validation/test split: 70/15/15
- Stratified sampling by health class
- Cross-validation (5-fold)
- Hyperparameter tuning via GridSearchCV

### 4.4 Optional CNN Approach

For image patch classification:

```python
# CNN architecture (experimental)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,6)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 health classes
])
```

---

## 5. Temporal Analysis & Early Warning

### 5.1 Time-Series Components

**Trend Detection**:
- Linear regression over 30-day window
- Mann-Kendall trend test (non-parametric)
- Seasonal decomposition (STL)

**Anomaly Detection**:
```python
def detect_anomalies(ndvi_timeseries):
    """
    Identify sudden drops or unusual patterns
    """
    rolling_mean = ndvi_timeseries.rolling(window=3).mean()
    rolling_std = ndvi_timeseries.rolling(window=3).std()
    
    # Points beyond 2 std deviations
    anomalies = abs(ndvi_timeseries - rolling_mean) > (2 * rolling_std)
    return anomalies
```

### 5.2 Early Warning System

**Alert Triggers**:
1. **Rapid Decline**: NDVI drop > 0.15 over 7 days
2. **Sustained Decline**: Negative trend over 14 days
3. **Below Threshold**: NDVI < 0.4 for 3+ consecutive observations
4. **Water Stress Pattern**: NDWI increase + NDVI decrease

**Alert Levels**:
- 🟢 **Normal**: No concerning patterns
- 🟡 **Watch**: Mild stress indicators
- 🟠 **Warning**: Moderate stress, investigation recommended
- 🔴 **Critical**: Severe stress, immediate action needed

---

## 6. Yield Potential Estimation

### 6.1 Methodology (Proxy-Based)

**Concept**: Cumulative vegetation vigor correlates with biomass production

```python
def estimate_yield_potential(ndvi_timeseries, baseline=None):
    """
    Relative yield index based on integrated NDVI
    NOT an absolute yield prediction
    """
    # Integrate NDVI over growing season
    integrated_ndvi = np.trapz(ndvi_timeseries, dx=5)  # 5-day intervals
    
    # Compare to baseline (historical average or regional norm)
    if baseline:
        relative_yield = (integrated_ndvi / baseline) * 100
    else:
        # Use current season max as reference
        relative_yield = (integrated_ndvi / integrated_ndvi.max()) * 100
    
    return {
        'relative_yield_index': relative_yield,
        'confidence': 'low',  # Acknowledge uncertainty
        'assumptions': [
            'Linear NDVI-yield relationship assumed',
            'No crop-specific calibration',
            'Weather/management factors not included',
            'For comparative analysis only'
        ]
    }
```

### 6.2 Limitations (CRITICAL)

⚠️ **This is NOT precision agriculture yield prediction**:
- No ground-truth calibration
- No crop type differentiation
- Ignores: soil quality, irrigation, fertilization, pests, weather extremes
- For research/education demonstration only

---

## 7. Evaluation Metrics

### 7.1 Classification Performance

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP + TN) / Total | Overall correctness |
| Precision | TP / (TP + FP) | Positive prediction reliability |
| Recall | TP / (TP + FN) | Capture rate per class |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | Balanced metric |
| Confusion Matrix | Class-by-class errors | Error pattern analysis |

### 7.2 Temporal Analysis Validation

- **Trend Accuracy**: Compare detected trends with visual inspection
- **Alert Precision**: False positive rate for warnings
- **Lead Time**: Days before visible stress manifestation

### 7.3 Qualitative Error Analysis

**Expected Failure Modes**:
1. Cloud contamination in tropical regions
2. Mixed pixels at field boundaries
3. Crop type confusion (e.g., stressed wheat vs healthy grassland)
4. Seasonal phenology misinterpretation
5. Atmospheric effects not fully corrected

---

## 8. Web Interface Design

### 8.1 Frontend Technology Stack

- **Framework**: HTML5 + CSS3 + JavaScript (Vanilla or Vue.js)
- **Mapping**: Leaflet.js with tile layers
- **Charts**: Chart.js or Plotly.js
- **Backend**: Python Flask API (or static JSON for demo)
- **Hosting**: GitHub Pages, Netlify, or Vercel

### 8.2 User Interface Components

**Main Dashboard**:
```
┌─────────────────────────────────────────────────────────┐
│  [Logo] Crop Health Monitoring System     [Help] [About]│
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────┐  ┌────────────────────────┐  │
│  │   Interactive Map    │  │   Health Summary       │  │
│  │   (Leaflet.js)       │  │   🟢 Healthy: 65%      │  │
│  │                      │  │   🟡 Mild: 25%         │  │
│  │   [Color-coded       │  │   🟠 Severe: 8%        │  │
│  │    field polygons]   │  │   🔴 Critical: 2%      │  │
│  │                      │  │                        │  │
│  └──────────────────────┘  └────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  NDVI Time Series                                │  │
│  │  [Line chart with trend line and anomaly markers]│  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Active Alerts                                    │  │
│  │  🔴 Field A3: Severe stress detected (2024-02-05)│  │
│  │  🟠 Field B7: Declining trend (2024-02-07)       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 8.3 Interactive Features

- Click field → View detailed health report
- Date slider → Temporal animation
- Layer toggles → NDVI, NDWI, RGB composite
- Export → Download reports as PDF/CSV

---

## 9. Implementation Phases

### Phase 1: Data Pipeline (Weeks 1-2)
- Set up Sentinel-2 API access (Google Earth Engine or Sentinel Hub)
- Implement preprocessing pipeline
- Calculate vegetation indices
- Store processed data in structured format

### Phase 2: Feature Engineering (Week 3)
- Extract temporal features
- Build feature vectors
- Create proxy labeling system
- Generate training dataset

### Phase 3: ML Model Development (Weeks 4-5)
- Train Random Forest classifier
- Hyperparameter tuning
- Evaluate performance metrics
- Implement uncertainty quantification

### Phase 4: Temporal Analysis (Week 6)
- Time-series trend detection
- Anomaly detection algorithms
- Early warning system logic

### Phase 5: Yield Estimation Module (Week 7)
- Implement integrated NDVI calculation
- Baseline comparison logic
- Clear limitation documentation

### Phase 6: Web Interface (Weeks 8-9)
- Leaflet.js map integration
- Chart visualizations
- Alert dashboard
- Responsive design

### Phase 7: Testing & Documentation (Week 10)
- End-to-end system testing
- Performance optimization
- Comprehensive documentation
- Research paper draft

---

## 10. Technical Stack Summary

**Backend**:
- Python 3.9+
- Libraries: `rasterio`, `geopandas`, `numpy`, `pandas`, `scikit-learn`
- Satellite API: `earthengine-api` or `sentinelhub`
- Optional: TensorFlow/Keras for CNN

**Frontend**:
- Leaflet.js 1.9+
- Chart.js 4.0+
- Bootstrap 5 (responsive UI)

**Data Storage**:
- GeoTIFF for rasters
- GeoJSON for vector data
- SQLite/PostgreSQL for features
- Cloud storage: AWS S3 or Google Cloud Storage

**Deployment**:
- GitHub repository (version control)
- GitHub Pages (static site)
- Docker containerization (optional)

---

## 11. Limitations & Ethical Considerations

### 11.1 Technical Limitations

1. **Spatial Resolution**: 10m pixels may miss sub-field variability
2. **Temporal Gaps**: Clouds can create data gaps during critical periods
3. **Indirect Measurement**: NDVI ≠ direct crop health
4. **No Ground Truth**: Proxy labels introduce systematic bias
5. **Crop Agnostic**: Model doesn't differentiate wheat, corn, rice, etc.
6. **Regional Specificity**: Thresholds need local calibration

### 11.2 Operational Constraints

- Requires technical expertise to interpret results
- Not a replacement for agronomic knowledge
- Should complement, not replace, field scouting
- Yield estimates are relative, not absolute

### 11.3 Ethical Considerations

- **Transparency**: All assumptions clearly stated
- **Data Privacy**: Farmer location data must be protected
- **Accessibility**: Free/low-cost tools prioritized
- **No Overpromising**: Acknowledge what system cannot do
- **Educational Focus**: Learning tool, not commercial product

---

## 12. Future Enhancements

### 12.1 Near-Term (3-6 months)

- Integrate weather data APIs (rainfall, temperature)
- Add crop type classification module
- Implement mobile app for field technicians
- Multi-language support

### 12.2 Medium-Term (6-12 months)

- Ground-truth data collection campaign
- Advanced time-series models (LSTM, Prophet)
- Soil moisture integration (Sentinel-1 SAR)
- Pest/disease detection from high-res imagery

### 12.3 Long-Term (1-2 years)

- Integration with autonomous agricultural robots
- Real-time alert system via SMS/WhatsApp
- Prescription mapping for variable-rate application
- Blockchain for data provenance and farmer records

---

## 13. Research Documentation Plan

### 13.1 Paper Structure

1. **Abstract**: Problem, approach, results, impact
2. **Introduction**: Agricultural challenges, remote sensing opportunity
3. **Related Work**: Survey of crop monitoring systems
4. **Methodology**: Detailed pipeline description
5. **Results**: Performance metrics, case studies
6. **Discussion**: Limitations, lessons learned
7. **Conclusion**: Future work, societal impact
8. **Appendix**: Code repository, data sources

### 13.2 Reproducibility Checklist

- ✅ All code in public GitHub repository
- ✅ Requirements.txt with exact package versions
- ✅ Sample data and preprocessing scripts
- ✅ Model training notebooks (Jupyter)
- ✅ API documentation
- ✅ Video demonstration

---

## 14. Success Criteria

**Minimum Viable Product (MVP)**:
- ✅ Successfully download and process Sentinel-2 imagery
- ✅ Compute NDVI/NDWI for user-selected fields
- ✅ Train ML model with >70% accuracy on validation set
- ✅ Deploy interactive web map showing health classifications
- ✅ Generate temporal NDVI charts

**Stretch Goals**:
- 🎯 Achieve >80% classification accuracy
- 🎯 Demonstrate early warning 7+ days before visible stress
- 🎯 Validate results with at least one real farm dataset
- 🎯 Publish preprint on arXiv or AgriXiv
- 🎯 Present at agricultural technology conference

---

## 15. Conclusion

This system represents a bridge between cutting-edge space technology and practical agricultural needs. By maintaining scientific rigor, acknowledging limitations, and prioritizing transparency, the project aims to contribute meaningfully to the intersection of AI, remote sensing, and sustainable agriculture.

**Core Philosophy**: 
*"Not to replace farmers, but to empower them with data-driven insights derived from space."*

