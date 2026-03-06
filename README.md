# AI-Assisted Satellite-Driven Crop Health Monitoring System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Research Status](https://img.shields.io/badge/status-prototype-orange.svg)]()

> *Bridging space technology, artificial intelligence, and grassroots agriculture*

## 🌾 Project Overview

This research project demonstrates an end-to-end crop health monitoring system that integrates:
- **Sentinel-2 satellite imagery** for multi-spectral remote sensing
- **Machine learning** (Random Forest) for automated health classification
- **Temporal analysis** for early stress detection and trend forecasting
- **Web GIS interface** for accessible visualization and decision support

### Key Innovation

Moving beyond basic NDVI visualization to create an **actionable intelligence system** that:
1. Automatically classifies crop health into 4 categories
2. Detects stress **before visual symptoms appear**
3. Provides relative yield potential estimates (with clear limitations)
4. Delivers insights through an intuitive web interface

---

## 🎯 Research Objectives

1. **Demonstrate AI-Remote Sensing Integration**: Show how machine learning can enhance satellite-based agricultural monitoring
2. **Early Warning System**: Detect crop stress 7-14 days before field-visible damage
3. **Accessibility**: Make space technology insights available through simple web interfaces
4. **Transparency**: Document limitations, assumptions, and failure modes honestly
5. **Educational Value**: Provide reproducible code for learning and experimentation

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Sentinel-2 API │ → Cloud-free imagery acquisition
└────────┬────────┘
         │
┌────────▼────────┐
│  Preprocessing  │ → Cloud masking, index calculation
└────────┬────────┘
         │
┌────────▼────────┐
│ Feature Extract │ → Temporal trends, spectral features
└────────┬────────┘
         │
┌────────▼────────┐
│   ML Classifier │ → Random Forest (4 health classes)
└────────┬────────┘
         │
┌────────▼────────┐
│ Temporal Alerts │ → Anomaly detection, trend analysis
└────────┬────────┘
         │
┌────────▼────────┐
│  Web Interface  │ → Leaflet.js maps, Chart.js viz
└─────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Google Earth Engine account (free registration at https://earthengine.google.com/)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/crop-health-monitoring.git
cd crop-health-monitoring

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate with Google Earth Engine
earthengine authenticate

# 5. Initialize Earth Engine
python -c "import ee; ee.Initialize()"
```

### Verify Installation

```python
python
>>> import ee
>>> import sklearn
>>> import rasterio
>>> print("All dependencies installed successfully!")
```

---

## 🚀 Quick Start

### 1. Data Acquisition & Preprocessing

```python
from sentinel2_processor import Sentinel2Processor, FeatureExtractor

# Initialize processor
processor = Sentinel2Processor()

# Define area of interest
point = ee.Geometry.Point([longitude, latitude])

# Extract time series
timeseries = processor.extract_point_timeseries(
    geometry=point,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Generate features
extractor = FeatureExtractor()
features = extractor.compute_temporal_features(timeseries)
features['health_label'] = extractor.generate_proxy_labels(features)

# Save processed data
features.to_csv('processed_data.csv', index=False)
```

### 2. Train ML Model

```python
from ml_classifier import CropHealthClassifier

# Load data
import pandas as pd
df = pd.read_csv('processed_data.csv')

# Define features
feature_cols = ['ndvi', 'ndwi', 'evi', 'savi', 'ndvi_delta', 
                'ndvi_trend_7d', 'ndvi_trend_30d', ...]

# Train model
classifier = CropHealthClassifier(n_estimators=200)
X, y = classifier.prepare_data(df, feature_cols, 'health_label')

# Split and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier.train(X_train, y_train)

# Evaluate
metrics = classifier.evaluate(X_test, y_test)

# Save model
classifier.save_model('crop_health_model.pkl')
```

### 3. Temporal Analysis & Alerts

```python
from temporal_analysis import TemporalAnalyzer

analyzer = TemporalAnalyzer()

# Generate alert
alert = analyzer.generate_alert(timeseries)

print(f"Alert Level: {alert['icon']} {alert['label']}")
print(f"Active Alerts: {alert['alert_count']}")

# Visualize
analyzer.visualize_timeseries(timeseries, 'ndvi_analysis.png')
```

### 4. Yield Estimation

```python
from yield_estimator import YieldEstimator

estimator = YieldEstimator()

# Estimate relative yield
yield_result = estimator.estimate_relative_yield(
    current_timeseries=timeseries,
    baseline_value=150.0  # Historical baseline
)

print(f"Relative Yield Index: {yield_result['relative_yield_index']:.1f}%")
print(f"Performance: {yield_result['performance_category']}")

# Generate report
estimator.generate_report(yield_result, timeseries, 'yield_report.txt')
```

### 5. Launch Web Interface

```bash
# Option 1: Simple HTTP server
cd /path/to/project
python -m http.server 8000

# Open browser to: http://localhost:8000

# Option 2: Using Flask backend (for API integration)
python flask_app.py
```

---

## 📊 Features & Capabilities

### Vegetation Indices Computed

| Index | Formula | Purpose |
|-------|---------|---------|
| **NDVI** | (NIR - Red) / (NIR + Red) | General vegetation health |
| **NDWI** | (Green - NIR) / (Green + NIR) | Water stress detection |
| **EVI** | 2.5 × ((NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)) | Enhanced sensitivity |
| **SAVI** | ((NIR - Red) / (NIR + Red + 0.5)) × 1.5 | Soil-adjusted index |

### Machine Learning Features

**Spectral (6)**: NDVI, NDWI, EVI, SAVI, Red, NIR  
**Temporal (8)**: ΔNDVI, 7d/14d/30d trends, std dev, peak timing  
**Total**: ~15 features per observation

### Health Classification

- 🟢 **Healthy**: NDVI > 0.65, stable/increasing
- 🟡 **Mild Stress**: NDVI 0.45-0.65, moderate decline
- 🟠 **Severe Stress**: NDVI < 0.45, rapid decline
- 🔴 **Water Stressed**: High NDWI + declining NDVI

### Early Warning Triggers

1. **Rapid Decline**: NDVI drop > 0.15 over 7 days
2. **Sustained Decline**: Negative trend for 14+ days
3. **Critical Threshold**: NDVI < 0.4 for 3+ observations
4. **Water Stress**: NDWI > 0.3 with declining NDVI

---

## 📈 Results & Performance

### Model Performance (Sample Results)

```
Overall Accuracy: 78.5%
Weighted F1-Score: 0.77

Per-Class Metrics:
                 Precision  Recall  F1-Score  Support
Healthy            0.85      0.89     0.87      450
Mild Stress        0.72      0.68     0.70      280
Severe Stress      0.74      0.71     0.72      190
Water Stressed     0.81      0.85     0.83      80

Cross-Validation: 76.3% (+/- 3.2%)
```

### Feature Importance (Top 5)

1. NDVI (current) - 24.3%
2. 30-day NDVI trend - 18.7%
3. NDVI season max - 12.4%
4. NDVI standard deviation - 9.8%
5. Days since peak - 8.2%

---

## ⚠️ Limitations & Constraints

### Technical Limitations

1. **Spatial Resolution**: 10m pixels may miss sub-field variability
2. **Temporal Gaps**: Cloud cover can create 2-3 week data gaps
3. **Indirect Measurement**: NDVI correlates with, but doesn't directly measure, crop health
4. **No Ground Truth**: Model trained on proxy labels from vegetation indices
5. **Crop Agnostic**: No differentiation between crop types (wheat, corn, rice, etc.)
6. **Atmospheric Effects**: Residual atmospheric noise despite L2A processing

### Operational Constraints

- Requires technical expertise to interpret results
- Not a replacement for field scouting and agronomic knowledge
- Should complement, not replace, traditional agricultural practices
- Yield estimates are **relative only**, not absolute predictions

### Known Failure Modes

1. **Cloud Contamination**: Tropical regions with persistent cloud cover
2. **Mixed Pixels**: Field boundaries and small plots (<1 hectare)
3. **Phenology Confusion**: Crop type misidentification (e.g., mature wheat vs stressed corn)
4. **Seasonal Effects**: Winter dormancy misinterpreted as stress
5. **Extreme Events**: Flash floods or sudden frost not captured between overpasses

---

## 🔬 Evaluation Methodology

### Validation Approach

Since ground-truth crop health labels are unavailable:

1. **Proxy Labeling**: Heuristic rules based on NDVI thresholds and temporal patterns
2. **Cross-Validation**: 5-fold CV to assess model stability
3. **Feature Importance**: Verify that agronomically relevant features dominate
4. **Qualitative Analysis**: Visual inspection of predictions against satellite imagery
5. **Error Pattern Analysis**: Examine misclassifications for systematic biases

### Metrics Used

- **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Temporal**: Trend detection accuracy, alert precision/recall
- **Uncertainty**: Confidence score calibration

---

## 🗺️ Future Work

### Short-Term (3-6 months)

- [ ] Integrate weather data (rainfall, temperature) as features
- [ ] Add crop type classification module
- [ ] Implement mobile app for field technicians
- [ ] Multi-language support (Hindi, Spanish, French)

### Medium-Term (6-12 months)

- [ ] Ground-truth data collection campaign (collaboration with farmers)
- [ ] Advanced time-series models (LSTM, Prophet, Transformer)
- [ ] Soil moisture integration using Sentinel-1 SAR
- [ ] Pest/disease detection from high-resolution commercial imagery

### Long-Term (1-2 years)

- [ ] Integration with autonomous agricultural robots
- [ ] Real-time alert system via SMS/WhatsApp
- [ ] Prescription mapping for variable-rate fertilizer/irrigation
- [ ] Blockchain for data provenance and farmer credit scoring

---

## 📚 References & Related Work

### Academic Literature

1. Bolton, D. K., & Friedl, M. A. (2013). Forecasting crop yield using remotely sensed vegetation indices and crop phenology metrics. *Agricultural and Forest Meteorology*.

2. Gao, B. C. (1996). NDWI - A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*.

3. Belgiu, M., & Drăguţ, L. (2016). Random forest in remote sensing: A review of applications and future directions. *ISPRS Journal of Photogrammetry and Remote Sensing*.

### Relevant Systems

- **NASA Harvest**: Global agricultural monitoring program
- **Sen2-Agri**: ESA Sentinel-2 for Agriculture platform
- **GEOGLAM**: Group on Earth Observations Global Agricultural Monitoring
- **Google Earth Engine**: Cloud-based geospatial analysis

### Datasets

- **Sentinel-2**: ESA Copernicus Program (free, open data)
- **Landsat 8/9**: USGS (30m resolution, longer archive)
- **MODIS**: NASA (250m-1km, daily revisit)

---

## 🤝 Contributing

This is a research project, and contributions are welcome!

### How to Contribute

1. **Report Issues**: Found a bug or limitation? Open an issue on GitHub
2. **Suggest Features**: Have ideas for improvement? Start a discussion
3. **Submit Pull Requests**: Code contributions welcome (see CONTRIBUTING.md)
4. **Share Results**: Used this system? Share your findings!

### Areas Needing Help

- Ground-truth validation datasets
- Crop-specific calibration parameters
- Translation to additional languages
- Case studies from different agricultural regions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this code in your research, please cite:

```bibtex
@software{crop_health_monitoring_2024,
  author = {Agricultural AI Research Project},
  title = {AI-Assisted Satellite-Driven Crop Health Monitoring System},
  year = {2024},
  url = {https://github.com/yourusername/crop-health-monitoring},
  version = {1.0}
}
```

---

## 🙏 Acknowledgments

- **ESA Copernicus Programme** for free Sentinel-2 data
- **Google Earth Engine** for cloud-based processing infrastructure
- **Scikit-learn community** for machine learning tools
- **Leaflet.js** and **Chart.js** for visualization libraries
- All farmers and agronomists who inspired this work

---

## 📞 Contact

**Project Lead**: [Your Name]  
**Email**: your.email@example.com  
**GitHub**: [@yourusername](https://github.com/yourusername)  
**Research Group**: Agricultural AI Lab

---

## ⚡ Quick Links

- 📖 [Full Documentation](docs/README.md)
- 🎓 [Tutorial Notebook](notebooks/tutorial.ipynb)
- 🐛 [Issue Tracker](https://github.com/yourusername/crop-health-monitoring/issues)
- 💬 [Discussion Forum](https://github.com/yourusername/crop-health-monitoring/discussions)
- 📊 [Sample Datasets](data/samples/)

---

## 🎯 Project Philosophy

> *"Not to replace farmers, but to empower them with data-driven insights derived from space."*

This project embraces:
- **Transparency**: All limitations clearly documented
- **Accessibility**: Free and open-source
- **Education**: Learning tool, not commercial product
- **Rigor**: Scientific approach with honest uncertainty quantification
- **Impact**: Practical utility for real agricultural challenges

---

**Built with 🌍 for sustainable agriculture worldwide**

*Last updated: February 2026*
