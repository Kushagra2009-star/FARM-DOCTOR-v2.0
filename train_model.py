"""
Train crop health classification model
"""

from ml_classifier import CropHealthClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

print("="*60)
print("CROP HEALTH MODEL TRAINING")
print("="*60)

print("\n1. Loading data...")
df = pd.read_csv('processed_data.csv')
print(f"   Total observations: {len(df)}")

print("\n2. Checking columns...")
print(f"   Columns in CSV: {df.columns.tolist()}")

# CORRECT feature columns - matching your CSV exactly
feature_columns = [
    'NDVI',           # Normalized Difference Vegetation Index
    'NDWI',           # Normalized Difference Water Index
    'EVI',            # Enhanced Vegetation Index
    'SAVI',           # Soil-Adjusted Vegetation Index
    'B4',             # Red band reflectance
    'B8',             # NIR band reflectance
    'NDVI_delta',     # Change from previous observation
    'NDVI_7d_mean',   # 7-day rolling average
    'NDVI_30d_mean',  # 30-day rolling average
    'NDVI_30d_std',   # 30-day standard deviation
    'NDVI_trend_7d',  # 7-day trend slope
    'NDVI_trend_30d', # 30-day trend slope
    'NDVI_season_max',# Maximum NDVI this season
    'days_since_peak',# Days since peak NDVI
    'NDVI_rate'       # Rate of NDVI change
]

# Verify all columns exist
missing = [col for col in feature_columns if col not in df.columns]
if missing:
    print(f"\n❌ ERROR: Missing columns: {missing}")
    print("\nAvailable columns:")
    print(df.columns.tolist())
    exit(1)
else:
    print(f"   ✅ All {len(feature_columns)} features found!")

# Initialize classifier
print("\n3. Initializing classifier...")
classifier = CropHealthClassifier(n_estimators=200, max_depth=15)

# Prepare data
print("\n4. Preparing training data...")
X, y = classifier.prepare_data(df, feature_columns, 'health_label')

print(f"   Samples after cleaning: {len(X)}")

# Check if enough data
if len(X) < 30:
    print(f"\n⚠️  WARNING: Only {len(X)} samples!")
    print("   Recommendation: Collect 100+ observations for reliable model")
    print("   Continuing anyway for demonstration...")
    
    # Adjust split for small dataset
    test_size = 0.3
    use_stratify = False
else:
    test_size = 0.3
    use_stratify = True

# Split data
print(f"\n5. Splitting data (test_size={test_size})...")

if use_stratify:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

print(f"   Training:   {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Testing:    {len(X_test)} samples")

# Train
print("\n6. Training model...")
print("   (This may take 1-2 minutes)")
metrics = classifier.train(X_train, y_train, X_val, y_val)

# Evaluate
print("\n7. Evaluating model...")
test_metrics = classifier.evaluate(X_test, y_test, output_dir='evaluation_results')

# Save
print("\n8. Saving model...")
classifier.save_model('crop_health_model.pkl')

# Summary
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
print(f"Model file: crop_health_model.pkl")
print(f"Evaluation charts: evaluation_results/")
print("="*60)

if len(X) < 100:
    print("\n⚠️  NOTE: Model trained on limited data (<100 samples)")
    print("For production use, collect 100+ observations")
print()