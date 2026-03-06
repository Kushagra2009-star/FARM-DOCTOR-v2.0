"""
Crop Health Classification Model
=================================

Random Forest classifier for crop health status prediction with
uncertainty quantification and feature importance analysis.

Dependencies:
    - scikit-learn
    - numpy
    - pandas
    - matplotlib
    - seaborn

Author: Agricultural AI Research Project
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from pathlib import Path


class CropHealthClassifier:
    """
    Random Forest model for classifying crop health from spectral features.
    """
    
    def __init__(self, n_estimators=200, max_depth=15, random_state=42):
        """
        Initialize classifier with hyperparameters.
        
        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',  # Handle class imbalance
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.training_history = {}
        
    def prepare_data(self, df, feature_columns, target_column='health_label'):
        """
        Prepare data for training by handling missing values and encoding.
        
        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names
            target_column: Name of target label column
            
        Returns:
            X: Feature matrix (numpy array)
            y: Encoded labels (numpy array)
        """
        # Remove rows with missing values
        df_clean = df[feature_columns + [target_column]].dropna()
        
        print(f"Data preparation: {len(df_clean)}/{len(df)} rows retained after removing NaN")
        
        # Extract features
        X = df_clean[feature_columns].values
        
        # Encode labels
        y = self.label_encoder.fit_transform(df_clean[target_column])
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            dict: Training metrics
        """
        print("Training Random Forest classifier...")
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_acc,
            'n_samples_train': len(X_train)
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = val_acc
            metrics['n_samples_val'] = len(X_val)
            
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")
        else:
            print(f"Training accuracy: {train_acc:.4f}")
        
        self.training_history = metrics
        return metrics
    
    def predict(self, X):
        """
        Predict health classes.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy.array: Predicted class labels (decoded)
        """
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for uncertainty quantification.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy.array: Probability matrix (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X):
        """
        Predict with confidence scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            tuple: (predictions, confidence_scores)
                - predictions: Predicted class labels
                - confidence_scores: Maximum probability for each prediction
        """
        proba = self.predict_proba(X)
        predictions = self.predict(X)
        confidence = np.max(proba, axis=1)
        
        return predictions, confidence
    
    def evaluate(self, X_test, y_test, output_dir='evaluation_results'):
        """
        Comprehensive model evaluation with visualizations.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save evaluation artifacts
            
        Returns:
            dict: Evaluation metrics
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print("\nPer-Class Metrics:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        self._plot_confusion_matrix(cm, self.label_encoder.classes_, 
                                    output_dir)
        
        # Feature Importance
        self._plot_feature_importance(output_dir)
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nEvaluation results saved to '{output_dir}/'")
        
        return metrics
    
    def _plot_confusion_matrix(self, cm, class_names, output_dir):
        """
        Plot and save confusion matrix heatmap.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix - Crop Health Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized version
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, output_dir):
        """
        Plot and save feature importance rankings.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [self.feature_columns[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as CSV
        importance_df = pd.DataFrame({
            'feature': [self.feature_columns[i] for i in indices],
            'importance': importances[indices]
        })
        importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
        
        print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """
        Perform grid search for hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to search
            
        Returns:
            dict: Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [30, 50, 100],
                'min_samples_leaf': [10, 20, 30]
            }
        
        print("Starting hyperparameter tuning (this may take a while)...")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def analyze_errors(self, X_test, y_test, output_dir='error_analysis'):
        """
        Detailed error analysis for model improvement insights.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save analysis
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get predictions with confidence
        y_pred_encoded = self.model.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_true = self.label_encoder.inverse_transform(y_test)
        proba = self.predict_proba(X_test)
        confidence = np.max(proba, axis=1)
        
        # Create error analysis DataFrame
        errors = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'confidence': confidence,
            'correct': y_true == y_pred
        })
        
        # Add feature values
        for i, col in enumerate(self.feature_columns):
            errors[col] = X_test[:, i]
        
        # Analyze misclassifications
        misclassified = errors[errors['correct'] == False]
        
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        print(f"Total misclassifications: {len(misclassified)}/{len(errors)} ({len(misclassified)/len(errors)*100:.2f}%)")
        
        print("\nMisclassification patterns:")
        print(misclassified.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False))
        
        print("\nAverage confidence by correctness:")
        print(errors.groupby('correct')['confidence'].mean())
        
        # Save detailed error report
        misclassified.to_csv(f'{output_dir}/misclassified_samples.csv', index=False)
        
        # Plot confidence distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(errors[errors['correct']]['confidence'], bins=20, alpha=0.7, label='Correct')
        axes[0].hist(errors[~errors['correct']]['confidence'], bins=20, alpha=0.7, label='Incorrect')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution by Correctness')
        axes[0].legend()
        
        # Error rate by confidence bin
        bins = np.linspace(0, 1, 11)
        errors['confidence_bin'] = pd.cut(errors['confidence'], bins)
        error_rate = errors.groupby('confidence_bin')['correct'].apply(lambda x: (~x).mean())
        
        axes[1].plot(range(len(error_rate)), error_rate.values, marker='o')
        axes[1].set_xlabel('Confidence Bin')
        axes[1].set_ylabel('Error Rate')
        axes[1].set_title('Error Rate vs Confidence')
        axes[1].set_xticklabels([f'{b:.1f}' for b in bins[1:]], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nError analysis saved to '{output_dir}/'")
    
    def save_model(self, filepath='crop_health_model.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to '{filepath}'")
    
    @classmethod
    def load_model(cls, filepath='crop_health_model.pkl'):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            CropHealthClassifier: Loaded classifier
        """
        model_data = joblib.load(filepath)
        
        classifier = cls()
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_columns = model_data['feature_columns']
        classifier.training_history = model_data['training_history']
        
        print(f"Model loaded from '{filepath}'")
        return classifier


# Example usage and training script
if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv('processed_sentinel2_timeseries.csv')
    
    # Define features
    feature_columns = [
        'ndvi', 'ndwi', 'evi', 'savi', 'red', 'nir',
        'ndvi_delta', 'ndvi_7d_mean', 'ndvi_30d_mean', 'ndvi_30d_std',
        'ndvi_trend_7d', 'ndvi_trend_30d', 'ndvi_season_max',
        'days_since_peak', 'ndvi_rate'
    ]
    
    # Initialize classifier
    classifier = CropHealthClassifier(n_estimators=200, max_depth=15)
    
    # Prepare data
    X, y = classifier.prepare_data(df, feature_columns, 'health_label')
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Train model
    metrics = classifier.train(X_train, y_train, X_val, y_val)
    
    # Cross-validation
    cv_results = classifier.cross_validate(X_train, y_train, cv=5)
    
    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test)
    
    # Error analysis
    classifier.analyze_errors(X_test, y_test)
    
    # Save model
    classifier.save_model('crop_health_model.pkl')
    
    print("\n" + "="*60)
    print("Training complete! Model ready for deployment.")
    print("="*60)
