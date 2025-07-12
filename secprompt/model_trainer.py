"""
Model Trainer Module

This module provides comprehensive model training capabilities for prompt injection detection,
including data collection, preprocessing, training, and evaluation.
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn matplotlib seaborn")

from .data_collector import DataCollector, TrainingSample
from .detector import PromptDetector


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str = "balanced"
    cv_folds: int = 5
    use_tfidf: bool = True
    use_ngrams: bool = True
    max_ngram_range: Tuple[int, int] = (1, 3)


@dataclass
class TrainingResult:
    """Result of model training"""
    model: Any
    config: TrainingConfig
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_time: float
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str
    timestamp: datetime


class ModelTrainer:
    """Comprehensive model trainer for prompt injection detection"""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.data_collector = DataCollector()
        self.detector = PromptDetector()
        
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn is required for model training. Install with: pip install scikit-learn matplotlib seaborn")
    
    def collect_training_data(self, 
                            sources: List[str] = None,
                            max_samples_per_source: int = 500,
                            save_path: str = "data/training_data.json") -> List[TrainingSample]:
        """Collect training data from specified sources"""
        
        if sources is None:
            sources = ["synthetic", "github", "reddit", "huggingface", "arxiv"]
        
        all_samples = []
        
        print("🔄 Starting data collection...")
        
        for source in sources:
            try:
                if source == "synthetic":
                    print("🔧 Generating synthetic data...")
                    samples = self.data_collector.generate_synthetic_data(num_samples=1000)
                elif source == "github":
                    print("📚 Collecting from GitHub...")
                    samples = self.data_collector.collect_from_github(max_repos=3)
                elif source == "reddit":
                    print("💬 Collecting from Reddit...")
                    samples = self.data_collector.collect_from_reddit(max_posts=100)
                elif source == "huggingface":
                    print("🤗 Collecting from Hugging Face...")
                    samples = self.data_collector.collect_from_huggingface()
                elif source == "arxiv":
                    print("📄 Collecting from arXiv...")
                    samples = self.data_collector.collect_from_arxiv(max_papers=10)
                else:
                    print(f"⚠️ Unknown source: {source}")
                    continue
                
                # Limit samples per source
                samples = samples[:max_samples_per_source]
                all_samples.extend(samples)
                print(f"✅ Collected {len(samples)} samples from {source}")
                
            except Exception as e:
                print(f"❌ Error collecting from {source}: {e}")
                continue
        
        # Save collected data
        if save_path:
            self.data_collector.save_dataset(all_samples, save_path)
        
        # Print statistics
        stats = self.data_collector.get_dataset_statistics(all_samples)
        print(f"\n📊 Dataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Injection samples: {stats['injection_samples']}")
        print(f"Safe samples: {stats['safe_samples']}")
        print(f"Injection ratio: {stats['injection_ratio']:.2%}")
        
        return all_samples
    
    def load_training_data(self, file_path: str) -> List[TrainingSample]:
        """Load training data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = TrainingSample(
                text=item['text'],
                is_injection=item['is_injection'],
                category=item['category'],
                severity=item['severity'],
                source=item['source'],
                confidence=item['confidence'],
                metadata=item['metadata']
            )
            samples.append(sample)
        
        return samples
    
    def preprocess_data(self, samples: List[TrainingSample]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess training data for model training"""
        
        # Extract features and labels
        texts = [sample.text for sample in samples]
        labels = [1 if sample.is_injection else 0 for sample in samples]
        
        # Convert to numpy arrays
        X = np.array(texts)
        y = np.array(labels)
        
        return X, y
    
    def create_feature_pipeline(self) -> Pipeline:
        """Create feature extraction pipeline"""
        
        if self.config.use_tfidf:
            vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.max_ngram_range if self.config.use_ngrams else (1, 1),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        else:
            vectorizer = CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.max_ngram_range if self.config.use_ngrams else (1, 1),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        
        return vectorizer
    
    def create_model(self) -> Any:
        """Create model based on configuration"""
        
        if self.config.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "logistic_regression":
            model = LogisticRegression(
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                max_iter=1000
            )
        elif self.config.model_type == "naive_bayes":
            model = MultinomialNB()
        elif self.config.model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        return model
    
    def train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> TrainingResult:
        """Train the model and return results"""
        
        import time
        start_time = time.time()
        
        # Create pipeline
        vectorizer = self.create_feature_pipeline()
        model = self.create_model()
        
        # Fit vectorizer and transform data
        print("🔧 Extracting features...")
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)
        
        # Train model
        print(f"🚀 Training {self.config.model_type} model...")
        model.fit(X_train_features, y_train)
        
        # Make predictions
        train_time = time.time() - start_time
        print(f"⏱️ Training completed in {train_time:.2f} seconds")
        
        # Evaluate model
        print("📊 Evaluating model...")
        y_pred = model.predict(X_test_features)
        y_prob = model.predict_proba(X_test_features)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        # Get feature importance
        feature_importance = self._get_feature_importance(model, vectorizer)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=['Safe', 'Injection'])
        
        return TrainingResult(
            model=model,
            config=self.config,
            metrics=metrics,
            feature_importance=feature_importance,
            training_time=train_time,
            test_predictions=y_pred,
            test_probabilities=y_prob,
            confusion_matrix=cm,
            classification_report=report,
            timestamp=datetime.now()
        )
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }
        
        return metrics
    
    def _get_feature_importance(self, model: Any, vectorizer: Any) -> Dict[str, float]:
        """Get feature importance from model"""
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            feature_names = vectorizer.get_feature_names_out()
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            feature_names = vectorizer.get_feature_names_out()
            importances = np.abs(model.coef_[0])
        else:
            return {}
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        feature_importance = {}
        
        for i in indices[:20]:  # Top 20 features
            feature_importance[feature_names[i]] = float(importances[i])
        
        return feature_importance
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation"""
        
        vectorizer = self.create_feature_pipeline()
        model = self.create_model()
        
        # Create pipeline for cross-validation
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            pipeline, X, y, 
            cv=self.config.cv_folds, 
            scoring='f1',
            n_jobs=-1
        )
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV"""
        
        vectorizer = self.create_feature_pipeline()
        model = self.create_model()
        
        # Define parameter grid based on model type
        if self.config.model_type == "random_forest":
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        elif self.config.model_type == "logistic_regression":
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        else:
            return {"error": "Hyperparameter tuning not implemented for this model type"}
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, 
            cv=self.config.cv_folds, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def save_model(self, result: TrainingResult, file_path: str):
        """Save trained model and metadata"""
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            'model': result.model,
            'config': result.config,
            'metrics': result.metrics,
            'feature_importance': result.feature_importance,
            'training_time': result.training_time,
            'timestamp': result.timestamp
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {file_path}")
    
    def load_model(self, file_path: str) -> TrainingResult:
        """Load trained model from file"""
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return TrainingResult(
            model=model_data['model'],
            config=model_data['config'],
            metrics=model_data['metrics'],
            feature_importance=model_data['feature_importance'],
            training_time=model_data['training_time'],
            test_predictions=np.array([]),  # Not saved
            test_probabilities=np.array([]),  # Not saved
            confusion_matrix=np.array([]),  # Not saved
            classification_report="",  # Not saved
            timestamp=model_data['timestamp']
        )
    
    def generate_training_report(self, result: TrainingResult, save_path: str = None) -> str:
        """Generate comprehensive training report"""
        
        report = []
        report.append("=" * 60)
        report.append("MODEL TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Model Type: {result.config.model_type}")
        report.append(f"Training Time: {result.training_time:.2f} seconds")
        report.append("")
        
        # Metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 20)
        for metric, value in result.metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        report.append("")
        
        # Feature importance
        if result.feature_importance:
            report.append("TOP FEATURES:")
            report.append("-" * 20)
            for feature, importance in list(result.feature_importance.items())[:10]:
                report.append(f"{feature}: {importance:.4f}")
            report.append("")
        
        # Configuration
        report.append("TRAINING CONFIGURATION:")
        report.append("-" * 20)
        for key, value in result.config.__dict__.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Classification report
        if result.classification_report:
            report.append("DETAILED CLASSIFICATION REPORT:")
            report.append("-" * 20)
            report.append(result.classification_report)
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to {save_path}")
        
        return report_text
    
    def plot_training_results(self, result: TrainingResult, save_path: str = None):
        """Generate plots for training results"""
        
        if not ML_AVAILABLE:
            print("Warning: matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = result.confusion_matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Feature Importance
        if result.feature_importance:
            features = list(result.feature_importance.keys())[:10]
            importances = list(result.feature_importance.values())[:10]
            axes[0, 1].barh(range(len(features)), importances)
            axes[0, 1].set_yticks(range(len(features)))
            axes[0, 1].set_yticklabels(features)
            axes[0, 1].set_title('Top 10 Feature Importance')
            axes[0, 1].set_xlabel('Importance')
        
        # Metrics Bar Chart
        metrics = result.metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # ROC Curve (if available)
        if hasattr(result, 'test_probabilities') and len(result.test_probabilities) > 0:
            # This would need the actual test labels to plot ROC curve
            axes[1, 1].text(0.5, 0.5, 'ROC Curve\n(requires test labels)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ROC Curve')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {save_path}")
        
        plt.show() 