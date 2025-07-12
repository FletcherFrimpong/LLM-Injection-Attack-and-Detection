#!/usr/bin/env python3
"""
Test script for Model Training functionality
"""

import sys
from pathlib import Path

# Add the secprompt package to the path
sys.path.append(str(Path(__file__).parent))

from secprompt.model_trainer import ModelTrainer, TrainingConfig
from secprompt.data_collector import DataCollector


def main():
    print("🤖 SecPrompt Model Training Demo")
    print("=" * 50)
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        data_collector = DataCollector()
        trainer = ModelTrainer()
        
        # Test 1: Data Collection
        print("\n1. Testing Data Collection")
        print("-" * 30)
        
        print("📊 Collecting synthetic data...")
        samples = data_collector.generate_synthetic_data(num_samples=200)
        
        # Show statistics
        stats = data_collector.get_dataset_statistics(samples)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Injection samples: {stats['injection_samples']}")
        print(f"Safe samples: {stats['safe_samples']}")
        print(f"Injection ratio: {stats['injection_ratio']:.2%}")
        
        # Test 2: Model Training
        print("\n2. Testing Model Training")
        print("-" * 30)
        
        # Create training configuration
        config = TrainingConfig(
            model_type="random_forest",
            test_size=0.2,
            max_features=2000,
            n_estimators=50,
            use_tfidf=True,
            use_ngrams=True
        )
        
        # Create trainer with config
        trainer = ModelTrainer(config)
        
        # Preprocess data
        print("🔧 Preprocessing data...")
        X, y = trainer.preprocess_data(samples)
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("🚀 Training model...")
        result = trainer.train_model(X_train, y_train, X_test, y_test)
        
        # Display results
        print(f"\n📊 Training Results:")
        print(f"Model Type: {result.config.model_type}")
        print(f"Training Time: {result.training_time:.2f} seconds")
        print(f"Accuracy: {result.metrics['accuracy']:.4f}")
        print(f"Precision: {result.metrics['precision']:.4f}")
        print(f"Recall: {result.metrics['recall']:.4f}")
        print(f"F1 Score: {result.metrics['f1_score']:.4f}")
        print(f"ROC AUC: {result.metrics['roc_auc']:.4f}")
        
        # Feature importance
        if result.feature_importance:
            print(f"\n🔍 Top 5 Features:")
            for i, (feature, importance) in enumerate(list(result.feature_importance.items())[:5]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Test 3: Cross-validation
        print("\n3. Testing Cross-validation")
        print("-" * 30)
        
        cv_results = trainer.cross_validate_model(X, y)
        print(f"Cross-validation F1 score: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        
        # Test 4: Save and Load Model
        print("\n4. Testing Model Save/Load")
        print("-" * 30)
        
        model_path = "models/test_detector.pkl"
        trainer.save_model(result, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Load model
        loaded_result = trainer.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        print(f"Loaded model metrics: {loaded_result.metrics}")
        
        # Test 5: Generate Report
        print("\n5. Testing Report Generation")
        print("-" * 30)
        
        report = trainer.generate_training_report(result)
        print("📄 Training Report Generated")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        # Test 6: Online Data Collection (limited)
        print("\n6. Testing Online Data Collection")
        print("-" * 30)
        
        print("📚 Collecting from GitHub (limited)...")
        try:
            github_samples = data_collector.collect_from_github(max_repos=1)
            print(f"✅ Collected {len(github_samples)} samples from GitHub")
        except Exception as e:
            print(f"⚠️ GitHub collection failed: {e}")
        
        print("🔧 Generating additional synthetic data...")
        synthetic_samples = data_collector.generate_synthetic_data(num_samples=300)
        print(f"✅ Generated {len(synthetic_samples)} synthetic samples")
        
        # Combine samples
        all_samples = samples + synthetic_samples
        combined_stats = data_collector.get_dataset_statistics(all_samples)
        print(f"📊 Combined dataset: {combined_stats['total_samples']} samples")
        print(f"   Injection ratio: {combined_stats['injection_ratio']:.2%}")
        
        # Save combined dataset
        data_collector.save_dataset(all_samples, "data/combined_training_data.json")
        print("💾 Combined dataset saved to data/combined_training_data.json")
        
        print("\n✅ All tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install scikit-learn matplotlib seaborn")
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    main() 