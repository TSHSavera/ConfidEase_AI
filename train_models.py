"""
Training Script for Speech Analyzer AI Models

This script demonstrates how to start training your models using the collected data.
Run this after you've collected sufficient user feedback and training data.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# For emotion model fine-tuning (when ready)
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch datasets")

from modules.training.data_collection import TrainingDataCollector


class ModelTrainer:
    """Handles training of Speech Analyzer AI models."""
    
    def __init__(self, data_collector: TrainingDataCollector = None):
        """Initialize trainer with data collector."""
        self.data_collector = data_collector or TrainingDataCollector()
        self.training_dir = Path("model_training")
        self.training_dir.mkdir(exist_ok=True)
    
    def check_data_readiness(self) -> Dict[str, bool]:
        """Check if we have enough data for training."""
        
        stats = self.data_collector.get_training_data_stats()
        
        readiness = {
            'emotion_detection': stats['total_emotion_annotations'] >= 500,
            'user_feedback': stats['total_feedback_entries'] >= 100,
            'synthetic_data': stats['total_synthetic_entries'] >= 1000,
            'overall_ready': False
        }
        
        # Need at least some real user data
        readiness['overall_ready'] = (
            readiness['user_feedback'] and 
            (readiness['emotion_detection'] or readiness['synthetic_data'])
        )
        
        return readiness, stats
    
    def prepare_emotion_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare emotion detection training data."""
        
        conn = sqlite3.connect(self.data_collector.db_path)
        
        # Get real user corrections
        real_data = pd.read_sql_query('''
            SELECT text_snippet, user_corrected_emotion 
            FROM emotion_annotations 
            WHERE user_corrected_emotion IS NOT NULL
        ''', conn)
        
        # Get synthetic data
        synthetic_data = pd.read_sql_query('''
            SELECT text, emotion_label 
            FROM synthetic_data
        ''', conn)
        
        conn.close()
        
        # Combine datasets
        texts = []
        labels = []
        
        # Add real data (higher weight)
        for _, row in real_data.iterrows():
            texts.append(row['text_snippet'])
            labels.append(row['user_corrected_emotion'])
        
        # Add synthetic data
        for _, row in synthetic_data.iterrows():
            texts.append(row['text'])
            labels.append(row['emotion_label'])
        
        return texts, labels
    
    def train_emotion_model_basic(self):
        """Train a basic emotion classification model."""
        
        print("🎭 Training Emotion Detection Model...")
        
        # Get training data
        texts, labels = self.prepare_emotion_training_data()
        
        if len(texts) < 100:
            print(f"❌ Insufficient data: {len(texts)} samples. Need at least 100.")
            return None
        
        print(f"📊 Training with {len(texts)} samples")
        
        # Basic approach: Use sklearn with TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import joblib
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        # Train
        print("🔄 Training model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = self.training_dir / "emotion_model_basic.pkl"
        joblib.dump(pipeline, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        return pipeline
    
    def train_emotion_model_advanced(self):
        """Train advanced emotion model using transformers (if available)."""
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available. Using basic model instead.")
            return self.train_emotion_model_basic()
        
        print("🤖 Training Advanced Emotion Detection Model...")
        
        # Get training data
        texts, labels = self.prepare_emotion_training_data()
        
        if len(texts) < 500:
            print(f"❌ Insufficient data for advanced training: {len(texts)} samples. Need at least 500.")
            print("💡 Consider using basic model or collecting more data.")
            return None
        
        # Prepare for transformers
        label_list = list(set(labels))
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for label, i in label2id.items()}
        
        # Convert labels to ids
        label_ids = [label2id[label] for label in labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
        )
        
        # Initialize model and tokenizer
        model_name = "distilbert-base-uncased"  # Smaller, faster model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
        
        # Tokenize data
        def tokenize_function(texts):
            return tokenizer(texts, truncation=True, padding=True, max_length=512)
        
        train_encodings = tokenize_function(X_train)
        test_encodings = tokenize_function(X_test)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': y_train
        })
        
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'], 
            'labels': y_test
        })
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.training_dir / "emotion_model_advanced"),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.training_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("🔄 Training advanced model... (this may take a while)")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"✅ Advanced model trained! Eval loss: {eval_results['eval_loss']:.3f}")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(str(self.training_dir / "emotion_model_advanced"))
        
        print(f"💾 Advanced model saved to: {self.training_dir / 'emotion_model_advanced'}")
        
        return trainer
    
    def analyze_user_feedback(self):
        """Analyze collected user feedback for insights."""
        
        conn = sqlite3.connect(self.data_collector.db_path)
        
        feedback_df = pd.read_sql_query('''
            SELECT 
                ai_emotion_score, user_emotion_rating,
                ai_pacing_score, user_pacing_rating,
                ai_clarity_score, user_clarity_rating,
                ai_tone_score, user_tone_rating,
                feedback_helpful, audience_type, delivery_goal
            FROM user_feedback 
            WHERE user_emotion_rating IS NOT NULL
        ''', conn)
        
        conn.close()
        
        if len(feedback_df) == 0:
            print("❌ No user feedback available for analysis.")
            return
        
        print("📊 USER FEEDBACK ANALYSIS")
        print("=" * 50)
        
        # Calculate agreement between AI and users
        metrics = ['emotion', 'pacing', 'clarity', 'tone']
        
        for metric in metrics:
            ai_col = f'ai_{metric}_score'
            user_col = f'user_{metric}_rating'
            
            if ai_col in feedback_df.columns and user_col in feedback_df.columns:
                correlation = feedback_df[ai_col].corr(feedback_df[user_col])
                mean_diff = (feedback_df[ai_col] - feedback_df[user_col]).mean()
                
                print(f"\n{metric.title()} Analysis:")
                print(f"  Correlation: {correlation:.3f}")
                print(f"  Mean difference (AI - User): {mean_diff:.2f}")
        
        # Overall helpfulness
        if 'feedback_helpful' in feedback_df.columns:
            avg_helpful = feedback_df['feedback_helpful'].mean()
            print(f"\nOverall helpfulness: {avg_helpful:.2f}/10")
        
        # Save analysis
        analysis_path = self.training_dir / "feedback_analysis.json"
        
        analysis_results = {
            'total_feedback': len(feedback_df),
            'correlations': {},
            'mean_differences': {},
            'average_helpfulness': feedback_df['feedback_helpful'].mean() if 'feedback_helpful' in feedback_df else None
        }
        
        for metric in metrics:
            ai_col = f'ai_{metric}_score'
            user_col = f'user_{metric}_rating'
            
            if ai_col in feedback_df.columns and user_col in feedback_df.columns:
                analysis_results['correlations'][metric] = feedback_df[ai_col].corr(feedback_df[user_col])
                analysis_results['mean_differences'][metric] = (feedback_df[ai_col] - feedback_df[user_col]).mean()
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\n💾 Analysis saved to: {analysis_path}")


def main():
    """Main training script."""
    
    print("""
    🎓 SPEECH ANALYZER MODEL TRAINING
    ================================
    """)
    
    trainer = ModelTrainer()
    
    # Check data readiness
    print("🔍 Checking training data readiness...")
    readiness, stats = trainer.check_data_readiness()
    
    print(f"\n📊 Data Statistics:")
    print(f"  User feedback entries: {stats['total_feedback_entries']}")
    print(f"  Emotion annotations: {stats['total_emotion_annotations']}")
    print(f"  Synthetic data points: {stats['total_synthetic_entries']}")
    print(f"  Average helpfulness: {stats['avg_helpfulness_rating']:.2f}/10")
    
    print(f"\n✅ Training Readiness:")
    for component, ready in readiness.items():
        if component != 'overall_ready':
            status = "✅ Ready" if ready else "❌ Need more data"
            print(f"  {component}: {status}")
    
    if not readiness['overall_ready']:
        print("\n⚠️ Not enough data for training yet.")
        print("💡 Continue using the app and collecting user feedback!")
        return
    
    print("\n🎯 Training Options:")
    print("1. Analyze user feedback")
    print("2. Train basic emotion model")
    print("3. Train advanced emotion model (requires transformers)")
    print("4. All of the above")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice in ['1', '4']:
        trainer.analyze_user_feedback()
    
    if choice in ['2', '4']:
        trainer.train_emotion_model_basic()
    
    if choice in ['3', '4']:
        trainer.train_emotion_model_advanced()
    
    print("\n🎉 Training complete!")
    print("💡 Remember to backup your trained models and continue collecting feedback.")


if __name__ == "__main__":
    main()
