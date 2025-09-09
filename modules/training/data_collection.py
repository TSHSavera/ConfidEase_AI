"""
Data Collection Module for Speech Analyzer Training

This module implements user feedback collection and synthetic data generation
to build training datasets for improving the AI models.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3

from modules.parameter_input.parameter_validator import AudienceType, DeliveryGoal


class TrainingDataCollector:
    """Collects user feedback and training data for model improvement."""
    
    def __init__(self, db_path: str = "training_data.db"):
        """Initialize data collector with SQLite database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for training data storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                speech_text TEXT,
                audience_type TEXT,
                delivery_goal TEXT,
                ai_emotion_score REAL,
                ai_pacing_score REAL,
                ai_clarity_score REAL,
                ai_tone_score REAL,
                user_emotion_rating INTEGER,
                user_pacing_rating INTEGER,
                user_clarity_rating INTEGER,
                user_tone_rating INTEGER,
                feedback_helpful INTEGER,
                user_comments TEXT,
                audio_file_path TEXT
            )
        ''')
        
        # Emotion annotations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_annotations (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                text_snippet TEXT,
                sentence_index INTEGER,
                ai_predicted_emotion TEXT,
                ai_confidence REAL,
                user_corrected_emotion TEXT,
                audience_type TEXT,
                delivery_goal TEXT
            )
        ''')
        
        # Synthetic training data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synthetic_data (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                text TEXT,
                emotion_label TEXT,
                audience_type TEXT,
                delivery_goal TEXT,
                generation_prompt TEXT,
                quality_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_user_feedback(self, analysis_result, user_ratings: Dict, 
                            user_comments: str = "", helpful_rating: int = None):
        """Collect user feedback on AI analysis results."""
        
        feedback_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (
                id, timestamp, speech_text, audience_type, delivery_goal,
                ai_emotion_score, ai_pacing_score, ai_clarity_score, ai_tone_score,
                user_emotion_rating, user_pacing_rating, user_clarity_rating, user_tone_rating,
                feedback_helpful, user_comments, audio_file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            datetime.now().isoformat(),
            getattr(analysis_result.parameters_used, 'speech_text', '')[:1000],  # Limit text length
            analysis_result.parameters_used.audience_type.value,
            analysis_result.parameters_used.delivery_goal.value,
            analysis_result.emotion_section.rating,
            analysis_result.pacing_section.rating,
            analysis_result.clarity_section.rating,
            analysis_result.tone_section.rating,
            user_ratings.get('emotion'),
            user_ratings.get('pacing'),
            user_ratings.get('clarity'),
            user_ratings.get('tone'),
            helpful_rating,
            user_comments,
            getattr(analysis_result.parameters_used, 'audio_file_path', None)
        ))
        
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def collect_emotion_corrections(self, text_snippets: List[str], 
                                  ai_predictions: List[Dict], 
                                  user_corrections: List[str],
                                  audience_type: AudienceType,
                                  delivery_goal: DeliveryGoal):
        """Collect user corrections for emotion detection."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, (snippet, prediction, correction) in enumerate(
            zip(text_snippets, ai_predictions, user_corrections)):
            
            annotation_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO emotion_annotations (
                    id, timestamp, text_snippet, sentence_index,
                    ai_predicted_emotion, ai_confidence, user_corrected_emotion,
                    audience_type, delivery_goal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                annotation_id,
                datetime.now().isoformat(),
                snippet[:500],  # Limit snippet length
                i,
                prediction.get('label', ''),
                prediction.get('score', 0.0),
                correction,
                audience_type.value,
                delivery_goal.value
            ))
        
        conn.commit()
        conn.close()
    
    def get_training_data_stats(self) -> Dict:
        """Get statistics about collected training data."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # User feedback stats
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        stats['total_feedback_entries'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(feedback_helpful) FROM user_feedback WHERE feedback_helpful IS NOT NULL')
        result = cursor.fetchone()[0]
        stats['avg_helpfulness_rating'] = result if result else 0
        
        # Emotion annotations stats
        cursor.execute('SELECT COUNT(*) FROM emotion_annotations')
        stats['total_emotion_annotations'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT user_corrected_emotion, COUNT(*) FROM emotion_annotations GROUP BY user_corrected_emotion')
        emotion_distribution = dict(cursor.fetchall())
        stats['emotion_distribution'] = emotion_distribution
        
        # Synthetic data stats
        cursor.execute('SELECT COUNT(*) FROM synthetic_data')
        stats['total_synthetic_entries'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def export_training_data(self, output_dir: str = "training_exports"):
        """Export collected data for model training."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Export emotion training data
        emotion_data = []
        cursor = conn.cursor()
        cursor.execute('''
            SELECT text_snippet, user_corrected_emotion, audience_type, delivery_goal 
            FROM emotion_annotations 
            WHERE user_corrected_emotion IS NOT NULL
        ''')
        
        for row in cursor.fetchall():
            emotion_data.append({
                'text': row[0],
                'emotion': row[1],
                'audience': row[2],
                'goal': row[3]
            })
        
        with open(output_path / "emotion_training_data.json", 'w') as f:
            json.dump(emotion_data, f, indent=2)
        
        # Export user feedback for analysis
        feedback_data = []
        cursor.execute('SELECT * FROM user_feedback')
        columns = [desc[0] for desc in cursor.description]
        
        for row in cursor.fetchall():
            feedback_data.append(dict(zip(columns, row)))
        
        with open(output_path / "user_feedback_data.json", 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        conn.close()
        
        return output_path


class SyntheticDataGenerator:
    """Generate synthetic training data using LLMs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key (optional for now)."""
        self.api_key = api_key
        self.data_collector = TrainingDataCollector()
    
    def generate_emotion_examples(self, count: int = 100) -> List[Dict]:
        """Generate emotion-labeled text examples."""
        
        # For now, use predefined templates
        # In production, would use OpenAI/Claude API
        
        templates = {
            'confident': [
                "I am absolutely certain that our approach will succeed.",
                "This strategy has proven effective in similar situations.",
                "Our team has the expertise to deliver outstanding results."
            ],
            'anxious': [
                "I'm not entirely sure if this will work as planned.",
                "There might be some challenges we haven't considered.",
                "I hope we can address all the potential issues."
            ],
            'enthusiastic': [
                "This is an incredible opportunity for innovation!",
                "I'm thrilled to share these exciting developments with you.",
                "The possibilities ahead are absolutely amazing!"
            ],
            'serious': [
                "We must carefully consider all implications.",
                "This matter requires our immediate attention.",
                "The consequences of this decision are significant."
            ]
        }
        
        synthetic_data = []
        
        for emotion, examples in templates.items():
            for example in examples:
                for audience in AudienceType:
                    for goal in DeliveryGoal:
                        synthetic_data.append({
                            'text': example,
                            'emotion': emotion,
                            'audience': audience.value,
                            'goal': goal.value,
                            'generation_method': 'template'
                        })
        
        return synthetic_data[:count]
    
    def save_synthetic_data(self, synthetic_examples: List[Dict]):
        """Save synthetic data to database."""
        
        conn = sqlite3.connect(self.data_collector.db_path)
        cursor = conn.cursor()
        
        for example in synthetic_examples:
            data_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO synthetic_data (
                    id, timestamp, text, emotion_label, audience_type, 
                    delivery_goal, generation_prompt, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_id,
                datetime.now().isoformat(),
                example['text'],
                example['emotion'],
                example['audience'],
                example['goal'],
                example.get('generation_method', ''),
                example.get('quality_score', 1.0)
            ))
        
        conn.commit()
        conn.close()


# Integration with main application
class FeedbackInterface:
    """User interface for collecting feedback."""
    
    def __init__(self, data_collector: TrainingDataCollector):
        self.data_collector = data_collector
    
    def ask_for_feedback(self, analysis_result) -> bool:
        """Ask user if they want to provide feedback."""
        
        print("\n" + "="*60)
        print("🔄 HELP IMPROVE THE AI")
        print("="*60)
        print("Your feedback helps make the AI more accurate!")
        print("This is completely optional and anonymous.")
        
        provide_feedback = input("\nWould you like to provide feedback? (y/n): ").strip().lower()
        
        if provide_feedback in ['y', 'yes']:
            self.collect_feedback_interactive(analysis_result)
            return True
        
        return False
    
    def collect_feedback_interactive(self, analysis_result):
        """Interactive feedback collection."""
        
        print("\n📊 Please rate the AI's analysis (1-10 scale):")
        
        user_ratings = {}
        
        # Collect ratings
        try:
            print(f"\n🎭 Emotion Analysis (AI scored: {analysis_result.emotion_section.rating:.1f}/10)")
            user_ratings['emotion'] = int(input("Your rating (1-10): "))
            
            print(f"\n⏱️ Pacing Analysis (AI scored: {analysis_result.pacing_section.rating:.1f}/10)")
            user_ratings['pacing'] = int(input("Your rating (1-10): "))
            
            print(f"\n🔍 Clarity Analysis (AI scored: {analysis_result.clarity_section.rating:.1f}/10)")
            user_ratings['clarity'] = int(input("Your rating (1-10): "))
            
            print(f"\n🎯 Tone Delivery (AI scored: {analysis_result.tone_section.rating:.1f}/10)")
            user_ratings['tone'] = int(input("Your rating (1-10): "))
            
            print("\n💭 Overall feedback helpfulness:")
            helpful_rating = int(input("How helpful was this analysis? (1-10): "))
            
            comments = input("\nAny additional comments (optional): ").strip()
            
            # Save feedback
            feedback_id = self.data_collector.collect_user_feedback(
                analysis_result, user_ratings, comments, helpful_rating
            )
            
            print(f"\n✅ Thank you! Your feedback has been recorded (ID: {feedback_id[:8]})")
            
        except (ValueError, KeyboardInterrupt):
            print("\n⚠️ Feedback collection cancelled.")
    
    def show_data_stats(self):
        """Show training data collection statistics."""
        
        stats = self.data_collector.get_training_data_stats()
        
        print("\n📈 TRAINING DATA STATISTICS")
        print("="*40)
        print(f"Total feedback entries: {stats['total_feedback_entries']}")
        print(f"Average helpfulness: {stats['avg_helpfulness_rating']:.2f}/10")
        print(f"Emotion annotations: {stats['total_emotion_annotations']}")
        print(f"Synthetic data points: {stats['total_synthetic_entries']}")
        
        if stats['emotion_distribution']:
            print("\nEmotion distribution:")
            for emotion, count in stats['emotion_distribution'].items():
                print(f"  {emotion}: {count}")
