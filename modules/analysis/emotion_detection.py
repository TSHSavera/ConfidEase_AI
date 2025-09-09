"""
Emotion Detection Sub-Module

Uses NLP techniques and transformer models to detect emotions in speech text.
Provides sentence-level emotion analysis and overall emotional tone assessment.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import re
import warnings
from collections import Counter

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", message=".*return_all_scores.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model.*")

logger = logging.getLogger(__name__)


@dataclass
class EmotionScore:
    """Represents emotion scores for a text segment."""
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float
    neutral: float
    
    def get_dominant_emotion(self) -> str:
        """Get the emotion with the highest score."""
        emotions = {
            'joy': self.joy,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust,
            'neutral': self.neutral
        }
        return max(emotions, key=emotions.get)
    
    def get_confidence(self) -> float:
        """Get confidence score (highest emotion score)."""
        emotions = [self.joy, self.sadness, self.anger, self.fear, 
                   self.surprise, self.disgust, self.neutral]
        return max(emotions)


@dataclass
class SentenceAnalysis:
    """Analysis result for a single sentence."""
    text: str
    emotion_scores: EmotionScore
    sentiment_polarity: float  # -1 to 1
    emotional_intensity: float  # 0 to 1


@dataclass
class EmotionAnalysisResult:
    """Complete emotion analysis result."""
    sentences: List[SentenceAnalysis]
    overall_emotion_scores: EmotionScore
    overall_sentiment: float
    emotional_consistency: float  # How consistent emotions are throughout
    emotion_transitions: List[Tuple[str, str]]  # Emotion changes between sentences
    likert_rating: int  # 1-5 scale


class EmotionDetector:
    """Detects emotions in speech text using transformer models."""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion detector.
        
        Args:
            model_name: Name of the pre-trained emotion detection model
        """
        # Suppress transformers warnings for cleaner output
        import transformers
        transformers.logging.set_verbosity_error()
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emotion_pipeline = None
        self.sentiment_pipeline = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load the emotion detection and sentiment analysis models."""
        try:
            logger.info(f"Loading emotion detection model: {self.model_name}")
            
            # Load emotion detection model (using top_k instead of deprecated return_all_scores)
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                top_k=None  # Returns all scores (replaces return_all_scores=True)
            )
            
            # Load sentiment analysis model (using top_k instead of deprecated return_all_scores)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                top_k=None  # Returns all scores (replaces return_all_scores=True)
            )
            
            logger.info("Emotion detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading emotion detection models: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text and split into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences using basic rules
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Ensure minimum sentence length
        filtered_sentences = []
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # At least 3 words
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _is_likely_english(self, text: str) -> bool:
        """
        Simple heuristic to check if text is likely English.
        
        Args:
            text: Input text
            
        Returns:
            True if text appears to be English
        """
        # Check for common English words
        common_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 
                       'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
                       'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
                       'by', 'but', 'not', 'what', 'all', 'were', 'we', 'when'}
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return True  # Assume English if no words found
        
        english_word_count = sum(1 for word in words if word in common_words)
        english_ratio = english_word_count / len(words)
        
        # Also check character distribution - English uses mostly ASCII
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / len(text) if text else 1
        
        return english_ratio > 0.1 and ascii_ratio > 0.9
    
    def _analyze_sentence_emotion(self, sentence: str) -> EmotionScore:
        """
        Analyze emotions in a single sentence.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            EmotionScore object
        """
        try:
            # Get emotion predictions
            emotion_results = self.emotion_pipeline(sentence)[0]
            
            # Initialize emotion scores
            emotion_scores = {
                'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0,
                'surprise': 0.0, 'disgust': 0.0, 'neutral': 0.0
            }
            
            # Map model outputs to our emotion categories
            emotion_mapping = {
                'joy': 'joy',
                'sadness': 'sadness',
                'anger': 'anger',
                'fear': 'fear',
                'surprise': 'surprise',
                'disgust': 'disgust',
                'neutral': 'neutral',
                'love': 'joy',  # Map love to joy
                'optimism': 'joy',  # Map optimism to joy
                'pessimism': 'sadness',  # Map pessimism to sadness
            }
            
            # Extract scores
            for result in emotion_results:
                emotion_label = result['label'].lower()
                score = result['score']
                
                if emotion_label in emotion_mapping:
                    target_emotion = emotion_mapping[emotion_label]
                    emotion_scores[target_emotion] = max(emotion_scores[target_emotion], score)
            
            return EmotionScore(**emotion_scores)
            
        except Exception as e:
            logger.warning(f"Error analyzing sentence emotion: {str(e)}")
            # Return neutral emotion scores as fallback
            return EmotionScore(
                joy=0.0, sadness=0.0, anger=0.0, fear=0.0,
                surprise=0.0, disgust=0.0, neutral=1.0
            )
    
    def _analyze_sentence_sentiment(self, sentence: str) -> float:
        """
        Analyze sentiment polarity of a sentence.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            Sentiment polarity (-1 to 1)
        """
        try:
            sentiment_results = self.sentiment_pipeline(sentence)[0]
            
            # Calculate polarity score
            polarity = 0.0
            for result in sentiment_results:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    polarity += score
                elif 'negative' in label:
                    polarity -= score
                # Neutral contributes 0 to polarity
            
            return max(-1.0, min(1.0, polarity))  # Clamp to [-1, 1]
            
        except Exception as e:
            logger.warning(f"Error analyzing sentence sentiment: {str(e)}")
            return 0.0  # Neutral sentiment as fallback
    
    def _calculate_emotional_intensity(self, emotion_scores: EmotionScore) -> float:
        """
        Calculate emotional intensity from emotion scores.
        
        Args:
            emotion_scores: EmotionScore object
            
        Returns:
            Emotional intensity (0 to 1)
        """
        # Sum of all non-neutral emotions
        non_neutral_sum = (emotion_scores.joy + emotion_scores.sadness + 
                          emotion_scores.anger + emotion_scores.fear + 
                          emotion_scores.surprise + emotion_scores.disgust)
        
        return min(1.0, non_neutral_sum)
    
    def analyze_text(self, text: str) -> EmotionAnalysisResult:
        """
        Perform complete emotion analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            EmotionAnalysisResult object
        """
        if not self.emotion_pipeline or not self.sentiment_pipeline:
            raise RuntimeError("Emotion detection models not loaded")
        
        logger.info("Starting emotion analysis...")
        
        # Validate that text is likely English
        if not self._is_likely_english(text):
            logger.warning("Input text may not be in English. Speech Analyzer is optimized for English text.")
        
        # Preprocess and split text
        sentences = self._preprocess_text(text)
        
        if not sentences:
            # Return neutral analysis for empty text
            neutral_score = EmotionScore(0, 0, 0, 0, 0, 0, 1.0)
            return EmotionAnalysisResult(
                sentences=[],
                overall_emotion_scores=neutral_score,
                overall_sentiment=0.0,
                emotional_consistency=1.0,
                emotion_transitions=[],
                likert_rating=3
            )
        
        # Analyze each sentence
        sentence_analyses = []
        for sentence in sentences:
            emotion_scores = self._analyze_sentence_emotion(sentence)
            sentiment = self._analyze_sentence_sentiment(sentence)
            intensity = self._calculate_emotional_intensity(emotion_scores)
            
            analysis = SentenceAnalysis(
                text=sentence,
                emotion_scores=emotion_scores,
                sentiment_polarity=sentiment,
                emotional_intensity=intensity
            )
            sentence_analyses.append(analysis)
        
        # Calculate overall metrics
        overall_scores = self._calculate_overall_emotions(sentence_analyses)
        overall_sentiment = np.mean([s.sentiment_polarity for s in sentence_analyses])
        consistency = self._calculate_emotional_consistency(sentence_analyses)
        transitions = self._analyze_emotion_transitions(sentence_analyses)
        likert_rating = self._calculate_likert_rating(overall_scores, overall_sentiment, consistency)
        
        result = EmotionAnalysisResult(
            sentences=sentence_analyses,
            overall_emotion_scores=overall_scores,
            overall_sentiment=overall_sentiment,
            emotional_consistency=consistency,
            emotion_transitions=transitions,
            likert_rating=likert_rating
        )
        
        logger.info(f"Emotion analysis completed. Overall emotion: {overall_scores.get_dominant_emotion()}, "
                   f"Sentiment: {overall_sentiment:.2f}, Rating: {likert_rating}/5")
        
        return result
    
    def _calculate_overall_emotions(self, sentence_analyses: List[SentenceAnalysis]) -> EmotionScore:
        """Calculate overall emotion scores from sentence analyses."""
        if not sentence_analyses:
            return EmotionScore(0, 0, 0, 0, 0, 0, 1.0)
        
        # Average emotion scores across sentences
        joy = np.mean([s.emotion_scores.joy for s in sentence_analyses])
        sadness = np.mean([s.emotion_scores.sadness for s in sentence_analyses])
        anger = np.mean([s.emotion_scores.anger for s in sentence_analyses])
        fear = np.mean([s.emotion_scores.fear for s in sentence_analyses])
        surprise = np.mean([s.emotion_scores.surprise for s in sentence_analyses])
        disgust = np.mean([s.emotion_scores.disgust for s in sentence_analyses])
        neutral = np.mean([s.emotion_scores.neutral for s in sentence_analyses])
        
        return EmotionScore(joy, sadness, anger, fear, surprise, disgust, neutral)
    
    def _calculate_emotional_consistency(self, sentence_analyses: List[SentenceAnalysis]) -> float:
        """Calculate how consistent emotions are throughout the text."""
        if len(sentence_analyses) <= 1:
            return 1.0
        
        dominant_emotions = [s.emotion_scores.get_dominant_emotion() for s in sentence_analyses]
        most_common_emotion = Counter(dominant_emotions).most_common(1)[0][0]
        consistency = dominant_emotions.count(most_common_emotion) / len(dominant_emotions)
        
        return consistency
    
    def _analyze_emotion_transitions(self, sentence_analyses: List[SentenceAnalysis]) -> List[Tuple[str, str]]:
        """Analyze emotion transitions between sentences."""
        if len(sentence_analyses) <= 1:
            return []
        
        transitions = []
        for i in range(len(sentence_analyses) - 1):
            current_emotion = sentence_analyses[i].emotion_scores.get_dominant_emotion()
            next_emotion = sentence_analyses[i + 1].emotion_scores.get_dominant_emotion()
            
            if current_emotion != next_emotion:
                transitions.append((current_emotion, next_emotion))
        
        return transitions
    
    def _calculate_likert_rating(self, overall_scores: EmotionScore, 
                                overall_sentiment: float, consistency: float) -> int:
        """
        Calculate Likert scale rating (1-5) for emotional delivery.
        
        Args:
            overall_scores: Overall emotion scores
            overall_sentiment: Overall sentiment polarity
            consistency: Emotional consistency score
            
        Returns:
            Likert scale rating (1-5)
        """
        # Base score from emotional appropriateness
        base_score = 3  # Neutral starting point
        
        # Adjust based on emotional clarity (confidence in dominant emotion)
        confidence = overall_scores.get_confidence()
        if confidence > 0.7:
            base_score += 1
        elif confidence < 0.3:
            base_score -= 1
        
        # Adjust based on consistency
        if consistency > 0.8:
            base_score += 0.5
        elif consistency < 0.4:
            base_score -= 0.5
        
        # Adjust based on emotional engagement (non-neutral emotions)
        emotional_engagement = 1 - overall_scores.neutral
        if emotional_engagement > 0.6:
            base_score += 0.5
        elif emotional_engagement < 0.2:
            base_score -= 0.5
        
        # Clamp to valid range and round
        return max(1, min(5, round(base_score)))
