"""
Analysis Module for Speech Analyzer

Core analysis module containing sub-modules for emotion detection, pacing analysis,
clarity assessment, and tone delivery evaluation.
"""

from .emotion_detection import EmotionDetector
from .pacing_analysis import PacingAnalyzer
from .clarity_analysis import ClarityAnalyzer
from .tone_delivery import ToneDeliveryAnalyzer

__all__ = ['EmotionDetector', 'PacingAnalyzer', 'ClarityAnalyzer', 'ToneDeliveryAnalyzer']
