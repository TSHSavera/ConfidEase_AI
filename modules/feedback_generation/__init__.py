"""
Feedback Generation Module for Speech Analyzer

Generates personalized feedback based on analysis results using template-based
and AI-generated approaches. Provides specific, actionable recommendations.
"""

from .feedback_generator import FeedbackGenerator
from .feedback_templates import FeedbackTemplates

__all__ = ['FeedbackGenerator', 'FeedbackTemplates']
