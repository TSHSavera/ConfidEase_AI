"""
Output Module for Speech Analyzer

Handles presentation of analysis results including Likert scale ratings,
detailed feedback, and optional visual representations.
"""

from .result_presenter import ResultPresenter
from .visualization import VisualizationGenerator

__all__ = ['ResultPresenter', 'VisualizationGenerator']
