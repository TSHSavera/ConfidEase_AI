"""
Audio Input Module for Speech Analyzer

This module handles audio file input, preprocessing, and speech-to-text conversion.
Supports MP3 and WAV file formats with noise reduction and normalization.
"""

from .audio_processor import AudioProcessor
from .speech_to_text import SpeechToText

__all__ = ['AudioProcessor', 'SpeechToText']
