"""
Parameter Input Module for Speech Analyzer

This module handles user input for target audience, delivery goals, and speech parameters.
Provides validation and structured data for the analysis modules.
"""

from .parameter_validator import ParameterValidator
from .user_interface import UserInterface

__all__ = ['ParameterValidator', 'UserInterface']
