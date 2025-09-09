"""
Parameter Validator Module

Validates and structures user input parameters for speech analysis.
Ensures all required parameters are provided and within valid ranges.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class AudienceType(Enum):
    """Enumeration of possible audience types."""
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    APATHETIC = "apathetic"
    UNINFORMED = "uninformed"
    EXPERTS = "experts"
    LAYPEOPLE = "laypeople"
    YOUNG_ONES = "young_ones"
    TEENS = "teens"
    ADULTS = "adults"


class DeliveryGoal(Enum):
    """Enumeration of possible delivery goals."""
    PERSUADE = "persuade"
    INFORM = "inform"
    ENTERTAIN = "entertain"
    IMPLORE = "implore"
    DELIVER = "deliver"
    COMMEMORATE = "commemorate"
    MOTIVATE = "motivate"


@dataclass
class SpeechParameters:
    """Structured speech analysis parameters."""
    audience_type: AudienceType
    delivery_goal: DeliveryGoal
    speech_text: str
    audio_file_path: Optional[str] = None
    custom_notes: Optional[str] = None


class ParameterValidator:
    """Validates and processes user input parameters."""
    
    def __init__(self):
        """Initialize the parameter validator."""
        self.audience_descriptions = {
            AudienceType.FRIENDLY: "Receptive souls, their views align",
            AudienceType.HOSTILE: "Unyielding, with minds that disagree",
            AudienceType.APATHETIC: "Minds that lack engagement's spark",
            AudienceType.UNINFORMED: "Those seeking knowledge, wanting to be free",
            AudienceType.EXPERTS: "Masters of the field, whose wisdom marks",
            AudienceType.LAYPEOPLE: "Without specialized lore, just a simple mark",
            AudienceType.YOUNG_ONES: "Little ones, their minds still in the dark",
            AudienceType.TEENS: "Those growing up, learning about what marks",
            AudienceType.ADULTS: "Grown-ups, who listen and think about the mark"
        }
        
        self.goal_descriptions = {
            DeliveryGoal.PERSUADE: "To change minds and win hearts through compelling argument",
            DeliveryGoal.INFORM: "To educate and enlighten with clear knowledge",
            DeliveryGoal.ENTERTAIN: "To delight and amuse the listening audience",
            DeliveryGoal.IMPLORE: "To earnestly request or beseech with passion",
            DeliveryGoal.DELIVER: "To present information clearly and effectively",
            DeliveryGoal.COMMEMORATE: "To honor and remember with reverence",
            DeliveryGoal.MOTIVATE: "To inspire action and drive forward movement"
        }
    
    def validate_audience_type(self, audience_input: str) -> AudienceType:
        """
        Validate and convert audience type input.
        
        Args:
            audience_input: String representation of audience type
            
        Returns:
            AudienceType enum value
            
        Raises:
            ValueError: If audience type is invalid
        """
        audience_input = audience_input.lower().strip()
        
        # Direct enum value match
        for audience_type in AudienceType:
            if audience_type.value == audience_input:
                return audience_type
        
        # Flexible matching for common variations
        audience_mappings = {
            "friend": AudienceType.FRIENDLY,
            "friendly": AudienceType.FRIENDLY,
            "receptive": AudienceType.FRIENDLY,
            "hostile": AudienceType.HOSTILE,
            "unfriendly": AudienceType.HOSTILE,
            "disagreeable": AudienceType.HOSTILE,
            "apathetic": AudienceType.APATHETIC,
            "disinterested": AudienceType.APATHETIC,
            "bored": AudienceType.APATHETIC,
            "uninformed": AudienceType.UNINFORMED,
            "novice": AudienceType.UNINFORMED,
            "beginner": AudienceType.UNINFORMED,
            "expert": AudienceType.EXPERTS,
            "experts": AudienceType.EXPERTS,
            "professional": AudienceType.EXPERTS,
            "layperson": AudienceType.LAYPEOPLE,
            "laypeople": AudienceType.LAYPEOPLE,
            "general": AudienceType.LAYPEOPLE,
            "children": AudienceType.YOUNG_ONES,
            "kids": AudienceType.YOUNG_ONES,
            "young": AudienceType.YOUNG_ONES,
            "teen": AudienceType.TEENS,
            "teens": AudienceType.TEENS,
            "teenager": AudienceType.TEENS,
            "adult": AudienceType.ADULTS,
            "adults": AudienceType.ADULTS,
            "grown-up": AudienceType.ADULTS
        }
        
        if audience_input in audience_mappings:
            return audience_mappings[audience_input]
        
        # If no match found, raise an error with available options
        valid_options = [audience.value for audience in AudienceType]
        raise ValueError(f"Invalid audience type: '{audience_input}'. "
                        f"Valid options: {', '.join(valid_options)}")
    
    def validate_delivery_goal(self, goal_input: str) -> DeliveryGoal:
        """
        Validate and convert delivery goal input.
        
        Args:
            goal_input: String representation of delivery goal
            
        Returns:
            DeliveryGoal enum value
            
        Raises:
            ValueError: If delivery goal is invalid
        """
        goal_input = goal_input.lower().strip()
        
        # Direct enum value match
        for goal in DeliveryGoal:
            if goal.value == goal_input:
                return goal
        
        # Flexible matching for common variations
        goal_mappings = {
            "convince": DeliveryGoal.PERSUADE,
            "persuade": DeliveryGoal.PERSUADE,
            "argue": DeliveryGoal.PERSUADE,
            "inform": DeliveryGoal.INFORM,
            "educate": DeliveryGoal.INFORM,
            "teach": DeliveryGoal.INFORM,
            "explain": DeliveryGoal.INFORM,
            "entertain": DeliveryGoal.ENTERTAIN,
            "amuse": DeliveryGoal.ENTERTAIN,
            "delight": DeliveryGoal.ENTERTAIN,
            "implore": DeliveryGoal.IMPLORE,
            "beg": DeliveryGoal.IMPLORE,
            "plead": DeliveryGoal.IMPLORE,
            "deliver": DeliveryGoal.DELIVER,
            "present": DeliveryGoal.DELIVER,
            "report": DeliveryGoal.DELIVER,
            "commemorate": DeliveryGoal.COMMEMORATE,
            "honor": DeliveryGoal.COMMEMORATE,
            "remember": DeliveryGoal.COMMEMORATE,
            "motivate": DeliveryGoal.MOTIVATE,
            "inspire": DeliveryGoal.MOTIVATE,
            "encourage": DeliveryGoal.MOTIVATE
        }
        
        if goal_input in goal_mappings:
            return goal_mappings[goal_input]
        
        # If no match found, raise an error with available options
        valid_options = [goal.value for goal in DeliveryGoal]
        raise ValueError(f"Invalid delivery goal: '{goal_input}'. "
                        f"Valid options: {', '.join(valid_options)}")
    
    def validate_speech_text(self, text: str) -> str:
        """
        Validate speech text input.
        
        Args:
            text: Speech text to validate
            
        Returns:
            Cleaned and validated text
            
        Raises:
            ValueError: If text is invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Speech text must be a non-empty string")
        
        text = text.strip()
        if len(text) < 10:
            raise ValueError("Speech text must be at least 10 characters long")
        
        if len(text) > 50000:
            raise ValueError("Speech text must be less than 50,000 characters")
        
        return text
    
    def create_parameters(self, 
                         audience_type: str,
                         delivery_goal: str,
                         speech_text: str,
                         audio_file_path: Optional[str] = None,
                         custom_notes: Optional[str] = None) -> SpeechParameters:
        """
        Create validated SpeechParameters object.
        
        Args:
            audience_type: Target audience type
            delivery_goal: Speech delivery goal
            speech_text: Speech content
            audio_file_path: Optional path to audio file
            custom_notes: Optional custom user notes
            
        Returns:
            Validated SpeechParameters object
            
        Raises:
            ValueError: If any parameter is invalid
        """
        try:
            validated_audience = self.validate_audience_type(audience_type)
            validated_goal = self.validate_delivery_goal(delivery_goal)
            validated_text = self.validate_speech_text(speech_text)
            
            parameters = SpeechParameters(
                audience_type=validated_audience,
                delivery_goal=validated_goal,
                speech_text=validated_text,
                audio_file_path=audio_file_path,
                custom_notes=custom_notes
            )
            
            logger.info(f"Parameters validated successfully: "
                       f"Audience={validated_audience.value}, "
                       f"Goal={validated_goal.value}")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            raise
    
    def get_audience_options(self) -> Dict[str, str]:
        """
        Get all available audience types with descriptions.
        
        Returns:
            Dictionary mapping audience types to descriptions
        """
        return {
            audience.value: self.audience_descriptions[audience]
            for audience in AudienceType
        }
    
    def get_goal_options(self) -> Dict[str, str]:
        """
        Get all available delivery goals with descriptions.
        
        Returns:
            Dictionary mapping delivery goals to descriptions
        """
        return {
            goal.value: self.goal_descriptions[goal]
            for goal in DeliveryGoal
        }
    
    def get_ideal_pacing_range(self, audience: AudienceType, goal: DeliveryGoal) -> tuple[int, int]:
        """
        Get ideal words per minute range based on audience and goal.
        
        Args:
            audience: Target audience type
            goal: Delivery goal
            
        Returns:
            Tuple of (min_wpm, max_wpm)
        """
        # Base ranges for different goals
        goal_ranges = {
            DeliveryGoal.PERSUADE: (140, 180),
            DeliveryGoal.INFORM: (150, 190),
            DeliveryGoal.ENTERTAIN: (160, 200),
            DeliveryGoal.IMPLORE: (120, 160),
            DeliveryGoal.DELIVER: (150, 180),
            DeliveryGoal.COMMEMORATE: (120, 150),
            DeliveryGoal.MOTIVATE: (160, 200)
        }
        
        base_min, base_max = goal_ranges.get(goal, (150, 180))
        
        # Adjust for audience
        if audience in [AudienceType.YOUNG_ONES, AudienceType.UNINFORMED]:
            # Slower for children and uninformed audiences
            return (base_min - 20, base_max - 20)
        elif audience == AudienceType.EXPERTS:
            # Can handle faster pace
            return (base_min + 10, base_max + 20)
        elif audience == AudienceType.HOSTILE:
            # Slower to be more persuasive
            return (base_min - 10, base_max - 10)
        
        return (base_min, base_max)
