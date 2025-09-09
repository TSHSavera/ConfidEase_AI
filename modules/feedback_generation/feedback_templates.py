"""
Feedback Templates Module

Contains templates and patterns for generating structured feedback
based on analysis results and user parameters.
"""

from typing import Dict, List, Any
from ..parameter_input.parameter_validator import AudienceType, DeliveryGoal


class FeedbackTemplates:
    """Templates for generating structured feedback."""
    
    def __init__(self):
        """Initialize feedback templates."""
        self.emotion_templates = self._create_emotion_templates()
        self.pacing_templates = self._create_pacing_templates()
        self.clarity_templates = self._create_clarity_templates()
        self.tone_templates = self._create_tone_templates()
        self.improvement_templates = self._create_improvement_templates()
    
    def _create_emotion_templates(self) -> Dict[str, Dict[str, str]]:
        """Create templates for emotion feedback."""
        return {
            'high_rating': {
                'opening': "Excellent emotional delivery! Your speech demonstrates {dominant_emotion} effectively.",
                'body': "Your emotional tone is well-suited for {goal} with {audience} audience. "
                       "The {confidence:.0%} confidence in your emotional expression shows authenticity.",
                'specific': "Particularly strong in sentences like: '{example_sentence}'"
            },
            'medium_rating': {
                'opening': "Good emotional foundation with room for enhancement.",
                'body': "Your {dominant_emotion} comes through clearly, though the emotional range could be "
                       "expanded for greater impact with {audience} audience.",
                'specific': "Consider strengthening emotional expression in sections discussing {topic_area}."
            },
            'low_rating': {
                'opening': "Your emotional delivery needs development to better serve your {goal} goal.",
                'body': "The current emotional tone appears too {current_issue} for {audience} audience. "
                       "Focus on conveying more {recommended_emotion} to enhance your message.",
                'specific': "Practice emotional variation, especially when transitioning between key points."
            }
        }
    
    def _create_pacing_templates(self) -> Dict[str, Dict[str, str]]:
        """Create templates for pacing feedback."""
        return {
            'optimal': {
                'opening': "Excellent pacing! Your {wpm:.0f} words per minute is ideal for {audience} audience.",
                'body': "This pace allows your audience to process information effectively while maintaining engagement.",
                'consistency': "Your pacing consistency of {consistency:.0%} shows good control."
            },
            'too_fast': {
                'opening': "Your speaking pace of {wpm:.0f} WPM is faster than ideal for {audience} audience.",
                'body': "Consider slowing to {ideal_min}-{ideal_max} WPM. Fast pacing can overwhelm {audience} "
                       "and reduce comprehension, especially for {goal} presentations.",
                'technique': "Practice with a metronome or record yourself to develop better pace awareness."
            },
            'too_slow': {
                'opening': "Your speaking pace of {wpm:.0f} WPM is slower than optimal for {audience} audience.",
                'body': "Increasing to {ideal_min}-{ideal_max} WPM would maintain better engagement. "
                       "Slow pacing can cause {audience} audience to lose interest in {goal} content.",
                'technique': "Practice reading aloud with energy and purpose to naturally increase pace."
            },
            'inconsistent': {
                'opening': "Your pacing varies significantly throughout the speech.",
                'body': "While some variation is natural, the current inconsistency ({consistency:.0%} consistency) "
                       "may distract from your {goal} message.",
                'technique': "Work on maintaining steadier rhythm while allowing natural emphasis on key points."
            }
        }
    
    def _create_clarity_templates(self) -> Dict[str, Dict[str, str]]:
        """Create templates for clarity feedback."""
        return {
            'high_clarity': {
                'opening': "Outstanding clarity! Your message is highly comprehensible.",
                'vocabulary': "Your vocabulary choices are appropriate for {audience} with {diversity:.0%} word diversity.",
                'structure': "Sentence structure supports understanding with an average of {avg_length:.1f} words per sentence."
            },
            'moderate_clarity': {
                'opening': "Good clarity with opportunities for improvement.",
                'vocabulary': "Consider simplifying {complex_words} complex words for better {audience} comprehension.",
                'structure': "Some sentences exceed optimal length. Aim for 10-20 words per sentence."
            },
            'low_clarity': {
                'opening': "Clarity needs significant improvement for effective {goal} communication.",
                'vocabulary': "Reduce complex vocabulary and eliminate {filler_count} filler words.",
                'structure': "Simplify sentence structure and address ambiguity issues: {ambiguity_issues}."
            },
            'pronunciation': {
                'challenges': "Pay attention to pronunciation of: {difficult_words}",
                'practice': "Practice these challenging words separately before incorporating into speech."
            }
        }
    
    def _create_tone_templates(self) -> Dict[str, Dict[str, str]]:
        """Create templates for tone delivery feedback."""
        return {
            'excellent_alignment': {
                'opening': "Exceptional tone alignment! Your delivery perfectly matches your {goal} objective.",
                'audience': "The tone is highly appropriate for {audience} audience with {appropriateness:.0%} suitability.",
                'consistency': "Consistent tone throughout maintains audience engagement and trust."
            },
            'good_alignment': {
                'opening': "Strong tone alignment with your {goal} goal.",
                'audience': "Generally appropriate for {audience}, with some areas for fine-tuning.",
                'suggestion': "Focus on {improvement_area} to enhance overall effectiveness."
            },
            'misalignment': {
                'opening': "Your tone doesn't fully support your {goal} objective with {audience} audience.",
                'emotional': "Current emotional approach: {current_emotion}. Recommended: {ideal_emotion}.",
                'adjustment': "Adjust {specific_aspects} to better achieve your communication goals."
            }
        }
    
    def _create_improvement_templates(self) -> Dict[str, List[str]]:
        """Create templates for specific improvement suggestions."""
        return {
            'emotion_improvement': [
                "Practice emotional expression exercises to develop range and authenticity.",
                "Record yourself reading emotional content to hear your current expression.",
                "Study speeches by effective {goal} speakers to observe emotional techniques.",
                "Work with a coach to develop more natural emotional variation."
            ],
            'pacing_improvement': [
                "Use a metronome during practice to develop consistent rhythm.",
                "Mark your script with pacing cues (fast, slow, pause).",
                "Practice reading news articles aloud to develop steady pace.",
                "Record and time different sections to monitor pace variation."
            ],
            'clarity_improvement': [
                "Read your speech aloud to identify unclear passages.",
                "Ask others to repeat back key points to test comprehension.",
                "Replace complex words with simpler alternatives where possible.",
                "Break long sentences into shorter, clearer statements."
            ],
            'audience_adaptation': [
                "Research your specific audience's background and interests.",
                "Adjust technical language based on audience expertise level.",
                "Consider cultural factors that might affect message reception.",
                "Practice with a sample audience similar to your target group."
            ]
        }
    
    def get_emotion_template(self, rating: int, **kwargs) -> str:
        """Get emotion feedback template based on rating."""
        if rating >= 4:
            template_key = 'high_rating'
        elif rating >= 3:
            template_key = 'medium_rating'
        else:
            template_key = 'low_rating'
        
        template = self.emotion_templates[template_key]
        
        feedback = template['opening'].format(**kwargs)
        if 'body' in template:
            feedback += " " + template['body'].format(**kwargs)
        if 'specific' in template and 'example_sentence' in kwargs:
            feedback += " " + template['specific'].format(**kwargs)
        
        return feedback
    
    def get_pacing_template(self, wpm: float, ideal_range: tuple, consistency: float, **kwargs) -> str:
        """Get pacing feedback template based on analysis."""
        ideal_min, ideal_max = ideal_range
        
        if ideal_min <= wpm <= ideal_max:
            if consistency >= 0.7:
                template_key = 'optimal'
            else:
                template_key = 'inconsistent'
        elif wpm > ideal_max:
            template_key = 'too_fast'
        else:
            template_key = 'too_slow'
        
        template = self.pacing_templates[template_key]
        
        # Prepare formatting arguments
        format_args = {
            'wpm': wpm,
            'ideal_min': ideal_min,
            'ideal_max': ideal_max,
            'consistency': consistency,
            **kwargs
        }
        
        feedback = template['opening'].format(**format_args)
        if 'body' in template:
            feedback += " " + template['body'].format(**format_args)
        if 'technique' in template:
            feedback += " " + template['technique'].format(**format_args)
        
        return feedback
    
    def get_clarity_template(self, clarity_score: float, **kwargs) -> str:
        """Get clarity feedback template based on score."""
        if clarity_score >= 0.7:
            template_key = 'high_clarity'
        elif clarity_score >= 0.4:
            template_key = 'moderate_clarity'
        else:
            template_key = 'low_clarity'
        
        template = self.clarity_templates[template_key]
        
        feedback = template['opening'].format(**kwargs)
        if 'vocabulary' in template:
            feedback += " " + template['vocabulary'].format(**kwargs)
        if 'structure' in template:
            feedback += " " + template['structure'].format(**kwargs)
        
        return feedback
    
    def get_tone_template(self, alignment_score: float, **kwargs) -> str:
        """Get tone feedback template based on alignment score."""
        if alignment_score >= 0.8:
            template_key = 'excellent_alignment'
        elif alignment_score >= 0.6:
            template_key = 'good_alignment'
        else:
            template_key = 'misalignment'
        
        template = self.tone_templates[template_key]
        
        feedback = template['opening'].format(**kwargs)
        if 'audience' in template:
            feedback += " " + template['audience'].format(**kwargs)
        if 'consistency' in template:
            feedback += " " + template['consistency'].format(**kwargs)
        if 'suggestion' in template:
            feedback += " " + template['suggestion'].format(**kwargs)
        if 'adjustment' in template:
            feedback += " " + template['adjustment'].format(**kwargs)
        
        return feedback
    
    def get_improvement_suggestions(self, improvement_areas: List[str]) -> List[str]:
        """Get specific improvement suggestions based on identified areas."""
        suggestions = []
        
        area_mappings = {
            'emotion': 'emotion_improvement',
            'pacing': 'pacing_improvement',
            'clarity': 'clarity_improvement',
            'audience': 'audience_adaptation'
        }
        
        for area in improvement_areas:
            # Map general area to specific template key
            template_key = None
            for key, template_name in area_mappings.items():
                if key in area.lower():
                    template_key = template_name
                    break
            
            if template_key and template_key in self.improvement_templates:
                area_suggestions = self.improvement_templates[template_key]
                suggestions.extend(area_suggestions[:2])  # Take first 2 suggestions per area
        
        return suggestions
    
    def create_section_header(self, section_name: str, rating: int) -> str:
        """Create a formatted section header with rating."""
        stars = "⭐" * rating + "☆" * (5 - rating)
        return f"\n{'='*50}\n{section_name.upper()} {stars} ({rating}/5)\n{'='*50}"
    
    def create_summary_template(self, overall_rating: int, strengths: List[str], 
                               improvements: List[str]) -> str:
        """Create overall summary template."""
        stars = "⭐" * overall_rating + "☆" * (5 - overall_rating)
        
        summary = f"\n{'='*60}\n"
        summary += f"OVERALL ASSESSMENT {stars} ({overall_rating}/5)\n"
        summary += f"{'='*60}\n"
        
        if strengths:
            summary += "\n🎯 KEY STRENGTHS:\n"
            for i, strength in enumerate(strengths[:3], 1):
                summary += f"{i}. {strength}\n"
        
        if improvements:
            summary += "\n🔧 PRIORITY IMPROVEMENTS:\n"
            for i, improvement in enumerate(improvements[:3], 1):
                summary += f"{i}. {improvement}\n"
        
        return summary
