"""
Tone Delivery Analysis Sub-Module

Combines results from emotion detection, pacing analysis, and clarity assessment
to evaluate how well the speech aligns with the delivery goal and target audience.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from ..parameter_input.parameter_validator import AudienceType, DeliveryGoal
from .emotion_detection import EmotionAnalysisResult
from .pacing_analysis import PacingAnalysisResult
from .clarity_analysis import ClarityAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ToneAlignment:
    """Represents how well the tone aligns with goals and audience."""
    emotion_alignment: float  # 0-1 scale
    pacing_alignment: float  # 0-1 scale
    clarity_alignment: float  # 0-1 scale
    overall_alignment: float  # 0-1 scale
    alignment_description: str


@dataclass
class AudienceAppropriatenesss:
    """Evaluation of appropriateness for target audience."""
    emotion_appropriateness: float  # 0-1 scale
    complexity_appropriateness: float  # 0-1 scale
    pacing_appropriateness: float  # 0-1 scale
    overall_appropriateness: float  # 0-1 scale
    appropriateness_description: str


@dataclass
class ToneDeliveryResult:
    """Complete tone delivery analysis result."""
    tone_alignment: ToneAlignment
    audience_appropriateness: AudienceAppropriatenesss
    goal_achievement_score: float  # 0-1 scale
    consistency_score: float  # How consistent is the tone throughout
    improvement_areas: List[str]
    strengths: List[str]
    likert_rating: int  # 1-5 scale
    recommendations: List[str]


class ToneDeliveryAnalyzer:
    """Analyzes overall tone delivery by combining all analysis results."""
    
    def __init__(self):
        """Initialize the tone delivery analyzer."""
        self.goal_emotion_mappings = self._create_goal_emotion_mappings()
        self.audience_complexity_mappings = self._create_audience_complexity_mappings()
    
    def _create_goal_emotion_mappings(self) -> Dict[DeliveryGoal, Dict[str, float]]:
        """Create ideal emotion profiles for each delivery goal."""
        return {
            DeliveryGoal.PERSUADE: {
                'joy': 0.4, 'sadness': 0.1, 'anger': 0.2, 'fear': 0.1,
                'surprise': 0.1, 'disgust': 0.0, 'neutral': 0.1
            },
            DeliveryGoal.INFORM: {
                'joy': 0.2, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0,
                'surprise': 0.1, 'disgust': 0.0, 'neutral': 0.7
            },
            DeliveryGoal.ENTERTAIN: {
                'joy': 0.6, 'sadness': 0.1, 'anger': 0.0, 'fear': 0.0,
                'surprise': 0.2, 'disgust': 0.0, 'neutral': 0.1
            },
            DeliveryGoal.IMPLORE: {
                'joy': 0.1, 'sadness': 0.3, 'anger': 0.0, 'fear': 0.2,
                'surprise': 0.0, 'disgust': 0.0, 'neutral': 0.4
            },
            DeliveryGoal.DELIVER: {
                'joy': 0.2, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0,
                'surprise': 0.0, 'disgust': 0.0, 'neutral': 0.8
            },
            DeliveryGoal.COMMEMORATE: {
                'joy': 0.3, 'sadness': 0.2, 'anger': 0.0, 'fear': 0.0,
                'surprise': 0.0, 'disgust': 0.0, 'neutral': 0.5
            },
            DeliveryGoal.MOTIVATE: {
                'joy': 0.5, 'sadness': 0.0, 'anger': 0.1, 'fear': 0.0,
                'surprise': 0.1, 'disgust': 0.0, 'neutral': 0.3
            }
        }
    
    def _create_audience_complexity_mappings(self) -> Dict[AudienceType, Dict[str, float]]:
        """Create appropriate complexity levels for each audience type."""
        return {
            AudienceType.YOUNG_ONES: {
                'max_readability_grade': 5.0,
                'max_avg_sentence_length': 12.0,
                'max_complex_word_ratio': 0.05,
                'ideal_wpm_adjustment': -20
            },
            AudienceType.TEENS: {
                'max_readability_grade': 8.0,
                'max_avg_sentence_length': 15.0,
                'max_complex_word_ratio': 0.10,
                'ideal_wpm_adjustment': -10
            },
            AudienceType.ADULTS: {
                'max_readability_grade': 12.0,
                'max_avg_sentence_length': 18.0,
                'max_complex_word_ratio': 0.15,
                'ideal_wpm_adjustment': 0
            },
            AudienceType.EXPERTS: {
                'max_readability_grade': 16.0,
                'max_avg_sentence_length': 22.0,
                'max_complex_word_ratio': 0.25,
                'ideal_wpm_adjustment': 10
            },
            AudienceType.LAYPEOPLE: {
                'max_readability_grade': 10.0,
                'max_avg_sentence_length': 16.0,
                'max_complex_word_ratio': 0.12,
                'ideal_wpm_adjustment': -5
            },
            AudienceType.UNINFORMED: {
                'max_readability_grade': 8.0,
                'max_avg_sentence_length': 14.0,
                'max_complex_word_ratio': 0.08,
                'ideal_wpm_adjustment': -15
            },
            AudienceType.FRIENDLY: {
                'max_readability_grade': 12.0,
                'max_avg_sentence_length': 18.0,
                'max_complex_word_ratio': 0.15,
                'ideal_wpm_adjustment': 5
            },
            AudienceType.HOSTILE: {
                'max_readability_grade': 10.0,
                'max_avg_sentence_length': 15.0,
                'max_complex_word_ratio': 0.10,
                'ideal_wpm_adjustment': -10
            },
            AudienceType.APATHETIC: {
                'max_readability_grade': 10.0,
                'max_avg_sentence_length': 16.0,
                'max_complex_word_ratio': 0.12,
                'ideal_wpm_adjustment': 0
            }
        }
    
    def analyze_tone_delivery(self, 
                            emotion_result: EmotionAnalysisResult,
                            pacing_result: PacingAnalysisResult,
                            clarity_result: ClarityAnalysisResult,
                            audience: AudienceType,
                            goal: DeliveryGoal) -> ToneDeliveryResult:
        """
        Perform comprehensive tone delivery analysis.
        
        Args:
            emotion_result: Results from emotion analysis
            pacing_result: Results from pacing analysis
            clarity_result: Results from clarity analysis
            audience: Target audience type
            goal: Delivery goal
            
        Returns:
            ToneDeliveryResult object
        """
        logger.info(f"Starting tone delivery analysis for {goal.value} goal and {audience.value} audience...")
        
        # Analyze tone alignment with goal
        tone_alignment = self._analyze_tone_alignment(emotion_result, pacing_result, clarity_result, goal)
        
        # Analyze appropriateness for audience
        audience_appropriateness = self._analyze_audience_appropriateness(
            emotion_result, pacing_result, clarity_result, audience
        )
        
        # Calculate goal achievement score
        goal_achievement = self._calculate_goal_achievement(tone_alignment, audience_appropriateness)
        
        # Calculate consistency score
        consistency = self._calculate_consistency(emotion_result, pacing_result)
        
        # Identify strengths and improvement areas
        strengths = self._identify_strengths(tone_alignment, audience_appropriateness, goal_achievement)
        improvement_areas = self._identify_improvement_areas(
            emotion_result, pacing_result, clarity_result, tone_alignment, audience_appropriateness
        )
        
        # Calculate Likert rating
        likert_rating = self._calculate_tone_rating(goal_achievement, consistency)
        
        # Generate recommendations
        recommendations = self._generate_tone_recommendations(
            emotion_result, pacing_result, clarity_result, 
            tone_alignment, audience_appropriateness, audience, goal
        )
        
        result = ToneDeliveryResult(
            tone_alignment=tone_alignment,
            audience_appropriateness=audience_appropriateness,
            goal_achievement_score=goal_achievement,
            consistency_score=consistency,
            improvement_areas=improvement_areas,
            strengths=strengths,
            likert_rating=likert_rating,
            recommendations=recommendations
        )
        
        logger.info(f"Tone delivery analysis completed. Achievement score: {goal_achievement:.2f}, "
                   f"Rating: {likert_rating}/5")
        
        return result
    
    def _analyze_tone_alignment(self, emotion_result: EmotionAnalysisResult,
                              pacing_result: PacingAnalysisResult,
                              clarity_result: ClarityAnalysisResult,
                              goal: DeliveryGoal) -> ToneAlignment:
        """Analyze how well the tone aligns with the delivery goal."""
        
        # Emotion alignment
        ideal_emotions = self.goal_emotion_mappings[goal]
        actual_emotions = {
            'joy': emotion_result.overall_emotion_scores.joy,
            'sadness': emotion_result.overall_emotion_scores.sadness,
            'anger': emotion_result.overall_emotion_scores.anger,
            'fear': emotion_result.overall_emotion_scores.fear,
            'surprise': emotion_result.overall_emotion_scores.surprise,
            'disgust': emotion_result.overall_emotion_scores.disgust,
            'neutral': emotion_result.overall_emotion_scores.neutral
        }
        
        # Calculate emotion alignment using cosine similarity
        emotion_alignment = self._calculate_cosine_similarity(ideal_emotions, actual_emotions)
        
        # Pacing alignment
        ideal_min, ideal_max = pacing_result.ideal_wpm_range
        ideal_wpm = (ideal_min + ideal_max) / 2
        wpm_deviation = abs(pacing_result.overall_wpm - ideal_wpm) / ideal_wpm
        pacing_alignment = max(0.0, 1.0 - wpm_deviation)
        
        # Clarity alignment (higher clarity generally better for all goals)
        clarity_alignment = clarity_result.clarity_score
        
        # Overall alignment
        weights = self._get_goal_weights(goal)
        overall_alignment = (
            emotion_alignment * weights['emotion'] + 
            pacing_alignment * weights['pacing'] + 
            clarity_alignment * weights['clarity']
        )
        
        alignment_description = self._get_alignment_description(overall_alignment, goal)
        
        return ToneAlignment(
            emotion_alignment=emotion_alignment,
            pacing_alignment=pacing_alignment,
            clarity_alignment=clarity_alignment,
            overall_alignment=overall_alignment,
            alignment_description=alignment_description
        )
    
    def _analyze_audience_appropriateness(self, emotion_result: EmotionAnalysisResult,
                                        pacing_result: PacingAnalysisResult,
                                        clarity_result: ClarityAnalysisResult,
                                        audience: AudienceType) -> AudienceAppropriatenesss:
        """Analyze appropriateness for the target audience."""
        
        audience_specs = self.audience_complexity_mappings[audience]
        
        # Emotion appropriateness (varies by audience attitude)
        emotion_appropriateness = self._calculate_emotion_appropriateness(emotion_result, audience)
        
        # Complexity appropriateness
        readability_grade = clarity_result.overall_readability.get('flesch_kincaid_grade', 8.0)
        max_grade = audience_specs['max_readability_grade']
        complexity_score = max(0.0, 1.0 - max(0, readability_grade - max_grade) / max_grade)
        
        avg_sentence_length = clarity_result.structure_analysis.get('avg_sentence_length', 15)
        max_length = audience_specs['max_avg_sentence_length']
        length_score = max(0.0, 1.0 - max(0, avg_sentence_length - max_length) / max_length)
        
        complex_word_ratio = (clarity_result.vocabulary_analysis.complex_words / 
                            max(1, clarity_result.vocabulary_analysis.total_words))
        max_ratio = audience_specs['max_complex_word_ratio']
        word_score = max(0.0, 1.0 - max(0, complex_word_ratio - max_ratio) / max_ratio)
        
        complexity_appropriateness = (complexity_score + length_score + word_score) / 3
        
        # Pacing appropriateness
        pacing_appropriateness = self._calculate_pacing_appropriateness(pacing_result, audience)
        
        # Overall appropriateness
        overall_appropriateness = (
            emotion_appropriateness * 0.3 + 
            complexity_appropriateness * 0.4 + 
            pacing_appropriateness * 0.3
        )
        
        appropriateness_description = self._get_appropriateness_description(overall_appropriateness, audience)
        
        return AudienceAppropriatenesss(
            emotion_appropriateness=emotion_appropriateness,
            complexity_appropriateness=complexity_appropriateness,
            pacing_appropriateness=pacing_appropriateness,
            overall_appropriateness=overall_appropriateness,
            appropriateness_description=appropriateness_description
        )
    
    def _calculate_cosine_similarity(self, ideal: Dict[str, float], actual: Dict[str, float]) -> float:
        """Calculate cosine similarity between ideal and actual emotion profiles."""
        ideal_vector = np.array([ideal[key] for key in sorted(ideal.keys())])
        actual_vector = np.array([actual[key] for key in sorted(actual.keys())])
        
        # Normalize vectors
        ideal_norm = np.linalg.norm(ideal_vector)
        actual_norm = np.linalg.norm(actual_vector)
        
        if ideal_norm == 0 or actual_norm == 0:
            return 0.0
        
        similarity = np.dot(ideal_vector, actual_vector) / (ideal_norm * actual_norm)
        return max(0.0, similarity)  # Ensure non-negative
    
    def _get_goal_weights(self, goal: DeliveryGoal) -> Dict[str, float]:
        """Get importance weights for different aspects based on goal."""
        weights = {
            DeliveryGoal.PERSUADE: {'emotion': 0.4, 'pacing': 0.3, 'clarity': 0.3},
            DeliveryGoal.INFORM: {'emotion': 0.2, 'pacing': 0.3, 'clarity': 0.5},
            DeliveryGoal.ENTERTAIN: {'emotion': 0.5, 'pacing': 0.3, 'clarity': 0.2},
            DeliveryGoal.IMPLORE: {'emotion': 0.5, 'pacing': 0.2, 'clarity': 0.3},
            DeliveryGoal.DELIVER: {'emotion': 0.2, 'pacing': 0.3, 'clarity': 0.5},
            DeliveryGoal.COMMEMORATE: {'emotion': 0.4, 'pacing': 0.2, 'clarity': 0.4},
            DeliveryGoal.MOTIVATE: {'emotion': 0.5, 'pacing': 0.3, 'clarity': 0.2}
        }
        return weights.get(goal, {'emotion': 0.33, 'pacing': 0.33, 'clarity': 0.34})
    
    def _calculate_emotion_appropriateness(self, emotion_result: EmotionAnalysisResult, 
                                         audience: AudienceType) -> float:
        """Calculate how appropriate the emotions are for the audience."""
        # Base appropriateness from emotional intensity
        intensity = 1 - emotion_result.overall_emotion_scores.neutral
        
        # Adjust based on audience type
        if audience == AudienceType.YOUNG_ONES:
            # Children appreciate more emotional expression
            return min(1.0, intensity + 0.2)
        elif audience == AudienceType.EXPERTS:
            # Experts prefer more neutral, professional tone
            return max(0.3, 1.0 - intensity * 0.5)
        elif audience == AudienceType.HOSTILE:
            # Hostile audiences need calm, controlled emotions
            return max(0.2, 1.0 - intensity * 0.7)
        elif audience == AudienceType.APATHETIC:
            # Apathetic audiences need more emotional engagement
            return min(1.0, intensity + 0.3)
        else:
            # Default: moderate emotional expression is appropriate
            return max(0.3, 1.0 - abs(intensity - 0.5))
    
    def _calculate_pacing_appropriateness(self, pacing_result: PacingAnalysisResult,
                                        audience: AudienceType) -> float:
        """Calculate pacing appropriateness for audience."""
        # This is already calculated in the pacing analysis
        # We can use the Likert rating as a proxy
        return pacing_result.likert_rating / 5.0
    
    def _calculate_goal_achievement(self, tone_alignment: ToneAlignment,
                                  audience_appropriateness: AudienceAppropriatenesss) -> float:
        """Calculate overall goal achievement score."""
        return (tone_alignment.overall_alignment * 0.6 + 
                audience_appropriateness.overall_appropriateness * 0.4)
    
    def _calculate_consistency(self, emotion_result: EmotionAnalysisResult,
                             pacing_result: PacingAnalysisResult) -> float:
        """Calculate overall consistency score."""
        emotion_consistency = emotion_result.emotional_consistency
        pacing_consistency = pacing_result.pacing_consistency
        
        return (emotion_consistency + pacing_consistency) / 2
    
    def _identify_strengths(self, tone_alignment: ToneAlignment,
                          audience_appropriateness: AudienceAppropriatenesss,
                          goal_achievement: float) -> List[str]:
        """Identify strengths in the speech delivery."""
        strengths = []
        
        if tone_alignment.emotion_alignment > 0.8:
            strengths.append("Excellent emotional alignment with your delivery goal")
        
        if tone_alignment.pacing_alignment > 0.8:
            strengths.append("Optimal pacing for your intended message")
        
        if tone_alignment.clarity_alignment > 0.8:
            strengths.append("Very clear and comprehensible delivery")
        
        if audience_appropriateness.complexity_appropriateness > 0.8:
            strengths.append("Well-suited complexity level for your target audience")
        
        if audience_appropriateness.emotion_appropriateness > 0.8:
            strengths.append("Appropriate emotional tone for your audience")
        
        if goal_achievement > 0.85:
            strengths.append("Overall excellent alignment between delivery and objectives")
        
        return strengths
    
    def _identify_improvement_areas(self, emotion_result: EmotionAnalysisResult,
                                  pacing_result: PacingAnalysisResult,
                                  clarity_result: ClarityAnalysisResult,
                                  tone_alignment: ToneAlignment,
                                  audience_appropriateness: AudienceAppropriatenesss) -> List[str]:
        """Identify areas needing improvement."""
        areas = []
        
        if tone_alignment.emotion_alignment < 0.6:
            areas.append("Emotional tone doesn't match delivery goal")
        
        if tone_alignment.pacing_alignment < 0.6:
            areas.append("Speaking pace needs adjustment")
        
        if tone_alignment.clarity_alignment < 0.6:
            areas.append("Speech clarity needs improvement")
        
        if audience_appropriateness.complexity_appropriateness < 0.6:
            areas.append("Content complexity not well-suited for audience")
        
        if audience_appropriateness.emotion_appropriateness < 0.6:
            areas.append("Emotional approach not optimal for audience")
        
        if emotion_result.emotional_consistency < 0.6:
            areas.append("Emotional tone lacks consistency")
        
        if pacing_result.pacing_consistency < 0.6:
            areas.append("Speaking pace lacks consistency")
        
        return areas
    
    def _calculate_tone_rating(self, goal_achievement: float, consistency: float) -> int:
        """Calculate Likert scale rating for tone delivery."""
        # Weighted combination
        base_score = goal_achievement * 0.7 + consistency * 0.3
        
        # Convert to 1-5 scale
        rating = base_score * 5
        
        return max(1, min(5, round(rating)))
    
    def _get_alignment_description(self, alignment_score: float, goal: DeliveryGoal) -> str:
        """Get description of tone alignment."""
        if alignment_score >= 0.8:
            return f"Excellent alignment with {goal.value} goal"
        elif alignment_score >= 0.6:
            return f"Good alignment with {goal.value} goal"
        elif alignment_score >= 0.4:
            return f"Moderate alignment with {goal.value} goal"
        else:
            return f"Poor alignment with {goal.value} goal"
    
    def _get_appropriateness_description(self, appropriateness_score: float, 
                                       audience: AudienceType) -> str:
        """Get description of audience appropriateness."""
        if appropriateness_score >= 0.8:
            return f"Highly appropriate for {audience.value} audience"
        elif appropriateness_score >= 0.6:
            return f"Generally appropriate for {audience.value} audience"
        elif appropriateness_score >= 0.4:
            return f"Somewhat appropriate for {audience.value} audience"
        else:
            return f"Not well-suited for {audience.value} audience"
    
    def _generate_tone_recommendations(self, emotion_result: EmotionAnalysisResult,
                                     pacing_result: PacingAnalysisResult,
                                     clarity_result: ClarityAnalysisResult,
                                     tone_alignment: ToneAlignment,
                                     audience_appropriateness: AudienceAppropriatenesss,
                                     audience: AudienceType,
                                     goal: DeliveryGoal) -> List[str]:
        """Generate comprehensive tone delivery recommendations."""
        recommendations = []
        
        # Goal-specific recommendations
        if tone_alignment.emotion_alignment < 0.6:
            ideal_emotions = self.goal_emotion_mappings[goal]
            dominant_ideal = max(ideal_emotions, key=ideal_emotions.get)
            recommendations.append(f"For {goal.value} speeches, focus more on {dominant_ideal} "
                                 f"emotions. Adjust your emotional expression accordingly.")
        
        # Audience-specific recommendations
        if audience_appropriateness.complexity_appropriateness < 0.6:
            if audience in [AudienceType.YOUNG_ONES, AudienceType.UNINFORMED]:
                recommendations.append("Simplify your language and sentence structure for better "
                                     "comprehension by your target audience.")
            elif audience == AudienceType.EXPERTS:
                recommendations.append("You can use more technical language and complex concepts "
                                     "appropriate for expert audiences.")
        
        # Consistency recommendations
        if emotion_result.emotional_consistency < 0.6:
            recommendations.append("Maintain more consistent emotional tone throughout your speech. "
                                 "Sudden emotional shifts can confuse your audience.")
        
        # Integration recommendations
        if tone_alignment.overall_alignment > 0.7 and audience_appropriateness.overall_appropriateness > 0.7:
            recommendations.append("Excellent work! Your tone effectively serves both your goal "
                                 "and audience. Fine-tune the details for even better impact.")
        elif tone_alignment.overall_alignment > 0.7:
            recommendations.append("Your tone aligns well with your goal. Focus on better "
                                 "tailoring the content and style to your specific audience.")
        elif audience_appropriateness.overall_appropriateness > 0.7:
            recommendations.append("Your speech is well-suited for your audience. Work on "
                                 "better aligning your emotional tone with your delivery goal.")
        
        return recommendations
