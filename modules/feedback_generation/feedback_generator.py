"""
Feedback Generator Module

Generates comprehensive, personalized feedback by combining analysis results
with templates and AI-generated insights. Creates actionable recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .feedback_templates import FeedbackTemplates
from ..parameter_input.parameter_validator import SpeechParameters
from ..analysis.emotion_detection import EmotionAnalysisResult
from ..analysis.pacing_analysis import PacingAnalysisResult
from ..analysis.clarity_analysis import ClarityAnalysisResult
from ..analysis.tone_delivery import ToneDeliveryResult

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSection:
    """Represents a section of feedback."""
    title: str
    rating: int  # 1-5 Likert scale
    summary: str
    detailed_feedback: str
    recommendations: List[str]
    strengths: List[str]
    improvements: List[str]
    missed_words: Optional[List[str]] = None
    potentially_mispronounced: Optional[List[Tuple[str, str]]] = None


@dataclass
class ComprehensiveFeedback:
    """Complete feedback report."""
    overall_rating: int
    overall_summary: str
    emotion_section: FeedbackSection
    pacing_section: FeedbackSection
    clarity_section: FeedbackSection
    tone_section: FeedbackSection
    actionable_steps: List[str]
    next_practice_focus: List[str]
    generated_timestamp: str
    parameters_used: SpeechParameters


class FeedbackGenerator:
    """Generates personalized feedback from analysis results."""
    
    def __init__(self):
        """Initialize the feedback generator."""
        self.templates = FeedbackTemplates()
        
    def generate_comprehensive_feedback(self,
                                      emotion_result: EmotionAnalysisResult,
                                      pacing_result: PacingAnalysisResult,
                                      clarity_result: ClarityAnalysisResult,
                                      tone_result: ToneDeliveryResult,
                                      parameters: SpeechParameters) -> ComprehensiveFeedback:
        """
        Generate comprehensive feedback from all analysis results.
        
        Args:
            emotion_result: Results from emotion analysis
            pacing_result: Results from pacing analysis
            clarity_result: Results from clarity analysis
            tone_result: Results from tone delivery analysis
            parameters: Original speech parameters
            
        Returns:
            ComprehensiveFeedback object
        """
        logger.info("Generating comprehensive feedback...")
        
        # Generate individual sections
        emotion_section = self._generate_emotion_feedback(emotion_result, parameters)
        pacing_section = self._generate_pacing_feedback(pacing_result, parameters)
        clarity_section = self._generate_clarity_feedback(clarity_result, parameters)
        tone_section = self._generate_tone_feedback(tone_result, parameters)
        
        # Calculate overall rating
        overall_rating = self._calculate_overall_rating(
            emotion_result.likert_rating,
            pacing_result.likert_rating,
            clarity_result.likert_rating,
            tone_result.likert_rating
        )
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(
            overall_rating, tone_result, parameters
        )
        
        # Generate actionable steps
        actionable_steps = self._generate_actionable_steps(
            emotion_result, pacing_result, clarity_result, tone_result
        )
        
        # Generate next practice focus
        next_practice_focus = self._generate_practice_focus(
            emotion_result, pacing_result, clarity_result, tone_result, parameters
        )
        
        feedback = ComprehensiveFeedback(
            overall_rating=overall_rating,
            overall_summary=overall_summary,
            emotion_section=emotion_section,
            pacing_section=pacing_section,
            clarity_section=clarity_section,
            tone_section=tone_section,
            actionable_steps=actionable_steps,
            next_practice_focus=next_practice_focus,
            generated_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            parameters_used=parameters
        )
        
        logger.info(f"Comprehensive feedback generated. Overall rating: {overall_rating}/5")
        return feedback
    
    def _generate_emotion_feedback(self, emotion_result: EmotionAnalysisResult,
                                 parameters: SpeechParameters) -> FeedbackSection:
        """Generate emotion-specific feedback section."""
        
        # Get example sentence with strongest emotion
        example_sentence = ""
        if emotion_result.sentences:
            strongest_sentence = max(emotion_result.sentences, 
                                   key=lambda s: s.emotional_intensity)
            example_sentence = strongest_sentence.text[:100] + "..." if len(strongest_sentence.text) > 100 else strongest_sentence.text
        
        # Prepare template arguments
        template_args = {
            'dominant_emotion': emotion_result.overall_emotion_scores.get_dominant_emotion(),
            'confidence': emotion_result.overall_emotion_scores.get_confidence(),
            'goal': parameters.delivery_goal.value,
            'audience': parameters.audience_type.value,
            'example_sentence': example_sentence,
            'current_issue': self._identify_emotion_issue(emotion_result),
            'recommended_emotion': self._get_recommended_emotion(parameters.delivery_goal),
            'topic_area': "key concepts"
        }
        
        # Generate detailed feedback using templates
        detailed_feedback = self.templates.get_emotion_template(
            emotion_result.likert_rating, **template_args
        )
        
        # Extract strengths and improvements
        strengths = []
        improvements = []
        
        if emotion_result.likert_rating >= 4:
            strengths.append(f"Strong {template_args['dominant_emotion']} expression")
            strengths.append(f"Good emotional consistency ({emotion_result.emotional_consistency:.0%})")
        
        if emotion_result.emotional_consistency < 0.6:
            improvements.append("Maintain more consistent emotional tone")
        
        if emotion_result.likert_rating < 3:
            improvements.append(f"Develop stronger {template_args['recommended_emotion']} expression")
        
        return FeedbackSection(
            title="Emotional Delivery",
            rating=emotion_result.likert_rating,
            summary=f"Your emotional delivery shows {template_args['dominant_emotion']} with "
                   f"{emotion_result.overall_emotion_scores.get_confidence():.0%} confidence.",
            detailed_feedback=detailed_feedback,
            recommendations=self._get_emotion_recommendations(emotion_result, parameters),
            strengths=strengths,
            improvements=improvements
        )
    
    def _generate_pacing_feedback(self, pacing_result: PacingAnalysisResult,
                                parameters: SpeechParameters) -> FeedbackSection:
        """Generate pacing-specific feedback section."""
        
        template_args = {
            'audience': parameters.audience_type.value,
            'goal': parameters.delivery_goal.value
        }
        
        detailed_feedback = self.templates.get_pacing_template(
            pacing_result.overall_wpm,
            pacing_result.ideal_wpm_range,
            pacing_result.pacing_consistency,
            **template_args
        )
        
        # Extract strengths and improvements
        strengths = []
        improvements = []
        
        ideal_min, ideal_max = pacing_result.ideal_wpm_range
        if ideal_min <= pacing_result.overall_wpm <= ideal_max:
            strengths.append("Optimal speaking pace for your audience")
        
        if pacing_result.pacing_consistency >= 0.7:
            strengths.append("Good pacing consistency")
        else:
            improvements.append("Improve pacing consistency")
        
        if pacing_result.overall_wpm < ideal_min:
            improvements.append("Increase speaking pace")
        elif pacing_result.overall_wpm > ideal_max:
            improvements.append("Slow down speaking pace")
        
        return FeedbackSection(
            title="Pacing & Rhythm",
            rating=pacing_result.likert_rating,
            summary=f"Speaking at {pacing_result.overall_wpm:.0f} WPM with "
                   f"{pacing_result.pacing_consistency:.0%} consistency.",
            detailed_feedback=detailed_feedback,
            recommendations=pacing_result.recommendations,
            strengths=strengths,
            improvements=improvements
        )
    
    def _generate_clarity_feedback(self, clarity_result: ClarityAnalysisResult,
                                 parameters: SpeechParameters) -> FeedbackSection:
        """Generate clarity-specific feedback section."""
        
        template_args = {
            'audience': parameters.audience_type.value,
            'diversity': clarity_result.vocabulary_analysis.vocabulary_diversity,
            'avg_length': clarity_result.structure_analysis.get('avg_sentence_length', 15),
            'complex_words': clarity_result.vocabulary_analysis.complex_words,
            'filler_count': len(clarity_result.vocabulary_analysis.filler_words),
            'ambiguity_issues': len([s for s in clarity_result.sentences if s.ambiguity_flags]),
            'difficult_words': ', '.join(clarity_result.pronunciation_flags[:3]) if clarity_result.pronunciation_flags else "none identified"
        }
        
        detailed_feedback = self.templates.get_clarity_template(
            clarity_result.clarity_score, **template_args
        )
        
        # Add pronunciation feedback if needed
        if clarity_result.pronunciation_flags:
            pronunciation_feedback = self.templates.clarity_templates['pronunciation']
            detailed_feedback += " " + pronunciation_feedback['challenges'].format(**template_args)
            detailed_feedback += " " + pronunciation_feedback['practice']
        
        # Add transcription information if available
        transcription_feedback = ""
        if clarity_result.transcription_info:
            transcription_feedback = self._generate_transcription_feedback(clarity_result.transcription_info)
            detailed_feedback += "\n\n" + transcription_feedback
        
        # Extract strengths and improvements based on transcription analysis
        strengths = []
        improvements = []
        
        # Prioritize transcription-based feedback if available
        if clarity_result.transcription_info and clarity_result.transcription_info.get('comparison_available'):
            accuracy_metrics = clarity_result.transcription_info.get('accuracy_metrics', {})
            word_accuracy = accuracy_metrics.get('word_accuracy', 0)
            
            if clarity_result.clarity_score >= 0.8:
                strengths.append("Excellent speech clarity - AI transcription matched your text very well")
                strengths.append("Clear pronunciation and articulation")
            elif clarity_result.clarity_score >= 0.6:
                strengths.append("Good speech clarity - Most words transcribed accurately")
                strengths.append("Generally clear pronunciation")
            elif clarity_result.clarity_score >= 0.4:
                strengths.append("Moderate clarity with room for improvement")
            
            if word_accuracy >= 0.8:
                strengths.append("High word recognition accuracy")
            
            if clarity_result.clarity_score < 0.6:
                improvements.append("Improve pronunciation clarity for better comprehension")
                improvements.append("Speak more slowly and enunciate words clearly")
            
            if word_accuracy < 0.5:
                improvements.append("Focus on clearer articulation of individual words")
                
        else:
            # Fallback to traditional analysis
            if clarity_result.clarity_score >= 0.7:
                strengths.append("High overall clarity")
                strengths.append("Appropriate vocabulary complexity")
            
            if clarity_result.vocabulary_analysis.vocabulary_diversity >= 0.6:
                strengths.append("Good vocabulary diversity")
            
            if clarity_result.clarity_score < 0.5:
                improvements.append("Simplify sentence structure")
                improvements.append("Use clearer vocabulary")
        
        # Common improvements regardless of analysis method
        if len(clarity_result.vocabulary_analysis.filler_words) > clarity_result.vocabulary_analysis.total_words * 0.05:
            improvements.append("Reduce filler words")
        
        # Add specific feedback for missed and mispronounced words
        if clarity_result.missed_words:
            improvements.append(f"Practice pronunciation of missed words: {', '.join(clarity_result.missed_words[:5])}")
            if len(clarity_result.missed_words) > 5:
                improvements.append(f"(+{len(clarity_result.missed_words) - 5} more words to review)")
        
        if clarity_result.potentially_mispronounced:
            mispronounced_list = [f"'{pair[0]}' (heard as '{pair[1]}')" for pair in clarity_result.potentially_mispronounced[:3]]
            improvements.append(f"Clarify pronunciation: {', '.join(mispronounced_list)}")
            if len(clarity_result.potentially_mispronounced) > 3:
                improvements.append(f"(+{len(clarity_result.potentially_mispronounced) - 3} more pronunciation issues)")
        
        # Create summary based on analysis method
        if clarity_result.transcription_info and clarity_result.transcription_info.get('comparison_available'):
            accuracy_metrics = clarity_result.transcription_info.get('accuracy_metrics', {})
            word_accuracy = accuracy_metrics.get('word_accuracy', 0)
            summary = f"Speech clarity score: {clarity_result.clarity_score:.1f}/1.0 based on AI transcription analysis (Word accuracy: {word_accuracy:.0%})"
        else:
            summary = f"Clarity score of {clarity_result.clarity_score:.1f} with {clarity_result.vocabulary_analysis.vocabulary_diversity:.0%} vocabulary diversity."
        
        return FeedbackSection(
            title="Clarity & Comprehension",
            rating=clarity_result.likert_rating,
            summary=summary,
            detailed_feedback=detailed_feedback,
            recommendations=clarity_result.recommendations,
            strengths=strengths,
            improvements=improvements,
            missed_words=clarity_result.missed_words,
            potentially_mispronounced=clarity_result.potentially_mispronounced
        )
    
    def _generate_tone_feedback(self, tone_result: ToneDeliveryResult,
                              parameters: SpeechParameters) -> FeedbackSection:
        """Generate tone delivery feedback section."""
        
        template_args = {
            'goal': parameters.delivery_goal.value,
            'audience': parameters.audience_type.value,
            'appropriateness': tone_result.audience_appropriateness.overall_appropriateness,
            'improvement_area': tone_result.improvement_areas[0] if tone_result.improvement_areas else "fine-tuning details",
            'current_emotion': tone_result.tone_alignment.emotion_alignment,
            'ideal_emotion': "balanced emotional expression",
            'specific_aspects': "emotional consistency and audience alignment"
        }
        
        detailed_feedback = self.templates.get_tone_template(
            tone_result.tone_alignment.overall_alignment, **template_args
        )
        
        return FeedbackSection(
            title="Tone Delivery",
            rating=tone_result.likert_rating,
            summary=f"Goal achievement score: {tone_result.goal_achievement_score:.1f}, "
                   f"Audience appropriateness: {tone_result.audience_appropriateness.overall_appropriateness:.1f}",
            detailed_feedback=detailed_feedback,
            recommendations=tone_result.recommendations,
            strengths=tone_result.strengths,
            improvements=tone_result.improvement_areas
        )
    
    def _calculate_overall_rating(self, emotion_rating: int, pacing_rating: int,
                                clarity_rating: int, tone_rating: int) -> int:
        """Calculate overall rating from individual ratings."""
        # Weighted average with tone being most important
        weights = {'emotion': 0.25, 'pacing': 0.25, 'clarity': 0.25, 'tone': 0.25}
        
        weighted_sum = (emotion_rating * weights['emotion'] + 
                       pacing_rating * weights['pacing'] + 
                       clarity_rating * weights['clarity'] + 
                       tone_rating * weights['tone'])
        
        return max(1, min(5, round(weighted_sum)))
    
    def _generate_overall_summary(self, overall_rating: int, tone_result: ToneDeliveryResult,
                                parameters: SpeechParameters) -> str:
        """Generate overall performance summary."""
        
        if overall_rating >= 4:
            summary = f"Excellent speech analysis results! Your {parameters.delivery_goal.value} presentation " \
                     f"is well-suited for {parameters.audience_type.value} audience."
        elif overall_rating >= 3:
            summary = f"Good foundation for your {parameters.delivery_goal.value} speech. " \
                     f"With focused improvements, you'll achieve excellent results with {parameters.audience_type.value} audience."
        else:
            summary = f"Your {parameters.delivery_goal.value} speech needs development to effectively " \
                     f"reach {parameters.audience_type.value} audience. Focus on the priority areas identified below."
        
        # Add specific achievement note
        if tone_result.goal_achievement_score >= 0.7:
            summary += " Your tone effectively supports your communication objectives."
        else:
            summary += " Consider adjusting your approach to better align with your communication goals."
        
        return summary
    
    def _generate_actionable_steps(self, emotion_result: EmotionAnalysisResult,
                                 pacing_result: PacingAnalysisResult,
                                 clarity_result: ClarityAnalysisResult,
                                 tone_result: ToneDeliveryResult) -> List[str]:
        """Generate specific, actionable improvement steps."""
        steps = []
        
        # Prioritize by lowest ratings
        ratings = [
            (emotion_result.likert_rating, "emotion"),
            (pacing_result.likert_rating, "pacing"),
            (clarity_result.likert_rating, "clarity"),
            (tone_result.likert_rating, "tone")
        ]
        
        # Sort by rating (lowest first)
        ratings.sort(key=lambda x: x[0])
        
        for rating, area in ratings[:2]:  # Top 2 priority areas
            if area == "emotion" and rating < 4:
                steps.append("Practice emotional expression by reading dramatic texts aloud daily")
                steps.append("Record yourself speaking to develop awareness of your emotional tone")
            
            elif area == "pacing" and rating < 4:
                if pacing_result.overall_wpm < pacing_result.ideal_wpm_range[0]:
                    steps.append("Practice speaking with more energy and purpose to increase pace")
                elif pacing_result.overall_wpm > pacing_result.ideal_wpm_range[1]:
                    steps.append("Use intentional pauses and slower delivery on key points")
                steps.append("Mark your script with pacing cues during practice")
            
            elif area == "clarity" and rating < 4:
                steps.append("Simplify complex sentences and replace difficult vocabulary")
                if len(clarity_result.vocabulary_analysis.filler_words) > 10:
                    steps.append("Practice eliminating filler words through conscious awareness")
            
            elif area == "tone" and rating < 4:
                steps.append("Align emotional expression more closely with your speech goal")
                steps.append("Practice adapting your style for your specific audience")
        
        # Add general steps if all ratings are good
        if all(rating >= 4 for rating, _ in ratings):
            steps.append("Fine-tune timing and emphasis on key points")
            steps.append("Practice with live audience for feedback")
        
        return steps[:5]  # Limit to 5 most important steps
    
    def _generate_practice_focus(self, emotion_result: EmotionAnalysisResult,
                               pacing_result: PacingAnalysisResult,
                               clarity_result: ClarityAnalysisResult,
                               tone_result: ToneDeliveryResult,
                               parameters: SpeechParameters) -> List[str]:
        """Generate focused practice recommendations."""
        focus_areas = []
        
        # Identify the most critical area
        lowest_rating = min(
            emotion_result.likert_rating,
            pacing_result.likert_rating,
            clarity_result.likert_rating,
            tone_result.likert_rating
        )
        
        if emotion_result.likert_rating == lowest_rating:
            focus_areas.append(f"Emotional expression for {parameters.delivery_goal.value} speeches")
        
        if pacing_result.likert_rating == lowest_rating:
            ideal_min, ideal_max = pacing_result.ideal_wpm_range
            focus_areas.append(f"Speaking pace: aim for {ideal_min}-{ideal_max} words per minute")
        
        if clarity_result.likert_rating == lowest_rating:
            focus_areas.append("Sentence structure and vocabulary simplification")
        
        if tone_result.likert_rating == lowest_rating:
            focus_areas.append(f"Tone alignment for {parameters.audience_type.value} audience")
        
        # Add consistency focus if needed
        if (emotion_result.emotional_consistency < 0.6 or 
            pacing_result.pacing_consistency < 0.6):
            focus_areas.append("Maintaining consistent delivery throughout speech")
        
        return focus_areas[:3]  # Top 3 focus areas
    
    def _identify_emotion_issue(self, emotion_result: EmotionAnalysisResult) -> str:
        """Identify the main emotion issue."""
        if emotion_result.overall_emotion_scores.neutral > 0.7:
            return "neutral"
        elif emotion_result.emotional_consistency < 0.5:
            return "inconsistent"
        else:
            return "unclear"
    
    def _get_recommended_emotion(self, goal: Any) -> str:
        """Get recommended emotion for delivery goal."""
        emotion_map = {
            'persuade': 'confidence and passion',
            'inform': 'clarity and engagement',
            'entertain': 'joy and enthusiasm',
            'motivate': 'inspiration and energy',
            'implore': 'sincerity and urgency',
            'deliver': 'professionalism and clarity',
            'commemorate': 'respect and warmth'
        }
        return emotion_map.get(goal.value, 'appropriate emotional expression')
    
    def _get_emotion_recommendations(self, emotion_result: EmotionAnalysisResult,
                                   parameters: SpeechParameters) -> List[str]:
        """Get specific emotion recommendations."""
        recommendations = []
        
        recommended_emotion = self._get_recommended_emotion(parameters.delivery_goal)
        recommendations.append(f"Focus on expressing {recommended_emotion} for {parameters.delivery_goal.value} goals")
        
        if emotion_result.emotional_consistency < 0.6:
            recommendations.append("Practice maintaining consistent emotional tone throughout")
        
        if len(emotion_result.emotion_transitions) > len(emotion_result.sentences) * 0.5:
            recommendations.append("Reduce frequent emotional shifts for better audience connection")
        
        return recommendations
    
    def format_feedback_report(self, feedback: ComprehensiveFeedback) -> str:
        """Format feedback into a readable report."""
        
        report = f"""
{'='*80}
🎤 SPEECH ANALYSIS REPORT
{'='*80}

Generated: {feedback.generated_timestamp}
Speech Goal: {feedback.parameters_used.delivery_goal.value.title()}
Target Audience: {feedback.parameters_used.audience_type.value.title()}
Word Count: {len(feedback.parameters_used.speech_text.split())} words

{self.templates.create_summary_template(
    feedback.overall_rating,
    feedback.emotion_section.strengths + feedback.pacing_section.strengths + 
    feedback.clarity_section.strengths + feedback.tone_section.strengths,
    feedback.actionable_steps
)}

{self.templates.create_section_header("Emotional Delivery", feedback.emotion_section.rating)}
{feedback.emotion_section.detailed_feedback}

Strengths:
{chr(10).join(f"• {strength}" for strength in feedback.emotion_section.strengths)}

{self.templates.create_section_header("Pacing & Rhythm", feedback.pacing_section.rating)}
{feedback.pacing_section.detailed_feedback}

Strengths:
{chr(10).join(f"• {strength}" for strength in feedback.pacing_section.strengths)}

{self.templates.create_section_header("Clarity & Comprehension", feedback.clarity_section.rating)}
{feedback.clarity_section.detailed_feedback}

Strengths:
{chr(10).join(f"• {strength}" for strength in feedback.clarity_section.strengths)}

{self.templates.create_section_header("Tone Delivery", feedback.tone_section.rating)}
{feedback.tone_section.detailed_feedback}

Strengths:
{chr(10).join(f"• {strength}" for strength in feedback.tone_section.strengths)}

{'='*80}
🎯 ACTIONABLE IMPROVEMENT PLAN
{'='*80}

Priority Actions:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(feedback.actionable_steps))}

Next Practice Focus:
{chr(10).join(f"• {focus}" for focus in feedback.next_practice_focus)}

{'='*80}
Thank you for using Speech Analyzer! 
Practice these recommendations and re-analyze to track your progress.
{'='*80}
"""
        
        return report
    
    def _generate_transcription_feedback(self, transcription_info: dict) -> str:
        """
        Generate feedback about AI transcription accuracy and quality.
        
        Args:
            transcription_info: Dictionary containing transcription analysis
            
        Returns:
            Formatted transcription feedback string
        """
        feedback_lines = []
        feedback_lines.append("🎙️ AI TRANSCRIPTION ANALYSIS")
        feedback_lines.append("─" * 40)
        
        # Show transcribed text
        transcribed_text = transcription_info.get('transcribed_text', '').strip()
        if transcribed_text:
            # Truncate if too long for display
            display_text = transcribed_text if len(transcribed_text) <= 200 else transcribed_text[:200] + "..."
            feedback_lines.append(f"📝 Transcribed Text: \"{display_text}\"")
        else:
            feedback_lines.append("📝 No transcription available")
            return "\n".join(feedback_lines)
        
        # Basic transcription stats
        word_count = transcription_info.get('word_count', 0)
        confidence = transcription_info.get('confidence_score')
        feedback_lines.append(f"📊 Word Count: {word_count}")
        
        if confidence is not None:
            confidence_pct = confidence * 100 if confidence <= 1.0 else confidence
            confidence_desc = self._get_confidence_description(confidence)
            feedback_lines.append(f"🎯 Confidence: {confidence_pct:.1f}% ({confidence_desc})")
        
        # Segments information
        if transcription_info.get('has_segments', False):
            segment_count = transcription_info.get('segment_count', 0)
            feedback_lines.append(f"⏱️ Speech Segments: {segment_count}")
        
        # Accuracy comparison if available
        accuracy_metrics = transcription_info.get('accuracy_metrics')
        if accuracy_metrics and transcription_info.get('comparison_available', False):
            feedback_lines.append("\n🔍 TRANSCRIPTION ACCURACY:")
            
            similarity = accuracy_metrics.get('similarity', 0)
            word_accuracy = accuracy_metrics.get('word_accuracy', 0)
            length_ratio = accuracy_metrics.get('length_ratio', 0)
            
            feedback_lines.append(f"  • Text Similarity: {similarity:.1%}")
            feedback_lines.append(f"  • Word Match Rate: {word_accuracy:.1%}")
            feedback_lines.append(f"  • Length Ratio: {length_ratio:.2f}")
            
            # Provide accuracy assessment
            if similarity >= 0.8:
                feedback_lines.append("  ✅ Excellent transcription accuracy")
            elif similarity >= 0.6:
                feedback_lines.append("  ✅ Good transcription accuracy")
            elif similarity >= 0.4:
                feedback_lines.append("  ⚠️ Fair transcription accuracy")
            else:
                feedback_lines.append("  ❌ Low transcription accuracy - consider clearer speech")
        
        else:
            feedback_lines.append("\n💡 TIP: Provide your original text for accuracy comparison!")
        
        return "\n".join(feedback_lines)
    
    def _get_confidence_description(self, confidence: float) -> str:
        """Get description for transcription confidence score."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Good"
        elif confidence >= 0.6:
            return "Fair"
        else:
            return "Low"
    
    def _generate_transcription_feedback(self, transcription_info: dict) -> str:
        """Generate feedback about AI transcription quality."""
        if not transcription_info:
            return ""
        
        feedback_parts = []
        feedback_parts.append("🤖 AI TRANSCRIPTION ANALYSIS:")
        feedback_parts.append("-" * 40)
        
        # Show transcribed text
        transcribed_text = transcription_info.get('transcribed_text', '')
        if transcribed_text:
            # Truncate if too long for display
            display_text = transcribed_text[:200] + "..." if len(transcribed_text) > 200 else transcribed_text
            feedback_parts.append(f"📝 Transcribed Text: \"{display_text}\"")
        
        # Show basic metrics
        word_count = transcription_info.get('word_count', 0)
        feedback_parts.append(f"📊 Word Count: {word_count}")
        
        # Show confidence if available
        confidence = transcription_info.get('confidence_score')
        if confidence is not None:
            confidence_level = self._get_confidence_description(confidence)
            feedback_parts.append(f"🎯 Transcription Confidence: {confidence:.2f} ({confidence_level})")
        
        # Show segment information
        if transcription_info.get('has_segments'):
            segment_count = transcription_info.get('segment_count', 0)
            feedback_parts.append(f"🔍 Audio Segments Processed: {segment_count}")
        
        # Show accuracy comparison if available
        accuracy_metrics = transcription_info.get('accuracy_metrics')
        if accuracy_metrics and transcription_info.get('comparison_available'):
            similarity = accuracy_metrics.get('similarity', 0)
            word_accuracy = accuracy_metrics.get('word_accuracy', 0)
            length_ratio = accuracy_metrics.get('length_ratio', 0)
            
            feedback_parts.append("\n📈 SPEECH CLARITY ASSESSMENT (Primary Analysis):")
            feedback_parts.append(f"   • Word Recognition Rate: {word_accuracy:.1%}")
            feedback_parts.append(f"   • Overall Text Similarity: {similarity:.1%}")
            feedback_parts.append(f"   • Length Consistency: {length_ratio:.2f}")
            
            # Provide detailed interpretation
            if word_accuracy >= 0.9:
                feedback_parts.append("   🏆 EXCELLENT: Nearly perfect word recognition!")
                feedback_parts.append("   → Your speech is extremely clear and easy to understand.")
            elif word_accuracy >= 0.8:
                feedback_parts.append("   ✅ VERY GOOD: High word recognition accuracy.")
                feedback_parts.append("   → Your speech clarity is above average.")
            elif word_accuracy >= 0.6:
                feedback_parts.append("   ✓ GOOD: Solid word recognition with minor issues.")
                feedback_parts.append("   → Consider speaking slightly slower or more clearly.")
            elif word_accuracy >= 0.4:
                feedback_parts.append("   ⚠️ FAIR: Moderate recognition - clarity needs improvement.")
                feedback_parts.append("   → Focus on enunciation and pacing.")
            else:
                feedback_parts.append("   ❌ NEEDS WORK: Low recognition rate.")
                feedback_parts.append("   → Significant clarity improvement needed.")
            
            # Add detailed word-level analysis
            missed_words = transcription_info.get('missed_words', [])
            mispronounced = transcription_info.get('potentially_mispronounced', [])
            
            if missed_words or mispronounced:
                feedback_parts.append("\n🔍 DETAILED WORD ANALYSIS:")
                
                if missed_words:
                    feedback_parts.append(f"   📵 Missed Words ({len(missed_words)}): {', '.join(missed_words[:10])}")
                    if len(missed_words) > 10:
                        feedback_parts.append(f"   ... and {len(missed_words) - 10} more")
                    feedback_parts.append("   → These words were not captured in the transcription")
                
                if mispronounced:
                    feedback_parts.append(f"   🗣️ Potential Mispronunciations ({len(mispronounced)}):")
                    for intended, heard in mispronounced[:5]:
                        feedback_parts.append(f"      • '{intended}' → heard as '{heard}'")
                    if len(mispronounced) > 5:
                        feedback_parts.append(f"      ... and {len(mispronounced) - 5} more")
                    feedback_parts.append("   → Consider practicing these specific words")
            
        else:
            feedback_parts.append("\n💡 This transcription was generated from your audio input.")
            feedback_parts.append("   Review the transcribed text above to verify accuracy.")
            feedback_parts.append("   📊 For detailed clarity analysis, provide both audio and text.")
        
        return "\n".join(feedback_parts)
