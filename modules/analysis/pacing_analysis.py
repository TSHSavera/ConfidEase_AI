"""
Pacing Analysis Sub-Module

Analyzes speech pacing by calculating words per minute (WPM) and comparing
against ideal ranges based on delivery goal and target audience.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re
from ..parameter_input.parameter_validator import AudienceType, DeliveryGoal

logger = logging.getLogger(__name__)


@dataclass
class PacingSegment:
    """Represents pacing information for a text segment."""
    text: str
    start_time: Optional[float]
    end_time: Optional[float]
    word_count: int
    duration: Optional[float]
    wpm: Optional[float]


@dataclass
class PacingAnalysisResult:
    """Complete pacing analysis result."""
    overall_wpm: float
    ideal_wpm_range: Tuple[int, int]
    segments: List[PacingSegment]
    pacing_consistency: float  # Variance in WPM across segments
    speech_duration: float
    pause_analysis: Dict[str, float]
    likert_rating: int  # 1-5 scale
    recommendations: List[str]


class PacingAnalyzer:
    """Analyzes speech pacing and rhythm."""
    
    def __init__(self):
        """Initialize the pacing analyzer."""
        self.silence_threshold = 0.5  # Minimum pause duration to consider
        
    def analyze_text_pacing(self, 
                           text: str, 
                           audience: AudienceType, 
                           goal: DeliveryGoal,
                           audio_duration: Optional[float] = None) -> PacingAnalysisResult:
        """
        Analyze pacing from text and optional audio duration.
        
        Args:
            text: Speech text to analyze
            audience: Target audience type
            goal: Delivery goal
            audio_duration: Optional duration of audio in seconds
            
        Returns:
            PacingAnalysisResult object
        """
        logger.info("Starting pacing analysis...")
        
        # Calculate word count and estimated WPM
        word_count = self._count_words(text)
        
        if audio_duration and audio_duration > 0:
            overall_wpm = (word_count / audio_duration) * 60
            speech_duration = audio_duration
        else:
            # Estimate duration based on average speaking rate
            estimated_duration = self._estimate_duration(text)
            overall_wpm = (word_count / estimated_duration) * 60 if estimated_duration > 0 else 0
            speech_duration = estimated_duration
        
        # Get ideal range for this audience and goal
        ideal_range = self._get_ideal_wpm_range(audience, goal)
        
        # Analyze text segments (sentences/paragraphs)
        segments = self._analyze_text_segments(text, speech_duration)
        
        # Calculate consistency
        consistency = self._calculate_pacing_consistency(segments)
        
        # Basic pause analysis (estimated from punctuation when no audio)
        pause_analysis = self._analyze_pauses_from_text(text)
        
        # Calculate Likert rating
        likert_rating = self._calculate_pacing_rating(overall_wpm, ideal_range, consistency)
        
        # Generate recommendations
        recommendations = self._generate_pacing_recommendations(
            overall_wpm, ideal_range, consistency, audience, goal
        )
        
        result = PacingAnalysisResult(
            overall_wpm=overall_wpm,
            ideal_wpm_range=ideal_range,
            segments=segments,
            pacing_consistency=consistency,
            speech_duration=speech_duration,
            pause_analysis=pause_analysis,
            likert_rating=likert_rating,
            recommendations=recommendations
        )
        
        logger.info(f"Pacing analysis completed. WPM: {overall_wpm:.1f}, "
                   f"Ideal range: {ideal_range[0]}-{ideal_range[1]}, Rating: {likert_rating}/5")
        
        return result
    
    def analyze_audio_pacing(self, 
                           transcription_segments: List,
                           text: str,
                           audience: AudienceType, 
                           goal: DeliveryGoal) -> PacingAnalysisResult:
        """
        Analyze pacing from audio transcription with timestamps.
        
        Args:
            transcription_segments: List of transcription segments with timestamps
            text: Full speech text
            audience: Target audience type
            goal: Delivery goal
            
        Returns:
            PacingAnalysisResult object
        """
        logger.info("Starting audio-based pacing analysis...")
        
        if not transcription_segments:
            # Fall back to text-only analysis
            return self.analyze_text_pacing(text, audience, goal)
        
        # Calculate overall metrics
        total_duration = transcription_segments[-1].end_time if transcription_segments else 0
        total_words = sum(len(seg.text.split()) for seg in transcription_segments)
        overall_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
        
        # Get ideal range
        ideal_range = self._get_ideal_wpm_range(audience, goal)
        
        # Create pacing segments from transcription (Check transcription logs for details of how segments are structured)
        pacing_segments = []
        for seg in transcription_segments:
            word_count = len(seg.text.split())
            duration = seg.end_time - seg.start_time
            wpm = (word_count / duration) * 60 if duration > 0 else 0
            
            pacing_segment = PacingSegment(
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                word_count=word_count,
                duration=duration,
                wpm=wpm
            )
            pacing_segments.append(pacing_segment)
        
        # Calculate consistency
        consistency = self._calculate_pacing_consistency(pacing_segments)
        
        # Analyze pauses between segments
        pause_analysis = self._analyze_audio_pauses(transcription_segments)
        
        # Calculate Likert rating
        likert_rating = self._calculate_pacing_rating(overall_wpm, ideal_range, consistency)
        
        # Generate recommendations
        recommendations = self._generate_pacing_recommendations(
            overall_wpm, ideal_range, consistency, audience, goal
        )
        
        result = PacingAnalysisResult(
            overall_wpm=overall_wpm,
            ideal_wpm_range=ideal_range,
            segments=pacing_segments,
            pacing_consistency=consistency,
            speech_duration=total_duration,
            pause_analysis=pause_analysis,
            likert_rating=likert_rating,
            recommendations=recommendations
        )
        
        logger.info(f"Audio pacing analysis completed. WPM: {overall_wpm:.1f}, "
                   f"Ideal range: {ideal_range[0]}-{ideal_range[1]}, Rating: {likert_rating}/5")
        
        return result
    
    def _count_words(self, text: str) -> int:
        """Count words in text, excluding punctuation."""
        # Remove extra whitespace and split
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    def _estimate_duration(self, text: str, base_wpm: float = 160) -> float:
        """
        Estimate speech duration based on text length.
        
        Args:
            text: Text to analyze
            base_wpm: Base words per minute for estimation
            
        Returns:
            Estimated duration in seconds
        """
        word_count = self._count_words(text)
        
        # Adjust for punctuation (pauses)
        pause_count = len(re.findall(r'[.!?;,]', text))
        estimated_pause_time = pause_count * 0.3  # 0.3 seconds per punctuation mark
        
        # Calculate base speaking time
        speaking_time = (word_count / base_wpm) * 60
        
        return speaking_time + estimated_pause_time
    
    def _get_ideal_wpm_range(self, audience: AudienceType, goal: DeliveryGoal) -> Tuple[int, int]:
        """
        Get ideal WPM range based on audience and goal.
        
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
        elif audience == AudienceType.APATHETIC:
            # Vary pace to maintain interest
            return (base_min - 5, base_max + 10)
        
        return (base_min, base_max)
    
    def _analyze_text_segments(self, text: str, total_duration: float) -> List[PacingSegment]:
        """
        Analyze pacing of text segments (sentences).
        
        Args:
            text: Text to analyze
            total_duration: Total estimated duration
            
        Returns:
            List of PacingSegment objects
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        segments = []
        total_words = self._count_words(text)
        current_time = 0.0
        
        for sentence in sentences:
            word_count = self._count_words(sentence)
            
            # Estimate duration proportional to word count
            if total_words > 0:
                segment_duration = (word_count / total_words) * total_duration
            else:
                segment_duration = 0.0
            
            wpm = (word_count / segment_duration) * 60 if segment_duration > 0 else 0
            
            segment = PacingSegment(
                text=sentence,
                start_time=current_time,
                end_time=current_time + segment_duration,
                word_count=word_count,
                duration=segment_duration,
                wpm=wpm
            )
            segments.append(segment)
            current_time += segment_duration
        
        return segments
    
    def _calculate_pacing_consistency(self, segments: List[PacingSegment]) -> float:
        """
        Calculate pacing consistency across segments.
        
        Args:
            segments: List of pacing segments
            
        Returns:
            Consistency score (0 to 1, where 1 is most consistent)
        """
        if len(segments) <= 1:
            return 1.0
        
        # Get WPM values for segments with valid WPM
        wpm_values = [seg.wpm for seg in segments if seg.wpm is not None and seg.wpm > 0]
        
        if len(wpm_values) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (std dev / mean)
        mean_wpm = np.mean(wpm_values)
        std_wpm = np.std(wpm_values)
        
        if mean_wpm == 0:
            return 1.0
        
        cv = std_wpm / mean_wpm
        
        # Convert to consistency score (lower variation = higher consistency)
        # CV of 0.2 or less is considered highly consistent
        consistency = max(0.0, 1.0 - (cv / 0.5))  # Scale so CV=0.5 gives consistency=0
        
        return min(1.0, consistency)
    
    def _analyze_pauses_from_text(self, text: str) -> Dict[str, float]:
        """
        Analyze pauses from text punctuation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with pause statistics
        """
        # Count different types of punctuation
        periods = len(re.findall(r'\.', text))
        commas = len(re.findall(r',', text))
        semicolons = len(re.findall(r';', text))
        colons = len(re.findall(r':', text))
        exclamations = len(re.findall(r'!', text))
        questions = len(re.findall(r'\?', text))
        
        # Estimate pause durations
        short_pauses = commas + semicolons + colons  # ~0.3 seconds each
        medium_pauses = periods  # ~0.5 seconds each
        long_pauses = exclamations + questions  # ~0.7 seconds each
        
        total_pause_time = (short_pauses * 0.3 + 
                           medium_pauses * 0.5 + 
                           long_pauses * 0.7)
        
        return {
            'total_pause_time': total_pause_time,
            'short_pauses': short_pauses,
            'medium_pauses': medium_pauses,
            'long_pauses': long_pauses,
            'avg_pause_duration': total_pause_time / max(1, short_pauses + medium_pauses + long_pauses)
        }
    
    def _analyze_audio_pauses(self, transcription_segments: List) -> Dict[str, float]:
        """
        Analyze pauses from audio transcription timestamps.
        
        Args:
            transcription_segments: List of transcription segments
            
        Returns:
            Dictionary with pause statistics
        """
        if len(transcription_segments) <= 1:
            return {'total_pause_time': 0, 'pause_count': 0, 'avg_pause_duration': 0}
        
        pauses = []
        for i in range(len(transcription_segments) - 1):
            pause_duration = (transcription_segments[i + 1].start_time - 
                            transcription_segments[i].end_time)
            if pause_duration > self.silence_threshold:
                pauses.append(pause_duration)
        
        total_pause_time = sum(pauses)
        pause_count = len(pauses)
        avg_pause_duration = total_pause_time / max(1, pause_count)
        
        return {
            'total_pause_time': total_pause_time,
            'pause_count': pause_count,
            'avg_pause_duration': avg_pause_duration,
            'longest_pause': max(pauses) if pauses else 0,
            'shortest_pause': min(pauses) if pauses else 0
        }
    
    def _calculate_pacing_rating(self, wpm: float, ideal_range: Tuple[int, int], 
                                consistency: float) -> int:
        """
        Calculate Likert scale rating (1-5) for pacing.
        
        Args:
            wpm: Actual words per minute
            ideal_range: Ideal WPM range (min, max)
            consistency: Pacing consistency score
            
        Returns:
            Likert scale rating (1-5)
        """
        min_wpm, max_wpm = ideal_range
        
        # Base score from WPM appropriateness
        if min_wpm <= wpm <= max_wpm:
            base_score = 5  # Perfect range
        else:
            # Calculate how far outside the ideal range
            if wpm < min_wpm:
                deviation = (min_wpm - wpm) / min_wpm
            else:
                deviation = (wpm - max_wpm) / max_wpm
            
            # Score decreases with deviation
            if deviation <= 0.1:
                base_score = 4
            elif deviation <= 0.25:
                base_score = 3
            elif deviation <= 0.5:
                base_score = 2
            else:
                base_score = 1
        
        # Adjust based on consistency
        if consistency >= 0.8:
            consistency_bonus = 0
        elif consistency >= 0.6:
            consistency_bonus = -0.5
        else:
            consistency_bonus = -1
        
        final_score = base_score + consistency_bonus
        return max(1, min(5, round(final_score)))
    
    def _generate_pacing_recommendations(self, wpm: float, ideal_range: Tuple[int, int],
                                       consistency: float, audience: AudienceType,
                                       goal: DeliveryGoal) -> List[str]:
        """
        Generate pacing improvement recommendations.
        
        Args:
            wpm: Actual words per minute
            ideal_range: Ideal WPM range
            consistency: Pacing consistency score
            audience: Target audience
            goal: Delivery goal
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        min_wpm, max_wpm = ideal_range
        
        # WPM recommendations
        if wpm < min_wpm:
            recommendations.append(f"Increase speaking pace to {min_wpm}-{max_wpm} WPM. "
                                 f"Current pace ({wpm:.0f} WPM) may be too slow for your audience.")
        elif wpm > max_wpm:
            recommendations.append(f"Slow down speaking pace to {min_wpm}-{max_wpm} WPM. "
                                 f"Current pace ({wpm:.0f} WPM) may be too fast for your audience.")
        else:
            recommendations.append(f"Excellent pacing! Your {wpm:.0f} WPM is ideal for "
                                 f"{audience.value} audience with {goal.value} goal.")
        
        # Consistency recommendations
        if consistency < 0.6:
            recommendations.append("Work on maintaining more consistent pacing throughout your speech. "
                                 "Practice with a metronome or pacing exercises.")
        elif consistency < 0.8:
            recommendations.append("Your pacing is fairly consistent, but could be improved with "
                                 "more practice maintaining steady rhythm.")
        
        # Audience-specific recommendations
        if audience == AudienceType.YOUNG_ONES:
            recommendations.append("For young audiences, use varied pacing to maintain attention. "
                                 "Slow down for important points, speed up for exciting parts.")
        elif audience == AudienceType.EXPERTS:
            recommendations.append("Expert audiences can handle faster pace, but ensure clarity "
                                 "is not sacrificed for speed.")
        elif audience == AudienceType.HOSTILE:
            recommendations.append("For hostile audiences, speak slowly and deliberately to "
                                 "appear more credible and allow time for processing.")
        
        # Goal-specific recommendations
        if goal == DeliveryGoal.PERSUADE:
            recommendations.append("For persuasive speaking, vary your pace: slow for important "
                                 "points, normal for explanation, faster for building excitement.")
        elif goal == DeliveryGoal.INFORM:
            recommendations.append("For informative speaking, maintain steady pace with "
                                 "strategic pauses after key concepts.")
        
        return recommendations
