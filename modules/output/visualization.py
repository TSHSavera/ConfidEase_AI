"""
Visualization Generator Module

Creates visual representations of speech analysis results including
charts, graphs, and visual feedback displays.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from ..feedback_generation.feedback_generator import ComprehensiveFeedback
from ..analysis.emotion_detection import EmotionAnalysisResult
from ..analysis.pacing_analysis import PacingAnalysisResult

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Generates visual representations of speech analysis results."""
    
    def __init__(self):
        """Initialize the visualization generator."""
        # Set up matplotlib style and font configuration
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Configure font to avoid Unicode glyph warnings
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
        
        # Define color scheme
        self.colors = {
            'excellent': '#28a745',
            'good': '#ffc107', 
            'fair': '#fd7e14',
            'poor': '#dc3545',
            'primary': '#007bff',
            'secondary': '#6c757d',
            'background': '#f8f9fa'
        }
    
    def create_overall_dashboard(self, feedback: ComprehensiveFeedback, 
                               save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive dashboard visualization.
        
        Args:
            feedback: Comprehensive feedback object
            save_path: Optional path to save the visualization
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Creating overall dashboard visualization...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Speech Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Overall score gauge (top-left)
        ax_gauge = fig.add_subplot(gs[0, 0])
        self._create_gauge_chart(ax_gauge, feedback.overall_rating, "Overall Score")
        
        # 2. Category ratings bar chart (top-center, spanning 2 columns)
        ax_bars = fig.add_subplot(gs[0, 1:3])
        self._create_category_bars(ax_bars, feedback)
        
        # 3. Speech metadata (top-right)
        ax_meta = fig.add_subplot(gs[0, 3])
        self._create_metadata_panel(ax_meta, feedback)
        
        # 4. Radar chart for detailed analysis (middle-left, spanning 2 columns)
        ax_radar = fig.add_subplot(gs[1, :2])
        self._create_radar_chart(ax_radar, feedback)
        
        # 5. Strengths vs Improvements (middle-right, spanning 2 columns)
        ax_strengths = fig.add_subplot(gs[1, 2:])
        self._create_strengths_improvements_chart(ax_strengths, feedback)
        
        # 6. Action plan priorities (bottom, spanning all columns)
        ax_actions = fig.add_subplot(gs[2, :])
        self._create_action_plan_chart(ax_actions, feedback)
        
        # Save or return the figure
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Dashboard saved to: {output_path}")
            plt.close()
            return str(output_path)
        else:
            plt.tight_layout()
            plt.show()
            return "displayed"
    
    def create_emotion_timeline(self, emotion_result: EmotionAnalysisResult,
                              save_path: Optional[str] = None) -> str:
        """
        Create a timeline showing emotional progression through the speech.
        
        Args:
            emotion_result: Emotion analysis result
            save_path: Optional path to save the visualization
            
        Returns:
            Path to the generated visualization file
        """
        if not emotion_result.sentences:
            logger.warning("No sentence data available for emotion timeline")
            return "no_data"
        
        logger.info("Creating emotion timeline visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Emotional Journey Through Your Speech', fontsize=16, fontweight='bold')
        
        # Prepare data
        sentences = emotion_result.sentences
        sentence_numbers = list(range(1, len(sentences) + 1))
        
        # Extract emotion data
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        emotion_colors = {
            'joy': '#FFD700', 'sadness': '#4169E1', 'anger': '#DC143C',
            'fear': '#8B008B', 'surprise': '#FF8C00', 'disgust': '#556B2F', 'neutral': '#708090'
        }
        
        # Plot 1: Dominant emotion per sentence
        dominant_emotions = [s.emotion_scores.get_dominant_emotion() for s in sentences]
        emotion_intensities = [s.emotional_intensity for s in sentences]
        
        colors = [emotion_colors[emotion] for emotion in dominant_emotions]
        bars = ax1.bar(sentence_numbers, emotion_intensities, color=colors, alpha=0.7)
        
        ax1.set_title('Emotional Intensity by Sentence', fontweight='bold')
        ax1.set_xlabel('Sentence Number')
        ax1.set_ylabel('Emotional Intensity')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add emotion labels on bars
        for i, (bar, emotion) in enumerate(zip(bars, dominant_emotions)):
            if emotion_intensities[i] > 0.1:  # Only label if significant intensity
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        emotion.title(), ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Plot 2: Emotion composition over time
        emotion_data = []
        for emotion in emotions:
            values = [getattr(s.emotion_scores, emotion) for s in sentences]
            emotion_data.append(values)
        
        ax2.stackplot(sentence_numbers, *emotion_data, 
                     labels=emotions, colors=[emotion_colors[e] for e in emotions], alpha=0.8)
        
        ax2.set_title('Emotion Composition Throughout Speech', fontweight='bold')
        ax2.set_xlabel('Sentence Number')
        ax2.set_ylabel('Emotion Proportion')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Emotion timeline saved to: {output_path}")
            plt.close()
            return str(output_path)
        else:
            plt.show()
            return "displayed"
    
    def create_pacing_analysis(self, pacing_result: PacingAnalysisResult,
                             save_path: Optional[str] = None) -> str:
        """
        Create visualizations for pacing analysis.
        
        Args:
            pacing_result: Pacing analysis result
            save_path: Optional path to save the visualization
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Creating pacing analysis visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Speech Pacing Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall WPM vs Ideal Range
        ideal_min, ideal_max = pacing_result.ideal_wpm_range
        categories = ['Your Pace', 'Ideal Min', 'Ideal Max']
        values = [pacing_result.overall_wpm, ideal_min, ideal_max]
        colors = [self._get_rating_color(pacing_result.likert_rating), 
                 self.colors['secondary'], self.colors['secondary']]
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_title('Speaking Pace Comparison', fontweight='bold')
        ax1.set_ylabel('Words Per Minute')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Add ideal range shading
        ax1.axhspan(ideal_min, ideal_max, alpha=0.2, color=self.colors['good'], 
                   label='Ideal Range')
        ax1.legend()
        
        # 2. Pacing consistency gauge
        self._create_gauge_chart(ax2, pacing_result.pacing_consistency * 5, 
                                "Pacing Consistency", max_value=5)
        
        # 3. Segment-by-segment pacing (if available)
        if pacing_result.segments and len(pacing_result.segments) > 1:
            segment_nums = list(range(1, len(pacing_result.segments) + 1))
            segment_wpm = [seg.wpm for seg in pacing_result.segments if seg.wpm]
            
            if segment_wpm:
                ax3.plot(segment_nums[:len(segment_wpm)], segment_wpm, 'o-', 
                        color=self.colors['primary'], linewidth=2, markersize=6)
                ax3.axhspan(ideal_min, ideal_max, alpha=0.2, color=self.colors['good'])
                ax3.set_title('Pacing Variation Throughout Speech', fontweight='bold')
                ax3.set_xlabel('Segment Number')
                ax3.set_ylabel('Words Per Minute')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient segment data', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Segment Pacing (No Data)', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Single segment analysis', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Segment Pacing (Single Segment)', fontweight='bold')
        
        # 4. Pause analysis
        pause_data = pacing_result.pause_analysis
        if pause_data and 'pause_count' in pause_data:
            pause_metrics = ['Total Pause Time', 'Pause Count', 'Avg Pause Duration']
            pause_values = [
                pause_data.get('total_pause_time', 0),
                pause_data.get('pause_count', 0),
                pause_data.get('avg_pause_duration', 0)
            ]
            
            # Normalize values for display
            normalized_values = []
            for i, value in enumerate(pause_values):
                if i == 0:  # Total pause time
                    normalized_values.append(min(value, 60))  # Cap at 60 seconds
                elif i == 1:  # Pause count
                    normalized_values.append(min(value, 50))  # Cap at 50 pauses
                else:  # Average duration
                    normalized_values.append(min(value, 5))   # Cap at 5 seconds
            
            bars = ax4.bar(pause_metrics, normalized_values, 
                          color=[self.colors['secondary']] * 3, alpha=0.7)
            ax4.set_title('Pause Analysis', fontweight='bold')
            ax4.set_ylabel('Normalized Values')
            
            # Add actual values as labels
            for bar, actual_value in zip(bars, pause_values):
                label = f'{actual_value:.1f}'
                if bar.get_height() > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            label, ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No pause data available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Pause Analysis (No Data)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pacing analysis saved to: {output_path}")
            plt.close()
            return str(output_path)
        else:
            plt.show()
            return "displayed"
    
    def _create_gauge_chart(self, ax, value: float, title: str, max_value: float = 5):
        """Create a gauge chart for displaying ratings."""
        # Normalize value to 0-1 range
        normalized_value = value / max_value
        
        # Create semicircle
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=10)
        
        # Value arc
        value_theta = np.linspace(0, np.pi * normalized_value, int(100 * normalized_value))
        color = self._get_rating_color_from_value(normalized_value)
        ax.plot(r * np.cos(value_theta), r * np.sin(value_theta), color=color, linewidth=10)
        
        # Needle
        needle_angle = np.pi * normalized_value
        needle_x = 0.9 * np.cos(needle_angle)
        needle_y = 0.9 * np.sin(needle_angle)
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.05, head_length=0.05, 
                fc='black', ec='black')
        
        # Labels
        ax.text(0, -0.3, f'{value:.1f}/{max_value}', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12)
        
        # Formatting
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _create_category_bars(self, ax, feedback: ComprehensiveFeedback):
        """Create horizontal bar chart for category ratings."""
        categories = ['Emotional\nDelivery', 'Pacing &\nRhythm', 
                     'Clarity &\nComprehension', 'Tone\nDelivery']
        ratings = [feedback.emotion_section.rating, feedback.pacing_section.rating,
                  feedback.clarity_section.rating, feedback.tone_section.rating]
        colors = [self._get_rating_color(rating) for rating in ratings]
        
        bars = ax.barh(categories, ratings, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, rating in zip(bars, ratings):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{rating}/5', va='center', fontweight='bold')
        
        ax.set_xlim(0, 5.5)
        ax.set_xlabel('Rating (1-5)')
        ax.set_title('Category Performance', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
    
    def _create_metadata_panel(self, ax, feedback: ComprehensiveFeedback):
        """Create metadata information panel."""
        ax.axis('off')
        
        metadata_text = f"""
Analysis Summary

Goal: {feedback.parameters_used.delivery_goal.value.title()}
Audience: {feedback.parameters_used.audience_type.value.title()}
Words: {len(feedback.parameters_used.speech_text.split())}
Date: {feedback.generated_timestamp.split()[0]}
Score: {feedback.overall_rating}/5

Key Insights:
• {len(feedback.emotion_section.strengths + feedback.pacing_section.strengths + 
       feedback.clarity_section.strengths + feedback.tone_section.strengths)} Strengths
• {len(feedback.actionable_steps)} Action Items
"""
        
        ax.text(0.05, 0.95, metadata_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor=self.colors['background'], alpha=0.8))
    
    def _create_radar_chart(self, ax, feedback: ComprehensiveFeedback):
        """Create radar chart for detailed skill assessment."""
        # Define skill areas (more granular than main categories)
        skills = ['Emotional\nExpression', 'Speaking\nPace', 'Voice\nClarity', 
                 'Audience\nAlignment', 'Content\nStructure', 'Overall\nImpact']
        
        # Map feedback to skill scores (0-5 scale)
        scores = [
            feedback.emotion_section.rating,
            feedback.pacing_section.rating, 
            feedback.clarity_section.rating,
            feedback.tone_section.rating,
            (feedback.clarity_section.rating + feedback.tone_section.rating) / 2,  # Structure
            feedback.overall_rating  # Overall impact
        ]
        
        # Number of skills
        N = len(skills)
        
        # Compute angles for each skill
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add scores for the first skill at the end to complete the circle
        scores += scores[:1]
        
        # Plot
        ax.plot(angles, scores, 'o-', linewidth=2, color=self.colors['primary'])
        ax.fill(angles, scores, alpha=0.25, color=self.colors['primary'])
        
        # Add skill labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(skills)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.grid(True)
        
        ax.set_title('Skill Assessment Radar', fontweight='bold', pad=20)
    
    def _create_strengths_improvements_chart(self, ax, feedback: ComprehensiveFeedback):
        """Create strengths vs improvements comparison."""
        # Count strengths and improvements by category
        categories = ['Emotion', 'Pacing', 'Clarity', 'Tone']
        strengths_count = [
            len(feedback.emotion_section.strengths),
            len(feedback.pacing_section.strengths),
            len(feedback.clarity_section.strengths),
            len(feedback.tone_section.strengths)
        ]
        improvements_count = [
            len(feedback.emotion_section.improvements),
            len(feedback.pacing_section.improvements),
            len(feedback.clarity_section.improvements),
            len(feedback.tone_section.improvements)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strengths_count, width, label='Strengths', 
                      color=self.colors['excellent'], alpha=0.8)
        bars2 = ax.bar(x + width/2, improvements_count, width, label='Improvements Needed',
                      color=self.colors['poor'], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Count')
        ax.set_title('Strengths vs Areas for Improvement', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    def _create_action_plan_chart(self, ax, feedback: ComprehensiveFeedback):
        """Create action plan priority visualization."""
        ax.axis('off')
        
        # Create action plan text
        action_text = "Priority Action Plan:\n\n"
        
        for i, action in enumerate(feedback.actionable_steps[:5], 1):
            # Truncate long actions
            short_action = action[:60] + "..." if len(action) > 60 else action
            action_text += f"{i}. {short_action}\n"
        
        if feedback.next_practice_focus:
            action_text += f"\nNext Practice Focus:\n"
            for focus in feedback.next_practice_focus[:3]:
                short_focus = focus[:50] + "..." if len(focus) > 50 else focus
                action_text += f"• {short_focus}\n"
        
        # Display in a styled text box
        ax.text(0.02, 0.98, action_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor=self.colors['background'], alpha=0.9))
    
    def _get_rating_color(self, rating: int) -> str:
        """Get color based on rating value."""
        if rating >= 4:
            return self.colors['excellent']
        elif rating >= 3:
            return self.colors['good']
        elif rating >= 2:
            return self.colors['fair']
        else:
            return self.colors['poor']
    
    def _get_rating_color_from_value(self, normalized_value: float) -> str:
        """Get color based on normalized value (0-1)."""
        if normalized_value >= 0.8:
            return self.colors['excellent']
        elif normalized_value >= 0.6:
            return self.colors['good']
        elif normalized_value >= 0.4:
            return self.colors['fair']
        else:
            return self.colors['poor']
