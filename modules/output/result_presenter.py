"""
Result Presenter Module

Handles formatting and presentation of analysis results in various formats
including console output, file export, and structured data.
"""

import json
import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from ..feedback_generation.feedback_generator import ComprehensiveFeedback
from ..parameter_input.parameter_validator import SpeechParameters
from ..analysis.emotion_detection import EmotionAnalysisResult
from ..analysis.pacing_analysis import PacingAnalysisResult
from ..analysis.clarity_analysis import ClarityAnalysisResult
from ..analysis.tone_delivery import ToneDeliveryResult

logger = logging.getLogger(__name__)


class ResultPresenter:
    """Presents analysis results in various formats."""
    
    def __init__(self):
        """Initialize the result presenter."""
        self.output_formats = ['console', 'text', 'json', 'csv', 'html']
    
    def present_results(self, feedback: ComprehensiveFeedback, 
                       output_format: str = 'console',
                       output_path: Optional[str] = None) -> str:
        """
        Present results in the specified format.
        
        Args:
            feedback: Comprehensive feedback object
            output_format: Format for presentation ('console', 'text', 'json', 'csv', 'html')
            output_path: Optional path to save output file
            
        Returns:
            Formatted output string
        """
        if output_format not in self.output_formats:
            raise ValueError(f"Unsupported output format: {output_format}. "
                           f"Supported formats: {', '.join(self.output_formats)}")
        
        logger.info(f"Presenting results in {output_format} format")
        
        if output_format == 'console':
            output = self._format_console_output(feedback)
            print(output)
        elif output_format == 'text':
            output = self._format_text_output(feedback)
        elif output_format == 'json':
            output = self._format_json_output(feedback)
        elif output_format == 'csv':
            output = self._format_csv_output(feedback)
        elif output_format == 'html':
            output = self._format_html_output(feedback)
        
        # Save to file if path provided
        if output_path and output_format != 'console':
            self._save_output(output, output_path, output_format)
        
        return output
    
    def _format_console_output(self, feedback: ComprehensiveFeedback) -> str:
        """Format output for console display with colors and formatting."""
        
        # ANSI color codes
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        END = '\033[0m'
        
        def get_rating_color(rating: int) -> str:
            if rating >= 4:
                return GREEN
            elif rating >= 3:
                return YELLOW
            else:
                return RED
        
        def format_stars(rating: int) -> str:
            return "⭐" * rating + "☆" * (5 - rating)
        
        output = f"""
{BOLD}{CYAN}{'='*80}{END}
{BOLD}{CYAN}🎤 SPEECH ANALYSIS REPORT{END}
{BOLD}{CYAN}{'='*80}{END}

{BOLD}Analysis Details:{END}
📅 Generated: {feedback.generated_timestamp}
🎯 Speech Goal: {feedback.parameters_used.delivery_goal.value.title()}
👥 Target Audience: {feedback.parameters_used.audience_type.value.title()}
📝 Word Count: {len(feedback.parameters_used.speech_text.split())} words

{BOLD}{UNDERLINE}OVERALL ASSESSMENT{END}
{get_rating_color(feedback.overall_rating)}{BOLD}{format_stars(feedback.overall_rating)} ({feedback.overall_rating}/5){END}
{feedback.overall_summary}

{BOLD}{BLUE}{'='*50}{END}
{BOLD}{BLUE}📊 DETAILED ANALYSIS{END}
{BOLD}{BLUE}{'='*50}{END}

{BOLD}1. EMOTIONAL DELIVERY{END} {get_rating_color(feedback.emotion_section.rating)}{format_stars(feedback.emotion_section.rating)} ({feedback.emotion_section.rating}/5){END}
{feedback.emotion_section.summary}

{BOLD}💪 Strengths:{END}
{chr(10).join(f"  ✅ {strength}" for strength in feedback.emotion_section.strengths)}

{BOLD}🔧 Areas for Improvement:{END}
{chr(10).join(f"  🔸 {improvement}" for improvement in feedback.emotion_section.improvements)}

{BOLD}2. PACING & RHYTHM{END} {get_rating_color(feedback.pacing_section.rating)}{format_stars(feedback.pacing_section.rating)} ({feedback.pacing_section.rating}/5){END}
{feedback.pacing_section.summary}

{BOLD}💪 Strengths:{END}
{chr(10).join(f"  ✅ {strength}" for strength in feedback.pacing_section.strengths)}

{BOLD}🔧 Areas for Improvement:{END}
{chr(10).join(f"  🔸 {improvement}" for improvement in feedback.pacing_section.improvements)}

{BOLD}3. CLARITY & COMPREHENSION{END} {get_rating_color(feedback.clarity_section.rating)}{format_stars(feedback.clarity_section.rating)} ({feedback.clarity_section.rating}/5){END}
{feedback.clarity_section.summary}

{BOLD}💪 Strengths:{END}
{chr(10).join(f"  ✅ {strength}" for strength in feedback.clarity_section.strengths)}

{BOLD}🔧 Areas for Improvement:{END}
{chr(10).join(f"  🔸 {improvement}" for improvement in feedback.clarity_section.improvements)}

{BOLD}4. TONE DELIVERY{END} {get_rating_color(feedback.tone_section.rating)}{format_stars(feedback.tone_section.rating)} ({feedback.tone_section.rating}/5){END}
{feedback.tone_section.summary}

{BOLD}💪 Strengths:{END}
{chr(10).join(f"  ✅ {strength}" for strength in feedback.tone_section.strengths)}

{BOLD}🔧 Areas for Improvement:{END}
{chr(10).join(f"  🔸 {improvement}" for improvement in feedback.tone_section.improvements)}

{BOLD}{YELLOW}{'='*50}{END}
{BOLD}{YELLOW}🎯 ACTION PLAN{END}
{BOLD}{YELLOW}{'='*50}{END}

{BOLD}Priority Actions:{END}
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(feedback.actionable_steps))}

{BOLD}Next Practice Focus:{END}
{chr(10).join(f"  🎯 {focus}" for focus in feedback.next_practice_focus)}

{BOLD}{CYAN}{'='*80}{END}
{BOLD}{CYAN}Thank you for using Speech Analyzer!{END}
{BOLD}{CYAN}Practice these recommendations and re-analyze to track your progress.{END}
{BOLD}{CYAN}{'='*80}{END}
"""
        return output
    
    def _format_text_output(self, feedback: ComprehensiveFeedback) -> str:
        """Format output as plain text for file saving."""
        from ..feedback_generation.feedback_generator import FeedbackGenerator
        generator = FeedbackGenerator()
        return generator.format_feedback_report(feedback)
    
    def _format_json_output(self, feedback: ComprehensiveFeedback) -> str:
        """Format output as JSON for API or data exchange."""
        
        # Convert feedback to JSON-serializable dictionary
        json_data = {
            "analysis_metadata": {
                "timestamp": feedback.generated_timestamp,
                "overall_rating": feedback.overall_rating,
                "overall_summary": feedback.overall_summary
            },
            "parameters": {
                "delivery_goal": feedback.parameters_used.delivery_goal.value,
                "audience_type": feedback.parameters_used.audience_type.value,
                "word_count": len(feedback.parameters_used.speech_text.split()),
                "has_audio": feedback.parameters_used.audio_file_path is not None,
                "custom_notes": feedback.parameters_used.custom_notes
            },
            "analysis_results": {
                "emotion": {
                    "rating": feedback.emotion_section.rating,
                    "summary": feedback.emotion_section.summary,
                    "detailed_feedback": feedback.emotion_section.detailed_feedback,
                    "strengths": feedback.emotion_section.strengths,
                    "improvements": feedback.emotion_section.improvements,
                    "recommendations": feedback.emotion_section.recommendations
                },
                "pacing": {
                    "rating": feedback.pacing_section.rating,
                    "summary": feedback.pacing_section.summary,
                    "detailed_feedback": feedback.pacing_section.detailed_feedback,
                    "strengths": feedback.pacing_section.strengths,
                    "improvements": feedback.pacing_section.improvements,
                    "recommendations": feedback.pacing_section.recommendations
                },
                "clarity": {
                    "rating": feedback.clarity_section.rating,
                    "summary": feedback.clarity_section.summary,
                    "detailed_feedback": feedback.clarity_section.detailed_feedback,
                    "strengths": feedback.clarity_section.strengths,
                    "improvements": feedback.clarity_section.improvements,
                    "recommendations": feedback.clarity_section.recommendations
                },
                "tone": {
                    "rating": feedback.tone_section.rating,
                    "summary": feedback.tone_section.summary,
                    "detailed_feedback": feedback.tone_section.detailed_feedback,
                    "strengths": feedback.tone_section.strengths,
                    "improvements": feedback.tone_section.improvements,
                    "recommendations": feedback.tone_section.recommendations
                }
            },
            "action_plan": {
                "priority_actions": feedback.actionable_steps,
                "practice_focus": feedback.next_practice_focus
            }
        }
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    def _format_csv_output(self, feedback: ComprehensiveFeedback) -> str:
        """Format output as CSV for spreadsheet analysis."""
        
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers and basic info
        writer.writerow(["Speech Analysis Results"])
        writer.writerow(["Timestamp", feedback.generated_timestamp])
        writer.writerow(["Delivery Goal", feedback.parameters_used.delivery_goal.value])
        writer.writerow(["Audience Type", feedback.parameters_used.audience_type.value])
        writer.writerow(["Word Count", len(feedback.parameters_used.speech_text.split())])
        writer.writerow(["Overall Rating", feedback.overall_rating])
        writer.writerow([])
        
        # Write detailed ratings
        writer.writerow(["Category", "Rating", "Summary"])
        writer.writerow(["Emotional Delivery", feedback.emotion_section.rating, feedback.emotion_section.summary])
        writer.writerow(["Pacing & Rhythm", feedback.pacing_section.rating, feedback.pacing_section.summary])
        writer.writerow(["Clarity & Comprehension", feedback.clarity_section.rating, feedback.clarity_section.summary])
        writer.writerow(["Tone Delivery", feedback.tone_section.rating, feedback.tone_section.summary])
        writer.writerow([])
        
        # Write clarity word analysis if available
        if hasattr(feedback.clarity_section, 'missed_words') and feedback.clarity_section.missed_words:
            writer.writerow(["Clarity Analysis - Missed Words"])
            writer.writerow(["Word", "Status"])
            for word in feedback.clarity_section.missed_words:
                writer.writerow([word, "Not captured in transcription"])
            writer.writerow([])
        
        if hasattr(feedback.clarity_section, 'potentially_mispronounced') and feedback.clarity_section.potentially_mispronounced:
            writer.writerow(["Clarity Analysis - Potential Mispronunciations"])
            writer.writerow(["Intended Word", "Transcribed As"])
            for intended, transcribed in feedback.clarity_section.potentially_mispronounced:
                writer.writerow([intended, transcribed])
            writer.writerow([])
        
        # Write strengths
        writer.writerow(["Strengths"])
        all_strengths = (feedback.emotion_section.strengths + 
                        feedback.pacing_section.strengths + 
                        feedback.clarity_section.strengths + 
                        feedback.tone_section.strengths)
        for strength in all_strengths:
            writer.writerow([strength])
        writer.writerow([])
        
        # Write action items
        writer.writerow(["Priority Actions"])
        for action in feedback.actionable_steps:
            writer.writerow([action])
        writer.writerow([])
        
        # Write practice focus
        writer.writerow(["Practice Focus"])
        for focus in feedback.next_practice_focus:
            writer.writerow([focus])
        
        return output.getvalue()
    
    def _format_html_output(self, feedback: ComprehensiveFeedback) -> str:
        """Format output as HTML for web viewing."""
        
        def format_stars_html(rating: int) -> str:
            stars = "⭐" * rating + "☆" * (5 - rating)
            return f'<span class="rating-{rating}">{stars} ({rating}/5)</span>'
        
        def get_rating_class(rating: int) -> str:
            if rating >= 4:
                return "rating-excellent"
            elif rating >= 3:
                return "rating-good"
            else:
                return "rating-needs-improvement"
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .overall-rating {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .section {{
            background: white;
            margin-bottom: 25px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section-header {{
            padding: 20px;
            font-weight: bold;
            font-size: 1.2em;
            border-bottom: 1px solid #eee;
        }}
        .section-content {{
            padding: 20px;
        }}
        .rating-excellent {{ color: #28a745; }}
        .rating-good {{ color: #ffc107; }}
        .rating-needs-improvement {{ color: #dc3545; }}
        .strengths, .improvements {{
            margin: 15px 0;
        }}
        .strengths h4 {{
            color: #28a745;
            margin-bottom: 10px;
        }}
        .improvements h4 {{
            color: #dc3545;
            margin-bottom: 10px;
        }}
        .action-plan {{
            background: #e8f5e8;
            padding: 25px;
            border-radius: 10px;
            margin-top: 30px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎤 Speech Analysis Report</h1>
        <p>AI-Powered Voice Assessment</p>
    </div>
    
    <div class="metadata">
        <strong>Analysis Details:</strong><br>
        📅 Generated: {feedback.generated_timestamp}<br>
        🎯 Speech Goal: {feedback.parameters_used.delivery_goal.value.title()}<br>
        👥 Target Audience: {feedback.parameters_used.audience_type.value.title()}<br>
        📝 Word Count: {len(feedback.parameters_used.speech_text.split())} words
    </div>
    
    <div class="overall-rating">
        <h2>Overall Assessment</h2>
        <div class="{get_rating_class(feedback.overall_rating)}" style="font-size: 2em; margin: 15px 0;">
            {format_stars_html(feedback.overall_rating)}
        </div>
        <p>{feedback.overall_summary}</p>
    </div>
    
    <div class="section">
        <div class="section-header {get_rating_class(feedback.emotion_section.rating)}">
            1. Emotional Delivery {format_stars_html(feedback.emotion_section.rating)}
        </div>
        <div class="section-content">
            <p><strong>Summary:</strong> {feedback.emotion_section.summary}</p>
            <p>{feedback.emotion_section.detailed_feedback}</p>
            
            {self._format_strengths_improvements_html(feedback.emotion_section.strengths, feedback.emotion_section.improvements)}
        </div>
    </div>
    
    <div class="section">
        <div class="section-header {get_rating_class(feedback.pacing_section.rating)}">
            2. Pacing & Rhythm {format_stars_html(feedback.pacing_section.rating)}
        </div>
        <div class="section-content">
            <p><strong>Summary:</strong> {feedback.pacing_section.summary}</p>
            <p>{feedback.pacing_section.detailed_feedback}</p>
            
            {self._format_strengths_improvements_html(feedback.pacing_section.strengths, feedback.pacing_section.improvements)}
        </div>
    </div>
    
    <div class="section">
        <div class="section-header {get_rating_class(feedback.clarity_section.rating)}">
            3. Clarity & Comprehension {format_stars_html(feedback.clarity_section.rating)}
        </div>
        <div class="section-content">
            <p><strong>Summary:</strong> {feedback.clarity_section.summary}</p>
            <p>{feedback.clarity_section.detailed_feedback}</p>
            
            {self._format_strengths_improvements_html(feedback.clarity_section.strengths, feedback.clarity_section.improvements)}
        </div>
    </div>
    
    <div class="section">
        <div class="section-header {get_rating_class(feedback.tone_section.rating)}">
            4. Tone Delivery {format_stars_html(feedback.tone_section.rating)}
        </div>
        <div class="section-content">
            <p><strong>Summary:</strong> {feedback.tone_section.summary}</p>
            <p>{feedback.tone_section.detailed_feedback}</p>
            
            {self._format_strengths_improvements_html(feedback.tone_section.strengths, feedback.tone_section.improvements)}
        </div>
    </div>
    
    <div class="action-plan">
        <h2>🎯 Action Plan</h2>
        
        <h3>Priority Actions:</h3>
        <ol>
            {chr(10).join(f"<li>{step}</li>" for step in feedback.actionable_steps)}
        </ol>
        
        <h3>Next Practice Focus:</h3>
        <ul>
            {chr(10).join(f"<li>{focus}</li>" for focus in feedback.next_practice_focus)}
        </ul>
    </div>
    
    <div style="text-align: center; margin-top: 40px; padding: 20px; background: white; border-radius: 10px;">
        <h3>Thank you for using Speech Analyzer!</h3>
        <p>Practice these recommendations and re-analyze to track your progress.</p>
    </div>
</body>
</html>
"""
        return html
    
    def _format_strengths_improvements_html(self, strengths: List[str], improvements: List[str]) -> str:
        """Format strengths and improvements as HTML."""
        html = ""
        
        if strengths:
            html += '<div class="strengths"><h4>💪 Strengths:</h4><ul>'
            html += chr(10).join(f"<li>{strength}</li>" for strength in strengths)
            html += '</ul></div>'
        
        if improvements:
            html += '<div class="improvements"><h4>🔧 Areas for Improvement:</h4><ul>'
            html += chr(10).join(f"<li>{improvement}</li>" for improvement in improvements)
            html += '</ul></div>'
        
        return html
    
    def _save_output(self, output: str, output_path: str, output_format: str) -> None:
        """Save output to file."""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure correct file extension
            extensions = {
                'text': '.txt',
                'json': '.json',
                'csv': '.csv',
                'html': '.html'
            }
            
            if output_format in extensions:
                expected_ext = extensions[output_format]
                if not path.suffix:
                    path = path.with_suffix(expected_ext)
                elif path.suffix.lower() != expected_ext:
                    path = path.with_suffix(expected_ext)
            
            # Write file with appropriate encoding
            encoding = 'utf-8'
            with open(path, 'w', encoding=encoding) as f:
                f.write(output)
            
            logger.info(f"Output saved to: {path}")
            
        except Exception as e:
            logger.error(f"Error saving output to {output_path}: {str(e)}")
            raise
    
    def create_summary_dashboard(self, feedback: ComprehensiveFeedback) -> Dict[str, Any]:
        """Create a dashboard-style summary of results."""
        
        ratings = {
            'emotion': feedback.emotion_section.rating,
            'pacing': feedback.pacing_section.rating,
            'clarity': feedback.clarity_section.rating,
            'tone': feedback.tone_section.rating
        }
        
        dashboard = {
            'overall_score': feedback.overall_rating,
            'overall_percentage': (feedback.overall_rating / 5) * 100,
            'category_scores': ratings,
            'category_percentages': {k: (v / 5) * 100 for k, v in ratings.items()},
            'strengths_count': len(feedback.emotion_section.strengths + 
                                 feedback.pacing_section.strengths + 
                                 feedback.clarity_section.strengths + 
                                 feedback.tone_section.strengths),
            'improvement_areas_count': len(feedback.emotion_section.improvements + 
                                         feedback.pacing_section.improvements + 
                                         feedback.clarity_section.improvements + 
                                         feedback.tone_section.improvements),
            'action_items_count': len(feedback.actionable_steps),
            'analysis_date': feedback.generated_timestamp,
            'speech_metadata': {
                'goal': feedback.parameters_used.delivery_goal.value,
                'audience': feedback.parameters_used.audience_type.value,
                'word_count': len(feedback.parameters_used.speech_text.split()),
                'has_audio': feedback.parameters_used.audio_file_path is not None
            }
        }
        
        return dashboard
