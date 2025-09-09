"""
Speech Analyzer - AI-Powered Voice Assessment Tool

Main application entry point that orchestrates all modules for comprehensive
speech analysis including emotion detection, pacing analysis, clarity assessment,
and tone delivery evaluation.
"""

import logging
import sys
import traceback
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

# Import all modules
from modules.parameter_input import UserInterface, ParameterValidator
from modules.audio_input import AudioProcessor, SpeechToText
from modules.analysis import EmotionDetector, PacingAnalyzer, ClarityAnalyzer, ToneDeliveryAnalyzer
from modules.feedback_generation import FeedbackGenerator
from modules.output import ResultPresenter, VisualizationGenerator
from modules.training.data_collection import TrainingDataCollector, FeedbackInterface

# Create a custom stream handler that can handle Unicode
class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            # Remove emojis and special characters for console output
            msg = re.sub(r'[^\x00-\x7F]+', '?', msg)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging (simplified to avoid Unicode issues)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('speech_analyzer.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class SpeechAnalyzer:
    """Main application class for speech analysis."""
    
    def __init__(self):
        """Initialize the Speech Analyzer application."""
        logger.info("Initializing Speech Analyzer...")
        
        # Initialize modules
        self.ui = UserInterface()
        self.validator = ParameterValidator()
        self.audio_processor = AudioProcessor()
        self.speech_to_text = None  # Initialized on demand
        self.emotion_detector = None  # Initialized on demand
        self.pacing_analyzer = PacingAnalyzer()
        self.clarity_analyzer = ClarityAnalyzer()
        self.tone_analyzer = ToneDeliveryAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        self.result_presenter = ResultPresenter()
        self.visualizer = VisualizationGenerator()
        
        # Initialize training data collection
        self.data_collector = TrainingDataCollector()
        self.feedback_interface = FeedbackInterface(self.data_collector)
        
        logger.info("Speech Analyzer initialized successfully")
    
    def run(self):
        """Run the main application flow."""
        try:
            self.ui.display_welcome()
            
            # Step 1: Collect user parameters
            logger.info("Collecting user parameters...")
            parameters = self.ui.collect_parameters()
            
            # Step 2: Process audio if provided
            transcription_result = None
            audio_features = None
            
            if parameters.audio_file_path:
                logger.info("Processing audio file...")
                transcription_result, audio_features = self._process_audio(parameters.audio_file_path)
                
                # Update speech text with transcription if no text provided
                if not parameters.speech_text.strip() and transcription_result:
                    parameters.speech_text = transcription_result.full_text
                    self.ui.display_info("Using transcribed text for analysis")
            
            # Step 3: Perform comprehensive analysis
            logger.info("Starting comprehensive speech analysis...")
            self.ui.display_info("Analyzing your speech... This may take a moment.")
            
            # Emotion Analysis
            self.ui.display_info("🎭 Analyzing emotional delivery...")
            emotion_result = self._analyze_emotions(parameters.speech_text)
            
            # Pacing Analysis
            self.ui.display_info("⏱️ Analyzing pacing and rhythm...")
            pacing_result = self._analyze_pacing(
                parameters.speech_text, 
                parameters.audience_type, 
                parameters.delivery_goal,
                transcription_result
            )
            
            # Clarity Analysis
            self.ui.display_info("🔍 Analyzing clarity and comprehension...")
            clarity_result = self._analyze_clarity(parameters.speech_text, audio_features, transcription_result)
            
            # Tone Delivery Analysis
            self.ui.display_info("🎯 Analyzing tone delivery alignment...")
            tone_result = self._analyze_tone_delivery(
                emotion_result, pacing_result, clarity_result,
                parameters.audience_type, parameters.delivery_goal
            )
            
            # Step 4: Generate comprehensive feedback
            self.ui.display_info("📝 Generating personalized feedback...")
            feedback = self.feedback_generator.generate_comprehensive_feedback(
                emotion_result, pacing_result, clarity_result, tone_result, parameters
            )
            
            # Step 5: Present results
            logger.info("Presenting analysis results...")
            self._present_results(feedback, parameters)
            
            # Step 6: Collect user feedback for training (optional)
            self.feedback_interface.ask_for_feedback(feedback)
            
            # Step 7: Optional visualizations
            self._offer_visualizations(feedback, emotion_result, pacing_result, clarity_result, tone_result)
            
            # Step 8: Save options
            self._offer_save_options(feedback)
            
            self.ui.display_success("Analysis complete! Thank you for using Speech Analyzer.")
            
        except KeyboardInterrupt:
            self.ui.display_info("Analysis interrupted by user.")
            logger.info("Application interrupted by user")
        except Exception as e:
            self.ui.display_error(f"An error occurred during analysis: {str(e)}")
            logger.error(f"Application error: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    def _process_audio(self, audio_file_path: str):
        """Process audio file and return transcription results."""
        try:
            # Load and preprocess audio
            audio_data, duration = self.audio_processor.load_audio(audio_file_path)
            processed_audio = self.audio_processor.preprocess_audio(audio_data)
            audio_features = self.audio_processor.get_audio_features(processed_audio)
            
            # Initialize speech-to-text if needed
            if not self.speech_to_text:
                self.ui.display_info("Loading speech recognition model...")
                self.speech_to_text = SpeechToText()
            
            # Transcribe audio
            self.ui.display_info("Transcribing audio...")
            transcription_result = self.speech_to_text.transcribe(processed_audio)
            
            # Save transcription to file
            if transcription_result and transcription_result.full_text:
                self._save_transcription(transcription_result, audio_file_path)
            
            logger.info(f"Audio processed successfully. Duration: {duration:.2f}s, "
                       f"Transcribed {transcription_result.word_count} words")
            
            return transcription_result, audio_features
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            self.ui.display_error(f"Audio processing failed: {str(e)}")
            return None, None
    
    def _analyze_emotions(self, speech_text: str):
        """Analyze emotions in the speech text."""
        if not self.emotion_detector:
            self.ui.display_info("Loading emotion detection model...")
            self.emotion_detector = EmotionDetector()
        
        return self.emotion_detector.analyze_text(speech_text)
    
    def _analyze_pacing(self, speech_text: str, audience_type, delivery_goal, transcription_result=None):
        """Analyze speech pacing."""
        if transcription_result and transcription_result.segments:
            return self.pacing_analyzer.analyze_audio_pacing(
                transcription_result.segments, speech_text, audience_type, delivery_goal
            )
        else:
            return self.pacing_analyzer.analyze_text_pacing(
                speech_text, audience_type, delivery_goal
            )
    
    def _analyze_clarity(self, speech_text: str, audio_features=None, transcription_result=None):
        """Analyze speech clarity."""
        return self.clarity_analyzer.analyze_clarity(speech_text, audio_features, transcription_result)
    
    def _analyze_tone_delivery(self, emotion_result, pacing_result, clarity_result, 
                              audience_type, delivery_goal):
        """Analyze tone delivery alignment."""
        return self.tone_analyzer.analyze_tone_delivery(
            emotion_result, pacing_result, clarity_result, audience_type, delivery_goal
        )
    
    def _present_results(self, feedback, parameters):
        """Present analysis results to the user."""
        # Display results in console
        self.result_presenter.present_results(feedback, output_format='console')
    
    def _offer_visualizations(self, feedback, emotion_result=None, pacing_result=None, 
                             clarity_result=None, tone_result=None):
        """Offer to create visual representations."""
        try:
            create_viz = input("\n📊 Would you like to create visual charts? (y/n): ").strip().lower()
            
            if create_viz in ['y', 'yes']:
                self.ui.display_info("Generating visualizations...")
                
                # Create output directory
                output_dir = Path("speech_analysis_output")
                output_dir.mkdir(exist_ok=True)
                
                # Generate dashboard
                dashboard_path = output_dir / "analysis_dashboard.png"
                self.visualizer.create_overall_dashboard(feedback, str(dashboard_path))
                
                self.ui.display_success(f"Dashboard saved to: {dashboard_path}")
                
                # Offer emotion timeline if emotion analysis was performed
                if emotion_result:
                    emotion_viz = input("🎭 Create emotion timeline? (y/n): ").strip().lower()
                    if emotion_viz in ['y', 'yes']:
                        self.ui.display_info("Generating emotion timeline...")
                        
                        # Generate timestamp for unique filename
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        timeline_path = output_dir / f"emotion_timeline_{timestamp}.png"
                        
                        # Create emotion timeline visualization
                        self.visualizer.create_emotion_timeline(emotion_result, str(timeline_path))
                        self.ui.display_success(f"Emotion timeline saved to: {timeline_path}")
                else:
                    self.ui.display_info("💡 Emotion timeline requires emotion analysis data")
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            self.ui.display_error("Visualization generation failed")
    
    def _offer_save_options(self, feedback):
        """Offer to save results in different formats."""
        try:
            save_results = input("\n💾 Would you like to save the detailed report? (y/n): ").strip().lower()
            
            if save_results in ['y', 'yes']:
                # Create output directory
                output_dir = Path("speech_analysis_output")
                output_dir.mkdir(exist_ok=True)
                
                # Offer format choices
                print("\nAvailable formats:")
                print("1. Text (.txt)")
                print("2. HTML (.html)")
                print("3. JSON (.json)")
                print("4. CSV (.csv)")
                
                format_choice = input("Choose format (1-4): ").strip()
                
                format_map = {
                    '1': ('text', '.txt'),
                    '2': ('html', '.html'),
                    '3': ('json', '.json'),
                    '4': ('csv', '.csv')
                }
                
                if format_choice in format_map:
                    output_format, extension = format_map[format_choice]
                    
                    timestamp = feedback.generated_timestamp.replace(':', '-').replace(' ', '_')
                    filename = f"speech_analysis_{timestamp}{extension}"
                    output_path = output_dir / filename
                    
                    self.result_presenter.present_results(
                        feedback, output_format=output_format, output_path=str(output_path)
                    )
                    
                    self.ui.display_success(f"Report saved to: {output_path}")
                else:
                    self.ui.display_info("Invalid format choice. Skipping save.")
                    
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            self.ui.display_error("Failed to save results")
    
    def _save_transcription(self, transcription_result, audio_file_path: str):
        """Save the transcribed text to a file for evaluation purposes."""
        try:
            # Create output directory
            output_dir = Path("speech_analysis_output")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename based on audio file and timestamp
            audio_filename = Path(audio_file_path).stem
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            transcription_filename = f"transcription_{audio_filename}_{timestamp}.txt"
            transcription_path = output_dir / transcription_filename
            
            # Prepare transcription content
            content = []
            content.append("=" * 60)
            content.append("SPEECH TRANSCRIPTION RESULTS")
            content.append("=" * 60)
            content.append(f"Audio File: {audio_file_path}")
            content.append(f"Transcription Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content.append(f"Word Count: {transcription_result.word_count}")
            content.append(f"Confidence Score: {getattr(transcription_result, 'confidence', 'N/A')}")
            content.append("-" * 60)
            content.append("FULL TRANSCRIPTION:")
            content.append("-" * 60)
            content.append(transcription_result.full_text)
            
            # Add segment details if available
            if hasattr(transcription_result, 'segments') and transcription_result.segments:
                content.append("\n" + "-" * 60)
                content.append("DETAILED SEGMENTS:")
                content.append("-" * 60)
                for i, segment in enumerate(transcription_result.segments, 1):
                    start_time = getattr(segment, 'start_time', 'N/A')
                    end_time = getattr(segment, 'end_time', 'N/A')
                    text = getattr(segment, 'text', str(segment))
                    content.append(f"Segment {i}: [{start_time}-{end_time}] {text}")
            
            content.append("\n" + "=" * 60)
            content.append("END OF TRANSCRIPTION")
            content.append("=" * 60)
            
            # Write to file
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            
            self.ui.display_success(f"📄 Transcription saved to: {transcription_path}")
            logger.info(f"Transcription saved to: {transcription_path}")
            
        except Exception as e:
            logger.error(f"Error saving transcription: {str(e)}")
            self.ui.display_error(f"Failed to save transcription: {str(e)}")


def main():
    """Main entry point for the application."""
    
    try:
        app = SpeechAnalyzer()
        app.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logger.critical(f"Fatal application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
