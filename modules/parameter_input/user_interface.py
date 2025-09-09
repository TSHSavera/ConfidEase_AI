"""
User Interface Module

Provides a simple command-line interface for user input collection.
Can be extended to support web UI or GUI in the future.
"""

import sys
from typing import Optional, Dict, Any
from .parameter_validator import ParameterValidator, SpeechParameters
import logging

logger = logging.getLogger(__name__)


class UserInterface:
    """Command-line user interface for parameter collection."""
    
    def __init__(self):
        """Initialize the user interface."""
        self.validator = ParameterValidator()
        self.parameters = None
    
    def display_welcome(self) -> None:
        """Display welcome message and instructions."""
        print("\n" + "="*60)
        print("  🎤 SPEECH ANALYZER - AI Voice Assessment Tool 🎤")
        print("="*60)
        print("\nWelcome! This tool will analyze your speech for:")
        print("• Emotional delivery")
        print("• Pacing and rhythm")
        print("• Clarity of expression")
        print("• Tone alignment with your goals")
        print("\n📝 LANGUAGE SUPPORT:")
        print("• Optimized for English language speech")
        print("• Audio transcription forced to English")
        print("• Best results with clear English pronunciation")
        print("\n⚠️  REQUIREMENTS:")
        print("• Both speech text AND audio file are required")
        print("• This ensures accurate analysis and ratings")
        print("\nLet's begin by gathering some information about your speech...")
        print()
    
    def display_audience_options(self) -> None:
        """Display available audience type options."""
        print("\n📊 AUDIENCE TYPES:")
        print("-" * 40)
        options = self.validator.get_audience_options()
        for i, (audience_type, description) in enumerate(options.items(), 1):
            print(f"{i:2d}. {audience_type.title():<12} - {description}")
    
    def display_goal_options(self) -> None:
        """Display available delivery goal options."""
        print("\n🎯 DELIVERY GOALS:")
        print("-" * 40)
        options = self.validator.get_goal_options()
        for i, (goal, description) in enumerate(options.items(), 1):
            print(f"{i:2d}. {goal.title():<12} - {description}")
    
    def get_user_choice(self, prompt: str, options: list, allow_text: bool = True) -> str:
        """
        Get user choice from numbered options or direct text input.
        
        Args:
            prompt: Prompt to display to user
            options: List of available options
            allow_text: Whether to allow direct text input
            
        Returns:
            Selected option as string
        """
        while True:
            try:
                user_input = input(f"\n{prompt}: ").strip()
                
                if not user_input:
                    print("❌ Please provide an input.")
                    continue
                
                # Check if input is a number (option selection)
                if user_input.isdigit():
                    choice_num = int(user_input)
                    if 1 <= choice_num <= len(options):
                        return options[choice_num - 1]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(options)}")
                        continue
                
                # If text input is allowed, return the input
                if allow_text:
                    return user_input
                else:
                    print("❌ Please enter a valid number from the options above.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def get_audience_type(self) -> str:
        """Get audience type from user."""
        self.display_audience_options()
        audience_options = list(self.validator.get_audience_options().keys())
        
        return self.get_user_choice(
            "Select your target audience (number or type name)",
            audience_options,
            allow_text=True
        )
    
    def get_delivery_goal(self) -> str:
        """Get delivery goal from user."""
        self.display_goal_options()
        goal_options = list(self.validator.get_goal_options().keys())
        
        return self.get_user_choice(
            "Select your delivery goal (number or type name)",
            goal_options,
            allow_text=True
        )
    
    def get_speech_text(self) -> str:
        """Get speech text from user."""
        print("\n📝 SPEECH TEXT (Required):")
        print("-" * 40)
        print("Please enter your speech text below.")
        print("Both text and audio are required for accurate ratings.")
        print("(You can paste multiple lines. Press Enter twice to finish)")
        
        lines = []
        empty_lines = 0
        
        while True:
            try:
                line = input()
                if line.strip() == "":
                    empty_lines += 1
                    if empty_lines >= 2:
                        break
                else:
                    empty_lines = 0
                    lines.append(line)
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
        
        speech_text = "\n".join(lines).strip()
        if not speech_text:
            print("❌ Speech text cannot be empty. Please try again.")
            return self.get_speech_text()
        
        return speech_text
    
    def get_audio_file(self) -> Optional[str]:
        """Get audio file path from user."""
        print("\n🎵 AUDIO FILE (Required):")
        print("-" * 40)
        print("Please enter the path to your audio file.")
        print("Supported formats: MP3, WAV, M4A, FLAC")
        print("Both text and audio are required for accurate ratings.")
        
        file_path = input("\nAudio file path: ").strip()
        return file_path if file_path else None
    
    def get_custom_notes(self) -> Optional[str]:
        """Get optional custom notes from user."""
        print("\n📋 CUSTOM NOTES (Optional):")
        print("-" * 40)
        print("Any additional context or specific areas you'd like analyzed?")
        print("Press Enter to skip.")
        
        notes = input("\nCustom notes: ").strip()
        return notes if notes else None
    
    def collect_parameters(self) -> SpeechParameters:
        """
        Collect all parameters from user through interactive prompts.
        
        Returns:
            Validated SpeechParameters object
        """        
        try:
            # Collect all parameters
            audience_type = self.get_audience_type()
            delivery_goal = self.get_delivery_goal()
            speech_text = self.get_speech_text()
            audio_file = self.get_audio_file()
            
            # Validate that both text and audio are provided (now required for accurate ratings)
            if not speech_text or not speech_text.strip():
                print("\n❌ ERROR: Speech text is required for accurate analysis.")
                print("👋 Goodbye!")
                sys.exit(0)
            
            if not audio_file or not audio_file.strip():
                print("\n❌ ERROR: Audio file is required for accurate analysis.")
                print("👋 Goodbye!")
                sys.exit(0)
            
            custom_notes = self.get_custom_notes()
            
            # Validate and create parameters object
            print("\n⚙️  Validating parameters...")
            self.parameters = self.validator.create_parameters(
                audience_type=audience_type,
                delivery_goal=delivery_goal,
                speech_text=speech_text,
                audio_file_path=audio_file,
                custom_notes=custom_notes
            )
            
            # Display summary
            self.display_parameter_summary()
            
            return self.parameters
            
        except Exception as e:
            logger.error(f"Error collecting parameters: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")
            return self.collect_parameters()
    
    def display_parameter_summary(self) -> None:
        """Display summary of collected parameters."""
        if not self.parameters:
            return
        
        print("\n" + "="*60)
        print("  📋 PARAMETER SUMMARY")
        print("="*60)
        print(f"Target Audience: {self.parameters.audience_type.value.title()}")
        print(f"Delivery Goal:   {self.parameters.delivery_goal.value.title()}")
        print(f"Speech Length:   {len(self.parameters.speech_text)} characters")
        print(f"Word Count:      {len(self.parameters.speech_text.split())} words")
        
        if self.parameters.audio_file_path:
            print(f"Audio File:      {self.parameters.audio_file_path}")
        else:
            print("Audio File:      None (text-only analysis)")
        
        if self.parameters.custom_notes:
            print(f"Custom Notes:    {self.parameters.custom_notes[:50]}{'...' if len(self.parameters.custom_notes) > 50 else ''}")
        
        # Display ideal pacing range
        min_wpm, max_wpm = self.validator.get_ideal_pacing_range(
            self.parameters.audience_type,
            self.parameters.delivery_goal
        )
        print(f"Ideal WPM Range: {min_wpm}-{max_wpm} words per minute")
        
        print("="*60)
        
        # Confirm with user
        confirm = input("\n✅ Proceed with analysis? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Let's try again...")
            self.collect_parameters()
    
    def display_error(self, error_message: str) -> None:
        """
        Display error message to user.
        
        Args:
            error_message: Error message to display
        """
        print(f"\n❌ Error: {error_message}")
    
    def display_info(self, info_message: str) -> None:
        """
        Display info message to user.
        
        Args:
            info_message: Info message to display
        """
        print(f"\nℹ️  {info_message}")
    
    def display_success(self, success_message: str) -> None:
        """
        Display success message to user.
        
        Args:
            success_message: Success message to display
        """
        print(f"\n✅ {success_message}")


def create_simple_gui_interface():
    """
    Future extension point for GUI interface.
    Could use tkinter, PyQt, or web-based interface.
    """
    pass


def create_web_interface():
    """
    Future extension point for web interface.
    Could use Flask, FastAPI, or Streamlit.
    """
    pass
