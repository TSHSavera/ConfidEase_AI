"""
Speech-to-Text Module

Handles transcription of audio to text using OpenAI's Whisper model.
Provides word-level timestamps and confidence scores when available.
"""

import whisper
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed speech."""
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    full_text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    word_count: int


class SpeechToText:
    """Speech-to-text converter using Whisper model."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the speech-to-text converter.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ("cuda", "cpu", or None for auto)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe audio data to text with timestamps.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            TranscriptionResult object with full transcription and segments
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Whisper expects audio to be normalized to [-1, 1]
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            logger.info("Starting transcription...")
            
            # Transcribe with word-level timestamps and force English language
            result = self.model.transcribe(
                audio_data,
                word_timestamps=True,
                verbose=False,
                language="en"  # Force English language
            )
            
            # Process segments
            segments = []
            for segment in result.get("segments", []):
                seg = TranscriptionSegment(
                    text=segment["text"].strip(),
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=segment.get("avg_logprob")
                )
                segments.append(seg)
            
            # Calculate duration and word count
            duration = len(audio_data) / sample_rate
            full_text = result["text"].strip()
            word_count = len(full_text.split()) if full_text else 0
            
            transcription_result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language="en",  # Always English since we force this
                duration=duration,
                word_count=word_count
            )
            
            logger.info(f"Transcription completed. Language: English, "
                       f"Words: {word_count}, Duration: {duration:.2f}s")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Transcribe audio file directly.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            logger.info(f"Transcribing file: {file_path}")
            result = self.model.transcribe(
                file_path, 
                word_timestamps=True,
                language="en"  # Force English language
            )
            
            # Process segments
            segments = []
            for segment in result.get("segments", []):
                seg = TranscriptionSegment(
                    text=segment["text"].strip(),
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=segment.get("avg_logprob")
                )
                segments.append(seg)
            
            full_text = result["text"].strip()
            word_count = len(full_text.split()) if full_text else 0
            
            # Estimate duration from segments if available
            duration = segments[-1].end_time if segments else 0.0
            
            transcription_result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language="en",  # Always English since we force this
                duration=duration,
                word_count=word_count
            )
            
            logger.info(f"File transcription completed. Language: English, "
                       f"Words: {word_count}, Duration: {duration:.2f}s")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            raise
    
    def get_words_per_minute(self, transcription_result: TranscriptionResult) -> float:
        """
        Calculate words per minute from transcription result.
        
        Args:
            transcription_result: TranscriptionResult object
            
        Returns:
            Words per minute as float
        """
        if transcription_result.duration <= 0:
            return 0.0
        
        wpm = (transcription_result.word_count / transcription_result.duration) * 60
        return round(wpm, 2)
    
    def get_speech_segments_info(self, transcription_result: TranscriptionResult) -> Dict:
        """
        Extract speech timing information from transcription.
        
        Args:
            transcription_result: TranscriptionResult object
            
        Returns:
            Dictionary with speech timing statistics
        """
        if not transcription_result.segments:
            return {}
        
        segment_durations = [
            seg.end_time - seg.start_time 
            for seg in transcription_result.segments
        ]
        
        # Calculate pauses between segments
        pauses = []
        for i in range(len(transcription_result.segments) - 1):
            pause_duration = (transcription_result.segments[i + 1].start_time - 
                            transcription_result.segments[i].end_time)
            if pause_duration > 0:
                pauses.append(pause_duration)
        
        return {
            "segment_count": len(transcription_result.segments),
            "avg_segment_duration": np.mean(segment_durations) if segment_durations else 0,
            "total_speech_time": sum(segment_durations),
            "total_pause_time": sum(pauses),
            "avg_pause_duration": np.mean(pauses) if pauses else 0,
            "speech_rate": transcription_result.word_count / sum(segment_durations) * 60 if segment_durations else 0
        }
