"""
Audio Processor Module

Handles audio file loading, preprocessing, and cleanup operations.
Supports MP3 and WAV formats with noise reduction and normalization.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Processes audio files for speech analysis."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load audio file and return audio data with duration.
        
        Args:
            file_path: Path to the audio file (MP3 or WAV)
            
        Returns:
            Tuple of (audio_data, duration_in_seconds)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        supported_formats = {'.mp3', '.wav', '.m4a', '.flac'}
        if file_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")
            
        try:
            # Load audio with librosa
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            duration = len(audio_data) / sr
            
            logger.info(f"Loaded audio file: {file_path.name}, Duration: {duration:.2f}s")
            return audio_data, duration
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data with noise reduction and normalization.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Normalize audio to [-1, 1] range
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply basic noise reduction using spectral gating
        audio_data = self._reduce_noise(audio_data)
        
        # Apply pre-emphasis filter to improve high-frequency components
        audio_data = self._apply_preemphasis(audio_data)
        
        logger.info("Audio preprocessing completed")
        return audio_data
    
    def _reduce_noise(self, audio_data: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Apply basic noise reduction using spectral subtraction.
        
        Args:
            audio_data: Input audio data
            noise_factor: Factor for noise reduction
            
        Returns:
            Noise-reduced audio data
        """
        # Compute STFT
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor from first 0.5 seconds
        noise_duration = min(int(0.5 * self.sample_rate / 512), magnitude.shape[1])
        noise_profile = np.mean(magnitude[:, :noise_duration], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        cleaned_magnitude = magnitude - noise_factor * noise_profile
        cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft)
        
        return cleaned_audio
    
    def _apply_preemphasis(self, audio_data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to enhance high frequencies.
        
        Args:
            audio_data: Input audio data
            alpha: Pre-emphasis coefficient
            
        Returns:
            Pre-emphasized audio data
        """
        return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
    
    def save_processed_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save processed audio to file.
        
        Args:
            audio_data: Processed audio data
            output_path: Path to save the processed audio
        """
        try:
            sf.write(output_path, audio_data, self.sample_rate)
            logger.info(f"Processed audio saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed audio: {str(e)}")
            raise
    
    def get_audio_features(self, audio_data: np.ndarray) -> dict:
        """
        Extract basic audio features for analysis.
        
        Args:
            audio_data: Audio data
            
        Returns:
            Dictionary containing audio features
        """
        features = {}
        
        # Basic statistics
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        return features
