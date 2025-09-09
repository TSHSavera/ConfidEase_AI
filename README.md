# ConfidEase AI

## 🤖 How the AI Analyzes Your Speech

### 1. **Audio Processing Pipeline**
```
Audio Input → Whisper STT → Text Comparison → Clarity Scoring
```
- **Whisper AI** transcribes your speech with high accuracy
- **Librosa** processes audio signals for acoustic features
- **Noise reduction** and normalization improve transcription quality
- **Multi-format support** handles various audio file types

### 2. **Emotion Detection Process**
```
Text → Sentence Segmentation → Transformer Models → Emotion Classification
```
- **HuggingFace Transformers** analyze emotional content
- **Sentence-level mapping** tracks emotional progression
- **VADER sentiment** provides additional emotional context
- **Consistency scoring** measures emotional stability throughout speech

### 3. **Clarity Analysis Method**
```
User Text ↔ AI Transcription → Word Matching → Pronunciation Assessment
```
- **Word-level comparison** between intended and transcribed text
- **Edit distance algorithms** identify potential mispronunciations
- **Phonetic similarity matching** catches sound-based errors
- **Missing word detection** highlights unclear pronunciation
- **Clarity score calculation**:
  - 50% Word accuracy (exact matches)
  - 25% Sequence preservation (word order)
  - 15% Length consistency
  - 10% Character similarity

### 4. **Pacing Analysis Engine**
```
Audio Segments → Timing Analysis → WPM Calculation → Rhythm Assessment
```
- **Words per minute** calculation from audio timestamps
- **Rhythm consistency** analysis across speech segments
- **Pause detection** and emphasis identification
- **Audience-specific optimization** recommendations

### 5. **Tone Delivery Assessment**
```
All Metrics → Context Analysis → Goal Alignment → Effectiveness Score
```
- **Multi-dimensional analysis** combining all previous metrics
- **Audience-goal matching** for contextual appropriateness
- **Delivery effectiveness** scoring based on intended outcome

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **4GB+ RAM** (8GB+ for optimal performance)
- **Internet connection** (for initial model downloads ~1-3GB)
- **English language speech** (primary optimization)

### Installation

1. **Download and Setup**
   ```bash
   # Navigate to project directory
   cd speech_analyzer
   
   # Create virtual environment
   python -m venv speech_analyzer_env
   
   # Activate environment
   speech_analyzer_env\Scripts\activate  # Windows
   source speech_analyzer_env/bin/activate  # macOS/Linux
   ```

2. **Install Dependencies**
   ```bash
   # Install all packages
   pip install -r requirements.txt
   
   # If issues occur, try:
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Download Required Data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('vader_lexicon')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   ```
   or run
   ```bash
   python nltk_download.py
   ```

### Run the Application
```bash
python main.py
```

## 📋 Usage Guide

### Step-by-Step Process
1. **Choose Input**: Text entry or audio file upload
2. **Set Parameters**:
   - **Audience**: Friendly, Hostile, Uninformed, etc.
   - **Goal**: Inform, Persuade, Entertain, Motivate, etc.
3. **AI Analysis**: Comprehensive processing across all dimensions
4. **Review Results**: Detailed feedback with specific recommendations
5. **Generate Visuals**: Optional charts and emotion timelines
6. **Export Reports**: Multiple formats (TXT, HTML, JSON, CSV)

### Audio Requirements
- **Formats**: WAV, MP3, FLAC, M4A
- **Quality**: Clear audio, minimal background noise
- **Duration**: 30 seconds to 10 minutes (optimal)
- **Language**: English (required for best accuracy)

## 📊 Understanding Results

### Emotion Analysis Output
- **Primary emotions** with confidence scores
- **Sentence-by-sentence** emotional progression
- **Emotional consistency** throughout speech
- **Audience alignment** recommendations

### Clarity Assessment Details
- **AI Transcription** vs. your intended text
- **Word accuracy percentage** and specific missed words
- **Potential mispronunciations** with suggested corrections
- **Pronunciation difficulty** identification

### Pacing Analysis Results
- **Speaking rate** (WPM) with audience-specific recommendations
- **Rhythm consistency** percentage
- **Pause patterns** and emphasis detection
- **Timing optimization** suggestions

### Comprehensive Scoring
Each dimension receives a **1-5 Likert scale rating**:
- **5**: Excellent performance
- **4**: Good with minor improvements
- **3**: Satisfactory, room for enhancement
- **2**: Needs improvement
- **1**: Significant work required

## 📈 Output Formats & Files

### Generated Files
- `speech_analyzer.log` - Application logs and debugging
- `speech_analysis_output/` - All results and visualizations
  - `analysis_dashboard.png` - Visual performance dashboard
  - `emotion_timeline_[timestamp].png` - Emotional progression chart
  - `transcription_[audio]_[timestamp].txt` - AI transcription comparison
  - `speech_analysis_[timestamp].[format]` - Detailed reports

### Available Export Formats
- **Console**
- **Text**
- **HTML**
- **JSON**
- **CSV**

## 🛠️ Technical Architecture

### Core AI Models
- **OpenAI Whisper**
- **HuggingFace Transformers**
- **NLTK**
- **Custom Algorithms**
- **RoBERTa-base**

## 🔧 Troubleshooting

### Common Issues & Solutions

**Installation Problems**
```bash
# SSL/Certificate errors
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Version conflicts
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --force-reinstall
```

**GPU/CUDA Issues**
```bash
# For NVIDIA GPU acceleration
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Audio Processing Issues**
```bash
# Additional codec support
pip install ffmpeg-python
# Download FFmpeg from: https://ffmpeg.org/download.html
```

**Memory Issues**
- Close other applications to free RAM
- Use shorter audio files (< 5 minutes)
- Consider upgrading to 8GB+ RAM

### System Diagnostics
```python
import sys, platform
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch not installed")
```

### Getting Help
1. Check `speech_analyzer.log` for detailed error messages
2. Ensure Python 3.8+ is installed
3. Verify all dependencies are correctly installed
4. Test with a small, clear audio file first

