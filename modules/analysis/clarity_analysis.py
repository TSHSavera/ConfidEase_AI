"""
Clarity Analysis Sub-Module

Analyzes speech clarity through pronunciation assessment and sentence structure analysis.
Evaluates readability, complexity, and potential areas of confusion.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from textstat import flesch_reading_ease, flesch_kincaid_grade, coleman_liau_index
import nltk
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SentenceClarity:
    """Clarity analysis for a single sentence."""
    text: str
    length_score: float  # 0-1, based on optimal length
    complexity_score: float  # 0-1, based on word/structure complexity
    readability_score: float  # 0-1, based on readability metrics
    ambiguity_flags: List[str]  # Potential ambiguity issues
    clarity_rating: float  # Overall 0-1 clarity score


@dataclass
class VocabularyAnalysis:
    """Analysis of vocabulary usage."""
    total_words: int
    unique_words: int
    vocabulary_diversity: float  # Type-token ratio
    avg_word_length: float
    complex_words: int
    jargon_words: List[str]
    filler_words: List[str]
    repetitive_phrases: List[Tuple[str, int]]


@dataclass
class ClarityAnalysisResult:
    """Complete clarity analysis result."""
    sentences: List[SentenceClarity]
    vocabulary_analysis: VocabularyAnalysis
    overall_readability: Dict[str, float]
    structure_analysis: Dict[str, any]
    pronunciation_flags: List[str]  # Potential pronunciation difficulties
    clarity_score: float  # Overall 0-1 clarity score
    likert_rating: int  # 1-5 scale
    recommendations: List[str]
    transcription_info: Optional[Dict[str, any]] = None  # AI transcription details
    missed_words: List[str] = None  # Words not captured in transcription
    potentially_mispronounced: List[Tuple[str, str]] = None  # (intended, transcribed) pairs


class ClarityAnalyzer:
    """Analyzes speech clarity and comprehensibility."""
    
    def __init__(self):
        """Initialize the clarity analyzer."""
        self._download_nltk_data()
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 'kind of',
            'basically', 'actually', 'literally', 'totally', 'really', 'very',
            'just', 'so', 'well', 'okay', 'right'
        }
        
        self.complex_words_threshold = 7  # Words with 7+ characters considered complex
        self.optimal_sentence_length = (10, 20)  # Optimal sentence length range
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
    
    def analyze_clarity(self, text: str, audio_features: Optional[Dict] = None, 
                       transcription_result: Optional[any] = None) -> ClarityAnalysisResult:
        """
        Perform comprehensive clarity analysis.
        
        Args:
            text: Speech text to analyze (user-provided)
            audio_features: Optional audio features for pronunciation analysis
            transcription_result: AI transcription result for comparison (primary method)
            
        Returns:
            ClarityAnalysisResult object
        """
        logger.info("Starting clarity analysis...")
        
        # Process transcription information and calculate primary clarity score
        transcription_info = None
        primary_clarity_score = None
        
        if transcription_result and text.strip():
            # Primary method: Compare transcription with user text
            transcription_info = self._process_transcription_info(transcription_result, text)
            primary_clarity_score = self._calculate_transcription_based_clarity(
                transcription_result, text, transcription_info
            )
            logger.info(f"Transcription-based clarity score: {primary_clarity_score:.2f}")
        
        # Fallback analysis using traditional methods
        sentences = self._analyze_sentences(text)
        vocabulary_analysis = self._analyze_vocabulary(text)
        readability_metrics = self._calculate_readability(text)
        structure_analysis = self._analyze_structure(text)
        pronunciation_flags = self._identify_pronunciation_challenges(text)
        
        # Use transcription-based score if available, otherwise use traditional calculation
        if primary_clarity_score is not None:
            clarity_score = primary_clarity_score
        else:
            clarity_score = self._calculate_overall_clarity(
                sentences, vocabulary_analysis, readability_metrics
            )
            logger.info("Using traditional clarity calculation (no transcription available)")
        
        # Convert to Likert rating
        likert_rating = self._calculate_clarity_rating(clarity_score, structure_analysis)
        
        # Generate recommendations
        recommendations = self._generate_clarity_recommendations(
            sentences, vocabulary_analysis, readability_metrics, structure_analysis
        )
        
        # Extract missed and mispronounced words from transcription info
        missed_words = []
        potentially_mispronounced = []
        
        if transcription_info:
            missed_words = transcription_info.get('missed_words', [])
            potentially_mispronounced = transcription_info.get('potentially_mispronounced', [])
        
        result = ClarityAnalysisResult(
            sentences=sentences,
            vocabulary_analysis=vocabulary_analysis,
            overall_readability=readability_metrics,
            structure_analysis=structure_analysis,
            pronunciation_flags=pronunciation_flags,
            clarity_score=clarity_score,
            likert_rating=likert_rating,
            recommendations=recommendations,
            transcription_info=transcription_info,
            missed_words=missed_words,
            potentially_mispronounced=potentially_mispronounced
        )
        
        logger.info(f"Clarity analysis completed. Overall score: {clarity_score:.2f}, "
                   f"Rating: {likert_rating}/5")
        
        return result
    
    def _analyze_sentences(self, text: str) -> List[SentenceClarity]:
        """Analyze clarity of individual sentences."""
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except:
            # Fallback to basic sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_analyses = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Length analysis
            word_count = len(sentence.split())
            min_len, max_len = self.optimal_sentence_length
            
            if min_len <= word_count <= max_len:
                length_score = 1.0
            elif word_count < min_len:
                length_score = word_count / min_len
            else:
                # Penalty for overly long sentences
                length_score = max(0.3, max_len / word_count)
            
            # Complexity analysis
            complexity_score = self._analyze_sentence_complexity(sentence)
            
            # Readability analysis
            try:
                readability_score = min(1.0, flesch_reading_ease(sentence) / 100.0)
            except:
                readability_score = 0.7  # Default moderate readability
            
            # Ambiguity detection
            ambiguity_flags = self._detect_ambiguity(sentence)
            
            # Overall clarity rating
            clarity_rating = (length_score + complexity_score + readability_score) / 3
            
            sentence_clarity = SentenceClarity(
                text=sentence,
                length_score=length_score,
                complexity_score=complexity_score,
                readability_score=readability_score,
                ambiguity_flags=ambiguity_flags,
                clarity_rating=clarity_rating
            )
            sentence_analyses.append(sentence_clarity)
        
        return sentence_analyses
    
    def _analyze_sentence_complexity(self, sentence: str) -> float:
        """Analyze grammatical and lexical complexity of a sentence."""
        words = sentence.split()
        
        # Word length complexity
        avg_word_length = np.mean([len(word.strip('.,!?;:')) for word in words])
        word_length_score = max(0.2, 1.0 - (avg_word_length - 4) / 6)  # Optimal ~4 chars
        
        # Syllable complexity (estimated)
        avg_syllables = np.mean([self._estimate_syllables(word) for word in words])
        syllable_score = max(0.2, 1.0 - (avg_syllables - 1.5) / 2)  # Optimal ~1.5 syllables
        
        # Punctuation complexity (nested clauses)
        punctuation_density = len(re.findall(r'[,;:]', sentence)) / len(words)
        punct_score = max(0.3, 1.0 - punctuation_density * 5)
        
        return (word_length_score + syllable_score + punct_score) / 3
    
    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower().strip('.,!?;:"\'')
        if not word:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _detect_ambiguity(self, sentence: str) -> List[str]:
        """Detect potential ambiguity issues in a sentence."""
        flags = []
        
        # Check for pronoun ambiguity
        pronouns = re.findall(r'\b(it|this|that|they|them|these|those)\b', sentence.lower())
        if len(pronouns) > 2:
            flags.append("Multiple pronouns may create confusion")
        
        # Check for passive voice (simplified detection)
        if re.search(r'\b(was|were|is|are|been)\s+\w+ed\b', sentence):
            flags.append("Passive voice may reduce clarity")
        
        # Check for double negatives
        negatives = re.findall(r'\b(not|no|never|nothing|nobody|nowhere|neither)\b', sentence.lower())
        if len(negatives) > 1:
            flags.append("Multiple negatives may confuse meaning")
        
        # Check for complex subordinate clauses
        if sentence.count(',') > 3 or sentence.count(';') > 0:
            flags.append("Complex sentence structure may impact clarity")
        
        return flags
    
    def _analyze_vocabulary(self, text: str) -> VocabularyAnalysis:
        """Analyze vocabulary usage and diversity."""
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        total_words = len(words)
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Complex words (7+ characters)
        complex_words = sum(1 for word in words if len(word) >= self.complex_words_threshold)
        
        # Identify jargon (words with 3+ syllables)
        jargon_words = [word for word in set(words) if self._estimate_syllables(word) >= 3]
        
        # Identify filler words
        filler_words = [word for word in words if word in self.filler_words]
        
        # Find repetitive phrases
        word_counts = Counter(words)
        repetitive_phrases = [(word, count) for word, count in word_counts.items() 
                             if count > max(2, total_words // 50)]  # Repeated more than 2% of speech
        
        return VocabularyAnalysis(
            total_words=total_words,
            unique_words=unique_words,
            vocabulary_diversity=vocabulary_diversity,
            avg_word_length=avg_word_length,
            complex_words=complex_words,
            jargon_words=jargon_words[:10],  # Top 10 jargon words
            filler_words=filler_words,
            repetitive_phrases=repetitive_phrases[:5]  # Top 5 repetitive phrases
        )
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability metrics."""
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_grade = flesch_kincaid_grade(text)
            coleman_liau = coleman_liau_index(text)
            
            return {
                'flesch_reading_ease': flesch_ease,
                'flesch_kincaid_grade': flesch_grade,
                'coleman_liau_index': coleman_liau,
                'readability_score': min(100, max(0, flesch_ease)) / 100  # Normalized 0-1
            }
        except Exception as e:
            logger.warning(f"Error calculating readability: {e}")
            return {
                'flesch_reading_ease': 60.0,
                'flesch_kincaid_grade': 8.0,
                'coleman_liau_index': 8.0,
                'readability_score': 0.6
            }
    
    def _analyze_structure(self, text: str) -> Dict[str, any]:
        """Analyze overall text structure."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Sentence length variation
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variation = np.std(sentence_lengths) / np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Paragraph structure (assuming double line breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Transition analysis
        transition_words = re.findall(
            r'\b(however|therefore|furthermore|moreover|additionally|consequently|thus|hence)\b',
            text.lower()
        )
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'sentence_length_variation': length_variation,
            'paragraph_count': paragraph_count,
            'transition_word_count': len(transition_words),
            'structure_score': self._calculate_structure_score(sentence_lengths, paragraph_count, len(transition_words))
        }
    
    def _calculate_structure_score(self, sentence_lengths: List[int], 
                                 paragraph_count: int, transition_count: int) -> float:
        """Calculate overall structure clarity score."""
        if not sentence_lengths:
            return 0.5
        
        # Sentence length consistency
        avg_length = np.mean(sentence_lengths)
        ideal_avg = 15  # Ideal average sentence length
        length_score = max(0.2, 1.0 - abs(avg_length - ideal_avg) / ideal_avg)
        
        # Paragraph organization (assuming reasonable text length)
        total_sentences = len(sentence_lengths)
        ideal_para_ratio = 0.2  # ~5 sentences per paragraph
        actual_para_ratio = paragraph_count / total_sentences if total_sentences > 0 else 0
        para_score = max(0.3, 1.0 - abs(actual_para_ratio - ideal_para_ratio) / ideal_para_ratio)
        
        # Transition usage
        transition_score = min(1.0, transition_count / max(1, paragraph_count * 0.5))
        
        return (length_score + para_score + transition_score) / 3
    
    def _identify_pronunciation_challenges(self, text: str) -> List[str]:
        """Identify words that may be challenging to pronounce."""
        challenges = []
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Long words (8+ characters)
        long_words = [word for word in set(words) if len(word) >= 8]
        if len(long_words) > len(set(words)) * 0.1:  # More than 10% long words
            challenges.append(f"Many long words may be challenging: {', '.join(long_words[:5])}")
        
        # Words with difficult consonant clusters
        difficult_patterns = [
            r'\w*ths\w*',  # 'ths' sounds
            r'\w*str\w*',  # 'str' clusters
            r'\w*spr\w*',  # 'spr' clusters
            r'\w*scr\w*',  # 'scr' clusters
        ]
        
        for pattern in difficult_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                unique_matches = list(set(matches))[:3]  # Convert set to list for slicing
                challenges.append(f"Words with challenging consonant clusters: {', '.join(unique_matches)}")
                break
        
        # Technical terms (words with numbers or specialized endings)
        technical_words = re.findall(r'\b\w*(?:tion|sion|ment|ology|ical|ness)\b', text.lower())
        if len(technical_words) > 5:
            unique_technical = list(set(technical_words))[:3]  # Convert set to list for slicing
            challenges.append(f"Technical terms may need careful pronunciation: {', '.join(unique_technical)}")
        
        return challenges
    
    def _calculate_overall_clarity(self, sentences: List[SentenceClarity],
                                 vocabulary: VocabularyAnalysis,
                                 readability: Dict[str, float]) -> float:
        """Calculate overall clarity score."""
        if not sentences:
            return 0.5
        
        # Average sentence clarity
        sentence_score = np.mean([s.clarity_rating for s in sentences])
        
        # Vocabulary clarity (diversity vs complexity balance)
        vocab_score = min(1.0, vocabulary.vocabulary_diversity * 2)  # Reward diversity
        vocab_score *= max(0.3, 1.0 - vocabulary.complex_words / vocabulary.total_words)  # Penalize complexity
        
        # Readability score
        readability_score = readability.get('readability_score', 0.6)
        
        # Weighted combination
        overall_score = (sentence_score * 0.4 + vocab_score * 0.3 + readability_score * 0.3)
        
        return max(0.0, min(1.0, overall_score))
    
    def _calculate_clarity_rating(self, clarity_score: float, 
                                structure_analysis: Dict[str, any]) -> int:
        """Convert clarity score to Likert scale rating."""
        base_rating = clarity_score * 5  # Scale to 1-5
        
        # Adjust based on structure
        structure_score = structure_analysis.get('structure_score', 0.5)
        if structure_score > 0.8:
            base_rating += 0.3
        elif structure_score < 0.4:
            base_rating -= 0.3
        
        return max(1, min(5, round(base_rating)))
    
    def _generate_clarity_recommendations(self, sentences: List[SentenceClarity],
                                        vocabulary: VocabularyAnalysis,
                                        readability: Dict[str, float],
                                        structure: Dict[str, any]) -> List[str]:
        """Generate clarity improvement recommendations."""
        recommendations = []
        
        # Sentence-level recommendations
        long_sentences = [s for s in sentences if len(s.text.split()) > 25]
        if long_sentences:
            recommendations.append("Consider breaking down long sentences for better clarity. "
                                 f"Found {len(long_sentences)} sentences with 25+ words.")
        
        ambiguous_sentences = [s for s in sentences if s.ambiguity_flags]
        if ambiguous_sentences:
            recommendations.append("Address potential ambiguity in sentence structure. "
                                 "Consider revising sentences with multiple pronouns or complex clauses.")
        
        # Vocabulary recommendations
        if vocabulary.complex_words / vocabulary.total_words > 0.15:
            recommendations.append("Consider simplifying vocabulary. More than 15% of words are complex. "
                                 "Use shorter, more common words when possible.")
        
        if len(vocabulary.filler_words) > vocabulary.total_words * 0.05:
            recommendations.append("Reduce filler words like 'um', 'uh', 'like', and 'you know'. "
                                 f"Found {len(vocabulary.filler_words)} filler words.")
        
        if vocabulary.repetitive_phrases:
            top_repetitive = vocabulary.repetitive_phrases[0]
            recommendations.append(f"Avoid excessive repetition. The word '{top_repetitive[0]}' "
                                 f"appears {top_repetitive[1]} times. Use synonyms for variety.")
        
        # Readability recommendations
        flesch_score = readability.get('flesch_reading_ease', 60)
        if flesch_score < 30:
            recommendations.append("Text is very difficult to read. Simplify sentence structure "
                                 "and vocabulary for better comprehension.")
        elif flesch_score < 50:
            recommendations.append("Text is fairly difficult to read. Consider shortening sentences "
                                 "and using simpler words.")
        
        # Structure recommendations
        avg_sentence_length = structure.get('avg_sentence_length', 15)
        if avg_sentence_length > 20:
            recommendations.append("Average sentence length is high. Aim for 10-20 words per sentence "
                                 "for optimal clarity.")
        
        transition_count = structure.get('transition_word_count', 0)
        sentence_count = structure.get('sentence_count', 1)
        if transition_count / sentence_count < 0.1:
            recommendations.append("Add more transition words to improve flow and connection "
                                 "between ideas (however, therefore, furthermore, etc.).")
        
        return recommendations
    
    def _process_transcription_info(self, transcription_result, original_text: str) -> Dict[str, any]:
        """
        Process and compare transcription result with original text.
        
        Args:
            transcription_result: AI transcription result
            original_text: The original/intended text for comparison
            
        Returns:
            Dictionary containing transcription analysis
        """
        if not transcription_result:
            return None
            
        transcription_info = {
            'transcribed_text': '',
            'word_count': 0,
            'confidence_score': None,
            'has_segments': False,
            'segment_count': 0,
            'comparison_available': bool(original_text and original_text.strip()),
            'accuracy_metrics': None
        }
        
        # Extract basic transcription info
        if hasattr(transcription_result, 'full_text'):
            transcription_info['transcribed_text'] = transcription_result.full_text
        elif hasattr(transcription_result, 'text'):
            transcription_info['transcribed_text'] = transcription_result.text
        elif isinstance(transcription_result, str):
            transcription_info['transcribed_text'] = transcription_result
            
        if hasattr(transcription_result, 'word_count'):
            transcription_info['word_count'] = transcription_result.word_count
        else:
            transcription_info['word_count'] = len(transcription_info['transcribed_text'].split())
            
        if hasattr(transcription_result, 'confidence'):
            transcription_info['confidence_score'] = transcription_result.confidence
            
        # Check for segments
        if hasattr(transcription_result, 'segments') and transcription_result.segments:
            transcription_info['has_segments'] = True
            transcription_info['segment_count'] = len(transcription_result.segments)
            
        # Compare with original text if available (for user-provided text)
        if transcription_info['comparison_available']:
            transcription_info['accuracy_metrics'] = self._calculate_transcription_accuracy(
                transcription_info['transcribed_text'], original_text
            )
            
        return transcription_info
    
    def _calculate_transcription_accuracy(self, transcribed_text: str, original_text: str) -> Dict[str, float]:
        """
        Calculate basic accuracy metrics between transcribed and original text.
        
        Args:
            transcribed_text: AI transcribed text
            original_text: Original/intended text
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not transcribed_text or not original_text:
            return {'similarity': 0.0, 'word_accuracy': 0.0, 'length_ratio': 0.0}
            
        # Normalize texts for comparison
        def normalize_text(text):
            return re.sub(r'[^\w\s]', '', text.lower()).strip()
            
        norm_transcribed = normalize_text(transcribed_text)
        norm_original = normalize_text(original_text)
        
        # Word-level accuracy
        transcribed_words = set(norm_transcribed.split())
        original_words = set(norm_original.split())
        
        if original_words:
            word_accuracy = len(transcribed_words.intersection(original_words)) / len(original_words)
        else:
            word_accuracy = 0.0
            
        # Length ratio
        if len(norm_original) > 0:
            length_ratio = len(norm_transcribed) / len(norm_original)
        else:
            length_ratio = 0.0
            
        # Simple character-based similarity (Jaccard similarity)
        def jaccard_similarity(text1, text2):
            if not text1 and not text2:
                return 1.0
            if not text1 or not text2:
                return 0.0
            set1 = set(text1)
            set2 = set(text2)
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union) if union else 0.0
            
        similarity = jaccard_similarity(norm_transcribed, norm_original)
        
        return {
            'similarity': similarity,
            'word_accuracy': word_accuracy,
            'length_ratio': length_ratio
        }
    
    def _calculate_transcription_based_clarity(self, transcription_result, original_text: str, 
                                             transcription_info: Dict) -> float:
        """
        Calculate clarity score based on transcription accuracy.
        This is the primary clarity assessment method.
        
        Args:
            transcription_result: AI transcription result
            original_text: User-provided original text
            transcription_info: Processed transcription information
            
        Returns:
            Clarity score between 0.0 and 1.0
        """
        if not transcription_result or not original_text.strip():
            return 0.5  # Neutral score when comparison not possible
        
        # Get transcribed text
        transcribed_text = transcription_info.get('transcribed_text', '')
        if not transcribed_text.strip():
            return 0.3  # Low score for failed transcription
        
        # Normalize both texts for comparison
        def normalize_text(text):
            # Convert to lowercase, remove punctuation, extra spaces
            text = re.sub(r'[^\w\s]', '', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        norm_original = normalize_text(original_text)
        norm_transcribed = normalize_text(transcribed_text)
        
        if not norm_original or not norm_transcribed:
            return 0.4  # Low score for empty normalized text
        
        # Calculate multiple accuracy metrics
        original_words = norm_original.split()
        transcribed_words = norm_transcribed.split()
        
        # Track word-level issues
        missed_words = []
        potentially_mispronounced = []
        
        # 1. Word-level accuracy (primary metric)
        original_word_set = set(original_words)
        transcribed_word_set = set(transcribed_words)
        
        if len(original_word_set) == 0:
            word_accuracy = 0.0
        else:
            matched_words = original_word_set.intersection(transcribed_word_set)
            missed_words = list(original_word_set - transcribed_word_set)
            word_accuracy = len(matched_words) / len(original_word_set)
        
        # Identify potential mispronunciations (words that appear in transcription but not in original)
        extra_transcribed_words = list(transcribed_word_set - original_word_set)
        
        # Try to match missed words with extra transcribed words using similarity
        for missed_word in missed_words[:]:  # Use slice copy to allow modification during iteration
            best_match = None
            best_similarity = 0.6  # Minimum similarity threshold
            
            for extra_word in extra_transcribed_words:
                similarity = self._calculate_word_similarity(missed_word, extra_word)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = extra_word
            
            if best_match:
                potentially_mispronounced.append((missed_word, best_match))
                missed_words.remove(missed_word)
                extra_transcribed_words.remove(best_match)
        
        # Store the word analysis results
        transcription_info['missed_words'] = missed_words
        transcription_info['potentially_mispronounced'] = potentially_mispronounced
        
        # 2. Sequence accuracy (considers word order)
        sequence_matches = 0
        max_length = max(len(original_words), len(transcribed_words))
        min_length = min(len(original_words), len(transcribed_words))
        
        for i in range(min_length):
            if original_words[i] == transcribed_words[i]:
                sequence_matches += 1
        
        sequence_accuracy = sequence_matches / max_length if max_length > 0 else 0.0
        
        # 3. Length similarity
        length_ratio = min(len(transcribed_words), len(original_words)) / max(len(transcribed_words), len(original_words)) if max(len(transcribed_words), len(original_words)) > 0 else 0.0
        
        # 4. Character-level similarity (for partial word matches)
        def calculate_char_similarity(text1, text2):
            if not text1 or not text2:
                return 0.0
            # Simple character overlap
            chars1 = set(text1)
            chars2 = set(text2)
            intersection = chars1.intersection(chars2)
            union = chars1.union(chars2)
            return len(intersection) / len(union) if union else 0.0
        
        char_similarity = calculate_char_similarity(norm_original, norm_transcribed)
        
        # Weighted combination of metrics for final clarity score
        clarity_score = (
            word_accuracy * 0.5 +        # 50% weight on word matching
            sequence_accuracy * 0.25 +   # 25% weight on sequence preservation
            length_ratio * 0.15 +        # 15% weight on length similarity
            char_similarity * 0.1        # 10% weight on character similarity
        )
        
        # Apply confidence boost if available
        if transcription_info.get('confidence_score'):
            confidence = transcription_info['confidence_score']
            # Boost score slightly for high confidence transcriptions
            confidence_bonus = (confidence - 0.5) * 0.1 if confidence > 0.5 else 0
            clarity_score = min(1.0, clarity_score + confidence_bonus)
        
        # Ensure score is within valid range
        clarity_score = max(0.0, min(1.0, clarity_score))
        
        logger.info(f"Clarity calculation - Word accuracy: {word_accuracy:.2f}, "
                   f"Sequence accuracy: {sequence_accuracy:.2f}, "
                   f"Length ratio: {length_ratio:.2f}, "
                   f"Final clarity score: {clarity_score:.2f}")
        
        return clarity_score
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words to identify potential mispronunciations.
        
        Args:
            word1: Original intended word
            word2: Transcribed word
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not word1 or not word2:
            return 0.0
        
        if word1 == word2:
            return 1.0
        
        # Calculate edit distance (Levenshtein distance)
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Calculate similarity based on edit distance
        distance = edit_distance(word1.lower(), word2.lower())
        max_length = max(len(word1), len(word2))
        
        if max_length == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_length)
        
        # Bonus for phonetic similarity (simple heuristics)
        phonetic_bonus = 0.0
        
        # Check for common sound confusions
        sound_pairs = [
            ('b', 'p'), ('d', 't'), ('g', 'k'), ('f', 'v'), ('s', 'z'),
            ('th', 'f'), ('th', 's'), ('sh', 's'), ('ch', 'sh')
        ]
        
        for sound1, sound2 in sound_pairs:
            if (sound1 in word1.lower() and sound2 in word2.lower()) or \
               (sound2 in word1.lower() and sound1 in word2.lower()):
                phonetic_bonus += 0.1
                break
        
        # Check for similar starting/ending sounds
        if len(word1) > 1 and len(word2) > 1:
            if word1[0].lower() == word2[0].lower():
                phonetic_bonus += 0.1
            if word1[-1].lower() == word2[-1].lower():
                phonetic_bonus += 0.1
        
        return min(1.0, similarity + phonetic_bonus)
