"""Unified data validation and quality monitoring module."""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    message: str
    metrics: Dict[str, Any]
    timestamp: str = datetime.utcnow().isoformat()

class DataValidator:
    """Unified data validation and quality monitoring."""
    
    def __init__(
        self,
        min_sequence_length: int = 1,
        max_sequence_length: int = 512,
        min_examples: int = 5,
        max_examples: int = 10000,
        min_embedding_dim: int = 384,
        max_embedding_dim: int = 768,
        quality_threshold: float = 0.7,
        min_unique_words: int = 10,
        max_duplicate_ratio: float = 0.3,
        min_tfidf_score: float = 0.1,
        outlier_threshold: float = 2.0
    ):
        """Initialize the validator.
        
        Args:
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length
            min_examples: Minimum number of examples
            max_examples: Maximum number of examples
            min_embedding_dim: Minimum embedding dimension
            max_embedding_dim: Maximum embedding dimension
            quality_threshold: Quality score threshold
            min_unique_words: Minimum number of unique words
            max_duplicate_ratio: Maximum ratio of duplicate sequences
            min_tfidf_score: Minimum TF-IDF score
            outlier_threshold: Threshold for outlier detection
        """
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.min_examples = min_examples
        self.max_examples = max_examples
        self.min_embedding_dim = min_embedding_dim
        self.max_embedding_dim = max_embedding_dim
        self.quality_threshold = quality_threshold
        self.min_unique_words = min_unique_words
        self.max_duplicate_ratio = max_duplicate_ratio
        self.min_tfidf_score = min_tfidf_score
        self.outlier_threshold = outlier_threshold
    
    def validate_embedding_data(
        self,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> ValidationResult:
        """Validate embedding data.
        
        Args:
            texts: List of input texts
            embeddings: List of embeddings
            
        Returns:
            Validation result
        """
        metrics = self._analyze_embedding_data(texts, embeddings)
        
        # Check data size
        if len(texts) < self.min_examples:
            return ValidationResult(
                is_valid=False,
                message=f"Insufficient examples: {len(texts)} < {self.min_examples}",
                metrics=metrics
            )
        
        if len(texts) > self.max_examples:
            return ValidationResult(
                is_valid=False,
                message=f"Too many examples: {len(texts)} > {self.max_examples}",
                metrics=metrics
            )
        
        # Check embedding dimensions
        if embeddings and len(embeddings[0]) < self.min_embedding_dim:
            return ValidationResult(
                is_valid=False,
                message=f"Embedding dimension too small: {len(embeddings[0])} < {self.min_embedding_dim}",
                metrics=metrics
            )
        
        if embeddings and len(embeddings[0]) > self.max_embedding_dim:
            return ValidationResult(
                is_valid=False,
                message=f"Embedding dimension too large: {len(embeddings[0])} > {self.max_embedding_dim}",
                metrics=metrics
            )
        
        # Check text quality
        quality_score = self._compute_text_quality(texts)
        if quality_score < self.quality_threshold:
            return ValidationResult(
                is_valid=False,
                message=f"Text quality too low: {quality_score:.2f} < {self.quality_threshold}",
                metrics=metrics
            )
        
        # Check for outliers
        if self._detect_outliers(embeddings):
            return ValidationResult(
                is_valid=False,
                message="Outliers detected in embeddings",
                metrics=metrics
            )
        
        return ValidationResult(
            is_valid=True,
            message="Data validation successful",
            metrics=metrics
        )
    
    def validate_llm_data(
        self,
        examples: List[Dict[str, str]]
    ) -> ValidationResult:
        """Validate LLM training data.
        
        Args:
            examples: List of training examples
            
        Returns:
            Validation result
        """
        metrics = self._analyze_llm_data(examples)
        
        # Check data size
        if len(examples) < self.min_examples:
            return ValidationResult(
                is_valid=False,
                message=f"Insufficient examples: {len(examples)} < {self.min_examples}",
                metrics=metrics
            )
        
        if len(examples) > self.max_examples:
            return ValidationResult(
                is_valid=False,
                message=f"Too many examples: {len(examples)} > {self.max_examples}",
                metrics=metrics
            )
        
        # Check example quality
        quality_score = self._compute_example_quality(examples)
        if quality_score < self.quality_threshold:
            return ValidationResult(
                is_valid=False,
                message=f"Example quality too low: {quality_score:.2f} < {self.quality_threshold}",
                metrics=metrics
            )
        
        # Check for duplicate examples
        if self._check_duplicates(examples):
            return ValidationResult(
                is_valid=False,
                message="Too many duplicate examples detected",
                metrics=metrics
            )
        
        return ValidationResult(
            is_valid=True,
            message="Data validation successful",
            metrics=metrics
        )
    
    def _analyze_embedding_data(
        self,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """Analyze embedding data.
        
        Args:
            texts: List of input texts
            embeddings: List of embeddings
            
        Returns:
            Analysis metrics
        """
        sequence_lengths = [len(text.split()) for text in texts]
        unique_words = len(set(word for text in texts for word in text.split()))
        
        return {
            "num_samples": len(texts),
            "avg_sequence_length": np.mean(sequence_lengths),
            "std_sequence_length": np.std(sequence_lengths),
            "min_sequence_length": min(sequence_lengths),
            "max_sequence_length": max(sequence_lengths),
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "unique_words": unique_words,
            "quality_score": self._compute_text_quality(texts),
            "duplicate_ratio": self._compute_duplicate_ratio(texts),
            "tfidf_score": self._compute_tfidf_score(texts)
        }
    
    def _analyze_llm_data(
        self,
        examples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Analyze LLM training data.
        
        Args:
            examples: List of training examples
            
        Returns:
            Analysis metrics
        """
        input_lengths = [len(example["input"].split()) for example in examples]
        output_lengths = [len(example["output"].split()) for example in examples]
        unique_words = len(set(word for example in examples 
                             for word in example["input"].split() + example["output"].split()))
        
        return {
            "num_samples": len(examples),
            "avg_input_length": np.mean(input_lengths),
            "std_input_length": np.std(input_lengths),
            "avg_output_length": np.mean(output_lengths),
            "std_output_length": np.std(output_lengths),
            "unique_words": unique_words,
            "quality_score": self._compute_example_quality(examples),
            "duplicate_ratio": self._compute_duplicate_ratio([e["input"] for e in examples]),
            "tfidf_score": self._compute_tfidf_score([e["input"] for e in examples])
        }
    
    def _compute_text_quality(self, texts: List[str]) -> float:
        """Compute text quality score.
        
        Args:
            texts: List of texts
            
        Returns:
            Quality score between 0 and 1
        """
        if not texts:
            return 0.0
        
        # Compute various quality metrics
        length_scores = []
        for text in texts:
            words = text.split()
            if self.min_sequence_length <= len(words) <= self.max_sequence_length:
                length_scores.append(1.0)
            else:
                length_scores.append(0.0)
        
        # Average the scores
        return np.mean(length_scores)
    
    def _compute_example_quality(self, examples: List[Dict[str, str]]) -> float:
        """Compute example quality score.
        
        Args:
            examples: List of training examples
            
        Returns:
            Quality score between 0 and 1
        """
        if not examples:
            return 0.0
        
        # Compute various quality metrics
        quality_scores = []
        for example in examples:
            input_words = example["input"].split()
            output_words = example["output"].split()
            
            # Check input length
            if self.min_sequence_length <= len(input_words) <= self.max_sequence_length:
                input_score = 1.0
            else:
                input_score = 0.0
            
            # Check output length
            if self.min_sequence_length <= len(output_words) <= self.max_sequence_length:
                output_score = 1.0
            else:
                output_score = 0.0
            
            # Average the scores
            quality_scores.append((input_score + output_score) / 2)
        
        return np.mean(quality_scores)
    
    def _compute_duplicate_ratio(self, texts: List[str]) -> float:
        """Compute ratio of duplicate sequences.
        
        Args:
            texts: List of texts
            
        Returns:
            Ratio of duplicate sequences
        """
        if not texts:
            return 0.0
        
        # Count unique sequences
        unique_sequences = set(texts)
        return 1 - (len(unique_sequences) / len(texts))
    
    def _compute_tfidf_score(self, texts: List[str]) -> float:
        """Compute TF-IDF score for text diversity.
        
        Args:
            texts: List of texts
            
        Returns:
            TF-IDF score
        """
        if not texts:
            return 0.0
        
        # Count word frequencies
        word_freq = Counter()
        for text in texts:
            word_freq.update(text.split())
        
        # Compute average TF-IDF score
        total_words = sum(word_freq.values())
        if total_words == 0:
            return 0.0
        
        tfidf_scores = []
        for text in texts:
            words = text.split()
            if not words:
                continue
            
            # Compute TF
            tf = word_freq[words[0]] / total_words
            
            # Compute IDF (simplified)
            idf = np.log(len(texts) / (word_freq[words[0]] + 1))
            
            tfidf_scores.append(tf * idf)
        
        return np.mean(tfidf_scores)
    
    def _detect_outliers(self, embeddings: List[List[float]]) -> bool:
        """Detect outliers in embeddings.
        
        Args:
            embeddings: List of embeddings
            
        Returns:
            True if outliers detected, False otherwise
        """
        if not embeddings:
            return False
        
        # Convert embeddings to numpy array
        X = np.array(embeddings)
        
        # Use DBSCAN to detect outliers
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        # Count outliers (points labeled as -1)
        num_outliers = np.sum(labels == -1)
        outlier_ratio = num_outliers / len(embeddings)
        
        return outlier_ratio > 0.1  # More than 10% outliers
    
    def _check_duplicates(self, examples: List[Dict[str, str]]) -> bool:
        """Check for duplicate examples.
        
        Args:
            examples: List of training examples
            
        Returns:
            True if too many duplicates, False otherwise
        """
        if not examples:
            return False
        
        # Count unique input-output pairs
        unique_pairs = set((e["input"], e["output"]) for e in examples)
        duplicate_ratio = 1 - (len(unique_pairs) / len(examples))
        
        return duplicate_ratio > self.max_duplicate_ratio
    
    def clean_embedding_data(
        self,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> Tuple[List[str], List[List[float]]]:
        """Clean embedding data.
        
        Args:
            texts: List of input texts
            embeddings: List of embeddings
            
        Returns:
            Cleaned texts and embeddings
        """
        cleaned_texts = []
        cleaned_embeddings = []
        
        for text, embedding in zip(texts, embeddings):
            words = text.split()
            if self.min_sequence_length <= len(words) <= self.max_sequence_length:
                cleaned_texts.append(text)
                cleaned_embeddings.append(embedding)
        
        return cleaned_texts, cleaned_embeddings
    
    def clean_llm_data(
        self,
        examples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Clean LLM training data.
        
        Args:
            examples: List of training examples
            
        Returns:
            Cleaned examples
        """
        cleaned_examples = []
        
        for example in examples:
            input_words = example["input"].split()
            output_words = example["output"].split()
            
            if (self.min_sequence_length <= len(input_words) <= self.max_sequence_length and
                self.min_sequence_length <= len(output_words) <= self.max_sequence_length):
                cleaned_examples.append(example)
        
        return cleaned_examples 