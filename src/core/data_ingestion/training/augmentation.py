"""Data augmentation module for model training."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

logger = logging.getLogger(__name__)

class DataAugmentor:
    """Implements data augmentation techniques for training data."""
    
    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_sequence_length: int = 512,
        augmentation_prob: float = 0.3
    ):
        """Initialize the data augmentor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_sequence_length: Maximum sequence length
            augmentation_prob: Probability of applying augmentation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_sequence_length = max_sequence_length
        self.augmentation_prob = augmentation_prob
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('wordnet')
    
    def augment_embedding_data(
        self,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> Tuple[List[str], List[List[float]]]:
        """Augment embedding training data.
        
        Args:
            texts: List of input texts
            embeddings: List of target embeddings
            
        Returns:
            Tuple of (augmented_texts, augmented_embeddings)
        """
        augmented_texts = []
        augmented_embeddings = []
        
        for text, embedding in zip(texts, embeddings):
            # Add original data
            augmented_texts.append(text)
            augmented_embeddings.append(embedding)
            
            # Apply augmentation with probability
            if random.random() < self.augmentation_prob:
                # Back-translation augmentation
                back_translated = self._back_translation_augmentation(text)
                if back_translated:
                    augmented_texts.append(back_translated)
                    augmented_embeddings.append(embedding)
                
                # Synonym replacement
                synonym_text = self._synonym_replacement(text)
                if synonym_text:
                    augmented_texts.append(synonym_text)
                    augmented_embeddings.append(embedding)
                
                # Random insertion
                inserted_text = self._random_insertion(text)
                if inserted_text:
                    augmented_texts.append(inserted_text)
                    augmented_embeddings.append(embedding)
        
        return augmented_texts, augmented_embeddings
    
    def augment_llm_data(
        self,
        examples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Augment LLM training data.
        
        Args:
            examples: List of training examples
            
        Returns:
            List of augmented training examples
        """
        augmented_examples = []
        
        for example in examples:
            # Add original example
            augmented_examples.append(example)
            
            # Apply augmentation with probability
            if random.random() < self.augmentation_prob:
                # Input augmentation
                augmented_input = self._augment_input(example["input"])
                if augmented_input:
                    augmented_examples.append({
                        "input": augmented_input,
                        "output": example["output"]
                    })
                
                # Output paraphrasing
                paraphrased_output = self._paraphrase_output(example["output"])
                if paraphrased_output:
                    augmented_examples.append({
                        "input": example["input"],
                        "output": paraphrased_output
                    })
        
        return augmented_examples
    
    def _back_translation_augmentation(self, text: str) -> Optional[str]:
        """Apply back-translation augmentation.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text or None if failed
        """
        try:
            # TODO: Implement back-translation using a translation service
            # For now, return None
            return None
        except Exception as e:
            logger.warning(f"Back-translation augmentation failed: {str(e)}")
            return None
    
    def _synonym_replacement(self, text: str) -> Optional[str]:
        """Apply synonym replacement augmentation.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text or None if failed
        """
        try:
            words = word_tokenize(text)
            augmented_words = []
            
            for word in words:
                if random.random() < 0.3:  # 30% chance to replace word
                    synonyms = []
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            if lemma.name() != word:
                                synonyms.append(lemma.name())
                    
                    if synonyms:
                        augmented_words.append(random.choice(synonyms))
                    else:
                        augmented_words.append(word)
                else:
                    augmented_words.append(word)
            
            return " ".join(augmented_words)
            
        except Exception as e:
            logger.warning(f"Synonym replacement failed: {str(e)}")
            return None
    
    def _random_insertion(self, text: str) -> Optional[str]:
        """Apply random insertion augmentation.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text or None if failed
        """
        try:
            words = word_tokenize(text)
            if len(words) < 2:
                return None
            
            # Get random word from the text
            random_word = random.choice(words)
            
            # Get synonyms for the random word
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    if lemma.name() != random_word:
                        synonyms.append(lemma.name())
            
            if not synonyms:
                return None
            
            # Insert random synonym at random position
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(synonyms))
            
            return " ".join(words)
            
        except Exception as e:
            logger.warning(f"Random insertion failed: {str(e)}")
            return None
    
    def _augment_input(self, text: str) -> Optional[str]:
        """Augment input text for LLM training.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text or None if failed
        """
        try:
            # Apply synonym replacement
            synonym_text = self._synonym_replacement(text)
            if synonym_text:
                return synonym_text
            
            # Apply random insertion
            inserted_text = self._random_insertion(text)
            if inserted_text:
                return inserted_text
            
            return None
            
        except Exception as e:
            logger.warning(f"Input augmentation failed: {str(e)}")
            return None
    
    def _paraphrase_output(self, text: str) -> Optional[str]:
        """Paraphrase output text for LLM training.
        
        Args:
            text: Output text
            
        Returns:
            Paraphrased text or None if failed
        """
        try:
            # Apply back-translation
            back_translated = self._back_translation_augmentation(text)
            if back_translated:
                return back_translated
            
            # Apply synonym replacement
            synonym_text = self._synonym_replacement(text)
            if synonym_text:
                return synonym_text
            
            return None
            
        except Exception as e:
            logger.warning(f"Output paraphrasing failed: {str(e)}")
            return None 