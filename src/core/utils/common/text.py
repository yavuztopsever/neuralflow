"""
Text processing utilities for the LangGraph project.
This module provides text processing capabilities.
"""

from typing import Any, Dict, List, Optional, Union
import re
import unicodedata
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextProcessor:
    """Utility class for text processing."""
    
    def __init__(self):
        """Initialize the text processor."""
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words("english"))
        
        # Load sentiment analysis model
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing diacritics and converting to ASCII.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        # Remove diacritics
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        
        return text
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of words
        """
        return word_tokenize(text)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of sentences
        """
        return sent_tokenize(text)
    
    def remove_stopwords(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text or list of words
            
        Returns:
            Union[str, List[str]]: Text without stopwords
        """
        if isinstance(text, str):
            words = self.tokenize_words(text)
        else:
            words = text
        
        words = [word for word in words if word not in self.stop_words]
        
        if isinstance(text, str):
            return " ".join(words)
        return words
    
    def lemmatize(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Lemmatize text.
        
        Args:
            text: Input text or list of words
            
        Returns:
            Union[str, List[str]]: Lemmatized text
        """
        if isinstance(text, str):
            words = self.tokenize_words(text)
        else:
            words = text
        
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        if isinstance(text, str):
            return " ".join(words)
        return words
    
    def stem(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Stem text.
        
        Args:
            text: Input text or list of words
            
        Returns:
            Union[str, List[str]]: Stemmed text
        """
        if isinstance(text, str):
            words = self.tokenize_words(text)
        else:
            words = text
        
        words = [self.stemmer.stem(word) for word in words]
        
        if isinstance(text, str):
            return " ".join(words)
        return words
    
    def get_pos_tags(self, text: str) -> List[tuple[str, str]]:
        """
        Get part-of-speech tags for text.
        
        Args:
            text: Input text
            
        Returns:
            List[tuple[str, str]]: List of (word, tag) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_named_entities(self, text: str) -> List[tuple[str, str]]:
        """
        Get named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List[tuple[str, str]]: List of (entity, label) tuples
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def get_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        return self.sia.polarity_scores(text)
    
    def get_sentiment_transformers(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores using transformers.
        
        Args:
            text: Input text
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.sentiment_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        return {
            "positive": float(scores[0][1]),
            "negative": float(scores[0][0])
        }
    
    def get_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List[str]: List of keywords
        """
        doc = self.nlp(text)
        
        # Get noun chunks and named entities
        keywords = []
        keywords.extend([chunk.text for chunk in doc.noun_chunks])
        keywords.extend([ent.text for ent in doc.ents])
        
        # Remove duplicates and sort by frequency
        keywords = list(set(keywords))
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = text.lower().count(keyword.lower())
        
        return sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_summary(self, text: str, ratio: float = 0.3) -> str:
        """
        Generate a summary of text.
        
        Args:
            text: Input text
            ratio: Ratio of sentences to keep
            
        Returns:
            str: Generated summary
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Score sentences based on word frequency
        word_freq = {}
        for word in doc:
            if word.is_stop or word.is_punct:
                continue
            word_freq[word.text] = word_freq.get(word.text, 0) + 1
        
        sentence_scores = {}
        for sent in doc.sents:
            for word in sent:
                if word.text in word_freq:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_freq[word.text]
                    else:
                        sentence_scores[sent] += word_freq[word.text]
        
        # Select top sentences
        summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:int(len(sentences) * ratio)]
        summary_sentences = [sent.text for sent, _ in summary_sentences]
        
        # Sort sentences by original order
        summary_sentences.sort(key=lambda x: sentences.index(x))
        
        return " ".join(summary_sentences)

__all__ = ['TextProcessor'] 