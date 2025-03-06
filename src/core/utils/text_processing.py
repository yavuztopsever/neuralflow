"""
Text processing utilities for the LangGraph project.
These utilities provide text processing capabilities integrated with LangChain.
"""

from typing import List, Dict, Any, Optional, Union
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: The text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using NLTK.
    
    Args:
        text: The text to process
        
    Returns:
        List[Dict[str, Any]]: List of entities with their types
    """
    # Tokenize and tag parts of speech
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Extract named entities
    named_entities = ne_chunk(pos_tags)
    
    # Convert to list of dictionaries
    entities = []
    current_entity = None
    
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            if current_entity is None:
                current_entity = {
                    'text': ' '.join([token for token, tag in chunk]),
                    'type': chunk.label()
                }
            else:
                current_entity['text'] += ' ' + ' '.join([token for token, tag in chunk])
        elif current_entity is not None:
            entities.append(current_entity)
            current_entity = None
    
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Generate a summary of the text.
    
    Args:
        text: The text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        str: Summarized text
    """
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Calculate sentence scores based on word frequency
    word_freq = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word not in stopwords.words('english'):
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words if word not in stopwords.words('english'))
        sentence_scores[sentence] = score
    
    # Get top sentences
    summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
    summary_sentences = [sentence for sentence, _ in sorted(summary_sentences, key=lambda x: sentences.index(x[0]))]
    
    return ' '.join(summary_sentences)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks using LangChain's text splitter.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Update text splitter parameters
    text_splitter.chunk_size = chunk_size
    text_splitter.chunk_overlap = chunk_overlap
    
    # Create a document
    doc = Document(page_content=text)
    
    # Split into chunks
    chunks = text_splitter.split_documents([doc])
    
    return [chunk.page_content for chunk in chunks]

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using NLTK's VADER.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict[str, float]: Sentiment scores
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: The text to process
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List[str]: List of keywords
    """
    # Tokenize and tag parts of speech
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    
    # Extract nouns and adjectives
    keywords = []
    for word, tag in pos_tags:
        if tag.startswith(('NN', 'JJ')) and word not in stopwords.words('english'):
            keywords.append(word)
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to target language using transformers.
    
    Args:
        text: The text to translate
        target_lang: Target language code
        
    Returns:
        str: Translated text
    """
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-" + target_lang)
    return translator(text)[0]['translation_text']

def extract_phrases(text: str, min_length: int = 2, max_length: int = 5) -> List[str]:
    """
    Extract meaningful phrases from text.
    
    Args:
        text: The text to process
        min_length: Minimum phrase length
        max_length: Maximum phrase length
        
    Returns:
        List[str]: List of phrases
    """
    # Tokenize and tag parts of speech
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    
    # Extract phrases
    phrases = []
    current_phrase = []
    
    for word, tag in pos_tags:
        if tag.startswith(('NN', 'JJ', 'VB')):
            current_phrase.append(word)
        elif current_phrase:
            if min_length <= len(current_phrase) <= max_length:
                phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    if current_phrase and min_length <= len(current_phrase) <= max_length:
        phrases.append(' '.join(current_phrase))
    
    return phrases

__all__ = [
    'clean_text',
    'extract_entities',
    'summarize_text',
    'chunk_text',
    'analyze_sentiment',
    'extract_keywords',
    'translate_text',
    'extract_phrases'
] 