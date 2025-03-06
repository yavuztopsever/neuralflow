"""
Text processing utilities specific to the LangGraph project.
These utilities handle text processing that is not covered by LangChain's built-in capabilities.
"""

from typing import List, Dict, Any, Optional, Union
import re
import unicodedata
from dataclasses import dataclass
import json

@dataclass
class TextProcessingResult:
    text: str
    metadata: Optional[Dict[str, Any]] = None
    tokens: Optional[List[str]] = None
    entities: Optional[List[Dict[str, Any]]] = None

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and normalizing unicode characters."""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities from text using regex patterns."""
    entities = []
    
    # Common entity patterns
    patterns = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)',
        'phone': r'\+?[\d\s-()]{10,}',
        'date': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        'time': r'\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                'type': entity_type,
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
    
    return entities

def tokenize_text(text: str, method: str = 'simple') -> List[str]:
    """Tokenize text using different methods."""
    if method == 'simple':
        # Simple whitespace-based tokenization
        return text.split()
    elif method == 'word':
        # Word-based tokenization with punctuation handling
        return re.findall(r'\b\w+\b', text)
    elif method == 'sentence':
        # Sentence tokenization
        return re.split(r'[.!?]+', text)
    else:
        raise ValueError(f"Unsupported tokenization method: {method}")

def clean_text(text: str, options: Optional[Dict[str, bool]] = None) -> str:
    """Clean text by applying various cleaning options."""
    options = options or {}
    
    if options.get('normalize', True):
        text = normalize_text(text)
    
    if options.get('remove_urls', False):
        text = re.sub(r'https?://\S+', '', text)
    
    if options.get('remove_emails', False):
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    
    if options.get('remove_numbers', False):
        text = re.sub(r'\d+', '', text)
    
    if options.get('remove_punctuation', False):
        text = re.sub(r'[^\w\s]', '', text)
    
    return text

def format_text(text: str, format_type: str = 'plain') -> str:
    """Format text according to specified format type."""
    if format_type == 'plain':
        return text
    elif format_type == 'markdown':
        # Basic markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        return text
    elif format_type == 'json':
        try:
            # Try to parse as JSON and pretty print
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, wrap in a JSON object
            return json.dumps({"text": text}, indent=2)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text using simple frequency-based approach."""
    # Tokenize text
    tokens = tokenize_text(text, method='word')
    
    # Count word frequencies
    word_freq = {}
    for token in tokens:
        token = token.lower()
        if len(token) > 2:  # Ignore very short words
            word_freq[token] = word_freq.get(token, 0) + 1
    
    # Sort by frequency and get top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in keywords[:max_keywords]]

def process_text(text: str, options: Optional[Dict[str, Any]] = None) -> TextProcessingResult:
    """Process text with various options and return a structured result."""
    options = options or {}
    
    # Clean text if requested
    if options.get('clean', True):
        text = clean_text(text, options.get('clean_options', {}))
    
    # Extract entities if requested
    entities = None
    if options.get('extract_entities', True):
        entities = extract_entities(text)
    
    # Tokenize if requested
    tokens = None
    if options.get('tokenize', True):
        tokens = tokenize_text(text, options.get('tokenization_method', 'simple'))
    
    # Format text if requested
    if options.get('format'):
        text = format_text(text, options['format'])
    
    return TextProcessingResult(
        text=text,
        metadata={
            'length': len(text),
            'word_count': len(text.split()),
            'entity_count': len(entities) if entities else 0,
            'token_count': len(tokens) if tokens else 0
        },
        tokens=tokens,
        entities=entities
    ) 