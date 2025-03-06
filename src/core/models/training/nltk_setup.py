"""
NLTK Setup Script

This script downloads the necessary NLTK packages for the model training scripts.
Run this once before running any model training or inference code.
"""

import nltk
import os
import ssl

def download_nltk_packages():
    """Download required NLTK packages."""
    print("Downloading NLTK resources...")
    
    required_packages = [
        'punkt',         # For sentence and word tokenization
        'stopwords',     # Common words to filter out
        'wordnet',       # For lemmatization
        'averaged_perceptron_tagger', # For part-of-speech tagging
    ]
    
    # Create data directory if it doesn't exist
    nltk_data_dir = os.getenv('NLTK_DATA', os.path.expanduser('~/nltk_data'))
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Fix SSL certificate issue
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Download packages
    for package in required_packages:
        try:
            nltk.download(package, quiet=False)
            print(f"✓ Downloaded {package}")
        except Exception as e:
            print(f"✗ Failed to download {package}: {e}")
    
    print("NLTK setup complete.")

if __name__ == "__main__":
    download_nltk_packages()