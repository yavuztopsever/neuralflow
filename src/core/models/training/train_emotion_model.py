import pickle
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from models.model_manager import ModelManager
from config.config import Config  # Import the centralized configuration

# Try to import NLTK components with fallbacks if not available
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    NLTK_AVAILABLE = False
    print("NLTK components not available, using simple tokenization and default stopwords")

# Fallback stopwords if NLTK is not available
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
}

# Simple tokenizer function as fallback
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    # Split on whitespace and filter out empty strings
    return [token for token in text.lower().split() if token]

def preprocess_text(text):
    """Preprocesses text data for the model."""
    if NLTK_AVAILABLE:
        try:
            # Try to use NLTK components
            tokens = word_tokenize(text)
            tokens = [token.lower() for token in tokens if token not in string.punctuation]
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except LookupError:
            # Fall back to simple processing if NLTK data not downloaded
            tokens = simple_tokenize(text)
            tokens = [token for token in tokens if token not in DEFAULT_STOPWORDS]
    else:
        # Use simple tokenization and default stopwords
        tokens = simple_tokenize(text)
        tokens = [token for token in tokens if token not in DEFAULT_STOPWORDS]
    
    return ' '.join(tokens)

def create_user_profiles(interactions):
    """Creates user profiles from interaction data."""
    user_profiles = {}
    for interaction in interactions:
        user_id = interaction.get("user_id")
        if user_id not in user_profiles:
            user_profiles[user_id] = {
                "preferences": {},
                "interests": [],
                "interaction_history": []
            }
        user_profiles[user_id]["interaction_history"].append(interaction)
    return user_profiles

def train_emotion_model(texts, labels):
    """Trains and evaluates the emotion model."""
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    tfidf_vectorizer = TfidfVectorizer()
    train_features = tfidf_vectorizer.fit_transform(train_texts)
    test_features = tfidf_vectorizer.transform(test_texts)
    model = SVC(kernel="linear")
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return model, tfidf_vectorizer, accuracy, report

def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    """Saves the model and vectorizer to the specified paths."""
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print("Model and vectorizer saved successfully.")

def main(model_manager, config):
    preprocessed_data = model_manager.fetch_and_preprocess_interaction_data()
    if not preprocessed_data:
        print("No data available for training. Exiting.")
        return
        
    texts = [data["user_query"] for data in preprocessed_data if "user_query" in data]
    labels = [data["sentiment"] for data in preprocessed_data if "sentiment" in data]
    
    if not texts or not labels or len(texts) != len(labels):
        print(f"Invalid training data. Found {len(texts)} texts and {len(labels)} labels.")
        return
        
    model, tfidf_vectorizer, accuracy, report = train_emotion_model(texts, labels)
    print(f"Emotion model accuracy: {accuracy}")
    print(f"Classification report:\n{report}")
    save_model_and_vectorizer(model, tfidf_vectorizer, config.EMOTION_MODEL_PATH, config.TFIDF_VECTORIZER_PATH)
    
    # For SVC models, the state is stored differently
    # No need to assert for a specific attribute as different model types have different attributes

if __name__ == "__main__":
    model_manager = ModelManager(auto_update=False)
    config = Config()
    main(model_manager, config)
