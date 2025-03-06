from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.memory_manager import create_memory_manager  # Import the factory function
from datetime import datetime
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import subprocess
import os
import json
from config.config import Config
import threading  # Add threading import
import ssl
from models.gguf_wrapper.llm_wrapper import get_llm  # Import the GGUF model wrapper
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFaceHub
from utils.common.hardware import HardwareManager

# Initialize necessary components
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')

@dataclass
class ModelConfig:
    name: str
    type: str
    max_tokens: int
    temperature: float
    embedding_dim: int

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.embedding_model = None
        self.llm = None  # Initialize LLM attribute
        self.models = {}
        self.loaded_models = set()
        self.model_last_used = {}
        self.max_loaded_models = getattr(config, 'MAX_LOADED_MODELS', 1)
        self.lock = threading.Lock()
        
        # Initialize hardware manager
        self.hardware_manager = HardwareManager()
        self.device, self.device_settings = self.hardware_manager.get_device_for_model(
            self.config.name
        )
        
        # Initialize the default LLM
        self.initialize_llm()

    def initialize(self):
        """Initialize the model and embedding model."""
        try:
            # Initialize models with hardware-optimized settings
            if self.config.embedding_model:
                self.embedding_model = self._load_model(
                    self.config.embedding_model,
                    self.device_settings
                )
            
            if self.config.model:
                self.model = self._load_model(
                    self.config.model,
                    self.device_settings
                )
            
            logger.info(f"Initialized models with device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _load_model(self, model_name: str, settings: Dict[str, Any]) -> Any:
        """Load a model with hardware-optimized settings.
        
        Args:
            model_name: Name of the model to load
            settings: Hardware-specific settings
            
        Returns:
            Any: The loaded model
        """
        try:
            # Apply hardware-specific settings
            if settings['device'] == 'cuda':
                # CUDA-specific optimizations
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            elif settings['device'] == 'mps':
                # MPS-specific optimizations
                torch.backends.mps.enable_fallback_to_cpu = True
            
            # Load model with appropriate settings
            model = self._load_model_implementation(
                model_name,
                device=settings['device'],
                dtype=settings['dtype'],
                num_workers=settings['num_workers'],
                pin_memory=settings['pin_memory']
            )
            
            logger.info(f"Loaded model {model_name} on {settings['device']}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _load_model_implementation(self, model_name: str, **kwargs) -> Any:
        """Implementation of model loading with hardware-specific optimizations.
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional arguments for model loading
            
        Returns:
            Any: The loaded model
        """
        # This is a placeholder for actual model loading implementation
        # The actual implementation would depend on the specific model framework being used
        pass

    def initialize_llm(self, provider: str = "local-gguf"):
        """Initialize the LLM based on the selected provider."""
        try:
            # Get hardware-optimized settings for LLM
            _, llm_settings = self.hardware_manager.get_device_for_model(
                self.config.name
            )
            
            # Initialize LLM with appropriate settings
            if provider == "local-gguf":
                # Local GGUF model initialization with hardware settings
                self.llm = self._initialize_local_llm(llm_settings)
            elif provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=getattr(self.config, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
                    temperature=getattr(self.config, 'DEFAULT_TEMPERATURE', 0.7),
                    max_tokens=getattr(self.config, 'OPENAI_MAX_TOKENS', 2000),
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
                logging.info("Initialized OpenAI model")
                
            elif provider == "anthropic":
                self.llm = ChatAnthropic(
                    model=getattr(self.config, 'ANTHROPIC_MODEL', 'claude-2'),
                    temperature=getattr(self.config, 'DEFAULT_TEMPERATURE', 0.7),
                    max_tokens=getattr(self.config, 'ANTHROPIC_MAX_TOKENS', 2000),
                    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                logging.info("Initialized Anthropic model")
                
            elif provider == "huggingface":
                self.llm = HuggingFaceHub(
                    repo_id=getattr(self.config, 'HUGGINGFACE_MODEL', 'gpt2'),
                    model_kwargs={
                        "temperature": getattr(self.config, 'DEFAULT_TEMPERATURE', 0.7),
                        "max_tokens": getattr(self.config, 'HUGGINGFACE_MAX_TOKENS', 1000)
                    },
                    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN')
                )
                logging.info("Initialized HuggingFace model")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            logger.info(f"Initialized LLM on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _initialize_local_llm(self, settings: Dict[str, Any]) -> Any:
        """Initialize a local LLM with hardware-specific settings.
        
        Args:
            settings: Hardware-specific settings
            
        Returns:
            Any: The initialized LLM
        """
        # This is a placeholder for actual LLM initialization
        # The actual implementation would depend on the specific LLM framework being used
        pass

    def generate_response(self, prompt: str) -> str:
        """Generate a response from the current LLM."""
        if not self.llm:
            self.initialize_llm()
            
        try:
            if hasattr(self.llm, 'generate'):
                return self.llm.generate(prompt)
            elif hasattr(self.llm, 'predict'):
                return self.llm.predict(prompt)
            else:
                return str(self.llm(prompt))
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def get_llm(self):
        """Return the initialized LLM."""
        if not self.llm:
            self.initialize_llm()
        return self.llm

    def stream_response(self, prompt, **kwargs):
        """Stream a response from the LLM."""
        if not self.llm:
            self.initialize_llm()
        return self.llm.generate(prompt, **kwargs)

    def get_embeddings(self, text):
        """Get embeddings for the given text."""
        if not self.llm:
            self.initialize_llm()
        return self.llm.get_embeddings(text)

    def analyze_content(self, text):
        """Analyze the content of the given text."""
        if 'content' not in self.models:
            self.load_model('content')
        return self.models['content'](text)

    def analyze_emotion(self, text):
        """Analyze the emotion in the given text."""
        if 'emotion' not in self.models:
            self.load_model('emotion')
        return self.models['emotion'](text)

    def apply_style(self, text, style):
        """Apply the given style to the text."""
        if 'style' not in self.models:
            self.load_model('style')
        return self.models['style'](text, style=style)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "name": self.config.name,
            "type": self.config.type,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "embedding_dim": self.config.embedding_dim
        }

    def load_last_processed_timestamp(self):
        """Loads the last processed timestamp from a log file."""
        try:
            # Get path with fallback
            timestamp_path = getattr(self.config, 'LAST_PROCESSED_TIMESTAMP_LOG', None)
            
            # If no path or path is invalid, use a default path
            if not timestamp_path or not isinstance(timestamp_path, str):
                if hasattr(self.config, 'MODELS_DIR'):
                    models_dir = self.config.MODELS_DIR
                    if hasattr(models_dir, 'as_posix'):
                        models_dir = models_dir.as_posix()
                    timestamp_path = os.path.join(models_dir, "last_processed_timestamp.log")
                else:
                    timestamp_path = "last_processed_timestamp.log"
            
            # Ensure it's a string if it's a Path object
            if hasattr(timestamp_path, 'as_posix'):
                timestamp_path = timestamp_path.as_posix()
                
            # Create directory if needed
            os.makedirs(os.path.dirname(timestamp_path), exist_ok=True)
                
            # Try to open and read
            if os.path.exists(timestamp_path):
                with open(timestamp_path, "r") as file:
                    return float(file.read().strip())
            else:
                # If file doesn't exist yet, just return 0.0
                return 0.0
                
        except (FileNotFoundError, ValueError, TypeError, OSError) as e:
            print(f"Error loading timestamp: {e}. Using default timestamp.")
            return 0.0

    def save_last_processed_timestamp(self, timestamp):
        """Saves the last processed timestamp to a log file."""
        try:
            # Get path with fallback
            timestamp_path = getattr(self.config, 'LAST_PROCESSED_TIMESTAMP_LOG', None)
            
            # If no path or path is invalid, use a default path
            if not timestamp_path or not isinstance(timestamp_path, str):
                if hasattr(self.config, 'MODELS_DIR'):
                    models_dir = self.config.MODELS_DIR
                    if hasattr(models_dir, 'as_posix'):
                        models_dir = models_dir.as_posix()
                    timestamp_path = os.path.join(models_dir, "last_processed_timestamp.log")
                else:
                    timestamp_path = "last_processed_timestamp.log"
            
            # Ensure it's a string if it's a Path object
            if hasattr(timestamp_path, 'as_posix'):
                timestamp_path = timestamp_path.as_posix()
                
            # Create directory if needed
            os.makedirs(os.path.dirname(timestamp_path), exist_ok=True)
                
            # Save the timestamp
            with open(timestamp_path, "w") as file:
                file.write(str(timestamp))
                
        except (TypeError, OSError) as e:
            print(f"Error saving timestamp: {e}")
            # Continue without saving the timestamp

    def _monitor_memory_usage(self):
        """Monitors memory usage and unloads models if it gets too high."""
        import psutil
        import time
        import gc
        
        # Get monitoring config
        memory_check_interval = getattr(self.config, 'MEMORY_CHECK_INTERVAL', 60)
        memory_high_threshold = getattr(self.config, 'MEMORY_HIGH_THRESHOLD', 60)
        memory_critical_threshold = getattr(self.config, 'MEMORY_CRITICAL_THRESHOLD', 75)
        memory_monitor_enabled = getattr(self.config, 'MEMORY_MONITOR_ENABLED', True)
        
        if not memory_monitor_enabled:
            print("Memory monitoring disabled in config")
            return
            
        print(f"Memory monitor started with check interval {memory_check_interval}s, " +
              f"high threshold {memory_high_threshold}%, critical threshold {memory_critical_threshold}%")
        
        while True:
            try:
                # Get current memory usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # Log memory usage
                print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB ({memory_percent:.1f}%)")
                
                # If memory usage is critical, emergency unload all models
                if memory_percent > memory_critical_threshold:
                    print(f"CRITICAL: Memory usage exceeds {memory_critical_threshold}%, emergency unloading ALL models...")
                    with self.lock:
                        for model_type in list(self.loaded_models):
                            self._unload_model(model_type)
                        self.models.clear()  # Ensure all references are gone
                        
                        # Aggressive garbage collection
                        for _ in range(3):
                            gc.collect()
                        print("All models unloaded, memory emergency handled")
                
                # If memory usage is high, unload low-priority models
                elif memory_percent > memory_high_threshold:
                    print(f"HIGH MEMORY: Usage exceeds {memory_high_threshold}%, unloading unused models...")
                    with self.lock:
                        # If we have loaded models, sort by priority and recency
                        if self.loaded_models:
                            # Combine priority and recency for smart unloading
                            models_to_sort = []
                            for model in self.loaded_models:
                                priority = self.model_priority.get(model, 999)  # Default to low priority
                                last_used = self.model_last_used.get(model, 0)  # When was it used
                                # Score combines priority and recency (lower is better to keep)
                                score = priority * 1000000 - last_used  # Priority dominant, recency secondary
                                models_to_sort.append((model, score))
                            
                            # Sort by score (higher score = unload first)
                            sorted_models = sorted(models_to_sort, key=lambda x: -x[1])
                            
                            # Keep only the highest priority, most recently used model
                            models_to_unload = sorted_models[1:]  # Keep first model
                            
                            for model_type, _ in models_to_unload:
                                self._unload_model(model_type)
                                print(f"Unloaded model: {model_type}")
                            
                            # Force garbage collection
                            gc.collect()
                            print(f"Unloaded {len(models_to_unload)} models")
                
                # Sleep before checking again
                time.sleep(memory_check_interval)
            except Exception as e:
                print(f"Error in memory monitor: {e}")
                time.sleep(60)  # Wait longer if there was an error
    
    def _unload_model(self, model_type):
        """Unloads a model from memory."""
        if model_type in self.models:
            print(f"Unloading model: {model_type}")
            del self.models[model_type]
            self.loaded_models.discard(model_type)
            import gc
            gc.collect()
            
    def load_model(self, model_type):
        """Loads a trained machine learning model with memory management."""
        # First check if the model is already loaded
        if model_type in self.models:
            # Update last used time
            self.model_last_used[model_type] = time.time()
            self.loaded_models.add(model_type)
            return self.models[model_type]

        # Check if we need to free memory before loading a new model
        import gc
        if len(self.loaded_models) >= self.max_loaded_models:
            with self.lock:
                # Find the least recently used model to unload
                if self.loaded_models and self.model_last_used:
                    oldest_model = min(
                        self.loaded_models, 
                        key=lambda m: self.model_last_used.get(m, 0)
                    )
                    self._unload_model(oldest_model)
                    gc.collect()

        try:
            model_paths = self.config.MODEL_PATHS
            if model_type not in model_paths:
                print(f"Error: Model type '{model_type}' not found in config.MODEL_PATHS")
                return None
                
            model_path = model_paths[model_type]
            
            # Check if model file/directory exists
            if not os.path.exists(model_path):
                print(f"Warning: Model path '{model_path}' does not exist for {model_type}")
                return None
            
            print(f"Loading model: {model_type}")
            
            # Use smaller model variants or quantization when possible
            if model_type == "content":
                model = torch.load(model_path, map_location=self.device)
            elif model_type == "style":
                # Use a smaller, optimized pipeline
                try:
                    model = pipeline(
                        "text-generation", 
                        model=model_path,
                        device_map="auto",  # Auto device mapping
                        torch_dtype=torch.float16  # Use half precision
                    )
                except:
                    model = pipeline("text-generation", model=model_path)
            elif model_type == "emotion":
                model = self.load_emotion_model(model_path)
            elif model_type == "relation":
                try:
                    model = pipeline(
                        "text-classification", 
                        model=model_path,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                except:
                    model = pipeline("text-classification", model=model_path)
            elif model_type == "linking":
                model = self.load_linking_model(model_path)
            else:
                print(f"Warning: Unknown model type '{model_type}'")
                model = None

            if model is not None:
                with self.lock:
                    self.models[model_type] = model
                    self.loaded_models.add(model_type)
                    self.model_last_used[model_type] = time.time()
                    
            # Force garbage collection after loading
            gc.collect()
            return model
            
        except FileNotFoundError as e:
            print(f"Error loading {model_type} model: File not found: {e}")
            return None
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return None

    def load_emotion_model(self, path):
        """Loads the emotion model and its vectorizer."""
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            
            # Use the config path for vectorizer
            vectorizer_path = self.config.TFIDF_VECTORIZER_PATH
            if not os.path.exists(vectorizer_path):
                print(f"Warning: Vectorizer file not found at {vectorizer_path}")
                return None
                
            with open(vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            return model
        except FileNotFoundError as e:
            print(f"Error loading emotion model: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading emotion model: {e}")
            return None

    def load_linking_model(self, path):
        """Loads the linking model and its tokenizer."""
        try:
            if not os.path.exists(path):
                print(f"Warning: Linking model directory not found at {path}")
                return None
                
            model = AutoModelForTokenClassification.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            return model, tokenizer
        except FileNotFoundError as e:
            print(f"Error loading linking model: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading linking model: {e}")
            return None

    def preprocess_text(self, text):
        """Preprocesses text data for the model."""
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]

    def create_user_profiles(self, interactions):
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

    def start_auto_update(self):
        """Starts a loop to periodically update the models."""
        while self.auto_update:
            new_interactions = self.fetch_new_interaction_data()
            if new_interactions:
                preprocessed_data = self.preprocess_interaction_data(new_interactions)
                self.trigger_training(preprocessed_data)
            time.sleep(self.update_interval)

    def fetch_new_interaction_data(self):
        """Fetches new interaction data from the stored interactions based on the last processed timestamp."""
        new_interactions = [
            interaction for interaction in self.interactions
            if interaction["timestamp"] > self.last_processed_timestamp
        ]
        if new_interactions:
            self.last_processed_timestamp = max(interaction["timestamp"] for interaction in new_interactions)
            self.save_last_processed_timestamp(self.last_processed_timestamp)
        return new_interactions

    def add_interaction(self, interaction):
        """Adds a new interaction to the stored interactions."""
        self.interactions.append(interaction)
        self.save_last_processed_timestamp(interaction["timestamp"])

    def get_interactions(self):
        """Fetches all interaction data from the database."""
        return self.memory_manager.get_interactions()

    def preprocess_interaction_data(self, interactions):
        """Preprocesses the interaction data for model training."""
        preprocessed_data = []
        for interaction in interactions:
            preprocessed_data.append({
                "user_query": self.preprocess_text(interaction["user_query"]),
                "agent_response": self.preprocess_text(interaction["response"]),
                "time_of_day": datetime.fromtimestamp(interaction["timestamp"]).hour,
                "user_location": interaction.get("user_location"),
                "sentiment": interaction.get("sentiment")
            })
        return preprocessed_data

    def fetch_and_preprocess_interaction_data(self):
        """Fetches new interaction data and preprocesses it for training."""
        new_interactions = self.fetch_new_interaction_data()
        if new_interactions:
            return self.preprocess_interaction_data(new_interactions)
        return []

    def fetch_and_preprocess_interaction_data_for_linking(self):
        """Fetches new interaction data and preprocesses it specifically for the linking model."""
        new_interactions = self.fetch_new_interaction_data()
        if new_interactions:
            preprocessed_data = []
            for interaction in new_interactions:
                user_query = self.preprocess_text(interaction["user_query"])
                agent_response = self.preprocess_text(interaction["response"])
                preprocessed_data.append({
                    "user_query": user_query,
                    "agent_response": agent_response
                })
            return preprocessed_data
        return []

    def fetch_and_preprocess_interaction_data_for_relation(self):
        """Fetches new interaction data and preprocesses it specifically for the relation model."""
        new_interactions = self.fetch_new_interaction_data()
        if new_interactions:
            preprocessed_data = []
            for interaction in new_interactions:
                user_query = self.preprocess_text(interaction["user_query"])
                preprocessed_data.append({
                    "user_query": user_query,
                    "label": "no_relation"
                })
            return preprocessed_data
        return []

    def trigger_training(self, preprocessed_data):
        """Triggers the training scripts with the preprocessed data."""
        try:
            self.save_preprocessed_data(preprocessed_data, self.config.PREPROCESSED_DATA_PATH)
            style_data = [(data["user_query"], "informal") for data in preprocessed_data]
            self.save_preprocessed_data(style_data, self.config.PREPROCESSED_STYLE_DATA_PATH)
            self.run_training_scripts()
        except subprocess.CalledProcessError as e:
            print(f"Error triggering training scripts: {e}")

    def save_preprocessed_data(self, data, path):
        """Saves preprocessed data to a file."""
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def run_training_scripts(self):
        """Runs the training scripts."""
        scripts = self.config.TRAINING_SCRIPTS
        for script in scripts:
            subprocess.run(["python", script], check=True)

    # Inference methods
    def infer_style(self, text):
        """Infers the style of the given text using the style model."""
        try:
            with self.lock:
                model = self.load_model("style")
                result = model(text)
            return result[0]['generated_text'] if result else "unknown"
        except Exception as e:
            print(f"Error inferring style: {e}")
            return "unknown"

    def infer_relation(self, text):
        """Infers the relation of the given text using the relation model."""
        try:
            with self.lock:
                model = self.load_model("relation")
                if model is None:
                    print("Relation model not available")
                    return None
                    
                inputs = model[1](text, return_tensors="pt", padding=True, truncation=True)
                outputs = model[0](**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.item()
        except Exception as e:
            print(f"Error inferring relation: {e}")
            return None

    def infer_linking(self, text):
        """Infers linking entities using the linking model."""
        try:
            with self.lock:
                model_package = self.load_model("linking")
                if model_package is None:
                    print("Linking model not available")
                    return None
                    
                model, tokenizer = model_package
                ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
                result = ner_pipeline(text)
            return result
        except Exception as e:
            print(f"Error inferring linking entities: {e}")
            return None

    def infer_content(self, text):
        """Infers the content of the given text using the content model."""
        try:
            with self.lock:
                model = self.load_model("content")
                if model is None:
                    print("Content model not available")
                    return None
                    
                inputs = torch.tensor([self.preprocess_text(text)], device=self.device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=-1)
            return predictions.item()
        except Exception as e:
            print(f"Error inferring content: {e}")
            return None
            
    def get_content_recommendation_model(self):
        """Returns the content recommendation model.
        
        This is used by the ContextHandler to get recommendations 
        based on user queries.
        """
        try:
            return self.load_model("content")
        except Exception as e:
            print(f"Error loading content recommendation model: {e}")
            # Return a mock model as fallback
            class MockRecommendationModel:
                def __call__(self, *args, **kwargs):
                    return 0  # Default recommendation category
            return MockRecommendationModel()

