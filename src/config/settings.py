import os
import logging
import gc
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not installed. Memory monitoring will be limited.")

def _safe_parse_int(env_var, default_value):
    """Safely parse an integer from environment variable, stripping any comments."""
    value = os.getenv(env_var)
    if value is None:
        return default_value
    
    # Strip any comments or extra spaces
    try:
        # Extract just the first part before any #
        clean_value = value.split('#')[0].strip()
        return int(clean_value)
    except (ValueError, AttributeError):
        logging.warning(f"Could not parse {env_var}={value} as integer. Using default {default_value}")
        return default_value

def _safe_parse_float(env_var, default_value):
    """Safely parse a float from environment variable, stripping any comments."""
    value = os.getenv(env_var)
    if value is None:
        return default_value
    
    # Strip any comments or extra spaces
    try:
        # Extract just the first part before any #
        clean_value = value.split('#')[0].strip()
        return float(clean_value)
    except (ValueError, AttributeError):
        logging.warning(f"Could not parse {env_var}={value} as float. Using default {default_value}")
        return default_value

def _safe_parse_bool(env_var, default_value):
    """Safely parse a boolean from environment variable."""
    value = os.getenv(env_var)
    if value is None:
        return default_value
        
    # Get the lowercase clean value
    clean_value = value.split('#')[0].strip().lower()
    
    # Check for truthy values
    if clean_value in ('true', 'yes', '1', 'y', 'on'):
        return True
    # Check for falsy values
    elif clean_value in ('false', 'no', '0', 'n', 'off'):
        return False
    # Default
    else:
        logging.warning(f"Could not parse {env_var}={value} as boolean. Using default {default_value}")
        return default_value

class Config:
    """Configuration settings for NeuralFlow-based agent."""

    # Base directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Load environment variables from .env file
    env_path = BASE_DIR / '.env'
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)
    
    # ========== Memory Management Settings ==========
    # Memory usage thresholds and monitoring
    MEMORY_CHECK_INTERVAL = 30  # Check memory every 30 seconds
    MEMORY_HIGH_THRESHOLD = 50  # Start unloading models at 50%
    MEMORY_CRITICAL_THRESHOLD = 65  # Emergency unload at 65%
    MEMORY_MONITOR_ENABLED = True  # Enable memory monitoring
    
    # Model loading settings
    MAX_LOADED_MODELS = 1  # Only allow 1 model loaded at a time on M4 Mac
    MODEL_PRIORITY = {  # Model loading priority (lower = higher priority)
        "content": 1,
        "style": 3,
        "emotion": 2,
        "relation": 4,
        "linking": 5
    }
    
    # Thread management
    MAX_THREADS = min(4, os.cpu_count() or 4)  # Maximum worker threads
    THREAD_POOL_ENABLED = True  # Use thread pool instead of individual threads
    
    # GGUF optimization settings
    GGUF_N_THREADS = _safe_parse_int("GGUF_N_THREADS", min(4, os.cpu_count() or 4))
    GGUF_N_BATCH = _safe_parse_int("GGUF_N_BATCH", 512)
    GGUF_N_GPU_LAYERS = _safe_parse_int("GGUF_N_GPU_LAYERS", -1)  # -1 means auto-detect
    
    # SQLAlchemy connection pool settings
    DB_POOL_SIZE = 2  # Reduced from default 5
    DB_MAX_OVERFLOW = 1  # Reduced from default 5
    DB_POOL_RECYCLE = 900  # Recycle connections after 15 mins

    # Validation helper for required environment variables
    @staticmethod
    def _validate_env_var(name, default=None, required=False):
        """Validate and get environment variable with proper error handling."""
        value = os.getenv(name)
        
        if value is None and required:
            # Log error for required variables
            logging.error(f"Required environment variable {name} is missing")
            # Use default if provided, otherwise raise exception
            if default is not None:
                logging.warning(f"Using default value for {name}: {default}")
                return default
            raise ValueError(f"Required environment variable {name} is missing")
            
        return value if value is not None else default
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    
    REDIS_PORT = _safe_parse_int("REDIS_PORT", 6379)
    REDIS_DB = _safe_parse_int("REDIS_DB", 0)
    REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    
    # ========== Directory Structure ==========
    # Core data directories
    DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
    VECTOR_DB_DIR = BASE_DIR / os.getenv("VECTOR_DB_DIR", "vector_store")
    GRAPH_DB_DIR = BASE_DIR / os.getenv("GRAPH_DB_DIR", "graph_store")
    LOGS_DIR = BASE_DIR / os.getenv("LOGS_DIR", "logs")
    MODELS_DIR = BASE_DIR / os.getenv("MODELS_DIR", "models")
    STORAGE_DIR = BASE_DIR / os.getenv("STORAGE_DIR", "storage")
    
    # Sub-directories
    VERSION_DIR = DATA_DIR / "versions"
    MEMORY_DIR = BASE_DIR / os.getenv("MEMORY_DIR", "memory")
    TRAINING_DATA_DIR = MODELS_DIR / "training_data"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    
    # File paths
    GRAPH_PATH = GRAPH_DB_DIR / "knowledge_graph.json"
    MEMORY_DB_PATH = MEMORY_DIR / "memory.db"
    SQLALCHEMY_DB_URL = f"sqlite:///{MEMORY_DB_PATH}"
    LONG_TERM_MEMORY_DB = f"sqlite:///{MEMORY_DIR / 'long_term_memory.db'}"
    USER_PREFERENCES_PATH = DATA_DIR / "user_preferences.json"
    USER_PREFERENCES_CACHE_TTL = 3600  # 1 hour cache TTL for user preferences
    
    # ========== LLM Settings ==========
    LLM_MODEL = os.getenv("LLM_MODEL", "local-gguf")  # Using local GGUF model by default
    DEFAULT_TEMPERATURE = _safe_parse_float("DEFAULT_TEMPERATURE", 0.7)
    
    # ========== GGUF LLM Settings ==========
    GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", str(BASE_DIR / "models" / "gguf_llm" / "mistral-7b-v0.1.Q4_K_M.gguf"))
    GGUF_MAX_TOKENS = _safe_parse_int("GGUF_MAX_TOKENS", 512)
    GGUF_TOP_P = _safe_parse_float("GGUF_TOP_P", 0.95)
    GGUF_TOP_K = _safe_parse_int("GGUF_TOP_K", 40)
    GGUF_CONTEXT_WINDOW = _safe_parse_int("GGUF_CONTEXT_WINDOW", 4096)
    
    # ========== OpenAI Settings ==========
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS = _safe_parse_int("OPENAI_MAX_TOKENS", 2000)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # ========== Anthropic Settings ==========
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-2")
    ANTHROPIC_MAX_TOKENS = _safe_parse_int("ANTHROPIC_MAX_TOKENS", 2000)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # ========== HuggingFace Settings ==========
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "gpt2")
    HUGGINGFACE_MAX_TOKENS = _safe_parse_int("HUGGINGFACE_MAX_TOKENS", 1000)
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    
    # ========== Vector Search ==========
    VECTOR_SEARCH_MODEL = os.getenv("VECTOR_SEARCH_MODEL", "text-embedding-ada-002")
    VECTOR_SEARCH_TOP_K = _safe_parse_int("VECTOR_SEARCH_TOP_K", 3)
    EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")  # Default sentence transformer model
    
    # ========== UI Settings ==========
    UI_THEME = os.getenv("UI_THEME", "dark")  # Options: "dark", "light"
    UI_TITLE = os.getenv("UI_TITLE", "NeuralFlow Agent")
    UI_SUBTITLE = os.getenv("UI_SUBTITLE", "Advanced LLM Workflow Orchestration")
    UI_AVATAR = os.getenv("UI_AVATAR", "🤖")
    
    # ========== Model Paths ==========
    MODEL_PATHS = {
        "content": MODELS_DIR / "content_model" / "pytorch_content_model.pth",
        "style": MODELS_DIR / "style_model",
        "emotion": MODELS_DIR / "emotion_model" / "emotion_model.pkl",
        "relation": MODELS_DIR / "relation_model",
        "linking": MODELS_DIR / "linking_model",
    }
    
    # Model setup
    TFIDF_VECTORIZER_PATH = MODELS_DIR / "emotion_model" / "tfidf_vectorizer.pkl"
    MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
    INTENT_CLASSIFIER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # Default sentiment model for intent
    
    # ========== Memory Settings ==========
    
    SHORT_TERM_LIMIT = _safe_parse_int("SHORT_TERM_LIMIT", 100)
    MID_TERM_LIMIT = _safe_parse_int("MID_TERM_LIMIT", 1000)
    SESSION_EXPIRATION_THRESHOLD = _safe_parse_int("SESSION_EXPIRATION_THRESHOLD", 3600)
    CACHE_TTL = _safe_parse_int("CACHE_TTL", 86400)  # 1 day
    RELATIONSHIP_CACHE_TTL = _safe_parse_int("RELATIONSHIP_CACHE_TTL", 3600)  # 1 hour
    
    # ========== Context Settings ==========
    CONTEXT_TOP_K = _safe_parse_int("CONTEXT_TOP_K", 10)
    SIMILARITY_THRESHOLD = _safe_parse_float("SIMILARITY_THRESHOLD", 0.5)
    FOLLOW_UP_THRESHOLD = _safe_parse_float("FOLLOW_UP_THRESHOLD", 0.8)
    DOCUMENT_QUERY_BOOST = _safe_parse_float("DOCUMENT_QUERY_BOOST", 2.0)
    
    # ========== Training Scripts ==========
    TRAINING_SCRIPTS = [
        BASE_DIR / "models" / "training" / "train_content_model.py",
        BASE_DIR / "models" / "training" / "train_style_model.py",
        BASE_DIR / "models" / "training" / "train_relation_model.py",
        BASE_DIR / "models" / "training" / "train_linking_model.py",
        BASE_DIR / "models" / "training" / "train_emotion_model.py"
    ]
    
    # ========== Model Training Settings ==========
    AUTO_UPDATE_MODELS = bool(os.getenv("AUTO_UPDATE_MODELS", True))
    MODEL_UPDATE_INTERVAL = int(os.getenv("MODEL_UPDATE_INTERVAL", 3600))
    LAST_PROCESSED_TIMESTAMP_LOG = MODELS_DIR / "last_processed_timestamp.log"
    PREPROCESSED_DATA_PATH = TRAINING_DATA_DIR / "preprocessed_data.pkl"
    PREPROCESSED_STYLE_DATA_PATH = TRAINING_DATA_DIR / "preprocessed_style_data.pkl"
    
    # Model hyperparameters
    EMBEDDING_DIM = _safe_parse_int("EMBEDDING_DIM", 64)
    HIDDEN_DIM = _safe_parse_int("HIDDEN_DIM", 32)
    BATCH_SIZE = _safe_parse_int("BATCH_SIZE", 32)
    EPOCHS = _safe_parse_int("EPOCHS", 10)
    
    # Training configuration
    TRAINING_ARGS = {
        "output_dir": str(CHECKPOINTS_DIR),
        "num_train_epochs": _safe_parse_int("TRAINING_NUM_EPOCHS", 3),
        "per_device_train_batch_size": _safe_parse_int("TRAINING_BATCH_SIZE", 16),
        "per_device_eval_batch_size": _safe_parse_int("TRAINING_BATCH_SIZE", 16),
        "warmup_steps": _safe_parse_int("TRAINING_WARMUP_STEPS", 500),
        "weight_decay": _safe_parse_float("TRAINING_WEIGHT_DECAY", 0.01),
        "logging_dir": str(LOGS_DIR / "training"),
        "logging_steps": _safe_parse_int("TRAINING_LOGGING_STEPS", 10),
    }
    
    # ========== Response Generation ==========
    RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "local-gguf")
    RESPONSE_TEMPERATURE = _safe_parse_float("RESPONSE_TEMPERATURE", 0.7)
    SENTIMENT_ANALYSIS_PIPELINE = "sentiment-analysis"
    
    # Response prompts
    RESPONSE_PROMPT_NEGATIVE = """
    User is expressing negative sentiment and the style is {style_label}. Emotion detected: {emotion_label}.
    Given the following information:
    Execution Result: {execution_result}
    Retrieved Context: {retrieved_context}

    Provide a structured, clear response.
    Acknowledge the negative sentiment and offer support or solutions.
    """
    
    RESPONSE_PROMPT_POSITIVE = """
    User query style is {style_label}. Emotion detected: {emotion_label}.
    Given the following information:
    Execution Result: {execution_result}
    Retrieved Context: {retrieved_context}

    Provide a structured, clear response.
    Adapt the response style to match the user's style.
    """
    
    # ========== Task Execution ==========
    DEFAULT_SEARCH_PARAMS = {}
    DEFAULT_EXECUTION_MODE = "safe"
    
    # Resource limits
    RESOURCE_LIMITS = {
        "safe": {"cpu": 5, "memory": 50},
        "unsafe": {"cpu": 10, "memory": 100}
    }
    
    # Code validation patterns
    CODE_VALIDATION_PATTERNS = {
        "python": [
            r"import\s+(os|subprocess|sys|shutil)",
            r"os\.system|subprocess\.call|sys\.exit|shutil\.rmtree",
            r"open\(|read\(|write\(",
            r"socket|requests"
        ],
        "javascript": [
            r"require\s*\(\s*[\'\"]fs[\'\"]\s*\)|require\s*\(\s*[\'\"]child_process[\'\"]\s*\)",
            r"fs\.open|fs\.read|fs\.write",
            r"http|net"
        ]
    }
    
    # ========== Distributed Processing ==========
    DISTRIBUTED_BACKEND = os.getenv("DISTRIBUTED_BACKEND", "local")  # Default to local execution
    DISTRIBUTED_ADDRESS = os.getenv("DISTRIBUTED_ADDRESS", "localhost:6379")
    DISTRIBUTED_NUM_WORKERS = _safe_parse_int("DISTRIBUTED_NUM_WORKERS", 4)
    DISTRIBUTED_TIMEOUT_MS = _safe_parse_int("DISTRIBUTED_TIMEOUT_MS", 3000)  # 3 second timeout
    STATE_SAVE_INTERVAL = _safe_parse_int("STATE_SAVE_INTERVAL", 300)  # Save state every 5 minutes by default
    
    # ========== User Preferences ==========
    CHAT_LOGS_PREFERENCE = _safe_parse_float("CHAT_LOGS_PREFERENCE", 1.5)
    DEFAULT_CHAT_LOGS_PREFERENCE = _safe_parse_float("DEFAULT_CHAT_LOGS_PREFERENCE", 1.0)
    DEFAULT_LONG_TERM_MEMORY_PREFERENCE = _safe_parse_float("DEFAULT_LONG_TERM_MEMORY_PREFERENCE", 0.8)
    DEFAULT_VECTOR_RESULTS_PREFERENCE = _safe_parse_float("DEFAULT_VECTOR_RESULTS_PREFERENCE", 0.9)
    DEFAULT_GRAPH_RESULTS_PREFERENCE = _safe_parse_float("DEFAULT_GRAPH_RESULTS_PREFERENCE", 0.7)
    DEFAULT_AVAILABLE_DOCUMENTS_PREFERENCE = _safe_parse_float("DEFAULT_AVAILABLE_DOCUMENTS_PREFERENCE", 0.6)
    DEFAULT_FUNCTION_RESULTS_PREFERENCE = _safe_parse_float("DEFAULT_FUNCTION_RESULTS_PREFERENCE", 1.0)
    
    # ========== Logging Settings ==========
    @staticmethod
    def _safe_parse_log_level(env_var, default_value):
        """Safely parse a log level from environment variable."""
        value = os.getenv(env_var)
        if value is None:
            return default_value
            
        # Get the clean value
        try:
            clean_value = value.split('#')[0].strip().upper()
            
            # Check for valid log levels
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if clean_value in valid_levels:
                return clean_value
            else:
                logging.warning(f"Invalid log level: {clean_value}. Using default: {default_value}")
                return default_value
        except (AttributeError, ValueError):
            return default_value
    
    LOG_LEVEL = _safe_parse_log_level("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # ========== Device Settings ==========
    USE_MPS = _safe_parse_bool("USE_MPS", False)
    USE_CPU = _safe_parse_bool("USE_CPU", True)
    PYTORCH_MPS_HIGH_WATERMARK_RATIO = _safe_parse_float("PYTORCH_MPS_HIGH_WATERMARK_RATIO", 0.0)
    
    # New attributes
    memory_db_url = None
    memory_backup_dir = None
    memory_prune_age_days = None
    debug = None
    vector_search_dimension = None
    vector_search_collection = None
    web_search_api_key = None
    web_search_engine_id = None
    function_calling_enabled = None
    function_calling_timeout = None
    response_style = None
    response_priority = None
    
    def __init__(self):
        # Model configuration
        self.model_path = "models/gguf_llm/mistral-7b-v0.1.Q4_K_M.gguf"
        self.model_type = "gguf"
        self.max_tokens = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Memory configuration
        self.memory_db_url = "sqlite+aiosqlite:///memory.db"
        self.memory_backup_dir = "backups"
        self.memory_prune_age_days = 30
        
        # Debug configuration
        self.debug = False
        
        # Vector search configuration
        self.vector_search_dimension = 384
        self.vector_search_collection = "documents"
        
        # Web search configuration
        self.web_search_api_key = None
        self.web_search_engine_id = None
        
        # Function calling configuration
        self.function_calling_enabled = True
        self.function_calling_timeout = 30
        
        # Response generation configuration
        self.response_style = "default"
        self.response_priority = "normal"
        
        # Logging configuration
        self.log_level = "INFO"
        self.log_file = "app.log"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all necessary directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.VECTOR_DB_DIR,
            cls.GRAPH_DB_DIR,
            cls.LOGS_DIR,
            cls.MODELS_DIR,
            cls.VERSION_DIR,
            cls.MEMORY_DIR,
            cls.TRAINING_DATA_DIR,
            cls.CHECKPOINTS_DIR,
            cls.STORAGE_DIR,
            # Ensure model directories exist
            cls.MODELS_DIR / "content_model",
            cls.MODELS_DIR / "style_model",
            cls.MODELS_DIR / "emotion_model",
            cls.MODELS_DIR / "relation_model",
            cls.MODELS_DIR / "linking_model",
            # Training logs
            cls.LOGS_DIR / "training",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        cls.setup_logging()
    
    @classmethod
    def setup_logging(cls) -> None:
        """Set up logging configuration."""
        # Convert string level to actual logging level
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        # Use level name if valid, or default to INFO
        level = log_level_map.get(cls.LOG_LEVEL, logging.INFO)
        
        try:
            # Ensure log directory exists
            os.makedirs(cls.LOGS_DIR, exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=level,
                format=cls.LOG_FORMAT,
                datefmt=cls.LOG_DATE_FORMAT,
                handlers=[
                    logging.FileHandler(os.path.join(cls.LOGS_DIR, "app.log")),
                    logging.StreamHandler()
                ]
            )
            logging.info(f"Logging initialized at level: {cls.LOG_LEVEL}")
        except Exception as e:
            # Simple console logging fallback
            print(f"Error setting up logging: {e}. Using basic console logging.")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()]
            )
        
    @classmethod
    def get_model_paths(cls) -> Dict[str, str]:
        """Get model paths converted to strings for compatibility."""
        return {k: str(v) for k, v in cls.MODEL_PATHS.items()}
        
    @classmethod
    def get_training_scripts(cls) -> List[str]:
        """Get training script paths converted to strings."""
        return [str(script) for script in cls.TRAINING_SCRIPTS]
    
    @classmethod
    def get_memory_usage(cls) -> Dict[str, Any]:
        """Get current memory usage information."""
        # Initialize result with basic information
        memory_info = {
            "available": True,
            "percent": 0,
            "used_mb": 0,
            "total_mb": 0,
            "process_mb": 0,
            "process_percent": 0,
        }
        
        # If psutil is available, get detailed memory info
        if HAS_PSUTIL:
            try:
                # System memory info
                system_memory = psutil.virtual_memory()
                memory_info["percent"] = system_memory.percent
                memory_info["used_mb"] = system_memory.used / (1024 * 1024)
                memory_info["total_mb"] = system_memory.total / (1024 * 1024)
                
                # Process memory info
                process = psutil.Process(os.getpid())
                process_memory = process.memory_info()
                memory_info["process_mb"] = process_memory.rss / (1024 * 1024)
                memory_info["process_percent"] = process.memory_percent()
            except Exception as e:
                logging.warning(f"Error getting memory info: {e}")
                memory_info["available"] = False
        else:
            memory_info["available"] = False
        
        return memory_info
    
    @classmethod
    def force_gc(cls) -> None:
        """Force garbage collection to free memory."""
        gc.collect()

# Initialize directories and logging when module is imported 
# We'll wrap this in a try/except to handle any initialization errors gracefully
try:
    Config.ensure_directories()
except Exception as e:
    print(f"Warning: Error during initial configuration setup: {e}")
    print("The application will attempt to continue with default settings.")
