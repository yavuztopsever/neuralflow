import json
import redis
import torch
import threading
import os
import logging
import asyncio
from collections import defaultdict
from heapq import nlargest
from typing import Dict, Any, List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from config.config import Config

from tools.memory_manager import MemoryManager
from tools.vector_search import VectorSearch
from tools.graph_search import GraphSearch
from tools.document_handler import DocumentHandler
from tools.function_caller import FunctionCaller  # Ensure this import is correct
from models.model_manager import ModelManager
from config.config import Config
from tools.utils.error_handling import with_error_handling

# Initialize lock for cache operations
cache_lock = threading.Lock()

logger = logging.getLogger(__name__)

class ContextHandler:
    """
    Handles context retrieval, analysis, user preferences, and context prioritization.
    """

    def __init__(self, memory_manager=None, document_handler=None, vector_search=None, web_search=None, graph_search=None, 
                 embedder=None, model_manager=None, function_caller=None, intent_classifier=None, 
                 content_recommendation_model=None, config=None):
                 
        import logging
        self.logger = logging.getLogger(__name__)
        """
        Initializes the ContextHandler with necessary components.
        """
        # Use provided config or global Config
        self.config = config or Config
        
        self.memory_manager = memory_manager or MemoryManager()
        self.document_handler = document_handler or DocumentHandler()
        
        # Handle either vector_search or web_search parameter
        if vector_search is not None:
            self.web_search = vector_search  # For vector search capability
        else:
            self.web_search = web_search or VectorSearch()
            
        self.graph_search = graph_search or GraphSearch()
        
        # Initialize embedder with proper fallback if SentenceTransformer fails
        if embedder:
            self.embedder = embedder
        else:
            try:
                # Try to load the specified embedder model
                self.embedder = SentenceTransformer(self.config.EMBEDDER_MODEL)
            except Exception as e:
                self.logger.error(f"Error initializing SentenceTransformer with {self.config.EMBEDDER_MODEL}: {e}")
                # Try to load a minimal lightweight model as fallback
                try:
                    self.logger.info("Trying to load lightweight 'all-MiniLM-L6-v2' model as fallback...")
                    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    self.logger.info("Successfully loaded lightweight embedding model")
                except Exception as e2:
                    self.logger.error(f"Error initializing fallback embedder: {e2}")
                    raise ImportError(f"Failed to initialize any embedding model. Original error: {e}. Fallback error: {e2}")
        
        self.model_manager = model_manager or ModelManager(memory_manager=self.memory_manager, config=self.config)
        self.function_caller = function_caller or FunctionCaller()
        
        # Initialize intent classifier with proper fallback
        if intent_classifier:
            self.intent_classifier = intent_classifier
        else:
            try:
                self.intent_classifier = pipeline(
                    "text-classification",
                    model=AutoModelForSequenceClassification.from_pretrained(self.config.INTENT_CLASSIFIER_MODEL),
                    tokenizer=AutoTokenizer.from_pretrained(self.config.INTENT_CLASSIFIER_MODEL)
                )
            except Exception as e:
                self.logger.error(f"Error initializing intent classifier with {self.config.INTENT_CLASSIFIER_MODEL}: {e}")
                # Try to load a lightweight classification model as fallback
                try:
                    self.logger.info("Trying to load lightweight classification model as fallback...")
                    # Use a tiny sentiment classifier as fallback
                    self.intent_classifier = pipeline(
                        "text-classification", 
                        model="distilbert-base-uncased-finetuned-sst-2-english"
                    )
                    self.logger.info("Successfully loaded lightweight classification model")
                except Exception as e2:
                    self.logger.error(f"Error initializing fallback classifier: {e2}")
                    raise ImportError(f"Failed to initialize any classification model. Original error: {e}. Fallback error: {e2}")
        self.content_recommendation_model = content_recommendation_model or self.model_manager.get_content_recommendation_model()
        self.user_preferences = self.load_user_preferences()
        self.lock = threading.Lock()  # Initialize a lock for thread safety

    @with_error_handling(error_message="Error retrieving context", default_return={})
    async def get_context(self, query, weights=None, top_k=None, conversation_id: Optional[str] = None):
        """
        Enhanced method to retrieve context for a query with LangGraph integration.
        
        Args:
            query: The user query to get context for
            weights: Dictionary of weight values for different context sources
            top_k: Maximum number of results to return per source
            conversation_id: Optional conversation ID for contextual retrieval
            
        Returns:
            Dictionary containing retrieved context from different sources with reasoning
        """
        self.logger.info(f"Getting context for query: {query[:50]}...")
        print(f"ContextHandler: Getting context for query: {query[:50]}...")
        
        # Use default weights if none provided
        if weights is None:
            weights = {
                "chat_logs": Config.CHAT_LOGS_PREFERENCE,
                "long_term_memory": Config.DEFAULT_LONG_TERM_MEMORY_PREFERENCE,
                "vector_results": Config.DEFAULT_VECTOR_RESULTS_PREFERENCE,
                "graph_results": Config.DEFAULT_GRAPH_RESULTS_PREFERENCE,
                "available_documents": Config.DEFAULT_AVAILABLE_DOCUMENTS_PREFERENCE,
                "function_results": Config.DEFAULT_FUNCTION_RESULTS_PREFERENCE
            }
            
        # Use default top_k if none provided
        if top_k is None:
            top_k = Config.CONTEXT_TOP_K
            
        # Initialize context result
        context = {}
        
        # Track reasoning for context retrieval
        reasoning = []
        
        # Determine if the query is likely to need specific context types
        context_needs = await self._analyze_query_needs(query)
        reasoning.append(f"Query analyzed as needing: {', '.join(context_needs)}")
        
        try:
            # Get chat history (highest priority) with conversation awareness
            if self.memory_manager and (weights.get("chat_logs", 0) > 0 or "memory" in context_needs):
                try:
                    # If conversation_id is provided, prioritize conversation-specific memory
                    if conversation_id and hasattr(self.memory_manager, 'get_session_memory'):
                        reasoning.append(f"Retrieving conversation-specific memory for conversation {conversation_id}")
                        try:
                            session_memory = await asyncio.to_thread(
                                lambda: self.memory_manager.get_session_memory(conversation_id)
                            )
                            context["conversation_memory"] = session_memory
                            reasoning.append(f"Retrieved {len(session_memory)} conversation-specific memory items")
                        except Exception as e:
                            self.logger.error(f"Error retrieving session memory: {e}")
                            context["conversation_memory"] = []
                    
                    # Also get general chat history
                    context["chat_history"] = self._get_chat_history_safely()
                    reasoning.append(f"Retrieved {len(context['chat_history'])} chat history items")
                except Exception as e:
                    self.logger.error(f"Error retrieving chat history: {e}")
                    context["chat_history"] = []
            else:
                context["chat_history"] = []
                
            # Get vector search results if available
            if hasattr(self, 'web_search') and self.web_search and (weights.get("vector_results", 0) > 0 or "vector" in context_needs):
                try:
                    vector_results = self.web_search.search_similar(query, top_k=top_k)
                    context["vector_results"] = vector_results
                    reasoning.append(f"Retrieved {len(vector_results)} vector search results")
                except Exception as e:
                    self.logger.error(f"Error in vector search: {e}")
                    # Try to at least return a mock result if search fails
                    try:
                        mock_result = [
                            f"Vector search failed with error: {e}",
                            "I can still help with your query using other knowledge sources."
                        ]
                        context["vector_results"] = mock_result
                        reasoning.append("Added mock vector search results after error")
                    except:
                        context["vector_results"] = []
            else:
                context["vector_results"] = []
                
            # Get knowledge graph results if available
            if self.graph_search and (weights.get("graph_results", 0) > 0 or "graph" in context_needs):
                try:
                    # Call the appropriate graph search method
                    if hasattr(self.graph_search, 'search'):
                        graph_results = self.graph_search.search(query, top_k=top_k)
                    else:
                        # Fall back to find_related_concepts if search is not available
                        graph_results = self.graph_search.find_related_concepts(query, depth=2)
                    context["graph_results"] = graph_results
                    reasoning.append(f"Retrieved knowledge graph results")
                except Exception as e:
                    self.logger.error(f"Error in graph search: {e}")
                    context["graph_results"] = []
            else:
                context["graph_results"] = []
                
            # Add document context if relevant
            if self.document_handler and (weights.get("available_documents", 0) > 0 or "document" in context_needs):
                try:
                    if hasattr(self.document_handler, 'is_document_query') and self.document_handler.is_document_query(query):
                        available_docs = self.document_handler.list_documents()
                        context["available_documents"] = available_docs
                        reasoning.append(f"Retrieved {len(available_docs)} available documents")
                    else:
                        context["available_documents"] = []
                except Exception as e:
                    self.logger.error(f"Error getting document list: {e}")
                    context["available_documents"] = []
            else:
                context["available_documents"] = []
                
            # Add reasoning to context for LangGraph state tracking
            context["reasoning"] = reasoning
            
            # Evaluate the quality of retrieved context
            sufficiency = await self.evaluate_context_quality(context, query)
            context["quality_evaluation"] = sufficiency
                
        except Exception as e:
            self.logger.error(f"Error in get_context: {e}")
            # Provide minimal context to avoid complete failure
            context = {
                "chat_history": [],
                "vector_results": [],
                "graph_results": [],
                "available_documents": [],
                "reasoning": [f"Error retrieving context: {e}"],
                "error": str(e)
            }
            
        return context
        
    async def _analyze_query_needs(self, query: str) -> List[str]:
        """Analyze the query to determine what types of context it likely needs."""
        # Simple keyword-based analysis as fallback
        needs = []
        
        # Memory is useful for follow-up questions
        if any(term in query.lower() for term in ["earlier", "before", "previous", "last time", "you said"]):
            needs.append("memory")
            
        # Document retrieval for document specific questions
        if any(term in query.lower() for term in ["document", "file", "pdf", "read", "content of"]):
            needs.append("document")
            
        # Vector search for semantic similarity queries
        if any(term in query.lower() for term in ["similar", "related", "like", "such as"]):
            needs.append("vector")
            
        # Graph search for relationship questions
        if any(term in query.lower() for term in ["connected", "relationship", "linked", "between", "how does", "why is"]):
            needs.append("graph")
            
        # Web search for recent information
        if any(term in query.lower() for term in ["current", "latest", "news", "recent", "up to date"]):
            needs.append("web")
        
        return needs if needs else ["memory", "vector", "graph", "document", "web"]
        
    async def evaluate_context_quality(self, context: Dict[str, Any], query: str) -> Dict[str, Union[bool, str]]:
        """Evaluate the quality and sufficiency of retrieved context."""
        result = {
            "is_sufficient": False,
            "reasoning": "",
            "context_sources": [],
            "missing_information": []
        }
        
        # Count non-empty context sources
        non_empty_sources = []
        for source, items in context.items():
            if source not in ["reasoning", "quality_evaluation", "error"] and items and len(items) > 0:
                non_empty_sources.append(source)
                
        result["context_sources"] = non_empty_sources
        
        # Basic sufficiency check - at least one non-empty source
        result["is_sufficient"] = len(non_empty_sources) > 0
        result["reasoning"] = f"Basic check: found {len(non_empty_sources)} non-empty context sources"
        
        if not result["is_sufficient"]:
            result["missing_information"] = ["No context sources returned results"]
            
        return result
            
    def load_user_preferences(self):
        """
        Loads user preferences from cache or persistent storage.
        """
        cache_key = "user_preferences"
        try:
            with cache_lock:
                preferences = self.memory_manager.get_cache(cache_key)
                if preferences is None:
                    try:
                        # Ensure USER_PREFERENCES_PATH is a string
                        preferences_path = Config.USER_PREFERENCES_PATH
                        if hasattr(preferences_path, 'as_posix'):
                            preferences_path = preferences_path.as_posix()
                            
                        # Make sure the parent directory exists
                        os.makedirs(os.path.dirname(preferences_path), exist_ok=True)
                            
                        try:
                            with open(preferences_path, "r") as f:
                                preferences = json.load(f)
                        except FileNotFoundError:
                            # Create default preferences and save them
                            preferences = self.default_preferences()
                            # Save preferences to file
                            with open(preferences_path, "w") as f:
                                json.dump(preferences, f)
                                
                        # Cache preferences
                        try:
                            self.memory_manager.set_cache(cache_key, preferences, ttl=Config.USER_PREFERENCES_CACHE_TTL)
                        except:
                            # If caching fails, just continue with the preferences we have
                            pass
                            
                    except Exception as e:
                        print(f"Error loading preferences: {e}, using defaults")
                        preferences = self.default_preferences()
        except Exception as e:
            print(f"Failed to load user preferences: {e}")
            preferences = self.default_preferences()
            
        return preferences

    @staticmethod
    def default_preferences():
        """
        Returns default user preferences.
        """
        return {
            "chat_logs": Config.DEFAULT_CHAT_LOGS_PREFERENCE,
            "long_term_memory": Config.DEFAULT_LONG_TERM_MEMORY_PREFERENCE,
            "vector_results": Config.DEFAULT_VECTOR_RESULTS_PREFERENCE,
            "graph_results": Config.DEFAULT_GRAPH_RESULTS_PREFERENCE,
            "available_documents": Config.DEFAULT_AVAILABLE_DOCUMENTS_PREFERENCE,
            "function_results": Config.DEFAULT_FUNCTION_RESULTS_PREFERENCE
        }

    def update_user_preferences(self, new_preferences):
        """
        Updates user preferences with new values.
        """
        self.user_preferences.update(new_preferences)
        self.save_user_preferences()

    def save_user_preferences(self):
        """
        Saves user preferences to persistent storage.
        """
        with open(Config.USER_PREFERENCES_PATH, "w") as f:
            json.dump(self.user_preferences, f)
        self.memory_manager.set_cache("user_preferences", self.user_preferences, ttl=Config.USER_PREFERENCES_CACHE_TTL)  # Update cache

    def analyze_query(self, user_query):
        """
        Analyzes the user's query for intents, keywords, and relations.
        """
        # Add reasoning step
        reasoning_steps = ["Analyzing query"]

        # Use a try/except block to handle each model separately
        predicted_intent = "neutral"
        intent_score = 0.5
        relation_label = None
        entities = None
        
        try:
            # Use the intent classifier (most likely to work)
            intent_result = self.intent_classifier(user_query)
            predicted_intent = intent_result[0]["label"]
            intent_score = intent_result[0]["score"]
            self.logger.info(f"Predicted intent: {predicted_intent} ({intent_score})")
            reasoning_steps.append(f"Predicted intent: {predicted_intent} ({intent_score})")
        except Exception as e:
            self.logger.error(f"Error in intent classification: {e}")
            # Continue with default value
        
        # Skip potentially missing models to avoid freezing
        if hasattr(self, 'model_manager'):
            # Check if models directory exists before trying to load models
            try:
                # Only try to infer relation if the model exists
                # Use Config directly instead of self.config
                model_paths = getattr(Config, 'MODEL_PATHS', {})
                if model_paths and 'relation' in model_paths and os.path.exists(model_paths.get('relation', '')):
                    relation_label = self.model_manager.infer_relation(user_query)
                    self.logger.info(f"Inferred relation: {relation_label}")
                    reasoning_steps.append(f"Inferred relation: {relation_label}")
                else:
                    self.logger.warning("Relation model not found, skipping relation inference")
                    
                # Only try to infer linking if the model exists
                if model_paths and 'linking' in model_paths and os.path.exists(model_paths.get('linking', '')):
                    entities = self.model_manager.infer_linking(user_query)
                    self.logger.info(f"Inferred entities: {entities}")
                    reasoning_steps.append(f"Inferred entities: {entities}")
                else:
                    self.logger.warning("Linking model not found, skipping entity linking")
            except Exception as e:
                self.logger.error(f"Error in model inference: {e}")
                # Continue with default values
        
        intents = {
            "document_retrieval": predicted_intent == "document_retrieval",
            "web_search": predicted_intent == "web_search",
            "code_execution": predicted_intent == "code_execution",
            "content_recommendation": predicted_intent == "content_recommendation"
        }

        keywords = user_query.lower().split()

        # Add reasoning steps to the result
        return {"intents": intents, "keywords": keywords, "relation": relation_label, "entities": entities, "reasoning_steps": reasoning_steps}

    def enhance_knowledge_graph(self, user_query, relation_label, entities):
        """
        Enhances the knowledge graph with extracted relations from the user query.
        """
        try:
            if len(entities) >= 2:
                self.graph_search.add_relation(entities[0]['word'], entities[1]['word'], relation_label)
                print(f"Added relation to knowledge graph: {entities[0]['word']} - {relation_label} - {entities[1]['word']}")
            else:
                print("Not enough entities found to add a relation.")
        except Exception as e:
            print(f"Error enhancing knowledge graph: {e}")

    def get_context_sync(self, user_query, context_weights=None, search_params="default"):
        """
        Retrieves context based on user query and intent analysis (synchronous version).
        """
        with self.lock:
            try:
                analysis = self.analyze_query(user_query)
                intents = analysis["intents"]
                keywords = analysis["keywords"]
                relation_label = analysis["relation"]
                entities = analysis["entities"]
            except Exception as e:
                intents, keywords, relation_label, entities = {}, [], None, []
                print(f"Error analyzing query: {e}")

            context = {"intents": intents, "keywords": keywords, "entities": entities}

            if relation_label and entities:
                self.enhance_knowledge_graph(user_query, relation_label, entities)

            context.update(self.retrieve_context_based_on_intents(intents, user_query))

            try:
                if not intents.get("document_retrieval") and not intents.get("code_execution"):
                    context["chat_logs"] = self.memory_manager.get_short_term_memory()
            except Exception as e:
                context["chat_logs"] = []
                print(f"Error retrieving chat logs: {e}")

            try:
                if not intents.get("document_retrieval") and not intents.get("code_execution"):
                    context["long_term_memory"] = self.memory_manager.get_interactions(query=user_query)
            except Exception as e:
                context["long_term_memory"] = []
                print(f"Error retrieving long-term memory: {e}")

            try:
                context["graph_results"] = self.graph_search.find_related_concepts(user_query)
            except Exception as e:
                context["graph_results"] = []
                print(f"Error performing graph search: {e}")

            try:
                context["content_recommendations"] = self.get_content_recommendations(user_query)
            except Exception as e:
                context["content_recommendations"] = []
                print(f"Error getting content recommendations: {e}")

            combined_context = defaultdict(list)
            for key, value in context.items():
                if isinstance(value, list):
                    combined_context[key].extend(value)

            # Calculate relevance safely
            try:
                prioritized_context = nlargest(
                    Config.CONTEXT_TOP_K,
                    combined_context.items(),
                    key=lambda item: self._calculate_relevance(item, user_query)
                )
            except Exception as e:
                print(f"Error calculating relevance: {e}, using all context")
                # If relevance calculation fails, just return all context
                prioritized_context = list(combined_context.items())

        return prioritized_context
        
    def _calculate_relevance(self, item, user_query):
        """Calculate relevance of context item to user query with adaptive weights and caching."""
        # Return a fixed value for tests that mock this method
        if hasattr(self, '_mock_calculate_relevance_return') and self._mock_calculate_relevance_return is not None:
            return self._mock_calculate_relevance_return
            
        cache_key = f"relevance_{hash(user_query)}_{hash(str(item))}"
        
        # Try to get from cache first
        try:
            if hasattr(self, 'memory_manager') and self.memory_manager is not None:
                cached_relevance = self.memory_manager.get_cache(cache_key)
                if cached_relevance is not None:
                    return cached_relevance
        except Exception:
            # Continue if cache retrieval fails
            pass
            
        try:
            context_type, context_data = item
            
            # Get dynamic weights based on query characteristics
            base_weights = self._get_dynamic_weights(user_query, context_type)
            
            # Adjust relevance based on content matching to user query
            keywords = user_query.lower().split()
            
            # Calculate recency boost for time-sensitive content types
            recency_boost = 1.0
            if context_type in ["chat_logs", "function_results"]:
                recency_boost = 1.2  # Boost recent results
            
            # Calculate urgency boost if query indicates urgency
            urgency_boost = 1.0
            if any(urgent_term in user_query.lower() for urgent_term in ["urgent", "immediately", "asap", "emergency"]):
                urgency_boost = 1.5
                
            # If context_data is a list, calculate relevance for each item
            if isinstance(context_data, list):
                item_relevance = 0
                for context_item in context_data:
                    if isinstance(context_item, str):
                        # Calculate keyword match rate
                        keyword_match_count = sum(1 for keyword in keywords if keyword in context_item.lower())
                        keyword_relevance = keyword_match_count / len(keywords) if keywords else 0
                        
                        # Calculate semantic similarity if embedder is available
                        semantic_boost = 1.0
                        try:
                            if hasattr(self, 'embedder') and self.embedder is not None:
                                query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
                                item_embedding = self.embedder.encode(context_item[:min(len(context_item), 512)], convert_to_tensor=True)
                                
                                # Import util only when needed and if available
                                try:
                                    from sentence_transformers import util
                                    similarity = float(util.pytorch_cos_sim(query_embedding, item_embedding)[0][0])
                                    semantic_boost = 1.0 + similarity
                                except ImportError:
                                    # Fall back if util is not available
                                    pass
                        except Exception:
                            pass
                        
                        # Combine all factors
                        item_score = base_weights * (1 + keyword_relevance) * recency_boost * urgency_boost * semantic_boost
                        item_relevance = max(item_relevance, item_score)
                
                # Store in cache before returning
                try:
                    if hasattr(self, 'memory_manager') and self.memory_manager is not None:
                        self.memory_manager.set_cache(cache_key, item_relevance, ttl=300)
                except Exception:
                    pass
                    
                return item_relevance
            else:
                # If context_data is a single item
                if isinstance(context_data, str):
                    # Calculate keyword match rate
                    keyword_match_count = sum(1 for keyword in keywords if keyword in context_data.lower())
                    keyword_relevance = keyword_match_count / len(keywords) if keywords else 0
                    
                    # Calculate semantic similarity if embedder is available
                    semantic_boost = 1.0
                    try:
                        if hasattr(self, 'embedder') and self.embedder is not None:
                            query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
                            item_embedding = self.embedder.encode(context_data[:min(len(context_data), 512)], convert_to_tensor=True)
                            
                            # Import util only when needed and if available
                            try:
                                from sentence_transformers import util
                                similarity = float(util.pytorch_cos_sim(query_embedding, item_embedding)[0][0])
                                semantic_boost = 1.0 + similarity
                            except ImportError:
                                # Fall back if util is not available
                                pass
                    except Exception:
                        pass
                    
                    # Combine all factors
                    relevance = base_weights * (1 + keyword_relevance) * recency_boost * urgency_boost * semantic_boost
                    
                    # Store in cache before returning
                    try:
                        if hasattr(self, 'memory_manager') and self.memory_manager is not None:
                            self.memory_manager.set_cache(cache_key, relevance, ttl=300)
                    except Exception:
                        pass
                        
                    return relevance
                
            # Default case
            return float(base_weights)
        except Exception as e:
            # Default relevance if calculation fails
            print(f"Error calculating relevance: {e}")
            return self._fallback_relevance(item)
            
    def _get_dynamic_weights(self, user_query, context_type):
        """Get dynamically adjusted weights based on query characteristics."""
        # Base weights
        base_weights = {
            "chat_logs": 1.5,         # Recent interactions are highly relevant
            "long_term_memory": 1.0,  # Long-term memory is moderately relevant
            "vector_results": 1.2,    # Vector search results are quite relevant
            "graph_results": 1.1,     # Graph search results are relevant for connected concepts
            "document_content": 1.3,  # Document content is highly relevant
            "function_results": 1.4,  # Function results are highly relevant
        }
        
        # Get base weight with fallback
        base_weight = base_weights.get(context_type, 0.5)
        
        # Adjust weight based on query characteristics
        query_lower = user_query.lower()
        
        # Boost weights for specific intents
        if "document" in query_lower and context_type == "document_content":
            base_weight *= 1.5
        elif "remember" in query_lower and context_type in ["long_term_memory", "chat_logs"]:
            base_weight *= 1.4
        elif "search" in query_lower and context_type == "vector_results":
            base_weight *= 1.3
        elif "related" in query_lower and context_type == "graph_results":
            base_weight *= 1.4
        elif "run" in query_lower and context_type == "function_results":
            base_weight *= 1.5
            
        return base_weight
        
    def _fallback_relevance(self, item):
        """Calculate fallback relevance when primary method fails."""
        try:
            context_type, _ = item
            # Simple fallback using static weights
            weights = {
                "chat_logs": 0.8,
                "long_term_memory": 0.6,
                "vector_results": 0.7,
                "graph_results": 0.6,
                "document_content": 0.8,
                "function_results": 0.9,
            }
            return weights.get(context_type, 0.5)
        except Exception:
            return 0.5

    def retrieve_context_based_on_intents(self, intents, user_query):
        """
        Retrieves context based on the identified intents.
        """
        context = {}
        if intents.get("document_retrieval"):
            context["document_content"] = self.retrieve_document_content(user_query)
        if intents.get("web_search"):
            context["vector_results"] = self.perform_vector_search(user_query)
        if intents.get("code_execution"):
            context["function_results"] = self.execute_code(user_query)
        return context

    def _extract_document_name(self, user_query):
        """
        Extracts document name from user query.
        """
        # Simple extraction based on keywords
        words = user_query.lower().split()
        
        # Look for patterns like "show document X" or "get file Y"
        if "document" in words or "file" in words or "doc" in words:
            # Try to find the word after document/file/doc
            for i, word in enumerate(words):
                if word in ["document", "file", "doc"] and i < len(words) - 1:
                    return words[i + 1]
        
        # If no document name is found, return a default document name or None
        return "default_document.txt"
        
    def retrieve_document_content(self, user_query):
        """
        Retrieves document content based on the user query.
        """
        try:
            document_name = self._extract_document_name(user_query)
            return self.document_handler.retrieve_document(document_name)
        except Exception as e:
            print(f"Error retrieving document content: {e}")
            return ""

    def perform_vector_search(self, user_query):
        """
        Performs a vector search based on the user query.
        """
        try:
            return self.web_search.search_similar(user_query, top_k=Config.VECTOR_SEARCH_TOP_K)
        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []

    def execute_code(self, user_query):
        """
        Executes code based on the user query.
        """
        try:
            return FunctionCaller().execute_code(user_query)
        except Exception as e:
            print(f"Error executing code: {e}")
            return []

    def get_content_recommendations(self, user_query):
        """
        Uses the ContentRecommendationModel to get content recommendations based on the user query.
        """
        try:
            return self.model_manager.infer_content(user_query)
        except Exception as e:
            print(f"Error getting content recommendations: {e}")
            return []

    def prioritize_context(self, context, user_query):
        """
        Prioritizes and prunes context based on relevance and user preferences.
        """
        try:
            weighted_context = []

            base_weights = {
                "chat_logs": Config.DEFAULT_CHAT_LOGS_PREFERENCE,
                "long_term_memory": Config.DEFAULT_LONG_TERM_MEMORY_PREFERENCE,
                "vector_results": Config.DEFAULT_VECTOR_RESULTS_PREFERENCE,
                "graph_results": Config.DEFAULT_GRAPH_RESULTS_PREFERENCE,
                "available_documents": Config.DEFAULT_AVAILABLE_DOCUMENTS_PREFERENCE,
                "function_results": Config.DEFAULT_FUNCTION_RESULTS_PREFERENCE
            }

            # Get user preferences with fallback to defaults
            weights = {}
            for key in base_weights:
                try:
                    weights[key] = base_weights[key] * self.user_preferences.get(key, 1.0)
                except:
                    weights[key] = base_weights[key]

            # Boost document preference if query mentions documents
            if "document" in user_query.lower():
                weights["available_documents"] = weights.get("available_documents", 1.0) * Config.DOCUMENT_QUERY_BOOST

            # Process each context item
            for key, weight in weights.items():
                if key in context and context[key]:
                    # Handle both list and dict types in context
                    if isinstance(context[key], list):
                        items = context[key]
                    elif isinstance(context[key], tuple) and len(context[key]) == 2:
                        # If this is a (key, value) tuple from nlargest
                        continue  # Skip processing this item
                    else:
                        # Single item
                        items = [context[key]]
                        
                    for item in items:
                        if item:  # Skip empty items
                            weighted_context.append((item, weight))

            # Sort by weight
            weighted_context.sort(key=lambda x: x[1], reverse=True)

            # Calculate similarity if embedder is available
            pruned_context = []
            
            try:
                # Create query embedding
                query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
                
                # Process each item
                for item, weight in weighted_context:
                    if isinstance(item, str):
                        try:
                            # Create item embedding
                            item_embedding = self.embedder.encode(item, convert_to_tensor=True)
                            
                            # Calculate similarity safely
                            try:
                                similarity_tensor = util.pytorch_cos_sim(query_embedding, item_embedding)
                                
                                # Extract the scalar value safely
                                if hasattr(similarity_tensor, 'item'):
                                    similarity = similarity_tensor.item()
                                elif isinstance(similarity_tensor, (int, float)):
                                    similarity = similarity_tensor
                                else:
                                    # If it's a tensor with multiple values, get the first one
                                    similarity = similarity_tensor[0][0].item() if hasattr(similarity_tensor[0][0], 'item') else float(similarity_tensor[0][0])
                                
                                if similarity > Config.SIMILARITY_THRESHOLD:
                                    pruned_context.append(item)
                            except Exception as e:
                                # If similarity calculation fails, include item anyway
                                pruned_context.append(item)
                                print(f"Error calculating similarity: {e}")
                        except Exception as e:
                            # If embedding fails, include item anyway
                            pruned_context.append(item)
                            print(f"Error embedding item: {e}")
                    elif item:  # If item is not a string but is truthy, include it
                        pruned_context.append(str(item))
            except Exception as e:
                # If embedding fails completely, just return the weighted items sorted by weight
                print(f"Error using embedder: {e}. Using weight-only prioritization.")
                pruned_context = [item for item, _ in weighted_context]

            return pruned_context
        except Exception as e:
            # Fallback if everything fails
            print(f"Error in prioritize_context: {e}. Returning raw context.")
            
            # Create a simple flat list of all context items
            flat_context = []
            if isinstance(context, dict):
                for items in context.values():
                    if isinstance(items, list):
                        flat_context.extend(items)
                    else:
                        flat_context.append(str(items))
            elif isinstance(context, list):
                flat_context = context
                
            return flat_context

    def _get_chat_history_safely(self):
        """Gets chat history in a safe way that handles missing attributes."""
        try:
            # Try the standard way first
            if hasattr(self.memory_manager, 'get_short_term_memory'):
                return self.memory_manager.get_short_term_memory()
            
            # If that fails, try to access any available interactions
            if hasattr(self.memory_manager, '_local_lists') and hasattr(self.memory_manager._local_lists, 'get'):
                # Try to get from local lists
                interactions = self.memory_manager._local_lists.get('interactions', [])
                if interactions:
                    return [json.loads(interaction) for interaction in interactions]
            
            # If no memory is available, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving chat history: {str(e)}")
            return []
            
    def check_context(self, user_query):
        """
        Checks the context of the user's query using semantic similarity and knowledge graph traversal.
        """
        recent_interactions = self._get_chat_history_safely()
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)

        for interaction in recent_interactions:
            interaction_embedding = self.embedder.encode(interaction, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, interaction_embedding).item()
            if similarity > Config.FOLLOW_UP_THRESHOLD:
                return 'follow-up'

        related_concepts = self.graph_search.traverse_graph(user_query)
        for concept in related_concepts:
            if concept in recent_interactions:
                return 'follow-up'

        return 'new-topic'

    def get_combined_context(self, user_query, context_weights=None, search_params="default"):
        """
        Retrieves and prioritizes the combined context for response generation.
        """
        context = self.get_context(user_query, context_weights, search_params)
        return self.prioritize_context(context, user_query)
